"""Core coding-agent loop implemented without an agent framework."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .history import truncate_messages
from .llm import ChatModel, ModelResponse
from .tools import ToolRegistry

SYSTEM_PROMPT = """You are a careful coding agent operating in one configured local workspace.
The host's absolute workspace path is intentionally not exposed to you.

Complete the user's programming task by inspecting the workspace, editing files, and running
relevant checks. Use only the provided tools for local operations. All file and working-directory
paths must be relative to the workspace. Commands accept a structured argv array and do not run
through a shell, so do not use pipes, redirection, or shell operators.

Before editing, inspect relevant files. Prefer focused changes that preserve existing conventions.
After editing, run proportionate tests or checks. Read tool results carefully, recover from errors
when possible, and never claim a command or test passed unless its tool result says so. You may call
multiple tools in one response when they are independent. When the task is complete or cannot make
further progress, answer with a concise summary, verification results, and any remaining issue.
"""


RunStatus = Literal["completed", "max_steps", "model_error"]


@dataclass(frozen=True)
class ProgressEvent:
    """Non-sensitive progress information suitable for terminal display."""

    step: int
    kind: Literal["model", "tool"]
    message: str
    ok: bool | None = None


@dataclass(frozen=True)
class AgentResult:
    """Final outcome of an agent run."""

    final_answer: str
    status: RunStatus
    steps: int
    messages: list[dict[str, Any]]


ProgressCallback = Callable[[ProgressEvent], None]


class CodingAgent:
    """Drive model/tool turns until a final answer or the step limit is reached."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        *,
        workspace: str | Path,
        max_steps: int = 20,
        max_history_messages: int = 100,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_history_messages < 2:
            raise ValueError("max_history_messages must be at least 2")
        self.model = model
        self.tools = tools
        self.workspace = Path(workspace).resolve()
        self.max_steps = max_steps
        self.max_history_messages = max_history_messages
        self.on_progress = on_progress

    def run(
        self,
        task: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if history is not None:
            if not isinstance(history, list) or any(
                not isinstance(message, dict) for message in history
            ):
                raise ValueError("history must be a list of message objects")
            # Keep the caller's list and nested values untouched while trimming old context.
            messages = copy.deepcopy(history)
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            messages = truncate_messages(messages, self.max_history_messages - 1)
            messages.append({"role": "user", "content": task.strip()})
        else:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": task.strip()},
            ]

        for step in range(1, self.max_steps + 1):
            self._emit(ProgressEvent(step=step, kind="model", message="Requesting model response"))
            try:
                messages = truncate_messages(messages, self.max_history_messages)
                response = self.model.complete(messages, self.tools.schemas)
            except Exception as exc:
                answer = f"Model request failed: {exc}"
                return AgentResult(
                    final_answer=answer,
                    status="model_error",
                    steps=step,
                    messages=messages,
                )

            messages.append(_assistant_message(response))
            if not response.tool_calls:
                answer = response.content or "The model ended without a final text response."
                return AgentResult(
                    final_answer=answer,
                    status="completed",
                    steps=step,
                    messages=messages,
                )

            for call in response.tool_calls:
                result = self.tools.dispatch(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.to_json(),
                    }
                )
                self._emit(
                    ProgressEvent(
                        step=step,
                        kind="tool",
                        message=call.name,
                        ok=result.ok,
                    )
                )

        answer = (
            f"Stopped after reaching the maximum of {self.max_steps} model steps. "
            "The task may be incomplete."
        )
        return AgentResult(
            final_answer=answer,
            status="max_steps",
            steps=self.max_steps,
            messages=messages,
        )

    def _emit(self, event: ProgressEvent) -> None:
        if self.on_progress is not None:
            self.on_progress(event)


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in response.tool_calls
        ]
    return message
