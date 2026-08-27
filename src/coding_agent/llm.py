"""Model abstraction and OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


class ModelRequestError(RuntimeError):
    """Raised when a model request fails after secret redaction."""


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call returned by a chat model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse:
    """Normalized assistant response used by the agent loop."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    """Minimal interface required by :class:`CodingAgent`."""

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        """Return the next assistant response for the conversation."""


class OpenAIChatModel:
    """Adapter for any OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "The 'openai' package is required for live model calls. "
                "Install project dependencies first."
            ) from exc
        try:
            self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        except Exception as exc:
            safe_message = str(exc).replace(settings.api_key, "[redacted]")
            raise ModelRequestError(safe_message) from exc
        self._model = settings.model
        self._api_key = settings.api_key

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
            )
        except Exception as exc:
            safe_message = str(exc).replace(self._api_key, "[redacted]")
            raise ModelRequestError(safe_message) from exc
        if not response.choices:
            raise ModelRequestError("Model returned no choices")
        message = response.choices[0].message
        normalized_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            function = call.function
            normalized_calls.append(
                ToolCall(
                    id=str(call.id),
                    name=str(function.name),
                    arguments=str(function.arguments),
                )
            )
        return ModelResponse(
            content=message.content,
            tool_calls=normalized_calls,
        )
