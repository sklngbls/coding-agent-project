from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coding_agent.agent import SYSTEM_PROMPT, CodingAgent, ProgressEvent
from coding_agent.llm import ModelResponse, ToolCall
from coding_agent.tools import LocalTools, ToolRegistry


class FakeModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        self.calls.append((list(messages), list(tools)))
        if not self.responses:
            raise AssertionError("FakeModel has no response left")
        return self.responses.pop(0)


def build_agent(
    tmp_path: Path,
    responses: list[ModelResponse],
    *,
    max_steps: int = 10,
    events: list[ProgressEvent] | None = None,
) -> tuple[CodingAgent, FakeModel]:
    model = FakeModel(responses)
    registry = ToolRegistry(LocalTools(tmp_path).definitions())
    callback = events.append if events is not None else None
    return (
        CodingAgent(
            model,
            registry,
            workspace=tmp_path,
            max_steps=max_steps,
            on_progress=callback,
        ),
        model,
    )


def call(call_id: str, name: str, arguments: dict[str, object] | str) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, name=name, arguments=raw)


def test_multiple_rounds_of_tools_until_final_answer(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("before", encoding="utf-8")
    responses = [
        ModelResponse(tool_calls=[call("read-1", "read_file", {"path": "input.txt"})]),
        ModelResponse(
            content="I will update it.",
            tool_calls=[
                call(
                    "write-1",
                    "write_file",
                    {"path": "input.txt", "content": "after"},
                )
            ],
        ),
        ModelResponse(content="Updated input.txt and verified the operation."),
    ]
    agent, model = build_agent(tmp_path, responses)

    result = agent.run("Change before to after")

    assert result.status == "completed"
    assert result.steps == 3
    assert result.final_answer.startswith("Updated")
    assert (tmp_path / "input.txt").read_text(encoding="utf-8") == "after"
    third_request_messages = model.calls[2][0]
    assert [message["role"] for message in third_request_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    first_tool_payload = json.loads(third_request_messages[3]["content"])
    assert first_tool_payload["data"]["content"] == "before"


def test_multiple_tool_calls_in_one_response_are_all_dispatched(tmp_path: Path) -> None:
    responses = [
        ModelResponse(
            tool_calls=[
                call("one", "write_file", {"path": "one.txt", "content": "1"}),
                call("two", "write_file", {"path": "two.txt", "content": "2"}),
            ]
        ),
        ModelResponse(content="Created both files."),
    ]
    agent, model = build_agent(tmp_path, responses)

    result = agent.run("Create two files")

    assert result.status == "completed"
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "1"
    assert (tmp_path / "two.txt").read_text(encoding="utf-8") == "2"
    second_request_messages = model.calls[1][0]
    assert [message["tool_call_id"] for message in second_request_messages[-2:]] == ["one", "two"]


def test_unknown_tool_and_bad_json_are_returned_to_model(tmp_path: Path) -> None:
    responses = [
        ModelResponse(
            tool_calls=[
                call("unknown", "not_a_tool", {}),
                call("bad-json", "read_file", "{"),
            ]
        ),
        ModelResponse(content="Recovered from both tool errors."),
    ]
    agent, model = build_agent(tmp_path, responses)

    result = agent.run("Try tools")

    assert result.status == "completed"
    tool_messages = model.calls[1][0][-2:]
    payloads = [json.loads(message["content"]) for message in tool_messages]
    assert payloads[0]["ok"] is False
    assert "Unknown tool" in payloads[0]["message"]
    assert payloads[1]["ok"] is False
    assert "Invalid JSON" in payloads[1]["message"]


def test_max_steps_stops_loop_after_executing_last_tools(tmp_path: Path) -> None:
    responses = [
        ModelResponse(tool_calls=[call("one", "list_files", {"path": "."})]),
        ModelResponse(tool_calls=[call("two", "list_files", {"path": "."})]),
        ModelResponse(content="This response must never be consumed."),
    ]
    agent, model = build_agent(tmp_path, responses, max_steps=2)

    result = agent.run("Keep looking")

    assert result.status == "max_steps"
    assert result.steps == 2
    assert "maximum of 2" in result.final_answer
    assert len(model.calls) == 2
    assert result.messages[-1]["role"] == "tool"


def test_model_failure_becomes_agent_result(tmp_path: Path) -> None:
    agent, _ = build_agent(tmp_path, [])

    result = agent.run("Do something")

    assert result.status == "model_error"
    assert result.steps == 1
    assert "Model request failed" in result.final_answer


def test_progress_events_do_not_contain_tool_arguments(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    secret_content = "content-that-should-not-be-in-progress"
    responses = [
        ModelResponse(
            tool_calls=[
                call(
                    "write",
                    "write_file",
                    {"path": "file.txt", "content": secret_content},
                )
            ]
        ),
        ModelResponse(content="Done"),
    ]
    agent, _ = build_agent(tmp_path, responses, events=events)

    agent.run("Write a file")

    assert any(event.kind == "tool" and event.message == "write_file" for event in events)
    assert all(secret_content not in event.message for event in events)


def test_empty_model_content_is_handled(tmp_path: Path) -> None:
    agent, _ = build_agent(tmp_path, [ModelResponse()])
    result = agent.run("Finish")
    assert result.status == "completed"
    assert "without a final text" in result.final_answer


def test_run_continues_existing_history_without_duplicate_system_or_mutation(
    tmp_path: Path,
) -> None:
    history = [
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "earlier task"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    original_history = json.loads(json.dumps(history))
    agent, model = build_agent(tmp_path, [ModelResponse(content="continued")])

    result = agent.run("next task", history=history)

    assert result.status == "completed"
    assert history == original_history
    sent = model.calls[0][0]
    assert [message["role"] for message in sent] == ["system", "user", "assistant", "user"]
    assert sum(message["role"] == "system" for message in sent) == 1
    assert sent[-1]["content"] == "next task"


def test_run_adds_system_prompt_when_history_has_none(tmp_path: Path) -> None:
    history: list[dict[str, Any]] = [{"role": "user", "content": "earlier task"}]
    agent, model = build_agent(tmp_path, [ModelResponse(content="done")])

    result = agent.run("latest task", history=history)

    assert result.status == "completed"
    sent = model.calls[0][0]
    assert [message["role"] for message in sent] == ["system", "user", "user"]
    assert sent[0]["content"] == SYSTEM_PROMPT


def test_long_history_is_bounded_before_model_call(tmp_path: Path) -> None:
    history = [{"role": "system", "content": "system"}]
    history.extend({"role": "user", "content": str(index)} for index in range(20))
    agent, model = build_agent(tmp_path, [ModelResponse(content="done")])
    agent.max_history_messages = 5

    result = agent.run("latest", history=history)

    assert result.status == "completed"
    sent = model.calls[0][0]
    assert len(sent) == 5
    assert sent[0]["role"] == "system"
    assert sent[-1]["content"] == "latest"


def test_history_limit_does_not_split_tool_call_round(tmp_path: Path) -> None:
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old-call",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old-call", "content": "old result"},
        {"role": "user", "content": "current task"},
    ]
    agent, model = build_agent(tmp_path, [ModelResponse(content="done")])
    agent.max_history_messages = 4

    result = agent.run("latest task", history=history)

    assert result.status == "completed"
    sent = model.calls[0][0]
    assert [message["role"] for message in sent] == ["system", "user", "user"]
    assert sent[-2]["content"] == "current task"
    assert sent[-1]["content"] == "latest task"
