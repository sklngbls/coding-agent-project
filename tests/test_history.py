from __future__ import annotations

from typing import Any

from coding_agent.history import estimate_tokens, truncate_messages_by_tokens


def test_estimate_tokens_handles_text_and_structured_values() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("中文") == 2
    assert estimate_tokens({"content": "abcd"}) >= 2


def test_token_truncation_keeps_system_and_complete_tool_round() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old task " + "x" * 40},
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

    trimmed = truncate_messages_by_tokens(messages, estimate_tokens(messages[-2:]) + 20)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "current task"
    assert all(message.get("role") != "tool" for message in trimmed[1:])
