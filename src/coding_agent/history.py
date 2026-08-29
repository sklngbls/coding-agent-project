"""Utilities for keeping chat histories within a bounded context window."""

from __future__ import annotations

from typing import Any

Message = dict[str, Any]


def truncate_messages(messages: list[Message], limit: int) -> list[Message]:
    """Keep the newest complete message sequence within ``limit`` messages.

    Chat Completions requires every assistant tool call to be followed by its
    matching tool results. The suffix is therefore selected only at valid
    message boundaries instead of slicing arbitrary message indexes.
    """

    if limit < 1:
        raise ValueError("message limit must be at least 1")
    if len(messages) <= limit:
        return messages

    system: Message | None = None
    body = messages
    if messages and messages[0].get("role") == "system":
        system = messages[0]
        body = messages[1:]
    budget = limit - (1 if system is not None else 0)
    if budget <= 0:
        return [system] if system is not None else []

    # Pick the longest valid suffix so old context is discarded only as needed.
    for start in range(len(body)):
        candidate = body[start:]
        if len(candidate) <= budget and _is_complete_sequence(candidate):
            return ([system] if system is not None else []) + candidate

    # A single tool-call round can be larger than the configured budget. Keep
    # the newest user turn so the next request still contains the active task.
    for message in reversed(body):
        if message.get("role") == "user":
            return ([system] if system is not None else []) + [message]
    return [system] if system is not None else []


def _is_complete_sequence(messages: list[Message]) -> bool:
    """Return whether a suffix has no orphaned or incomplete tool messages."""

    if messages and messages[0].get("role") == "tool":
        return False
    if messages and messages[0].get("role") == "assistant" and messages[0].get(
        "tool_calls"
    ):
        # The assistant tool call's preceding user turn was truncated.
        return False

    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if pending:
            if role != "tool":
                return False
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or tool_call_id not in pending:
                return False
            pending.remove(tool_call_id)
            continue

        if role == "tool":
            return False
        if role != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        if not isinstance(tool_calls, list):
            return False
        call_ids: set[str] = set()
        for call in tool_calls:
            if not isinstance(call, dict):
                return False
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                return False
            call_ids.add(call_id)
        pending = call_ids
    return not pending
