"""Model abstraction and OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from .config import Settings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam


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
            # The core keeps provider-neutral JSON-like messages. Cast only at this
            # SDK boundary to match OpenAI Python 2.x's generated parameter unions.
            sdk_messages = cast("Iterable[ChatCompletionMessageParam]", messages)
            sdk_tools = cast("Iterable[ChatCompletionToolUnionParam]", tools)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=sdk_messages,
                tools=sdk_tools,
            )
        except Exception as exc:
            safe_message = str(exc).replace(self._api_key, "[redacted]")
            raise ModelRequestError(safe_message) from exc

        if isinstance(response, str):
            normalized_response = response.strip().replace("\r", " ").replace("\n", " ")
            preview = self._redact(normalized_response)[:160]
            detail = f"; response preview: {preview!r}" if preview else ""
            raise ModelRequestError(
                "OpenAI-compatible endpoint returned plain text instead of a Chat "
                "Completion object"
                + detail
                + ". Check LLM_BASE_URL points to the provider API root (usually ending "
                "in /v1), LLM_MODEL is valid, and the gateway has Chat Completions "
                "compatibility enabled."
            )
        choices = getattr(response, "choices", None)
        if choices is None:
            response_type = type(response).__name__
            raise ModelRequestError(
                "OpenAI-compatible endpoint returned an unsupported response shape "
                f"({response_type}; expected a Chat Completion object with choices). "
                "Check LLM_BASE_URL, LLM_MODEL, and the provider's Chat Completions "
                "compatibility mode."
            )
        if not choices:
            raise ModelRequestError("Model returned no choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ModelRequestError(
                "Chat Completion response contained a choice without a message; "
                "check the provider's Chat Completions response format."
            )
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

    def _redact(self, value: str) -> str:
        """Remove the configured credential from any diagnostic text."""

        return value.replace(self._api_key, "[redacted]") if self._api_key else value
