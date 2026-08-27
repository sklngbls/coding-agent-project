from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from coding_agent.config import Settings
from coding_agent.llm import ModelRequestError, OpenAIChatModel


class FakeCompletions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    completions: FakeCompletions,
    *,
    init_error: Exception | None = None,
) -> type:
    module = ModuleType("openai")

    class FakeOpenAI:
        last_init: dict[str, str] | None = None

        def __init__(self, *, api_key: str, base_url: str) -> None:
            if init_error is not None:
                raise init_error
            type(self).last_init = {"api_key": api_key, "base_url": base_url}
            self.chat = SimpleNamespace(completions=completions)

    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return FakeOpenAI


def settings(tmp_path: Path, api_key: str = "test-secret") -> Settings:
    return Settings(
        workspace=tmp_path,
        api_key=api_key,
        base_url="https://example.invalid/v1",
        model="fake-model",
    )


def test_adapter_normalizes_multiple_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="working",
                    tool_calls=[
                        SimpleNamespace(
                            id="one",
                            function=SimpleNamespace(name="read_file", arguments='{"path":"a"}'),
                        ),
                        SimpleNamespace(
                            id="two",
                            function=SimpleNamespace(name="list_files", arguments="{}"),
                        ),
                    ],
                )
            )
        ]
    )
    completions = FakeCompletions(response=response)
    fake_client = install_fake_openai(monkeypatch, completions)

    adapter = OpenAIChatModel(settings(tmp_path))
    result = adapter.complete([{"role": "user", "content": "task"}], [{"type": "function"}])

    assert result.content == "working"
    assert [item.name for item in result.tool_calls] == ["read_file", "list_files"]
    assert completions.kwargs is not None
    assert completions.kwargs["model"] == "fake-model"
    assert fake_client.last_init == {
        "api_key": "test-secret",
        "base_url": "https://example.invalid/v1",
    }


def test_adapter_redacts_api_key_from_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "actual-secret-value"
    completions = FakeCompletions(error=RuntimeError(f"request rejected for {secret}"))
    install_fake_openai(monkeypatch, completions)
    adapter = OpenAIChatModel(settings(tmp_path, api_key=secret))

    with pytest.raises(ModelRequestError) as error:
        adapter.complete([], [])

    assert secret not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_adapter_redacts_api_key_from_initialization_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "initialization-secret"
    install_fake_openai(
        monkeypatch,
        FakeCompletions(),
        init_error=RuntimeError(f"invalid client value {secret}"),
    )
    with pytest.raises(ModelRequestError) as error:
        OpenAIChatModel(settings(tmp_path, api_key=secret))
    assert secret not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_adapter_rejects_empty_choices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions(response=SimpleNamespace(choices=[]))
    install_fake_openai(monkeypatch, completions)
    adapter = OpenAIChatModel(settings(tmp_path))
    with pytest.raises(ModelRequestError, match="no choices"):
        adapter.complete([], [])
