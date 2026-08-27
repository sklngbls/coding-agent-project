from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.config import ConfigurationError, Settings


def test_settings_load_required_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    settings = Settings.from_environment(tmp_path, max_steps=4, command_timeout=3)

    assert settings.workspace == tmp_path.resolve()
    assert settings.api_key == "test-key"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.model == "test-model"
    assert settings.max_steps == 4
    assert settings.command_timeout == 3


def test_settings_report_all_missing_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError) as error:
        Settings.from_environment(".")

    message = str(error.value)
    assert "LLM_API_KEY" in message
    assert "LLM_BASE_URL" in message
    assert "LLM_MODEL" in message


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"command_timeout": 0}, "command_timeout"),
        ({"max_output_chars": 0}, "max_output_chars"),
    ],
)
def test_settings_validate_limits(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, int],
    expected: str,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with pytest.raises(ConfigurationError, match=expected):
        Settings.from_environment(".", **overrides)
