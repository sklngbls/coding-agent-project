from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.config import ConfigurationError, Settings, load_environment_file


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


def test_settings_loads_dotenv_when_environment_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "export LLM_API_KEY='file-key'\n"
        "LLM_BASE_URL=https://example.invalid/v1\n"
        'LLM_MODEL="file-model"\n',
        encoding="utf-8",
    )

    settings = Settings.from_environment(tmp_path)

    assert settings.api_key == "file-key"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.model == "file-model"


def test_environment_values_override_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=file-key\nLLM_BASE_URL=https://file.invalid/v1\nLLM_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "process-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://process.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "process-model")

    settings = Settings.from_environment(tmp_path)

    assert settings.api_key == "process-key"
    assert settings.base_url == "https://process.invalid/v1"
    assert settings.model == "process-model"


def test_dotenv_parser_ignores_comments_and_supports_quoted_values(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nKEY=value # inline comment\nOTHER='a # b'\n",
        encoding="utf-8",
    )

    assert load_environment_file(path) == {"KEY": "value", "OTHER": "a # b"}


def test_settings_report_all_missing_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
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
        ({"max_history_tokens": 0}, "max_history_tokens"),
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
