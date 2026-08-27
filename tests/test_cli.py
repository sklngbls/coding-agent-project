from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent import cli
from coding_agent.llm import ModelResponse

from .test_agent import FakeModel


def configure_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def test_cli_runs_agent_with_fake_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch)
    fake = FakeModel([ModelResponse(content="Task complete.")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: fake)

    exit_code = cli.main(["--workspace", str(tmp_path), "Do the task"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Task complete." in captured.out
    assert "[step 1]" in captured.err


def test_cli_reports_missing_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    exit_code = cli.main(["--workspace", str(tmp_path), "Do the task"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error" in captured.err
    assert "LLM_API_KEY" in captured.err


def test_cli_reports_missing_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch)
    exit_code = cli.main(["--workspace", str(tmp_path / "missing"), "Do the task"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Workspace does not exist" in captured.err
