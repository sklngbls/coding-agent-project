from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent import cli
from coding_agent.llm import ModelResponse
from coding_agent.sessions import SessionStore

from .test_agent import FakeModel


def configure_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path.parent / f"{tmp_path.name}-localappdata"))


def test_cli_runs_agent_with_fake_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
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
    configure_environment(monkeypatch, tmp_path)
    exit_code = cli.main(["--workspace", str(tmp_path / "missing"), "Do the task"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Workspace does not exist" in captured.err


def test_cli_session_option_continues_history_without_duplicate_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
    real_store = cli.SessionStore

    def make_store(workspace: Path, *, api_key: str | None = None) -> SessionStore:
        return real_store(
            workspace,
            storage_root=tmp_path.parent / f"{tmp_path.name}-sessions",
            api_key=api_key,
        )

    monkeypatch.setattr(cli, "SessionStore", make_store)
    first_model = FakeModel([ModelResponse(content="first")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: first_model)
    assert cli.main(["--workspace", str(tmp_path), "first task"]) == 0
    session_id = make_store(tmp_path).get_recent()
    assert session_id is not None

    second_model = FakeModel([ModelResponse(content="second")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: second_model)
    exit_code = cli.main(
        ["--workspace", str(tmp_path), "--session", session_id.session_id, "second task"]
    )

    assert exit_code == 0
    sent = second_model.calls[0][0]
    assert sum(message["role"] == "system" for message in sent) == 1
    assert sent[-1]["content"] == "second task"
    assert "[session" in capsys.readouterr().err


def test_cli_continue_without_task_accepts_multiple_tasks_and_saves_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
    real_store = cli.SessionStore

    def make_store(workspace: Path, *, api_key: str | None = None) -> SessionStore:
        return real_store(
            workspace,
            storage_root=tmp_path.parent / f"{tmp_path.name}-sessions",
            api_key=api_key,
        )

    monkeypatch.setattr(cli, "SessionStore", make_store)
    model = FakeModel([ModelResponse(content="one"), ModelResponse(content="two")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: model)
    inputs = iter(["first task", "second task", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.main(["--workspace", str(tmp_path), "--continue"])

    assert exit_code == 0
    recent = make_store(tmp_path).get_recent()
    assert recent is not None
    user_tasks = [
        message["content"] for message in recent.messages if message.get("role") == "user"
    ]
    assert user_tasks == ["first task", "second task"]
    assert "continuing" not in capsys.readouterr().err.splitlines()[0]


def test_cli_default_tasks_are_separate_and_continue_uses_recent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    real_store = cli.SessionStore

    def make_store(workspace: Path, *, api_key: str | None = None) -> SessionStore:
        return real_store(
            workspace,
            storage_root=tmp_path.parent / f"{tmp_path.name}-sessions",
            api_key=api_key,
        )

    monkeypatch.setattr(cli, "SessionStore", make_store)
    models = [
        FakeModel([ModelResponse(content="first answer")]),
        FakeModel([ModelResponse(content="second answer")]),
        FakeModel([ModelResponse(content="continued answer")]),
    ]
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: models.pop(0))

    assert cli.main(["--workspace", str(tmp_path), "first task"]) == 0
    assert cli.main(["--workspace", str(tmp_path), "second task"]) == 0
    assert len(make_store(tmp_path).list()) == 2
    assert cli.main(["--workspace", str(tmp_path), "--continue", "third task"]) == 0

    sessions = make_store(tmp_path).list()
    assert len(sessions) == 2
    recent_user_tasks = [
        message["content"] for message in sessions[0].messages if message.get("role") == "user"
    ]
    assert recent_user_tasks == ["second task", "third task"]


def test_cli_session_mode_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--workspace", ".", "--continue", "--new-session", "task"])
