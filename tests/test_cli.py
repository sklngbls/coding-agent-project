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
    assert "Do the task" in captured.err


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


def test_cli_rejects_removed_session_id_option() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--workspace", ".", "--session", "a" * 32, "second task"]
        )


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


def test_cli_manual_title_is_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    fake = FakeModel([ModelResponse(content="done")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: fake)

    exit_code = cli.main(
        ["--workspace", str(tmp_path), "--title", "Session management", "Do the task"]
    )

    assert exit_code == 0
    recent = SessionStore(tmp_path).get_recent()
    assert recent is not None
    assert recent.title == "Session management"


def test_cli_lists_sessions_without_model_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
    store = SessionStore(tmp_path)
    session = store.create([{"role": "user", "content": "Saved conversation"}])
    store.save(session)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def fail_model(_settings: object) -> None:
        raise AssertionError("list mode must not initialize the model")

    monkeypatch.setattr(cli, "OpenAIChatModel", fail_model)
    exit_code = cli.main(["--workspace", str(tmp_path), "--list-sessions"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "UPDATED | TITLE | SESSION ID" in captured.out
    assert "Saved conversation" in captured.out
    assert session.session_id in captured.out


def test_cli_list_reports_no_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path.parent / f"{tmp_path.name}-localappdata"))

    exit_code = cli.main(["--workspace", str(tmp_path), "--list-sessions"])

    assert exit_code == 0
    assert "No sessions found" in capsys.readouterr().out


def test_cli_selects_numbered_session_and_enters_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
    store = SessionStore(tmp_path)
    older = store.create([{"role": "user", "content": "older task"}], title="Older")
    store.save(older)
    newer = store.create([{"role": "user", "content": "newer task"}], title="Newer")
    store.save(newer)
    model = FakeModel([ModelResponse(content="continued")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: model)
    inputs = iter(["1", "follow-up task", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.main(["--workspace", str(tmp_path), "--select-session"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.index("[1] Newer") < captured.out.index("[2] Older")
    loaded = store.load(newer.session_id)
    user_tasks = [
        message["content"] for message in loaded.messages if message.get("role") == "user"
    ]
    assert user_tasks == ["newer task", "follow-up task"]


def test_cli_selector_can_create_new_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    model = FakeModel([ModelResponse(content="created")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: model)
    inputs = iter(["0", "first task in new session", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.main(["--workspace", str(tmp_path), "--select-session"])

    assert exit_code == 0
    recent = SessionStore(tmp_path).get_recent()
    assert recent is not None
    assert recent.title == "first task in new session"


def test_cli_selector_retries_invalid_input_and_cancels_before_model_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path.parent / f"{tmp_path.name}-localappdata"))
    store = SessionStore(tmp_path)
    store.save(store.create(title="Existing"))
    inputs = iter(["invalid", "9", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def fail_model(_settings: object) -> None:
        raise AssertionError("cancelled selection must not initialize the model")

    monkeypatch.setattr(cli, "OpenAIChatModel", fail_model)
    exit_code = cli.main(["--workspace", str(tmp_path), "--select-session"])

    assert exit_code == 0
    assert capsys.readouterr().out.count("Invalid selection") == 2


def test_cli_selector_eof_cancels_without_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path.parent / f"{tmp_path.name}-localappdata"))

    def end_input(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)

    assert cli.main(["--workspace", str(tmp_path), "--select-session"]) == 0


def test_cli_selector_with_task_runs_once_in_selected_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    store = SessionStore(tmp_path)
    session = store.create([{"role": "user", "content": "original"}], title="Selected")
    store.save(session)
    model = FakeModel([ModelResponse(content="done")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: model)
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    exit_code = cli.main(
        ["--workspace", str(tmp_path), "--select-session", "one follow-up task"]
    )

    assert exit_code == 0
    loaded = store.load(session.session_id)
    assert loaded.messages[-2]["content"] == "one follow-up task"
    assert loaded.messages[-1]["content"] == "done"


def test_cli_rejects_title_when_continuing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        ["--workspace", str(tmp_path), "--continue", "--title", "invalid", "task"]
    )

    assert exit_code == 2
    assert "--title can only be used" in capsys.readouterr().err


def test_cli_session_mode_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--workspace", ".", "--continue", "--new-session", "task"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--workspace", ".", "--continue", "--list-sessions"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--workspace", ".", "--continue", "--select-session"])
