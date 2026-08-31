from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent import cli
from coding_agent.llm import ModelResponse, ToolCall
from coding_agent.sessions import SessionStore
from coding_agent.workspaces import WorkspaceStore

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


def test_cli_workspace_is_optional() -> None:
    args = cli.build_parser().parse_args([])

    assert args.workspace is None
    assert args.yes is False


def test_cli_yes_flag_skips_operation_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    fake = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write_file",
                        arguments='{"path":"created.txt","content":"created"}',
                    )
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: fake)

    def fail_input(_prompt: str) -> str:
        raise AssertionError("--yes should not request confirmation")

    monkeypatch.setattr("builtins.input", fail_input)

    assert cli.main(["--yes", "--workspace", str(tmp_path), "Create a file"]) == 0
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created"


def test_cli_prompts_before_file_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
    fake = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write_file",
                        arguments='{"path":"created.txt","content":"created"}',
                    )
                ]
            ),
            ModelResponse(content="The write was not approved."),
        ]
    )
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: fake)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert cli.main(["--workspace", str(tmp_path), "Create a file"]) == 0
    captured = capsys.readouterr()
    assert "[approval required] write_file: created.txt" in captured.err
    assert "The write was not approved." in captured.out
    assert not (tmp_path / "created.txt").exists()


def test_xx_main_accepts_code_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        received.append(argv)
        return 0

    monkeypatch.setattr(cli, "main", fake_main)

    assert cli.xx_main(["code", "--workspace", "project"]) == 0
    assert received == [["--workspace", "project"]]


def test_banner_uses_figlet_style_ascii_title() -> None:
    banner = cli._render_banner()

    assert "XXCODE | Coding Agent" not in banner
    assert "-" * 44 in banner
    assert banner.count("\n") == 7
    assert "___  ______  ___" in banner


def test_cli_without_workspace_uses_recent_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    workspace = tmp_path / "recent-project"
    workspace.mkdir()
    WorkspaceStore().remember(workspace)
    fake = FakeModel([ModelResponse(content="Task complete.")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: fake)
    inputs = iter(["1", "0", "Do the task", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.main(["--select-session"])

    assert exit_code == 0
    recent = SessionStore(workspace).get_recent()
    assert recent is not None
    assert recent.title == "Do the task"


def test_cli_without_arguments_selects_workspace_then_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    workspace = tmp_path / "recent-project"
    workspace.mkdir()
    WorkspaceStore().remember(workspace)
    fake = FakeModel([ModelResponse(content="Task complete.")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: fake)
    inputs = iter(["1", "0", "Do the task", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.main([])

    assert exit_code == 0
    recent = SessionStore(workspace).get_recent()
    assert recent is not None
    assert recent.title == "Do the task"


def test_cli_without_workspace_accepts_a_new_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    workspace = tmp_path / "new-project"
    workspace.mkdir()
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "0" if prompt == "Workspace: " else str(workspace),
    )

    selected = cli._resolve_workspace(None)

    assert selected == workspace.resolve()


def test_cli_without_workspace_can_cancel_before_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")

    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "-" * 39 in output


def test_cli_rejects_removed_session_id_option() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--workspace", ".", "--session", "a" * 32, "second task"]
        )


def test_cli_rejects_removed_list_sessions_option() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--workspace", ".", "--list-sessions"])


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
    captured = capsys.readouterr()
    assert "continuing" not in captured.err.splitlines()[0]
    assert captured.out.count("Agent:") == 2
    assert captured.out.count("-" * 56) == 2


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


def test_cli_selects_numbered_session_and_enters_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_environment(monkeypatch, tmp_path)
    store = SessionStore(tmp_path)
    older = store.create([{"role": "user", "content": "older task"}], title="Older")
    store.save(older)
    newer = store.create(
        [
            {"role": "user", "content": "newer task"},
            {"role": "assistant", "content": "previous answer"},
        ],
        title="Newer",
    )
    store.save(newer)
    model = FakeModel([ModelResponse(content="continued")])
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: model)
    inputs = iter(["1", "follow-up task", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = cli.main(["--workspace", str(tmp_path), "--select-session"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Session selection" in captured.out
    assert "Updated:" in captured.out
    assert "Last user: newer task" in captured.out
    assert "--------------------------------" in captured.out
    assert "Last agent: previous answer" in captured.out
    assert "[previous session]" in captured.err
    assert "-" * 56 in captured.err
    status_position = captured.err.index("[session]")
    assert captured.err.index("-" * 56) < status_position
    assert "Last user: newer task" in captured.err
    assert "Last agent: previous answer" in captured.err
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
        cli.build_parser().parse_args(["--workspace", ".", "--continue", "--select-session"])
