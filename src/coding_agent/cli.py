"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from .agent import CodingAgent, ProgressEvent
from .config import ConfigurationError, Settings
from .llm import OpenAIChatModel
from .sessions import (
    UNTITLED_SESSION_TITLE,
    Session,
    SessionError,
    SessionStore,
    make_session_title,
)
from .tools import LocalTools, ToolInputError, ToolRegistry
from .workspaces import WorkspaceError, WorkspaceStore

_BANNER_ART = r"""                                  .___
___  ______  ___   ____  ____   __| _/____
\  \/  /\  \/  / _/ ___\/  _ \ / __ |/ __ \
 >    <  >    <  \  \__(  <_> ) /_/ \  ___/
/__/\_ \/__/\_ \  \___  >____/\____ |\___  >
      \/      \/      \/           \/    \/"""
_BANNER_LINES = tuple(_BANNER_ART.splitlines())
_SESSION_PREVIEW_CHARS = 72
_PREVIEW_DIVIDER = "-" * 32
_TURN_DIVIDER = "-" * 56


def _render_banner() -> str:
    """Render the built-in FIGlet-style startup title."""

    rows = list(_BANNER_LINES)
    rows.extend(("", "-" * max(len(row) for row in rows)))
    return "\n".join(rows)


def _print_banner() -> None:
    print(_render_banner())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Run a coding agent in a restricted local workspace.",
    )
    parser.add_argument(
        "--workspace",
        help="Workspace directory; omit to choose one after startup",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout", type=float, default=20.0)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts for file changes and commands",
    )
    session_modes = parser.add_mutually_exclusive_group()
    session_modes.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Continue the most recent workspace session",
    )
    session_modes.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new session explicitly",
    )
    session_modes.add_argument(
        "--select-session",
        action="store_true",
        help="Choose a session from an interactive numbered menu",
    )
    session_modes.add_argument(
        "--rename-session",
        action="store_true",
        help="Rename an existing session from an interactive menu",
    )
    session_modes.add_argument(
        "--delete-session",
        action="store_true",
        help="Delete an existing session from an interactive menu",
    )
    parser.add_argument("--title", help="Title for a new or renamed session")
    parser.add_argument("task", nargs="?", help="Natural-language programming task")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_error = _validate_session_arguments(args)
    if validation_error is not None:
        print(f"Argument error: {validation_error}", file=sys.stderr)
        return 2
    _print_banner()
    try:
        workspace = _resolve_workspace(args.workspace)
    except WorkspaceError as exc:
        print(f"Workspace error: {exc}", file=sys.stderr)
        return 2
    if workspace is None:
        return 0
    if args.rename_session:
        _remember_workspace(workspace)
        try:
            return _rename_session_interactively(str(workspace), args.title)
        except (OSError, SessionError, ValueError) as exc:
            print(f"Session error: {exc}", file=sys.stderr)
            return 2
    if args.delete_session:
        _remember_workspace(workspace)
        try:
            return _delete_session_interactively(str(workspace), skip_confirmation=args.yes)
        except (OSError, SessionError, ValueError) as exc:
            print(f"Session error: {exc}", file=sys.stderr)
            return 2
    session_choice: tuple[Literal["existing", "new"], str | None] | None = None
    choose_session = args.select_session or (
        args.workspace is None
        and args.task is None
        and not args.continue_session
        and not args.new_session
        and not args.rename_session
        and not args.delete_session
        and args.title is None
    )
    if choose_session:
        try:
            selected = _prompt_session_choice(str(workspace), skip_confirmation=args.yes)
        except (OSError, SessionError, ValueError) as exc:
            print(f"Session error: {exc}", file=sys.stderr)
            return 2
        if selected is None:
            return 0
        session_choice = selected

    try:
        settings = Settings.from_environment(
            workspace,
            max_steps=args.max_steps,
            command_timeout=args.command_timeout,
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    _remember_workspace(settings.workspace)

    try:
        session_store = SessionStore(settings.workspace, api_key=settings.api_key)
        confirm_action = (
            None
            if args.yes
            else lambda operation, target, preview: _confirm_action(
                operation,
                target,
                preview,
                settings.api_key,
            )
        )
        local_tools = LocalTools(
            settings.workspace,
            command_timeout=settings.command_timeout,
            max_output_chars=settings.max_output_chars,
            confirm_action=confirm_action,
        )
        registry = ToolRegistry(local_tools.definitions())
        model = OpenAIChatModel(settings)
    except (SessionError, ToolInputError, RuntimeError, ValueError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2

    agent = CodingAgent(
        model,
        registry,
        workspace=settings.workspace,
        max_steps=settings.max_steps,
        on_progress=_print_progress,
    )

    if session_choice is not None:
        action, session_id = session_choice
        try:
            if action == "existing" and session_id is not None:
                selected_session = session_store.load(session_id)
                continuing = True
            else:
                selected_session = session_store.create()
                continuing = False
        except SessionError as exc:
            print(f"Session error: {exc}", file=sys.stderr)
            return 2
        if args.task is None:
            return _conversation_loop(
                agent,
                session_store,
                selected_session,
                continuing,
                settings.api_key,
            )
        _print_session_status(selected_session, continuing)
        return _run_and_save(
            agent,
            session_store,
            selected_session,
            args.task,
            continuing,
            settings.api_key,
        )

    if args.continue_session and args.task is None:
        return _interactive_loop(agent, session_store, settings.api_key)

    task = args.task
    if task is None:
        try:
            task = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo task provided.", file=sys.stderr)
            return 2
    if not task:
        print("Task must not be empty.", file=sys.stderr)
        return 2

    try:
        session, continuing = _select_session(session_store, args, task)
    except SessionError as exc:
        print(f"Session error: {exc}", file=sys.stderr)
        return 2
    _print_session_status(session, continuing)
    return _run_and_save(agent, session_store, session, task, continuing, settings.api_key)


def xx_main(argv: list[str] | None = None) -> int:
    """Launch the agent through the short xx code command alias."""

    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args and command_args[0].casefold() == "code":
        command_args.pop(0)
    return main(command_args)


def _validate_session_arguments(args: argparse.Namespace) -> str | None:
    if args.select_session and args.title is not None:
        return "--title cannot be used with --select-session"
    if args.title is not None and not args.title.strip():
        return "--title must not be empty"
    if args.title is not None and args.continue_session:
        return "--title can only be used when starting a new session"
    if args.rename_session and args.task is not None:
        return "--rename-session does not accept a task; use --title for the new name"
    if args.delete_session and args.title is not None:
        return "--title cannot be used with --delete-session"
    if args.delete_session and args.task is not None:
        return "--delete-session does not accept a task"
    return None


def _resolve_workspace(workspace_value: str | None) -> Path | None:
    """Use an explicit workspace or ask the user to choose one."""

    if workspace_value is not None:
        workspace = _validate_workspace_path(workspace_value)
        return workspace
    try:
        selected = _prompt_workspace_choice()
    except WorkspaceError as exc:
        print(f"Workspace error: {exc}", file=sys.stderr)
        return None
    if selected is None:
        return None
    return selected


def _prompt_workspace_choice() -> Path | None:
    """Prompt for a recent workspace or a new path."""

    try:
        records = WorkspaceStore().list()
    except WorkspaceError as exc:
        print(f"Could not read recent workspaces: {exc}", file=sys.stderr)
        records = []

    _print_menu_header("Workspace selection")
    for index, record in enumerate(records, start=1):
        print(f"[{index}] {record.path}")
    if not records:
        print("No recent workspaces.")
    print()
    print("[0] Enter a workspace path")
    print("[q] Cancel")

    while True:
        print()
        try:
            value = input("Workspace: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("")
            return None
        if not value or value in {"q", "quit", "exit"}:
            return None
        if value in {"0", "n", "new"}:
            try:
                raw_path = input("Workspace path (Enter for current directory): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                return None
            if raw_path.casefold() in {"q", "quit", "exit"}:
                return None
            try:
                return _validate_workspace_path(raw_path or Path.cwd())
            except WorkspaceError as exc:
                print(f"Workspace error: {exc}")
                continue
        try:
            index = int(value)
        except ValueError:
            index = -1
        if 1 <= index <= len(records):
            try:
                return _validate_workspace_path(records[index - 1].path)
            except WorkspaceError as exc:
                print(f"Workspace error: {exc}")
                continue
        print(f"Invalid selection. Enter 0-{len(records)} or q to cancel.")


def _validate_workspace_path(value: str | Path) -> Path:
    workspace = Path(value).expanduser().resolve()
    if not workspace.exists():
        raise WorkspaceError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise WorkspaceError(f"Workspace is not a directory: {workspace}")
    return workspace


def _remember_workspace(workspace: Path) -> None:
    try:
        WorkspaceStore().remember(workspace)
    except WorkspaceError as exc:
        print(f"Warning: could not save recent workspace: {exc}", file=sys.stderr)


def _prompt_session_choice(
    workspace_value: str,
    *,
    skip_confirmation: bool = False,
) -> tuple[Literal["existing", "new"], str | None] | None:
    store = _open_session_store(workspace_value)

    while True:
        sessions = store.list()
        _print_session_selection_menu(sessions)
        print()
        try:
            value = input("Selection: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("")
            return None
        if not value or value in {"q", "quit", "exit"}:
            return None
        if value in {"0", "n", "new"}:
            return "new", None
        if value in {"r", "rename"}:
            if not sessions:
                print("No existing sessions to rename.")
                continue
            _prompt_session_rename(store)
            continue
        if value in {"d", "delete", "remove"}:
            if not sessions:
                print("No existing sessions to delete.")
                continue
            _prompt_session_delete(store, skip_confirmation=skip_confirmation)
            continue
        try:
            index = int(value)
        except ValueError:
            index = -1
        if 1 <= index <= len(sessions):
            return "existing", sessions[index - 1].session_id
        print(f"Invalid selection. Enter 0-{len(sessions)} or q to cancel.")


def _print_session_selection_menu(sessions: list[Session]) -> None:
    _print_menu_header("Session selection")
    for index, session in enumerate(sessions, start=1):
        print(f"[{index}] {session.title}")
        print(f"    Updated: {_format_timestamp(session.updated_at)}")
        previews = _session_history_preview(session)
        if previews:
            for preview_index, (label, content) in enumerate(previews):
                if preview_index:
                    print(f"    {_PREVIEW_DIVIDER}")
                print(f"    Last {label}: {content}")
        else:
            print("    No conversation yet")
        print()
    if not sessions:
        print("No existing sessions for this workspace.")
    print()
    print("[0] Start a new session")
    print("[r] Rename a session")
    print("[d] Delete a session")
    print("[q] Cancel")


def _rename_session_interactively(workspace_value: str, title: str | None) -> int:
    store = _open_session_store(workspace_value)
    if _prompt_session_rename(store, title=title):
        return 0
    return 0


def _delete_session_interactively(workspace_value: str, *, skip_confirmation: bool) -> int:
    store = _open_session_store(workspace_value)
    _prompt_session_delete(store, skip_confirmation=skip_confirmation)
    return 0


def _prompt_session_rename(store: SessionStore, *, title: str | None = None) -> bool:
    sessions = store.list()
    _print_menu_header("Rename session")
    for index, session in enumerate(sessions, start=1):
        print(f"[{index}] {session.title}")
    if not sessions:
        print("No existing sessions for this workspace.")
        return False
    print("[q] Cancel")

    while True:
        print()
        try:
            value = input("Session to rename: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("")
            return False
        if not value or value in {"q", "quit", "exit"}:
            return False
        try:
            index = int(value)
        except ValueError:
            index = -1
        if 1 <= index <= len(sessions):
            selected = sessions[index - 1]
            break
        print(f"Invalid selection. Enter 1-{len(sessions)} or q to cancel.")

    provided_title = title is not None
    new_title = title
    if new_title is None:
        try:
            new_title = input("New title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return False
    if not new_title or (
        not provided_title and new_title.casefold() in {"q", "quit", "exit"}
    ):
        print("Rename cancelled.")
        return False
    previous_title = selected.title
    renamed = store.rename(selected.session_id, new_title)
    print(f"Session renamed: {previous_title} -> {renamed.title}")
    return True


def _prompt_session_delete(store: SessionStore, *, skip_confirmation: bool = False) -> bool:
    sessions = store.list()
    _print_menu_header("Delete session")
    for index, session in enumerate(sessions, start=1):
        print(f"[{index}] {session.title}")
    if not sessions:
        print("No existing sessions for this workspace.")
        return False
    print("[q] Cancel")

    while True:
        print()
        try:
            value = input("Session to delete: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("")
            return False
        if not value or value in {"q", "quit", "exit"}:
            return False
        try:
            index = int(value)
        except ValueError:
            index = -1
        if 1 <= index <= len(sessions):
            selected = sessions[index - 1]
            break
        print(f"Invalid selection. Enter 1-{len(sessions)} or q to cancel.")

    if not skip_confirmation:
        try:
            answer = input(
                f'Delete session "{selected.title}" and its entire conversation? [y/N]: '
            ).strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print("")
            return False
        if answer not in {"y", "yes"}:
            print("Deletion cancelled.")
            return False
    deleted = store.delete(selected.session_id)
    print(f"Session deleted: {deleted.title}")
    return True


def _print_menu_header(title: str) -> None:
    print()
    print("=" * 56)
    print(title)
    print("=" * 56)


def _session_history_preview(session: Session) -> tuple[tuple[str, str], ...]:
    """Return compact previews of the latest user and agent messages."""

    latest: dict[str, str] = {}
    for message in reversed(session.messages):
        role = message.get("role")
        if role not in {"user", "assistant"} or role in latest:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = " ".join(content.split())
        if not normalized:
            continue
        latest[role] = _truncate_preview(normalized)
        if len(latest) == 2:
            break

    previews: list[tuple[str, str]] = []
    if "user" in latest:
        previews.append(("user", latest["user"]))
    if "assistant" in latest:
        previews.append(("agent", latest["assistant"]))
    return tuple(previews)


def _truncate_preview(value: str, limit: int = _SESSION_PREVIEW_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _open_session_store(workspace_value: str) -> SessionStore:
    workspace = Path(workspace_value).expanduser().resolve()
    if not workspace.exists():
        raise SessionError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise SessionError(f"Workspace is not a directory: {workspace}")
    return SessionStore(workspace)


def _format_timestamp(value: str) -> str:
    return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")


def _select_session(
    session_store: SessionStore,
    args: argparse.Namespace,
    task: str | None = None,
) -> tuple[Session, bool]:
    if args.continue_session:
        recent = session_store.get_recent()
        if recent is not None:
            return recent, True
        print("[session] no previous session; creating a new session", file=sys.stderr)
        return session_store.create(), False
    return session_store.create(title=args.title or task), False


def _interactive_loop(agent: CodingAgent, session_store: SessionStore, api_key: str) -> int:
    try:
        session, continuing = _select_session(
            session_store,
            argparse.Namespace(
                continue_session=True,
                new_session=False,
                title=None,
            ),
        )
    except SessionError as exc:
        print(f"Session error: {exc}", file=sys.stderr)
        return 2
    return _conversation_loop(agent, session_store, session, continuing, api_key)


def _conversation_loop(
    agent: CodingAgent,
    session_store: SessionStore,
    session: Session,
    continuing: bool,
    api_key: str,
) -> int:
    _print_session_status(session, continuing)
    exit_code = 0
    while True:
        try:
            task = input("User task (blank/exit/quit to stop): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            break
        if not task or task.casefold() in {"exit", "quit"}:
            break
        result_code = _run_and_save(
            agent,
            session_store,
            session,
            task,
            continuing=continuing,
            api_key=api_key,
        )
        if result_code != 0:
            exit_code = result_code
        continuing = True
    return exit_code


def _run_and_save(
    agent: CodingAgent,
    session_store: SessionStore,
    session: Session,
    task: str,
    continuing: bool,
    api_key: str,
) -> int:
    if session.title == UNTITLED_SESSION_TITLE and not any(
        message.get("role") == "user" for message in session.messages
    ):
        session.title = make_session_title(task)
    result = agent.run(task, history=session.messages if continuing else None)
    session.messages = result.messages
    print("\nAgent:\n" + _redact(result.final_answer, api_key))
    print(_TURN_DIVIDER)
    try:
        session_store.save(session)
    except SessionError as exc:
        print(f"Session save error: {exc}", file=sys.stderr)
        return 2
    return 0 if result.status == "completed" else 1


def _print_session_status(session: Session, continuing: bool) -> None:
    state = "continuing" if continuing else "new"
    print(_TURN_DIVIDER, file=sys.stderr)
    print(f"[session] {state}: {session.title}", file=sys.stderr)
    if not continuing:
        return
    previews = _session_history_preview(session)
    if not previews:
        return
    print(_TURN_DIVIDER, file=sys.stderr)
    print("[previous session]", file=sys.stderr)
    for preview_index, (label, content) in enumerate(previews):
        if preview_index:
            print(f"    {_PREVIEW_DIVIDER}", file=sys.stderr)
        print(f"    Last {label}: {content}", file=sys.stderr)


def _redact(value: str, api_key: str) -> str:
    return value.replace(api_key, "[redacted]") if api_key else value


def _confirm_action(operation: str, target: str, preview: str, api_key: str) -> bool:
    print(_TURN_DIVIDER, file=sys.stderr)
    print(f"[approval required] {operation}: {target}", file=sys.stderr)
    print(_redact(preview, api_key), file=sys.stderr)
    print(_TURN_DIVIDER, file=sys.stderr)
    try:
        answer = input("Allow this operation? [y/N]: ").strip().casefold()
    except (EOFError, KeyboardInterrupt):
        print("", file=sys.stderr)
        return False
    return answer in {"y", "yes"}


def _print_progress(event: ProgressEvent) -> None:
    if event.kind == "model":
        print(f"[step {event.step}] {event.message}...", file=sys.stderr)
        return
    outcome = "ok" if event.ok else "error"
    print(f"[step {event.step}] tool {event.message}: {outcome}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
