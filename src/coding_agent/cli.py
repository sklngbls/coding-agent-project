"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from .agent import CodingAgent, ProgressEvent
from .config import ConfigurationError, Settings
from .llm import OpenAIChatModel
from .sessions import Session, SessionError, SessionStore
from .tools import LocalTools, ToolInputError, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Run a coding agent in a restricted local workspace.",
    )
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout", type=float, default=20.0)
    session_modes = parser.add_mutually_exclusive_group()
    session_modes.add_argument("--session", metavar="SESSION_ID", help="Continue a session")
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
    parser.add_argument("task", nargs="?", help="Natural-language programming task")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_environment(
            args.workspace,
            max_steps=args.max_steps,
            command_timeout=args.command_timeout,
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        session_store = SessionStore(settings.workspace, api_key=settings.api_key)
        local_tools = LocalTools(
            settings.workspace,
            command_timeout=settings.command_timeout,
            max_output_chars=settings.max_output_chars,
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
        session, continuing = _select_session(session_store, args)
    except SessionError as exc:
        print(f"Session error: {exc}", file=sys.stderr)
        return 2
    _print_session_status(session, continuing)
    return _run_and_save(agent, session_store, session, task, continuing, settings.api_key)


def _select_session(session_store: SessionStore, args: argparse.Namespace) -> tuple[Session, bool]:
    if args.session:
        return session_store.load(args.session), True
    if args.continue_session:
        recent = session_store.get_recent()
        if recent is not None:
            return recent, True
        print("[session] no previous session; creating a new session", file=sys.stderr)
        return session_store.create(), False
    return session_store.create(), False


def _interactive_loop(agent: CodingAgent, session_store: SessionStore, api_key: str) -> int:
    try:
        session, continuing = _select_session(
            session_store,
            argparse.Namespace(session=None, continue_session=True, new_session=False),
        )
    except SessionError as exc:
        print(f"Session error: {exc}", file=sys.stderr)
        return 2
    _print_session_status(session, continuing)
    exit_code = 0
    while True:
        try:
            task = input("Task (blank/exit/quit to stop): ").strip()
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
            continuing=True,
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
    result = agent.run(task, history=session.messages if continuing else None)
    session.messages = result.messages
    print("\n" + _redact(result.final_answer, api_key))
    try:
        session_store.save(session)
    except SessionError as exc:
        print(f"Session save error: {exc}", file=sys.stderr)
        return 2
    return 0 if result.status == "completed" else 1


def _print_session_status(session: Session, continuing: bool) -> None:
    state = "continuing" if continuing else "new"
    print(f"[session {session.session_id}] {state}", file=sys.stderr)


def _redact(value: str, api_key: str) -> str:
    return value.replace(api_key, "[redacted]") if api_key else value


def _print_progress(event: ProgressEvent) -> None:
    if event.kind == "model":
        print(f"[step {event.step}] {event.message}...", file=sys.stderr)
        return
    outcome = "ok" if event.ok else "error"
    print(f"[step {event.step}] tool {event.message}: {outcome}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
