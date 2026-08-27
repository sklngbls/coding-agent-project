"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from .agent import CodingAgent, ProgressEvent
from .config import ConfigurationError, Settings
from .llm import OpenAIChatModel
from .tools import LocalTools, ToolInputError, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Run a coding agent in a restricted local workspace.",
    )
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--command-timeout", type=float, default=20.0)
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
        local_tools = LocalTools(
            settings.workspace,
            command_timeout=settings.command_timeout,
            max_output_chars=settings.max_output_chars,
        )
        registry = ToolRegistry(local_tools.definitions())
        model = OpenAIChatModel(settings)
    except (ToolInputError, RuntimeError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2

    agent = CodingAgent(
        model,
        registry,
        workspace=settings.workspace,
        max_steps=settings.max_steps,
        on_progress=_print_progress,
    )
    result = agent.run(task)
    print("\n" + result.final_answer)
    return 0 if result.status == "completed" else 1


def _print_progress(event: ProgressEvent) -> None:
    if event.kind == "model":
        print(f"[step {event.step}] {event.message}...", file=sys.stderr)
        return
    outcome = "ok" if event.ok else "error"
    print(f"[step {event.step}] tool {event.message}: {outcome}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
