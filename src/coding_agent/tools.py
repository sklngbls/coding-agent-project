"""Local tools exposed to the model.

File paths and command working directories are confined to the configured workspace.
Commands use structured argument vectors and never invoke a shell.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolInputError(ValueError):
    """Raised when a tool receives invalid or unsafe input."""


@dataclass(frozen=True)
class ToolResult:
    """Structured result returned to the model for every tool invocation."""

    ok: bool
    message: str
    data: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return json.dumps(payload, ensure_ascii=False)


ToolHandler = Callable[[dict[str, Any]], ToolResult]
ConfirmationCallback = Callable[[str, str, str], bool]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Workspace:
    """Resolve model-provided paths without allowing workspace escape."""

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).expanduser()
        try:
            self.root = requested.resolve(strict=True)
        except OSError as exc:
            raise ToolInputError(f"Workspace does not exist: {requested}") from exc
        if not self.root.is_dir():
            raise ToolInputError(f"Workspace is not a directory: {requested}")

    def resolve(self, relative_path: str, *, must_exist: bool = False) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ToolInputError("path must be a non-empty string")
        supplied = Path(relative_path)
        if supplied.is_absolute() or supplied.drive or ".." in supplied.parts:
            raise ToolInputError("path must be relative and must not contain '..'")
        try:
            resolved = (self.root / supplied).resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise ToolInputError(f"Path does not exist: {relative_path}") from exc
        except OSError as exc:
            raise ToolInputError(f"Cannot resolve path {relative_path!r}: {exc}") from exc
        if not resolved.is_relative_to(self.root):
            raise ToolInputError("path escapes the configured workspace")
        return resolved

    def display_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


class LocalTools:
    """Implementation of the model's local file and process tools."""

    _PRUNED_DIRECTORIES = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "node_modules",
    }

    def __init__(
        self,
        workspace: str | Path,
        *,
        command_timeout: float = 20.0,
        max_output_chars: int = 20_000,
        max_list_entries: int = 1_000,
        confirm_action: ConfirmationCallback | None = None,
    ) -> None:
        if command_timeout <= 0:
            raise ValueError("command_timeout must be greater than 0")
        if max_output_chars < 1 or max_list_entries < 1:
            raise ValueError("output and listing limits must be positive")
        self.workspace = Workspace(workspace)
        self.command_timeout = command_timeout
        self.max_output_chars = max_output_chars
        self.max_list_entries = max_list_entries
        self.confirm_action = confirm_action
        self._checkpoint_info: dict[str, Any] | None = None

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="list_files",
                description=(
                    "List files and directories under a relative workspace path. "
                    "Directory symlinks are listed but never followed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path; default '.'"},
                        "recursive": {"type": "boolean", "default": True},
                    },
                    "additionalProperties": False,
                },
                handler=self.list_files,
            ),
            ToolDefinition(
                name="read_file",
                description="Read a UTF-8 text file from the workspace.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self.read_file,
            ),
            ToolDefinition(
                name="write_file",
                description=(
                    "Write complete UTF-8 text content to a workspace file, replacing it "
                    "if present. "
                    "Missing parent directories are created."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self.write_file,
            ),
            ToolDefinition(
                name="replace_text",
                description=(
                    "Replace one exact text occurrence in a UTF-8 workspace file. "
                    "Fails if the old text is absent or occurs more than once."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
                handler=self.replace_text,
            ),
            ToolDefinition(
                name="run_command",
                description=(
                    "Run a command in the workspace without a shell. Pass each argument "
                    "separately. "
                    "The working directory must be relative to the workspace."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "cwd": {"type": "string", "description": "Relative directory; default '.'"},
                        "timeout": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "description": "Optional timeout capped by the configured maximum",
                        },
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
                handler=self.run_command,
            ),
        ]

    def list_files(self, arguments: dict[str, Any]) -> ToolResult:
        relative = _optional_string(arguments, "path", ".")
        recursive = _optional_bool(arguments, "recursive", True)
        base = self.workspace.resolve(relative, must_exist=True)
        if not base.is_dir():
            raise ToolInputError(f"Not a directory: {relative}")

        entries: list[str] = []
        truncated = False

        def add_entry(path: Path) -> bool:
            nonlocal truncated
            if len(entries) >= self.max_list_entries:
                truncated = True
                return False
            suffix = "/" if path.is_dir() and not path.is_symlink() else ""
            if path.is_symlink():
                suffix = "@"
            entries.append(self.workspace.display_path(path) + suffix)
            return True

        if recursive:
            for current, directory_names, file_names in os.walk(base, followlinks=False):
                current_path = Path(current)
                directory_names.sort()
                file_names.sort()
                for name in directory_names:
                    if not add_entry(current_path / name):
                        break
                if truncated:
                    break
                directory_names[:] = [
                    name
                    for name in directory_names
                    if name not in self._PRUNED_DIRECTORIES
                    and not (current_path / name).is_symlink()
                ]
                for name in file_names:
                    if not add_entry(current_path / name):
                        break
                if truncated:
                    break
        else:
            for path in sorted(base.iterdir(), key=lambda item: item.name):
                if not add_entry(path):
                    break
        return ToolResult(
            ok=True,
            message=f"Listed {len(entries)} entries" + (" (truncated)" if truncated else ""),
            data={"entries": entries, "truncated": truncated},
        )

    def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        relative = _required_string(arguments, "path")
        path = self.workspace.resolve(relative, must_exist=True)
        if not path.is_file():
            raise ToolInputError(f"Not a file: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolInputError(f"File is not valid UTF-8: {relative}") from exc
        except OSError as exc:
            raise ToolInputError(f"Cannot read {relative!r}: {exc}") from exc
        content, truncated = _truncate(content, self.max_output_chars)
        return ToolResult(
            ok=True,
            message="File read" + (" (content truncated)" if truncated else ""),
            data={
                "path": self.workspace.display_path(path),
                "content": content,
                "truncated": truncated,
            },
        )

    def write_file(self, arguments: dict[str, Any]) -> ToolResult:
        relative = _required_string(arguments, "path")
        content = _required_string(arguments, "content", allow_empty=True)
        path = self.workspace.resolve(relative)
        if path.exists() and not path.is_file():
            raise ToolInputError(f"Not a file: {relative}")
        previous = ""
        if path.exists():
            try:
                previous = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ToolInputError(f"Cannot read UTF-8 file {relative!r}: {exc}") from exc
        if previous == content:
            return ToolResult(
                ok=True,
                message="File already contains the requested content",
                data={"path": self.workspace.display_path(path), "changed": False},
            )
        diff = _text_diff(
            self.workspace.display_path(path),
            previous,
            content,
            self.max_output_chars,
        )
        if not self._confirm("write_file", self.workspace.display_path(path), diff):
            return ToolResult(
                ok=False,
                message="File write was rejected by the user",
                data={"path": self.workspace.display_path(path), "diff": diff},
            )
        checkpoint = self._ensure_git_checkpoint()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Resolve again after directory creation to catch symlinks in newly visible parents.
            path = self.workspace.resolve(relative)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolInputError(f"Cannot write {relative!r}: {exc}") from exc
        return ToolResult(
            ok=True,
            message=f"Wrote {len(content)} characters" + _checkpoint_suffix(checkpoint),
            data={
                "path": self.workspace.display_path(path),
                "changed": True,
                "diff": diff,
                "checkpoint": checkpoint,
            },
        )

    def replace_text(self, arguments: dict[str, Any]) -> ToolResult:
        relative = _required_string(arguments, "path")
        old_text = _required_string(arguments, "old_text")
        new_text = _required_string(arguments, "new_text", allow_empty=True)
        path = self.workspace.resolve(relative, must_exist=True)
        if not path.is_file():
            raise ToolInputError(f"Not a file: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolInputError(f"Cannot read UTF-8 file {relative!r}: {exc}") from exc
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise ToolInputError("old_text was not found")
        if occurrences > 1:
            raise ToolInputError(f"old_text occurs {occurrences} times; provide a unique match")
        updated = content.replace(old_text, new_text, 1)
        if updated == content:
            return ToolResult(
                ok=True,
                message="File already contains the requested replacement",
                data={"path": self.workspace.display_path(path), "changed": False},
            )
        diff = _text_diff(
            self.workspace.display_path(path),
            content,
            updated,
            self.max_output_chars,
        )
        if not self._confirm("replace_text", self.workspace.display_path(path), diff):
            return ToolResult(
                ok=False,
                message="Text replacement was rejected by the user",
                data={"path": self.workspace.display_path(path), "diff": diff},
            )
        checkpoint = self._ensure_git_checkpoint()
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            raise ToolInputError(f"Cannot write {relative!r}: {exc}") from exc
        return ToolResult(
            ok=True,
            message="Replaced one occurrence" + _checkpoint_suffix(checkpoint),
            data={
                "path": self.workspace.display_path(path),
                "changed": True,
                "diff": diff,
                "checkpoint": checkpoint,
            },
        )

    def run_command(self, arguments: dict[str, Any]) -> ToolResult:
        argv = arguments.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
        ):
            raise ToolInputError("argv must be a non-empty array of non-empty strings")
        relative_cwd = _optional_string(arguments, "cwd", ".")
        cwd = self.workspace.resolve(relative_cwd, must_exist=True)
        if not cwd.is_dir():
            raise ToolInputError(f"Command cwd is not a directory: {relative_cwd}")
        requested_timeout = arguments.get("timeout", self.command_timeout)
        if (
            isinstance(requested_timeout, bool)
            or not isinstance(requested_timeout, int | float)
            or not math.isfinite(requested_timeout)
        ):
            raise ToolInputError("timeout must be a positive number")
        if requested_timeout <= 0:
            raise ToolInputError("timeout must be a positive number")
        timeout = min(float(requested_timeout), self.command_timeout)
        command_preview = (
            f"Command: {shlex.join(argv)}\n"
            f"Working directory: {self.workspace.display_path(cwd)}\n"
            f"Timeout: {timeout:g} seconds"
        )
        if not self._confirm("run_command", shlex.join(argv), command_preview):
            return ToolResult(
                ok=False,
                message="Command execution was rejected by the user",
                data={
                    "argv": argv,
                    "cwd": self.workspace.display_path(cwd),
                    "approved": False,
                },
            )
        checkpoint = self._ensure_git_checkpoint()

        environment = os.environ.copy()
        environment.pop("LLM_API_KEY", None)
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_timeout_output(exc.stdout)
            stderr = _coerce_timeout_output(exc.stderr)
            result = self._command_result(
                ok=False,
                message=f"Command timed out after {timeout:g} seconds",
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
            if result.data is not None:
                result.data["checkpoint"] = checkpoint
            return ToolResult(
                ok=result.ok,
                message=result.message + _checkpoint_suffix(checkpoint),
                data=result.data,
            )
        except OSError as exc:
            return ToolResult(
                ok=False,
                message=f"Could not start command: {exc}" + _checkpoint_suffix(checkpoint),
                data={
                    "argv": argv,
                    "cwd": self.workspace.display_path(cwd),
                    "checkpoint": checkpoint,
                },
            )
        result = self._command_result(
            ok=completed.returncode == 0,
            message=(
                "Command completed successfully"
                if completed.returncode == 0
                else f"Command failed with exit code {completed.returncode}"
            ),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )
        if result.data is not None:
            result.data["checkpoint"] = checkpoint
        result = ToolResult(
            ok=result.ok,
            message=result.message + _checkpoint_suffix(checkpoint),
            data=result.data,
        )
        return result

    def _confirm(self, operation: str, target: str, preview: str) -> bool:
        if self.confirm_action is None:
            return True
        try:
            return bool(self.confirm_action(operation, target, preview))
        except Exception as exc:
            raise ToolInputError(f"Operation confirmation failed: {exc}") from exc

    def _ensure_git_checkpoint(self) -> dict[str, Any]:
        if self._checkpoint_info is not None:
            return self._checkpoint_info

        command_prefix = ["git", "-C", str(self.workspace.root)]
        environment = os.environ.copy()
        environment.pop("LLM_API_KEY", None)
        try:
            repository = subprocess.run(
                [*command_prefix, "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                env=environment,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            repository = None
        if repository is None or repository.returncode != 0:
            self._checkpoint_info = {"available": False, "reason": "not a Git repository"}
            return self._checkpoint_info

        status = subprocess.run(
            [*command_prefix, "status", "--porcelain=v1"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=environment,
            check=False,
            shell=False,
        )
        if status.returncode != 0:
            raise ToolInputError("Cannot inspect Git status; operation cancelled")

        head = subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=environment,
            check=False,
            shell=False,
        )
        reference = head.stdout.strip() if head.returncode == 0 else ""
        if status.stdout.strip():
            stash = subprocess.run(
                [*command_prefix, "stash", "create", "coding-agent checkpoint"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                env=environment,
                check=False,
                shell=False,
            )
            if stash.returncode != 0:
                raise ToolInputError("Cannot create Git checkpoint; operation cancelled")
            reference = stash.stdout.strip() or reference

        if reference:
            updated = subprocess.run(
                [
                    *command_prefix,
                    "update-ref",
                    "refs/coding-agent/checkpoint",
                    reference,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                env=environment,
                check=False,
                shell=False,
            )
            if updated.returncode != 0:
                raise ToolInputError("Cannot record Git checkpoint; operation cancelled")

        status_lines = [line for line in status.stdout.splitlines() if line.strip()]
        untracked = sum(line.startswith("?? ") for line in status_lines)
        self._checkpoint_info = {
            "available": bool(reference),
            "reference": reference or None,
            "dirty_before": bool(status_lines),
            "untracked_before": untracked,
        }
        return self._checkpoint_info

    def _command_result(
        self,
        *,
        ok: bool,
        message: str,
        returncode: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> ToolResult:
        combined = f"stdout:\n{stdout}\nstderr:\n{stderr}"
        combined, truncated = _truncate(combined, self.max_output_chars)
        return ToolResult(
            ok=ok,
            message=message + (" (output truncated)" if truncated else ""),
            data={
                "returncode": returncode,
                "output": combined,
                "timed_out": timed_out,
                "truncated": truncated,
            },
        )


def _text_diff(path: str, before: str, after: str, limit: int) -> str:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if not diff:
        return "(no textual changes)"
    return _truncate(diff, limit)[0]


def _checkpoint_suffix(checkpoint: dict[str, Any]) -> str:
    reference = checkpoint.get("reference")
    if reference:
        return f"; Git checkpoint {reference[:12]} recorded"
    if checkpoint.get("available") is False:
        return "; Git checkpoint unavailable"
    return ""


class ToolRegistry:
    """Expose tool schemas and convert all dispatch errors into tool results."""

    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("Tool names must be unique")

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [definition.as_openai_tool() for definition in self._definitions.values()]

    def dispatch(self, name: str, raw_arguments: str) -> ToolResult:
        definition = self._definitions.get(name)
        if definition is None:
            return ToolResult(ok=False, message=f"Unknown tool: {name}")
        try:
            arguments = json.loads(raw_arguments, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            return ToolResult(ok=False, message=f"Invalid JSON arguments: {detail}")
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, message="Tool arguments must be a JSON object")
        try:
            return definition.handler(arguments)
        except ToolInputError as exc:
            return ToolResult(ok=False, message=str(exc))
        except Exception as exc:  # pragma: no cover - last-resort isolation boundary
            return ToolResult(ok=False, message=f"Tool execution failed unexpectedly: {exc}")


def _required_string(arguments: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = " (may be empty)" if allow_empty else ""
        raise ToolInputError(f"{name} must be a string{suffix}")
    return value


def _optional_string(arguments: dict[str, Any], name: str, default: str) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or not value:
        raise ToolInputError(f"{name} must be a non-empty string")
    return value


def _optional_bool(arguments: dict[str, Any], name: str, default: bool) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolInputError(f"{name} must be a boolean")
    return value


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = "\n... output truncated ...\n"
    if limit <= len(marker):
        return marker[:limit], True
    head_length = (limit - len(marker)) // 2
    tail_length = limit - len(marker) - head_length
    return value[:head_length] + marker + value[-tail_length:], True


def _coerce_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")
