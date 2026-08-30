"""Persistence for recently used local workspaces."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .sessions import default_storage_root, normalize_workspace


class WorkspaceError(RuntimeError):
    """Raised when a workspace cannot be selected or recorded."""


@dataclass(frozen=True)
class WorkspaceRecord:
    """A workspace path and the time it was last selected."""

    path: Path
    updated_at: str


WorkspaceRecords = list[WorkspaceRecord]


class WorkspaceStore:
    """Store recently used workspaces outside project directories."""

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.registry_path = Path(registry_path or default_registry_path()).expanduser().resolve()

    def list(self) -> list[WorkspaceRecord]:
        """Return existing workspaces ordered from newest to oldest."""

        if not self.registry_path.exists():
            return []
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError(f"Cannot read workspace registry: {exc}") from exc
        if not isinstance(payload, list):
            raise WorkspaceError("Workspace registry must contain a JSON list")

        records: WorkspaceRecords = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            updated_at = item.get("updated_at")
            if not isinstance(raw_path, str) or not isinstance(updated_at, str):
                continue
            try:
                path = _validate_workspace(raw_path)
                datetime.fromisoformat(updated_at)
            except (OSError, ValueError, WorkspaceError):
                continue
            records.append(WorkspaceRecord(path=path, updated_at=updated_at))
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records

    def remember(self, workspace: str | Path) -> Path:
        """Record an existing directory and return its normalized path."""

        path = _validate_workspace(workspace)
        records = [item for item in self.list() if not _same_path(item.path, path)]
        records.insert(0, WorkspaceRecord(path=path, updated_at=_utc_now()))
        self._save(records)
        return path

    def _save(self, records: WorkspaceRecords) -> None:
        payload = [
            {"path": str(record.path), "updated_at": record.updated_at} for record in records
        ]
        encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        temporary_path: Path | None = None
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.registry_path.parent,
                prefix=f".{self.registry_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.registry_path)
        except (OSError, TypeError, ValueError) as exc:
            raise WorkspaceError(f"Cannot save workspace registry: {exc}") from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


def default_registry_path() -> Path:
    """Return the per-user registry path next to persisted sessions."""

    return default_storage_root().parent / "workspaces.json"


def _validate_workspace(value: str | Path) -> Path:
    path = normalize_workspace(value)
    if not path.exists():
        raise WorkspaceError(f"Workspace does not exist: {path}")
    if not path.is_dir():
        raise WorkspaceError(f"Workspace is not a directory: {path}")
    return path


def _same_path(first: Path, second: Path) -> bool:
    if os.name == "nt":
        return str(first).casefold() == str(second).casefold()
    return first == second


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["WorkspaceError", "WorkspaceRecord", "WorkspaceStore", "default_registry_path"]
