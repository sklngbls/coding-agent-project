"""Persistent local conversation sessions.

Session files live outside the coding workspace and are written atomically. The
storage layer intentionally contains no model or Agent logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .history import truncate_messages


class SessionError(RuntimeError):
    """Base class for session storage failures."""


class SessionNotFoundError(SessionError):
    """Raised when a requested session does not exist."""


class SessionWorkspaceError(SessionError):
    """Raised when a session belongs to another workspace."""


class SessionCorruptError(SessionError):
    """Raised when a session file is not valid or has an unexpected shape."""


@dataclass
class Session:
    """A persisted conversation and its workspace ownership metadata."""

    session_id: str
    title: str
    workspace: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]


MessageHistory = list[dict[str, Any]]
SessionList = list[Session]


_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_TITLE_CHARS = 60
UNTITLED_SESSION_TITLE = "Untitled session"


def make_session_title(value: str | None) -> str:
    """Create a compact, single-line title suitable for terminal display."""

    if value is None:
        return UNTITLED_SESSION_TITLE
    normalized = " ".join(value.split())
    if not normalized:
        return UNTITLED_SESSION_TITLE
    if len(normalized) <= _MAX_TITLE_CHARS:
        return normalized
    return normalized[: _MAX_TITLE_CHARS - 3].rstrip() + "..."


def normalize_workspace(workspace: str | Path) -> Path:
    """Return the canonical workspace path used for ownership and hashing."""

    return Path(workspace).expanduser().resolve(strict=False)


def workspace_key(workspace: str | Path) -> str:
    """Return a stable, collision-resistant directory key for a workspace."""

    normalized = str(normalize_workspace(workspace))
    if os.name == "nt":
        normalized = normalized.casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def default_storage_root() -> Path:
    """Choose a per-user storage root outside normal workspaces."""

    if os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data).expanduser() / "coding-agent" / "sessions"
    xdg_data_home = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "coding-agent" / "sessions"
    return Path.home() / ".local" / "share" / "coding-agent" / "sessions"


class SessionStore:
    """Create and manage sessions belonging to one normalized workspace."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        api_key: str | None = None,
        storage_root: str | Path | None = None,
        max_messages: int = 200,
    ) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.workspace_path = normalize_workspace(workspace)
        requested_root = normalize_workspace(storage_root or default_storage_root())
        if requested_root == self.workspace_path or requested_root.is_relative_to(
            self.workspace_path
        ):
            raise ValueError("session storage must be outside the workspace")
        self.storage_root = requested_root
        self.workspace_dir = self.storage_root / workspace_key(self.workspace_path)
        self.api_key = api_key or ""
        self.max_messages = max_messages

    def create(
        self,
        messages: MessageHistory | None = None,
        *,
        title: str | None = None,
    ) -> Session:
        """Create an unsaved session with a fresh ID and current timestamps."""

        now = _utc_now()
        prepared_messages = self._prepare_messages(messages or [])
        return Session(
            session_id=uuid.uuid4().hex,
            title=self._prepare_title(title, prepared_messages),
            workspace=str(self.workspace_path),
            created_at=now,
            updated_at=now,
            messages=prepared_messages,
        )

    def save(self, session: Session) -> Session:
        """Atomically persist a session and return its sanitized state."""

        self._validate_session_id(session.session_id)
        session_workspace = normalize_workspace(session.workspace)
        if session_workspace != self.workspace_path:
            raise SessionWorkspaceError(
                f"Session belongs to {session_workspace}, not {self.workspace_path}"
            )
        try:
            created_at = _validate_timestamp(session.created_at, "created_at")
            messages = self._prepare_messages(session.messages)
            title = self._prepare_title(session.title, messages)
            updated_at = _utc_now()
            payload = {
                "session_id": session.session_id,
                "title": title,
                "workspace": str(self.workspace_path),
                "created_at": created_at,
                "updated_at": updated_at,
                "messages": messages,
            }
            encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        except (TypeError, ValueError) as exc:
            raise SessionError(f"Cannot serialize session {session.session_id}: {exc}") from exc

        path = self._session_path(session.session_id)
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{session.session_id}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except OSError as exc:
            raise SessionError(
                f"Cannot atomically save session {session.session_id}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        session.workspace = str(self.workspace_path)
        session.title = title
        session.created_at = created_at
        session.updated_at = updated_at
        session.messages = messages
        return session

    def load(self, session_id: str) -> Session:
        """Load a session and reject IDs belonging to another workspace."""

        self._validate_session_id(session_id)
        path = self._session_path(session_id)
        if not path.exists():
            foreign_path = self._find_session_elsewhere(session_id)
            if foreign_path is not None:
                raise SessionWorkspaceError(f"Session {session_id} belongs to another workspace")
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return self._read(path, expected_id=session_id)

    def list(self) -> SessionList:
        """List this workspace's sessions from newest to oldest."""

        if not self.workspace_dir.exists():
            return []
        sessions: SessionList = []
        for path in sorted(self.workspace_dir.glob("*.json")):
            if not _SESSION_ID.fullmatch(path.stem):
                continue
            sessions.append(self._read(path, expected_id=path.stem))
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions

    def get_recent(self) -> Session | None:
        """Return the most recently updated session, if any."""

        sessions = self.list()
        return sessions[0] if sessions else None

    def rename(self, session_id: str, title: str) -> Session:
        """Rename a persisted session and return its updated state."""

        if not isinstance(title, str) or not title.strip():
            raise SessionError("title must not be empty")
        session = self.load(session_id)
        session.title = self._prepare_title(title, session.messages)
        return self.save(session)

    def _session_path(self, session_id: str) -> Path:
        return self.workspace_dir / f"{session_id}.json"

    def _read(self, path: Path, *, expected_id: str) -> Session:
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionCorruptError(f"Cannot read session file {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SessionCorruptError(f"Session file {path} must contain a JSON object")
        try:
            session_id = _require_string(payload, "session_id")
            workspace = _require_string(payload, "workspace")
            created_at = _validate_timestamp(_require_string(payload, "created_at"), "created_at")
            updated_at = _validate_timestamp(_require_string(payload, "updated_at"), "updated_at")
            messages = payload["messages"]
            if not isinstance(messages, list) or any(
                not isinstance(item, dict) for item in messages
            ):
                raise ValueError("messages must be a list of JSON objects")
            raw_title = payload.get("title")
            if raw_title is not None and not isinstance(raw_title, str):
                raise ValueError("title must be a string")
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionCorruptError(f"Invalid session file {path}: {exc}") from exc
        if _SESSION_ID.fullmatch(session_id) is None:
            raise SessionCorruptError(f"Invalid session ID in file: {path}")
        if session_id != expected_id:
            raise SessionCorruptError(f"Session ID does not match file name: {path}")
        normalized_workspace = normalize_workspace(workspace)
        if normalized_workspace != self.workspace_path:
            raise SessionWorkspaceError(
                f"Session belongs to {normalized_workspace}, not {self.workspace_path}"
            )
        return Session(
            session_id=session_id,
            title=self._prepare_title(raw_title, messages),
            workspace=str(normalized_workspace),
            created_at=created_at,
            updated_at=updated_at,
            messages=self._prepare_messages(messages),
        )

    def _find_session_elsewhere(self, session_id: str) -> Path | None:
        if not self.storage_root.exists():
            return None
        for candidate in self.storage_root.glob(f"*/{session_id}.json"):
            if candidate.parent != self.workspace_dir:
                return candidate
        return None

    def _prepare_messages(self, messages: MessageHistory) -> MessageHistory:
        sanitized = [_redact_value(message, self.api_key) for message in messages]
        if any(not isinstance(message, dict) for message in sanitized):
            raise ValueError("messages must be a list of JSON objects")
        return truncate_messages(sanitized, self.max_messages)

    def _prepare_title(self, title: str | None, messages: MessageHistory) -> str:
        if title is not None and not isinstance(title, str):
            raise ValueError("title must be a string")
        candidate = title if title and title.strip() else _first_user_message(messages)
        redacted = _redact_value(candidate, self.api_key)
        return make_session_title(redacted if isinstance(redacted, str) else None)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise SessionError("session_id must be a 32-character lowercase hexadecimal ID")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_timestamp(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty timestamp")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    return value


def _first_user_message(messages: MessageHistory) -> str | None:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _redact_value(value: Any, api_key: str) -> Any:
    if isinstance(value, str):
        return value.replace(api_key, "[redacted]") if api_key else value
    if isinstance(value, list):
        return [_redact_value(item, api_key) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item, api_key) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_redact_value(item, api_key) for item in value]
    return value
