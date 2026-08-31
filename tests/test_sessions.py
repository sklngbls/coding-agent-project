from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.sessions import (
    UNTITLED_SESSION_TITLE,
    SessionCorruptError,
    SessionError,
    SessionNotFoundError,
    SessionStore,
    SessionWorkspaceError,
    make_session_title,
    workspace_key,
)


def test_create_save_load_list_and_recent(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    first = store.create([{"role": "system", "content": "system"}])
    store.save(first)
    second = store.create([{"role": "system", "content": "system"}])
    store.save(second)

    loaded = store.load(first.session_id)
    listed = store.list()

    assert loaded.session_id == first.session_id
    assert loaded.title == UNTITLED_SESSION_TITLE
    assert {item.session_id for item in listed} == {first.session_id, second.session_id}
    recent = store.get_recent()
    assert recent is not None
    assert recent.session_id == second.session_id


def test_workspace_hash_is_stable_and_isolates_same_named_directories(tmp_path: Path) -> None:
    workspace_one = tmp_path / "one" / "project"
    workspace_two = tmp_path / "two" / "project"
    assert workspace_key(workspace_one) == workspace_key(workspace_one)
    assert workspace_key(workspace_one) != workspace_key(workspace_two)

    root = tmp_path / "sessions"
    store_one = SessionStore(workspace_one, storage_root=root)
    store_two = SessionStore(workspace_two, storage_root=root)
    session = store_one.create()
    store_one.save(session)
    assert store_two.list() == []
    with pytest.raises(SessionWorkspaceError):
        store_two.load(session.session_id)


def test_corrupt_json_and_invalid_shape_have_clear_errors(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    store.workspace_dir.mkdir(parents=True)
    path = store.workspace_dir / ("a" * 32 + ".json")
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(SessionCorruptError, match="Cannot read session file"):
        store.load("a" * 32)

    path.write_text(json.dumps({"session_id": "a" * 32}), encoding="utf-8")
    with pytest.raises(SessionCorruptError, match="Invalid session file"):
        store.load("a" * 32)


def test_save_is_atomic_and_redacts_api_key_recursively(tmp_path: Path) -> None:
    secret = "secret-api-key"
    store = SessionStore(
        tmp_path / "workspace",
        storage_root=tmp_path / "sessions",
        api_key=secret,
    )
    session = store.create(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"value={secret}"},
            {
                "role": "tool",
                "content": {"stdout": ["nested", secret], "stderr": secret},
            },
        ],
        title=f"title={secret}",
        summary=f"summary={secret}",
    )
    store.save(session)

    path = store.workspace_dir / f"{session.session_id}.json"
    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    assert "[redacted]" in raw
    assert not list(store.workspace_dir.glob(".*.tmp"))
    assert session.title == "title=[redacted]"
    assert session.summary == "summary=[redacted]"


def test_summary_round_trips_for_legacy_and_new_sessions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    session = store.create(title="With summary", summary="Important project facts")
    store.save(session)

    loaded = store.load(session.session_id)

    assert loaded.summary == "Important project facts"


def test_title_is_generated_from_first_user_message_and_normalized(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    task = "  Add   session titles\n and list existing conversations  "

    session = store.create([{"role": "user", "content": task}])
    manual = store.create(title="  Manual   title  ")

    assert session.title == "Add session titles and list existing conversations"
    assert manual.title == "Manual title"
    assert len(make_session_title("x" * 100)) == 60
    assert make_session_title("x" * 100).endswith("...")


def test_rename_updates_persisted_title(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    session = store.create([{"role": "user", "content": "Original task"}], title="Old title")
    store.save(session)

    renamed = store.rename(session.session_id, "  New   title  ")

    assert renamed.title == "New title"
    assert store.load(session.session_id).title == "New title"


def test_rename_rejects_empty_title(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    session = store.create(title="Existing")
    store.save(session)

    with pytest.raises(SessionError, match="title must not be empty"):
        store.rename(session.session_id, "   ")


def test_delete_removes_persisted_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    session = store.create(title="To delete")
    store.save(session)

    deleted = store.delete(session.session_id)

    assert deleted.title == "To delete"
    assert store.list() == []
    with pytest.raises(SessionNotFoundError):
        store.load(session.session_id)


def test_legacy_session_without_title_uses_first_user_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    session = store.create([{"role": "user", "content": "Legacy task title"}])
    store.save(session)
    path = store.workspace_dir / f"{session.session_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["title"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load(session.session_id)

    assert loaded.title == "Legacy task title"


def test_failed_atomic_replace_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "workspace", storage_root=tmp_path / "sessions")
    session = store.create([{"role": "user", "content": "original"}])
    store.save(session)
    path = store.workspace_dir / f"{session.session_id}.json"
    original = path.read_text(encoding="utf-8")
    session.messages.append({"role": "assistant", "content": "replacement"})

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("coding_agent.sessions.os.replace", fail_replace)
    with pytest.raises(SessionError, match="Cannot atomically save"):
        store.save(session)

    assert path.read_text(encoding="utf-8") == original
    assert not list(store.workspace_dir.glob(".*.tmp"))


def test_history_limit_keeps_system_and_newest_messages(tmp_path: Path) -> None:
    store = SessionStore(
        tmp_path / "workspace",
        storage_root=tmp_path / "sessions",
        max_messages=3,
    )
    session = store.create(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "middle"},
            {"role": "user", "content": "new"},
        ]
    )
    assert [item["content"] for item in session.messages] == ["system", "middle", "new"]


def test_history_limit_drops_incomplete_tool_round(tmp_path: Path) -> None:
    store = SessionStore(
        tmp_path / "workspace",
        storage_root=tmp_path / "sessions",
        max_messages=3,
    )
    session = store.create(
        [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "user", "content": "new task"},
        ]
    )

    assert [item["role"] for item in session.messages] == ["system", "user"]
    assert session.messages[-1]["content"] == "new task"
