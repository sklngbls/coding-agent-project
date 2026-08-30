from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.workspaces import WorkspaceError, WorkspaceStore


def test_remember_lists_existing_workspaces_newest_first(tmp_path: Path) -> None:
    registry_path = tmp_path / "workspaces.json"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    store = WorkspaceStore(registry_path)

    store.remember(first)
    store.remember(second)
    store.remember(first)

    records = store.list()
    assert [record.path for record in records] == [first.resolve(), second.resolve()]


def test_missing_workspace_is_not_returned_from_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "workspaces.json"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WorkspaceStore(registry_path)
    store.remember(workspace)
    workspace.rmdir()

    assert store.list() == []


def test_remember_rejects_file_and_missing_path(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "workspaces.json")
    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="not a directory"):
        store.remember(file_path)
    with pytest.raises(WorkspaceError, match="does not exist"):
        store.remember(tmp_path / "missing")


def test_corrupt_registry_is_reported(tmp_path: Path) -> None:
    registry_path = tmp_path / "workspaces.json"
    registry_path.write_text(json.dumps({"path": "invalid"}), encoding="utf-8")

    with pytest.raises(WorkspaceError, match="must contain a JSON list"):
        WorkspaceStore(registry_path).list()
