from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from coding_agent.tools import LocalTools, ToolRegistry, Workspace


def result_payload(registry: ToolRegistry, name: str, arguments: dict[str, object]) -> dict:
    return json.loads(registry.dispatch(name, json.dumps(arguments)).to_json())


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    tools = LocalTools(tmp_path, command_timeout=0.25, max_output_chars=2_000)
    return ToolRegistry(tools.definitions())


def test_file_write_read_replace_and_list(registry: ToolRegistry, tmp_path: Path) -> None:
    written = result_payload(
        registry,
        "write_file",
        {"path": "src/hello.txt", "content": "hello world\n"},
    )
    assert written["ok"] is True
    assert (tmp_path / "src" / "hello.txt").read_text(encoding="utf-8") == "hello world\n"

    read = result_payload(registry, "read_file", {"path": "src/hello.txt"})
    assert read["data"]["content"] == "hello world\n"

    replaced = result_payload(
        registry,
        "replace_text",
        {"path": "src/hello.txt", "old_text": "world", "new_text": "agent"},
    )
    assert replaced["ok"] is True
    assert (tmp_path / "src" / "hello.txt").read_text(encoding="utf-8") == "hello agent\n"

    listing = result_payload(registry, "list_files", {"path": ".", "recursive": True})
    assert "src/" in listing["data"]["entries"]
    assert "src/hello.txt" in listing["data"]["entries"]


@pytest.mark.parametrize("path", ["../secret.txt", "folder/../../secret.txt"])
def test_relative_path_escape_is_rejected(registry: ToolRegistry, path: str) -> None:
    result = result_payload(registry, "write_file", {"path": path, "content": "no"})
    assert result["ok"] is False
    assert "relative" in result["message"]


def test_absolute_path_is_rejected(registry: ToolRegistry, tmp_path: Path) -> None:
    result = result_payload(
        registry,
        "read_file",
        {"path": str((tmp_path / "outside.txt").resolve())},
    )
    assert result["ok"] is False
    assert "relative" in result["message"]


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this system")
    registry = ToolRegistry(LocalTools(workspace).definitions())
    result = result_payload(registry, "write_file", {"path": "link/secret.txt", "content": "no"})
    assert result["ok"] is False
    assert "escapes" in result["message"]


def test_workspace_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        Workspace(tmp_path / "missing")


def test_unknown_tool_and_bad_json_are_results(registry: ToolRegistry) -> None:
    unknown = registry.dispatch("missing", "{}").to_json()
    invalid = registry.dispatch("read_file", "{").to_json()
    assert json.loads(unknown)["ok"] is False
    assert "Unknown tool" in unknown
    assert json.loads(invalid)["ok"] is False
    assert "Invalid JSON" in invalid


def test_non_standard_json_constant_is_rejected(registry: ToolRegistry) -> None:
    result = registry.dispatch("run_command", '{"argv":["test"],"timeout":NaN}')
    assert result.ok is False
    assert "non-standard JSON constant" in result.message


def test_command_success(registry: ToolRegistry) -> None:
    result = result_payload(
        registry,
        "run_command",
        {"argv": [sys.executable, "-c", "print('ready')"]},
    )
    assert result["ok"] is True
    assert result["data"]["returncode"] == 0
    assert "ready" in result["data"]["output"]


def test_command_failure(registry: ToolRegistry) -> None:
    result = result_payload(
        registry,
        "run_command",
        {"argv": [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"]},
    )
    assert result["ok"] is False
    assert result["data"]["returncode"] == 7
    assert "bad" in result["data"]["output"]


def test_command_timeout(registry: ToolRegistry) -> None:
    result = result_payload(
        registry,
        "run_command",
        {"argv": [sys.executable, "-c", "import time; time.sleep(2)"], "timeout": 0.1},
    )
    assert result["ok"] is False
    assert result["data"]["timed_out"] is True
    assert "timed out" in result["message"]


def test_command_output_is_limited(tmp_path: Path) -> None:
    registry = ToolRegistry(LocalTools(tmp_path, max_output_chars=120).definitions())
    result = result_payload(
        registry,
        "run_command",
        {"argv": [sys.executable, "-c", "print('x' * 1000)"]},
    )
    assert result["ok"] is True
    assert result["data"]["truncated"] is True
    assert len(result["data"]["output"]) == 120


def test_command_does_not_inherit_llm_api_key(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "must-not-leak")
    result = result_payload(
        registry,
        "run_command",
        {"argv": [sys.executable, "-c", "import os; print(os.getenv('LLM_API_KEY'))"]},
    )
    assert result["ok"] is True
    assert "must-not-leak" not in result["data"]["output"]
    assert "None" in result["data"]["output"]


def test_command_cwd_is_confined(registry: ToolRegistry) -> None:
    result = result_payload(
        registry,
        "run_command",
        {"argv": [sys.executable, "-c", "print('no')"], "cwd": ".."},
    )
    assert result["ok"] is False
    assert "relative" in result["message"]


def test_registry_schemas_have_required_tools(registry: ToolRegistry) -> None:
    names = {schema["function"]["name"] for schema in registry.schemas}
    assert {"list_files", "read_file", "write_file", "run_command"} <= names


def test_recursive_listing_prunes_internal_directories(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "internal").write_text("hidden", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("shown", encoding="utf-8")
    result = result_payload(registry, "list_files", {"path": ".", "recursive": True})
    assert ".git/" in result["data"]["entries"]
    assert ".git/objects/" not in result["data"]["entries"]
    assert "visible.txt" in result["data"]["entries"]


def test_command_missing_executable_is_a_result(registry: ToolRegistry) -> None:
    missing_name = "definitely-not-a-real-command-for-coding-agent-tests"
    result = result_payload(registry, "run_command", {"argv": [missing_name]})
    assert result["ok"] is False
    assert "Could not start command" in result["message"]


def test_read_missing_file_is_a_result(registry: ToolRegistry) -> None:
    result = result_payload(registry, "read_file", {"path": "missing.txt"})
    assert result["ok"] is False
    assert "does not exist" in result["message"]


def test_llm_api_key_remains_in_parent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "parent-value")
    assert os.environ["LLM_API_KEY"] == "parent-value"
