"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


_ENV_NAMES: Final = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_environment_file(path: str | Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries from a dotenv-style file."""

    environment_path = Path(path).expanduser()
    try:
        lines = environment_path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigurationError(f"Cannot read environment file {environment_path}: {exc}") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(
                f"Invalid environment file {environment_path} line {line_number}: "
                "expected KEY=VALUE"
            )
        name, raw_value = (part.strip() for part in line.split("=", 1))
        if _ENV_KEY.fullmatch(name) is None:
            raise ConfigurationError(
                f"Invalid environment variable name {name!r} in "
                f"{environment_path} line {line_number}"
            )
        values[name] = _parse_environment_value(raw_value)
    return values


def _parse_environment_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def _load_environment_values(workspace: str | Path) -> dict[str, str]:
    """Load dotenv files with workspace values taking precedence over cwd values."""

    values: dict[str, str] = {}
    candidates = [Path.cwd() / ".env", Path(workspace).expanduser().resolve() / ".env"]
    for candidate in candidates:
        if candidate.exists():
            values.update(load_environment_file(candidate))
    return values


@dataclass(frozen=True)
class Settings:
    """Settings shared by the CLI, model adapter and agent."""

    workspace: Path
    api_key: str
    base_url: str
    model: str
    max_steps: int = 20
    command_timeout: float = 20.0
    max_output_chars: int = 20_000
    max_history_tokens: int = 50_000

    @classmethod
    def from_environment(
        cls,
        workspace: str | Path,
        *,
        max_steps: int = 20,
        command_timeout: float = 20.0,
        max_output_chars: int = 20_000,
        max_history_tokens: int = 50_000,
    ) -> Settings:
        file_values = _load_environment_values(workspace)
        values = {
            name: os.getenv(name, "").strip() or file_values.get(name, "").strip()
            for name in _ENV_NAMES
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ConfigurationError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )
        if max_steps < 1:
            raise ConfigurationError("max_steps must be at least 1")
        if command_timeout <= 0:
            raise ConfigurationError("command_timeout must be greater than 0")
        if max_output_chars < 1:
            raise ConfigurationError("max_output_chars must be at least 1")
        if max_history_tokens < 1:
            raise ConfigurationError("max_history_tokens must be at least 1")
        return cls(
            workspace=Path(workspace).expanduser().resolve(),
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"],
            model=values["LLM_MODEL"],
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_output_chars=max_output_chars,
            max_history_tokens=max_history_tokens,
        )
