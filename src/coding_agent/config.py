"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


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

    @classmethod
    def from_environment(
        cls,
        workspace: str | Path,
        *,
        max_steps: int = 20,
        command_timeout: float = 20.0,
        max_output_chars: int = 20_000,
    ) -> Settings:
        values = {
            "LLM_API_KEY": os.getenv("LLM_API_KEY", "").strip(),
            "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").strip(),
            "LLM_MODEL": os.getenv("LLM_MODEL", "").strip(),
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
        return cls(
            workspace=Path(workspace).expanduser().resolve(),
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"],
            model=values["LLM_MODEL"],
            max_steps=max_steps,
            command_timeout=command_timeout,
            max_output_chars=max_output_chars,
        )
