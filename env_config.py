"""Shared .env loader for local simulator and inference runs."""

from __future__ import annotations

import os
from pathlib import Path


_LOADED = False


def _candidate_env_files() -> list[Path]:
    root = Path(__file__).resolve().parent
    return [root / ".env.local", root / ".env"]


def load_repo_env() -> None:
    global _LOADED
    if _LOADED:
        return

    for path in _candidate_env_files():
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)

    _LOADED = True
