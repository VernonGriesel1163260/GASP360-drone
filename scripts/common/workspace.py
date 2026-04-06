from __future__ import annotations

import os
from pathlib import Path


def resolve_code_root(caller_file: str | Path) -> Path:
    return Path(caller_file).resolve().parent.parent


def resolve_workspace_root(explicit_root: str | Path | None = None, caller_file: str | Path | None = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    env_root = os.environ.get("GASP_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    if caller_file is None:
        raise ValueError("caller_file is required when no explicit root or env root is provided")

    return resolve_code_root(caller_file)