"""Loading of the YAML files in `config/` (conventions.md section 9).

Every tunable value lives in `config/*.yaml`, so this module is the one place
that knows where they are. Loads are cached: configs are read-only at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
CONVENTIONS_VERSION = 2


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load ``config/<name>.yaml``, checking that it declares our conventions version."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    found = cfg.get("conventions_version")
    if found is not None and found != CONVENTIONS_VERSION:
        raise ValueError(
            f"{path.name} declares conventions_version {found}, code expects {CONVENTIONS_VERSION}"
        )
    return cfg


def skills() -> dict[str, Any]:
    return load("skills")


def joint_names() -> dict[str, Any]:
    return load("joint_names")


def calibration() -> dict[str, Any]:
    return load("calibration")


def home_q() -> np.ndarray:
    """HOME as the 16-value robot-interface vector (conventions.md section 4)."""
    return np.asarray(load("home")["q"], dtype=np.float64)


def path(*parts: str) -> Path:
    """A repo-relative path, so scripts work from any working directory."""
    return REPO_ROOT.joinpath(*parts)
