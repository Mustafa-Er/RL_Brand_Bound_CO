"""Shared infrastructure: YAML config, seeding, logging, paths.

Used by every stage. Kept deliberately small so the rest of the codebase has
exactly one place to look for these primitives.
"""
from __future__ import annotations

import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Return the absolute path of the repository root."""
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    """Resolve a config-relative path to an absolute path under repo_root."""
    path = Path(p)
    return path if path.is_absolute() else repo_root() / path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files and merge them left-to-right."""
    if not paths:
        raise ValueError("load_config requires at least one path")
    cfg: dict[str, Any] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML in {p} must be a mapping")
        cfg = _deep_merge(cfg, data)
    return cfg


def resolve_device(device_cfg: str) -> str:
    """Resolve ``device: auto`` to ``cuda`` or ``cpu`` at runtime."""
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    """Configure the root logger to emit to stderr and (optionally) a file."""
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_LOG_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / filename, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root
