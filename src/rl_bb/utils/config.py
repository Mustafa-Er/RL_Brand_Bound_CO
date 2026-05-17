"""YAML config loader with deep-merge override support."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``. Non-dict values overwrite."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files and merge them left-to-right.

    Later files override earlier ones. Empty files are treated as ``{}``.
    """
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
