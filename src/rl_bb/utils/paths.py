"""Project path helpers. All paths derived from the repo root."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the absolute path of the repository root."""
    return Path(__file__).resolve().parents[3]


def resolve(p: str | Path) -> Path:
    """Resolve a config-relative path to an absolute Path under repo_root."""
    path = Path(p)
    return path if path.is_absolute() else repo_root() / path
