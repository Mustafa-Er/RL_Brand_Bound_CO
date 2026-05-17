"""Simple file + stream logging setup."""
from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure(log_dir: Path | None = None, level: int = logging.INFO,
              filename: str = "run.log") -> logging.Logger:
    """Configure the root logger to emit to stderr and (optionally) a file.

    Safe to call multiple times; existing handlers are cleared first.
    """
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
