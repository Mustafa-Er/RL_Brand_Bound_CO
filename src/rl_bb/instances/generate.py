"""Instance generation for Set Covering and Combinatorial Auction problems.

Uses Ecole's built-in instance generators. Instances are written to disk in
MPS format and grouped by problem type, size regime, and split.

Directory layout produced::

    <data_dir>/<problem>/<regime>/<split>/instance_0000.mps
                                          instance_0001.mps
                                          ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import ecole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Size regimes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetCoverSize:
    n_rows: int
    n_cols: int
    density: float = 0.05


@dataclass(frozen=True)
class AuctionSize:
    n_items: int
    n_bids: int


REGIMES = ("train_size", "transfer_medium", "transfer_large")
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------

def _set_cover_generator(size: SetCoverSize, seed: int) -> ecole.instance.SetCoverGenerator:
    return ecole.instance.SetCoverGenerator(
        n_rows=size.n_rows,
        n_cols=size.n_cols,
        density=size.density,
        rng=ecole.RandomGenerator(seed),
    )


def _auction_generator(size: AuctionSize, seed: int) -> ecole.instance.CombinatorialAuctionGenerator:
    return ecole.instance.CombinatorialAuctionGenerator(
        n_items=size.n_items,
        n_bids=size.n_bids,
        rng=ecole.RandomGenerator(seed),
    )


def make_generator(problem_type: str, size_cfg: dict, seed: int):
    """Return a configured Ecole generator for the requested problem/size."""
    if problem_type == "set_covering":
        return _set_cover_generator(SetCoverSize(**size_cfg), seed)
    if problem_type == "combinatorial_auction":
        return _auction_generator(AuctionSize(**size_cfg), seed)
    raise ValueError(f"Unknown problem_type: {problem_type!r}")


# ---------------------------------------------------------------------------
# Writing instances
# ---------------------------------------------------------------------------

def _iter_models(generator, n: int) -> Iterator:
    """Yield ``n`` SCIP models from an Ecole generator."""
    for i in range(n):
        yield i, next(generator)


def write_instances(
    problem_type: str,
    size_cfg: dict,
    out_dir: Path,
    n: int,
    seed: int,
) -> int:
    """Generate ``n`` instances and write them as ``.mps`` files to ``out_dir``.

    Returns the number of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(problem_type, size_cfg, seed)
    written = 0
    for i, model in _iter_models(generator, n):
        target = out_dir / f"instance_{i:04d}.mps"
        # ecole.scip.Model wraps a SCIP model; expose underlying pyscipopt Model
        # via as_pyscipopt() and use writeProblem with .mps extension.
        model.as_pyscipopt().writeProblem(str(target))
        written += 1
    logger.info("Wrote %d instances to %s", written, out_dir)
    return written


def generate_all(
    problem_type: str,
    sizes: dict,           # {regime: size_cfg}
    counts: dict,          # {split: int}
    data_dir: Path,
    base_seed: int,
    regimes: tuple[str, ...] = REGIMES,
) -> dict[tuple[str, str], int]:
    """Generate every (regime, split) combination.

    Training generates only the ``train_size`` regime; ``val`` and ``test``
    splits are produced for all regimes (training-size eval + transfer eval).

    Returns a mapping ``{(regime, split): n_written}``.
    """
    results: dict[tuple[str, str], int] = {}
    for regime in regimes:
        if regime not in sizes:
            raise KeyError(f"Missing size config for regime {regime!r}")
        size_cfg = sizes[regime]
        for split in SPLITS:
            # Only training-size regime gets a training split.
            if split == "train" and regime != "train_size":
                continue
            n = counts.get(split, 0)
            if n <= 0:
                continue
            # Distinct seed per (problem, regime, split) avoids overlap.
            seed = _seed_for(base_seed, problem_type, regime, split)
            out_dir = data_dir / problem_type / regime / split
            written = write_instances(problem_type, size_cfg, out_dir, n, seed)
            results[(regime, split)] = written
    return results


def _seed_for(base_seed: int, problem_type: str, regime: str, split: str) -> int:
    """Derive a stable per-bucket seed from the base seed."""
    key = f"{problem_type}|{regime}|{split}"
    # Simple stable hash: sum of code points mixed with base_seed.
    h = base_seed
    for ch in key:
        h = (h * 1315423911) ^ ord(ch)
    return h & 0x7FFFFFFF
