"""Stage 1 — Generate MILP instances and (optionally) smoke-test the env.

Per problem type and size regime, produces ``.mps`` files for the train,
val, and test splits. The training split is only populated for the
``train_size`` regime; ``val`` and ``test`` are populated for all regimes so
Stage 4 can measure transfer.

Layout::

    data/<experiment.name>/<problem>/<regime>/<split>/instance_XXXX.mps

Run::

    python -m rl_bb.stage_1_instances \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem combinatorial_auction
    # add --smoke to also roll a random policy through one instance
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import ecole

from rl_bb.envs import env_config_from_dict, make_branching_env
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.stage_1")

PROBLEMS = ("set_covering", "combinatorial_auction")
REGIMES = ("train_size", "transfer_medium", "transfer_large")
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Generators
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


def _set_cover_generator(size: SetCoverSize, seed: int):
    return ecole.instance.SetCoverGenerator(
        n_rows=size.n_rows,
        n_cols=size.n_cols,
        density=size.density,
        rng=ecole.RandomGenerator(seed),
    )


def _auction_generator(size: AuctionSize, seed: int):
    return ecole.instance.CombinatorialAuctionGenerator(
        n_items=size.n_items,
        n_bids=size.n_bids,
        rng=ecole.RandomGenerator(seed),
    )


def make_generator(problem_type: str, size_cfg: dict, seed: int):
    if problem_type == "set_covering":
        return _set_cover_generator(SetCoverSize(**size_cfg), seed)
    if problem_type == "combinatorial_auction":
        return _auction_generator(AuctionSize(**size_cfg), seed)
    raise ValueError(f"Unknown problem_type: {problem_type!r}")


def _iter_models(generator, n: int) -> Iterator:
    for i in range(n):
        yield i, next(generator)


def write_instances(
    problem_type: str,
    size_cfg: dict,
    out_dir: Path,
    n: int,
    seed: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(problem_type, size_cfg, seed)
    written = 0
    for i, model in _iter_models(generator, n):
        target = out_dir / f"instance_{i:04d}.mps"
        model.as_pyscipopt().writeProblem(str(target))
        written += 1
    logger.info("Wrote %d instances to %s", written, out_dir)
    return written


def _seed_for(base_seed: int, problem_type: str, regime: str, split: str) -> int:
    key = f"{problem_type}|{regime}|{split}"
    h = base_seed
    for ch in key:
        h = (h * 1315423911) ^ ord(ch)
    return h & 0x7FFFFFFF


def generate_all(
    problem_type: str,
    sizes: dict,
    counts: dict,
    data_dir: Path,
    base_seed: int,
    regimes: tuple[str, ...] = REGIMES,
) -> dict[tuple[str, str], int]:
    """Generate every (regime, split) combination defined by ``counts``."""
    results: dict[tuple[str, str], int] = {}
    for regime in regimes:
        if regime not in sizes:
            raise KeyError(f"Missing size config for regime {regime!r}")
        size_cfg = sizes[regime]
        for split in SPLITS:
            if split == "train" and regime != "train_size":
                continue
            n = counts.get(split, 0)
            if n <= 0:
                continue
            seed = _seed_for(base_seed, problem_type, regime, split)
            out_dir = data_dir / problem_type / regime / split
            results[(regime, split)] = write_instances(
                problem_type, size_cfg, out_dir, n, seed
            )
    return results


# ---------------------------------------------------------------------------
# Env smoke test (replaces former scripts.run_env_smoke)
# ---------------------------------------------------------------------------

def smoke_random_rollouts(env, instances: list[Path], rng: random.Random) -> None:
    for inst in instances:
        t0 = time.perf_counter()
        obs, action_set, _r, done, info = env.reset(str(inst))
        cum_reward = 0.0
        steps = 0
        while not done:
            action = rng.choice(list(action_set))
            obs, action_set, reward, done, info = env.step(action)
            cum_reward += float(reward)
            steps += 1
        logger.info(
            "smoke | %s | steps=%d nodes=%.0f wall=%.2fs reward=%.3f",
            inst.name, steps, float(info.get("nb_nodes", 0.0)),
            time.perf_counter() - t0, cum_reward,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stage_1(cfg: dict, problem: str | None = None, smoke: bool = False) -> dict:
    """Programmatic entry point used by the notebook."""
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    data_dir = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    counts = cfg["instances"]["counts"]

    problems = (problem,) if problem else PROBLEMS
    summary: dict = {}
    for p in problems:
        if p not in cfg["instances"]:
            logger.warning("No size config for problem %s, skipping.", p)
            continue
        logger.info("Generating problem=%s under %s", p, data_dir)
        summary[p] = generate_all(
            problem_type=p,
            sizes=cfg["instances"][p],
            counts=counts,
            data_dir=data_dir,
            base_seed=seed,
        )

    if smoke:
        env_cfg = env_config_from_dict(cfg.get("env", {}))
        env = make_branching_env(env_cfg)
        env.seed(seed)
        rng = random.Random(seed)
        for p in problems:
            bucket = data_dir / p / "train_size" / "train"
            insts = sorted(bucket.glob("instance_*.mps"))[:3]
            if insts:
                logger.info("Smoke rollout: %s/train_size/train (%d instances)", p, len(insts))
                smoke_random_rollouts(env, insts, rng)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 — instance generation + env smoke.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, default=None)
    p.add_argument("--smoke", action="store_true", help="Random-policy rollout after generation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    log_dir = resolve_path(cfg["paths"]["log_dir"]) / cfg["experiment"]["name"]
    configure_logging(log_dir=log_dir, filename="stage_1.log")
    t0 = time.perf_counter()
    summary = run_stage_1(cfg, problem=args.problem, smoke=args.smoke)
    total = sum(sum(v.values()) for v in summary.values())
    logger.info("Done: %d instances in %.1fs.", total, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
