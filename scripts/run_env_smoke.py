"""Smoke test for the Stage-2 branching environment.

Loads a handful of instances and rolls out a uniform-random branching policy
under DFS node selection. Logs nodes solved, LP iterations, wall time, and
cumulative dual-bound gain per instance.

Example
-------
    python -m scripts.run_env_smoke \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem set_covering --n-instances 3
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

from rl_bb.envs import env_config_from_dict, make_branching_env
from rl_bb.utils import configure_logging, load_config, resolve_path, set_seed

logger = logging.getLogger("rl_bb.env_smoke")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-policy rollout smoke test.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument(
        "--problem",
        choices=("set_covering", "combinatorial_auction"),
        default="set_covering",
    )
    p.add_argument("--regime", default="train_size")
    p.add_argument("--split", default="train")
    p.add_argument("--n-instances", type=int, default=3)
    return p.parse_args()


def list_instances(cfg: dict, args: argparse.Namespace) -> list[Path]:
    data_root = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    bucket = data_root / args.problem / args.regime / args.split
    if not bucket.exists():
        raise FileNotFoundError(
            f"No instances at {bucket} — run scripts.generate_instances first."
        )
    files = sorted(bucket.glob("instance_*.mps"))
    if not files:
        raise FileNotFoundError(f"Bucket {bucket} is empty.")
    return files[: args.n_instances]


def rollout(env, instance: Path, rng: random.Random) -> dict:
    t0 = time.perf_counter()
    obs, action_set, reward, done, info = env.reset(str(instance))
    cum_reward = 0.0
    steps = 0
    while not done:
        action = rng.choice(list(action_set))
        obs, action_set, reward, done, info = env.step(action)
        cum_reward += float(reward)
        steps += 1
    return {
        "instance": instance.name,
        "steps": steps,
        "cum_reward": cum_reward,
        "wall_time_s": time.perf_counter() - t0,
        "scip_wall_time_s": float(info.get("wall_time", 0.0)),
        "nb_nodes": float(info.get("nb_nodes", 0.0)),
        "lp_iterations": float(info.get("lp_iterations", 0.0)),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    log_dir = resolve_path(cfg["paths"]["log_dir"])
    configure_logging(log_dir=log_dir, filename="env_smoke.log")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    env_cfg = env_config_from_dict(cfg.get("env", {}))
    env = make_branching_env(env_cfg)
    env.seed(seed)

    rng = random.Random(seed)
    instances = list_instances(cfg, args)
    logger.info(
        "Rolling out %d instance(s) from %s/%s/%s",
        len(instances), args.problem, args.regime, args.split,
    )
    for inst in instances:
        result = rollout(env, inst, rng)
        logger.info(
            "%s | steps=%d nodes=%.0f lp_iter=%.0f wall=%.2fs reward=%.3f",
            result["instance"],
            result["steps"],
            result["nb_nodes"],
            result["lp_iterations"],
            result["wall_time_s"],
            result["cum_reward"],
        )


if __name__ == "__main__":
    main()
