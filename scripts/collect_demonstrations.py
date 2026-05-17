"""CLI to roll out the Reliability Branching expert and dump trajectories.

Reliability Branching is the *only* demonstration source used to pretrain
the policy. FSB and random policies exist as evaluation baselines (see
:mod:`rl_bb.experts.policies`); they are not collected here.

Example
-------
    python -m scripts.collect_demonstrations \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem combinatorial_auction --split train
"""
from __future__ import annotations

import argparse
import logging

from rl_bb.envs import env_config_from_dict
from rl_bb.experts import RBPolicy, collect_many
from rl_bb.utils import configure_logging, load_config, resolve_path, set_seed

logger = logging.getLogger("rl_bb.collect")

PROBLEMS = ("set_covering", "combinatorial_auction")
EXPERT_NAME = "rb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Reliability Branching demonstrations.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--regime", default="train_size")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit to first N instances in the bucket (default: all).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    log_dir = resolve_path(cfg["paths"]["log_dir"])
    configure_logging(log_dir=log_dir, filename="collect_demonstrations.log")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    data_root = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    bucket = data_root / args.problem / args.regime / args.split
    if not bucket.exists():
        raise FileNotFoundError(
            f"No instances at {bucket} — run scripts.generate_instances first."
        )
    instances = sorted(bucket.glob("instance_*.mps"))
    if args.max_instances is not None:
        instances = instances[: args.max_instances]
    if not instances:
        raise FileNotFoundError(f"No .mps files in {bucket}.")

    env_cfg = env_config_from_dict(cfg.get("env", {}))
    policy = RBPolicy()

    out_dir = (
        resolve_path(cfg["paths"]["data_dir"])
        / cfg["experiment"]["name"]
        / "demonstrations"
        / args.problem
        / args.regime
        / args.split
        / EXPERT_NAME
    )
    logger.info(
        "Collecting %d RB demos for %s/%s/%s -> %s",
        len(instances), args.problem, args.regime, args.split, out_dir,
    )
    written = collect_many(instances, policy, env_cfg, out_dir, seed=seed)
    logger.info("Done: %d trajectories written.", written)


if __name__ == "__main__":
    main()
