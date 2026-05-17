"""End-to-end evaluation: Random, FSB, and PPO across all size regimes.

For each (policy, regime, seed) the script runs every test-split instance and
records wall-clock time, B&B node count, LP iterations, and the dual
integral. Per-instance rows go to ``logs/<exp>/eval_detail.csv``; the
mean/std summary goes to ``eval_summary.csv`` and ``eval_summary.json``.

Example
-------
    python -m scripts.eval \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem combinatorial_auction
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rl_bb.envs import env_config_from_dict, make_branching_env, make_expert_env
from rl_bb.eval import InstanceResult, aggregate, evaluate_on_instance, write_results
from rl_bb.experts import FSBPolicy, RandomPolicy
from rl_bb.models import RLPolicy
from rl_bb.training import load_pretrained_gcnn
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_device,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.eval")

PROBLEMS = ("set_covering", "combinatorial_auction")
REGIMES = ("train_size", "transfer_medium", "transfer_large")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Random, FSB, and PPO policies.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument(
        "--split", default="test",
        help="Instance split to evaluate on (default: test).",
    )
    p.add_argument(
        "--policies",
        nargs="+",
        default=["random", "fsb", "ppo"],
        choices=("random", "fsb", "ppo"),
        help="Subset of policies to evaluate.",
    )
    p.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap instances per bucket (useful for quick smoke runs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    log_dir = resolve_path(cfg["paths"]["log_dir"]) / cfg["experiment"]["name"]
    configure_logging(log_dir=log_dir, filename="eval.log")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))

    data_root = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    ckpt_dir = resolve_path(cfg["paths"]["ckpt_dir"]) / cfg["experiment"]["name"]
    env_cfg = env_config_from_dict(cfg.get("env", {}))
    seeds = list(cfg.get("eval", {}).get("seeds", [0, 1, 2, 3, 4]))
    logger.info("seeds: %s, device: %s", seeds, device)

    # ------------------------------------------------------------------
    # Policy factories: (name, env_factory, policy_factory)
    # The env factory varies because FSB needs strong-branching scores
    # in the observation, while Random and PPO only need the bipartite.
    # ------------------------------------------------------------------
    def policies_to_run():
        if "random" in args.policies:
            yield "random", make_branching_env, lambda: RandomPolicy(seed=seed)
        if "fsb" in args.policies:
            yield "fsb", make_expert_env, lambda: FSBPolicy()
        if "ppo" in args.policies:
            ckpt = ckpt_dir / "ppo_best.pt"
            if not ckpt.exists():
                logger.warning("No PPO checkpoint at %s — falling back to ppo_latest.pt", ckpt)
                ckpt = ckpt_dir / "ppo_latest.pt"
            if not ckpt.exists():
                logger.warning("No PPO checkpoint available; skipping PPO.")
                return
            model = load_pretrained_gcnn(ckpt, device=device)
            model.eval()
            yield "ppo", make_branching_env, lambda: RLPolicy(model, device=device, stochastic=False)

    per_instance: list[InstanceResult] = []
    for policy_name, env_factory, policy_factory in policies_to_run():
        env = env_factory(env_cfg)
        for regime in REGIMES:
            bucket = data_root / args.problem / regime / args.split
            if not bucket.exists():
                logger.warning("Skip %s/%s/%s — directory missing.", args.problem, regime, args.split)
                continue
            instances = sorted(bucket.glob("instance_*.mps"))
            if args.max_instances is not None:
                instances = instances[: args.max_instances]
            if not instances:
                logger.warning("No instances in %s.", bucket)
                continue

            for s in seeds:
                policy = policy_factory()
                for inst in instances:
                    res = evaluate_on_instance(
                        env, inst, policy,
                        policy_name=policy_name,
                        regime=regime,
                        split=args.split,
                        seed=s,
                    )
                    per_instance.append(res)
                    logger.debug(
                        "%-6s | %-15s | seed=%d | %s | nodes=%.0f wall=%.2fs",
                        policy_name, regime, s, inst.name, res.n_nodes, res.wall_time_s,
                    )
            logger.info("done: policy=%s regime=%s instances=%d seeds=%d",
                        policy_name, regime, len(instances), len(seeds))

    summary = aggregate(per_instance)
    detail_path, summary_path = write_results(per_instance, summary, log_dir)
    logger.info("Wrote %d detail rows -> %s", len(per_instance), detail_path)
    logger.info("Wrote %d summary rows -> %s", len(summary), summary_path)

    # Pretty-print summary for quick reading.
    for row in summary:
        logger.info(
            "%-6s %-15s %-5s n=%2d | nodes=%7.1f±%-6.1f | wall=%5.2fs±%-4.2f | DI=%s",
            row["policy"], row["regime"], row["split"], row["n"],
            row["n_nodes_mean"], row["n_nodes_std"],
            row["wall_time_s_mean"], row["wall_time_s_std"],
            "n/a" if row["dual_integral_mean"] != row["dual_integral_mean"]  # NaN check
            else f"{row['dual_integral_mean']:.2f}±{row['dual_integral_std']:.2f}",
        )


if __name__ == "__main__":
    main()
