"""Stage 4 — Evaluation: Random vs FSB vs PPO across all size regimes.

For each (policy, regime, seed, instance) we record wall-clock time, SCIP
solving time, B&B node count, LP iterations, and the dual integral.

Outputs (under ``logs/<experiment.name>/``):

* ``eval_detail.csv``  — one row per (instance, policy, seed)
* ``eval_summary.csv`` — mean / std per (policy, regime, split)
* ``eval_summary.json`` — same summary as JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rl_bb.envs import env_config_from_dict, make_branching_env, make_expert_env
from rl_bb.experts import FSBPolicy, RandomPolicy
from rl_bb.model import RLPolicy, load_gcnn
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_device,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.stage_4")

PROBLEMS = ("set_covering", "combinatorial_auction")
REGIMES = ("train_size", "transfer_medium", "transfer_large")


# ===========================================================================
# Per-instance evaluation
# ===========================================================================

@dataclass
class InstanceResult:
    instance: str
    policy: str
    regime: str
    split: str
    seed: int
    wall_time_s: float
    scip_wall_time_s: float
    n_nodes: float
    lp_iterations: float
    dual_integral: float | None
    n_decisions: int

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["dual_integral"] is None:
            d["dual_integral"] = ""
        return d


def evaluate_on_instance(
    env,
    instance_path: Path,
    policy,
    *,
    policy_name: str,
    regime: str,
    split: str,
    seed: int,
) -> InstanceResult:
    env.seed(seed)
    policy.reset()
    t0 = time.perf_counter()
    obs, action_set, _r, done, info = env.reset(str(instance_path))
    n_decisions = 0
    while not done:
        action = int(policy.act(obs, action_set, env.model))
        obs, action_set, _r, done, info = env.step(action)
        n_decisions += 1
    wall = time.perf_counter() - t0

    dual_integral = info.get("dual_integral")
    return InstanceResult(
        instance=instance_path.name,
        policy=policy_name,
        regime=regime,
        split=split,
        seed=seed,
        wall_time_s=wall,
        scip_wall_time_s=float(info.get("wall_time", 0.0)),
        n_nodes=float(info.get("nb_nodes", 0.0)),
        lp_iterations=float(info.get("lp_iterations", 0.0)),
        dual_integral=None if dual_integral is None else float(dual_integral),
        n_decisions=n_decisions,
    )


# ===========================================================================
# Aggregation
# ===========================================================================

METRIC_FIELDS = (
    "wall_time_s", "scip_wall_time_s", "n_nodes",
    "lp_iterations", "dual_integral", "n_decisions",
)


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def _safe_std(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def aggregate(results: Iterable[InstanceResult]) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[InstanceResult]] = defaultdict(list)
    for r in results:
        buckets[(r.policy, r.regime, r.split)].append(r)

    rows: list[dict] = []
    for (policy, regime, split), group in sorted(buckets.items()):
        row = {"policy": policy, "regime": regime, "split": split, "n": len(group)}
        for field in METRIC_FIELDS:
            values = [getattr(g, field) for g in group]
            row[f"{field}_mean"] = _safe_mean(values)
            row[f"{field}_std"] = _safe_std(values)
        rows.append(row)
    return rows


def write_results(per_instance: list[InstanceResult], summary: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / "eval_detail.csv"
    summary_path = out_dir / "eval_summary.csv"

    with open(detail, "w", newline="", encoding="utf-8") as f:
        if per_instance:
            writer = csv.DictWriter(f, fieldnames=list(per_instance[0].as_dict().keys()))
            writer.writeheader()
            for r in per_instance:
                writer.writerow(r.as_dict())

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return detail, summary_path


# ===========================================================================
# Top-level driver
# ===========================================================================

def run_stage_4(
    cfg: dict,
    problem: str,
    split: str = "test",
    policies: tuple[str, ...] = ("random", "fsb", "ppo"),
    max_instances: int | None = None,
) -> tuple[list[InstanceResult], list[dict]]:
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))

    data_root = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    ckpt_dir = resolve_path(cfg["paths"]["ckpt_dir"]) / cfg["experiment"]["name"]
    env_cfg = env_config_from_dict(cfg.get("env", {}))
    seeds = list(cfg.get("eval", {}).get("seeds", [0, 1, 2, 3, 4]))
    logger.info("seeds=%s device=%s policies=%s", seeds, device, policies)

    def policies_iter():
        if "random" in policies:
            yield "random", make_branching_env, lambda: RandomPolicy(seed=seed)
        if "fsb" in policies:
            yield "fsb", make_expert_env, lambda: FSBPolicy()
        if "ppo" in policies:
            ckpt = ckpt_dir / "ppo_best.pt"
            if not ckpt.exists():
                logger.warning("No %s; falling back to ppo_latest.pt", ckpt)
                ckpt = ckpt_dir / "ppo_latest.pt"
            if not ckpt.exists():
                logger.warning("No PPO checkpoint available; skipping PPO.")
                return
            model = load_gcnn(ckpt, device=device)
            model.eval()
            yield "ppo", make_branching_env, lambda: RLPolicy(model, device=device, stochastic=False)

    per_instance: list[InstanceResult] = []
    for policy_name, env_factory, policy_factory in policies_iter():
        env = env_factory(env_cfg)
        for regime in REGIMES:
            bucket = data_root / problem / regime / split
            if not bucket.exists():
                logger.warning("Skip %s/%s/%s — directory missing.", problem, regime, split)
                continue
            instances = sorted(bucket.glob("instance_*.mps"))
            if max_instances is not None:
                instances = instances[:max_instances]
            if not instances:
                logger.warning("No instances in %s.", bucket)
                continue
            for s in seeds:
                policy = policy_factory()
                for inst in instances:
                    per_instance.append(evaluate_on_instance(
                        env, inst, policy,
                        policy_name=policy_name, regime=regime, split=split, seed=s,
                    ))
            logger.info("done: policy=%s regime=%s instances=%d seeds=%d",
                        policy_name, regime, len(instances), len(seeds))

    summary = aggregate(per_instance)
    return per_instance, summary


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — Random / FSB / PPO evaluation.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--policies",
        nargs="+",
        default=["random", "fsb", "ppo"],
        choices=("random", "fsb", "ppo"),
    )
    p.add_argument("--max-instances", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    log_dir = resolve_path(cfg["paths"]["log_dir"]) / cfg["experiment"]["name"]
    configure_logging(log_dir=log_dir, filename="stage_4.log")

    per_instance, summary = run_stage_4(
        cfg, problem=args.problem, split=args.split,
        policies=tuple(args.policies), max_instances=args.max_instances,
    )
    detail_path, summary_path = write_results(per_instance, summary, log_dir)
    logger.info("Wrote %d detail rows -> %s", len(per_instance), detail_path)
    logger.info("Wrote %d summary rows -> %s", len(summary), summary_path)

    for row in summary:
        di = row["dual_integral_mean"]
        di_str = "n/a" if di != di else f"{di:.2f}±{row['dual_integral_std']:.2f}"
        logger.info(
            "%-6s %-15s %-5s n=%2d | nodes=%7.1f±%-6.1f | wall=%5.2fs±%-4.2f | DI=%s",
            row["policy"], row["regime"], row["split"], row["n"],
            row["n_nodes_mean"], row["n_nodes_std"],
            row["wall_time_s_mean"], row["wall_time_s_std"], di_str,
        )


if __name__ == "__main__":
    main()
