"""Per-instance evaluation runner.

The runner drives a ``Policy`` through one instance under a configured env,
records wall-clock time, B&B node count, LP iterations, and the dual
integral (when Ecole exposes it). It is intentionally minimal: aggregation
over seeds and instances lives in :mod:`rl_bb.eval.aggregate`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    instance: str
    policy: str
    regime: str
    split: str
    seed: int
    wall_time_s: float          # Python-side wall time
    scip_wall_time_s: float     # SCIP-reported solving time
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
