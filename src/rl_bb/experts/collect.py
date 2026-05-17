"""Demonstration collection: roll out a policy and dump (obs, action) pairs.

In this project the only caller is :mod:`scripts.collect_demonstrations`,
which always passes :class:`~rl_bb.experts.policies.RBPolicy`. The
``collect_many`` helper is policy-agnostic for testability, but the
on-disk demonstration tree under ``data/<exp>/demonstrations/...`` only
contains RB rollouts.

Output layout::

    <out_dir>/<problem>/<regime>/<split>/rb/instance_XXXX.pkl

Each pickle is a dict with the keys::

    {
        "instance": "<filename>.mps",
        "expert": "RBPolicy",                   # type(policy).__name__
        "observations": [bipartite_obs, ...],   # one per branching decision
        "actions":      [int, ...],             # SCIP variable index
        "action_sets":  [array, ...],           # candidate sets per step
        "n_nodes":      float,
        "lp_iterations": float,
        "wall_time_s":  float,
    }

Bipartite observations are stored as plain Python objects (numpy arrays
inside an Ecole namedtuple-like struct); pickling handles them transparently.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Iterable

from rl_bb.envs import EnvConfig, make_expert_env

logger = logging.getLogger(__name__)


def collect_one(env, instance_path: Path, policy) -> dict:
    """Roll out ``policy`` on a single instance and return the trajectory dict."""
    policy.reset()
    t0 = time.perf_counter()
    observation, action_set, _reward, done, info = env.reset(str(instance_path))

    bipartite_traj: list = []
    action_traj: list[int] = []
    action_set_traj: list = []

    while not done:
        bipartite, _sb_scores = observation
        action = int(policy.act(observation, action_set, env.model))
        bipartite_traj.append(bipartite)
        action_traj.append(action)
        action_set_traj.append(action_set)
        observation, action_set, _reward, done, info = env.step(action)

    return {
        "instance": instance_path.name,
        "expert": type(policy).__name__,
        "observations": bipartite_traj,
        "actions": action_traj,
        "action_sets": action_set_traj,
        "n_nodes": float(info.get("nb_nodes", 0.0)),
        "lp_iterations": float(info.get("lp_iterations", 0.0)),
        "wall_time_s": time.perf_counter() - t0,
    }


def collect_many(
    instances: Iterable[Path],
    policy,
    env_cfg: EnvConfig,
    out_dir: Path,
    seed: int = 0,
) -> int:
    """Collect demonstrations for every instance; write one pickle each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_expert_env(env_cfg)
    env.seed(seed)
    written = 0
    for inst in instances:
        traj = collect_one(env, inst, policy)
        target = out_dir / f"{Path(inst).stem}.pkl"
        with open(target, "wb") as f:
            pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "  %s | decisions=%d nodes=%.0f wall=%.2fs",
            inst.name, len(traj["actions"]), traj["n_nodes"], traj["wall_time_s"],
        )
        written += 1
    return written
