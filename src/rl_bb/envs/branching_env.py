"""Ecole branching environment factory.

This module produces a configured ``ecole.environment.Branching`` instance
with:

* DFS node selection forced via SCIP parameters (see :mod:`rl_bb.envs.dfs`)
* The Gasse et al. 2019 bipartite node observation
* The :class:`~rl_bb.envs.rewards.DualBoundGain` reward
* Solver time and gap limits driven by config

The factory does not load any instance; callers iterate over instance files
and call :meth:`ecole.environment.Branching.reset` themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ecole

from rl_bb.envs.dfs import DFS_SCIP_PARAMS
from rl_bb.envs.rewards import DualBoundGain


@dataclass(frozen=True)
class EnvConfig:
    time_limit_s: float = 3600.0
    gap_limit: float = 0.0
    extra_scip_params: dict[str, Any] | None = None


def _build_scip_params(cfg: EnvConfig) -> dict[str, Any]:
    params: dict[str, Any] = dict(DFS_SCIP_PARAMS)
    params["limits/time"] = float(cfg.time_limit_s)
    if cfg.gap_limit and cfg.gap_limit > 0:
        params["limits/gap"] = float(cfg.gap_limit)
    if cfg.extra_scip_params:
        params.update(cfg.extra_scip_params)
    return params


def _info_functions() -> dict:
    info = {
        "nb_nodes": ecole.reward.NNodes().cumsum(),
        "lp_iterations": ecole.reward.LpIterations().cumsum(),
        "wall_time": ecole.reward.SolvingTime().cumsum(),
    }
    # Ecole renamed the dual-integral reward across versions; try a few names.
    for name in ("DualIntegral", "PrimalDualIntegral", "PrimalIntegral"):
        cls = getattr(ecole.reward, name, None)
        if cls is not None:
            info["dual_integral"] = cls().cumsum()
            break
    return info


def make_branching_env(cfg: EnvConfig) -> ecole.environment.Branching:
    """Construct a Branching env with DFS, bipartite obs, and DB-gain reward.

    Intended for RL training and inference; observation is the bipartite
    representation alone.
    """
    return ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        reward_function=DualBoundGain(),
        information_function=_info_functions(),
        scip_params=_build_scip_params(cfg),
    )


def make_expert_env(cfg: EnvConfig) -> ecole.environment.Branching:
    """Branching env whose observation is ``(bipartite, sb_scores)``.

    Used by FSB and RB experts (which need strong-branching scores to score
    candidates) and by demonstration collection (so the stored observation
    matches the RL env's bipartite tensor).
    """
    return ecole.environment.Branching(
        observation_function=(
            ecole.observation.NodeBipartite(),
            ecole.observation.StrongBranchingScores(),
        ),
        reward_function=DualBoundGain(),
        information_function=_info_functions(),
        scip_params=_build_scip_params(cfg),
    )


def env_config_from_dict(d: dict) -> EnvConfig:
    return EnvConfig(
        time_limit_s=float(d.get("time_limit_s", 3600.0)),
        gap_limit=float(d.get("gap_limit", 0.0)),
        extra_scip_params=d.get("extra_scip_params"),
    )
