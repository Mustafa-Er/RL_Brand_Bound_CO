"""Ecole branching environment: DFS node selection, dual-bound-gain reward.

Two factories share a single base configuration:

* :func:`make_branching_env` — bipartite observation only; for RL training
  and inference where strong-branching scores are not needed.
* :func:`make_expert_env`    — ``(bipartite, sb_scores)`` tuple observation;
  required by :class:`~rl_bb.experts.FSBPolicy` and
  :class:`~rl_bb.experts.RBPolicy`.

The reward, ``DualBoundGain``, is a sign-corrected one-step approximation to
the spec's ``min(DB(left), DB(right)) - DB(parent)`` formula. Ecole's
Branching env does not surface the two children separately, so we use the
change in the *global* dual bound between successive branching calls, which
under DFS is dominated by the most recently opened subtree. A SCIP event
handler could compute the exact spec formula at the cost of additional
boilerplate; we keep the public reward interface stable so this can be
swapped in later without breaking callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ecole


# ---------------------------------------------------------------------------
# DFS enforcement
# ---------------------------------------------------------------------------

_DFS_PRIORITY = 10_000_000

DFS_SCIP_PARAMS: dict[str, int] = {
    "nodeselection/dfs/stdpriority": _DFS_PRIORITY,
    "nodeselection/dfs/memsavepriority": _DFS_PRIORITY,
}


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_SENSE_FACTOR: dict[str, float] = {"minimize": 1.0, "maximize": -1.0}


def _is_nonfinite(x: float) -> bool:
    return x != x or x == float("inf") or x == float("-inf")


class DualBoundGain:
    """Signed dual-bound improvement between consecutive branching calls.

    Ecole reward functions are duck-typed: any object with ``before_reset``
    and ``extract`` methods is accepted. Positive ⇒ the bound moved in the
    improving direction.
    """

    def __init__(self) -> None:
        self._prev: float | None = None
        self._sense: float = 1.0

    def before_reset(self, model) -> None:
        self._prev = None
        sense = model.as_pyscipopt().getObjectiveSense().lower()
        self._sense = _SENSE_FACTOR.get(sense, 1.0)

    def extract(self, model, done: bool) -> float:
        pyscip = model.as_pyscipopt()
        try:
            db = pyscip.getDualbound()
        except Exception:
            return 0.0
        if self._prev is None or _is_nonfinite(db) or _is_nonfinite(self._prev):
            self._prev = db
            return 0.0
        reward = self._sense * (db - self._prev)
        self._prev = db
        return float(reward)


# ---------------------------------------------------------------------------
# Env config + factories
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    time_limit_s: float = 3600.0
    gap_limit: float = 0.0
    extra_scip_params: dict[str, Any] | None = None


def env_config_from_dict(d: dict) -> EnvConfig:
    return EnvConfig(
        time_limit_s=float(d.get("time_limit_s", 3600.0)),
        gap_limit=float(d.get("gap_limit", 0.0)),
        extra_scip_params=d.get("extra_scip_params"),
    )


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
    """Bipartite-only observation; for RL training and inference."""
    return ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        reward_function=DualBoundGain(),
        information_function=_info_functions(),
        scip_params=_build_scip_params(cfg),
    )


def make_expert_env(cfg: EnvConfig) -> ecole.environment.Branching:
    """Tuple observation ``(bipartite, sb_scores)``; for FSB and RB experts."""
    return ecole.environment.Branching(
        observation_function=(
            ecole.observation.NodeBipartite(),
            ecole.observation.StrongBranchingScores(),
        ),
        reward_function=DualBoundGain(),
        information_function=_info_functions(),
        scip_params=_build_scip_params(cfg),
    )
