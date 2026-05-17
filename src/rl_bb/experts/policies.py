"""Branching policies usable inside the Branching env.

All policies share the same call signature::

    action = policy.act(observation, action_set, model)

Roles in this project:

* :class:`RBPolicy` — *the* imitation-learning expert. Pretraining
  demonstrations are produced by rolling this policy out on training
  instances (Stage 3 + Stage 5).
* :class:`FSBPolicy` — Stage 7 evaluation baseline (Full Strong Branching).
  Not used for demonstration collection.
* :class:`RandomPolicy` — sanity-check baseline.

Observation type depends on the env factory used:

* :func:`rl_bb.envs.make_branching_env` returns the bipartite observation
  alone — suitable for RL inference (:class:`RandomPolicy` works here).
* :func:`rl_bb.envs.make_expert_env` returns a tuple
  ``(bipartite, strong_branching_scores)`` — needed by :class:`FSBPolicy`
  and :class:`RBPolicy`.

:class:`RBPolicy` implements reliability pseudocost branching
(Achterberg 2007): for each candidate variable, do full strong branching
until it has been branched ``reliability`` times in each direction, then
switch to SCIP's running pseudocost score.
"""
from __future__ import annotations

import random
from typing import Protocol


class Policy(Protocol):
    """Anything that maps (observation, action_set, model) -> action."""

    def act(self, observation, action_set, model) -> int: ...

    def reset(self) -> None: ...


class RandomPolicy:
    """Uniform random over the LP-fractional branching candidates."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, observation, action_set, model) -> int:
        return int(self._rng.choice(list(action_set)))


class FSBPolicy:
    """Argmax over Full Strong Branching scores.

    Expects the expert env's tuple observation ``(bipartite, sb_scores)``;
    ``sb_scores`` is an array indexed by SCIP variable position.
    """

    def reset(self) -> None:
        pass

    def act(self, observation, action_set, model) -> int:
        _, sb_scores = observation
        return int(max(action_set, key=lambda v: float(sb_scores[v])))


class RBPolicy:
    """Reliability pseudocost branching (Achterberg 2007).

    For each candidate, count up- and down-branching observations via SCIP's
    pseudocost machinery. If the minimum count is below ``reliability``, score
    the candidate with its strong-branching score; otherwise use SCIP's
    running pseudocost score. Pick the argmax.
    """

    def __init__(self, reliability: int = 4) -> None:
        self.reliability = int(reliability)

    def reset(self) -> None:
        pass

    def act(self, observation, action_set, model) -> int:
        _, sb_scores = observation
        pyscip = model.as_pyscipopt()
        scip_vars = pyscip.getVars()

        best_idx = None
        best_score = -float("inf")
        for v_idx in action_set:
            var = scip_vars[int(v_idx)]
            n_down, n_up = _pseudocost_counts(pyscip, var)
            if min(n_down, n_up) < self.reliability:
                score = float(sb_scores[int(v_idx)])
            else:
                score = _pseudocost_score(pyscip, var)
            if score > best_score:
                best_score = score
                best_idx = int(v_idx)
        # Fall back to first candidate if all scores were -inf (shouldn't happen).
        return int(best_idx if best_idx is not None else next(iter(action_set)))


# ---------------------------------------------------------------------------
# Defensive PySCIPOpt accessors
# ---------------------------------------------------------------------------

def _pseudocost_counts(model, var) -> tuple[float, float]:
    """Return (down, up) pseudocost observation counts; 0 if unavailable."""
    for name in ("getVarPseudocostCountCurrentRun", "getVarPseudocostCount"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            return float(fn(var, 0)), float(fn(var, 1))
        except TypeError:
            try:
                return float(fn(var, downwards=True)), float(fn(var, downwards=False))
            except Exception:
                continue
        except Exception:
            continue
    return 0.0, 0.0


def _pseudocost_score(model, var) -> float:
    """Return a pseudocost-based variable score; 0 if unavailable."""
    for name in ("getVarPseudocostScoreCurrentRun", "getVarPseudocostScore"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            return float(fn(var))
        except Exception:
            continue
    return 0.0
