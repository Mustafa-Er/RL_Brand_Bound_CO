"""Branching policies usable inside the Branching env.

Roles:

* :class:`RBPolicy` — the imitation-learning expert. Stage 2 collects
  demonstrations by rolling this policy out on training instances and trains
  the GCNN on its decisions.
* :class:`FSBPolicy` — Stage 4 evaluation baseline (Full Strong Branching).
* :class:`RandomPolicy` — sanity-check baseline.

All three share the signature ``act(observation, action_set, model) -> int``.
"""
from __future__ import annotations

import random
from typing import Protocol


class Policy(Protocol):
    def act(self, observation, action_set, model) -> int: ...
    def reset(self) -> None: ...


class RandomPolicy:
    """Uniform random over LP-fractional branching candidates."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, observation, action_set, model) -> int:
        return int(self._rng.choice(list(action_set)))


class FSBPolicy:
    """Argmax over Full Strong Branching scores.

    Expects the expert env's tuple observation ``(bipartite, sb_scores)``.
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
    running pseudocost score.
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
        return int(best_idx if best_idx is not None else next(iter(action_set)))


# ---------------------------------------------------------------------------
# Defensive PySCIPOpt accessors (signatures vary across versions)
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
