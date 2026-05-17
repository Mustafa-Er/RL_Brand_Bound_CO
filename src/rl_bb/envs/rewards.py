"""Reward functions for the branching MDP.

The spec sets the per-step reward to the *local LP gain*::

    r_t = min(DB(left_child), DB(right_child)) - DB(parent)

For minimization problems SCIP keeps a non-decreasing dual bound, so the raw
expression is non-negative. For maximization problems the dual bound is
non-increasing, making the raw expression non-positive. To keep "improvement
is positive" in both senses, we multiply by the objective-sense factor
(+1 for minimize, -1 for maximize).

Implementation note
-------------------
Ecole's :class:`ecole.environment.Branching` env hands control back to the
caller only at branching decisions, not directly after each child node has
been processed. We therefore approximate the local LP gain by the change in
the *global dual bound* between two successive branching calls under DFS
node selection — which, in DFS, is dominated by progress in the most recently
opened subtree. The exact "min over the two newly created children" reading
of the spec would require a SCIP event handler that snapshots both child
LP relaxations; this can be layered in later without changing the reward's
public interface.
"""
from __future__ import annotations

_SENSE_FACTOR: dict[str, float] = {"minimize": 1.0, "maximize": -1.0}


class DualBoundGain:
    """Signed dual-bound improvement between consecutive branching calls.

    Ecole reward functions are duck-typed: any object with ``before_reset`` and
    ``extract`` methods is accepted. Returns 0 on the first call after reset
    (no previous bound to compare). Positive ⇒ the bound moved in the
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
            # SCIP may refuse to expose the bound in transient states
            # (e.g. immediately after solving). Treat as no signal.
            return 0.0

        if self._prev is None or _is_nonfinite(db) or _is_nonfinite(self._prev):
            self._prev = db
            return 0.0

        reward = self._sense * (db - self._prev)
        self._prev = db
        return float(reward)


def _is_nonfinite(x: float) -> bool:
    return x != x or x == float("inf") or x == float("-inf")
