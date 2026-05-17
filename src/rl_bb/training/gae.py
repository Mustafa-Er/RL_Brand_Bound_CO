"""Generalized Advantage Estimation (Schulman et al. 2016) for the branching MDP.

The spec defines a B&B-adapted TD error::

    δ_t = r_t + γ · (V(s_t^left) + V(s_t^right)) / 2 − V(s_t)

Ecole's Branching env does not surface the two children's value estimates
separately to the agent — it returns control at the next branching node,
which under DFS is one descendant in the open subtree. We therefore
approximate the spec with the standard one-step TD error::

    δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

This mirrors the local-LP-gain reward approximation already documented in
``rl_bb.envs.rewards`` and stays consistent across the two estimators. The
public interface accepts ``last_value`` so callers can bootstrap from an
unterminated trajectory if needed; episodes here always reach SCIP's
``done=True`` so the default ``0.0`` is correct.
"""
from __future__ import annotations

from typing import Iterable


def compute_gae(
    rewards: Iterable[float],
    values: Iterable[float],
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Compute per-step advantages and value targets.

    Returns ``(advantages, returns)``. ``returns[t] = advantages[t] + values[t]``
    is the standard "return" used as the value-head regression target.
    """
    rewards = list(rewards)
    values = list(values)
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    advantages = [0.0] * len(rewards)
    next_value = float(last_value)
    next_adv = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        adv = delta + gamma * lam * next_adv
        advantages[t] = adv
        next_value = values[t]
        next_adv = adv
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
