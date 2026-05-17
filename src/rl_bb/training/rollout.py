"""Rollout collection for PPO.

A *rollout* is one full episode of branching decisions on a single MILP
instance. At each branching call we record the bipartite observation, the
action sampled from the policy, the candidate set, the log-probability and
value the network assigned, and the reward returned by the env on the
following step.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rl_bb.models import GCNN
from rl_bb.models.obs_to_tensors import obs_to_tensors

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    observation: object
    action: int
    action_set: list[int]
    log_prob: float
    value: float
    reward: float


@dataclass
class Trajectory:
    steps: list[StepRecord]
    n_nodes: float
    wall_time_s: float
    sum_reward: float


def _sample_action(model: GCNN, bipartite, action_set, device) -> tuple[int, float, float]:
    """Sample an action from the masked softmax; return (action, log_prob, value)."""
    tensors = obs_to_tensors(bipartite).to(device)
    logits, value = model.forward_with_mask(tensors, list(action_set))
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # Numerically stable categorical sampling restricted to finite-logit support.
    idx = torch.multinomial(probs, num_samples=1)
    action = int(idx.item())
    return action, float(log_probs[idx].item()), float(value.item())


def collect_trajectory(env, instance_path: Path, model: GCNN, device: str) -> Trajectory:
    """Run ``model`` on one instance and return the full trajectory."""
    model.eval()
    t0 = time.perf_counter()
    obs, action_set, _r, done, info = env.reset(str(instance_path))
    steps: list[StepRecord] = []
    sum_reward = 0.0
    while not done:
        bipartite = obs[0] if isinstance(obs, tuple) else obs
        with torch.no_grad():
            action, log_prob, value = _sample_action(model, bipartite, action_set, device)
        next_obs, next_action_set, reward, done, info = env.step(action)
        steps.append(StepRecord(
            observation=bipartite,
            action=action,
            action_set=[int(v) for v in action_set],
            log_prob=log_prob,
            value=value,
            reward=float(reward),
        ))
        sum_reward += float(reward)
        obs, action_set = next_obs, next_action_set
    return Trajectory(
        steps=steps,
        n_nodes=float(info.get("nb_nodes", 0.0)),
        wall_time_s=time.perf_counter() - t0,
        sum_reward=sum_reward,
    )
