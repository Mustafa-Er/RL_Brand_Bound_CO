"""PPO training with GAE on the branching environment.

The model is warm-started from the Stage-5 imitation-learning checkpoint;
:func:`run_ppo` raises if it does not find one. Each iteration:

1. Roll out the current policy on ``rollouts_per_iter`` training instances.
2. Compute GAE advantages and value targets per trajectory.
3. Concatenate all transitions into one big buffer.
4. Run ``update_epochs`` passes over shuffled minibatches, applying the
   clipped PPO surrogate + value MSE + entropy bonus.

Checkpoints are saved per-iteration to ``<ckpt_dir>/ppo_latest.pt`` and the
best-by-mean-reward to ``ppo_best.pt``.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rl_bb.envs import EnvConfig, make_branching_env
from rl_bb.models import GCNN
from rl_bb.training.dataset import BCSample, collate_bipartite
from rl_bb.training.gae import compute_gae
from rl_bb.training.pretrain import load_pretrained_gcnn
from rl_bb.training.rollout import StepRecord, Trajectory, collect_trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Buffer + flat tensors
# ---------------------------------------------------------------------------

@dataclass
class PPOSample:
    sample: BCSample        # observation/action/action_set
    log_prob: float
    advantage: float
    return_: float


def trajectories_to_samples(
    trajectories: list[Trajectory], gamma: float, lam: float
) -> list[PPOSample]:
    out: list[PPOSample] = []
    for traj in trajectories:
        if not traj.steps:
            continue
        adv, ret = compute_gae(
            rewards=[s.reward for s in traj.steps],
            values=[s.value for s in traj.steps],
            gamma=gamma,
            lam=lam,
        )
        for s, a, r in zip(traj.steps, adv, ret):
            out.append(PPOSample(
                sample=BCSample(observation=s.observation, action=s.action, action_set=s.action_set),
                log_prob=s.log_prob,
                advantage=a,
                return_=r,
            ))
    return out


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    clip_frac: float
    approx_kl: float


def ppo_update_step(
    model: GCNN,
    optimizer: torch.optim.Optimizer,
    minibatch: list[PPOSample],
    *,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    device: str,
) -> UpdateStats:
    batch = collate_bipartite([s.sample for s in minibatch]).to(device)
    logits, values = model(batch.tensors, batch.graph_ids)

    new_log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for action, aset in zip(batch.actions, batch.action_sets):
        cand = torch.as_tensor(aset, dtype=torch.long, device=logits.device)
        cand_logits = logits[cand]
        log_softmax = F.log_softmax(cand_logits, dim=-1)
        aset_to_pos = {v: i for i, v in enumerate(aset)}   # O(1) lookup instead of O(N) .index()
        target_pos = aset_to_pos[action]
        new_log_probs.append(log_softmax[target_pos])
        probs = log_softmax.exp()
        entropies.append(-(probs * log_softmax).sum())
    new_log_probs_t = torch.stack(new_log_probs)
    entropies_t = torch.stack(entropies)

    old_log_probs_t = torch.tensor([s.log_prob for s in minibatch], device=device, dtype=torch.float32)
    advantages_t = torch.tensor([s.advantage for s in minibatch], device=device, dtype=torch.float32)
    returns_t = torch.tensor([s.return_ for s in minibatch], device=device, dtype=torch.float32)

    # Advantage normalization (per-minibatch, common PPO trick).
    if advantages_t.numel() > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    ratio = torch.exp(new_log_probs_t - old_log_probs_t)
    surr1 = ratio * advantages_t
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_t
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns_t)
    entropy_mean = entropies_t.mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
        approx_kl = (old_log_probs_t - new_log_probs_t).mean()

    return UpdateStats(
        policy_loss=float(policy_loss),
        value_loss=float(value_loss),
        entropy=float(entropy_mean),
        clip_frac=float(clip_frac),
        approx_kl=float(approx_kl),
    )


# ---------------------------------------------------------------------------
# Top-level loop
# ---------------------------------------------------------------------------

@dataclass
class PPOPaths:
    instance_dir: Path
    pretrain_ckpt: Path
    ckpt_dir: Path


def run_ppo(
    paths: PPOPaths,
    env_cfg: EnvConfig,
    *,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    lr: float,
    iterations: int,
    rollouts_per_iter: int,
    update_epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if not paths.pretrain_ckpt.exists():
        raise FileNotFoundError(
            f"Expected BC checkpoint at {paths.pretrain_ckpt} — run scripts.pretrain first."
        )
    model = load_pretrained_gcnn(paths.pretrain_ckpt, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    instances = sorted(paths.instance_dir.glob("instance_*.mps"))
    if len(instances) < rollouts_per_iter:
        raise RuntimeError(
            f"Need at least {rollouts_per_iter} training instances at {paths.instance_dir}, "
            f"found {len(instances)}."
        )

    env = make_branching_env(env_cfg)
    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_mean_reward = -float("inf")
    history: list[dict] = []

    for it in range(1, iterations + 1):
        t0 = time.perf_counter()

        # ---- Rollout ----
        picks = rng.sample(instances, rollouts_per_iter)
        trajectories: list[Trajectory] = []
        for inst in picks:
            env.seed(rng.randrange(2**31))
            trajectories.append(collect_trajectory(env, inst, model, device))
        samples = trajectories_to_samples(trajectories, gamma, gae_lambda)

        non_empty = [t for t in trajectories if t.steps]
        mean_reward = (
            sum(t.sum_reward for t in non_empty) / max(1, len(non_empty))
        )
        mean_steps = (
            sum(len(t.steps) for t in non_empty) / max(1, len(non_empty))
        )
        mean_nodes = (
            sum(t.n_nodes for t in non_empty) / max(1, len(non_empty))
        )

        if not samples:
            logger.warning(
                "iteration %d: every rollout terminated at the root (no branching)."
                " Skipping update; consider larger instances.",
                it,
            )
            history.append({
                "iter": it, "mean_reward": mean_reward, "mean_steps": mean_steps,
                "mean_nodes": mean_nodes, "n_samples": 0, "elapsed_s": time.perf_counter() - t0,
            })
            continue

        # ---- PPO updates ----
        update_records: list[UpdateStats] = []
        model.train()   # rollout sets eval(); restore train mode before gradient updates
        for _ in range(update_epochs):
            rng.shuffle(samples)
            for start in range(0, len(samples), minibatch_size):
                mb = samples[start:start + minibatch_size]
                if not mb:
                    continue
                update_records.append(ppo_update_step(
                    model, optimizer, mb,
                    clip_eps=clip_eps, value_coef=value_coef,
                    entropy_coef=entropy_coef, device=device,
                ))

        def _mean(field: str) -> float:
            return sum(getattr(r, field) for r in update_records) / max(1, len(update_records))

        record = {
            "iter": it,
            "mean_reward": mean_reward,
            "mean_steps": mean_steps,
            "mean_nodes": mean_nodes,
            "n_samples": len(samples),
            "policy_loss": _mean("policy_loss"),
            "value_loss": _mean("value_loss"),
            "entropy": _mean("entropy"),
            "clip_frac": _mean("clip_frac"),
            "approx_kl": _mean("approx_kl"),
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(record)
        logger.info(
            "iter %3d | reward=%.3f steps=%.1f nodes=%.1f | "
            "pol=%.3f val=%.3f ent=%.3f clip=%.2f kl=%.4f | %.1fs",
            it, mean_reward, mean_steps, mean_nodes,
            record["policy_loss"], record["value_loss"], record["entropy"],
            record["clip_frac"], record["approx_kl"], record["elapsed_s"],
        )

        # ---- Checkpoints ----
        ckpt_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "feature_dims": (model.var_embed[1][0].in_features,
                             model.cons_embed[1][0].in_features,
                             model.edge_norm.normalized_shape[0]),
            "model_config": {"hidden": model.hidden, "n_layers": model.n_layers},
            "iter": it,
            "mean_reward": mean_reward,
        }
        torch.save(ckpt_payload, paths.ckpt_dir / "ppo_latest.pt")
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(ckpt_payload, paths.ckpt_dir / "ppo_best.pt")

    (paths.ckpt_dir / "ppo_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {"history": history, "best_mean_reward": best_mean_reward}
