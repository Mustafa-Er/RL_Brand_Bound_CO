"""Stage 3 — PPO+GAE training, warm-started from the Stage 2 checkpoint.

Each iteration:
  1. Roll out the current policy on ``rollouts_per_iter`` training instances.
  2. Compute GAE advantages and value targets per trajectory.
  3. Concatenate transitions into one buffer.
  4. Run ``update_epochs`` shuffled-minibatch passes with the clipped PPO
     surrogate + value MSE + entropy bonus + grad-clip.

Outputs::

    checkpoints/<exp>/ppo_latest.pt
    checkpoints/<exp>/ppo_best.pt        # best-by-mean-reward
    checkpoints/<exp>/ppo_history.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from rl_bb.data import BCSample, collate_bipartite
from rl_bb.envs import EnvConfig, env_config_from_dict, make_branching_env
from rl_bb.model import GCNN, load_gcnn, obs_to_tensors, save_gcnn
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_device,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.stage_3")

PROBLEMS = ("set_covering", "combinatorial_auction")
_GRAD_CLIP_NORM = 1.0


# ===========================================================================
# Config + dataclasses
# ===========================================================================

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-5
    iterations: int = 50
    rollouts_per_iter: int = 8
    update_epochs: int = 4
    minibatch_size: int = 16
    value_coef: float = 0.25
    entropy_coef: float = 0.01
    device: str = "cpu"
    seed: int = 0


@dataclass
class PPOPaths:
    instance_dir: Path
    pretrain_ckpt: Path
    ckpt_dir: Path


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


@dataclass
class PPOSample:
    sample: BCSample
    log_prob: float
    advantage: float
    return_: float


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    clip_frac: float
    approx_kl: float


# ===========================================================================
# GAE
# ===========================================================================

def compute_gae(
    rewards: Iterable[float],
    values: Iterable[float],
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Return ``(advantages, returns)`` for one trajectory."""
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


# ===========================================================================
# Rollout
# ===========================================================================

def _sample_action(model: GCNN, bipartite, action_set, device) -> tuple[int, float, float]:
    tensors = obs_to_tensors(bipartite).to(device)
    logits, value = model.forward_with_mask(tensors, list(action_set))
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item()), float(log_probs[idx].item()), float(value.item())


def collect_trajectory(env, instance_path: Path, model: GCNN, device: str) -> Trajectory:
    """Roll out ``model`` on one instance; return the full trajectory."""
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
            gamma=gamma, lam=lam,
        )
        for s, a, r in zip(traj.steps, adv, ret):
            out.append(PPOSample(
                sample=BCSample(observation=s.observation, action=s.action, action_set=s.action_set),
                log_prob=s.log_prob, advantage=a, return_=r,
            ))
    return out


# ===========================================================================
# PPO update
# ===========================================================================

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
        aset_to_pos = {v: i for i, v in enumerate(aset)}
        target_pos = aset_to_pos[action]
        new_log_probs.append(log_softmax[target_pos])
        probs = log_softmax.exp()
        entropies.append(-(probs * log_softmax).sum())

    new_log_probs_t = torch.stack(new_log_probs)
    entropies_t = torch.stack(entropies)
    old_log_probs_t = torch.tensor([s.log_prob for s in minibatch], device=device, dtype=torch.float32)
    advantages_t = torch.tensor([s.advantage for s in minibatch], device=device, dtype=torch.float32)
    returns_t = torch.tensor([s.return_ for s in minibatch], device=device, dtype=torch.float32)

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_GRAD_CLIP_NORM)
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


def _stats_mean(records: list[UpdateStats], field: str) -> float:
    return sum(getattr(r, field) for r in records) / max(1, len(records))


# ===========================================================================
# Top-level PPO loop
# ===========================================================================

def run_ppo(paths: PPOPaths, env_cfg: EnvConfig, cfg: PPOConfig) -> dict:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    if not paths.pretrain_ckpt.exists():
        raise FileNotFoundError(
            f"Expected BC checkpoint at {paths.pretrain_ckpt}; run Stage 2 first."
        )
    model = load_gcnn(paths.pretrain_ckpt, device=cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    instances = sorted(paths.instance_dir.glob("instance_*.mps"))
    if len(instances) < cfg.rollouts_per_iter:
        raise RuntimeError(
            f"Need at least {cfg.rollouts_per_iter} instances at {paths.instance_dir}, "
            f"found {len(instances)}."
        )

    env = make_branching_env(env_cfg)
    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_mean_reward = -float("inf")
    history: list[dict] = []

    for it in range(1, cfg.iterations + 1):
        t0 = time.perf_counter()
        picks = rng.sample(instances, cfg.rollouts_per_iter)
        trajectories: list[Trajectory] = []
        for inst in picks:
            env.seed(rng.randrange(2**31))
            trajectories.append(collect_trajectory(env, inst, model, cfg.device))
        samples = trajectories_to_samples(trajectories, cfg.gamma, cfg.gae_lambda)

        non_empty = [t for t in trajectories if t.steps]
        mean_reward = sum(t.sum_reward for t in non_empty) / max(1, len(non_empty))
        mean_steps  = sum(len(t.steps)  for t in non_empty) / max(1, len(non_empty))
        mean_nodes  = sum(t.n_nodes     for t in non_empty) / max(1, len(non_empty))

        if not samples:
            logger.warning(
                "iter %d: every rollout terminated at the root — skipping update.", it,
            )
            history.append({
                "iter": it, "mean_reward": mean_reward, "mean_steps": mean_steps,
                "mean_nodes": mean_nodes, "n_samples": 0,
                "elapsed_s": time.perf_counter() - t0,
            })
            continue

        update_records: list[UpdateStats] = []
        model.train()
        for _ in range(cfg.update_epochs):
            rng.shuffle(samples)
            for start in range(0, len(samples), cfg.minibatch_size):
                mb = samples[start : start + cfg.minibatch_size]
                if mb:
                    update_records.append(ppo_update_step(
                        model, optimizer, mb,
                        clip_eps=cfg.clip_eps,
                        value_coef=cfg.value_coef,
                        entropy_coef=cfg.entropy_coef,
                        device=cfg.device,
                    ))

        record = {
            "iter": it,
            "mean_reward": mean_reward, "mean_steps": mean_steps, "mean_nodes": mean_nodes,
            "n_samples": len(samples),
            "policy_loss": _stats_mean(update_records, "policy_loss"),
            "value_loss":  _stats_mean(update_records, "value_loss"),
            "entropy":     _stats_mean(update_records, "entropy"),
            "clip_frac":   _stats_mean(update_records, "clip_frac"),
            "approx_kl":   _stats_mean(update_records, "approx_kl"),
            "elapsed_s":   time.perf_counter() - t0,
        }
        history.append(record)
        logger.info(
            "iter %3d | reward=%.3f steps=%.1f nodes=%.1f | "
            "pol=%.3f val=%.3f ent=%.3f clip=%.2f kl=%.4f | %.1fs",
            it, mean_reward, mean_steps, mean_nodes,
            record["policy_loss"], record["value_loss"], record["entropy"],
            record["clip_frac"], record["approx_kl"], record["elapsed_s"],
        )

        save_gcnn(
            model, paths.ckpt_dir / "ppo_latest.pt",
            optimizer_state=optimizer.state_dict(), iter=it, mean_reward=mean_reward,
        )
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            save_gcnn(
                model, paths.ckpt_dir / "ppo_best.pt",
                optimizer_state=optimizer.state_dict(), iter=it, mean_reward=mean_reward,
            )

    (paths.ckpt_dir / "ppo_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {"history": history, "best_mean_reward": best_mean_reward}


# ===========================================================================
# Public entry point + CLI
# ===========================================================================

def run_stage_3(cfg: dict, problem: str, regime: str = "train_size", split: str = "train") -> dict:
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))

    data_root = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    instance_dir = data_root / problem / regime / split
    ckpt_dir = resolve_path(cfg["paths"]["ckpt_dir"]) / cfg["experiment"]["name"]
    paths = PPOPaths(
        instance_dir=instance_dir,
        pretrain_ckpt=ckpt_dir / "pretrain_best.pt",
        ckpt_dir=ckpt_dir,
    )

    env_cfg = env_config_from_dict(cfg.get("env", {}))
    pcfg = cfg.get("ppo", {})
    ppo_cfg = PPOConfig(
        gamma=float(pcfg.get("gamma", 0.99)),
        gae_lambda=float(pcfg.get("gae_lambda", 0.95)),
        clip_eps=float(pcfg.get("clip_eps", 0.2)),
        lr=float(pcfg.get("lr") or 3e-5),
        iterations=int(pcfg.get("iterations") or 50),
        rollouts_per_iter=int(pcfg.get("rollouts_per_iter") or 8),
        update_epochs=int(pcfg.get("update_epochs") or 4),
        minibatch_size=int(pcfg.get("minibatch_size") or 16),
        value_coef=float(pcfg.get("value_coef", 0.25)),
        entropy_coef=float(pcfg.get("entropy_coef", 0.01)),
        device=device,
        seed=seed,
    )
    return run_ppo(paths, env_cfg, ppo_cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — PPO+GAE training.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--regime", default="train_size")
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    log_dir = resolve_path(cfg["paths"]["log_dir"]) / cfg["experiment"]["name"]
    configure_logging(log_dir=log_dir, filename="stage_3.log")
    out = run_stage_3(cfg, problem=args.problem, regime=args.regime, split=args.split)
    logger.info("Stage 3 done | best_mean_reward=%.4f", out["best_mean_reward"])


if __name__ == "__main__":
    main()
