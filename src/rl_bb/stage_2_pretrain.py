"""Stage 2 — GCNN + Supervised Learning (behavioral cloning on RB demos).

This stage rolls the Reliability Branching expert over training and val
instances to produce ``.pkl`` demonstration trajectories, then trains a
bipartite GCNN on those (observation → expert action) pairs with per-sample
cross-entropy over each sample's action set.

Cache modes (``pretrain.mode`` in config)::

    auto           Load ``pretrain_best.pt`` if it exists, otherwise train.
    force_retrain  Always retrain; overwrite any existing checkpoint.
    load_only      Require an existing checkpoint; raise if missing.

If demonstrations are missing under
``data/<exp>/demonstrations/<problem>/<regime>/<split>/rb/`` they are
generated on the fly before training.

Outputs::

    checkpoints/<exp>/pretrain_best.pt
    checkpoints/<exp>/pretrain_history.json

Run::

    python -m rl_bb.stage_2_pretrain \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem combinatorial_auction
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl_bb.data import BCDataset, BipartiteBatch, collate_bipartite
from rl_bb.envs import EnvConfig, env_config_from_dict, make_expert_env
from rl_bb.experts import RBPolicy
from rl_bb.model import GCNN, infer_feature_dims, save_gcnn
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_device,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.stage_2")

PROBLEMS = ("set_covering", "combinatorial_auction")
EXPERT_NAME = "rb"


# ===========================================================================
# Demonstration collection (RB only)
# ===========================================================================

def _collect_one(env, instance_path: Path, policy) -> dict:
    policy.reset()
    t0 = time.perf_counter()
    observation, action_set, _r, done, info = env.reset(str(instance_path))
    bipartite_traj, action_traj, action_set_traj = [], [], []
    while not done:
        bipartite, _sb = observation
        action = int(policy.act(observation, action_set, env.model))
        bipartite_traj.append(bipartite)
        action_traj.append(action)
        action_set_traj.append(action_set)
        observation, action_set, _r, done, info = env.step(action)
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


def collect_demonstrations(
    instances: Iterable[Path],
    policy,
    env_cfg: EnvConfig,
    out_dir: Path,
    seed: int = 0,
    overwrite: bool = False,
) -> int:
    """Roll out ``policy`` on each instance and write one ``.pkl`` per instance.

    Resumable: instances whose pickle already exists are skipped unless
    ``overwrite=True``. This makes killing and restarting a long collection
    safe — work already on disk is reused.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = list(instances)
    # Build the list of instances that still need work.
    targets: list[tuple[int, Path, Path]] = []
    skipped = 0
    for i, inst in enumerate(instances):
        target = out_dir / f"{Path(inst).stem}.pkl"
        if target.exists() and not overwrite:
            skipped += 1
            continue
        targets.append((i, inst, target))
    if skipped:
        logger.info(
            "  resume: %d/%d demos already on disk, collecting %d remaining",
            skipped, len(instances), len(targets),
        )

    if not targets:
        return 0

    env = make_expert_env(env_cfg)
    written = 0
    for i, inst, target in targets:
        env.seed(seed + i)
        traj = _collect_one(env, inst, policy)
        with open(target, "wb") as f:
            pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "  demo %s | decisions=%d nodes=%.0f wall=%.2fs",
            inst.name, len(traj["actions"]), traj["n_nodes"], traj["wall_time_s"],
        )
        written += 1
    return written


def collect_rb_demonstrations(
    instances: Iterable[Path],
    env_cfg: EnvConfig,
    out_dir: Path,
    seed: int = 0,
) -> int:
    """Shorthand: roll out :class:`RBPolicy` on each instance."""
    return collect_demonstrations(instances, RBPolicy(), env_cfg, out_dir, seed=seed)


def _ensure_demos(
    split_dir: Path,
    instance_dir: Path,
    env_cfg: EnvConfig,
    seed: int,
) -> None:
    """Generate demonstrations for ``split_dir``, resuming any partial state.

    For each ``.mps`` instance we expect a matching ``.pkl`` in ``split_dir``.
    Missing pickles are collected; existing ones are kept. This makes the
    collection step idempotent and crash-safe.
    """
    instances = sorted(instance_dir.glob("instance_*.mps"))
    if not instances:
        raise FileNotFoundError(
            f"No instances at {instance_dir}; run Stage 1 first."
        )
    split_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in split_dir.glob("*.pkl")}
    needed = [inst for inst in instances if inst.stem not in existing]
    if not needed:
        logger.info("Demos already complete (%d files) at %s", len(instances), split_dir)
        return
    logger.info(
        "Collecting %d/%d RB demos (skipping %d already on disk) -> %s",
        len(needed), len(instances), len(existing), split_dir,
    )
    # collect_rb_demonstrations re-applies the same resume guard internally,
    # but feeding only the needed instances keeps the log tidy.
    collect_rb_demonstrations(needed, env_cfg, split_dir, seed=seed)


# ===========================================================================
# Per-batch loss / accuracy
# ===========================================================================

@dataclass
class StepMetrics:
    loss: float
    n_samples: int
    top1: int
    top5: int


def _step_loss(logits: torch.Tensor, batch: BipartiteBatch) -> tuple[torch.Tensor, StepMetrics]:
    losses: list[torch.Tensor] = []
    top1 = top5 = 0
    for action, action_set in zip(batch.actions, batch.action_sets):
        candidate_idx = torch.as_tensor(action_set, dtype=torch.long, device=logits.device)
        candidate_logits = logits[candidate_idx]
        aset_to_pos = {v: i for i, v in enumerate(action_set)}
        target_pos = aset_to_pos[action]
        target = torch.tensor([target_pos], device=logits.device, dtype=torch.long)
        losses.append(F.cross_entropy(candidate_logits.unsqueeze(0), target))

        order = candidate_logits.argsort(descending=True)
        if int(order[0]) == target_pos:
            top1 += 1
        k = min(5, len(action_set))
        if target_pos in order[:k].tolist():
            top5 += 1

    loss = torch.stack(losses).mean()
    return loss, StepMetrics(loss=float(loss), n_samples=len(batch.actions), top1=top1, top5=top5)


@dataclass
class EpochStats:
    loss: float
    top1: float
    top5: float
    samples: int

    @classmethod
    def from_steps(cls, accum: list[StepMetrics]) -> "EpochStats":
        n = sum(s.n_samples for s in accum)
        if n == 0:
            return cls(loss=float("nan"), top1=0.0, top5=0.0, samples=0)
        return cls(
            loss=sum(s.loss * s.n_samples for s in accum) / n,
            top1=sum(s.top1 for s in accum) / n,
            top5=sum(s.top5 for s in accum) / n,
            samples=n,
        )


def _train_one_epoch(model, loader, optimizer, device) -> EpochStats:
    model.train()
    accum: list[StepMetrics] = []
    for batch in loader:
        batch = batch.to(device)
        logits, _v = model(batch.tensors, batch.graph_ids)
        loss, m = _step_loss(logits, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accum.append(m)
    return EpochStats.from_steps(accum)


def _evaluate(model, loader, device) -> EpochStats:
    model.eval()
    accum: list[StepMetrics] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _v = model(batch.tensors, batch.graph_ids)
            _loss, m = _step_loss(logits, batch)
            accum.append(m)
    return EpochStats.from_steps(accum)


# ===========================================================================
# Training driver with cache modes
# ===========================================================================

@dataclass
class PretrainPaths:
    train_root: Path
    val_root: Path
    train_instance_dir: Path | None = None
    val_instance_dir: Path | None = None
    ckpt_dir: Path | None = None


@dataclass
class PretrainConfig:
    hidden: int = 64
    n_layers: int = 2
    lr: float = 1e-2
    epochs: int = 2000
    batch_size: int = 32
    device: str = "cpu"
    seed: int = 0
    log_every: int = 10
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 5e-5
    scheduler_threshold: float = 1e-4


def run_pretrain(paths: PretrainPaths, cfg: PretrainConfig) -> dict:
    """Train the GCNN with supervised loss on the demonstrations under ``paths``.

    Demonstrations must already exist; this is the bare training loop. The
    higher-level :func:`run_stage_2` handles demo collection and cache modes.
    """
    torch.manual_seed(cfg.seed)
    hidden = cfg.hidden
    n_layers = cfg.n_layers
    lr = cfg.lr
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    device = cfg.device
    seed = cfg.seed
    train_set = BCDataset(paths.train_root)
    val_set = BCDataset(paths.val_root)
    if len(train_set) == 0:
        raise RuntimeError(f"No training samples found under {paths.train_root}")

    sample = train_set[0]
    d_var, d_cons, d_edge = infer_feature_dims(sample.observation)
    logger.info("Feature dims: var=%d cons=%d edge=%d", d_var, d_cons, d_edge)

    model = GCNN(d_var, d_cons, d_edge, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
        threshold=cfg.scheduler_threshold,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_bipartite
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_bipartite
    ) if len(val_set) > 0 else None

    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict] = []
    log_every = max(1, int(cfg.log_every))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_stats = _train_one_epoch(model, train_loader, optimizer, device)
        val_stats = _evaluate(model, val_loader, device) if val_loader is not None else None
        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Scheduler step on whichever loss drives "improvement".
        score = val_stats.loss if val_stats else train_stats.loss
        scheduler.step(score)

        improved = score < best_val
        if improved:
            best_val = score
            save_gcnn(
                model,
                paths.ckpt_dir / "pretrain_best.pt",
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                val_loss=best_val,
            )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_stats.loss,
            "train_top1": train_stats.top1, "train_top5": train_stats.top5,
            "val_loss": val_stats.loss if val_stats else None,
            "val_top1": val_stats.top1 if val_stats else None,
            "val_top5": val_stats.top5 if val_stats else None,
            "elapsed_s": elapsed,
            "best_val_so_far": best_val,
        })

        # Log on cadence, on the first/last epoch, or when a new best is found.
        if epoch == 1 or epoch == epochs or epoch % log_every == 0 or improved:
            tag = " *best*" if improved else ""
            logger.info(
                "epoch %4d | lr=%.2e | train loss=%.4f top1=%.3f | val %s | %.2fs%s",
                epoch, current_lr,
                train_stats.loss, train_stats.top1,
                (f"loss={val_stats.loss:.4f} top1={val_stats.top1:.3f}" if val_stats else "—"),
                elapsed, tag,
            )

    (paths.ckpt_dir / "pretrain_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {"history": history, "best_val_loss": best_val}


def run_stage_2(
    cfg: dict,
    problem: str,
    regime: str = "train_size",
    mode: str | None = None,
) -> dict:
    """Programmatic entry point. Returns dict with ``mode`` and either
    ``history``/``best_val_loss`` (if training ran) or ``ckpt_path``
    (if cache hit)."""
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))
    pcfg = cfg.get("pretrain", {})
    mode = mode or pcfg.get("mode", "auto")
    if mode not in ("auto", "force_retrain", "load_only"):
        raise ValueError(f"Unknown pretrain.mode: {mode!r}")

    data_dir = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    demos_root = data_dir / "demonstrations" / problem / regime
    ckpt_dir = resolve_path(cfg["paths"]["ckpt_dir"]) / cfg["experiment"]["name"]
    paths = PretrainPaths(
        train_root=demos_root / "train" / EXPERT_NAME,
        val_root=demos_root / "val" / EXPERT_NAME,
        train_instance_dir=data_dir / problem / regime / "train",
        val_instance_dir=data_dir / problem / regime / "val",
        ckpt_dir=ckpt_dir,
    )
    ckpt_path = paths.ckpt_dir / "pretrain_best.pt"

    if mode == "load_only":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"mode=load_only but no checkpoint at {ckpt_path}.")
        logger.info("Loaded cached checkpoint (load_only): %s", ckpt_path)
        return {"mode": mode, "ckpt_path": str(ckpt_path)}

    if mode == "auto" and ckpt_path.exists():
        logger.info("Loaded cached checkpoint (auto): %s", ckpt_path)
        return {"mode": mode, "ckpt_path": str(ckpt_path), "cached": True}

    # Need to train. Ensure demonstrations exist first.
    env_cfg = env_config_from_dict(cfg.get("env", {}))
    _ensure_demos(paths.train_root, paths.train_instance_dir, env_cfg, seed)
    _ensure_demos(paths.val_root, paths.val_instance_dir, env_cfg, seed + 1)

    mcfg = cfg.get("model", {}).get("gcnn", {})
    sched = pcfg.get("scheduler") or {}
    pre_cfg = PretrainConfig(
        hidden=int(mcfg.get("embedding_size", 64)),
        n_layers=int(mcfg.get("n_layers", 2)),
        lr=float(pcfg.get("lr") or 1e-2),
        epochs=int(pcfg.get("epochs") or 2000),
        batch_size=int(pcfg.get("batch_size") or 32),
        device=device,
        seed=seed,
        log_every=int(pcfg.get("log_every", 10)),
        scheduler_patience=int(sched.get("patience", 50)),
        scheduler_factor=float(sched.get("factor", 0.5)),
        scheduler_min_lr=float(sched.get("min_lr", 5e-5)),
        scheduler_threshold=float(sched.get("threshold", 1e-4)),
    )
    logger.info("Training (mode=%s) -> %s", mode, ckpt_path)
    result = run_pretrain(paths, pre_cfg)
    result["mode"] = mode
    result["ckpt_path"] = str(ckpt_path)
    return result


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — GCNN + SL pretraining.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--regime", default="train_size")
    p.add_argument(
        "--mode",
        choices=("auto", "force_retrain", "load_only"),
        default=None,
        help="Override pretrain.mode from config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    log_dir = resolve_path(cfg["paths"]["log_dir"]) / cfg["experiment"]["name"]
    configure_logging(log_dir=log_dir, filename="stage_2.log")
    out = run_stage_2(cfg, problem=args.problem, regime=args.regime, mode=args.mode)
    if "best_val_loss" in out:
        logger.info("Stage 2 done | best_val_loss=%.4f | ckpt=%s",
                    out["best_val_loss"], out["ckpt_path"])
    else:
        logger.info("Stage 2 done | cache hit | ckpt=%s", out["ckpt_path"])


if __name__ == "__main__":
    main()
