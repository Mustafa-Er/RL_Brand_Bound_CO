"""Imitation pretraining (behavioral cloning) on expert demonstrations.

Pipeline
--------
1. Load all RB demonstrations under ``<demos_root>/<problem>/<regime>/<split>/rb/``.
2. Forward each batch through the GCNN; compute per-sample cross-entropy
   over each sample's *action set* (not all variables).
3. Track CE loss and top-1 / top-5 accuracy on train and val splits.
4. Save the best-by-val-loss checkpoint to ``<ckpt_dir>/pretrain_best.pt``.

The checkpoint stores model weights, the feature-dimension tuple needed to
re-instantiate the GCNN, and the optimizer state.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl_bb.models import GCNN, infer_feature_dims
from rl_bb.training.dataset import (
    BCDataset,
    BipartiteBatch,
    collate_bipartite,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-batch loss / accuracy
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    loss: float
    n_samples: int
    top1: int
    top5: int


def _step_loss(
    logits: torch.Tensor,
    batch: BipartiteBatch,
) -> tuple[torch.Tensor, StepMetrics]:
    """Compute mean CE over a batch and top-k accuracy counts."""
    losses: list[torch.Tensor] = []
    top1 = 0
    top5 = 0
    for action, action_set in zip(batch.actions, batch.action_sets):
        candidate_idx = torch.as_tensor(action_set, dtype=torch.long, device=logits.device)
        candidate_logits = logits[candidate_idx]                # (k,)
        aset_to_pos = {v: i for i, v in enumerate(action_set)}  # O(1) lookup
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    loss: float
    top1: float
    top5: float
    samples: int

    @classmethod
    def from_step(cls, accum: list[StepMetrics]) -> "EpochStats":
        n = sum(s.n_samples for s in accum)
        if n == 0:
            return cls(loss=float("nan"), top1=0.0, top5=0.0, samples=0)
        return cls(
            loss=sum(s.loss * s.n_samples for s in accum) / n,
            top1=sum(s.top1 for s in accum) / n,
            top5=sum(s.top5 for s in accum) / n,
            samples=n,
        )


def evaluate(model: GCNN, loader: DataLoader, device: str) -> EpochStats:
    model.eval()
    accum: list[StepMetrics] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _value = model(batch.tensors, batch.graph_ids)
            _loss, m = _step_loss(logits, batch)
            accum.append(m)
    return EpochStats.from_step(accum)


def train_one_epoch(
    model: GCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> EpochStats:
    model.train()
    accum: list[StepMetrics] = []
    for batch in loader:
        batch = batch.to(device)
        logits, _value = model(batch.tensors, batch.graph_ids)
        loss, m = _step_loss(logits, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accum.append(m)
    return EpochStats.from_step(accum)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class PretrainPaths:
    train_root: Path
    val_root: Path
    ckpt_dir: Path


def run_pretrain(
    paths: PretrainPaths,
    *,
    hidden: int,
    n_layers: int,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)

    train_set = BCDataset(paths.train_root)
    val_set = BCDataset(paths.val_root)
    if len(train_set) == 0:
        raise RuntimeError(f"No training samples found under {paths.train_root}")

    sample = train_set[0]
    d_var, d_cons, d_edge = infer_feature_dims(sample.observation)
    logger.info("Feature dims: var=%d cons=%d edge=%d", d_var, d_cons, d_edge)

    model = GCNN(d_var, d_cons, d_edge, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_bipartite,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_bipartite,
    ) if len(val_set) > 0 else None

    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device) if val_loader is not None else None
        elapsed = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_top1": train_stats.top1,
            "train_top5": train_stats.top5,
            "val_loss": val_stats.loss if val_stats else None,
            "val_top1": val_stats.top1 if val_stats else None,
            "val_top5": val_stats.top5 if val_stats else None,
            "elapsed_s": elapsed,
        }
        history.append(record)
        logger.info(
            "epoch %3d | train loss=%.4f top1=%.3f top5=%.3f | val %s | %.1fs",
            epoch, train_stats.loss, train_stats.top1, train_stats.top5,
            (f"loss={val_stats.loss:.4f} top1={val_stats.top1:.3f} top5={val_stats.top5:.3f}"
             if val_stats else "—"),
            elapsed,
        )

        # Save best by val loss (or last if no val).
        score = val_stats.loss if val_stats else train_stats.loss
        if score < best_val_loss:
            best_val_loss = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "feature_dims": (d_var, d_cons, d_edge),
                    "model_config": {"hidden": hidden, "n_layers": n_layers},
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                paths.ckpt_dir / "pretrain_best.pt",
            )

    (paths.ckpt_dir / "pretrain_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {"history": history, "best_val_loss": best_val_loss}


def load_pretrained_gcnn(ckpt_path: Path, device: str = "cpu") -> GCNN:
    """Re-instantiate a GCNN from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    d_var, d_cons, d_edge = ckpt["feature_dims"]
    mcfg = ckpt["model_config"]
    model = GCNN(d_var, d_cons, d_edge, hidden=mcfg["hidden"], n_layers=mcfg["n_layers"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model
