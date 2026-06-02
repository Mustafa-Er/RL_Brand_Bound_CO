"""Bipartite GCNN (Gasse et al. 2019) + observation conversion + inference wrapper.

* :func:`obs_to_tensors` turns an Ecole ``NodeBipartiteObs`` into a
  :class:`BipartiteTensors` dataclass (pure torch).
* :class:`GCNN` is a shared-backbone policy + value network. The policy head
  emits per-variable logits; callers mask non-candidate logits to ``-inf``
  before sampling. The value head mean-pools variable embeddings per graph.
* :class:`RLPolicy` wraps a trained ``GCNN`` in the ``Policy`` interface so
  Stage 4 evaluation can drop it in next to Random and FSB.
* :func:`save_gcnn` / :func:`load_gcnn` provide the on-disk checkpoint
  protocol shared by Stage 2 (BC) and Stage 3 (PPO).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Observation → tensors
# ===========================================================================

@dataclass
class BipartiteTensors:
    var_features: torch.Tensor      # (n_vars,  d_var)
    cons_features: torch.Tensor     # (n_cons,  d_cons)
    edge_index: torch.Tensor        # (2, n_edges); row 0: cons idx, row 1: var idx
    edge_features: torch.Tensor     # (n_edges, d_edge)
    n_vars: int
    n_cons: int

    def to(self, device) -> "BipartiteTensors":
        return BipartiteTensors(
            var_features=self.var_features.to(device),
            cons_features=self.cons_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            n_vars=self.n_vars,
            n_cons=self.n_cons,
        )


def _var_features(obs):
    """Ecole renamed ``column_features`` -> ``variable_features``; accept both."""
    for name in ("variable_features", "column_features"):
        if hasattr(obs, name):
            return getattr(obs, name)
    raise AttributeError(
        "NodeBipartiteObs has neither 'variable_features' nor 'column_features'"
    )


def obs_to_tensors(obs) -> BipartiteTensors:
    """Convert an Ecole ``NodeBipartiteObs`` to torch tensors (CPU)."""
    var_np = np.asarray(_var_features(obs), dtype=np.float32)
    cons_np = np.asarray(obs.row_features, dtype=np.float32)
    edge_idx_np = np.asarray(obs.edge_features.indices, dtype=np.int64)
    edge_val_np = np.asarray(obs.edge_features.values, dtype=np.float32)
    if edge_val_np.ndim == 1:
        edge_val_np = edge_val_np[:, None]
    return BipartiteTensors(
        var_features=torch.from_numpy(var_np),
        cons_features=torch.from_numpy(cons_np),
        edge_index=torch.from_numpy(edge_idx_np),
        edge_features=torch.from_numpy(edge_val_np),
        n_vars=int(var_np.shape[0]),
        n_cons=int(cons_np.shape[0]),
    )


def infer_feature_dims(obs) -> tuple[int, int, int]:
    """Return ``(d_var, d_cons, d_edge)`` from a single observation."""
    t = obs_to_tensors(obs)
    return t.var_features.shape[1], t.cons_features.shape[1], t.edge_features.shape[1]


# ===========================================================================
# GCNN
# ===========================================================================

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class BipartiteConv(nn.Module):
    """One half-round of message passing between cons and var nodes."""

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        self.msg = _mlp(2 * hidden + edge_dim, hidden, hidden)
        self.update = _mlp(2 * hidden, hidden, hidden)

    def forward(
        self,
        dst_h: torch.Tensor,
        src_h: torch.Tensor,
        edge_index_src: torch.Tensor,
        edge_index_dst: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        msg_in = torch.cat(
            [src_h[edge_index_src], dst_h[edge_index_dst], edge_features],
            dim=-1,
        )
        msg = self.msg(msg_in)
        agg = _scatter_sum(msg, edge_index_dst, dst_h.size(0))
        return self.update(torch.cat([dst_h, agg], dim=-1))


class GCNN(nn.Module):
    """Shared-backbone bipartite GCNN with policy + value heads."""

    def __init__(
        self,
        var_dim: int,
        cons_dim: int,
        edge_dim: int = 1,
        hidden: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers

        self.var_embed = nn.Sequential(nn.LayerNorm(var_dim), _mlp(var_dim, hidden, hidden))
        self.cons_embed = nn.Sequential(nn.LayerNorm(cons_dim), _mlp(cons_dim, hidden, hidden))
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.cons_updates = nn.ModuleList([BipartiteConv(hidden, edge_dim) for _ in range(n_layers)])
        self.var_updates = nn.ModuleList([BipartiteConv(hidden, edge_dim) for _ in range(n_layers)])
        self.policy_head = _mlp(hidden, hidden, 1)
        self.value_head = _mlp(hidden, hidden, 1)

    def _backbone(self, t: BipartiteTensors) -> torch.Tensor:
        cons_h = self.cons_embed(t.cons_features)
        var_h = self.var_embed(t.var_features)
        edge_h = self.edge_norm(t.edge_features)
        cons_idx = t.edge_index[0]
        var_idx = t.edge_index[1]
        for cons_conv, var_conv in zip(self.cons_updates, self.var_updates):
            cons_h = cons_conv(cons_h, var_h, var_idx, cons_idx, edge_h)
            var_h = var_conv(var_h, cons_h, cons_idx, var_idx, edge_h)
        return var_h

    def forward(
        self,
        t: BipartiteTensors,
        graph_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)``.

        ``logits`` is ``(n_vars,)``. ``value`` is a scalar if ``graph_ids`` is
        ``None``, else a ``(n_graphs,)`` per-graph vector.
        """
        var_h = self._backbone(t)
        logits = self.policy_head(var_h).squeeze(-1)
        if graph_ids is None:
            pooled = var_h.mean(dim=0, keepdim=True)
            value = self.value_head(pooled).squeeze(-1).squeeze(-1)
        else:
            n_graphs = int(graph_ids.max().item()) + 1
            sums = _scatter_sum(var_h, graph_ids, n_graphs)
            counts = _scatter_sum(
                torch.ones(var_h.size(0), 1, device=var_h.device, dtype=var_h.dtype),
                graph_ids,
                n_graphs,
            )
            pooled = sums / counts.clamp(min=1.0)
            value = self.value_head(pooled).squeeze(-1)
        return logits, value

    def forward_with_mask(
        self, t: BipartiteTensors, action_set
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_logits, value)`` with non-candidates set to ``-inf``."""
        logits, value = self.forward(t)
        mask = torch.full_like(logits, float("-inf"))
        idx = torch.as_tensor(list(action_set), dtype=torch.long, device=logits.device)
        mask[idx] = 0.0
        return logits + mask, value


# ===========================================================================
# Checkpoint I/O (shared by Stage 2 + Stage 3)
# ===========================================================================

def save_gcnn(model: GCNN, path: Path, **meta) -> None:
    """Save model weights + the metadata needed to rebuild the architecture."""
    payload = {
        "model_state": model.state_dict(),
        "feature_dims": (
            model.var_embed[1][0].in_features,
            model.cons_embed[1][0].in_features,
            model.edge_norm.normalized_shape[0],
        ),
        "model_config": {"hidden": model.hidden, "n_layers": model.n_layers},
    }
    payload.update(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_gcnn(path: Path, device: str = "cpu") -> GCNN:
    """Reconstruct a ``GCNN`` from a checkpoint produced by :func:`save_gcnn`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    d_var, d_cons, d_edge = ckpt["feature_dims"]
    mcfg = ckpt["model_config"]
    model = GCNN(d_var, d_cons, d_edge, hidden=mcfg["hidden"], n_layers=mcfg["n_layers"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model


# ===========================================================================
# Inference wrapper
# ===========================================================================

class RLPolicy:
    """``Policy``-compatible wrapper around a trained :class:`GCNN`."""

    def __init__(self, model: GCNN, device: str = "cpu", stochastic: bool = False) -> None:
        self.model = model
        self.device = device
        self.stochastic = stochastic
        self.model.to(device)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, observation, action_set, model) -> int:
        bipartite = observation[0] if isinstance(observation, tuple) else observation
        tensors = obs_to_tensors(bipartite).to(self.device)
        logits, _value = self.model.forward_with_mask(tensors, action_set)
        if self.stochastic:
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(logits).item())
