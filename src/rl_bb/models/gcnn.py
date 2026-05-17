"""Bipartite GCNN with shared backbone, policy head, and value head.

Architecture follows Gasse et al. 2019 (NeurIPS) with one change: a value
head is added on top of the shared variable embeddings for PPO.

Layout::

    var_features  ──► VarEmbed ─┐
                                ├── ConsMP ──► cons_h
    cons_features ──► ConsEmbed ┘                  │
                                                   ▼
                                            VarMP (uses cons_h, edges)
                                                   │
                                                   ▼
                                            shared variable embeddings
                                                   │
                                ┌──────────────────┴─────────────────┐
                                ▼                                    ▼
                        PolicyHead (per-var)              ValueHead (mean-pool → MLP)
                        → logits[n_vars], action mask     → V(s) scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_bb.models.obs_to_tensors import BipartiteTensors


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum-aggregate ``src`` rows into ``dim_size`` buckets by ``index``."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class BipartiteConv(nn.Module):
    """One half-round of message passing between cons and var nodes.

    Aggregates ``src`` node embeddings into ``dst`` buckets, mixing in
    per-edge features through a small edge MLP.
    """

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        self.msg = _mlp(2 * hidden + edge_dim, hidden, hidden)
        self.update = _mlp(2 * hidden, hidden, hidden)

    def forward(
        self,
        dst_h: torch.Tensor,        # (n_dst, hidden)
        src_h: torch.Tensor,        # (n_src, hidden)
        edge_index_src: torch.Tensor,  # (n_edges,)
        edge_index_dst: torch.Tensor,  # (n_edges,)
        edge_features: torch.Tensor,   # (n_edges, edge_dim)
    ) -> torch.Tensor:
        msg_in = torch.cat(
            [src_h[edge_index_src], dst_h[edge_index_dst], edge_features],
            dim=-1,
        )
        msg = self.msg(msg_in)                                     # (n_edges, hidden)
        agg = _scatter_sum(msg, edge_index_dst, dst_h.size(0))      # (n_dst, hidden)
        return self.update(torch.cat([dst_h, agg], dim=-1))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GCNN(nn.Module):
    """Shared-backbone bipartite GCNN with policy and value heads.

    The policy head returns *logits* over all variables. Callers must mask
    out non-candidates (set their logits to ``-inf``) before sampling or
    taking an argmax — see :meth:`forward_with_mask`.
    """

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

        # One "round" = cons-update followed by var-update.
        self.cons_updates = nn.ModuleList([BipartiteConv(hidden, edge_dim) for _ in range(n_layers)])
        self.var_updates = nn.ModuleList([BipartiteConv(hidden, edge_dim) for _ in range(n_layers)])

        # Per-variable policy logit (single scalar).
        self.policy_head = _mlp(hidden, hidden, 1)

        # Value head: mean-pool variable embeddings, then MLP → scalar.
        self.value_head = _mlp(hidden, hidden, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _backbone(self, t: BipartiteTensors) -> torch.Tensor:
        """Return the final variable embeddings ``(n_vars, hidden)``."""
        cons_h = self.cons_embed(t.cons_features)
        var_h = self.var_embed(t.var_features)
        edge_h = self.edge_norm(t.edge_features)

        # edge_index row 0: constraint index; row 1: variable index.
        cons_idx = t.edge_index[0]
        var_idx = t.edge_index[1]

        for cons_conv, var_conv in zip(self.cons_updates, self.var_updates):
            # cons ← var
            cons_h = cons_conv(cons_h, var_h, var_idx, cons_idx, edge_h)
            # var ← cons
            var_h = var_conv(var_h, cons_h, cons_idx, var_idx, edge_h)
        return var_h

    def forward(
        self,
        t: BipartiteTensors,
        graph_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)``.

        ``logits`` has shape ``(n_vars,)`` whether or not the input is
        batched. ``value`` is a scalar for a single graph or a vector of
        per-graph values when ``graph_ids`` is supplied.
        """
        var_h = self._backbone(t)
        logits = self.policy_head(var_h).squeeze(-1)
        if graph_ids is None:
            pooled = var_h.mean(dim=0, keepdim=True)               # (1, hidden)
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
            value = self.value_head(pooled).squeeze(-1)            # (n_graphs,)
        return logits, value

    def forward_with_mask(
        self, t: BipartiteTensors, action_set
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_logits, value)``.

        Non-candidate variables' logits are set to ``-inf`` so a downstream
        softmax/argmax ignores them.
        """
        logits, value = self.forward(t)
        mask = torch.full_like(logits, float("-inf"))
        idx = torch.as_tensor(list(action_set), dtype=torch.long, device=logits.device)
        mask[idx] = 0.0
        return logits + mask, value
