"""Convert Ecole's ``NodeBipartiteObs`` into torch tensors.

The bipartite observation produced by ``ecole.observation.NodeBipartite()``
groups information into three blocks:

* ``column_features``  — per-variable static features (LP solution value,
  reduced cost, type, etc.)
* ``row_features``     — per-constraint features (right-hand side, slack…)
* ``edge_features``    — sparse non-zero coefficients of the constraint
  matrix at the current LP relaxation.

This module exposes :class:`BipartiteTensors` (a plain dataclass holding
``Tensor``\\ s) and :func:`obs_to_tensors`. Conversions happen on CPU; move
the result to the training device with ``.to(device)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BipartiteTensors:
    var_features: torch.Tensor      # (n_vars,  d_var)
    cons_features: torch.Tensor     # (n_cons,  d_cons)
    edge_index: torch.Tensor        # (2, n_edges) — row 0: cons idx, row 1: var idx
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
    """Convert an :class:`ecole.observation.NodeBipartiteObs` to tensors."""
    var_np = np.asarray(_var_features(obs), dtype=np.float32)
    cons_np = np.asarray(obs.row_features, dtype=np.float32)

    edge_idx_np = np.asarray(obs.edge_features.indices, dtype=np.int64)
    edge_val_np = np.asarray(obs.edge_features.values, dtype=np.float32)
    # Ecole emits edge values as a 1D vector of coefficients; promote to 2D
    # so the per-edge feature axis is explicit.
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
