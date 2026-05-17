"""Behavioral-cloning dataset and collate function for branching demos.

Each ``.pkl`` produced by Stage 3 contains one full episode's worth of
``(observation, action, action_set)`` tuples. We flatten the episodes so
the BC training loop sees one branching decision per sample.

For dummy-scale data (hundreds to a few thousand decisions) we load every
pickle eagerly into memory at construction time — this keeps the data loop
simple and avoids per-step pickle I/O. For full-scale runs (10⁴+ instances)
we will need an on-disk index file; that's a Stage-5+ optimization.

The :func:`collate_bipartite` function stitches a list of single-graph
:class:`BipartiteTensors` samples into one block-diagonal big graph with the
appropriate index offsets, plus a ``graph_ids`` vector so the value head
can mean-pool per-graph.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from rl_bb.models.obs_to_tensors import BipartiteTensors, obs_to_tensors

logger = logging.getLogger(__name__)


@dataclass
class BCSample:
    observation: object              # ecole NodeBipartiteObs
    action: int                      # SCIP variable index
    action_set: list[int]            # candidate variable indices


@dataclass
class BipartiteBatch:
    tensors: BipartiteTensors
    actions: list[int]               # global (post-offset) action per sample
    action_sets: list[list[int]]     # global candidate sets per sample
    graph_ids: torch.Tensor          # (total_n_vars,) which graph each var belongs to

    def to(self, device) -> "BipartiteBatch":
        return BipartiteBatch(
            tensors=self.tensors.to(device),
            actions=self.actions,
            action_sets=self.action_sets,
            graph_ids=self.graph_ids.to(device),
        )


class BCDataset(Dataset):
    """Flat (observation, action, action_set) view over all demonstrations."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.samples: list[BCSample] = []
        if not self.root.exists():
            raise FileNotFoundError(f"Demonstration root {self.root} does not exist.")
        pickles = sorted(self.root.glob("**/*.pkl"))
        for p in pickles:
            with open(p, "rb") as f:
                traj = pickle.load(f)
            for obs, act, aset in zip(
                traj["observations"], traj["actions"], traj["action_sets"]
            ):
                self.samples.append(
                    BCSample(
                        observation=obs,
                        action=int(act),
                        action_set=[int(v) for v in aset],
                    )
                )
        logger.info(
            "Loaded %d branching decisions from %d pickle(s) in %s",
            len(self.samples), len(pickles), self.root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BCSample:
        return self.samples[idx]


def collate_bipartite(batch: Iterable[BCSample]) -> BipartiteBatch:
    """Stack a list of :class:`BCSample` into a single batched graph."""
    var_offsets = [0]
    cons_offsets = [0]
    all_var, all_cons, all_eidx, all_eattr, graph_ids = [], [], [], [], []
    actions: list[int] = []
    action_sets: list[list[int]] = []

    for i, s in enumerate(batch):
        t = obs_to_tensors(s.observation)
        all_var.append(t.var_features)
        all_cons.append(t.cons_features)

        shifted = t.edge_index.clone()
        shifted[0] += cons_offsets[-1]
        shifted[1] += var_offsets[-1]
        all_eidx.append(shifted)
        all_eattr.append(t.edge_features)

        v_off = var_offsets[-1]
        actions.append(s.action + v_off)
        action_sets.append([v + v_off for v in s.action_set])
        graph_ids.append(torch.full((t.n_vars,), i, dtype=torch.long))

        var_offsets.append(v_off + t.n_vars)
        cons_offsets.append(cons_offsets[-1] + t.n_cons)

    tensors = BipartiteTensors(
        var_features=torch.cat(all_var, dim=0),
        cons_features=torch.cat(all_cons, dim=0),
        edge_index=torch.cat(all_eidx, dim=1),
        edge_features=torch.cat(all_eattr, dim=0),
        n_vars=var_offsets[-1],
        n_cons=cons_offsets[-1],
    )
    return BipartiteBatch(
        tensors=tensors,
        actions=actions,
        action_sets=action_sets,
        graph_ids=torch.cat(graph_ids, dim=0),
    )
