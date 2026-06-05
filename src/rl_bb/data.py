"""Behavioral-cloning dataset + manual bipartite batching.

A ``.pkl`` produced by Stage 2 demonstration collection holds one full
episode's worth of ``(observation, action, action_set)`` tuples. We flatten
these so the BC training loop sees one branching decision per sample.

``collate_bipartite`` stitches a list of single-graph samples into one
block-diagonal big graph with the right index offsets, plus a ``graph_ids``
vector so the value head can mean-pool per-graph.

Loading is **lazy**: only one pickle (plus a small LRU cache of recently
used ones) is held in RAM at a time. This lets us train on 10⁴+ instances
without exhausting memory. An index sidecar ``.bcdataset_index.json`` is
written alongside the demonstrations the first time the dataset is built,
so subsequent runs start instantly instead of re-deserializing every
pickle just to count decisions.
"""
from __future__ import annotations

import json
import logging
import pickle
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset

from rl_bb.model import BipartiteTensors, obs_to_tensors

logger = logging.getLogger(__name__)


@dataclass
class BCSample:
    observation: object
    action: int
    action_set: list[int]


@dataclass
class BipartiteBatch:
    tensors: BipartiteTensors
    actions: list[int]
    action_sets: list[list[int]]
    graph_ids: torch.Tensor

    def to(self, device) -> "BipartiteBatch":
        return BipartiteBatch(
            tensors=self.tensors.to(device),
            actions=self.actions,
            action_sets=self.action_sets,
            graph_ids=self.graph_ids.to(device),
        )


# ---------------------------------------------------------------------------
# Index sidecar (so we don't unpickle every file just to count decisions)
# ---------------------------------------------------------------------------

_INDEX_NAME = ".bcdataset_index.json"


def _load_index_sidecar(root: Path) -> dict | None:
    p = root / _INDEX_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_index_sidecar(root: Path, counts: dict[str, int]) -> None:
    p = root / _INDEX_NAME
    try:
        p.write_text(
            json.dumps({"counts": counts}, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Could not write index sidecar at %s: %s", p, e)


def _build_index(
    root: Path, pickle_paths: list[Path]
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Return ``(index, counts)``.

    ``index`` is a flat list of (pickle_path, decision_idx) tuples — one
    entry per branching decision. ``counts`` maps pickle filename to the
    number of decisions it holds, ready to be cached as a sidecar.

    Tries the sidecar first; if it covers the current pickle set, uses it.
    Otherwise opens every pickle once (peak RAM = one pickle at a time) to
    count decisions, then writes a fresh sidecar.
    """
    cached = _load_index_sidecar(root)
    pickle_names = [p.name for p in pickle_paths]
    cached_counts: dict[str, int] = (cached or {}).get("counts", {})

    # Reuse sidecar entries that already cover our (possibly subset) request;
    # only count the pickles whose entry is missing.
    counts: dict[str, int] = {}
    to_count: list[Path] = []
    for p in pickle_paths:
        if p.name in cached_counts:
            counts[p.name] = cached_counts[p.name]
        else:
            to_count.append(p)

    if to_count:
        for i, p in enumerate(to_count):
            with open(p, "rb") as f:
                traj = pickle.load(f)
            counts[p.name] = len(traj["actions"])
            if (i + 1) % 500 == 0:
                logger.info("  indexing %d/%d new pickles…", i + 1, len(to_count))
        # Merge into the sidecar so subsequent runs hit cache.
        merged = dict(cached_counts)
        merged.update(counts)
        _save_index_sidecar(root, merged)

    by_name: dict[str, Path] = {p.name: p for p in pickle_paths}
    index: list[tuple[Path, int]] = []
    for name in pickle_names:
        n = counts[name]
        path = by_name[name]
        for k in range(n):
            index.append((path, k))
    return index, counts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """Lazy view over flat ``(observation, action, action_set)`` decisions.

    Pickles are loaded on demand; an LRU cache of size ``cache_size`` keeps
    the most-recently-used ones in memory so contiguous access patterns
    don't incur repeated I/O. Peak RAM usage scales with
    ``cache_size × (size of one pickle)``, not with the total dataset size.

    Parameters
    ----------
    root:
        Directory holding ``.pkl`` demonstration files.
    cache_size:
        How many pickles to keep cached. Default 32 trades modest RAM
        (tens to a few hundred MB on full-scale data) for fewer reloads.
    """

    def __init__(
        self,
        root: Path,
        cache_size: int = 32,
        max_pickles: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Demonstration root {self.root} does not exist.")
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[Path, dict] = OrderedDict()

        all_pickles = sorted(self.root.glob("**/*.pkl"))
        if max_pickles is not None and max_pickles < len(all_pickles):
            pickle_paths = all_pickles[: int(max_pickles)]
            logger.info(
                "Using first %d/%d pickles from %s (max_pickles cap)",
                len(pickle_paths), len(all_pickles), self.root,
            )
        else:
            pickle_paths = all_pickles
        self._index, _counts = _build_index(self.root, pickle_paths)
        logger.info(
            "Indexed %d branching decisions across %d pickle(s) in %s "
            "(lazy, lru_cache=%d)",
            len(self._index), len(pickle_paths), self.root, self.cache_size,
        )

    def __len__(self) -> int:
        return len(self._index)

    def _load_pickle(self, path: Path) -> dict:
        traj = self._cache.get(path)
        if traj is not None:
            self._cache.move_to_end(path)
            return traj
        with open(path, "rb") as f:
            traj = pickle.load(f)
        self._cache[path] = traj
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return traj

    def __getitem__(self, idx: int) -> BCSample:
        path, dec_idx = self._index[idx]
        traj = self._load_pickle(path)
        return BCSample(
            observation=traj["observations"][dec_idx],
            action=int(traj["actions"][dec_idx]),
            action_set=[int(v) for v in traj["action_sets"][dec_idx]],
        )


# ---------------------------------------------------------------------------
# Sampler — groups decisions by source pickle so each pickle is read once/epoch
# ---------------------------------------------------------------------------

class PickleGroupedBatchSampler:
    """Yield batches of dataset indices that all come from the same pickle.

    Why: the lazy :class:`BCDataset` reads one pickle per cache miss. With a
    fully-random shuffle on a dataset spanning 10⁴ pickles, every batch
    triggers ~``batch_size`` cache misses → I/O dominates training time.

    By grouping batches at the pickle level we read each pickle once per
    epoch. Pickle order and the order of decisions inside a pickle are both
    shuffled when ``shuffle=True``, so the model still sees varied
    minibatches; only the *batch composition* loses cross-instance mixing,
    which is acceptable for behavioral cloning.
    """

    def __init__(
        self,
        dataset: "BCDataset",
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._rng = random.Random(seed)

        groups: dict[Path, list[int]] = defaultdict(list)
        for sample_idx, (path, _dec) in enumerate(dataset._index):
            groups[path].append(sample_idx)
        self._groups: list[list[int]] = list(groups.values())

        n_batches = 0
        for g in self._groups:
            if self.drop_last:
                n_batches += len(g) // self.batch_size
            else:
                n_batches += (len(g) + self.batch_size - 1) // self.batch_size
        self._n_batches = n_batches

    def __iter__(self) -> Iterator[list[int]]:
        groups = [list(g) for g in self._groups]  # work on a copy
        if self.shuffle:
            self._rng.shuffle(groups)
            for g in groups:
                self._rng.shuffle(g)
        for group in groups:
            for i in range(0, len(group), self.batch_size):
                batch = group[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        return self._n_batches


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_bipartite(batch: Iterable[BCSample]) -> BipartiteBatch:
    """Stack a list of :class:`BCSample` into one batched graph."""
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
