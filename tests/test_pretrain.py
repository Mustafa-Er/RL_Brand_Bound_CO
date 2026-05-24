"""Stage 5 tests: BC dataset, batched collate, overfit-one-sample, ckpt I/O."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from rl_bb.envs import EnvConfig, make_expert_env
from rl_bb.experts import RBPolicy, collect_many
from rl_bb.instances import write_instances
from rl_bb.models import GCNN, infer_feature_dims
from rl_bb.training import (
    BCDataset,
    PretrainConfig,
    PretrainPaths,
    collate_bipartite,
    load_pretrained_gcnn,
    run_pretrain,
)
from rl_bb.training.pretrain import _step_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_demo_dir(tmp_path_factory) -> Path:
    """Produce a small RB demonstration dataset under tmp/<split>/rb/."""
    root = tmp_path_factory.mktemp("demos")
    inst_dir = root / "inst"
    inst_dir.mkdir()
    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        inst_dir,
        n=3,
        seed=7,
    )
    instances = sorted(inst_dir.glob("*.mps"))
    env_cfg = EnvConfig(time_limit_s=30.0, extra_scip_params={"presolving/maxrounds": 0})

    train_out = root / "train" / "rb"
    val_out = root / "val" / "rb"
    collect_many(instances[:2], RBPolicy(), env_cfg, train_out, seed=0)
    collect_many(instances[2:], RBPolicy(), env_cfg, val_out, seed=1)
    return root


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

def test_bcdataset_collects_samples(tiny_demo_dir: Path):
    ds = BCDataset(tiny_demo_dir / "train" / "rb")
    if len(ds) == 0:
        pytest.skip("no branching decisions in the generated demos")
    s = ds[0]
    assert s.action in s.action_set


def test_collate_batches_two_samples(tiny_demo_dir: Path):
    ds = BCDataset(tiny_demo_dir / "train" / "rb")
    if len(ds) < 2:
        pytest.skip("need at least 2 decisions")
    batch = collate_bipartite([ds[0], ds[1]])
    # graph_ids must label each variable with 0 or 1, two graphs total.
    assert set(batch.graph_ids.tolist()) == {0, 1}
    # actions live in the global indexing scheme.
    n_vars = batch.tensors.var_features.shape[0]
    for a in batch.actions:
        assert 0 <= a < n_vars
    # action_sets contain valid global indices.
    for aset in batch.action_sets:
        for v in aset:
            assert 0 <= v < n_vars


# ---------------------------------------------------------------------------
# Loss + overfit-one-sample
# ---------------------------------------------------------------------------

def test_overfit_one_sample(tiny_demo_dir: Path):
    ds = BCDataset(tiny_demo_dir / "train" / "rb")
    if len(ds) == 0:
        pytest.skip("no decisions to fit")
    sample = ds[0]
    d_var, d_cons, d_edge = infer_feature_dims(sample.observation)
    model = GCNN(d_var, d_cons, d_edge, hidden=32, n_layers=2)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch = collate_bipartite([sample])

    initial_loss = None
    for _ in range(200):
        logits, _v = model(batch.tensors, batch.graph_ids)
        loss, _m = _step_loss(logits, batch)
        if initial_loss is None:
            initial_loss = float(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert float(loss) < initial_loss * 0.2, (
        f"loss did not shrink: {initial_loss:.3f} -> {float(loss):.3f}"
    )


# ---------------------------------------------------------------------------
# End-to-end run + checkpoint round-trip
# ---------------------------------------------------------------------------

def test_run_pretrain_end_to_end(tiny_demo_dir: Path, tmp_path: Path):
    paths = PretrainPaths(
        train_root=tiny_demo_dir / "train" / "rb",
        val_root=tiny_demo_dir / "val" / "rb",
        ckpt_dir=tmp_path / "ckpts",
    )
    train_ds = BCDataset(paths.train_root)
    if len(train_ds) == 0:
        pytest.skip("no training samples")

    out = run_pretrain(
        paths,
        PretrainConfig(hidden=16, n_layers=1, lr=1e-3, epochs=2, batch_size=4, device="cpu", seed=0),
    )
    assert len(out["history"]) == 2
    ckpt = paths.ckpt_dir / "pretrain_best.pt"
    assert ckpt.exists()

    model = load_pretrained_gcnn(ckpt, device="cpu")
    assert isinstance(model, GCNN)
