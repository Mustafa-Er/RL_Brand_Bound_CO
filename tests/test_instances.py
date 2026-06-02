"""Stage 1 sanity tests for the instance generator."""
from __future__ import annotations

from pathlib import Path

import pytest

from rl_bb.stage_1_instances import generate_all, make_generator, write_instances


# ---------------------------------------------------------------------------
# Generator construction
# ---------------------------------------------------------------------------

def test_make_generator_set_covering():
    gen = make_generator(
        "set_covering",
        {"n_rows": 50, "n_cols": 100, "density": 0.05},
        seed=42,
    )
    model = next(gen)
    pyscip = model.as_pyscipopt()
    # set covering has n_cols binary variables and n_rows constraints.
    assert pyscip.getNVars() == 100
    assert pyscip.getNConss() == 50


def test_make_generator_combinatorial_auction():
    gen = make_generator(
        "combinatorial_auction",
        {"n_items": 20, "n_bids": 50},
        seed=42,
    )
    model = next(gen)
    assert model.as_pyscipopt().getNVars() > 0


def test_make_generator_rejects_unknown_problem():
    with pytest.raises(ValueError):
        make_generator("travelling_salesman", {}, seed=0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_generator_is_deterministic(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cfg = {"n_rows": 50, "n_cols": 100, "density": 0.1}
    write_instances("set_covering", cfg, out_a, n=2, seed=123)
    write_instances("set_covering", cfg, out_b, n=2, seed=123)
    for i in range(2):
        a = (out_a / f"instance_{i:04d}.mps").read_bytes()
        b = (out_b / f"instance_{i:04d}.mps").read_bytes()
        assert a == b, f"instance {i} differs across runs with same seed"


# ---------------------------------------------------------------------------
# End-to-end mini run
# ---------------------------------------------------------------------------

def test_generate_all_end_to_end(tmp_path: Path):
    sizes = {
        "train_size":      {"n_rows": 50, "n_cols": 100, "density": 0.1},
        "transfer_medium": {"n_rows": 75, "n_cols": 150, "density": 0.1},
        "transfer_large":  {"n_rows": 100, "n_cols": 200, "density": 0.1},
    }
    counts = {"train": 2, "val": 1, "test": 1}
    results = generate_all(
        problem_type="set_covering",
        sizes=sizes,
        counts=counts,
        data_dir=tmp_path,
        base_seed=7,
    )
    # train split only for train_size regime; val/test for all three regimes.
    assert results[("train_size", "train")] == 2
    assert results[("train_size", "val")] == 1
    assert results[("transfer_medium", "test")] == 1
    assert results[("transfer_large", "test")] == 1
    assert ("transfer_medium", "train") not in results
