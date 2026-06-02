"""Stage 7 sanity tests for the evaluation pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from rl_bb.envs import make_branching_env, make_expert_env
from rl_bb.experts import FSBPolicy, RandomPolicy
from rl_bb.stage_1_instances import write_instances
from rl_bb.stage_4_eval import (
    InstanceResult,
    aggregate,
    evaluate_on_instance,
    write_results,
)


@pytest.fixture(scope="module")
def hard_instance(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("eval_inst")
    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        out_dir,
        n=1,
        seed=5,
    )
    return out_dir / "instance_0000.mps"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_runner_returns_finite_metrics(hard_instance: Path, no_presolve_env_cfg):
    env = make_branching_env(no_presolve_env_cfg)
    res = evaluate_on_instance(
        env, hard_instance, RandomPolicy(seed=0),
        policy_name="random", regime="train_size", split="test", seed=0,
    )
    assert res.wall_time_s >= 0
    assert res.n_nodes >= 1
    assert res.n_decisions >= 0


def test_fsb_runs_with_expert_env(hard_instance: Path, no_presolve_env_cfg):
    env = make_expert_env(no_presolve_env_cfg)
    res = evaluate_on_instance(
        env, hard_instance, FSBPolicy(),
        policy_name="fsb", regime="train_size", split="test", seed=0,
    )
    assert res.n_nodes >= 1


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def test_aggregate_groups_by_policy_regime_split():
    results = [
        InstanceResult(
            instance=f"a{i}.mps", policy="random", regime="train_size", split="test",
            seed=0, wall_time_s=1.0 + 0.1 * i, scip_wall_time_s=0.9,
            n_nodes=10.0 + i, lp_iterations=100.0, dual_integral=None,
            n_decisions=5,
        )
        for i in range(3)
    ]
    summary = aggregate(results)
    assert len(summary) == 1
    row = summary[0]
    assert row["policy"] == "random"
    assert row["n"] == 3
    assert abs(row["n_nodes_mean"] - 11.0) < 1e-9


def test_write_results_creates_files(tmp_path: Path):
    results = [
        InstanceResult(
            instance="a.mps", policy="random", regime="train_size", split="test",
            seed=0, wall_time_s=1.0, scip_wall_time_s=0.5,
            n_nodes=10.0, lp_iterations=50.0, dual_integral=0.0,
            n_decisions=3,
        )
    ]
    summary = aggregate(results)
    detail, summary_path = write_results(results, summary, tmp_path)
    assert detail.exists()
    assert summary_path.exists()
    assert (tmp_path / "eval_summary.json").exists()
