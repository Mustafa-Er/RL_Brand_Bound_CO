"""Stage 2 sanity tests for the Branching environment wrapper."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from rl_bb.envs import (
    DFS_SCIP_PARAMS,
    EnvConfig,
    env_config_from_dict,
    make_branching_env,
)
from rl_bb.stage_1_instances import write_instances


@pytest.fixture(scope="module")
def small_instance(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("env_test_instances")
    write_instances(
        "set_covering",
        {"n_rows": 50, "n_cols": 100, "density": 0.1},
        out_dir,
        n=1,
        seed=2024,
    )
    return out_dir / "instance_0000.mps"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_env_config_defaults_from_empty_dict():
    cfg = env_config_from_dict({})
    assert cfg.time_limit_s == 3600.0
    assert cfg.gap_limit == 0.0


def test_dfs_params_present():
    assert "nodeselection/dfs/stdpriority" in DFS_SCIP_PARAMS
    assert DFS_SCIP_PARAMS["nodeselection/dfs/stdpriority"] > 200_000


# ---------------------------------------------------------------------------
# End-to-end rollout
# ---------------------------------------------------------------------------

def test_random_policy_solves_small_instance(small_instance: Path):
    env = make_branching_env(EnvConfig(time_limit_s=30.0))
    env.seed(0)
    rng = random.Random(0)

    obs, action_set, reward, done, info = env.reset(str(small_instance))
    steps = 0
    cum_reward = 0.0
    while not done and steps < 5000:
        action = rng.choice(list(action_set))
        obs, action_set, reward, done, info = env.step(action)
        cum_reward += float(reward)
        steps += 1

    assert done, "env should terminate within the time/node budget"
    # SCIP's presolver may solve very small instances at the root, in which
    # case zero branching decisions is expected. We only require finiteness
    # and that SCIP reported at least one node processed.
    assert cum_reward == cum_reward, "reward must not be NaN"
    assert info.get("nb_nodes", 0.0) >= 1


def test_branching_triggers_on_harder_instance(tmp_path: Path):
    """Use a larger combinatorial auction so SCIP cannot solve at the root."""
    from rl_bb.stage_1_instances import write_instances

    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        tmp_path,
        n=1,
        seed=99,
    )
    inst = tmp_path / "instance_0000.mps"

    # Disable presolve so branching is forced (test focuses on env wiring).
    env = make_branching_env(
        EnvConfig(
            time_limit_s=30.0,
            extra_scip_params={"presolving/maxrounds": 0},
        )
    )
    env.seed(0)
    rng = random.Random(0)

    obs, action_set, reward, done, info = env.reset(str(inst))
    steps = 0
    while not done and steps < 200:
        action = rng.choice(list(action_set))
        obs, action_set, reward, done, info = env.step(action)
        steps += 1

    assert steps > 0, "branching should be exercised on a harder instance"
