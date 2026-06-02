"""Stage 3 sanity tests for expert policies and demonstration collection."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import pytest

from rl_bb.envs import make_expert_env
from rl_bb.experts import FSBPolicy, RBPolicy, RandomPolicy
from rl_bb.stage_1_instances import write_instances
from rl_bb.stage_2_pretrain import collect_demonstrations as collect_many


@pytest.fixture(scope="module")
def harder_instance(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("expert_test_instances")
    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        out_dir,
        n=1,
        seed=2024,
    )
    return out_dir / "instance_0000.mps"


# ---------------------------------------------------------------------------
# Policy contract
# ---------------------------------------------------------------------------

def test_random_policy_returns_valid_action():
    rp = RandomPolicy(seed=0)
    action = rp.act(observation=None, action_set=[3, 7, 11], model=None)
    assert action in (3, 7, 11)


@pytest.mark.parametrize("policy_cls", [FSBPolicy, RBPolicy])
def test_expert_picks_action_from_action_set(harder_instance: Path, no_presolve_env_cfg, policy_cls):
    env = make_expert_env(no_presolve_env_cfg)
    env.seed(0)
    policy = policy_cls()
    policy.reset()

    obs, action_set, _r, done, _info = env.reset(str(harder_instance))
    if done:
        pytest.skip("instance solved at root; cannot exercise expert")

    action = policy.act(obs, action_set, env.model)
    assert action in list(action_set)


# ---------------------------------------------------------------------------
# Demonstration collection end-to-end
# ---------------------------------------------------------------------------

def test_collect_many_writes_one_pickle_per_instance(tmp_path: Path, no_presolve_env_cfg):
    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        inst_dir,
        n=2,
        seed=11,
    )
    instances = sorted(inst_dir.glob("*.mps"))
    out_dir = tmp_path / "demos"

    written = collect_many(instances, FSBPolicy(), no_presolve_env_cfg, out_dir, seed=0)
    assert written == 2
    pkls = sorted(out_dir.glob("*.pkl"))
    assert len(pkls) == 2

    with open(pkls[0], "rb") as f:
        traj = pickle.load(f)
    assert traj["expert"] == "FSBPolicy"
    assert len(traj["observations"]) == len(traj["actions"])
    assert all(isinstance(a, int) for a in traj["actions"])
