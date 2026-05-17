"""Stage 6 tests: GAE math, rollout, and a single PPO iteration."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from rl_bb.envs import EnvConfig, make_branching_env
from rl_bb.instances import write_instances
from rl_bb.models import GCNN, infer_feature_dims
from rl_bb.training import PPOPaths, collect_trajectory, compute_gae, run_ppo


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def test_gae_zero_rewards_zero_values_zero_advantages():
    adv, ret = compute_gae([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], gamma=0.99, lam=0.95)
    assert adv == [0.0, 0.0, 0.0]
    assert ret == [0.0, 0.0, 0.0]


def test_gae_constant_reward_matches_closed_form():
    """With r_t = 1, V_t = 0, last_value = 0:

    δ_t  = 1 + γ·0 - 0 = 1      for t < T
    A_T-1 = 1
    A_t   = 1 + γλ A_{t+1}
    """
    gamma, lam = 0.99, 0.95
    T = 4
    rewards = [1.0] * T
    values = [0.0] * T
    adv, _ = compute_gae(rewards, values, gamma=gamma, lam=lam)
    # closed-form: A_t = sum_{k>=0} (γλ)^k for k=0..(T-1-t)
    expected = []
    for t in range(T):
        n = T - t
        s = sum((gamma * lam) ** k for k in range(n))
        expected.append(s)
    for a, e in zip(adv, expected):
        assert math.isclose(a, e, rel_tol=1e-6), (a, e)


def test_gae_returns_equal_advantages_plus_values():
    rewards = [0.5, -0.2, 1.0]
    values = [0.1, 0.0, -0.4]
    adv, ret = compute_gae(rewards, values, gamma=0.9, lam=0.8)
    for a, v, r in zip(adv, values, ret):
        assert math.isclose(r, a + v, rel_tol=1e-7)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def harder_instance(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("ppo_test_instances")
    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        out_dir,
        n=2,
        seed=2024,
    )
    return out_dir / "instance_0000.mps"


def test_collect_trajectory_runs(harder_instance: Path):
    env = make_branching_env(
        EnvConfig(time_limit_s=30.0, extra_scip_params={"presolving/maxrounds": 0})
    )
    env.seed(0)
    # Probe feature dims once via a fresh reset.
    obs, _aset, _r, _d, _i = env.reset(str(harder_instance))
    bipartite = obs[0] if isinstance(obs, tuple) else obs
    d_var, d_cons, d_edge = infer_feature_dims(bipartite)
    model = GCNN(d_var, d_cons, d_edge, hidden=16, n_layers=1)

    traj = collect_trajectory(env, harder_instance, model, device="cpu")
    if not traj.steps:
        pytest.skip("instance solved at root")
    s = traj.steps[0]
    assert s.action in s.action_set
    assert math.isfinite(s.log_prob)
    assert math.isfinite(s.value)


# ---------------------------------------------------------------------------
# End-to-end PPO iteration
# ---------------------------------------------------------------------------

def test_run_ppo_one_iteration(tmp_path: Path):
    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    write_instances(
        "combinatorial_auction",
        {"n_items": 50, "n_bids": 200},
        inst_dir,
        n=3,
        seed=11,
    )

    # Fabricate a "pretrained" checkpoint by saving a freshly-initialized GCNN.
    sample_obs = None
    env = make_branching_env(
        EnvConfig(time_limit_s=30.0, extra_scip_params={"presolving/maxrounds": 0})
    )
    env.seed(0)
    obs, *_ = env.reset(str(next(inst_dir.glob("*.mps"))))
    sample_obs = obs[0] if isinstance(obs, tuple) else obs
    d_var, d_cons, d_edge = infer_feature_dims(sample_obs)
    model = GCNN(d_var, d_cons, d_edge, hidden=16, n_layers=1)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    pretrain_ckpt = ckpt_dir / "pretrain_best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": {},
        "feature_dims": (d_var, d_cons, d_edge),
        "model_config": {"hidden": 16, "n_layers": 1},
        "epoch": 0,
        "val_loss": 0.0,
    }, pretrain_ckpt)

    paths = PPOPaths(instance_dir=inst_dir, pretrain_ckpt=pretrain_ckpt, ckpt_dir=ckpt_dir)
    out = run_ppo(
        paths,
        EnvConfig(time_limit_s=30.0, extra_scip_params={"presolving/maxrounds": 0}),
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2, lr=1e-4,
        iterations=1, rollouts_per_iter=2, update_epochs=1, minibatch_size=4,
        value_coef=0.5, entropy_coef=0.01,
        device="cpu", seed=0,
    )
    assert len(out["history"]) == 1
    assert (ckpt_dir / "ppo_latest.pt").exists()
