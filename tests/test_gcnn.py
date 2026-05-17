"""Stage 4 sanity tests for the GCNN."""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from rl_bb.envs import EnvConfig, make_branching_env
from rl_bb.instances import write_instances
from rl_bb.models import GCNN, RLPolicy, infer_feature_dims, obs_to_tensors


@pytest.fixture(scope="module")
def real_observation():
    """Drive a branching env until we get an observation with candidates."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        td_path = Path(td)
        write_instances(
            "combinatorial_auction",
            {"n_items": 50, "n_bids": 200},
            td_path,
            n=1,
            seed=2024,
        )
        env = make_branching_env(
            EnvConfig(time_limit_s=30.0, extra_scip_params={"presolving/maxrounds": 0})
        )
        env.seed(0)
        obs, action_set, _r, done, _info = env.reset(str(td_path / "instance_0000.mps"))
        rng = random.Random(0)
        # Step a few times to land at a representative interior node.
        for _ in range(3):
            if done:
                break
            obs, action_set, _r, done, _info = env.step(rng.choice(list(action_set)))
        if done:
            pytest.skip("could not reach a non-terminal branching state")
        return obs, list(action_set)


# ---------------------------------------------------------------------------
# Shape and gradient flow
# ---------------------------------------------------------------------------

def test_obs_to_tensors_shapes(real_observation):
    obs, _ = real_observation
    t = obs_to_tensors(obs)
    assert t.var_features.ndim == 2
    assert t.cons_features.ndim == 2
    assert t.edge_index.shape[0] == 2
    assert t.edge_features.shape[0] == t.edge_index.shape[1]


def test_gcnn_forward_returns_expected_shapes(real_observation):
    obs, action_set = real_observation
    d_var, d_cons, d_edge = infer_feature_dims(obs)
    model = GCNN(d_var, d_cons, d_edge, hidden=32, n_layers=2)
    t = obs_to_tensors(obs)
    logits, value = model.forward_with_mask(t, action_set)
    assert logits.shape == (t.n_vars,)
    assert value.shape == ()
    # Masked logits: only candidate positions are finite.
    finite_mask = torch.isfinite(logits)
    assert finite_mask.sum().item() == len(action_set)
    assert int(torch.argmax(logits).item()) in action_set


def test_gcnn_gradients_flow(real_observation):
    obs, action_set = real_observation
    d_var, d_cons, d_edge = infer_feature_dims(obs)
    model = GCNN(d_var, d_cons, d_edge, hidden=16, n_layers=1)
    t = obs_to_tensors(obs)
    logits, _value = model.forward_with_mask(t, action_set)
    target = torch.tensor([action_set[0]], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached any parameter"
    assert all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# RLPolicy wrapper
# ---------------------------------------------------------------------------

def test_rl_policy_picks_action_from_set(real_observation):
    obs, action_set = real_observation
    d_var, d_cons, d_edge = infer_feature_dims(obs)
    model = GCNN(d_var, d_cons, d_edge, hidden=16, n_layers=1).eval()
    policy = RLPolicy(model, device="cpu", stochastic=False)
    action = policy.act(observation=obs, action_set=action_set, model=None)
    assert action in action_set
