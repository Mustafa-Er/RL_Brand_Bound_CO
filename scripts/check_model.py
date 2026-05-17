"""Forward-pass sanity check for the GCNN.

Loads one demonstration pickle, constructs a randomly-initialized GCNN with
the right input dims, and runs forward/backward to verify shapes and gradient
flow.

Example
-------
    python -m scripts.check_model \\
        --demo data/rl_bb_dummy/demonstrations/combinatorial_auction/train_size/train/rb/instance_0018.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import torch

from rl_bb.models import GCNN, infer_feature_dims, obs_to_tensors
from rl_bb.utils import configure_logging, resolve_device

logger = logging.getLogger("rl_bb.check_model")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GCNN forward-pass smoke test.")
    p.add_argument("--demo", type=Path, required=True,
                   help="Path to a .pkl demonstration trajectory.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    device = resolve_device(args.device)

    with open(args.demo, "rb") as f:
        traj = pickle.load(f)
    if not traj["observations"]:
        raise SystemExit(
            f"{args.demo} has zero branching decisions; pick another demo."
        )

    sample_obs = traj["observations"][0]
    sample_action_set = traj["action_sets"][0]

    d_var, d_cons, d_edge = infer_feature_dims(sample_obs)
    logger.info("Feature dims: var=%d cons=%d edge=%d", d_var, d_cons, d_edge)

    model = GCNN(
        var_dim=d_var,
        cons_dim=d_cons,
        edge_dim=d_edge,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("GCNN params: %d  hidden=%d  layers=%d",
                n_params, args.hidden, args.n_layers)

    t = obs_to_tensors(sample_obs).to(device)
    logits, value = model.forward_with_mask(t, sample_action_set)
    logger.info("logits shape: %s  value: %.4f", tuple(logits.shape), float(value))
    logger.info("argmax variable: %d  finite logits: %d/%d",
                int(torch.argmax(logits)),
                int(torch.isfinite(logits).sum()),
                logits.numel())

    # Quick backward to verify gradients flow.
    target = torch.tensor([sample_action_set[0]], device=device, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    logger.info("dummy CE loss=%.4f  grad-norm=%.4f", float(loss), grad_norm)


if __name__ == "__main__":
    main()
