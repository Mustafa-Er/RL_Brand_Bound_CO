"""CLI for PPO training (Stage 6).

Requires the Stage-5 BC checkpoint at
``checkpoints/<experiment.name>/pretrain_best.pt``.

Example
-------
    python -m scripts.ppo \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem combinatorial_auction
"""
from __future__ import annotations

import argparse
import logging

from rl_bb.envs import env_config_from_dict
from rl_bb.training import PPOConfig, PPOPaths, run_ppo
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_device,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.ppo")

PROBLEMS = ("set_covering", "combinatorial_auction")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training on the branching env.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--regime", default="train_size")
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    log_dir = resolve_path(cfg["paths"]["log_dir"])
    configure_logging(log_dir=log_dir, filename="ppo.log")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))
    logger.info("Device: %s", device)

    data_root = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    instance_dir = data_root / args.problem / args.regime / args.split
    ckpt_dir = resolve_path(cfg["paths"]["ckpt_dir"]) / cfg["experiment"]["name"]
    paths = PPOPaths(
        instance_dir=instance_dir,
        pretrain_ckpt=ckpt_dir / "pretrain_best.pt",
        ckpt_dir=ckpt_dir,
    )
    logger.info("instances:    %s", paths.instance_dir)
    logger.info("BC ckpt:      %s", paths.pretrain_ckpt)
    logger.info("PPO ckpt dir: %s", paths.ckpt_dir)

    env_cfg = env_config_from_dict(cfg.get("env", {}))
    pcfg = cfg.get("ppo", {})
    ppo_cfg = PPOConfig(
        gamma=float(pcfg.get("gamma", 0.99)),
        gae_lambda=float(pcfg.get("gae_lambda", 0.95)),
        clip_eps=float(pcfg.get("clip_eps", 0.2)),
        lr=float(pcfg.get("lr") or 3e-5),
        iterations=int(pcfg.get("iterations") or 50),
        rollouts_per_iter=int(pcfg.get("rollouts_per_iter") or 8),
        update_epochs=int(pcfg.get("update_epochs") or 4),
        minibatch_size=int(pcfg.get("minibatch_size") or 16),
        value_coef=float(pcfg.get("value_coef", 0.25)),
        entropy_coef=float(pcfg.get("entropy_coef", 0.01)),
        device=device,
        seed=seed,
    )
    out = run_ppo(paths, env_cfg, ppo_cfg)
    logger.info("Best mean reward: %.4f", out["best_mean_reward"])


if __name__ == "__main__":
    main()
