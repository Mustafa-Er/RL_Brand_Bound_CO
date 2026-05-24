"""CLI for behavioral-cloning pretraining on RB demonstrations.

Example
-------
    python -m scripts.pretrain \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem combinatorial_auction
"""
from __future__ import annotations

import argparse
import logging

from rl_bb.training import PretrainConfig, PretrainPaths, run_pretrain
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_device,
    resolve_path,
    set_seed,
)

logger = logging.getLogger("rl_bb.pretrain")

PROBLEMS = ("set_covering", "combinatorial_auction")


EXPERT_NAME = "rb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioral cloning pretraining on RB demos.")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--problem", choices=PROBLEMS, required=True)
    p.add_argument("--regime", default="train_size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    log_dir = resolve_path(cfg["paths"]["log_dir"])
    configure_logging(log_dir=log_dir, filename="pretrain.log")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))
    logger.info("Device: %s", device)

    demos_root = (
        resolve_path(cfg["paths"]["data_dir"])
        / cfg["experiment"]["name"]
        / "demonstrations"
        / args.problem
        / args.regime
    )
    ckpt_dir = resolve_path(cfg["paths"]["ckpt_dir"]) / cfg["experiment"]["name"]
    paths = PretrainPaths(
        train_root=demos_root / "train" / EXPERT_NAME,
        val_root=demos_root / "val" / EXPERT_NAME,
        ckpt_dir=ckpt_dir,
    )
    logger.info("train demos: %s", paths.train_root)
    logger.info("val demos:   %s", paths.val_root)
    logger.info("ckpt dir:    %s", paths.ckpt_dir)

    mcfg = cfg.get("model", {}).get("gcnn", {})
    pcfg = cfg.get("pretrain", {})
    pretrain_cfg = PretrainConfig(
        hidden=int(mcfg.get("embedding_size", 64)),
        n_layers=int(mcfg.get("n_layers", 2)),
        lr=float(pcfg.get("lr") or 1e-3),
        epochs=int(pcfg.get("epochs") or 30),
        batch_size=int(pcfg.get("batch_size") or 32),
        device=device,
        seed=seed,
    )
    out = run_pretrain(paths, pretrain_cfg)
    logger.info("Best val loss: %.4f", out["best_val_loss"])


if __name__ == "__main__":
    main()
