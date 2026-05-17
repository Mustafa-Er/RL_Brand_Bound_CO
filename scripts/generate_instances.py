"""CLI entry point for instance generation.

Examples
--------
Generate dummy instances for both problem types::

    python -m scripts.generate_instances \\
        --config config/base.yaml --config config/dummy.yaml

Generate only set covering::

    python -m scripts.generate_instances \\
        --config config/base.yaml --config config/dummy.yaml \\
        --problem set_covering
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from rl_bb.instances import generate_all
from rl_bb.utils import (
    configure_logging,
    load_config,
    resolve_path,
    set_seed,
)

PROBLEMS = ("set_covering", "combinatorial_auction")
logger = logging.getLogger("rl_bb.generate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MILP instances via Ecole.")
    p.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config file. Repeat for layered overrides.",
    )
    p.add_argument(
        "--problem",
        choices=PROBLEMS,
        default=None,
        help="Limit generation to a single problem type (default: both).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)

    log_dir = resolve_path(cfg["paths"]["log_dir"])
    configure_logging(log_dir=log_dir, filename="generate_instances.log")

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    data_dir = resolve_path(cfg["paths"]["data_dir"]) / cfg["experiment"]["name"]
    counts = cfg["instances"]["counts"]

    problems = (args.problem,) if args.problem else PROBLEMS
    grand_total = 0
    t0 = time.perf_counter()

    for problem in problems:
        if problem not in cfg["instances"]:
            logger.warning("No size config for problem %s, skipping.", problem)
            continue
        sizes = cfg["instances"][problem]
        logger.info("Generating problem=%s under %s", problem, data_dir)
        results = generate_all(
            problem_type=problem,
            sizes=sizes,
            counts=counts,
            data_dir=data_dir,
            base_seed=seed,
        )
        for (regime, split), n in results.items():
            logger.info("  %-18s %-5s -> %d", regime, split, n)
            grand_total += n

    elapsed = time.perf_counter() - t0
    logger.info("Done: %d instances in %.1fs.", grand_total, elapsed)


if __name__ == "__main__":
    main()
