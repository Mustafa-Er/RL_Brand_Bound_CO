"""Aggregate evaluation results over seeds and instances."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from rl_bb.eval.runner import InstanceResult

METRIC_FIELDS = (
    "wall_time_s",
    "scip_wall_time_s",
    "n_nodes",
    "lp_iterations",
    "dual_integral",
    "n_decisions",
)


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def _safe_std(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate(results: Iterable[InstanceResult]) -> list[dict]:
    """Group by (policy, regime, split) and compute mean ± std per metric."""
    buckets: dict[tuple[str, str, str], list[InstanceResult]] = defaultdict(list)
    for r in results:
        buckets[(r.policy, r.regime, r.split)].append(r)

    rows: list[dict] = []
    for (policy, regime, split), group in sorted(buckets.items()):
        row = {"policy": policy, "regime": regime, "split": split, "n": len(group)}
        for field in METRIC_FIELDS:
            values = [getattr(g, field) for g in group]
            row[f"{field}_mean"] = _safe_mean(values)
            row[f"{field}_std"] = _safe_std(values)
        rows.append(row)
    return rows


def write_results(
    per_instance: list[InstanceResult],
    summary: list[dict],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / "eval_detail.csv"
    summary_path = out_dir / "eval_summary.csv"

    with open(detail, "w", newline="", encoding="utf-8") as f:
        if per_instance:
            writer = csv.DictWriter(f, fieldnames=list(per_instance[0].as_dict().keys()))
            writer.writeheader()
            for r in per_instance:
                writer.writerow(r.as_dict())

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return detail, summary_path
