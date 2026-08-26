#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_evaluations(evaluation_root: Path) -> list[dict[str, Any]]:
    metric_paths = sorted(evaluation_root.glob("seed-*/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No seed-*/metrics.json files found under {evaluation_root}")

    evaluations = []
    for metric_path in metric_paths:
        with metric_path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
        evaluation["_source"] = str(metric_path)
        evaluations.append(evaluation)
    return sorted(evaluations, key=lambda item: int(item["seed"]))


def aggregate_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(item["seed"]) for item in evaluations]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Evaluation seeds must be unique, got {seeds}")

    checkpoints = {item["checkpoint"] for item in evaluations}
    if len(checkpoints) != 1:
        raise ValueError("All evaluations must use the same checkpoint")

    episode_counts = [int(item["num_eval_episodes"]) for item in evaluations]
    metric_names = set(evaluations[0]["metrics"])
    if any(set(item["metrics"]) != metric_names for item in evaluations[1:]):
        raise ValueError("All evaluations must contain the same metrics")

    metrics = {}
    for metric_name in sorted(metric_names):
        values = [float(item["metrics"][metric_name]) for item in evaluations]
        weighted_mean = sum(
            value * count for value, count in zip(values, episode_counts, strict=True)
        ) / sum(episode_counts)
        metrics[metric_name] = {
            "per_seed": {str(seed): value for seed, value in zip(seeds, values, strict=True)},
            "mean": weighted_mean,
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "checkpoint": checkpoints.pop(),
        "checkpoint_iteration": evaluations[0].get("iteration"),
        "num_seeds": len(seeds),
        "seeds": seeds,
        "episodes_per_seed": episode_counts,
        "total_eval_episodes": sum(episode_counts),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed eval_dp.py metrics into one JSON report."
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()

    evaluation_root = args.evaluation_root.expanduser()
    summary = aggregate_evaluations(load_evaluations(evaluation_root))
    summary_path = evaluation_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Aggregate evaluation: {summary_path}")


if __name__ == "__main__":
    main()
