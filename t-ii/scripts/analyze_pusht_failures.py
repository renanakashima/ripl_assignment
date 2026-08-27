#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

T_II_ROOT = Path(__file__).resolve().parents[1]
T_I_ROOT = T_II_ROOT.parent / "t-i"
for root in (T_II_ROOT, T_I_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from ripl.runtime import json_ready

from ripl_t2.failure_analysis import analyze_failure_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Push-T failure diagnostics and propose targeted pose regions."
    )
    parser.add_argument("episodes", nargs="+", type=Path, help="episodes.jsonl files or folders")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-bins", type=int, default=2)
    parser.add_argument("--y-bins", type=int, default=3)
    parser.add_argument("--theta-bins", type=int, default=4)
    parser.add_argument("--min-cell-episodes", type=int, default=3)
    return parser.parse_args()


def resolve_episode_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("episodes.jsonl")))
        elif item.is_file():
            paths.append(item)
        else:
            raise FileNotFoundError(f"Episode input not found: {item}")
    if not paths:
        raise FileNotFoundError("No episodes.jsonl inputs were found")
    return paths


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def write_cell_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fieldnames = [
        "x_rel_min",
        "x_rel_max",
        "y_rel_min",
        "y_rel_max",
        "theta_deg_min",
        "theta_deg_max",
        "episodes",
        "success_rate",
        "failure_rate",
        "failure_rate_wilson_95_low",
        "failure_rate_wilson_95_high",
        "failure_tag_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "x_rel_min": cell["x_rel_range"][0],
                    "x_rel_max": cell["x_rel_range"][1],
                    "y_rel_min": cell["y_rel_range"][0],
                    "y_rel_max": cell["y_rel_range"][1],
                    "theta_deg_min": cell["theta_deg_range"][0],
                    "theta_deg_max": cell["theta_deg_range"][1],
                    "episodes": cell["episodes"],
                    "success_rate": cell["success_rate"],
                    "failure_rate": cell["failure_rate"],
                    "failure_rate_wilson_95_low": cell["failure_rate_wilson_95"][0],
                    "failure_rate_wilson_95_high": cell["failure_rate_wilson_95"][1],
                    "failure_tag_counts": json.dumps(cell["failure_tag_counts"], sort_keys=True),
                }
            )


def main() -> None:
    args = parse_args()
    paths = resolve_episode_paths(args.episodes)
    records = load_records(paths)
    analysis = analyze_failure_records(
        records,
        x_bins=args.x_bins,
        y_bins=args.y_bins,
        theta_bins=args.theta_bins,
        min_cell_episodes=args.min_cell_episodes,
    )
    analysis["episode_files"] = [str(path) for path in paths]
    analysis["num_records"] = len(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(analysis), handle, indent=2)
    write_cell_csv(args.output_dir / "high_failure_cells.csv", analysis["high_failure_cells"])

    print(json.dumps(json_ready(analysis["overall"]), indent=2))
    print(f"Failure analysis: {analysis_path}")


if __name__ == "__main__":
    main()
