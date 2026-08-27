from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import sqrt
from typing import Any

import numpy as np

TRAJECTORY_COLUMNS = (
    "tee_x",
    "tee_y",
    "tee_yaw_deg",
    "overlap",
    "tcp_x",
    "tcp_y",
    "tcp_z",
)


def raw_pose_to_xy_yaw(raw_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ManiSkill [x, y, z, qw, qx, qy, qz] poses to planar values."""
    pose = np.asarray(raw_pose, dtype=np.float64)
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected raw poses ending in 7 values, got {pose.shape}")
    yaw = np.degrees(2.0 * np.arctan2(pose[..., 6], pose[..., 3])) % 360.0
    return pose[..., 0], pose[..., 1], yaw


def angular_distance_deg(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    return np.abs((first_array - second_array + 180.0) % 360.0 - 180.0)


def trajectory_summary(
    trajectory: np.ndarray,
    goal_x: float,
    goal_y: float,
    goal_yaw_deg: float,
) -> dict[str, float]:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(TRAJECTORY_COLUMNS):
        raise ValueError(
            f"Expected a [steps, {len(TRAJECTORY_COLUMNS)}] trajectory, got {values.shape}"
        )
    goal_distance = np.linalg.norm(values[:, :2] - np.array([goal_x, goal_y]), axis=1)
    angular_error = angular_distance_deg(values[:, 2], goal_yaw_deg)
    tcp_distance = np.linalg.norm(values[:, 4:6] - values[:, :2], axis=1)
    initial_xy = values[0, :2]
    displacement = np.linalg.norm(values[:, :2] - initial_xy, axis=1)
    best_overlap_step = int(np.argmax(values[:, 3]))
    return {
        "initial_goal_distance": float(goal_distance[0]),
        "minimum_goal_distance": float(goal_distance.min()),
        "final_goal_distance": float(goal_distance[-1]),
        "initial_angular_error_deg": float(angular_error[0]),
        "minimum_angular_error_deg": float(angular_error.min()),
        "final_angular_error_deg": float(angular_error[-1]),
        "maximum_overlap": float(values[:, 3].max()),
        "final_overlap": float(values[-1, 3]),
        "overlap_regression": float(values[:, 3].max() - values[-1, 3]),
        "best_overlap_step": best_overlap_step,
        "maximum_displacement": float(displacement.max()),
        "minimum_tcp_to_tee_distance": float(tcp_distance.min()),
        "final_tcp_to_tee_distance": float(tcp_distance[-1]),
    }


def candidate_failure_tags(success_once: bool, summary: dict[str, float]) -> list[str]:
    """Attach non-exclusive, auditable behavior tags for later human mode selection."""
    if success_once:
        return []

    tags: list[str] = []
    if summary["maximum_overlap"] >= 0.60 and summary["overlap_regression"] >= 0.15:
        tags.append("progress_regression")
    if (
        summary["minimum_goal_distance"] <= 0.08
        and summary["maximum_overlap"] < 0.90
        and summary["final_angular_error_deg"] >= 35.0
    ):
        tags.append("near_goal_misalignment")
    if (
        summary["final_goal_distance"] >= 0.15
        or summary["final_goal_distance"] - summary["minimum_goal_distance"] >= 0.08
    ):
        tags.append("overshoot_or_displacement")
    if summary["maximum_overlap"] < 0.40:
        tags.append("low_progress_or_contact_loss")
    if not tags:
        tags.append("stalled_partial_overlap")
    return tags


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z * sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total**2)) / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successes = sum(bool(record["success_once"]) for record in records)
    tag_counts = Counter(tag for record in records for tag in record.get("failure_tags", []))
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_wilson_95": wilson_interval(successes, total),
        "failure_rate": (total - successes) / total if total else 0.0,
        "failure_rate_wilson_95": wilson_interval(total - successes, total),
        "failure_tag_counts": dict(sorted(tag_counts.items())),
        "failure_tag_rates": {tag: count / total for tag, count in sorted(tag_counts.items())},
    }


def _bin_index(value: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


def analyze_failure_records(
    records: Iterable[dict[str, Any]],
    x_bins: int = 2,
    y_bins: int = 3,
    theta_bins: int = 4,
    min_cell_episodes: int = 3,
) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError("At least one episode record is required")
    if min(x_bins, y_bins, theta_bins, min_cell_episodes) < 1:
        raise ValueError("Bin counts and min_cell_episodes must be positive")

    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        seed = int(record.get("evaluation_seed", 0))
        records_by_seed.setdefault(seed, []).append(record)
    per_seed = {
        str(seed): _group_summary(seed_records)
        for seed, seed_records in sorted(records_by_seed.items())
    }
    seed_success_rates = [summary["success_rate"] for summary in per_seed.values()]
    across_seed = {
        "num_seeds": len(seed_success_rates),
        "mean_success_rate": float(np.mean(seed_success_rates)),
        "sample_std_success_rate": (
            float(np.std(seed_success_rates, ddof=1)) if len(seed_success_rates) > 1 else 0.0
        ),
    }

    edges = {
        "initial_x_rel": np.linspace(-0.10, 0.10, x_bins + 1),
        "initial_y_rel": np.linspace(-0.10, 0.20, y_bins + 1),
        "initial_theta_deg": np.linspace(0.0, 360.0, theta_bins + 1),
    }
    marginal_bins: dict[str, list[dict[str, Any]]] = {}
    for key, key_edges in edges.items():
        groups: list[list[dict[str, Any]]] = [[] for _ in range(len(key_edges) - 1)]
        for record in records:
            groups[_bin_index(float(record[key]), key_edges)].append(record)
        marginal_bins[key] = []
        for index, group in enumerate(groups):
            item = {
                "range": [float(key_edges[index]), float(key_edges[index + 1])],
                **_group_summary(group),
            }
            marginal_bins[key].append(item)

    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        cell_key = (
            _bin_index(float(record["initial_x_rel"]), edges["initial_x_rel"]),
            _bin_index(float(record["initial_y_rel"]), edges["initial_y_rel"]),
            _bin_index(float(record["initial_theta_deg"]), edges["initial_theta_deg"]),
        )
        cells.setdefault(cell_key, []).append(record)

    cell_summaries = []
    for (x_index, y_index, theta_index), group in cells.items():
        if len(group) < min_cell_episodes:
            continue
        cell_summaries.append(
            {
                "x_rel_range": [
                    float(edges["initial_x_rel"][x_index]),
                    float(edges["initial_x_rel"][x_index + 1]),
                ],
                "y_rel_range": [
                    float(edges["initial_y_rel"][y_index]),
                    float(edges["initial_y_rel"][y_index + 1]),
                ],
                "theta_deg_range": [
                    float(edges["initial_theta_deg"][theta_index]),
                    float(edges["initial_theta_deg"][theta_index + 1]),
                ],
                **_group_summary(group),
            }
        )
    cell_summaries.sort(
        key=lambda item: (item["failure_rate_wilson_95"][0], item["episodes"]), reverse=True
    )

    all_tags = sorted({tag for record in records for tag in record.get("failure_tags", [])})
    candidate_regions_by_tag: dict[str, list[dict[str, Any]]] = {}
    for tag in all_tags:
        candidates = []
        for cell in cell_summaries:
            tag_count = int(cell["failure_tag_counts"].get(tag, 0))
            candidate = {
                "x_rel_range": cell["x_rel_range"],
                "y_rel_range": cell["y_rel_range"],
                "theta_deg_range": cell["theta_deg_range"],
                "episodes": cell["episodes"],
                "tagged_failures": tag_count,
                "tag_rate": tag_count / cell["episodes"],
                "tag_rate_wilson_95": wilson_interval(tag_count, cell["episodes"]),
                "base_success_rate": cell["success_rate"],
            }
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (item["tag_rate_wilson_95"][0], item["tagged_failures"]),
            reverse=True,
        )
        candidate_regions_by_tag[tag] = candidates[:10]

    return {
        "overall": _group_summary(records),
        "per_seed": per_seed,
        "across_seed": across_seed,
        "marginal_bins": marginal_bins,
        "high_failure_cells": cell_summaries,
        "candidate_regions_by_tag": candidate_regions_by_tag,
        "binning": {
            "x_bins": x_bins,
            "y_bins": y_bins,
            "theta_bins": theta_bins,
            "min_cell_episodes": min_cell_episodes,
        },
    }
