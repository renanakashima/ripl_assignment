import numpy as np
import pytest

from ripl_t2.config import parse_failure_eval_config
from ripl_t2.failure_analysis import (
    TRAJECTORY_COLUMNS,
    analyze_failure_records,
    angular_distance_deg,
    candidate_failure_tags,
    raw_pose_to_xy_yaw,
    trajectory_summary,
    wilson_interval,
)


def test_failure_eval_config_accepts_target_pose_range():
    config = parse_failure_eval_config(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--failure-mode-name",
            "orientation-stall",
            "--theta-deg-min",
            "120",
            "--theta-deg-max",
            "210",
        ]
    )
    assert config.failure_mode_name == "orientation-stall"
    assert config.theta_deg_min == 120
    assert config.theta_deg_max == 210


def test_pose_and_circular_angle_helpers():
    theta = np.deg2rad(300.0)
    pose = np.array([[0.2, -0.1, 0.02, np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]])
    x, y, yaw = raw_pose_to_xy_yaw(pose)
    assert x.tolist() == [0.2]
    assert y.tolist() == [-0.1]
    assert yaw[0] == pytest.approx(300.0)
    assert angular_distance_deg(350.0, 10.0) == pytest.approx(20.0)


def test_trajectory_summary_and_failure_tags_are_auditable():
    assert len(TRAJECTORY_COLUMNS) == 7
    trajectory = np.array(
        [
            [-0.25, 0.05, 120.0, 0.10, -0.32, 0.28, 0.03],
            [-0.16, -0.08, 210.0, 0.72, -0.20, -0.02, 0.03],
            [-0.15, -0.09, 220.0, 0.45, -0.18, -0.05, 0.03],
        ]
    )
    summary = trajectory_summary(trajectory, -0.156, -0.1, 300.0)
    assert summary["maximum_overlap"] == pytest.approx(0.72)
    assert summary["overlap_regression"] == pytest.approx(0.27)
    tags = candidate_failure_tags(False, summary)
    assert "progress_regression" in tags
    assert "near_goal_misalignment" in tags


def test_failure_record_analysis_reports_pose_cells_and_tag_candidates():
    records = []
    for index in range(8):
        failure = index < 4
        records.append(
            {
                "initial_x_rel": -0.08 if index < 4 else 0.08,
                "initial_y_rel": -0.05,
                "initial_theta_deg": 45.0,
                "evaluation_seed": index % 2,
                "success_once": not failure,
                "failure_tags": ["low_progress_or_contact_loss"] if failure else [],
            }
        )
    analysis = analyze_failure_records(records, x_bins=2, y_bins=1, theta_bins=1)
    assert analysis["overall"]["success_rate"] == pytest.approx(0.5)
    assert analysis["across_seed"]["num_seeds"] == 2
    assert analysis["across_seed"]["mean_success_rate"] == pytest.approx(0.5)
    assert analysis["high_failure_cells"][0]["failure_rate"] == pytest.approx(1.0)
    candidate = analysis["candidate_regions_by_tag"]["low_progress_or_contact_loss"][0]
    assert candidate["x_rel_range"] == [-0.1, 0.0]
    assert candidate["tag_rate"] == pytest.approx(1.0)


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(30, 100)
    assert low < 0.30 < high
