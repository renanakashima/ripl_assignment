from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from mani_skill.utils import common
from tqdm import tqdm

from ripl_t2.failure_analysis import (
    TRAJECTORY_COLUMNS,
    candidate_failure_tags,
    raw_pose_to_xy_yaw,
    trajectory_summary,
)


@dataclass
class FailureEvaluationResult:
    metrics: dict[str, np.ndarray]
    episodes: list[dict[str, Any]]
    trajectories: list[np.ndarray]


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _diagnostic_rows(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tee_pose = _to_numpy(info["tee_pose"])
    tcp_pose = _to_numpy(info["tcp_pose"])
    overlap = _to_numpy(info["overlap"]).reshape(-1)
    tee_x, tee_y, tee_yaw = raw_pose_to_xy_yaw(tee_pose)
    rows = np.column_stack(
        [
            tee_x,
            tee_y,
            tee_yaw,
            overlap,
            tcp_pose[:, 0],
            tcp_pose[:, 1],
            tcp_pose[:, 2],
        ]
    ).astype(np.float32)
    return rows, tee_pose


def _initial_diagnostics(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("initial_tee_pose", "initial_tcp_pose", "initial_overlap", "goal_tee_pose")
    missing = [key for key in required if key not in info]
    if missing:
        raise RuntimeError(f"Push-T reset diagnostics are missing: {', '.join(missing)}")
    initial_info = {
        "tee_pose": info["initial_tee_pose"],
        "tcp_pose": info["initial_tcp_pose"],
        "overlap": info["initial_overlap"],
    }
    rows, tee_pose = _diagnostic_rows(initial_info)
    return rows, tee_pose, _to_numpy(info["goal_tee_pose"])


def _terminal_episode_metrics(info: dict[str, Any]) -> dict[str, np.ndarray]:
    terminal_info = info.get("final_info", info)
    episode = terminal_info.get("episode")
    if not isinstance(episode, dict):
        raise TypeError("Terminal Push-T info does not contain vector episode metrics")
    return {key: _to_numpy(value).reshape(-1) for key, value in episode.items()}


def evaluate_pusht_failures(
    num_episodes: int,
    policy: Any,
    envs: Any,
    device: torch.device,
    sim_backend: str,
    seed: int,
    failure_mode_name: str,
    progress_bar: bool = True,
) -> FailureEvaluationResult:
    """Evaluate synchronized Push-T episodes and retain initial states and trajectories."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if not failure_mode_name.strip():
        raise ValueError("failure_mode_name must not be empty")

    was_training = policy.training
    policy.eval()
    metric_values: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_records: list[dict[str, Any]] = []
    saved_trajectories: list[np.ndarray] = []
    progress = tqdm(total=num_episodes, desc="failure evaluation") if progress_bar else None

    with torch.no_grad():
        observations, reset_info = envs.reset(seed=seed)
        initial_rows, initial_poses, goal_poses = _initial_diagnostics(reset_info)
        batch_trajectories = [[row.copy()] for row in initial_rows]
        completed = 0

        while completed < num_episodes:
            tensor_observations = common.to_tensor(observations, device)
            action_sequence = policy.get_action(tensor_observations)
            if sim_backend == "physx_cpu":
                action_sequence = action_sequence.cpu().numpy()

            terminal_info: dict[str, Any] | None = None
            for action_index in range(action_sequence.shape[1]):
                observations, _, _, truncated, info = envs.step(action_sequence[:, action_index])
                is_terminal = bool(_to_numpy(truncated).any())
                diagnostic_info = info.get("final_info", info) if is_terminal else info
                rows, _ = _diagnostic_rows(diagnostic_info)
                for env_index, row in enumerate(rows):
                    batch_trajectories[env_index].append(row.copy())
                if is_terminal:
                    if not bool(_to_numpy(truncated).all()):
                        raise RuntimeError(
                            "Vector episodes desynchronized during failure evaluation"
                        )
                    terminal_info = info
                    break

            if terminal_info is None:
                continue

            episode_metrics = _terminal_episode_metrics(terminal_info)
            for key, value in episode_metrics.items():
                metric_values[key].append(value)

            batch_count = min(envs.num_envs, num_episodes - completed)
            goal_x, goal_y, goal_yaw = raw_pose_to_xy_yaw(goal_poses)
            initial_x, initial_y, initial_yaw = raw_pose_to_xy_yaw(initial_poses)
            for env_index in range(batch_count):
                trajectory = np.asarray(batch_trajectories[env_index], dtype=np.float32)
                summary = trajectory_summary(
                    trajectory,
                    float(goal_x[env_index]),
                    float(goal_y[env_index]),
                    float(goal_yaw[env_index]),
                )
                per_episode_metrics = {
                    key: float(values[env_index]) for key, values in episode_metrics.items()
                }
                success_once = bool(per_episode_metrics.get("success_once", 0.0))
                record = {
                    "episode_index": completed + env_index,
                    "evaluation_seed": seed,
                    "vector_index": env_index,
                    "failure_mode_name": failure_mode_name,
                    "initial_x": float(initial_x[env_index]),
                    "initial_y": float(initial_y[env_index]),
                    "initial_theta_deg": float(initial_yaw[env_index]),
                    "initial_x_rel": float(initial_x[env_index] - goal_x[env_index]),
                    "initial_y_rel": float(initial_y[env_index] - goal_y[env_index]),
                    "goal_x": float(goal_x[env_index]),
                    "goal_y": float(goal_y[env_index]),
                    "goal_theta_deg": float(goal_yaw[env_index]),
                    "success_once": success_once,
                    "success_at_end": bool(per_episode_metrics.get("success_at_end", 0.0)),
                    "episode_metrics": per_episode_metrics,
                    **summary,
                }
                record["failure_tags"] = candidate_failure_tags(success_once, summary)
                episode_records.append(record)
                saved_trajectories.append(trajectory)

            completed += batch_count
            if progress:
                progress.update(batch_count)
            if completed >= num_episodes:
                break

            if "initial_tee_pose" in terminal_info:
                reset_info = terminal_info
            else:
                observations, reset_info = envs.reset()
            initial_rows, initial_poses, goal_poses = _initial_diagnostics(reset_info)
            batch_trajectories = [[row.copy()] for row in initial_rows]

    if progress:
        progress.close()
    policy.train(was_training)
    metrics = {
        key: np.concatenate([np.atleast_1d(value) for value in values])[:num_episodes]
        for key, values in metric_values.items()
    }
    if len(episode_records) != num_episodes or len(saved_trajectories) != num_episodes:
        raise RuntimeError(
            f"Expected {num_episodes} episode records, got "
            f"{len(episode_records)} records and {len(saved_trajectories)} trajectories"
        )
    return FailureEvaluationResult(metrics, episode_records, saved_trajectories)


__all__ = [
    "TRAJECTORY_COLUMNS",
    "FailureEvaluationResult",
    "evaluate_pusht_failures",
]
