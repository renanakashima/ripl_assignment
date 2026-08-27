#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

T_I_ROOT = Path(__file__).resolve().parents[1] / "t-i"
if str(T_I_ROOT) not in sys.path:
    sys.path.insert(0, str(T_I_ROOT))

from ripl.config import (
    TrainConfig,
    validate_policy_config,
)
from ripl.policy import DiffusionPolicy
from ripl.runtime import ensure_supported_python, json_ready, seed_everything, select_device

from ripl_t2.config import parse_failure_eval_config
from ripl_t2.envs import PushTPoseRange, make_failure_envs
from ripl_t2.failure_analysis import TRAJECTORY_COLUMNS, wilson_interval
from ripl_t2.failure_evaluation import evaluate_pusht_failures


def main() -> None:
    ensure_supported_python()
    config = parse_failure_eval_config()
    checkpoint_path = Path(config.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = select_device(config.cuda)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_values = checkpoint["train_config"]
    valid_fields = {field.name for field in fields(TrainConfig)}
    train_config = TrainConfig(
        **{key: value for key, value in train_values.items() if key in valid_fields}
    )
    if train_config.env_id != "PushT-v1":
        raise ValueError("Failure evaluation requires a PushT-v1 checkpoint")
    if config.act_horizon is not None:
        train_config.act_horizon = config.act_horizon
    validate_policy_config(train_config)

    seed = train_config.seed if config.seed is None else config.seed
    seed_everything(seed, train_config.torch_deterministic)
    pose_range = PushTPoseRange(
        x_rel_min=config.x_rel_min,
        x_rel_max=config.x_rel_max,
        y_rel_min=config.y_rel_min,
        y_rel_max=config.y_rel_max,
        theta_deg_min=config.theta_deg_min,
        theta_deg_max=config.theta_deg_max,
    )
    sim_backend = config.sim_backend or train_config.sim_backend
    output_dir = (
        Path(config.output_dir).expanduser()
        if config.output_dir
        else checkpoint_path.parent.parent / "failure-evaluation" / config.failure_mode_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = make_failure_envs(
        env_id=train_config.env_id,
        num_envs=config.num_eval_envs,
        sim_backend=sim_backend,
        control_mode=train_config.control_mode,
        obs_mode=train_config.obs_mode,
        obs_horizon=train_config.obs_horizon,
        max_episode_steps=train_config.max_episode_steps,
        video_dir=output_dir / "videos" if config.capture_video else None,
        pusht_pose_range=pose_range.to_dict(),
    )
    try:
        policy = DiffusionPolicy(envs, train_config).to(device)
        policy.load_state_dict(checkpoint["ema_policy"])
        evaluation = evaluate_pusht_failures(
            config.num_eval_episodes,
            policy,
            envs,
            device,
            sim_backend,
            seed=seed,
            failure_mode_name=config.failure_mode_name,
        )
    finally:
        envs.close()

    successes = sum(record["success_once"] for record in evaluation.episodes)
    tag_counts = Counter(
        tag for record in evaluation.episodes for tag in record.get("failure_tags", [])
    )
    results = {
        "checkpoint": str(checkpoint_path),
        "iteration": checkpoint.get("iteration"),
        "failure_mode_name": config.failure_mode_name,
        "pose_range": pose_range.to_dict(),
        "num_eval_episodes": config.num_eval_episodes,
        "num_eval_envs": config.num_eval_envs,
        "seed": seed,
        "act_horizon": train_config.act_horizon,
        "successes": successes,
        "success_rate": successes / config.num_eval_episodes,
        "success_rate_wilson_95": wilson_interval(successes, config.num_eval_episodes),
        "metrics": {key: float(np.mean(values)) for key, values in evaluation.metrics.items()},
        "failure_tag_counts": dict(sorted(tag_counts.items())),
        "trajectory_columns": list(TRAJECTORY_COLUMNS),
        "capture_video": config.capture_video,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(results), handle, indent=2)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in evaluation.episodes:
            handle.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
    if config.save_trajectories:
        np.savez_compressed(
            output_dir / "trajectories.npz",
            **{
                f"episode_{index:06d}": trajectory
                for index, trajectory in enumerate(evaluation.trajectories)
            },
        )

    print(json.dumps(json_ready(results), indent=2))
    print(f"Failure-evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
