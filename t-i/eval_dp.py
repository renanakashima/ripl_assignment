#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from ripl.config import EvalConfig, TrainConfig, parse_config, validate_policy_config
from ripl.envs import make_eval_envs
from ripl.evaluation import evaluate_policy
from ripl.policy import DiffusionPolicy
from ripl.runtime import ensure_supported_python, json_ready, seed_everything, select_device


def main() -> None:
    ensure_supported_python()
    config = parse_config(EvalConfig)
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
    if config.act_horizon is not None:
        train_config.act_horizon = config.act_horizon
    validate_policy_config(train_config)
    seed = train_config.seed if config.seed is None else config.seed
    seed_everything(seed, train_config.torch_deterministic)

    sim_backend = config.sim_backend or train_config.sim_backend
    output_dir = (
        Path(config.output_dir).expanduser()
        if config.output_dir
        else checkpoint_path.parent.parent / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    envs = make_eval_envs(
        env_id=train_config.env_id,
        num_envs=config.num_eval_envs,
        sim_backend=sim_backend,
        control_mode=train_config.control_mode,
        obs_mode=train_config.obs_mode,
        obs_horizon=train_config.obs_horizon,
        max_episode_steps=train_config.max_episode_steps,
        video_dir=output_dir / "videos" if config.capture_video else None,
    )
    try:
        policy = DiffusionPolicy(envs, train_config).to(device)
        policy.load_state_dict(checkpoint["ema_policy"])
        raw_metrics = evaluate_policy(
            config.num_eval_episodes,
            policy,
            envs,
            device,
            sim_backend,
            seed=seed,
        )
        results = {
            "checkpoint": str(checkpoint_path),
            "iteration": checkpoint.get("iteration"),
            "num_eval_episodes": config.num_eval_episodes,
            "seed": seed,
            "act_horizon": train_config.act_horizon,
            "metrics": {key: float(np.mean(values)) for key, values in raw_metrics.items()},
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(json_ready(results), handle, indent=2)
        print(json.dumps(results, indent=2))
        print(f"Evaluation artifacts: {output_dir}")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
