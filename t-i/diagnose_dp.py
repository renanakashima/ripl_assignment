#!/usr/bin/env python3
"""Diagnose visual conditioning and offline action quality for a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Subset

from ripl.config import TrainConfig, validate_policy_config
from ripl.data import DiffusionDemoDataset, move_batch_to_device
from ripl.envs import get_unwrapped_observation_metadata, make_eval_envs
from ripl.policy import DiffusionPolicy
from ripl.runtime import ensure_supported_python, json_ready, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_supported_python()
    device = select_device(cuda=True)
    checkpoint_path = args.checkpoint.expanduser()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    valid_fields = {field.name for field in fields(TrainConfig)}
    config = TrainConfig(
        **{key: value for key, value in checkpoint["train_config"].items() if key in valid_fields}
    )
    validate_policy_config(config)
    seed_everything(args.seed, config.torch_deterministic)

    envs = make_eval_envs(
        env_id=config.env_id,
        num_envs=1,
        sim_backend=config.sim_backend,
        control_mode=config.control_mode,
        obs_mode=config.obs_mode,
        obs_horizon=config.obs_horizon,
        max_episode_steps=config.max_episode_steps,
    )
    try:
        source_space, include_rgb, include_depth = get_unwrapped_observation_metadata(
            config.env_id,
            config.control_mode,
            config.obs_mode,
            config.max_episode_steps,
        )
        dataset = DiffusionDemoDataset(
            data_path=config.demo_path,
            observation_space=source_space,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            control_mode=config.control_mode,
            include_rgb=include_rgb,
            include_depth=include_depth,
            device=torch.device("cpu"),
            num_trajectories=config.num_demos,
        )
        sample_count = min(args.num_samples, len(dataset))
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
        batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=sample_count)))
        batch = move_batch_to_device(batch, device)

        policy = DiffusionPolicy(envs, config).to(device)
        policy.load_state_dict(checkpoint["ema_policy"])
        policy.eval()

        observations = batch["observations"]
        shuffled_observations = {key: value.clone() for key, value in observations.items()}
        permutation = torch.randperm(sample_count, generator=generator).to(device)
        shuffled_observations["rgb"] = observations["rgb"][permutation]
        if "depth" in observations:
            shuffled_observations["depth"] = observations["depth"][permutation]

        with torch.no_grad():
            condition = policy.encode_observations(observations, channel_last=False)
            shuffled_condition = policy.encode_observations(
                shuffled_observations, channel_last=False
            )

            torch.manual_seed(args.seed)
            original_loss = policy.compute_loss(observations, batch["actions"])
            torch.manual_seed(args.seed)
            shuffled_loss = policy.compute_loss(shuffled_observations, batch["actions"])

            online_observations = {
                key: (value.permute(0, 1, 3, 4, 2) if key in {"rgb", "depth"} else value)
                for key, value in observations.items()
            }
            shuffled_online_observations = {
                key: (value.permute(0, 1, 3, 4, 2) if key in {"rgb", "depth"} else value)
                for key, value in shuffled_observations.items()
            }
            torch.manual_seed(args.seed)
            predicted = policy.get_action(online_observations)
            torch.manual_seed(args.seed)
            shuffled_predicted = policy.get_action(shuffled_online_observations)

        start = config.obs_horizon - 1
        target = batch["actions"][:, start : start + config.act_horizon]
        flat_predicted = predicted.flatten(1)
        flat_target = target.flatten(1)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "iteration": checkpoint.get("iteration"),
            "num_samples": sample_count,
            "act_horizon": config.act_horizon,
            "pool_visual_feature_map": config.pool_visual_feature_map,
            "diffusion_loss": float(original_loss),
            "diffusion_loss_with_shuffled_rgb": float(shuffled_loss),
            "condition_mean_absolute_change_after_rgb_shuffle": float(
                (condition - shuffled_condition).abs().mean()
            ),
            "action_mean_absolute_change_after_rgb_shuffle": float(
                (predicted - shuffled_predicted).abs().mean()
            ),
            "sampled_action_mse_to_demo": float(functional.mse_loss(predicted, target)),
            "sampled_action_cosine_to_demo": float(
                functional.cosine_similarity(flat_predicted, flat_target).mean()
            ),
            "sampled_action_mean_absolute_value": float(predicted.abs().mean()),
            "demo_action_mean_absolute_value": float(target.abs().mean()),
            "sampled_action_saturation_fraction": float((predicted.abs() > 0.95).float().mean()),
        }
        print(json.dumps(json_ready(metrics), indent=2))
    finally:
        envs.close()


if __name__ == "__main__":
    main()
