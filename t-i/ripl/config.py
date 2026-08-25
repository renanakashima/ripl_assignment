from __future__ import annotations

import argparse
import sys
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import TypeVar

import tyro
import yaml


@dataclass
class PolicyConfig:
    obs_horizon: int = 2
    act_horizon: int = 1
    pred_horizon: int = 16
    diffusion_steps: int = 100
    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    visual_feature_dim: int = 256
    # True preserves compatibility with checkpoints created before this option existed.
    # New visual Push-T experiments disable it so the encoder retains object location.
    pool_visual_feature_map: bool = True


@dataclass
class TrainConfig(PolicyConfig):
    exp_name: str = "pusht-rgb-diffusion"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "PushT-v1"
    demo_path: str = "~/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"
    num_demos: int | None = None
    control_mode: str = "pd_ee_delta_pos"
    obs_mode: str = "rgb"
    sim_backend: str = "physx_cuda"
    max_episode_steps: int = 150

    total_iters: int = 50_000
    batch_size: int = 128
    lr: float = 1e-4
    warmup_steps: int = 500
    num_dataloader_workers: int = 0
    dataset_device: str = "cpu"

    log_freq: int = 100
    eval_freq: int = 5_000
    save_freq: int | None = 5_000
    num_eval_episodes: int = 20
    num_eval_envs: int = 10
    capture_video: bool = True
    output_dir: str = "runs"
    resume: str | None = None

    track: bool = False
    wandb_project_name: str = "RIPL-ManiSkill"
    wandb_entity: str | None = None
    demo_type: str = "rl"


@dataclass
class EvalConfig:
    checkpoint: str
    num_eval_episodes: int = 20
    num_eval_envs: int = 10
    sim_backend: str | None = None
    capture_video: bool = True
    output_dir: str | None = None
    seed: int | None = None
    cuda: bool = True


T = TypeVar("T")


def _required_field_names(config_type: type) -> set[str]:
    return {
        field.name
        for field in fields(config_type)
        if field.default is MISSING and field.default_factory is MISSING
    }


def parse_config(config_type: type[T], argv: list[str] | None = None) -> T:
    """Parse an optional YAML config, then let explicit CLI flags override it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, remaining = parser.parse_known_args(argv)

    defaults = None
    if known.config is not None:
        with known.config.expanduser().open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError(f"{known.config} must contain a YAML mapping")
        valid = {field.name for field in fields(config_type)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"Unknown keys in {known.config}: {', '.join(unknown)}")
        for field in fields(config_type):
            if field.name in values and isinstance(field.default, tuple):
                values[field.name] = tuple(values[field.name])
        missing = _required_field_names(config_type) - set(values)
        if missing:
            raise ValueError(
                f"Missing required keys in {known.config}: {', '.join(sorted(missing))}"
            )
        defaults = config_type(**values)

    if defaults is None:
        return tyro.cli(config_type, args=remaining)
    return tyro.cli(config_type, default=defaults, args=remaining)


def validate_policy_config(config: PolicyConfig) -> None:
    if config.obs_horizon < 1 or config.act_horizon < 1 or config.pred_horizon < 1:
        raise ValueError("All horizons must be positive")
    if config.obs_horizon + config.act_horizon - 1 > config.pred_horizon:
        raise ValueError("obs_horizon + act_horizon - 1 must not exceed pred_horizon")
    if config.diffusion_steps < 1:
        raise ValueError("diffusion_steps must be positive")
    if any(dim % config.n_groups for dim in config.unet_dims):
        raise ValueError("Every U-Net dimension must be divisible by n_groups")
