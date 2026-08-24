#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch import optim
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ripl.config import TrainConfig, parse_config, validate_policy_config
from ripl.data import (
    DiffusionDemoDataset,
    IterationBasedBatchSampler,
    move_batch_to_device,
    worker_init_fn,
)
from ripl.envs import get_unwrapped_observation_metadata, make_eval_envs
from ripl.evaluation import evaluate_policy
from ripl.policy import DiffusionPolicy
from ripl.runtime import (
    ensure_supported_python,
    json_ready,
    seed_everything,
    select_device,
    validate_demo_metadata,
)


def main() -> None:
    ensure_supported_python()
    config = parse_config(TrainConfig)
    validate_policy_config(config)
    demo_path = validate_demo_metadata(config.demo_path, config.control_mode)
    seed_everything(config.seed, config.torch_deterministic)
    device = select_device(config.cuda)
    dataset_device = device if config.dataset_device == "cuda" else torch.device("cpu")
    if dataset_device.type == "cuda" and config.num_dataloader_workers:
        raise ValueError("Use num_dataloader_workers=0 when storing the dataset on CUDA")

    resume_checkpoint = None
    start_iteration = 0
    if config.resume:
        resume_path = Path(config.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        start_iteration = int(resume_checkpoint["iteration"])
        run_dir = resume_path.parent.parent
        run_name = run_dir.name
        print(f"Resuming {run_name} from iteration {start_iteration}")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.exp_name}__seed{config.seed}__{timestamp}"
        run_dir = Path(config.output_dir).expanduser() / run_name
    if start_iteration >= config.total_iters:
        raise ValueError(
            f"Checkpoint is already at iteration {start_iteration}; total_iters must be larger"
        )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(asdict(config)), handle, indent=2)

    envs = make_eval_envs(
        env_id=config.env_id,
        num_envs=config.num_eval_envs,
        sim_backend=config.sim_backend,
        control_mode=config.control_mode,
        obs_mode=config.obs_mode,
        obs_horizon=config.obs_horizon,
        max_episode_steps=config.max_episode_steps,
        video_dir=run_dir / "videos" if config.capture_video else None,
    )
    source_space, include_rgb, include_depth = get_unwrapped_observation_metadata(
        config.env_id,
        config.control_mode,
        config.obs_mode,
        config.max_episode_steps,
    )
    dataset = DiffusionDemoDataset(
        data_path=demo_path,
        observation_space=source_space,
        obs_horizon=config.obs_horizon,
        pred_horizon=config.pred_horizon,
        control_mode=config.control_mode,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=dataset_device,
        num_trajectories=config.num_demos,
    )
    if len(dataset) < config.batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} windows, fewer than batch_size={config.batch_size}"
        )
    sampler = RandomSampler(dataset, replacement=False)
    finite_batches = BatchSampler(sampler, batch_size=config.batch_size, drop_last=True)
    batches = IterationBasedBatchSampler(finite_batches, config.total_iters - start_iteration)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=config.num_dataloader_workers,
        worker_init_fn=partial(worker_init_fn, base_seed=config.seed),
        persistent_workers=config.num_dataloader_workers > 0,
        pin_memory=device.type == "cuda" and dataset_device.type == "cpu",
    )

    policy = DiffusionPolicy(envs, config).to(device)
    ema_policy = DiffusionPolicy(envs, config).to(device)
    optimizer = optim.AdamW(
        policy.parameters(), lr=config.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_iters,
    )
    ema = EMAModel(policy.parameters(), power=0.75)
    if resume_checkpoint is not None:
        policy.load_state_dict(resume_checkpoint["policy"])
        ema_policy.load_state_dict(resume_checkpoint["ema_policy"])
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "lr_scheduler" in resume_checkpoint:
            lr_scheduler.load_state_dict(resume_checkpoint["lr_scheduler"])
        if "ema" in resume_checkpoint:
            ema.load_state_dict(resume_checkpoint["ema"])
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in asdict(config).items()),
    )

    if config.track:
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "Install tracking support with `pip install -e '.[tracking]'`"
            ) from error
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=asdict(config),
            name=run_name,
            group="DiffusionPolicy",
            tags=["diffusion_policy", "visual", config.env_id],
        )

    def save_checkpoint(tag: str, iteration: int, metrics: dict[str, float]) -> Path:
        ema.copy_to(ema_policy.parameters())
        path = checkpoint_dir / f"{tag}.pt"
        torch.save(
            {
                "policy": policy.state_dict(),
                "ema_policy": ema_policy.state_dict(),
                "train_config": json_ready(asdict(config)),
                "iteration": iteration,
                "metrics": metrics,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "ema": ema.state_dict(),
            },
            path,
        )
        print(f"Saved checkpoint: {path}")
        return path

    best_metrics: dict[str, float] = defaultdict(lambda: -1.0)
    latest_metrics: dict[str, float] = {}
    if resume_checkpoint is not None:
        latest_metrics = dict(resume_checkpoint.get("metrics", {}))
        best_metrics.update(latest_metrics)
    progress = tqdm(
        total=config.total_iters,
        initial=start_iteration,
        desc="training",
    )
    policy.train()
    try:
        for iteration, batch in enumerate(dataloader, start=start_iteration + 1):
            batch = move_batch_to_device(batch, device)
            loss = policy.compute_loss(batch["observations"], batch["actions"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(policy.parameters())

            if iteration % config.log_freq == 0 or iteration == 1:
                writer.add_scalar("loss/train", loss.item(), iteration)
                writer.add_scalar("charts/learning_rate", lr_scheduler.get_last_lr()[0], iteration)
            if iteration % config.eval_freq == 0 or iteration == config.total_iters:
                ema.copy_to(ema_policy.parameters())
                raw_metrics = evaluate_policy(
                    config.num_eval_episodes,
                    ema_policy,
                    envs,
                    device,
                    config.sim_backend,
                )
                latest_metrics = {key: float(np.mean(value)) for key, value in raw_metrics.items()}
                print(f"iteration={iteration} evaluation={latest_metrics}")
                for key, value in latest_metrics.items():
                    writer.add_scalar(f"eval/{key}", value, iteration)
                for key in ("success_once", "success_at_end"):
                    if key in latest_metrics and latest_metrics[key] > best_metrics[key]:
                        best_metrics[key] = latest_metrics[key]
                        save_checkpoint(f"best_{key}", iteration, latest_metrics)
                policy.train()
            if config.save_freq and iteration % config.save_freq == 0:
                save_checkpoint(f"step_{iteration}", iteration, latest_metrics)

            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    finally:
        progress.close()
        save_checkpoint("final", min(progress.n, config.total_iters), latest_metrics)
        envs.close()
        writer.close()


if __name__ == "__main__":
    main()
