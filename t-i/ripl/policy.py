from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from torch.nn import functional

from ripl.config import PolicyConfig
from ripl.networks import ConditionalUnet1D, PlainConv


class DiffusionPolicy(nn.Module):
    """RGB-conditioned action diffusion model for ManiSkill vector environments."""

    def __init__(self, env: Any, config: PolicyConfig):
        super().__init__()
        self.obs_horizon = config.obs_horizon
        self.act_horizon = config.act_horizon
        self.pred_horizon = config.pred_horizon
        self.diffusion_steps = config.diffusion_steps

        observation_space = env.single_observation_space
        action_space = env.single_action_space
        if len(observation_space["state"].shape) != 2:
            raise ValueError("Expected frame-stacked state shape (obs_horizon, state_dim)")
        if len(action_space.shape) != 1:
            raise ValueError("Expected a flat action space")
        if not (np.all(action_space.low == -1) and np.all(action_space.high == 1)):
            raise ValueError("DDPM clipping requires actions normalized to [-1, 1]")

        self.act_dim = action_space.shape[0]
        state_dim = observation_space["state"].shape[1]
        self.include_rgb = "rgb" in observation_space.spaces
        self.include_depth = "depth" in observation_space.spaces
        if not self.include_rgb:
            raise ValueError("Visual Diffusion Policy requires RGB observations")

        rgb_shape = observation_space["rgb"].shape
        visual_channels = rgb_shape[-1]
        if self.include_depth:
            visual_channels += observation_space["depth"].shape[-1]
        self.visual_encoder = PlainConv(
            visual_channels,
            config.visual_feature_dim,
            image_size=(rgb_shape[-3], rgb_shape[-2]),
            pool_feature_map=config.pool_visual_feature_map,
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=config.obs_horizon * (config.visual_feature_dim + state_dim),
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def encode_observations(
        self, observations: dict[str, torch.Tensor], channel_last: bool
    ) -> torch.Tensor:
        rgb = observations["rgb"]
        if channel_last:
            rgb = rgb.permute(0, 1, 4, 2, 3)
        image_sequence = rgb.float() / 255.0

        if self.include_depth:
            depth = observations["depth"]
            if channel_last:
                depth = depth.permute(0, 1, 4, 2, 3)
            image_sequence = torch.cat([image_sequence, depth.float() / 1024.0], dim=2)

        batch_size = image_sequence.shape[0]
        visual_features = self.visual_encoder(image_sequence.flatten(end_dim=1))
        visual_features = visual_features.reshape(batch_size, self.obs_horizon, -1)
        features = torch.cat([visual_features, observations["state"].float()], dim=-1)
        return features.flatten(start_dim=1)

    def compute_loss(
        self, observations: dict[str, torch.Tensor], action_sequence: torch.Tensor
    ) -> torch.Tensor:
        condition = self.encode_observations(observations, channel_last=False)
        noise = torch.randn_like(action_sequence)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (action_sequence.shape[0],),
            device=action_sequence.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(action_sequence, noise, timesteps)
        predicted_noise = self.noise_pred_net(noisy_actions, timesteps, condition)
        return functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def get_action(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        condition = self.encode_observations(observations, channel_last=True)
        noisy_actions = torch.randn(
            (observations["state"].shape[0], self.pred_horizon, self.act_dim),
            device=observations["state"].device,
        )
        self.noise_scheduler.set_timesteps(self.diffusion_steps)
        for timestep in self.noise_scheduler.timesteps:
            predicted_noise = self.noise_pred_net(noisy_actions, timestep, condition)
            noisy_actions = self.noise_scheduler.step(
                model_output=predicted_noise,
                timestep=timestep,
                sample=noisy_actions,
            ).prev_sample

        start = self.obs_horizon - 1
        return noisy_actions[:, start : start + self.act_horizon]
