"""Neural-network building blocks adapted from ManiSkill's Diffusion Policy baseline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("The diffusion embedding dimension must be even and >= 4")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10_000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale
        )
        embedding = timesteps[:, None].float() * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.blocks[0](inputs)
        scale_bias = self.cond_encoder(condition).reshape(
            condition.shape[0], 2, self.out_channels, 1
        )
        output = scale_bias[:, 0] * output + scale_bias[:, 1]
        output = self.blocks[1](output)
        return output + self.residual_conv(inputs)


class ConditionalUnet1D(nn.Module):
    """1-D U-Net that predicts action noise with FiLM observation conditioning."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 64,
        down_dims: Sequence[int] = (64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        in_out = list(pairwise(all_dims))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        reversed_levels = list(reversed(in_out[1:]))
        for dim_in, dim_out in reversed_levels:
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        Upsample1d(dim_in),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=inputs.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(inputs.device)
        timesteps = timesteps.expand(inputs.shape[0])

        condition = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            condition = torch.cat([condition, global_cond], dim=-1)

        skips: list[torch.Tensor] = []
        for first, second, downsample in self.down_modules:
            inputs = second(first(inputs, condition), condition)
            skips.append(inputs)
            inputs = downsample(inputs)

        for middle in self.mid_modules:
            inputs = middle(inputs, condition)

        for first, second, upsample in self.up_modules:
            inputs = torch.cat((inputs, skips.pop()), dim=1)
            inputs = upsample(second(first(inputs, condition), condition))

        return self.final_conv(inputs).moveaxis(-1, -2)


def make_mlp(
    in_channels: int,
    mlp_channels: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
    last_activation: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_channels
    for index, output in enumerate(mlp_channels):
        layers.append(nn.Linear(current, output))
        if last_activation or index < len(mlp_channels) - 1:
            layers.append(activation())
        current = output
    return nn.Sequential(*layers)


class PlainConv(nn.Module):
    """Small camera encoder with optional spatial feature-map pooling."""

    def __init__(
        self,
        in_channels: int,
        out_dim: int = 256,
        image_size: tuple[int, int] = (128, 128),
        pool_feature_map: bool = True,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
        )
        self.pool_feature_map = pool_feature_map
        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            flattened_dim = 128
        else:
            self.pool = nn.Identity()
            height, width = image_size
            flattened_dim = 128 * (height // 16) * (width // 16)
        self.fc = make_mlp(flattened_dim, [out_dim])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.cnn(image)).flatten(1))
