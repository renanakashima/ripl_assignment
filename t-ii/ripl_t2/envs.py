from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

TARGETED_PUSHT_ENV_ID = "RIPL-PushTTargeted-v1"


@dataclass(frozen=True)
class PushTPoseRange:
    """Axis-aligned subset of PushT-v1's nominal initial-pose distribution."""

    x_rel_min: float = -0.10
    x_rel_max: float = 0.10
    y_rel_min: float = -0.10
    y_rel_max: float = 0.20
    theta_deg_min: float = 0.0
    theta_deg_max: float = 360.0

    def __post_init__(self) -> None:
        if not -0.10 <= self.x_rel_min <= self.x_rel_max <= 0.10:
            raise ValueError("x-relative range must lie within PushT's nominal [-0.10, 0.10] m")
        if not -0.10 <= self.y_rel_min <= self.y_rel_max <= 0.20:
            raise ValueError("y-relative range must lie within PushT's nominal [-0.10, 0.20] m")
        if not all(
            torch.isfinite(torch.tensor(value)).item()
            for value in (self.theta_deg_min, self.theta_deg_max)
        ):
            raise ValueError("theta bounds must be finite")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> PushTPoseRange:
        return cls() if values is None else cls(**dict(values))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def theta_span_deg(self) -> float:
        raw_span = self.theta_deg_max - self.theta_deg_min
        if isclose(raw_span, 0.0, abs_tol=1e-12):
            return 0.0
        if abs(raw_span) >= 360.0:
            return 360.0
        return raw_span % 360.0


@register_env(TARGETED_PUSHT_ENV_ID, max_episode_steps=100)
class TargetedPushTEnv(PushTEnv):
    """PushT-v1 with a reproducible, pose-conditioned initial-state sampler."""

    def __init__(self, *args, pose_range: Mapping[str, Any] | None = None, **kwargs):
        self.ripl_pose_range = PushTPoseRange.from_mapping(pose_range)
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            batch_size = len(env_idx)
            pose_range = self.ripl_pose_range

            x_rel = pose_range.x_rel_min + torch.rand(batch_size) * (
                pose_range.x_rel_max - pose_range.x_rel_min
            )
            y_rel = pose_range.y_rel_min + torch.rand(batch_size) * (
                pose_range.y_rel_max - pose_range.y_rel_min
            )
            theta_deg = torch.full((batch_size,), pose_range.theta_deg_min)
            if pose_range.theta_span_deg:
                theta_deg += torch.rand(batch_size) * pose_range.theta_span_deg
            theta = torch.deg2rad(theta_deg.remainder(360.0))

            position = torch.zeros((batch_size, 3))
            position[:, 0] = self.goal_offset[0] + x_rel
            position[:, 1] = self.goal_offset[1] + y_rel
            position[:, 2] = 0.04 / 2 + 1e-3

            quaternion = torch.zeros((batch_size, 4))
            quaternion[:, 0] = (theta / 2).cos()
            quaternion[:, -1] = (theta / 2).sin()
            self.tee.set_pose(Pose.create_from_pq(p=position, q=quaternion))


class PushTDiagnosticsWrapper(gym.Wrapper):
    """Expose Push-T poses and overlap before vector-environment autoreset."""

    def _diagnostics(self) -> dict[str, Any]:
        base_env = self.unwrapped
        return {
            "tee_pose": base_env.tee.pose.raw_pose.clone(),
            "goal_tee_pose": base_env.goal_tee.pose.raw_pose.clone(),
            "tcp_pose": base_env.agent.tcp.pose.raw_pose.clone(),
            "overlap": base_env.pseudo_render_intersection().clone(),
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        diagnostics = self._diagnostics()
        info.update(diagnostics)
        info["initial_tee_pose"] = diagnostics["tee_pose"].clone()
        info["initial_tcp_pose"] = diagnostics["tcp_pose"].clone()
        info["initial_overlap"] = diagnostics["overlap"].clone()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(self._diagnostics())
        return observation, reward, terminated, truncated, info


def make_failure_envs(
    env_id: str,
    num_envs: int,
    sim_backend: str,
    control_mode: str,
    obs_mode: str,
    obs_horizon: int,
    max_episode_steps: int,
    pusht_pose_range: dict[str, float],
    video_dir: str | Path | None = None,
):
    """Create pose-conditioned visual Push-T environments with diagnostic metadata."""
    if env_id != "PushT-v1":
        raise ValueError("Failure evaluation is supported only for PushT-v1")
    pose_range = PushTPoseRange.from_mapping(pusht_pose_range)
    env_kwargs = {
        "control_mode": control_mode,
        "reward_mode": "sparse",
        "obs_mode": obs_mode,
        "render_mode": "rgb_array",
        "human_render_camera_configs": {"shader_pack": "default"},
        "max_episode_steps": max_episode_steps,
        "pose_range": pose_range.to_dict(),
    }
    video_dir = str(video_dir) if video_dir is not None else None

    if sim_backend == "physx_cpu":

        def make_one(seed: int):
            def thunk():
                env = gym.make(TARGETED_PUSHT_ENV_ID, reconfiguration_freq=1, **env_kwargs)
                env = FlattenRGBDObservationWrapper(env)
                env = FrameStack(env, num_stack=obs_horizon)
                env = PushTDiagnosticsWrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir and seed == 0:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="T-II Push-T failure evaluation",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_class = gym.vector.SyncVectorEnv
        if num_envs > 1:
            vector_class = lambda constructors: gym.vector.AsyncVectorEnv(
                constructors, context="forkserver"
            )
        return vector_class([make_one(seed) for seed in range(num_envs)])

    env = gym.make(
        TARGETED_PUSHT_ENV_ID,
        num_envs=num_envs,
        sim_backend=sim_backend,
        reconfiguration_freq=1,
        **env_kwargs,
    )
    episode_steps = gym_utils.find_max_episode_steps_value(env)
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=obs_horizon)
    env = PushTDiagnosticsWrapper(env)
    if video_dir:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            source_type="diffusion_policy",
            source_desc="T-II Push-T failure evaluation",
            max_steps_per_video=episode_steps,
        )
    return ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
