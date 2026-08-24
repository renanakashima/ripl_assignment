from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - importing registers ManiSkill environments
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def make_eval_envs(
    env_id: str,
    num_envs: int,
    sim_backend: str,
    control_mode: str,
    obs_mode: str,
    obs_horizon: int,
    max_episode_steps: int,
    video_dir: str | Path | None = None,
):
    """Create the visual, frame-stacked vector environment used for evaluation."""
    env_kwargs = {
        "control_mode": control_mode,
        "reward_mode": "sparse",
        "obs_mode": obs_mode,
        "render_mode": "rgb_array",
        "human_render_camera_configs": {"shader_pack": "default"},
        "max_episode_steps": max_episode_steps,
    }
    video_dir = str(video_dir) if video_dir is not None else None

    if sim_backend == "physx_cpu":

        def make_one(seed: int):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                env = FlattenRGBDObservationWrapper(env)
                env = FrameStack(env, num_stack=obs_horizon)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir and seed == 0:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="visual Diffusion Policy evaluation",
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
        env_id,
        num_envs=num_envs,
        sim_backend=sim_backend,
        reconfiguration_freq=1,
        **env_kwargs,
    )
    episode_steps = gym_utils.find_max_episode_steps_value(env)
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=obs_horizon)
    if video_dir:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            source_type="diffusion_policy",
            source_desc="visual Diffusion Policy evaluation",
            max_steps_per_video=episode_steps,
        )
    return ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)


def get_unwrapped_observation_metadata(
    env_id: str,
    control_mode: str,
    obs_mode: str,
    max_episode_steps: int,
):
    """Return source observation space and enabled visual modalities for demo conversion."""
    env = gym.make(
        env_id,
        control_mode=control_mode,
        reward_mode="sparse",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    try:
        observation_space = env.observation_space
        include_rgb = bool(env.unwrapped.obs_mode_struct.visual.rgb)
        include_depth = bool(env.unwrapped.obs_mode_struct.visual.depth)
        return observation_space, include_rgb, include_depth
    finally:
        env.close()
