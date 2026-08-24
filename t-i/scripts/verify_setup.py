from __future__ import annotations

import sys

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch


def main() -> None:
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError("Use Python 3.10-3.12")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a GPU runtime")
    env = gym.make(
        "PushT-v1",
        num_envs=1,
        sim_backend="physx_cuda",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
    )
    try:
        observation, _ = env.reset(seed=0)
        cameras = observation["sensor_data"]
        shapes = {name: tuple(data["rgb"].shape) for name, data in cameras.items()}
        print(f"ManiSkill PushT RGB setup is ready; camera tensors: {shapes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
