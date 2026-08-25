from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from mani_skill.utils import common
from tqdm import tqdm


def evaluate_policy(
    num_episodes: int,
    policy: Any,
    envs: Any,
    device: torch.device,
    sim_backend: str,
    progress_bar: bool = True,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate complete synchronized episodes and return per-episode metrics."""
    was_training = policy.training
    policy.eval()
    metrics: dict[str, list[np.ndarray]] = defaultdict(list)
    progress = tqdm(total=num_episodes, desc="evaluation") if progress_bar else None
    completed = 0

    with torch.no_grad():
        observations, _ = envs.reset(seed=seed)
        while completed < num_episodes:
            tensor_observations = common.to_tensor(observations, device)
            action_sequence = policy.get_action(tensor_observations)
            if sim_backend == "physx_cpu":
                action_sequence = action_sequence.cpu().numpy()

            for action_index in range(action_sequence.shape[1]):
                observations, _, _, truncated, info = envs.step(action_sequence[:, action_index])
                if truncated.any():
                    break

            if truncated.any():
                if not truncated.all():
                    raise RuntimeError("Vector episodes desynchronized during evaluation")
                final_info = info.get("final_info")
                if final_info is None:
                    # Gymnasium's CPU vector environments can return terminal metrics
                    # directly and reset on the following step. ManiSkill's GPU vector
                    # wrapper uses same-step autoreset and nests them in final_info.
                    for key, value in info["episode"].items():
                        item = value.float().cpu().numpy() if torch.is_tensor(value) else value
                        metrics[key].append(np.asarray(item))
                elif isinstance(final_info, dict):
                    for key, value in final_info["episode"].items():
                        item = value.float().cpu().numpy() if torch.is_tensor(value) else value
                        metrics[key].append(np.asarray(item))
                else:
                    for episode_info in final_info:
                        for key, value in episode_info["episode"].items():
                            metrics[key].append(np.asarray(value))
                completed += envs.num_envs
                if progress:
                    progress.update(min(envs.num_envs, num_episodes - progress.n))
                if final_info is None and completed < num_episodes:
                    observations, _ = envs.reset()

    if progress:
        progress.close()
    policy.train(was_training)
    return {
        key: np.concatenate([np.atleast_1d(v) for v in values])[:num_episodes]
        for key, values in metrics.items()
    }
