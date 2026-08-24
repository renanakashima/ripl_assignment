"""ManiSkill demonstration loading and Diffusion Policy sequence sampling."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

TARGET_TO_H5_KEY = {
    "observations": "obs",
    "actions": "actions",
}


def _load_h5_node(node: h5py.File | h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, (h5py.File, h5py.Group)):
        return {key: _load_h5_node(node[key]) for key in node}
    if isinstance(node, h5py.Dataset):
        return node[()]
    raise TypeError(f"Unsupported HDF5 node: {type(node)!r}")


def load_demo_dataset(
    path: str | Path, num_trajectories: int | None = None
) -> dict[str, list[Any]]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Demonstration file not found: {path}\n"
            "Run scripts/prepare_pusht_demos.sh in Colab first."
        )
    with h5py.File(path, "r") as handle:
        trajectory_keys = sorted(handle.keys(), key=lambda key: int(key.rsplit("_", 1)[-1]))
        if num_trajectories is not None:
            if num_trajectories > len(trajectory_keys):
                raise ValueError(
                    f"Requested {num_trajectories} demos, but {path} has {len(trajectory_keys)}"
                )
            trajectory_keys = trajectory_keys[:num_trajectories]
        trajectories = {key: _load_h5_node(handle[key]) for key in trajectory_keys}

    first = trajectories[trajectory_keys[0]]
    for source_key in TARGET_TO_H5_KEY.values():
        if source_key not in first:
            raise KeyError(f"Expected '{source_key}' in {trajectory_keys[0]}; found {first.keys()}")
    return {
        target_key: [trajectories[key][source_key] for key in trajectory_keys]
        for target_key, source_key in TARGET_TO_H5_KEY.items()
    }


def reorder_like(data: dict[str, Any], reference: Any) -> dict[str, Any]:
    """Recursively match Gym Dict-space key order before flattening observations."""
    output = {}
    for key, value in reference.items():
        output[key] = reorder_like(data[key], value) if hasattr(value, "items") else data[key]
    return output


def build_state_obs_extractor() -> Callable[[dict[str, Any]], list[Any]]:
    return lambda observation: [
        *observation["agent"].values(),
        *observation["extra"].values(),
    ]


def convert_visual_observation(
    observation: dict[str, Any],
    include_depth: bool,
    state_obs_extractor: Callable[[dict[str, Any]], list[Any]],
) -> dict[str, np.ndarray]:
    """Convert a full trajectory from ManiSkill's nested format to channel-first arrays."""
    camera_data = observation["sensor_data"]
    modalities = ["rgb", "depth"] if include_depth else ["rgb"]
    images = {}
    for modality in modalities:
        joined = np.concatenate([camera[modality] for camera in camera_data.values()], axis=-1)
        images[modality] = np.transpose(joined, axes=(0, 3, 1, 2))

    state_parts = state_obs_extractor(observation)
    state_parts = [
        part.astype(np.float32) if part.dtype == np.float64 else part for part in state_parts
    ]
    state = np.column_stack(state_parts).astype(np.float32, copy=False)
    return {"state": state, **images}


class DiffusionDemoDataset(Dataset):
    """Sequence-window dataset with the padding convention from the ManiSkill baseline."""

    def __init__(
        self,
        data_path: str | Path,
        observation_space: Any,
        obs_horizon: int,
        pred_horizon: int,
        control_mode: str,
        include_rgb: bool,
        include_depth: bool,
        device: torch.device,
        num_trajectories: int | None,
    ):
        if not include_rgb:
            raise ValueError("This project implements a visual RGB policy; RGB must be enabled")
        if "delta_pos" not in control_mode and "delta_pose" not in control_mode:
            raise NotImplementedError(
                f"Padding for control mode {control_mode!r} is not implemented; use a delta controller"
            )

        raw = load_demo_dataset(data_path, num_trajectories)
        state_extractor = build_state_obs_extractor()
        observations = []
        for trajectory in raw["observations"]:
            ordered = reorder_like(trajectory, observation_space)
            converted = convert_visual_observation(ordered, include_depth, state_extractor)
            tensor_trajectory = {
                "state": torch.from_numpy(converted["state"]).to(device=device),
                "rgb": torch.from_numpy(converted["rgb"]).to(device=device),
            }
            if include_depth:
                tensor_trajectory["depth"] = torch.from_numpy(
                    converted["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            observations.append(tensor_trajectory)

        actions = [
            torch.as_tensor(action, dtype=torch.float32, device=device) for action in raw["actions"]
        ]
        self.trajectories = {"observations": observations, "actions": actions}
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.pad_action_arm = torch.zeros(actions[0].shape[1] - 1, device=device)
        self.slices: list[tuple[int, int, int]] = []

        total_transitions = 0
        for trajectory_index, action_trajectory in enumerate(actions):
            length = action_trajectory.shape[0]
            observation_length = observations[trajectory_index]["state"].shape[0]
            if observation_length != length + 1:
                raise ValueError(
                    f"Trajectory {trajectory_index}: {observation_length} observations for "
                    f"{length} actions; expected actions + 1"
                )
            total_transitions += length
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices.extend(
                (trajectory_index, start, start + pred_horizon)
                for start in range(-pad_before, length - pred_horizon + pad_after)
            )

        print(
            f"Loaded {len(actions)} trajectories, {total_transitions} transitions, "
            f"and {len(self.slices)} training windows on {device}."
        )

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_index, start, end = self.slices[index]
        actions = self.trajectories["actions"][trajectory_index]
        observations = self.trajectories["observations"][trajectory_index]
        length = actions.shape[0]

        observation_sequence = {}
        for key, values in observations.items():
            sequence = values[max(0, start) : start + self.obs_horizon]
            if start < 0:
                sequence = torch.cat(
                    [sequence[0].repeat(-start, *([1] * (sequence.ndim - 1))), sequence]
                )
            observation_sequence[key] = sequence

        action_sequence = actions[max(0, start) : end]
        if start < 0:
            action_sequence = torch.cat([action_sequence[0].repeat(-start, 1), action_sequence])
        if end > length:
            padded_action = torch.cat([self.pad_action_arm, action_sequence[-1, -1, None]])
            action_sequence = torch.cat([action_sequence, padded_action.repeat(end - length, 1)])

        return {"observations": observation_sequence, "actions": action_sequence}


class IterationBasedBatchSampler(Sampler[list[int]]):
    """Repeat a finite batch sampler until exactly `num_iterations` batches are emitted."""

    def __init__(self, batch_sampler: Any, num_iterations: int):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations

    def __iter__(self) -> Iterator[list[int]]:
        iteration = 0
        while iteration < self.num_iterations:
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self) -> int:
        return self.num_iterations


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    np.random.seed(base_seed + worker_id)


def move_batch_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    return value
