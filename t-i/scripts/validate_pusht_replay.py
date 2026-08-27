#!/usr/bin/env python3
"""Validate native and replayed Push-T demonstration datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


def _load_metadata(trajectory_path: Path) -> dict[str, Any]:
    metadata_path = trajectory_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing trajectory metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("env_info", {}).get("env_id") != "PushT-v1":
        raise ValueError("Expected PushT-v1 trajectory metadata")
    return metadata


def _control_mode(metadata: dict[str, Any]) -> str | None:
    env_kwargs = metadata.get("env_info", {}).get("env_kwargs", {})
    if env_kwargs.get("control_mode") is not None:
        return str(env_kwargs["control_mode"])
    episodes = metadata.get("episodes", [])
    return str(episodes[0].get("control_mode")) if episodes else None


def validate_source(
    trajectory_path: Path,
    count: int,
    control_mode: str,
    replay_envs: int,
    allow_replay_env_mismatch: bool = False,
) -> int:
    """Validate selected successful source episodes and replay parallelism."""
    if not trajectory_path.is_file():
        raise ValueError(f"Missing source trajectory: {trajectory_path}")
    metadata = _load_metadata(trajectory_path)
    actual_control_mode = _control_mode(metadata)
    if actual_control_mode != control_mode:
        raise ValueError(f"Source controller is {actual_control_mode!r}; expected {control_mode!r}")

    collection_envs = metadata.get("env_info", {}).get("env_kwargs", {}).get("num_envs")
    if collection_envs is None:
        raise ValueError("Source metadata does not record collection num_envs")
    collection_envs = int(collection_envs)
    if replay_envs != collection_envs and not allow_replay_env_mismatch:
        raise ValueError(
            f"Replay requested {replay_envs} environments, but the source was collected with "
            f"{collection_envs}. Set REPLAY_ENVS={collection_envs}, or explicitly set "
            "ALLOW_REPLAY_ENV_MISMATCH=1 for a lower-fidelity smoke run."
        )

    episodes = metadata.get("episodes", [])[:count]
    if len(episodes) != count:
        raise ValueError(f"Requested {count} source episodes, but found {len(episodes)}")

    invalid: list[int] = []
    with h5py.File(trajectory_path, "r") as handle:
        for episode in episodes:
            episode_id = int(episode["episode_id"])
            key = f"traj_{episode_id}"
            if key not in handle:
                invalid.append(episode_id)
                continue
            group = handle[key]
            if "success" not in group or not bool(group["success"][-1]):
                invalid.append(episode_id)
    if invalid:
        raise ValueError(f"Selected source episodes are not successful at the end: {invalid}")
    return collection_envs


def _rgb_dataset(group: h5py.Group) -> h5py.Dataset:
    try:
        cameras = group["obs"]["sensor_data"]
    except KeyError as error:
        raise ValueError(f"{group.name} does not contain visual observations") from error
    for camera in cameras.values():
        if "rgb" in camera:
            return camera["rgb"]
    raise ValueError(f"{group.name} does not contain an RGB camera stream")


def validate_output(
    trajectory_path: Path,
    count: int,
    control_mode: str,
) -> None:
    """Validate exact count, controller, RGB streams, and sequence lengths."""
    if not trajectory_path.is_file():
        raise ValueError(f"Missing replayed trajectory: {trajectory_path}")
    metadata = _load_metadata(trajectory_path)
    actual_control_mode = _control_mode(metadata)
    if actual_control_mode != control_mode:
        raise ValueError(
            f"Replayed controller is {actual_control_mode!r}; expected {control_mode!r}"
        )
    obs_mode = metadata.get("env_info", {}).get("env_kwargs", {}).get("obs_mode")
    if obs_mode != "rgb":
        raise ValueError(f"Replayed observation mode is {obs_mode!r}; expected 'rgb'")
    episodes = metadata.get("episodes", [])
    if len(episodes) != count:
        raise ValueError(f"Replayed metadata has {len(episodes)} episodes; expected {count}")

    with h5py.File(trajectory_path, "r") as handle:
        trajectory_keys = [key for key in handle if key.startswith("traj_")]
        if len(trajectory_keys) != count:
            raise ValueError(
                f"Replayed HDF5 has {len(trajectory_keys)} trajectories; expected {count}"
            )
        for key in trajectory_keys:
            group = handle[key]
            action_count = len(group["actions"])
            rgb_count = len(_rgb_dataset(group))
            if rgb_count != action_count + 1:
                raise ValueError(
                    f"{key} has {rgb_count} RGB frames for {action_count} actions; "
                    "expected actions + 1"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source")
    source.add_argument("--trajectory", type=Path, required=True)
    source.add_argument("--count", type=int, required=True)
    source.add_argument("--control-mode", required=True)
    source.add_argument("--replay-envs", type=int, required=True)
    source.add_argument("--allow-replay-env-mismatch", action="store_true")

    output = subparsers.add_parser("output")
    output.add_argument("--trajectory", type=Path, required=True)
    output.add_argument("--count", type=int, required=True)
    output.add_argument("--control-mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "source":
        collection_envs = validate_source(
            args.trajectory,
            args.count,
            args.control_mode,
            args.replay_envs,
            args.allow_replay_env_mismatch,
        )
        print(
            f"Verified {args.count} successful {args.control_mode} source trajectories; "
            f"collection environments: {collection_envs}; replay environments: "
            f"{args.replay_envs}"
        )
    else:
        validate_output(args.trajectory, args.count, args.control_mode)
        print(
            f"Verified {args.count} RGB {args.control_mode} trajectories with aligned "
            "observation/action lengths"
        )


if __name__ == "__main__":
    main()
