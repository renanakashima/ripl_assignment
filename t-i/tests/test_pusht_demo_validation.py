import json

import h5py
import numpy as np
import pytest

from scripts.validate_pusht_replay import validate_output, validate_source


def _write_metadata(path, *, control_mode, num_envs, obs_mode, count):
    metadata = {
        "env_info": {
            "env_id": "PushT-v1",
            "env_kwargs": {
                "control_mode": control_mode,
                "num_envs": num_envs,
                "obs_mode": obs_mode,
            },
        },
        "episodes": [
            {
                "episode_id": index,
                "control_mode": control_mode,
                "success": True,
            }
            for index in range(count)
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_source(path, *, count=2, control_mode="pd_ee_delta_pose", num_envs=1024):
    with h5py.File(path, "w") as handle:
        for index in range(count):
            trajectory = handle.create_group(f"traj_{index}")
            trajectory.create_dataset("success", data=np.array([False, True]))
    _write_metadata(
        path,
        control_mode=control_mode,
        num_envs=num_envs,
        obs_mode="state",
        count=count,
    )


def _write_output(path, *, count=2, control_mode="pd_ee_delta_pose"):
    with h5py.File(path, "w") as handle:
        for index in range(count):
            trajectory = handle.create_group(f"traj_{index}")
            trajectory.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
            camera = trajectory.create_group("obs/sensor_data/base_camera")
            camera.create_dataset("rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
    _write_metadata(
        path,
        control_mode=control_mode,
        num_envs=1024,
        obs_mode="rgb",
        count=count,
    )


def test_validate_source_requires_collection_parallelism(tmp_path):
    path = tmp_path / "trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
    _write_source(path)

    assert validate_source(path, 2, "pd_ee_delta_pose", 1024) == 1024
    with pytest.raises(ValueError, match="source was collected with 1024"):
        validate_source(path, 2, "pd_ee_delta_pose", 64)
    assert validate_source(path, 2, "pd_ee_delta_pose", 64, True) == 1024


def test_validate_source_rejects_failed_selected_episode(tmp_path):
    path = tmp_path / "trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
    _write_source(path)
    with h5py.File(path, "r+") as handle:
        handle["traj_1/success"][-1] = False

    with pytest.raises(ValueError, match="not successful"):
        validate_source(path, 2, "pd_ee_delta_pose", 1024)


def test_validate_output_checks_rgb_sequence_alignment(tmp_path):
    path = tmp_path / "trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
    _write_output(path)

    validate_output(path, 2, "pd_ee_delta_pose")
    with h5py.File(path, "r+") as handle:
        del handle["traj_1/obs/sensor_data/base_camera/rgb"]
        handle["traj_1/obs/sensor_data/base_camera"].create_dataset(
            "rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8)
        )

    with pytest.raises(ValueError, match=r"expected actions \+ 1"):
        validate_output(path, 2, "pd_ee_delta_pose")
