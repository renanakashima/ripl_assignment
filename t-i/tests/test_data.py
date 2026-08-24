import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ripl.data import DiffusionDemoDataset


def test_sequence_windows_pad_beginning_and_end(tmp_path):
    path = tmp_path / "trajectory.rgb.test.h5"
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        trajectory.create_dataset(
            "actions",
            data=np.arange(15, dtype=np.float32).reshape(5, 3),
        )
        observations = trajectory.create_group("obs")
        agent = observations.create_group("agent")
        agent.create_dataset("qpos", data=np.arange(12, dtype=np.float32).reshape(6, 2))
        extra = observations.create_group("extra")
        extra.create_dataset("goal", data=np.arange(6, dtype=np.float32))
        sensor_data = observations.create_group("sensor_data")
        camera = sensor_data.create_group("cam")
        camera.create_dataset("rgb", data=np.zeros((6, 4, 4, 3), dtype=np.uint8))

    observation_space = {
        "agent": {"qpos": None},
        "extra": {"goal": None},
        "sensor_data": {"cam": {"rgb": None}},
    }
    dataset = DiffusionDemoDataset(
        data_path=path,
        observation_space=observation_space,
        obs_horizon=2,
        pred_horizon=4,
        control_mode="pd_ee_delta_pos",
        include_rgb=True,
        include_depth=False,
        device=torch.device("cpu"),
        num_trajectories=1,
    )

    assert len(dataset) == 4
    first = dataset[0]
    last = dataset[-1]
    assert first["observations"]["rgb"].shape == (2, 3, 4, 4)
    assert first["observations"]["state"].shape == (2, 3)
    assert torch.equal(first["actions"][0], first["actions"][1])
    assert last["actions"].shape == (4, 3)
    assert torch.equal(last["actions"][-1, :-1], torch.zeros(2))
    assert last["actions"][-1, -1] == last["actions"][-2, -1]
