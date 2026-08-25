from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("diffusers")

from gymnasium import spaces

from ripl.config import PolicyConfig
from ripl.policy import DiffusionPolicy


def test_policy_trains_channel_first_and_infers_channel_last():
    fake_env = SimpleNamespace(
        single_observation_space=spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, shape=(2, 5), dtype=np.float32),
                "rgb": spaces.Box(0, 255, shape=(2, 128, 128, 3), dtype=np.uint8),
            }
        ),
        single_action_space=spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
    )
    config = PolicyConfig(
        obs_horizon=2,
        act_horizon=1,
        pred_horizon=8,
        diffusion_steps=2,
        diffusion_step_embed_dim=16,
        unet_dims=(16, 32, 64),
        n_groups=8,
        visual_feature_dim=32,
        pool_visual_feature_map=False,
    )
    policy = DiffusionPolicy(fake_env, config)

    training_observations = {
        "state": torch.randn(2, 2, 5),
        "rgb": torch.randint(0, 256, (2, 2, 3, 128, 128), dtype=torch.uint8),
    }
    loss = policy.compute_loss(training_observations, torch.rand(2, 8, 4) * 2 - 1)
    assert loss.ndim == 0 and torch.isfinite(loss)

    evaluation_observations = {
        "state": torch.randn(2, 2, 5),
        "rgb": torch.randint(0, 256, (2, 2, 128, 128, 3), dtype=torch.uint8),
    }
    assert policy.get_action(evaluation_observations).shape == (2, 1, 4)
