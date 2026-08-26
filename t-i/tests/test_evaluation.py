import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")

from ripl.evaluation import evaluate_policy


class _FakePolicy:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def get_action(self, observations):
        return torch.zeros((1, 1, 1))


class _FakeEnv:
    num_envs = 1

    def __init__(self):
        self.reset_seed = None
        self.nest_final_info = True

    def reset(self, seed=None):
        self.reset_seed = seed
        return np.zeros((1, 1), dtype=np.float32), {}

    def step(self, action):
        episode = {
            "success_once": torch.tensor([1.0]),
            "episode_len": torch.tensor([1.0]),
        }
        info = (
            {"final_info": {"episode": episode, "overlap": torch.tensor([0.75])}}
            if self.nest_final_info
            else {"episode": episode, "overlap": np.array([0.75], dtype=np.float32)}
        )
        return (
            np.zeros((1, 1), dtype=np.float32),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            info,
        )


def test_evaluation_seeds_environment_reset():
    env = _FakeEnv()
    metrics = evaluate_policy(
        1,
        _FakePolicy(),
        env,
        torch.device("cpu"),
        "physx_cuda",
        progress_bar=False,
        seed=123,
    )
    assert env.reset_seed == 123
    assert metrics["success_once"].tolist() == [1.0]
    assert metrics["max_overlap"].tolist() == [0.75]
    assert metrics["final_overlap"].tolist() == [0.75]


def test_evaluation_accepts_direct_cpu_terminal_metrics():
    env = _FakeEnv()
    env.nest_final_info = False
    metrics = evaluate_policy(
        1,
        _FakePolicy(),
        env,
        torch.device("cpu"),
        "physx_cpu",
        progress_bar=False,
        seed=7,
    )
    assert env.reset_seed == 7
    assert metrics["success_once"].tolist() == [1.0]
    assert metrics["max_overlap"].tolist() == [0.75]
    assert metrics["final_overlap"].tolist() == [0.75]
