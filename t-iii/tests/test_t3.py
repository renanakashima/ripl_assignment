import json
from pathlib import Path

import pytest

from ripl_t3.pipeline import FailureEvidence, PoseRegion, RecordedLLM, generate_reward_artifact
from ripl_t3.validation import inspect_source, validate_reward_code, validate_sampler_code

REWARD_CODE = """def compute_dense_reward(state):
    position_error = torch.linalg.vector_norm(state["tee_xy"] - state["goal_xy"], dim=-1)
    angle_delta = state["tee_yaw"] - state["goal_yaw"]
    angle_error = torch.abs(torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta)))
    progress = state["overlap"] - state["previous_overlap"]
    success = (state["overlap"] >= 0.90).to(state["overlap"].dtype)
    position = torch.exp(-10.0 * position_error)
    angle = torch.exp(-2.0 * angle_error)
    reward = position + angle + 2.0 * progress + 5.0 * success
    return {"reward": reward, "position": position, "angle": angle, "success": success}
"""

SAMPLER_CODE = """def sample_initial_poses(num_envs, device, generator):
    choose_failure = torch.rand(num_envs, device=device, generator=generator) < 0.8
    failure_x = -0.10 + 0.10 * torch.rand(num_envs, device=device, generator=generator)
    failure_y = -0.10 + 0.20 * torch.rand(num_envs, device=device, generator=generator)
    failure_theta = torch.deg2rad(90.0 + 90.0 * torch.rand(num_envs, device=device, generator=generator))
    global_x = -0.10 + 0.20 * torch.rand(num_envs, device=device, generator=generator)
    global_y = -0.10 + 0.30 * torch.rand(num_envs, device=device, generator=generator)
    global_theta = 2.0 * torch.pi * torch.rand(num_envs, device=device, generator=generator)
    return {
        "x_rel": torch.where(choose_failure, failure_x, global_x),
        "y_rel": torch.where(choose_failure, failure_y, global_y),
        "theta_rad": torch.where(choose_failure, failure_theta, global_theta),
    }
"""

REGION = PoseRegion(-0.10, 0.0, -0.10, 0.10, 90.0, 180.0)


def test_pose_region_rejects_values_outside_nominal_domain() -> None:
    with pytest.raises(ValueError, match="x-relative"):
        PoseRegion(-0.11, 0.0, -0.10, 0.20, 0.0, 360.0)


def test_generated_program_contracts() -> None:
    reward = validate_reward_code(REWARD_CODE)
    sampler = validate_sampler_code(SAMPLER_CODE, REGION.to_dict())
    assert reward.passed, reward.errors
    assert sampler.passed, sampler.errors


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import os\ndef compute_dense_reward(state):\n    return {}\n", "Imports are forbidden"),
        (
            'def compute_dense_reward(state):\n    return {"reward": torch.load("x.pt")}\n',
            "Forbidden torch capability",
        ),
    ],
)
def test_static_inspection_rejects_unsafe_code(source: str, expected: str) -> None:
    _, errors = inspect_source(source, "compute_dense_reward")
    assert any(expected in error for error in errors)


def test_recorded_pipeline_preserves_reproducibility_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "failure.mp4"
    video.write_bytes(b"synthetic-test-video")
    evidence = FailureEvidence(
        "contact_loss",
        video,
        REGION,
        {"episode_index": 7, "success_once": False},
    )
    llm = RecordedLLM(
        {
            "grounding": {
                "observed_failure": "The tool loses contact before useful rotation.",
                "visual_evidence": ["The T remains far from the goal."],
                "hypothesized_mechanism": "The policy approaches from an ineffective side.",
                "recovery_objective": "Maintain contact while increasing overlap.",
                "uncertainties": ["Contact force is not visible."],
            },
            "planning": {
                "stages": [{"name": "approach", "activation": "far", "objective": "contact"}],
                "reward_terms": [{"name": "overlap", "formula": "overlap", "weight": 1.0}],
                "sampler_strategy": {
                    "distribution": "mixture",
                    "failure_mass": 0.8,
                    "exploration_mass": 0.2,
                },
                "verification_questions": ["Does goal alignment dominate return?"],
            },
            "coding": {
                "reward_code": REWARD_CODE,
                "sampler_code": SAMPLER_CODE,
                "rationale": "Progress and alignment target the grounded failure.",
                "llm_failure_modes": ["The agent may oscillate for progress reward."],
                "manual_effort": [],
            },
        }
    )

    result = generate_reward_artifact(evidence, llm, tmp_path / "artifact")
    assert result.validation_passed
    assert (result.output_dir / "reward.py").read_text().startswith("def compute_dense_reward")
    assert json.loads((result.output_dir / "validation.json").read_text())["passed"] is True
