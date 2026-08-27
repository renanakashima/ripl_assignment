from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from ripl_t3.state_api import PUSHT_STATE_API
from ripl_t3.validation import validate_reward_code, validate_sampler_code

SYSTEM_PROMPT = """You are a robotics reward-design researcher. Be conservative about what a
rollout video establishes. Never invent simulator state fields. Return valid JSON only. Generated
code may use torch but may not import packages or access files, networks, subprocesses, reflection,
serialization, or dynamic execution."""


@dataclass(frozen=True)
class PoseRegion:
    """A failure-conditioned subset of PushT-v1's nominal reset domain."""

    x_rel_min: float
    x_rel_max: float
    y_rel_min: float
    y_rel_max: float
    theta_deg_min: float
    theta_deg_max: float

    def __post_init__(self) -> None:
        if not all(isfinite(float(value)) for value in asdict(self).values()):
            raise ValueError("Pose-region bounds must be finite")
        if not -0.10 <= self.x_rel_min <= self.x_rel_max <= 0.10:
            raise ValueError("x-relative range must stay within [-0.10, 0.10] m")
        if not -0.10 <= self.y_rel_min <= self.y_rel_max <= 0.20:
            raise ValueError("y-relative range must stay within [-0.10, 0.20] m")
        if abs(self.theta_deg_max - self.theta_deg_min) > 360.0:
            raise ValueError("Angular range must span at most 360 degrees")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PoseRegion:
        return cls(**{key: float(value) for key, value in values.items()})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FailureEvidence:
    failure_mode_name: str
    video_path: Path
    pose_region: PoseRegion
    episode_record: dict[str, Any] = field(default_factory=dict)
    analyst_description: str = ""

    def __post_init__(self) -> None:
        if not self.failure_mode_name.strip():
            raise ValueError("failure_mode_name must not be empty")
        if self.video_path.suffix.lower() != ".mp4" or not self.video_path.is_file():
            raise ValueError(f"Expected an existing mp4 rollout: {self.video_path}")

    def metadata(self) -> dict[str, Any]:
        return {
            "failure_mode_name": self.failure_mode_name,
            "pose_region": self.pose_region.to_dict(),
            "episode_record": self.episode_record,
            "analyst_description": self.analyst_description,
        }


class MultimodalLLM(Protocol):
    model: str

    def complete(self, messages: list[dict[str, Any]], *, stage: str) -> dict[str, Any]: ...


@dataclass
class OpenAICompatibleLLM:
    """Adapter for vLLM, Qwen Cloud, or another OpenAI-compatible API."""

    model: str
    base_url: str
    api_key: str

    def complete(self, messages: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the client with `pip install -e '.[llm]'`") from error
        response = OpenAI(base_url=self.base_url, api_key=self.api_key).chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"Empty LLM response during {stage}")
        return json.loads(content)


@dataclass
class RecordedLLM:
    """Replay saved responses in tests or exact reproductions."""

    responses: dict[str, dict[str, Any]]
    model: str = "recorded-response"

    def complete(self, messages: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
        del messages
        return self.responses[stage]


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    validation_passed: bool


def _grounding_prompt(metadata: dict[str, Any]) -> str:
    return f"""Ground one Push-T failure from the attached rollout and metadata. Separate visible
evidence from hypotheses. Official success means overlap reaches 0.90 at any time.

Metadata:
{json.dumps(metadata, indent=2, sort_keys=True)}

Return: {{"observed_failure": str, "visual_evidence": [str],
"hypothesized_mechanism": str, "recovery_objective": str, "uncertainties": [str]}}"""


def _planning_prompt(grounding: dict[str, Any], metadata: dict[str, Any]) -> str:
    return f"""Design an interpretable staged dense reward and failure-focused reset distribution.
Use ROSETTA-style staging and preserve the unchanged success metric. Avoid rewarding inactivity.
Include progress/regression terms where relevant and retain some full-domain reset coverage.

Grounding:
{json.dumps(grounding, indent=2, sort_keys=True)}

Metadata:
{json.dumps(metadata, indent=2, sort_keys=True)}

State API:
{json.dumps(PUSHT_STATE_API, indent=2, sort_keys=True)}

Return: {{"stages": [{{"name": str, "activation": str, "objective": str}}],
"reward_terms": [{{"name": str, "formula": str, "weight": float, "purpose": str}}],
"sampler_strategy": {{"distribution": str, "failure_mass": float,
"exploration_mass": float}}, "verification_questions": [str]}}"""


def _coding_prompt(
    grounding: dict[str, Any], plan: dict[str, Any], metadata: dict[str, Any]
) -> str:
    return f"""Implement this plan as two batched, device-safe functions. Use torch without imports.

Grounding:
{json.dumps(grounding, indent=2, sort_keys=True)}

Plan:
{json.dumps(plan, indent=2, sort_keys=True)}

Failure region:
{json.dumps(metadata["pose_region"], indent=2, sort_keys=True)}

State API:
{json.dumps(PUSHT_STATE_API, indent=2, sort_keys=True)}

`compute_dense_reward(state)` must return a dict of finite [N] tensors containing `reward`, use
wrapped angular errors, and prefer overlap >= 0.90. `sample_initial_poses(num_envs, device,
generator)` must return [N] tensors `x_rel`, `y_rel`, `theta_rad`; stay inside PushT's nominal
domain; place most probability in the failure region; and retain exploration.

Return: {{"reward_code": str, "sampler_code": str, "rationale": str,
"llm_failure_modes": [str], "manual_effort": [str]}}"""


def _messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _require_keys(values: dict[str, Any], keys: set[str], stage: str) -> None:
    missing = keys - values.keys()
    if missing:
        raise ValueError(f"{stage} response is missing: {', '.join(sorted(missing))}")


def _video_data_url(path: Path, max_bytes: int = 25 * 1024 * 1024) -> str:
    if path.stat().st_size > max_bytes:
        raise ValueError("Video exceeds 25 MiB; compress it or pass --video-url")
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_grounding_prompt(evidence: FailureEvidence) -> dict[str, str]:
    return {"system": SYSTEM_PROMPT, "grounding": _grounding_prompt(evidence.metadata())}


def generate_reward_artifact(
    evidence: FailureEvidence,
    llm: MultimodalLLM,
    output_dir: str | Path,
    *,
    video_url: str | None = None,
) -> PipelineResult:
    """Run grounding, planning, coding, validation, and artifact preservation."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    metadata = evidence.metadata()
    prompts = render_grounding_prompt(evidence)
    grounding = llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url or _video_data_url(evidence.video_path)},
                    },
                    {"type": "text", "text": prompts["grounding"]},
                ],
            },
        ],
        stage="grounding",
    )
    _require_keys(
        grounding,
        {
            "observed_failure",
            "visual_evidence",
            "hypothesized_mechanism",
            "recovery_objective",
            "uncertainties",
        },
        "grounding",
    )

    prompts["planning"] = _planning_prompt(grounding, metadata)
    plan = llm.complete(_messages(prompts["planning"]), stage="planning")
    _require_keys(
        plan,
        {"stages", "reward_terms", "sampler_strategy", "verification_questions"},
        "planning",
    )

    prompts["coding"] = _coding_prompt(grounding, plan, metadata)
    programs = llm.complete(_messages(prompts["coding"]), stage="coding")
    _require_keys(
        programs,
        {"reward_code", "sampler_code", "rationale", "llm_failure_modes", "manual_effort"},
        "coding",
    )

    reward_report = validate_reward_code(str(programs["reward_code"]))
    sampler_report = validate_sampler_code(
        str(programs["sampler_code"]), evidence.pose_region.to_dict()
    )
    validation = {
        "passed": reward_report.passed and sampler_report.passed,
        "reward": reward_report.to_dict(),
        "sampler": sampler_report.to_dict(),
    }

    (root / "reward.py").write_text(str(programs["reward_code"]).rstrip() + "\n")
    (root / "episode_sampler.py").write_text(str(programs["sampler_code"]).rstrip() + "\n")
    _write_json(root / "prompts.json", prompts)
    _write_json(
        root / "responses.json",
        {"grounding": grounding, "planning": plan, "coding": programs},
    )
    _write_json(root / "validation.json", validation)
    _write_json(
        root / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": llm.model,
            "video": str(evidence.video_path.resolve()),
            "video_sha256": _sha256(evidence.video_path),
            "failure_evidence": metadata,
            "validation_passed": validation["passed"],
        },
    )
    return PipelineResult(root, bool(validation["passed"]))
