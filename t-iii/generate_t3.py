#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ripl_t3.pipeline import (
    FailureEvidence,
    OpenAICompatibleLLM,
    PoseRegion,
    RecordedLLM,
    generate_reward_artifact,
    render_grounding_prompt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate a Push-T reward and failure-focused reset sampler"
    )
    parser.add_argument("--video", type=Path, required=True, help="Representative T-II mp4")
    parser.add_argument("--failure-config", type=Path, required=True)
    parser.add_argument("--episode-record", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=os.getenv("RIPL_T3_MODEL", "Qwen/Qwen3.8-27B"))
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--video-url", help="HTTP URL visible to the model server; otherwise use a data URL"
    )
    parser.add_argument(
        "--recorded-responses", type=Path, help="Replay JSON responses without an API call"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the first-stage prompt only")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return values


def _read_episode(path: Path | None, index: int) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        matches = [record for record in records if int(record.get("episode_index", -1)) == index]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one episode_index={index} record in {path}, got {len(matches)}"
            )
        return dict(matches[0])
    return _read_json(path)


def main() -> None:
    args = _parse_args()
    config = _read_json(args.failure_config)
    evidence = FailureEvidence(
        failure_mode_name=str(config["failure_mode_name"]),
        video_path=args.video,
        pose_region=PoseRegion.from_dict(config["pose_region"]),
        episode_record=_read_episode(args.episode_record, args.episode_index),
        analyst_description=str(config.get("analyst_description", "")),
    )
    if args.dry_run:
        print(json.dumps(render_grounding_prompt(evidence), indent=2))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is set")

    if args.recorded_responses:
        llm = RecordedLLM(_read_json(args.recorded_responses))
    else:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {args.api_key_env} before calling the model API")
        llm = OpenAICompatibleLLM(
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
        )
    result = generate_reward_artifact(
        evidence,
        llm,
        args.output_dir,
        video_url=args.video_url,
    )
    print(f"T-III artifact: {result.output_dir}")
    print(f"Validation passed: {result.validation_passed}")
    if not result.validation_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
