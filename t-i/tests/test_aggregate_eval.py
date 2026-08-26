import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_aggregate_eval_script(tmp_path):
    evaluation_root = tmp_path / "evaluation-final"
    checkpoint = "/tmp/best_success_once.pt"
    for seed, success in [(0, 0.2), (1, 0.4), (2, 0.6)]:
        output_dir = evaluation_root / f"seed-{seed}"
        output_dir.mkdir(parents=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "checkpoint": checkpoint,
                    "iteration": 25000,
                    "num_eval_episodes": 100,
                    "seed": seed,
                    "metrics": {"success_once": success, "episode_len": 100.0},
                }
            ),
            encoding="utf-8",
        )

    script = Path(__file__).parents[1] / "scripts" / "aggregate_eval.py"
    subprocess.run(
        [sys.executable, str(script), "--evaluation-root", str(evaluation_root)],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((evaluation_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [0, 1, 2]
    assert summary["total_eval_episodes"] == 300
    assert summary["checkpoint_iteration"] == 25000
    assert summary["metrics"]["success_once"]["mean"] == pytest.approx(0.4)
    assert summary["metrics"]["success_once"]["sample_std"] == pytest.approx(0.2)
