# RIPL Prospective Student Assignment - Fall 2026

Code, experiment configuration, and report source for an RGB Diffusion Policy on ManiSkill
PushT-v1.

**Status:** T-I is complete. T-II evaluation tooling is implemented, but discovery and targeted
experiments did not complete. T-III contains an unexecuted validation scaffold. T-IV was not
implemented. No T-II, T-III, or T-IV results are claimed.

GitHub: https://github.com/renanakashima/ripl_assignment

## 1. Environment setup

Python 3.10-3.12 is supported. From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install -e './t-i[dev]' -e './t-ii[dev]'
python -m pytest -q t-i/tests t-ii/tests
```

## 2. Dataset and demonstrations

The Push-T run uses 100 successful `pd_ee_delta_pose` demonstrations replayed to RGB with the
`env_states[t + 1]` temporal-alignment correction. On a Runpod Linux/NVIDIA machine:

```bash
cd t-i
bash scripts/runpod_pusht.sh setup
bash scripts/runpod_pusht.sh prepare
bash scripts/runpod_pusht.sh smoke
```

Preparation validates controller, observation count, source success, replay parallelism, and
trajectory alignment before training.

## 3. Train the Diffusion Policy

```bash
cd t-i
bash scripts/runpod_pusht.sh train
```

The canonical configuration is `t-i/configs/pusht_rgb_delta_pose.yaml`. Resume only from a saved
periodic checkpoint:

```bash
RIPL_RESUME=/path/to/step_15000.pt \
  bash scripts/runpod_pusht.sh train --total-iters 50000
```

## 4. Evaluate the fixed checkpoint

```bash
cd t-i
RIPL_RUN_DIR=/workspace/ripl-artifacts/runs/YOUR_RUN \
  bash scripts/runpod_pusht.sh eval
```

This loads `checkpoints/best_success_once.pt`, evaluates seeds 0, 1, and 2 for 100 episodes each,
and writes `evaluation-final/summary.json`. The fixed 15k checkpoint obtains 11.0% mean
`success_once`, with 33/300 pooled successes.

## 5. Failure-mode evaluation (T-II; not completed)

The T-II implementation supports exact pose-conditioned resets, JSONL episode records, NPZ
trajectories, screening analysis, videos, and three-seed targeted evaluation:

```bash
cd t-ii
RIPL_CHECKPOINT=/path/to/best_success_once.pt \
RIPL_T2_ROOT=/workspace/ripl-artifacts/t2-pusht \
RIPL_EPISODES=200 RIPL_NUM_ENVS=20 \
  bash scripts/run_pusht_t2.sh discover

RIPL_T2_ROOT=/workspace/ripl-artifacts/t2-pusht \
  bash scripts/run_pusht_t2.sh analyze
```

Do not run `targeted` until trajectory and video inspection identifies two behaviorally distinct,
contiguous initial-pose regimes. No completed T-II rollout dataset, per-mode rates, or
representative failure videos are included.

## 6. Videos and artifacts

- T-I diagnostics and backups: `t-i/output/runpod_backups/pusht_final_20260827/`
- T-II infrastructure log: `t-ii/output/runpod_backups/pusht_t2_20260827/`
- Report source and generated PDF: `t-i/report/` and `t-i/output/pdf/`

T-I diagnostic videos are not relabeled as T-II failure-mode evidence.

## 7. Known limitations

- Only one policy-training seed was run; the three reported seeds vary evaluation environments.
- The 15k checkpoint was selected by a noisy 20-episode diagnostic; the baseline uses 300 fresh
  episodes.
- T-II could not obtain a graphics-capable replacement GPU before the deadline. An A100 and the
  local Apple M4 both failed the required Vulkan RGB-rendering path.
- T-III and T-IV were not experimentally executed. Their methodology is proposed in the report
  without fabricated outcomes.

## Code map

- [`t-i/`](t-i/): aligned replay, visual policy, training, evaluation, tests, and report.
- [`t-ii/`](t-ii/): pose-conditioned resets, trajectory capture, screening, and targeted evaluator.
- [`t-iii/`](t-iii/): provider-independent LLM reward/sampler validation scaffold; not executed.
