# T-I: RGB Diffusion Policy for ManiSkill

This directory contains the trained `PushT-v1` baseline used by T-II–T-IV. The canonical
experiment uses 100 RGB demonstrations, the native `pd_ee_delta_pose` controller, spatial visual
features, one-step replanning, and 50,000 updates.

The implementation follows [Diffusion Policy](https://arxiv.org/abs/2303.04137) and is adapted
from ManiSkill's
[official baseline](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/diffusion_policy).
Python 3.10–3.12 is supported; Python 3.13 is intentionally rejected.

## Minimal file map

```text
t-i/
├── train_dp.py                    # train or resume one policy
├── eval_dp.py                     # evaluate one fixed checkpoint
├── diagnose_dp.py                 # optional visual/action diagnostics
├── configs/
│   └── pusht_rgb_delta_pose.yaml  # canonical Push-T experiment
├── ripl/                          # config, data, model, policy, env, evaluation
├── scripts/
│   ├── runpod_pusht.sh            # setup/prepare/smoke/train/eval workflow
│   ├── prepare_pusht_delta_pose_demos.sh
│   ├── aggregate_eval.py
│   └── README.md                  # Colab, HCE, and historical launcher index
├── results/                       # experiment decisions and reported metrics
└── tests/
```

The other configs and launchers are retained as experiment provenance. They are not required for
the final Push-T reproduction path.

## Install and verify

Run commands from this directory:

```bash
python -m pip install -e '.[dev]'
python scripts/verify_setup.py
python -m pytest -q
```

On Runpod, `bash scripts/runpod_pusht.sh setup` installs the same package plus the headless Vulkan
requirements. The tested container is
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`.

## Reproduce the fixed Push-T experiment

The Runpod launcher is the shortest complete path. It keeps datasets, logs, checkpoints, and
evaluations under `/workspace/ripl-artifacts` by default.

```bash
cd /workspace/ripl_assignment/t-i
bash scripts/runpod_pusht.sh setup
bash scripts/runpod_pusht.sh prepare
bash scripts/runpod_pusht.sh smoke
bash scripts/runpod_pusht.sh train
```

The preparation step checks all of the conditions that materially affected Push-T reproducibility:

- 100 successful source trajectories using `pd_ee_delta_pose`;
- 1,024 collection and replay environments for the reportable dataset;
- aligned observations from `env_states[t + 1]`, yielding `T + 1` RGB frames for `T` actions.

For a small-GPU smoke test only, lower the replay parallelism explicitly:

```bash
ALLOW_REPLAY_ENV_MISMATCH=1 REPLAY_ENVS=64 \
  bash scripts/prepare_pusht_delta_pose_demos.sh
```

Do not use that mismatched replay for the reported experiment.

### Resume safely

Periodic checkpoints are written every 5,000 iterations. Resume the newest validated checkpoint
without changing the original configuration:

```bash
RIPL_RESUME=/workspace/ripl-artifacts/runs/RUN/checkpoints/step_15000.pt \
  bash scripts/runpod_pusht.sh train --total-iters 50000
```

### Final evaluation

Evaluate the same `best_success_once.pt` checkpoint for 100 episodes at each seed 0, 1, and 2:

```bash
RIPL_RUN_DIR=/workspace/ripl-artifacts/runs/RUN \
  bash scripts/runpod_pusht.sh eval
```

This produces `evaluation-final/seed-{0,1,2}/metrics.json` and
`evaluation-final/summary.json`. Report the three seed-level rates, their mean and sample standard
deviation, and the pooled successes out of 300 episodes.

## Metric semantics

The primary metric is `success_once`: whether T/goal overlap reaches at least 0.90 at any point in
the episode. `success_at_end`, return, episode length, maximum overlap, and final overlap are also
saved. A rollout can therefore count as a success even if later actions move the block away; this
distinction matters when interpreting videos and defining T-II failure modes.

Training loss and offline action agreement are diagnostics, not evidence of closed-loop success.
Use rollout metrics and videos to select a checkpoint.

## Other execution environments

The policy and configuration are platform-independent; only the launcher changes.

| Environment | Entry point |
|---|---|
| Colab | `scripts/setup_colab.sh`, then `train_dp.py` |
| Georgia Tech HCE | `scripts/prepare_hce_pusht_delta_pose.sbatch`, then `train_hce_pusht_delta_pose.sbatch` |
| Runpod | `scripts/runpod_pusht.sh {setup,prepare,smoke,train,eval}` |

On HCE, create `logs/` before `sbatch` because Slurm opens the output file before running the job.
Run `python` directly inside the batch allocation; do not add a nested `srun`.

## What to preserve

For a reproducible submission, save:

- the exact YAML config and selected checkpoint hash;
- demonstration metadata and the replay-validation log;
- training log, wall time, and sampled GPU-memory CSV;
- every seed's `metrics.json`, aggregate `summary.json`, and representative videos;
- checkpoint lineage when training was resumed.

Large runtime artifacts belong in backed-up output storage, not ordinary Git history. Source,
configs, tests, and concise result summaries belong in Git.

## Historical experiments

These files explain the debugging sequence but are not alternative instructions for the final run:

| File | Purpose |
|---|---|
| `configs/pusht_rgb.yaml` | original globally pooled Push-T baseline |
| `configs/pusht_rgb_spatial.yaml` | spatial-feature correction |
| `configs/pusht_rgb_20pct.yaml` | historical task-tuned attempt |
| `configs/pushcube_rgb.yaml` | completed PushCube control experiment |
| `results/pusht_progress_2026-08-26.md` | checkpoint and replay-alignment decisions |
| `results/pushcube_ti_results.md` | completed three-seed PushCube result |

## Attribution

The U-Net, visual encoder, sequence-padding convention, DDPM schedule, EMA, and fair-evaluation
wrappers are adapted from the Apache-2.0 ManiSkill baseline at commit
`62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` (2026-08-01).
