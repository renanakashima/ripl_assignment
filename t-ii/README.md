# Push-T T-II failure-identification and characterization

## Purpose

Use one fixed `best_success_once.pt` checkpoint to discover and validate two behaviorally distinct,
reproducible failure modes. Do not define modes solely from unsuccessful final frames or training
loss. The official primary metric is episode-level `success_once`, which corresponds to reaching
at least 0.90 T/goal overlap at any point during the rollout. Also retain `success_at_end`, maximum
overlap, and final overlap.

## Minimal setup

T-II reuses the exact policy implementation from sibling directory `t-i`; it does not copy model
code. Install both editable packages from the repository root:

```bash
python -m pip install -e './t-i[dev]' -e './t-ii[dev]'
python -m pytest -q t-ii/tests
```

The small source surface has one responsibility per file:

```text
t-ii/
├── eval_pusht_failures.py       # rollout entry point
├── ripl_t2/
│   ├── config.py                # typed CLI options
│   ├── envs.py                  # targeted reset sampler and ManiSkill wrappers
│   ├── failure_evaluation.py    # per-episode metrics and trajectories
│   └── failure_analysis.py      # pose bins, tags, and confidence intervals
├── scripts/
│   ├── run_pusht_t2.sh          # discover, analyze, or targeted protocol
│   └── analyze_pusht_failures.py
└── tests/test_t2.py
```

## Initial-state domain

PushT-v1 fixes the goal at world XY `(-0.156, -0.100)` m and yaw 300 degrees. Its nominal T-block
initial distribution is:

- relative X: `[-0.10, 0.10]` m
- relative Y: `[-0.10, 0.20]` m
- yaw: `[0, 360)` degrees

Robot joint initialization noise remains seed-controlled. `RIPL-PushTTargeted-v1` samples only
within the nominal task distribution and supports wrapped angular ranges such as 330 to 30 degrees.

## Phase 1: discovery

Run at least 200 rollouts with the final fixed checkpoint and save per-step T pose, TCP pose, and
overlap. Use vectorized simulation but keep one JSONL record and one trajectory array per episode.

```bash
cd t-ii
RIPL_CHECKPOINT=/path/to/best_success_once.pt \
RIPL_T2_ROOT=/workspace/ripl-artifacts/t2-pusht \
RIPL_EPISODES=200 RIPL_NUM_ENVS=20 \
bash scripts/run_pusht_t2.sh discover

RIPL_T2_ROOT=/workspace/ripl-artifacts/t2-pusht \
bash scripts/run_pusht_t2.sh analyze
```

The analyzer reports Wilson 95% intervals for success/failure rates, marginal pose bins, joint pose
cells, and non-exclusive behavior tags. The tags are screening aids, not final scientific labels:

- `progress_regression`: substantial overlap was achieved and then lost.
- `near_goal_misalignment`: the block approached the goal but retained a large angular error.
- `overshoot_or_displacement`: the block ended far from its closest approach or outside 0.15 m.
- `low_progress_or_contact_loss`: maximum overlap stayed below 0.40.
- `stalled_partial_overlap`: residual failed episodes not covered above.

Inspect trajectories and videos before selecting two modes. Each selected mode must have a coherent
mechanism and a contiguous initial-pose region with enough discovery support.

## Phase 2: representative videos

Rerun a small number of episodes inside each candidate pose region with
`RIPL_CAPTURE_VIDEO=1`. Preserve the associated `episodes.jsonl` so every MP4 has its exact sampled
initial pose and outcome. Prefer videos that exhibit the typical behavior, not only the most
dramatic outlier.

## Phase 3: targeted base-policy evaluation

For each selected mode, set its name and pose bounds, then run 100 episodes at each seed 0, 1, and
2. This is 300 episodes per mode and 600 targeted episodes total.

```bash
RIPL_CHECKPOINT=/path/to/best_success_once.pt \
RIPL_T2_ROOT=/workspace/ripl-artifacts/t2-pusht \
RIPL_FAILURE_MODE=mode_name \
RIPL_X_REL_MIN=-0.10 RIPL_X_REL_MAX=0.00 \
RIPL_Y_REL_MIN=-0.10 RIPL_Y_REL_MAX=0.10 \
RIPL_THETA_DEG_MIN=90 RIPL_THETA_DEG_MAX=180 \
RIPL_EPISODES=100 RIPL_NUM_ENVS=20 \
bash scripts/run_pusht_t2.sh targeted
```

Replace the example bounds with the empirically selected ranges. Report each seed separately, the
three-seed mean and sample standard deviation, pooled successes out of 300, and a binomial confidence
interval. Preserve the checkpoint hash, pose sampler definition, exact seeds, JSONL records, NPZ
trajectories, analysis JSON/CSV, and representative MP4s.
