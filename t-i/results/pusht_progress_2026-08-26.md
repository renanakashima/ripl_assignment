# Push-T Progress Log — 2026-08-26

This document preserves the current state of the ManiSkill visual Diffusion Policy assignment,
including completed controls, failed Push-T experiments, diagnostic evidence, dataset findings,
implemented fixes, HCE/Slurm details, and the exact continuation point.

## Assignment target

The required T-I deliverable is a visual-input Diffusion Policy trained with imitation learning,
with training loss, hyperparameters, wall time, VRAM, and success over 100 evaluation rollouts for
each of three random seeds. T-II subsequently requires two reproducible Push-T failure modes,
quantitative initial-state characterizations, representative MP4s, and targeted 100-rollout by
three-seed evaluations for each mode.

## Completed PushCube control

PushCube established that the general RGB policy, spatial encoder, training loop, checkpointing,
and evaluation code can work:

- Run: `pushcube-rgb-diffusion-spatial__seed1__20260825-150140`
- Dataset: 100 motion-planning trajectories, 6,870 transitions, 6,770 windows
- Controller/backend: `pd_ee_delta_pos`, `physx_cpu`
- Training: 30,000 iterations, batch 256, observation horizon 2, action horizon 8,
  prediction horizon 16, 100 DDPM steps
- Selected checkpoint: first 95% diagnostic checkpoint at iteration 20,000
- Training wall time: 1,863 seconds on NVIDIA L40S
- Peak sampled VRAM: 13,736 MiB
- Final evaluation: seed 0 = 78%, seed 1 = 81%, seed 2 = 87%
- Reported result: **82.0% +/- 4.6 percentage points**, 246/300 successes

Full evidence is in `results/pushcube_ti_results.md`.

## Why PushCube does not exercise the Push-T replay defect

PushCube demonstrations are prepared from motion-planning data with `--use-first-env-state` on
`physx_cpu`. Only the initial state is restored, after which actions and observations evolve
sequentially through the CPU replay path.

Push-T demonstrations are RL trajectories prepared with `--use-env-states` on `physx_cuda`.
This enters ManiSkill's GPU-parallel replay path and restores a recorded environment state after
every action. The affected implementation restored `env_states[t]` after action `t`, although a
trajectory with `T` actions contains `T + 1` states and the next action must begin from
`env_states[t + 1]`.

## Historical delta-position Push-T configuration

Historical config: `configs/pusht_rgb_20pct.yaml`

- Environment: `PushT-v1`
- Observation: RGB plus robot proprioception
- Controller: `pd_ee_delta_pos`
- Backend: `physx_cuda`
- Maximum episode length: 150
- Observation/action/prediction horizons: 2 / 1 / 16
- Spatial visual encoder: enabled (`pool_visual_feature_map: false`)
- Diffusion steps: 100
- U-Net dimensions: `[64, 128, 256]`
- Batch size: 256
- Learning rate: 0.0001 with 500-step warmup and cosine decay
- Training budget: 50,000 iterations
- Diagnostic evaluation: 20 episodes every 5,000 iterations

The one-step action horizon matches ManiSkill's tuned state-based Push-T command. ManiSkill does
not publish a tuned RGB Push-T baseline in its current `baselines.sh`; its published Push-T command
is state based, uses `pd_ee_delta_pose`, 100 RL demonstrations, 50,000 iterations, and action
horizon 1. ManiSkill issue #882 reports only approximately 15–20% success for one state-based
reproduction, so 20% RGB is an ambitious target rather than an established expected result.

## Demonstration provenance

Downloaded raw file:

```text
~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5
```

Audit of the raw data:

| Metric | Value |
|---|---:|
| Raw trajectories | 888 |
| Trajectories with success labels | 888 |
| `success_once` | 888 |
| `success_at_end` | 888 |

Thus the source data is valid. Initial GPU RGB replay attempts requested 100 trajectories but
retained only 90 because ManiSkill defaults to filtering on replay-time final success. Repeating
with 64 and 256 parallel environments produced the same 90/100 result, ruling out replay
parallelism as the primary explanation.

Commit `e00e39c` changed the preparation workflow to:

1. verify that the selected raw source trajectories end successfully;
2. use `--allow-failure` only after that verification;
3. require exactly 100 RGB output trajectories; and
4. default to 256 replay environments.

This produced 100 RGB trajectories, 9,743 transitions, and 9,643 training windows. However, that
100-trajectory RGB dataset was generated before the state-index alignment fix below and must be
treated as suspect.

## Failed 100-demo Push-T run

Run:

```text
runs/pusht-rgb-diffusion-spatial-replan1__seed1__20260826-124014
```

Training evaluation remained at zero through the observed checkpoints:

| Iteration | Loss shown near evaluation | `success_once` | `success_at_end` |
|---:|---:|---:|---:|
| 5,000 | 0.0405 | 0% | 0% |
| 10,000 | 0.0301 | 0% | 0% |
| 15,000 | 0.0267 | 0% | 0% |
| 20,000 | 0.0245 | 0% | 0% |
| 25,000 | 0.0193 | 0% | 0% |
| 30,000 | 0.0188 | 0% | 0% |
| 35,000 | 0.0075 | 0% | 0% |

The terminal log also showed Slurm step `6909.0` killed at its time limit around iteration 36,751,
but the run directory contains `step_50000.pt`, and the diagnostic checkpoint reports iteration
50,000. Preserve both facts; use `sacct` if exact job lineage is needed for the final report.

### Offline diagnostic at iteration 50,000

Command:

```bash
python diagnose_dp.py \
  --checkpoint runs/pusht-rgb-diffusion-spatial-replan1__seed1__20260826-124014/checkpoints/step_50000.pt
```

Results:

| Diagnostic | Value |
|---|---:|
| Diffusion loss | 0.0106576 |
| Loss with shuffled RGB | 0.7345653 |
| Condition mean absolute change after RGB shuffle | 0.0228238 |
| Action mean absolute change after RGB shuffle | 0.1473866 |
| Sampled-action MSE to demonstration | 0.0222317 |
| Sampled-action cosine to demonstration | 0.9962674 |
| Sampled-action mean absolute value | 0.3115980 |
| Demonstration action mean absolute value | 0.3459596 |
| Sampled-action saturation fraction | 0.109375 |

Interpretation: the model strongly uses RGB and fits the saved offline action targets extremely
well. The zero closed-loop result is not explained by simple visual neglect or gross underfitting.
Because this diagnostic is computed on the same pre-fix dataset used for training, it is also
consistent with the model accurately learning a temporally misaligned observation/action mapping.
Do not resume this checkpoint on the corrected dataset.

## GPU replay state-index defect and fix

ManiSkill's GPU `--use-env-states` replay path in the inspected implementation executes action
`t` and then restores `env_states[t]`. The required next state is `env_states[t + 1]`. Restoring the
previous state means later actions can be executed from the wrong recorded state, corrupting the
visual observation/action relationship even while producing a low offline training loss.

Commit `0d369dc` added:

- `scripts/replay_trajectory_aligned.py`, which inspects the installed replay implementation and
  changes exactly one `env_states[t]` restoration to `env_states[t + 1]` in process;
- a fail-closed guard if the installed source does not match the expected pattern;
- three passing focused unit tests; and
- integration of the aligned wrapper into `scripts/prepare_pusht_20pct_demos.sh`.

The workaround does not modify the installed virtual environment on disk.

## Exact continuation point

Latest interactive allocation snapshot:

- Slurm job: `6921`
- Requested node/GPU: `nvidia-l4`, one L4
- Compute hostname: `doop`
- Time allocation: 45 minutes
- Repository already pulled through commit `0d369dc`
- Old iteration-50,000 diagnostic completed successfully
- The current canonical RGB file still needs to be moved aside and regenerated with the aligned
  wrapper unless this was done after this progress log was written

Run inside the GPU allocation:

```bash
cd ~/ripl_assignment/t-i
source .venv/bin/activate

DEMO_DIR="$HOME/.maniskill/demos/PushT-v1/rl"

mv -i \
  "$DEMO_DIR/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5" \
  "$DEMO_DIR/trajectory.rgb.pd_ee_delta_pos.physx_cuda.before-alignment-fix.h5"

mv -i \
  "$DEMO_DIR/trajectory.rgb.pd_ee_delta_pos.physx_cuda.json" \
  "$DEMO_DIR/trajectory.rgb.pd_ee_delta_pos.physx_cuda.before-alignment-fix.json"

NUM_DEMOS=100 REPLAY_ENVS=64 \
bash scripts/prepare_pusht_20pct_demos.sh
```

Required confirmation:

```text
Applied PushT GPU replay alignment: env_states[t] -> env_states[t + 1]
Verified 100 successful source trajectories
Verified 100 RGB trajectories
```

Then leave the short interactive allocation and submit a new run from a login node. Prefer H200;
L40/L40S are also suitable when available. Do not resume a pre-fix checkpoint.

```bash
exit
cd ~/ripl_assignment/t-i
mkdir -p logs

sbatch \
  --nodelist=nvidia-h200 \
  --gres=gpu:h200:1 \
  --time=02:00:00 \
  scripts/train_hce_pusht_20pct.sbatch
```

Monitor the new job at 5,000-iteration intervals. Any nonzero result among the 20 diagnostic
episodes is useful evidence that replay alignment was causal. If the corrected run remains at 0%
through approximately 15,000–20,000 iterations, pause before another full run and perform:

1. the same `diagnose_dp.py` shuffled-RGB and offline-action audit;
2. rollout-video inspection for contact direction, overshoot, oscillation, and missed contact;
3. an official state-based Push-T control using 100 demonstrations; and
4. only then a controlled visual-policy change, such as a coordinate-aware encoder or a different
   controller/dataset, rather than changing several variables simultaneously.

## T-II status

T-II remains feasible but has not been implemented. The current evaluator records aggregate
episode metrics and videos but does not yet save initial T pose or constrain reset configurations.
After a defensible Push-T base policy exists, add a survey evaluator that records initial T
`x/y/yaw`, outcome, overlap, final pose errors, action reversals, and video paths. Use survey data
to identify two empirical failure modes, then implement constrained-reset evaluation with 100
rollouts under seeds 0, 1, and 2 for each failure configuration.

Do not invent the second failure mode before inspecting rollouts. One previously observed behavior
is pushing the T away from the target; the other must be established empirically.

## Native delta-pose recovery plan

The next controlled experiment supersedes the older continuation command above. A provenance
audit found an official Push-T RL dataset whose controller is already `pd_ee_delta_pose`, matching
ManiSkill's published tuned state-based Push-T command. Its metadata records collection with 1,024
parallel environments, 719 episodes, and 719 successful episode labels. This avoids unsupported
controller conversion and lets replay use the same parallelism as collection, which ManiSkill
identifies as important for Push-T fidelity.

New experiment lineage:

- Config: `configs/pusht_rgb_delta_pose.yaml`
- Raw data: `trajectory.none.pd_ee_delta_pose.physx_cuda.h5`
- RGB data: `trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5`
- Demonstrations: exactly 100 successful trajectories
- Collection/replay environments: 1,024 / 1,024 for the reportable dataset
- Controller: native `pd_ee_delta_pose` throughout
- Observation: RGB plus robot proprioception
- Replay alignment: recorded `env_states[t + 1]` after action `t`
- Horizons: observation/action/prediction = 2 / 1 / 16
- Training: 50,000 iterations, batch 256, 20 diagnostic episodes every 5,000 iterations

The pipeline adds `scripts/validate_pusht_replay.py`, which checks source provenance and success,
collection/replay parallelism, output controller and RGB mode, exact trajectory count, and one
more RGB observation than actions in every episode. A reduced replay is allowed only by setting
`ALLOW_REPLAY_ENV_MISMATCH=1`, and is suitable for integration testing rather than final results.

Queue the faithful replay, then train from scratch only after all validator messages appear:

```bash
cd ~/ripl_assignment/t-i
source .venv/bin/activate
mkdir -p logs

sbatch scripts/prepare_hce_pusht_delta_pose.sbatch
# after successful replay validation:
sbatch scripts/train_hce_pusht_delta_pose.sbatch
```

Do not reuse the delta-position RGB file or resume either zero-success checkpoint. At iterations
15,000–20,000, require both rollout video review and overlap diagnostics; loss alone is not a
closed-loop metric. If native delta-pose remains at zero, establish a state-based control on the
same 100 source episodes before changing the RGB architecture. If the state control succeeds but
RGB fails, the next isolated variable should be the image encoder, not data/controller/backend
simultaneously. A checkpoint that clears the 20% diagnostic target should be evaluated with
`scripts/eval_hce_pusht_delta_pose.sbatch` over 100 episodes for each seed 0, 1, and 2.
