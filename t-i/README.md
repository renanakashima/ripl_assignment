# Visual Diffusion Policy for ManiSkill

This repository implements imitation learning with an RGB-conditioned Diffusion Policy for
ManiSkill's `PushT-v1` and `PushCube-v1` tasks. It is organized for editing in Cursor and training
on Google Colab or Georgia Tech HCE; no local NVIDIA server is required.

The policy follows the core formulation from the
[Diffusion Policy paper](https://arxiv.org/abs/2303.04137): it encodes the two latest visual and
proprioceptive observations, conditions a 1-D U-Net with those features through FiLM, denoises a
16-step action sequence with a cosine-schedule DDPM, executes a configurable action chunk, and
replans. “Visual” here means RGB is the main scene input while low-dimensional robot state is
retained, as in a standard visuomotor policy.

The implementation is adapted from ManiSkill's
[official Diffusion Policy baseline](https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/diffusion_policy),
and the environment/data workflow follows the
[official imitation-learning setup](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html).

## Repository layout

```text
.
├── train_dp.py                  # training entry point
├── eval_dp.py                   # checkpoint evaluation entry point
├── configs/
│   ├── pushcube_rgb.yaml        # official-aligned visual PushCube experiment
│   ├── pusht_rgb.yaml           # original PushT experiment
│   ├── pusht_rgb_spatial.yaml   # spatial PushT follow-up
│   ├── pusht_rgb_20pct.yaml     # task-tuned spatial PushT attempt
│   ├── pusht_rgb_delta_pose.yaml # native-controller visual PushT experiment
│   └── smoke.yaml               # ten-update integration test
├── scripts/
│   ├── aggregate_eval.py        # summarize final metrics across evaluation seeds
│   ├── eval_hce_pushcube.sbatch # three-seed, 300-rollout PushCube evaluation
│   ├── setup_colab.sh           # Python dependencies + headless Vulkan
│   ├── prepare_pushcube_demos.sh
│   ├── prepare_pusht_demos.sh   # download and replay RGB demonstrations
│   ├── prepare_pusht_20pct_demos.sh
│   ├── prepare_pusht_delta_pose_demos.sh
│   ├── prepare_hce_pusht_delta_pose.sbatch
│   ├── train_hce_pushcube.sbatch
│   ├── train_hce_pusht_20pct.sbatch
│   ├── eval_hce_pusht_20pct.sbatch
│   ├── train_hce_pusht_delta_pose.sbatch
│   ├── eval_hce_pusht_delta_pose.sbatch
│   ├── train_colab.sh
│   └── verify_setup.py
├── notebooks/
│   └── pusht_diffusion_colab.ipynb
├── ripl/                        # model, dataset, environments, and evaluation
├── results/
│   ├── pushcube_ti_results.md   # completed 3-seed PushCube baseline
│   └── pusht_progress_2026-08-26.md # Push-T experiments and continuation point
└── tests/
```

## Why Python 3.12

The project declares `requires-python = ">=3.10,<3.13"`. Google Colab's Python 3.12 runtime is
supported. Python 3.13 is rejected early because the present ManiSkill/PyTorch/rendering stack is
not the target of this setup. Do not downgrade Colab below 3.12 unless a dependency regression
specifically requires it.

## Colab workflow

First, push this folder to a GitHub repository from Cursor. In Colab select **Runtime > Change
runtime type > T4 GPU**. Confirm that `!python --version` reports 3.12.x, then run these cells:

```python
!git clone https://github.com/renanakashima/ripl_assignment.git
%cd ripl_assignment/t-i
!bash scripts/setup_colab.sh
```

The recommended Push-T vision experiment uses the native `pd_ee_delta_pose` source selected by
ManiSkill's published Push-T command. Its demonstrations were collected with 1,024 GPU
environments. Generate the high-fidelity RGB replay once on a large HCE GPU, then copy the HDF5
and JSON files to Colab or Google Drive. A T4-conscious replay is available for smoke testing, but
must be marked explicitly because it does not match collection parallelism:

```python
!ALLOW_REPLAY_ENV_MISMATCH=1 NUM_DEMOS=100 REPLAY_ENVS=64 \
  bash scripts/prepare_pusht_delta_pose_demos.sh
```

The script validates the native controller, the first 100 source success labels, collection versus
replay parallelism, output controller and observation mode, exact trajectory count, and the
required `T + 1` RGB frames for `T` actions. Push-T is unusually sensitive to tiny simulator
differences, so use the 1,024-environment HCE replay for the actual experiment.

Run the short end-to-end test before committing to a full job:

```python
!python train_dp.py \
  --config configs/pusht_rgb_delta_pose.yaml \
  --exp-name pusht-rgb-delta-pose-smoke \
  --num-demos 4 --batch-size 8 \
  --total-iters 1 --warmup-steps 1 \
  --eval-freq 1 --save-freq 1 \
  --num-eval-episodes 1 --num-eval-envs 1 \
  --no-capture-video
```

Then train the full experiment:

```python
!python train_dp.py --config configs/pusht_rgb_delta_pose.yaml --batch-size 128
```

The full config requests 50,000 gradient updates. Runtime depends on the assigned Colab GPU and
evaluation frequency, and may exceed one free-session window. For durable checkpoints, mount
Google Drive and override the output directory:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
!python train_dp.py --config configs/pusht_rgb_delta_pose.yaml --batch-size 128 \
  --output-dir /content/drive/MyDrive/ripl-pusht-runs
```

Resume after a disconnect by setting a larger or unchanged `total_iters` and loading a periodic
checkpoint:

```python
!python train_dp.py --config configs/pusht_rgb_delta_pose.yaml --batch-size 128 \
  --resume /content/drive/MyDrive/ripl-pusht-runs/RUN_NAME/checkpoints/step_5000.pt \
  --total-iters 50000
```

To evaluate the exponential-moving-average policy and save videos/metrics:

```python
!python eval_dp.py \
  --checkpoint /content/drive/MyDrive/ripl-pusht-runs/RUN_NAME/checkpoints/final.pt \
  --num-eval-episodes 20 --num-eval-envs 10
```

To test temporal execution without retraining, pass `--act-horizon 8`. This changes only how many
actions from the checkpoint's 16-step prediction are executed before replanning; it does not alter
the learned weights. Push-T evaluation also reports `max_overlap` and `final_overlap`; success
requires overlap of at least 0.90.

The reported research metric is `success_once`. Training also logs `success_at_end`, return, loss,
and learning rate. Inspect logs with:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/ripl-pusht-runs
```

## Configuration notes

- `obs_mode: rgb` is intentional. ManiSkill documents its RGB+Depth Diffusion Policy path as
  untuned, while RGB is the supported visual baseline.
- `act_horizon: 1` follows ManiSkill's tuned PushT state-policy command. It makes the controller
  replan every step, which is useful for this precise contact-rich task.
- `max_episode_steps: 150` follows the official PushT baseline rather than the task's default 100.
- `dataset_device: cpu` conserves Colab GPU memory. Images stay compressed as `uint8` in RAM and
  minibatches are transferred asynchronously.
- The official baseline assumes 128×128 camera images. The legacy configuration uses adaptive
  pooling, while the spatial follow-up preserves the resulting 8×8 convolutional feature map.
- To log to Weights & Biases, run `pip install -e '.[tracking]'` and set `track: true`.

## Expected-results caveat

[`PushT-v1` issue #882](https://github.com/mani-skill/ManiSkill/issues/882) reports roughly
0.15–0.20 success for one state-based reproduction and asks why a paper reports 0.4–0.5; the issue
is closed without a maintainer answer. It does not establish an expected RGB result. Treat seeds,
demo replay parallelism, dataset provenance, simulator backend, and checkpoint selection as part of
the experiment. Report your exact configuration and mean success over multiple seeds rather than
claiming that one run reproduces a separate paper.

## Spatial Push-T correction

`configs/pusht_rgb.yaml` preserves the first completed baseline configuration, whose visual
encoder globally pools its convolutional feature map. That can discard the T block's image
location: in RGB mode, Push-T exposes robot/TCP state but withholds privileged object and goal
poses. The paper-aligned follow-up `configs/pusht_rgb_spatial.yaml` therefore retains the 8×8
convolutional feature map before projecting it to the 256-dimensional visual embedding and uses
an eight-step execution horizon for temporally consistent pushes. It is a new architecture and
must be trained from scratch; old checkpoints remain loadable because pooled encoding stays the
default when the new configuration field is absent.

On Georgia Tech HCE, create `logs/` before submission because Slurm opens the output file before
the job script runs:

```bash
mkdir -p logs
sbatch scripts/train_hce_spatial.sbatch
```

If a trained visual checkpoint still has low success, measure whether sampled actions react to
the RGB input before changing the architecture again:

```bash
python diagnose_dp.py --checkpoint runs/RUN_NAME/checkpoints/final.pt
```

## PushCube HCE workflow

PushCube is the recommended assignment experiment after the two visual Push-T policies achieved
zero closed-loop success despite using RGB and matching demonstration actions offline. The
PushCube configuration follows ManiSkill's published RGB baseline settings: motion-planning
demonstrations, `pd_ee_delta_pos`, `physx_cpu`, 100-step episodes, and 30,000 training updates. It
retains spatial image features and uses the visual baseline's eight-step execution horizon.

On an interactive L40S allocation, download and replay the first 100 demonstrations with RGB:

```bash
NUM_DEMOS=100 REPLAY_ENVS=10 bash scripts/prepare_pushcube_demos.sh
```

Run a one-update integration test:

```bash
python train_dp.py \
  --config configs/pushcube_rgb.yaml \
  --exp-name pushcube-rgb-smoke \
  --num-demos 4 \
  --batch-size 8 \
  --total-iters 1 \
  --warmup-steps 1 \
  --eval-freq 1 \
  --save-freq 1 \
  --num-eval-episodes 1 \
  --num-eval-envs 1 \
  --no-capture-video
```

If that succeeds, submit the persistent full run:

```bash
mkdir -p logs
bash -n scripts/train_hce_pushcube.sbatch
sbatch scripts/train_hce_pushcube.sbatch
```

After training, inspect the run log and submit the final evaluation. This evaluates the checkpoint
with the best diagnostic `success_once` on 100 fresh episodes under each of seeds 0, 1, and 2. The
three runs all use the same checkpoint; these are environment/evaluation seeds, not three separate
training runs.

```bash
tail -n 40 "$(ls -t logs/pushcube-dp-*.out | head -n 1)"
mkdir -p logs
bash -n scripts/eval_hce_pushcube.sbatch
sbatch scripts/eval_hce_pushcube.sbatch
```

The batch job writes each seed's `metrics.json` plus `evaluation-final/summary.json`, which contains
the mean and sample standard deviation across the three 100-rollout success rates. To target a
specific run instead of the newest completed PushCube run, export its path when submitting:

```bash
RIPL_RUN_DIR="$HOME/ripl_assignment/t-i/runs/RUN_NAME" \
  sbatch --export=ALL,RIPL_RUN_DIR scripts/eval_hce_pushcube.sbatch
```

For the report, preserve the training log, `configs/pushcube_rgb.yaml`, checkpoint iteration,
TensorBoard loss curve, wall time, GPU model, and peak `memory.used` from the corresponding
`logs/gpu-pushcube-JOB_ID.csv`. Report `success_once` per evaluation seed and its mean ± sample
standard deviation from `evaluation-final/summary.json`.

## Recommended Push-T vision workflow

The previous `configs/pusht_rgb_20pct.yaml` experiment is preserved as historical evidence, but
it uses a `pd_ee_delta_pos` dataset and should not be the next run. The controlled recovery
experiment is `configs/pusht_rgb_delta_pose.yaml`: native `pd_ee_delta_pose` source trajectories,
100 demonstrations, the corrected `env_states[t + 1]` RGB replay, 1,024 replay environments,
spatial RGB features, one-step replanning, 150-step episodes, and 50,000 updates. The controller,
demonstration count, training horizon, and budget now match ManiSkill's published state-based
Push-T command; the observation remains RGB as required by this assignment.

The source metadata reports that these Push-T trajectories were collected with 1,024 parallel
environments. ManiSkill warns that Push-T replay is sensitive to parallel simulation differences,
so the preparation script fails if collection and replay parallelism differ unless an explicit
smoke-test override is supplied. It also verifies the native controller, all selected source
success labels, output controller and observation mode, exact trajectory count, and `T + 1` RGB
observations for `T` actions.

For the reportable dataset, queue the replay on HCE from the `t-i` directory:

```bash
mkdir -p logs
bash -n scripts/prepare_hce_pusht_delta_pose.sbatch
sbatch scripts/prepare_hce_pusht_delta_pose.sbatch
```

Monitor it with `squeue -j JOB_ID` and `tail -F logs/pusht-replay-delta-pose-JOB_ID.out`. Do not
start training until the log contains all three confirmations:

```text
Verified 100 successful pd_ee_delta_pose source trajectories; collection environments: 1024; replay environments: 1024
Applied PushT GPU replay alignment: env_states[t] -> env_states[t + 1]
Verified 100 RGB pd_ee_delta_pose trajectories with aligned observation/action lengths
```

If a partial or older native-controller output already exists, move both its HDF5 and JSON files
aside before resubmitting. `ALLOW_REPLAY_ENV_MISMATCH=1` is reserved for a quick Colab/T4
integration test and must not be used for the final dataset.

Run a one-update integration test before spending the full training budget:

```bash
python train_dp.py \
  --config configs/pusht_rgb_delta_pose.yaml \
  --exp-name pusht-rgb-delta-pose-smoke \
  --num-demos 4 \
  --batch-size 8 \
  --total-iters 1 \
  --warmup-steps 1 \
  --eval-freq 1 \
  --save-freq 1 \
  --num-eval-episodes 1 \
  --num-eval-envs 1 \
  --no-capture-video
```

Submit training only after the smoke test succeeds:

```bash
mkdir -p logs
bash -n scripts/train_hce_pusht_delta_pose.sbatch
sbatch scripts/train_hce_pusht_delta_pose.sbatch
```

Use the 20-episode evaluations at iterations 5,000, 10,000, and 15,000 as an early signal. Keep
training while the policy shows increasingly purposeful contact in videos, even if the first
successes have not appeared. If actions still ignore RGB or consistently push the T away from the
goal at iteration 15,000, run `diagnose_dp.py` and inspect the videos before spending the remaining
budget. Do not use training loss by itself as the stop/go criterion.

If a checkpoint reaches at least 20% diagnostic `success_once`, finish the scheduled run and submit
the independent three-seed evaluation:

```bash
mkdir -p logs
bash -n scripts/eval_hce_pusht_delta_pose.sbatch
sbatch scripts/eval_hce_pusht_delta_pose.sbatch
```

The evaluation uses the best diagnostic checkpoint for 100 fresh episodes under each of seeds 0,
1, and 2, then writes `evaluation-final/summary.json`. Treat 20% on the small diagnostic set as a
checkpoint-selection signal; the reportable result is the mean and sample standard deviation over
the final 300 rollouts.

## Attribution

The U-Net, visual encoder, sequence-padding convention, DDPM schedule, EMA, and fair-evaluation
wrappers are adapted from the Apache-2.0 ManiSkill baseline at commit
`62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` (2026-08-01). See the paper and upstream repository
for citation details.
