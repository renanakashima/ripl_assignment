# Visual Diffusion Policy for ManiSkill `PushT-v1`

This repository implements imitation learning with an RGB-conditioned Diffusion Policy for
ManiSkill's `PushT-v1` task. It is organized for editing in Cursor and training in Google Colab;
no local NVIDIA server is required.

The policy follows the core formulation from the
[Diffusion Policy paper](https://arxiv.org/abs/2303.04137): it encodes the two latest visual and
proprioceptive observations, conditions a 1-D U-Net with those features through FiLM, denoises a
16-step action sequence with a cosine-schedule DDPM, executes one action, and replans. “Visual”
here means RGB is the main scene input while low-dimensional robot state is retained, as in a
standard visuomotor policy.

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
│   ├── pusht_rgb.yaml           # full experiment
│   └── smoke.yaml               # ten-update integration test
├── scripts/
│   ├── setup_colab.sh           # Python dependencies + headless Vulkan
│   ├── prepare_pusht_demos.sh   # download and replay RGB demonstrations
│   ├── train_colab.sh
│   └── verify_setup.py
├── notebooks/
│   └── pusht_diffusion_colab.ipynb
├── ripl/                        # model, dataset, environments, and evaluation
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
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO/t-i
!bash scripts/setup_colab.sh
```

Prepare 100 RGB demonstrations:

```python
!NUM_DEMOS=100 REPLAY_ENVS=64 bash scripts/prepare_pusht_demos.sh
```

This first downloads ManiSkill's compressed PushT demonstrations and then replays the first 100
with RGB observations. Replay is a required preprocessing step—the downloaded files omit images.
ManiSkill's upstream script uses 256 parallel GPU environments for PushT. The Colab default here
is 64 to fit a T4; if it succeeds and memory permits, `REPLAY_ENVS=256` most closely matches the
upstream recipe. PushT is unusually sensitive to tiny simulator differences, so do not change the
simulation backend between replay and evaluation.

Run the short end-to-end test before committing to a full job:

```python
!python train_dp.py --config configs/smoke.yaml
```

Then train the full experiment:

```python
!python train_dp.py --config configs/pusht_rgb.yaml
```

The full config requests 50,000 gradient updates. Runtime depends on the assigned Colab GPU and
evaluation frequency, and may exceed one free-session window. For durable checkpoints, mount
Google Drive and override the output directory:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
!python train_dp.py --config configs/pusht_rgb.yaml \
  --output-dir /content/drive/MyDrive/ripl-pusht-runs
```

Resume after a disconnect by setting a larger or unchanged `total_iters` and loading a periodic
checkpoint:

```python
!python train_dp.py --config configs/pusht_rgb.yaml \
  --resume /content/drive/MyDrive/ripl-pusht-runs/RUN_NAME/checkpoints/step_5000.pt \
  --total-iters 50000
```

To evaluate the exponential-moving-average policy and save videos/metrics:

```python
!python eval_dp.py \
  --checkpoint /content/drive/MyDrive/ripl-pusht-runs/RUN_NAME/checkpoints/final.pt \
  --num-eval-episodes 20 --num-eval-envs 10
```

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
- The official baseline assumes 128×128 camera images; this implementation uses adaptive pooling
  but the prepared ManiSkill data still supplies the official camera resolution.
- To log to Weights & Biases, run `pip install -e '.[tracking]'` and set `track: true`.

## Expected-results caveat

[`PushT-v1` issue #882](https://github.com/mani-skill/ManiSkill/issues/882) reports roughly
0.15–0.20 success for one state-based reproduction and asks why a paper reports 0.4–0.5; the issue
is closed without a maintainer answer. It does not establish an expected RGB result. Treat seeds,
demo replay parallelism, dataset provenance, simulator backend, and checkpoint selection as part of
the experiment. Report your exact configuration and mean success over multiple seeds rather than
claiming that one run reproduces a separate paper.

## Attribution

The U-Net, visual encoder, sequence-padding convention, DDPM schedule, EMA, and fair-evaluation
wrappers are adapted from the Apache-2.0 ManiSkill baseline at commit
`62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` (2026-08-01). See the paper and upstream repository
for citation details.
