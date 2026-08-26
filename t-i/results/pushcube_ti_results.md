# T-I Visual Diffusion Policy Results: PushCube-v1

## Experiment summary

An RGB-conditioned Diffusion Policy was trained with imitation learning on ManiSkill's
`PushCube-v1` task. The policy uses RGB observations together with robot proprioception and
predicts action sequences for the `pd_ee_delta_pos` controller. The visual encoder retains its
spatial feature map rather than globally pooling it.

- HCE run: `pushcube-rgb-diffusion-spatial__seed1__20260825-150140`
- Training job: `6887`
- Final evaluation job: `6894`
- Training GPU: NVIDIA L40S
- Python: 3.12.14
- Simulation backend: `physx_cpu`

## Dataset

The dataset was produced by replaying ManiSkill motion-planning demonstrations with RGB
observations.

| Item | Value |
|---|---:|
| Retained trajectories | 100 |
| Transitions | 6,870 |
| Training windows | 6,770 |
| Observation mode | RGB |
| Controller | `pd_ee_delta_pos` |

## Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Training iterations | 30,000 |
| Batch size | 256 |
| Learning rate | 0.0001 |
| Warmup steps | 500 |
| Observation horizon | 2 |
| Action horizon | 8 |
| Prediction horizon | 16 |
| Diffusion steps | 100 |
| Diffusion embedding dimension | 64 |
| U-Net dimensions | `[64, 128, 256]` |
| Visual feature dimension | 256 |
| Spatial feature map retained | Yes |
| Maximum episode length | 100 |
| Optimizer | AdamW |
| Learning-rate schedule | Cosine with 500-step warmup |
| EMA power | 0.75 |

The full configuration is preserved in `configs/pushcube_rgb.yaml` and in the run's
`config.json`.

## Training results

| Iteration | Diagnostic `success_once` | Diagnostic `success_at_end` |
|---:|---:|---:|
| 5,000 | 80% | 80% |
| 10,000 | 90% | 90% |
| 15,000 | 90% | 90% |
| 20,000 | 95% | 95% |
| 25,000 | 95% | 95% |
| 30,000 | 95% | 95% |

- Final displayed minibatch diffusion loss: **0.0005**
- Training wall time: **1,863 seconds (31 minutes 3 seconds)**
- Average displayed training throughput: **16.37 iterations/second**, including evaluation
  overhead in the completed progress display
- Selected checkpoint: **`best_success_once.pt`, iteration 20,000**

The selected checkpoint is from iteration 20,000 because it was the first checkpoint to reach the
maximum diagnostic success of 95%. Later checkpoints tied rather than exceeded that value. The
diagnostic evaluations used 20 episodes and were used only for checkpoint selection, not as the
final baseline.

## GPU and VRAM usage

GPU statistics were sampled every 10 seconds with `nvidia-smi` and saved in
`logs/gpu-pushcube-6887.csv`.

| Measurement | Value |
|---|---:|
| GPU | NVIDIA L40S |
| Available VRAM | 46,068 MiB (44.99 GiB) |
| Peak sampled VRAM | **13,736 MiB (13.41 GiB)** |
| Fraction of available VRAM | **29.8%** |
| Peak sampled GPU utilization | **83%** |

Because measurements were sampled at 10-second intervals, 13,736 MiB is the peak sampled value;
an instantaneous peak between samples could have been slightly higher.

## Final three-seed evaluation

The same EMA checkpoint from iteration 20,000 was evaluated for 100 episodes under each of three
environment seeds. Videos were disabled for this large evaluation. These are evaluation seeds,
not three independently trained policies.

| Evaluation seed | Episodes | `success_once` | `success_at_end` |
|---:|---:|---:|---:|
| 0 | 100 | 78% | 78% |
| 1 | 100 | 81% | 81% |
| 2 | 100 | 87% | 87% |
| **Mean** | **300 total** | **82.0%** | **82.0%** |
| **Sample standard deviation** |  | **4.58 percentage points** | **4.58 percentage points** |

The pooled result was **246 successful episodes out of 300**. Mean episode return was 30.617 with
an across-seed sample standard deviation of 0.451. `success_once` and `success_at_end` were equal
for every seed, indicating that episodes counted as successful remained within the goal region at
the final timestep rather than succeeding only transiently.

The reportable baseline is therefore:

> The RGB visual Diffusion Policy achieved an average `PushCube-v1` success rate of **82.0% ± 4.6
> percentage points** across three evaluation seeds, using 100 evaluation rollouts per seed.

ManiSkill defines PushCube success as the cube's XY center being within the target's 0.10 m radius
while the cube remains on the table. It does not require the cube to cover the target's central
bullseye. See the official
[`PushCube-v1` implementation](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py).

## Result artifacts on HCE

```text
runs/pushcube-rgb-diffusion-spatial__seed1__20260825-150140/
├── config.json
├── checkpoints/best_success_once.pt
├── evaluation-final/
│   ├── seed-0/metrics.json
│   ├── seed-1/metrics.json
│   ├── seed-2/metrics.json
│   └── summary.json
└── videos/

logs/pushcube-dp-6887.out
logs/gpu-pushcube-6887.csv
logs/pushcube-eval-6894.out
```

Fields beginning with an underscore, such as `_success_once`, are Gymnasium vector-environment
validity masks and are not experimental metrics.
