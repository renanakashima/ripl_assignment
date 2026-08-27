# Experiment scripts

The scripts are grouped below by purpose. Run commands from the `t-i` directory so relative
configuration and module paths resolve consistently.

## Data preparation

- `prepare_pushcube_demos.sh`: download and replay PushCube demonstrations as RGB observations.
- `prepare_pusht_demos.sh`: reproduce the original PushT RGB replay.
- `prepare_pusht_20pct_demos.sh`: prepare the historical task-tuned PushT variant.
- `prepare_pusht_delta_pose_demos.sh`: prepare the current native-controller PushT replay.
- `replay_trajectory_aligned.py` and `validate_pusht_replay.py`: apply and verify the PushT
  observation/action alignment correction.

## Execution environments

- `setup_colab.sh` and `train_colab.sh`: Colab setup and the original PushT launcher.
- `*_hce_*.sbatch` and `train_hce_*.sbatch`: HCE replay, training, and evaluation jobs.
- `setup_runpod.sh`: provision a Runpod GPU Pod.
- `runpod_pushcube.sh` and `runpod_pusht.sh`: stable task-specific Runpod entry points. They share
  the private `_runpod_experiment.sh` implementation to avoid duplicating orchestration logic.

## Verification and evaluation

- `verify_setup.py`: check imports, CUDA, and ManiSkill environment registration.
- `aggregate_eval.py`: combine fixed-checkpoint evaluation metrics across seeds.

Shell launchers fail fast on errors. You can validate them without starting an experiment:

```bash
bash -n scripts/*.sh
for file in scripts/*.sbatch; do bash -n "$file"; done
```
