# TWMR HPC Training (Job-Only)

This branch is set up to run the full training pipeline on HPC via one Slurm job script:

- `jobs/train_full_curriculum.sbatch`

The job runs `scripts/full_curriculum.py`, which includes:

1. Stage A/B/C curriculum
2. Optional Stage-C bridge
3. Dual-obstacle training (`box_dual`)
4. Checkpoint evaluation
5. Deterministic rollout rendering

## Submit the Job

> [!IMPORTANT]
> Before running the job, update `jobs/train_full_curriculum.sbatch` for your setup:
> - Line 11: set `#SBATCH --mail-user=...` to your email address.
> - Line 15: set `cd ...` to the path of your local clone on the cluster.
> - Review other cluster-specific `#SBATCH` settings (partition, GPU type, wall time, memory) as needed.

From repo root:

```bash
sbatch jobs/train_full_curriculum.sbatch
```

## Submit with Custom Full-Curriculum Flags

To pass arguments through to `scripts/full_curriculum.py`, use:

```bash
sbatch jobs/train_full_curriculum.sbatch -- --suffix_prefix=myrun --skip_bridge
```

Another example (continue from a known checkpoint):

```bash
sbatch jobs/train_full_curriculum.sbatch -- --dual_start_checkpoint_path=/path/to/checkpoints
```

Bridge fallback example (continue even if bridge has no promoted stage):

```bash
sbatch jobs/train_full_curriculum.sbatch -- --bridge_no_promotion_action=fallback_stagec
```

## Monitor

```bash
squeue -u "$USER"
tail -f logs/twmr-full-<jobid>.out
tail -f logs/twmr-full-<jobid>.err
```

Cancel if needed:

```bash
scancel <jobid>
```

## Output Artifacts

Primary top-level record:

- `logs/full-curriculum-*.json`

Quick lookup:

```bash
latest_manifest="$(ls -1t logs/full-curriculum-*.json | head -n 1)"
echo "$latest_manifest"
```

The full manifest links to:

- Stage A/B/C manifest
- Bridge manifest (if run)
- Dual training experiment/checkpoints
- Eval JSON and metrics
- Render experiment directory and rollout files

# What Happens During Full-Curriculum Training

The full-curriculum run is a checkpoint-handoff pipeline across multiple PPO trainings:

1. Stage A (`stageA-flat`): trains on `xml_variant=flat` (no obstacle) to learn baseline locomotion/control.
2. Stage B (`stageB-box-easy`): resumes from Stage A and trains on an easier obstacle: `xml_variant=box`, `obstacle_height=0.03`, `obstacle_x_position=0.55`, `obstacle_half_length=0.15`, `success_x_margin=0.08`.
3. Stage C (`stageC-box-target`): resumes from Stage B and trains on the target harder obstacle: `xml_variant=box`, `obstacle_height=0.06`, `obstacle_x_position=0.60`, `obstacle_half_length=0.20`, `success_x_margin=0.10`.
4. Optional bridge curriculum: runs intermediate box settings (default `mode=height` with heights `0.045,0.05,0.055,0.06`), evaluates each bridge stage, and only promotes checkpoints that pass success-rate gates.
5. If bridge runs but produces no promoted stage, full-curriculum now fails fast by default (`--bridge_no_promotion_action=stop`) instead of silently continuing from a failed bridge checkpoint.
6. Dual-obstacle phase (`box_dual`): starts from the selected promoted bridge checkpoint (or Stage C if bridge is skipped), enables a second obstacle, and trains with obstacle randomization ranges for obstacle position and obstacle gap.
7. You can override bridge failure behavior with `--bridge_no_promotion_action=fallback_stagec`, `fallback_stageb`, or `continue_last` (legacy behavior).
8. Post-training evaluation: runs `scripts/eval_twmr_checkpoint.py` on the final dual checkpoint and reports `success_rate`, `mean_return`, and `mean_x_distance`.
9. Deterministic render run: loads the final checkpoint and runs `train_jax_ppo.py` with `--num_timesteps=0` to generate rollout videos without additional learning.

With the current full-curriculum defaults in this repo:
- Stage A/B/C: 1,000,000 timesteps per stage
- Bridge: 1,500,000 timesteps per bridge stage
- Dual-obstacle training: 10,000,000 timesteps

## Potential Shortcoming
The policy is currently trained in sim with a richer proprioceptive state than an IMU-only robot, so there is an observation mismatch at deployment time. For example, there is currently a near-obstacle extension bonus during training, which is something that is not possible for the real robot to detect.
