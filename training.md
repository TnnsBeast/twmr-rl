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
