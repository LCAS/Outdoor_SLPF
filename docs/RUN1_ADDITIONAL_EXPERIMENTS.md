# Run1 Additional Experiments (Option A + Option B)

This protocol adds two robustness studies on top of the run1 SLPF setup.

## Option A — Detection degradation

- Randomly drop semantic detections before LiDAR semantic association.
- Default sweep: `20%` and `40%`.
- Report:
  - `ape_align_rmse`
  - `row_correct_fraction`

## Option B — Landmark removal from map

- Remove `30%` and `50%` of landmarks from one map section.
- The section is a contiguous band along the map's dominant axis
  (default quantiles: `35%` to `65%` of axis extent).
- Report recovery behavior after exiting that section:
  - section APE (`section_ape_mean`, `section_ape_peak`)
  - post-section APE (`post_ape_mean`)
  - recovery time/distance (`recovery_frames`, `recovery_distance_m`)
  - recovery success fraction

## Command (full run)

```bash
source .venv/bin/activate
python scripts/run_run1_robustness_experiments.py \
  --seeds 11,22,33 \
  --drop-rates 0.2,0.4 \
  --remove-rates 0.3,0.5 \
  --require-cuda
```

## Command (reuse existing run1 baseline trajectories)

If you already have a baseline run1 directory (for example
`results/iros_rh1/20260225_130359_multiseed_slpf_ngps_run1_recomputed`),
reuse those SLPF trajectories:

```bash
source .venv/bin/activate
python scripts/run_run1_robustness_experiments.py \
  --seeds 11,22,33 \
  --drop-rates 0.2,0.4 \
  --remove-rates 0.3,0.5 \
  --reuse-baseline-dir results/iros_rh1/20260225_130359_multiseed_slpf_ngps_run1_recomputed \
  --require-cuda
```

## Key outputs

In `results/iros_rh1_robustness/<timestamp>_run1_robustness_<cpu|gpu>/`:

- `run1_robustness_per_seed.csv`: all runs (baseline + Option A + Option B).
- `run1_robustness_aggregate.csv`: grouped mean/std + deltas vs baseline.
- `option_a_detection_drop_summary.csv`: APE + row correctness table for Option A.
- `option_b_recovery_per_seed.csv`: per-seed recovery metrics for Option B.
- `option_b_recovery_aggregate.csv`: recovery aggregates.
- `option_b_recovery_summary.png`: compact recovery figure.
- `run_protocol.json`: exact commands and map-variant metadata for reproducibility.

