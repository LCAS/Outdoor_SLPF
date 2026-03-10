# SPF++ Hyperparameter Tuning Plan (Grid + Bayesian Optuna)

## Goal
Improve SPF++ on three axes simultaneously:
- localization consistency (aligned APE/RPE),
- row adherence (cross-track and row assignment),
- trajectory smoothness (jerk and heading acceleration).

## Strategy
Use a two-stage search with Optuna:
1. **Stage 1: Coarse grid search**
   - broad exploration of key knobs (`gps_weight`, `miss_penalty`, `wrong_hit_penalty`, `corridor_weight`)
   - objective: find stable regions and good initialization points.
2. **Stage 2: Bayesian refinement (TPE)**
   - warm-start from top grid trials (`enqueue_trial`)
   - refine around promising regions with continuous sampling.

This keeps the total number of expensive SPF++ runs lower than pure grid while being less initialization-sensitive than pure Bayesian search.

## Objective
Each trial runs SPF++ and computes aligned metrics from the generated trajectory:
- `ape_align_rmse`
- `rpe_2m_align_rmse`, `rpe_5m_align_rmse`, `rpe_10m_align_rmse`
- `cross_track_mean`, `row_correct_fraction`, `row_switch_events`
- `jerk_rms`, `heading_accel_rms`

Single-score minimization (lower is better):
- weighted sum favoring row adherence and smoothness while constraining localization drift.

## Tuned Parameters
### Grid (coarse)
- `gps_weight`
- `miss_penalty`
- `wrong_hit_penalty`
- `corridor_weight`

### Bayesian (refinement)
- `miss_penalty`
- `wrong_hit_penalty`
- `gps_weight`
- `semantic_sigma`
- `gps_sigma`
- `corridor_weight`
- `background_class_weight`
- `pose_smooth_alpha_pos`
- `pose_smooth_alpha_theta`
- `odom_yaw_filter_alpha`

## Runtime controls for efficiency
`scripts/spf_lidar.py` supports:
- `--frame-stride`
- `--max-frames`
- `--no-visualization`

Recommended workflow:
- tune on shortened runs first (`max_frames`, larger stride),
- re-run top configs on full runs before final selection.

## Command template
```bash
MPLCONFIGDIR=.tmp_mpl \
.venv/bin/python scripts/tune_spfpp_optuna.py \
  --search-seeds 22 \
  --grid-max-trials 18 \
  --bayes-trials 24 \
  --topk-enqueue 6 \
  --frame-stride 4 \
  --max-frames 180 \
  --no-visualization \
  --require-cuda
```

## Outputs
Per tuning run (timestamped directory under `results/spf_lidar++/tuning_optuna/`):
- `tuning_plan.json`
- `grid_trials.csv`
- `bayes_trials.csv`
- `stage_summary.json`
- `best_config.json`
- per-trial artifacts/logs under `grid/` and `bayes/`.
