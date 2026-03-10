# Runtime Experiment Protocol (Reviewer Reporting)

This protocol reports:
- `Full pipeline Hz` (end-to-end processed-frame throughput).
- `Main component runtime` (mean/p95 ms and time share per stage).
- Hardware snapshot for the machine used in the experiment.

## 1) Run repeated trials

Use the dedicated runner:

```bash
source .venv/bin/activate
python3 scripts/run_runtime_profile_experiment.py \
  --trials 3 \
  --measured-frames 200 \
  --warmup-frames 20 \
  --frame-stride 4 \
  --data-path data/2025/rh_run1 \
  --require-cuda
```

Notes:
- `--warmup-frames` excludes startup/JIT effects from metrics.
- `--measured-frames` controls statistical confidence.
- `--no-visualization` is automatically enforced by the runner for fair online runtime numbers.
- If GPU is unavailable, replace `--require-cuda` with `--allow-cpu`.

## 2) Output artifacts

The runner creates a timestamped folder under `results/runtime_profile/` containing:
- `runtime_profile_trials.csv`: per-trial full pipeline Hz and per-component stats.
- `runtime_profile_components.csv`: component means/std aggregated across trials.
- `runtime_profile_experiment_summary.json`: experiment config + hardware + aggregate pipeline Hz.
- Per-trial folders with:
  - `runtime_profile_frames.csv` (per-frame timings after warmup).
  - `runtime_profile_summary.json` (single-trial summary used by the aggregator).

## 3) What to report in paper/rebuttal

Report at minimum:
1. Full pipeline throughput:
   - `processed_frame_hz_mean ± std`
2. Effective input-rate throughput:
   - `effective_input_hz_with_stride_mean ± std`
3. Main component timing (from `runtime_profile_components.csv`):
   - `semantic_inference_sec`
   - `measurement_update_sec`
   - `lidar_association_sec`
   - `motion_update_sec`
   - `resample_sec`
4. Hardware:
   - CPU model, RAM, GPU model/driver, Torch/CUDA versions (in summary JSON).

## 4) Stage definitions

Per-frame component times in `spf_lidar.py`:
- `io_sec`: RGB/depth/LiDAR file load and decode.
- `semantic_inference_sec`: YOLO inference + mask post-processing.
- `lidar_association_sec`: semantic-circle assignment of LiDAR beams.
- `motion_update_sec`: odometry delta and particle motion update.
- `measurement_update_sec`: particle likelihood update (`measurement_likelihood_gpu`).
- `pose_post_sec`: pose extraction/smoothing + trajectory bookkeeping.
- `resample_sec`: adaptive resampling.
- `visualization_sec`: debug rendering (expected near zero in protocol).
- `stats_write_sec`: per-frame stats CSV append.
- `other_sec`: frame-time remainder not covered by explicit stages.
