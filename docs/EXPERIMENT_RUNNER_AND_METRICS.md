# Experiment runners and metrics

This document explains the scripts used to run the experiments and compute metrics in this repository, and gives precise definitions for every metric produced by the pipeline.

**Scope**
- **Repository scripts** referenced: `scripts/compute_metrics.py`, `scripts/run_evo_all.sh`, `scripts/aggregate_evo_results.py`, `scripts/plot_trajectories.py`.
- **Baseline policy**: AMCL, RTAB-Map, and ORB-SLAM3 are evaluated from exported trajectory files already present in the repository layout. The public release does not require shipping their native runtime stacks.

**How the runner works (high level)**
- **Prepare environment**: activate the project's Python virtualenv (`.venv`) so required Python packages and the `evo` CLI are available.
- **Compute geometric/row metrics**: `scripts/compute_metrics.py` reads TUM trajectory files (methods + GPS ground-truth), aligns trajectories when requested (first-pose 2D yaw+translation or Umeyama), computes ATE-style statistics and row/cross-track metrics, and writes `results/trajectory_metrics.csv` and a PDF summary.
- **Run evo evaluations**: `scripts/run_evo_all.sh` executes the evo CLI tools (`evo_ape`, `evo_rpe`) for each method and for several alignment variants:
  - APE modes: raw (no alignment), Umeyama (Sim(3) without scale correction), Umeyama with scale (Sim(3) allowing scale).
  - RPE evaluations for travel-distance deltas (e.g., 2m, 5m, 10m).
  The runner saves evo result archives (JSON) into `results/` such as `results/evo_<method>_ape_umey.json` and `results/evo_<method>_rpe_2m.json`.
- **Aggregate results**: `scripts/aggregate_evo_results.py` reads the saved evo archives, extracts key statistics (RMSE values for APE and RPE variants), merges the row/cross-track values from `results/trajectory_metrics.csv`, and writes `results/evo_aggregated_metrics.csv` and `results/evo_aggregated_metrics.pdf` (a table plus the `results/trajectory_comparison.png` image appended).
- **Optional plots**: `scripts/plot_trajectories.py` produces color-coded trajectory visualizations (`results/trajectory_comparison.png`) where per-point/segment Euclidean errors vs the GPS ground truth are used for the color map.

**Typical command sequence (copyable)**
```bash
source .venv/bin/activate
python3 scripts/compute_metrics.py
bash scripts/run_evo_all.sh
python3 scripts/aggregate_evo_results.py
```

**Alignment options used by the pipeline**
- **First-pose 2D yaw+translation** — a lightweight 2D alignment placing the first pose of the evaluated trajectory onto the first pose of the reference by applying a yaw rotation (around vertical axis) plus 2D translation. This keeps scale and local geometry intact and is useful when the trajectories share orientation and scale but differ by an initial heading/offset.
- **Umeyama (Sim(3))** — a full similarity transform estimation in 3D using the Umeyama algorithm. Two variants are used:
  - **Umeyama (no scale)** — similarity transform constrained to rotation + translation (no scale change), equivalent to SE(3) rigid-body alignment.
  - **Umeyama (with scale)** — full Sim(3) allowing a uniform scale factor in addition to rotation and translation. This is useful when evaluated and reference trajectories have proportional scaling differences.

**Files produced**
- `results/trajectory_metrics.csv`: per-method row metrics and geometric statistics computed by `compute_metrics.py`.
- `results/trajectory_comparison.png`: color-coded overlay of trajectories vs GPS ground truth.
- `results/evo_<method>_ape_*.json` / `results/evo_<method>_rpe_*.json`: saved evo result archives from `evo_ape`/`evo_rpe` runs.
- `results/evo_aggregated_metrics.csv` and `results/evo_aggregated_metrics.pdf`: aggregated table of APE & RPE RMSE values and row metrics.

**Metric Definitions (explicit)**

- **Absolute Trajectory Error (ATE) / APE (evo APE)**
  - APE is the pointwise Euclidean distance between corresponding poses of the estimated trajectory and the reference (after the chosen alignment/transform has been applied). When reported as RMSE, it is:
  $$\text{APE}_{\mathrm{RMSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|p_i^{est} - p_i^{ref}\|^2 }$$
  where $p_i^{est}$ and $p_i^{ref}$ are the 3D positions for pose index $i$, and $N$ is the number of matched poses. The pipeline reports three APE variants for each method: raw (no alignment), Umeyama (no scale), and Umeyama with scale.

- **Relative Pose Error (RPE)**
  - RPE measures the error in relative motion over a fixed travel-distance or fixed time interval. For the distance-based variant used in this pipeline (delta = 2m/5m/10m), the relative pose between pose $i$ and pose $j$ (where the cumulative traveled distance between them ≈ delta) is compared between estimate and reference. The positional RPE RMSE is computed as:
  $$\text{RPE}_{\mathrm{RMSE}} = \sqrt{\frac{1}{M} \sum_{k=1}^M \| (p_{k+\delta}^{est} - p_k^{est}) - (p_{k+\delta}^{ref} - p_k^{ref}) \|^2 }$$
  where $M$ is the number of valid pairs used for the delta-based computation. The pipeline reports RPE RMSE for delta = 2m, 5m, and 10m.

- **RMSE vs mean/median**
  - RMSE (root mean square error) is used to summarize APE and RPE distributions because it amplifies large errors and is commonly reported in SLAM/trajectory literature. The pipeline also computes other summary stats where useful (mean, median, max) for row/cross-track metrics.

- **Cross-track distance to row centerline**
  - For each trajectory point the script computes the shortest Euclidean distance in the horizontal plane from the robot position to the nearest vineyard row centerline (extracted from `data/riseholme_poles_trunk.geojson`). The pipeline reports the cross-track mean, median, and max (in meters):
  $$\text{cross\_track\_mean} = \frac{1}{N}\sum_{i=1}^N d_i$$
  where $d_i$ is the horizontal-plane distance from pose $i$ to the matched row centerline.

- **Row-correct fraction**
  - The fraction of poses for which the estimated trajectory is assigned to the same vineyard row as the ground-truth pose. Computed as:
  $$\text{row\_correct\_fraction} = \frac{\text{# poses with same assigned row}}{\text{total # poses}}$$

- **Row-switch events**
  - The number of times the evaluated trajectory switches its assigned row while the ground truth remains inside the same row (an indicator of lost row-following behavior). This is counted as discrete transitions in the per-sample row assignment sequence. The script reports a single integer count per trajectory.

- **Per-point Euclidean error (used for plotting)**
  - For color-coded trajectory plots, the script computes the instantaneous Euclidean distance in the horizontal plane (or full 3D Euclidean distance per current settings) between each evaluated pose and the interpolated GPS ground truth. These values are used to color segments/points on `results/trajectory_comparison.png`.

**Notes, caveats and interpretation guidance**
- APE raw vs Umeyama: raw APE reflects global offsets and scale mismatch; Umeyama-aligned APE (especially with scale) is the appropriate comparison when trajectories differ by an affine similarity transform (e.g., different scale or coordinate magnitude). Choose the variant that matches your desired invariance.
- RPE is a local-motion error metric; low RPE with high APE can indicate accurate local odometry but a global drift or offset.
- Cross-track metrics focus on how closely the robot follows vineyard rows and are orthogonal to APE/RPE — they answer a different operational question (row-following performance).
- All numeric outputs appear in `results/evo_aggregated_metrics.csv`. The PDF `results/evo_aggregated_metrics.pdf` is a visual table summary (and includes the trajectory comparison image appended).
