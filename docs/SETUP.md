# Setup Guide

This repository supports two install modes:

- evaluation-only, for reproducing metrics and figures from existing trajectory files
- full SPF pipeline, for rerunning the semantic particle filter itself

## 1. Evaluation-only environment

Use this when you only need the paper metrics, tables, and plots.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This covers the scripts used for:

- `scripts/compute_metrics.py`
- `scripts/run_evo_all.sh`
- `scripts/aggregate_evo_results.py`
- `scripts/plot_trajectories.py`
- `scripts/plot_trajectories_2x4_experiment_comparison.py`

You also need the `evo` CLI available in the same environment.

## 2. Full SPF pipeline environment

The main localisation pipeline in `scripts/spf_lidar.py` has heavier dependencies than `requirements.txt` alone. In addition to the evaluation packages, install:

- PyTorch, matched to your CUDA setup if you want GPU execution
- `opencv-python`
- `ultralytics`
- `tqdm`
- `geopandas`

Suggested order:

```bash
pip install torch
pip install opencv-python ultralytics tqdm geopandas
```

If you need CUDA acceleration, replace the first command with the PyTorch install command that matches your CUDA version.

The script loads the trained segmentation weights from `models/yolo.pt`.

## 3. Optional experiment extras

Some optional experiment runners require additional packages:

- `optuna` for `scripts/tune_spfpp_optuna.py`

These are not required for the recommended public release workflow if you are consuming baseline trajectories from existing CSV or TUM files.

## 4. Expected data layout

For SPF execution, the default dataset layout is:

```text
<data-root>/
  data.csv
  rgb/
  depth/
  lidar/
```

The default map file is `data/riseholme_poles_trunk.geojson`.

## 5. Quick validation

Check that the main entry points resolve correctly:

```bash
python3 scripts/spf_lidar.py --help
python3 scripts/compute_metrics.py --help
python3 scripts/run_runtime_profile_experiment.py --help
```

## 6. Release-scope note

The public release is intentionally focused on the SPF pipeline and the evaluation stack. Native AMCL and ORB-SLAM3 runtime helpers are not required when the repository already ships their exported trajectories or CSV summaries.
