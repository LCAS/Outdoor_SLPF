#!/usr/bin/env bash
set -euo pipefail

# Runner to reproduce metrics, plots and merged summaries.
# Usage: ./scripts/run_all_experiments.sh

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$BASE_DIR/.venv"

if [ -d "$VENV" ]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
else
    echo "Warning: virtualenv not found at $VENV. Ensure dependencies are installed and 'evo' is on PATH." >&2
fi

echo "Running SPF++ localization pipeline..."
python3 "$BASE_DIR/scripts/spf_lidar.py" --gps-weight 0.5

echo "Running compute_metrics.py..."
python3 "$BASE_DIR/scripts/compute_metrics.py"

echo "Running plot_trajectories.py..."
python3 "$BASE_DIR/scripts/plot_trajectories.py"

echo "Merging EVO and RTE results..."
python3 "$BASE_DIR/scripts/merge_evo_and_rte.py"

echo "Done. Outputs are in $BASE_DIR/results"

# Optional: run evo APE evaluations for each method (requires 'evo_ape' on PATH).
# Uncomment and adapt paths below if you want to re-run evo evaluations from raw TUM files.
# echo "Running evo APE (example for Noisy GPS)..."
# evo_ape tum "$BASE_DIR/results/ngps_only/gps_pose.tum" "$BASE_DIR/results/ngps_only/noisy_gnss.tum" --save_results "$BASE_DIR/results/evo_ngps_raw.json" --rmse
