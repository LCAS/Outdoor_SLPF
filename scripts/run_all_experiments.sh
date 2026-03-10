#!/usr/bin/env bash
set -euo pipefail

# Runner to reproduce metrics, plots and merged summaries.
# Usage: ./scripts/run_all_experiments.sh

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$BASE_DIR/.venv"
RESULTS_DIR="${RESULTS_DIR:-$BASE_DIR/results}"
SPF_TRAJ_SRC="${SPF_TRAJ_SRC:-$BASE_DIR/results/spf_lidar/spf_lidar.tum}"
SPF_GT_SRC="${SPF_GT_SRC:-$BASE_DIR/results/spf_lidar/gps_pose.tum}"
SPFPP_TRAJ_SRC="${SPFPP_TRAJ_SRC:-$BASE_DIR/results/spf_lidar++/0.5/trajectory_0.5.tum}"
SPFPP_GT_SRC="${SPFPP_GT_SRC:-$BASE_DIR/results/spf_lidar++/0.5/gps_pose.tum}"
NOISY_GPS_TUM_SRC="${NOISY_GPS_TUM_SRC:-}"
NOISY_GPS_GT_SRC="${NOISY_GPS_GT_SRC:-}"

prepare_input_trajectories() {
    local src="$BASE_DIR/results"
    local dst="$RESULTS_DIR"
    local files=(
        "spf_lidar/spf_lidar.tum"
        "spf_lidar/gps_pose.tum"
        "spf_lidar++/0.5/trajectory_0.5.tum"
        "spf_lidar++/0.5/gps_pose.tum"
        "amcl/tum1/amcl_pose.tum"
        "amcl/tum1/gps_pose.tum"
        "rtabmap/rgbd_run1_3runs/run1/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum"
        "rtabmap/rgbd_run1_3runs/run1/rtabmap/rgbd/tum1/gps_pose.tum"
        "rtabmap/rgb_run1_3runs/run1/rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum"
        "rtabmap/rgb_run1_3runs/run1/rtabmap/rgb/tum1/gps_pose.tum"
        "rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum"
        "rtabmap/rgbd/tum1/gps_pose.tum"
        "rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum"
        "rtabmap/rgb/tum1/gps_pose.tum"
    )

    mkdir -p "$dst"

    # Allow explicitly pinning SPF/SPF++ inputs (e.g., from A/B validation outputs).
    if [ -f "$SPF_TRAJ_SRC" ] && [ -f "$SPF_GT_SRC" ]; then
        mkdir -p "$dst/spf_lidar"
        cp -f "$SPF_TRAJ_SRC" "$dst/spf_lidar/spf_lidar.tum"
        cp -f "$SPF_GT_SRC" "$dst/spf_lidar/gps_pose.tum"
    else
        echo "Warning: missing SPF override inputs ($SPF_TRAJ_SRC or $SPF_GT_SRC)" >&2
    fi

    if [ -f "$SPFPP_TRAJ_SRC" ] && [ -f "$SPFPP_GT_SRC" ]; then
        mkdir -p "$dst/spf_lidar++/0.5"
        cp -f "$SPFPP_TRAJ_SRC" "$dst/spf_lidar++/0.5/trajectory_0.5.tum"
        cp -f "$SPFPP_GT_SRC" "$dst/spf_lidar++/0.5/gps_pose.tum"
    else
        echo "Warning: missing SPF++ override inputs ($SPFPP_TRAJ_SRC or $SPFPP_GT_SRC)" >&2
    fi

    # Prefer canonical noisy GPS location under results/noisy_gps.
    local ngps_traj_src="$NOISY_GPS_TUM_SRC"
    local ngps_gt_src="$NOISY_GPS_GT_SRC"
    if [ -z "$ngps_traj_src" ]; then
        if [ -f "$src/noisy_gps/noisy_gps_seed_11.tum" ]; then
            ngps_traj_src="$src/noisy_gps/noisy_gps_seed_11.tum"
        elif [ -f "$src/noisy_gps/noisy_gnss.tum" ]; then
            ngps_traj_src="$src/noisy_gps/noisy_gnss.tum"
        elif [ -f "$src/ngps_only/noisy_gnss.tum" ]; then
            ngps_traj_src="$src/ngps_only/noisy_gnss.tum"
        elif [ -f "$src/ngps_only-deprecated/noisy_gnss.tum" ]; then
            ngps_traj_src="$src/ngps_only-deprecated/noisy_gnss.tum"
        fi
    fi
    if [ -z "$ngps_gt_src" ]; then
        if [ -f "$src/noisy_gps/gps_pose.tum" ]; then
            ngps_gt_src="$src/noisy_gps/gps_pose.tum"
        elif [ -f "$src/ngps_only/gps_pose.tum" ]; then
            ngps_gt_src="$src/ngps_only/gps_pose.tum"
        elif [ -f "$src/ngps_only-deprecated/gps_pose.tum" ]; then
            ngps_gt_src="$src/ngps_only-deprecated/gps_pose.tum"
        fi
    fi
    if [ -n "$ngps_traj_src" ] && [ -n "$ngps_gt_src" ] && [ -f "$ngps_traj_src" ] && [ -f "$ngps_gt_src" ]; then
        mkdir -p "$dst/noisy_gps"
        cp -f "$ngps_traj_src" "$dst/noisy_gps/noisy_gps_seed_11.tum"
        cp -f "$ngps_gt_src" "$dst/noisy_gps/gps_pose.tum"
    else
        echo "Warning: missing noisy GPS inputs (traj=$ngps_traj_src gt=$ngps_gt_src)" >&2
    fi

    for rel in "${files[@]}"; do
        # SPF/SPF++ are copied from explicit sources above.
        if [ "$rel" = "spf_lidar/spf_lidar.tum" ] || \
           [ "$rel" = "spf_lidar/gps_pose.tum" ] || \
           [ "$rel" = "spf_lidar++/0.5/trajectory_0.5.tum" ] || \
           [ "$rel" = "spf_lidar++/0.5/gps_pose.tum" ]; then
            continue
        fi
        if [ -f "$src/$rel" ]; then
            mkdir -p "$dst/$(dirname "$rel")"
            cp -f "$src/$rel" "$dst/$rel"
        else
            echo "Warning: missing input trajectory $src/$rel" >&2
        fi
    done
}

if [ -d "$VENV" ]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
else
    echo "Warning: virtualenv not found at $VENV. Ensure dependencies are installed and 'evo' is on PATH." >&2
fi

echo "Preparing trajectory inputs in $RESULTS_DIR..."
prepare_input_trajectories

echo "Running compute_metrics.py..."
RESULTS_DIR="$RESULTS_DIR" python3 "$BASE_DIR/scripts/compute_metrics.py"

echo "Running plot_trajectories.py..."
RESULTS_DIR="$RESULTS_DIR" python3 "$BASE_DIR/scripts/plot_trajectories.py"

echo "Running evo APE/RPE evaluations..."
RESULTS_DIR="$RESULTS_DIR" SKIP_COMPUTE_METRICS=1 bash "$BASE_DIR/scripts/run_evo_all.sh"

echo "Merging EVO and RTE results..."
RESULTS_DIR="$RESULTS_DIR" python3 "$BASE_DIR/scripts/merge_evo_and_rte.py"

echo "Done. Outputs are in $RESULTS_DIR"

# Optional: run evo APE evaluations for each method (requires 'evo_ape' on PATH).
# Uncomment and adapt paths below if you want to re-run evo evaluations from raw TUM files.
# echo "Running evo APE (example for Noisy GPS)..."
# evo_ape tum "$BASE_DIR/results/ngps_only/gps_pose.tum" "$BASE_DIR/results/ngps_only/noisy_gnss.tum" --save_results "$BASE_DIR/results/evo_ngps_raw.json" --rmse
