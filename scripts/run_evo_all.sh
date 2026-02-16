#!/usr/bin/env bash
set -euo pipefail

# Run evo APE and RPE for all methods and save results into results/
# Requires `evo_ape` and `evo_rpe` on PATH (install via pip install evo)

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$BASE_DIR/results"


get_ref() {
    case "$1" in
        spf) echo "$RESULTS/spf_lidar/gps_pose.tum" ;;
        spfpp) echo "$RESULTS/spf_lidar++/0.5/gps_pose.tum" ;;
        ngps) echo "$RESULTS/ngps_only/gps_pose.tum" ;;
        amcl) echo "$RESULTS/amcl/tum1/gps_pose.tum" ;;
        rtab_rgbd) echo "$RESULTS/rtabmap/rgbd/tum1/gps_pose.tum" ;;
        rtab_rgb) echo "$RESULTS/rtabmap/rgb/tum1/gps_pose.tum" ;;
        *) echo "" ;;
    esac
}

get_est() {
    case "$1" in
        spf) echo "$RESULTS/spf_lidar/spf_lidar.tum" ;;
        spfpp) echo "$RESULTS/spf_lidar++/0.5/trajectory_0.5.tum" ;;
        ngps) echo "$RESULTS/ngps_only/noisy_gnss.tum" ;;
        amcl) echo "$RESULTS/amcl/tum1/amcl_pose.tum" ;;
        rtab_rgbd) echo "$RESULTS/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum" ;;
        rtab_rgb) echo "$RESULTS/rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum" ;;
        *) echo "" ;;
    esac
}

METHODS=(spf spfpp ngps amcl rtab_rgbd rtab_rgb)

if [ -d "$BASE_DIR/.venv" ]; then
    # shellcheck disable=SC1090
    source "$BASE_DIR/.venv/bin/activate"
fi

echo "Computing ATE/RTE and row metrics with compute_metrics.py..."
python3 "$BASE_DIR/scripts/compute_metrics.py"


for m in "${METHODS[@]}"; do
    ref=$(get_ref "$m")
    est=$(get_est "$m")
    if [ ! -f "$ref" ] || [ ! -f "$est" ]; then
        echo "Skipping $m: missing files ($ref or $est)" >&2
        continue
    fi

    echo "Running evo APE (raw) for $m..."
    evo_ape tum "$ref" "$est" --save_results "$RESULTS/evo_${m}_ape_raw.json" --no_warnings || echo "evo_ape failed for $m (raw)"

    echo "Running evo APE (Umeyama, no scale) for $m..."
    evo_ape tum "$ref" "$est" -a --save_results "$RESULTS/evo_${m}_ape_umey.json" --no_warnings || echo "evo_ape failed for $m (umey)"

    echo "Running evo APE (Umeyama, with scale) for $m..."
    evo_ape tum "$ref" "$est" -a --correct_scale --save_results "$RESULTS/evo_${m}_ape_umey_scale.json" --no_warnings || echo "evo_ape failed for $m (umey_scale)"

    # run RPE at distances 2m,5m,10m (delta unit meters)
    for d in 2 5 10; do
        echo "Running evo RPE (delta=${d}m) for $m..."
        evo_rpe tum "$ref" "$est" --delta $d --delta_unit m --save_results "$RESULTS/evo_${m}_rpe_${d}m.json" --no_warnings || echo "evo_rpe failed for $m delta $d"
    done
done

echo "Finished evo runs. Output JSONs are in $RESULTS"

echo "Aggregating evo results into CSV + PDF..."
python3 "$BASE_DIR/scripts/aggregate_evo_results.py"

echo "Aggregation complete. See $RESULTS/evo_aggregated_metrics.csv and $RESULTS/evo_aggregated_metrics.pdf"
