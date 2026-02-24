#!/usr/bin/env bash
set -euo pipefail

# Run evo APE and RPE for all methods and save results into results/
# Requires `evo_ape` and `evo_rpe` on PATH (install via pip install evo)

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$BASE_DIR/results}"
RESULTS="$RESULTS_DIR"

get_ngps_ref() {
    if [ -n "${NOISY_GPS_GT_TUM:-}" ]; then
        echo "$NOISY_GPS_GT_TUM"
    elif [ -f "$RESULTS/noisy_gps/gps_pose.tum" ]; then
        echo "$RESULTS/noisy_gps/gps_pose.tum"
    elif [ -f "$RESULTS/ngps_only/gps_pose.tum" ]; then
        echo "$RESULTS/ngps_only/gps_pose.tum"
    else
        echo "$RESULTS/ngps_only-deprecated/gps_pose.tum"
    fi
}

get_ngps_est() {
    if [ -n "${NOISY_GPS_TUM:-}" ]; then
        echo "$NOISY_GPS_TUM"
    elif [ -f "$RESULTS/noisy_gps/noisy_gps_seed_11.tum" ]; then
        echo "$RESULTS/noisy_gps/noisy_gps_seed_11.tum"
    elif [ -f "$RESULTS/noisy_gps/noisy_gnss.tum" ]; then
        echo "$RESULTS/noisy_gps/noisy_gnss.tum"
    elif [ -f "$RESULTS/ngps_only/noisy_gnss.tum" ]; then
        echo "$RESULTS/ngps_only/noisy_gnss.tum"
    else
        echo "$RESULTS/ngps_only-deprecated/noisy_gnss.tum"
    fi
}


get_ref() {
    case "$1" in
        spf) echo "$RESULTS/spf_lidar/gps_pose.tum" ;;
        spfpp) echo "$RESULTS/spf_lidar++/0.5/gps_pose.tum" ;;
        ngps) get_ngps_ref ;;
        amcl) echo "$RESULTS/amcl/tum1/gps_pose.tum" ;;
        amcl_ngps)
            if [ -f "$RESULTS/amcl_ngps/tum1/gps_pose.tum" ]; then
                echo "$RESULTS/amcl_ngps/tum1/gps_pose.tum"
            else
                echo "$RESULTS/iros/amcl_ngps/tum1/gps_pose.tum"
            fi
            ;;
        rtab_rgbd) echo "$RESULTS/rtabmap/rgbd/tum1/gps_pose.tum" ;;
        rtab_rgb) echo "$RESULTS/rtabmap/rgb/tum1/gps_pose.tum" ;;
        orb_rgbd_s4) echo "$RESULTS/orbslam3/rgbd/s4/gps_pose.tum" ;;
        orb_rgbd_full) echo "$RESULTS/orbslam3/rgbd/full/gps_pose.tum" ;;
        orb_mono_s4) echo "$RESULTS/orbslam3/mono/s4/gps_pose.tum" ;;
        orb_mono_full) echo "$RESULTS/orbslam3/mono/full/gps_pose.tum" ;;
        *) echo "" ;;
    esac
}

get_est() {
    case "$1" in
        spf) echo "$RESULTS/spf_lidar/spf_lidar.tum" ;;
        spfpp) echo "$RESULTS/spf_lidar++/0.5/trajectory_0.5.tum" ;;
        ngps) get_ngps_est ;;
        amcl) echo "$RESULTS/amcl/tum1/amcl_pose.tum" ;;
        amcl_ngps)
            if [ -f "$RESULTS/amcl_ngps/tum1/trajectory_0.5.tum" ]; then
                echo "$RESULTS/amcl_ngps/tum1/trajectory_0.5.tum"
            else
                echo "$RESULTS/iros/amcl_ngps/tum1/trajectory_0.5.tum"
            fi
            ;;
        rtab_rgbd) echo "$RESULTS/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum" ;;
        rtab_rgb) echo "$RESULTS/rtabmap/rgb/tum1/rtabmap_rgb_filtered.tum" ;;
        orb_rgbd_s4) echo "$RESULTS/orbslam3/rgbd/s4/orbslam3_rgbd.tum" ;;
        orb_rgbd_full) echo "$RESULTS/orbslam3/rgbd/full/orbslam3_rgbd.tum" ;;
        orb_mono_s4) echo "$RESULTS/orbslam3/mono/s4/orbslam3_mono.tum" ;;
        orb_mono_full) echo "$RESULTS/orbslam3/mono/full/orbslam3_mono.tum" ;;
        *) echo "" ;;
    esac
}

METHODS_DEFAULT=(spf spfpp ngps amcl amcl_ngps rtab_rgbd rtab_rgb orb_rgbd_s4 orb_rgbd_full orb_mono_s4 orb_mono_full)
if [ -n "${EVO_METHODS:-}" ]; then
    IFS=',' read -r -a METHODS <<< "$EVO_METHODS"
else
    METHODS=("${METHODS_DEFAULT[@]}")
fi

method_label() {
    case "$1" in
        spf) echo "SPF LiDAR" ;;
        spfpp) echo "SPF LiDAR++" ;;
        ngps) echo "Noisy GPS" ;;
        amcl) echo "AMCL" ;;
        amcl_ngps) echo "AMCL+GPS" ;;
        rtab_rgbd) echo "RTABMap RGBD" ;;
        rtab_rgb) echo "RTABMap RGB" ;;
        orb_rgbd_s4) echo "ORB-SLAM3 RGBD (s4)" ;;
        orb_rgbd_full) echo "ORB-SLAM3 RGBD (full)" ;;
        orb_mono_s4) echo "ORB-SLAM3 Mono (s4)" ;;
        orb_mono_full) echo "ORB-SLAM3 Mono (full)" ;;
        *) echo "" ;;
    esac
}

if [ -d "$BASE_DIR/.venv" ]; then
    # shellcheck disable=SC1090
    source "$BASE_DIR/.venv/bin/activate"
fi

if [ "${SKIP_COMPUTE_METRICS:-0}" != "1" ]; then
    METRIC_LABELS="${METRICS_METHOD_LABELS:-}"
    if [ -z "$METRIC_LABELS" ] && [ -n "${EVO_METHODS:-}" ]; then
        labels=()
        for m in "${METHODS[@]}"; do
            lbl="$(method_label "$m")"
            if [ -n "$lbl" ]; then
                labels+=("$lbl")
            fi
        done
        if [ "${#labels[@]}" -gt 0 ]; then
            IFS=',' METRIC_LABELS="${labels[*]}"
        fi
    fi

    echo "Computing ATE/RTE and row metrics with compute_metrics.py..."
    RESULTS_DIR="$RESULTS_DIR" METRICS_METHOD_LABELS="$METRIC_LABELS" python3 "$BASE_DIR/scripts/compute_metrics.py"
fi


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

    # Run RPE at distances 2m,5m,10m with alignment (A/B convention).
    for d in 2 5 10; do
        echo "Running evo RPE (delta=${d}m) for $m..."
        evo_rpe tum "$ref" "$est" -a --delta $d --delta_unit m --save_results "$RESULTS/evo_${m}_rpe_${d}m.json" --no_warnings || echo "evo_rpe failed for $m delta $d"
    done
done

echo "Finished evo runs. Output JSONs are in $RESULTS"

echo "Aggregating evo results into CSV + PDF..."
RESULTS_DIR="$RESULTS_DIR" EVO_METHODS="${EVO_METHODS:-}" python3 "$BASE_DIR/scripts/aggregate_evo_results.py"

echo "Aggregation complete. See $RESULTS/evo_aggregated_metrics.csv and $RESULTS/evo_aggregated_metrics.pdf"
