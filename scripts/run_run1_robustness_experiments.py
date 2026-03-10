#!/usr/bin/env python3
"""Run additional run1 robustness experiments:

Option A: detection degradation
  - Randomly drop X% semantic detections before LiDAR association.
  - Report APE and row correctness.

Option B: landmark removal from map
  - Remove 30-50% landmarks inside one map section.
  - Report recovery behavior after leaving the removed section.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geojson_rows import extract_row_id as extract_geojson_row_id
from run_ab_validation import (
    BASE_DIR,
    aligned_estimate,
    check_cuda_available,
    compute_row_metrics,
    compute_smoothness_metrics,
    evaluate_run,
    interpolate_positions,
    load_rows_from_geojson,
    read_tum_file,
    run_cmd,
)


DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_rh1_robustness"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "2025" / "rh_run1"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
DEFAULT_SEEDS = "11,22,33"
DEFAULT_DROP_RATES = "0.2,0.4"
DEFAULT_REMOVE_RATES = "0.3,0.5"
DEFAULT_SECTION_QMIN = 0.35
DEFAULT_SECTION_QMAX = 0.65


@dataclass
class SectionDefinition:
    axis: int
    axis_name: str
    lo_abs: float
    hi_abs: float
    lo_centered: float
    hi_centered: float
    center_x: float
    center_y: float
    n_total: int
    n_in_section: int


@dataclass
class MapRemovalVariant:
    remove_rate: float
    geojson_path: Path
    n_removed: int
    n_in_section: int
    section: SectionDefinition


def parse_int_list(text: str) -> List[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def parse_float_list(text: str) -> List[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def safe_float(v, default=float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return x


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_slpf_cmd(
    *,
    python_exec: Path,
    out_dir: Path,
    seed: int,
    data_path: Path,
    geojson: Path,
    detection_drop_rate: float,
    require_cuda: bool,
    max_frames: int | None,
) -> List[str]:
    cmd = [
        str(python_exec),
        str(BASE_DIR / "scripts" / "spf_lidar.py"),
        "--miss-penalty", "4.0",
        "--wrong-hit-penalty", "4.0",
        "--gps-weight", "0.5",
        "--seed", str(seed),
        "--output-folder", str(out_dir),
        "--frame-stride", "4",
        "--semantic-sigma", "0.05",
        "--gps-sigma", "1.1",
        "--corridor-weight", "0.30",
        "--corridor-dist-sigma", "1.50",
        "--corridor-heading-sigma", "0.35",
        "--background-class-weight", "0.20",
        "--max-background-obs", "120",
        "--expected-obs-count", "150",
        "--pose-smooth-alpha-pos", "0.55",
        "--pose-smooth-alpha-theta", "0.50",
        "--odom-yaw-filter-alpha", "0.90",
        "--particle-count", "100",
        "--segment-chunk", "4096",
        "--detection-drop-rate", f"{max(0.0, min(1.0, float(detection_drop_rate))):.6f}",
        "--no-visualization",
        "--data-path", str(data_path),
        "--geojson-path", str(geojson),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def _prepare_landmarks_for_spf(gdf_raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf_raw.copy()

    def _extract_row_id(row) -> str:
        return extract_geojson_row_id(
            {
                "feature_type": row.get("feature_type", ""),
                "vine_vine_row_id": row.get("vine_vine_row_id", ""),
                "row_post_id": row.get("row_post_id", ""),
                "feature_name": row.get("feature_name", ""),
            }
        )

    gdf["row_id"] = gdf.apply(_extract_row_id, axis=1)
    gdf = gdf[gdf["row_id"].astype(str).str.len() > 0]
    gdf = gdf[gdf["row_id"] != "unknown"]
    if gdf.empty:
        raise ValueError("No valid row landmarks available after filtering.")
    if not gdf.crs or not gdf.crs.is_projected:
        utm_crs = gdf.estimate_utm_crs()
        if utm_crs is None:
            raise ValueError("Failed to estimate projected CRS for map.")
        gdf = gdf.to_crs(utm_crs)
    return gdf


def _section_from_landmarks(
    gdf_landmarks: gpd.GeoDataFrame,
    *,
    qmin: float | None = None,
    qmax: float | None = None,
    axis: int | None = None,
    lo_abs: float | None = None,
    hi_abs: float | None = None,
) -> SectionDefinition:
    x = gdf_landmarks.geometry.x.to_numpy(dtype=np.float64)
    y = gdf_landmarks.geometry.y.to_numpy(dtype=np.float64)
    if axis is None:
        axis = 1 if np.ptp(y) >= np.ptp(x) else 0
    axis_name = "y" if axis == 1 else "x"
    vals = y if axis == 1 else x

    if lo_abs is None or hi_abs is None:
        if qmin is None or qmax is None:
            raise ValueError("Either (qmin,qmax) or (lo_abs,hi_abs) must be provided.")
        lo_abs = float(np.quantile(vals, qmin))
        hi_abs = float(np.quantile(vals, qmax))

    center = gdf_landmarks.geometry.union_all().centroid
    center_x = float(center.x)
    center_y = float(center.y)
    center_axis = center_y if axis == 1 else center_x
    in_section = (vals >= lo_abs) & (vals <= hi_abs)

    return SectionDefinition(
        axis=int(axis),
        axis_name=axis_name,
        lo_abs=float(lo_abs),
        hi_abs=float(hi_abs),
        lo_centered=float(lo_abs - center_axis),
        hi_centered=float(hi_abs - center_axis),
        center_x=center_x,
        center_y=center_y,
        n_total=int(len(gdf_landmarks)),
        n_in_section=int(np.sum(in_section)),
    )


def create_section_removed_map(
    *,
    src_geojson: Path,
    dst_geojson: Path,
    remove_rate: float,
    axis: int,
    lo_abs: float,
    hi_abs: float,
    random_seed: int,
) -> MapRemovalVariant:
    gdf_raw = gpd.read_file(src_geojson)
    gdf_landmarks = _prepare_landmarks_for_spf(gdf_raw)

    vals = (
        gdf_landmarks.geometry.y.to_numpy(dtype=np.float64)
        if axis == 1
        else gdf_landmarks.geometry.x.to_numpy(dtype=np.float64)
    )
    section_mask = (vals >= lo_abs) & (vals <= hi_abs)
    section_indices = np.asarray(gdf_landmarks.index[section_mask].to_list(), dtype=np.int64)

    remove_rate = float(max(0.0, min(1.0, remove_rate)))
    n_remove = int(round(remove_rate * float(section_indices.size)))
    # Keep at least two landmarks so segment construction remains valid.
    n_remove = min(n_remove, max(0, len(gdf_landmarks) - 2))

    if n_remove > 0 and section_indices.size > 0:
        rng = np.random.default_rng(int(random_seed))
        remove_idx = rng.choice(section_indices, size=n_remove, replace=False)
        gdf_out = gdf_raw.drop(index=remove_idx)
    else:
        remove_idx = np.asarray([], dtype=np.int64)
        gdf_out = gdf_raw.copy()

    dst_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(dst_geojson, driver="GeoJSON")

    gdf_out_landmarks = _prepare_landmarks_for_spf(gdf_out)
    section_out = _section_from_landmarks(
        gdf_out_landmarks,
        axis=axis,
        lo_abs=lo_abs,
        hi_abs=hi_abs,
    )

    return MapRemovalVariant(
        remove_rate=remove_rate,
        geojson_path=dst_geojson,
        n_removed=int(len(remove_idx)),
        n_in_section=int(section_indices.size),
        section=section_out,
    )


def _summarize(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def aggregate_main(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["option"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)

    out_rows: List[Dict[str, object]] = []
    metric_keys = [
        "ape_align_rmse",
        "row_correct_fraction",
        "cross_track_mean",
        "row_switch_events",
        "runtime_sec",
    ]
    for (option, variant), group in sorted(grouped.items()):
        out: Dict[str, object] = {
            "option": option,
            "variant": variant,
            "n_runs": len(group),
        }
        for m in metric_keys:
            mean, std = _summarize(safe_float(r.get(m)) for r in group)
            out[f"{m}_mean"] = mean
            out[f"{m}_std"] = std
        out_rows.append(out)

    baseline = next(
        (r for r in out_rows if str(r["option"]) == "baseline" and str(r["variant"]) == "full_map"),
        None,
    )
    if baseline is not None:
        base_ape = safe_float(baseline.get("ape_align_rmse_mean"))
        base_row = safe_float(baseline.get("row_correct_fraction_mean"))
        for row in out_rows:
            row["delta_ape_align_vs_baseline"] = safe_float(row.get("ape_align_rmse_mean")) - base_ape
            row["delta_row_correct_vs_baseline"] = safe_float(row.get("row_correct_fraction_mean")) - base_row
    return out_rows


def compute_recovery_metrics(
    *,
    variant_aligned: Dict[str, np.ndarray],
    baseline_aligned: Dict[str, np.ndarray],
    variant_section: SectionDefinition,
    baseline_section: SectionDefinition,
    hold_frames: int = 12,
    post_window: int = 80,
) -> Dict[str, float | int]:
    err_var = np.asarray(variant_aligned["errors"], dtype=np.float64)
    gt_var = np.asarray(variant_aligned["gt_interp"], dtype=np.float64)
    err_base = np.asarray(baseline_aligned["errors"], dtype=np.float64)
    gt_base = np.asarray(baseline_aligned["gt_interp"], dtype=np.float64)

    axis = int(variant_section.axis)
    var_vals = gt_var[:, axis]
    base_vals = gt_base[:, axis]

    var_mask = (var_vals >= variant_section.lo_centered) & (var_vals <= variant_section.hi_centered)
    base_mask = (base_vals >= baseline_section.lo_centered) & (base_vals <= baseline_section.hi_centered)

    if not np.any(var_mask):
        return {
            "section_frames": 0,
            "section_ape_mean": float("nan"),
            "section_ape_peak": float("nan"),
            "post_ape_mean": float("nan"),
            "baseline_post_ape_mean": float("nan"),
            "post_to_section_ratio": float("nan"),
            "recovery_threshold_ape": float("nan"),
            "recovered_flag": 0,
            "recovery_frames": float("nan"),
            "recovery_distance_m": float("nan"),
        }

    var_idx = np.flatnonzero(var_mask)
    var_exit = int(var_idx[-1])
    section_err = err_var[var_idx]
    section_mean = float(np.mean(section_err))
    section_peak = float(np.max(section_err))

    if np.any(base_mask):
        base_exit = int(np.flatnonzero(base_mask)[-1])
    else:
        base_exit = int(max(0, min(len(err_base) - 1, round(0.6 * len(err_base)))))

    var_post = err_var[var_exit + 1: var_exit + 1 + post_window]
    base_post = err_base[base_exit + 1: base_exit + 1 + post_window]
    post_ape_mean = float(np.mean(var_post)) if var_post.size > 0 else float("nan")
    baseline_post_mean = float(np.mean(base_post)) if base_post.size > 0 else float(np.mean(err_base))

    recovery_threshold = float(baseline_post_mean * 1.15) if math.isfinite(baseline_post_mean) else float("nan")
    recovered_flag = 0
    recovery_frames = float("nan")
    recovery_distance = float("nan")

    if math.isfinite(recovery_threshold):
        start = var_exit + 1
        end = len(err_var) - hold_frames + 1
        if start < end:
            for k in range(start, end):
                window = err_var[k:k + hold_frames]
                if np.all(window <= recovery_threshold):
                    recovered_flag = 1
                    recovery_frames = float(k - var_exit)
                    gt_xy = gt_var[:, :2]
                    cumdist = np.concatenate(
                        [[0.0], np.cumsum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1))]
                    )
                    recovery_distance = float(max(0.0, cumdist[k] - cumdist[var_exit]))
                    break

    ratio = (
        float(post_ape_mean / section_mean)
        if math.isfinite(post_ape_mean) and math.isfinite(section_mean) and section_mean > 1e-12
        else float("nan")
    )

    return {
        "section_frames": int(var_idx.size),
        "section_ape_mean": section_mean,
        "section_ape_peak": section_peak,
        "post_ape_mean": post_ape_mean,
        "baseline_post_ape_mean": baseline_post_mean,
        "post_to_section_ratio": ratio,
        "recovery_threshold_ape": recovery_threshold,
        "recovered_flag": int(recovered_flag),
        "recovery_frames": recovery_frames,
        "recovery_distance_m": recovery_distance,
    }


def evaluate_run_safe(
    *,
    name: str,
    est_tum: Path,
    gt_tum: Path,
    out_dir: Path,
    rows: Dict[str, np.ndarray],
    evo_ape_bin: Path,
    evo_rpe_bin: Path,
    env: Dict[str, str],
) -> Dict[str, object]:
    try:
        return evaluate_run(
            name=name,
            est_tum=est_tum,
            gt_tum=gt_tum,
            out_dir=out_dir,
            rows=rows,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
    except RuntimeError as exc:
        # Keep debug/smoke runs usable when evo RPE fails on too-short trajectories.
        est = read_tum_file(est_tum)
        gt = read_tum_file(gt_tum)
        gt_interp = interpolate_positions(gt.timestamps, gt.positions, est.timestamps)
        scale, rot, trans = 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        try:
            aligned = aligned_estimate(est_tum, gt_tum)
            est_aligned = aligned["est_aligned"]
            err = aligned["errors"]
        except Exception:
            est_aligned = (scale * (rot @ est.positions.T)).T + trans
            err = np.linalg.norm(est_aligned - gt_interp, axis=1)

        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")
        row_metrics = compute_row_metrics(est_aligned, gt_interp, rows) if rows else {
            "cross_track_mean": float("nan"),
            "cross_track_median": float("nan"),
            "cross_track_max": float("nan"),
            "row_correct_fraction": float("nan"),
            "row_switch_events": float("nan"),
        }
        smooth = compute_smoothness_metrics(est.timestamps, est_aligned)

        fallback = {
            "run_name": name,
            "est_tum": str(est_tum),
            "gt_tum": str(gt_tum),
            "ape_raw_rmse": rmse,
            "ape_align_rmse": rmse,
            "rpe_2m_align_rmse": float("nan"),
            "rpe_5m_align_rmse": float("nan"),
            "rpe_10m_align_rmse": float("nan"),
            **row_metrics,
            **smooth,
            "eval_warning": str(exc),
        }
        return fallback


def aggregate_recovery(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    out_rows: List[Dict[str, object]] = []
    for variant, group in sorted(grouped.items()):
        out: Dict[str, object] = {"variant": variant, "n_runs": len(group)}
        for metric in [
            "section_ape_mean",
            "section_ape_peak",
            "post_ape_mean",
            "baseline_post_ape_mean",
            "post_to_section_ratio",
            "recovery_frames",
            "recovery_distance_m",
        ]:
            mean, std = _summarize(safe_float(r.get(metric)) for r in group)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        recovered_vals = [safe_float(r.get("recovered_flag")) for r in group]
        recovered_vals = [v for v in recovered_vals if math.isfinite(v)]
        out["recovered_fraction"] = float(np.mean(recovered_vals)) if recovered_vals else float("nan")
        out_rows.append(out)
    return out_rows


def plot_recovery_summary(
    recovery_agg_rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    if not recovery_agg_rows:
        return

    labels = [str(r["variant"]) for r in recovery_agg_rows]
    x = np.arange(len(labels))
    width = 0.28

    section_mean = np.asarray([safe_float(r["section_ape_mean_mean"]) for r in recovery_agg_rows], dtype=np.float64)
    post_mean = np.asarray([safe_float(r["post_ape_mean_mean"]) for r in recovery_agg_rows], dtype=np.float64)
    post_ref = np.asarray([safe_float(r["baseline_post_ape_mean_mean"]) for r in recovery_agg_rows], dtype=np.float64)

    rec_frames = np.asarray([safe_float(r["recovery_frames_mean"]) for r in recovery_agg_rows], dtype=np.float64)
    rec_frac = np.asarray([safe_float(r["recovered_fraction"]) for r in recovery_agg_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax0 = axes[0]
    ax0.bar(x - width, section_mean, width=width, color="tab:red", alpha=0.85, label="APE in removed section")
    ax0.bar(x, post_mean, width=width, color="tab:blue", alpha=0.85, label="APE after section")
    ax0.bar(x + width, post_ref, width=width, color="tab:green", alpha=0.85, label="Baseline post-section APE")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylabel("APE (m)")
    ax0.set_title("Section Error vs Post-section Error")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="best")

    ax1 = axes[1]
    bars = ax1.bar(x, rec_frames, color="tab:purple", alpha=0.85)
    for i, b in enumerate(bars):
        frac = rec_frac[i]
        if math.isfinite(frac):
            ax1.text(
                b.get_x() + b.get_width() * 0.5,
                b.get_height() if math.isfinite(b.get_height()) else 0.0,
                f"{100.0 * frac:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Frames to recover")
    ax1.set_title("Recovery Time (label: recovery success rate)")
    ax1.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Option B Recovery Summary", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def resolve_reuse_baseline_paths(reuse_dir: Path, seed: int) -> Tuple[Path, Path] | None:
    candidates = [
        reuse_dir / f"seed_{seed}",
        reuse_dir / "slpf" / f"seed_{seed}",
    ]
    for c in candidates:
        est = c / "trajectory_0.5.tum"
        gt = c / "gps_pose.tum"
        if est.exists() and gt.exists():
            return est, gt
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run run1 robustness experiments (Option A + Option B).")
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv" / "bin" / "python")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--drop-rates", type=str, default=DEFAULT_DROP_RATES)
    parser.add_argument("--remove-rates", type=str, default=DEFAULT_REMOVE_RATES)
    parser.add_argument("--section-qmin", type=float, default=DEFAULT_SECTION_QMIN)
    parser.add_argument("--section-qmax", type=float, default=DEFAULT_SECTION_QMAX)
    parser.add_argument("--map-random-seed", type=int, default=2026)
    parser.add_argument("--reuse-baseline-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    args = parser.parse_args()

    python_exec = args.python_exec.expanduser()
    if not python_exec.is_absolute():
        python_exec = (BASE_DIR / python_exec).resolve()
    data_path = args.data_path.expanduser()
    if not data_path.is_absolute():
        data_path = (BASE_DIR / data_path).resolve()
    geojson = args.geojson.expanduser()
    if not geojson.is_absolute():
        geojson = (BASE_DIR / geojson).resolve()
    reuse_baseline_dir = None
    if args.reuse_baseline_dir is not None:
        reuse_baseline_dir = args.reuse_baseline_dir.expanduser()
        if not reuse_baseline_dir.is_absolute():
            reuse_baseline_dir = (BASE_DIR / reuse_baseline_dir).resolve()

    if args.section_qmin < 0.0 or args.section_qmax > 1.0 or args.section_qmin >= args.section_qmax:
        raise ValueError("Section quantiles must satisfy 0 <= qmin < qmax <= 1.")

    seeds = parse_int_list(args.seeds)
    drop_rates = [float(x) for x in parse_float_list(args.drop_rates)]
    remove_rates = [float(x) for x in parse_float_list(args.remove_rates)]
    for r in [*drop_rates, *remove_rates]:
        if r < 0.0 or r > 1.0:
            raise ValueError("All rates must be in [0, 1].")

    data_csv = data_path / "data.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing dataset csv: {data_csv}")
    if not geojson.exists():
        raise FileNotFoundError(f"Missing map geojson: {geojson}")

    if args.require_cuda and not check_cuda_available(python_exec):
        raise RuntimeError("CUDA is required but not visible in the selected Python runtime.")

    evo_ape_bin = python_exec.parent / "evo_ape"
    evo_rpe_bin = python_exec.parent / "evo_rpe"
    if not evo_ape_bin.exists() or not evo_rpe_bin.exists():
        raise FileNotFoundError("Missing evo_ape/evo_rpe in selected virtualenv bin directory.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hw_tag = "gpu" if args.require_cuda else "cpu"
    run_dir = args.output_root.resolve() / f"{timestamp}_run1_robustness_{hw_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    rows_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def get_rows(map_path: Path) -> Dict[str, np.ndarray]:
        key = str(map_path.resolve())
        if key not in rows_cache:
            rows_cache[key] = load_rows_from_geojson(map_path)
        return rows_cache[key]

    full_landmarks = _prepare_landmarks_for_spf(gpd.read_file(geojson))
    baseline_section = _section_from_landmarks(
        full_landmarks,
        qmin=float(args.section_qmin),
        qmax=float(args.section_qmax),
    )

    protocol: Dict[str, object] = {
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "geojson": str(geojson),
        "seeds": seeds,
        "drop_rates": drop_rates,
        "remove_rates": remove_rates,
        "section_qmin": float(args.section_qmin),
        "section_qmax": float(args.section_qmax),
        "baseline_section": {
            "axis": baseline_section.axis,
            "axis_name": baseline_section.axis_name,
            "lo_abs": baseline_section.lo_abs,
            "hi_abs": baseline_section.hi_abs,
            "lo_centered": baseline_section.lo_centered,
            "hi_centered": baseline_section.hi_centered,
        },
        "commands": [],
        "map_variants": [],
    }

    all_rows: List[Dict[str, object]] = []
    baseline_aligned_cache: Dict[int, Dict[str, np.ndarray]] = {}

    for seed in seeds:
        if reuse_baseline_dir is not None:
            reused = resolve_reuse_baseline_paths(reuse_baseline_dir, seed)
        else:
            reused = None

        if reused is not None:
            est_tum, gt_tum = reused
            runtime_sec = 0.0
            protocol["commands"].append(
                {
                    "option": "baseline",
                    "seed": seed,
                    "command": [f"reuse baseline est={est_tum}", f"reuse baseline gt={gt_tum}"],
                    "runtime_sec": runtime_sec,
                }
            )
        else:
            out_dir = run_dir / "baseline" / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_slpf_cmd(
                python_exec=python_exec,
                out_dir=out_dir,
                seed=seed,
                data_path=data_path,
                geojson=geojson,
                detection_drop_rate=0.0,
                require_cuda=args.require_cuda,
                max_frames=args.max_frames,
            )
            runtime_sec = run_cmd(cmd, out_dir / "run_baseline.log", cwd=BASE_DIR, env=env)
            protocol["commands"].append(
                {"option": "baseline", "seed": seed, "command": cmd, "runtime_sec": runtime_sec}
            )
            est_tum = out_dir / "trajectory_0.5.tum"
            gt_tum = out_dir / "gps_pose.tum"

        if not est_tum.exists() or not gt_tum.exists():
            raise FileNotFoundError(f"Missing baseline outputs for seed {seed}: {est_tum}, {gt_tum}")

        metrics = evaluate_run_safe(
            name=f"baseline_seed_{seed}",
            est_tum=est_tum,
            gt_tum=gt_tum,
            out_dir=run_dir / "baseline_eval" / f"seed_{seed}",
            rows=get_rows(geojson),
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        metrics.update(
            {
                "option": "baseline",
                "variant": "full_map",
                "seed": seed,
                "detection_drop_rate": 0.0,
                "map_remove_rate": 0.0,
                "map_geojson": str(geojson),
                "runtime_sec": runtime_sec,
            }
        )
        all_rows.append(metrics)
        baseline_aligned_cache[seed] = aligned_estimate(est_tum, gt_tum)

    # Option A: detection degradation
    for drop_rate in drop_rates:
        pct = int(round(100.0 * drop_rate))
        variant = f"drop_{pct:02d}pct"
        for seed in seeds:
            out_dir = run_dir / "option_a_detection_drop" / variant / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_slpf_cmd(
                python_exec=python_exec,
                out_dir=out_dir,
                seed=seed,
                data_path=data_path,
                geojson=geojson,
                detection_drop_rate=drop_rate,
                require_cuda=args.require_cuda,
                max_frames=args.max_frames,
            )
            runtime_sec = run_cmd(cmd, out_dir / "run.log", cwd=BASE_DIR, env=env)
            protocol["commands"].append(
                {
                    "option": "option_a_detection_drop",
                    "variant": variant,
                    "seed": seed,
                    "command": cmd,
                    "runtime_sec": runtime_sec,
                }
            )

            est_tum = out_dir / "trajectory_0.5.tum"
            gt_tum = out_dir / "gps_pose.tum"
            if not est_tum.exists() or not gt_tum.exists():
                raise FileNotFoundError(f"Missing Option A outputs for {variant} seed {seed}: {out_dir}")

            metrics = evaluate_run_safe(
                name=f"{variant}_seed_{seed}",
                est_tum=est_tum,
                gt_tum=gt_tum,
                out_dir=out_dir / "eval",
                rows=get_rows(geojson),
                evo_ape_bin=evo_ape_bin,
                evo_rpe_bin=evo_rpe_bin,
                env=env,
            )
            metrics.update(
                {
                    "option": "option_a_detection_drop",
                    "variant": variant,
                    "seed": seed,
                    "detection_drop_rate": float(drop_rate),
                    "map_remove_rate": 0.0,
                    "map_geojson": str(geojson),
                    "runtime_sec": runtime_sec,
                }
            )
            all_rows.append(metrics)

    # Option B: map landmark removal inside one section
    option_b_variants: List[MapRemovalVariant] = []
    maps_dir = run_dir / "option_b_map_variants"
    for i, remove_rate in enumerate(remove_rates):
        pct = int(round(100.0 * remove_rate))
        map_path = maps_dir / f"riseholme_poles_trunk_remove_{pct:02d}pct.geojson"
        variant = create_section_removed_map(
            src_geojson=geojson,
            dst_geojson=map_path,
            remove_rate=remove_rate,
            axis=baseline_section.axis,
            lo_abs=baseline_section.lo_abs,
            hi_abs=baseline_section.hi_abs,
            random_seed=int(args.map_random_seed + 1000 * i + pct),
        )
        option_b_variants.append(variant)
        protocol["map_variants"].append(
            {
                "remove_rate": variant.remove_rate,
                "path": str(variant.geojson_path),
                "n_removed": variant.n_removed,
                "n_in_section": variant.n_in_section,
                "section": {
                    "axis": variant.section.axis,
                    "axis_name": variant.section.axis_name,
                    "lo_abs": variant.section.lo_abs,
                    "hi_abs": variant.section.hi_abs,
                    "lo_centered": variant.section.lo_centered,
                    "hi_centered": variant.section.hi_centered,
                    "center_x": variant.section.center_x,
                    "center_y": variant.section.center_y,
                },
            }
        )

    recovery_rows: List[Dict[str, object]] = []
    for variant in option_b_variants:
        pct = int(round(100.0 * variant.remove_rate))
        variant_name = f"remove_{pct:02d}pct"
        for seed in seeds:
            out_dir = run_dir / "option_b_landmark_removal" / variant_name / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_slpf_cmd(
                python_exec=python_exec,
                out_dir=out_dir,
                seed=seed,
                data_path=data_path,
                geojson=variant.geojson_path,
                detection_drop_rate=0.0,
                require_cuda=args.require_cuda,
                max_frames=args.max_frames,
            )
            runtime_sec = run_cmd(cmd, out_dir / "run.log", cwd=BASE_DIR, env=env)
            protocol["commands"].append(
                {
                    "option": "option_b_landmark_removal",
                    "variant": variant_name,
                    "seed": seed,
                    "command": cmd,
                    "runtime_sec": runtime_sec,
                }
            )

            est_tum = out_dir / "trajectory_0.5.tum"
            gt_tum = out_dir / "gps_pose.tum"
            if not est_tum.exists() or not gt_tum.exists():
                raise FileNotFoundError(f"Missing Option B outputs for {variant_name} seed {seed}: {out_dir}")

            metrics = evaluate_run_safe(
                name=f"{variant_name}_seed_{seed}",
                est_tum=est_tum,
                gt_tum=gt_tum,
                out_dir=out_dir / "eval",
                rows=get_rows(variant.geojson_path),
                evo_ape_bin=evo_ape_bin,
                evo_rpe_bin=evo_rpe_bin,
                env=env,
            )
            metrics.update(
                {
                    "option": "option_b_landmark_removal",
                    "variant": variant_name,
                    "seed": seed,
                    "detection_drop_rate": 0.0,
                    "map_remove_rate": float(variant.remove_rate),
                    "map_geojson": str(variant.geojson_path),
                    "runtime_sec": runtime_sec,
                }
            )
            all_rows.append(metrics)

            variant_aligned = aligned_estimate(est_tum, gt_tum)
            recovery = compute_recovery_metrics(
                variant_aligned=variant_aligned,
                baseline_aligned=baseline_aligned_cache[seed],
                variant_section=variant.section,
                baseline_section=baseline_section,
            )
            recovery_rows.append(
                {
                    "variant": variant_name,
                    "seed": seed,
                    "remove_rate": float(variant.remove_rate),
                    "n_removed": int(variant.n_removed),
                    "n_in_section": int(variant.n_in_section),
                    **recovery,
                }
            )

    all_rows.sort(
        key=lambda r: (
            str(r["option"]),
            str(r["variant"]),
            int(r["seed"]),
        )
    )
    write_csv(run_dir / "run1_robustness_per_seed.csv", all_rows)

    agg_rows = aggregate_main(all_rows)
    write_csv(run_dir / "run1_robustness_aggregate.csv", agg_rows)

    option_a_summary = [
        {
            "variant": str(r["variant"]),
            "n_runs": int(r["n_runs"]),
            "ape_align_rmse_mean": safe_float(r.get("ape_align_rmse_mean")),
            "ape_align_rmse_std": safe_float(r.get("ape_align_rmse_std")),
            "row_correct_fraction_mean": safe_float(r.get("row_correct_fraction_mean")),
            "row_correct_fraction_std": safe_float(r.get("row_correct_fraction_std")),
            "delta_ape_align_vs_baseline": safe_float(r.get("delta_ape_align_vs_baseline")),
            "delta_row_correct_vs_baseline": safe_float(r.get("delta_row_correct_vs_baseline")),
        }
        for r in agg_rows
        if str(r.get("option")) in {"baseline", "option_a_detection_drop"}
    ]
    write_csv(run_dir / "option_a_detection_drop_summary.csv", option_a_summary)

    write_csv(run_dir / "option_b_recovery_per_seed.csv", recovery_rows)
    recovery_agg = aggregate_recovery(recovery_rows)
    write_csv(run_dir / "option_b_recovery_aggregate.csv", recovery_agg)
    plot_recovery_summary(recovery_agg, run_dir / "option_b_recovery_summary.png")

    (run_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2))

    print(f"[INFO] Run1 robustness experiments complete: {run_dir}")
    best_a = [r for r in option_a_summary if r["variant"] != "full_map"]
    if best_a:
        best = min(best_a, key=lambda x: safe_float(x.get("ape_align_rmse_mean"), 1e9))
        print(
            f"[INFO] Option A best APE variant: {best['variant']} "
            f"(APE={safe_float(best['ape_align_rmse_mean']):.4f}, "
            f"RowAcc={safe_float(best['row_correct_fraction_mean']):.4f})"
        )
    if recovery_agg:
        print("[INFO] Option B recovery summary written to option_b_recovery_aggregate.csv")


if __name__ == "__main__":
    main()
