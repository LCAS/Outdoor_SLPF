#!/usr/bin/env python3
"""
Run A/B validation for SPF LiDAR option 1+2 improvements.

Baseline:
  - existing artifacts in results/ab_runs/main_gpu

Candidate:
  - current branch, executed N times with different seeds

Outputs:
  - timestamped folder under results/ab_runs/
  - per-run and aggregate CSV/JSON metrics
  - aligned overlay and error panel plots
  - markdown validation summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from geojson_rows import iter_projected_points


BASE_DIR = Path(__file__).parent.parent
DEFAULT_BASELINE_DIR = BASE_DIR / "results" / "ab_runs" / "main_gpu"
DEFAULT_DATA_ROOT = BASE_DIR / "data" / "2025" / "ICRA2"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "ab_runs"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"


KEY_METRICS = [
    "ape_align_rmse",
    "rpe_2m_align_rmse",
    "rpe_5m_align_rmse",
    "rpe_10m_align_rmse",
    "cross_track_mean",
    "row_correct_fraction",
    "row_switch_events",
    "speed_mean",
    "accel_rms",
    "jerk_rms",
    "heading_rate_rms",
    "heading_accel_rms",
]


@dataclass
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray


def run_cmd(cmd: List[str], log_path: Path, cwd: Path, env: Dict[str, str] | None = None) -> float:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    duration = time.time() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.write(proc.stdout or "")
        f.write(f"\n[exit_code={proc.returncode} duration_sec={duration:.2f}]\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (see {log_path})")
    return duration


def check_cuda_available(python_exec: Path) -> bool:
    probe = [
        str(python_exec),
        "-c",
        "import torch; print('1' if torch.cuda.is_available() else '0')",
    ]
    proc = subprocess.run(
        probe,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = (proc.stdout or "").strip().splitlines()
    return proc.returncode == 0 and len(out) > 0 and out[-1].strip() == "1"


def read_tum_file(path: Path) -> Trajectory:
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    if not data:
        raise ValueError(f"No valid poses in {path}")
    arr = np.asarray(data, dtype=np.float64)
    return Trajectory(
        timestamps=arr[:, 0],
        positions=arr[:, 1:4],
        quaternions=arr[:, 4:8],
    )


def interpolate_positions(gt_ts: np.ndarray, gt_pos: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    out = np.zeros((len(target_ts), 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = np.interp(target_ts, gt_ts, gt_pos[:, i])
    return out


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scaling: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src={src.shape}, dst={dst.shape}")

    n, m = src.shape
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_c = src - mean_src
    dst_c = dst - mean_dst
    cov = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    if with_scaling:
        var_src = (src_c ** 2).sum() / n
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0
    t = mean_dst - scale * R @ mean_src
    return float(scale), R, t


def apply_transform(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def parse_evo_stats(archive_path: Path) -> Dict[str, float | None]:
    if not archive_path.exists():
        return {"rmse": None, "mean": None, "median": None, "max": None, "std": None}
    with zipfile.ZipFile(archive_path, "r") as z:
        stats = {}
        if "stats.json" in z.namelist():
            stats = json.loads(z.read("stats.json").decode("utf-8"))
        return {
            "rmse": stats.get("rmse"),
            "mean": stats.get("mean"),
            "median": stats.get("median"),
            "max": stats.get("max"),
            "std": stats.get("std"),
        }


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    pa = p - a
    ba = b - a
    denom = float(ba.dot(ba))
    if denom <= 1e-12:
        return float(np.linalg.norm(pa))
    t = max(0.0, min(1.0, float(pa.dot(ba) / denom)))
    proj = a + t * ba
    return float(np.linalg.norm(p - proj))


def nearest_row_and_distance(pt: np.ndarray, rows: Dict[str, np.ndarray]) -> Tuple[str | None, float]:
    best_row = None
    best_dist = float("inf")
    for rid, pts in rows.items():
        for i in range(len(pts) - 1):
            d = point_segment_distance(pt, pts[i], pts[i + 1])
            if d < best_dist:
                best_dist = d
                best_row = rid
    return best_row, best_dist


def load_rows_from_geojson(path: Path, target_crs: str = "epsg:32630") -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    for item in iter_projected_points(path, target_crs=target_crs):
        records.append((item["x"], item["y"], item["row_id"]))

    if not records:
        raise ValueError("No row points found in geojson")

    all_xy = np.array([[r[0], r[1]] for r in records], dtype=np.float64)
    center = all_xy.mean(axis=0)
    rows: Dict[str, List[np.ndarray]] = {}
    for x, y, rid in records:
        rows.setdefault(rid, []).append(np.array([x - center[0], y - center[1]], dtype=np.float64))

    rows_sorted: Dict[str, np.ndarray] = {}
    for rid, pts in rows.items():
        pts_arr = np.asarray(pts)
        if pts_arr.shape[0] < 2:
            continue
        sort_dim = 0 if np.ptp(pts_arr[:, 0]) >= np.ptp(pts_arr[:, 1]) else 1
        order = np.argsort(pts_arr[:, sort_dim])
        rows_sorted[rid] = pts_arr[order]
    return rows_sorted


def compute_row_metrics(est_aligned: np.ndarray, gt_interp: np.ndarray, rows: Dict[str, np.ndarray]) -> Dict[str, float]:
    cross_track_errs = []
    est_rows = []
    gt_rows = []

    for i in range(len(gt_interp)):
        gt_pt = gt_interp[i, :2]
        est_pt = est_aligned[i, :2]
        gt_row, _ = nearest_row_and_distance(gt_pt, rows)
        est_row, est_row_dist = nearest_row_and_distance(est_pt, rows)
        gt_rows.append(gt_row)
        est_rows.append(est_row)

        if gt_row in rows:
            pts = rows[gt_row]
            best_d = float("inf")
            for j in range(len(pts) - 1):
                d = point_segment_distance(est_pt, pts[j], pts[j + 1])
                if d < best_d:
                    best_d = d
            cross_track_errs.append(best_d)
        else:
            cross_track_errs.append(est_row_dist)

    ct_arr = np.asarray([c for c in cross_track_errs if c is not None], dtype=np.float64)
    matches = [1 if (e == g and e is not None) else 0 for e, g in zip(est_rows, gt_rows)]
    wrong = [1 if (e != g and e is not None and g is not None) else 0 for e, g in zip(est_rows, gt_rows)]
    switches = 0
    for i, val in enumerate(wrong):
        if val and (i == 0 or not wrong[i - 1]):
            switches += 1

    return {
        "cross_track_mean": float(np.mean(ct_arr)) if ct_arr.size else float("nan"),
        "cross_track_median": float(np.median(ct_arr)) if ct_arr.size else float("nan"),
        "cross_track_max": float(np.max(ct_arr)) if ct_arr.size else float("nan"),
        "row_correct_fraction": float(np.mean(matches)) if matches else float("nan"),
        "row_switch_events": float(switches),
    }


def _rms(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_smoothness_metrics(timestamps: np.ndarray, aligned_positions: np.ndarray) -> Dict[str, float]:
    if len(timestamps) < 4:
        return {
            "speed_mean": float("nan"),
            "accel_rms": float("nan"),
            "jerk_rms": float("nan"),
            "heading_rate_rms": float("nan"),
            "heading_accel_rms": float("nan"),
        }

    xy = aligned_positions[:, :2]
    dxy = np.diff(xy, axis=0)
    dt = np.diff(timestamps)
    valid = dt > 1e-6
    if not np.any(valid):
        return {
            "speed_mean": float("nan"),
            "accel_rms": float("nan"),
            "jerk_rms": float("nan"),
            "heading_rate_rms": float("nan"),
            "heading_accel_rms": float("nan"),
        }

    dxy = dxy[valid]
    dt = dt[valid]
    speed = np.linalg.norm(dxy, axis=1) / dt

    if speed.size >= 2:
        dt_acc = 0.5 * (dt[1:] + dt[:-1])
        accel = np.diff(speed) / dt_acc
    else:
        accel = np.array([], dtype=np.float64)

    if accel.size >= 2:
        dt_jerk = 0.5 * (dt_acc[1:] + dt_acc[:-1])
        jerk = np.diff(accel) / dt_jerk
    else:
        jerk = np.array([], dtype=np.float64)

    heading = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if heading.size >= 2:
        dt_head = 0.5 * (dt[1:] + dt[:-1])
        heading_rate = np.diff(heading) / dt_head
    else:
        heading_rate = np.array([], dtype=np.float64)

    if heading_rate.size >= 2:
        dt_head_acc = 0.5 * (dt_head[1:] + dt_head[:-1])
        heading_accel = np.diff(heading_rate) / dt_head_acc
    else:
        heading_accel = np.array([], dtype=np.float64)

    return {
        "speed_mean": float(np.mean(speed)) if speed.size else float("nan"),
        "accel_rms": _rms(accel),
        "jerk_rms": _rms(jerk),
        "heading_rate_rms": _rms(heading_rate),
        "heading_accel_rms": _rms(heading_accel),
    }


def aligned_estimate(est_tum: Path, gt_tum: Path) -> Dict[str, np.ndarray]:
    est = read_tum_file(est_tum)
    gt = read_tum_file(gt_tum)
    gt_interp = interpolate_positions(gt.timestamps, gt.positions, est.timestamps)
    scale, rot, trans = umeyama_alignment(est.positions, gt_interp, with_scaling=False)
    est_aligned = apply_transform(est.positions, scale, rot, trans)
    err = np.linalg.norm(est_aligned - gt_interp, axis=1)
    return {
        "timestamps": est.timestamps,
        "est_aligned": est_aligned,
        "gt_interp": gt_interp,
        "errors": err,
    }


def run_evo_bundle(
    est_tum: Path,
    gt_tum: Path,
    prefix: str,
    out_dir: Path,
    evo_ape_bin: Path,
    evo_rpe_bin: Path,
    env: Dict[str, str],
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_json = out_dir / f"evo_{prefix}_ape_raw.json"
    align_json = out_dir / f"evo_{prefix}_ape_align.json"
    rpe2_json = out_dir / f"evo_{prefix}_rpe_2m_align.json"
    rpe5_json = out_dir / f"evo_{prefix}_rpe_5m_align.json"
    rpe10_json = out_dir / f"evo_{prefix}_rpe_10m_align.json"

    run_cmd(
        [str(evo_ape_bin), "tum", str(gt_tum), str(est_tum), "--save_results", str(raw_json), "--no_warnings"],
        out_dir / f"evo_{prefix}_ape_raw.log",
        cwd=BASE_DIR,
        env=env,
    )
    run_cmd(
        [str(evo_ape_bin), "tum", str(gt_tum), str(est_tum), "-a", "--save_results", str(align_json), "--no_warnings"],
        out_dir / f"evo_{prefix}_ape_align.log",
        cwd=BASE_DIR,
        env=env,
    )
    for d, out_json in [(2, rpe2_json), (5, rpe5_json), (10, rpe10_json)]:
        run_cmd(
            [
                str(evo_rpe_bin),
                "tum",
                str(gt_tum),
                str(est_tum),
                "-a",
                "--delta",
                str(d),
                "--delta_unit",
                "m",
                "--save_results",
                str(out_json),
                "--no_warnings",
            ],
            out_dir / f"evo_{prefix}_rpe_{d}m_align.log",
            cwd=BASE_DIR,
            env=env,
        )

    raw = parse_evo_stats(raw_json)
    align = parse_evo_stats(align_json)
    r2 = parse_evo_stats(rpe2_json)
    r5 = parse_evo_stats(rpe5_json)
    r10 = parse_evo_stats(rpe10_json)

    return {
        "ape_raw_rmse": raw["rmse"],
        "ape_align_rmse": align["rmse"],
        "rpe_2m_align_rmse": r2["rmse"],
        "rpe_5m_align_rmse": r5["rmse"],
        "rpe_10m_align_rmse": r10["rmse"],
    }


def evaluate_run(
    name: str,
    est_tum: Path,
    gt_tum: Path,
    out_dir: Path,
    rows: Dict[str, np.ndarray],
    evo_ape_bin: Path,
    evo_rpe_bin: Path,
    env: Dict[str, str],
) -> Dict[str, float | str]:
    evo = run_evo_bundle(est_tum, gt_tum, name, out_dir, evo_ape_bin, evo_rpe_bin, env)
    aligned = aligned_estimate(est_tum, gt_tum)
    row_metrics = compute_row_metrics(aligned["est_aligned"], aligned["gt_interp"], rows)
    smooth = compute_smoothness_metrics(aligned["timestamps"], aligned["est_aligned"])

    return {
        "run_name": name,
        "est_tum": str(est_tum),
        "gt_tum": str(gt_tum),
        **evo,
        **row_metrics,
        **smooth,
    }


def add_error_line(ax, xy: np.ndarray, err: np.ndarray, norm: Normalize, label: str):
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=2.0)
    lc.set_array(err[:-1])
    ax.add_collection(lc)
    ax.plot([], [], color="gray", linestyle="--", label="GT")
    ax.plot([], [], color="black", linewidth=2, label=label)
    return lc


def plot_overlay(
    baseline_aligned: Dict[str, np.ndarray],
    candidate_aligned: Dict[str, np.ndarray],
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 8))
    gt_xy = baseline_aligned["gt_interp"][:, :2]
    b_xy = baseline_aligned["est_aligned"][:, :2]
    c_xy = candidate_aligned["est_aligned"][:, :2]

    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k--", linewidth=2, label="GPS ground truth")
    ax.plot(b_xy[:, 0], b_xy[:, 1], color="tab:blue", linewidth=2, label="baseline main_gpu (aligned)")
    ax.plot(c_xy[:, 0], c_xy[:, 1], color="tab:red", linewidth=2, label="candidate median run (aligned)")
    ax.set_title("A/B Overlay Aligned to Ground Truth (Umeyama SE(3), no scale)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_error_panels(
    baseline_aligned: Dict[str, np.ndarray],
    candidate_aligned: Dict[str, np.ndarray],
    out_path: Path,
):
    b_xy = baseline_aligned["est_aligned"][:, :2]
    c_xy = candidate_aligned["est_aligned"][:, :2]
    b_gt = baseline_aligned["gt_interp"][:, :2]
    c_gt = candidate_aligned["gt_interp"][:, :2]
    b_err = baseline_aligned["errors"]
    c_err = candidate_aligned["errors"]
    vmax = max(float(np.max(b_err)), float(np.max(c_err)))
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, xy, gt, err, title in [
        (axes[0], b_xy, b_gt, b_err, "baseline main_gpu (aligned)"),
        (axes[1], c_xy, c_gt, c_err, "candidate median run (aligned)"),
    ]:
        ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=1.7, alpha=0.75, label="GT")
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=2.4)
        lc.set_array(err[:-1])
        ax.add_collection(lc)
        ax.set_title(title)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    cbar = fig.colorbar(lc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Position error (m)")
    fig.suptitle("Aligned Trajectories with Error Colormap", fontsize=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_seed_variability(candidate_rows: List[Dict[str, float | str]], baseline_row: Dict[str, float | str], out_path: Path):
    seeds = [str(int(r["seed"])) for r in candidate_rows]
    metrics = [
        ("ape_align_rmse", "APE align RMSE (m)"),
        ("cross_track_mean", "Cross-track mean (m)"),
        ("jerk_rms", "Jerk RMS"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(r[key]) for r in candidate_rows]
        x = np.arange(len(vals))
        ax.bar(x, vals, color="tab:blue", alpha=0.8)
        ax.axhline(float(baseline_row[key]), color="tab:red", linestyle="--", linewidth=1.8, label="baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.set_title(title)
        ax.set_xlabel("Seed")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="best")

    fig.suptitle("Seed Variability vs Baseline", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, float | str]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def make_summary(
    run_dir: Path,
    baseline_row: Dict[str, float | str],
    candidate_rows: List[Dict[str, float | str]],
    aggregate_rows: List[Dict[str, float | str]],
    representative: Dict[str, float | str],
) -> Dict[str, object]:
    cand_metrics = {}
    for key in KEY_METRICS:
        vals = np.array([safe_float(r[key]) for r in candidate_rows], dtype=np.float64)
        cand_metrics[key] = {
            "median": float(np.nanmedian(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "std": float(np.nanstd(vals)),
        }

    baseline = {k: safe_float(v) for k, v in baseline_row.items() if k in KEY_METRICS}

    loc_gate = (
        cand_metrics["ape_align_rmse"]["median"] <= baseline["ape_align_rmse"] * 1.01
        and cand_metrics["rpe_5m_align_rmse"]["median"] <= baseline["rpe_5m_align_rmse"] * 1.01
    )
    row_gate = (
        cand_metrics["cross_track_mean"]["median"] <= baseline["cross_track_mean"] * 0.95
        and cand_metrics["row_correct_fraction"]["median"] >= baseline["row_correct_fraction"] + 0.02
    )
    smooth_gate = (
        cand_metrics["jerk_rms"]["median"] <= baseline["jerk_rms"] * 0.95
        and cand_metrics["heading_accel_rms"]["median"] <= baseline["heading_accel_rms"] * 0.95
    )

    no_major_regression = (
        cand_metrics["ape_align_rmse"]["median"] <= baseline["ape_align_rmse"] * 1.10
        and cand_metrics["cross_track_mean"]["median"] <= baseline["cross_track_mean"] * 1.10
        and cand_metrics["jerk_rms"]["median"] <= baseline["jerk_rms"] * 1.10
    )

    gates = {
        "localization_gate": bool(loc_gate),
        "row_gate": bool(row_gate),
        "smoothness_gate": bool(smooth_gate),
        "no_major_regression": bool(no_major_regression),
    }
    pass_count = int(loc_gate) + int(row_gate) + int(smooth_gate)
    overall_pass = bool(pass_count >= 2 and no_major_regression)
    gates["overall_pass"] = overall_pass

    summary = {
        "baseline_run": baseline_row["run_name"],
        "representative_candidate_run": representative["run_name"],
        "candidate_seed_count": len(candidate_rows),
        "gates": gates,
        "candidate_median_metrics": {k: v["median"] for k, v in cand_metrics.items()},
    }

    json_path = run_dir / "ab_delta_vs_baseline.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = run_dir / "validation_summary.md"
    with open(md_path, "w") as f:
        f.write("# A/B Validation Summary\n\n")
        f.write(f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`\n")
        f.write(f"- Baseline: `{baseline_row['run_name']}`\n")
        f.write(f"- Candidate runs: `{len(candidate_rows)}` seeds\n")
        f.write(f"- Representative qualitative run: `{representative['run_name']}`\n\n")

        f.write("## Decision Gates\n\n")
        for k, v in gates.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\n")

        f.write("## Key Metrics (Baseline vs Candidate Median)\n\n")
        f.write("| Metric | Baseline | Candidate Median | Delta |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in aggregate_rows:
            metric = row["metric"]
            b = safe_float(row["baseline"])
            m = safe_float(row["candidate_median"])
            d = safe_float(row["delta_abs"])
            f.write(f"| {metric} | {b:.6f} | {m:.6f} | {d:+.6f} |\n")
        f.write("\n")

        f.write("## Artifacts\n\n")
        f.write("- `ab_metrics_per_run.csv`\n")
        f.write("- `ab_metrics_aggregate.csv`\n")
        f.write("- `ab_delta_vs_baseline.json`\n")
        f.write("- `ab_trajectory_overlay_aligned_gt.png`\n")
        f.write("- `ab_aligned_error_panels.png`\n")
        f.write("- `ab_seed_variability.png`\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run A/B validation for SPF LiDAR option 1+2.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv" / "bin" / "python")
    parser.add_argument("--run-count", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--gps-weight", type=float, default=0.5)
    parser.add_argument("--miss-penalty", type=float, default=4.0)
    parser.add_argument("--wrong-hit-penalty", type=float, default=4.0)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    # Keep the provided python path as-is (no resolve), because `.venv/bin/python`
    # may be a symlink and we need sibling tools (`evo_ape`, `evo_rpe`) in `.venv/bin`.
    python_exec = args.python_exec.expanduser()
    if not python_exec.is_absolute():
        python_exec = (BASE_DIR / python_exec).absolute()
    scripts_dir = BASE_DIR / "scripts"

    baseline_est = baseline_dir / f"trajectory_{args.gps_weight}.tum"
    baseline_gt = baseline_dir / "gps_pose.tum"
    if not baseline_est.exists():
        raise FileNotFoundError(f"Baseline est missing: {baseline_est}")
    if not baseline_gt.exists():
        raise FileNotFoundError(f"Baseline GT missing: {baseline_gt}")

    if not (data_root / "data.csv").exists():
        raise FileNotFoundError(f"Missing dataset csv at {(data_root / 'data.csv')}")
    for sub in ("rgb", "depth", "lidar"):
        if not (data_root / sub).exists():
            raise FileNotFoundError(f"Missing dataset folder: {data_root / sub}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < args.run_count:
        raise ValueError(f"Need at least {args.run_count} seeds, got {len(seeds)}")
    seeds = seeds[: args.run_count]

    if args.require_cuda and not check_cuda_available(python_exec):
        raise RuntimeError("CUDA is required but not visible in this Python runtime.")

    evo_ape_bin = python_exec.parent / "evo_ape"
    evo_rpe_bin = python_exec.parent / "evo_rpe"
    if not evo_ape_bin.exists() or not evo_rpe_bin.exists():
        raise FileNotFoundError("evo_ape/evo_rpe not found in virtualenv bin directory.")

    rows = load_rows_from_geojson(args.geojson.resolve())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{ts}_option12_validation"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["MPLBACKEND"] = "Agg"

    # Baseline evaluation (existing files only).
    baseline_eval_dir = run_dir / "baseline_eval"
    baseline_eval_dir.mkdir(parents=True, exist_ok=True)
    baseline_row = evaluate_run(
        name="baseline_main_gpu",
        est_tum=baseline_est,
        gt_tum=baseline_gt,
        out_dir=baseline_eval_dir,
        rows=rows,
        evo_ape_bin=evo_ape_bin,
        evo_rpe_bin=evo_rpe_bin,
        env=env,
    )
    baseline_row["seed"] = -1
    baseline_row["runtime_sec"] = 0.0

    candidate_rows: List[Dict[str, float | str]] = []
    for seed in seeds:
        run_name = f"candidate_seed_{seed}"
        out_dir = run_dir / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        run_log = out_dir / "run_spf_lidar.log"
        cmd = [
            str(python_exec),
            str(scripts_dir / "spf_lidar.py"),
            "--miss-penalty",
            str(args.miss_penalty),
            "--wrong-hit-penalty",
            str(args.wrong_hit_penalty),
            "--gps-weight",
            str(args.gps_weight),
            "--seed",
            str(seed),
            "--output-folder",
            str(out_dir),
        ]
        if args.require_cuda:
            cmd.append("--require-cuda")
        runtime = run_cmd(cmd, run_log, cwd=BASE_DIR, env=env)

        est_tum = out_dir / f"trajectory_{args.gps_weight}.tum"
        gt_tum = out_dir / "gps_pose.tum"
        if not est_tum.exists() or not gt_tum.exists():
            raise FileNotFoundError(f"Run {run_name} did not produce trajectory files.")

        row = evaluate_run(
            name=run_name,
            est_tum=est_tum,
            gt_tum=gt_tum,
            out_dir=out_dir,
            rows=rows,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        row["seed"] = seed
        row["runtime_sec"] = runtime
        candidate_rows.append(row)

    # Representative run by median ape_align_rmse.
    sorted_candidates = sorted(candidate_rows, key=lambda r: float(r["ape_align_rmse"]))
    representative = sorted_candidates[len(sorted_candidates) // 2]
    rep_est = Path(str(representative["est_tum"]))
    rep_gt = Path(str(representative["gt_tum"]))

    baseline_aligned = aligned_estimate(baseline_est, baseline_gt)
    rep_aligned = aligned_estimate(rep_est, rep_gt)
    plot_overlay(baseline_aligned, rep_aligned, run_dir / "ab_trajectory_overlay_aligned_gt.png")
    plot_error_panels(baseline_aligned, rep_aligned, run_dir / "ab_aligned_error_panels.png")
    plot_seed_variability(candidate_rows, baseline_row, run_dir / "ab_seed_variability.png")

    per_run_rows = [baseline_row] + candidate_rows
    write_csv(run_dir / "ab_metrics_per_run.csv", per_run_rows)

    aggregate_rows = []
    for metric in KEY_METRICS:
        baseline_v = safe_float(baseline_row[metric])
        cand_vals = np.array([safe_float(r[metric]) for r in candidate_rows], dtype=np.float64)
        cand_median = float(np.nanmedian(cand_vals))
        cand_min = float(np.nanmin(cand_vals))
        cand_max = float(np.nanmax(cand_vals))
        cand_std = float(np.nanstd(cand_vals))
        delta_abs = cand_median - baseline_v
        if math.isfinite(baseline_v) and abs(baseline_v) > 1e-12:
            delta_rel_pct = 100.0 * delta_abs / baseline_v
        else:
            delta_rel_pct = float("nan")
        aggregate_rows.append(
            {
                "metric": metric,
                "baseline": baseline_v,
                "candidate_median": cand_median,
                "candidate_min": cand_min,
                "candidate_max": cand_max,
                "candidate_std": cand_std,
                "delta_abs": delta_abs,
                "delta_rel_pct": delta_rel_pct,
            }
        )
    write_csv(run_dir / "ab_metrics_aggregate.csv", aggregate_rows)

    summary = make_summary(run_dir, baseline_row, candidate_rows, aggregate_rows, representative)

    print(f"[INFO] Validation complete: {run_dir}")
    print(f"[INFO] Overall pass: {summary['gates']['overall_pass']}")
    print(f"[INFO] Representative run: {summary['representative_candidate_run']}")


if __name__ == "__main__":
    main()
