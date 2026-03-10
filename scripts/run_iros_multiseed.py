#!/usr/bin/env python3
"""
Run IROS benchmark as a multi-seed study (3 seeds by default).

This script:
1. Regenerates SPF and SLPF trajectories for each seed.
2. Evaluates deterministic baselines (Noisy GPS / AMCL / RTAB-Map) per seed.
3. Computes aligned EVO + row + smoothness metrics with the same utilities used
   by run_ab_validation.py.
4. Exports per-seed + aggregate CSVs and a compact summary plot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_ab_validation import (
    BASE_DIR,
    KEY_METRICS,
    check_cuda_available,
    evaluate_run,
    interpolate_positions,
    load_rows_from_geojson,
    read_tum_file,
    run_cmd,
)


DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
DEFAULT_SEEDS = "11,22,33"
DEFAULT_AMCL_NGPS_AMCL_STD = 0.35
DEFAULT_AMCL_NGPS_GPS_STD = 1.8
DEFAULT_AMCL_NGPS_PROCESS_STD = 0.8

# Static baselines copied into each seed folder for uniform evaluation.
FIXED_BASELINES = {
    "amcl": {
        "est": BASE_DIR / "results" / "iros" / "amcl" / "tum1" / "amcl_pose.tum",
        "gt": BASE_DIR / "results" / "iros" / "amcl" / "tum1" / "gps_pose.tum",
    },
    "rtab_rgbd": {
        "est": BASE_DIR / "results" / "iros" / "rtabmap" / "rgbd" / "tum1" / "rtabmap_rgbd_filtered.tum",
        "gt": BASE_DIR / "results" / "iros" / "rtabmap" / "rgbd" / "tum1" / "gps_pose.tum",
    },
    "rtab_rgb": {
        "est": BASE_DIR / "results" / "iros" / "rtabmap" / "rgb" / "tum1" / "rtabmap_rgb_filtered.tum",
        "gt": BASE_DIR / "results" / "iros" / "rtabmap" / "rgb" / "tum1" / "gps_pose.tum",
    },
}
NOISY_GPS_GT_SOURCE = BASE_DIR / "results" / "iros" / "ngps_only" / "gps_pose.tum"
NOISY_GPS_SCRIPT = BASE_DIR / "scripts" / "degrade_gps_vineyard.py"


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(v, default=float("nan")) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_by_method(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    metric_keys = [
        "ape_raw_rmse",
        "ape_align_rmse",
        "rpe_2m_align_rmse",
        "rpe_5m_align_rmse",
        "rpe_10m_align_rmse",
        *KEY_METRICS[4:],  # row + smoothness terms
    ]

    agg_rows: List[Dict[str, object]] = []
    for method, group in by_method.items():
        out: Dict[str, object] = {
            "method": method,
            "n_runs": len(group),
            "runtime_sec_mean": float(np.mean([safe_float(r.get("runtime_sec"), 0.0) for r in group])),
            "runtime_sec_std": float(np.std([safe_float(r.get("runtime_sec"), 0.0) for r in group])),
        }
        for metric in metric_keys:
            vals = np.asarray([safe_float(r.get(metric)) for r in group], dtype=np.float64)
            valid = vals[np.isfinite(vals)]
            out[f"{metric}_mean"] = float(np.mean(valid)) if valid.size else float("nan")
            out[f"{metric}_std"] = float(np.std(valid)) if valid.size else float("nan")
            out[f"{metric}_median"] = float(np.median(valid)) if valid.size else float("nan")
        agg_rows.append(out)

    agg_rows.sort(key=lambda r: str(r["method"]))
    return agg_rows


def plot_multiseed_summary(agg_rows: List[Dict[str, object]], out_path: Path):
    if not agg_rows:
        return

    # If fused AMCL+GPS is available, hide plain AMCL in the summary figure.
    use_amcl_gps = any(str(r.get("method")) == "AMCL+NGPS" for r in agg_rows)
    rows_for_plot = [
        r for r in agg_rows
        if not (use_amcl_gps and str(r.get("method")) == "amcl")
    ]

    method_order = ["slpf", "spf", "AMCL+NGPS", "ngps", "amcl", "rtab_rgbd", "rtab_rgb"]
    rows_sorted = sorted(rows_for_plot, key=lambda r: method_order.index(r["method"]) if r["method"] in method_order else 999)
    label_map = {"AMCL+NGPS": "amcl+GPS"}
    labels = [label_map.get(str(r["method"]), str(r["method"])) for r in rows_sorted]
    x = np.arange(len(labels))

    panels = [
        ("ape_align_rmse_mean", "ape_align_rmse_std", "APE aligned RMSE (m)", True),
        ("rpe_5m_align_rmse_mean", "rpe_5m_align_rmse_std", "RPE 5m aligned RMSE (m)", True),
        ("cross_track_mean_mean", "cross_track_mean_std", "Cross-track mean (m)", True),
        ("row_correct_fraction_mean", "row_correct_fraction_std", "Row correct fraction", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, (mkey, skey, title, lower_better) in zip(axes, panels):
        means = np.asarray([safe_float(r.get(mkey)) for r in rows_sorted], dtype=np.float64)
        stds = np.asarray([safe_float(r.get(skey)) for r in rows_sorted], dtype=np.float64)
        bars = ax.bar(x, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.9)

        if np.any(np.isfinite(means)):
            best_idx = int(np.nanargmin(means)) if lower_better else int(np.nanargmax(means))
            bars[best_idx].set_color("tab:green")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("IROS Multi-seed Benchmark (mean ± std over seeds)", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_spf_cmd(python_exec: Path, spf_script: Path, seed: int, out_dir: Path, require_cuda: bool, max_frames: int | None):
    # Legacy-style SPF approximation (semantic+GPS, no corridor/background extras).
    cmd = [
        str(python_exec),
        str(spf_script),
        "--miss-penalty", "4.0",
        "--wrong-hit-penalty", "4.0",
        "--gps-weight", "0.5",
        "--seed", str(seed),
        "--output-folder", str(out_dir),
        "--frame-stride", "5",
        "--semantic-sigma", "0.20",
        "--gps-sigma", "1.1",
        "--particle-count", "300",
        "--disable-corridor",
        "--disable-background",
        "--disable-dynamic-gps-weight",
        "--disable-pose-smoothing",
        "--no-visualization",
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def build_slpf_cmd(python_exec: Path, spf_script: Path, seed: int, out_dir: Path, require_cuda: bool, max_frames: int | None):
    # Frozen IROS SLPF (ours) configuration.
    cmd = [
        str(python_exec),
        str(spf_script),
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
        "--no-visualization",
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if require_cuda:
        cmd.append("--require-cuda")
    return cmd


def copy_fixed_tum(src_est: Path, src_gt: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    est = dst_dir / "trajectory_0.5.tum"
    gt = dst_dir / "gps_pose.tum"
    shutil.copy2(src_est, est)
    shutil.copy2(src_gt, gt)
    return est, gt


def build_noisy_gps_cmd(python_exec: Path, seed: int, gt_tum: Path, out_tum: Path):
    return [
        str(python_exec),
        str(NOISY_GPS_SCRIPT),
        str(gt_tum),
        str(out_tum),
        "--seed",
        str(seed),
    ]


def write_tum(path: Path, timestamps: np.ndarray, positions: np.ndarray, quaternions: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, p, q in zip(timestamps, positions, quaternions):
            f.write(
                f"{float(t)} {float(p[0])} {float(p[1])} {float(p[2])} "
                f"{float(q[0])} {float(q[1])} {float(q[2])} {float(q[3])}\n"
            )


def build_amcl_ngps_fused_tum(
    amcl_est: Path,
    ngps_est: Path,
    out_est: Path,
    *,
    amcl_pos_std: float,
    gps_pos_std: float,
    process_accel_std: float,
):
    """
    AMCL+NGPS: constant-velocity Kalman fusion of AMCL and noisy GPS positions.

    State x = [px, py, vx, vy]^T
    Measurement model for both sensors: z = [px, py]^T
    Sequential updates: AMCL update, then GPS update at each time step.
    Orientation is kept from AMCL.
    """
    amcl = read_tum_file(amcl_est)
    ngps = read_tum_file(ngps_est)
    ngps_interp = interpolate_positions(ngps.timestamps, ngps.positions, amcl.timestamps)

    ts = amcl.timestamps
    z_amcl = amcl.positions[:, :2]
    z_gps = ngps_interp[:, :2]
    n = len(ts)
    if n == 0:
        raise ValueError("Empty AMCL trajectory")

    I4 = np.eye(4, dtype=np.float64)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    R_amcl = (max(amcl_pos_std, 1e-6) ** 2) * np.eye(2, dtype=np.float64)
    R_gps = (max(gps_pos_std, 1e-6) ** 2) * np.eye(2, dtype=np.float64)
    q = max(process_accel_std, 1e-6) ** 2

    x = np.zeros(4, dtype=np.float64)
    x[:2] = z_amcl[0]
    if n > 1:
        dt0 = max(float(ts[1] - ts[0]), 1e-3)
        x[2:] = (z_amcl[1] - z_amcl[0]) / dt0
    P = np.diag([1.0, 1.0, 2.0, 2.0]).astype(np.float64)
    fused_xy = np.zeros((n, 2), dtype=np.float64)

    for k in range(n):
        if k > 0:
            dt = max(float(ts[k] - ts[k - 1]), 1e-3)
            F = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            Q = q * np.array(
                [
                    [0.25 * dt**4, 0.0, 0.5 * dt**3, 0.0],
                    [0.0, 0.25 * dt**4, 0.0, 0.5 * dt**3],
                    [0.5 * dt**3, 0.0, dt**2, 0.0],
                    [0.0, 0.5 * dt**3, 0.0, dt**2],
                ],
                dtype=np.float64,
            )
            x = F @ x
            P = F @ P @ F.T + Q

        for z, R in ((z_amcl[k], R_amcl), (z_gps[k], R_gps)):
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ y)
            P = (I4 - K @ H) @ P

        fused_xy[k] = x[:2]

    fused_pos = np.array(amcl.positions, copy=True)
    fused_pos[:, :2] = fused_xy
    write_tum(out_est, ts, fused_pos, amcl.quaternions)


def main():
    parser = argparse.ArgumentParser(description="Run IROS comparison with 3-seed protocol.")
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv" / "bin" / "python")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--amcl-ngps-amcl-std", type=float, default=DEFAULT_AMCL_NGPS_AMCL_STD)
    parser.add_argument("--amcl-ngps-gps-std", type=float, default=DEFAULT_AMCL_NGPS_GPS_STD)
    parser.add_argument("--amcl-ngps-process-std", type=float, default=DEFAULT_AMCL_NGPS_PROCESS_STD)
    args = parser.parse_args()

    python_exec = args.python_exec.expanduser()
    if not python_exec.is_absolute():
        python_exec = (BASE_DIR / python_exec).resolve()

    amcl_ngps_amcl_std = float(max(1e-6, args.amcl_ngps_amcl_std))
    amcl_ngps_gps_std = float(max(1e-6, args.amcl_ngps_gps_std))
    amcl_ngps_process_std = float(max(1e-6, args.amcl_ngps_process_std))
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    if args.require_cuda and not check_cuda_available(python_exec):
        raise RuntimeError("CUDA is required but not visible in the selected Python runtime.")

    evo_ape_bin = python_exec.parent / "evo_ape"
    evo_rpe_bin = python_exec.parent / "evo_rpe"
    if not evo_ape_bin.exists() or not evo_rpe_bin.exists():
        raise FileNotFoundError("evo_ape/evo_rpe not found in virtualenv bin directory.")

    if not NOISY_GPS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing noisy GPS generator script: {NOISY_GPS_SCRIPT}")
    if not NOISY_GPS_GT_SOURCE.exists():
        raise FileNotFoundError(f"Missing noisy GPS GT source: {NOISY_GPS_GT_SOURCE}")

    for method, paths in FIXED_BASELINES.items():
        if not paths["est"].exists() or not paths["gt"].exists():
            raise FileNotFoundError(f"Missing deterministic baseline input for {method}: {paths}")

    rows_map = load_rows_from_geojson(args.geojson.resolve())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / f"{timestamp}_multiseed_main"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    scripts_dir = BASE_DIR / "scripts"
    spf_script = scripts_dir / "spf_lidar.py"

    per_seed_rows: List[Dict[str, object]] = []
    protocol = {
        "run_dir": str(run_dir),
        "seeds": seeds,
        "require_cuda": bool(args.require_cuda),
        "cuda_visible_devices": args.cuda_visible_devices,
        "amcl_ngps": {
            "amcl_std_m": amcl_ngps_amcl_std,
            "gps_std_m": amcl_ngps_gps_std,
            "process_accel_std_mps2": amcl_ngps_process_std,
        },
        "commands": [],
    }

    for seed in seeds:
        # 1) SPF (legacy-style config, regenerated)
        spf_dir = run_dir / "spf" / f"seed_{seed}"
        spf_dir.mkdir(parents=True, exist_ok=True)
        spf_cmd = build_spf_cmd(python_exec, spf_script, seed, spf_dir, args.require_cuda, args.max_frames)
        spf_runtime = run_cmd(spf_cmd, spf_dir / "run_spf.log", cwd=BASE_DIR, env=env)
        spf_est = spf_dir / "trajectory_0.5.tum"
        spf_gt = spf_dir / "gps_pose.tum"
        if not spf_est.exists() or not spf_gt.exists():
            raise FileNotFoundError(f"Missing SPF outputs for seed {seed}: {spf_dir}")

        spf_metrics = evaluate_run(
            name=f"spf_seed_{seed}",
            est_tum=spf_est,
            gt_tum=spf_gt,
            out_dir=spf_dir / "eval",
            rows=rows_map,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        spf_metrics.update({"method": "spf", "seed": seed, "runtime_sec": spf_runtime})
        per_seed_rows.append(spf_metrics)
        protocol["commands"].append({"method": "spf", "seed": seed, "command": spf_cmd, "runtime_sec": spf_runtime})

        # 2) SLPF (ours, regenerated)
        slpf_dir = run_dir / "slpf" / f"seed_{seed}"
        slpf_dir.mkdir(parents=True, exist_ok=True)
        slpf_cmd = build_slpf_cmd(python_exec, spf_script, seed, slpf_dir, args.require_cuda, args.max_frames)
        slpf_runtime = run_cmd(slpf_cmd, slpf_dir / "run_slpf.log", cwd=BASE_DIR, env=env)
        slpf_est = slpf_dir / "trajectory_0.5.tum"
        slpf_gt = slpf_dir / "gps_pose.tum"
        if not slpf_est.exists() or not slpf_gt.exists():
            raise FileNotFoundError(f"Missing SLPF outputs for seed {seed}: {slpf_dir}")

        slpf_metrics = evaluate_run(
            name=f"slpf_seed_{seed}",
            est_tum=slpf_est,
            gt_tum=slpf_gt,
            out_dir=slpf_dir / "eval",
            rows=rows_map,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        slpf_metrics.update({"method": "slpf", "seed": seed, "runtime_sec": slpf_runtime})
        per_seed_rows.append(slpf_metrics)
        protocol["commands"].append({"method": "slpf", "seed": seed, "command": slpf_cmd, "runtime_sec": slpf_runtime})

        # 3) Noisy GPS baseline generated from GT (seed-specific).
        ngps_dir = run_dir / "ngps" / f"seed_{seed}"
        ngps_dir.mkdir(parents=True, exist_ok=True)
        ngps_est = ngps_dir / "trajectory_0.5.tum"
        ngps_gt = ngps_dir / "gps_pose.tum"
        shutil.copy2(NOISY_GPS_GT_SOURCE, ngps_gt)
        ngps_cmd = build_noisy_gps_cmd(
            python_exec=python_exec,
            seed=seed,
            gt_tum=ngps_gt,
            out_tum=ngps_est,
        )
        ngps_runtime = run_cmd(ngps_cmd, ngps_dir / "run_ngps.log", cwd=BASE_DIR, env=env)
        if not ngps_est.exists():
            raise FileNotFoundError(f"Missing generated NGPS trajectory for seed {seed}: {ngps_est}")

        ngps_metrics = evaluate_run(
            name=f"ngps_seed_{seed}",
            est_tum=ngps_est,
            gt_tum=ngps_gt,
            out_dir=ngps_dir / "eval",
            rows=rows_map,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        ngps_metrics.update({"method": "ngps", "seed": seed, "runtime_sec": ngps_runtime})
        per_seed_rows.append(ngps_metrics)
        protocol["commands"].append(
            {
                "method": "ngps",
                "seed": seed,
                "command": ngps_cmd,
                "runtime_sec": ngps_runtime,
            }
        )

        # 4) AMCL+NGPS fused baseline, using seed-specific generated noisy GPS.
        amcl_ngps_dir = run_dir / "amcl_ngps" / f"seed_{seed}"
        amcl_ngps_dir.mkdir(parents=True, exist_ok=True)
        amcl_ngps_est = amcl_ngps_dir / "trajectory_0.5.tum"
        amcl_ngps_gt = amcl_ngps_dir / "gps_pose.tum"
        shutil.copy2(FIXED_BASELINES["amcl"]["gt"], amcl_ngps_gt)
        build_amcl_ngps_fused_tum(
            amcl_est=FIXED_BASELINES["amcl"]["est"],
            ngps_est=ngps_est,
            out_est=amcl_ngps_est,
            amcl_pos_std=amcl_ngps_amcl_std,
            gps_pos_std=amcl_ngps_gps_std,
            process_accel_std=amcl_ngps_process_std,
        )
        amcl_ngps_metrics = evaluate_run(
            name=f"amcl_ngps_seed_{seed}",
            est_tum=amcl_ngps_est,
            gt_tum=amcl_ngps_gt,
            out_dir=amcl_ngps_dir / "eval",
            rows=rows_map,
            evo_ape_bin=evo_ape_bin,
            evo_rpe_bin=evo_rpe_bin,
            env=env,
        )
        amcl_ngps_metrics.update({"method": "AMCL+NGPS", "seed": seed, "runtime_sec": 0.0})
        per_seed_rows.append(amcl_ngps_metrics)
        protocol["commands"].append(
            {
                    "method": "AMCL+NGPS",
                    "seed": seed,
                    "command": [
                    (
                        "kalman_fusion amcl+ngps "
                        f"amcl_std={amcl_ngps_amcl_std} "
                        f"gps_std={amcl_ngps_gps_std} "
                        f"process_accel_std={amcl_ngps_process_std}"
                    ),
                    f"source_amcl={FIXED_BASELINES['amcl']['est']}",
                    f"source_ngps={ngps_est}",
                    f"gt={FIXED_BASELINES['amcl']['gt']}",
                    ],
                    "runtime_sec": 0.0,
                }
        )

        # 5) Fixed baselines (copied per seed for uniform eval protocol)
        for method, paths in FIXED_BASELINES.items():
            method_dir = run_dir / method / f"seed_{seed}"
            est_tum, gt_tum = copy_fixed_tum(paths["est"], paths["gt"], method_dir)
            metrics = evaluate_run(
                name=f"{method}_seed_{seed}",
                est_tum=est_tum,
                gt_tum=gt_tum,
                out_dir=method_dir / "eval",
                rows=rows_map,
                evo_ape_bin=evo_ape_bin,
                evo_rpe_bin=evo_rpe_bin,
                env=env,
            )
            metrics.update({"method": method, "seed": seed, "runtime_sec": 0.0})
            per_seed_rows.append(metrics)
            protocol["commands"].append(
                {
                    "method": method,
                    "seed": seed,
                    "command": [f"copy {paths['est']} -> {est_tum}", f"copy {paths['gt']} -> {gt_tum}"],
                    "runtime_sec": 0.0,
                }
            )

    # Deterministic output order
    method_order = ["slpf", "spf", "AMCL+NGPS", "ngps", "amcl", "rtab_rgbd", "rtab_rgb"]
    per_seed_rows.sort(key=lambda r: (method_order.index(str(r["method"])) if str(r["method"]) in method_order else 999, int(r["seed"])))

    write_csv(run_dir / "trajectory_metrics_multiseed_per_seed.csv", per_seed_rows)
    agg_rows = aggregate_by_method(per_seed_rows)
    write_csv(run_dir / "trajectory_metrics_multiseed_aggregate.csv", agg_rows)
    plot_multiseed_summary(agg_rows, run_dir / "trajectory_metrics_multiseed_summary.pdf")
    (run_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2))

    print(f"[INFO] Multi-seed IROS run complete: {run_dir}")
    best = min(agg_rows, key=lambda r: safe_float(r.get("ape_align_rmse_mean"), 1e9))
    print(f"[INFO] Best method by APE aligned mean: {best['method']} ({safe_float(best['ape_align_rmse_mean']):.6f})")


if __name__ == "__main__":
    main()
