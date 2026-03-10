#!/usr/bin/env python3
"""Fuse RTABMAP with noisy GPS (3 seeds) and compute metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def read_tum_file(path: Path) -> Trajectory:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    if not data:
        raise ValueError(f"No valid poses in {path}")

    arr = np.asarray(data, dtype=np.float64)
    order = np.argsort(arr[:, 0], kind="mergesort")
    arr = arr[order]
    rev = arr[::-1]
    _, rev_idx = np.unique(rev[:, 0], return_index=True)
    keep_last = np.sort(arr.shape[0] - 1 - rev_idx)
    arr = arr[keep_last]

    return Trajectory(
        timestamps=arr[:, 0],
        positions=arr[:, 1:4],
        quaternions=arr[:, 4:8],
    )


def write_tum(path: Path, traj: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, p, q in zip(traj.timestamps, traj.positions, traj.quaternions):
            f.write(
                f"{float(t):.9f} "
                f"{float(p[0]):.9f} {float(p[1]):.9f} {float(p[2]):.9f} "
                f"{float(q[0]):.9f} {float(q[1]):.9f} {float(q[2]):.9f} {float(q[3]):.9f}\n"
            )


def interpolate_positions(ts_src: np.ndarray, pos_src: np.ndarray, ts_q: np.ndarray) -> np.ndarray:
    out = np.zeros((len(ts_q), 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = np.interp(ts_q, ts_src, pos_src[:, i])
    return out


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scaling: bool = False):
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


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    pa = p - a
    ba = b - a
    denom = float(ba.dot(ba))
    if denom <= 1e-12:
        return float(np.linalg.norm(pa))
    t = max(0.0, min(1.0, float(pa.dot(ba) / denom)))
    proj = a + t * ba
    return float(np.linalg.norm(p - proj))


def nearest_row_and_distance(pt: np.ndarray, rows: Dict[str, np.ndarray]):
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
    import sys

    scripts_dir = BASE_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from geojson_rows import iter_projected_points  # pylint: disable=import-outside-toplevel

    records = []
    for item in iter_projected_points(path, target_crs=target_crs):
        records.append((item["x"], item["y"], item["row_id"]))
    if not records:
        raise ValueError(f"No row points found in {path}")

    all_xy = np.array([[r[0], r[1]] for r in records], dtype=np.float64)
    center = all_xy.mean(axis=0)

    rows: Dict[str, List[np.ndarray]] = {}
    for x, y, rid in records:
        rows.setdefault(rid, []).append(np.array([x - center[0], y - center[1]], dtype=np.float64))

    out: Dict[str, np.ndarray] = {}
    for rid, pts in rows.items():
        pts_arr = np.asarray(pts)
        if pts_arr.shape[0] < 2:
            continue
        sort_dim = 0 if np.ptp(pts_arr[:, 0]) >= np.ptp(pts_arr[:, 1]) else 1
        order = np.argsort(pts_arr[:, sort_dim])
        out[rid] = pts_arr[order]
    return out


def compute_row_metrics(est_aligned: np.ndarray, gt_interp: np.ndarray, rows: Dict[str, np.ndarray]):
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

    cross_track_mean = float(np.mean(cross_track_errs)) if cross_track_errs else float("nan")
    correct = [
        1.0 if (g is not None and e is not None and g == e) else 0.0
        for g, e in zip(gt_rows, est_rows)
    ]
    row_correct_fraction = float(np.mean(correct)) if correct else float("nan")
    switches = 0
    prev = None
    for rid in est_rows:
        if rid is None:
            continue
        if prev is not None and rid != prev:
            switches += 1
        prev = rid
    return {
        "cross_track_mean": cross_track_mean,
        "row_correct_fraction": row_correct_fraction,
        "row_switch_events": float(switches),
    }


def run_cmd(cmd: List[str], log_path: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.write(proc.stdout or "")
        f.write(f"\n[exit_code={proc.returncode}]\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def parse_evo_stats(path: Path):
    with zipfile.ZipFile(path, "r") as z:
        stats = json.loads(z.read("stats.json").decode("utf-8"))
    return {
        "rmse": float(stats.get("rmse", math.nan)),
        "mean": float(stats.get("mean", math.nan)),
        "median": float(stats.get("median", math.nan)),
        "max": float(stats.get("max", math.nan)),
    }


def run_evo_bundle(name: str, est_tum: Path, gt_tum: Path, out_dir: Path, evo_ape_bin: Path, evo_rpe_bin: Path):
    raw_json = out_dir / f"evo_{name}_ape_raw.zip"
    align_json = out_dir / f"evo_{name}_ape_align.zip"
    r2_json = out_dir / f"evo_{name}_rpe_2m_align.zip"
    r5_json = out_dir / f"evo_{name}_rpe_5m_align.zip"
    r10_json = out_dir / f"evo_{name}_rpe_10m_align.zip"

    run_cmd(
        [str(evo_ape_bin), "tum", str(gt_tum), str(est_tum), "--save_results", str(raw_json), "--no_warnings"],
        out_dir / f"{name}_ape_raw.log",
    )
    run_cmd(
        [str(evo_ape_bin), "tum", str(gt_tum), str(est_tum), "-a", "--save_results", str(align_json), "--no_warnings"],
        out_dir / f"{name}_ape_align.log",
    )
    for d, out_json in [(2, r2_json), (5, r5_json), (10, r10_json)]:
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
            out_dir / f"{name}_rpe_{d}m_align.log",
        )

    raw = parse_evo_stats(raw_json)
    align = parse_evo_stats(align_json)
    r2 = parse_evo_stats(r2_json)
    r5 = parse_evo_stats(r5_json)
    r10 = parse_evo_stats(r10_json)
    return {
        "ape_raw_rmse": raw["rmse"],
        "ape_align_rmse": align["rmse"],
        "rpe_2m_align_rmse": r2["rmse"],
        "rpe_5m_align_rmse": r5["rmse"],
        "rpe_10m_align_rmse": r10["rmse"],
        "ape_align_mean": align["mean"],
        "ape_align_median": align["median"],
        "ape_align_max": align["max"],
    }


def fuse_kalman(
    rtab: Trajectory,
    gps: Trajectory,
    *,
    rtab_std: float,
    gps_std: float,
    process_std: float,
) -> Trajectory:
    gps_interp = interpolate_positions(gps.timestamps, gps.positions, rtab.timestamps)
    z_rtab = rtab.positions[:, :2]
    z_gps = gps_interp[:, :2]

    ts = rtab.timestamps
    n = len(ts)
    if n == 0:
        raise ValueError("Empty RTAB trajectory")

    x = np.zeros(4, dtype=np.float64)
    P = np.eye(4, dtype=np.float64)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    I4 = np.eye(4, dtype=np.float64)

    R_rtab = (max(rtab_std, 1e-6) ** 2) * np.eye(2, dtype=np.float64)
    R_gps = (max(gps_std, 1e-6) ** 2) * np.eye(2, dtype=np.float64)
    qacc2 = max(process_std, 1e-6) ** 2

    x[:2] = z_rtab[0]
    if n > 1:
        dt0 = max(1e-3, float(ts[1] - ts[0]))
        x[2:] = (z_rtab[1] - z_rtab[0]) / dt0

    fused_xy = np.zeros((n, 2), dtype=np.float64)
    for k in range(n):
        if k > 0:
            dt = max(1e-3, float(ts[k] - ts[k - 1]))
            F = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            Q = qacc2 * np.array(
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

        for z, R in ((z_rtab[k], R_rtab), (z_gps[k], R_gps)):
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ y)
            P = (I4 - K @ H) @ P

        fused_xy[k] = x[:2]

    out_pos = np.array(rtab.positions, copy=True)
    out_pos[:, :2] = fused_xy
    return Trajectory(
        timestamps=np.array(rtab.timestamps, copy=True),
        positions=out_pos,
        quaternions=np.array(rtab.quaternions, copy=True),
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTABMAP + NoisyGPS 3-seed Kalman fusion and metrics.")
    parser.add_argument(
        "--rtab-tum",
        type=Path,
        default=BASE_DIR / "results" / "rtabmap" / "rgbd_run1_3runs" / "run1" / "rtabmap" / "rgbd" / "tum1" / "rtabmap_rgbd_filtered.tum",
    )
    parser.add_argument(
        "--gt-tum",
        type=Path,
        default=BASE_DIR / "results" / "rtabmap" / "rgbd_run1_3runs" / "run1" / "rtabmap" / "rgbd" / "tum1" / "gps_pose.tum",
    )
    parser.add_argument("--noisy-gps-dir", type=Path, default=BASE_DIR / "results" / "noisy_gps")
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results" / "rtabmap" / "rgbd_ngps_3runs_metrics")
    parser.add_argument("--geojson", type=Path, default=BASE_DIR / "data" / "riseholme_poles_trunk.geojson")
    parser.add_argument("--rtab-std", type=float, default=0.35)
    parser.add_argument("--gps-std", type=float, default=1.8)
    parser.add_argument("--process-std", type=float, default=0.8)
    args = parser.parse_args()

    rtab_tum = args.rtab_tum.expanduser()
    gt_tum = args.gt_tum.expanduser()
    rtab_tum = first_existing(
        rtab_tum,
        BASE_DIR / "results" / "rtabmap" / "rgbd" / "tum1" / "rtabmap_rgbd_filtered.tum",
    )
    gt_tum = first_existing(
        gt_tum,
        BASE_DIR / "results" / "rtabmap" / "rgbd" / "tum1" / "gps_pose.tum",
    )
    noisy_dir = args.noisy_gps_dir.expanduser()
    out_dir = args.output_dir.expanduser()
    geojson = args.geojson.expanduser()
    seeds = parse_int_list(args.seeds)

    if not rtab_tum.exists():
        raise FileNotFoundError(f"Missing RTAB trajectory: {rtab_tum}")
    if not gt_tum.exists():
        raise FileNotFoundError(f"Missing GT trajectory: {gt_tum}")
    if not geojson.exists():
        raise FileNotFoundError(f"Missing geojson: {geojson}")

    evo_ape = BASE_DIR / ".venv" / "bin" / "evo_ape"
    evo_rpe = BASE_DIR / ".venv" / "bin" / "evo_rpe"
    if not evo_ape.exists() or not evo_rpe.exists():
        raise FileNotFoundError("evo_ape/evo_rpe not found in .venv/bin")

    rows_map = load_rows_from_geojson(geojson)
    rtab = read_tum_file(rtab_tum)
    gt = read_tum_file(gt_tum)

    fused_dir = out_dir / "fused_tum"
    eval_dir = out_dir / "evo"
    fused_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    per_rows: List[Dict[str, object]] = []

    for seed in seeds:
        noisy_tum = noisy_dir / f"noisy_gps_seed_{seed}.tum"
        if not noisy_tum.exists():
            raise FileNotFoundError(f"Missing noisy GPS for seed {seed}: {noisy_tum}")

        gps = read_tum_file(noisy_tum)
        fused = fuse_kalman(
            rtab,
            gps,
            rtab_std=float(args.rtab_std),
            gps_std=float(args.gps_std),
            process_std=float(args.process_std),
        )

        run_name = f"rtab_ngps_s{seed}"
        fused_tum = fused_dir / f"{run_name}.tum"
        write_tum(fused_tum, fused)

        evo = run_evo_bundle(run_name, fused_tum, gt_tum, eval_dir, evo_ape, evo_rpe)
        gt_interp = interpolate_positions(gt.timestamps, gt.positions, fused.timestamps)
        scale, rot, trans = umeyama_alignment(fused.positions, gt_interp, with_scaling=False)
        est_aligned = apply_transform(fused.positions, scale, rot, trans)
        row_metrics = compute_row_metrics(est_aligned, gt_interp, rows_map)

        per_rows.append(
            {
                "run": run_name,
                "seed": int(seed),
                "poses_used": int(len(fused.timestamps)),
                "fused_tum": str(fused_tum),
                "noisy_gps_tum": str(noisy_tum),
                **evo,
                **row_metrics,
            }
        )

    per_csv = out_dir / "rtab_ngps_metrics_3runs_per_run.csv"
    write_csv(per_csv, per_rows)

    metric_cols = [
        "ape_raw_rmse",
        "ape_align_rmse",
        "rpe_2m_align_rmse",
        "rpe_5m_align_rmse",
        "rpe_10m_align_rmse",
        "ape_align_mean",
        "ape_align_median",
        "ape_align_max",
        "cross_track_mean",
        "row_correct_fraction",
        "row_switch_events",
    ]
    agg = {
        "group": "rtab_ngps",
        "n_runs": len(per_rows),
        "rtab_tum": str(rtab_tum),
        "gt_tum": str(gt_tum),
        "noisy_gps_dir": str(noisy_dir),
    }
    for col in metric_cols:
        vals = np.asarray([float(r[col]) for r in per_rows], dtype=np.float64)
        agg[f"{col}_mean"] = float(np.mean(vals))
        agg[f"{col}_std"] = float(np.std(vals))
        agg[f"{col}_min"] = float(np.min(vals))
        agg[f"{col}_max"] = float(np.max(vals))

    agg_csv = out_dir / "rtab_ngps_metrics_3runs_aggregate.csv"
    write_csv(agg_csv, [agg])

    config_json = out_dir / "run_config.json"
    config_json.write_text(
        json.dumps(
            {
                "rtab_std": float(args.rtab_std),
                "gps_std": float(args.gps_std),
                "process_std": float(args.process_std),
                "seeds": seeds,
                "geojson": str(geojson),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Per-run metrics: {per_csv}")
    print(f"[INFO] Aggregate metrics: {agg_csv}")
    for row in per_rows:
        print(
            f"[INFO] {row['run']}: "
            f"ape_align_rmse={float(row['ape_align_rmse']):.6f}, "
            f"rpe_5m_align_rmse={float(row['rpe_5m_align_rmse']):.6f}, "
            f"cross_track_mean={float(row['cross_track_mean']):.6f}"
        )


if __name__ == "__main__":
    main()
