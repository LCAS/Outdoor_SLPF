#!/usr/bin/env python3
"""
Multi-stage SPF++ hyperparameter optimization pipeline.

Stages:
  A) Broad random search on validation split (cheap eval)
  B) Local refinement around top configs (medium eval)
  C) Full-run confirmation of best candidates (expensive eval)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import compute_metrics as cm


GLOBAL_SPACE = {
    "gps_weight": (0.05, 0.95),
    "gps_sigma": (0.2, 5.0),
    "semantic_sigma": (0.01, 0.8),
    "semantic_radius": (0.2, 1.8),
    "miss_penalty": (0.5, 10.0),
    "wrong_hit_penalty": (0.5, 10.0),
    "update_min_d": (0.01, 0.4),
    "update_min_a_deg": (1.0, 20.0),
    "ess_ratio_threshold": (0.3, 0.95),
    "resample_interval": (1, 5),  # integer
    "pose_smooth_alpha_pos": (0.1, 0.95),
    "pose_smooth_alpha_theta": (0.1, 0.95),
}

INT_PARAMS = {"resample_interval"}


def sample_params(rng: random.Random, bounds: dict) -> dict:
    out = {}
    for k, (lo, hi) in bounds.items():
        if k in INT_PARAMS:
            out[k] = int(rng.randint(int(lo), int(hi)))
        else:
            out[k] = float(rng.uniform(float(lo), float(hi)))
    return out


def local_bounds(anchor: dict, global_bounds: dict, frac: float = 0.2) -> dict:
    out = {}
    for k, (glo, ghi) in global_bounds.items():
        if k in INT_PARAMS:
            center = int(anchor[k])
            span = max(1, int(round((ghi - glo) * frac * 0.5)))
            lo = max(int(glo), center - span)
            hi = min(int(ghi), center + span)
            if lo > hi:
                lo, hi = hi, lo
            out[k] = (lo, hi)
        else:
            center = float(anchor[k])
            span = (float(ghi) - float(glo)) * frac * 0.5
            lo = max(float(glo), center - span)
            hi = min(float(ghi), center + span)
            if lo > hi:
                lo, hi = hi, lo
            out[k] = (lo, hi)
    return out


def find_estimate_file(trial_dir: Path) -> Path:
    candidates = sorted(trial_dir.glob("trajectory_*.tum"))
    if not candidates:
        raise FileNotFoundError(f"No trajectory_*.tum found in {trial_dir}")
    return candidates[0]


def compute_ate_umey_rmse(gt_tum: Path, est_tum: Path) -> float:
    gt_ts, gt_pos, gt_q = cm.read_tum_file(str(gt_tum))
    est_ts, est_pos, est_q = cm.read_tum_file(str(est_tum))
    if gt_pos is None or est_pos is None:
        raise RuntimeError("Missing trajectory/ground-truth data for ATE computation.")

    gt_interp = cm.interpolate_ground_truth(gt_ts, gt_pos, est_ts)
    est_aligned = cm.align_first_pose(est_pos, est_q, gt_interp, gt_q, mirror=False)
    s, R, t = cm.umeyama_alignment(est_aligned, gt_interp, with_scaling=True)
    est_umey = (s * (R @ est_aligned.T)).T + t
    ate = cm.compute_ate(est_umey, gt_interp)
    rmse = float(ate["rmse"])
    if not np.isfinite(rmse):
        raise RuntimeError("Non-finite ATE RMSE encountered.")
    return rmse


def run_spf_trial(
    base_dir: Path,
    out_dir: Path,
    trial_idx: int,
    stage: str,
    params: dict,
    frame_start: int,
    frame_end: int | None,
    frame_stride: int,
    max_processed_frames: int | None,
    timeout_s: int,
    save_visualizations: bool,
) -> dict:
    trial_dir = out_dir / f"{stage.lower()}_trial_{trial_idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "spf_lidar.py"),
        "--gps-weight", str(params["gps_weight"]),
        "--gps-sigma", str(params["gps_sigma"]),
        "--semantic-sigma", str(params["semantic_sigma"]),
        "--semantic-radius", str(params["semantic_radius"]),
        "--miss-penalty", str(params["miss_penalty"]),
        "--wrong-hit-penalty", str(params["wrong_hit_penalty"]),
        "--update-min-d", str(params["update_min_d"]),
        "--update-min-a-deg", str(params["update_min_a_deg"]),
        "--ess-ratio-threshold", str(params["ess_ratio_threshold"]),
        "--resample-interval", str(params["resample_interval"]),
        "--pose-smooth-alpha-pos", str(params["pose_smooth_alpha_pos"]),
        "--pose-smooth-alpha-theta", str(params["pose_smooth_alpha_theta"]),
        "--frame-start", str(frame_start),
        "--frame-stride", str(max(1, frame_stride)),
        "--output-folder", str(trial_dir),
    ]
    if frame_end is not None:
        cmd += ["--frame-end", str(frame_end)]
    if max_processed_frames is not None:
        cmd += ["--max-processed-frames", str(max_processed_frames)]
    if not save_visualizations:
        cmd += ["--no-visualizations"]

    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(base_dir),
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_s,
    )
    elapsed = time.time() - t0

    with open(trial_dir / "trial.log", "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr or "")

    base_row = {
        "stage": stage,
        "trial_id": trial_idx,
        "status": "ok" if proc.returncode == 0 else "failed",
        "runtime_s": elapsed,
        "return_code": proc.returncode,
        "ate_umey_rmse": None,
        **params,
    }

    if proc.returncode != 0:
        return base_row

    try:
        gt_tum = trial_dir / "gps_pose.tum"
        est_tum = find_estimate_file(trial_dir)
        base_row["ate_umey_rmse"] = compute_ate_umey_rmse(gt_tum, est_tum)
    except Exception:
        base_row["status"] = "failed_metric"
        base_row["ate_umey_rmse"] = None
    return base_row


def save_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_plot(rows: list[dict], out_pdf: Path) -> None:
    ok = [r for r in rows if r["status"] == "ok" and r["ate_umey_rmse"] is not None]
    if not ok:
        return
    stages = ["A", "B", "C"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in stages:
        ys = [r["ate_umey_rmse"] for r in ok if r["stage"] == s]
        xs = [s] * len(ys)
        if ys:
            ax.scatter(xs, ys, alpha=0.7, label=f"Stage {s}")
    ax.set_ylabel("ATE RMSE (Umeyama)")
    ax.set_title("SPF++ Multi-stage Tuning Results")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def get_split_indices(csv_path: Path) -> tuple[int, int, int]:
    n = len(pd.read_csv(csv_path))
    if n < 4:
        raise RuntimeError(f"Dataset too small for split: {n}")
    i50 = int(0.50 * (n - 1))
    i75 = int(0.75 * (n - 1))
    return n, i50, i75


def top_k(rows: list[dict], k: int, stage_filter: str | None = None) -> list[dict]:
    ok = [r for r in rows if r["status"] == "ok" and r["ate_umey_rmse"] is not None]
    if stage_filter is not None:
        ok = [r for r in ok if r["stage"] == stage_filter]
    ok.sort(key=lambda r: r["ate_umey_rmse"])
    return ok[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-stage SPF++ hyperparameter tuning.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stage-a-trials", type=int, default=80)
    parser.add_argument("--stage-b-trials", type=int, default=60)
    parser.add_argument("--stage-c-topk", type=int, default=5)
    parser.add_argument("--stage-b-anchor-topk", type=int, default=10)
    parser.add_argument("--local-frac", type=float, default=0.25, help="Local search span fraction of global range.")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--output-subdir", type=str, default="results/spf_lidar++/tuning_multistage")
    parser.add_argument("--dataset-csv", type=str, default="data/2025/ICRA2/data.csv")
    parser.add_argument("--run-stage-c", action="store_true", help="Run full-confirmation stage C.")
    args = parser.parse_args()

    base_dir = REPO_ROOT
    out_dir = base_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    dataset_csv = base_dir / args.dataset_csv
    n, i50, i75 = get_split_indices(dataset_csv)
    # Tune on validation block (middle-late segment), hold out final quarter.
    stage_a_start, stage_a_end = i50 + 1, i75
    stage_b_start, stage_b_end = i50 + 1, i75
    stage_c_start, stage_c_end = 0, n - 1
    holdout_start, holdout_end = i75 + 1, n - 1

    rows: list[dict] = []
    trial_counter = 0

    print(f"[PIPE] Dataset frames: {n}")
    print(f"[PIPE] Stage A/B split: [{stage_a_start}, {stage_a_end}]")
    print(f"[PIPE] Holdout split: [{holdout_start}, {holdout_end}]")

    # ---------- Stage A ----------
    print(f"[PIPE] Stage A: broad search ({args.stage_a_trials} trials)")
    for _ in range(args.stage_a_trials):
        trial_counter += 1
        params = sample_params(rng, GLOBAL_SPACE)
        row = run_spf_trial(
            base_dir=base_dir,
            out_dir=out_dir,
            trial_idx=trial_counter,
            stage="A",
            params=params,
            frame_start=stage_a_start,
            frame_end=stage_a_end,
            frame_stride=8,
            max_processed_frames=180,
            timeout_s=args.timeout_s,
            save_visualizations=False,
        )
        rows.append(row)
        save_rows(rows, out_dir / "leaderboard.csv")
        if row["status"] == "ok":
            print(f"[A:{trial_counter}] ATE={row['ate_umey_rmse']:.4f}")
        else:
            print(f"[A:{trial_counter}] {row['status']}")

    anchors = top_k(rows, args.stage_b_anchor_topk, stage_filter="A")
    if not anchors:
        raise RuntimeError("Stage A produced no valid trials.")

    # ---------- Stage B ----------
    print(f"[PIPE] Stage B: local refinement ({args.stage_b_trials} trials)")
    for i in range(args.stage_b_trials):
        trial_counter += 1
        anchor = anchors[i % len(anchors)]
        bounds = local_bounds(anchor, GLOBAL_SPACE, frac=args.local_frac)
        params = sample_params(rng, bounds)
        row = run_spf_trial(
            base_dir=base_dir,
            out_dir=out_dir,
            trial_idx=trial_counter,
            stage="B",
            params=params,
            frame_start=stage_b_start,
            frame_end=stage_b_end,
            frame_stride=6,
            max_processed_frames=260,
            timeout_s=args.timeout_s,
            save_visualizations=False,
        )
        rows.append(row)
        save_rows(rows, out_dir / "leaderboard.csv")
        if row["status"] == "ok":
            print(f"[B:{trial_counter}] ATE={row['ate_umey_rmse']:.4f}")
        else:
            print(f"[B:{trial_counter}] {row['status']}")

    # ---------- Stage C ----------
    stage_c_rows = []
    if args.run_stage_c:
        candidates = top_k(rows, args.stage_c_topk)
        print(f"[PIPE] Stage C: full confirmation ({len(candidates)} candidates)")
        for cand in candidates:
            trial_counter += 1
            params = {k: cand[k] for k in GLOBAL_SPACE.keys()}
            row = run_spf_trial(
                base_dir=base_dir,
                out_dir=out_dir,
                trial_idx=trial_counter,
                stage="C",
                params=params,
                frame_start=stage_c_start,
                frame_end=stage_c_end,
                frame_stride=4,
                max_processed_frames=None,
                timeout_s=args.timeout_s,
                save_visualizations=False,
            )
            rows.append(row)
            stage_c_rows.append(row)
            save_rows(rows, out_dir / "leaderboard.csv")
            if row["status"] == "ok":
                print(f"[C:{trial_counter}] FULL ATE={row['ate_umey_rmse']:.4f}")
            else:
                print(f"[C:{trial_counter}] {row['status']}")

    # Best selection
    preferred_pool = [r for r in rows if r["stage"] == "C" and r["status"] == "ok"] if args.run_stage_c else []
    if not preferred_pool:
        preferred_pool = [r for r in rows if r["status"] == "ok"]
    preferred_pool.sort(key=lambda r: r["ate_umey_rmse"])
    best = preferred_pool[0]

    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    summary = {
        "dataset_frames": n,
        "splits": {
            "stage_a_b": [stage_a_start, stage_a_end],
            "holdout": [holdout_start, holdout_end],
        },
        "counts": {
            "stage_a_trials": args.stage_a_trials,
            "stage_b_trials": args.stage_b_trials,
            "stage_c_trials": len(stage_c_rows),
            "total_trials": len(rows),
        },
        "best_trial": best,
    }
    with open(out_dir / "stage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_summary_plot(rows, out_dir / "tuning_multistage_summary.pdf")
    print(f"[PIPE] Done. Best ATE={best['ate_umey_rmse']:.4f} (stage {best['stage']})")
    print(f"[PIPE] Outputs: {out_dir}")


if __name__ == "__main__":
    main()

