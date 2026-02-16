#!/usr/bin/env python3
"""
Random-search tuner for SPF++ hyperparameters using ATE RMSE (Umeyama-aligned)
as objective.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import compute_metrics as cm


def sample_params(rng: random.Random) -> dict:
    return {
        "gps_weight": rng.uniform(0.1, 0.9),
        "gps_sigma": rng.uniform(0.3, 4.0),
        "semantic_sigma": rng.uniform(0.01, 0.5),
        "semantic_radius": rng.uniform(0.3, 1.5),
        "miss_penalty": rng.uniform(1.0, 8.0),
        "wrong_hit_penalty": rng.uniform(1.0, 8.0),
        "update_min_d": rng.uniform(0.02, 0.3),
        "update_min_a_deg": rng.uniform(2.0, 15.0),
        "ess_ratio_threshold": rng.uniform(0.4, 0.9),
        "resample_interval": int(rng.randint(1, 4)),
        "pose_smooth_alpha_pos": rng.uniform(0.2, 0.9),
        "pose_smooth_alpha_theta": rng.uniform(0.2, 0.9),
    }


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
    return float(ate["rmse"])


def find_estimate_file(trial_dir: Path) -> Path:
    candidates = sorted(trial_dir.glob("trajectory_*.tum"))
    if not candidates:
        raise FileNotFoundError(f"No trajectory_*.tum found in {trial_dir}")
    return candidates[0]


def run_trial(
    trial_id: int,
    params: dict,
    args: argparse.Namespace,
    base_dir: Path,
) -> dict:
    trial_dir = base_dir / args.output_subdir / f"trial_{trial_id:03d}"
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
        "--frame-stride", str(args.frame_stride),
        "--frame-start", str(args.frame_start),
        "--output-folder", str(trial_dir),
        "--no-visualizations",
    ]
    if args.frame_end is not None:
        cmd += ["--frame-end", str(args.frame_end)]
    if args.max_processed_frames is not None:
        cmd += ["--max-processed-frames", str(args.max_processed_frames)]

    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(base_dir),
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
        env=env,
    )
    elapsed = time.time() - start

    log_path = trial_dir / "trial.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr or "")

    if proc.returncode != 0:
        return {
            "trial_id": trial_id,
            "status": "failed",
            "ate_umey_rmse": None,
            "runtime_s": elapsed,
            "return_code": proc.returncode,
            **params,
        }

    gt_tum = trial_dir / "gps_pose.tum"
    est_tum = find_estimate_file(trial_dir)
    ate_rmse = compute_ate_umey_rmse(gt_tum, est_tum)
    if not np.isfinite(ate_rmse):
        return {
            "trial_id": trial_id,
            "status": "failed_metric",
            "ate_umey_rmse": None,
            "runtime_s": elapsed,
            "return_code": 0,
            **params,
        }

    return {
        "trial_id": trial_id,
        "status": "ok",
        "ate_umey_rmse": ate_rmse,
        "runtime_s": elapsed,
        "return_code": 0,
        **params,
    }


def write_trials_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary_pdf(rows: list[dict], out_pdf: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("ate_umey_rmse") is not None]
    if not ok_rows:
        return
    trial_ids = [r["trial_id"] for r in ok_rows]
    ates = [r["ate_umey_rmse"] for r in ok_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_ids, ates, marker="o", linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("ATE RMSE (Umeyama)")
    ax.set_title("SPF++ Hyperparameter Search")
    ax.grid(True, alpha=0.3)
    best_idx = int(np.argmin(ates))
    ax.scatter([trial_ids[best_idx]], [ates[best_idx]], color="red", s=80, label="Best")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune SPF++ parameters with ATE RMSE objective.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random-search trials.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for search.")
    parser.add_argument("--timeout-s", type=int, default=3600, help="Per-trial timeout (seconds).")
    parser.add_argument("--frame-start", type=int, default=0, help="First frame index to evaluate.")
    parser.add_argument("--frame-end", type=int, default=900, help="Last frame index to evaluate.")
    parser.add_argument("--frame-stride", type=int, default=8, help="Evaluate every Nth frame.")
    parser.add_argument("--max-processed-frames", type=int, default=180, help="Hard cap on processed frames per trial.")
    parser.add_argument("--output-subdir", type=str, default="results/spf_lidar++/tuning", help="Relative output folder for tuning runs.")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    best = None

    for trial_id in range(1, args.trials + 1):
        params = sample_params(rng)
        print(f"[TUNE] Trial {trial_id}/{args.trials} ...")
        row = run_trial(trial_id, params, args, base_dir)
        rows.append(row)
        if row["status"] == "ok":
            print(f"[TUNE] Trial {trial_id}: ATE RMSE={row['ate_umey_rmse']:.4f}")
            if best is None or row["ate_umey_rmse"] < best["ate_umey_rmse"]:
                best = row
        else:
            print(f"[TUNE] Trial {trial_id}: FAILED (return_code={row['return_code']})")

        write_trials_csv(rows, out_dir / "tuning_trials.csv")

    if best is not None:
        with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"[TUNE] Best trial={best['trial_id']} ATE RMSE={best['ate_umey_rmse']:.4f}")
    else:
        print("[TUNE] No successful trials.")

    write_summary_pdf(rows, out_dir / "tuning_summary.pdf")
    print(f"[TUNE] Results written under: {out_dir}")


if __name__ == "__main__":
    main()
