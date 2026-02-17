#!/usr/bin/env python3
"""
Run SPF++ ablation study for IROS paper.

- Regenerates SPF++ baseline (full) and ablations on fixed seeds.
- Evaluates each run with aligned EVO + row + smoothness metrics.
- Exports CSV, plot, LaTeX table, protocol JSON, and non-wall formulation snippet.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    aligned_estimate,
    check_cuda_available,
    evaluate_run,
    load_rows_from_geojson,
    run_cmd,
)


DEFAULT_OUTPUT_ROOT = BASE_DIR / "results" / "iros_ablation"
DEFAULT_GEOJSON = BASE_DIR / "data" / "riseholme_poles_trunk.geojson"
DEFAULT_SEEDS = "11,22,33"

VARIANT_ORDER = [
    "full",
    "no_gps",
    "no_semantic",
    "no_corridor",
    "non_wall_points",
    "no_background",
    "static_gps_weight",
    "no_pose_smoothing",
]

VARIANT_ARGS = {
    "full": [],
    "no_gps": ["--disable-gps"],
    "no_semantic": ["--disable-semantic"],
    "no_corridor": ["--disable-corridor"],
    "non_wall_points": ["--semantic-model", "point"],
    "no_background": ["--disable-background"],
    "static_gps_weight": ["--disable-dynamic-gps-weight"],
    "no_pose_smoothing": ["--disable-pose-smoothing"],
}

CORE_METRICS = [
    "ape_align_rmse",
    "rpe_2m_align_rmse",
    "rpe_5m_align_rmse",
    "rpe_10m_align_rmse",
    "cross_track_mean",
    "row_correct_fraction",
    "row_switch_events",
]

LOWER_IS_BETTER = {
    "ape_align_rmse",
    "rpe_2m_align_rmse",
    "rpe_5m_align_rmse",
    "rpe_10m_align_rmse",
    "cross_track_mean",
    "row_switch_events",
    "speed_mean",
    "accel_rms",
    "jerk_rms",
    "heading_rate_rms",
    "heading_accel_rms",
}


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


def primary_score(metrics: Dict[str, float]) -> float:
    # Lower is better.
    return (
        1.00 * safe_float(metrics["ape_align_rmse"], 1e9)
        + 0.20 * safe_float(metrics["rpe_2m_align_rmse"], 1e9)
        + 0.35 * safe_float(metrics["rpe_5m_align_rmse"], 1e9)
        + 0.20 * safe_float(metrics["rpe_10m_align_rmse"], 1e9)
        + 0.80 * safe_float(metrics["cross_track_mean"], 1e9)
        + 2.00 * max(0.0, 1.0 - safe_float(metrics["row_correct_fraction"], 0.0))
        + 0.03 * safe_float(metrics["row_switch_events"], 1e9)
        + 0.40 * safe_float(metrics["jerk_rms"], 1e9)
        + 0.20 * safe_float(metrics["heading_accel_rms"], 1e9)
    )


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_by_variant(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_variant.setdefault(str(r["variant"]), []).append(r)

    agg_rows: List[Dict[str, object]] = []
    for variant in VARIANT_ORDER:
        if variant not in by_variant:
            continue
        group = by_variant[variant]
        out: Dict[str, object] = {
            "variant": variant,
            "n_runs": len(group),
            "runtime_sec_mean": float(np.mean([safe_float(r["runtime_sec"], 0.0) for r in group])),
            "runtime_sec_std": float(np.std([safe_float(r["runtime_sec"], 0.0) for r in group])),
        }

        metric_means = {}
        for metric in KEY_METRICS:
            vals = np.asarray([safe_float(r.get(metric)) for r in group], dtype=np.float64)
            valid = vals[np.isfinite(vals)]
            out[f"{metric}_mean"] = float(np.mean(valid)) if valid.size else float("nan")
            out[f"{metric}_std"] = float(np.std(valid)) if valid.size else float("nan")
            out[f"{metric}_median"] = float(np.median(valid)) if valid.size else float("nan")
            metric_means[metric] = out[f"{metric}_mean"]

        out["primary_score_mean"] = float(primary_score(metric_means))
        agg_rows.append(out)

    # Delta against full (means).
    full = next((r for r in agg_rows if r["variant"] == "full"), None)
    if full is not None:
        for row in agg_rows:
            for metric in KEY_METRICS:
                row[f"delta_vs_full_{metric}"] = safe_float(row[f"{metric}_mean"]) - safe_float(full[f"{metric}_mean"])
            row["delta_vs_full_primary_score"] = safe_float(row["primary_score_mean"]) - safe_float(full["primary_score_mean"])

    # Ranking by primary score and each metric.
    def rank_variants(key: str, lower_better: bool):
        ranked = sorted(agg_rows, key=lambda r: safe_float(r[key], 1e9 if lower_better else -1e9), reverse=not lower_better)
        return {r["variant"]: i + 1 for i, r in enumerate(ranked)}

    rank_primary = rank_variants("primary_score_mean", lower_better=True)
    for row in agg_rows:
        row["rank_primary_score"] = int(rank_primary[row["variant"]])

    for metric in KEY_METRICS:
        mean_key = f"{metric}_mean"
        lower = metric in LOWER_IS_BETTER
        rank_map = rank_variants(mean_key, lower_better=lower)
        for row in agg_rows:
            row[f"rank_{metric}"] = int(rank_map[row["variant"]])

    agg_rows.sort(key=lambda r: int(r["rank_primary_score"]))
    return agg_rows


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.3f} \\pm {std:.3f}"


def write_latex_table(agg_rows: List[Dict[str, object]], out_path: Path):
    latex_metrics = CORE_METRICS

    best = {}
    for m in latex_metrics:
        key = f"{m}_mean"
        vals = [(r["variant"], safe_float(r[key])) for r in agg_rows]
        if m in LOWER_IS_BETTER:
            best_val = min(v for _, v in vals)
        else:
            best_val = max(v for _, v in vals)
        best[m] = best_val

    headers = [
        ("ape_align_rmse", "APE$\\downarrow$"),
        ("rpe_2m_align_rmse", "RPE2$\\downarrow$"),
        ("rpe_5m_align_rmse", "RPE5$\\downarrow$"),
        ("rpe_10m_align_rmse", "RPE10$\\downarrow$"),
        ("cross_track_mean", "XTE$\\downarrow$"),
        ("row_correct_fraction", "RowAcc$\\uparrow$"),
        ("row_switch_events", "Switch$\\downarrow$"),
    ]

    lines = []
    lines.append("% Auto-generated by scripts/run_spfpp_ablation.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append("Variant & " + " & ".join([h for _, h in headers]) + " \\\\")
    lines.append("\\midrule")

    for row in agg_rows:
        cells = []
        for metric, _ in headers:
            mean = safe_float(row[f"{metric}_mean"]) 
            std = safe_float(row[f"{metric}_std"]) 
            cell = format_pm(mean, std)
            if abs(mean - best[metric]) <= 1e-12:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        variant_name = str(row["variant"]).replace("_", "\\_")
        lines.append(f"{variant_name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{SPF++ ablation study (mean $\\pm$ std across 3 seeds, aligned metrics).}")
    lines.append("\\label{tab:spfpp_ablation}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def plot_ablation_main(agg_rows: List[Dict[str, object]], out_path: Path):
    plot_metrics = [
        ("ape_align_rmse", "APE RMSE", True),
        ("rpe_5m_align_rmse", "RPE 5m RMSE", True),
        ("cross_track_mean", "Cross-track", True),
        ("row_correct_fraction", "Row Correct", False),
        ("row_switch_events", "Row Switches", True),
        ("jerk_rms", "Jerk RMS", True),
    ]

    labels = [str(r["variant"]) for r in agg_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    for ax, (metric, title, lower_better) in zip(axes, plot_metrics):
        means = np.asarray([safe_float(r[f"{metric}_mean"]) for r in agg_rows], dtype=np.float64)
        stds = np.asarray([safe_float(r[f"{metric}_std"]) for r in agg_rows], dtype=np.float64)
        bars = ax.bar(x, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.85)

        best_idx = int(np.nanargmin(means)) if lower_better else int(np.nanargmax(means))
        bars[best_idx].set_color("tab:green")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("SPF++ Ablation (Variants Only)", fontsize=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay_seed22(per_seed_rows: List[Dict[str, object]], out_path: Path):
    seed_rows = [r for r in per_seed_rows if int(r["seed"]) == 22]
    if not seed_rows:
        return

    seed_by_variant = {str(r["variant"]): r for r in seed_rows}
    if "full" not in seed_by_variant:
        return

    full = seed_by_variant["full"]
    full_aligned = aligned_estimate(Path(str(full["est_tum"])), Path(str(full["gt_tum"])))

    fig, ax = plt.subplots(figsize=(11, 8))
    gt_xy = full_aligned["gt_interp"][:, :2]
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k--", linewidth=2, label="GPS ground truth")

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    cidx = 0
    for variant in VARIANT_ORDER:
        row = seed_by_variant.get(variant)
        if row is None:
            continue
        aligned = aligned_estimate(Path(str(row["est_tum"])), Path(str(row["gt_tum"])))
        xy = aligned["est_aligned"][:, :2]
        color = colors[cidx % len(colors)] if colors else None
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.7, color=color, label=variant)
        cidx += 1

    ax.set_title("Ablation Overlay (Seed 22, Aligned to GT)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_nonwall_formulation_tex(out_path: Path):
    text = r"""% Auto-generated by scripts/run_spfpp_ablation.py
\subsection{Point-Based Semantic Matching (No Semantic Walls)}
To ablate semantic walls, we model poles and trunks as individual map objects and score each observation ray against class-consistent point hypotheses.

Given particle $i$ and observation ray $j$ with class $c_j$, observed range $r_j$, and local bearing $\alpha_j$, we transform all map points of class $c\in\{\text{pole},\text{trunk}\}$ into the particle local frame, obtaining $(r_{ik},\alpha_{ik})$.

Candidates satisfy:
\begin{equation}
|\Delta\alpha_{ijk}| \le \alpha_{\text{gate}},\qquad
|\Delta r_{ijk}| \le r_{\text{gate}},\qquad
0 < r_{ik} \le r_{\max},
\end{equation}
with $\Delta\alpha_{ijk}=\mathrm{wrap}(\alpha_{ik}-\alpha_j)$ and $\Delta r_{ijk}=r_{ik}-r_j$.

If class-consistent candidates exist, we use the best (maximum log-likelihood):
\begin{equation}
s_{ij}=\max_k\left[-\frac{1}{2}\left(\frac{\Delta\alpha_{ijk}^2}{\sigma_\alpha^2}+\frac{\Delta r_{ijk}^2}{\sigma_r^2}\right)\right].
\end{equation}
If only opposite-class candidates exist, a wrong-hit penalty is used; if no candidate exists, a miss penalty is used.

For background rays, if any map point falls along the ray before the observed background range, we treat the ray as a hit-like free-space consistency event; otherwise a miss penalty is applied.

The per-particle semantic score is averaged across rays and fused with GPS and corridor terms using the same normalized fusion strategy used in SPF++.
"""
    out_path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Run SPF++ ablation study (IROS protocol).")
    parser.add_argument("--python-exec", type=Path, default=BASE_DIR / ".venv" / "bin" / "python")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    parser.add_argument("--max-frames", type=int, default=None)

    # Frozen baseline config defaults.
    parser.add_argument("--miss-penalty", type=float, default=4.0)
    parser.add_argument("--wrong-hit-penalty", type=float, default=4.0)
    parser.add_argument("--gps-weight", type=float, default=0.5)
    parser.add_argument("--semantic-sigma", type=float, default=0.05)
    parser.add_argument("--gps-sigma", type=float, default=1.1)
    parser.add_argument("--corridor-weight", type=float, default=0.30)
    parser.add_argument("--corridor-dist-sigma", type=float, default=1.50)
    parser.add_argument("--corridor-heading-sigma", type=float, default=0.35)
    parser.add_argument("--background-class-weight", type=float, default=0.20)
    parser.add_argument("--max-background-obs", type=int, default=120)
    parser.add_argument("--expected-obs-count", type=float, default=150.0)
    parser.add_argument("--pose-smooth-alpha-pos", type=float, default=0.55)
    parser.add_argument("--pose-smooth-alpha-theta", type=float, default=0.50)
    parser.add_argument("--odom-yaw-filter-alpha", type=float, default=0.90)
    parser.add_argument("--particle-count", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=4)

    # Point model defaults.
    parser.add_argument("--point-ang-sigma", type=float, default=0.08)
    parser.add_argument("--point-range-sigma", type=float, default=0.35)
    parser.add_argument("--point-ang-gate", type=float, default=0.20)
    parser.add_argument("--point-max-range-diff", type=float, default=1.5)

    args = parser.parse_args()

    python_exec = args.python_exec.expanduser()
    if not python_exec.is_absolute():
        python_exec = (BASE_DIR / python_exec).resolve()

    output_root = args.output_root.resolve()
    geojson = args.geojson.resolve()
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not variants:
        raise ValueError("At least one variant is required.")
    unknown = [v for v in variants if v not in VARIANT_ARGS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    if args.require_cuda and not check_cuda_available(python_exec):
        raise RuntimeError("CUDA is required but not visible in the selected Python runtime.")

    evo_ape_bin = python_exec.parent / "evo_ape"
    evo_rpe_bin = python_exec.parent / "evo_rpe"
    if not evo_ape_bin.exists() or not evo_rpe_bin.exists():
        raise FileNotFoundError("evo_ape/evo_rpe not found in virtualenv bin directory.")

    rows_map = load_rows_from_geojson(geojson)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_ablation"
    if args.max_frames is not None:
        run_name += "_smoke"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(BASE_DIR / ".tmp_mpl")
    (BASE_DIR / ".tmp_mpl").mkdir(parents=True, exist_ok=True)

    protocol = {
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "seeds": seeds,
        "variants": variants,
        "require_cuda": bool(args.require_cuda),
        "cuda_visible_devices": args.cuda_visible_devices,
        "frozen_config": {
            "miss_penalty": args.miss_penalty,
            "wrong_hit_penalty": args.wrong_hit_penalty,
            "gps_weight": args.gps_weight,
            "semantic_sigma": args.semantic_sigma,
            "gps_sigma": args.gps_sigma,
            "corridor_weight": args.corridor_weight,
            "corridor_dist_sigma": args.corridor_dist_sigma,
            "corridor_heading_sigma": args.corridor_heading_sigma,
            "background_class_weight": args.background_class_weight,
            "max_background_obs": args.max_background_obs,
            "expected_obs_count": args.expected_obs_count,
            "pose_smooth_alpha_pos": args.pose_smooth_alpha_pos,
            "pose_smooth_alpha_theta": args.pose_smooth_alpha_theta,
            "odom_yaw_filter_alpha": args.odom_yaw_filter_alpha,
            "particle_count": args.particle_count,
            "frame_stride": args.frame_stride,
            "point_ang_sigma": args.point_ang_sigma,
            "point_range_sigma": args.point_range_sigma,
            "point_ang_gate": args.point_ang_gate,
            "point_max_range_diff": args.point_max_range_diff,
        },
        "commands": [],
    }

    scripts_dir = BASE_DIR / "scripts"
    spf_script = scripts_dir / "spf_lidar.py"

    per_seed_rows: List[Dict[str, object]] = []

    for variant in variants:
        for seed in seeds:
            seed_dir = run_dir / variant / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(python_exec),
                str(spf_script),
                "--miss-penalty", str(args.miss_penalty),
                "--wrong-hit-penalty", str(args.wrong_hit_penalty),
                "--gps-weight", str(args.gps_weight),
                "--seed", str(seed),
                "--output-folder", str(seed_dir),
                "--frame-stride", str(args.frame_stride),
                "--semantic-sigma", str(args.semantic_sigma),
                "--gps-sigma", str(args.gps_sigma),
                "--corridor-weight", str(args.corridor_weight),
                "--corridor-dist-sigma", str(args.corridor_dist_sigma),
                "--corridor-heading-sigma", str(args.corridor_heading_sigma),
                "--background-class-weight", str(args.background_class_weight),
                "--max-background-obs", str(args.max_background_obs),
                "--expected-obs-count", str(args.expected_obs_count),
                "--pose-smooth-alpha-pos", str(args.pose_smooth_alpha_pos),
                "--pose-smooth-alpha-theta", str(args.pose_smooth_alpha_theta),
                "--odom-yaw-filter-alpha", str(args.odom_yaw_filter_alpha),
                "--particle-count", str(args.particle_count),
                "--point-ang-sigma", str(args.point_ang_sigma),
                "--point-range-sigma", str(args.point_range_sigma),
                "--point-ang-gate", str(args.point_ang_gate),
                "--point-max-range-diff", str(args.point_max_range_diff),
                "--no-visualization",
            ]
            if args.max_frames is not None:
                cmd.extend(["--max-frames", str(args.max_frames)])
            if args.require_cuda:
                cmd.append("--require-cuda")
            cmd.extend(VARIANT_ARGS[variant])

            run_log = seed_dir / "run_spf_lidar.log"
            runtime = run_cmd(cmd, run_log, cwd=BASE_DIR, env=env)

            est_tum = seed_dir / f"trajectory_{args.gps_weight}.tum"
            gt_tum = seed_dir / "gps_pose.tum"
            if not est_tum.exists() or not gt_tum.exists():
                raise FileNotFoundError(f"Missing trajectory outputs for {variant} seed {seed}: {seed_dir}")

            metrics = evaluate_run(
                name=f"{variant}_seed_{seed}",
                est_tum=est_tum,
                gt_tum=gt_tum,
                out_dir=seed_dir / "eval",
                rows=rows_map,
                evo_ape_bin=evo_ape_bin,
                evo_rpe_bin=evo_rpe_bin,
                env=env,
            )

            metrics["variant"] = variant
            metrics["seed"] = seed
            metrics["runtime_sec"] = runtime
            metrics["command"] = " ".join(cmd)
            per_seed_rows.append(metrics)

            protocol["commands"].append({
                "variant": variant,
                "seed": seed,
                "command": cmd,
                "runtime_sec": runtime,
                "output_dir": str(seed_dir),
            })

    per_seed_rows.sort(key=lambda r: (VARIANT_ORDER.index(str(r["variant"])), int(r["seed"])))
    write_csv(run_dir / "ablation_metrics_per_seed.csv", per_seed_rows)

    agg_rows = aggregate_by_variant(per_seed_rows)
    write_csv(run_dir / "ablation_metrics_aggregate.csv", agg_rows)

    write_latex_table(agg_rows, run_dir / "ablation_primary_table.tex")
    plot_ablation_main(agg_rows, run_dir / "ablation_main_plot.pdf")
    plot_overlay_seed22(per_seed_rows, run_dir / "ablation_trajectory_overlay_seed22.png")
    write_nonwall_formulation_tex(run_dir / "ablation_nonwall_formulation.tex")

    (run_dir / "ablation_protocol.json").write_text(json.dumps(protocol, indent=2))

    print(f"[INFO] Ablation complete: {run_dir}")
    if agg_rows:
        best = min(agg_rows, key=lambda r: safe_float(r["primary_score_mean"], 1e9))
        print(f"[INFO] Best primary-score variant: {best['variant']} ({safe_float(best['primary_score_mean']):.6f})")


if __name__ == "__main__":
    main()
