#!/usr/bin/env python3
"""Build a 2x4 trajectory comparison figure for two experiments with RTK overlays."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from pyproj import Transformer

from geojson_rows import iter_projected_points
from plot_trajectories import (
    anchor_start_to_ground_truth,
    apply_transform,
    compute_errors,
    interpolate_ground_truth,
    load_landmark_points,
    read_tum_file,
    stride_keep_end,
    umeyama_alignment,
)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    trajectory_path: Path
    ground_truth_path: Path
    stride: int = 1
    align_umeyama_scale: bool = False
    anchor_start: bool = False
    error_vmin: float = 0.0
    error_vmax: float = 10.0


def resolve_path(base_dir: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base_dir / path)


def compute_geojson_center_xy(geojson_path: Path, target_crs: str = "epsg:32630") -> np.ndarray:
    points = []
    for item in iter_projected_points(geojson_path, target_crs=target_crs):
        points.append((item["x"], item["y"]))
    if not points:
        raise ValueError(f"No projected points found in geojson: {geojson_path}")
    return np.asarray(points, dtype=np.float64).mean(axis=0)


def load_rtk_csv_as_centered_xy(
    csv_path: Path,
    map_center_xy: np.ndarray,
    target_crs: str = "epsg:32630",
) -> np.ndarray:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing RTK CSV: {csv_path}")

    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    xy = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        has_latlon = {"latitude", "longitude"}.issubset(set(reader.fieldnames))
        has_utm = {"utm_easting", "utm_northing"}.issubset(set(reader.fieldnames))
        if not has_latlon and not has_utm:
            raise ValueError(
                f"CSV missing coordinate fields (expected lat/lon or utm_easting/utm_northing): {csv_path}"
            )

        for row in reader:
            try:
                if has_utm and row.get("utm_easting") and row.get("utm_northing"):
                    x = float(row["utm_easting"])
                    y = float(row["utm_northing"])
                else:
                    lon = float(row["longitude"])
                    lat = float(row["latitude"])
                    x, y = transformer.transform(lon, lat)
            except (TypeError, ValueError):
                continue
            xy.append((x - map_center_xy[0], y - map_center_xy[1]))

    if not xy:
        raise ValueError(f"No valid RTK points parsed from {csv_path}")

    arr = np.asarray(xy, dtype=np.float64)
    if len(arr) > 1:
        step = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        keep = np.ones(len(arr), dtype=bool)
        keep[1:] = step > 1e-9
        arr = arr[keep]
    return arr


def load_method_plot_data(spec: MethodSpec) -> dict:
    if not spec.trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {spec.trajectory_path}")
    if not spec.ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {spec.ground_truth_path}")

    gt_ts, gt_pos, _ = read_tum_file(str(spec.ground_truth_path))
    traj_ts, traj_pos, _ = read_tum_file(str(spec.trajectory_path))
    if gt_pos is None or traj_pos is None:
        raise ValueError(f"Failed to read TUM data for method {spec.label}")

    gt_interp = interpolate_ground_truth(gt_ts, gt_pos, traj_ts)
    traj_plot = np.copy(traj_pos)

    if spec.align_umeyama_scale:
        scale, rot, trans = umeyama_alignment(traj_plot, gt_interp, with_scaling=True)
        traj_plot = apply_transform(traj_plot, scale, rot, trans)
    if spec.anchor_start:
        traj_plot = anchor_start_to_ground_truth(traj_plot, gt_interp)
    if spec.stride > 1:
        traj_plot = stride_keep_end(traj_plot, spec.stride)
        gt_interp = stride_keep_end(gt_interp, spec.stride)

    errors = compute_errors(traj_plot, gt_interp)
    return {"spec": spec, "trajectory": traj_plot, "ground_truth": gt_interp, "errors": errors}


def add_landmarks(ax, landmark_points: dict | None) -> None:
    if landmark_points is None:
        return

    poles = landmark_points.get("poles", np.empty((0, 2)))
    trunks = landmark_points.get("trunks", np.empty((0, 2)))
    seg_pole = landmark_points.get("segments_pole", np.empty((0, 2, 2)))
    seg_trunk = landmark_points.get("segments_trunk", np.empty((0, 2, 2)))

    if seg_pole.size:
        ax.add_collection(LineCollection(seg_pole, colors="orange", linewidths=0.8, alpha=0.5, zorder=1))
    if seg_trunk.size:
        ax.add_collection(LineCollection(seg_trunk, colors="forestgreen", linewidths=0.8, alpha=0.5, zorder=1))
    if poles.size:
        ax.scatter(
            poles[:, 0],
            poles[:, 1],
            s=12,
            edgecolor="black",
            facecolor="orange",
            linewidth=0.3,
            zorder=2,
        )
    if trunks.size:
        ax.scatter(
            trunks[:, 0],
            trunks[:, 1],
            s=8,
            marker="s",
            color="forestgreen",
            alpha=0.7,
            zorder=2,
        )


def plot_subplot(
    ax,
    method_data: dict,
    landmark_points: dict | None,
    rtk_xy: np.ndarray,
    show_title: bool,
    show_ylabel: bool,
):
    spec: MethodSpec = method_data["spec"]
    traj = method_data["trajectory"]
    gt = method_data["ground_truth"]
    errors = method_data["errors"]

    add_landmarks(ax, landmark_points)

    points = traj[:, :2]
    gt_xy = gt[:, :2]
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="gray", linestyle="--", linewidth=1.4, alpha=0.8, zorder=2)

    # Explicit RTK overlay for the row (run1 on row-1, run2 on row-2).
    ax.plot(rtk_xy[:, 0], rtk_xy[:, 1], color="tab:red", linestyle="-.", linewidth=1.5, alpha=0.9, zorder=3)
    ax.scatter(
        rtk_xy[0, 0],
        rtk_xy[0, 1],
        s=42,
        marker="^",
        color="#ffcc00",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
    )
    ax.scatter(
        rtk_xy[-1, 0],
        rtk_xy[-1, 1],
        s=42,
        marker="v",
        color="#00bcd4",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
    )

    if len(points) > 1:
        segments = np.array([points[i : i + 2] for i in range(len(points) - 1)])
        lc = LineCollection(
            segments,
            cmap="viridis",
            norm=Normalize(vmin=spec.error_vmin, vmax=spec.error_vmax),
            linewidths=2.4,
            zorder=4,
        )
        lc.set_array(errors[:-1])
        ax.add_collection(lc)
    else:
        lc = None
        ax.plot(points[:, 0], points[:, 1], color="tab:blue", linewidth=2.2, zorder=4)

    ax.scatter(
        points[0, 0],
        points[0, 1],
        s=40,
        marker="o",
        color="limegreen",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
    )
    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        s=48,
        marker="X",
        color="crimson",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
    )

    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_xticks(np.arange(-18, 19, 6))
    ax.set_yticks(np.arange(-18, 19, 6))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)" if show_ylabel else "")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    if show_title:
        ax.set_title(spec.label)

    if lc is not None:
        cbar = plt.colorbar(lc, ax=ax, fraction=0.048, pad=0.02)
        cbar.set_label("Error (m)")


def build_method_specs(base_dir: Path, exp1_root: Path, exp2_root: Path) -> list[list[MethodSpec]]:
    exp1 = [
        MethodSpec(
            label="SLPF(ours)",
            trajectory_path=exp1_root / "spf_lidar++" / "0.5" / "trajectory_0.5.tum",
            ground_truth_path=exp1_root / "spf_lidar++" / "0.5" / "gps_pose.tum",
            error_vmin=0.0,
            error_vmax=5.0,
        ),
        MethodSpec(
            label="AMCL+GPS",
            trajectory_path=exp1_root / "amcl_ngps" / "tum1" / "trajectory_0.5.tum",
            ground_truth_path=exp1_root / "amcl_ngps" / "tum1" / "gps_pose.tum",
            error_vmin=0.0,
            error_vmax=5.0,
        ),
        MethodSpec(
            label="RTABMAP RGBD",
            trajectory_path=exp1_root / "rtabmap" / "rgbd" / "tum1" / "rtabmap_rgbd_filtered.tum",
            ground_truth_path=exp1_root / "rtabmap" / "rgbd" / "tum1" / "gps_pose.tum",
            align_umeyama_scale=True,
            anchor_start=True,
            error_vmin=0.0,
            error_vmax=15.0,
        ),
        MethodSpec(
            label="Noisy GPS",
            trajectory_path=base_dir / "data" / "2025" / "noisy_gps" / "run1" / "noisy_gps_seed_11.tum",
            ground_truth_path=base_dir / "data" / "2025" / "amcl" / "tum1" / "gps_pose.tum",
            stride=20,
            error_vmin=0.0,
            error_vmax=20.0,
        ),
    ]

    exp2 = [
        MethodSpec(
            label="SLPF(ours)",
            trajectory_path=exp2_root / "slpf" / "seed_11" / "trajectory_0.5.tum",
            ground_truth_path=exp2_root / "slpf" / "seed_11" / "gps_pose.tum",
            error_vmin=0.0,
            error_vmax=5.0,
        ),
        MethodSpec(
            label="AMCL+GPS",
            trajectory_path=exp2_root / "amcl_ngps" / "seed_11" / "trajectory_0.5.tum",
            ground_truth_path=exp2_root / "amcl_ngps" / "seed_11" / "gps_pose.tum",
            error_vmin=0.0,
            error_vmax=5.0,
        ),
        MethodSpec(
            label="RTABMAP RGBD",
            trajectory_path=exp2_root / "rtab_rgbd" / "seed_11" / "rtabmap_rgbd_filtered.tum",
            ground_truth_path=exp2_root / "rtab_rgbd" / "seed_11" / "gps_pose.tum",
            align_umeyama_scale=True,
            anchor_start=True,
            error_vmin=0.0,
            error_vmax=15.0,
        ),
        MethodSpec(
            label="Noisy GPS",
            trajectory_path=exp2_root / "ngps" / "seed_11" / "trajectory_0.5.tum",
            ground_truth_path=exp2_root / "ngps" / "seed_11" / "gps_pose.tum",
            stride=20,
            error_vmin=0.0,
            error_vmax=20.0,
        ),
    ]
    return [exp1, exp2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1-root", default="results/iros_rh1")
    parser.add_argument("--exp2-root", default="results/iros_rh2/20260225_105822_multiseed_all_methods")
    parser.add_argument("--geojson", default="data/riseholme_poles_trunk.geojson")
    parser.add_argument("--run1-csv", default="data/2025/rh_run1/data.csv")
    parser.add_argument("--run2-csv", default="data/2025/rh_run2/data.csv")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument(
        "--output-stem",
        default="trajectory_comparison_2x4_experiment1_vs_experiment2_vertical_labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    exp1_root = resolve_path(base_dir, args.exp1_root)
    exp2_root = resolve_path(base_dir, args.exp2_root)
    geojson_path = resolve_path(base_dir, args.geojson)
    run1_csv = resolve_path(base_dir, args.run1_csv)
    run2_csv = resolve_path(base_dir, args.run2_csv)
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_center_xy = compute_geojson_center_xy(geojson_path)
    landmark_points = load_landmark_points(geojson_path)
    rtk_run1 = load_rtk_csv_as_centered_xy(run1_csv, map_center_xy)
    rtk_run2 = load_rtk_csv_as_centered_xy(run2_csv, map_center_xy)

    method_specs_by_row = build_method_specs(base_dir, exp1_root, exp2_root)
    method_data_by_row = []
    for row_specs in method_specs_by_row:
        row_data = [load_method_plot_data(spec) for spec in row_specs]
        method_data_by_row.append(row_data)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.08, wspace=0.16, hspace=0.14)

    for row_idx in range(2):
        rtk_xy = rtk_run1 if row_idx == 0 else rtk_run2
        for col_idx in range(4):
            plot_subplot(
                axes[row_idx, col_idx],
                method_data_by_row[row_idx][col_idx],
                landmark_points=landmark_points,
                rtk_xy=rtk_xy,
                show_title=(row_idx == 0),
                show_ylabel=(col_idx == 0),
            )

    fig.text(0.023, 0.63, "Experiment 1", rotation=90, fontsize=18, fontweight="bold", va="center")
    fig.text(0.023, 0.285, "Experiment 2", rotation=90, fontsize=18, fontweight="bold", va="center")

    legend_handles = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.6, label="GPS ground truth"),
        Line2D([0], [0], color="tab:red", linestyle="-.", linewidth=1.6, label="RTK-GPS ground truth"),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor="#ffcc00",
            markeredgecolor="black",
            markersize=8,
            label="RTK start",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            linestyle="None",
            markerfacecolor="#00bcd4",
            markeredgecolor="black",
            markersize=8,
            label="RTK end",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="limegreen",
            markeredgecolor="black",
            markersize=7,
            label="Robot start",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markerfacecolor="crimson",
            markeredgecolor="black",
            markersize=7,
            label="Robot end",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="black",
            markersize=6,
            label="Row posts",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="forestgreen",
            markeredgecolor="forestgreen",
            markersize=6,
            label="Vines",
        ),
        Line2D([0], [0], color="orange", linewidth=1.2, label="Semantic wall (Pole)"),
        Line2D([0], [0], color="forestgreen", linewidth=1.2, label="Semantic wall (Trunk)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=10,
        frameon=False,
        fontsize=13,
        handlelength=2.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    output_stem = args.output_stem
    output_png = output_dir / f"{output_stem}.png"
    output_pdf = output_dir / f"{output_stem}.pdf"
    fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
