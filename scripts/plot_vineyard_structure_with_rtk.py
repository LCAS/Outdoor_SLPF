#!/usr/bin/env python3
"""Plot vineyard structure (posts/vines/semantic walls) with RTK trajectories only."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from pyproj import Transformer

from geojson_rows import iter_projected_points
from plot_trajectories import load_landmark_points


def resolve_path(base_dir: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base_dir / path)


def compute_geojson_center_xy(geojson_path: Path, target_crs: str = "epsg:32630") -> np.ndarray:
    xy = []
    for item in iter_projected_points(geojson_path, target_crs=target_crs):
        xy.append((item["x"], item["y"]))
    if not xy:
        raise ValueError(f"No projected points found in geojson: {geojson_path}")
    return np.asarray(xy, dtype=np.float64).mean(axis=0)


def load_rtk_csv_as_centered_xy(
    csv_path: Path,
    map_center_xy: np.ndarray,
    target_crs: str = "epsg:32630",
) -> np.ndarray:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing RTK CSV: {csv_path}")

    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    points = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        has_latlon = {"latitude", "longitude"}.issubset(set(reader.fieldnames))
        has_utm = {"utm_easting", "utm_northing"}.issubset(set(reader.fieldnames))
        if not has_latlon and not has_utm:
            raise ValueError(
                f"CSV missing coordinates (expected latitude/longitude or utm_easting/utm_northing): {csv_path}"
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
            points.append((x - map_center_xy[0], y - map_center_xy[1]))

    if not points:
        raise ValueError(f"No valid RTK points parsed from {csv_path}")
    return np.asarray(points, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", default="data/riseholme_poles_trunk.geojson")
    parser.add_argument("--run1-csv", default="data/2025/rh_run1/data.csv")
    parser.add_argument("--run2-csv", default="data/2025/rh_run2/data.csv")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--output-stem", default="vineyard_structure_with_rtk_run1_run2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    geojson_path = resolve_path(base_dir, args.geojson)
    run1_csv = resolve_path(base_dir, args.run1_csv)
    run2_csv = resolve_path(base_dir, args.run2_csv)
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_center_xy = compute_geojson_center_xy(geojson_path)
    landmarks = load_landmark_points(geojson_path)
    run1_xy = load_rtk_csv_as_centered_xy(run1_csv, map_center_xy)
    run2_xy = load_rtk_csv_as_centered_xy(run2_csv, map_center_xy)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.0, 8.0), constrained_layout=True)
    semantic_wall_lw = 1.8
    rtk_lw = 2.6

    poles = landmarks.get("poles", np.empty((0, 2)))
    trunks = landmarks.get("trunks", np.empty((0, 2)))
    seg_pole = landmarks.get("segments_pole", np.empty((0, 2, 2)))
    seg_trunk = landmarks.get("segments_trunk", np.empty((0, 2, 2)))

    if seg_pole.size:
        ax.add_collection(
            LineCollection(
                seg_pole,
                colors="orange",
                linewidths=semantic_wall_lw,
                alpha=0.7,
                label="Semantic wall (Pole)",
            )
        )
    if seg_trunk.size:
        ax.add_collection(
            LineCollection(
                seg_trunk,
                colors="forestgreen",
                linewidths=semantic_wall_lw,
                alpha=0.7,
                label="Semantic wall (Trunk)",
            )
        )
    if poles.size:
        ax.scatter(poles[:, 0], poles[:, 1], s=14, edgecolor="black", facecolor="orange", linewidth=0.35, label="Row posts")
    if trunks.size:
        ax.scatter(trunks[:, 0], trunks[:, 1], s=9, marker="s", color="forestgreen", alpha=0.75, label="Vines")

    ax.plot(run1_xy[:, 0], run1_xy[:, 1], color="tab:red", linewidth=rtk_lw, label="RTK run1")
    ax.plot(run2_xy[:, 0], run2_xy[:, 1], color="tab:blue", linewidth=rtk_lw, label="RTK run2")

    ax.scatter(
        run1_xy[0, 0],
        run1_xy[0, 1],
        s=90,
        marker="^",
        color="#ffcc00",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
        label="Run1 start",
    )
    ax.scatter(
        run1_xy[-1, 0],
        run1_xy[-1, 1],
        s=90,
        marker="v",
        color="#00bcd4",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
        label="Run1 end",
    )
    ax.scatter(
        run2_xy[0, 0],
        run2_xy[0, 1],
        s=86,
        marker=">",
        color="#ff9f1a",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
        label="Run2 start",
    )
    ax.scatter(
        run2_xy[-1, 0],
        run2_xy[-1, 1],
        s=86,
        marker="<",
        color="#7fdbff",
        edgecolor="black",
        linewidth=0.6,
        zorder=7,
        label="Run2 end",
    )

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xticks(np.arange(-15, 16, 5))
    ax.set_yticks(np.arange(-15, 16, 5))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax.tick_params(width=1.4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.0,
    )

    output_png = output_dir / f"{args.output_stem}.png"
    output_pdf = output_dir / f"{args.output_stem}.pdf"
    fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
