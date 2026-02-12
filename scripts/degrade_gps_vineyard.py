#!/usr/bin/env python3
"""
degrade_gps_vineyard.py

Creates a "fake noisy GNSS" baseline from RTK ground truth by injecting:
  1) Slow time-correlated bias (1st-order Gauss–Markov)  -> low-frequency wander
  2) Fast white noise                                    -> receiver jitter
  3) Occasional heavy-tailed outliers (Student-t)        -> sporadic multipath-like jumps
  4) Optional dropouts/outages (burst model)             -> GNSS unavailable periods

Input (your format):
  # timestamp tx ty tz qx qy qz qw
  0  -10.7352  -11.8230  0.0  0 0 0 1
  4  -10.7224  -11.8090  0.0  0 0 0 1

Output:
  Same columns; if dropout_mode == "remove" some rows are removed.
  If dropout_mode == "hold" or "nan", row count stays the same.

--------------------------------------------------------------------------------
DEFAULTS (tuned for: small vineyard rows, open sky, but "non-RTK GNSS baseline"):
  - White noise (sigma_w_xy): 1.0 m        (fast jitter)
  - Correlated bias (sigma_b_xy): 1.0 m    (slow wander, steady-state std)
  - Correlation time (tau): 30 s           (how slowly the wander changes)
  - Outliers: rare (0.5%) with scale 4 m   (sporadic multipath-like jumps)
  - Dropouts: disabled by default          (open sky; enable for stress tests)

--------------------------------------------------------------------------------
TUNING GUIDE (quick heuristics):

A) "Cleaner" GNSS baseline (closer to good open-sky receiver):
   - sigma_w_xy: 0.3–0.7
   - sigma_b_xy: 0.3–0.8
   - outlier_prob: 0–0.002 and/or outlier_scale_xy: 2–3
   - keep dropouts disabled (dropout_rate = 0)

B) "Worse" GNSS baseline (e.g., near foliage, partial occlusions):
   - sigma_w_xy: 1.5–3.0
   - sigma_b_xy: 2.0–5.0
   - outlier_prob: 0.01–0.05 and outlier_scale_xy: 5–15
   - enable dropouts: dropout_rate 0.001–0.01, mean_dur 1–5 s

C) Time correlation:
   - tau small (5–15 s): bias changes quickly (more “shaky wander”)
   - tau large (60–300 s): bias drifts slowly (more “smooth drift”)

D) Height channel:
   If tz is always 0 and you only care about planar GNSS:
   - set sigma_w_z=0 and sigma_b_z=0 (or ignore z downstream).

Plotting:
  --plot will show:
    - XY trajectories (RTK vs noisy)
    - error magnitude over time
    - XY error components over time
    - dropout markers (if any)

Examples:
  python degrade_gps_vineyard.py groundtruth.tum noisy_gnss.tum
  python degrade_gps_vineyard.py groundtruth.tum noisy_gnss.tum --plot
  python degrade_gps_vineyard.py groundtruth.tum noisy_gnss.tum --plot --sigma-w-xy 0.6 --dropout-rate 0
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class NoiseParams:
    # --- Fast receiver jitter (metres) ---
    sigma_w_xy: float = 1.0
    sigma_w_z: float = 1.5

    # --- Slow correlated wander (metres) ---
    sigma_b_xy: float = 1.0
    sigma_b_z: float = 1.5
    tau: float = 30.0  # seconds

    # --- Multipath/NLOS-like outliers (heavy-tailed) ---
    outlier_prob: float = 0.005
    outlier_scale_xy: float = 4.0
    outlier_scale_z: float = 6.0
    outlier_dof: float = 3.0

    # --- Dropouts / outages (optional) ---
    dropout_rate: float = 0.0  # per second; 0 disables
    dropout_mean_dur: float = 2.0  # seconds
    dropout_mode: str = "remove"  # remove | hold | nan


def read_tum_with_comments(path: str) -> np.ndarray:
    """Reads whitespace-separated numeric data, ignoring lines starting with '#'."""
    data = np.loadtxt(path, dtype=float, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError("Input must have at least 4 columns: t x y z")
    return data


def write_tum(path: str, data: np.ndarray, header: str | None = None) -> None:
    """Writes whitespace-separated numeric data; optional commented header line."""
    fmt = ["{:.9f}"] + ["{:.6f}"] * (data.shape[1] - 1)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("# " + header.strip() + "\n")
        for row in data:
            f.write(" ".join(fmt[i].format(row[i]) for i in range(data.shape[1])) + "\n")


def build_dropout_mask(t: np.ndarray, rng: np.random.Generator, rate_per_s: float, mean_dur_s: float) -> np.ndarray:
    """
    Boolean mask of length N: True means "dropout at this sample".
    Burst model:
      - start outage with prob ~= 1-exp(-rate*dt)
      - duration ~ exponential(mean_dur_s)
    """
    N = len(t)
    mask = np.zeros(N, dtype=bool)
    if rate_per_s <= 0.0 or mean_dur_s <= 0.0 or N == 0:
        return mask

    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 1e-3, None)

    i = 0
    while i < N:
        p_start = 1.0 - math.exp(-rate_per_s * float(dt[i]))
        if rng.random() < p_start:
            dur = rng.exponential(mean_dur_s)
            t_end = t[i] + dur
            j = i
            while j < N and t[j] <= t_end:
                mask[j] = True
                j += 1
            i = j
        else:
            i += 1
    return mask


def apply_noise(t: np.ndarray, pos: np.ndarray, params: NoiseParams, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply correlated bias + white noise + outliers + (optional) dropouts.

    Returns:
      pos_noisy: (N,3)
      dropout_mask: (N,)
      err_components: (N,3) = (pos_noisy - pos) before "remove" is applied
    """
    N = pos.shape[0]
    if N == 0:
        return pos.copy(), np.zeros(0, dtype=bool), np.zeros_like(pos)

    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 1e-3, None)

    # 1) Gauss–Markov bias
    b = np.zeros((N, 3), dtype=float)
    sigma_b = np.array([params.sigma_b_xy, params.sigma_b_xy, params.sigma_b_z], dtype=float)
    tau = max(params.tau, 1e-3)

    for k in range(1, N):
        phi = math.exp(-float(dt[k]) / tau)
        innov_scale = math.sqrt(max(1.0 - phi * phi, 0.0))
        b[k] = phi * b[k - 1] + innov_scale * sigma_b * rng.standard_normal(3)

    # 2) White noise
    sigma_w = np.array([params.sigma_w_xy, params.sigma_w_xy, params.sigma_w_z], dtype=float)
    w = rng.standard_normal((N, 3)) * sigma_w

    # 3) Outliers
    out = np.zeros((N, 3), dtype=float)
    if params.outlier_prob > 0.0:
        mask_out = rng.random(N) < params.outlier_prob
        if np.any(mask_out):
            dof = max(params.outlier_dof, 1.0)
            scale = np.array([params.outlier_scale_xy, params.outlier_scale_xy, params.outlier_scale_z], dtype=float)
            out[mask_out] = rng.standard_t(dof, size=(mask_out.sum(), 3)) * scale

    pos_noisy = pos + b + w + out
    err = (pos_noisy - pos).copy()

    # 4) Dropouts
    dropout_mask = build_dropout_mask(t, rng, params.dropout_rate, params.dropout_mean_dur)

    if np.any(dropout_mask):
        if params.dropout_mode == "nan":
            pos_noisy[dropout_mask] = np.nan
        elif params.dropout_mode == "hold":
            for k in range(N):
                if dropout_mask[k] and k > 0:
                    pos_noisy[k] = pos_noisy[k - 1]
        elif params.dropout_mode == "remove":
            pass
        else:
            raise ValueError("dropout_mode must be one of: remove | hold | nan")

    return pos_noisy, dropout_mask, err


def plot_results(t: np.ndarray, pos: np.ndarray, pos_noisy: np.ndarray, dropout_mask: np.ndarray, err: np.ndarray, title: str) -> None:
    """
    Quick sanity-check plots.
    Uses matplotlib; imported only if --plot is requested.
    """
    import matplotlib.pyplot as plt

    # If dropouts are removed later, plotting here still uses full-length arrays
    # (dropout points are marked).
    xy_err = np.linalg.norm(err[:, 0:2], axis=1)
    z_err = np.abs(err[:, 2])

    fig1 = plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], label="RTK (ground truth)")
    plt.plot(pos_noisy[:, 0], pos_noisy[:, 1], label="Noisy GNSS baseline")
    if np.any(dropout_mask):
        # Mark dropout samples on noisy trajectory
        plt.scatter(pos_noisy[dropout_mask, 0], pos_noisy[dropout_mask, 1], marker="x", label="Dropout samples")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{title} — XY trajectory")
    plt.legend()
    plt.grid(True)

    fig2 = plt.figure()
    plt.plot(t, xy_err, label="|XY error| [m]")
    plt.plot(t, z_err, label="|Z error| [m]")
    if np.any(dropout_mask):
        # Add vertical markers at dropout times
        ymin, ymax = plt.ylim()
        plt.vlines(t[dropout_mask], ymin=ymin, ymax=ymax, linestyles="dotted", label="Dropout times")
        plt.ylim(ymin, ymax)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title(f"{title} — Error magnitude")
    plt.legend()
    plt.grid(True)

    fig3 = plt.figure()
    plt.plot(t, err[:, 0], label="ex [m]")
    plt.plot(t, err[:, 1], label="ey [m]")
    if np.any(dropout_mask):
        ymin, ymax = plt.ylim()
        plt.vlines(t[dropout_mask], ymin=ymin, ymax=ymax, linestyles="dotted", label="Dropout times")
        plt.ylim(ymin, ymax)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title(f"{title} — XY error components")
    plt.legend()
    plt.grid(True)

    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input RTK file (TUM-like; supports '#' comment header)")
    ap.add_argument("output", help="Output noisy GNSS file (TUM-like)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    ap.add_argument("--plot", action="store_true", help="Show sanity-check plots (requires matplotlib)")
    ap.add_argument("--plot-title", default="GNSS degradation", help="Title prefix for plots")

    # Key parameters exposed for easy overrides
    ap.add_argument("--sigma-w-xy", type=float, default=NoiseParams.sigma_w_xy)
    ap.add_argument("--sigma-w-z", type=float, default=NoiseParams.sigma_w_z)
    ap.add_argument("--sigma-b-xy", type=float, default=NoiseParams.sigma_b_xy)
    ap.add_argument("--sigma-b-z", type=float, default=NoiseParams.sigma_b_z)
    ap.add_argument("--tau", type=float, default=NoiseParams.tau)

    ap.add_argument("--outlier-prob", type=float, default=NoiseParams.outlier_prob)
    ap.add_argument("--outlier-scale-xy", type=float, default=NoiseParams.outlier_scale_xy)
    ap.add_argument("--outlier-scale-z", type=float, default=NoiseParams.outlier_scale_z)
    ap.add_argument("--outlier-dof", type=float, default=NoiseParams.outlier_dof)

    ap.add_argument("--dropout-rate", type=float, default=NoiseParams.dropout_rate)
    ap.add_argument("--dropout-mean-dur", type=float, default=NoiseParams.dropout_mean_dur)
    ap.add_argument("--dropout-mode", default=NoiseParams.dropout_mode, choices=["remove", "hold", "nan"])

    args = ap.parse_args()

    params = NoiseParams(
        sigma_w_xy=args.sigma_w_xy,
        sigma_w_z=args.sigma_w_z,
        sigma_b_xy=args.sigma_b_xy,
        sigma_b_z=args.sigma_b_z,
        tau=args.tau,
        outlier_prob=args.outlier_prob,
        outlier_scale_xy=args.outlier_scale_xy,
        outlier_scale_z=args.outlier_scale_z,
        outlier_dof=args.outlier_dof,
        dropout_rate=args.dropout_rate,
        dropout_mean_dur=args.dropout_mean_dur,
        dropout_mode=args.dropout_mode,
    )

    rng = np.random.default_rng(args.seed)

    data = read_tum_with_comments(args.input)
    t = data[:, 0].astype(float)
    pos = data[:, 1:4].copy()

    pos_noisy, dropout_mask, err = apply_noise(t, pos, params, rng)

    # Plot before "remove" is applied (so you can see where dropouts would occur)
    if args.plot:
        plot_results(t, pos, pos_noisy, dropout_mask, err, args.plot_title)

    # Write output: preserve extra columns (e.g., quaternion), only replace xyz.
    if np.any(dropout_mask) and params.dropout_mode == "remove":
        keep = ~dropout_mask
        data_out = data[keep].copy()
        data_out[:, 1:4] = pos_noisy[keep]
    else:
        data_out = data.copy()
        data_out[:, 1:4] = pos_noisy

    header = "timestamp tx ty tz qx qy qz qw (noisy baseline generated from RTK)"
    write_tum(args.output, data_out, header=header)

    # Terminal summary
    expected_outliers = params.outlier_prob * len(t)
    dropout_count = int(dropout_mask.sum())
    print(f"Wrote: {args.output}")
    print(f"Samples in:  {len(t)}")
    print(f"Samples out: {len(data_out)} (dropout_mode={params.dropout_mode})")
    print(f"Expected outliers ~ {expected_outliers:.1f} (p={params.outlier_prob})")
    print(f"Dropout samples (pre-remove): {dropout_count} (rate={params.dropout_rate}/s, mean_dur={params.dropout_mean_dur}s)")


if __name__ == "__main__":
    main()