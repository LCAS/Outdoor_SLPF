#!/usr/bin/env python3
"""Diagnostic plots and stats comparing SPF LiDAR vs Noisy GPS against ground truth.

Produces:
- results/spf_vs_gps_errors.png (time series + histograms)
- results/spf_vs_gps_trajectory.png (overlay on map)
- prints summary statistics and top outliers
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import json
from pyproj import Transformer


def read_tum_file(p):
    data = []
    with open(p, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    if not data:
        return None, None, None
    a = np.array(data)
    return a[:,0], a[:,1:4], a[:,4:8]


def quaternion_to_yaw(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def interpolate(gt_ts, gt_pos, targ_ts):
    p = np.zeros((len(targ_ts), 3))
    for i in range(3):
        p[:, i] = np.interp(targ_ts, gt_ts, gt_pos[:, i])
    return p


def umeyama_alignment(src, dst, with_scaling=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
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
    return scale, R, t


def compute_errors(est, gt):
    e = np.linalg.norm(est - gt, axis=1)
    return e


def main():
    base = Path(__file__).parent.parent
    data = base / 'data'
    results = base / 'results'
    results.mkdir(exist_ok=True)

    # files
    gt_f = results / 'spf_lidar' / 'gps_pose.tum'
    spf_f = results / 'spf_lidar' / 'spf_lidar.tum'
    gps_f = results / 'ngps_only' / 'trajectory_pf.tum'

    gt_ts, gt_pos, _ = read_tum_file(gt_f)
    spf_ts, spf_pos, spf_q = read_tum_file(spf_f)
    gps_ts, gps_pos, gps_q = read_tum_file(gps_f)

    # interpolate GT to each estimator timestamps
    gt_on_spf = interpolate(gt_ts, gt_pos, spf_ts)
    gt_on_gps = interpolate(gt_ts, gt_pos, gps_ts)

    # Umeyama-align both estimators to GT
    s_spf, R_spf, t_spf = umeyama_alignment(spf_pos, gt_on_spf, with_scaling=True)
    spf_umey = (s_spf * (R_spf @ spf_pos.T)).T + t_spf

    s_gps, R_gps, t_gps = umeyama_alignment(gps_pos, gt_on_gps, with_scaling=True)
    gps_umey = (s_gps * (R_gps @ gps_pos.T)).T + t_gps

    # errors
    err_spf = compute_errors(spf_umey, gt_on_spf)
    err_gps = compute_errors(gps_umey, gt_on_gps)

    # summary stats
    def stats(arr):
        return dict(rmse=float(np.sqrt(np.mean(arr**2))), mean=float(arr.mean()), median=float(np.median(arr)), max=float(arr.max()))

    s_spf_stats = stats(err_spf)
    s_gps_stats = stats(err_gps)

    print('SPF (Umeyama) RMSE: {:.3f} m'.format(s_spf_stats['rmse']))
    print('Noisy GPS (Umeyama) RMSE: {:.3f} m'.format(s_gps_stats['rmse']))

    # identify top outliers for SPF
    top_idx = np.argsort(-err_spf)[:10]
    print('\nTop SPF error timestamps and values:')
    for i in top_idx:
        print(f"{i}: t={spf_ts[i]:.3f}, err={err_spf[i]:.3f} m")

    # plot time series and histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_ts = axs[0,0]
    ax_hist = axs[0,1]
    ax_cdf = axs[1,0]
    ax_map = axs[1,1]

    ax_ts.plot(spf_ts - spf_ts[0], err_spf, label='SPF LiDAR', color='C0')
    ax_ts.plot(gps_ts - gps_ts[0], err_gps, label='Noisy GPS', color='C1')
    ax_ts.set_xlabel('time (s)')
    ax_ts.set_ylabel('position error (m)')
    ax_ts.legend()

    ax_hist.hist(err_spf, bins=50, alpha=0.7, label='SPF')
    ax_hist.hist(err_gps, bins=50, alpha=0.7, label='Noisy GPS')
    ax_hist.set_xlabel('error (m)')
    ax_hist.set_ylabel('count')
    ax_hist.legend()

    # CDF
    for arr, label, color in [(err_spf, 'SPF', 'C0'), (err_gps, 'GPS', 'C1')]:
        sorted_a = np.sort(arr)
        p = np.linspace(0,1,len(sorted_a))
        ax_cdf.plot(sorted_a, p, label=label, color=color)
    ax_cdf.set_xlabel('error (m)')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.legend()

    # overlay trajectories (centered using GT mean like plotting script)
    center = np.vstack((gt_pos, spf_pos, gps_pos)).mean(axis=0)
    gt_xy = gt_pos[:, :2] - center[:2]
    spf_xy = spf_pos[:, :2] - center[:2]
    gps_xy = gps_pos[:, :2] - center[:2]
    ax_map.plot(gt_xy[:,0], gt_xy[:,1], '--', color='gray', label='GT')
    ax_map.plot(spf_xy[:,0], spf_xy[:,1], '-', color='C0', label='SPF raw')
    ax_map.plot(gps_xy[:,0], gps_xy[:,1], '-', color='C1', label='GPS raw')
    ax_map.set_aspect('equal', 'box')
    ax_map.set_title('Trajectory overlay (raw poses)')
    ax_map.legend()

    plt.tight_layout()
    out = results / 'spf_vs_gps_errors.png'
    fig.savefig(out, dpi=200)
    print(f"Saved diagnostic plot to {out}")


if __name__ == '__main__':
    main()
