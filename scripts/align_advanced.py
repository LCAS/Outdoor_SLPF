#!/usr/bin/env python3
"""Advanced alignment comparisons:
- continuous time-shift optimization + Umeyama
- Huber-weighted Umeyama (IRLS)

Writes results to `results/align_advanced.csv` and prints concise summary.
"""
from pathlib import Path
import numpy as np
import csv
import math
from functools import partial

try:
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None


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
        return np.array([]), np.array([])
    a = np.array(data)
    return a[:,0], a[:,1:4]


def interpolate(gt_ts, gt_pos, targ_ts):
    p = np.zeros((len(targ_ts), 3))
    for i in range(3):
        p[:, i] = np.interp(targ_ts, gt_ts, gt_pos[:, i])
    return p


def umeyama_alignment(src, dst, with_scaling=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
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


def apply_transform(p, scale, R, t):
    return (scale * (R @ p.T)).T + t


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def time_shift_optimize(est_ts, est_pos, gt_ts, gt_pos, bounds=(-2.0, 2.0)):
    """Optimize a continuous time shift (seconds) to minimize Umeyama RMSE.
    Returns best_shift and best_rmse.
    """
    def obj(shift):
        shifted = est_ts + shift
        gt_on_est = interpolate(gt_ts, gt_pos, shifted)
        s, R, t = umeyama_alignment(est_pos, gt_on_est, with_scaling=True)
        transformed = apply_transform(est_pos, s, R, t)
        return rmse(transformed, gt_on_est)

    if minimize_scalar is not None:
        res = minimize_scalar(obj, bounds=bounds, method='bounded')
        return float(res.x), float(res.fun)
    # fallback: simple grid
    shifts = np.linspace(bounds[0], bounds[1], 201)
    vals = [obj(s) for s in shifts]
    idx = int(np.argmin(vals))
    return float(shifts[idx]), float(vals[idx])


def huber_weights(residuals, delta):
    # residuals: positive norms
    w = np.ones_like(residuals)
    mask = residuals > delta
    w[mask] = delta / residuals[mask]
    return w


def weighted_umeyama(src, dst, weights=None, with_scaling=True, max_iter=20, tol=1e-6, delta=1.0):
    """Weighted/IRLS Umeyama using Huber weights if weights is None (iterative).
    Returns s,R,t
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if weights is None:
        # initialize with ordinary Umeyama
        s, R, t = umeyama_alignment(src, dst, with_scaling=with_scaling)
        for it in range(max_iter):
            transformed = apply_transform(src, s, R, t)
            res = np.linalg.norm(transformed - dst, axis=1)
            w = huber_weights(res, delta)
            if w.sum() == 0:
                break
            W = w / w.sum()
            mean_src = (W[:,None] * src).sum(axis=0)
            mean_dst = (W[:,None] * dst).sum(axis=0)
            src_c = src - mean_src
            dst_c = dst - mean_dst
            cov = (dst_c.T @ (W[:,None] * src_c))
            U, D, Vt = np.linalg.svd(cov)
            S = np.eye(3)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S[-1, -1] = -1
            R_new = U @ S @ Vt
            if with_scaling:
                var_src = (W * np.sum(src_c**2, axis=1)).sum()
                scale_new = np.trace(np.diag(D) @ S) / var_src
            else:
                scale_new = 1.0
            t_new = mean_dst - scale_new * R_new @ mean_src
            # check convergence
            if np.allclose(R, R_new, atol=tol) and abs(s - scale_new) < tol and np.allclose(t, t_new, atol=tol):
                s, R, t = scale_new, R_new, t_new
                break
            s, R, t = scale_new, R_new, t_new
        return s, R, t
    else:
        w = np.asarray(weights, dtype=np.float64)
        W = w / w.sum()
        mean_src = (W[:,None] * src).sum(axis=0)
        mean_dst = (W[:,None] * dst).sum(axis=0)
        src_c = src - mean_src
        dst_c = dst - mean_dst
        cov = (dst_c.T @ (W[:,None] * src_c))
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1
        R = U @ S @ Vt
        if with_scaling:
            var_src = (W * np.sum(src_c**2, axis=1)).sum()
            scale = np.trace(np.diag(D) @ S) / var_src
        else:
            scale = 1.0
        t = mean_dst - scale * R @ mean_src
        return scale, R, t


def main():
    base = Path(__file__).parent.parent
    data = base / 'data'
    results = base / 'results'
    results.mkdir(exist_ok=True)

    gt_f = results / 'spf_lidar' / 'gps_pose.tum'
    spf_f = results / 'spf_lidar' / 'spf_lidar.tum'
    gps_f = results / 'ngps_only' / 'trajectory_pf.tum'

    gt_ts, gt_pos = read_tum_file(gt_f)
    spf_ts, spf_pos = read_tum_file(spf_f)
    gps_ts, gps_pos = read_tum_file(gps_f)

    # 1) continuous time-shift + Umeyama
    sh_spf, rmse_spf = time_shift_optimize(spf_ts, spf_pos, gt_ts, gt_pos, bounds=(-2.0, 2.0))
    sh_gps, rmse_gps = time_shift_optimize(gps_ts, gps_pos, gt_ts, gt_pos, bounds=(-2.0, 2.0))

    # 2) Huber-weighted Umeyama (no time shift)
    gt_on_spf = interpolate(gt_ts, gt_pos, spf_ts)
    s_h_spf, R_h_spf, t_h_spf = weighted_umeyama(spf_pos, gt_on_spf, with_scaling=True, delta=1.0)
    rmse_h_spf = rmse(apply_transform(spf_pos, s_h_spf, R_h_spf, t_h_spf), gt_on_spf)

    gt_on_gps = interpolate(gt_ts, gt_pos, gps_ts)
    s_h_gps, R_h_gps, t_h_gps = weighted_umeyama(gps_pos, gt_on_gps, with_scaling=True, delta=1.0)
    rmse_h_gps = rmse(apply_transform(gps_pos, s_h_gps, R_h_gps, t_h_gps), gt_on_gps)

    outp = results / 'align_advanced.csv'
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'spf_rmse', 'ngps_rmse', 'extra'])
        w.writerow(['timeshift_umeyama', f'{rmse_spf:.6f}', f'{rmse_gps:.6f}', f'spf_shift={sh_spf:.3f},ngps_shift={sh_gps:.3f}'])
        w.writerow(['huber_umeyama', f'{rmse_h_spf:.6f}', f'{rmse_h_gps:.6f}', 'delta=1.0'])

    print('Results:')
    print(f'Time-shift+Umeyama: SPF RMSE={rmse_spf:.3f} m (shift={sh_spf:.3f}s), NGPS RMSE={rmse_gps:.3f} m (shift={sh_gps:.3f}s)')
    print(f'Huber-Umeyama:     SPF RMSE={rmse_h_spf:.3f} m, NGPS RMSE={rmse_h_gps:.3f} m')
    print('Wrote', outp)


if __name__ == '__main__':
    main()
