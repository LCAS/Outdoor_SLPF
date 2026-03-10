#!/usr/bin/env python3
"""Compare alignment strategies and report ATE for SPF and Noisy GPS.

Methods:
- Umeyama similarity (with scale)
- Umeyama without scaling
- RANSAC-Umeyama (robust to outliers)
- Time-shift search + Umeyama

Prints RMSE for each method and saves a small CSV summary to results/align_comparison.csv
"""
from pathlib import Path
import numpy as np
import csv


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


def ransac_umeyama(src, dst, iters=200, sample=20, inlier_thresh=2.0):
    best = None
    n = len(src)
    idxs = np.arange(n)
    for _ in range(iters):
        if n <= sample:
            samp = idxs
        else:
            samp = np.random.choice(idxs, sample, replace=False)
        try:
            s, R, t = umeyama_alignment(src[samp], dst[samp], with_scaling=True)
        except Exception:
            continue
        transformed = apply_transform(src, s, R, t)
        errs = np.linalg.norm(transformed - dst, axis=1)
        inliers = errs <= inlier_thresh
        score = inliers.sum()
        if best is None or score > best[0]:
            best = (score, s, R, t, inliers, errs)
    return best


def time_shift_search(est_ts, est_pos, gt_ts, gt_pos, shifts, with_scaling=True):
    best = None
    for sh in shifts:
        shifted_ts = est_ts + sh
        gt_on_est = interpolate(gt_ts, gt_pos, shifted_ts)
        s, R, t = umeyama_alignment(est_pos, gt_on_est, with_scaling=with_scaling)
        transformed = apply_transform(est_pos, s, R, t)
        e = rmse(transformed, gt_on_est)
        if best is None or e < best[0]:
            best = (e, sh, s, R, t)
    return best


def compare_one(name, est_ts, est_pos, gt_ts, gt_pos, results_dir):
    out = []
    gt_on_est = interpolate(gt_ts, gt_pos, est_ts)

    # Umeyama (with scale)
    s1, R1, t1 = umeyama_alignment(est_pos, gt_on_est, with_scaling=True)
    e1 = rmse(apply_transform(est_pos, s1, R1, t1), gt_on_est)
    out.append(('umeyama', e1))

    # Umeyama no scale
    s2, R2, t2 = umeyama_alignment(est_pos, gt_on_est, with_scaling=False)
    e2 = rmse(apply_transform(est_pos, s2, R2, t2), gt_on_est)
    out.append(('umeyama_noscale', e2))

    # RANSAC Umeyama
    r = ransac_umeyama(est_pos, gt_on_est, iters=300, sample=max(10, min(50, len(est_pos)//4)), inlier_thresh=2.0)
    if r:
        score, s3, R3, t3, inliers, errs = r
        e3 = rmse(apply_transform(est_pos, s3, R3, t3), gt_on_est)
        out.append(('ransac_umeyama', e3, int(score), float(errs.mean())))
    else:
        out.append(('ransac_umeyama', None, 0, None))

    # time shift search
    shifts = np.linspace(-2.0, 2.0, 41)
    best = time_shift_search(est_ts, est_pos, gt_ts, gt_pos, shifts, with_scaling=True)
    if best:
        e_shift, sh, s4, R4, t4 = best
        out.append(('timeshift_umeyama', e_shift, float(sh)))
    else:
        out.append(('timeshift_umeyama', None, None))

    # write brief CSV
    csvp = results_dir / f'align_{name}.csv'
    with open(csvp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'rmse', 'extra1', 'extra2'])
        for row in out:
            w.writerow(row)
    return out


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

    print('Running comparisons...')
    res_spf = compare_one('spf', spf_ts, spf_pos, gt_ts, gt_pos, results)
    res_gps = compare_one('ngps', gps_ts, gps_pos, gt_ts, gt_pos, results)

    summary = results / 'align_comparison.csv'
    with open(summary, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'variant', 'spf_rmse', 'ngps_rmse'])
        # collect keys
        keys = [r[0] for r in res_spf]
        for i, k in enumerate(keys):
            a = res_spf[i]
            b = res_gps[i]
            # normalize row lengths
            rmse_a = a[1] if len(a) > 1 else ''
            rmse_b = b[1] if len(b) > 1 else ''
            w.writerow([k, '', rmse_a, rmse_b])

    print('Wrote results to', summary)
    for r in res_spf:
        print('SPF:', r)
    for r in res_gps:
        print('NGPS:', r)


if __name__ == '__main__':
    main()
