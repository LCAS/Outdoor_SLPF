#!/usr/bin/env python3
"""Compute Absolute Trajectory Error (ATE) and Relative Trajectory Error (RTE).

Writes a CSV to `results/trajectory_metrics.csv` and prints a summary to stdout.
"""
from pathlib import Path
import numpy as np
import csv
import math
import json
import os

from geojson_rows import iter_projected_points


def read_tum_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    if not data:
        return None, None, None
    data = np.array(data)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]
    return timestamps, positions, quaternions


def quaternion_to_yaw(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return yaw_z


def interpolate_ground_truth(gt_ts, gt_pos, target_ts):
    p = np.zeros((len(target_ts), 3))
    for i in range(3):
        p[:, i] = np.interp(target_ts, gt_ts, gt_pos[:, i])
    return p


def align_first_pose(traj_pos, traj_q, gt_pos_interp, gt_q_interp, mirror=False):
    # Align trajectory so first pose matches ground truth first pose
    # Use yaw from quaternions and a 2D rotation + translation
    traj_pos = traj_pos.copy()
    if mirror:
        traj_pos[:, 1] = -traj_pos[:, 1]

    yaw_traj = quaternion_to_yaw(*traj_q[0])
    yaw_gt = quaternion_to_yaw(*gt_q_interp[0])
    yaw_diff = yaw_gt - yaw_traj
    c, s = math.cos(yaw_diff), math.sin(yaw_diff)
    R = np.array([[c, -s], [s, c]])

    local = traj_pos[:, :2] - traj_pos[0, :2]
    rotated = (R @ local.T).T
    aligned = np.zeros_like(traj_pos)
    aligned[:, :2] = rotated + gt_pos_interp[0, :2]
    aligned[:, 2] = traj_pos[:, 2] - traj_pos[0, 2] + gt_pos_interp[0, 2]
    return aligned


def compute_ate(est_aligned, gt_interp):
    errors = np.linalg.norm(est_aligned - gt_interp, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return {
        'rmse': float(rmse),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'max': float(np.max(errors))
    }


def umeyama_alignment(src, dst, with_scaling=True):
    """Compute similarity (s,R,t) that maps src -> dst using Umeyama method.
    src and dst are N x D arrays.
    Returns s, R, t where transformed = s * R @ src.T + t[:,None]
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    n, m = src.shape

    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)

    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov = (dst_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scaling:
        var_src = (src_centered ** 2).sum() / n
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0

    t = mean_dst - scale * R @ mean_src

    return scale, R, t


def apply_transform(points, scale, rot, trans):
    return (scale * (rot @ points.T)).T + trans


def _rms(arr):
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_smoothness_metrics(timestamps, aligned_positions):
    if len(timestamps) < 4:
        return {
            'speed_mean': None,
            'accel_rms': None,
            'jerk_rms': None,
            'heading_rate_rms': None,
            'heading_accel_rms': None
        }

    xy = aligned_positions[:, :2]
    dxy = np.diff(xy, axis=0)
    dt = np.diff(timestamps)
    valid = dt > 1e-6
    if not np.any(valid):
        return {
            'speed_mean': None,
            'accel_rms': None,
            'jerk_rms': None,
            'heading_rate_rms': None,
            'heading_accel_rms': None
        }

    dxy = dxy[valid]
    dt = dt[valid]
    speed = np.linalg.norm(dxy, axis=1) / dt

    if speed.size >= 2:
        dt_acc = 0.5 * (dt[1:] + dt[:-1])
        accel = np.diff(speed) / dt_acc
    else:
        accel = np.array([], dtype=np.float64)
        dt_acc = np.array([], dtype=np.float64)

    if accel.size >= 2:
        dt_jerk = 0.5 * (dt_acc[1:] + dt_acc[:-1])
        jerk = np.diff(accel) / dt_jerk
    else:
        jerk = np.array([], dtype=np.float64)

    heading = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
    if heading.size >= 2:
        dt_head = 0.5 * (dt[1:] + dt[:-1])
        heading_rate = np.diff(heading) / dt_head
    else:
        heading_rate = np.array([], dtype=np.float64)
        dt_head = np.array([], dtype=np.float64)

    if heading_rate.size >= 2:
        dt_head_acc = 0.5 * (dt_head[1:] + dt_head[:-1])
        heading_accel = np.diff(heading_rate) / dt_head_acc
    else:
        heading_accel = np.array([], dtype=np.float64)

    return {
        'speed_mean': float(np.mean(speed)) if speed.size else None,
        'accel_rms': _rms(accel),
        'jerk_rms': _rms(jerk),
        'heading_rate_rms': _rms(heading_rate),
        'heading_accel_rms': _rms(heading_accel)
    }


def compute_rte_at_distances(est_aligned, gt_interp, distances):
    # cumulative distances along gt_interp
    diffs = np.linalg.norm(gt_interp[1:] - gt_interp[:-1], axis=1)
    cum = np.concatenate(([0.0], np.cumsum(diffs)))
    results = {}
    n = len(gt_interp)
    for d in distances:
        errs = []
        counts = 0
        for i in range(n):
            # find j s.t. cum[j] - cum[i] >= d
            target = cum[i] + d
            # binary search
            j = np.searchsorted(cum, target, side='left')
            if j < n:
                gt_rel = gt_interp[j] - gt_interp[i]
                est_rel = est_aligned[j] - est_aligned[i]
                err = np.linalg.norm(est_rel - gt_rel)
                errs.append(err)
                counts += 1
        if errs:
            errs = np.array(errs)
            results[d] = {
                'mean': float(errs.mean()),
                'rmse': float(np.sqrt(np.mean(errs ** 2))),
                'count': int(counts)
            }
        else:
            results[d] = {'mean': None, 'rmse': None, 'count': 0}
    return results


def load_rows_from_geojson(geojson_path, target_crs='epsg:32630'):
    """Load rows from geojson and return a dict row_id -> ordered points (centered).
    Points are projected to target_crs and then centered by the mean of all features
    to match plotting conventions.
    """
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        return {}
    records = []
    for item in iter_projected_points(geojson_path, target_crs=target_crs):
        records.append({'x': item['x'], 'y': item['y'], 'row_id': item['row_id']})
    if not records:
        return {}
    all_xy = np.array([[r['x'], r['y']] for r in records])
    center = all_xy.mean(axis=0)
    rows = {}
    for r in records:
        displaced = (r['x'] - center[0], r['y'] - center[1])
        rows.setdefault(r['row_id'], []).append(displaced)

    # sort points along primary axis for each row and return as numpy arrays
    rows_sorted = {}
    for rid, pts in rows.items():
        pts_arr = np.array(pts)
        if pts_arr.shape[0] < 2:
            continue
        sort_dim = 0 if np.ptp(pts_arr[:, 0]) >= np.ptp(pts_arr[:, 1]) else 1
        order = np.argsort(pts_arr[:, sort_dim])
        rows_sorted[rid] = pts_arr[order]
    return rows_sorted


def point_segment_distance(p, a, b):
    # p, a, b are 2D
    pa = p - a
    ba = b - a
    denom = ba.dot(ba)
    if denom == 0:
        return np.linalg.norm(pa)
    t = max(0.0, min(1.0, pa.dot(ba) / denom))
    proj = a + t * ba
    return np.linalg.norm(p - proj)


def nearest_row_and_distance(pt, rows):
    # rows: dict row_id -> Nx2 array
    best_row = None
    best_dist = float('inf')
    for rid, pts in rows.items():
        # compute distance to polyline segments
        for i in range(len(pts) - 1):
            d = point_segment_distance(np.array(pt), pts[i], pts[i+1])
            if d < best_dist:
                best_dist = d
                best_row = rid
    return best_row, best_dist


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    results_override = os.environ.get('RESULTS_DIR')
    if results_override:
        results_dir = Path(results_override)
        if not results_dir.is_absolute():
            results_dir = base_dir / results_dir
    else:
        results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    def first_existing(*candidates):
        for p in candidates:
            path = Path(p)
            if path.exists():
                return path
        return Path(candidates[0])

    amcl_ngps_traj = first_existing(
        results_dir / 'amcl_ngps' / 'tum1' / 'trajectory_0.5.tum',
        results_dir / 'iros' / 'amcl_ngps' / 'tum1' / 'trajectory_0.5.tum',
    )
    amcl_ngps_gt = first_existing(
        results_dir / 'amcl_ngps' / 'tum1' / 'gps_pose.tum',
        results_dir / 'iros' / 'amcl_ngps' / 'tum1' / 'gps_pose.tum',
    )
    noisy_gps_traj_override = os.environ.get('NOISY_GPS_TUM', '').strip()
    if noisy_gps_traj_override:
        noisy_gps_traj = Path(noisy_gps_traj_override)
        if not noisy_gps_traj.is_absolute():
            noisy_gps_traj = base_dir / noisy_gps_traj
    else:
        noisy_gps_traj = first_existing(
            results_dir / 'noisy_gps' / 'noisy_gps_seed_11.tum',
            results_dir / 'noisy_gps' / 'noisy_gnss.tum',
            results_dir / 'ngps_only' / 'noisy_gnss.tum',
            results_dir / 'ngps_only-deprecated' / 'noisy_gnss.tum',
        )

    noisy_gps_gt_override = os.environ.get('NOISY_GPS_GT_TUM', '').strip()
    if noisy_gps_gt_override:
        noisy_gps_gt = Path(noisy_gps_gt_override)
        if not noisy_gps_gt.is_absolute():
            noisy_gps_gt = base_dir / noisy_gps_gt
    else:
        noisy_gps_gt = first_existing(
            results_dir / 'noisy_gps' / 'gps_pose.tum',
            results_dir / 'ngps_only' / 'gps_pose.tum',
            results_dir / 'ngps_only-deprecated' / 'gps_pose.tum',
        )
    rtab_rgbd_traj = first_existing(
        results_dir / 'rtabmap' / 'rgbd_run1_3runs' / 'run1' / 'rtabmap' / 'rgbd' / 'tum1' / 'rtabmap_rgbd_filtered.tum',
        results_dir / 'rtabmap' / 'rgbd' / 'tum1' / 'rtabmap_rgbd_filtered.tum',
    )
    rtab_rgbd_gt = first_existing(
        results_dir / 'rtabmap' / 'rgbd_run1_3runs' / 'run1' / 'rtabmap' / 'rgbd' / 'tum1' / 'gps_pose.tum',
        results_dir / 'rtabmap' / 'rgbd' / 'tum1' / 'gps_pose.tum',
    )
    rtab_rgb_traj = first_existing(
        results_dir / 'rtabmap' / 'rgb_run1_3runs' / 'run1' / 'rtabmap' / 'rgb' / 'tum1' / 'rtabmap_rgb_filtered.tum',
        results_dir / 'rtabmap' / 'rgb' / 'tum1' / 'rtabmap_rgb_filtered.tum',
    )
    rtab_rgb_gt = first_existing(
        results_dir / 'rtabmap' / 'rgb_run1_3runs' / 'run1' / 'rtabmap' / 'rgb' / 'tum1' / 'gps_pose.tum',
        results_dir / 'rtabmap' / 'rgb' / 'tum1' / 'gps_pose.tum',
    )

    trajectories = {
        'SPF LiDAR': {
            'trajectory': results_dir / 'spf_lidar' / 'spf_lidar.tum',
            'ground_truth': results_dir / 'spf_lidar' / 'gps_pose.tum'
        },
        'SPF LiDAR++': {
            'trajectory': results_dir / 'spf_lidar++' / '0.5' / 'trajectory_0.5.tum',
            'ground_truth': results_dir / 'spf_lidar++' / '0.5' / 'gps_pose.tum'
        },
        'Noisy GPS': {
            # treat the synthetic noisy GNSS (already in results) as the method trajectory
            'trajectory': noisy_gps_traj,
            # compare against the common GPS ground truth stored with ngps results
            'ground_truth': noisy_gps_gt
        },
        'AMCL': {
            'trajectory': results_dir / 'amcl' / 'tum1' / 'amcl_pose.tum',
            # Compare AMCL against the shared GPS pose file like other methods
            'ground_truth': results_dir / 'amcl' / 'tum1' / 'gps_pose.tum'
        },
        'AMCL+GPS': {
            'trajectory': amcl_ngps_traj,
            'ground_truth': amcl_ngps_gt
        },
        'RTABMap RGBD': {
            'trajectory': rtab_rgbd_traj,
            'ground_truth': rtab_rgbd_gt
        },
        'RTABMap RGB': {
            'trajectory': rtab_rgb_traj,
            'ground_truth': rtab_rgb_gt
        },
        'ORB-SLAM3 RGBD (s4)': {
            'trajectory': results_dir / 'orbslam3' / 'rgbd' / 's4' / 'orbslam3_rgbd.tum',
            'ground_truth': results_dir / 'orbslam3' / 'rgbd' / 's4' / 'gps_pose.tum'
        },
        'ORB-SLAM3 RGBD (full)': {
            'trajectory': results_dir / 'orbslam3' / 'rgbd' / 'full' / 'orbslam3_rgbd.tum',
            'ground_truth': results_dir / 'orbslam3' / 'rgbd' / 'full' / 'gps_pose.tum'
        },
        'ORB-SLAM3 Mono (s4)': {
            'trajectory': results_dir / 'orbslam3' / 'mono' / 's4' / 'orbslam3_mono.tum',
            'ground_truth': results_dir / 'orbslam3' / 'mono' / 's4' / 'gps_pose.tum'
        },
        'ORB-SLAM3 Mono (full)': {
            'trajectory': results_dir / 'orbslam3' / 'mono' / 'full' / 'orbslam3_mono.tum',
            'ground_truth': results_dir / 'orbslam3' / 'mono' / 'full' / 'gps_pose.tum'
        }
    }

    # Optional filter: comma-separated human-readable method labels.
    # Example:
    # METRICS_METHOD_LABELS="ORB-SLAM3 RGBD (full),ORB-SLAM3 Mono (full)"
    labels_filter = os.environ.get('METRICS_METHOD_LABELS', '').strip()
    if labels_filter:
        allowed = {x.strip() for x in labels_filter.split(',') if x.strip()}
        trajectories = {k: v for k, v in trajectories.items() if k in allowed}

    distances = [2.0, 5.0, 10.0]

    out_rows = []
    # load row centerlines from geojson
    geojson_override = os.environ.get('METRICS_GEOJSON_PATH', '').strip()
    if geojson_override:
        geojson_path = Path(geojson_override)
        if not geojson_path.is_absolute():
            geojson_path = base_dir / geojson_path
    else:
        geojson_path = data_dir / 'riseholme_poles_trunk.geojson'
    rows = load_rows_from_geojson(geojson_path)
    for label, paths in trajectories.items():
        print(f"Processing {label}...")
        traj_path = Path(paths['trajectory'])
        gt_path = Path(paths['ground_truth']) if paths.get('ground_truth') is not None else None

        if not traj_path.exists():
            print(f"  Skipping, trajectory file not found: {traj_path}")
            continue
        if gt_path is not None and not gt_path.exists():
            print(f"  Skipping, ground truth file not found: {gt_path}")
            continue

        # handle lidar-only methods with no ground truth specified
        if paths.get('ground_truth') is None:
            gt_ts = gt_pos = gt_q = None
        else:
            gt_ts, gt_pos, gt_q = read_tum_file(str(gt_path))

        traj_ts, traj_pos, traj_q = read_tum_file(str(traj_path))
        if traj_pos is None:
            print("  Skipping, could not read trajectory file.")
            continue
        if paths.get('ground_truth') is not None and gt_pos is None:
            print("  Skipping, could not read ground truth file.")
            continue

        if gt_pos is not None:
            # Interpolate GT to trajectory timestamps, then align estimator with
            # Umeyama SE(3) no-scale (same convention used by A/B validation).
            gt_interp = interpolate_ground_truth(gt_ts, gt_pos, traj_ts)
            try:
                s, R, t = umeyama_alignment(traj_pos, gt_interp, with_scaling=False)
                est_aligned = apply_transform(traj_pos, s, R, t)
            except Exception:
                est_aligned = traj_pos

            ate = compute_ate(est_aligned, gt_interp)
            rte = compute_rte_at_distances(est_aligned, gt_interp, distances)

            # Keep legacy umey columns for compatibility, now identical to aligned metrics.
            ate_umey = dict(ate)
            rte_umey = dict(rte)
            smooth = compute_smoothness_metrics(traj_ts, est_aligned)
        else:
            # lidar-only method: skip GT-based metrics
            gt_interp = None
            est_aligned = None
            ate = {'rmse': None, 'mean': None, 'median': None, 'max': None}
            rte = {d: {'mean': None, 'rmse': None, 'count': 0} for d in distances}
            ate_umey = {'rmse': None, 'mean': None, 'median': None, 'max': None}
            rte_umey = {d: {'mean': None, 'rmse': None, 'count': 0} for d in distances}
            smooth = {
                'speed_mean': None,
                'accel_rms': None,
                'jerk_rms': None,
                'heading_rate_rms': None,
                'heading_accel_rms': None
            }

        # Row-based metrics: cross-track error, row correctness, wrong-row events
        cross_track_errs = []
        est_rows = []
        gt_rows = []
        if gt_interp is not None and rows:
            n = len(gt_interp)
            for i in range(n):
                gt_pt = gt_interp[i, :2]
                est_pt = est_aligned[i, :2]
                # nearest row for GT and estimator
                gt_row, gt_row_dist = (None, None)
                gt_row, _ = nearest_row_and_distance(gt_pt, rows)
                est_row, est_row_dist = nearest_row_and_distance(est_pt, rows)
                gt_rows.append(gt_row)
                est_rows.append(est_row)
                # compute cross-track error as distance from estimator to GT row centerline
                if gt_row in rows:
                    # compute minimal distance to segments of gt_row
                    pts = rows[gt_row]
                    best_d = float('inf')
                    for j in range(len(pts) - 1):
                        d = point_segment_distance(est_pt, pts[j], pts[j+1])
                        if d < best_d:
                            best_d = d
                    cross_track_errs.append(best_d)
                else:
                    # fallback to distance to nearest row
                    cross_track_errs.append(est_row_dist)

            # summarize row metrics
            if cross_track_errs:
                ct_arr = np.array([c for c in cross_track_errs if c is not None])
                ct_mean = float(ct_arr.mean())
                ct_median = float(np.median(ct_arr))
                ct_max = float(ct_arr.max())
                # row correctness fraction
                matches = [1 if (e == g and e is not None) else 0 for e, g in zip(est_rows, gt_rows)]
                row_correct_frac = float(np.mean(matches)) if matches else None
                # wrong-row events: count entries into wrong state
                wrong = [1 if (e != g and e is not None and g is not None) else 0 for e, g in zip(est_rows, gt_rows)]
                switches = 0
                for idx_w in range(len(wrong)):
                    if wrong[idx_w] and (idx_w == 0 or not wrong[idx_w - 1]):
                        switches += 1
                row_switch_events = int(switches)
            else:
                ct_mean = ct_median = ct_max = None
                row_correct_frac = None
                row_switch_events = 0
        else:
            ct_mean = ct_median = ct_max = None
            row_correct_frac = None
            row_switch_events = 0

        row = {
            'method': label,
            'ate_rmse': ate['rmse'],
            'ate_mean': ate['mean'],
            'ate_median': ate['median'],
            'ate_max': ate['max']
        }
        for d in distances:
            row[f'rte_{int(d)}m_mean'] = rte[d]['mean']
            row[f'rte_{int(d)}m_rmse'] = rte[d]['rmse']
            row[f'rte_{int(d)}m_count'] = rte[d]['count']

        # RTE after Umeyama alignment
        for d in distances:
            row[f'rte_{int(d)}m_umey_mean'] = rte_umey[d]['mean']
            row[f'rte_{int(d)}m_umey_rmse'] = rte_umey[d]['rmse']
            row[f'rte_{int(d)}m_umey_count'] = rte_umey[d]['count']

        # append row-based metrics
        row['cross_track_mean'] = ct_mean
        row['cross_track_median'] = ct_median
        row['cross_track_max'] = ct_max
        row['row_correct_fraction'] = row_correct_frac
        row['row_switch_events'] = row_switch_events
        # append Umeyama ATE
        row['ate_umeyama_rmse'] = ate_umey['rmse']
        row['ate_umeyama_mean'] = ate_umey['mean']
        row['ate_umeyama_median'] = ate_umey['median']
        row['ate_umeyama_max'] = ate_umey['max']
        # Explicit aligned metric aliases for consistency with A/B naming.
        row['ate_align_rmse'] = ate['rmse']
        for d in distances:
            row[f'rpe_{int(d)}m_align_rmse'] = rte[d]['rmse']
        # Smoothness metrics used by A/B validation.
        row['speed_mean'] = smooth['speed_mean']
        row['accel_rms'] = smooth['accel_rms']
        row['jerk_rms'] = smooth['jerk_rms']
        row['heading_rate_rms'] = smooth['heading_rate_rms']
        row['heading_accel_rms'] = smooth['heading_accel_rms']

        out_rows.append(row)

        if row['ate_rmse'] is not None:
            print(f"  ATE RMSE: {row['ate_rmse']:.4f} m | mean: {row['ate_mean']:.4f} m | max: {row['ate_max']:.4f} m")
            for d in distances:
                r = rte[d]
                if r['count']:
                    print(f"  RTE @ {int(d)}m -> mean: {r['mean']:.4f} m, rmse: {r['rmse']:.4f} m, pairs: {r['count']}")
                else:
                    print(f"  RTE @ {int(d)}m -> no pairs found")
        else:
            print("  Note: lidar-only method; skipping GT-based metrics.")

    # Write CSV
    if out_rows:
        keys = list(out_rows[0].keys())
        out_path = results_dir / 'trajectory_metrics.csv'
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)
        print(f"\nMetrics written to {out_path}")
        # Generate a clean PDF summary reporting RMSE for ATE and RTE@X (Umeyama-aligned)
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import csv as _csv

            # Columns we want to present
            want_cols = ['method', 'ate_umeyama_rmse', 'rte_2m_umey_rmse', 'rte_5m_umey_rmse', 'rte_10m_umey_rmse']

            with open(out_path, 'r') as _f:
                reader = _csv.DictReader(_f)
                rows = list(reader)

            # Build table data with formatted numbers
            table_header = ['Method', 'ATE RMSE (m)', 'RTE@2m RMSE (m)', 'RTE@5m RMSE (m)', 'RTE@10m RMSE (m)']
            table_rows = []
            for r in rows:
                def fmt(val):
                    try:
                        v = float(r.get(val, ''))
                        return f"{v:.3f}"
                    except Exception:
                        return '-'
                table_rows.append([r.get('method', ''), fmt('ate_umeyama_rmse'), fmt('rte_2m_umey_rmse'), fmt('rte_5m_umey_rmse'), fmt('rte_10m_umey_rmse')])

            # Create a neat table on A4 landscape
            fig_width, fig_height = 11.7, 8.27
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis('off')

            # Title
            ax.text(0.5, 0.95, 'Trajectory Metrics Summary (RMSE)', ha='center', va='center', fontsize=14, weight='bold')

            # Create table
            table = ax.table(cellText=table_rows, colLabels=table_header, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            pdf_path = results_dir / 'trajectory_metrics_summary.pdf'
            pp = PdfPages(str(pdf_path))
            pp.savefig(fig, bbox_inches='tight')
            pp.close()
            plt.close(fig)
            print(f"PDF summary written to {pdf_path}")
        except Exception as e:
            print(f"Could not generate PDF summary: {e}")


if __name__ == '__main__':
    main()
