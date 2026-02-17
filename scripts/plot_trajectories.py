import json
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pyproj import Transformer


def read_tum_file(filepath):
    """
    Read a TUM format file.
    
    Format: timestamp tx ty tz qx qy qz qw
    Returns: timestamps, positions (Nx3), quaternions (Nx4)
    """
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


def interpolate_ground_truth(gt_timestamps, gt_positions, query_timestamps):
    """
    Interpolate ground truth positions to match trajectory timestamps.
    """
    interpolated_positions = np.zeros((len(query_timestamps), 3))
    for i in range(3):
        interpolated_positions[:, i] = np.interp(
            query_timestamps,
            gt_timestamps,
            gt_positions[:, i]
        )
    return interpolated_positions


def compute_errors(trajectory_positions, gt_positions):
    """
    Compute position errors (Euclidean distance) between trajectory and ground truth.
    """
    errors = np.linalg.norm(trajectory_positions - gt_positions, axis=1)
    return errors


def quaternion_to_yaw(q):
    """Convert a quaternion [qx, qy, qz, qw] to yaw."""
    x, y, z, w = q
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)


def align_trajectory(traj_pos, traj_q, gt_pos, gt_q, mirror=False):
    """
    Align the trajectory to the ground truth using the first point's translation and rotation.
    Optionally mirrors the trajectory across the local Y-axis before alignment.
    """
    # Use first points for alignment
    p0_traj = traj_pos[0, :2]
    p0_gt = gt_pos[0, :2]
    
    yaw0_traj = quaternion_to_yaw(traj_q[0])
    if mirror:
        yaw0_traj = -yaw0_traj
        
    yaw0_gt = quaternion_to_yaw(gt_q[0])
    
    # Calculate relative rotation
    delta_yaw = yaw0_gt - yaw0_traj
    
    # Create rotation matrix
    cos_y = np.cos(delta_yaw)
    sin_y = np.sin(delta_yaw)
    R = np.array([
        [cos_y, -sin_y],
        [sin_y, cos_y]
    ])
    
    # Transform coordinates
    traj_pos_2d = traj_pos[:, :2]
    
    # Shift to local origin
    local_pos = traj_pos_2d - p0_traj
    
    # Mirror Y if requested
    if mirror:
        local_pos[:, 1] = -local_pos[:, 1]
    
    # Rotate and shift to GT's start
    aligned_pos_2d = local_pos @ R.T + p0_gt
    
    # Update only X and Y, keep Z as is
    aligned_pos = np.copy(traj_pos)
    aligned_pos[:, :2] = aligned_pos_2d
    
    return aligned_pos


def umeyama_alignment(src, dst, with_scaling=False):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch src={src.shape}, dst={dst.shape}")

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


def apply_transform(points, scale, rot, trans):
    return (scale * (rot @ points.T)).T + trans


def anchor_start_to_ground_truth(traj_pos, gt_pos):
    """
    Shift an already-aligned trajectory so its first sample matches GT start.
    """
    if traj_pos is None or gt_pos is None or len(traj_pos) == 0 or len(gt_pos) == 0:
        return traj_pos
    delta = gt_pos[0] - traj_pos[0]
    return traj_pos + delta


def stride_keep_end(arr, stride):
    """
    Downsample an array with fixed stride while always preserving first/last element.
    """
    if arr is None or stride <= 1 or len(arr) <= 2:
        return arr
    idx = np.arange(0, len(arr), stride)
    if idx[-1] != len(arr) - 1:
        idx = np.append(idx, len(arr) - 1)
    return arr[idx]


def load_landmark_points(geojson_path, target_crs='epsg:32630'):
    """Return centered poles, vines, and row segments in projected coordinates."""
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        return {'poles': np.empty((0, 2)), 'trunks': np.empty((0, 2)), 'segments_pole': np.empty((0, 2, 2)), 'segments_trunk': np.empty((0, 2, 2))}

    transformer = Transformer.from_crs('epsg:4326', target_crs, always_xy=True)
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    features = data.get('features', [])
    records = []
    for feature in features:
        geom = feature.get('geometry') or {}
        if geom.get('type') != 'Point':
            continue
        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        x, y = transformer.transform(lon, lat)
        props = feature.get('properties', {})
        feature_type = props.get('feature_type', '').lower()
        row_id = props.get('vine_vine_row_id') or ''
        if not row_id:
            row_post_id = props.get('row_post_id', '')
            row_id = row_post_id.rsplit('_post_', 1)[0] if '_post_' in row_post_id else row_post_id
        row_id = row_id or 'unknown'
        records.append({'x': x, 'y': y, 'type': feature_type, 'row_id': row_id})

    if not records:
        return {'poles': np.empty((0, 2)), 'trunks': np.empty((0, 2)), 'segments_pole': np.empty((0, 2, 2)), 'segments_trunk': np.empty((0, 2, 2))}

    all_xy = np.array([[r['x'], r['y']] for r in records])
    center = all_xy.mean(axis=0)

    categories = {'poles': [], 'trunks': []}
    rows = {}
    for r in records:
        displaced = (r['x'] - center[0], r['y'] - center[1])
        if r['type'] == 'row_post':
            categories['poles'].append(displaced)
        else:
            categories['trunks'].append(displaced)
        
        rows.setdefault(r['row_id'], []).append({
            'coords': displaced,
            'type': r['type']
        })

    segments_pole = []
    segments_trunk = []
    for pts in rows.values():
        if len(pts) < 2:
            continue
        
        # Sort points in the row
        pts_coords = np.array([p['coords'] for p in pts])
        sort_dim = 0 if np.ptp(pts_coords[:, 0]) >= np.ptp(pts_coords[:, 1]) else 1
        order = np.argsort(pts_coords[:, sort_dim])
        sorted_pts = [pts[i] for i in order]

        for i in range(len(sorted_pts) - 1):
            p1 = sorted_pts[i]
            p2 = sorted_pts[i+1]
            seg = np.array([p1['coords'], p2['coords']])
            
            # If segment connects at least one pole -> pole class
            if p1['type'] == 'row_post' or p2['type'] == 'row_post':
                segments_pole.append(seg)
            else:
                segments_trunk.append(seg)

    return {
        'poles': np.array(categories['poles']) if categories['poles'] else np.empty((0, 2)),
        'trunks': np.array(categories['trunks']) if categories['trunks'] else np.empty((0, 2)),
        'segments_pole': np.array(segments_pole) if segments_pole else np.empty((0, 2, 2)),
        'segments_trunk': np.array(segments_trunk) if segments_trunk else np.empty((0, 2, 2))
    }


def plot_trajectory_with_error_colors(
    ax,
    trajectory_pos,
    gt_positions=None,
    errors=None,
    label=None,
    landmark_points=None,
    cmap='viridis',
    vmin=None,
    vmax=None,
    fixed_xlim=None,
    fixed_ylim=None,
    show_ylabel=True,
):
    """
    Plot a trajectory with colors representing errors, overlay its GPS ground truth path,
    and optionally add mapped landmarks.
    """
    points = trajectory_pos[:, :2]
    segments = np.array([points[i:i+2] for i in range(len(points)-1)]) if len(points) > 1 else np.empty((0, 2, 2))
    
    # Use provided vmin/vmax if available, else derive from current errors
    plot_vmin = vmin if vmin is not None else errors.min()
    plot_vmax = vmax if vmax is not None else errors.max()
    norm = Normalize(vmin=plot_vmin, vmax=plot_vmax)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2)
    if len(points) > 1:
        if errors is not None:
            lc.set_array(errors[:-1])
            ax.add_collection(lc)
        else:
            ax.plot(points[:, 0], points[:, 1], linestyle='-', color='tab:blue')
    else:
        ax.plot(points[:, 0], points[:, 1], linestyle='-', color='tab:blue')

    if gt_positions is not None:
        gt_xy = gt_positions[:, :2]
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color='gray', linestyle='--', linewidth=1.5, label='GPS ground truth')
    else:
        gt_xy = np.empty((0, 2))

    extra_points = []
    if landmark_points is not None:
        poles = landmark_points.get('poles', np.empty((0, 2)))
        trunks = landmark_points.get('trunks', np.empty((0, 2)))
        if poles.size:
            ax.scatter(poles[:, 0], poles[:, 1], s=12, edgecolor='black', facecolor='orange', linewidth=0.3, label='Row posts')
            extra_points.append(poles)
        if trunks.size:
            ax.scatter(trunks[:, 0], trunks[:, 1], s=8, marker='s', color='forestgreen', alpha=0.7, label='Vines')
            extra_points.append(trunks)
            
        seg_pole = landmark_points.get('segments_pole', np.empty((0, 2, 2)))
        if seg_pole.size:
            lc_pole = LineCollection(seg_pole, colors='orange', linewidths=0.8, alpha=0.5, label='Semantic wall (Pole)')
            ax.add_collection(lc_pole)
            extra_points.append(seg_pole.reshape(-1, 2))
            
        seg_trunk = landmark_points.get('segments_trunk', np.empty((0, 2, 2)))
        if seg_trunk.size:
            lc_trunk = LineCollection(seg_trunk, colors='forestgreen', linewidths=0.8, alpha=0.5, label='Semantic wall (Trunk)')
            ax.add_collection(lc_trunk)
            extra_points.append(seg_trunk.reshape(-1, 2))

    extra_mat = np.vstack(extra_points) if extra_points else None
    # gather points for axis limits
    parts = [points]
    if gt_xy.size:
        parts.append(gt_xy)
    if extra_mat is not None:
        parts.append(extra_mat)
    all_points = np.vstack(parts)

    if fixed_xlim is not None and fixed_ylim is not None:
        ax.set_xlim(*fixed_xlim)
        ax.set_ylim(*fixed_ylim)
        ax.set_xticks(np.arange(fixed_xlim[0], fixed_xlim[1] + 0.001, 5.0))
        ax.set_yticks(np.arange(fixed_ylim[0], fixed_ylim[1] + 0.001, 5.0))
    else:
        ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
        ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)' if show_ylabel else '')
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='lower right')
    
    if errors is not None and len(points) > 1:
        cbar = plt.colorbar(lc, ax=ax, label='Error (m)')
        return cbar
    return None


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    data_dir = base_dir / 'data'
    results_override = os.environ.get('RESULTS_DIR')
    if results_override:
        results_dir = Path(results_override)
        if not results_dir.is_absolute():
            results_dir = base_dir / results_dir
    else:
        results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    noisy_gps_stride = int(os.environ.get('NOISY_GPS_STRIDE', '4'))

    amcl_gps_traj = results_dir / 'amcl_ngps' / 'tum1' / 'trajectory_0.5.tum'
    amcl_gps_gt = results_dir / 'amcl_ngps' / 'tum1' / 'gps_pose.tum'
    use_amcl_gps = amcl_gps_traj.exists() and amcl_gps_gt.exists()
    amcl_label = 'AMCL+GPS' if use_amcl_gps else 'AMCL'
    amcl_traj = amcl_gps_traj if use_amcl_gps else (results_dir / 'amcl' / 'tum1' / 'amcl_pose.tum')
    amcl_gt = amcl_gps_gt if use_amcl_gps else (results_dir / 'amcl' / 'tum1' / 'gps_pose.tum')

    trajectories = {
        'SLPF(ours)': {
            'trajectory': results_dir / 'spf_lidar++' / '0.5' / 'trajectory_0.5.tum',
            'ground_truth': results_dir / 'spf_lidar++' / '0.5' / 'gps_pose.tum',
            'stride': 1
        },
        amcl_label: {
            'trajectory': amcl_traj,
            # Compare against the shared GPS pose file like other methods
            'ground_truth': amcl_gt,
            'stride': 1
        },
        'RTABMAP RGBD': {
            'trajectory': results_dir / 'rtabmap' / 'rgbd' / 'tum1' / 'rtabmap_rgbd_filtered.tum',
            'ground_truth': results_dir / 'rtabmap' / 'rgbd' / 'tum1' / 'gps_pose.tum',
            'stride': 1
        },
        'Noisy GPS': {
            # use the synthetic noisy GNSS (already in results) as the method trajectory
            'trajectory': results_dir / 'ngps_only' / 'noisy_gnss.tum',
            # compare against the common GPS ground truth stored with ngps results
            'ground_truth': results_dir / 'ngps_only' / 'gps_pose.tum',
            'stride': noisy_gps_stride
        }
    }
    # Keep explicit start-point anchoring only for RTAB-Map.
    anchor_start_labels = {'RTABMAP RGBD'}

    plot_data = []
    for label, paths in trajectories.items():
        print(f"Processing {label}...")
        traj_path = Path(paths['trajectory'])
        gt_path = Path(paths['ground_truth']) if paths.get('ground_truth') is not None else None
        stride = max(1, int(paths.get('stride', 1)))

        if not traj_path.exists():
            print(f"  Warning: trajectory file not found, skipping: {traj_path}")
            continue
        if gt_path is not None and not gt_path.exists():
            print(f"  Warning: ground-truth file not found, skipping: {gt_path}")
            continue

        gt_ts = gt_pos = gt_q = None
        if paths.get('ground_truth') is not None:
            gt_ts, gt_pos, gt_q = read_tum_file(str(gt_path))
        traj_ts, traj_pos, traj_q = read_tum_file(str(traj_path))

        if traj_pos is None:
            print(f"  Warning: Could not read trajectory for {label}")
            continue

        if gt_pos is not None:
            gt_interp = interpolate_ground_truth(gt_ts, gt_pos, traj_ts)
            # Keep visualization in the same aligned frame used for metrics:
            # Umeyama SE(3) alignment without scaling.
            try:
                scale, rot, trans = umeyama_alignment(traj_pos, gt_interp, with_scaling=False)
                traj_plot = apply_transform(traj_pos, scale, rot, trans)
            except Exception:
                traj_plot = traj_pos

            if label in anchor_start_labels:
                applied_shift = gt_interp[0] - traj_plot[0]
                traj_plot = anchor_start_to_ground_truth(traj_plot, gt_interp)
                print(
                    f"  Start anchored to GT "
                    f"(applied dx={applied_shift[0]:.4f}, dy={applied_shift[1]:.4f}, dz={applied_shift[2]:.4f})"
                )

            if stride > 1:
                traj_plot = stride_keep_end(traj_plot, stride)
                gt_interp = stride_keep_end(gt_interp, stride)
                print(f"  Stride applied: every {stride} samples")

            errors = compute_errors(traj_plot, gt_interp)
            print(f"  Mean error: {errors.mean():.4f} m")
            print(f"  Max error: {errors.max():.4f} m")
            print(f"  Min error: {errors.min():.4f} m")
        else:
            traj_plot = traj_pos
            errors = None
            print(f"  Note: {label} is lidar-only; skipping GT comparison.")

        plot_data.append({
            'label': label,
            'trajectory': traj_plot,
            'ground_truth': gt_interp if gt_pos is not None else None,
            'errors': errors
        })

    if not plot_data:
        print("No trajectories to plot.")
        return

    landmark_points = load_landmark_points(data_dir / 'riseholme_poles_trunk.geojson')

    n_plots = len(plot_data)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 5.0 * n_rows),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    axes = np.atleast_1d(axes).flatten()

    # Per-method colormap ranges keep low-error and high-error methods readable.
    method_vrange = {
        'SLPF(ours)': (0.0, 5.0),
        'Noisy GPS': (0.0, 10.0),
        'AMCL': (0.0, 5.0),
        'AMCL+GPS': (0.0, 5.0),
        'RTABMAP RGBD': (0.0, 40.0),
    }

    for idx, item in enumerate(plot_data):
        v_min, v_max = method_vrange.get(item['label'], (0.0, 10.0))
        is_left_col = (idx % n_cols) == 0

        plot_trajectory_with_error_colors(
            axes[idx],
            item['trajectory'],
            item['ground_truth'],
            item['errors'],
            item['label'],
            landmark_points=landmark_points,
            cmap='viridis',
            vmin=v_min,
            vmax=v_max,
            show_ylabel=is_left_col,
        )

    for idx in range(len(plot_data), len(axes)):
        axes[idx].set_visible(False)

    output_png = results_dir / 'trajectory_comparison.png'
    output_pdf = results_dir / 'trajectory_comparison.pdf'
    plt.savefig(output_png, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.03)
    print(f"\nPlot saved to {output_png}")
    print(f"Plot saved to {output_pdf}")


if __name__ == '__main__':
    main()
