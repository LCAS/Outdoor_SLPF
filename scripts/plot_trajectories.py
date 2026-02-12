import json
from pathlib import Path

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


def plot_trajectory_with_error_colors(ax, trajectory_pos, gt_positions=None, errors=None, label=None, landmark_points=None, cmap='viridis', vmin=None, vmax=None):
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
    ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
    ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    if errors is not None and len(points) > 1:
        cbar = plt.colorbar(lc, ax=ax, label='Error (m)')
        return cbar
    return None


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    trajectories = {
        'SPF LiDAR': {
            'trajectory': results_dir / 'spf_lidar' / 'spf_lidar.tum',
            'ground_truth': results_dir / 'spf_lidar' / 'gps_pose.tum'
        },
        'Noisy GPS': {
            # use the synthetic noisy GNSS (already in results) as the method trajectory
            'trajectory': results_dir / 'ngps_only' / 'noisy_gnss.tum',
            # compare against the common GPS ground truth stored with ngps results
            'ground_truth': results_dir / 'ngps_only' / 'gps_pose.tum'
        },
        'AMCL': {
            'trajectory': results_dir / 'amcl' / 'tum1' / 'amcl_pose.tum',
            # Compare AMCL against the shared GPS pose file like other methods
            'ground_truth': results_dir / 'amcl' / 'tum1' / 'gps_pose.tum'
        },
        'RTABMap RGBD': {
            'trajectory': results_dir / 'rtabmap' / 'rgbd' / 'tum1' / 'rtabmap_rgbd_filtered.tum',
            'ground_truth': results_dir / 'rtabmap' / 'rgbd' / 'tum1' / 'gps_pose.tum'
        },
        'RTABMap RGB': {
            'trajectory': results_dir / 'rtabmap' / 'rgb' / 'tum1' / 'rtabmap_rgb_filtered.tum',
            'ground_truth': results_dir / 'rtabmap' / 'rgb' / 'tum1' / 'gps_pose.tum'
        }
    }

    plot_data = []
    for label, paths in trajectories.items():
        print(f"Processing {label}...")

        gt_ts = gt_pos = gt_q = None
        if paths.get('ground_truth') is not None:
            gt_ts, gt_pos, gt_q = read_tum_file(str(paths['ground_truth']))
        traj_ts, traj_pos, traj_q = read_tum_file(str(paths['trajectory']))

        if traj_pos is None:
            print(f"  Warning: Could not read trajectory for {label}")
            continue

        # Initial alignment for RTABMap methods
        if 'RTABMap' in label:
            # Find closest GT index for the start of the trajectory
            idx0_gt = np.argmin(np.abs(gt_ts - traj_ts[0]))
            
            # Align only if we have reasonable overlap start
            if np.abs(gt_ts[idx0_gt] - traj_ts[0]) < 1.0:
                print(f"  Aligning {label} to ground truth start pose (with mirroring)...")
                traj_pos = align_trajectory(
                    traj_pos, traj_q, 
                    gt_pos[idx0_gt:], gt_q[idx0_gt:],
                    mirror=True
                )

        if gt_pos is not None:
            gt_interp = interpolate_ground_truth(gt_ts, gt_pos, traj_ts)
            errors = compute_errors(traj_pos, gt_interp)
            print(f"  Mean error: {errors.mean():.4f} m")
            print(f"  Max error: {errors.max():.4f} m")
            print(f"  Min error: {errors.min():.4f} m")
        else:
            errors = None
            print(f"  Note: {label} is lidar-only; skipping GT comparison.")

        plot_data.append({
            'label': label,
            'trajectory': traj_pos,
            'ground_truth': gt_pos,
            'errors': errors
        })

    if not plot_data:
        print("No trajectories to plot.")
        return

    landmark_points = load_landmark_points(data_dir / 'riseholme_poles_trunk.geojson')

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()

    # Define explicit min/max values for row-wise colormaps
    # Row 1 (indices 0, 1, 2): SPF LiDAR, Noisy GPS, AMCL
    vmin1, vmax1 = 0.0, 5.0

    # Row 2 (indices 3, 4): RTABMap methods
    vmin2, vmax2 = 0.0, 40.0

    for idx, item in enumerate(plot_data):
        # Assign shared range based on row
        if idx < 3:
            v_min, v_max = vmin1, vmax1
        else:
            v_min, v_max = vmin2, vmax2

        plot_trajectory_with_error_colors(
            axes[idx],
            item['trajectory'],
            item['ground_truth'],
            item['errors'],
            item['label'],
            landmark_points=landmark_points,
            cmap='viridis',
            vmin=v_min,
            vmax=v_max
        )

    for idx in range(len(plot_data), len(axes)):
        axes[idx].set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.25)
    output_path = results_dir / 'trajectory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == '__main__':
    main()
