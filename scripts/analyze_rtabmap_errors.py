import numpy as np
import os

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

def compute_stats(traj_path, gt_path):
    t_traj, p_traj, _ = read_tum_file(traj_path)
    t_gt, p_gt, _ = read_tum_file(gt_path)

    if t_traj is None or t_gt is None:
        print("Error: Could not read one of the files.")
        return

    # Filter trajectory to be within ground truth time range
    mask = (t_traj >= t_gt[0]) & (t_traj <= t_gt[-1])
    t_traj = t_traj[mask]
    p_traj = p_traj[mask]

    if len(t_traj) == 0:
        print("Error: No overlapping timestamps.")
        return

    # Interpolate GT to match trajectory timestamps
    p_gt_interp = np.zeros((len(t_traj), 3))
    for i in range(3):
        p_gt_interp[:, i] = np.interp(t_traj, t_gt, p_gt[:, i])

    errors = np.linalg.norm(p_traj - p_gt_interp, axis=1)

    print(f"Min error: {np.min(errors):.4f}")
    print(f"Max error: {np.max(errors):.4f}")
    print(f"Median error: {np.median(errors):.4f}")
    print(f"90th percentile: {np.percentile(errors, 90):.4f}")

if __name__ == "__main__":
    from pathlib import Path

    def first_existing(*candidates):
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    base_dir = Path(__file__).parent.parent
    traj_path = first_existing(
        base_dir / "results/rtabmap/rgbd_run1_3runs/run1/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum",
        base_dir / "results/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum",
    )
    gt_path = first_existing(
        base_dir / "results/rtabmap/rgbd_run1_3runs/run1/rtabmap/rgbd/tum1/gps_pose.tum",
        base_dir / "results/rtabmap/rgbd/tum1/gps_pose.tum",
    )
    compute_stats(traj_path, gt_path)
