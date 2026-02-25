# Plot Sources Manifest

This folder centralizes generated plot artifacts.

## Plot Batch
- Output folder: `results/plots`
- Reference metrics/log run folder:
  - `results/iros_rh2/20260225_105822_multiseed_all_methods`

## Main 2x4 Comparison Figure
Files:
- `trajectory_comparison_2x4_experiment1_vs_experiment2_vertical_labels.png`
- `trajectory_comparison_2x4_experiment1_vs_experiment2_vertical_labels.pdf`

Generation script:
- `scripts/plot_trajectories_2x4_experiment_comparison.py`

Source folders used:
- Experiment 1 root:
  - `results/iros_rh1`
- Experiment 2 root:
  - `results/iros_rh2/20260225_105822_multiseed_all_methods`

Method-level trajectory inputs:
- Experiment 1
  - `SLPF(ours)`: `results/iros_rh1/spf_lidar++/0.5/trajectory_0.5.tum`
  - `AMCL+GPS`: `results/iros_rh1/amcl_ngps/tum1/trajectory_0.5.tum`
  - `RTABMAP RGBD`: `results/iros_rh1/rtabmap/rgbd/tum1/rtabmap_rgbd_filtered.tum`
  - `Noisy GPS`: `data/2025/noisy_gps/run1/noisy_gps_seed_11.tum`
- Experiment 2
  - `SLPF(ours)`: `results/iros_rh2/20260225_105822_multiseed_all_methods/slpf/seed_11/trajectory_0.5.tum`
  - `AMCL+GPS`: `results/iros_rh2/20260225_105822_multiseed_all_methods/amcl_ngps/seed_11/trajectory_0.5.tum`
  - `RTABMAP RGBD`: `results/iros_rh2/20260225_105822_multiseed_all_methods/rtab_rgbd/seed_11/rtabmap_rgbd_filtered.tum`
  - `Noisy GPS`: `results/iros_rh2/20260225_105822_multiseed_all_methods/ngps/seed_11/trajectory_0.5.tum`

Ground-truth pairing:
- All methods use their sibling `gps_pose.tum` in the same method folder.
- Exception: Experiment 1 `Noisy GPS` uses `data/2025/amcl/tum1/gps_pose.tum`.

Map/landmark source:
- `data/riseholme_poles_trunk.geojson`

RTK-GPS overlay inputs:
- Experiment 1 RTK trajectory: `data/2025/rh_run1/data.csv`
- Experiment 2 RTK trajectory: `data/2025/rh_run2/data.csv`
- RTK CSV coordinates are read from `latitude/longitude`, projected to `EPSG:32630`, then centered using the GeoJSON map centroid (same reference as landmarks).

Rendering settings for the main 2x4:
- Layout: `2 rows x 4 columns`
- Row labels: `Experiment 1`, `Experiment 2` (vertical on left)
- Method titles: shown only on the first row (second-row titles removed)
- RTABMAP alignment: Umeyama with scaling + start anchoring
- RTABMAP RGBD colormap range: `0-15 m`
- Noisy GPS downsampling:
  - Experiment 1: stride `20` (from full-resolution source)
  - Experiment 2: stride `20`
- RTK-GPS overlay: dash-dot red line on every subplot in the corresponding row
- RTK direction markers:
  - start marker: yellow `^`
  - end marker: cyan `v`
- Start/end robot markers: enabled
- Per-subplot legends: disabled
- Single global legend row: enabled at top
- Legend readability: font size `13 pt`
- Row-label spacing: moved closer to `Y (m)` labels
- Axis/grid centering: limits forced symmetric around `(0, 0)` on both axes
- Tick centering: `X` and `Y` ticks are symmetric around `0`
- Fixed plot extent: all subplots forced to `[-18, +18]` on both `X` and `Y`
- Tick set for fixed extent: `-18, -12, -6, 0, 6, 12, 18`
- Outer padding adjustment: reduced left margin, increased right margin
- Export crop: near-tight crop enabled (`bbox_inches='tight'`, `pad_inches=0.01`) for paper space efficiency
- Row spacing: reduced vertical gap between row 1 and row 2 (`hspace=0.14`)

## Vineyard Structure + RTK-Only Figure
Files:
- `vineyard_structure_with_rtk_run1_run2.png`
- `vineyard_structure_with_rtk_run1_run2.pdf`

Generation script:
- `scripts/plot_vineyard_structure_with_rtk.py`

Inputs:
- Map structure source: `data/riseholme_poles_trunk.geojson`
- RTK run1 source: `data/2025/rh_run1/data.csv`
- RTK run2 source: `data/2025/rh_run2/data.csv`

Rendering settings:
- Single-panel map view
- Landmarks: row posts + vines + semantic wall segments
- RTK overlay only (no estimated-method trajectories)
- RTK start/end markers shown for both runs
