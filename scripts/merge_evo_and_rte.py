#!/usr/bin/env python3
from pathlib import Path
import csv
import os
import pandas as pd

base = Path(__file__).parent.parent
results_override = os.environ.get('RESULTS_DIR')
if results_override:
    results = Path(results_override)
    if not results.is_absolute():
        results = base / results
else:
    results = base / 'results'

# map display names to method keys used in evo_aggregated_metrics.csv
method_key_map = {
    'SPF LiDAR': 'spf',
    'SPF LiDAR++': 'spfpp',
    'Noisy GPS': 'ngps',
    'AMCL': 'amcl',
    'RTABMap RGBD': 'rtab_rgbd',
    'RTABMap RGB': 'rtab_rgb'
}

# read row-based metrics CSV
rte_df = pd.read_csv(results / 'trajectory_metrics.csv')
# build dict method -> row adherence values
rte_map = {}
for _, row in rte_df.iterrows():
    method = row['method']
    rte_map[method] = {
        'cross_track_mean': row.get('cross_track_mean'),
        'row_correct_fraction': row.get('row_correct_fraction'),
        'row_switch_events': row.get('row_switch_events')
    }

out_rows = []

# Read evo aggregate metrics to get deterministic ATE values
evo_csv_path = results / 'evo_aggregated_metrics.csv'
if not evo_csv_path.exists():
    raise FileNotFoundError(f'Missing {evo_csv_path}. Run scripts/aggregate_evo_results.py first.')

evo_df = pd.read_csv(evo_csv_path)
evo_map = {}
for _, row in evo_df.iterrows():
    key = str(row.get('method_key', '')).strip().lower()
    if key:
        evo_map[key] = row

method_list = ['SPF LiDAR', 'SPF LiDAR++', 'Noisy GPS', 'AMCL', 'RTABMap RGBD', 'RTABMap RGB']

for method in method_list:
    key = method_key_map.get(method)
    evo_row = evo_map.get(key, {})
    ate_rmse = evo_row.get('ape_umey_rmse')
    if pd.isna(ate_rmse):
        ate_rmse = evo_row.get('ape_raw_rmse')
    if pd.isna(ate_rmse):
        ate_rmse = None
    rte2 = evo_row.get('rpe_2m_rmse')
    if pd.isna(rte2):
        rte2 = None
    rte5 = evo_row.get('rpe_5m_rmse')
    if pd.isna(rte5):
        rte5 = None
    rte10 = evo_row.get('rpe_10m_rmse')
    if pd.isna(rte10):
        rte10 = None

    rte_vals = rte_map.get(method, {})
    out_rows.append({
        'method': method,
        'ate_evo_rmse': ate_rmse,
        'rte2_evo_rmse': rte2,
        'rte5_evo_rmse': rte5,
        'rte10_evo_rmse': rte10,
        'cross_track_mean': rte_vals.get('cross_track_mean'),
        'row_correct_fraction': rte_vals.get('row_correct_fraction'),
        'row_switch_events': rte_vals.get('row_switch_events')
    })

out_path = results / 'trajectory_metrics_evo.csv'
fieldnames = ['method','ate_evo_rmse','rte2_evo_rmse','rte5_evo_rmse','rte10_evo_rmse','cross_track_mean','row_correct_fraction','row_switch_events']
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)
print('Wrote', out_path)

# Generate a PDF summary table
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    table_header = ['Method', 'ATE RMSE (m)', 'RTE@2m RMSE (m)', 'RTE@5m RMSE (m)', 'RTE@10m RMSE (m)', 'Cross-track mean (m)', 'Row correct frac', 'Row switch events']
    table_rows = []
    for r in out_rows:
        def fmt(val):
            try:
                if val is None:
                    return '-'
                v = float(val)
                return f"{v:.3f}"
            except Exception:
                return str(val)
        table_rows.append([
            r.get('method',''),
            fmt(r.get('ate_evo_rmse')),
            fmt(r.get('rte2_evo_rmse')),
            fmt(r.get('rte5_evo_rmse')),
            fmt(r.get('rte10_evo_rmse')),
            fmt(r.get('cross_track_mean')),
            fmt(r.get('row_correct_fraction')),
            fmt(r.get('row_switch_events'))
        ])

    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Trajectory Metrics (EVO + Row Metrics)', ha='center', va='center', fontsize=14, weight='bold')
    table = ax.table(cellText=table_rows, colLabels=table_header, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    pdf_path = results / 'trajectory_metrics_evo_summary.pdf'
    pp = PdfPages(str(pdf_path))
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    plt.close(fig)
    print('PDF summary written to', pdf_path)
except Exception as e:
    print('Could not generate PDF summary:', e)
