#!/usr/bin/env python3
from pathlib import Path
import zipfile, json, csv
import pandas as pd

base = Path(__file__).parent.parent
results = base / 'results'

# map method names between our CSV and evo files
method_map = {
    'SPF LiDAR': 'SPF',
    'SPF++': 'SPF++',
    'Noisy GPS': 'NoisyGPS',
    'AMCL': 'AMCL',
    'RTABMap RGBD': 'RTAB_RGBD',
    'RTABMap RGB': 'RTAB_RGB'
}

evo_files = {v: results / f'evo_{k.lower()}_raw.json' for k, v in [('spf','SPF'), ('ngps','NoisyGPS'), ('amcl','AMCL'), ('rtab_rgbd','RTAB_RGBD'), ('rtab_rgb','RTAB_RGB')]}
# fallback to previously-saved names if raw absent
for k in list(evo_files.keys()):
    if not evo_files[k].exists():
        # try umey file names
        evo_files[k] = results / evo_files[k].name.replace('_raw','_umey')

# read our RTE CSV
rte_df = pd.read_csv(results / 'trajectory_metrics.csv')
# build dict method -> rte values
rte_map = {}
def first_valid(*vals):
    for v in vals:
        if pd.notna(v):
            return v
    return None

for _, row in rte_df.iterrows():
    method = row['method']
    rte_map[method] = {
        # prefer Umeyama-aligned RTE columns (consistent with compute_metrics PDF)
        'rte2': first_valid(row.get('rte_2m_umey_rmse'), row.get('rte_2m_rmse')),
        'rte5': first_valid(row.get('rte_5m_umey_rmse'), row.get('rte_5m_rmse')),
        'rte10': first_valid(row.get('rte_10m_umey_rmse'), row.get('rte_10m_rmse')),
        'cross_track_mean': row.get('cross_track_mean'),
        'row_correct_fraction': row.get('row_correct_fraction'),
        'row_switch_events': row.get('row_switch_events')
    }

out_rows = []

# Pre-scan all evo archives and extract their info/stats to map to methods robustly
file_info = {}
for p in results.glob('evo_*json'):
    try:
        with zipfile.ZipFile(p, 'r') as z:
            info = None
            stats = None
            if 'info.json' in z.namelist():
                info = json.loads(z.read('info.json'))
            if 'stats.json' in z.namelist():
                stats = json.loads(z.read('stats.json'))
            file_info[p] = {'info': info, 'stats': stats}
    except Exception:
        # not a zip archive or unreadable; skip
        continue

# keywords to match archives to methods
keywords = {
    'SPF LiDAR': ['spf', 'spf_lidar'],
    'SPF++': ['spfpp', 'spf_lidar++', 'spf++'],
    'Noisy GPS': ['ngps', 'noisy', 'noisy_gnss', 'noisy-gps'],
    'AMCL': ['amcl'],
    'RTABMap RGBD': ['rtabmap_rgbd', 'rgbd'],
    'RTABMap RGB': ['rtabmap_rgb', 'rgb']
}

method_list = ['SPF LiDAR', 'SPF++', 'Noisy GPS', 'AMCL', 'RTABMap RGBD', 'RTABMap RGB']

for method in method_list:
    ate_rmse = None
    # search file_info for matching keywords in info.ref_name or info.est_name or filename
    for p, vals in file_info.items():
        if '_ape_' not in p.name:
            continue
        info = vals.get('info') or {}
        stats = vals.get('stats') or {}
        hay = ' '.join([p.name, str(info.get('ref_name','')), str(info.get('est_name','')), info.get('title','')]).lower()
        for kw in keywords.get(method, []):
            if kw in hay:
                ate_rmse = stats.get('rmse') if stats else None
                break
        if ate_rmse is not None:
            break

    rte_vals = rte_map.get(method, {})
    out_rows.append({
        'method': method,
        'ate_evo_rmse': ate_rmse,
        'rte2_evo_rmse': rte_vals.get('rte2'),
        'rte5_evo_rmse': rte_vals.get('rte5'),
        'rte10_evo_rmse': rte_vals.get('rte10'),
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
