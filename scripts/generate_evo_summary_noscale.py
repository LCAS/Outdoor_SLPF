#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

base = Path(__file__).parent.parent
results = base / 'results'

files = {
    'SPF': results / 'evo_spf_umey_noscale.json',
    'NoisyGPS': results / 'evo_ngps_umey_noscale.json',
    'AMCL': results / 'evo_amcl_umey_noscale.json',
    'RTAB_RGBD': results / 'evo_rtab_rgbd_umey_noscale.json',
    'RTAB_RGB': results / 'evo_rtab_rgb_umey_noscale.json'
}

rows = []
for name, p in files.items():
    if not p.exists():
        print('Missing', p)
        continue
    with zipfile.ZipFile(p, 'r') as z:
        if 'stats.json' in z.namelist():
            with z.open('stats.json') as fh:
                data = json.load(fh)
            stats = data.get('summary', {}) if isinstance(data, dict) else data
        elif 'info.json' in z.namelist():
            with z.open('info.json') as fh:
                data = json.load(fh)
            stats = data.get('summary', {})
        else:
            stats = {}
    rows.append({
        'method': name,
        'rmse': stats.get('rmse'),
        'mean': stats.get('mean'),
        'median': stats.get('median'),
        'max': stats.get('max')
    })

if not rows:
    print('No data'); raise SystemExit(1)

df = pd.DataFrame(rows)
df.to_csv(results / 'evo_umey_noscale_summary.csv', index=False)
print('Wrote', results / 'evo_umey_noscale_summary.csv')

# plot
fig, ax = plt.subplots(figsize=(8,4))
df_sorted = df.dropna(subset=['rmse']).sort_values('rmse')
ax.bar(df_sorted['method'], df_sorted['rmse'], color='C1')
ax.set_ylabel('APE RMSE (m) - evo Umeyama (no scale)')
ax.set_title('evo APE RMSE after Umeyama alignment (no scale)')
plt.tight_layout()
plt.savefig(results / 'evo_umey_noscale_summary.png', dpi=150)
print('Wrote', results / 'evo_umey_noscale_summary.png')
