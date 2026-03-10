#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

base = Path(__file__).parent.parent
results = base / 'results'

files = {
    'SPF': results / 'evo_spf_umey.json',
    'NoisyGPS': results / 'evo_ngps_umey.json',
    'AMCL': results / 'evo_amcl_umey.json',
    'RTAB_RGBD': results / 'evo_rtab_rgbd_umey.json',
    'RTAB_RGB': results / 'evo_rtab_rgb_umey.json'
}

rows = []
for name, p in files.items():
    if not p.exists():
        continue
    # evo saved a zip archive; open stats.json inside
    import zipfile
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
    print('No evo umey results found in results/'); raise SystemExit(1)

df = pd.DataFrame(rows)
df.to_csv(results / 'evo_umey_summary.csv', index=False)
print('Wrote', results / 'evo_umey_summary.csv')

# simple plot
fig, ax = plt.subplots(figsize=(8,4))
df_sorted = df.dropna(subset=['rmse']).sort_values('rmse')
ax.bar(df_sorted['method'], df_sorted['rmse'], color='C0')
ax.set_ylabel('APE RMSE (m) - evo Umeyama')
ax.set_title('evo APE RMSE after Umeyama alignment')
plt.tight_layout()
plt.savefig(results / 'evo_umey_summary.png', dpi=150)
print('Wrote', results / 'evo_umey_summary.png')
