#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path(__file__).parent.parent
results_dir = base / 'results'
csv_path = results_dir / 'trajectory_metrics.csv'

if not csv_path.exists():
    print('Missing', csv_path)
    raise SystemExit(1)

df = pd.read_csv(csv_path)
# Keep only methods with a numeric ate_umeyama_rmse
df = df[df['ate_umeyama_rmse'].notna()]
# Sort by RMSE
df = df.sort_values('ate_umeyama_rmse')

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(df['method'], df['ate_umeyama_rmse'], color='C0')
ax.set_ylabel('ATE RMSE after Umeyama (m)')
ax.set_title('Umeyama-aligned ATE RMSE by Method')
ax.set_xticklabels(df['method'], rotation=30, ha='right')
plt.tight_layout()
out = results_dir / 'ume_ate_umeyama_comparison.png'
plt.savefig(out, dpi=150)
print('Saved', out)
