#!/usr/bin/env python3
"""Aggregate evo JSON results (APE + RPE) into CSV and a PDF summary table.

This script expects evo result archives saved via `evo_ape --save_results` and
`evo_rpe --save_results`, e.g. `evo_spf_ape_raw.json`, `evo_spf_ape_umey.json`,
`evo_spf_rpe_2m.json`.
"""
from pathlib import Path
import zipfile
import json
import csv
import os

BASE = Path(__file__).parent.parent
RESULTS_OVERRIDE = os.environ.get('RESULTS_DIR')
if RESULTS_OVERRIDE:
    RESULTS = Path(RESULTS_OVERRIDE)
    if not RESULTS.is_absolute():
        RESULTS = BASE / RESULTS
else:
    RESULTS = BASE / 'results'


def parse_evo_archive(path: Path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            info = json.loads(z.read('info.json').decode()) if 'info.json' in z.namelist() else {}
            stats = json.loads(z.read('stats.json').decode()) if 'stats.json' in z.namelist() else {}
            return {'info': info, 'stats': stats}
    except Exception:
        return None


def main():
    files = list(RESULTS.glob('evo_*.json'))
    rows = {}
    allowed_methods_csv = os.environ.get('EVO_METHODS', '').strip()
    allowed_methods = {m.strip() for m in allowed_methods_csv.split(',') if m.strip()} if allowed_methods_csv else None

    for f in files:
        parsed = parse_evo_archive(f)
        if not parsed:
            continue
        stats = parsed.get('stats') or {}
        rmse = stats.get('rmse')

        # filename patterns: evo_{method}_ape_{variant}.json or evo_{method}_rpe_{delta}.json
        parts = f.stem.split('_')
        if len(parts) < 3:
            continue
        # method key may contain underscores (e.g., rtab_rgbd)
        try:
            idx = parts.index('ape')
            kind = 'ape'
        except ValueError:
            try:
                idx = parts.index('rpe')
                kind = 'rpe'
            except ValueError:
                continue

        method_key = '_'.join(parts[1:idx])
        if allowed_methods is not None and method_key not in allowed_methods:
            continue
        tail = parts[idx+1:]
        if method_key not in rows:
            rows[method_key] = {}

        if kind == 'ape':
            variant = '_'.join(tail) if tail else 'raw'
            if variant in ('raw', ''):
                rows[method_key]['ape_raw_rmse'] = rmse
            elif 'umey' in variant and 'scale' in variant:
                rows[method_key]['ape_umey_scale_rmse'] = rmse
            elif 'umey' in variant:
                rows[method_key]['ape_umey_rmse'] = rmse
            else:
                rows[method_key][f'ape_{variant}_rmse'] = rmse
        elif kind == 'rpe':
            delta = tail[0] if tail else ''
            rows[method_key][f'rpe_{delta}_rmse'] = rmse

    # Merge row-based metrics from compute_metrics output (trajectory_metrics.csv)
    traj_metrics_path = RESULTS / 'trajectory_metrics.csv'
    row_metrics = {}
    if traj_metrics_path.exists():
        import csv as _csv
        with open(traj_metrics_path, 'r') as _f:
            reader = _csv.DictReader(_f)
            # map human label -> row metrics
            for r in reader:
                row_metrics[r['method']] = {
                    'cross_track_mean': r.get('cross_track_mean'),
                    'cross_track_median': r.get('cross_track_median'),
                    'cross_track_max': r.get('cross_track_max'),
                    'row_correct_fraction': r.get('row_correct_fraction'),
                    'row_switch_events': r.get('row_switch_events')
                }

    # Write CSV (include row-based metrics)
    out_path = RESULTS / 'evo_aggregated_metrics.csv'
    fieldnames = [
        'method_key',
        'ape_raw_rmse', 'ape_umey_rmse', 'ape_umey_scale_rmse',
        'rpe_2m_rmse', 'rpe_5m_rmse', 'rpe_10m_rmse',
        'cross_track_mean', 'cross_track_median', 'cross_track_max', 'row_correct_fraction', 'row_switch_events'
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in rows.items():
            row = {
                'method_key': k,
                'ape_raw_rmse': v.get('ape_raw_rmse'),
                'ape_umey_rmse': v.get('ape_umey_rmse'),
                'ape_umey_scale_rmse': v.get('ape_umey_scale_rmse'),
                'rpe_2m_rmse': v.get('rpe_2m_rmse'),
                'rpe_5m_rmse': v.get('rpe_5m_rmse'),
                'rpe_10m_rmse': v.get('rpe_10m_rmse')
            }
            # attach row-based metrics if available
            # map method_key to human label used in trajectory_metrics.csv
            key_map = {
                'spf': 'SPF LiDAR',
                'spfpp': 'SPF LiDAR++',
                'ngps': 'Noisy GPS',
                'amcl': 'AMCL',
                'amcl_ngps': 'AMCL+GPS',
                'rtab_rgbd': 'RTABMap RGBD',
                'rtab_rgb': 'RTABMap RGB',
                'orb_rgbd_s4': 'ORB-SLAM3 RGBD (s4)',
                'orb_rgbd_full': 'ORB-SLAM3 RGBD (full)',
                'orb_mono_s4': 'ORB-SLAM3 Mono (s4)',
                'orb_mono_full': 'ORB-SLAM3 Mono (full)'
            }
            human = key_map.get(k)
            if human and human in row_metrics:
                rm = row_metrics[human]
                row['cross_track_mean'] = rm.get('cross_track_mean')
                row['cross_track_median'] = rm.get('cross_track_median')
                row['cross_track_max'] = rm.get('cross_track_max')
                row['row_correct_fraction'] = rm.get('row_correct_fraction')
                row['row_switch_events'] = rm.get('row_switch_events')
            else:
                row['cross_track_mean'] = None
                row['cross_track_median'] = None
                row['cross_track_max'] = None
                row['row_correct_fraction'] = None
                row['row_switch_events'] = None
            writer.writerow(row)
    print('Wrote', out_path)

    # Create PDF summary
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        table_header = [
            'Method',
            'APE\nraw',
            'APE\nUmeyama',
            'APE\nUmeyama (scale)',
            'RPE@2m',
            'RPE@5m',
            'RPE@10m',
            'Cross-track\nmean (m)',
            'Row correct\nfrac',
            'Row switch\nevents'
        ]
        table_rows = []
        for k, v in rows.items():
            def fmt(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return '-'
            # enrich with row-based metrics if available
            key_map = {
                'spf': 'SPF LiDAR',
                'spfpp': 'SPF LiDAR++',
                'ngps': 'Noisy GPS',
                'amcl': 'AMCL',
                'amcl_ngps': 'AMCL+GPS',
                'rtab_rgbd': 'RTABMap RGBD',
                'rtab_rgb': 'RTABMap RGB',
                'orb_rgbd_s4': 'ORB-SLAM3 RGBD (s4)',
                'orb_rgbd_full': 'ORB-SLAM3 RGBD (full)',
                'orb_mono_s4': 'ORB-SLAM3 Mono (s4)',
                'orb_mono_full': 'ORB-SLAM3 Mono (full)'
            }
            human = key_map.get(k)
            rm = row_metrics.get(human, {}) if human else {}

            table_rows.append([
                k,
                fmt(v.get('ape_raw_rmse')),
                fmt(v.get('ape_umey_rmse')),
                fmt(v.get('ape_umey_scale_rmse')),
                fmt(v.get('rpe_2m_rmse')),
                fmt(v.get('rpe_5m_rmse')),
                fmt(v.get('rpe_10m_rmse')),
                fmt(rm.get('cross_track_mean')),
                fmt(rm.get('row_correct_fraction')),
                fmt(rm.get('row_switch_events'))
            ])

        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        ax.axis('off')
        ax.text(0.5, 0.95, 'EVO Aggregated Metrics (APE + RPE)', ha='center', va='center', fontsize=14, weight='bold')
        table = ax.table(cellText=table_rows, colLabels=table_header, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        pdf_path = RESULTS / 'evo_aggregated_metrics.pdf'
        pp = PdfPages(str(pdf_path))
        pp.savefig(fig, bbox_inches='tight')
        # append trajectory comparison image as an extra page if exists
        comp_img = RESULTS / 'trajectory_comparison.png'
        if comp_img.exists():
            try:
                import matplotlib.image as mpimg
                fig2 = plt.figure(figsize=(11.7, 8.27))
                img = mpimg.imread(str(comp_img))
                plt.imshow(img)
                plt.axis('off')
                pp.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            except Exception:
                pass
        pp.close()
        plt.close(fig)
        print('PDF written to', pdf_path)
    except Exception as e:
        print('Could not create PDF:', e)


if __name__ == '__main__':
    main()
