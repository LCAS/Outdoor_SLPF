# Run1 Robustness Add-on (for ablation table)

- Source folder: `results/iros_rh1_robustness/20260225_151229_run1_robustness_gpu`

## Primary metrics

| Variant | APE_align (m) | RowCorrect | Cross-track (m) |
|---|---:|---:|---:|
| Baseline (full map) | 0.996 ± 0.028 | 0.724 ± 0.018 | 1.306 ± 0.047 |
| DetDrop 20% | 1.021 ± 0.082 | 0.733 ± 0.008 | 1.306 ± 0.052 |
| DetDrop 40% | 1.049 ± 0.090 | 0.735 ± 0.027 | 1.263 ± 0.049 |
| MapRemove 30% | 1.045 ± 0.036 | 0.721 ± 0.011 | 1.265 ± 0.016 |
| MapRemove 50% | 1.127 ± 0.129 | 0.716 ± 0.005 | 1.323 ± 0.024 |

## Recovery metrics (Option B)

| Variant | Section APE (m) | Post APE (m) | Recovery frames | Recovery distance (m) | Recovery rate |
|---|---:|---:|---:|---:|---:|
| MapRemove 30% | 0.744 ± 0.057 | 0.824 ± 0.043 | 18.67 ± 6.85 | 11.92 ± 4.81 | 100% |
| MapRemove 50% | 1.067 ± 0.165 | 0.869 ± 0.062 | 18.67 ± 6.13 | 11.79 ± 4.46 | 100% |
