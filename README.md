# Outdoor_SLPF — Semantic Landmark Particle Filter (SPF)

Bringing robust, semantics-aware localisation to outdoor natural environments.

Submitted to ICRA 2026 — preprint: https://arxiv.org/pdf/2509.18342

## Why this repo matters

Outdoor environments (orchards, vineyards, parks) challenge conventional localisation due to appearance variability and ambiguous geometry. This project demonstrates a particle-filter-based approach that uses semantic landmarks (poles, trunks) to maintain accurate localisation where odometry or visual methods drift.

## Latest state (February 25, 2026)

This repository now includes the paper-facing bundles for:

- Experiment 1 (run1): multiseed benchmark and recomputed SLPF/Noisy-GPS baseline.
- Experiment 2 (run2): multiseed benchmark across methods.
- Main SPF++ ablation study.
- Runtime profiling experiment (pipeline Hz + component timings).
- Run1 robustness add-on experiments (detection drop and landmark removal).

Canonical result folders:

- `results/iros_rh1/20260217_172656_multiseed_main`
- `results/iros_rh1/20260225_130359_multiseed_slpf_ngps_run1_recomputed`
- `results/iros_rh2/20260225_105822_multiseed_all_methods`
- `results/iros_ablation/20260217_085404_ablation`
- `results/runtime_profile/20260225_142949_runtime_gpu`
- `results/iros_rh1_robustness/20260225_151229_run1_robustness_gpu`
- `results/plots`

## Quick start

1. Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Compute baseline metrics + evo aggregation:

```bash
python3 scripts/compute_metrics.py
bash scripts/run_evo_all.sh
python3 scripts/aggregate_evo_results.py
```

3. Generate paper plotting artifacts:

```bash
python3 scripts/plot_trajectories_2x4_experiment_comparison.py
python3 scripts/plot_vineyard_structure_with_rtk.py
```

## Core scripts (paper workflow)

- `scripts/spf_lidar.py` — main SPF/SLPF implementation and runtime profiling hooks.
- `scripts/run_ab_validation.py` — shared evaluation utilities used by experiment runners.
- `scripts/run_spfpp_ablation.py` — SPF++ ablation pipeline and table outputs.
- `scripts/run_run1_robustness_experiments.py` — run1 robustness experiments (Option A/B).
- `scripts/run_runtime_profile_experiment.py` — repeated runtime trial runner + aggregation.
- `scripts/aggregate_evo_results.py` — aggregates evo archives into comparison tables.
- `scripts/merge_evo_and_rte.py` — merges evo trajectory metrics with row/cross-track metrics.
- `scripts/plot_trajectories.py` — trajectory overlays and summary plots.
- `scripts/plot_trajectories_2x4_experiment_comparison.py` — main 2x4 experiment comparison figure.
- `scripts/plot_vineyard_structure_with_rtk.py` — vineyard structure + RTK reference figure.
- `scripts/geojson_rows.py` — row-id extraction utilities for GeoJSON landmarks.

## Repository layout (short)

- `data/` — curated processed trajectories and map artifacts used by experiments.
- `scripts/` — experiment runners, evaluation, aggregation, and plotting utilities.
- `results/` — generated figures, tables, per-seed outputs, and reproducibility bundles.
- `docs/` — protocols and metric definitions.

## Data/results policy

- Processed experiment data is tracked for reproducibility.
- Large raw sensor recordings are intentionally excluded.
- Evo zip archives under `data/` are excluded via `.gitignore` (`data/**/*.zip`).

## Documentation

- Metrics and evaluation definitions: `docs/EXPERIMENT_RUNNER_AND_METRICS.md`
- Runtime protocol: `docs/RUNTIME_EXPERIMENT.md`
- Run1 robustness add-on protocol: `docs/RUN1_ADDITIONAL_EXPERIMENTS.md`
- ORB-SLAM3 note: `docs/ORB_SLAM3_ON_ICRA2.md`

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{de2025semantic,
  title={Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation},
  author={de Silva, Rajitha and Cox, Jonathan and Heselden, James R and Popovic, Marija and Cadena, Cesar and Polvara, Riccardo},
  journal={arXiv preprint arXiv:2509.18342},
  year={2025}
}
```

## License

This repository is distributed under the Polyform Noncommercial License 1.0.0. Use for noncommercial research and teaching with attribution.

## Contact

Open an issue here or contact the corresponding author listed on the preprint for questions or reproduction help.
