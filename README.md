# Outdoor_SLPF

Semantic landmark particle filtering for outdoor row-structured environments.

This repository accompanies the paper `Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation` and packages the core SPF/SPF++ implementation, the evaluation pipeline, and the paper-facing documentation needed to understand and reproduce the published workflows.

Preprint: https://arxiv.org/pdf/2509.18342

## Repository scope

This public release is centered on:

- the main localisation pipeline in `scripts/spf_lidar.py`
- the evaluation and aggregation workflow used in the paper
- curated map, trajectory, and model assets required by those workflows

Baseline methods such as AMCL, RTAB-Map, and ORB-SLAM3 are treated here as precomputed trajectory inputs. The recommended public release does not depend on shipping their native runtime stacks.

## Repository layout

- `data/` contains processed traverses, trajectory inputs, and semantic map artifacts
- `models/` contains trained weights used by the SPF pipeline
- `configs/` contains sensor and camera configuration files
- `scripts/` contains localisation, evaluation, aggregation, and plotting code
- `docs/` contains setup notes, methodology, and experiment protocols
- `results/` contains generated outputs and paper figures

## Quick start

For evaluation-only workflows:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/compute_metrics.py
bash scripts/run_evo_all.sh
python3 scripts/aggregate_evo_results.py
```

For the full SPF pipeline, including semantic inference and runtime profiling, see `docs/SETUP.md`.

## Main workflows

Run the semantic particle filter on a processed traverse:

```bash
python3 scripts/spf_lidar.py \
  --data-path data/2025/rh_run1 \
  --output-folder results/example_run \
  --no-visualization
```

Generate paper figures and tables:

```bash
python3 scripts/plot_trajectories_2x4_experiment_comparison.py
python3 scripts/plot_vineyard_structure_with_rtk.py
```

Run release-facing experiment extensions:

- `scripts/run_spfpp_ablation.py` for SPF++ ablations
- `scripts/run_runtime_profile_experiment.py` for throughput and stage timing
- `scripts/run_run1_robustness_experiments.py` for robustness studies

## Representative results

Qualitative comparison across the two main experiments:

![Trajectory comparison across Experiment 1 and Experiment 2](results/plots/trajectory_comparison_2x4_experiment1_vs_experiment2_vertical_labels.png)

Plot provenance and inputs are documented in `results/plots/PLOT_SOURCES.md`.

The tables below reproduce the paper-reported values, shown as mean `±` std over three runs. Lower is better for all metrics except `Row correct`.

Experiment 1:

| Method | APE Raw (m) | APE Aligned (m) | RPE 2m (m) | RPE 5m (m) | Cross-track Mean (m) | Median | Max | Row correct | Row mislocalisation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Noisy GNSS | 3.04 ± 0.13 | 2.98 ± 0.13 | 3.56 ± 0.29 | 3.99 ± 0.43 | 1.77 ± 0.12 | 1.78 | 1.78 | 0.64 ± 0.04 | 64.3 ± 5.4 |
| AMCL | 1.37 ± 0.47 | 1.02 ± 0.30 | 4.69 ± 0.05 | 5.39 ± 0.11 | 1.40 ± 0.10 | 0.84 | 1.54 | 0.67 ± 0.13 | 27.3 ± 3.1 |
| AMCL+NoisyGNSS | 1.33 ± 0.46 | 0.99 ± 0.30 | 4.66 ± 0.04 | 5.35 ± 0.10 | 1.39 ± 0.09 | 0.80 | 1.52 | 0.67 ± 0.13 | 27.3 ± 3.1 |
| RGB RTAB-Map | 59.62 ± 0.48 | 6.68 ± 0.03 | 1.19 ± 0.01 | 2.21 ± 0.01 | 6.21 ± 0.03 | 6.15 | 19.49 | 0.45 ± 0.00 | 14.0 ± 0.0 |
| RGBD RTAB-Map | 61.33 ± 0.28 | 10.02 ± 0.03 | 8.25 ± 0.08 | 10.79 ± 0.11 | 6.78 ± 0.01 | 6.13 | 46.10 | 0.48 ± 0.00 | 13.3 ± 0.5 |
| SLPF (ours) | 1.07 ± 0.09 | 1.04 ± 0.10 | 3.33 ± 0.07 | 6.92 ± 0.09 | 1.26 ± 0.06 | 1.25 | 3.85 | 0.73 ± 0.01 | 34.67 ± 1.70 |

Experiment 2:

| Method | APE Raw (m) | APE Aligned (m) | RPE 2m (m) | RPE 5m (m) | Cross-track Mean (m) | Median | Max | Row correct | Row mislocalisation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Noisy GNSS | 3.16 ± 0.22 | 3.09 ± 0.23 | 3.76 ± 0.32 | 4.22 ± 0.55 | 1.99 ± 0.04 | 1.80 | 36.99 | 0.58 ± 0.02 | 713.3 ± 23.2 |
| AMCL | 3.50 ± 0.94 | 2.04 ± 0.31 | 2.78 ± 0.79 | 3.43 ± 0.92 | 1.55 ± 0.12 | 1.86 | 7.01 | 0.55 ± 0.05 | 26.7 ± 2.1 |
| AMCL+NoisyGNSS | 3.38 ± 0.91 | 1.98 ± 0.31 | 2.80 ± 0.78 | 3.42 ± 0.92 | 1.51 ± 0.11 | 1.77 | 6.38 | 0.55 ± 0.05 | 26.7 ± 1.2 |
| RGB RTAB-Map | 85.95 ± 0.19 | 9.12 ± 0.41 | 1.63 ± 0.01 | 2.48 ± 0.00 | 6.81 ± 0.31 | 6.26 | 20.80 | 0.39 ± 0.02 | 17.0 ± 0.8 |
| RGBD RTAB-Map | 87.17 ± 0.01 | 9.06 ± 0.00 | 1.81 ± 0.00 | 3.43 ± 0.00 | 7.25 ± 0.08 | 6.73 | 38.15 | 0.42 ± 0.00 | 18.7 ± 0.5 |
| SLPF (ours) | 1.24 ± 0.04 | 1.11 ± 0.06 | 3.34 ± 0.02 | 6.82 ± 0.07 | 1.46 ± 0.03 | 1.31 | 4.35 | 0.67 ± 0.02 | 28.0 ± 3.3 |

The ablation study and the full paper artifact bundle are prepared separately from this source-focused release commit.

## Documentation

- `docs/SETUP.md` explains dependencies, expected data layout, and install variants
- `docs/PIPELINE.md` explains the working principle of the semantic particle filter
- `docs/EXPERIMENT_RUNNER_AND_METRICS.md` defines the evaluation pipeline and all reported metrics
- `docs/RUNTIME_EXPERIMENT.md` documents the runtime profiling protocol
- `docs/RUN1_ADDITIONAL_EXPERIMENTS.md` documents the additional robustness study
- `docs/PUBLIC_RELEASE_SCOPE.md` records what should and should not be bundled in the public release commit

## Included assets and exclusions

- Processed experiment assets and semantic maps are tracked for reproducibility
- Raw sensor logs and large intermediate archives are intentionally excluded
- Native AMCL and ORB-SLAM3 runtime helpers are outside the recommended public release scope

## Citation

If you use this repository, please cite the accompanying paper. A machine-readable citation file is provided in `CITATION.cff`.

```bibtex
@article{de2025semantic,
  title={Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation},
  author={de Silva, Rajitha and Cox, Jonathan and Heselden, James R and Popovic, Marija and Cadena, Cesar and Polvara, Riccardo},
  journal={arXiv preprint arXiv:2509.18342},
  year={2025}
}
```

## License

This repository is distributed under the PolyForm Noncommercial License 1.0.0. See `LICENSE.md`.
