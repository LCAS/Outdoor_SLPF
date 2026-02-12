# Outdoor_SLPF — Semantic Landmark Particle Filter (SPF)

Bringing robust, semantics-aware localisation to outdoor natural environments.

Submitted to ICRA 2026 — preprint: https://arxiv.org/pdf/2509.18342

Why this repo matters
--------------------
Outdoor environments (orchards, vineyards, parks) challenge conventional localisation due to appearance variability and ambiguous geometry. This project demonstrates a particle-filter-based approach that uses semantic landmarks (poles, trunks) to maintain accurate localisation where odometry or visual methods drift.

What you'll find here
---------------------
- Cleaned trajectory traces and semantic maps (GeoJSON) used in the experiments.
- Analysis and plotting scripts to reproduce figures and compute metrics (ATE, RPE, cross-track errors).
- Diagnostic tools to compare SPF against baselines (Noisy GPS, AMCL, RTAB-Map).
- Example outputs and CSV summaries in `results/`.

Quick start
-----------
1. Create a Python virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate the main trajectory comparison figure:

```bash
python scripts/plot_trajectories.py
```

3. Compute metrics and diagnostics:

```bash
python scripts/compute_metrics.py
python scripts/diagnose_spf_vs_gps.py
```

4. Run the full evaluation pipeline (metrics + evo + aggregation):

```bash
source .venv/bin/activate
python3 scripts/compute_metrics.py
bash scripts/run_evo_all.sh
python3 scripts/aggregate_evo_results.py
```

Repository layout (short)
------------------------
- `data/` — processed trajectory TUM files, GeoJSON maps, and small logs used for evaluation.
  - We track processed files (TUM traces, filtered RTAB-Map outputs and GeoJSON). Large raw recordings (e.g., raw bag files) are not included here.
- `scripts/` — analysis, plotting and conversion tools
  - `plot_trajectories.py` — 2x3 comparison grid, semantic overlay, per-row shared colormaps
  - `compute_metrics.py` — ATE/RPE, Umeyama alignment, row/cross-track metrics, CSV/PDF summaries
  - `diagnose_spf_vs_gps.py` — focused diagnostics and outlier reporting
- `results/` — figures and CSVs produced by scripts (tracked in repo)

Data and results policy
-----------------------
- `data/` now contains processed and curated traces used to reproduce paper plots; you can commit these files so collaborators can run analyses without downloading large raw logs.
- `results/` contains example outputs (figures, CSV summaries). It is included so project deliverables remain alongside code.
- Large raw sensor recordings are intentionally excluded; keep heavy raw datasets in external storage and point scripts to them if needed.

Evaluation notes
----------------
- Metric alignment: scripts support either simple first-pose yaw+translate alignment (useful for visual overlays) or Umeyama 3D similarity alignment — the latter reproduces evo-style similarity APE.
- For formal APE/RPE evaluation, install `evo` and run its CLI on TUM files to obtain standard metrics.

Documentation
-------------
- See the experiment runner and metric definitions for details: `docs/EXPERIMENT_RUNNER_AND_METRICS.md`.

Citation
--------
If you use this code or data, please cite the paper:

```
@article{de2025semantic,
  title={Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation},
  author={de Silva, Rajitha and Cox, Jonathan and Heselden, James R and Popovic, Marija and Cadena, Cesar and Polvara, Riccardo},
  journal={arXiv preprint arXiv:2509.18342},
  year={2025}
}
```

License
-------
This repository is distributed under the Polyform Noncommercial License 1.0.0. Use for noncommercial research and teaching with attribution.

Contact
-------
Open an issue here or contact the corresponding author listed on the preprint for questions or reproduction help.


