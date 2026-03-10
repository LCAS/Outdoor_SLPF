# Public Release Scope

This note records the recommended split between release-worthy repository content and local-only working files in the current working tree.

## Include in the public release commit

- top-level repo docs: `README.md`, `LICENSE.md`, `CITATION.cff`
- setup and method docs under `docs/`
- the core SPF and evaluation scripts under `scripts/`
- publication-relevant plotting and aggregation script updates
- sensor configuration needed by the released pipeline, such as `configs/realsense_d435i.yaml`

## Exclude from the public release commit

- `docker/`
- native AMCL and ORB-SLAM3 helper stacks:
  - `scripts/amcl/`
  - `scripts/orbslam3/`
  - `configs/amcl_jojo/`
  - `configs/orbslam3/`
  - `docs/ORB_SLAM3_ON_ICRA2.md`
- local temporary directories and caches
- reviewer notes and paper-drafting files
- bulk regenerated `results/` folders unless you intentionally want to ship a curated artifact bundle

## Keep only if you want a separate artifact release

- final paper plots under `results/plots/`
- compact aggregate CSV and PDF summaries
- provenance notes such as `results/plots/PLOT_SOURCES.md`

The code release should stay focused on method, evaluation, and reproducibility. If you want to ship large generated results, do that as a separate curated artifact snapshot rather than mixing it into the main source release commit.
