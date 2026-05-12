# Modular Data Reconstruction — Stage 00

This repository contains the data construction implementation for the US equity fundamental factor-investing pipeline. It reconstructs the CPZ data foundation from
raw CRSP and Compustat inputs, validates the panel with internal audits and
external benchmarks, and writes the final Stage 00 artifact for downstream stages.

## What is included

- `src/data_reconstruction/` — modular Stage 00 package with raw download,
  CRSP cleaning, accounting merge, monthly characteristic computation, risk
  construction, panel assembly, and validation.
- `notebooks/00_data_foundation.ipynb` — walkthrough notebook for Stage 00.
- `configs/stage00_data_reconstruction.yaml` — Stage 00 configuration, paths,
  universe settings, and accepted caveats.
- `data/foundation/` — generated Stage 00 artifacts (`factor_panel_v2.parquet`,
  validation summaries, acceptance gate records, and diagnostics).


## Stage 00 summary

- Stage 00 is complete and accepted under the caveat-aware gate.
- The final Stage 00 artifact is `data/foundation/factor_panel_v2.parquet`.
- Validation includes:
  - Arm 3 internal audits (PIT bounds, rank-normalization, schema, coverage stability)
  - Arm 1 CPZ distributional comparison diagnostics
  - Arm 2 FF5+UMD replication against Ken French
- Acceptance is recorded in `data/foundation/stage0_acceptance.json`.

## Quick start

```bash
cd equity-fundamental-factor-investing
conda env create -f environment.yml
conda activate factor-pipeline
pytest tests/ -v
```

## Running Stage 00

When raw intermediates already exist:

```bash
python -m data_reconstruction.pipeline --config configs/stage00_data_reconstruction.yaml --skip-build
```

For a one-time WRDS refresh or Ken French pull:

```bash
python -m data_reconstruction.pipeline --config configs/stage00_data_reconstruction.yaml --pull-raw
python -m data_reconstruction.pipeline --config configs/stage00_data_reconstruction.yaml --pull-kf
```

Optional flags:

- `--no-validation` — skip validation artifacts
- `--no-audits` — skip Arm 3 audits
- `--no-arm1` — skip CPZ diagnostics
- `--no-arm2` — skip Ken French replication
- `--no-figures` — skip figure generation
- `--verbose` — verbose logging

## Package structure

- `src/data_reconstruction/`
  - `download.py` — WRDS raw pulls
  - `crsp.py` — CRSP monthly cleaning and Shumway delisting logic
  - `accounting.py` — Compustat PIT merge and book equity / accruals
  - `monthly.py` — momentum, NI, Rel2High, SUV and monthly characteristics
  - `risk.py` — 252-day CAPM regression and Roll spread
  - `assemble.py` — final panel assembly, completeness filter, rank-normalization
  - `pipeline.py` — Stage 00 CLI and orchestrator
  - `validation/` — Arm 1/2/3 validation and diagnostics


## Development commands

```bash
pytest tests/
ruff check src/ tests/
black src/ tests/
```

## Environment requirements

Python 3.11+, Anaconda recommended.

## Contact

Project owner: Amos Anderson
University: Stony Brook University

## Related documents

- ROADMAP.txt — full 11-stage pipeline plan
- STATE.md — current project state snapshot
- DECISION_LOG.md — append-only decision history
- COLLABORATOR_PROMPT.txt — instructions for continuation across platforms


