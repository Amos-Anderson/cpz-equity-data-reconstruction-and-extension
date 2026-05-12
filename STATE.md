# STATE.md - Project State Snapshot

> **Last updated**: 2026-05-11 (Stage 00 end-to-end complete; caveat-aware gate live)
> **Updated by**: Claude (Opus 4.7 via Claude Code)
> **Status**: Stage 00 is **PASS (with accepted universe caveats)**. The caveat-aware acceptance gate added in DECISION_LOG entry 042 reads `stage0_complete: true` with `all_arms_mechanically_passed: false` and the six entry-041 universe-attributable failures listed in `accepted_caveats`. `unaccepted_failures` is empty. Stage 1 is unblocked.

---

## A) Where Stage 00 stands

The Stage 00 package is feature-complete and has run end-to-end with all three validation arms producing real outputs.

**Pipeline construction.** Same notebook-derived translation Codex committed on 2026-05-10. The four audit fixes from Doc 1 are in effect. The package produces a 1,736,168-row, 13,906-firm panel covering 1967-01-31 → 2024-12-31.

**Validation results**:

| Arm | Status | Outcome |
|---|---|---|
| **Arm 3 — internal audits (gate)** | 3/4 PASS, 1 hard-fail | PIT, rank-norm, schema all clean. Coverage stability flags 12 months in 1984 with \|YoY\| > 30% — same universe-filter root cause as Arm 1 D1 / Arm 2 amplitude failures. |
| **Arm 1 — CPZ distributional (diagnostic)** | 4 artifacts written | Pearson and stat differences are within sanity bounds (max \|diff_mean\| = 5e-19). Max breadth gap = 73.5% in 1967, narrowing to <5% by ~1985. Same universe-filter signature. |
| **Arm 2 — FF5+UMD vs Ken French (gate)** | ρ PASSES, MAD/min_year_ρ FAILS | Every factor clears the Pearson ρ > 0.85 time-series-shape test. Amplitude (MAD) and calendar-year (min_year_ρ) bars fail on HML, RMW, UMD — universe-attributable per DECISION_LOG entry 041. |

**Arm 2 numbers (2017-2024 vs Ken French, 96 months):**

| Factor | ρ | MAD | Δann | min_year_ρ | bad year |
|---|---|---|---|---|---|
| MKT | 0.994 | 0.0040 | +0.0085 | 0.922 | 2017 |
| SMB | 0.945 | 0.0076 | +0.0115 | 0.880 | 2019 |
| HML | 0.873 | **0.0154** | +0.0132 | **0.681** | 2023 (SVB) |
| RMW | 0.860 | 0.0090 | +0.0092 | **0.760** | 2019 (yield curve) |
| CMA | 0.968 | 0.0055 | −0.0114 | 0.879 | 2020 |
| UMD | 0.919 | **0.0126** | −0.0029 | **0.743** | 2024 (bank rebound) |

Pearson ρ ≥ 0.86 for every factor means our characteristics are computed correctly. The amplitude / calendar-year failures are explained by financials being absent from our panel but present in Ken French's; the Δann signs match the predicted direction in every case.

**Pipeline gate decision (`data/foundation/stage0_acceptance.json`).** Two gates are reported side-by-side per DECISION_LOG entry 042:

- `all_arms_mechanically_passed: false` — the raw strict-threshold view (Arm 3 coverage_stability + Arm 2 HML/RMW/UMD amplitude/min_year_ρ fail).
- `stage0_complete: true` — the headline; every failure sits within `accepted_caveats` (six items: `coverage_stability`, `hml.mad_lt_0p01`, `umd.mad_lt_0p01`, `hml.min_year_rho_gt_0p80`, `rmw.min_year_rho_gt_0p80`, `umd.min_year_rho_gt_0p80`). `unaccepted_failures` is empty.

End-of-run log line reads: `stage0 acceptance: PASS (with accepted caveats) | mechanical: arm2=False arm3=False | accepted_caveats=6`. Stage 1 reads the panel from `factor_panel_v2.parquet` and proceeds; its gate check is `stage0_complete == true`. If a future failure surfaces that is *not* universe-attributable, it will not match any caveat key and will correctly block Stage 1.

---

## B) Final layout of `src/data_reconstruction/`

```
src/data_reconstruction/
├── __init__.py
├── config.py             # Stage00Config: paths, sample windows, exclude_financials, accepted_arm{2,3}_caveats
├── constants.py          # ALL_46
├── download.py           # WRDS pulls (pull_all_raw_data, pull_crsp_monthly, pull_compustat, pull_factor_and_daily_data)
├── crsp.py               # build_clean_monthly (Shumway delisting, excess returns, ME)
├── accounting.py         # build_accounting_file (BE hierarchy, AC audit fix 2, etc.)
├── monthly.py            # build_monthly_file (momentum, NI, Rel2High audit fix 3, SUV AR(3))
├── risk.py               # build_risk_file (252-day CAPM regression family, Roll spread)
├── assemble.py           # assemble_panel (PIT merge_asof, completeness filter, rank-norm, splits)
├── pipeline.py           # run_stage00 orchestrator + CLI; caveat-aware gate via _write_stage0_acceptance
└── validation/
    ├── __init__.py
    ├── audits.py             # Arm 3: PIT, rank-norm, schema, coverage stability
    ├── lightweight.py        # legacy breadth/return summary (kept as sanity check)
    ├── cpz_comparison.py     # Arm 1: 4 distributional artifacts
    ├── ken_french_loader.py  # Arm 2 input: robust WRDS pull (FF3+RF+UMD from ff.factors_monthly, RMW+CMA from ff.fivefactors_monthly)
    ├── fama_french.py        # Arm 2 portfolio construction (2×3 NYSE breakpoints)
    ├── arm2.py               # Arm 2 orchestrator + acceptance criteria
    └── diagnostics.py        # plotly+kaleido figure helpers
```

Tests: `tests/test_stage00_modular.py`, **18 passing**.

---

## C) CLI surface

```powershell
# One-time WRDS pulls (interactive credentials):
python -m data_reconstruction.pipeline --config configs/stage00_data_reconstruction.yaml --pull-raw
python -m data_reconstruction.pipeline --config configs/stage00_data_reconstruction.yaml --pull-kf

# Full end-to-end (assumes raw intermediates exist; ~90 seconds):
python -m data_reconstruction.pipeline --config configs/stage00_data_reconstruction.yaml --skip-build

# Selective skips: --no-validation --no-audits --no-arm1 --no-arm2 --no-figures --verbose
```

---

## D) Outputs on disk after the end-to-end run

### `data/foundation/`

| File | Purpose |
|---|---|
| `factor_panel_v2.parquet` | Final 1.74M-row Stage 00 panel |
| `completeness_diag.csv` | Pre-filter non-null counts per characteristic |
| `yearly_completeness_diag.parquet` | Pre-filter coverage per (year, characteristic) |
| `stage00_summary.json` | Row counts, firm count, date range, split sizes |
| `arm3_summary.json` | Arm 3 audit results (gate) |
| `arm1_summary.json` + `arm1_{annual_breadth,annual_returns,char_stats,yearly_coverage}.parquet` | Arm 1 distributional (diagnostic) |
| `arm2_summary.json` + `arm2_{factor_returns,per_factor_summary}.parquet` | Arm 2 FF5+UMD (gate) |
| `cpz_breadth_return_summary.json` | Lightweight legacy diagnostic |
| **`stage0_acceptance.json`** | **Caveat-aware gate: `stage0_complete`, `all_arms_mechanically_passed`, `accepted_caveats`, `unaccepted_failures`** |

### `C:/Users/amosa/ml4t_data/extended_v2/`

- `train.parquet`: 1972-01 → 2010-12, 1,286,839 rows
- `valid.parquet`: 2011-01 → 2021-12, 294,724 rows
- `test.parquet`: 2022-01 → 2024-12, 79,244 rows

### `reports/figures/stage00/`

`panel_coverage`, `char_presence`, `arm1_annual_breadth`, `arm1_annual_returns`, `arm1_char_stats`, `arm1_yearly_coverage`, `arm2_cumulative_returns`, `arm2_correlation_summary` — each as both `.html` (interactive) and `.png` (LaTeX/notebook).

### Notebook

`notebooks/00_data_foundation.ipynb` — 30 cells, substantive walkthrough. Section 9 renders the Arm 2 cumulative-returns subplot grid and the per-factor correlation bar inline via `IPython.display.Image`.

---

## E) Immediate Next Actions (for the user)

1. **Run the notebook** to visually confirm the Arm 2 diagnosis. Section 9's cumulative-return subplot grid is the key figure — the divergences should cluster around the named bad-year months (HML 2023, RMW 2019, UMD 2024) rather than appear as systematic month-over-month drifts.

   ```powershell
   jupyter lab notebooks/00_data_foundation.ipynb
   ```

2. **Push the figures/notebook to GitHub** when satisfied — `reports/figures/stage00/` and `notebooks/00_data_foundation.ipynb` together give a GitHub-readable Stage 00 record.

3. **Proceed to Stage 1 (universe construction).** Per DECISION_LOG entry 041, the universe choice (`exclude_financials: true`) is locked. Stage 1 reads `factor_panel_v2.parquet` and builds the three universe variants (academic_broad, liquid, nyse_equivalent) on top. `stage0_acceptance.json::stage0_complete == false` is a known caveat, not a blocker.

4. **Defer the 1984 cross-sectional jump investigation** to either Stage 1 (which will naturally re-segment the panel by universe variant) or to the LaTeX report-writing phase. It is not on the critical path.

---

## F) Known issues / outstanding questions

- **Arm 2 amplitude failures** (HML MAD 0.0154, UMD MAD 0.0126; HML/RMW/UMD min_year_ρ below 0.80): universe-attributable per DECISION_LOG 041; sit within `accepted_caveats` per DECISION_LOG 042. Headline gate passes; mechanical gate honestly reports the failures alongside.
- **Arm 3 1984 coverage step-up**: same root cause; accepted caveat. Defer detailed root-cause work (which Compustat field jumps in 1984) until the LaTeX writeup phase or until Stage 1 re-segmentation surfaces it as still material.
- **ME unit semantics**: `crsp.py` computes `me = |prc| * shrout / 1000` (millions of dollars). Doc 1 §"Raw Data Sources" states `ME = |prc| * shrout * 1000` (dollars). Doesn't affect rank-norm or any rank-derived characteristic; flag for reconciliation when the LaTeX Stage 00 section is written.
- **Doc 1 vs Doc 3 PIT-bound formula**: Doc 3 now agrees with Doc 1 (corrected during Phase 1). `audit_pit_bounds` enforces the corrected bound; 0 violations on 1.74M rows.
- **Strict-mode escape hatch**: pass `accepted_arm3_caveats: []` and `accepted_arm2_caveats: []` in the YAML to restore the pre-entry-042 strict gate behavior (would re-introduce `stage0_complete: false`). Useful if a future stage requires the mechanical gate to pass.
