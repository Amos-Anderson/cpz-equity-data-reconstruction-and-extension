# 00 DATA FOUNDATION DOC3 VALIDATION

---



# Stage 0 (cont.): Validation Methodology

## Overview and Acceptance Bar

> **Note on the acceptance numbers below.** Where numerical thresholds
> appear (e.g., ``$\rho > 0.85$'' for Arm 2), they are *pre-registered
> targets* stated before any code is run, as the bar we intend to meet.
> 
> After implementation, the actual measured results will be substituted
> into the LaTeX report. If the actuals match or exceed the targets, the
> panel passes. If they fall short, the failure protocol
> (§(subsec:stage0-failure-protocol)) governs whether we proceed
> with documented caveats or block on the discrepancy. We do
> *not* adjust the thresholds to match the results --- this would
> violate v2 Principle 4 (no post-hoc filtering, DECISION_LOG entry 011).
> 
> Arm 1 below is a **diagnostic layer** with no pre-registered
> thresholds. See the Arm 1 subsection for the rationale: the
> CPZ reference panel is permno-anonymized, so the per-(permno, date)
> comparison originally specified in this document cannot be
> implemented. We compute four distributional comparisons instead, and
> report them as observed without pass/fail gating. The acceptance bar
> shifts to Arms 2 and 3.

A reconstructed dataset is only as useful as our confidence in its
correctness. Document 3 specifies three validation arms that
collectively establish that confidence:

- **Arm 1 — CPZ distributional comparison (1967--2016).** For
      the period where [ChenPelgerZhu2024] provide a published
      characteristic panel, we compare our reconstruction against
      theirs *distributionally*: annual breadth, annual equal-weighted
      excess return, per-characteristic full-period stats, and a
      yearly coverage diagnostic. Arm 1 is a diagnostic layer (no
      formal pass/fail), because the published CPZ panel is
      permno-anonymized and per-firm matching is impossible.

- **Arm 2 — Fama-French replication (2017--2024).** For the
      8-year extension where no published characteristic panel exists,
      we instead construct the FF5+UMD long-short factor portfolios
      from our data and compare their returns to Kenneth French's
      published series. **This is the load-bearing per-firm gate** for
      Stage 0.

- **Arm 3 — Internal consistency audits.** Independent of any
      external reference: PIT correctness audit (does any merged
      `datadate` exceed the panel date?), look-ahead audit (do
      monthly characteristics use only past information?), and
      coverage stability audit (does the firm count change smoothly
      over time, or are there structural breaks?).

The acceptance bar is intentionally strict on Arms 2 and 3. A hard
failure on either blocks Stage 0 from proceeding to Stage 1; we do not
accept "close enough" on the data foundation, because errors compound
through the pipeline. The specific numerical thresholds are stated in
each arm's subsection.

## Arm 1 --- CPZ Distributional Comparison (1967-2016)

### Reference dataset and the permno-anonymization constraint

We use the published CPZ characteristic panel from
[ChenPelgerZhu2024], distributed via Stefan Jansen's replication
archive as `firm_characteristics_all.parquet`. The panel covers
1967-01 through 2016-12, with one row per firm-month and 48 columns:
`date`, `ret` (excess return), and the 46 rank-normalized
characteristics from FNW.

**Critical constraint.** The published panel is permno-anonymized at
source. The row order does not encode firm identity. There is no
column or index that maps a row to a specific CRSP permno. This is
consistent with the CRSP / Compustat licensing constraint preventing
redistribution of identifiable joins.

Consequence: the per-(permno, date) intersection comparison originally
contemplated in this document --- with $\mathcal{I}_t = \mathrm{permno}(\mathbf{P}_{\mathrm{CPZ},t}) \cap \mathrm{permno}(\mathbf{P}_{\mathrm{ours},t})$
and $d_{i,t,f} = |z^{\mathrm{ours}}_{i,t,f} - z^{\mathrm{CPZ}}_{i,t,f}|$
--- cannot be implemented. The CPZ panel does not carry the firm
identifier required to compute either of those quantities.

We therefore reformulate Arm 1 as a **distributional comparison**.

### Why distributional and why diagnostic-only

Two observations bound what a distributional comparison can claim:

**(1) Rank-normalization enforces uniform marginals.** Both panels
are rank-normalized cross-sectionally per month to the open interval
$(-1/2, +1/2)$. By construction, each characteristic's monthly
marginal distribution is approximately discrete-uniform on that
interval in *both* panels, regardless of how the underlying
characteristics were computed. Per-marginal moment comparisons
therefore detect only differences in *which firms are included*
(universe / missing-value handling), not differences in *how the
characteristic was computed*.

**(2) The genuinely informative tests are joint or return-predictive.**
Joint distributional structure (cross-characteristic correlations) and
return-predictive structure (decile sorts on a characteristic vs forward
returns) are not collapsed by rank-normalization. They are stronger
tests, but they overlap with Arm 2's portfolio replication, which is
itself per-firm in construction (uses our permnos to build value-weighted
portfolios) and compared against an external published benchmark
(Kenneth French).

For these two reasons, Arm 1 reports as a *diagnostic layer*:
the four metrics below are computed and reported as observed, without
pass/fail gating. The acceptance gate for Stage 0 sits at Arms 2 (FF5
+ UMD replication) and 3 (internal audits). This deviation from the
original "all three arms must pass" framing is recorded in
`DECISION_LOG.md` and is forced by the data, not chosen freely.

### Four distributional diagnostics

The implementation follows
`Previous Data Pipeline Versions/Data_Pipeline_v2.ipynb`, which
established the same four-artifact pattern.

**(D1) Annual breadth.** Per calendar year $y$:

$$
\mathrm{breadth}_y = \frac{1}{|\mathcal{T}_y|} \sum_{t \in \mathcal{T}_y} N_t,
\qquad
\mathcal{T}_y = \{\text{month-ends in year } y\},
$$

where $N_t$ is the firm count in the panel at month $t$. We compute
this for ours and for CPZ separately, then report the difference and
percent difference per year. A persistent breadth gap on one side
indicates a universe-filter or missing-value-handling difference.

**(D2) Annual equal-weighted excess return.** Per calendar year $y$:

$$
\overline{r}_y = \frac{1}{|\mathcal{T}_y|} \sum_{t \in \mathcal{T}_y}
   \frac{1}{N_t} \sum_{i} r^{\mathrm{ex}}_{i,t}.
$$

Each panel produces its own $\overline{r}_y$. The CPZ panel stores
excess returns in its `ret` column; ours uses `ret_excess`.
Compared as a time series, with per-year `diff` reported.

**(D3) Per-characteristic full-period stats.** For each characteristic
$f$ on each panel separately, compute the full-period mean, median, and
std of the rank-normalized values across all firm-months. Report
$\Delta_{f}^{\mathrm{mean}}, \Delta_{f}^{\mathrm{median}},
\Delta_{f}^{\mathrm{std}}$ = ours minus CPZ. By the uniform-marginal
property in §"Why distributional and why diagnostic-only" above, all
six raw stats sit at theoretical values near $\overline{x} \approx 0$
and $\sigma \approx 1/\sqrt{12} \approx 0.289$. Material deviations
in either panel indicate that some firms have been dropped
asymmetrically (e.g., one panel drops more low-rank firms than the
other due to missing-value handling).

**(D4) Per-(year, characteristic) pre-filter coverage of our panel.**
Before the completeness filter is applied, what fraction of firm-months
have a non-null value for each characteristic, broken down by year? A
heatmap of $(46 \text{ chars}) \times (\text{years})$. There is no
direct CPZ analog (the published CPZ panel is post-filter, so its
coverage is trivially 1.0 everywhere); this diagnostic instead
explains *our* universe size at each point in time and shows when
specific characteristics become available.

### Output artifacts

Four parquet files written to `data/foundation/`:

| **File** | **Index** | **Columns** |
|---|---|---|
| `arm1_annual_breadth.parquet` | `year` | `ours_breadth`, `cpz_breadth`, `diff`, `pct_diff` |
| `arm1_annual_returns.parquet` | `year` | `ours_ann_ret`, `cpz_ann_ret`, `diff` |
| `arm1_char_stats.parquet`     | `characteristic` | `ours_{mean,median,std}`, `cpz_{mean,median,std}`, `diff_{mean,median,std}` |
| `arm1_yearly_coverage.parquet` | `(year, characteristic)` | `coverage` (pre-filter fraction non-null) |

A summary JSON `arm1_summary.json` records aggregate findings: max
absolute breadth pct-diff and the year it occurs in; max absolute
return diff; characteristics with the largest stat differences; etc.
The `arm1_severity` field is always `"diagnostic"` --- this arm does
not gate the pipeline.

### Plotly figures (Phase 4 inputs)

Four figures rendered as both HTML (interactive) and PNG (LaTeX),
written to `reports/figures/stage00/`:

- `arm1_annual_breadth.{html,png}` --- two-line time series.
- `arm1_annual_returns.{html,png}` --- two-line time series.
- `arm1_char_stats.{html,png}` --- side-by-side horizontal bars of
      $|\Delta^{\mathrm{mean}}|$ and $|\Delta^{\mathrm{std}}|$.
- `arm1_yearly_coverage.{html,png}` --- characteristic-by-year
      heatmap.

### Look-ahead bias

Arm 1's diagnostics are all cross-sectional or within-year aggregates.
No metric references information that postdates the date being
characterized. The rank-normalization that produces both panels is
itself within-month, so no future information feeks back into the
inputs. The comparison window (1967-01 through 2016-12) is the CPZ
panel's full coverage; we do not extrapolate Arm 1 into 2017-2024.

## Arm 2 --- Fama-French Replication (2017-2024)

### Why Fama-French replication

The CPZ panel ends at 2016. To validate our 2017--2024 extension, we
need an external reference, but no published characteristic panel
exists for this period. Instead we use Kenneth French's published
factor return series, which are constructed from a known protocol and
updated continuously through the present.

The strategy: construct the FF5 factors (HML, RMW, CMA, plus SMB, plus
the market) and the UMD momentum factor from our reconstructed data,
following Fama-French's exact protocol. If our data are correct, our
factor returns should closely match French's published returns on the
overlap.

This is an indirect test --- we are validating the data via its
ability to reproduce known factor returns rather than via direct
comparison of characteristics. But it is rigorous: a 0.85+ correlation
between our factor returns and French's, sustained over 8 years, is
strong evidence that our underlying characteristics are correctly
constructed.

### Factor construction protocol

The FF5+UMD factors are constructed from monthly portfolios formed via
the standard 2$\times$3 size-and-characteristic sort with NYSE
breakpoints (FamaFrench1993,FamaFrench2015). The protocol
for each factor is summarized below; the full mathematical specification
follows.

**Universe rule.** At each month-end $t$, restrict to common
stocks (`shrcd` $\in {10, 11}$) on NYSE/AMEX/NASDAQ
(`exchcd` $\in {1, 2, 3}$) with non-missing
$\mathrm{ME}_{i,t}$, $\mathrm{BE}_{i,\tau(t)}$, and the relevant
characteristic. We use the same universe rule as Stage 0's general
panel (Document 1, §(subsec:stage0-completeness)) to ensure
consistency.

**Size sort.** At June-end of each year, sort stocks by their
market equity at end of June. Use the median of NYSE-only
`exchcd`=1 stocks as the breakpoint:

$$
b^{\mathrm{NYSE}}_{\mathrm{ME}, t} = \mathrm{median}\bigl(
  {\mathrm{ME}_{i, t} : i \in \mathcal{U}_{\mathrm{NYSE}, t}}
\bigr),
$$

where $\mathcal{U}_{\mathrm{NYSE}, t}$ is the set of NYSE common stocks
at $t$. Stocks in our panel with $\mathrm{ME}_{i,t} \le b^{\mathrm{NYSE}}_{\mathrm{ME}, t}$
are classified as Small (S); the rest as Big (B).

**Characteristic sort (HML, RMW, CMA, UMD).** The relevant
characteristic uses NYSE-only 30th and 70th percentile breakpoints:

$$
b^{\mathrm{NYSE}, 30}_{f, t} &= P_{30}\bigl({x_{i,t,f} : i \in \mathcal{U}_{\mathrm{NYSE}, t}}\bigr), \notag \\
b^{\mathrm{NYSE}, 70}_{f, t} &= P_{70}\bigl({x_{i,t,f} : i \in \mathcal{U}_{\mathrm{NYSE}, t}}\bigr).
$$

Stocks below the 30th percentile go in the Low (L) bucket; above the
70th, in the High (H) bucket; in between, the Medium (M) bucket. This
yields a 2$\times$3 partition with 6 portfolios: SL, SM, SH, BL, BM, BH.

**Portfolio returns.** Each of the 6 portfolios is value-weighted
(weights proportional to $\mathrm{ME}_{i,t-1}$ at portfolio formation
date):

$$
r^{P}_{t} = \sum_{i \in P_{t-1}} w_{i, t-1} \cdot r_{i, t},
   w_{i, t-1} = \frac{\mathrm{ME}_{i, t-1}}{\sum_{j \in P_{t-1}} \mathrm{ME}_{j, t-1}}.
$$

**Factor returns.** The factor is the long-short combination of
extreme buckets, averaged across size:

- **HML** (High BEME minus Low BEME):
      $\mathrm{HML}_t = \frac{1}{2}(r^{SH}_t + r^{BH}_t) - \frac{1}{2}(r^{SL}_t + r^{BL}_t)$.

- **RMW** (Robust minus Weak profitability, OP):
      $\mathrm{RMW}_t = \frac{1}{2}(r^{SR}_t + r^{BR}_t) - \frac{1}{2}(r^{SW}_t + r^{BW}_t)$.

- **CMA** (Conservative minus Aggressive investment):
      $\mathrm{CMA}_t = \frac{1}{2}(r^{SC}_t + r^{BC}_t) - \frac{1}{2}(r^{SA}_t + r^{BA}_t)$.

- **UMD** (Up minus Down momentum, $r_{12,2}$):
      $\mathrm{UMD}_t = \frac{1}{2}(r^{SU}_t + r^{BU}_t) - \frac{1}{2}(r^{SD}_t + r^{BD}_t)$.

- **SMB** (Small minus Big, averaging across all 6 portfolios):
      $\mathrm{SMB}_t = \frac{1}{3}(r^{SL}_t + r^{SM}_t + r^{SH}_t)
                       - \frac{1}{3}(r^{BL}_t + r^{BM}_t + r^{BH}_t)$
      using the BEME sorting (Fama-French's convention; SMB is invariant
      across the choice of secondary sort if all are 2$\times$3).

- **Market (RMRF)**: value-weighted return of the universe
      minus the risk-free rate, $\mathrm{MKT}^{ex}_t = R^{VW}_t - r_{f,t}$.

For each factor, the sort is rebalanced at the June-end of each year
(except UMD, which uses monthly rebalancing) and held until the next
rebalance.

### Comparison metrics

Let $f^{\mathrm{ours}}_t$ denote our reconstructed factor return at
month $t$, and $f^{\mathrm{KF}}_t$ the corresponding return from
French's data library.

**Monthly correlation.** The primary metric:

$$
\rho^{\mathrm{Pearson}}_f = \mathrm{Pearson}\bigl(
   {f^{\mathrm{ours}}_t}_{t=2017\text{-}01}^{2024\text{-}12},
   {f^{\mathrm{KF}}_t}_{t=2017\text{-}01}^{2024\text{-}12}
\bigr).
$$

**Mean absolute difference, monthly:**

$$
\overline{d}_f^{\mathrm{FF}} = \frac{1}{T_{2017\text{-}24}}
   \sum_{t} |f^{\mathrm{ours}}_t - f^{\mathrm{KF}}_t|.
$$

**Annualized return divergence:**

$$
\Delta^{\mathrm{ann}}_f = \bigl(\overline{f^{\mathrm{ours}}}\bigr)^{12}
                          - \bigl(\overline{f^{\mathrm{KF}}}\bigr)^{12},
$$

where overlines denote sample means. This converts the per-month
divergence into the annualized return units stakeholders typically
interpret.

**Diebold-Mariano test for predictive accuracy:** Test the null
that two factor return series have equal predictive power for the
market. [DieboldMariano1995]. Used as a secondary check when
correlation is borderline.

### Acceptance criteria

The Fama-French replication passes when ALL of the following hold for
the period 2017-01 through 2024-12:

1. **Pearson correlation: ** $\rho^{\mathrm{Pearson}}_f > 0.85$
      for all five FF5 factors and UMD.
2. **MAD: ** $\overline{d}_f^{\mathrm{FF}} < 0.01$ (1% per
      month absolute) for all factors.
3. **Annualized divergence: ** $|\Delta^{\mathrm{ann}}_f| < 0.03$
      (3% per year absolute) for all factors.
4. **No structural break: ** The Pearson correlation is
      sustained ($> 0.80$) in each calendar year individually, not just
      the full-period statistic. Tests for whether 2021 (the GameStop
      year) drives the result.

### The GameStop episode

January 2021 saw the well-documented GameStop short-squeeze that
generated extraordinary returns on small-capitalization stocks
(GME, AMC, others). This affects the Fama-French SMB and HML factors
materially, since both involve small-stock weights.

We expect divergence between our HML and SMB and French's series in
January 2021 to be larger than in other months. The acceptance
criteria above (Criterion 4 in particular) specifically tests whether
this episode breaks the multi-year correlation. If 2021's per-year
correlation falls below 0.80 while other years remain $> 0.85$, we
report 2021 as a known caveat in the LaTeX section but the panel
passes.

**Mitigation.** We do NOT attempt to "fix" GameStop in the data.
We document it as a real economic event that the FF protocol naturally
encodes in extreme returns. Our reconstruction will encode it the
same way if the underlying CRSP returns and ME values are correct.

**Robustness check.** We run a parallel comparison excluding
microcap stocks (`ME` below the NYSE 20th percentile) for
robustness. If the full-sample comparison fails but the
ex-microcap comparison passes, the issue is microcap-specific
(GME-like) and we accept the panel with the microcap caveat noted.

### Output artifacts

`data/foundation/ff_validation.parquet`:

| **Column** | **Type** | **Description** |
|---|---|---|
| `factor` | string | Factor name (HML, RMW, CMA, SMB, MKT, UMD) |
| `date` | datetime | Month-end |
| `ours` | float64 | Our reconstructed factor return |
| `kf` | float64 | Kenneth French's published return |
| `abs_diff` | float64 | $|f^{\mathrm{ours}}_t - f^{\mathrm{KF}}_t|$ |

*Table: Schema of `ff_validation.parquet`.*

`data/foundation/ff_summary.json`: per-factor pass/fail status with
$\rho^{\mathrm{Pearson}}$, MAD, $\Delta^{\mathrm{ann}}$, and a
GameStop-sensitivity flag.

## Arm 3 --- Internal Consistency Audits

Internal audits are independent of any external reference. They verify
that the panel is self-consistent --- that we did not introduce
errors during construction even if the resulting numbers happen to
match CPZ.

### Audit 3.1 --- PIT correctness

For every (`permno`, `date`) pair in the final panel,
the merged `datadate` must satisfy:

$$
\mathrm{avail_date}_{i,t}
   \le \mathrm{date}
   \le \mathrm{avail_date}_{i,t} + 380\text{ days},
\quad \text{where } \mathrm{avail_date}_{i,t} \equiv \mathrm{datadate}_{i,t} + 6\text{ months (rounded to month-end)}.
$$

Equivalently in terms of `datadate`:

$$
\mathrm{datadate}_{i,t} + 6\text{ months}
   \le \mathrm{date}
   \le \mathrm{datadate}_{i,t} + 6\text{ months} + 380\text{ days}.
$$

The lower bound is the 6-month accounting lag (Document 1,
§(subsec:stage0-pit)); the upper bound is the `merge_asof`
tolerance (audit fix 1), measured from `avail_date` (i.e., "12.5 months
*stale*" in Document 1's language --- not 12.5 months after the fiscal
year-end itself).

**Implementation.** A function `audit_pit_bounds()`
in `src/data_reconstruction/pit/audit.py` iterates over the
panel and asserts the bound for every row. A violation triggers a
hard error (not a warning) with the offending row reported. The
function runs as part of the test suite.

**Acceptance.** Zero PIT violations.

### Audit 3.2 --- Look-ahead audit for monthly characteristics

For each monthly characteristic (Momentum family + Risk family +
Liquidity family), verify that the value at panel month-end $t$ uses
ONLY information observable through end-of-day on the last trading day
of month $t$.

The audit is conducted by examining each characteristic's input
specification:

- **Momentum family ($r_{12,2**$, $r_{2,1}$, etc.):} verify
      that the rolling window endpoints are $[t - L, t - k]$ for some
      lookback $L$ and skip $k \ge 1$. The skip $k$ ensures we do not
      use the contemporaneous month-$t$ return when forming a signal
      at month-end $t$ (the standard convention for momentum).

- **Risk family (Beta, IdioVol, Variance):** verify that the
      252-day rolling window ends at month-end $t$ and does not extend
      into the future.

- **Liquidity family (Spread, LTurnover, LME):** verify that
      Spread uses daily returns within month $t$ (not future), that
      LTurnover uses month-$t$ `vol` and `shrout`,
      and that LME uses month-$(t-1)$ market equity.

**Acceptance.** All monthly characteristics pass formal
inspection. This is a code-review check, not a data check; the
implementation tests verify the conventions are correctly applied.

### Audit 3.3 --- Coverage stability

The number of firms in the panel should change smoothly over time,
absent specific known events (e.g., the 1973 NYSE-AMEX listing
expansion, the 1986 NASDAQ-Compustat integration, the 2000 dot-com
bust, the 2007--2009 financial crisis).

We compute the monthly firm count $N_t$ and inspect:

- Time series plot of $N_t$ vs. $t$.

- Year-over-year percent change $\Delta_t = (N_t - N_{t-12}) / N_{t-12}$.

- Months where $|\Delta_t| > 0.10$ are flagged for inspection.

**Acceptance.** The 1986 NASDAQ-Compustat integration produces a
visible step-up in firm count (expected). No other unexplained jumps
of magnitude $> 10%$ year-over-year.

### Audit 3.4 --- Rank-normalization integrity

For each characteristic $f$ at each month-end $t$:

$$
\mathrm{rank}\bigl(z_{i,t,f}\bigr) \in {1, 2, \ldots, N_{t,f}},
   z_{i,t,f} \in (-1/2, 1/2).
$$

I.e., ranks are exactly the integers $1$ to $N_{t,f}$ (with ties
averaged), and the normalized values lie strictly within the open
interval. Equality at $\pm 1/2$ would indicate a normalization bug.

**Acceptance.** 100% of (firm, month, characteristic) triples
satisfy the open interval condition.

### Audit 3.5 --- Output schema integrity

Verify that the final panel matches the schema documented in
Document 1 (§(subsec:stage0-schema), Table~(tab:stage0-schema)):

- Required identifier columns are non-null.

- Numeric columns have the correct dtype (float64 for
      characteristics, int8/int32 for codes).

- No duplicate (`permno`, `date`) rows.

- Date column is monotonic within each `permno` group.

**Acceptance.** Schema matches; zero duplicate keys.

## Validation Implementation Architecture

The validation suite lives in
`src/data_reconstruction/validation/`:

- `cpz_comparison.py`: Arm 1 implementation. Top-level
      function `compare_to_cpz(ours, cpz, output_dir)` produces
      the validation parquet and summary JSON, asserts acceptance
      criteria.

- `fama_french.py`: Arm 2 implementation. Builds FF5+UMD
      from our data using the protocol in
      §(subsec:stage0-arm2); compares to French's published series.

- `ken_french_loader.py`: Pulls Ken French's published
      factor data via WRDS (table `ff.factors_monthly` for
      monthly, `ff.factors_daily` for daily). Verifies dates
      match our sample and converts to our internal format.

- `audits.py`: Arm 3 implementations (PIT, look-ahead,
      coverage, rank, schema). One function per audit; `run_all_audits()`
      orchestrates them.

- `diagnostics.py`: Plotting utilities for the LaTeX figures
      (per-year breadth charts, factor return time series comparisons,
      coverage time series).

The validation pipeline is invoked at the end of Stage 0's
construction pipeline:

- `pipeline.py` produces the final panel.

- `validation/` is run on the produced panel.

- All three arms must pass before `stage0_complete = True`
      is set in the run summary.

## Documentation in the LaTeX Report

The `reports/sections/00_data_foundation.tex` section
incorporates the validation results in a dedicated subsection with:

- **Table 1: Arm 1 summary.** Per-characteristic full-period
      MAD, max-year MAD, mean Spearman correlation, and pass/fail.

- **Figure 1: Arm 1 yearly breadth.** Median MAD across
      characteristics, plotted by year. Shows when the reconstruction
      most closely matches CPZ and when it diverges.

- **Table 2: Arm 2 summary.** Per-factor (HML, RMW, CMA,
      SMB, MKT, UMD) Pearson correlation, MAD, annualized divergence,
      pass/fail.

- **Figure 2: Arm 2 cumulative factor returns.** Time series
      of cumulative returns from our reconstruction vs. French's
      published series, one panel per factor.

- **Table 3: Arm 3 audit results.** PIT, look-ahead,
      coverage, rank, schema pass/fail with violation counts where
      applicable.

- **Discussion.** Documents any known caveats (GameStop in
      2021, the 1986 NASDAQ-Compustat step) and how they are handled.

This subsection is approximately 5--6 pages of compiled LaTeX,
including figures.

## Validation Failure Protocol

If any arm fails its acceptance criteria, Stage 0 implementation does
NOT proceed to Stage 1 until the failure is diagnosed and either
fixed or explicitly accepted with documentation. Specifically:

**Hard fail (block proceeding).**

- PIT correctness violations (Audit 3.1)

- Schema integrity violations (Audit 3.5)

- Coverage stability anomalies of magnitude $> 30%$ year-over-year
      that are not pre-1973 or 1986

**Soft fail (proceed with documented caveat).**

- Single-year violations of CPZ comparison Criterion 1 (yearly
      breadth) localized to a known regime change (e.g., 1986
      NASDAQ-Compustat integration)

- GameStop-2021 effect on FF5 SMB/HML correlation, with ex-microcap
      comparison passing

- One or two characteristics where CPZ Criterion 2 fails (full-period
      agreement) but Criterion 3 (Spearman rank correlation) passes ---
      this indicates correct rank ordering with scaling differences
      that are not material for downstream factor work

**Accept (document and continue).**

- Discrepancies in micro-cap stocks (`ME` below the
      NYSE 5th percentile) due to vendor-data idiosyncrasies; these
      are not the universe we trade.

- Known data revisions in Compustat that postdate CPZ's snapshot
      but predate our snapshot.

For all soft-fail and accept cases, the LaTeX report explicitly
documents the issue, its scope, the diagnosis, and the rationale for
proceeding. This is the operational implementation of v2 Principle 5
(honest results reporting).

## Document 3 Conclusion

Documents 1, 2, and 3 collectively specify Stage 0's complete
mathematical framework:

- Document 1: methodological scaffolding (data sources, PIT,
      audit fixes, Roll spread, rank-normalization).

- Document 2: 46 characteristic formulas with academic references
      and sign conventions.

- Document 3: validation methodology (CPZ overlap, Fama-French
      replication, internal consistency audits).

The framework is now complete. Implementation can begin once Document 3
is approved. The implementation roadmap follows the modular package
structure specified in Document 1 (§(subsec:stage0-context)):
each submodule of `src/data_reconstruction/` is independently
testable, and the pipeline orchestrator wires them together but
contains no construction logic itself.

The validation suite is the final gate: all three arms must pass
before Stage 0 produces its final panel and yields control to Stage 1.
A panel that fails validation does not get used downstream regardless
of how convenient that would be for project timeline. This is
the operational implementation of v2 Principle 4 (no post-hoc
filtering): we lock the validation criteria before the test runs and
accept the result it produces, not the result we wanted.
