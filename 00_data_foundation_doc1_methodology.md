# 00 DATA FOUNDATION DOC1 METHODOLOGY

> **Note for Kimi (or any non-LaTeX-capable platform)**:
> This is a Markdown conversion of the original LaTeX file
> `00_data_foundation_doc1_methodology.tex`. Math is in `$...$` (inline) or `$$...$$` (display)
> notation. Tables and equations are best-effort conversions —
> the LaTeX source is the canonical version when available.

---



# Stage 0: Data Foundation

## Overview and Project Context

Stage 0 reconstructs the firm-characteristic panel of [ChenPelgerZhu2024]
(hereafter *CPZ*) from raw CRSP and Compustat sources, extending it
through 2024. The output is a monthly panel of US common stocks
(1967--2024) carrying \(46\) firm characteristics constructed per
[FreybergerNeuhierlWeber2020] (hereafter *FNW*), excess returns,
market equity, and identifiers.

The choice of CPZ as the reference panel is not incidental. CPZ provides
one of the most carefully validated academic factor datasets for US
equities, and its factor list has become a de facto standard in the
asset-pricing replication literature. Reconstructing CPZ rather than
defining a fresh panel grants two benefits: *(i)* every nontrivial
construction choice is anchored in published methodology, and *(ii)*
we obtain a quantitative validation target via direct comparison against
the published CPZ panel over its 1967--2016 sample.

The 8-year extension (2017--2024) covers economic regimes that the
original CPZ sample does not include: the 2018 trade tensions, the 2020
COVID crash and recovery, the 2022 inflation/rate-shock regime, and the
post-hike recovery. These years are validated against published
Fama-French factor returns from Kenneth French's data library, since
no published characteristic panel exists for direct comparison there.

This is the only stage of the project that touches raw vendor data.
Stages 1--11 consume Stage 0's output panel as a fixed input. The
correctness of Stage 0 is therefore a structural prerequisite for the
entire pipeline.

### Module Structure

Stage 0 is organized as a standalone Python package
`src/data_reconstruction/` with the following submodules
(specified in v2 ROADMAP §4.2):

- `raw_data/`: vendor data loaders (CRSP monthly, CRSP daily,
      Compustat fundamentals, Fama-French factors, delisting adjustments)

- `characteristics/`: one module per family
      (`value.py`, `profitability.py`, `investment.py`,
      `momentum.py`, `risk.py`, `liquidity.py`,
      `other.py`)

- `pit/`: point-in-time merge logic and PIT-correctness audits

- `validation/`: CPZ comparison (1967--2016) and Fama-French
      replication (2017--2024)

- `pipeline.py`: orchestrator that produces the final panel

Each submodule is independently testable. The `pipeline.py`
orchestrator wires them together but contains no construction logic
itself.

## Raw Data Sources

### CRSP Monthly Stock File (MSF)

The CRSP MSF is the canonical record of US common-stock returns and
market equity at monthly frequency. We pull
1960-01--2024-12 (extending to 1960 even though our sample starts in
1967, so that 12-month momentum signals
$r_{12,2}$, $r_{12,7}$, etc.\ have a full lookback at sample start).

Required fields: `permno`, `date` (month-end),
`ret` (raw monthly return), `prc` (closing price,
sign-coded for bid-ask midpoint), `shrout` (shares outstanding,
in thousands), `vol` (monthly volume, in shares),
`cfacshr` (cumulative share-adjustment factor),
`cfacpr` (cumulative price-adjustment factor),
`shrcd` (share code), `exchcd` (exchange code),
`siccd` (SIC industry code), `ticker`, `comnam`.

We restrict to common stock (`shrcd` $\in {10, 11}$) on NYSE,
AMEX, or NASDAQ (`exchcd` $\in {1, 2, 3}$). This filter is
applied at load time, not at universe construction (Stage 1), because
non-common-stock observations carry vendor inconsistencies that pollute
construction even when we never trade them.

Market equity is computed as

$$
\mathrm{ME}_{i,t} = |\mathrm{prc}_{i,t}| \cdot \mathrm{shrout}_{i,t} \cdot 1000,
$$

expressed in dollars (CRSP reports `shrout` in thousands).

### CRSP Daily Stock File (DSF)

The DSF is required for risk characteristics computed via 252-day
rolling regressions: \(\beta\), \(\mathrm{IdioVol}\),
\(\mathrm{Resid_Var}\), \(\mathrm{Variance}\), and the
Roll [Roll1984] implied spread.

Pulled fields: `permno`, `date`, `ret` (daily raw
return). The DSF is large (\(\sim\)60M rows over 1963--2024, \(\sim\)715
MB on disk). We pull from 1963 onward so that 252-day windows for
1967-01-31 estimates are fully populated.

### Compustat Annual Fundamentals

Compustat `funda` provides the balance-sheet, income-statement,
and cash-flow data behind the 20 accounting characteristics. Pulled
fields cover the items required by FNW definitions, including
`at`, `lt`, `seq`, `ceq`, `pstk`,
`pstkrv`, `pstkl`, `txditc`, `ib`,
`ni`, `revt`, `sale`, `cogs`, `xsga`,
`xint`, `dp`, `dv`, `dltt`, `dlc`,
`act`, `lct`, `che`, `intan`,
`ppent`, and `invt`.

We retain only primary-link records (`linkprim` $\in {P, C}$)
from the CRSP-Compustat link table. This eliminates duplicate
permno-datadate observations that arise when a single CRSP security
maps to multiple Compustat `gvkey` records (typically through
mergers or restatements). Primary-link selection is the standard
WRDS-CCM convention.

### Fama-French Factors

Used at two layers. The risk-free rate \(r_{f,t}\) is required to
construct excess returns
\(r^{ex}_{i,t} = r_{i,t} - r_{f,t}\). Daily and monthly FF factors
both come directly from Kenneth French's data library via WRDS.

The FF research factors (HML, RMW, CMA, UMD) themselves are not
Stage 0 outputs. They appear in Stage 0 only as validation targets
in §(subsec:stage0-rationale-summary) below; the construction
methodology for our FF5+UMD replication is detailed in Document 3.

### CRSP Delisting Returns

CRSP separately reports `dlret`, the return realized in the
delisting month. This must be folded into the standard return series;
otherwise we systematically lose the delisting-month return and incur
survivorship bias. The merge logic is detailed in
§(subsec:stage0-shumway) below.

## Excess Returns and the Delisting Adjustment

### The delisting bias

CRSP `ret` reports the holding-period return for stocks alive
through the month-end. For stocks that delist mid-month, no entry
appears in `ret` for the delisting month; the delisting-month
return is reported separately in `dlret`. Failing to merge
`dlret` into the return series biases backtests upward by
omitting losses from delisted stocks.

[Shumway1997] documented this bias and proposed an imputation
rule: when a stock delists for cause and `dlret` is missing,
assign a return based on the delisting code (`dlstcd`). The
canonical imputations are:

- \(\mathrm{dlstcd} \in {500} \cup [520, 584]\): performance-related
      delistings (e.g., insufficient capital, liquidation). Impute
      \(r = -0.30\) for NYSE/AMEX, \(r = -0.55\) for NASDAQ.

- \(\mathrm{dlstcd} = 100\): still-active or merged. Use
      `dlret` if available, else 0 (no return information lost).

- Other codes (e.g., 200--499 for ordinary delistings): use
      `dlret` if available, else \(r = 0\).

### Merge logic

For a stock-month \((i, t)\) where \(i\) delisted in month \(t\):

$$
r_{i,t} =
\begin{cases}
\mathrm{dlret}_{i,t}, & \text{if } \mathrm{dlret}_{i,t} \neq \text{NA} \\
-0.30, & \text{if } \mathrm{dlstcd}_i \in \mathcal{D}_{\text{perf}}
        \text{ and exchange} \in {\text{NYSE/AMEX}} \\
-0.55, & \text{if } \mathrm{dlstcd}_i \in \mathcal{D}_{\text{perf}}
        \text{ and exchange} = \text{NASDAQ} \\
0, & \text{otherwise}
\end{cases}
$$

where \(\mathcal{D}_{\text{perf}} = {500} \cup {520, 521, \ldots, 584}\)
is the set of performance-related delisting codes.

After this imputation, the excess return is computed as

$$
r^{ex}_{i,t} = r_{i,t} - r_{f,t}.
$$

## Book Equity: Davis-Fama-French (2000) Definition

Book equity \(\mathrm{BE}_{i,t}\) is the denominator of multiple value
characteristics (BEME, $\mathrm{BE}/\mathrm{ME}$). Compustat does not
report book equity directly; it must be constructed from balance-sheet
items, and several legitimate definitions exist. We follow
[DavisFamaFrench2000], which is the dominant academic convention.

### Stockholders' equity hierarchy

Stockholders' equity \(\mathrm{SE}\) is constructed with the following
priority:

$$
\mathrm{SE}_{i,t} =
\begin{cases}
\mathrm{seq}_{i,t}, & \text{if } \mathrm{seq}_{i,t} \neq \text{NA} \\
\mathrm{ceq}_{i,t} + \mathrm{pstk}_{i,t}, & \text{else if } \mathrm{ceq}_{i,t} \neq \text{NA} \\
\mathrm{at}_{i,t} - \mathrm{lt}_{i,t}, & \text{otherwise}.
\end{cases}
$$

The hierarchy reflects increasing approximation. `seq` is the
direct stockholders' equity figure (best). `ceq + pstk` is the
common-equity figure plus preferred stock (still good). The book-value
identity `at - lt` is the residual approximation, used only when
neither of the prior two is available.

### Preferred stock hierarchy

Preferred stock \(\mathrm{PS}\) is constructed in priority order:

$$
\mathrm{PS}_{i,t} =
\begin{cases}
\mathrm{pstkrv}_{i,t}, & \text{if } \mathrm{pstkrv}_{i,t} \neq \text{NA}
                          \text{(redemption value)}\\
\mathrm{pstkl}_{i,t}, & \text{else if } \mathrm{pstkl}_{i,t} \neq \text{NA}
                          \text{(liquidating preference)}\\
\mathrm{pstk}_{i,t}, & \text{otherwise (carrying value)}.
\end{cases}
$$

The redemption value is the most economically relevant figure; the
others are accounting approximations of decreasing fidelity.

### Book equity formula

Book equity is then

$$
\mathrm{BE}_{i,t} = \mathrm{SE}_{i,t} + \mathrm{txditc}_{i,t} - \mathrm{PS}_{i,t},
$$

where `txditc` (deferred taxes plus investment tax credits) is
added back as part of equity per the
[DavisFamaFrench2000] convention. Missing `txditc` is
treated as zero.

We retain only \(\mathrm{BE}_{i,t} > 0\); negative book equity is not
economically meaningful for valuation ratios and we assign `NaN`
in such cases. Approximately 0.5--1.5% of firm-years have negative book
equity in the modern sample; these are dropped from value-characteristic
computations but retained for the panel (other characteristics are
unaffected).

## Point-in-Time (PIT) Mechanics

### The PIT requirement

A factor signal at month \(t\) must be computable using *only*
information that would have been available to a real investor at time
\(t\). Violation of this rule (*look-ahead bias*) inflates
backtested performance because the model effectively peeks at future
data.

For market data (CRSP daily and monthly), the PIT timestamp is the
trade date itself: a price for 2015-06-30 was knowable at 2015-06-30.
For accounting data (Compustat), there is a structural lag between the
fiscal year-end (when the data refers to) and the publication date
(when investors learn the data). The standard academic convention,
following Fama-French, is a *6-month lag*.

### The 6-month lag

For a fiscal year ending on `datadate`, we set the data's
"available date" to

$$
\mathrm{avail_date}_{i,t} = \mathrm{datadate}_{i,t} +
                              \text{6 months, rounded to month-end}.
$$

This is a conservative bound. Empirically, most 10-K filings occur
3--4 months after fiscal year-end, but tail cases (extension filings,
restatements, smaller firms) can extend to 5--6 months. The 6-month
convention ensures that a filing was almost certainly public by
`avail_date`. For a fiscal year ending 2014-12-31, the avail
date is 2015-06-30; the data is usable starting from the 2015-06-30
panel snapshot.

A more aggressive lag (e.g., 3 months) would slightly improve the
freshness of accounting data at the cost of risking forward-leakage in
late-filing firms. The 6-month convention has decades of academic use
and matches the CPZ benchmark, so we adopt it without modification.

### Merge methodology: `pandas.merge_asof`

The CRSP monthly panel and the Compustat panel have different time
indices. CRSP is observed at every month-end; Compustat is observed at
each firm's fiscal year-end (which can be any month). We need to merge
the most recent available Compustat record into each CRSP month-end
observation, separately for each firm.

`pandas.merge_asof` implements this directly. For each
permno-date pair in the panel, it finds the most recent Compustat
observation for that permno where `avail_date` \(\le\) panel
date, then attaches that record's columns. Crucially, this is a
*backward*-direction merge: it only uses information from the
past, never the future.

### The `tolerance parameter and the 380-day choice`

`merge_asof` allows a maximum staleness via the `tolerance`
argument. If no Compustat record within the tolerance window exists for
a given firm-month, the merge yields NA for the accounting columns.

We set `tolerance` to 380 days. Justification:

- Accounting data is annual. The expected gap between consecutive
      `avail_date` entries for a firm is 12 months
      (\(\sim\)365 days).

- A small slack above 365 days accommodates fiscal-year-end shifts
      and irregular reporting schedules: a firm changing its fiscal
      year may have one 11-month and one 13-month gap, but consecutive
      gaps over 380 days would imply skipping an annual filing entirely.

- 380 days is approximately 12.5 months. Larger tolerances
      (e.g., 548 days $\approx$ 18 months, used in some research
      pipelines) would let an 18-month-old filing populate a current
      month, which is too stale to be informative for valuation.

- This choice forces firms with missed filings (typically firms
      under SEC enforcement actions or in financial distress) to drop
      out of the panel until they file again. This is the correct
      behavior: such firms have impaired information transparency and
      a real investor would not trade them on stale data.

### Audit fix 1: tolerance from 548 to 380 days

An earlier v1 implementation used `tolerance = 548 days`. This
allowed accounting data up to 18 months stale to populate a current
month. The fix to 380 days shortens this window to a more appropriate
12.5 months, dropping stale-data firm-months that would otherwise
contaminate value and profitability characteristics.

### PIT correctness audit

We perform an explicit PIT audit on the merged panel: for each
firm-month, we verify that the merged `datadate` satisfies
\(\mathrm{datadate}\) + 6 months
$\le$ panel date $<$
\(\mathrm{datadate}\) + 380 days. Any violation triggers a hard error,
not a warning. The audit is implemented in
`src/data_reconstruction/pit/audit.py` and runs in the test
suite.

## The Other Two Audit Fixes

In addition to the merge_asof tolerance reduction
(§(subsec:stage0-pit)), v1 implementation revealed two
construction bugs that v2 retains as fixes.

### Audit fix 2: AC (Total Accruals) without `fillna(0)`

Total Accruals (AC) following [RichardsonSloanSoliman2005] is
defined as

$$
\mathrm{AC}_{i,t} = \frac{\mathrm{NOA}_{i,t} - \mathrm{NOA}_{i,t-1}}
                          {\mathrm{AT}_{i,t-1}},
$$

where NOA is Net Operating Assets. The denominator \(\mathrm{AT}_{i,t-1}\)
must be available, which requires both:

1. Current and prior-period NOA are computable, and
2. \(\mathrm{AT}_{i,t-2}\) (used to scale prior NOA) is non-missing.

A v1 bug applied `NOA.fillna(0)` on the prior-period NOA term.
This produced spurious AC values for first-observation firms and firms
with missing `at_lag2`: the formula effectively reduced to
\(\mathrm{AC}_{i,t} = \mathrm{NOA}_{i,t} \cdot
\mathrm{at_lag1}_{i,t} / \mathrm{at_lag1}_{i,t}\), which is just
the level of NOA, not its change.

The v2 fix returns NA when either prior NOA term is unavailable, letting
the completeness filter drop those firm-months instead of imputing a
fake value.

### Audit fix 3: `Rel2High uses split-adjusted price`

\(\mathrm{Rel2High}\) per [GeorgeHwang2004] is the ratio of current
price to its 52-week (12-month) high:

$$
\mathrm{Rel2High}_{i,t}
   = \frac{\mathrm{prc}^{\mathrm{adj}}_{i,t}}{
       \max_{s \in [t-12, t-1]} \mathrm{prc}^{\mathrm{adj}}_{i,s}}.
$$

The v1 bug used raw, unadjusted prices. When a stock undergoes a stock
split during the lookback window (e.g., 2-for-1), the pre-split prices
are 2x larger than post-split prices, producing a 12-month "high" that
the post-split price cannot exceed. The result is a Rel2High value of
0.5 driven entirely by the split, not by price weakness.

The fix uses split-adjusted prices throughout:

$$
\mathrm{prc}^{\mathrm{adj}}_{i,t} = \frac{|\mathrm{prc}_{i,t}|}{\mathrm{cfacpr}_{i,t}},
$$

where `cfacpr` is CRSP's cumulative price-adjustment factor.
After this adjustment, all prices are on a single comparable scale and
splits do not contaminate the rolling maximum.

## Roll (1984) Implied Bid-Ask Spread

The bid-ask spread \(\mathrm{Spread}_{i,t}\) is required as a liquidity
characteristic. Direct bid-ask quote data is unavailable in CRSP for
NYSE/AMEX stocks before 1983 and for NASDAQ before 1993. To maintain
coverage from 1967, we use the [Roll1984] implied spread
estimator, which infers spread from the serial covariance of daily
returns.

### Derivation

Roll's estimator rests on a specific market-microstructure model. Assume
the true (unobserved) price \(p^*_t\) follows a random walk with no
drift, and the observed price \(p_t\) is the true price plus a buy-sell
indicator scaled by half the bid-ask spread:

$$
p_t = p^*_t + \tfrac{1}{2} s \cdot q_t,
   q_t \in {-1, +1},
   q_t \stackrel{\mathrm{iid}}{\sim} \text{Bernoulli}(0.5).
$$

The observed return is

$$
r_t = p_t - p_{t-1} = (p^*_t - p^*_{t-1}) + \tfrac{1}{2} s (q_t - q_{t-1}).
$$

Computing the lag-1 autocovariance of \(r_t\):

$$
\mathrm{Cov}(r_t, r_{t-1})
&= \mathrm{Cov}\left(\tfrac{1}{2} s (q_t - q_{t-1}),
                    \tfrac{1}{2} s (q_{t-1} - q_{t-2})\right) \notag\\
&= \tfrac{1}{4} s^2 \mathrm{Cov}(q_t - q_{t-1}, q_{t-1} - q_{t-2}) \notag\\
&= \tfrac{1}{4} s^2 \cdot (-1) \notag\\
&= -\tfrac{1}{4} s^2.
$$

Solving for \(s\):

$$
s = 2 \sqrt{-\mathrm{Cov}(r_t, r_{t-1})}.
$$

This is the Roll (1984) estimator. It is well-defined only when the
sample covariance is negative; positive covariance can arise from
trending markets or noise and renders the estimator inapplicable.

### Implementation

For each firm-month, we compute the daily-return autocovariance over
that month's daily observations:

$$
\widehat{\mathrm{Spread}}_{i,t} =
\begin{cases}
2 \sqrt{-\widehat{\gamma}_1(r_{i,t})}, &
   \text{if } \widehat{\gamma}_1(r_{i,t}) < 0
   \text{ and } |\mathcal{D}_{i,t}| \ge 10 \\
0, & \text{if } \widehat{\gamma}_1(r_{i,t}) \ge 0 \\
\text{NA}, & \text{otherwise.}
\end{cases}
$$

Here \(\widehat{\gamma}_1\) denotes the lag-1 sample autocovariance and
\(\mathcal{D}_{i,t}\) is the set of valid daily-return observations
within firm-month \((i,t)\). We require at least 10 daily observations
to ensure the autocovariance estimate is stable.

A nonnegative autocovariance is set to a spread of zero rather than NA
because the estimator structurally cannot accommodate this case but the
firm-month is otherwise observed and tradable. Roll spread is therefore
non-negative by construction.

### Limitations

Roll spread is a noisy proxy. [Goyenko2009] document that it
captures the true effective spread less reliably than alternatives like
the Effective Tick estimator or High-Low spread, particularly for
illiquid stocks. We use Roll because it is the only estimator available
on the full 1963--2024 daily sample without separate quote data;
post-1993 sensitivity analyses can replace it with effective spread for
robustness checks but are not required for the core panel.

## Rank Normalization

After all 46 raw characteristics are computed and merged onto the
monthly panel, we cross-sectionally rank-normalize each characteristic
to a common scale. This is the final step before output.

### Definition

For characteristic \(f\) at date \(t\), let \(N_t\) be the number of
firms with non-missing observations of \(f\) at date \(t\). For each
firm \(i\) with rank \(R_{i,t,f}\) (in ascending order, ties averaged):

$$
z_{i,t,f} = \frac{R_{i,t,f}}{N_t + 1} - \tfrac{1}{2}.
$$

This places \(z_{i,t,f}\) in the open interval
\((-1/2, 1/2)\). The smallest non-missing observation maps to
approximately \(-1/2 + 1/(N_t+1)\); the largest maps to approximately
\(1/2 - 1/(N_t+1)\).

### Why rank normalize?

Rank normalization confers four properties critical to factor analysis:

**(1) Scale invariance.** Different characteristics are measured
in different units (BEME is unitless; AT is log dollars; Beta is
unitless but bounded near \(\pm 3\)). Rank-normalization places all
factors on a single bounded scale, making them comparable across families
and combinable in any composite signal without arbitrary rescaling.

**(2) Outlier resistance.** Heavy-tailed financial data
(e.g., extreme value ratios from low-ME stocks) would dominate
weighted combinations if used in raw form. Rank normalization caps any
observation's influence at \(1/2\), eliminating outlier dominance.

**(3) Cross-sectional standardization within month.**
Equation~(eq:rank-norm) is computed independently per month \(t\).
This removes time-series drift in characteristic levels (e.g., long-run
changes in average leverage) while preserving cross-sectional rankings
within each month. The factor signal is therefore "high BEME relative
to other firms this month" rather than "high BEME on an absolute scale."

**(4) Compatibility with downstream methods.** Stage 4's
combination methods (PCA, regression, etc.) all expect inputs on
comparable scales. Rank-normalized characteristics satisfy this
without further preprocessing.

### Comparison to z-scoring

An alternative is cross-sectional standardization (z-scoring):

$$
z_{i,t,f} = \frac{x_{i,t,f} - \overline{x}_{t,f}}{\sigma_{t,f}}.
$$

Z-scoring preserves the shape of the distribution, while rank-normalization
imposes a uniform shape regardless of input. For factor analysis with
heavy-tailed inputs (which most firm characteristics are), the uniformity
of rank-normalization is preferable; CPZ and FNW both follow this
convention. We adopt rank-normalization to match.

## Completeness Filter

The CPZ panel applies a strict universe rule: a firm-month is retained
in the panel only if *all 46 characteristics* are simultaneously
non-missing.

$$
(i, t) \in \mathcal{U}_{\text{CPZ}}
\iff
z_{i,t,f} \neq \text{NA}    \forall f \in {1, \ldots, 46}.
$$

This is restrictive. A firm with missing Compustat data for any one
fiscal year drops out of the panel for that year. Most missingness
arises from short Compustat history (newly-public firms) or filing
gaps (firms in financial distress).

The filter has two consequences. First, the panel is balanced in the
sense that every firm-month has every factor; downstream stages do not
need to handle factor missingness. Second, the panel is biased toward
firms with longer accounting histories; very small or recently-IPO'd
firms are underrepresented. Stage 1 universe filters can choose to
mitigate or amplify this depending on the desired research universe.

We apply the filter *before* rank-normalization, so that ranks are
computed on the filtered universe. Rank-normalizing first and then
filtering would produce ranks computed on a larger but inconsistent
sample, contaminating the cross-sectional standardization.

## The Final Panel: Schema

The output of Stage 0 is a single panel saved as Parquet (split into
train/valid/test for ML convenience but otherwise contiguous). One row
per (`permno`, `date`) pair satisfying the completeness
filter.

| **Column** | **Type** | **Description** |
|---|---|---|
| `permno` | int64 | CRSP permanent security identifier |
| `date` | datetime | Month-end date |
| `ret` | float64 | Raw monthly return (post-Shumway) |
| `ret_excess` | float64 | Excess return ($r_{i,t} - r_{f,t}$) |
| `me` | float64 | Market equity, dollars |
| `prc` | float64 | Closing price (signed CRSP value) |
| `shrout` | float64 | Shares outstanding (thousands) |
| `ticker` | string | Current ticker symbol |
| `ticker_current` | string | Most recent ticker for permno (for Stage 10) |
| `exchcd` | int8 | Exchange code (1=NYSE, 2=AMEX, 3=NASDAQ) |
| `siccd` | int32 | SIC industry code |
| `shrcd` | int8 | Share code |
| `datadate` | datetime | Compustat fiscal year-end (PIT audit) |
| `avail_date` | datetime | Earliest panel date for accounting data |
| \multicolumn{3}{l}{46 rank-normalized characteristic columns:} |
| `BEME, E2P, ...` | float64 | In open interval $(-1/2, 1/2)$ |

*Table: Stage 0 final panel schema. Approximately 1.6M rows over
1967--2024.*

## Summary of Methodological Choices

The methodological choices in Stage 0 are summarized in
Table~(tab:stage0-rationale). Each choice has a published
academic basis and is empirically validated in Stage 0 Document 3
(forthcoming).

| p{5cm}p{4cm}}

**Choice** | **Convention** | **Reference** |
|---|---|---|
| Reference panel | CPZ extended factor panel | [ChenPelgerZhu2024] |
| Characteristic definitions | FNW (46 characteristics) | [FreybergerNeuhierlWeber2020] |
| Book equity formula | Davis-Fama-French (2000) hierarchy | [DavisFamaFrench2000] |
| Delisting return imputation | Shumway (1997) by exchange | [Shumway1997] |
| Accounting data lag | 6-month from fiscal year-end | [FamaFrench1993] convention |
| Merge tolerance | 380 days (12.5 months) | v1 audit fix |
| Bid-ask spread proxy | Roll (1984) implied spread | [Roll1984] |
| Cross-sectional scaling | Rank-normalize to $(-1/2, 1/2)$ | CPZ/FNW convention |
| Universe filter | All 46 chars non-missing | CPZ convention |
| Audit fix 2: AC formula | No `fillna(0)` on lagged NOA | v1 audit fix |
| Audit fix 3: Rel2High | Split-adjusted prices | v1 audit fix |

*Table: Stage 0 methodological choices and their academic basis.*

## Document 1 Conclusion

Document 1 has established the methodological scaffolding for Stage 0:
data sources, point-in-time mechanics, key adjustments (Shumway
delisting, book equity, audit fixes), and the rank-normalization
that delivers the final panel. It does not yet specify the 46
characteristic formulas; that is the content of Document 2. Validation
methodology (CPZ comparison, Fama-French replication) is detailed in
Document 3.

The methodological choices in this document are tightly anchored to
published academic conventions, and the audit fixes are documented with
their mathematical justifications. Implementation can proceed once
Documents 2 and 3 are also approved.
