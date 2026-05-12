# 00 DATA FOUNDATION DOC2 CHARACTERISTICS

---



# Stage 0 (cont.): Characteristic Definitions

## Overview and Notation

Document 1 established the methodological scaffolding: data sources,
PIT mechanics, the Davis-Fama-French book equity construction, the
Shumway delisting adjustment, the Roll spread derivation, and the
rank-normalization that produces our final outputs. Document 2 specifies
the construction formulas for the 46 firm characteristics that populate
the panel.

The 46 characteristics decompose into seven families based on economic
content:

| **Family** | **Count** | **Members** |
|---|---|---|
| Value | 6 | BEME, E2P, CF2P, D2P, S2P, A2ME |
| Profitability | 7 | PROF, ROE, ROA, OP, PM, PCM, RNA |
| Investment | 6 | Investment, NOA, DPI2A, NI, OA, AC |
| Momentum | 8 | r12_2, r2_1, r12_7, r36_13, ST_REV, LT_Rev, SUV, Rel2High |
| Risk | 5 | Beta, MktBeta, IdioVol, Resid_Var, Variance |
| Liquidity | 3 | Spread, LTurnover, LME |
| Other | 11 | Q, C, CF, AT, ATO, CTO, D2A, FC2Y, Lev, OL, SGA2S |
| **Total** | **46** |  |

*Table: The 46 characteristics in the CPZ-extended panel, organized by
economic family.*

The "Other" family is heterogeneous: it contains size (AT), Tobin's Q,
cash-to-assets, leverage variants, and turnover-style measures that do
not fit cleanly into the other six families.

Notation conventions for the rest of this document:

- $i$ indexes firms (CRSP `permno`); $t$ indexes time
      (month-end for monthly data, fiscal-year-end for annual).

- $r_{i,t}$ denotes the raw monthly return; $r^{ex}_{i,t}$ the
      excess return after subtracting the risk-free rate.

- $\mathrm{ME}_{i,t}$ denotes month-end market equity in dollars
      (eq.~(eq:me-defn) in Document 1).

- $\mathrm{BE}_{i,t}$ denotes Davis-Fama-French book equity
      (eq.~(eq:be-defn) in Document 1).

- Compustat balance-sheet items are denoted by their lowercase
      Compustat names (e.g., `at`, `ib`, `cogs`).

- For an annual Compustat item observed at fiscal year-end,
      $X_{i,t-1}$ denotes the value from the previous fiscal year.

- All ME-dependent ratios (Value family) use *contemporaneous*
      market equity at the panel month-end, paired with the PIT-merged
      accounting numerator. The numerator's `avail_date` must
      be $\le$ the panel month-end (i.e., already public).

- For monthly (price-based) characteristics, signals at month $t$
      use information observable through month-end $t$.

The "Sign convention" rows in each subsection state the predicted
correlation between the characteristic and forward returns following
the published source. Positive ($+$) means high values predict positive
returns; negative ($-$) means high values predict negative returns.
These are theoretical predictions; empirical signs in our sample are
established in Stage 3 (factor diagnostics).

## Value Family (6)

The value family contains characteristics that scale a fundamental
quantity (book value, earnings, sales, cash flow, dividends, assets) by
market equity. The economic logic is that low market values relative to
fundamentals signal undervaluation and predict positive future returns.
All six are computed at panel-merge time, since the denominator
$\mathrm{ME}_{i,t}$ is the panel month-end market equity, not the
`datadate` value.

### BEME --- Book-to-Market Equity

**Definition:**

$$
\mathrm{BEME}_{i,t} = \frac{\mathrm{BE}_{i,\tau(t)}}{\mathrm{ME}_{i,t}},
$$

where $\tau(t)$ denotes the most recent fiscal year-end with
$`avail_date` \le t$ (i.e., the PIT-merged accounting record).

**Required inputs:**
`seq`, `ceq`, `pstk`, `pstkrv`,
`pstkl`, `txditc`, `at`, `lt` (for BE);
`prc`, `shrout` (for ME).

**Reference:** [FamaFrench1992,FamaFrench1993].

**Sign convention:** positive. High BEME indicates the market
values the firm at less than its book equity, the canonical value
signal.

**Known issues:** Negative BE (eq.~(eq:be-defn) drops these)
removes the firm-month from the value-family characteristics. Approximately
0.5--1.5% of firm-years have negative BE in modern data; these tend to
be distressed firms. The completeness filter
(§(subsec:stage0-completeness)) drops the firm-month from the
panel when any characteristic is missing, including BEME.

### E2P --- Earnings-to-Price

**Definition:**

$$
\mathrm{E2P}_{i,t} = \frac{\mathrm{IB}_{i,\tau(t)}}{\mathrm{ME}_{i,t}}.
$$

where IB is income before extraordinary items.

**Required inputs:** `ib`, `prc`, `shrout`.

**Reference:** [Basu1977,FamaFrench1992].

**Sign convention:** positive. High earnings yield (low P/E) is a
classical value signal.

**Known issues:** Negative IB produces negative E2P. We retain
negative-earnings firms (do not set NA), since the rank-normalization
will assign them low cross-sectional ranks, which is the appropriate
treatment in a long-short factor.

### CF2P --- Cash-Flow-to-Price

**Definition:**

$$
\mathrm{CF2P}_{i,t} = \frac{\mathrm{IB}_{i,\tau(t)} + \mathrm{DP}_{i,\tau(t)}}
                            {\mathrm{ME}_{i,t}},
$$

where DP is depreciation and amortization. The numerator
$\mathrm{IB} + \mathrm{DP}$ is a commonly-used cash-flow proxy:
income plus non-cash depreciation.

**Required inputs:** `ib`, `dp`, `prc`,
`shrout`.

**Reference:** [Lakonishok1994].

**Sign convention:** positive.

**Known issues:** As with E2P, negative numerators are retained
and rank-normalized.

### D2P --- Dividend Yield

**Definition:**

$$
\mathrm{D2P}_{i,t} = \frac{\mathrm{DV}_{i,\tau(t)}}{\mathrm{ME}_{i,t}},
$$

where DV is total cash dividends paid (Compustat `dv`).

**Required inputs:** `dv`, `prc`, `shrout`.

**Reference:** [LitzenbergerRamaswamy1979,FamaFrench1988].

**Sign convention:** positive (mild). Dividend yield is a weak but
historically positive predictor.

**Known issues:** Many firms pay no dividends ($\mathrm{DV} = 0$
or NA). In v2, missing `dv` is treated as zero, ensuring D2P is
computable for all firms (potentially equal to zero). This is the
standard convention for dividend-yield characteristics.

### S2P --- Sales-to-Price

**Definition:**

$$
\mathrm{S2P}_{i,t} = \frac{\mathrm{SALE}_{i,\tau(t)}}{\mathrm{ME}_{i,t}}.
$$

**Required inputs:** `sale`, `prc`, `shrout`.

**Reference:** [Barbee1996].

**Sign convention:** positive. Like other value ratios, but uses
sales rather than earnings or book value as the fundamental anchor;
robust for firms with negative or volatile earnings.

### A2ME --- Assets-to-Market-Equity

**Definition:**

$$
\mathrm{A2ME}_{i,t} = \frac{\mathrm{AT}_{i,\tau(t)}}{\mathrm{ME}_{i,t}},
$$

where AT is total assets.

**Required inputs:** `at`, `prc`, `shrout`.

**Reference:** [BharyMcLean2014].

**Sign convention:** positive. Conceptually similar to BEME but
uses total assets in the numerator instead of book equity. Highly
correlated with BEME but not identical.

## Profitability Family (7)

Profitability characteristics measure efficiency at converting inputs
(revenue, equity, assets) into earnings or operating profits. The
economic logic is that more profitable firms generate higher returns
than less profitable firms with similar valuations, controlling for
risk. All seven are computed from Compustat with the 6-month lag.

### PROF --- Gross Profitability

**Definition:**

$$
\mathrm{PROF}_{i,t} = \frac{\mathrm{SALE}_{i,t} - \mathrm{COGS}_{i,t}}
                           {\mathrm{BE}_{i,t}},
$$

defined when $\mathrm{BE}_{i,t} > 0$.

**Required inputs:** `sale`, `cogs`, `seq`,
`ceq`, `pstk`, `pstkrv`, `pstkl`,
`txditc`, `at`, `lt`.

**Reference:** [NovyMarx2013].

**Sign convention:** positive. Novy-Marx's central finding is that
gross profitability is an alternative measure of "quality" that predicts
returns in excess of value and momentum.

### ROE --- Return on Equity

**Definition:**

$$
\mathrm{ROE}_{i,t} = \frac{\mathrm{IB}_{i,t}}{\mathrm{BE}_{i,t-1}},
$$

defined when $\mathrm{BE}_{i,t-1} > 0$. The lagged book equity in the
denominator follows the standard accounting convention: ROE measures
the return generated this period on the equity available at the start
of the period.

**Required inputs:** `ib`, current and prior-year
`seq`, etc.

**Reference:** [HouXueZhang2015].

**Sign convention:** positive.

**Known issues:** Requires prior-year BE to be available; the
first year of a firm's Compustat history will have NA ROE.

### ROA --- Return on Assets

**Definition:**

$$
\mathrm{ROA}_{i,t} = \frac{\mathrm{IB}_{i,t}}{\mathrm{AT}_{i,t-1}},
$$

defined when $\mathrm{AT}_{i,t-1} > 0$.

**Required inputs:** `ib`, prior-year `at`.

**Reference:** [HouXueZhang2015,BalakrishnanBartovFaurel2010].

**Sign convention:** positive.

### OP --- Operating Profitability

**Definition:**

$$
\mathrm{OP}_{i,t} = \frac{\mathrm{REVT}_{i,t} - \mathrm{COGS}_{i,t}
                          - \mathrm{XSGA}_{i,t} - \mathrm{XINT}_{i,t}}
                          {\mathrm{BE}_{i,t}},
$$

defined when $\mathrm{BE}_{i,t} > 0$. The numerator is
operating profit: revenue minus cost of goods sold, minus selling
general and administrative expenses, minus interest expense. Missing
values for COGS, XSGA, or XINT are treated as zero (the standard
Fama-French convention).

**Required inputs:** `revt` (or `sale` as fallback),
`cogs`, `xsga`, `xint`, BE inputs.

**Reference:** [FamaFrench2015].

**Sign convention:** positive. Operating profitability is one of
the five factors in FF5 and a robust positive return predictor.

### PM --- Profit Margin

**Definition:**

$$
\mathrm{PM}_{i,t} = \frac{\mathrm{IB}_{i,t}}{\mathrm{SALE}_{i,t}},
$$

defined when $|\mathrm{SALE}_{i,t}| > 0.01$ (a small threshold to avoid
division by near-zero).

**Required inputs:** `ib`, `sale`.

**Reference:** [SoliFridson1995].

**Sign convention:** positive.

### PCM --- Price-Cost Margin

**Definition:**

$$
\mathrm{PCM}_{i,t} = \frac{\mathrm{SALE}_{i,t} - \mathrm{COGS}_{i,t}}
                          {\mathrm{SALE}_{i,t}},
$$

defined when $|\mathrm{SALE}_{i,t}| > 0.01$. PCM is gross margin: gross
profit divided by sales.

**Required inputs:** `sale`, `cogs`.

**Reference:** [Gorodnichenko2010].

**Sign convention:** positive (mild).

### RNA --- Return on Net Assets

**Definition:**

$$
\mathrm{RNA}_{i,t} = \frac{\mathrm{IB}_{i,t}}{\mathrm{AT}_{i,t} - \mathrm{CHE}_{i,t}},
$$

defined when $|\mathrm{AT}_{i,t} - \mathrm{CHE}_{i,t}| > 0.01$. The
denominator is "operating assets" --- total assets minus cash and short-term
investments. Missing CHE is treated as zero.

**Required inputs:** `ib`, `at`, `che`.

**Reference:** [SoliFridson1995].

**Sign convention:** positive. RNA strips out cash holdings to
focus on the productive asset base; high RNA indicates efficient
operations.

## Investment Family (6)

The investment family contains characteristics measuring the rate of
firm-level capital investment (real, intangible, and accounting). The
core economic finding (CooperGulenSchill2008, formalized in
HouXueZhang2015,FamaFrench2015) is that high-investment firms
underperform low-investment firms after controlling for size and value.

### Investment --- Annual Asset Growth

**Definition:**

$$
\mathrm{Investment}_{i,t} = \frac{\mathrm{AT}_{i,t}}{\mathrm{AT}_{i,t-1}} - 1,
$$

defined when $\mathrm{AT}_{i,t-1} > 0$. The simplest investment
characteristic: the annual percentage change in total assets.

**Required inputs:** `at`, prior-year `at`.

**Reference:** [CooperGulenSchill2008].

**Sign convention:** negative. The "asset-growth anomaly" is one
of the most robustly documented effects in factor research.

### NOA --- Net Operating Assets

**Definition:**

$$
\mathrm{NOA}_{i,t} = \frac{\mathrm{OP}^{\text{assets}}_{i,t}
                            - \mathrm{OP}^{\text{liab}}_{i,t}}
                          {\mathrm{AT}_{i,t-1}},
$$

where the operating-assets and operating-liabilities terms are

$$
\mathrm{OP}^{\text{assets}}_{i,t} &= \mathrm{AT}_{i,t} - \mathrm{CHE}_{i,t}
                                    - \mathrm{INTAN}_{i,t}, \\
\mathrm{OP}^{\text{liab}}_{i,t} &= \mathrm{AT}_{i,t} - \mathrm{DLTT}_{i,t}
                                  - \mathrm{DLC}_{i,t} - \mathrm{PSTK}_{i,t}
                                  - \mathrm{SE}_{i,t}.
$$

The intuition is to compute net operating assets (productive assets net
of operating liabilities) and scale by lagged total assets. Missing
values for CHE, INTAN, DLTT, DLC, PSTK, and the various BE-related
items are treated as zero.

**Required inputs:** `at`, `che`, `intan`,
`dltt`, `dlc`, `pstk`, BE inputs, prior-year
`at`.

**Reference:** [HirshleiferHouTeoh2004].

**Sign convention:** negative. High NOA indicates aggressive
investment in operating assets, which on average underperforms.

### DPI2A --- Change in PPE and Inventory

**Definition:**

$$
\mathrm{DPI2A}_{i,t} = \frac{\Delta \mathrm{PPENT}_{i,t}
                              + \Delta \mathrm{INVT}_{i,t}}
                            {\mathrm{AT}_{i,t-1}},
$$

where $\Delta X_{i,t} = X_{i,t} - X_{i,t-1}$ and the PPE+Inventory
sum is set to zero where individual components are missing.

**Required inputs:** `ppent`, `invt`, prior-year
versions, `at`.

**Reference:** [LyandresSunZhang2008].

**Sign convention:** negative. A more direct measure of physical
investment than total asset growth.

### NI --- Net Share Issuance

**Definition:**

$$
\mathrm{NI}_{i,t} = \log\left(
   \frac{\mathrm{shrout}^{\mathrm{adj}}_{i,t}}
        {\mathrm{shrout}^{\mathrm{adj}}_{i,t-12}}
\right),
$$

where $\mathrm{shrout}^{\mathrm{adj}}_{i,t} = \mathrm{shrout}_{i,t}
\cdot \mathrm{cfacshr}_{i,t}$ is the split-adjusted share count. NI is
the 12-month log change in adjusted shares outstanding.

**Required inputs:** `shrout`, `cfacshr`.

**Reference:** [PontiffWoodgate2008,DanielTitman2006].

**Sign convention:** negative. Firms that issue shares
(positive NI) tend to underperform; firms that buy back shares
(negative NI) tend to outperform.

**Known issues:** NI is monthly (computed from monthly CRSP),
not annual like the other Investment-family members. We classify it
under Investment because it is an issuance/repurchase activity
indicator, not a market signal.

### OA --- Operating Accruals

**Definition:**

$$
\mathrm{OA}_{i,t} = \frac{(\Delta\mathrm{ACT}_{i,t} - \Delta\mathrm{CHE}_{i,t})
                          - (\Delta\mathrm{LCT}_{i,t} - \Delta\mathrm{DLC}_{i,t})
                          - \mathrm{DP}_{i,t}}
                          {\mathrm{AT}_{i,t-1}},
$$

where again $\Delta X_{i,t} = X_{i,t} - X_{i,t-1}$ and missing
components are treated as zero.

The numerator decomposes the change in non-cash working capital:

- $\Delta \mathrm{ACT} - \Delta \mathrm{CHE}$: change in non-cash
      current assets;

- $\Delta \mathrm{LCT} - \Delta \mathrm{DLC}$: change in
      non-debt current liabilities;

- $\mathrm{DP}$: depreciation expense.

**Required inputs:** `act`, `lct`, `che`,
`dlc`, `dp`, prior-year versions, `at`.

**Reference:** [Sloan1996].

**Sign convention:** negative. Sloan's original finding: high
accruals predict negative future returns because they reflect earnings
that are not yet realized in cash and tend to reverse.

### AC --- Total Accruals

**Definition:**

$$
\mathrm{AC}_{i,t} = \frac{\mathrm{NOA}_{i,t} \cdot \mathrm{AT}_{i,t-1}
                          - \mathrm{NOA}_{i,t-1} \cdot \mathrm{AT}_{i,t-2}}
                          {\mathrm{AT}_{i,t-1}},
$$

defined when both prior-period NOA and $\mathrm{AT}_{i,t-2}$ are
non-missing (audit fix 2 from Document 1, §(subsec:stage0-audit-fixes)).

The numerator is the dollar change in NOA from year $t-1$ to year $t$;
the denominator scales by lagged total assets to make the figure
comparable across firms.

**Required inputs:** NOA inputs (current and prior year),
prior-year and lag-2 `at`.

**Reference:** [RichardsonSloanSoliman2005].

**Sign convention:** negative. AC generalizes the operating
accruals concept to total firm-level accruals.

**Known issues:** Per audit fix 2, we do NOT impute zero for
missing NOA${}_{i,t-1}$. First-observation firms and firms with missing
`at_lag2` have NA AC and are dropped by the completeness filter.

## Momentum Family (8)

Momentum characteristics measure recent return patterns at various
horizons: short-term reversal (1 month), intermediate momentum (months
2--12 and 7--12), long-term reversal (months 13--36 and 13--60), price
position (52-week high), and volume anomalies (SUV). All are computed
from the CRSP monthly file and use *raw* returns rather than
excess returns, following the academic convention of
[JegadeeshTitman1993] and the FNW Appendix.

### r12_2 --- Standard Momentum

**Definition:**

$$
r_{12,2}^{\text{(i,t)}} = \prod_{s=t-12}^{t-2} (1 + r_{i,s}) - 1,
$$

the compound raw return from $t-12$ to $t-2$ inclusive (skipping the
most recent month $t-1$ to avoid contamination from short-term
reversal). 11-month formation window.

**Required inputs:** `ret_adj` (split-adjusted raw returns).

**Reference:** [JegadeeshTitman1993,FamaFrench1996].

**Sign convention:** positive. The canonical momentum signal:
recent winners continue winning at horizons of 6--12 months.

**Known issues:** Computation requires at least 8 of the 11 months
to be observed (`min_periods=8` in the rolling window) to
ensure adequate data; firms with insufficient history have NA r12_2.

### r2_1 --- Short-Term Reversal Signal

**Definition:**

$$
r_{2,1}^{\text{(i,t)}} = r_{i,t-1},
$$

the prior-month raw return.

**Required inputs:** `ret_adj`.

**Reference:** [Lehmann1990,JegadeeshTitman1993].

**Sign convention:** negative. High prior-month returns
predict low next-month returns (the short-term reversal effect).

### ST_REV --- Short-Term Reversal Factor

**Definition:**

$$
\mathrm{ST_REV}_{i,t} = -r_{2,1}^{\text{(i,t)}} = -r_{i,t-1}.
$$

ST_REV is just the negative of r2_1, recoded so the sign convention
matches the long-side direction.

**Sign convention:** positive (by construction).

**Note:** ST_REV and r2_1 are perfectly rank-correlated (Spearman
$=$ 1.0 with sign flip). They are kept as separate columns in the
panel as a CPZ convention. Stage 4 will see them as duplicates and
either method will handle the redundancy via its own mechanism (PCA
collapses; Ridge shrinks; etc.).

### r12_7 --- Intermediate Momentum

**Definition:**

$$
r_{12,7}^{\text{(i,t)}} = \prod_{s=t-12}^{t-7} (1 + r_{i,s}) - 1,
$$

the compound raw return from $t-12$ to $t-7$ inclusive. 6-month
formation window covering the older half of standard momentum.

**Required inputs:** `ret_adj`.

**Reference:** [NovyMarx2012].

**Sign convention:** positive. Novy-Marx (2012) shows that
intermediate momentum (months 7--12) carries the bulk of the standard
momentum signal, while months 2--6 carry weaker signal. r12_7
isolates this effect.

### r36_13 --- Long-Term Reversal

**Definition:**

$$
r_{36,13}^{\text{(i,t)}} = \prod_{s=t-36}^{t-13} (1 + r_{i,s}) - 1,
$$

24-month formation window covering months 13--36 prior to formation.

**Required inputs:** `ret_adj`.

**Reference:** [DeBondtThaler1985].

**Sign convention:** negative. The long-term reversal effect:
firms with strong returns 1--3 years ago tend to underperform going
forward.

### LT_Rev --- Very Long-Term Reversal

**Definition:**

$$
\mathrm{LT_Rev}_{i,t} = \prod_{s=t-60}^{t-13} (1 + r_{i,s}) - 1,
$$

48-month formation window covering months 13--60.

**Required inputs:** `ret_adj`.

**Reference:** [DeBondtThaler1985].

**Sign convention:** negative. Even longer-horizon reversal,
following the original DeBondt-Thaler 5-year horizon.

### Rel2High --- Price-to-52-Week-High

**Definition:**

$$
\mathrm{Rel2High}_{i,t} = \frac{\mathrm{prc}^{\mathrm{adj}}_{i,t}}
                                {\max_{s \in [t-12,  t-1]}
                                 \mathrm{prc}^{\mathrm{adj}}_{i,s}},
$$

where $\mathrm{prc}^{\mathrm{adj}}_{i,t} = |\mathrm{prc}_{i,t}|
/ \mathrm{cfacpr}_{i,t}$ is the split-adjusted price (audit fix 3 from
Document 1, §(subsec:stage0-audit-fixes)).

**Required inputs:** `prc`, `cfacpr`.

**Reference:** [GeorgeHwang2004].

**Sign convention:** positive. George and Hwang document that
prices near their 52-week highs predict positive future returns ---
a momentum-related signal that is somewhat orthogonal to standard
return-momentum.

**Known issues:** Per audit fix 3, raw prices are converted to
split-adjusted before the rolling maximum.

### SUV --- Standardized Unexplained Volume

**Definition:** For each firm-month, fit an AR(3) regression of
log volume on its three lags using the past 36 months of data:

$$
\log V_{i,s} = \alpha_i + \beta_1 \log V_{i,s-1} + \beta_2 \log V_{i,s-2}
              + \beta_3 \log V_{i,s-3} + \epsilon_{i,s},
$$

estimated over $s \in [t - 36, t - 1]$. Then SUV is the standardized
residual at month $t$:

$$
\mathrm{SUV}_{i,t} = \frac{\log V_{i,t} - \widehat{\log V}_{i,t}}
                          {\widehat{\sigma}_{\epsilon, i}},
$$

where $\widehat{\log V}_{i,t}$ is the AR(3) prediction at $t$ and
$\widehat{\sigma}_{\epsilon, i}$ is the regression standard error.

**Required inputs:** `vol`.

**Reference:** [Garfinkel2009], building on the
Lo-Wang volume literature.

**Sign convention:** positive (mild). SUV captures abnormal
trading volume; firms with high SUV (recent volume above their
typical level) tend to outperform in the short run.

**Known issues:** The minimum window is 12 months
(`min_len = 12`). Computationally expensive: each firm-month
fits an OLS, and the per-firm Python loop is the runtime bottleneck of
Stage 0 (\(\sim\)20 minutes for the full panel).

## Risk Family (5)

The risk family contains five characteristics computed from CRSP daily
data over a 252-trading-day rolling window: market beta, market beta
(duplicate naming), idiosyncratic volatility, residual variance, and
total variance. All five involve a daily-frequency OLS regression of
excess returns on market excess returns:

$$
r^{ex}_{i,d} = \alpha_{i,t} + \beta_{i,t} \cdot \mathrm{MKT}^{ex}_d + \epsilon_{i,d},
   d \in \mathcal{W}_{i,t},
$$

where $\mathcal{W}_{i,t}$ is the set of valid daily observations for
firm $i$ over the 252 trading days ending at month-end $t$. We require
at least 60 valid daily observations to compute the regression.

### Beta --- Market Beta (CAPM)

**Definition:**

$$
\mathrm{Beta}_{i,t} = \widehat{\beta}_{i,t}
   \text{from regression~(eq:risk-regression)}.
$$

**Required inputs:** CRSP daily `ret`, FF daily
`mktrf` and `rf`.

**Reference:** [Sharpe1964,FrazziniPedersen2014].

**Sign convention:** negative. The "betting against beta" anomaly
(Frazzini-Pedersen): low-beta stocks earn higher risk-adjusted returns
than high-beta stocks, the reverse of CAPM's prediction.

### MktBeta --- Market Beta (Duplicate)

**Definition:**

$$
\mathrm{MktBeta}_{i,t} = \widehat{\beta}_{i,t}.
$$

**Note:** MktBeta and Beta are identical by construction. Per
the v2 decision (DECISION_LOG entry 014), we keep both columns. They
will be perfect rank-duplicates in Stage 4 and Stage 4 methods will
handle the redundancy.

### IdioVol --- Idiosyncratic Volatility

**Definition:**

$$
\mathrm{IdioVol}_{i,t} = \sqrt{252} \cdot
\mathrm{std}\bigl(\widehat{\epsilon}_{i,d}\bigr),
   d \in \mathcal{W}_{i,t},
$$

where $\widehat{\epsilon}_{i,d}$ are the residuals from
regression~(eq:risk-regression).

**Required inputs:** CRSP daily `ret`, FF daily
`mktrf`.

**Reference:** [AngHodrickXing2006].

**Sign convention:** negative. The "idiosyncratic volatility
puzzle" (Ang et al. 2006): high-IdioVol stocks have lower returns,
contradicting standard portfolio theory.

### Resid_Var --- Residual Variance

**Definition:**

$$
\mathrm{Resid_Var}_{i,t} = 252 \cdot \mathrm{var}\bigl(\widehat{\epsilon}_{i,d}\bigr).
$$

**Note:** $\mathrm{Resid_Var}_{i,t} = (\mathrm{IdioVol}_{i,t})^2$,
so this is a deterministic function of IdioVol; perfectly rank-correlated
with IdioVol after the squaring (which preserves rank since IdioVol is
non-negative).

**Reference:** Same as IdioVol.

**Sign convention:** negative.

**Note on duplicates:** As with MktBeta, Resid_Var is a perfect
rank-duplicate of IdioVol. Per v2 decision (DECISION_LOG entry 014),
we keep both.

### Variance --- Total Variance

**Definition:**

$$
\mathrm{Variance}_{i,t} = 252 \cdot \mathrm{var}\bigl(r_{i,d}\bigr),
   d \in \mathcal{W}_{i,t}.
$$

The total annualized return variance, ignoring the market regression.

**Required inputs:** CRSP daily `ret`.

**Reference:** [Goyal2003].

**Sign convention:** negative.

## Liquidity Family (3)

The liquidity family contains three characteristics measuring trading
ease or its inverse (cost): the Roll-implied bid-ask spread (described
in Document 1), log turnover (volume scaled by shares outstanding),
and log market equity (a proxy for both size and liquidity).

### Spread --- Roll (1984) Implied Spread

**Definition:** See Document 1, §(subsec:stage0-roll),
eq.~(eq:roll-impl).

**Required inputs:** CRSP daily `ret`.

**Reference:** [Roll1984].

**Sign convention:** positive. Higher-spread (illiquid) stocks
earn higher returns to compensate for trading costs (Amihud-Mendelson
1986).

### LTurnover --- Log Turnover

**Definition:**

$$
\mathrm{LTurnover}_{i,t} = \log\left(
   \frac{\mathrm{vol}_{i,t}}{\mathrm{shrout}_{i,t} \cdot 1000}
\right),
$$

where the $1000$ factor converts `shrout` from thousands to raw
share counts. The argument is monthly volume divided by shares
outstanding: turnover.

**Required inputs:** `vol`, `shrout`.

**Reference:** [LeeSwaminathan2000,Datar1998].

**Sign convention:** negative (mild). High-turnover stocks tend
to underperform; this is sometimes interpreted as a proxy for
disagreement or speculative interest.

### LME --- Log Market Equity

**Definition:**

$$
\mathrm{LME}_{i,t} = \log(\mathrm{ME}_{i,t-1}),
$$

the log of the previous month-end market equity. The lag ensures the
characteristic is observable as of the formation date $t$.

**Required inputs:** `prc`, `shrout` (for ME).

**Reference:** [Banz1981,FamaFrench1992].

**Sign convention:** negative. The size effect: smaller firms
earn higher returns. Substantially weaker post-1980 in the empirical
record.

## Other Family (11)

The "Other" family is a heterogeneous group of accounting
characteristics that do not fit cleanly into the value, profitability,
or investment families.

### Q --- Tobin's Q

**Definition:**

$$
Q_{i,t} = \frac{\mathrm{AT}_{i,t} + \mathrm{ME}_{i,t} - \mathrm{BE}_{i,t}}
                {\mathrm{AT}_{i,t}},
$$

defined when $\mathrm{AT}_{i,t} > 0$ and $\mathrm{ME}_{i,t} > 0$. The
classic Tobin's Q approximation: total enterprise value divided by
replacement cost. Missing BE is treated as zero.

**Required inputs:** `at`, ME, BE.

**Reference:** [Tobin1969,LangStulz1994].

**Sign convention:** negative. High-Q (growth) firms tend to
underperform low-Q (value) firms; a value-related effect.

### C --- Cash-to-Assets

**Definition:**

$$
C_{i,t} = \frac{\mathrm{CHE}_{i,t}}{\mathrm{AT}_{i,t}},
$$

defined when $\mathrm{AT}_{i,t} > 0$. Missing CHE is treated as zero.

**Required inputs:** `che`, `at`.

**Reference:** [Palazzo2012].

**Sign convention:** positive (mild). High cash holdings can
indicate financial flexibility or, alternatively, agency costs from
inefficient capital allocation; the sign is empirically debated.

### CF --- Cash Flow Yield

**Definition:**

$$
\mathrm{CF}_{i,t} = \frac{\mathrm{IB}_{i,t} + \mathrm{DP}_{i,t}}
                          {\mathrm{ME}_{i,t}}.
$$

**Note:** CF and CF2P are identical by construction. They are kept
as separate columns per CPZ convention; both will appear in Stage 4
inputs and downstream methods will handle the duplicate.

**Required inputs:** `ib`, `dp`, ME.

**Reference:** Same as CF2P ([Lakonishok1994]).

**Sign convention:** positive.

### AT --- Log Total Assets

**Definition:**

$$
\mathrm{AT}_{i,t} = \log(\mathrm{at}_{i,t}),
$$

defined when $\mathrm{at}_{i,t} > 0$. We clip $\mathrm{at}_{i,t}$ to
$10^{-6}$ to handle rare zero or negative entries.

**Required inputs:** `at`.

**Reference:** [FreybergerNeuhierlWeber2020].

**Sign convention:** negative. AT is a size proxy similar to LME
but using book size rather than market size.

### ATO --- Asset Turnover

**Definition:**

$$
\mathrm{ATO}_{i,t} = \frac{\mathrm{SALE}_{i,t}}{\mathrm{AT}_{i,t-1}},
$$

defined when $\mathrm{AT}_{i,t-1} > 0$.

**Required inputs:** `sale`, prior-year `at`.

**Reference:** [SoliFridson1995].

**Sign convention:** positive. ATO measures revenue-generating
efficiency per dollar of assets.

### CTO --- Capital Turnover

**Definition:**

$$
\mathrm{CTO}_{i,t} = \frac{\mathrm{SALE}_{i,t}}{\mathrm{PPENT}_{i,t}},
$$

defined when $\mathrm{PPENT}_{i,t} > 0.01$. Capital turnover: sales
per dollar of net property, plant, and equipment.

**Required inputs:** `sale`, `ppent`.

**Reference:** [Haugen1996].

**Sign convention:** positive (mild).

### D2A --- Debt-to-Assets

**Definition:**

$$
\mathrm{D2A}_{i,t} = \frac{\mathrm{DLTT}_{i,t} + \mathrm{DLC}_{i,t}}
                          {\mathrm{AT}_{i,t}},
$$

defined when $\mathrm{AT}_{i,t} > 0$. Missing DLTT or DLC is treated as
zero.

**Required inputs:** `dltt`, `dlc`, `at`.

**Reference:** [Bhandari1988,FamaFrench1992].

**Sign convention:** positive (mild). Higher-leverage firms earn
slightly higher returns to compensate for financial risk; the effect
is small after controlling for other factors.

### FC2Y --- Fixed Cost to Year-Total Assets (Operating Leverage Proxy)

**Definition:**

$$
\mathrm{FC2Y}_{i,t} = \frac{\mathrm{COGS}_{i,t} + \mathrm{XSGA}_{i,t}}
                            {\mathrm{AT}_{i,t}},
$$

defined when $\mathrm{AT}_{i,t} > 0$. Missing COGS or XSGA is treated
as zero.

**Required inputs:** `cogs`, `xsga`, `at`.

**Reference:** [NovyMarx2011].

**Sign convention:** positive. Higher operating leverage implies
higher sensitivity to economic conditions and a corresponding return
premium.

### Lev --- Market Leverage

**Definition:**

$$
\mathrm{Lev}_{i,t} = \frac{\mathrm{DLTT}_{i,t} + \mathrm{DLC}_{i,t}}
                          {\mathrm{DLTT}_{i,t} + \mathrm{DLC}_{i,t} + \mathrm{ME}_{i,t}},
$$

defined when the denominator is positive. Market leverage: debt over
debt plus market equity.

**Required inputs:** `dltt`, `dlc`, ME.

**Reference:** [FamaFrench1992].

**Sign convention:** positive.

### OL --- Operating Leverage (Same Formula as FC2Y)

**Definition:**

$$
\mathrm{OL}_{i,t} = \mathrm{FC2Y}_{i,t}.
$$

**Note:** OL and FC2Y are identical by construction. Like the
other duplicates (MktBeta/Beta, Resid_Var/IdioVol, CF/CF2P, ST_REV/r2_1
sign-flipped), both columns are retained per CPZ convention.

**Sign convention:** positive (same as FC2Y).

### SGA2S --- SG&A to Sales

**Definition:**

$$
\mathrm{SGA2S}_{i,t} = \frac{\mathrm{XSGA}_{i,t}}{\mathrm{SALE}_{i,t}},
$$

defined when $|\mathrm{SALE}_{i,t}| > 0.01$. Missing XSGA is treated
as zero.

**Required inputs:** `xsga`, `sale`.

**Reference:** [ChenZhang2010].

**Sign convention:** negative. High SG&A intensity may indicate
operational inefficiency.

## Summary Table

Table~(tab:char-summary) consolidates all 46 characteristics with
their family, formula reference (equation in this document), academic
source, and predicted sign.

| **Characteristic** | **Family** | **Reference** | **Eq.** | **Sign** |
|---|---|---|---|---|
| BEME | Value | FamaFrench1992 | (eq:beme) | + |
| E2P | Value | Basu1977 | (eq:e2p) | + |
| CF2P | Value | Lakonishok1994 | (eq:cf2p) | + |
| D2P | Value | LitzenbergerRamaswamy1979 | (eq:d2p) | + |
| S2P | Value | Barbee1996 | (eq:s2p) | + |
| A2ME | Value | BharyMcLean2014 | (eq:a2me) | + |
| PROF | Profitability | NovyMarx2013 | (eq:prof) | + |
| ROE | Profitability | HouXueZhang2015 | (eq:roe) | + |
| ROA | Profitability | BalakrishnanBartovFaurel2010 | (eq:roa) | + |
| OP | Profitability | FamaFrench2015 | (eq:op) | + |
| PM | Profitability | SoliFridson1995 | (eq:pm) | + |
| PCM | Profitability | Gorodnichenko2010 | (eq:pcm) | + |
| RNA | Profitability | SoliFridson1995 | (eq:rna) | + |
| Investment | Investment | CooperGulenSchill2008 | (eq:investment) | --- |
| NOA | Investment | HirshleiferHouTeoh2004 | (eq:noa-ratio) | --- |
| DPI2A | Investment | LyandresSunZhang2008 | (eq:dpi2a) | --- |
| NI | Investment | PontiffWoodgate2008 | (eq:ni) | --- |
| OA | Investment | Sloan1996 | (eq:oa) | --- |
| AC | Investment | RichardsonSloanSoliman2005 | (eq:ac) | --- |
| r12_2 | Momentum | JegadeeshTitman1993 | (eq:r12-2) | + |
| r2_1 | Momentum | Lehmann1990 | (eq:r2-1) | --- |
| ST_REV | Momentum | Lehmann1990 | (eq:st-rev) | + |
| r12_7 | Momentum | NovyMarx2012 | (eq:r12-7) | + |
| r36_13 | Momentum | DeBondtThaler1985 | (eq:r36-13) | --- |
| LT_Rev | Momentum | DeBondtThaler1985 | (eq:lt-rev) | --- |
| Rel2High | Momentum | GeorgeHwang2004 | (eq:rel2high) | + |
| SUV | Momentum | Garfinkel2009 | (eq:suv) | + |
| Beta | Risk | FrazziniPedersen2014 | (eq:beta) | --- |
| MktBeta | Risk | FrazziniPedersen2014 | (eq:mktbeta) | --- |
| IdioVol | Risk | AngHodrickXing2006 | (eq:idiovol) | --- |
| Resid_Var | Risk | AngHodrickXing2006 | (eq:resid-var) | --- |
| Variance | Risk | Goyal2003 | (eq:variance) | --- |
| Spread | Liquidity | Roll1984 | (eq:roll-impl) | + |
| LTurnover | Liquidity | LeeSwaminathan2000 | (eq:lturnover) | --- |
| LME | Liquidity | Banz1981 | (eq:lme) | --- |
| Q | Other | Tobin1969 | (eq:tobins-q) | --- |
| C | Other | Palazzo2012 | (eq:c) | + |
| CF | Other | Lakonishok1994 | (eq:cf) | + |
| AT | Other | FreybergerNeuhierlWeber2020 | (eq:log-at) | --- |
| ATO | Other | SoliFridson1995 | (eq:ato) | + |
| CTO | Other | Haugen1996 | (eq:cto) | + |
| D2A | Other | FamaFrench1992 | (eq:d2a) | + |
| FC2Y | Other | NovyMarx2011 | (eq:fc2y) | + |
| Lev | Other | FamaFrench1992 | (eq:lev) | + |
| OL | Other | NovyMarx2011 | (eq:ol) | + |
| SGA2S | Other | ChenZhang2010 | (eq:sga2s) | --- |

*Table: All 46 characteristics with family, primary academic source,
equation number in this document, and predicted sign of forward-return
correlation. Sign of "+" means high values predict positive returns;
"---" means high values predict negative returns.*

## Known Rank-Duplicate Pairs

Five pairs of characteristics in the panel are perfectly rank-correlated
by construction. Per the v2 design (DECISION_LOG entry 014), all 46
columns are retained; Stage 4 methods handle the duplication.

| **Pair** | **Relationship** | **Notes** |
|---|---|---|
| MktBeta, Beta | Identical | Same regression slope, named both ways |
| Resid_Var, IdioVol | $\mathrm{Resid_Var} = \mathrm{IdioVol}^2$ | Square preserves rank since IdioVol $\ge 0$ |
| CF, CF2P | Identical | Same numerator and denominator |
| OL, FC2Y | Identical | Same formula, two names per CPZ |
| ST_REV, r2_1 | Sign-flipped | ST_REV $= -\mathrm{r2_1}$;
   Spearman $=-1$ |

*Table: Five rank-duplicate pairs in the 46-characteristic panel.
All retained per v2 decision; Stage 4 methods handle duplication.*

## Document 2 Conclusion

Document 2 has specified the construction formulas for all 46 firm
characteristics in the panel, organized by family with academic
references and predicted signs. Equations are numbered for cross-reference
from Stage 0's implementation code (each formula corresponds to a
function in `src/data_reconstruction/characteristics/`).

The characteristics divide into five rank-duplicate pairs by
construction (Table~(tab:duplicates)), reflecting CPZ's convention
of carrying multiple names for related quantities. We retain all
duplicates per the v2 design decision (DECISION_LOG entry 014).

Document 3 (forthcoming) details the validation methodology that will
verify our reconstruction matches CPZ on the 1967--2016 overlap and
matches Fama-French published factors on 2017--2024.
