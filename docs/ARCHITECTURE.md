# Stage 0 Modular Architecture Design

> **Status**: DRAFT — pending Amos review and approval
> **Date**: 2026-05-09
> **Depends on**: Documents 1, 2, 3 (all approved)
> **References**: ROADMAP.txt v2 §4.2, §4.4, §4.5; DECISION_LOG entries 001, 002, 014, 015, 017-020

---

## 1. Design Principles

1. **Math-to-code traceability**: Every function has a corresponding equation number from Documents 1-3. The implementation is a translation, not a reinterpretation.
2. **Submodule independence**: Each submodule (`raw_data/`, `characteristics/`, `pit/`, `validation/`) is independently importable and testable. `pipeline.py` only orchestrates; it contains no construction logic.
3. **Fail-fast validation**: The validation suite runs as a hard gate. `stage0_complete = False` until all three arms pass. No downstream stage consumes an unvalidated panel.
4. **Idempotency**: Re-running `pipeline.py` with the same config and raw data produces bitwise-identical outputs (deterministic, no randomness).
5. **Memory-conscious streaming**: The full raw panel is large (~1.6M rows × 60+ columns). Where possible, operations use chunked/grouped processing rather than full-materialization.

---

## 2. Module Layout

```
src/data_reconstruction/
├── __init__.py                     # Package version, public re-exports
├── pipeline.py                     # Orchestrator: end-to-end panel build
│
├── raw_data/
│   ├── __init__.py
│   ├── crsp_loader.py              # CRSP MSF + DSF from WRDS
│   ├── compustat_loader.py         # Compustat funda + CCM link
│   ├── delisting.py                # Shumway 1997 imputation
│   ├── fama_french.py              # FF factors from WRDS
│   └── cache.py                    # Versioned parquet cache with hash keys
│
├── characteristics/
│   ├── __init__.py                 # Public: compute_all_characteristics()
│   ├── _registry.py                # Maps char name → (family, func, inputs, eq_ref)
│   ├── value.py                    # 6 chars: BEME, E2P, CF2P, D2P, S2P, A2ME
│   ├── profitability.py            # 7 chars: PROF, ROE, ROA, OP, PM, PCM, RNA
│   ├── investment.py               # 6 chars: Investment, NOA, DPI2A, NI, OA, AC
│   ├── momentum.py                 # 8 chars: r12_2, r2_1, ST_REV, r12_7, r36_13, LT_Rev, Rel2High, SUV
│   ├── risk.py                     # 5 chars: Beta, MktBeta, IdioVol, Resid_Var, Variance
│   ├── liquidity.py                # 3 chars: Spread, LTurnover, LME
│   └── other.py                    # 11 chars: Q, C, CF, AT, ATO, CTO, D2A, FC2Y, Lev, OL, SGA2S
│
├── pit/
│   ├── __init__.py
│   ├── merge_asof.py               # Pandas merge_asof with 380-day tolerance
│   └── audit.py                    # PIT bounds, look-ahead, coverage, rank, schema audits
│
├── validation/
│   ├── __init__.py
│   ├── cpz_comparison.py           # Arm 1: per-char, per-year, per-firm CPZ comparison
│   ├── fama_french.py              # Arm 2: FF5+UMD construction from our data
│   ├── ken_french_loader.py        # Pull KF published series from WRDS
│   ├── audits.py                   # Arm 3: internal consistency audits
│   ├── diagnostics.py              # Plotting: breadth charts, factor return series, coverage
│   └── _acceptance.py              # Pre-registered thresholds + pass/fail logic
│
└── utils/
    ├── __init__.py
    ├── logging.py                  # Structured logging (per-module loggers)
    ├── constants.py                # Magic numbers with equation references
    └── schema.py                   # Panel schema dataclass + validators
```

---

## 3. Public APIs Per Submodule

### 3.1 `raw_data/`

#### `crsp_loader.py`

```python
def load_crsp_monthly(
    wrds_conn,
    start_date: str = "1960-01-01",
    end_date: str = "2024-12-31",
    share_codes: tuple[int, ...] = (10, 11),
    exchange_codes: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Pull CRSP Monthly Stock File.

    Returns DataFrame with columns:
    permno, date, ret, prc, shrout, vol, cfacshr, cfacpr,
    shrcd, exchcd, siccd, ticker, comnam, dlret, dlstcd
    """

def load_crsp_daily(
    wrds_conn,
    start_date: str = "1963-01-01",
    end_date: str = "2024-12-31",
    share_codes: tuple[int, ...] = (10, 11),
) -> pd.DataFrame:
    """Pull CRSP Daily Stock File. Returns permno, date, ret."""

def compute_market_equity(crsp_monthly: pd.DataFrame) -> pd.Series:
    """ME = |prc| * shrout * 1000.  Doc 1, Eq. 1."""

def split_adjusted_price(crsp_monthly: pd.DataFrame) -> pd.Series:
    """prc_adj = |prc| / cfacpr.  Doc 1, Eq. 7 (audit fix 3)."""
```

#### `compustat_loader.py`

```python
def load_compustat_annual(
    wrds_conn,
    start_date: str = "1959-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Pull Compustat funda with primary-link CCM filter.
    Returns all fields needed by characteristic families."""

def compute_book_equity(compustat: pd.DataFrame) -> pd.Series:
    """Davis-Fama-French book equity hierarchy. Doc 1, Eq. 4a-4c.
    Returns BE series with BE > 0 only (negative BE → NA).
    """

def compute_availability_dates(compustat: pd.DataFrame) -> pd.Series:
    """avail_date = datadate + 6 months, month-end. Doc 1, Eq. 2."""
```

#### `delisting.py`

```python
def impute_delisting_returns(
    crsp_monthly: pd.DataFrame,
    perf_codes: set[int] = None,  # default: {500} ∪ [520, 584]
    nyse_amex_imp: float = -0.30,
    nasdaq_imp: float = -0.55,
) -> pd.DataFrame:
    """Shumway 1997 delisting imputation. Doc 1, Eq. 3.
    Returns crsp_monthly with 'ret' column updated and 'ret_excess'
    computed as ret - rf.
    """
```

#### `fama_french.py`

```python
def load_ff_factors(
    wrds_conn,
    frequency: Literal["monthly", "daily"] = "monthly",
) -> pd.DataFrame:
    """Pull FF factors (mktrf, smb, hml, rmw, cma, umd, rf) from WRDS."""

def compute_excess_returns(
    crsp_monthly: pd.DataFrame,
    rf: pd.Series,
) -> pd.DataFrame:
    """ret_excess = ret - rf.  Doc 1, Eq. 3b."""
```

#### `cache.py`

```python
class VersionedCache:
    """Disk cache for raw data pulls with versioned keys.

    Key format: {dataset_name}_{start}_{end}_{content_hash}.parquet
    Content hash is computed from SQL query text + config dict.
    """

    def __init__(self, cache_dir: Path):
        ...

    def get(self, key: str) -> pd.DataFrame | None:
        """Return cached DataFrame if key exists, else None."""

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Write DataFrame to cache."""

    def invalidate(self, pattern: str) -> int:
        """Remove cached files matching pattern. Returns count removed."""
```

---

### 3.2 `characteristics/`

#### `_registry.py` — The Single Source of Truth

```python
@dataclass
class CharacteristicSpec:
    name: str                       # Column name in output panel
    family: str                     # One of 7 families
    func: Callable                  # The computation function
    inputs: list[str]               # Required input column names
    equation_ref: str               # LaTeX equation label (e.g., "eq:beme")
    sign_convention: Literal["+", "-"]  # Predicted return correlation
    is_duplicate: bool              # Is this a known rank-duplicate?
    duplicate_of: str | None        # If duplicate, which char?

REGISTRY: dict[str, CharacteristicSpec] = {
    # Populated at module import from all family modules
    "BEME": CharacteristicSpec(...),
    "E2P": CharacteristicSpec(...),
    # ... all 46
}

# Known duplicate pairs (Doc 2, Table 4)
DUPLICATE_PAIRS: list[tuple[str, str, str]] = [
    ("MktBeta", "Beta", "identical"),
    ("Resid_Var", "IdioVol", "squared"),
    ("CF", "CF2P", "identical"),
    ("OL", "FC2Y", "identical"),
    ("ST_REV", "r2_1", "sign_flip"),
]
```

#### Family modules (uniform interface)

Each family module exports a single public function:

```python
# value.py, profitability.py, investment.py, other.py
def compute_<family>(
    monthly_panel: pd.DataFrame,    # CRSP monthly with ME, prc, shrout
    compustat: pd.DataFrame,        # Compustat with BE and all BS/IS items
    be: pd.Series,                  # Pre-computed BE (Doc 1, Eq. 4)
) -> pd.DataFrame:
    """Compute all characteristics in this family.

    Parameters
    ----------
    monthly_panel : pd.DataFrame
        Must contain: permno, date, me, prc, shrout, cfacpr, ret
    compustat : pd.DataFrame
        Must contain all fields listed in Doc 2 for this family.
    be : pd.Series
        Book equity, indexed by (permno, datadate). NA for BE ≤ 0.

    Returns
    -------
    pd.DataFrame with columns: permno, date, {char1}, {char2}, ...
    One row per (permno, date) in monthly_panel. NA where inputs missing.
    """

# momentum.py (special: needs raw returns, not compustat)
def compute_momentum(
    monthly_panel: pd.DataFrame,    # CRSP monthly with ret, prc, cfacpr, vol
) -> pd.DataFrame:
    """Compute all 8 momentum-family characteristics.
    SUV (Doc 2, Eq. 38-39) fits AR(3) per firm; this is the runtime bottleneck.
    """

# risk.py (special: needs daily data)
def compute_risk(
    daily_returns: pd.DataFrame,    # CRSP daily: permno, date, ret
    market_daily: pd.Series,        # FF daily mktrf, indexed by date
    monthly_dates: pd.DatetimeIndex, # Month-ends to compute for
) -> pd.DataFrame:
    """Compute all 5 risk-family characteristics from 252-day rolling OLS.
    Doc 2, Eq. 40-44. Requires ≥ 60 valid daily observations per window.
    """

# liquidity.py
def compute_liquidity(
    monthly_panel: pd.DataFrame,    # CRSP monthly: permno, date, vol, shrout, me
    daily_returns: pd.DataFrame,    # CRSP daily: permno, date, ret (for Roll spread)
) -> pd.DataFrame:
    """Compute all 3 liquidity-family characteristics.
    Spread uses Roll 1984 estimator (Doc 1, Eq. 6). Requires ≥ 10 daily obs.
    """
```

#### `__init__.py` — Orchestrator for all characteristics

```python
def compute_all_characteristics(
    crsp_monthly: pd.DataFrame,
    crsp_daily: pd.DataFrame,
    compustat: pd.DataFrame,
    be: pd.Series,
    ff_daily: pd.Series,            # mktrf for risk family
) -> pd.DataFrame:
    """Compute all 46 characteristics, merge onto monthly panel.

    Returns DataFrame with columns: permno, date, {46 char columns}.
    Each family module is called independently; results are merged on
    (permno, date) via outer join. NA preserved (not filled).
    """
```

---

### 3.3 `pit/`

#### `merge_asof.py`

```python
def pit_merge(
    monthly_panel: pd.DataFrame,     # CRSP: permno, date (month-end)
    compustat: pd.DataFrame,         # With avail_date column
    tolerance_days: int = 380,       # Audit fix 1 (was 548)
) -> pd.DataFrame:
    """Point-in-time merge: attach most recent Compustat record to each
    firm-month where avail_date ≤ panel date ≤ avail_date + tolerance.

    Uses pandas.merge_asof with backward direction.
    Returns panel with accounting columns attached; NA where no match.
    Doc 1, §2.4.
    """
```

#### `audit.py`

```python
def audit_pit_bounds(
    panel: pd.DataFrame,
    tolerance_days: int = 380,
) -> pd.DataFrame:
    """Verify for every row: datadate + 6mo ≤ date < datadate + tolerance.
    Hard-assert: zero violations. Doc 1, Eq. 2 + Eq. 8."""

def audit_lookahead_monthly(
    panel: pd.DataFrame,
) -> dict[str, bool]:
    """Verify monthly characteristics use only past information.
    Checks: momentum skip ≥ 1 month, risk 252-day window ends at t,
    LME uses ME(t-1), Spread uses daily obs within month t.
    Returns {char_name: pass_bool}."""

def audit_coverage_stability(
    panel: pd.DataFrame,
    yoy_threshold: float = 0.10,
) -> tuple[pd.Series, list[str]]:
    """Return monthly firm count and list of flagged months where
    |YoY change| > threshold. Doc 3, Audit 3.3."""

def audit_rank_integrity(
    panel: pd.DataFrame,
    char_cols: list[str],
) -> dict[str, int]:
    """Verify rank-normalized values in (-1/2, 1/2) for all chars.
    Returns {char_name: n_violations}. Doc 3, Audit 3.4."""

def audit_schema(
    panel: pd.DataFrame,
) -> dict[str, bool]:
    """Verify: no duplicate (permno, date), correct dtypes, required
    columns present, dates monotonic within permno. Doc 3, Audit 3.5."""

def run_all_audits(
    panel: pd.DataFrame,
    char_cols: list[str],
) -> dict[str, dict]:
    """Run all Arm 3 audits. Returns structured report with pass/fail
    per audit. Hard-fail on PIT or schema violations."""
```

---

### 3.4 `validation/`

#### `cpz_comparison.py` — Arm 1

```python
def compare_to_cpz(
    ours: pd.DataFrame,              # Our reconstructed panel
    cpz: pd.DataFrame,               # Published CPZ panel (1967-2016)
    output_dir: Path,
) -> dict:
    """Arm 1: CPZ overlap validation.

    Computes per (year, characteristic):
      - MAD (Eq. M1, M2 in Doc 3)
      - Spearman rank correlation (Eq. M3)
      - Overlap counts

    Asserts acceptance criteria (Doc 3, §3.1.3):
      1. ≥ 49/50 years with median_f MAD < 0.005
      2. ≥ 44/46 chars with full-period MAD < 1e-4
      3. ≥ 95% of (year, char) cells with Spearman > 0.95

    Returns summary dict + writes cpz_validation.parquet + summary JSON.
    """
```

#### `fama_french.py` — Arm 2

```python
def construct_ff5_umd(
    panel: pd.DataFrame,
    char_col_map: dict[str, str],    # {'BEME': 'BEME', 'OP': 'OP', ...}
) -> pd.DataFrame:
    """Construct FF5 + UMD factor returns from our panel using NYSE
    breakpoint 2×3 sorts. Doc 3, §3.2.

    Returns monthly factor returns: MKT, SMB, HML, RMW, CMA, UMD.
    """

def compare_to_ken_french(
    ours_factors: pd.DataFrame,
    kf_factors: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Arm 2: Compare our factor returns to Kenneth French published series.

    Computes per factor:
      - Pearson correlation (Doc 3, Eq. 12)
      - Monthly MAD (Doc 3, Eq. 13)
      - Annualized divergence (Doc 3, Eq. 14)
      - Per-year correlation (for GameStop detection)

    Asserts acceptance criteria (Doc 3, §3.2.4):
      1. ρ > 0.85 for all 6 factors
      2. MAD < 0.01/month for all factors
      3. |annualized divergence| < 3%
      4. Per-year correlation > 0.80 (with GameStop caveat)

    Returns summary dict + writes ff_validation.parquet + summary JSON.
    """
```

#### `ken_french_loader.py`

```python
def load_ken_french_monthly(
    wrds_conn,
    factors: list[str] = ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"],
) -> pd.DataFrame:
    """Pull published FF monthly factors from WRDS ff.factors_monthly."""
```

#### `audits.py` — Arm 3

```python
# Re-exports from pit.audit for convenience
def run_all_audits(panel: pd.DataFrame, char_cols: list[str]) -> dict:
    """Run all Arm 3 audits (PIT, look-ahead, coverage, rank, schema).
    Returns structured report. Hard-fail on PIT or schema."""
```

#### `_acceptance.py` — Threshold Definitions

```python
# Pre-registered acceptance targets (Doc 3, §1.1 pre-registration note)
# These are TARGETS, not results. Results are measured at runtime.

CPZ_YEARLY_BREADTH_YEARS = 49           # of 50 years
CPZ_YEARLY_BREADTH_MAD = 0.005          # median across chars
CPZ_FULL_PERIOD_CHARS = 44              # of 46 characteristics
CPZ_FULL_PERIOD_MAD = 1e-4              # absolute
CPZ_SPEARMAN_CELLS = 0.95               # 95% of (year, char) cells
CPZ_SPEARMAN_THRESHOLD = 0.95

FF_PEARSON_MIN = 0.85
FF_MAD_MAX = 0.01
FF_ANNUAL_DIV_MAX = 0.03
FF_PER_YEAR_PEARSON_MIN = 0.80          # with GameStop caveat
```

---

### 3.5 `pipeline.py` — Orchestrator

```python
class Stage0Pipeline:
    """End-to-end Stage 0 panel builder.

    Usage:
        pipeline = Stage0Pipeline.from_yaml("configs/data_reconstruction.yaml")
        result = pipeline.run()  # Returns Stage0Result
        assert result.stage0_complete
    """

    def __init__(self, config: Stage0Config):
        ...

    @classmethod
    def from_yaml(cls, path: str | Path) -> Stage0Pipeline:
        """Load config from YAML and validate."""

    def run(self) -> Stage0Result:
        """Execute full pipeline in order:

        1. Load raw data (CRSP, Compustat, FF) — from WRDS or cache
        2. Compute book equity (Doc 1, Eq. 4)
        3. Apply Shumway delisting (Doc 1, Eq. 3)
        4. Compute excess returns (Doc 1, Eq. 3b)
        5. PIT merge: Compustat onto CRSP monthly (380-day tolerance)
        6. Compute all 46 characteristics (Doc 2)
        7. Apply completeness filter (Doc 1, Eq. 9)
        8. Rank-normalize (Doc 1, Eq. 5)
        9. Run validation suite (Doc 3)
        10. Write output panel + diagnostics

        Returns Stage0Result with paths, summary stats, validation reports.
        """

@dataclass
class Stage0Result:
    """Immutable result of Stage 0 pipeline run."""
    panel_path: Path                    # factor_panel_v2.parquet
    raw_panel_path: Path                # pre-PIT raw_panel.parquet
    cpz_validation_path: Path | None    # None if CPZ panel unavailable
    ff_validation_path: Path
    audit_report: dict
    stage0_complete: bool
    validation_summary: dict            # Per-arm pass/fail status
    n_rows: int
    n_firms: int
    date_range: tuple[str, str]
    characteristics: list[str]          # 46 column names
```

---

## 4. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 0 PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
  │  WRDS /      │    │  WRDS /      │    │  WRDS /          │
  │  Cache       │    │  Cache       │    │  Cache           │
  │              │    │              │    │                  │
  │ CRSP Monthly │    │ CRSP Daily   │    │ Compustat Annual │
  │ (1960-2024)  │    │ (1963-2024)  │    │ (1959-2024)      │
  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘
         │                   │                     │
         ▼                   ▼                     ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    raw_data/ MODULE                       │
  │  • crsp_loader: load, filter, compute ME, split-adj price│
  │  • compustat_loader: load, primary-link filter, compute BE│
  │  • delisting: Shumway imputation, excess returns         │
  │  • fama_french: load FF factors                            │
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                     pit/ MODULE                           │
  │  • merge_asof: PIT merge Compustat → CRSP (380-day tol)  │
  │  • audit: PIT bounds assertion (hard fail on violation)  │
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼ (monthly panel with accounting data)
  ┌──────────────────────────────────────────────────────────┐
  │              characteristics/ MODULE                      │
  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐ │
  │  │ value.py    │ │ profitability│ │ investment.py      │ │
  │  │ (6 chars)   │ │ .py (7 chars)│ │ (6 chars)          │ │
  │  └─────────────┘ └──────────────┘ └────────────────────┘ │
  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐ │
  │  │ momentum.py │ │ risk.py      │ │ liquidity.py       │ │
  │  │ (8 chars)   │ │ (5 chars)    │ │ (3 chars)          │ │
  │  └─────────────┘ └──────────────┘ └────────────────────┘ │
  │  ┌─────────────────────────────────────────────────────┐ │
  │  │ other.py (11 chars)                                  │ │
  │  └─────────────────────────────────────────────────────┘ │
  │  Registry: 46 CharacteristicSpec objects, 5 duplicate pairs│
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼ (46 raw characteristic columns)
  ┌──────────────────────────────────────────────────────────┐
  │              RANK-NORMALIZATION (pipeline.py)             │
  │  z_{i,t,f} = R_{i,t,f} / (N_t + 1) - 0.5                │
  │  Doc 1, Eq. 5 — applied per characteristic, per month    │
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │            COMPLETENESS FILTER (pipeline.py)              │
  │  Retain firm-months where ALL 46 chars are non-missing   │
  │  Doc 1, Eq. 9 — applied BEFORE rank-normalization        │
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │              validation/ MODULE (THREE ARMS)              │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ │
  │  │ Arm 1: CPZ      │  │ Arm 2: Fama-    │  │ Arm 3:   │ │
  │  │     Comparison  │  │     French      │  │ Internal │ │
  │  │     (1967-2016) │  │     Replication │  │ Audits   │ │
  │  │  cpz_comparison │  │     (2017-2024) │  │ audits.py│ │
  │  │     .py         │  │  fama_french.py │  │          │ │
  │  └─────────────────┘  └─────────────────┘  └──────────┘ │
  │                                                          │
  │  Acceptance: All 3 arms pass → stage0_complete = True   │
  │  Hard fail: PIT violations, schema violations            │
  │  Soft fail: Proceed with documented caveat               │
  └────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                      OUTPUTS                               │
  │  factor_panel_v2.parquet    — Final panel (1.6M rows)     │
  │  raw_panel.parquet          — Pre-PIT panel               │
  │  cpz_validation.parquet     — Arm 1 diagnostics           │
  │  ff_validation.parquet      — Arm 2 diagnostics           │
  │  stage0_summary.json        — Validation pass/fail + stats│
  └──────────────────────────────────────────────────────────┘
```

---

## 5. Test Organization

```
tests/
├── conftest.py                     # Shared fixtures (WRDS mock, sample data)
├── test_data_reconstruction.py     # Top-level: pipeline integration test
├── raw_data/
│   ├── test_crsp_loader.py         # ME computation, price adjustment
│   ├── test_compustat_loader.py    # BE hierarchy, avail_date
│   ├── test_delisting.py           # Shumway imputation edge cases
│   └── test_cache.py               # VersionedCache get/put/invalidate
├── characteristics/
│   ├── test_value.py               # BEME, E2P, CF2P, D2P, S2P, A2ME
│   ├── test_profitability.py       # PROF, ROE, ROA, OP, PM, PCM, RNA
│   ├── test_investment.py          # Investment, NOA, DPI2A, NI, OA, AC
│   ├── test_momentum.py            # r12_2, r2_1, ST_REV, r12_7, r36_13, LT_Rev, Rel2High, SUV
│   ├── test_risk.py                # Beta, MktBeta, IdioVol, Resid_Var, Variance
│   ├── test_liquidity.py           # Spread, LTurnover, LME
│   ├── test_other.py               # Q, C, CF, AT, ATO, CTO, D2A, FC2Y, Lev, OL, SGA2S
│   └── test_registry.py            # All 46 registered, 5 duplicate pairs
├── pit/
│   ├── test_merge_asof.py          # PIT merge with tolerance
│   └── test_audit.py               # All 5 audit functions
└── validation/
    ├── test_cpz_comparison.py      # Arm 1 metrics computation
    ├── test_fama_french.py         # Arm 2 FF5+UMD construction
    └── test_acceptance.py          # Threshold logic, pass/fail classification
```

### Key Test Fixtures (conftest.py)

```python
@pytest.fixture
def sample_crsp_monthly() -> pd.DataFrame:
    """Synthetic CRSP monthly data for 10 firms × 24 months.
    Covers: normal case, delisting month, missing price, stock split."""

@pytest.fixture
def sample_compustat() -> pd.DataFrame:
    """Synthetic Compustat annual data matching the CRSP firms.
    Covers: full BE hierarchy, missing seq, fiscal year shifts."""

@pytest.fixture
def sample_daily() -> pd.DataFrame:
    """Synthetic CRSP daily returns for risk/liquidity families.
    Covers: sufficient obs, insufficient obs, positive autocov."""

@pytest.fixture
def mock_wrds() -> MagicMock:
    """Mock WRDS connection that returns sample data from fixtures."""

@pytest.fixture
def sample_panel() -> pd.DataFrame:
    """Pre-built panel with all 46 characteristics (from sample data).
    Used for validation tests without re-running full pipeline."""
```

### Critical Test Cases

| Module | Test | What it verifies |
|---|---|---|
| `test_delisting` | NASDAQ perf delisting gets -0.55 | Doc 1, Eq. 3 branch 3 |
| `test_delisting` | Missing dlret, non-perf code → 0 | Doc 1, Eq. 3 branch 4 |
| `test_compustat` | BE hierarchy: seq → ceq+pstk → at-lt | Doc 1, Eq. 4a |
| `test_compustat` | BE ≤ 0 → NA (not negative) | Doc 1, §2.3.3 |
| `test_merge_asof` | 380-day tolerance excludes stale data | Doc 1, §2.4.4 |
| `test_audit` | PIT bound violation → hard error | Doc 3, Audit 3.1 |
| `test_value` | BEME = BE / ME with PIT-merged BE | Doc 2, Eq. 10 |
| `test_investment` | AC returns NA (not 0) when lag NOA missing | Audit fix 2 |
| `test_momentum` | Rel2High uses prc / cfacpr | Audit fix 3, Doc 2, Eq. 34 |
| `test_momentum` | r12_2 skips month t-1 | Doc 2, Eq. 27 |
| `test_risk` | IdioVol = sqrt(252) × std(resid) | Doc 2, Eq. 42 |
| `test_liquidity` | Spread = 0 when autocov ≥ 0 | Doc 1, Eq. 6 |
| `test_registry` | Exactly 46 characteristics registered | Doc 2, §1.2 |
| `test_registry` | Exactly 5 duplicate pairs identified | Doc 2, §8 |

---

## 6. Interface Contracts

### Contract 1: Raw Data → PIT Merge
- **Input**: CRSP monthly (permno, date, ret, prc, shrout, vol, cfacpr, cfacshr, shrcd, exchcd, siccd) + Compustat (permno, datadate, avail_date, {all accounting fields})
- **Output**: Monthly panel with accounting columns PIT-merged
- **Invariant**: Every row satisfies `datadate + 6mo ≤ date < datadate + 380 days`

### Contract 2: PIT Panel → Characteristics
- **Input**: PIT-merged panel + daily returns + FF daily mktrf
- **Output**: 46 raw characteristic columns (unnormalized, pre-filter)
- **Invariant**: Each column has a corresponding `CharacteristicSpec` in the registry with equation reference

### Contract 3: Characteristics → Normalized Panel
- **Input**: 46 raw columns on monthly panel
- **Output**: Rank-normalized columns in (-1/2, 1/2), completeness filter applied
- **Invariant**: No (permno, date) has any NA among the 46 columns; rank values are strictly within the open interval

### Contract 4: Panel → Validation
- **Input**: Final normalized panel + CPZ reference (optional) + KF published series
- **Output**: Pass/fail status + diagnostic parquets + summary JSON
- **Invariant**: `stage0_complete = False` unless all arms pass (or soft-fail with documented caveat)

---

## 7. File Locations

| File | Path | Purpose |
|---|---|---|
| Config | `configs/data_reconstruction.yaml` | All locked parameters |
| Source | `src/data_reconstruction/` | Package root |
| Tests | `tests/test_data_reconstruction/` | Test suite |
| Notebook | `notebooks/00_data_foundation.ipynb` | Walkthrough |
| LaTeX | `reports/sections/00_data_foundation.tex` | Math documentation |
| Output panel | `data/foundation/factor_panel_v2.parquet` | Main artifact |
| CPZ validation | `data/foundation/cpz_validation.parquet` | Arm 1 diagnostics |
| FF validation | `data/foundation/ff_validation.parquet` | Arm 2 diagnostics |
| Summary | `data/foundation/stage0_summary.json` | Pass/fail + stats |
