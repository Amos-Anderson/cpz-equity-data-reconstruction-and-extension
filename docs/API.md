# Public API Reference

## Main Pipeline

### `Stage0Pipeline`

Main orchestrator for the data reconstruction pipeline.

```python
from data_construction import Stage0Pipeline

# Load from YAML configuration
pipeline = Stage0Pipeline.from_yaml("configs/stage00.yaml")

# Run the full pipeline
result = pipeline.run()
```

#### Methods

- `from_yaml(path: str | Path) -> Stage0Pipeline` — Load configuration from YAML file
- `run() -> Stage0Result` — Execute the full pipeline end-to-end

#### Returns

`Stage0Result` dataclass with:
- `panel_path: Path` — Path to final factor panel (parquet)
- `raw_panel_path: Path` — Path to pre-PIT panel
- `cpz_validation_path: Path | None` — Arm 1 validation results
- `ff_validation_path: Path` — Arm 2 validation results
- `audit_report: dict` — Arm 3 internal audit results
- `stage0_complete: bool` — Overall success flag
- `validation_summary: dict` — Per-arm pass/fail status
- `n_rows: int` — Number of rows in final panel
- `n_firms: int` — Number of unique firms
- `date_range: tuple[str, str]` — (start_date, end_date)
- `characteristics: list[str]` — 46 characteristic column names

---

## Raw Data Module

### `raw_data.crsp`

CRSP stock data loading and transformations.

```python
from data_construction.raw_data import crsp
```

**Functions:**

- `load_crsp_monthly(wrds_conn, start_date: str, end_date: str, ...) -> pd.DataFrame`
  - Columns: permno, date, ret, prc, shrout, vol, cfacpr, cfacshr, shrcd, exchcd, siccd, ticker, comnam, dlret, dlstcd
  
- `load_crsp_daily(wrds_conn, start_date: str, end_date: str, ...) -> pd.DataFrame`
  - Columns: permno, date, ret
  
- `compute_market_equity(crsp_monthly: pd.DataFrame) -> pd.Series`
  - Returns: Market equity (|prc| × shrout × 1000) — Doc 1, Eq. 1
  
- `split_adjusted_price(crsp_monthly: pd.DataFrame) -> pd.Series`
  - Returns: Adjusted price (|prc| / cfacpr) — Doc 1, Eq. 7

---

### `raw_data.accounting`

Compustat data loading and accounting transformations.

```python
from data_construction.raw_data import accounting
```

**Functions:**

- `load_compustat_annual(wrds_conn, start_date: str, end_date: str) -> pd.DataFrame`
  - Returns full Compustat with CCM primary-link filter
  
- `compute_book_equity(compustat: pd.DataFrame) -> pd.Series`
  - Davis-Fama-French BE hierarchy: seq → ceq+pstk → at-lt
  - Returns: Book equity (NaN for BE ≤ 0) — Doc 1, Eq. 4
  
- `compute_availability_dates(compustat: pd.DataFrame) -> pd.Series`
  - Returns: Availability date = datadate + 6 months — Doc 1, Eq. 2

---

### `raw_data.download`

WRDS connection and data pulling utilities.

```python
from data_construction.raw_data import download
```

**Functions:**

- `get_wrds_connection() -> wrds.Connection`
  - Establishes WRDS connection using credentials from .env or environment variables
  
- `load_ff_factors(wrds_conn, frequency: str = "monthly") -> pd.DataFrame`
  - Loads Fama-French factors (mktrf, smb, hml, rmw, cma, umd, rf)

---

### `raw_data.cache`

Versioned data caching system.

```python
from data_construction.raw_data.cache import VersionedCache
```

**Class: `VersionedCache`**

```python
cache = VersionedCache(Path("data/.cache"))

# Get cached data
df = cache.get("crsp_monthly_1960_2024_abc123def")

# Store data
cache.put("crsp_monthly_1960_2024_abc123def", df)

# Invalidate matching patterns
n_removed = cache.invalidate("crsp_monthly_*")
```

---

## Characteristics Module

### `characteristics`

Main entry point for characteristic computation.

```python
from data_construction.characteristics import compute_all_characteristics
```

**Functions:**

- `compute_all_characteristics(crsp_monthly, crsp_daily, compustat, be, ff_daily) -> pd.DataFrame`
  - Computes all 46 characteristics in one call
  - Returns: Monthly panel with 46 characteristic columns (raw, unnormalized)

---

### Family-Specific Modules

Each family has a `compute_<family>()` function:

```python
from data_construction.characteristics import value, profitability, investment, momentum, risk, liquidity, other

# Value family (6 chars)
value_chars = value.compute_value(monthly_panel, compustat, be)

# Profitability family (7 chars)
prof_chars = profitability.compute_profitability(monthly_panel, compustat, be)

# Investment family (6 chars)
inv_chars = investment.compute_investment(monthly_panel, compustat, be)

# Momentum family (8 chars) — special, needs raw returns only
mom_chars = momentum.compute_momentum(monthly_panel)

# Risk family (5 chars) — special, needs daily data
risk_chars = risk.compute_risk(daily_returns, market_daily, monthly_dates)

# Liquidity family (3 chars) — special, needs daily data
liq_chars = liquidity.compute_liquidity(monthly_panel, daily_returns)

# Other family (11 chars)
other_chars = other.compute_other(monthly_panel, compustat, be)
```

### `characteristics._registry`

Registry of all 46 characteristics with metadata.

```python
from data_construction.characteristics._registry import REGISTRY, DUPLICATE_PAIRS

# Access characteristic spec
spec = REGISTRY["BEME"]
print(spec.name, spec.family, spec.equation_ref)

# Check for duplicates
for char1, char2, relation in DUPLICATE_PAIRS:
    print(f"{char1} is {relation} {char2}")
```

---

## Point-in-Time Module

### `pit.merge`

Point-in-time merging of accounting data onto monthly panel.

```python
from data_construction.pit import merge
```

**Functions:**

- `pit_merge(monthly_panel, compustat, tolerance_days: int = 380) -> pd.DataFrame`
  - Attaches most recent Compustat record to each firm-month
  - Tolerance window: avail_date ≤ date < avail_date + 380 days
  - Returns: Panel with accounting columns merged

---

### `pit.audit`

Point-in-time and data quality audits.

```python
from data_construction.pit import audit
```

**Functions:**

- `audit_pit_bounds(panel, tolerance_days: int = 380) -> pd.DataFrame`
  - Verifies: datadate + 6mo ≤ date < datadate + tolerance
  - Hard-fails on violations
  
- `audit_lookahead_monthly(panel) -> dict[str, bool]`
  - Checks momentum skip, risk window, price adjustments
  - Returns: {char_name: pass_bool}
  
- `audit_coverage_stability(panel, yoy_threshold: float = 0.10) -> tuple[pd.Series, list]`
  - Year-over-year coverage change detection
  
- `audit_rank_integrity(panel, char_cols: list) -> dict[str, int]`
  - Verifies rank values in (-0.5, 0.5)
  
- `audit_schema(panel) -> dict[str, bool]`
  - Column presence, dtypes, no duplicates, monotonic dates
  
- `run_all_audits(panel, char_cols) -> dict`
  - Runs all audit functions and returns structured report

---

## Validation Module

### `validation.cpz` (Arm 1)

CPZ (Hou, Xue, Zhang) characteristics comparison.

```python
from data_construction.validation import cpz_comparison
```

**Functions:**

- `compare_to_cpz(ours: pd.DataFrame, cpz: pd.DataFrame, output_dir: Path) -> dict`
  - Compares our characteristics against published CPZ panel (1967-2016 overlap)
  - Metrics: MAD, Spearman correlation, overlap counts
  - Writes: cpz_validation.parquet, summary JSON
  - Returns: Summary dict with pass/fail status

---

### `validation.fama_french` (Arm 2)

Fama-French factor replication.

```python
from data_construction.validation import fama_french
```

**Functions:**

- `construct_ff5_umd(panel, char_col_map) -> pd.DataFrame`
  - Constructs FF5+UMD factor returns using 2×3 NYSE sorts
  - Returns: Monthly factor returns (MKT, SMB, HML, RMW, CMA, UMD)
  
- `compare_to_ken_french(ours_factors, kf_factors, output_dir) -> dict`
  - Compares factor returns against Kenneth French published series
  - Metrics: Pearson correlation, MAD, annualized divergence
  - Writes: ff_validation.parquet, summary JSON
  - Returns: Summary dict with pass/fail status

---

### `validation.audits` (Arm 3)

Internal consistency audits.

```python
from data_construction.validation import audits
```

**Functions:**

- `run_all_audits(panel, char_cols) -> dict`
  - Runs PIT bounds, look-ahead, coverage, rank, schema audits
  - Hard-fails on PIT or schema violations
  - Returns: Structured report dict

---

## Configuration

### `utils.config`

Configuration dataclasses and validation.

```python
from data_construction.utils.config import Stage0Config
```

**Class: `Stage0Config`**

```python
config = Stage0Config.from_yaml("configs/stage00.yaml")

# Access config parameters
config.date_range         # ("1960-01-01", "2024-12-31")
config.wrds_cache_dir     # Path to cache
config.output_dir         # Path to outputs
config.validation_thresholds  # Dict of acceptance criteria
```

---

## Constants and Utilities

### `utils.constants`

Magic numbers and equation references.

```python
from data_construction.utils.constants import *

# Example constants
DELISTING_IMP_NASDAQ  # -0.55
DELISTING_IMP_NYSE_AMEX  # -0.30
PIT_TOLERANCE_DAYS  # 380
RANK_LOWER_BOUND  # -0.5
RANK_UPPER_BOUND  # 0.5
```

---

## Example Workflows

### Minimal: Just Load Raw Data

```python
from data_construction.raw_data import crsp, accounting, download

conn = download.get_wrds_connection()
crsp_monthly = crsp.load_crsp_monthly(conn)
compustat = accounting.load_compustat_annual(conn)
```

### Intermediate: Compute Characteristics Only

```python
from data_construction.characteristics import compute_all_characteristics
from data_construction.raw_data import crsp, accounting, download

# Load data...
all_chars = compute_all_characteristics(crsp_monthly, crsp_daily, compustat, be, ff_daily)
```

### Full: End-to-End Pipeline with Validation

```python
from data_construction import Stage0Pipeline

pipeline = Stage0Pipeline.from_yaml("configs/stage00.yaml")
result = pipeline.run()

if result.stage0_complete:
    print(f"✓ Success: {result.panel_path}")
else:
    print(f"✗ Failed validation: {result.validation_summary}")
```

---

## Advanced: Custom Configuration

```python
from data_construction.utils.config import Stage0Config
from data_construction import Stage0Pipeline

# Create config programmatically
config = Stage0Config(
    date_range=("2010-01-01", "2023-12-31"),
    wrds_cache_dir="./cache",
    output_dir="./output",
    # ... other parameters
)

pipeline = Stage0Pipeline(config)
result = pipeline.run()
```
