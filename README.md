# Data Construction

A production-grade data reconstruction pipeline for building fundamental factor datasets from US equity market data (CRSP, Compustat, Fama-French factors).

## Overview

This package reconstructs a monthly panel of 46 fundamental characteristics from raw market and accounting data, with three-arm validation against published benchmarks (CPZ characteristics, Fama-French factors, and internal audit checks).

**Key features:**
- Deterministic, idempotent pipeline with full traceability to academic equations
- 46 fundamental characteristics across 7 families (value, profitability, investment, momentum, risk, liquidity, other)
- Point-in-time merged accounting data (380-day tolerance window)
- Three-arm validation framework (CPZ overlap, FF5+UMD replication, internal audits)
- Structured logging and diagnostic outputs
- Full test coverage with synthetic fixtures

## Installation

### Prerequisites

- Python 3.11+
- WRDS account (for live data pulls)
- Conda (recommended) or pip

### Quick start

```bash
# Clone the repo
git clone https://github.com/Amos-Anderson/cpz-equity-data-reconstruction-and-extension/tree/main/src/data_construction.git
cd data-construction

# Install dependencies
conda env create -f environment.yml
conda activate data-construction

# Install package in development mode
pip install -e ".[dev]"

# Run tests
pytest -v
```

### Configuration

1. Set up your WRDS credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your WRDS username/password
   ```

2. Review the sample config:
   ```bash
   cat configs/stage00.yaml
   ```

## Usage

### Basic pipeline execution

```python
from data_construction import Stage0Pipeline

# Load configuration
pipeline = Stage0Pipeline.from_yaml("configs/stage00.yaml")

# Run the full pipeline
result = pipeline.run()

# Check validation status
if result.stage0_complete:
    print(f"✓ Pipeline complete: {result.n_rows} rows, {result.n_firms} firms")
    print(f"  Characteristics: {result.characteristics}")
    print(f"  Date range: {result.date_range}")
else:
    print("✗ Pipeline failed validation")
    print(result.validation_summary)

# Access outputs
print(f"Panel: {result.panel_path}")
print(f"Validation: {result.ff_validation_path}")
```

### Loading the constructed panel

```python
import pandas as pd

panel = pd.read_parquet("data/factor_panel_v2.parquet")
print(panel.head())
# Columns: permno, date, 46 characteristic columns (rank-normalized)
```

### Raw data access

```python
from data_construction.raw_data import crsp, accounting

# Load CRSP monthly stock data
crsp_monthly = crsp.load_crsp_monthly(wrds_conn)

# Load Compustat with CCM link
compustat = accounting.load_compustat_annual(wrds_conn)

# Compute book equity (Davis-Fama-French hierarchy)
be = accounting.compute_book_equity(compustat)
```

### Custom characteristic computation

```python
from data_construction.characteristics import value, profitability

# Compute value family characteristics (BEME, E2P, CF2P, D2P, S2P, A2ME)
value_chars = value.compute_value(monthly_panel, compustat, be)

# Compute profitability characteristics (PROF, ROE, ROA, OP, PM, PCM, RNA)
prof_chars = profitability.compute_profitability(monthly_panel, compustat, be)
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Detailed design, module layout, contracts, test organization
- **[Document 1: Methodology](docs/00_data_foundation_doc1_methodology.md)** — Math, equations, data sources
- **[Document 2: Characteristics](docs/00_data_foundation_doc2_characteristics.md)** — All 46 characteristics with definitions
- **[Document 3: Validation](docs/00_data_foundation_doc3_validation.md)** — Three-arm validation framework

## Pipeline stages

1. **Raw data loading** — Pull CRSP (monthly/daily), Compustat, FF factors from WRDS or cache
2. **Book equity** — Davis-Fama-French hierarchy (Doc 1, Eq. 4)
3. **Delisting adjustment** — Shumway 1997 imputation for missing returns (Doc 1, Eq. 3)
4. **Point-in-time merge** — Attach accounting data with 380-day tolerance (Doc 1, §2.4)
5. **Characteristic computation** — 46 characteristics across 7 families (Doc 2)
6. **Completeness filter** — Retain firm-months with all 46 characteristics (Doc 1, Eq. 9)
7. **Rank normalization** — Scale to (-0.5, 0.5) range per month (Doc 1, Eq. 5)
8. **Validation** — Three-arm checks (CPZ, Fama-French, internal audits)

## Validation framework

**Arm 1: CPZ Comparison** (1967-2016 overlap)
- Mean absolute deviation < 0.005 for 49/50 years
- Full-period MAD < 1e-4 for 44/46 characteristics
- Spearman rank correlation > 0.95 for 95% of cells

**Arm 2: Fama-French Replication** (2017-2024)
- Construct FF5+UMD from our characteristics using 2×3 sorts
- Pearson correlation > 0.85 with published series
- Monthly MAD < 0.01 for all factors
- Annualized divergence < 3%

**Arm 3: Internal Audits**
- PIT bounds integrity (6mo ≤ lag < 380 days)
- Look-ahead detection (momentum skip, risk window, price adjustments)
- Coverage stability (YoY change < 10%)
- Rank value bounds (-0.5, 0.5)
- Schema validation (no duplicates, correct dtypes, monotonic dates)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=data_construction

# Run specific test module
pytest tests/test_pipeline.py -v
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Code quality checks
ruff check src/
black --check src/

# Auto-fix formatting
ruff check --fix src/
black src/
```

## Citation

If you use this package in research, please cite:

```bibtex
@software{anderson2026dataconstruction,
  title={Data Construction: Fundamental Factor Pipeline for US Equities},
  author={Anderson, Amos},
  year={2026},
  url={https://github.com/Amos-Anderson/cpz-equity-data-reconstruction-and-extension}
}
```

## License

Proprietary — Academic use only. See LICENSE file for details.

## Contact

Questions or issues? Open an issue on GitHub or contact the author.
