# Installation Guide

This document covers detailed installation and setup instructions for the data-construction pipeline.

## System Requirements

- **Python**: 3.11 or higher
- **OS**: Linux, macOS, or Windows
- **Disk space**: ~20GB for raw data cache (optional)
- **Memory**: 16GB RAM recommended for full pipeline runs

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/data-construction.git
cd data-construction
```

## Step 2: Create a Python Environment

### Option A: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate data-construction
```

### Option B: Using venv + pip

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Step 3: Set Up WRDS Access

This package requires a WRDS (Wharton Research Data Services) account to pull raw data.

### 3a: Create a WRDS Account

1. Visit [WRDS Registration](https://wrds-www.wharton.upenn.edu/login/)
2. Create an account or log in
3. Subscribe to the following datasets:
   - CRSP (US Stock Database)
   - Compustat (Fundamentals Annual)
   - Fama/French Factors

### 3b: Configure Local Credentials

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your WRDS credentials:
   ```
   WRDS_USERNAME=your_username
   WRDS_PASSWORD=your_password
   ```

3. Verify connection (optional):
   ```python
   from data_construction.raw_data import download
   conn = download.get_wrds_connection()
   print("✓ WRDS connection successful")
   ```

## Step 4: Install Package

```bash
# Development install (editable)
pip install -e .

# Full install with dev tools
pip install -e ".[dev]"
```

## Step 5: Verify Installation

Run the test suite to ensure everything is working:

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=data_construction
```

Expected output:
```
========== test session starts ==========
collected 150+ items
tests/test_pipeline.py::test_stage0_pipeline_integration PASSED
tests/raw_data/test_crsp.py::test_load_crsp_monthly PASSED
...
========== 150+ passed in 45.2s ==========
```

## Configuration

### Main Config File

The pipeline is configured via YAML:

```bash
cp configs/stage00.yaml configs/my_config.yaml
# Edit as needed
```

Key parameters in `configs/stage00.yaml`:
- `date_range`: Start and end dates for data pull
- `wrds_cache_dir`: Where to cache raw data
- `output_dir`: Where to write processed data
- `validation_thresholds`: Acceptance criteria for three-arm validation

### Environment Variables

Additional configuration via `.env`:

```
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password
DATA_DIR=./data
CACHE_DIR=./data/.cache
LOG_LEVEL=INFO
```

## First Run

### Quick Test (Synthetic Data)

```python
from data_construction import Stage0Pipeline
from tests.conftest import sample_panel

# Uses test fixtures, no WRDS needed
panel = sample_panel()
print(f"✓ Sample panel: {panel.shape[0]} rows, {len(panel.columns)} columns")
```

### Full Pipeline (Requires WRDS)

```python
from data_construction import Stage0Pipeline

# Load configuration
pipeline = Stage0Pipeline.from_yaml("configs/stage00.yaml")

# Run pipeline
result = pipeline.run()

# Check status
if result.stage0_complete:
    print(f"✓ Pipeline complete")
    print(f"  Panel: {result.panel_path}")
    print(f"  Rows: {result.n_rows}")
    print(f"  Firms: {result.n_firms}")
    print(f"  Date range: {result.date_range[0]} to {result.date_range[1]}")
else:
    print("✗ Pipeline validation failed")
    print(result.validation_summary)
```

## Troubleshooting

### WRDS Connection Issues

```python
# Test connection
import os
from data_construction.raw_data import download

os.environ['WRDS_USERNAME'] = 'your_username'
os.environ['WRDS_PASSWORD'] = 'your_password'

try:
    conn = download.get_wrds_connection()
    print("✓ Connected")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

**Common issues:**
- Credentials incorrect → Check WRDS account settings
- Query timeout → Try smaller date range in config
- Permission denied → Verify dataset subscriptions on WRDS website

### Memory Issues

If running out of memory during characteristic computation:

```yaml
# In configs/stage00.yaml
processing:
  chunk_size: 5000  # Reduce from default 10000
  n_jobs: 1        # Use single process instead of parallel
```

### Slow Performance

To speed up subsequent runs, the pipeline caches raw data:

```bash
# See what's cached
ls -lh data/.cache/

# Clear cache if needed
rm data/.cache/crsp_*.parquet
```

## Docker (Optional)

To run in a container:

```bash
# Build image
docker build -t data-construction .

# Run container
docker run -it \
  -e WRDS_USERNAME=your_username \
  -e WRDS_PASSWORD=your_password \
  -v $(pwd)/data:/app/data \
  data-construction bash
```

## Getting Help

- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for design details
- Read the documentation files (doc1_methodology, doc2_characteristics, doc3_validation)
- Review test files for usage examples
- Open an issue on GitHub

## Next Steps

Once installed and verified:

1. Review the [ARCHITECTURE.md](ARCHITECTURE.md) for overview
2. Run the example in `configs/stage00.yaml`
3. Explore outputs in `data/`
4. Check validation reports in `data/stage0_summary.json`
