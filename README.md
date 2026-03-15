# Data Pipeline: CPZ Dataset Extension (1967–2024)

### A step-by-step guide to reproducing and extending the Chen, Pelger & Zhu (2020) dataset

#### By Amos Anderson and Nick Koukounas (March, 2026)

---

## What This Pipeline Does

This notebook constructs a cross-sectional equity dataset of 46 firm characteristics
for all US common stocks from **January 1967 to December 2024**. It replicates the
dataset used in Chen, Pelger & Zhu (2020), *"Deep Learning in Asset Pricing,"*
Management Science 70(2), 714–750 and extends it by 8 years beyond their original
2016 endpoint, adding the COVID crash, the 2022 rate-shock regime, and the post-hike
recovery period.

**Final output:** Three parquet files (`train.parquet`, `valid.parquet`, `test.parquet`)
containing rank-normalized characteristics and excess returns, ready for factor model
estimation and machine learning.

| Split | Period | Rows | Avg Stocks/Month |
|-------|--------|------|-----------------|
| Train | 1967–1989 | 489,885 | 1,775 |
| Valid | 1990–1999 | 376,018 | 3,133 |
| Test  | 2000–2024 | 739,286 | 2,464 |
| **Total** | **1967–2024** | **1,605,189** | **2,300** |

---

## Prerequisites

You need **four things** before running a single cell. Do not skip any of them.

---

### 1. WRDS Account

WRDS (Wharton Research Data Services) is the data provider for CRSP and Compustat.
You must have an active institutional account.

**How to get one:**
- Visit [wrds-web.wharton.upenn.edu](https://wrds-web.wharton.upenn.edu)
- Click **"Register"** and use your institutional email address
- Your institution must have a WRDS subscription (most research universities do)
- Account activation typically takes 1-2 business days

**What you need from WRDS:**
- Your WRDS **username** (not your email: the short username you chose at registration)
- Your WRDS **password**

**You will be prompted to enter these credentials every time you run Cells 2, 3, and 4.**
There is no way to pre-supply them in the notebook as WRDS handles authentication
interactively at connection time. This is by design.

**WRDS data subscriptions required:**
- CRSP Monthly and Daily Stock Files
- Compustat Fundamentals Annual
- Fama-French Factors

If your account does not have access to any of these, contact your institution's
WRDS administrator.

---

### 2. Python Environment

The pipeline requires **Python 3.10 or later**. We strongly recommend using a
dedicated conda environment to avoid package conflicts.

**Step 1 - Install Miniconda** (if you do not already have conda):

Download from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
and follow the installer instructions for your operating system.

**Step 2 - Create the environment:**

Open a terminal (Anaconda Prompt on Windows, Terminal on macOS/Linux) and run:

```bash
conda create -n ml4t python=3.11 -y
conda activate ml4t
```

**Step 3 - Install all required packages:**

```bash
pip install jupyterlab pandas numpy scipy wrds pyarrow fastparquet
```

Every package is required. Do not skip any:

| Package | Purpose |
|---------|---------|
| `jupyterlab` | Run the notebook |
| `pandas` | Data manipulation throughout |
| `numpy` | Numerical computation |
| `scipy` | Linear regression for Beta computation |
| `wrds` | WRDS database connection |
| `pyarrow` | Read/write parquet files |
| `fastparquet` | Parquet fallback |

**Step 4 - Verify the installation:**

```bash
python -c "import wrds, pandas, numpy, scipy, pyarrow; print('All packages OK')"
```

You should see: `All packages OK`

---

### 3. Storage Space

The pipeline downloads and creates large files. You need at least **15 GB of free disk
space** on the drive where you store the data.

Breakdown of what gets created:

| File | Size |
|------|------|
| `crsp_msf_raw.parquet` (CRSP monthly) | ~52 MB |
| `crsp_dsf_raw.parquet` (CRSP daily) | ~715 MB |
| `compustat_annual_raw.parquet` | ~47 MB |
| `ff_factors_*.parquet` (2 files) | ~5 MB |
| `crsp_market_index.parquet` | ~1 MB |
| `crsp_clean_monthly.parquet` | ~109 MB |
| `accounting_chars.parquet` | ~67 MB |
| `monthly_chars.parquet` | ~278 MB |
| `risk_chars.parquet` | ~129 MB |
| `crsp_delist_raw.parquet` | ~3 MB |
| `panel_final.parquet` | ~481 MB |
| `train.parquet` + `valid.parquet` + `test.parquet` | ~615 MB |
| **Total** | **~2.5 GB** |

---

### 4. Data Directories

The pipeline writes files to two fixed directories. **Create both before running:**

**On Windows:**
```
C:\Users\<your_username>\ml4t_data\raw\
C:\Users\<your_username>\ml4t_data\extended_v2\
```

To create them, open File Explorer and create the folders, or run in a terminal:
```cmd
mkdir C:\Users\%USERNAME%\ml4t_data\raw
mkdir C:\Users\%USERNAME%\ml4t_data\extended_v2
```

**On macOS/Linux:**
```
~/ml4t_data/raw/
~/ml4t_data/extended_v2/
```
```bash
mkdir -p ~/ml4t_data/raw ~/ml4t_data/extended_v2
```

> **Important:** If your username is not `amosa`, you must update the `RAW_DIR`,
> `OUT_DIR`, and `CPZ_PATH` variables at the top of each code cell before running.
> Search for `amosa` in the notebook and replace with your actual Windows username.

---

## The CPZ Reference Dataset

You also need the original CPZ dataset file. This is used **only** in Cell 1
to establish validation targets and it is never modified.

**File:** `firm_characteristics_all.parquet`

**Where it must be placed:**
```
C:\Users\<your_username>\ml4t_data\academic\firm_characteristics_all.parquet
```

This file is available from Stefan Nagel's course materials or the CPZ replication
package. If you obtained this notebook as part of a course, this file was provided
to you separately. Place it in the `academic` subfolder exactly as shown above.

---

## Running the Notebook

### Launch JupyterLab

With your conda environment active, navigate to the folder containing
`Data_Pipeline.ipynb` and launch JupyterLab:

```bash
conda activate ml4t
cd path/to/notebook/folder
jupyter lab
```

A browser window will open automatically. Double-click `Data_Pipeline.ipynb`
to open it.

---

### Cell Execution Guide

**Run every cell in order, top to bottom. Do not skip cells. Do not re-run
cells out of order.** Each cell writes files to disk that subsequent cells
depend on.

---

#### Cell 1 - Validation Targets *(~5 seconds, no WRDS required)*

Reads the CPZ reference dataset and prints validation targets. No data is
written to disk. This cell confirms your CPZ file is readable and shows the
exact numbers our pipeline must reproduce.

**Expected output:** Tables of stocks per month (584 in 1967, growing to 1971
in 2016), annual returns, and characteristic statistics.

---

#### Cell 2 - CRSP Markdown *(no code, just read)*

Explains the construction methodology. No action required.

---

#### Cell 3 - Pull CRSP Monthly Returns *(~3 minutes, WRDS required)*

Connects to WRDS and downloads the CRSP monthly stock file (1960–2024) and
delisting returns.

**You will see two prompts:**
```
Enter your WRDS username [previous_user]:
Enter your password:
```
Type your WRDS username and press Enter. Type your password (it will not be
visible as you type) and press Enter.

**You will also be asked:**
```
Create .pgpass file now [y/n]?:
```
Type `y` and press Enter. This saves your credentials for future sessions so
you only need to enter them once per session (not once per cell).

**Expected output:**
```
CRSP monthly shape: (2873923, 12)
Unique permnos: 22,150
Saved: ...crsp_msf_raw.parquet (51.7 MB)
Delisting returns: (38383, 4)
Step 1 complete.
```

---

#### Cell 4 - Pull Compustat *(~10 minutes, WRDS required)*

Downloads Compustat annual fundamentals (1960–2024). This is all balance sheet,
income statement, and cash flow data needed for the 20 accounting characteristics.

**You will be prompted for WRDS credentials again.** Enter them the same way
as in Cell 3. If you created the `.pgpass` file in Cell 3, you may not be
prompted, this is normal.

**Expected output:**
```
Compustat shape: (329368, 48)
Unique permnos: 27,919
Saved: ...compustat_annual_raw.parquet (46.4 MB)
Step 2 complete.
```

---

#### Cell 5 - Pull CRSP Daily + FF Factors *(~25 minutes, WRDS required)*

Downloads four datasets in sequence:
1. Fama-French monthly factors (1926–2024) : needed for the risk-free rate
2. Fama-French daily factors (1963–2024) : needed for Beta computation
3. CRSP value-weighted market index (1960–2024)
4. CRSP daily stock file (1963–2024) : this is the large one (~60M rows)

**You will be prompted for WRDS credentials once at the start of this cell.**

The daily stock file download dominates runtime. The terminal will be silent
for up to 25 minutes while it transfers. This is normal so do not interrupt it.

**Expected output:**
```
FF monthly: (1182, 5) | Saved: ...
FF daily: (15606, 5) | Saved: ...
Market index: (780, 3) | Saved: ...
CRSP daily shape: (59431009, 8)
Saved: ...crsp_dsf_raw.parquet (715.1 MB)
Step 3 complete.
```

---

#### Cell 6 - Build Clean Monthly Panel *(~2 minutes, no WRDS required)*

Merges delisting returns into CRSP monthly data, computes excess returns
using the full Fama-French risk-free rate history (ensuring pre-1967 months
have valid rf values for momentum signal construction), and computes market
equity.

**No WRDS connection required.** Reads from the files saved in Cell 5.

**Expected output:**
```
1960: ret=99.5%  rf=100.0%  ret_excess=99.5%
...
1967: ret=99.4%  rf=100.0%  ret_excess=99.4%
Clean monthly panel: (2873923, 18)
Step 4 complete.
```

---

#### Cell 7 - Characteristic Construction Markdown *(no code, just read)*

Explains all 46 characteristics and their academic sources. No action required.

---

#### Cell 8 - Construct Accounting Characteristics *(~3 minutes, no WRDS required)*

Computes all 20 annual accounting characteristics from Compustat following
Freyberger, Neuhierl & Weber (2017). Applies the 6-month fiscal year lag
(Fama-French convention).

**Expected output:**
```
Accounting characteristics: (329368, 33)
avail_date range: 1960-07-31 to 2025-06-30
Saved: ...accounting_chars.parquet (66.8 MB)
Step 5 complete.
```

---

#### Cell 9 - Construct Monthly + Risk Characteristics *(~90 minutes, no WRDS required)*

This is the longest-running cell. It has three sections:

**Section A -- Monthly characteristics (~25 minutes):**
Computes momentum signals (r12_2, r12_7, r36_13, LT_Rev, r2_1, ST_REV),
Rel2High, NI (net issuance), LME, LTurnover, and SUV. All momentum signals
use raw returns computed over the full history back to 1960 - this is what
ensures non-null values from January 1967 onward.

The SUV computation is the slow part of Section A. The terminal will print
progress every few minutes. This is normal.

**Section B -- Risk characteristics (~45 minutes):**
Computes Beta, MktBeta, IdioVol, Resid_Var, and Variance using 252-day
rolling OLS regressions over the daily return data. Processes ~22,000 stocks
in 44 chunks and prints progress every 5 chunks.

**Section C -- Roll (1984) spread (~20 minutes):**
Computes the implied bid-ask spread from the serial covariance of daily
returns. This is the only spread measure with coverage back to 1963 as direct
bid-ask quotes are not available in CRSP before 1983.

**Expected output (truncated):**
```
=== Section A: Monthly characteristics ===
SUV non-null: 2,267,254
Monthly chars saved: ...monthly_chars.parquet (278.3 MB)

=== Section B: Risk characteristics from daily data ===
Processing 21,825 permnos in 44 chunks...
  Chunk 1/44...
  ...
Risk computation complete: 2,696,268 observations

=== Section C: Roll (1984) implied spread ===
Roll spread: (2849588, 4) | non-null: 98.8%
Risk chars saved: ...risk_chars.parquet (128.6 MB)
Step 6 complete.
```

---

#### Cell 10 - Final Merge, Validate, Normalize, Save *(~20 minutes, no WRDS required)*

This cell assembles the final dataset in seven steps:

1. Merges all sources onto the base CRSP monthly panel
2. Computes the nine ME-dependent characteristics (BEME, E2P, CF2P, etc.)
3. Applies the CPZ completeness filter -- keeps only stock-months where all
   46 characteristics are simultaneously non-null
4. Validates breadth (stocks per month) and return distribution against CPZ
5. Rank-normalizes all 46 characteristics to [-0.5, 0.5] within each month
6. Splits into train/valid/test
7. Saves final parquet files

**Expected validation output:**
```
=== VALIDATION: Return distribution vs CPZ ===
mean     0.009467     0.009855   0.000388 ✓
std      0.168178     0.161131   0.007047 ← CHECK
p10     -0.146535    -0.142433   0.004102 ✓
p50     -0.002452    -0.001946   0.000506 ✓
p90      0.163267     0.161867   0.001400 ✓
```

**Expected final output:**
```
train:  489,885 rows | 1967-01-31 to 1989-12-31 | avg 1775 stocks/month
valid:  376,018 rows | 1990-01-31 to 1999-12-31 | avg 3133 stocks/month
test :  739,286 rows | 2000-01-31 to 2024-12-31 | avg 2464 stocks/month
Total rows: 1,605,189
Step 7 complete. Pipeline done.
```

---

#### Cell 11 - Validation Results Markdown *(no code, just read)*

Summarizes the validation results and documents the known remaining
differences from CPZ. No action required.

---

## Total Runtime Summary

| Cell | Task | Runtime | WRDS? |
|------|------|---------|-------|
| Cell 1 | Validation targets | ~5 sec | No |
| Cell 3 | CRSP monthly pull | ~3 min | **Yes** |
| Cell 4 | Compustat pull | ~10 min | **Yes** |
| Cell 5 | CRSP daily pull | ~25 min | **Yes** |
| Cell 6 | Build clean panel | ~2 min | No |
| Cell 8 | Accounting chars | ~3 min | No |
| Cell 9 | Monthly + risk chars | ~90 min | No |
| Cell 10 | Final merge + save | ~20 min | No |
| **Total** | | **~2.5 hours** | |

---

## Output File Structure

After Cell 10 completes successfully, your output directory contains:

```
C:\Users\<your_username>\ml4t_data\
│
├── academic\
│   └── firm_characteristics_all.parquet     <-- CPZ reference (you provide this)
│
├── raw\                                      <-- Intermediate files (safe to delete after)
│   ├── crsp_msf_raw.parquet
│   ├── crsp_dsf_raw.parquet
│   ├── crsp_delist_raw.parquet
│   ├── crsp_clean_monthly.parquet
│   ├── crsp_market_index.parquet
│   ├── compustat_annual_raw.parquet
│   ├── ff_factors_monthly_full.parquet
│   ├── ff_factors_daily.parquet
│   ├── accounting_chars.parquet
│   ├── monthly_chars.parquet
│   └── risk_chars.parquet
│
└── extended_v2\                              <-- FINAL OUTPUT
    ├── train.parquet                         1967–1989
    ├── valid.parquet                         1990–1999
    └── test.parquet                          2000–2024
```

---

## How to Load the Final Data

```python
import pandas as pd

EXTENDED_DIR = r"C:\Users\<your_username>\ml4t_data\extended_v2"

train = pd.read_parquet(f"{EXTENDED_DIR}\\train.parquet")
valid = pd.read_parquet(f"{EXTENDED_DIR}\\valid.parquet")
test  = pd.read_parquet(f"{EXTENDED_DIR}\\test.parquet")

# Key columns
# 'permno'     : CRSP permanent security identifier
# 'date'       : Month-end date (pandas Timestamp)
# 'ret_excess' : Monthly excess return (return - risk-free rate)
# 'me'         : Market equity in $millions (current month)
# All 46 characteristics: rank-normalized to [-0.5, 0.5]

print(train.columns.tolist())
print(train.shape)
```

---

## Troubleshooting

**"column does not exist" error from WRDS**

The WRDS schema was updated in 2022. The pipeline uses the current column
names (`date` instead of the legacy `caldt` in the market index table).
If you see this error in a different cell, check the table name in the SQL
query and verify it against the WRDS data dictionary for your subscription.

**WRDS connection times out**

WRDS connections expire after ~30 minutes of inactivity. If you pause
between cells and get a connection error, re-run the affected cell from
the top - it will re-authenticate. The `.pgpass` file means you will not
need to re-enter your password.

**"Permission denied" on a WRDS table**

Your WRDS account does not have access to that dataset. Contact your
institution's WRDS administrator and request access to CRSP, Compustat,
and Fama-French datasets.

**DeprecationWarning about DataFrameGroupBy.apply**

This is a pandas version warning, not an error. The computation completes
correctly. It will be suppressed in a future version of the pipeline.

**Cell 9 appears frozen during SUV computation**

The SUV calculation processes each stock's full time series in a Python loop.
It is genuinely slow (~20 minutes) and produces no progress output during
computation. The cell is running so do not interrupt it. Wait for the
`SUV non-null: 2,267,254` line to appear.

**Out of memory error**

The daily CRSP file loads ~3 GB into RAM during Cell 9. You need at least
16 GB of RAM available. Close other applications before running Cell 9.
If you have exactly 16 GB total, close your browser and all other programs.

**Numbers differ slightly from what is shown here**

WRDS updates their databases periodically with corrections and additions.
Minor differences (< 1%) in stock counts or return values from those shown
in this README are expected and do not indicate a problem.

---

## Known Differences from the Original CPZ Dataset

The pipeline is validated to match CPZ **closely but not perfectly**. The
following known differences are documented and understood:

| Issue | Magnitude | Explanation |
|-------|-----------|-------------|
| Breadth 1967–1982 | ~25% more stocks | CPZ applied minimum listing history filter not documented in their paper |
| Breadth 1983–1985 | ~27% fewer stocks | NASDAQ Compustat expansion timing difference |
| Breadth 2005–2016 | 4-8% more stocks | Minor universe filter differences |
| Return std | 0.168 vs 0.161 | Direct consequence of universe size difference |
| Annual returns | 49/50 years within ±0.5% | Only 1967 differs by 0.76% |

These differences do not affect the validity of the 2017–2024 extension.

---

## Citation

If you use this dataset in your research, please cite:

> Chen, L., Pelger, M., & Zhu, J. (2024). Deep learning in asset pricing.
> *Management Science*, 70(2), 714–750.

And cite the underlying data sources:

> Freyberger, J., Neuhierl, A., & Weber, M. (2020). Dissecting characteristics
> nonparametrically. *Review of Financial Studies*, 33(5), 2326–2377.

> Shumway, T. (1997). The delisting bias in CRSP data.
> *Journal of Finance*, 52(1), 327–340.

> Roll, R. (1984). A simple implicit measure of the effective bid-ask spread
> in an efficient market. *Journal of Finance*, 39(4), 1127–1139.

---

*For questions about this pipeline, refer to the accompanying LaTeX
documentation which describes every construction decision in full detail.*
