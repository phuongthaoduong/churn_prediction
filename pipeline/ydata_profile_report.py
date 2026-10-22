"""
Step 1 (supplement): ydata-profiling Report
============================================
Generates an interactive HTML profiling report on a sampled subset
of srv_sales.csv and srv_saleitems.csv using ydata-profiling.

The full datasets are too large (7.2 GB) for in-memory profiling,
so we sample rows to keep memory usage reasonable.
"""

import pandas as pd
import os
from ydata_profiling import ProfileReport

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sale_data")
SALES_PATH = os.path.join(DATA_DIR, "srv_sales.csv")
SALEITEMS_PATH = os.path.join(DATA_DIR, "srv_saleitems.csv")
INDUSTRY_PATH = os.path.join(DATA_DIR, "industry_servereid.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output", "step1_profiling")

SAMPLE_SIZE = 500_000  # rows to sample from each large file

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sample_large_csv(path: str, sample_size: int, **read_kwargs) -> pd.DataFrame:
    """Read a large CSV by first counting rows, then sampling uniformly."""
    # Count total rows (fast scan without parsing all columns)
    total = sum(1 for _ in open(path)) - 1  # subtract header
    print(f"  Total rows in {os.path.basename(path)}: {total:,}")

    if total <= sample_size:
        print(f"  Loading all rows (below sample threshold)")
        return pd.read_csv(path, **read_kwargs)

    # Sample row indices
    skip_indices = sorted(
        set(range(1, total + 1))
        - set(pd.Series(range(1, total + 1)).sample(n=sample_size, random_state=42))
    )
    print(f"  Sampling {sample_size:,} / {total:,} rows ({sample_size/total:.1%})")
    return pd.read_csv(path, skiprows=skip_indices, **read_kwargs)


def profile_sales():
    """Generate ydata-profiling report for srv_sales.csv (sampled)."""
    print("\n[1/3] Profiling srv_sales.csv ...")
    df = sample_large_csv(
        SALES_PATH,
        SAMPLE_SIZE,
        dtype={
            "server_id": str,
            "shop_id": "int32",
            "id": "int64",
            "code": str,
            "company_id": "int64",
            "company_type": str,
            "acct_id": "int64",
            "acct_name": str,
        },
        parse_dates=["opt_on", "company_create"],
    )

    profile = ProfileReport(
        df,
        title="srv_sales Profiling Report (sampled)",
        explorative=True,
        minimal=False,
    )
    out_path = os.path.join(OUTPUT_DIR, "ydata_srv_sales.html")
    profile.to_file(out_path)
    print(f"  Saved to {out_path}")


def profile_saleitems():
    """Generate ydata-profiling report for srv_saleitems.csv (sampled)."""
    print("\n[2/3] Profiling srv_saleitems.csv ...")
    df = sample_large_csv(
        SALEITEMS_PATH,
        SAMPLE_SIZE,
        dtype={
            "server_id": str,
            "shop_id": "int32",
            "id": "int64",
            "sale_id": "int64",
            "product_id": "int64",
        },
        parse_dates=["opt_on"],
    )

    profile = ProfileReport(
        df,
        title="srv_saleitems Profiling Report (sampled)",
        explorative=True,
        minimal=False,
    )
    out_path = os.path.join(OUTPUT_DIR, "ydata_srv_saleitems.html")
    profile.to_file(out_path)
    print(f"  Saved to {out_path}")


def profile_industry():
    """Generate ydata-profiling report for industry_servereid.csv (full)."""
    print("\n[3/3] Profiling industry_servereid.csv ...")
    df = pd.read_csv(INDUSTRY_PATH, dtype=str)
    print(f"  Loaded {len(df):,} rows")

    profile = ProfileReport(
        df,
        title="industry_servereid Profiling Report",
        explorative=True,
        minimal=False,
    )
    out_path = os.path.join(OUTPUT_DIR, "ydata_industry.html")
    profile.to_file(out_path)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    profile_sales()
    profile_saleitems()
    profile_industry()
    print("\nAll ydata-profiling reports generated.")
