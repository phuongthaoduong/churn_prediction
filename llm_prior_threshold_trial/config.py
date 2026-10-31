"""Shared settings for the LLM-prior threshold trial."""
from pathlib import Path

TRIAL_DIR = Path(__file__).resolve().parent
ROOT = TRIAL_DIR.parent                      # Data_Processing/
SALE_DATA = ROOT / "sale_data"
TEST_ACCOUNTS = ROOT / "profiling_output" / "step2_cleaning" / "exclusion_log_step2_2.csv"

DATA = TRIAL_DIR / "data"
LLM_RAW = DATA / "llm_raw"
RESULTS = TRIAL_DIR / "results"

DATA_START = "2024-05-01"
DATA_END = "2026-05-31"
TRAIN_END = "2025-03-31"          # last day of the 8 initial-train batches
FIRST_PP = "2024-08-01"
FIRST_EVAL_PP = "2025-04-01"
LAST_PP = "2026-03-01"

PERCENTILE = 95                   # same as L2_PERCENTILE in the pipeline
OBS_WINDOW_DAYS = 90              # customer must buy in the 90 days before PP
FOLLOWUP_DAYS = 180               # "came back" window for the false-churn check
MIN_IPI_FOR_REGRESSION = 200      # merchants used to fit the prior regression
KAPPA_GRID = [1, 3, 10, 30, 100, 300, 1000]
SIZE_BUCKETS = [0, 100, 500, 2000, float("inf")]

PILOT_SIZE = 100
SEED = 42
