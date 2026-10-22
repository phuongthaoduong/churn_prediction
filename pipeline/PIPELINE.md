# Pipeline Step Index

Scripts in this directory used to be named `stepN_*.py` / `stepN_M_*.py`. The
step number is now tracked only here — filenames no longer carry it. Run in
the order listed (each script's own docstring has the full input/output
contract).

| Step | Script | What it does |
|------|--------|---------------|
| 1 | `data_profiling.py` | Profiles the raw 7.2 GB `srv_sales.csv` / `srv_saleitems.csv` in chunks: data span/scale, verticals, median repurchase cycles, sufficient-history proportion, data quality checks. |
| 1 (supplement) | `ydata_profile_report.py` | Interactive ydata-profiling HTML report on a sampled subset of the raw data. |
| 2.1 | `remove_outliers.py` | Removes transactions with implausible financial amounts (bill_amt / amt outliers above a derived threshold). Runs first in the cleaning sequence. |
| 2.2 | `detect_test_accounts.py` | Multi-signal detection and removal of system/POS test accounts (extreme txn count, trivial-amount ratio, placeholder create date). |
| 2.3 | `handle_returns.py` | Analyzes and removes negative-amount returns/corrections; computes per-customer return stats before removal. |
| 2.4 | `remove_orphans.py` | Removes line items with no matching sale record (orphans left behind by upstream removals). |
| 3 | `temporal_structuring.py` | Partitions cleaned transactions into monthly batches with observation/label window separation; assigns customer-batch eligibility. |
| 3 (report) | `report_and_metrics.py` | Narrative report of Step 3 plus all metric categories recomputed on the cleaned data. |
| 4 | `feature_engineering.py` | Builds the 25-feature matrix (customer, store, time-level) per (customer, prediction_point), using only the observation window. |
| 5 | `label_construction.py` | Builds churn labels under two schemes: L1 (fixed 30-day threshold) and L2 (category-level dynamic P95 threshold). |
| 6 | `dataset_assembly.py` | Merges features (step 4) and labels (step 5) into `dataset_final.parquet`; validates for leakage/completeness/consistency. |
| 7.1 | `train_eval_split.py` | Assigns each monthly batch to `initial_train` or `evaluation`; records label-maturity dates per scheme; snapshots experiment constants. |
| 7.2 | `drift_baselines.py` | Trains provisional initial models, ranks features for drift monitoring, freezes PSI bin edges, precomputes per-batch bin proportions. |
| 7.3 | `preprocessing_log.py` | Assembles the consolidated preprocessing decision log (`docs/preprocessing_log.md`) from all step 2–7.2 artifacts. |
| 8.0 | `eval_labels.py` | Builds the shared 60-day evaluation label (`label_eval`) used as the primary cross-scheme comparison outcome. |
| 8.1 | `hyperparameter_tuning.py` | Checkpointed random search per label scheme over leakage-safe TSCV folds; freezes tuned hyperparameters. |
| 8.2 | `feature_set_comparison.py` | Compares the 25-feature set against an RFM baseline using tuned hyperparameters; freezes the feature set; reports mutual information. |
| 8.3 | `run_experiment.py` | Runs the prequential factorial experiment (L1/L2 × S1–S4 strategies) over both maturity modes (`scheme`, `equalized`); persists predictions under `experiment_output/predictions/<mode>/<strategy>/` and the retrain log. |
| 8.4 | `compute_metrics.py` | Computes per-batch AUC/F1/Recall@10%/Jaccard against the shared and own labels, and aggregates per-configuration summaries with retraining cost. |

Shared modules (not pipeline steps, imported by the scripts above):
`experiment_config.py`, `drift_utils.py`, `strategies.py`, `metrics.py`.
