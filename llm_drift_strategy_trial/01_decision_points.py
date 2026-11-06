"""Step 1: reconstruct the exact drift evidence S4 saw at each of its real
decision points, for both schemes and both maturity modes, using only
already-computed artifacts under experiment_output/ (read-only; this script
never touches pipeline/ or experiment_output/ outputs).

This is NOT a live re-simulation of a new S5 strategy — it holds S4's actual
realized reference-batch/retrain trajectory fixed and asks: given the exact
evidence S4 saw at each point, does an LLM reasoning over it reach a
different decision than S4's psi_fired OR auc_fired rule? This is a cheap,
fast diagnostic to run BEFORE investing in a full sequential S5 simulation
(which requires actually refitting XGBoost at whatever points the LLM
chooses, since a different retrain timeline changes every later batch's
PSI reference and AUC history too).

Output: data/decision_points.json
"""
import json
import sys

import config as C

sys.path.insert(0, str(C.PIPELINE_DIR))

import pandas as pd
from sklearn.metrics import roc_auc_score

from drift_utils import compute_psi
from experiment_config import (
    BATCH_METADATA, FIRST_EVAL_PP, L2_THRESHOLDS_PATH, PSI_DISTRIBUTIONS_PATH,
    PSI_REFERENCE_PATH, equalized_close_dates, label_close_dates,
)

EXP = str(C.EXPERIMENT_DIR)

batch_meta = pd.read_csv(BATCH_METADATA)
l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)
psi_ref = json.load(open(PSI_REFERENCE_PATH))
dist = pd.read_csv(PSI_DISTRIBUTIONS_PATH)

all_pps = sorted(pd.to_datetime(batch_meta["prediction_point"]).unique())
eval_pps = [pp for pp in all_pps if pd.Timestamp(pp) >= FIRST_EVAL_PP]
assert len(eval_pps) == 12

log = pd.read_csv(f"{EXP}/retrain_log.csv", parse_dates=["t"])


def psi_lookup(scheme, monitored, ref_pp, cur_pp):
    sub = dist[dist["scheme"] == scheme]
    worst, contribs = 0.0, []
    for feat in monitored:
        ref = sub[(sub["prediction_point"] == ref_pp) & (sub["feature"] == feat)]
        cur = sub[(sub["prediction_point"] == cur_pp) & (sub["feature"] == feat)]
        p = compute_psi(ref.sort_values("bin")["proportion"].to_numpy(),
                        cur.sort_values("bin")["proportion"].to_numpy())
        contribs.append((feat, round(p, 4)))
        worst = max(worst, p)
    contribs.sort(key=lambda x: -x[1])
    return round(worst, 4), contribs[:5]


def build(mode):
    close = (label_close_dates(batch_meta, l2_thresholds) if mode == "scheme"
             else equalized_close_dates(batch_meta))
    close_idx = close.set_index("prediction_point")
    rows = []
    for scheme in ("L1", "L2"):
        monitored = psi_ref[scheme]["monitored_features"]
        label_close_col = "l1_label_close" if scheme == "L1" else "l2_label_close"
        strat_dir = f"{EXP}/predictions/{mode}/S4"
        init_row = log[(log.maturity_mode == mode) & (log.scheme == scheme) &
                       (log.strategy == "S4") & (log.trigger == "initial")].iloc[0]
        ref_pp_dt = FIRST_EVAL_PP - pd.Timedelta(days=int(init_row.effective_lag_days))
        ref_pp = f"{ref_pp_dt:%Y-%m-%d}"
        last_train_t = FIRST_EVAL_PP

        for t in eval_pps:
            t = pd.Timestamp(t)
            if t <= FIRST_EVAL_PP:
                continue
            psi_max, top5 = psi_lookup(scheme, monitored, ref_pp, f"{t:%Y-%m-%d}")
            auc_hist = []
            for prev in eval_pps:
                prev = pd.Timestamp(prev)
                if not (last_train_t <= prev < t):
                    continue
                if close_idx.loc[prev, label_close_col] > t:
                    continue
                saved = pd.read_parquet(f"{strat_dir}/{scheme}_{prev:%Y-%m-%d}.parquet")
                auc_hist.append(round(float(roc_auc_score(
                    saved[f"label_{scheme}"], saved["y_prob"])), 4))
            actual_row = log[(log.maturity_mode == mode) & (log.scheme == scheme) &
                             (log.strategy == "S4") & (log.t == t)]
            actual_action = "retrain" if len(actual_row) else "hold"
            rows.append({
                "mode": mode, "scheme": scheme, "t": f"{t:%Y-%m-%d}",
                "ref_pp": ref_pp,
                "months_since_last_train": round((t - last_train_t).days / 30, 1),
                "psi_max": psi_max, "top5_psi_features": top5,
                "auc_hist": auc_hist, "s4_actual_action": actual_action,
            })
            if actual_action == "retrain":
                ref_pp = f"{t:%Y-%m-%d}"
                last_train_t = t
    return rows


if __name__ == "__main__":
    all_rows = build("scheme") + build("equalized")
    C.DATA.mkdir(parents=True, exist_ok=True)
    with open(C.DATA / "decision_points.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    n_retrain = sum(1 for r in all_rows if r["s4_actual_action"] == "retrain")
    print(f"{len(all_rows)} decision points written to data/decision_points.json "
          f"({n_retrain} were S4 retrains, {len(all_rows) - n_retrain} were holds)")
