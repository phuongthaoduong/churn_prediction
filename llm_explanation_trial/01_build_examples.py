"""Step 1: pick a sample of customers from pipeline/shap_analysis.py's
output (experiment_output/shap_values_<scheme>.parquet) and extract each
one's top-K SHAP-ranked features, for the LLM explanation trial.

For each example, also builds the ABLATED version: the same top-K list
with the #1 SHAP feature removed. This is what makes the faithfulness
test exact rather than fuzzy-text-matching: the ablated prompt genuinely
does not contain the top feature, so if the LLM still names it as the
primary driver, that's a clean, mechanical hallucination.

Output: data/examples.json
"""
import json

import numpy as np
import pandas as pd

import config as C

SCHEMES = ("L1", "L2")


def top_k_features(row, cols, k):
    contribs = [(c, float(row[f"shap_{c}"]), row[f"{c}_value"]) for c in cols]
    contribs.sort(key=lambda x: -abs(x[1]))
    return contribs[:k]


def main():
    rng = np.random.default_rng(C.SEED)
    examples = []
    for scheme in SCHEMES:
        df = pd.read_parquet(C.EXPERIMENT_DIR / f"shap_values_{scheme}.parquet")
        cols = [c[len("shap_"):] for c in df.columns if c.startswith("shap_")]

        top_risk = df.sort_values("y_prob", ascending=False).head(C.N_TOP_RISK)
        remaining = df.drop(top_risk.index)
        random_sample = remaining.sample(n=C.N_RANDOM, random_state=C.SEED)
        chosen = pd.concat([top_risk, random_sample])

        for i, (_, row) in enumerate(chosen.iterrows()):
            full = top_k_features(row, cols, C.TOP_K_FEATURES)
            ablated = full[1:]  # drop the #1 SHAP feature entirely
            examples.append({
                "example_id": f"{scheme}_{i}",
                "scheme": scheme,
                "y_prob": round(float(row["y_prob"]), 4),
                "label": int(row[f"label_{scheme}"]),
                "top_feature_removed_in_ablation": full[0][0],
                "full_features": [
                    {"feature": f, "shap": round(s, 4),
                     "value": (None if pd.isna(v) else round(float(v), 2))}
                    for f, s, v in full],
                "ablated_features": [
                    {"feature": f, "shap": round(s, 4),
                     "value": (None if pd.isna(v) else round(float(v), 2))}
                    for f, s, v in ablated],
            })

    C.DATA.mkdir(parents=True, exist_ok=True)
    with open(C.DATA / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"{len(examples)} examples written to data/examples.json "
          f"({len(SCHEMES)} schemes x {C.N_TOP_RISK + C.N_RANDOM} customers)")


if __name__ == "__main__":
    main()
