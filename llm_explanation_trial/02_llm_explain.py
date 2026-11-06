"""Step 2: ask an LLM to explain each customer's churn risk from their real
SHAP contributions (full prompt), and again with the #1 SHAP feature
removed entirely (ablated prompt). The LLM never sees customer identity,
merchant name, or anything beyond the numeric feature:value:SHAP triples —
it cannot repeat the falsified "ask the LLM what it knows" mistake because
there is nothing to know; everything it's given is the actual reason the
model made this prediction.

Faithfulness test (exact, not fuzzy text-matching): the LLM is asked to
report a structured `primary_driver` field (one of the feature names it
was shown) alongside its narrative. In the ablated run, the removed
feature is NOT in the candidate list — so if the LLM still names it,
that's an unambiguous hallucination, not just the case of the
LLM being unable to name it, since it genuinely cannot be recovered from
what it was given.

Usage
  export LLM_BASE_URL=https://api.deepseek.com/v1
  export LLM_API_KEY=sk-...
  python3 02_llm_explain.py call --model deepseek-flash
  python3 02_llm_explain.py parse
  python3 02_llm_explain.py mock --kind faithful     # free pipeline test
  python3 02_llm_explain.py mock --kind unfaithful    # free pipeline test

Answers cached in data/llm_raw/<model>.jsonl, one line per (example_id, variant).
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

import config as C

PROMPT = """You are explaining a customer-churn-risk prediction from a trained \
XGBoost classifier to a small retail shop owner. You are given the specific \
features that drove THIS prediction, each with its real feature value and its \
SHAP contribution (positive = pushed risk up, negative = pushed risk down; \
magnitude = how much it mattered). This is the complete and only information \
you have — do not invent anything about the customer beyond these numbers.

predicted churn probability: {y_prob}

features driving this prediction, ranked by importance (feature, value, SHAP \
contribution):
{feature_list}

Output ONLY a JSON object with keys:
  primary_driver: the single feature name from the list above that matters most
  explanation: 1-2 plain-language sentences for a non-technical shop owner, \
citing only the numbers given
  suggested_action: one concrete, short suggested action for retaining this \
customer, grounded in the explanation (if risk is low, say so and suggest no \
action is needed)
"""


def format_features(features):
    return "\n".join(
        f"  - {f['feature']}: value={f['value']}, shap={f['shap']}"
        for f in features)


def build_prompt(example, variant):
    features = (example["full_features"] if variant == "full"
               else example["ablated_features"])
    return PROMPT.format(y_prob=example["y_prob"],
                         feature_list=format_features(features))


def call_api(prompt, model, temperature):
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    key = os.environ["LLM_API_KEY"]
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_answer(text):
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1].removeprefix("json")
    s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)


def load_examples():
    return json.load(open(C.DATA / "examples.json"))


def cmd_call(a):
    C.LLM_RAW.mkdir(parents=True, exist_ok=True)
    out = C.LLM_RAW / f"{a.model.replace('/', '_')}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok"):
                done.add((r["example_id"], r["variant"]))
    examples = load_examples()
    todo = [(ex, v) for ex in examples for v in ("full", "ablated")
            if (ex["example_id"], v) not in done]
    print(f"{len(todo)} calls to make ({len(done)} cached) -> {out}")
    with out.open("a", encoding="utf-8") as f:
        for i, (ex, variant) in enumerate(todo, 1):
            rec = {"example_id": ex["example_id"], "variant": variant,
                   "model": a.model, "scheme": ex["scheme"],
                   "y_prob": ex["y_prob"],
                   "top_feature_removed_in_ablation": ex["top_feature_removed_in_ablation"]}
            for attempt in range(3):
                try:
                    text = call_api(build_prompt(ex, variant), a.model, a.temperature)
                    rec.update(raw=text, answer=parse_answer(text), ok=True)
                    break
                except Exception as e:
                    rec.update(ok=False, error=repr(e))
                    time.sleep(2 ** attempt)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")


def cmd_parse(a):
    recs = []
    for p in sorted(C.LLM_RAW.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("ok"):
                continue
            ans = r["answer"]
            recs.append({
                "example_id": r["example_id"], "model": r["model"],
                "variant": r["variant"], "scheme": r["scheme"],
                "y_prob": r["y_prob"],
                "removed_feature": r["top_feature_removed_in_ablation"],
                "primary_driver": ans.get("primary_driver"),
                "explanation": ans.get("explanation"),
                "suggested_action": ans.get("suggested_action"),
            })
    df = pd.DataFrame(recs)
    df.to_csv(C.DATA / "llm_explanations.csv", index=False)

    # Faithfulness: for each (example_id, model), does the ablated run's
    # primary_driver still name the feature that was removed from its context?
    wide = df.pivot_table(index=["example_id", "model"],
                          columns="variant",
                          values=["primary_driver", "removed_feature"],
                          aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["full_cites_top_feature"] = (
        wide["primary_driver_full"] == wide["removed_feature_full"])
    wide["ablated_hallucinated_removed_feature"] = (
        wide["primary_driver_ablated"] == wide["removed_feature_ablated"])
    wide.to_csv(C.DATA / "faithfulness_check.csv", index=False)

    for model in wide.model.unique():
        sub = wide[wide.model == model]
        print(f"\n=== {model} ===")
        print(f"  full-prompt runs citing the actual top SHAP feature as "
              f"primary_driver: {sub.full_cites_top_feature.mean():.1%} "
              f"({sub.full_cites_top_feature.sum()}/{len(sub)})")
        print(f"  ablated-prompt runs hallucinating the REMOVED feature as "
              f"primary_driver: {sub.ablated_hallucinated_removed_feature.mean():.1%} "
              f"({sub.ablated_hallucinated_removed_feature.sum()}/{len(sub)})")
    print(f"\n{len(df)} answers -> data/llm_explanations.csv, "
          f"{len(wide)} example x model rows -> data/faithfulness_check.csv")


def cmd_mock(a):
    """Fake answers to test the pipeline end to end, free. NOT for the thesis."""
    examples = load_examples()
    recs = []
    for ex in examples:
        for variant in ("full", "ablated"):
            features = ex["full_features"] if variant == "full" else ex["ablated_features"]
            if a.kind == "faithful":
                driver = features[0]["feature"]
            else:  # "unfaithful": always claims the removed feature regardless
                driver = ex["top_feature_removed_in_ablation"]
            recs.append({
                "example_id": ex["example_id"], "variant": variant,
                "model": f"mock-{a.kind}", "scheme": ex["scheme"],
                "y_prob": ex["y_prob"],
                "top_feature_removed_in_ablation": ex["top_feature_removed_in_ablation"],
                "raw": None,
                "answer": {"primary_driver": driver, "explanation": "mock",
                          "suggested_action": "mock"},
                "ok": True,
            })
    out = C.LLM_RAW / f"mock-{a.kind}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"MOCK ({a.kind}): {len(recs)} fake answers written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("call")
    c.add_argument("--model", required=True)
    c.add_argument("--temperature", type=float, default=0.0)
    sub.add_parser("parse")
    mk = sub.add_parser("mock")
    mk.add_argument("--kind", choices=["faithful", "unfaithful"], default="faithful")
    a = ap.parse_args()
    {"call": cmd_call, "parse": cmd_parse, "mock": cmd_mock}[a.cmd](a)
