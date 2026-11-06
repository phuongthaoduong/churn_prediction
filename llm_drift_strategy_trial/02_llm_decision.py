"""Step 2: ask an LLM to make the retrain/hold decision at each of S4's real
decision points, given the exact drift evidence S4 saw (data/decision_points.json).

This tests the DECISION POLICY only — the LLM sees the same evidence S4's
rule saw, holding the evidence-generation trajectory fixed. It never sees
anything about customer identity/knowledge; only numeric drift diagnostics.

Usage
  export LLM_BASE_URL=https://api.deepseek.com/v1
  export LLM_API_KEY=sk-...
  python3 02_llm_decision.py call --model deepseek-flash --reps 5
  python3 02_llm_decision.py parse
  python3 02_llm_decision.py mock --kind mimic-rule   # free pipeline test, NOT for the thesis

Answers cached in data/llm_raw/<model>.jsonl, one line per (decision_point, rep).
Re-running skips what's already cached (resumable).
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

import config as C

PROMPT = """You are a model-maintenance assistant for a customer-churn prediction \
system (XGBoost classifier, retrained periodically on retail transaction data). \
You are shown drift-monitoring evidence at one decision point and must decide \
whether to retrain the model, hold (do nothing), or reset the drift-reference \
batch without retraining.

Context: PSI (Population Stability Index) measures how much a feature's \
distribution has shifted versus a reference batch; PSI > 0.2 on any monitored \
feature is the project's existing rule-of-thumb trigger. The reference batch is \
the last batch the model was trained on. Note: the final evaluation batch in \
this study is known to show a measurement artifact (a sharp late-batch PSI \
spike in "recency" caused by the data boundary cutting off recent purchases \
near the end of the observation window, not necessarily genuine population \
drift) — take this into account when the current decision point IS the final \
batch (2026-03-01) and the spike is concentrated in "recency".

Decision point:
  scheme: {scheme}
  current batch (t): {t}
  current PSI reference batch: {ref_pp}
  months since model was last trained: {months_since_last_train}
  max PSI across monitored features: {psi_max}
  top-5 features by PSI contribution: {top5_psi_features}
  own-label AUC on each matured batch since the last training event, in \
chronological order (last = most recent): {auc_hist}
  is this the final evaluation batch in the study: {is_final_batch}

Decide: hold, retrain, or reset_reference (reset the PSI reference batch to \
the current batch without retraining the model — useful if you believe the \
population has permanently shifted but the current model still performs fine).

Output ONLY a JSON object with keys:
  decision: one of "hold", "retrain", "reset_reference"
  confidence: integer 1 (low) to 5 (high)
  reasoning: one sentence, in English, citing the specific numbers above that \
drove your decision
"""


def build_prompt(row):
    return PROMPT.format(
        scheme=row["scheme"], t=row["t"], ref_pp=row["ref_pp"],
        months_since_last_train=row["months_since_last_train"],
        psi_max=row["psi_max"], top5_psi_features=row["top5_psi_features"],
        auc_hist=row["auc_hist"],
        is_final_batch=(row["t"] == "2026-03-01"),
    )


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


def dp_id(row):
    return f"{row['mode']}|{row['scheme']}|{row['t']}"


def load_decision_points():
    return json.load(open(C.DATA / "decision_points.json"))


def cmd_call(a):
    C.LLM_RAW.mkdir(parents=True, exist_ok=True)
    out = C.LLM_RAW / f"{a.model.replace('/', '_')}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok"):
                done.add((r["dp_id"], r["rep"]))
    rows = load_decision_points()
    todo = [(row, k) for row in rows for k in range(a.reps)
            if (dp_id(row), k) not in done]
    print(f"{len(todo)} calls to make ({len(done)} cached) -> {out}")
    with out.open("a", encoding="utf-8") as f:
        for i, (row, k) in enumerate(todo, 1):
            rec = {"dp_id": dp_id(row), "rep": k, "model": a.model,
                   "mode": row["mode"], "scheme": row["scheme"], "t": row["t"],
                   "s4_actual_action": row["s4_actual_action"]}
            for attempt in range(3):
                try:
                    text = call_api(build_prompt(row), a.model, a.temperature)
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
                "dp_id": r["dp_id"], "model": r["model"], "rep": r["rep"],
                "mode": r["mode"], "scheme": r["scheme"], "t": r["t"],
                "s4_actual_action": r["s4_actual_action"],
                "llm_decision": ans.get("decision"),
                "confidence": ans.get("confidence"),
                "reasoning": ans.get("reasoning"),
            })
    df = pd.DataFrame(recs)
    df.to_csv(C.DATA / "llm_decisions_reps.csv", index=False)
    agg = (df.groupby(["dp_id", "model"])
             .agg(mode=("mode", "first"), scheme=("scheme", "first"),
                  t=("t", "first"), s4_actual_action=("s4_actual_action", "first"),
                  llm_decision=("llm_decision", lambda x: x.mode().iat[0] if len(x.mode()) else None),
                  agreement_rate=("llm_decision", lambda x: (x == x.mode().iat[0]).mean() if len(x.mode()) else np.nan),
                  n_reps=("rep", "size"))
             .reset_index())
    agg.to_csv(C.DATA / "llm_decisions.csv", index=False)
    print(f"{len(df)} answers -> {len(agg)} decision-point x model rows in data/llm_decisions.csv")
    print(agg[["t", "scheme", "mode", "s4_actual_action", "llm_decision", "agreement_rate"]].to_string(index=False))


def cmd_mock(a):
    """Fake answers to test the pipeline end to end, free. NOT for the thesis."""
    rng = np.random.default_rng(C.SEED)
    rows = load_decision_points()
    recs = []
    for row in rows:
        for k in range(5):
            if a.kind == "mimic-rule":
                decision = "retrain" if row["psi_max"] > 0.2 else "hold"
            else:  # "random": ignores evidence entirely
                decision = rng.choice(["hold", "retrain", "reset_reference"], p=[0.8, 0.15, 0.05])
            recs.append({
                "dp_id": dp_id(row), "rep": k, "model": f"mock-{a.kind}",
                "mode": row["mode"], "scheme": row["scheme"], "t": row["t"],
                "s4_actual_action": row["s4_actual_action"],
                "raw": None, "answer": {"decision": decision, "confidence": 3,
                                        "reasoning": "mock"}, "ok": True,
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
    c.add_argument("--reps", type=int, default=5)
    c.add_argument("--temperature", type=float, default=0.0)
    sub.add_parser("parse")
    mk = sub.add_parser("mock")
    mk.add_argument("--kind", choices=["mimic-rule", "random"], default="random")
    a = ap.parse_args()
    {"call": cmd_call, "parse": cmd_parse, "mock": cmd_mock}[a.cmd](a)
