"""Step 2: ask an LLM to estimate churn probability directly from a
customer's raw transaction sequence, in chronological order and again with
the same transactions shuffled. No merchant/customer identity, no feature
names, no RFM aggregates — just the raw (days_before_pp, bill_amt,
pay_amt, owe_amt) tuples. If the LLM's accuracy depends on order, that's
evidence it's using real sequence structure, not just reading off the
same aggregate statistics XGBoost already has.

Usage
  export LLM_BASE_URL=https://api.deepseek.com/v1
  export LLM_API_KEY=sk-...
  python3 02_llm_sequence_call.py call --model deepseek-flash
  python3 02_llm_sequence_call.py parse
  python3 02_llm_sequence_call.py mock --kind informative   # free pipeline test
  python3 02_llm_sequence_call.py mock --kind order-blind    # free pipeline test
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

LLM_RAW = "data/llm_raw"

PROMPT = """You are estimating the probability that a retail customer will \
make NO purchase in the next 60 days (i.e. will "churn"), using ONLY their \
raw transaction history below. Each entry is one past transaction: \
days_before_pp = how many days before today this transaction happened \
(0 = most recent), bill_amt = amount billed, pay_amt = amount paid, \
owe_amt = amount left owing. You are given nothing else — no customer \
identity, no industry, no computed statistics.

Transaction history, {n} transactions, in the order given below:
{txn_list}

Output ONLY a JSON object with keys:
  churn_probability: a number between 0 and 1
  reasoning: one sentence citing specific numbers from the data above
"""


def format_txns(txns):
    return "\n".join(
        f"  {i+1}. days_before_pp={t['days_before_pp']}, bill_amt={t['bill_amt']}, "
        f"pay_amt={t['pay_amt']}, owe_amt={t['owe_amt']}"
        for i, t in enumerate(txns))


def build_prompt(example, variant):
    txns = example[variant]
    return PROMPT.format(n=len(txns), txn_list=format_txns(txns))


def call_api(prompt, model, temperature):
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    key = os.environ["LLM_API_KEY"]
    # reasoning_effort="none" disables deepseek-flash's default thinking mode.
    # Without this, complex prompts (e.g. 100-transaction sequences) can take
    # 60s+ to generate and fail with IncompleteRead(0 bytes read) — confirmed
    # empirically: the same prompt took 63s and failed with thinking mode on,
    # vs 2.1s and succeeded with it off.
    body = {"model": model, "temperature": temperature, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_answer(text):
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1].removeprefix("json")
    s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)


def load_examples():
    return json.load(open("data/pilot_examples.json"))


def cmd_call(a):
    os.makedirs(LLM_RAW, exist_ok=True)
    out = f"{LLM_RAW}/{a.model.replace('/', '_')}.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok"):
                done.add((r["example_id"], r["variant"]))
    examples = load_examples()
    todo = [(ex, v) for ex in examples for v in ("chronological", "shuffled")
            if (ex["example_id"], v) not in done]
    print(f"{len(todo)} calls to make ({len(done)} cached) -> {out}")
    with open(out, "a", encoding="utf-8") as f:
        for i, (ex, variant) in enumerate(todo, 1):
            rec = {"example_id": ex["example_id"], "variant": variant,
                   "model": a.model, "label_eval": ex["label_eval"],
                   "xgb_y_prob": ex["xgb_y_prob"], "n_txns": ex["n_txns"]}
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
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")


def cmd_parse(a):
    from sklearn.metrics import roc_auc_score
    recs = []
    for fname in sorted(os.listdir(LLM_RAW)):
        if not fname.endswith(".jsonl"):
            continue
        for line in open(f"{LLM_RAW}/{fname}", encoding="utf-8"):
            r = json.loads(line)
            if not r.get("ok"):
                continue
            recs.append({
                "example_id": r["example_id"], "model": r["model"],
                "variant": r["variant"], "label_eval": r["label_eval"],
                "xgb_y_prob": r["xgb_y_prob"], "n_txns": r["n_txns"],
                "llm_churn_prob": r["answer"].get("churn_probability"),
                "reasoning": r["answer"].get("reasoning"),
            })
    df = pd.DataFrame(recs)
    df.to_csv("data/pilot_results.csv", index=False)

    print(f"{len(df)} answers parsed -> data/pilot_results.csv\n")
    for model in df.model.unique():
        sub = df[df.model == model]
        chrono = sub[sub.variant == "chronological"].set_index("example_id")
        shuf = sub[sub.variant == "shuffled"].set_index("example_id")
        common = chrono.index.intersection(shuf.index)
        chrono, shuf = chrono.loc[common], shuf.loc[common]
        y = chrono["label_eval"]
        print(f"=== {model}  (n={len(common)}) ===")
        try:
            print(f"  AUC, LLM chronological-order:  {roc_auc_score(y, chrono.llm_churn_prob):.4f}")
            print(f"  AUC, LLM shuffled-order:       {roc_auc_score(y, shuf.llm_churn_prob):.4f}")
            print(f"  AUC, existing XGBoost (same {len(common)} customers): "
                  f"{roc_auc_score(y, chrono.xgb_y_prob):.4f}")
        except ValueError as e:
            print(f"  AUC undefined: {e}")
        print(f"  class balance (label_eval=1): {y.mean():.1%}")
        print()


def cmd_mock(a):
    """Fake answers to test the pipeline end to end, free. NOT for the thesis."""
    rng = np.random.default_rng(42)
    examples = load_examples()
    os.makedirs(LLM_RAW, exist_ok=True)
    out = f"{LLM_RAW}/mock-{a.kind}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            for variant in ("chronological", "shuffled"):
                if a.kind == "informative":
                    # correlates with true label on chrono, degrades on shuffled
                    base = 0.7 if ex["label_eval"] else 0.2
                    noise = rng.normal(0, 0.05 if variant == "chronological" else 0.3)
                else:  # "order-blind": identical regardless of order
                    base = 0.7 if ex["label_eval"] else 0.2
                    noise = rng.normal(0, 0.05)
                prob = float(np.clip(base + noise, 0.01, 0.99))
                rec = {"example_id": ex["example_id"], "variant": variant,
                      "model": f"mock-{a.kind}", "label_eval": ex["label_eval"],
                      "xgb_y_prob": ex["xgb_y_prob"], "n_txns": ex["n_txns"],
                      "raw": None, "answer": {"churn_probability": prob,
                                              "reasoning": "mock"}, "ok": True}
                f.write(json.dumps(rec) + "\n")
    print(f"MOCK ({a.kind}) written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("call")
    c.add_argument("--model", required=True)
    c.add_argument("--temperature", type=float, default=0.0)
    sub.add_parser("parse")
    mk = sub.add_parser("mock")
    mk.add_argument("--kind", choices=["informative", "order-blind"], default="informative")
    a = ap.parse_args()
    {"call": cmd_call, "parse": cmd_parse, "mock": cmd_mock}[a.cmd](a)
