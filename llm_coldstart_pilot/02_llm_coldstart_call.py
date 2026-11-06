"""Step 2: ask an LLM to predict whether a cold-start customer (known only
by their first transaction) will make a second purchase within 60 days,
by reasoning over 20 real analog customers from the same store (their
first-transaction signature + their TRUE actual outcome). No customer or
merchant identity anywhere in the prompt.

Pre-registered falsification: compare this LLM prediction's AUC against
(b) the plain cohort-mean of the 20 analogs (no LLM at all) and (c) the
population base rate. If (a) doesn't beat (b), the LLM added nothing a
GROUP BY didn't already give.

Usage
  export LLM_BASE_URL=https://api.deepseek.com/v1
  export LLM_API_KEY=sk-...
  python3 02_llm_coldstart_call.py call --model deepseek-flash
  python3 02_llm_coldstart_call.py parse
  python3 02_llm_coldstart_call.py mock --kind beats-cohort    # free test
  python3 02_llm_coldstart_call.py mock --kind same-as-cohort   # free test
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

LLM_RAW = "data/llm_raw"

PROMPT = """A new customer just made their first-ever purchase at a retail \
shop. You must estimate the probability they will make a SECOND purchase \
within the next 60 days, using ONLY the real data below — no outside \
knowledge about any business or customer.

New customer's first purchase:
  bill_amt={bill_amt}, pay_amt={pay_amt}, owe_amt={owe_amt}, month={month}, \
day_of_week={day_of_week} (0=Monday)

Here are {n} OTHER customers from the SAME shop, each shown with their own \
first purchase AND whether they actually made a second purchase within 60 \
days (the real outcome) — use these as analogs to judge what kind of first \
purchase tends to predict a quick return at this shop:
{analog_list}

Output ONLY a JSON object with keys:
  return_probability: a number between 0 and 1 (probability the new \
customer returns within 60 days)
  reasoning: one sentence citing which analogs most resemble the new \
customer and why
"""


def format_analogs(analogs):
    lines = []
    for i, a in enumerate(analogs):
        s = a["signature"]
        lines.append(
            f"  {i+1}. bill_amt={s['bill_amt']}, pay_amt={s['pay_amt']}, "
            f"owe_amt={s['owe_amt']}, month={s['month']}, "
            f"day_of_week={s['day_of_week']} -> returned_within_60d="
            f"{a['returned_60d']}")
    return "\n".join(lines)


def build_prompt(example):
    s = example["test_signature"]
    return PROMPT.format(
        bill_amt=s["bill_amt"], pay_amt=s["pay_amt"], owe_amt=s["owe_amt"],
        month=s["month"], day_of_week=s["day_of_week"],
        n=len(example["analogs"]), analog_list=format_analogs(example["analogs"]))


def call_api(prompt, model, temperature):
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    key = os.environ["LLM_API_KEY"]
    body = {"model": model, "temperature": temperature, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_answer(text):
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1].removeprefix("json")
    s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)


def load_examples():
    return json.load(open("data/examples.json"))


def cmd_call(a):
    os.makedirs(LLM_RAW, exist_ok=True)
    out = f"{LLM_RAW}/{a.model.replace('/', '_')}.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["example_id"])
    examples = load_examples()
    todo = [ex for ex in examples if ex["example_id"] not in done]
    print(f"{len(todo)} calls to make ({len(done)} cached) -> {out}")
    with open(out, "a", encoding="utf-8") as f:
        for i, ex in enumerate(todo, 1):
            rec = {"example_id": ex["example_id"], "model": a.model,
                   "true_returned_60d": ex["true_returned_60d"],
                   "cohort_mean_prediction": ex["cohort_mean_prediction"],
                   "population_base_rate_prediction": ex["population_base_rate_prediction"]}
            for attempt in range(3):
                try:
                    text = call_api(build_prompt(ex), a.model, a.temperature)
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
                "true_returned_60d": r["true_returned_60d"],
                "cohort_mean_prediction": r["cohort_mean_prediction"],
                "population_base_rate_prediction": r["population_base_rate_prediction"],
                "llm_return_probability": r["answer"].get("return_probability"),
                "reasoning": r["answer"].get("reasoning"),
            })
    df = pd.DataFrame(recs)
    df.to_csv("data/pilot_results.csv", index=False)
    print(f"{len(df)} answers -> data/pilot_results.csv\n")
    for model in df.model.unique():
        sub = df[df.model == model]
        y = sub["true_returned_60d"]
        print(f"=== {model}  (n={len(sub)}, positive rate={y.mean():.1%}) ===")
        print(f"  AUC, LLM-with-analogs:        {roc_auc_score(y, sub.llm_return_probability):.4f}")
        print(f"  AUC, cohort-mean (no LLM):     {roc_auc_score(y, sub.cohort_mean_prediction):.4f}")
        print(f"  AUC, population base rate:     "
              f"{'undefined (constant)' if sub.population_base_rate_prediction.nunique()==1 else roc_auc_score(y, sub.population_base_rate_prediction)}")
        print()


def cmd_mock(a):
    """Fake answers to test the pipeline end to end, free. NOT for the thesis."""
    rng = np.random.default_rng(42)
    examples = load_examples()
    os.makedirs(LLM_RAW, exist_ok=True)
    out = f"{LLM_RAW}/mock-{a.kind}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            cohort = ex["cohort_mean_prediction"]
            if a.kind == "beats-cohort":
                # noisily closer to the true label than the cohort mean is
                base = 0.8 if ex["true_returned_60d"] else 0.2
                prob = float(np.clip(base + rng.normal(0, 0.1), 0.01, 0.99))
            else:  # "same-as-cohort": just echoes the cohort mean with tiny noise
                prob = float(np.clip(cohort + rng.normal(0, 0.02), 0.01, 0.99))
            rec = {"example_id": ex["example_id"], "model": f"mock-{a.kind}",
                  "true_returned_60d": ex["true_returned_60d"],
                  "cohort_mean_prediction": cohort,
                  "population_base_rate_prediction": ex["population_base_rate_prediction"],
                  "raw": None, "answer": {"return_probability": prob, "reasoning": "mock"},
                  "ok": True}
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
    mk.add_argument("--kind", choices=["beats-cohort", "same-as-cohort"], default="same-as-cohort")
    a = ap.parse_args()
    {"call": cmd_call, "parse": cmd_parse, "mock": cmd_mock}[a.cmd](a)
