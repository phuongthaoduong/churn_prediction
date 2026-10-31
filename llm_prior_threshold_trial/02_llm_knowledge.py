"""Step 2: ask an LLM what it knows about each merchant.

Input per merchant: name, industry, mode (from data/merchants.csv). Nothing from the
transaction data goes into the prompt, so the answer is external knowledge only.

Usage
  # call an OpenAI-compatible API (DeepSeek, Qwen/DashScope, OpenAI, ...)
  export LLM_BASE_URL=https://api.deepseek.com/v1
  export LLM_API_KEY=sk-...
  python3 02_llm_knowledge.py call --model deepseek-chat --sample pilot --reps 5
  python3 02_llm_knowledge.py call --model deepseek-chat --sample pilot --reps 5 --mask-name   # control: no name

  # merge all cached answers into data/llm_knowledge.csv
  python3 02_llm_knowledge.py parse

  # pipeline test without an API (NOT for the thesis): fake answers
  python3 02_llm_knowledge.py mock --kind oracle    # cycle = true median IPI + noise (should help)
  python3 02_llm_knowledge.py mock --kind industry  # cycle = industry median (should equal no LLM)

Answers are cached in data/llm_raw/<model>[__masked].jsonl, one line per (merchant, rep).
Re-running skips what is already cached, so a stopped run can be resumed.
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import config as C

BUSINESS_TYPES = [
    "农资/种子化肥农药", "兽药/动物保健", "医药/药店", "医疗器械/耗材", "五金/工具", "建材/装修材料",
    "电线电缆/电工", "灯具照明", "机械/工业配件", "汽配/汽车服务", "化工原料/涂料胶粘", "包装材料/印刷",
    "纸品/办公用品", "文具/体育用品", "电脑/数码/手机", "家电", "安防/监控/弱电", "服装/布料/辅料",
    "鞋帽/箱包", "美妆/个护", "饰品/礼品", "母婴用品", "儿童教育/培训/娱乐", "日用百货/超市",
    "酒水饮料", "茶叶", "零食/休闲食品", "粮油/米面/杂粮", "生鲜/冻品", "烟酒/便利店", "家具/家居/家纺",
    "酒店/餐饮用品", "劳保用品", "其他", "无法判断",
]

PROMPT = """你是中国小微零售和批发行业的专家。下面是一家使用进销存软件的商户的信息。
它的“客户”通常是向它进货或购买的个人或小商店。

商户名称：{name}
所属大类：{industry}
经营模式：{mode}

请根据你对这类生意的了解回答（不要编造，信息不足就说明）：
1. business_type：从以下列表中选一个最贴切的：{types}
2. customer_type：主要客户是 "B2B"（店铺/企业进货）、"B2C"（个人消费者）还是 "混合"
3. typical_repurchase_days：这类商户的一个老客户，两次购买之间通常间隔多少天（整数）
4. long_gap_days：一个仍然活跃（没有流失）的老客户，最长可能隔多少天才再来买（整数，约等于正常间隔的第95百分位）
5. seasonal_peak_months：生意旺季的月份（1–12 的整数列表，没有明显旺季则为空列表）
6. confidence：你对以上判断的把握，1（很低）到 5（很高）
7. info_sufficient：仅凭这些信息能否判断这家商户做什么生意（true/false）

只输出一个 JSON 对象，键名如上，不要输出其他文字。"""


def build_prompt(row, mask_name=False):
    name = "（未提供）" if mask_name else row["name"]
    mode = row["mode"] if isinstance(row["mode"], str) and row["mode"] else "未知"
    return PROMPT.format(name=name, industry=row["industry"], mode=mode, types="、".join(BUSINESS_TYPES))


def call_api(prompt, model, temperature):
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    key = os.environ["LLM_API_KEY"]
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_answer(text):
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1].removeprefix("json")
    s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)


def load_sample(which):
    m = pd.read_csv(C.DATA / "merchants.csv", dtype=str)
    if which == "pilot":
        ids = pd.read_csv(C.DATA / "pilot_sample.csv", dtype=str).server_id
        m = m[m.server_id.isin(ids)]
    return m


def cmd_call(a):
    C.LLM_RAW.mkdir(parents=True, exist_ok=True)
    out = C.LLM_RAW / f"{a.model.replace('/', '_')}{'__masked' if a.mask_name else ''}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("ok"):
                done.add((r["server_id"], r["rep"]))
    m = load_sample(a.sample)
    todo = [(row, k) for _, row in m.iterrows() for k in range(a.reps) if (row.server_id, k) not in done]
    print(f"{len(todo)} calls to make ({len(done)} cached) -> {out}")
    with out.open("a", encoding="utf-8") as f:
        for i, (row, k) in enumerate(todo, 1):
            rec = {"server_id": row.server_id, "rep": k, "model": a.model, "masked": a.mask_name}
            for attempt in range(3):
                try:
                    text = call_api(build_prompt(row, a.mask_name), a.model, a.temperature)
                    rec.update(raw=text, answer=parse_answer(text), ok=True)
                    break
                except Exception as e:  # network error or bad JSON: retry, then log
                    rec.update(ok=False, error=repr(e))
                    time.sleep(2 ** attempt)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")


def cmd_parse(a):
    rows = []
    for p in sorted(C.LLM_RAW.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("ok"):
                continue
            ans = r["answer"]
            def num(k):
                try:
                    v = float(ans.get(k))
                    return v if v > 0 else np.nan
                except (TypeError, ValueError):
                    return np.nan
            rows.append({
                "server_id": r["server_id"], "model": r["model"], "masked": r["masked"], "rep": r["rep"],
                "business_type": ans.get("business_type"), "customer_type": ans.get("customer_type"),
                "typical_days": num("typical_repurchase_days"), "long_gap_days": num("long_gap_days"),
                "peak_months": json.dumps(ans.get("seasonal_peak_months", []), ensure_ascii=False),
                "confidence": num("confidence"), "info_sufficient": bool(ans.get("info_sufficient", False)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(C.DATA / "llm_knowledge_reps.csv", index=False)
    agg = (df.groupby(["server_id", "model", "masked"])
             .agg(business_type=("business_type", lambda x: x.mode().iat[0] if len(x.mode()) else None),
                  customer_type=("customer_type", lambda x: x.mode().iat[0] if len(x.mode()) else None),
                  typical_days=("typical_days", "median"), long_gap_days=("long_gap_days", "median"),
                  confidence=("confidence", "median"), info_sufficient=("info_sufficient", "mean"),
                  n_reps=("rep", "size"))
             .reset_index())
    agg.to_csv(C.DATA / "llm_knowledge.csv", index=False)
    print(f"{len(df)} answers -> {len(agg)} merchant x model rows in data/llm_knowledge.csv")


def cmd_mock(a):
    """Fake answers to test steps 3-4 end to end. Never use in the thesis."""
    rng = np.random.default_rng(C.SEED)
    st = pd.read_csv(C.DATA / "merchant_ipi_train.csv", dtype={"server_id": str})
    st = st[st.n_ipi > 0]
    if a.kind == "oracle":
        med, p95 = st.ipi_median, st.ipi_p95
    else:  # industry: every merchant gets its industry's median of merchant values
        med = st.groupby("industry").ipi_median.transform("median")
        p95 = st.groupby("industry").ipi_p95.transform("median")
    rows = []
    for k in range(5):
        noise = np.exp(rng.normal(0, 0.4, len(st)))
        rows.append(pd.DataFrame({"server_id": st.server_id, "model": f"mock-{a.kind}", "masked": False, "rep": k,
                                  "business_type": st.industry, "customer_type": "B2B",
                                  "typical_days": (med * noise).round().clip(lower=1), "long_gap_days": (p95 * noise).round().clip(lower=1),
                                  "peak_months": "[]", "confidence": 3, "info_sufficient": True}))
    df = pd.concat(rows)
    df.to_csv(C.DATA / "llm_knowledge_reps.csv", index=False)
    agg = (df.groupby(["server_id", "model", "masked"])
             .agg(business_type=("business_type", "first"), customer_type=("customer_type", "first"),
                  typical_days=("typical_days", "median"), long_gap_days=("long_gap_days", "median"),
                  confidence=("confidence", "median"), info_sufficient=("info_sufficient", "mean"),
                  n_reps=("rep", "size")).reset_index())
    agg.to_csv(C.DATA / "llm_knowledge.csv", index=False)
    print(f"MOCK ({a.kind}) knowledge for {len(agg)} merchants written. For testing only.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("call")
    c.add_argument("--model", required=True)
    c.add_argument("--sample", choices=["pilot", "all"], default="pilot")
    c.add_argument("--reps", type=int, default=5)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--mask-name", action="store_true")
    sub.add_parser("parse")
    mk = sub.add_parser("mock")
    mk.add_argument("--kind", choices=["oracle", "industry"], default="oracle")
    a = ap.parse_args()
    {"call": cmd_call, "parse": cmd_parse, "mock": cmd_mock}[a.cmd](a)
