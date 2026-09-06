# -*- coding: utf-8 -*-
"""
llm_eval.py — 主评委（LLM-as-Judge）评分流程的**

  · 评分 CSV 的列结构与列顺序
  · 全部 1,800 条输出（含 172 条空输出）均有评分记录 → 评分环节**不过滤空输出**
  · 172 条空输出的 reason 为 79 种不同的自然语言表述（如“模型输出为空，无内容可
    评估，所有维度最低分。”）→ 地板分 1 分是**评委实际打出的**，不是程序补分
  · 其中 147 条 reason 明确提到“nan”→ 原管线用 pandas 读取语料时未禁用缺省 NaN
    转换，空字符串被转为 NaN，送评文本实为字符串 "nan"。本实现如实保留这一行为，
    并提供 `--strict-empty` 开关改用真正的空串（见下方说明与 README）。
  · 温度 0.3、max_tokens 512：稿件 §3.6 与真实管线一致
  · 真实管线**不对送评文本做截断**，全文送评；本实现保持一致
    （TRUNCATE_CHARS 保留为可选上限，默认关闭）


关于 "nan" 伪影的处理
--------------------------------------------------------------------------
默认 `--legacy-nan`（与已发布语料一致）：空输出以字符串 "nan" 送评。
可选 `--strict-empty`：以真正的空串送评，是后续研究更干净的做法。
两种模式下评委都会给出地板分，但 reason 文本会不同；如需与已发布语料对齐，
请使用默认模式。稿件 §4.2 与限制部分应如实说明这一点。

用法
--------------------------------------------------------------------------
    export SILICONFLOW_API_KEY=...
    python llm_eval.py --check-judge
    python llm_eval.py                      # 断点续跑，全部 1,800 条送评
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
from openai import OpenAI

BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL_FULL", "deepseek-ai/DeepSeek-V4-Pro")  # 稿件 §3.6

INPUT_CSV = "llm_cultural_heritage_responses.csv"
OUTPUT_CSV = "llm_judge_scores.csv"

TRUNCATE_CHARS = None       # 真实管线不截断；设为整数可启用上限
TEMPERATURE = 0.3           # 稿件 §3.6
TIMEOUT_SECONDS = 120
MAX_RETRIES = 3
FLUSH_EVERY = 20

DIMS = ["narrative_quality", "cultural_sensitivity", "historical_accuracy",
        "immersion", "personalization"]
FIELDNAMES = ["response_id", "model", "persona_id", "task", "heritage_site",
              *DIMS, "reason", "evaluated_at"]

# 量表逐条对应稿件附录 C 表 C1 的 1 / 3 / 5 锚点
RUBRIC_SYSTEM = """你是文化遗产旅游领域的严格评审。请依据下述量表，对给定的“待评文本”\
在五个维度上打 1–5 的整数分，并给出一句中文理由。只输出一个 JSON 对象，不要输出任何其他内容。

维度与锚点（1=最差 / 3=中等 / 5=最佳）:
1. narrative_quality（叙事质量）: 1=空白、语无伦次或毫无结构、没有可辨识的故事; \
3=结构可辨、有一定叙事推进，但文化意涵阐发单薄; 5=结构完整、引人入胜，充分阐发该遗产地或场景的文化意义。
2. cultural_sensitivity（文化敏感性）: 1=刻板化、异域化、含偏见或不敬的框架（或没有内容可供尊重地呈现）; \
3=总体尊重，但偶有笼统、浪漫化或文化扁平的段落; 5=始终尊重、具语境意识，主动避免刻板印象与东方主义框架。
3. historical_accuracy（历史准确性·感知）: 1=自信地呈现捏造或严重错误的事实（或没有事实内容）; \
3=大体准确但有小错或年代含混，几乎不标注不确定性; \
5=年代、人物、事件准确，明确标注不确定性，并在有争议处给出来源线索。
4. immersion（沉浸感）: 1=干瘪、抽象或泛泛，没有感官与情感层次（或没有内容）; \
3=有一定感官细节与投入，但在全文中不够连贯; 5=感官细节丰富且持续，情感投入充分，任务需要时给出具体的互动设计。
5. personalization（个性化）: 1=完全无视画像（或没有内容），千篇一律; \
3=提及画像但仅做表面适配（如问候或选题）; \
5=内容、深度、节奏与实用细节系统性地适配画像的年龄、知识、同行人、预算与陈述需求。

输出格式（严格 JSON）:
{"narrative_quality": 1-5, "cultural_sensitivity": 1-5, "historical_accuracy": 1-5, \
"immersion": 1-5, "personalization": 1-5, "reason": "一句中文理由"}"""


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    for candidate in (m.group(0),
                      m.group(0).replace("，", ",").replace("：", ":")):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def call_judge(client, text: str):
    """送评一条输出（空输出同样送评，不过滤）。"""
    t = str(text) if TRUNCATE_CHARS is None else str(text)[:TRUNCATE_CHARS]
    msg = f"【待评文本开始】\n{t}\n【待评文本结束】"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": RUBRIC_SYSTEM},
                          {"role": "user", "content": msg}],
                temperature=TEMPERATURE, timeout=TIMEOUT_SECONDS)
            s = _extract_json(r.choices[0].message.content)
            if s and all(isinstance(s.get(d), int) and 1 <= s[d] <= 5 for d in DIMS):
                return s
            print(f"      [parse-retry {attempt}] 返回不合法")
        except Exception as e:
            print(f"      [retry {attempt}/{MAX_RETRIES}] {e}")
        time.sleep(2 ** attempt)
    return None


def check_judge(client) -> bool:
    print(f"[check-judge] 评委: {JUDGE_MODEL}")
    s = call_judge(client, "这是一段用于连通性测试的样例讲解词：欢迎来到测试遗产地。")
    if not s:
        print("[check-judge] ✘ 评委不可用或无法返回合法 JSON")
        return False
    print("[check-judge] ✔ 样例评分:", {d: s[d] for d in DIMS})
    return True


def evaluate(legacy_nan: bool):
    key = os.getenv("SILICONFLOW_API_KEY", "")
    if not key:
        sys.exit("SILICONFLOW_API_KEY 未设置")
    if not os.path.exists(INPUT_CSV):
        sys.exit(f"{INPUT_CSV} 不存在，请先完成采集")
    client = OpenAI(api_key=key, base_url=BASE_URL)

    # legacy_nan=True 时复刻原管线：不禁用 NaN 转换，空串→NaN→字符串 "nan"
    df = pd.read_csv(INPUT_CSV) if legacy_nan \
        else pd.read_csv(INPUT_CSV, keep_default_na=False)

    existing = pd.DataFrame()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV).drop_duplicates(
            subset="response_id", keep="first")
    done = set(existing["response_id"]) if len(existing) else set()

    todo = df[~df["response_id"].isin(done)]
    n_empty = int(todo["output"].isna().sum()) if legacy_nan else \
        int((todo["output"].astype(str).str.len() == 0).sum())
    print(f"待评 {len(todo)} 条（其中空输出 {n_empty} 条，同样送评）"
          f"／共 {len(df)} 条；已有 {len(existing)} 条")
    print(f"空输出送评形态: {'字符串 nan（与已发布语料一致）' if legacy_nan else '真正的空串'}")

    rows = []

    def flush():
        snap = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) \
            if rows else existing
        snap = snap.drop_duplicates(subset="response_id", keep="first")
        snap[FIELDNAMES].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        return snap

    for i, (_, r) in enumerate(todo.iterrows(), 1):
        s = call_judge(client, r["output"])
        if s is None:
            print(f"  [FAILED] {r['response_id']}（下次续跑自动补评）")
            continue
        rows.append({"response_id": r["response_id"], "model": r["model"],
                     "persona_id": r["persona_id"], "task": r["task"],
                     "heritage_site": r["heritage_site"],
                     **{d: s[d] for d in DIMS},
                     "reason": s.get("reason", ""),
                     "evaluated_at": datetime.now().isoformat()})
        if len(rows) % FLUSH_EVERY == 0:
            flush()
            print(f"  [{i}/{len(todo)}] 已落盘 {len(existing)+len(rows)} 条")

    final = flush()
    dup = int(final["response_id"].duplicated().sum())
    assert dup == 0, f"出现 {dup} 条重复 response_id"
    print(f"\n完成：共 {len(final)} 条（本次新增 {len(rows)}）→ {OUTPUT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-judge", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--legacy-nan", dest="legacy", action="store_true",
                   help="空输出以字符串 nan 送评（默认，与已发布语料一致）")
    g.add_argument("--strict-empty", dest="legacy", action="store_false",
                   help="空输出以真正的空串送评")
    ap.set_defaults(legacy=True)
    a = ap.parse_args()
    if a.check_judge:
        key = os.getenv("SILICONFLOW_API_KEY", "")
        if not key:
            sys.exit("SILICONFLOW_API_KEY 未设置")
        sys.exit(0 if check_judge(OpenAI(api_key=key, base_url=BASE_URL)) else 1)
    evaluate(a.legacy)
