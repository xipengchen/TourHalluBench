# -*- coding: utf-8 -*-
"""
second_judge_validation.py — 附录 D.2 独立第二评委验证

严格按稿件附录 D.2 规定执行:
  · 分层样本 60 条 = 10 端点 × 3 任务 × 2 条，画像在每个 端点×任务 层内随机抽取
  · Nex-N2-Pro 的空输出保留在样本中（不剔除）
  · 第二评委: 与主评委不同家族/厂商，且本身不在被测 10 端点之列
  · 与主评委相同的量表（Table C1 锚点）与相同的 prompt 结构，端点身份盲化
  · 温度 0.3，每条输出重复评分 3 轮以估计评委内稳定性
输出: d2_out/second_judge_scores.csv（断点续跑）

用法: python second_judge_validation.py [--sample-only]
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI

RESP_CSV = "llm_cultural_heritage_responses.csv"
OUT_DIR = "d2_out"
SAMPLE_CSV = f"{OUT_DIR}/d2_sample.csv"
SCORES_CSV = f"{OUT_DIR}/second_judge_scores.csv"

SECOND_JUDGE = "Qwen/Qwen3.5-397B-A17B"   # 阿里家族；不在被测 10 端点内
BASE_URL = "https://api.siliconflow.cn/v1"
TEMPERATURE = 0.3
PASSES = 3
TRUNCATE = 3000
SEED = 20260802
DIMS = ["narrative_quality", "cultural_sensitivity", "historical_accuracy",
        "immersion", "personalization"]

# 与主评委一致的量表（逐条对应稿件 Table C1 的 1/3/5 锚点）
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


def build_sample() -> pd.DataFrame:
    """10 端点 × 3 任务 × 2 条；层内随机选画像；空输出保留。"""
    resp = pd.read_csv(RESP_CSV, keep_default_na=False)
    parts = []
    for (m, t), g in resp.groupby(["model", "task"]):
        parts.append(g.sample(n=2, random_state=SEED))
    picked = pd.concat(parts).reset_index(drop=True)
    assert len(picked) == 60, len(picked)
    cols = ["response_id", "model", "persona_id", "task", "heritage_site", "output"]
    picked = picked[cols]
    os.makedirs(OUT_DIR, exist_ok=True)
    picked.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
    return picked


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(0).replace("，", ",").replace("：", ":"))
        except json.JSONDecodeError:
            return None


def call_judge(client, text):
    """端点身份盲化：只送文本本身，不含任何模型/画像元数据。"""
    t = "" if text is None else str(text)
    truncated = len(t) > TRUNCATE
    t = t[:TRUNCATE]
    msg = f"【待评文本开始】\n{t}\n【待评文本结束】"
    for attempt in range(1, 4):
        try:
            r = client.chat.completions.create(
                model=SECOND_JUDGE,
                messages=[{"role": "system", "content": RUBRIC_SYSTEM},
                          {"role": "user", "content": msg}],
                temperature=TEMPERATURE, timeout=180)
            s = _extract_json(r.choices[0].message.content or "")
            if s and all(isinstance(s.get(d), int) and 1 <= s[d] <= 5 for d in DIMS):
                s["_truncated"] = truncated
                return s
        except Exception as e:
            print(f"      [retry {attempt}] {e}")
        time.sleep(2 ** attempt)
    return None


def main(sample_only=False, only_pass=None, workers=8):
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = build_sample()
    print(f"分层样本: {len(sample)} 条 | 每端点 {len(sample)//10} 条 | "
          f"空输出 {int((sample['output'].astype(str).str.len()==0).sum())} 条")
    if sample_only:
        print(sample.groupby(["model"]).size().to_string())
        return

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        sys.exit("SILICONFLOW_API_KEY 未设置")
    client = OpenAI(api_key=key, base_url=BASE_URL)

    done = set()
    if os.path.exists(SCORES_CSV):
        old = pd.read_csv(SCORES_CSV)
        done = set(zip(old["response_id"], old["pass_no"]))
        print(f"已有 {len(old)} 条评分，续跑")

    todo = [(r, p) for p in range(1, PASSES + 1) for _, r in sample.iterrows()
            if (r["response_id"], p) not in done]
    if only_pass:
        todo = [(r, p) for (r, p) in todo if p == only_pass]
    print(f"待评 {len(todo)} 次调用")

    def work(item):
        r, p = item
        s = call_judge(client, r["output"])
        if s is None:
            return None
        return {"response_id": r["response_id"], "pass_no": p,
                "model": r["model"], "task": r["task"],
                "persona_id": r["persona_id"],
                **{d: s[d] for d in DIMS},
                "reason": s.get("reason", ""),
                "judge_model": SECOND_JUDGE,
                "output_len": len(str(r["output"])),
                "truncated": bool(s["_truncated"]),
                "evaluated_at": datetime.now(timezone.utc).isoformat()}

    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(work, todo), 1):
            if res:
                rows.append(res)
            if i % 20 == 0:
                print(f"  进度 {i}/{len(todo)}，成功 {len(rows)}")
    if rows:
        snap = pd.concat([pd.read_csv(SCORES_CSV), pd.DataFrame(rows)],
                         ignore_index=True) if os.path.exists(SCORES_CSV) \
            else pd.DataFrame(rows)
        snap.drop_duplicates(subset=["response_id", "pass_no"]).to_csv(
            SCORES_CSV, index=False, encoding="utf-8-sig")
    print(f"本轮新增 {len(rows)}；累计 "
          f"{len(pd.read_csv(SCORES_CSV)) if os.path.exists(SCORES_CSV) else 0}/180")
    print("完成 →", SCORES_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--pass-no", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.sample_only, a.pass_no, a.workers)
