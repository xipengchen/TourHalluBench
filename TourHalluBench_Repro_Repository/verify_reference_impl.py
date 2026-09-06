# -*- coding: utf-8 -*-
"""
verify_reference_impl.py

  1. data_collection.py 的输出列名与列顺序 == 已发布语料
  2. 计划的 1,800 个单元 == 语料中的 (model, persona, task, replicate) 全集
  3. 生成的 prompt 与语料中对应单元的 prompt **逐字一致**（含 replicate 后缀）
  4. llm_eval.py 的输出列名与列顺序 == 已发布评分 CSV
  5. llm_eval.py 不过滤空输出（1,800 条全部送评）
  6. legacy-nan 模式下空输出确实以字符串 "nan" 送评

用法: python verify_reference_impl.py
"""

import sys
from unittest import mock

import pandas as pd

RESP = "llm_cultural_heritage_responses.csv"
JUDGE = "llm_judge_scores.csv"
ok_all = True


def chk(label, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'✔' if cond else '✘'}] {label}" + (f" — {detail}" if detail else ""))


def main():
    corpus = pd.read_csv(RESP, keep_default_na=False)
    judge = pd.read_csv(JUDGE, keep_default_na=False)

    import data_collection as dc
    import llm_eval as ev

    print("== data_collection.py ==")
    chk("输出列顺序与语料一致", dc.FIELDNAMES == list(corpus.columns),
        f"{len(dc.FIELDNAMES)} 列")
    chk("端点集合与语料一致", set(dc.MODELS) == set(corpus["model"]))
    chk("full_model 映射与语料一致",
        all(corpus[corpus.model == m]["full_model"].iloc[0] == full
            for m, full in dc.MODELS.items()))
    chk("温度/最大长度与稿件 §3.5 一致",
        dc.TEMPERATURE == 0.7 and dc.MAX_TOKENS == 2048)

    personas, prompts = dc.load_prompt_bank()
    chk("画像数 = 20", len(personas) == 20)
    chk("基础 prompt 数 = 60", len(prompts) == 60)

    # 计划单元 vs 语料单元
    tasks = ["博物馆导览讲解", "历史场景叙事", "沉浸式体验设计"]
    plan = {(m, p, t, rep) for rep in range(3) for m in dc.MODEL_ORDER
            for p in sorted(personas) for t in tasks}
    actual = set(zip(corpus.model, corpus.persona_id, corpus.task, corpus.replicate))
    chk("1,800 个单元与语料全集相同",
        plan == actual, f"计划 {len(plan)}／语料 {len(actual)}")

    # prompt 逐字一致
    mism = 0
    for row in corpus.itertuples():
        want = dc.build_prompt(prompts[(row.persona_id, row.task)], row.replicate)
        if want != row.prompt:
            mism += 1
    chk("1,800 条 prompt 与语料逐字一致", mism == 0, f"不一致 {mism} 条")

    print("\n== llm_eval.py ==")
    chk("输出列顺序与评分 CSV 一致", ev.FIELDNAMES == list(judge.columns),
        f"{len(ev.FIELDNAMES)} 列")
    chk("不截断送评 + 温度 0.3（与真实管线一致）",
        ev.TRUNCATE_CHARS is None and ev.TEMPERATURE == 0.3)
    chk("评委端点与稿件 §3.6 一致", ev.JUDGE_MODEL == "deepseek-ai/DeepSeek-V4-Pro")

    # 桩掉网络：统计送评条数与空输出送评形态
    sent = []

    def fake_judge(client, text):
        sent.append(str(text))
        return {d: 3 for d in ev.DIMS} | {"reason": "stub"}

    import os
    if os.path.exists("_verify_judge_out.csv"):
        os.remove("_verify_judge_out.csv")          # 确保从干净状态开始
    with mock.patch.object(ev, "call_judge", fake_judge), \
            mock.patch.object(ev, "OpenAI", lambda **kw: object()), \
            mock.patch.dict("os.environ", {"SILICONFLOW_API_KEY": "stub"}):
        ev.OUTPUT_CSV = "_verify_judge_out.csv"
        ev.evaluate(legacy_nan=True)

    chk("全部 1,800 条送评（不过滤空输出）", len(sent) == 1800, f"实际 {len(sent)}")
    n_nan = sum(1 for t in sent if t == "nan")
    n_empty_corpus = int((corpus["output"].astype(str).str.len() == 0).sum())
    chk("空输出以字符串 nan 送评（复刻原管线）",
        n_nan == n_empty_corpus, f"nan {n_nan}／语料空输出 {n_empty_corpus}")

    out = pd.read_csv("_verify_judge_out.csv")
    chk("评分输出 1,800 行且 id 唯一",
        len(out) == 1800 and out.response_id.is_unique)

    if os.path.exists("_verify_judge_out.csv"):
        os.remove("_verify_judge_out.csv")
    print("\n==>", "结构校验全部通过。" if ok_all else "存在未通过项。")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
