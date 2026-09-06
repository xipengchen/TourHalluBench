# -*- coding: utf-8 -*-
"""
data_collection.py — 语料采集流程的**


  · 语料 CSV 的列结构、列顺序与 response_id 格式（12 位十六进制）
  · 60 条画像开场白与 3 个固定任务指令块（逐字取自 prompt 列）
  · 第 2/3 次生成所附加的多样化指令（逐字取自 prompt 列）
  · 由 timestamp 列还原出的循环嵌套与轮询结构（见下）
  · 由 timestamp 间隔还原出的调用节奏与 5 次断点续跑

从语料还原出的执行结构
--------------------------------------------------------------------------
  外层：按固定顺序轮询 10 个端点（LongCat-2.0 → GLM-5.2 → Kimi-K2.7-Code →
        DeepSeek-V4-Pro → DeepSeek-V4-Flash → Pro-Kimi-K2.6 → Pro-GLM-5.1 →
        Nex-N2-Pro → MiniMax-M2.5 → Pro-MiniMax-M2.5），每轮为每个端点采集一
        个区块，全程共 41 个区块、5 次会话中断后续跑，故十个端点的采集时间窗
        **两两 100% 重叠**（这一点与稿件限制 (12) 的原表述不同，见 README）。
  内层：replicate（0→1→2） → persona（P01→P20） → task（导览→叙事→体验）

用法
--------------------------------------------------------------------------
    export SILICONFLOW_API_KEY=...
    python data_collection.py --check-models     # 先核对端点 ID
    python data_collection.py                    # 采集（可断点续跑）
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime

import pandas as pd
from openai import OpenAI

BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# 端点短名 → 网关完整 ID（取自语料的 model / full_model 两列，逐一对应）
MODELS = {
    "LongCat-2.0":       "meituan-longcat/LongCat-2.0",
    "GLM-5.2":           "zai-org/GLM-5.2",
    "Kimi-K2.7-Code":    "moonshotai/Kimi-K2.7-Code",
    "DeepSeek-V4-Pro":   "deepseek-ai/DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    "Pro-Kimi-K2.6":     "Pro/moonshotai/Kimi-K2.6",
    "Pro-GLM-5.1":       "Pro/zai-org/GLM-5.1",
    "Nex-N2-Pro":        "nex-agi/Nex-N2-Pro",
    "MiniMax-M2.5":      "MiniMaxAI/MiniMax-M2.5",
    "Pro-MiniMax-M2.5":  "Pro/MiniMaxAI/MiniMax-M2.5",
}
MODEL_ORDER = list(MODELS)          # 轮询顺序，与语料中的时间序一致

TEMPERATURE = 0.7                   # 与真实管线一致（稿件 §3.5）
MAX_TOKENS = 2048                   # 稿件 §3.5
REPLICATES = 3
PACING_SECONDS = 0.5
TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

PERSONAS_CSV = "repro_out/personas.csv"          # 由 extract_benchmark_materials.py 导出
PROMPTS_CSV = "repro_out/prompt_bank.csv"        # 同上（60 条开场白 + 任务块）
OUTPUT_CSV = "llm_cultural_heritage_responses.csv"
CHECKPOINT = "collection_checkpoint.json"

# 语料的列顺序（逐字对齐已发布的 CSV）
FIELDNAMES = ["response_id", "model", "full_model", "persona_id", "profile",
              "nationality", "gender", "age", "budget", "companion", "purpose",
              "heritage_site", "country", "duration", "task", "prompt",
              "output", "timestamp", "replicate"]

# 第 2/3 次生成附加的多样化指令（逐字取自语料 prompt 列）
REGEN_SUFFIX = ("\n[这是第 {n} 次生成，请在不重复前面内容的前提下，"
                "换一个角度、风格或细节重新生成一份全新的回答。]")


def load_prompt_bank():
    """读取 60 条 (persona, task) 基础 prompt 与画像字段。"""
    if not (os.path.exists(PERSONAS_CSV) and os.path.exists(PROMPTS_CSV)):
        sys.exit(f"缺少 {PERSONAS_CSV} 或 {PROMPTS_CSV}；"
                 f"请先运行 extract_benchmark_materials.py")
    personas = pd.read_csv(PERSONAS_CSV, keep_default_na=False) \
        .set_index("persona_id").to_dict("index")
    bank = pd.read_csv(PROMPTS_CSV, keep_default_na=False)
    prompts = {(r.persona_id, r.task): r.prompt for r in bank.itertuples()}
    return personas, prompts


def build_prompt(base_prompt: str, replicate: int) -> str:
    """replicate 0 用基础 prompt；1/2 追加多样化指令。"""
    if replicate == 0:
        return base_prompt
    return base_prompt + REGEN_SUFFIX.format(n=replicate + 1)


def check_models(client) -> bool:
    try:
        available = {m.id for m in client.models.list().data}
    except Exception as e:
        print(f"[check-models] 无法获取模型列表: {e}")
        return False
    ok = True
    for short, full in MODELS.items():
        hit = full in available
        ok &= hit
        print(f"  {'✔' if hit else '✘'} {short:<20} {full}")
    print("\n[check-models]", "全部有效，可以采集。" if ok else "存在无效端点 ID，请先修正。")
    return ok


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            return {tuple(k) for k in json.load(f)["completed"]}
    return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(map(list, done)),
                   "last_updated": datetime.now().isoformat()},
                  f, ensure_ascii=False, indent=1)


def call_model(client, full_model: str, prompt: str) -> str:
    """重试后仍失败返回空串——空输出是 RQ4 的研究对象，必须保留。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=full_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                timeout=TIMEOUT_SECONDS)
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"      [retry {attempt}/{MAX_RETRIES}] {full_model}: {e}")
            time.sleep(2 ** attempt)
    return ""


def collect():
    key = os.getenv("SILICONFLOW_API_KEY", "")
    if not key:
        sys.exit("SILICONFLOW_API_KEY 未设置")
    client = OpenAI(api_key=key, base_url=BASE_URL)
    personas, prompts = load_prompt_bank()
    pids = sorted(personas)
    tasks = ["博物馆导览讲解", "历史场景叙事", "沉浸式体验设计"]

    done = load_checkpoint()
    # 还原出的循环顺序：replicate → model（轮询）→ persona → task
    plan = [(m, p, t, rep)
            for rep in range(REPLICATES)
            for m in MODEL_ORDER
            for p in pids
            for t in tasks]
    todo = [k for k in plan if k not in done]
    print(f"计划 {len(plan)} 条，已完成 {len(plan)-len(todo)} 条，本次待采 {len(todo)} 条")

    new_file = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            w.writeheader()
        for i, (m, pid, task, rep) in enumerate(todo, 1):
            prompt = build_prompt(prompts[(pid, task)], rep)
            out = call_model(client, MODELS[m], prompt)
            pr = personas[pid]
            w.writerow({
                "response_id": uuid.uuid4().hex[:12],
                "model": m, "full_model": MODELS[m], "persona_id": pid,
                "profile": pr["profile"], "nationality": pr["nationality"],
                "gender": pr["gender"], "age": pr["age"], "budget": pr["budget"],
                "companion": pr["companion"], "purpose": pr["purpose"],
                "heritage_site": pr["heritage_site"], "country": pr["country"],
                "duration": pr["duration"], "task": task, "prompt": prompt,
                "output": out, "timestamp": datetime.now().isoformat(),
                "replicate": rep,
            })
            f.flush()
            done.add((m, pid, task, rep))
            if i % 20 == 0 or i == len(todo):
                save_checkpoint(done)
                print(f"  [{i}/{len(todo)}] {m} × {pid} × {task} rep{rep}（{len(out)} 字）")
            time.sleep(PACING_SECONDS)
    save_checkpoint(done)

    df = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
    print("\n===== 各端点采集摘要（静默失效在此当场可见）=====")
    for m in MODEL_ORDER:
        sub = df[df.model == m]
        if not len(sub):
            continue
        ln = sub["output"].astype(str).str.len()
        flag = "  ⚠ 疑似静默失效" if (ln < 20).mean() > 0.5 else ""
        print(f"  {m:<20} n={len(sub):<5} 空={int((ln==0).sum()):<4} "
              f"均长={ln.mean():.0f}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-models", action="store_true")
    a = ap.parse_args()
    if a.check_models:
        key = os.getenv("SILICONFLOW_API_KEY", "")
        if not key:
            sys.exit("SILICONFLOW_API_KEY 未设置")
        sys.exit(0 if check_models(OpenAI(api_key=key, base_url=BASE_URL)) else 1)
    collect()
