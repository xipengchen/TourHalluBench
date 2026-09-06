# -*- coding: utf-8 -*-
"""
extract_benchmark_materials.py — 从公开语料 CSV 派生论文承诺公开的基准材料

产出（全部由 llm_cultural_heritage_responses.csv 派生，任何人可复算核对）:
  repro_out/personas.csv        — 20 个画像定义（论文 Data availability 承诺项①）
  repro_out/prompt_structure.md — prompt 实际结构: 20×3 画像开场白 + 3 个固定任务
                                  指令块 + 第2/3次重复的多样化后缀（承诺项②）
用法: python extract_benchmark_materials.py <responses.csv>
"""

import os
import re
import sys

import pandas as pd

OUT = "repro_out"
os.makedirs(OUT, exist_ok=True)

TASK_SPLIT = r"(请为我撰写|请以第一人称|请为我设计|请将我正在参观)"
REP_SUFFIX = r"(\[这是第.*?\])$"


def main(csv_path: str):
    resp = pd.read_csv(csv_path, keep_default_na=False)

    # ---- ① 画像定义表 ------------------------------------------------------
    persona_cols = ["persona_id", "nationality", "gender", "age", "budget",
                    "companion", "purpose", "heritage_site", "country",
                    "duration", "profile"]
    personas = (resp.drop_duplicates("persona_id")[persona_cols]
                .sort_values("persona_id").reset_index(drop=True))
    assert len(personas) == 20, len(personas)
    personas.to_csv(os.path.join(OUT, "personas.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"[✔] personas.csv — {len(personas)} 个画像")

    # ---- ② prompt 结构 ------------------------------------------------------
    base = resp[resp.replicate == 0].drop_duplicates(["persona_id", "task"])
    assert len(base) == 60, len(base)

    def split(p):
        p = re.sub(REP_SUFFIX, "", str(p)).strip()
        m = re.search(TASK_SPLIT, p)
        return (p[:m.start()].strip(), p[m.start():].strip())

    parts = base[["persona_id", "task", "prompt"]].copy()
    parts[["preamble", "taskblock"]] = parts["prompt"].apply(
        lambda p: pd.Series(split(p)))

    taskblocks = parts.groupby("task")["taskblock"].agg(
        lambda s: s.unique().tolist())
    for t, blocks in taskblocks.items():
        assert len(blocks) == 1, f"任务块在画像间不一致: {t}"

    suffixes = {}
    for rep in (1, 2):
        s = (resp[resp.replicate == rep]["prompt"].astype(str)
             .str.extract(REP_SUFFIX)[0].dropna().unique())
        assert len(s) == 1, f"rep{rep} 后缀不唯一: {s}"
        suffixes[rep] = s[0]

    lines = [
        "# TourHalluBench prompt 实际结构（由公开语料逐条派生）",
        "",
        "每条 prompt = 画像开场白（画像专属、含轻微任务适配） + 固定任务指令块；",
        "第 2/3 次重复在末尾追加多样化后缀。以下全部内容从语料 CSV 提取，可复核。",
        "",
        "## 一、三个固定任务指令块（全部 20 画像一致）",
    ]
    for t, blocks in taskblocks.items():
        lines += [f"\n### {t}\n", "> " + blocks[0]]
    lines += ["\n## 二、重复生成后缀（追加于 prompt 末尾）\n",
              f"- 第 2 次: > {suffixes[1]}",
              f"- 第 3 次: > {suffixes[2]}",
              "\n## 三、20 × 3 画像开场白（画像专属）\n"]
    for pid in sorted(parts.persona_id.unique()):
        lines.append(f"\n### {pid}")
        for _, r in parts[parts.persona_id == pid].iterrows():
            lines.append(f"- **{r.task}**: {r.preamble}")

    with open(os.path.join(OUT, "prompt_structure.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[✔] prompt_structure.md — 3 任务块 + 2 后缀 + {len(parts)} 条开场白")

    # ---- ③ 60 条基础 prompt（供 data_collection.py 参考实现使用）--------------
    bank = base[["persona_id", "task", "prompt"]].copy()
    bank["prompt"] = bank["prompt"].apply(
        lambda p: re.sub(REP_SUFFIX, "", str(p)).strip())
    bank = bank.sort_values(["persona_id", "task"])
    bank.to_csv(os.path.join(OUT, "prompt_bank.csv"),
                index=False, encoding="utf-8-sig")
    print(f"[✔] prompt_bank.csv — {len(bank)} 条基础 prompt（replicate 0）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "llm_cultural_heritage_responses.csv")
