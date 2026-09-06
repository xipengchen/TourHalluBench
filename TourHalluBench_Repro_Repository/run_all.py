# -*- coding: utf-8 -*-
"""
run_all.py — 一键复现：在公开语料上重算稿件报告的全部数字、表与图

前置：把两份语料 CSV 放在本目录下
    llm_cultural_heritage_responses.csv
    llm_judge_scores.csv

依次执行：
    1. reproduce_paper_numbers.py   核验完整性 + 复算 Table 2/3、Figure 4 数值、§3.5/§4.5 数字
    2. extract_benchmark_materials.py  导出 20 画像定义与 prompt 结构
    3. analysis.py                  词汇/主题/情感/内容特征诊断  → analysis_results/
    4. stats_analysis.py            ANOVA、效应量、混合效应、Tukey → stats_results/
    5. robustness_analysis.py       序数回归、Bradley–Terry、TF-IDF 代理、剔除退化端点重算
    6. truncation_sensitivity.py    截断敏感性
    7. visualization.py             Figure 2–7 → figures_v2/

第二评委验证（附录 D.2）需要 API Key，单独运行：
    export SILICONFLOW_API_KEY=...
    python second_judge_validation.py --pass-no 1 --workers 20   # 依次 1/2/3
    python analyze_second_judge.py
"""

import os
import subprocess
import sys

STEPS = [
    ("复算稿件数字", "reproduce_paper_numbers.py"),
    ("导出基准材料", "extract_benchmark_materials.py"),
    ("词汇/主题诊断", "analysis.py"),
    ("统计检验", "stats_analysis.py"),
    ("稳健性分析", "robustness_analysis.py"),
    ("截断敏感性", "truncation_sensitivity.py"),
    ("图表生成", "visualization.py"),
]

REQUIRED = ["llm_cultural_heritage_responses.csv", "llm_judge_scores.csv"]


def main():
    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        sys.exit(f"缺少输入文件: {missing}\n请把两份语料 CSV 放在本目录下。")

    failed = []
    for i, (label, script) in enumerate(STEPS, 1):
        print(f"\n{'='*64}\n[{i}/{len(STEPS)}] {label} — {script}\n{'='*64}")
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            failed.append(script)
            print(f"  ✘ {script} 未成功（返回码 {r.returncode}）")

    print(f"\n{'='*64}\n完成情况")
    print(f"  成功 {len(STEPS)-len(failed)}/{len(STEPS)}")
    if failed:
        print(f"  失败: {failed}")
    print("\n产物目录：")
    for d in ["repro_out", "analysis_results", "stats_results", "figures_v2", "d2_out"]:
        if os.path.isdir(d):
            print(f"  {d}/  ({len(os.listdir(d))} 个文件)")


if __name__ == "__main__":
    main()
