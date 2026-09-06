# -*- coding: utf-8 -*-
"""
analyze_second_judge.py — 附录 D.2 第二评委验证的一致性分析

输入: d2_out/second_judge_scores.csv（本脚本产出）、公开的主评委评分 CSV
输出: d2_out/d2_report.txt、d2_agreement.csv、d2_endpoint_ranks.csv

报告口径（按稿件附录 D.2 与 D.3）:
  · 评委内稳定性: 同一输出重复评分轮次之间的二次加权 κ、完全一致率、MAD
  · 与主评委一致性: 逐维二次加权 κ、Spearman ρ、MAD、within-one-point 一致率
  · 退化（空）输出与正常输出分开报告
  · 端点配对排序一致率；D.3 规定不一致 >20% 触发升级
"""

import io
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

SEC_CSV = "d2_out/second_judge_scores.csv"
PRI_CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "llm_judge_scores.csv"
DIMS = ["narrative_quality", "cultural_sensitivity", "historical_accuracy",
        "immersion", "personalization"]
_buf = io.StringIO()


def P(*a):
    print(*a)
    print(*a, file=_buf)


def main():
    sec = pd.read_csv(SEC_CSV)
    pri = pd.read_csv(PRI_CSV, keep_default_na=False)
    counts = sec.groupby("pass_no").size().to_dict()
    complete = [p for p, n in counts.items() if n == 60]
    P(f"第二评委: {sec['judge_model'].iloc[0]}")
    P(f"评分轮次完成情况: {counts}；完整轮次 = {complete}")

    # ---- 评委内稳定性 ----
    rows = []
    if len(complete) >= 2:
        a_p, b_p = complete[0], complete[1]
        A = sec[sec.pass_no == a_p].set_index("response_id")
        B = sec[sec.pass_no == b_p].set_index("response_id")
        idx = A.index.intersection(B.index)
        P(f"\n== 评委内稳定性（pass{a_p} vs pass{b_p}, n={len(idx)}）==")
        for d in DIMS:
            x, y = A.loc[idx, d], B.loc[idx, d]
            P(f"  {d:<24} κ_w={cohen_kappa_score(x, y, weights='quadratic'):.3f}"
              f"  exact={np.mean(x == y)*100:.1f}%  MAD={np.abs(x-y).mean():.3f}")

    # ---- 与主评委一致性（第二评委取完整轮次均值）----
    sec_avg = sec[sec.pass_no.isin(complete)].groupby("response_id")[DIMS].mean()
    pri_s = pri.set_index("response_id").loc[sec_avg.index]
    P(f"\n== 与主评委一致性（n={len(sec_avg)}）==")
    for d in DIMS:
        a = pri_s[d].astype(float)
        b = sec_avg[d]
        rows.append(dict(dimension=d,
                         kappa_w=round(cohen_kappa_score(
                             a.astype(int), b.round().astype(int),
                             weights="quadratic"), 3),
                         spearman=round(spearmanr(a, b)[0], 3),
                         MAD=round(np.abs(a - b).mean(), 3),
                         within_one=round(np.mean(np.abs(a - b) <= 1) * 100, 1)))
        P(f"  {d:<24} κ_w={rows[-1]['kappa_w']:.3f}  ρ={rows[-1]['spearman']:.3f}"
          f"  MAD={rows[-1]['MAD']:.3f}  within-1={rows[-1]['within_one']:.1f}%")
    pd.DataFrame(rows).to_csv("d2_out/d2_agreement.csv", index=False,
                              encoding="utf-8-sig")

    # ---- 退化 vs 正常 ----
    lens = sec[sec.pass_no == complete[0]].set_index("response_id")["output_len"]
    empty = lens[lens == 0].index
    func = sec_avg.index.difference(empty)
    P(f"\n== 退化（空）输出 n={len(empty)} ==")
    P(f"  主评委均分 {pri_s.loc[empty, DIMS].astype(float).mean().mean():.2f}"
      f" ｜ 第二评委均分 {sec_avg.loc[empty].mean().mean():.2f}")
    P(f"== 正常输出 n={len(func)} ==")
    P(f"  主评委均分 {pri_s.loc[func, DIMS].astype(float).mean().mean():.2f}"
      f" ｜ 第二评委均分 {sec_avg.loc[func].mean().mean():.2f}")

    # ---- 端点排序与 D.3 ----
    sm = sec[sec.pass_no.isin(complete)].groupby(
        ["response_id", "model"])[DIMS].mean().reset_index()
    s_rank = sm.groupby("model")[DIMS].mean().mean(axis=1)
    p_rank = pri[pri.response_id.isin(set(sec_avg.index))].groupby(
        "model")[DIMS].mean().mean(axis=1)
    models = sorted(s_rank.index)
    tab = pd.DataFrame({"primary": p_rank, "second": s_rank}).sort_values(
        "primary", ascending=False).round(2)
    tab.to_csv("d2_out/d2_endpoint_ranks.csv", encoding="utf-8-sig")
    P("\n== 端点总均分（同一 60 条子样本）==")
    P(tab.to_string())

    pairs = list(itertools.combinations(models, 2))
    agree = sum(1 for a, b in pairs
                if np.sign(p_rank[a]-p_rank[b]) == np.sign(s_rank[a]-s_rank[b]))
    disagree_pct = (1 - agree/len(pairs)) * 100
    m9 = [m for m in models if m != "Nex-N2-Pro"]
    p9 = list(itertools.combinations(m9, 2))
    a9 = sum(1 for a, b in p9
             if np.sign(p_rank[a]-p_rank[b]) == np.sign(s_rank[a]-s_rank[b]))
    P(f"\n== D.3 判定 ==")
    P(f"  全部 {len(pairs)} 对: 一致 {agree} ({agree/len(pairs)*100:.1f}%)，"
      f"不一致 {disagree_pct:.1f}%")
    P(f"  九端点子集 {len(p9)} 对: 一致 {a9} ({a9/len(p9)*100:.1f}%)")
    P(f"  端点排序 Spearman ρ={spearmanr(p_rank[models], s_rank[models])[0]:.3f}"
      f"  Kendall τ={kendalltau(p_rank[models], s_rank[models])[0]:.3f}")
    P(f"  D.3 阈值（不一致 >20% 触发升级）: "
      f"{'触发 — 精细排序须按协议降级处理' if disagree_pct > 20 else '未触发'}")

    with open("d2_out/d2_report.txt", "w", encoding="utf-8") as f:
        f.write(_buf.getvalue())
    print("\n报告已写入 d2_out/d2_report.txt")


if __name__ == "__main__":
    main()
