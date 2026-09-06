# -*- coding: utf-8 -*-
"""
reproduce_paper_numbers.py — 从公开的两份 CSV 复算论文全部关键数字

复算并核对（对照稿件中的数值）:
  完整性: 1,800 行 / id 唯一且两侧对应 / Table 1 十端点 / 600 单元×3 / 每端点 180
  §3.5: 采集窗口、语料均长与 SD、长输出（>3,000 字符）361/1,800（20.1%）与其余 1,439
  §4.5: Nex-N2-Pro 空输出 170/180（94.4%）、均长、五维地板分
  Table 2: 每端点五维均值 + 95% bootstrap 百分位 CI（10,000 次）
  Table 3: model × task 双因素 ANOVA（Type II）的 F / η² / p
  §4.3: 剔除退化端点后的九端点子集 η²（稿件: 0.017–0.110）

输出: repro_out/verify_report.txt, table2_means_ci.csv, table3_anova.csv
用法: python reproduce_paper_numbers.py <responses.csv> <judge.csv>
"""

import io
import os
import sys

import numpy as np
import pandas as pd

OUT = "repro_out"
os.makedirs(OUT, exist_ok=True)

EXPECTED_MODELS = {
    "LongCat-2.0", "GLM-5.2", "Kimi-K2.7-Code", "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash", "Pro-Kimi-K2.6", "Pro-GLM-5.1", "Nex-N2-Pro",
    "MiniMax-M2.5", "Pro-MiniMax-M2.5",
}
DIMS = ["narrative_quality", "cultural_sensitivity", "historical_accuracy",
        "immersion", "personalization"]
TRUNC = 3000
BOOT = 10000
RNG = np.random.default_rng(20260802)

_buf = io.StringIO()


def P(*a):
    print(*a)
    print(*a, file=_buf)





def chk(label, cond, detail=""):
    P(f"  [{'✔' if cond else '✘'}] {label}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def main(resp_csv, judge_csv):
    resp = pd.read_csv(resp_csv, keep_default_na=False)
    judge = pd.read_csv(judge_csv, keep_default_na=False)
    ok = True

    P("== 完整性 ==")
    ok &= chk("采集/评分各 1,800 行", len(resp) == 1800 and len(judge) == 1800,
              f"{len(resp)}/{len(judge)}")
    ok &= chk("response_id 唯一且两侧对应",
              resp.response_id.is_unique and judge.response_id.is_unique
              and set(resp.response_id) == set(judge.response_id))
    ok &= chk("model 集合 = 稿件 Table 1",
              set(resp.model) == EXPECTED_MODELS)
    cell = resp.groupby(["model", "persona_id", "task"]).size()
    ok &= chk("600 单元 × 每单元 3 条", len(cell) == 600 and (cell == 3).all())
    ok &= chk("每端点 n = 180", (resp.groupby("model").size() == 180).all())
    sc = judge[DIMS].astype(float)
    ok &= chk("评分均为 1–5 整数",
              bool(((sc >= 1) & (sc <= 5) & (sc % 1 == 0)).all().all()))

    P("\n== §3.5 语料 ==")
    ts = pd.to_datetime(resp["timestamp"], utc=True, errors="coerce")
    P(f"  采集窗口: {ts.min()} → {ts.max()}")
    olen = resp["output"].astype(str).str.len()
    P(f"  语料均长 = {olen.mean():.1f}（SD = {olen.std():.1f}）"
      f"  ｜稿件: 1,903（SD 1,086）")
    n_tr = int((olen > TRUNC).sum())
    ok &= chk("长输出 >3,000 字符 361/1,800（20.1%），其余 1,439",
              n_tr == 361, f"实算 {n_tr}（{n_tr/18:.1f}%），其余 {1800-n_tr}")

    P("\n== §4.5 退化端点 ==")
    nx = resp[resp.model == "Nex-N2-Pro"]
    nlen = nx["output"].astype(str).str.len()
    ok &= chk("Nex 空输出 170/180", int((nlen == 0).sum()) == 170,
              f"空 {int((nlen == 0).sum())}，<20 字 {int((nlen < 20).sum())}，"
              f"均长 {nlen.mean():.0f}")
    jn = judge[judge.model == "Nex-N2-Pro"][DIMS].mean()
    P(f"  Nex 五维均值: " + "  ".join(f"{d.split('_')[0][:3].upper()}"
      f"={jn[d]:.2f}" for d in DIMS) + f"  总均 {jn.mean():.2f}")

    P("\n== Table 2 — 每端点五维均值（95% bootstrap CI, 10,000 次）==")
    rows = []
    for m in sorted(judge.model.unique()):
        sub = judge[judge.model == m]
        rec = {"model": m}
        cells = []
        for d in DIMS:
            v = sub[d].to_numpy(float)
            bs = RNG.choice(v, size=(BOOT, len(v)), replace=True).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec[d] = f"{v.mean():.2f} ({lo:.2f}–{hi:.2f})"
            cells.append(rec[d])
        rows.append(rec)
        P(f"  {m:<20} " + " | ".join(cells))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table2_means_ci.csv"),
                              index=False, encoding="utf-8-sig")

    P("\n== Table 3 — model × task 双因素 ANOVA（Type II）==")
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    t3 = []
    for d in DIMS:
        tab = anova_lm(ols(f"{d} ~ C(model)*C(task)", data=judge).fit(), typ=2)
        ss = tab["sum_sq"]
        rec = dict(dimension=d,
                   F_model=round(tab.loc["C(model)", "F"], 1),
                   eta2_model=round(ss["C(model)"] / ss.sum(), 3),
                   F_task=round(tab.loc["C(task)", "F"], 2),
                   p_task=float(tab.loc["C(task)", "PR(>F)"]),
                   F_inter=round(tab.loc["C(model):C(task)", "F"], 2),
                   p_inter=float(tab.loc["C(model):C(task)", "PR(>F)"]))
        t3.append(rec)
        P(f"  {d:<24} F_model={rec['F_model']:>7}  η²={rec['eta2_model']:.3f}"
          f"  F_task={rec['F_task']:>6}  F_int={rec['F_inter']}")
    pd.DataFrame(t3).to_csv(os.path.join(OUT, "table3_anova.csv"),
                            index=False, encoding="utf-8-sig")

    P("\n== §4.2 / Figure 4 — 词汇启发式诊断 ==")
    try:
        from lexical_diagnostics import HEUR_COLS, compute_diagnostics
        diag = compute_diagnostics(resp)
        tot = diag.groupby("model")["stereotype_total"].sum()
        nex_m = int(tot["Nex-N2-Pro"])
        fun = tot.drop("Nex-N2-Pro")
        ok &= chk("Nex 标记总数 = 3", nex_m == 3, f"实算 {nex_m}")
        ok &= chk("正常端点标记 43–120",
                  fun.min() == 43 and fun.max() == 120,
                  f"实算 {int(fun.min())}–{int(fun.max())}")
        hcols = [HEUR_COLS[d] for d in DIMS]
        h = diag.groupby("model")[hcols].mean().mean(axis=1)
        jm = judge.groupby("model")[DIMS].mean().mean(axis=1)
        delta = (h - jm).round(2)
        dfun = delta.drop("Nex-N2-Pro")
        ok &= chk("图4b Δ 正常端点 −1.83 ~ −1.32",
                  dfun.min() == -1.83 and dfun.max() == -1.32,
                  f"实算 {dfun.min():.2f} ~ {dfun.max():.2f}")
        ok &= chk("图4b Δ 退化端点 = +0.63",
                  delta["Nex-N2-Pro"] == 0.63, f"实算 {delta['Nex-N2-Pro']:+.2f}")
        pd.DataFrame({"judge_mean": jm.round(2), "heuristic_mean": h.round(2),
                      "delta": delta, "stereotype_markers": tot}) \
            .sort_values("delta").to_csv(
                os.path.join(OUT, "figure4_values.csv"), encoding="utf-8-sig")
        by_cat = diag.groupby("model")[[f"ste_{c}" for c in
                                        ("exotic", "religious_bias",
                                         "national_bias", "romanticize")]].sum()
        by_cat.to_csv(os.path.join(OUT, "figure4a_by_category.csv"),
                      encoding="utf-8-sig")
        P("  → figure4_values.csv / figure4a_by_category.csv 已写出")
    except ImportError:
        P("  （未找到 lexical_diagnostics.py，跳过）")

    P("\n== §4.3 九端点子集（剔除 Nex）η² ｜稿件: 0.017–0.110 ==")
    j9 = judge[judge.model != "Nex-N2-Pro"]
    e9 = {}
    for d in DIMS:
        tab = anova_lm(ols(f"{d} ~ C(model)*C(task)", data=j9).fit(), typ=2)
        ss = tab["sum_sq"]
        e9[d] = round(ss["C(model)"] / ss.sum(), 3)
    P("  " + "  ".join(f"{d.split('_')[0][:3].upper()}={v}" for d, v in e9.items())
      + f"  区间 {min(e9.values())}–{max(e9.values())}")

    P("\n==>", "全部核对通过，以上数值可直接誊入稿件。" if ok
      else "存在 ✘ 项，先解决再誊数。")
    with open(os.path.join(OUT, "verify_report.txt"), "w", encoding="utf-8") as f:
        f.write(_buf.getvalue())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "llm_cultural_heritage_responses.csv",
         sys.argv[2] if len(sys.argv) > 2 else "llm_judge_scores.csv")
