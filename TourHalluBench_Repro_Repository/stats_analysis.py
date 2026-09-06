"""
Statistical analysis — hypothesis testing and effect-size estimation across
the LLM-as-Judge evaluation scores.

Methods:
  1. One-way ANOVA (model as factor) on each of 5 dimensions
  2. Two-way ANOVA (model × task) for interaction effects
  3. Effect sizes: η² (eta-squared) and Cohen's d
  4. Mixed-effects models (model & task as fixed; persona as random)
  5. Pairwise Tukey HSD post-hoc comparisons
  6. 95% CI forest plots for per-model means

Outputs (stats_results/):
  anova_summary.csv       — one-way & two-way ANOVA tables
  effect_sizes.csv        — η² + pairwise Cohen's d
  mixed_effects.csv       — mixed-model coefficients
  posthoc_tukey_*.csv     — pairwise comparisons per dimension
  forest_plots_95ci.png   — aggregated forest plot
"""

import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---- Configuration -----------------------------------------------------------
INPUT_SCORES = "llm_judge_scores.csv"
OUT_DIR = "stats_results"
os.makedirs(OUT_DIR, exist_ok=True)

DIMS = [
    "narrative_quality",
    "cultural_sensitivity",
    "historical_accuracy",
    "immersion",
    "personalization",
]

DIM_LABELS = {
    "narrative_quality": "Narrative Quality",
    "cultural_sensitivity": "Cultural Sensitivity",
    "historical_accuracy": "Historical Accuracy",
    "immersion": "Immersion",
    "personalization": "Personalization",
}

# ---- Colour palette (Okabe-Ito desaturated) ----------------------------------
PALETTE = [
    "#B89B2F",  # deep amber
    "#5FA3C9",  # desaturated sky
    "#2E8B5E",  # forest green
    "#B86B8C",  # desaturated rose
    "#CAB93B",  # deep mustard
    "#8A5AAB",  # desaturated violet
    "#A77B3A",  # ochre
    "#4B7F9C",  # deep blue-gray
    "#5D8A6B",  # olive
]
OUTLIER_COLOR = "#C0392B"
GRID_COLOR = "#E0E0E0"
EDGE_COLOR = "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def _safe_savefig(fig, fname):
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor="white")
    plt.close(fig)


# ---- One-way ANOVA -----------------------------------------------------------
def run_oneway_anova(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for dim in DIMS:
        groups = [g[dim].dropna().values for _, g in df.groupby("model")]
        f_stat, p_val = stats.f_oneway(*groups)
        # η² = SS_model / SS_total
        grand_mean = df[dim].mean()
        ss_total = ((df[dim] - grand_mean) ** 2).sum()
        ss_model = sum(
            len(g) * (g[dim].mean() - grand_mean) ** 2
            for _, g in df.groupby("model")
        )
        eta2 = ss_model / ss_total if ss_total > 0 else 0
        results.append({
            "dimension": dim,
            "F": f_stat,
            "p": p_val,
            "eta_squared": eta2,
            "significant": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns",
        })
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUT_DIR, "anova_summary.csv"), index=False, encoding="utf-8-sig")
    print("[stats] ANOVA → anova_summary.csv")
    return out


# ---- Two-way ANOVA -----------------------------------------------------------
def run_twoway_anova(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for dim in DIMS:
        formula = f"{dim} ~ C(model) + C(task) + C(model):C(task)"
        model = ols(formula, data=df).fit()
        aov = anova_lm(model, typ=2)
        for source in aov.index:
            results.append({
                "dimension": dim,
                "source": source,
                "df": aov.loc[source, "df"],
                "sum_sq": aov.loc[source, "sum_sq"],
                "F": aov.loc[source, "F"],
                "p": aov.loc[source, "PR(>F)"],
            })
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUT_DIR, "twoway_anova.csv"), index=False, encoding="utf-8-sig")
    print("[stats] Two-way ANOVA → twoway_anova.csv")
    return out


# ---- Effect sizes ------------------------------------------------------------
def _cohens_d(x, y):
    """Pooled Cohen's d between two independent samples."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    s_pooled = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / s_pooled if s_pooled > 0 else np.nan

def run_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(df["model"].unique())
    records = []
    for dim in DIMS:
        for m1, m2 in combinations(models, 2):
            x = df[df["model"] == m1][dim].dropna()
            y = df[df["model"] == m2][dim].dropna()
            d = _cohens_d(x, y)
            records.append({
                "dimension": dim,
                "model_1": m1,
                "model_2": m2,
                "cohens_d": d,
                "abs_d": abs(d),
            })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "effect_sizes.csv"), index=False, encoding="utf-8-sig")
    print("[stats] Effect sizes → effect_sizes.csv")
    return out


# ---- Mixed-effects models ----------------------------------------------------
def run_mixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Model: score ~ C(model) + C(task); random intercept per persona."""
    try:
        import statsmodels.formula.api as smf
        results = []
        for dim in DIMS:
            try:
                model = smf.mixedlm(
                    f"{dim} ~ C(model) + C(task)",
                    df,
                    groups=df["persona_id"],
                ).fit(reml=True)
                coefs = model.params.to_dict()
                for k, v in coefs.items():
                    results.append({"dimension": dim, "term": k, "estimate": v})
            except Exception as e:
                print(f"  [MixedLM] {dim} failed: {e}")
        out = pd.DataFrame(results)
        out.to_csv(os.path.join(OUT_DIR, "mixed_effects.csv"), index=False, encoding="utf-8-sig")
        print("[stats] Mixed effects → mixed_effects.csv")
        return out
    except ImportError:
        print("[stats] statsmodels.formula.api not available; skip mixed effects")
        return pd.DataFrame()


# ---- Post-hoc Tukey HSD ------------------------------------------------------
def run_posthoc(df: pd.DataFrame):
    for dim in DIMS:
        try:
            tukey = pairwise_tukeyhsd(df[dim].dropna(), df["model"].astype(str), alpha=0.05)
            rows = []
            for res in tukey.summary().data[1:]:
                rows.append({
                    "group1": res[0],
                    "group2": res[1],
                    "meandiff": res[2],
                    "p": res[3],
                    "lower": res[4],
                    "upper": res[5],
                    "reject": res[6],
                })
            pd.DataFrame(rows).to_csv(
                os.path.join(OUT_DIR, f"posthoc_tukey_{dim}.csv"),
                index=False, encoding="utf-8-sig"
            )
        except Exception as e:
            print(f"  [Tukey] {dim} failed: {e}")
    print("[stats] Tukey HSD complete")


# ---- Forest plot (95% CI) ----------------------------------------------------
def plot_forest(df: pd.DataFrame):
    """Per-model means ± 95% CI for each dimension; one row per dimension."""
    n_dims = len(DIMS)
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=False)
    if n_dims == 1:
        axes = [axes]

    model_list = sorted(df["model"].unique())
    color_map = dict(zip(model_list, PALETTE * (len(model_list) // len(PALETTE) + 1)))

    for ax, dim in zip(axes, DIMS):
        stats_rows = []
        for m in model_list:
            vals = df[df["model"] == m][dim].dropna()
            n = len(vals)
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
            ci = 1.96 * se
            stats_rows.append({"model": m, "mean": mean, "se": se, "ci": ci})

        sdf = pd.DataFrame(stats_rows).sort_values("mean")
        y = np.arange(len(sdf))
        colors = [color_map.get(m, "#888888") for m in sdf["model"]]

        ax.errorbar(sdf["mean"], y, xerr=sdf["ci"], fmt="none",
                    ecolor=GRID_COLOR, capsize=3, zorder=1)
        ax.scatter(sdf["mean"], y, c=colors, s=50, edgecolor=EDGE_COLOR,
                   linewidth=0.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sdf["model"], fontsize=9)
        ax.set_title(DIM_LABELS[dim], fontsize=11, fontweight="bold")
        ax.set_xlabel("Score (1-5)")
        ax.axvline(x=3, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.3)
        ax.set_xlim(0.5, 5.5)

    fig.tight_layout(pad=2)
    _safe_savefig(fig, "forest_plots_95ci.png")
    print("[stats] Forest plot → forest_plots_95ci.png")


# ---- Main --------------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_SCORES):
        raise FileNotFoundError(f"{INPUT_SCORES} not found. Run llm_eval.py first.")

    df = pd.read_csv(INPUT_SCORES)
    print(f"[stats] 加载 {len(df)} 条评分记录")

    run_oneway_anova(df)
    run_twoway_anova(df)
    run_effect_sizes(df)
    run_mixed_effects(df)
    run_posthoc(df)
    plot_forest(df)

    print("\n[stats] 全部完成。结果保存至 stats_results/")


if __name__ == "__main__":
    main()