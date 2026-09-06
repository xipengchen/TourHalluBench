"""Truncation sensitivity analysis (fifth/sixth review, item 4).

Checks whether the 3,000-character judge-input truncation biases endpoint comparisons:
  A. Re-estimate endpoint means on untruncated outputs; Spearman rank correlation
     vs full-sample means (nine functioning endpoints) + functioning-subset ANOVA eta^2.
  B. Add a truncation indicator to endpoint-and-task OLS regressions
     (persona-clustered SEs) on the full sample.
"""
import pandas as pd, numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scs

DIMS = {"NQ": "narrative_quality", "CS": "cultural_sensitivity", "HA": "historical_accuracy",
        "IMM": "immersion", "PER": "personalization"}
DEGEN = "Nex-N2-Pro"

jf = pd.read_csv("llm_judge_scores.csv")
resp = pd.read_csv("llm_cultural_heritage_responses.csv")
resp["_len"] = resp["output"].fillna("").astype(str).str.len()
m = jf.merge(resp[["response_id", "_len"]], on="response_id", how="left")
m["truncated"] = (m["_len"] > 3000).astype(int)
func = [x for x in m["model"].unique() if x != DEGEN]
sub = m[m["truncated"] == 0]
rows = []
for dk, col in DIMS.items():
    means_full = m.groupby("model")[col].mean()
    means_sub = sub.groupby("model")[col].mean()
    rho = scs.spearmanr(means_full.loc[func], means_sub.loc[func]).statistic
    fs = sub[sub["model"].isin(func)]
    aov = smf.ols(f"{col} ~ C(model)", data=fs).fit()
    an = sm.stats.anova_lm(aov, typ=2)
    eta2 = an["sum_sq"]["C(model)"] / (an["sum_sq"]["C(model)"] + an["sum_sq"]["Residual"])
    fit = smf.ols(f"{col} ~ C(model) + C(task) + truncated", data=m).fit(
        cov_type="cluster", cov_kwds={"groups": m["persona_id"]})
    rows.append({"dimension": dk, "n_untruncated": len(sub),
                 "spearman_full_vs_untruncated": round(rho, 3),
                 "subset_anova_F": round(an["F"]["C(model)"], 2),
                 "subset_eta2": round(eta2, 3),
                 "trunc_coef": round(fit.params["truncated"], 3),
                 "trunc_p": f"{fit.pvalues['truncated']:.3g}"})
out = pd.DataFrame(rows)
out.to_csv("stats_results/truncation_sensitivity.csv", index=False)
print(out.to_string(index=False))