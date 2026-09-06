# -*- coding: utf-8 -*-
"""
robustness_analysis.py — Q1-review robustness add-on analyses for TourHalluBench
================================================================================
Runs three complementary analyses on top of llm_judge_scores.csv /
llm_cultural_heritage_responses.csv and writes CSV results to stats_results/:

1. ordinal_robustness.csv
   Ordinal logistic regression (statsmodels OrderedModel, logit link) of each
   1-5 judge dimension on model + task dummies, with persona-clustered standard
   errors.  Added in response to the review comment that ordinal 1-5 ratings
   should not rely on ANOVA alone.  Reports the coefficient and p-value of the
   degenerate endpoint (Nex-N2-Pro) relative to the reference model, plus
   convergence status.

2. bradley_terry.csv / pairwise_winrate.csv / pairwise_discrimination.csv
   Pairwise comparison analysis addressing the judge "ceiling effect" comment:
   within every matched scenario cell (persona x task; the heritage site is
   nested within persona, so each persona maps to exactly one site), the three
   stochastic replications of each endpoint are first averaged, and all
   C(10,2) = 45 endpoint pairs are then compared on these cell means
   (ties = 0.5 win), giving 60 x 45 = 2,700 matched pairs including the
   degenerate endpoint.  From the pair table we derive (i) a Bradley-Terry
   strength rating per model (minorization-maximization algorithm, ratings
   normalized to mean 1 for identification), (ii) the full pairwise win-rate
   matrix, and (iii) the per-dimension discrimination rate, i.e. the share of
   matched pairs whose scores actually differ — a direct measure of how much
   signal survives the near-ceiling means.

3. tfidf_surrogate.csv
   A stronger automated baseline than the keyword heuristic: TF-IDF features
   over jieba tokens (max_features=3000, min_df=3) + Ridge regression per
   dimension under persona-grouped 5-fold cross-validation (GroupKFold by persona, so near-duplicate prompts cannot leak across train/test splits).  Reports Spearman/Pearson
   correlations and MAE of out-of-fold predictions against the judge scores.

4. exclusion_anova.csv / exclusion_bradley_terry.csv /
   exclusion_pairwise_winrate.csv / exclusion_pairwise_discrimination.csv
   Model-quality analysis on the nine functioning endpoints (degenerate
   endpoint excluded): one-way ANOVA effect sizes per dimension, Bradley-Terry
   ratings over the 60 x 36 = 2,160 matched pairs, and per-dimension
   discrimination rates.  Distinguishes operational portfolio effects
   (10 endpoints, dominated by the quality-control failure) from genuine
   model-quality differentiation among functioning systems.

Usage:  python robustness_analysis.py        (run inside the pipeline folder)
Requires: pandas, numpy, statsmodels, scikit-learn, scipy, jieba
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SR = os.path.join(BASE, "stats_results")
os.makedirs(SR, exist_ok=True)

DIMS = ["narrative_quality", "cultural_sensitivity", "historical_accuracy",
        "immersion", "personalization"]
DIM_EN = dict(zip(DIMS, ["Narrative Quality", "Cultural Sensitivity",
                         "Historical Accuracy", "Immersion", "Personalization"]))
GRP = ["persona_id", "task", "heritage_site"]


def load():
    judge = pd.read_csv(os.path.join(BASE, "llm_judge_scores.csv"))
    judge["overall"] = judge[DIMS].mean(axis=1)
    resp = pd.read_csv(os.path.join(BASE, "llm_cultural_heritage_responses.csv"))
    return judge, resp


# ---------------------------------------------------------------------------
# 1. ordinal logistic regression with persona-clustered SEs
# ---------------------------------------------------------------------------
def ordinal_robustness(judge):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    rows = []
    for dim in DIMS:
        y = judge[dim].astype(int)
        X = pd.get_dummies(judge[["model", "task"]], drop_first=True).astype(float)
        m = OrderedModel(y, X, distr="logit")
        r = m.fit(method="bfgs", disp=False, maxiter=500, cov_type="cluster",
                  cov_kwds={"groups": judge["persona_id"]})
        rows.append({
            "dimension": DIM_EN[dim],
            "Nex_N2_Pro_coef": r.params.get("model_Nex-N2-Pro"),
            "Nex_N2_Pro_pvalue": r.pvalues.get("model_Nex-N2-Pro"),
            "converged": bool(r.mle_retvals.get("converged")),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SR, "ordinal_robustness.csv"), index=False)
    print("[1] ordinal_robustness.csv"); return out


# ---------------------------------------------------------------------------
# 2. pairwise win-rate, Bradley-Terry ratings, discrimination rate
# ---------------------------------------------------------------------------
def pairwise(judge, tag=""):
    models = sorted(judge["model"].unique())
    midx = {m: i for i, m in enumerate(models)}
    n = len(models)

    def pair_stats(col):
        W = np.zeros((n, n)); N = np.zeros((n, n)); disc = 0; tot = 0
        for _, g in judge.groupby(GRP):
            s = g.groupby("model")[col].mean()
            ms = s.index.tolist()
            for a in range(len(ms)):
                for b in range(a + 1, len(ms)):
                    sa, sb = s.iloc[a], s.iloc[b]
                    i, j = midx[ms[a]], midx[ms[b]]
                    tot += 1
                    if sa == sb:
                        wa = wb = 0.5
                    else:
                        wa, wb = (1.0, 0.0) if sa > sb else (0.0, 1.0)
                        disc += 1
                    W[i, j] += wa; W[j, i] += wb
                    N[i, j] += 1;   N[j, i] += 1
        return W, N, disc, tot

    W, N, disc, tot = pair_stats("overall")

    # Bradley-Terry via minorization-maximization
    g = np.ones(n)
    for _ in range(2000):
        gn = g.copy()
        for i in range(n):
            num = W[i].sum()
            den = sum((N[i, j] + N[j, i]) / (g[i] + g[j])
                      for j in range(n) if j != i and N[i, j] > 0)
            gn[i] = num / den if den > 0 else g[i]
        if np.max(np.abs(gn - g)) < 1e-10:
            g = gn; break
        g = gn
    g = g / g.mean()
    bt = pd.DataFrame({"model": models, "BT_score": g.round(3),
                       "wins": W.sum(axis=1).astype(int)}) \
           .sort_values("BT_score", ascending=False)
    bt.to_csv(os.path.join(SR, f"{tag}bradley_terry.csv"), index=False)

    wr_vals = np.divide(W, N, out=np.full_like(W, np.nan, dtype=float), where=N > 0)
    wr = pd.DataFrame(wr_vals.round(4), index=models, columns=models)
    wr.to_csv(os.path.join(SR, f"{tag}pairwise_winrate.csv"))

    rows = []
    for dim in DIMS + ["overall"]:
        _, _, dc, tt = pair_stats(dim)
        rows.append({"dimension": DIM_EN.get(dim, "Overall"), "pairs": tt,
                     "disc_rate_pct": round(100 * dc / tt, 1)})
    disc_df = pd.DataFrame(rows)
    disc_df.to_csv(os.path.join(SR, f"{tag}pairwise_discrimination.csv"), index=False)
    print(f"[2] {tag}bradley_terry.csv / {tag}pairwise_winrate.csv / {tag}pairwise_discrimination.csv")
    return bt, wr, disc_df


# ---------------------------------------------------------------------------
# 3. TF-IDF + Ridge surrogate baseline (5-fold CV)
# ---------------------------------------------------------------------------
def tfidf_surrogate(judge, resp):
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr, pearsonr

    m = resp.merge(judge[["response_id"] + DIMS + ["overall"]],
                   on="response_id", how="inner")
    texts = m["output"].fillna("").astype(str).tolist()

    def tok(t):
        return [w for w in jieba.lcut(t) if len(w.strip()) > 1]

    Xv = TfidfVectorizer(tokenizer=tok, token_pattern=None,
                         max_features=3000, min_df=3).fit_transform(texts)
    # GroupKFold by persona: all outputs of a visitor persona stay in the same
    # fold, so near-duplicate prompts cannot leak across train/test splits.
    gkf = GroupKFold(5)
    groups = m["persona_id"].values
    rows = []
    for dim in DIMS + ["overall"]:
        y = m[dim].values.astype(float)
        pred = np.zeros_like(y)
        for tr, te in gkf.split(Xv, y, groups):
            r = Ridge(alpha=1.0)
            r.fit(Xv[tr], y[tr])
            pred[te] = r.predict(Xv[te])
        rows.append({"dimension": DIM_EN.get(dim, "Overall"),
                     "spearman": round(spearmanr(y, pred).statistic, 3),
                     "pearson": round(pearsonr(y, pred).statistic, 3),
                     "mae": round(np.abs(y - pred).mean(), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SR, "tfidf_surrogate.csv"), index=False)
    print("[3] tfidf_surrogate.csv"); return out


# ---------------------------------------------------------------------------
# 4. functioning-endpoint subset (model-quality analysis, degenerate endpoint
#    excluded): ANOVA effect sizes, Bradley-Terry ratings, discrimination rates
# ---------------------------------------------------------------------------
def exclusion_analysis(judge):
    from scipy import stats as sst
    jf = judge[judge["model"] != "Nex-N2-Pro"].copy()
    rows = []
    for dim in DIMS + ["overall"]:
        groups = [g[dim].values for _, g in jf.groupby("model")]
        F, p = sst.f_oneway(*groups)
        grand = jf[dim].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((jf[dim] - grand) ** 2).sum()
        rows.append({"dimension": DIM_EN.get(dim, "Overall"), "F": round(F, 2),
                     "p_value": p, "eta_squared": round(ss_b / ss_t, 3)})
    anova9 = pd.DataFrame(rows)
    anova9.to_csv(os.path.join(SR, "exclusion_anova.csv"), index=False)

    bt9, wr9, disc9 = pairwise(jf, tag="exclusion_")
    print("[4] exclusion_anova / exclusion_bradley_terry / "
          "exclusion_pairwise_winrate / exclusion_pairwise_discrimination")
    return anova9, bt9, disc9


if __name__ == "__main__":
    JUDGE, RESP = load()
    print(ordinal_robustness(JUDGE).to_string(index=False))
    bt, wr, disc = pairwise(JUDGE)
    print(bt.to_string(index=False)); print(disc.to_string(index=False))
    print(tfidf_surrogate(JUDGE, RESP).to_string(index=False))
    a9, b9, d9 = exclusion_analysis(JUDGE)
    print(a9.to_string(index=False)); print(b9.to_string(index=False))
    print(d9.to_string(index=False))
    print("all robustness results ->", SR)