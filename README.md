
# TourHalluBench — Reproduction Package

Code and result files for **"TourHalluBench"**

Every number, table and figure in the paper is recomputed from two raw corpus files by
the scripts here. Nothing is transcribed by hand.

The benchmark evaluates ten commercial LLM API endpoints on Chinese-language heritage
interpretation: 10 endpoints × 20 visitor personas × 20 World Heritage sites × 3 tasks ×
3 generations = **1,800 outputs**, each scored by three independent evaluator families
(a rubric-based LLM judge, transparent lexical diagnostics, and reference-free factuality
diagnostics).

---

## 1. Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. The offline pipeline needs no API key and no network access.

## 2. Data

Two corpus files are **not included in this repository**:

```
llm_cultural_heritage_responses.csv     # 1,800 model outputs
llm_judge_scores.csv                    # 1,800 primary-judge scores
```

Place both in the repository root before running anything. They are available from the
authors on reasonable request during peer review, and will be deposited in a public
archive with a DOI on acceptance (see the paper's Data Availability statement).

## 3. Run everything

```bash
python run_all.py
```

Ten offline steps, a few minutes end to end. Prints `成功 10/10` when all pass.

Individual steps must run in order — later ones consume earlier outputs:

| # | Script | Produces |
|---|---|---|
| 1 | `reproduce_paper_numbers.py` | integrity checks + every headline number → `repro_out/` |
| 2 | `extract_benchmark_materials.py` | persona definitions, prompt structure → `repro_out/` |
| 3 | `analysis.py` | lexical, sentiment, topic and content diagnostics → `analysis_results/` |
| 4 | `stats_analysis.py` | ANOVA, effect sizes, mixed effects, Tukey HSD → `stats_results/` |
| 5 | `robustness_analysis.py` | ordinal regression, Bradley–Terry, TF-IDF surrogate → `stats_results/` |
| 6 | `truncation_sensitivity.py` | long-output sensitivity → `stats_results/` |
| 7 | `visualization.py` | Figures 2–7 → `figures_v2/` |
| 8 | `new_methods_2026.py` | reference-free diagnostics, degeneracy stress test, integrity gate → `newmethods_out/` |
| 9 | `analyze_three_judges.py` | three-judge agreement at two sample sizes → `d3_out/` |
| 10 | `verify_new_claims.py` | reconciles §4.7 / §4.10 prose against the result files |

`lexical_diagnostics.py` holds the lexicon definitions and the five-dimension heuristic
scoring rules. It is imported by the other scripts and is not run on its own; edit it if you
want to change what the transparent baseline measures.

Two extra utilities:

```bash
python graphical_abstract.py      # the submitted graphical abstract
python verify_reference_impl.py   # checks the collection/scoring scripts against the corpus
```

## 4. Scripts that call the API

The ten steps above are fully offline. These re-collect data and **do not need to be run**
— their outputs are already in the repository:

```bash
export SILICONFLOW_API_KEY=...

python second_judge_validation.py --pass-no 1 --workers 20   # then 2, 3
python analyze_second_judge.py                               # offline
python third_judge_validation.py --workers 3                 # third judge + expand to 150
python persistence_probe.py --workers 4                      # re-probe the failed endpoints

python data_collection.py --check-models                     # validate endpoint IDs first
python data_collection.py                                    # ~1–2 days for 1,800 outputs
python llm_eval.py --check-judge
python llm_eval.py
```

All collection scripts checkpoint and resume; re-running skips completed work. The gateway
enforces a per-model token-rate limit, so keep `--workers` low on the judge scripts — they
wait out a full rate-limit window before retrying.

## 5. Where each result lives

| Directory | Contents |
|---|---|
| `repro_out/` | `verify_report.txt` (main integrity + recomputation report), Table 2 means with bootstrap CIs, ANOVA table, Figure 4 values, persona definitions, prompt bank |
| `analysis_results/` | Text statistics, sentiment, LDA topics, stereotype-marker counts (source of Figure 4a), content-feature flags, word frequencies |
| `stats_results/` | One- and two-way ANOVA, η² and Cohen's *d*, mixed-effects estimates, Tukey HSD per dimension, Bradley–Terry strengths, pairwise win rates, ordinal robustness, TF-IDF surrogate, nine-endpoint re-analysis excluding the degenerate endpoint |
| `newmethods_out/` | Claim extraction, cross-endpoint corroboration (Table 8), degeneracy stress test (Table 3), integrity-gated effect sizes (Table 9), persistence re-probe |
| `d2_out/` | Second-judge validation on 60 outputs (Appendix D.2, stage 1) |
| `d3_out/` | Three-judge validation on 150 outputs (Appendix D.2 stage 2, Table 10) |
| `figures_v2/` | Figures 2–7 at 300 dpi, plus the graphical abstract. The submitted figures are 400 dpi versions of the same content |

Notes on scope that are easy to miss when reading the CSVs directly:

- Claim extraction yields **3,364** instances (2,667 temporal + 697 quantity), but both
  reference-free diagnostics — within-cell self-consistency and cross-endpoint
  corroboration — operate on the **temporal claims only**, so `corroboration_by_model.csv`
  sums to 2,667.
- The degenerate endpoint's mean output length is **69 characters**. Reading the corpus with
  default missing-value conversion turns empty strings into the literal `"nan"` and inflates
  this to 72; the paper reports 69 throughout.
- The three-judge analysis uses **first-pass scores only** for all judges, so that the
  original 60 outputs and the 90 added ones carry identical measurement noise.
- Endpoint effect sizes are reported at four levels of corpus curation. Do not quote the
  full-portfolio column alone — 83.8–92.8% of that effect comes from one failed endpoint.

## 6. Two safeguards in the code

`analyze_three_judges.py` refuses to emit a report unless every judge covers at least 98%
of the sample. Partial coverage from gateway rate limiting once fell disproportionately on
the degenerate endpoint, which would have biased agreement upward while the tables still
looked normal.

`verify_new_claims.py` compares each statistic written in the paper against the result
files and exits non-zero on any mismatch, so prose and data cannot drift apart silently.

## 7. Citation

```bibtex
@article{tourhallubench,
  title   = {TourHalluBench: A Multi-Dimensional Benchmark for LLM Hallucination
             in Cultural Heritage Tourism},
  journal = {Information Processing & Management},
  note    = {Under review},
  year    = {2026}
}
```

## 8. License

*To be confirmed by the authors before public release.* The customary split for this kind
of package is a permissive licence for the code (e.g. MIT) and a data licence for the
corpus and result files (e.g. CC BY 4.0). Add a `LICENSE` file before publishing.

The evaluated model endpoints are third-party commercial services; this repository
redistributes none of their weights, and endpoint names are used descriptively to identify
what was tested at a stated point in time.
