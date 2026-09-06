# TourHalluBench 复现仓库

## 一、怎么跑

### 准备

把两份语料 CSV 放在本目录下（和脚本同级）：

```
llm_cultural_heritage_responses.csv     # 1,800 条模型输出
llm_judge_scores.csv                    # 1,800 条主评委评分
```

装依赖：

```bash
pip install -r requirements.txt
```

### 一键跑完

```bash
python run_all.py
```

依次执行 7 步，几分钟结束，结果写进 5 个产物目录。跑完打印 `成功 7/7` 即全部通过。

### 也可以单步跑

顺序不能乱——后面的步骤依赖前面的产物：

```bash
python reproduce_paper_numbers.py       # 1. 核验数据 + 复算论文全部数字
python extract_benchmark_materials.py   # 2. 导出 20 个画像定义与 prompt 结构
python analysis.py                      # 3. 词汇/情感/主题/内容特征诊断
python stats_analysis.py                # 4. ANOVA、效应量、混合效应、Tukey
python robustness_analysis.py           # 5. 序数回归、Bradley–Terry、TF-IDF 代理
python truncation_sensitivity.py        # 6. 长输出敏感性
python visualization.py                 # 7. 生成 Figure 2–7
```

### 需要联网和 API Key 的脚本

以上 7 步都是离线的。下面这些会调用 SiliconFlow，**默认不用跑**（结果已在仓库里）：

```bash
export SILICONFLOW_API_KEY=你的key

# 第二评委验证（论文附录 D.2）：60 条 × 3 轮 = 180 次调用
python second_judge_validation.py --pass-no 1 --workers 20
python second_judge_validation.py --pass-no 2 --workers 20
python second_judge_validation.py --pass-no 3 --workers 20
python analyze_second_judge.py          # 算一致性指标，这一步不用 Key

# 重新采集语料（会覆盖现有数据，慎用）
python data_collection.py --check-models    # 先检查端点 ID 是否有效
python data_collection.py                   # 采集 1,800 条，约 1–2 天
python llm_eval.py --check-judge            # 先检查评委端点
python llm_eval.py                          # 评分 1,800 条
```

两个采集脚本都支持断点续跑，中断后重新执行会自动跳过已完成的部分。

### 校验脚本

```bash
python verify_reference_impl.py         # 不联网、不花钱，检查采集/评分脚本与语料是否对得上
```

---

## 二、每个文件夹是什么

### `repro_out/` — 复算论文数字的产物

| 文件 | 内容 |
|---|---|
| `verify_report.txt` | **主报告**。数据完整性核验 + 论文关键数字的复算结果，逐项打勾 |
| `table2_means_ci.csv` | 论文 Table 2：十个端点 × 五维均值 + 95% bootstrap 置信区间 |
| `table3_anova.csv` | 论文 Table 3：双因素 ANOVA 的 F 值、η²、p 值 |
| `figure4_values.csv` | 图 4b 的数值：各端点判分器均分、启发式均分、Δ 差值 |
| `figure4a_by_category.csv` | 图 4a 的数值：各端点四类刻板印象标记数 |
| `personas.csv` | 20 个游客画像的完整字段定义 |
| `prompt_structure.md` | prompt 的真实结构：3 个任务指令块 + 重复生成后缀 + 60 条画像开场白 |
| `prompt_bank.csv` | 60 条基础 prompt（`data_collection.py` 要读它） |

### `analysis_results/` — 文本诊断产物

对 1,800 条输出做的各类文本分析。文件名前的数字是分析步骤编号。

| 文件 | 内容 |
|---|---|
| `10_text_stats.csv` | 字数、句数、token 数等基础统计 |
| `11_sentiment.csv` | 情感极性得分 |
| `12_topic_distribution.csv` / `12_topic_keywords.csv` | LDA 五主题分布与各主题关键词 |
| `13_stereotype.csv` | **刻板印象标记计数**（四子类 + 总分），图 4a 的数据来源 |
| `14_topic_stream.csv` | 主题随端点/任务的分布变化 |
| `15_content_features.csv` | 内容特征二值标记（时间/人物/地点/AR-VR/声音等） |
| `16_immersion_tech.csv` | 沉浸技术词提及统计 |
| `17_word_freq.csv` | 全语料词频 |

### `stats_results/` — 统计检验产物

| 文件 | 内容 |
|---|---|
| `anova_summary.csv` / `twoway_anova.csv` | 单因素、双因素方差分析表 |
| `effect_sizes.csv` | η² 与两两 Cohen's d |
| `mixed_effects.csv` | 混合效应模型（画像随机截距）系数 |
| `posthoc_tukey_*.csv` | 五个维度各自的 Tukey HSD 事后比较（每份 45 对） |
| `bradley_terry.csv` | Bradley–Terry 端点强度评分 |
| `pairwise_winrate.csv` / `pairwise_discrimination.csv` | 配对胜率矩阵与判别率 |
| `exclusion_*.csv` | **剔除退化端点 Nex-N2-Pro 后**重算的同类结果（九端点子集） |
| `ordinal_robustness.csv` | 序数逻辑回归稳健性检验 |
| `truncation_sensitivity.csv` | 长输出敏感性：短输出子集重估 + 长输出协变量 |
| `tfidf_surrogate.csv` | TF-IDF 代理基线（论文附录 E） |
| `forest_plots_95ci.png` | 五维端点均值的 95% CI 森林图 |

### `figures_v2/` — 重新生成的图

`visualization.py` 跑出来的 Figure 2–7，300 DPI。与论文刊印版内容一致，仅分辨率更高。

### `d2_out/` — 第二评委验证（论文 §4.6 / 附录 D.2）

| 文件 | 内容 |
|---|---|
| `d2_report.txt` | **主报告**。评委内稳定性、双评委一致性、D.3 升级判定 |
| `second_judge_scores.csv` | 180 条原始评分（60 条样本 × 3 轮） |
| `d2_sample.csv` | 抽中的 60 条样本及其原文 |
| `d2_agreement.csv` | 逐维一致性指标（κ、Spearman ρ、MAD、within-one-point） |
| `d2_endpoint_ranks.csv` | 两个评委给出的端点排序对照 |

---

## 三、脚本清单

| 脚本 | 作用 |
|---|---|
| `run_all.py` | 一键依次跑完前 7 步 |
| `reproduce_paper_numbers.py` | 核验数据完整性 + 复算论文所有数字 |
| `extract_benchmark_materials.py` | 从语料导出画像定义与 prompt 结构 |
| `lexical_diagnostics.py` | 词汇启发式仪器（词表 + 五维评分公式），被其他脚本导入，不单独运行 |
| `analysis.py` | 文本诊断 |
| `stats_analysis.py` | 统计检验 |
| `robustness_analysis.py` | 稳健性分析 |
| `truncation_sensitivity.py` | 长输出敏感性 |
| `visualization.py` | 生成图表 |
| `second_judge_validation.py` | 第二评委打分（需 Key） |
| `analyze_second_judge.py` | 第二评委一致性分析 |
| `data_collection.py` | 语料采集（需 Key） |
| `llm_eval.py` | 主评委评分（需 Key） |
| `verify_reference_impl.py` | 校验上面两个参考实现与语料的结构一致性 |
