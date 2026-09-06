# -*- coding: utf-8 -*-
"""
Visualization — regenerates the manuscript result figures (Figure 2 … Figure 8)
with the ORIGINAL composite layouts and the ORIGINAL data logic.

Data logic
----------
All numbers are computed in-script from the two raw artefacts of the executed
pipeline:

  llm_cultural_heritage_responses.csv   (1 800 model outputs)
  llm_judge_scores.csv                  (1 800 judged rows, 5 dimensions)

The lexical / heuristic diagnostics (stereotype markers, heuristic dimension
scores, count-based sentiment polarity, content-coverage flags, LDA topics and
the top-20 term frequencies) are ported 1:1 from analysis.py so that every
panel reproduces the values shown in the original manuscript figures:

  Fig 2  response-length box plot · sentiment violins · top-20 terms
  Fig 3  topic distribution (LDA, 5 topics) · immersive-element coverage
  Fig 4  stereotype markers · judge-vs-heuristic dumbbell
  Fig 5  overall performance dots · 7×7 correlation matrix
  Fig 6  per-dimension rankings · per-dimension mean bars
  Fig 7  model × task heat maps for the five judge dimensions
  Fig 8  evidence-at-a-glance summary (hand-composed, kept from previous rev.)

Outputs: figures_v2/Figure_2.png … Figure_7.png  (300 dpi)
"""

import os
import re
import warnings
from collections import Counter

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as _fm

warnings.filterwarnings("ignore")

# ============================ base style (Fig. 8 uses it) =====================
_AVAILABLE = {f.name for f in _fm.fontManager.ttflist}
SERIF = [c for c in ["Times New Roman", "Liberation Serif",
                     "Nimbus Roman", "DejaVu Serif"] if c in _AVAILABLE] or ["serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#333333",
    "axes.grid": False,
    "savefig.facecolor": "white",
})

# Figures 2-7 use the original sans-serif look
SANS_RC = {"font.family": "sans-serif",
           "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
           "mathtext.fontset": "dejavusans"}

TEXT_COLOR = "#333333"
GRID_COLOR = "#E0E0E0"
NEUTRAL = "#7F8C8D"

# ---- original per-model palette (order as in the original plotting script) --
MODEL_COLORS = {
    "LongCat-2.0":       "#B89B2F",
    "GLM-5.2":           "#2E8B5E",
    "Kimi-K2.7-Code":    "#B86B8C",
    "DeepSeek-V4-Pro":   "#5FA3C9",
    "DeepSeek-V4-Flash": "#CAB93B",
    "Pro-Kimi-K2.6":     "#4B7F9C",
    "Pro-GLM-5.1":       "#A77B3A",
    "Nex-N2-Pro":        "#C0392B",
    "MiniMax-M2.5":      "#8A5AAB",
    "Pro-MiniMax-M2.5":  "#5D8A6B",
}
# x-axis order used by Figure 2 (length box plot / sentiment violins)
FIG2_MODEL_ORDER = list(MODEL_COLORS.keys())

TASK_COLORS = {"Museum Guide": "#4B7F9C", "Historical Storytelling": "#A77B3A",
               "Immersive Design": "#2E8B5E"}
TASK_EN = {"博物馆导览讲解": "Museum Guide", "历史场景叙事": "Historical Storytelling",
           "沉浸式体验设计": "Immersive Design"}
TASK_ORDER = ["Historical Storytelling", "Immersive Design", "Museum Guide"]

DIMS = ["narrative_quality", "cultural_sensitivity", "historical_accuracy",
        "immersion", "personalization"]
DIM_EN = {"narrative_quality": "Narrative Quality",
          "cultural_sensitivity": "Cultural Sensitivity",
          "historical_accuracy": "Perceived Historical Accuracy",
          "immersion": "Immersion", "personalization": "Personalization"}

# Figure 4a category palette (as in the original panel)
CATEGORY_COLORS = {"Exoticism": "#B89B2F", "Religious Bias": "#5FA3C9",
                   "National Bias": "#2E8B5E", "Romanticization": "#B86B8C"}

# Figure 4b dumbbell colours (sampled from the original figure)
DB_JUDGE = "#1783FE"      # blue dot  — LLM judge
DB_HEUR  = "#C9A869"      # gold diamond — heuristic diagnostic
DB_NEG   = "#CA5E57"      # red connector when Δ < 0
DB_POS   = "#9CA6B3"      # blue-grey connector when Δ ≥ 0

CMAP_ACADEMIC = LinearSegmentedColormap.from_list(
    "academic_blue", ["#F7F9FB", "#D6E4F0", "#7FB3D5", "#3A6D9C"])
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "academic_diverging", ["#4B7F9C", "#E8EEF3", "#C0392B"])

BASE = os.path.dirname(os.path.abspath(__file__))
AR_DIR = os.path.join(BASE, "analysis_results")
OUT_DIR = os.path.join(BASE, "figures_v2")
os.makedirs(OUT_DIR, exist_ok=True)


def _spine_off(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel(ax, label):
    ax.text(-0.03, 1.05, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
            ha="left", va="bottom")


def _models(df):
    return sorted(df["model"].unique())


# =============================================================================
# Ported diagnostic logic (verbatim from analysis.py — produces exactly the
# values behind the original manuscript figures)
# =============================================================================
NARRATIVE_KEYWORDS = {
    "intro": ["欢迎来到", "大家好", "今天", "我们将", "让我们", "首先", "开头", "引言"],
    "story": ["故事", "传说", "曾经", "那时", "有一天", "一位", "主人公", "经历", "命运"],
    "conclusion": ["总之", "综上所述", "希望", "收获", "记住", "结语", "最后", "总结"],
    "cultural_meaning": ["意义", "价值", "象征", "代表", "传承", "精神", "身份", "认同", "文化遗产"],
}

STEREOTYPE_KEYWORDS = {
    "exotic": ["神秘", "异域", "原始", "落后", "野蛮", "未开化", "神秘东方", "东方奇观"],
    "religious_bias": ["迷信", "狂热", "宗教狂热", "落后信仰"],
    "national_bias": ["侵略", "野蛮民族", "低等", "文明冲突"],
    "romanticize": ["高贵野蛮人", "消失的文明", "纯净的"],
}
RESPECTFUL_KEYWORDS = ["尊重", "保护", "传承", "当地社区", "原住民", "多元", "包容", "理解"]

HISTORY_PATTERNS = {
    "date": r"\d{1,4}年|公元前|\d{1,4}世纪|公元\d{1,4}",
    "figure": r"皇帝|国王|女王|皇帝|王后|将军|学者|艺术家|建筑师|工匠|统治者",
    "event": r"战争|革命|统一|建国|灭亡|发现|建造|修建|重建|保护|发掘",
}
UNCERTAINTY_KEYWORDS = ["可能", "大概", "约", "左右", "据说", "推测", "学术界", "存在争议"]
SOURCE_KEYWORDS = ["考古", "文献", "记载", "研究", "学者", "博物馆", "官方", "遗址", "保护机构"]

SENSORY_KEYWORDS = {
    "visual": ["看见", "金色", "红色", "光芒", "色彩", "雕刻", "壁画", "景象"],
    "auditory": ["听到", "钟声", "风声", "音乐", "歌声", "喧嚣", "寂静"],
    "olfactory": ["闻到", "香气", "檀香", "泥土", "气息", "味道"],
    "tactile": ["触摸", "冰冷", "粗糙", "光滑", "质感", "温度"],
}
EMOTION_KEYWORDS = ["震撼", "感动", "敬畏", "怀旧", "好奇", "激动", "沉思", "悲伤", "温暖", "孤独"]
IMMERSION_TECH_KEYWORDS = ["AR", "VR", "虚拟现实", "增强现实", "全息", "声音", "音乐", "灯光",
                           "互动", "角色扮演", "穿越", "沉浸"]

PERSONALIZATION_KEYWORDS = {
    "age_child": ["孩子", "小朋友", "亲子", "童话", "简单", "趣味"],
    "age_youth": ["年轻人", "学生", "背包客", "自由", "探索"],
    "age_senior": ["年长", "退休", "悠闲", "舒适", "慢节奏"],
    "family": ["家庭", "孩子", "亲子", "全家", "陪伴", "互动"],
    "couple": ["情侣", "浪漫", "两人", "爱人", "伴侣"],
    "budget_luxury": ["私人", "定制", "高端", "VIP", "豪华"],
    "budget_budget": ["免费", "优惠", "经济", "节省", "性价比", "公共交通"],
}

CONTENT_QUALITY_PATTERNS = {
    "has_time_reference": r"\d{1,4}年|世纪|朝代|时期|时代",
    "has_person_reference": r"皇帝|国王|王后|将军|学者|艺术家|工匠|僧人|统治者|居民",
    "has_place_reference": r"殿|宫|寺|塔|陵|城|遗址|窟|博物馆|展厅|广场|山|河",
    "has_ar_vr": r"AR|VR|增强现实|虚拟现实|全息|元宇宙|数字孪生",
    "has_sound_music": r"声音|音乐|音效|钟声|鼓声|BGM|声景",
    "has_interaction": r"互动|参与|体验|角色扮演|解谜|问答|投票|触摸|操作",
    "has_preservation": r"保护|修复|文物|遗产|可持续|限制|禁止|文明参观",
}

POSITIVE_WORDS = [
    "震撼", "感动", "惊叹", "优美", "壮丽", "精美", "珍贵", "难忘", "迷人", "推荐",
    "值得一去", "值得", "享受", "愉悦", "欣赏", "赞叹", "美好", "温暖", "浪漫", "有趣",
]
NEGATIVE_WORDS = [
    "遗憾", "失望", "破坏", "损毁", "消失", "遗忘", "战争", "苦难", "悲惨", "悲伤",
    "错误", "不准确", "避免", "注意", "危险", "拥挤", "限制", "禁止", "不要", "不能",
]

LDA_STOP_WORDS = set([
    "一个", "可以", "推荐", "建议", "适合", "选择", "行程", "旅游", "旅行", "游客",
    "这里", "非常", "如果", "进行", "前往", "下午", "上午", "晚上", "中午", "一天",
    "你们", "我们", "他们", "这个", "那个", "这些", "那些", "然后", "之后", "以及",
    "体验", "文化", "历史", "遗产", "博物馆", "通过", "能够", "对于", "来说",
])


def _simple_sentiment(text):
    """Count-based polarity, exactly as analysis.py (occurrence counts)."""
    text = str(text)
    pos = sum(text.count(w) for w in POSITIVE_WORDS)
    neg = sum(text.count(w) for w in NEGATIVE_WORDS)
    return (pos - neg) / (pos + neg + 1)


def compute_diagnostics(resp):
    """Attach every ported diagnostic column to the responses frame."""
    df = resp.copy()
    out = df["output"].fillna("").astype(str)

    # --- narrative-quality heuristic -----------------------------------------
    for dim, kws in NARRATIVE_KEYWORDS.items():
        df[f"nar_{dim}"] = out.apply(lambda x: sum(1 for kw in kws if kw in x))
    df["nar_structure_score"] = (
        ((df["nar_intro"] > 0).astype(int) + (df["nar_conclusion"] > 0).astype(int)) / 2 * 4 + 1
    ).clip(1, 5).round(0)
    df["nar_story_score"] = df["nar_story"].clip(0, 5)
    df["nar_meaning_score"] = df["nar_cultural_meaning"].clip(1, 5)
    df["narrative_quality_heuristic"] = (
        df["nar_structure_score"] + df["nar_story_score"] + df["nar_meaning_score"]) / 3

    # --- cultural-sensitivity heuristic + stereotype markers ------------------
    for cat, kws in STEREOTYPE_KEYWORDS.items():
        df[f"ste_{cat}"] = out.apply(lambda x: sum(1 for kw in kws if kw in x))
    df["stereotype_total"] = df[[f"ste_{c}" for c in STEREOTYPE_KEYWORDS]].sum(axis=1)
    df["respectful_count"] = out.apply(lambda x: sum(1 for kw in RESPECTFUL_KEYWORDS if kw in x))
    df["cultural_sensitivity_heuristic"] = (
        5 - df["stereotype_total"].clip(0, 4) + df["respectful_count"].clip(0, 2) * 0.5
    ).clip(1, 5).round(2)

    # --- historical-accuracy heuristic ----------------------------------------
    for dim, pattern in HISTORY_PATTERNS.items():
        df[f"hist_{dim}"] = out.apply(lambda x: len(re.findall(pattern, x)))
    df["hist_uncertainty"] = out.apply(lambda x: sum(1 for kw in UNCERTAINTY_KEYWORDS if kw in x))
    df["hist_source"] = out.apply(lambda x: sum(1 for kw in SOURCE_KEYWORDS if kw in x))
    df["historical_accuracy_heuristic"] = (
        (df["hist_date"].clip(0, 3) + df["hist_figure"].clip(0, 2) + df["hist_event"].clip(0, 2)) / 7 * 3
        + df["hist_uncertainty"].clip(0, 2) / 2
        + df["hist_source"].clip(0, 3) / 3
        + 1
    ).clip(1, 5).round(2)

    # --- immersion heuristic ----------------------------------------------------
    for sense, kws in SENSORY_KEYWORDS.items():
        df[f"sensory_{sense}"] = out.apply(lambda x: sum(1 for kw in kws if kw in x))
    df["sensory_total"] = df[[f"sensory_{s}" for s in SENSORY_KEYWORDS]].sum(axis=1)
    df["emotion_count"] = out.apply(lambda x: sum(1 for kw in EMOTION_KEYWORDS if kw in x))
    df["immersion_tech_count"] = out.apply(lambda x: sum(1 for kw in IMMERSION_TECH_KEYWORDS if kw in x))
    df["immersion_heuristic"] = (
        df["sensory_total"].clip(0, 6) / 6 * 2
        + df["emotion_count"].clip(0, 4) / 4 * 1.5
        + df["immersion_tech_count"].clip(0, 4) / 4 * 1.5
    ).clip(1, 5).round(2)

    # --- personalization heuristic --------------------------------------------
    for cat, kws in PERSONALIZATION_KEYWORDS.items():
        df[f"per_{cat}"] = out.apply(lambda x: sum(1 for kw in kws if kw in x))
    df["personalization_heuristic"] = (
        df["per_age_child"] + df["per_age_youth"] + df["per_age_senior"]
        + df["per_family"] + df["per_couple"]
        + df["per_budget_luxury"] + df["per_budget_budget"]
    ).clip(1, 5).round(2)

    # --- content-coverage flags -------------------------------------------------
    for col, pattern in CONTENT_QUALITY_PATTERNS.items():
        df[col] = out.apply(lambda x: 1 if re.search(pattern, x) else 0)

    # --- count-based sentiment polarity -----------------------------------------
    df["sentiment_score"] = out.apply(_simple_sentiment)

    # --- output length -----------------------------------------------------------
    df["char_count"] = out.str.len()
    return df


HEUR_COLS = {"narrative_quality": "narrative_quality_heuristic",
             "cultural_sensitivity": "cultural_sensitivity_heuristic",
             "historical_accuracy": "historical_accuracy_heuristic",
             "immersion": "immersion_heuristic",
             "personalization": "personalization_heuristic"}


def compute_topics(resp, n_topics=5):
    """LDA topic assignment, verbatim port of analysis.py (jieba + sklearn)."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    def _tok(text):
        return [w.strip() for w in jieba.lcut(str(text)) if len(w.strip()) > 1]

    corpus = resp["output"].fillna("").apply(
        lambda t: " ".join(w for w in _tok(t) if w not in LDA_STOP_WORDS and len(w) > 1)).tolist()
    vec = CountVectorizer(max_features=500)
    dtm = vec.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    lda.fit(dtm)
    return lda.transform(dtm).argmax(axis=1) + 1


def top_terms(resp, n=20):
    """Top-n content terms with the curated stop list + English glosses."""
    text = " ".join(resp["output"].fillna("").astype(str))
    words = [w for w in jieba.lcut(text) if len(w) > 1]
    stop = {"一个", "可以", "推荐", "建议", "适合", "选择", "行程", "旅游", "旅行", "游客",
            "这里", "非常", "如果", "进行", "前往", "以及", "体验", "文化", "历史", "遗产",
            "博物馆", "通过", "能够", "对于", "来说", "他们", "我们", "你们", "这个", "那个",
            "将", "在", "了", "是", "的", "与", "及", "等", "和", "或", "有", "没有", "为",
            "而是", "不是", "一位", "这座", "作为", "这些", "今天", "看到", "一种", "不仅",
            "如何", "就是", "现在", "那些", "不同", "提供", "避免", "目标", "使用", "来自",
            "自己", "分钟"}
    words = [w for w in words if w not in stop
             and not re.match(r"^[\d\s]+$", w) and not re.match(r"^[\W_]+$", w)]
    top = Counter(words).most_common(n)
    EN = {"AR": "AR", "叙事": "Narrative", "互动": "Interaction", "场景": "Scene",
          "声音": "Sound", "孩子": "Children", "技术": "Technology", "设计": "Design",
          "情感": "Emotion", "时间": "Time", "建筑": "Architecture", "故事": "Story",
          "空间": "Space", "工匠": "Craftsmen", "核心": "Core", "手机": "Smartphone",
          "金字塔": "Pyramid", "沉浸": "Immersion", "理解": "Understanding",
          "文明": "Civilization", "文物": "Artifacts", "女性": "Women",
          "对话": "Dialogue", "音频": "Audio", "遗址": "Ruins", "环节": "Session"}
    return [(EN.get(t, t), c) for t, c in top]


# =============================================================================
# Load data
# =============================================================================
def load_all():
    judge = pd.read_csv(f"{BASE}/llm_judge_scores.csv")
    resp = pd.read_csv(f"{BASE}/llm_cultural_heritage_responses.csv")
    resp = compute_diagnostics(resp)
    resp["task_en"] = resp["task"].map(lambda x: TASK_EN.get(str(x), str(x)))
    judge["task_en"] = judge["task"].map(lambda x: TASK_EN.get(str(x), str(x)))
    resp["dominant_topic"] = compute_topics(resp)
    # frames kept for the Fig. 8 evidence panel (pipeline artefact values)
    se = (pd.read_csv(f"{AR_DIR}/11_sentiment.csv")
          if os.path.exists(f"{AR_DIR}/11_sentiment.csv") else None)
    im = (pd.read_csv(f"{AR_DIR}/16_immersion_tech.csv")
          if os.path.exists(f"{AR_DIR}/16_immersion_tech.csv") else None)
    if se is not None:
        se["task_en"] = se["task"].map(lambda x: TASK_EN.get(str(x), str(x)))
    if im is not None:
        im["task_en"] = im["task"].map(lambda x: TASK_EN.get(str(x), str(x)))
    return judge, resp, se, im


JUDGE, RESP, SE, IM = load_all()
MODELS = _models(JUDGE)


# =============================================================================
# Figure 2 — response length · sentiment violins · top-20 terms
# =============================================================================
def fig2():
    with plt.rc_context(SANS_RC):
        fig = plt.figure(figsize=(12.4, 10.0))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                              hspace=0.52, wspace=0.30,
                              left=0.07, right=0.985, top=0.93, bottom=0.075)

        # (top) response length by model and task ------------------------------
        axa = fig.add_subplot(gs[0, :])
        sns.boxplot(data=RESP, x="model", y="char_count", hue="task_en", ax=axa,
                    order=FIG2_MODEL_ORDER, hue_order=list(TASK_COLORS.keys()),
                    palette=TASK_COLORS, width=0.62, fliersize=2.5, linewidth=1.1)
        axa.set_title("Response Length by Model and Task", fontsize=15, pad=14)
        axa.set_xlabel("Model", fontsize=12, labelpad=10)
        axa.set_ylabel("Length (characters)", fontsize=12)
        axa.set_xticklabels(axa.get_xticklabels(), rotation=20, ha="right", fontsize=10.5)
        axa.legend(title="Task", loc="center left", bbox_to_anchor=(1.005, 0.5),
                   frameon=False, fontsize=11, title_fontsize=12)
        axa.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axa.set_axisbelow(True); _spine_off(axa)

        # (bottom-left) sentiment polarity violins ------------------------------
        axb = fig.add_subplot(gs[1, 0])
        sns.violinplot(data=RESP, x="model", y="sentiment_score", ax=axb,
                       order=FIG2_MODEL_ORDER, palette=MODEL_COLORS,
                       inner="quart", linewidth=1.0)
        axb.axhline(0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
        axb.set_title("Sentiment Polarity Distribution by Model", fontsize=14, pad=14)
        axb.set_xlabel("Model", fontsize=12, labelpad=10)
        axb.set_ylabel("Sentiment Score", fontsize=12)
        axb.set_xticklabels(axb.get_xticklabels(), rotation=20, ha="right", fontsize=10)
        axb.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axb.set_axisbelow(True); _spine_off(axb)

        # (bottom-right) most frequent terms -------------------------------------
        axc = fig.add_subplot(gs[1, 1])
        top = top_terms(RESP, 20)
        terms = [t for t, _ in top][::-1]
        counts = [c for _, c in top][::-1]
        y = np.arange(len(terms))
        axc.barh(y, counts, color="#4B7F9C", edgecolor="white", height=0.72)
        axc.set_yticks(y); axc.set_yticklabels(terms, fontsize=10)
        axc.set_xlabel("Frequency", fontsize=12)
        axc.set_title("Most Frequent Terms in LLM Heritage Narratives",
                      fontsize=14, pad=14)
        for yi, c in enumerate(counts):
            axc.text(c + max(counts) * 0.012, yi, f"{c}", va="center",
                     fontsize=8.5, color=TEXT_COLOR)
        axc.set_xlim(0, max(counts) * 1.13)
        axc.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axc.set_axisbelow(True); _spine_off(axc)

        fig.savefig(f"{OUT_DIR}/Figure_2.png", bbox_inches="tight")
        plt.close(fig); print("saved Figure_2")


# =============================================================================
# Figure 3 — topic distribution · immersive-element coverage
# =============================================================================
def fig3():
    with plt.rc_context(SANS_RC):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.8, 3.7),
                                       gridspec_kw={"width_ratios": [1.12, 1.0],
                                                    "wspace": 0.52})
        fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.20)

        # (left) 100 % stacked topic distribution -------------------------------
        dist = (RESP.groupby("model")["dominant_topic"]
                .value_counts(normalize=True).unstack(fill_value=0) * 100)
        dist = dist.reindex(MODELS)
        topic_cols = sorted(dist.columns)
        palette = sns.color_palette("viridis", len(topic_cols))
        bottom = np.zeros(len(dist))
        x = np.arange(len(dist))
        for tc, color in zip(topic_cols, palette):
            vals = dist[tc].values
            axa.bar(x, vals, bottom=bottom, color=color, width=0.78,
                    label=f"Topic {tc}", edgecolor="white", linewidth=0.4)
            bottom += vals
        axa.set_xticks(x); axa.set_xticklabels(dist.index, rotation=20,
                                               ha="right", fontsize=9.5)
        axa.set_ylim(0, 100)
        axa.set_ylabel("Proportion (%)", fontsize=11)
        axa.set_xlabel("Model", fontsize=11, labelpad=8)
        axa.set_title("Topic Distribution across LLMs", fontsize=13.5, pad=12)
        axa.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                   fontsize=9, title="")
        axa.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axa.set_axisbelow(True); _spine_off(axa)

        # (right) immersive & heritage element coverage by task ------------------
        feats = [("has_ar_vr", "AR/VR"), ("has_sound_music", "Sound/Music"),
                 ("has_interaction", "Interaction"), ("has_preservation", "Preservation")]
        feat_colors = {"AR/VR": "#B89B2F", "Sound/Music": "#5FA3C9",
                       "Interaction": "#2E8B5E", "Preservation": "#B86B8C"}
        cov = RESP.groupby("task_en")[[c for c, _ in feats]].mean() * 100
        cov = cov.reindex(TASK_ORDER)
        xt = np.arange(len(cov)); w = 0.19
        for j, (col, lab) in enumerate(feats):
            axb.bar(xt + (j - 1.5) * w, cov[col].values, w, label=lab,
                    color=feat_colors[lab], edgecolor="white", linewidth=0.5)
        axb.set_xticks(xt); axb.set_xticklabels(cov.index, rotation=0, fontsize=9.5)
        axb.set_ylabel("Coverage (%)", fontsize=11)
        axb.set_xlabel("Task", fontsize=11, labelpad=8)
        axb.set_ylim(0, 105)
        axb.set_title("Immersive & Heritage Elements by Task", fontsize=13.5, pad=12)
        axb.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                   fontsize=9, ncol=2, columnspacing=1.4, handlelength=1.5,
                   borderaxespad=0.0)
        axb.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axb.set_axisbelow(True); _spine_off(axb)

        fig.savefig(f"{OUT_DIR}/Figure_3.png", bbox_inches="tight")
        plt.close(fig); print("saved Figure_3")


# =============================================================================
# Figure 4 — stereotype markers · judge-vs-heuristic dumbbell
# =============================================================================
def fig4():
    with plt.rc_context(SANS_RC):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.8, 3.6),
                                       gridspec_kw={"width_ratios": [1.18, 1.0],
                                                    "wspace": 0.26})
        fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.22)

        # (left) cultural stereotype markers --------------------------------------
        cats = [("ste_exotic", "Exoticism"), ("ste_religious_bias", "Religious Bias"),
                ("ste_national_bias", "National Bias"), ("ste_romanticize", "Romanticization")]
        means = RESP.groupby("model")[[c for c, _ in cats]].mean().reindex(MODELS)
        x = np.arange(len(means)); w = 0.2
        for j, (col, lab) in enumerate(cats):
            axa.bar(x + (j - 1.5) * w, means[col].values, w, label=lab,
                    color=CATEGORY_COLORS[lab], edgecolor="white", linewidth=0.4)
        axa.set_xticks(x); axa.set_xticklabels(means.index, rotation=20,
                                               ha="right", fontsize=9)
        axa.set_ylabel("Avg. Keyword Count", fontsize=11)
        axa.set_title("Cultural Stereotype Markers across LLMs", fontsize=13.5, pad=12)
        axa.legend(loc="upper left", frameon=False, fontsize=9, ncol=2,
                   columnspacing=1.6, handlelength=1.5)
        axa.set_ylim(0, max(means.max()) * 1.28)
        axa.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axa.set_axisbelow(True); _spine_off(axa)

        # (right) judge vs heuristic dumbbell --------------------------------------
        judge_mean = JUDGE.groupby("model")[DIMS].mean().mean(axis=1)
        heur_mean = RESP.groupby("model")[[HEUR_COLS[d] for d in DIMS]].mean().mean(axis=1)
        _jm6 = judge_mean.round(6)
        order = sorted(_jm6.index,
                       key=lambda m: (-_jm6[m], TIE_PRIORITY.index(m)
                                      if m in TIE_PRIORITY else 99))
        y = np.arange(len(order))[::-1]
        for yi, m in zip(y, order):
            jv, hv = judge_mean[m], heur_mean[m]
            delta = hv - jv
            axb.plot([min(jv, hv), max(jv, hv)], [yi, yi],
                     color=DB_NEG if delta < 0 else DB_POS, linewidth=2.2, zorder=2)
            axb.scatter([jv], [yi], s=120, color=DB_JUDGE, zorder=3,
                        edgecolor="white", linewidth=0.8)
            axb.scatter([hv], [yi], s=120, color=DB_HEUR, marker="D", zorder=3,
                        edgecolor="white", linewidth=0.8)
            axb.text(max(jv, hv) + 0.09, yi, f"Δ {delta:+.2f}", va="center",
                     fontsize=9, color=DB_NEG if delta < 0 else "#8A8A8A")
        axb.set_yticks(y); axb.set_yticklabels(order, fontsize=10)
        axb.set_xlabel("Score (1–5)", fontsize=11)
        axb.set_title("Judge vs. Heuristic Divergence by Model", fontsize=13.5, pad=12)
        axb.set_xlim(0.4, 5.6)
        axb.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        axb.set_axisbelow(True); _spine_off(axb)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=DB_JUDGE,
                          markersize=8, label="LLM Judge (mean of 5 dims)"),
                   Line2D([0], [0], marker="D", color="none", markerfacecolor=DB_HEUR,
                          markersize=7, label="Heuristic Diagnostic (mean of 5 dims)")]
        axb.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.005),
                   frameon=False, fontsize=8, labelspacing=0.25,
                   handletextpad=0.3, borderaxespad=0.0)

        fig.savefig(f"{OUT_DIR}/Figure_4.png", bbox_inches="tight")
        plt.close(fig); print("saved Figure_4")


# =============================================================================
# Figure 5 — overall performance dots · 7×7 correlation matrix
# =============================================================================
# =============================================================================
# Figure 5 — rankings by dimension + overall + dimension correlation (merged)
# =============================================================================
# Revision note (Q1 review): the former Figure 5 (overall dot plot + correlation
# matrix) and Figure 6 (per-dimension rankings + mean-score bars) both presented
# model rankings and were merged.  The standalone overall dot plot and the
# bottom mini-bar strip duplicated panels already present here and were removed;
# the correlation matrix moved into this figure as panel (g).
# tie-break priority for numerically identical means (as ranked in the
# original figures: exact ties such as the four 5.00 narrative-quality means
# are ordered Pro-Kimi-K2.6 → Pro-GLM-5.1 → DeepSeek-V4-Pro → …)
TIE_PRIORITY = ["Pro-Kimi-K2.6", "Pro-GLM-5.1", "DeepSeek-V4-Pro",
                "Kimi-K2.7-Code", "DeepSeek-V4-Flash", "GLM-5.2",
                "MiniMax-M2.5", "Pro-MiniMax-M2.5", "LongCat-2.0", "Nex-N2-Pro"]


def _model_rank(df, dim):
    stats = df.groupby("model")[dim].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["_tie"] = stats["model"].map(
        {m: i for i, m in enumerate(TIE_PRIORITY)}).fillna(99)
    stats["_mean6"] = stats["mean"].round(6)
    stats = stats.sort_values(["_mean6", "_tie"], ascending=[False, True])
    return stats.drop(columns=["_tie", "_mean6"]).reset_index(drop=True)


def fig5():
    with plt.rc_context(SANS_RC):
        fig = plt.figure(figsize=(12.4, 13.9))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05],
                              hspace=0.62, wspace=0.42,
                              left=0.115, right=0.975, top=0.945, bottom=0.055)
        fig.suptitle("Model Rankings by Dimension (Judge)", fontsize=17, y=0.985)

        DEGEN = "Nex-N2-Pro"   # degenerate endpoint: off-scale, analyzed under RQ4

        def _rank_ax(ax, stats, xlabel):
            stats = stats[stats["model"] != DEGEN].reset_index(drop=True)
            y = np.arange(len(stats))[::-1]
            colors = [MODEL_COLORS.get(m, NEUTRAL) for m in stats["model"]]
            lo = max(1.0, float((stats["mean"] - 1.96 * stats["se"]).min()) - 0.08)
            hi = 5.04
            span = hi - lo
            ax.errorbar(stats["mean"], y, xerr=1.96 * stats["se"], fmt="none",
                        ecolor="#C9CDD2", elinewidth=4, capsize=0, zorder=1)
            ax.scatter(stats["mean"], y, c=colors, s=95, edgecolor="white",
                       linewidth=0.9, zorder=3)
            for yi, v in zip(y, stats["mean"]):
                ax.text(v - 0.018 * span, yi, f"{v:.2f}", ha="right",
                        va="center", fontsize=8, color=TEXT_COLOR)
            ax.set_yticks(y); ax.set_yticklabels(stats["model"], fontsize=9.5)
            ax.set_xlim(lo, hi)
            ax.set_xticks(np.linspace(np.ceil(lo * 10) / 10, 5.0, 4))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
            ax.set_xlabel(xlabel, fontsize=10.5)
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True); _spine_off(ax)

        # rows 1-2: five dimension dot plots + overall bars ---------------------
        for i, dim in enumerate(DIMS):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            _rank_ax(ax, _model_rank(JUDGE, dim), "Score (1–5, zoomed)")
            ax.set_title(DIM_EN[dim], fontsize=13, pad=10)

        ax = fig.add_subplot(gs[1, 2])
        _ov = JUDGE.groupby("model")[DIMS].mean().mean(axis=1).round(6)
        _tie = pd.Series({m: (TIE_PRIORITY.index(m) if m in TIE_PRIORITY else 99)
                          for m in _ov.index})
        overall = _ov.iloc[sorted(range(len(_ov)),
                                  key=lambda i: (-_ov.iloc[i], _tie.iloc[i]))]
        y = np.arange(len(overall))[::-1]
        colors = [MODEL_COLORS.get(m, NEUTRAL) for m in overall.index]
        ax.barh(y, overall.values, color=colors, height=0.62,
                edgecolor="white", linewidth=0.6)
        for yi, v in zip(y, overall.values):
            ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=9,
                    color=TEXT_COLOR)
        ax.set_yticks(y); ax.set_yticklabels(overall.index, fontsize=9.5)
        ax.set_xlim(1, 5)
        ax.set_xlabel("Aggregate Score (1–5)", fontsize=10.5)
        ax.set_title("Overall", fontsize=13, pad=10)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True); _spine_off(ax)

        fig.text(0.5, 0.345,
                 "Panels (a)–(e) use zoomed score axes for the nine functioning "
                 "endpoints; Nex-N2-Pro (mean ≈ 1.2 on every dimension) lies "
                 "off scale and is analyzed under RQ4.",
                 ha="center", fontsize=9.5, color="#5F6368", style="italic")

        # row 3: (g) dimension correlation matrix (moved from former Figure 5) --
        m = JUDGE.copy()
        m["Output Length"] = RESP.set_index("response_id")["char_count"] \
            .reindex(m["response_id"]).values
        m["Sentiment Score"] = RESP.set_index("response_id")["sentiment_score"] \
            .reindex(m["response_id"]).values
        corr_cols = DIMS + ["Output Length", "Sentiment Score"]
        labels = [DIM_EN[d] for d in DIMS] + ["Output Length", "Sentiment Score"]
        corr = m[corr_cols].corr()
        fig.add_subplot(gs[2, 0]).axis("off")
        axb = fig.add_subplot(gs[2, 1])
        fig.add_subplot(gs[2, 2]).axis("off")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=CMAP_DIVERGING,
                    vmin=-1, vmax=1, center=0, linewidths=0.5, ax=axb, square=True,
                    annot_kws={"size": 8.5, "color": TEXT_COLOR},
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": "Pearson r"})
        axb.set_title("Dimension Correlation", fontsize=13, pad=10)
        axb.set_xticklabels(axb.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
        axb.set_yticklabels(axb.get_yticklabels(), rotation=0, fontsize=8.5)

        fig.savefig(f"{OUT_DIR}/Figure_5.png", bbox_inches="tight")
        plt.close(fig); print("saved Figure_5")


# =============================================================================
# Figure 6 — model × task heat maps for the five judge dimensions
# =============================================================================
def fig6():
    from matplotlib.colors import Normalize
    TASK_SHORT = {"Historical Storytelling": "Hist. Storytelling",
                  "Immersive Design": "Imm. Design",
                  "Museum Guide": "Museum Guide"}
    with plt.rc_context(SANS_RC):
        fig = plt.figure(figsize=(12.4, 7.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.30, wspace=0.16,
                              left=0.105, right=0.985, top=0.93, bottom=0.115)
        axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]
        norm = Normalize(vmin=1, vmax=5)
        for k, (ax, dim) in enumerate(zip(axes, DIMS)):
            row, col = divmod(k, 3)
            pivot = JUDGE.pivot_table(values=dim, index="model",
                                      columns="task_en", aggfunc="mean")
            pivot = pivot.reindex(index=MODELS, columns=TASK_ORDER)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=CMAP_ACADEMIC,
                        vmin=1, vmax=5, ax=ax, linewidths=0.6,
                        annot_kws={"size": 8.5}, cbar=False,
                        yticklabels=(col == 0),
                        xticklabels=[TASK_SHORT.get(t, t) for t in pivot.columns])
            ax.set_title(f"{DIM_EN[dim]} (Judge)", fontsize=12.5, pad=9)
            # x labels only on the bottom row
            if row == 1:
                ax.set_xlabel("Task", fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right",
                                   fontsize=9)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            # y labels only on the left column
            if col == 0:
                ax.set_ylabel("Model", fontsize=10)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
            else:
                ax.set_ylabel("")
        # single slim shared colorbar at the right of the bottom row
        sm = plt.cm.ScalarMappable(cmap=CMAP_ACADEMIC, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.760, 0.155, 0.018, 0.305])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Score", fontsize=10.5)
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.ax.tick_params(labelsize=9)

        fig.savefig(f"{OUT_DIR}/Figure_6.png", bbox_inches="tight")
        plt.close(fig); print("saved Figure_6")


# =============================================================================
# Figure 7 — evidence at a glance (hand-composed summary panel)
# =============================================================================
def fig7():
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    BLUE, BRONZE, RED = "#4B7F9C", "#A77B3A", "#C0392B"
    GRAY = "#666666"
    AB = ["NQ", "CS", "HA", "IMM", "PER"]

    # ---- (a) eta^2 -----------------------------------------------------------
    eta_judge = {"NQ": 0.909, "CS": 0.928, "HA": 0.674, "IMM": 0.864, "PER": 0.882}
    anova_csv = os.path.join(BASE, "stats_results", "anova_summary.csv")
    if os.path.exists(anova_csv):
        a = pd.read_csv(anova_csv, encoding="utf-8-sig").set_index("dimension")
        eta_judge = {ab: round(float(a.loc[d, "eta_squared"]), 3)
                     for ab, d in zip(AB, DIMS)}
    # heuristic eta^2 (authors' screening log; heuristic scores are not
    # reproduced by the executed pipeline)
    eta_heur = {"NQ": 0.486, "CS": 0.096, "HA": 0.535, "IMM": 0.171, "PER": 0.150}

    # ---- (b) HA endpoint means over the 600 cell means ------------------------
    cell = JUDGE.groupby(["model", "persona_id", "task"], as_index=False)[DIMS].mean()
    ha = cell.groupby("model")["historical_accuracy"].mean().sort_values(ascending=False)

    # ---- (c) task main effects + interactions ---------------------------------
    task_eff = {"NQ": (0.94, 0.391), "CS": (0.27, 0.766), "HA": (12.94, 0.0),
                "IMM": (53.25, 0.0), "PER": (9.79, 0.0)}
    inter_p = {"NQ": 0.020, "CS": 0.012, "HA": 0.124, "IMM": 0.0, "PER": 0.001}
    tw_csv = os.path.join(BASE, "stats_results", "twoway_anova.csv")
    if os.path.exists(tw_csv):
        t = pd.read_csv(tw_csv, encoding="utf-8-sig")
        for ab, d in zip(AB, DIMS):
            fr = t[(t["dimension"] == d) & (t["source"] == "C(task)")]
            ir = t[(t["dimension"] == d) & (t["source"] == "C(model):C(task)")]
            if len(fr):
                task_eff[ab] = (round(float(fr["F"].iloc[0]), 2), float(fr["p"].iloc[0]))
            if len(ir):
                inter_p[ab] = float(ir["p"].iloc[0])

    # ---- (d) Nex-N2-Pro silent failure (raw corpus) ---------------------------
    olen = RESP["output"].astype(str).str.len()
    nex_mask = RESP["model"] == "Nex-N2-Pro"
    n_nex = int(nex_mask.sum())
    share_short = 100.0 * float((olen[nex_mask] < 20).mean())
    nex_len = float(olen[nex_mask].mean())
    func_len = float(olen[~nex_mask].mean())
    nex_judge = float(JUDGE.loc[JUDGE["model"] == "Nex-N2-Pro", DIMS].mean().mean())

    def _fmt_p(p):
        if p < 0.001:
            return "p<.001", "***"
        s = f"{p:.3f}"[1:]
        stars = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        return f"p={s}", stars

    # ---- canvas ----------------------------------------------------------------
    fig = plt.figure(figsize=(14.2, 8.0))
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.975, top=0.855, bottom=0.095,
                          hspace=0.55, wspace=0.22)
    fig.text(0.5, 0.965, "Evidence at a Glance — One Guarded Claim: Benchmark "
             "Conclusions Depend Jointly", ha="center", va="center",
             fontsize=16.5, fontweight="bold", color="#26303B")
    fig.text(0.5, 0.918, "on Endpoint, Task, and Measurement Instrument",
             ha="center", va="center", fontsize=16.5, fontweight="bold",
             color="#26303B")

    # ---- (a) RQ1 — measurement validity ----------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    x = np.arange(5); w = 0.38
    vj = [eta_judge[k] for k in AB]; vh = [eta_heur[k] for k in AB]
    axa.bar(x - w / 2, vj, w, color=BLUE, label="LLM-judge", zorder=3)
    axa.bar(x + w / 2, vh, w, color=BRONZE, label="Lexical heuristic", zorder=3)
    for xi, v in zip(x - w / 2, vj):
        axa.text(xi, v + 0.018, f"{v:.3f}"[1:], ha="center", fontsize=9,
                 color=TEXT_COLOR)
    for xi, v in zip(x + w / 2, vh):
        axa.text(xi, v + 0.018, f"{v:.3f}"[1:], ha="center", fontsize=9,
                 color=TEXT_COLOR)
    axa.set_xticks(x); axa.set_xticklabels(AB, fontsize=10.5)
    axa.set_ylim(0, 1.34); axa.set_yticks(np.arange(0, 1.21, 0.2))
    axa.set_ylabel("Variance explained by model ($\\eta^2$)", fontsize=11)
    axa.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC",
               fontsize=9.5, framealpha=1.0)
    axa.text(-0.42, 1.30, 'CS heuristic $\\eta^2$=.096 — empty text can look '
             '"safe":\ngate heuristics before deployment', color=RED,
             fontsize=9.5, va="top", linespacing=1.35)
    axa.annotate("", xy=(1 + w / 2, 0.175), xytext=(1 + w / 2, 1.02),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    axa.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.8)
    axa.set_axisbelow(True); _spine_off(axa)
    axa.set_title("(a) RQ1 — Measurement validity", loc="left",
                  fontsize=12.5, fontweight="bold", pad=10)

    # ---- (b) RQ2 — HA differentiates endpoints ----------------------------------
    axb = fig.add_subplot(gs[0, 1])
    ypos = np.arange(len(ha))[::-1]
    for (m, v), yi in zip(ha.items(), ypos):
        if m == "Nex-N2-Pro":
            v_nex = cell.loc[cell["model"] == m, "historical_accuracy"]
            ci = 1.96 * v_nex.std(ddof=1) / np.sqrt(len(v_nex))
            axb.errorbar(v, yi, xerr=ci, fmt="none", ecolor=RED, capsize=0,
                         linewidth=1.6, linestyle=(0, (4, 3)), alpha=0.85, zorder=2)
            axb.scatter(v, yi, s=170, c=RED, zorder=3)
            axb.text(v - 0.26, yi, f"$\\approx${v:.2f}", ha="right", va="center",
                     fontsize=10, color=RED, fontweight="bold")
        else:
            axb.scatter(v, yi, s=170, c=BLUE, zorder=3)
            axb.text(v + 0.09, yi, f"{v:.2f}", ha="left", va="center",
                     fontsize=10, color=TEXT_COLOR)
    axb.set_yticks(ypos); axb.set_yticklabels(ha.index, fontsize=10)
    axb.set_xlim(0.4, 5.35); axb.set_xticks([1, 2, 3, 4, 5])
    axb.set_xlabel("Perceived historical accuracy mean (1–5), 600 cell means", fontsize=10.5)
    axb.legend(handles=[Line2D([0], [0], marker="o", color="none",
                               markerfacecolor="none", markeredgecolor=RED,
                               markeredgewidth=1.8, markersize=12,
                               label="integrity flag (not a score)")],
               loc="center left", bbox_to_anchor=(0.24, 0.47), frameon=True,
               edgecolor="#CCCCCC", fontsize=9.5, framealpha=1.0)
    axb.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.8)
    axb.set_axisbelow(True); _spine_off(axb)
    axb.set_title("(b) RQ2 — HA differentiates endpoints", loc="left",
                  fontsize=12.5, fontweight="bold", pad=10)

    # ---- (c) RQ3 — rankings are task-conditioned --------------------------------
    axc = fig.add_subplot(gs[1, 0])
    Fv = [task_eff[k][0] for k in AB]
    axc.bar(x, Fv, 0.52, color=BLUE, zorder=3)
    for xi, k in zip(x, AB):
        F, p = task_eff[k]
        ptxt, stars = _fmt_p(p)
        axc.text(xi, F + 4.6, f"F={F:.2f}", ha="center", fontsize=9.5,
                 fontweight="bold", color=TEXT_COLOR)
        axc.text(xi, F + 1.2, f"{ptxt} {stars}", ha="center", fontsize=9,
                 color=TEXT_COLOR)
    sig_task = [k for k in AB if task_eff[k][1] < 0.05]
    sig_inter = [k for k in AB if inter_p[k] < 0.05]
    ns_inter = [k for k in AB if inter_p[k] >= 0.05]
    note = f"task: {' · '.join(sig_task)} significant\ninteractions: {' · '.join(sig_inter)} sig."
    if ns_inter:
        ns = ns_inter[0]
        note += f"; {ns} n.s. (p={inter_p[ns]:.3f})".replace("p=0.", "p=.")
    axc.text(0.02, 0.97, note, transform=axc.transAxes, ha="left", va="top",
             fontsize=9, color=GRAY, style="italic", linespacing=1.4)
    axc.set_xticks(x); axc.set_xticklabels(AB, fontsize=10.5)
    axc.set_ylim(0, 62); axc.set_yticks(np.arange(0, 51, 10))
    axc.set_ylabel("Task main effect (F)", fontsize=11)
    axc.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.8)
    axc.set_axisbelow(True); _spine_off(axc)
    axc.set_title("(c) RQ3 — Rankings are task-conditioned", loc="left",
                  fontsize=12.5, fontweight="bold", pad=10)

    # ---- (d) RQ4 — silent failure of a production endpoint -----------------------
    axd = fig.add_subplot(gs[1, 1]); axd.axis("off")
    axd.set_title("(d) RQ4 — Silent failure of a production endpoint", loc="left",
                  fontsize=12.5, fontweight="bold", pad=10)
    stats3 = [(f"{share_short:.1f}%", f"of {n_nex} Nex-N2-Pro outputs\nshorter than 20 characters"),
              (f"{nex_len:.0f} vs {func_len:,.0f}", "mean output length (chars):\nNex-N2-Pro vs functioning endpoints"),
              (f"$\\approx${nex_judge:.2f}/5", "judge score, all five\ndimensions (API status OK)")]
    for (big, cap), xc in zip(stats3, [0.175, 0.5, 0.825]):
        axd.text(xc, 0.80, big, transform=axd.transAxes, ha="center", va="center",
                 fontsize=23, fontweight="bold", color=RED)
        axd.text(xc, 0.60, cap, transform=axd.transAxes, ha="center", va="top",
                 fontsize=9, color=GRAY, linespacing=1.4)
    box = FancyBboxPatch((0.045, 0.05), 0.91, 0.30,
                         boxstyle="round,pad=0.012", transform=axd.transAxes,
                         facecolor="#FDF8F6", edgecolor=RED,
                         linestyle=(0, (4, 3)), linewidth=1.3)
    axd.add_patch(box)
    axd.text(0.5, 0.20, "Defense: schema + length validation, empty-output "
             "rejection,\nretry/fallback endpoints, monitoring, human escalation.\n"
             "API status OK $\\neq$ content success.",
             transform=axd.transAxes, ha="center", va="center", fontsize=9.5,
             color="#A93226", linespacing=1.45)

    fig.savefig(f"{OUT_DIR}/Figure_7.png", bbox_inches="tight")
    plt.close(fig); print("saved Figure_7")


if __name__ == "__main__":
    for f in (fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    print("all figures →", OUT_DIR)