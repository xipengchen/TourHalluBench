"""
Heuristic content analysis — extracts quantitative features from LLM outputs
for downstream statistical tests.

Features extracted:
  - Text statistics: char/word/paragraph counts, sentence complexity
  - Topic modeling: LDA over tokenized Chinese text → dominant topic
  - Sentiment analysis: positive/neutral/negative ratios via keyword matching
  - Stereotype detection: keyword dictionaries for exoticization, primitivism,
    gendered stereotypes, and national clichés
  - Immersion & tech markers: sensory, emotional, interactive, AR/VR terms
  - N-gram features: top unigrams/bigrams for word frequency analysis

Outputs (analysis_results/):
  12_topic_distribution.csv      14_topic_stream.csv
  11_sentiment.csv               13_stereotype.csv
  15_content_features.csv        16_immersion_tech.csv
  10_text_stats.csv              17_word_freq.csv
  + various intermediate CSVs
"""

import os
import re
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ---- Paths -------------------------------------------------------------------
INPUT_CSV = "llm_cultural_heritage_responses.csv"
OUT_DIR = "analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

STOPWORDS_FILE = os.path.join(os.path.dirname(__file__), "stopwords.txt")


# ---- Stopwords ---------------------------------------------------------------
def _load_stopwords() -> set:
    base = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
            "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
            "所", "为", "所以", "因为", "但是", "然而", "而且", "或者", "如果",
            "虽然", "可以", "这个", "那个", "什么", "怎么", "哪", "吗", "啊", "吧",
            "呢", "哦", "嗯", "哈", "哇", "呀", "哎", "喂", "哼", "咚", "啦",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "and", "but",
            "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some", "such",
            "only", "own", "same", "than", "too", "very", "just", "now", "then"}
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
            base |= {ln.strip() for ln in f if ln.strip()}
    return base

STOPWORDS = _load_stopwords()


def tokenize(text: str) -> list[str]:
    """Chinese-aware tokenization with stopword removal."""
    if not isinstance(text, str) or not text.strip():
        return []
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in STOPWORDS]


# ---- 10. Text statistics -----------------------------------------------------
def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-output text metrics."""
    records = []
    for _, row in df.iterrows():
        text = str(row.get("output", ""))
        records.append({
            "response_id": row["response_id"],
            "model": row["model"],
            "task": row["task"],
            "char_count": len(text),
            "word_count": len(tokenize(text)),
            "sentence_count": len(re.split(r"[。！？.!?\n]+", text)),
            "avg_sentence_length": (
                len(text) / max(len(re.split(r"[。！？.!?\n]+", text)), 1)
            ),
        })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "10_text_stats.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 10_text_stats.csv")
    return out


# ---- 11. Sentiment analysis --------------------------------------------------
POS_WORDS = {"美丽", "宏伟", "精彩", "完美", "独特", "震撼", "惊叹", "辉煌", "珍贵",
             "绝妙", "迷人", "优雅", "赞叹", "伟大", "神秘", "吸引", "推荐", "值得",
             "神奇", "壮丽", "壮观", "匠心", "精湛", "瑰宝", "璀璨", "叹为观止"}
NEG_WORDS = {"破坏", "损失", "遗憾", "破坏性", "过度开发", "商业化", "褪色", "风化",
             "侵蚀", "消失", "遗忘", "忽略", "忽视", "污染", "拥挤", "喧闹"}

def _sentiment_counts(text: str) -> tuple[int, int, int]:
    words = set(tokenize(text))
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    neu = len(words) - pos - neg
    return pos, neg, max(neu, 0)

def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = str(row.get("output", ""))
        pos, neg, neu = _sentiment_counts(text)
        total = max(pos + neg + neu, 1)
        records.append({
            "response_id": row["response_id"],
            "model": row["model"],
            "task": row["task"],
            "positive_ratio": pos / total,
            "negative_ratio": neg / total,
            "neutral_ratio": neu / total,
            "sentiment_score": (pos - neg) / total,
        })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "11_sentiment.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 11_sentiment.csv")
    return out


# ---- 12-14. Topic modeling (LDA) ---------------------------------------------
def compute_topics(df: pd.DataFrame, n_topics: int = 6) -> pd.DataFrame:
    """Latent Dirichlet Allocation over tokenized outputs."""
    texts = [str(r.get("output", "")) for _, r in df.iterrows()]
    tokenized = [" ".join(tokenize(t)) for t in texts]

    vec = CountVectorizer(max_df=0.9, min_df=5, max_features=2000)
    dtm = vec.fit_transform(tokenized)
    feature_names = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    doc_topics = lda.fit_transform(dtm)

    # Assign dominant topic to each document
    dominant = np.argmax(doc_topics, axis=1)
    df = df.copy()
    df["dominant_topic"] = dominant

    # Topic distribution per model
    topic_dist = df.groupby(["model", "dominant_topic"]).size().unstack(fill_value=0)
    topic_dist.to_csv(os.path.join(OUT_DIR, "12_topic_distribution.csv"), encoding="utf-8-sig")
    print("[analysis] 12_topic_distribution.csv")

    # Topic stream (full document-level topic assignments)
    stream_cols = ["response_id", "model", "persona_id", "task", "dominant_topic"]
    df[stream_cols].to_csv(os.path.join(OUT_DIR, "14_topic_stream.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 14_topic_stream.csv")

    # Top words per topic
    topic_words = []
    for t_idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-15:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        topic_words.append({"topic": t_idx, "top_words": " ".join(top_words)})
    pd.DataFrame(topic_words).to_csv(
        os.path.join(OUT_DIR, "12_topic_keywords.csv"), index=False, encoding="utf-8-sig"
    )
    return df


# ---- 13. Stereotype detection ------------------------------------------------
# 词表统一（2026-08-02）：本模块此前自带一套与 visualization.py 不同的刻板印象词表，
# 造成正文与 Figure 4 的类别不一致。现统一从 lexical_diagnostics.py 导入产出刊印
# Figure 4 的那套词表（exotic / religious_bias / national_bias / romanticize），
# 并改用对原文的子串计数（中文无词界，短语与单词同权）。
from lexical_diagnostics import STEREOTYPE_KEYWORDS as _STE_LEX

def detect_stereotype(text: str) -> dict:
    """与 Figure 4 的仪器完全同口径：统计每个子类中**出现过的不同词条数**
    （每个词条至多计 1 次，子串匹配），而非出现次数累计。"""
    t = "" if text is None else str(text)
    return {cat: sum(1 for kw in kws if kw in t) for cat, kws in _STE_LEX.items()}

def compute_stereotype(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = str(row.get("output", ""))
        st = detect_stereotype(text)
        records.append({
            "response_id": row["response_id"],
            "model": row["model"],
            "task": row["task"],
            "persona_id": row["persona_id"],
            "exoticism_count": st["exotic"],
            "religious_bias_count": st["religious_bias"],
            "national_bias_count": st["national_bias"],
            "romanticization_count": st["romanticize"],
            "total_stereotype_score": sum(st.values()),
        })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "13_stereotype.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 13_stereotype.csv")
    return out


# ---- 15. Content features (binary markers) -----------------------------------
TIME_PAT = re.compile(r"\d{3,4}\s*年|[一二三四五六七八九十]+世纪|\d+世纪|公元前|公元")
PERSON_PAT = re.compile(r"皇帝|国王|法老|苏丹|可汗|将军|丞相|宦官|太监|王后|公主|王子|"
                         r"大臣|贵妃|太后|考古学家|历史学家|探险家|学者|诗人|画家|旅行家")
PLACE_PAT = re.compile(r"宫殿|庙宇|陵墓|城墙|塔|阁|楼|门|广场|园林|庭院|墓室|石窟|寺院|祭坛")
ARVR_PAT = re.compile(r"AR|VR|虚拟现实|增强现实|混合现实|3D|三维|投影|全息|数字|互动屏幕|触屏")
SOUND_PAT = re.compile(r"音乐|声音|钟声|鼓声|琴|笛|歌|唱|诵|吟|回声|旋律|节奏|乐器")
INTERACTION_PAT = re.compile(r"互动|参与|体验|触摸|点击|扫描|扫码|打卡|拍照|合影|角色扮演|答题")
PRESERVATION_PAT = re.compile(r"保护|修复|维护|保存|传承|非遗|世界遗产|禁止触摸|禁止拍照|"
                              r"文物|脆弱|风化|磨损")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = str(row.get("output", ""))
        records.append({
            "response_id": row["response_id"],
            "model": row["model"],
            "task": row["task"],
            "has_time_reference": int(bool(TIME_PAT.search(text))),
            "has_person_reference": int(bool(PERSON_PAT.search(text))),
            "has_place_reference": int(bool(PLACE_PAT.search(text))),
            "has_ar_vr": int(bool(ARVR_PAT.search(text))),
            "has_sound_music": int(bool(SOUND_PAT.search(text))),
            "has_interaction": int(bool(INTERACTION_PAT.search(text))),
            "has_preservation": int(bool(PRESERVATION_PAT.search(text))),
        })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "15_content_features.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 15_content_features.csv")
    return out


# ---- 16. Immersion & technology markers --------------------------------------
SENSORY_PAT = re.compile(r"看|听|闻|触摸|感受|温暖|寒冷|明亮|昏暗|香气|味道|粗糙|光滑|柔软")
EMOTIONAL_PAT = re.compile(r"敬畏|震撼|感动|怀旧|好奇|兴奋|悲伤|崇敬|惊叹|沉思|宁静|庄严")
TECH_PAT = re.compile(r"AR|VR|AI|APP|应用|程序|二维码|GPS|语音导览|耳机|投影|3D")

def compute_immersion_tech(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = str(row.get("output", ""))
        records.append({
            "response_id": row["response_id"],
            "model": row["model"],
            "task": row["task"],
            "sensory_count": len(SENSORY_PAT.findall(text)),
            "emotional_count": len(EMOTIONAL_PAT.findall(text)),
            "tech_count": len(TECH_PAT.findall(text)),
        })
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "16_immersion_tech.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 16_immersion_tech.csv")
    return out


# ---- 17. Word frequency ------------------------------------------------------
def compute_word_freq(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Top-N unigrams and bigrams across all outputs."""
    all_words = []
    all_bigrams = []
    for text in df["output"].dropna():
        tokens = tokenize(str(text))
        all_words.extend(tokens)
        for i in range(len(tokens) - 1):
            all_bigrams.append(f"{tokens[i]}_{tokens[i+1]}")

    word_freq = Counter(all_words).most_common(top_n)
    bigram_freq = Counter(all_bigrams).most_common(top_n)

    out = pd.DataFrame([
        {"rank": i+1, "term": w, "freq": f, "type": "unigram"}
        for i, (w, f) in enumerate(word_freq)
    ] + [
        {"rank": i+1, "term": w, "freq": f, "type": "bigram"}
        for i, (w, f) in enumerate(bigram_freq)
    ])
    out.to_csv(os.path.join(OUT_DIR, "17_word_freq.csv"), index=False, encoding="utf-8-sig")
    print("[analysis] 17_word_freq.csv")
    return out


# ---- Main --------------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"[analysis] 加载 {len(df)} 条记录")

    compute_text_stats(df)
    compute_sentiment(df)
    df_with_topics = compute_topics(df)
    compute_stereotype(df_with_topics)
    compute_features(df_with_topics)
    compute_immersion_tech(df_with_topics)
    compute_word_freq(df_with_topics)

    print("\n[analysis] 全部完成。结果保存至 analysis_results/")


if __name__ == "__main__":
    main()