# -*- coding: utf-8 -*-
"""
lexical_diagnostics.py — 透明词汇启发式诊断（论文 §3.6 / 附录 C.2 的仪器实现）

来源与地位:
  本模块的词表、正则与评分公式逐字取自产出刊印 Figure 4 的可视化脚本，并已在
  公开语料上验证可精确复现稿件中的全部相关数字:
    · 每端点刻板印象标记总数: Nex-N2-Pro = 3；其余端点 43–120
    · 图 4b 判分器 vs 启发式的 Δ: 正常端点 −1.83 ~ −1.32；退化端点 +0.63
  复现命令见 README_repro.md。

用法:
    from lexical_diagnostics import compute_diagnostics, HEUR_COLS
    d = compute_diagnostics(responses_df)      # 逐条附加全部诊断列
"""

import re

import jieba
import pandas as pd

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
