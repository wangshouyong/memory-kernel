# SPDX-License-Identifier: Apache-2.0
"""中文词法层（v4）单测：`memory_kernel/zh.py` 全部 API + skeleton/obligations 的 zh 接入。

契约：`docs/ALGO-V4-CONTRACT.md` §2 / §3 / §8.1。本文件只测**新增的 opt-in 能力**与
**默认路径回归门**，不改任何既有测试的断言。

回归门（本文件最重要的一节，见 `ZhFalseRegressionTest`）
────────────────────────────────────────────────────────────────────────────
v4 的硬约束是「所有新参数默认关闭时，行为与改动前逐字节一致」。这里用**固定样例 + 硬编码
期望集合**来锁（期望值来自改动前的 HEAD 版本实际输出，逐 token 抄录）：

    content_tokens("Alice bought a red car on March 5th.")        == {alice, bought, car, march, red}
    content_tokens("She wants his car and said it was fine")     == {car, fine}
    content_tokens("我们用 PostgreSQL 2023 年做过一个项目，...")  == {postgresql}
    norm_tokens("Joanna's project don't work; running, ran, runs")== {joanna, project, run, work}
    ...
    apply_obligations(...).detail                                 == 原文逐字
    build_idf(...).to_json()                                      == 版本 idf-s2

另有 `ZhFalseSnapshotTest`：把同一批固定样例的全部输出算一个 sha256，与改动前抓的快照
（`8d64f7b026c7c447d4e86952265e1afcde3a308a5f7a92d31a61bf4b72dde04b`）对比——这是"逐字节
一致"的机器证据，任何一处默认路径的意外改动都会让它变红。
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import zh
from memory_kernel.ledger import AnchorLedger
from memory_kernel.obligations import apply_obligations
from memory_kernel.skeleton import (
    POLICY_VERSION,
    POLICY_VERSION_V3,
    POLICY_VERSION_V31,
    POLICY_VERSION_V31_C3,
    POLICY_VERSION_V4,
    SkeletonPolicy,
    build_idf,
    content_tokens,
    idf_for_ledger,
    norm_tokens,
    skeleton_check,
)

KEY = hashlib.sha256(b"zh-test").digest()

# 改动前（HEAD）的默认路径输出快照 sha256：由 change 前的
# `content_tokens/norm_tokens/stem/build_idf/idf_for_ledger/apply_obligations/skeleton_check`
# 在固定样例上的结果计算。改动后重算必须完全相同。
PRE_CHANGE_SNAPSHOT_SHA256 = (
    "8d64f7b026c7c447d4e86952265e1afcde3a308a5f7a92d31a61bf4b72dde04b"
)


# --------------------------------------------------------------------------- #
# 1. CJK 判定
# --------------------------------------------------------------------------- #

class CjkTest(unittest.TestCase):
    def test_has_cjk(self):
        self.assertTrue(zh.has_cjk("我"))
        self.assertTrue(zh.has_cjk("abc中文def"))
        self.assertFalse(zh.has_cjk(""))
        self.assertFalse(zh.has_cjk("abc 123 !@#"))
        # CJK 标点/全角符号**不是**汉字：它们不承载内容/否定/时间
        self.assertFalse(zh.has_cjk("。，！？「」【】"))
        self.assertFalse(zh.has_cjk("ＡＢＣ１２３"))   # 全角 ASCII 也不是汉字

    def test_cjk_ratio(self):
        self.assertEqual(zh.cjk_ratio(""), 0.0)
        self.assertEqual(zh.cjk_ratio("   \t\n"), 0.0)      # 全空白 → 0.0（不是 1.0）
        self.assertEqual(zh.cjk_ratio("我们"), 1.0)
        self.assertAlmostEqual(zh.cjk_ratio("我们abc"), 0.4)  # 分母不含空白：2 CJK / 5 非空白
        self.assertAlmostEqual(zh.cjk_ratio("我们 abc"), 0.4)  # 空白不计入分母

    def test_cjk_tokens_follow_contract_example(self):
        # 契约 §8.1 的字面例子
        self.assertEqual(zh.zh_tokens("我上周去了杭州"),
                         ["我上", "上周", "周去", "去了", "了杭", "杭州"])


# --------------------------------------------------------------------------- #
# 2. zh_tokens：确定性、bigram、ASCII 段、顺序与重复
# --------------------------------------------------------------------------- #

class ZhTokensTest(unittest.TestCase):
    def test_deterministic_repeated_calls(self):
        text = "我们用 PostgreSQL 2023 年做过一个项目，但是后来没有继续维护。"
        first = zh.zh_tokens(text)
        for _ in range(5):
            self.assertEqual(zh.zh_tokens(text), first)

    def test_single_char_han_run_is_unigram(self):
        # 长度 1 的汉字段取 unigram（"5 个" 里的 "个"）
        self.assertEqual(zh.zh_tokens("5个"), ["5", "个"])
        self.assertEqual(zh.zh_tokens("a我b"), ["a", "我", "b"])

    def test_ascii_runs_are_lowercased_whole(self):
        self.assertEqual(zh.zh_tokens("PostgreSQL"), ["postgresql"])
        self.assertEqual(zh.zh_tokens("ABC2023x"), ["abc2023x"])
        # 数字也是 ASCII 字母数字段（硬单元另由 skeleton.hard_units 负责，这里只是 token）
        self.assertEqual(zh.zh_tokens("2026"), ["2026"])

    def test_punctuation_and_whitespace_are_separators(self):
        self.assertEqual(zh.zh_tokens("我们，去了 杭州。"), ["我们", "去了", "杭州"])

    def test_order_and_duplicates_are_preserved(self):
        # 契约：顺序 = 原文出现顺序，允许重复（调用方自己取集合）
        self.assertEqual(zh.zh_tokens("我们我们"), ["我们", "们我", "我们"])
        self.assertEqual(zh.zh_tokens("杭州西湖"), ["杭州", "州西", "西湖"])

    def test_no_stopword_filtering_in_tokenizer(self):
        # 分词层不过滤停用词（过滤是 content_tokens 的职责），否则无法解释"为什么少了词"
        self.assertIn("我们", zh.zh_tokens("我们"))
        self.assertIn("的", zh.zh_tokens("的"))

    def test_empty_and_pure_ascii(self):
        self.assertEqual(zh.zh_tokens(""), [])
        self.assertEqual(zh.zh_tokens("   "), [])
        self.assertEqual(zh.zh_tokens("hello world"), ["hello", "world"])


# --------------------------------------------------------------------------- #
# 3. ZH_STOP：只收 1~2 字虚词，绝不收否定词
# --------------------------------------------------------------------------- #

class ZhStopTest(unittest.TestCase):
    def test_stop_entries_are_one_or_two_chars(self):
        # zh_tokens 只产出 1~2 字汉 token；更长/更短的项永远不会命中 → 属于死条目
        for entry in zh.ZH_STOP:
            self.assertIn(len(entry), (1, 2), f"死条目（永远不会命中）: {entry!r}")

    def test_stop_never_contains_negation(self):
        # 硬约束：否定一旦混进停用词表，否定会在分词阶段被悄悄抹掉
        neg_chars = set("不没未无别非否甭莫勿")
        for entry in zh.ZH_STOP:
            self.assertFalse(set(entry) & neg_chars, f"停用词含否定字形: {entry!r}")
            self.assertFalse(zh.zh_negated(entry), f"停用词被判成否定: {entry!r}")

    def test_stop_covers_high_frequency_function_words(self):
        for w in ("的", "了", "是", "在", "我们", "这个", "那个", "一个", "因为", "所以"):
            self.assertIn(w, zh.ZH_STOP)

    def test_negations_are_not_stopwords(self):
        for w in zh.ZH_NEGATION_MARKERS:
            self.assertNotIn(w, zh.ZH_STOP, f"否定词不得进停用词表: {w!r}")


# --------------------------------------------------------------------------- #
# 4. zh_negated：14 个标记 + 词内同形词豁免
# --------------------------------------------------------------------------- #

class ZhNegatedTest(unittest.TestCase):
    def test_all_contract_markers_hit(self):
        # 契约 §2 的 14 个标记（没有/没/从未/不曾/并非/并不/绝不/不是/不用/未/无/别/甭/毫无）
        for marker in ("没有", "没", "从未", "不曾", "并非", "并不", "绝不", "不是",
                       "不用", "未", "无", "别", "甭", "毫无"):
            self.assertTrue(zh.zh_negated(marker), f"未命中否定标记: {marker!r}")

    def test_markers_in_sentences(self):
        for text in ("我们没有保存那条记录", "她从未去过那里", "这并非最终版本",
                     "数据绝不外泄", "他不是负责人", "别担心", "毫无意义"):
            self.assertTrue(zh.zh_negated(text), text)

    def test_no_negation_texts(self):
        for text in ("", "今天天气不错", "我们上周去了杭州", "数据库迁移已经完成",
                     "所有人都能接受"):
            self.assertFalse(zh.zh_negated(text), text)

    def test_problem_idiom_is_exempt(self):
        # 契约 §8.1 明确要求不命中「没问题的」。实现用封闭豁免表压掉：
        # 「没问题」是约定俗成的正面评价，判成否定会把"没问题的方案"整类判反。
        self.assertFalse(zh.zh_negated("没问题的"))
        self.assertFalse(zh.zh_negated("没问题的方案已经确认了"))

    def test_lexicalized_compounds_are_exempt(self):
        # 单字标记（未/无/别）的词内假阳性用封闭豁免表抑制；未收录的同形词仍会假阳性
        # （方向更严，且只在 zh=True 路径上）。
        for text in ("未来的计划", "无论结果如何", "特别行政区", "区别很大",
                     "别人说的", "性别字段", "别墅区", "无线网络", "道歉与告别"):
            self.assertFalse(zh.zh_negated(text), text)

    def test_negation_inside_compound_still_detected(self):
        # 豁免是"封闭且最小"的：真否定不被它吞掉
        for text in ("没法用", "尚未完成", "无法启动", "别再说", "不用改"):
            self.assertTrue(zh.zh_negated(text), text)


# --------------------------------------------------------------------------- #
# 5. zh_temporal_terms：归一化 + 不重叠 + 确定性
# --------------------------------------------------------------------------- #

class ZhTemporalTest(unittest.TestCase):
    def test_contract_normalization_examples(self):
        self.assertEqual(zh.zh_temporal_terms("2026年3月5日"), {"2026-03-05"})
        self.assertEqual(zh.zh_temporal_terms("上周三"), {"上周三"})
        self.assertEqual(zh.zh_temporal_terms("上周"), {"上周"})
        self.assertEqual(zh.zh_temporal_terms("3月5日"), {"03-05"})
        self.assertEqual(zh.zh_temporal_terms("2026年3月"), {"2026-03"})
        self.assertEqual(zh.zh_temporal_terms("2026年"), {"2026"})
        self.assertEqual(zh.zh_temporal_terms("2026-03-05"), {"2026-03-05"})

    def test_relative_day_week_month_year(self):
        for word in ("昨天", "前天", "今天", "明天", "后天"):
            self.assertEqual(zh.zh_temporal_terms(word), {word})
        self.assertEqual(zh.zh_temporal_terms("下周三"), {"下周三"})
        self.assertEqual(zh.zh_temporal_terms("本周五"), {"本周五"})
        self.assertEqual(zh.zh_temporal_terms("星期日"), {"周日"})
        self.assertEqual(zh.zh_temporal_terms("上个月"), {"上个月"})
        self.assertEqual(zh.zh_temporal_terms("下个月"), {"下个月"})
        self.assertEqual(zh.zh_temporal_terms("这个月"), {"本月"})
        for word in ("去年", "今年", "明年"):
            self.assertEqual(zh.zh_temporal_terms(word), {word})

    def test_offsets_and_clock(self):
        self.assertEqual(zh.zh_temporal_terms("3天前"), {"3天前"})
        self.assertEqual(zh.zh_temporal_terms("2个月后"), {"2个月后"})
        self.assertEqual(zh.zh_temporal_terms("3点"), {"03:00"})
        self.assertEqual(zh.zh_temporal_terms("3点30分"), {"03:30"})

    def test_no_overlapping_double_count(self):
        # "2026年3月5日" 不能同时产出 "2026年"/"3月"/"3月5日"（重叠项会让 ⊆ 比较自相矛盾）
        self.assertEqual(len(zh.zh_temporal_terms("2026年3月5日")), 1)

    def test_invalid_calendar_date_yields_nothing(self):
        self.assertEqual(zh.zh_temporal_terms("2026年2月30日"), set())
        self.assertEqual(zh.zh_temporal_terms("2026年13月5日"), set())

    def test_multiple_terms_in_one_text(self):
        # 时刻不做"下午 +12"这类语义换算：3点 就是 03:00（换算会让「3点 vs 下午3点」不可比，
        # 反而制造假杀；时段词「下午」单独作为一项保留）。
        self.assertEqual(zh.zh_temporal_terms("昨天下午3点我们见了面，上周三也见过"),
                         {"昨天", "下午", "03:00", "上周三"})

    def test_deterministic(self):
        text = "2026年3月5日与上周三、昨天下午3点、3天前"
        first = zh.zh_temporal_terms(text)
        for _ in range(5):
            self.assertEqual(zh.zh_temporal_terms(text), first)

    def test_empty_text(self):
        self.assertEqual(zh.zh_temporal_terms(""), set())


class ZhTemporalSubsetTest(unittest.TestCase):
    def test_plain_subset_holds(self):
        self.assertTrue(zh.zh_temporal_subset({"上周三"}, {"上周三", "明天"}))
        self.assertTrue(zh.zh_temporal_subset(set(), {"上周三"}))
        self.assertFalse(zh.zh_temporal_subset({"明天"}, {"上周三"}))

    def test_more_precise_evidence_covers_coarser_claim(self):
        # 证据给 2026年3月5日，断言只说 3月5日 → 覆盖（不假杀）
        self.assertTrue(zh.zh_temporal_subset({"03-05"}, {"2026-03-05"}))
        self.assertTrue(zh.zh_temporal_subset({"2026-03"}, {"2026-03-05"}))
        self.assertTrue(zh.zh_temporal_subset({"2026"}, {"2026-03-05"}))
        self.assertTrue(zh.zh_temporal_subset({"上周"}, {"上周三"}))

    def test_more_precise_claim_is_not_covered(self):
        # 断言比证据更精确 → 不覆盖（年份/周几没有被证据接地）
        self.assertFalse(zh.zh_temporal_subset({"2026-03-05"}, {"03-05"}))
        self.assertFalse(zh.zh_temporal_subset({"2026-03-05"}, {"2026-03"}))
        self.assertFalse(zh.zh_temporal_subset({"上周三"}, {"上周"}))
        self.assertFalse(zh.zh_temporal_subset({"下周三"}, {"上周三"}))
        self.assertFalse(zh.zh_temporal_subset({"3天前"}, {"2天前"}))

    def test_cross_family_is_not_covered(self):
        # 不同族（相对天 vs 绝对日 vs 时段）不做语义换算，只认字符串相等
        self.assertFalse(zh.zh_temporal_subset({"昨天"}, {"2026-03-05"}))
        self.assertFalse(zh.zh_temporal_subset({"03:00"}, {"2026-03-05"}))


# --------------------------------------------------------------------------- #
# 6. 更正 / 指代 / 推测
# --------------------------------------------------------------------------- #

class ZhScenarioSignalTest(unittest.TestCase):
    def test_corrected(self):
        for text in ("不是这样的", "改成周三", "其实我记错了", "更正一下", "说错了",
                     "他改口了", "记错了", "这样不对"):
            self.assertTrue(zh.zh_corrected(text), text)
        for text in ("", "今天天气不错", "已经确认了"):
            self.assertFalse(zh.zh_corrected(text), text)

    def test_anaphor(self):
        for text in ("上次说的", "上回那个", "之前提过", "刚才那件事", "前面说过",
                     "我们说好的", "你说过要改"):
            self.assertTrue(zh.zh_anaphor(text), text)
        for text in ("", "今天天气不错", "已经确认了"):
            self.assertFalse(zh.zh_anaphor(text), text)

    def test_hedge(self):
        for text in ("可能是这样", "也许会变", "大概三天", "或许可行", "建议改成周三",
                     "应该是他", "可以考虑", "不妨试试", "大概率能成"):
            self.assertTrue(zh.zh_hedge(text), text)
        for text in ("", "确定是它", "已经确认了"):
            self.assertFalse(zh.zh_hedge(text), text)


# --------------------------------------------------------------------------- #
# 7. content_tokens / norm_tokens 的 zh 接入
# --------------------------------------------------------------------------- #

class ContentTokensZhTest(unittest.TestCase):
    TEXTS = (
        "",
        "   ",
        "Alice bought a red car on March 5th.",
        "我们用 PostgreSQL 2023 年做过一个项目，但是后来没有继续维护。",
        "今天天气不错，我们上周三去了杭州西湖。",
        "Mixed 中文 and English 混合文本 bigram 测试 2026年3月5日",
        "SELECT * FROM users WHERE NOT deleted = 0",
    )

    def test_zh_true_is_union_of_english_and_zh_minus_stop(self):
        for text in self.TEXTS:
            expected = content_tokens(text) | (set(zh.zh_tokens(text)) - zh.ZH_STOP)
            self.assertEqual(content_tokens(text, zh=True), expected, text)
            expected_v1 = (content_tokens(text, stop_mode="v1")
                           | (set(zh.zh_tokens(text)) - zh.ZH_STOP))
            self.assertEqual(content_tokens(text, stop_mode="v1", zh=True), expected_v1, text)

    def test_zh_true_gives_chinese_text_nonzero_tokens(self):
        # 这就是 v4 要修的结构性缺口：zh=False 时中文 0 token、判定空转
        self.assertEqual(content_tokens("今天天气不错，我们上周三去了杭州西湖。"), set())
        self.assertTrue(content_tokens("今天天气不错，我们上周三去了杭州西湖。", zh=True))

    def test_zh_stop_removed_from_zh_tokens(self):
        toks = content_tokens("我们的这个方案很好", zh=True)
        self.assertNotIn("我们", toks)
        self.assertNotIn("这个", toks)
        self.assertIn("方案", toks)

    def test_norm_tokens_zh_stems_ascii_only(self):
        # 汉 token 是 1~2 字，stem() 原样返回；ASCII 照旧词干化——两侧同函数即一致
        toks = norm_tokens("running 杭州", zh=True)
        self.assertIn("run", toks)
        self.assertIn("杭州", toks)

    def test_deterministic(self):
        text = "我们用 PostgreSQL 做过项目，但是后来没有继续维护。"
        first = content_tokens(text, zh=True)
        for _ in range(5):
            self.assertEqual(content_tokens(text, zh=True), first)

    def test_unknown_stop_mode_still_raises(self):
        with self.assertRaises(ValueError):
            content_tokens("x", stop_mode="v3")
        with self.assertRaises(ValueError):
            content_tokens("x", stop_mode="v3", zh=True)


# --------------------------------------------------------------------------- #
# 8. IDF：版本标签与缓存键必须含 zh
# --------------------------------------------------------------------------- #

class IdTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.ledger.append("今天天气不错，我们上周三去了杭州西湖。", source_id="s", episode_id="e")
        self.ledger.append("Alice bought a red car on March 5th.", source_id="s", episode_id="e")
        self.ledger.append("数据库迁移已经完成，下周上线。", source_id="s", episode_id="e")

    def test_version_labels_are_distinguishable(self):
        self.assertEqual(build_idf([{"a"}]).to_json()["version"], "idf-s2")
        self.assertEqual(build_idf([{"a"}], zh=True).to_json()["version"], "idf-s2+zh")
        self.assertEqual(idf_for_ledger(self.ledger).version, "idf-s2")
        self.assertEqual(idf_for_ledger(self.ledger, zh=True).version, "idf-s2+zh")

    def test_cache_key_includes_zh(self):
        a = idf_for_ledger(self.ledger)
        b = idf_for_ledger(self.ledger)
        c = idf_for_ledger(self.ledger, zh=True)
        d = idf_for_ledger(self.ledger, zh=True)
        self.assertIs(a, b)          # 同键缓存命中
        self.assertIs(c, d)
        self.assertIsNot(a, c)       # 不同 zh → 不同模型（中文 bigram 不能混进英文 df）
        self.assertNotEqual(a.to_json(), c.to_json())

    def test_zh_idf_contains_han_tokens_only_when_zh(self):
        en = idf_for_ledger(self.ledger).to_json()["df"]
        zh_df = idf_for_ledger(self.ledger, zh=True).to_json()["df"]
        self.assertFalse(any(zh.has_cjk(t) for t in en))
        self.assertTrue(any(zh.has_cjk(t) for t in zh_df))
        self.assertEqual(idf_for_ledger(self.ledger).n_docs, idf_for_ledger(self.ledger, zh=True).n_docs)

    def test_build_order_does_not_change_result(self):
        # 先用 zh=False 再用 zh=True 建的账本，与反序结果一致（缓存不串味）
        other = AnchorLedger(None, KEY)
        other.append("今天天气不错，我们上周三去了杭州西湖。", source_id="s", episode_id="e")
        other.append("Alice bought a red car on March 5th.", source_id="s", episode_id="e")
        other.append("数据库迁移已经完成，下周上线。", source_id="s", episode_id="e")
        first = idf_for_ledger(self.ledger, zh=True).to_json()
        _ = idf_for_ledger(self.ledger)                 # 中间插一次 zh=False
        self.assertEqual(idf_for_ledger(self.ledger, zh=True).to_json(), first)
        self.assertEqual(idf_for_ledger(other, zh=True).to_json(), first)


# --------------------------------------------------------------------------- #
# 9. SkeletonPolicy：新字段、v4()、describe()
# --------------------------------------------------------------------------- #

class PolicyV4Test(unittest.TestCase):
    def test_contract_v4_factory(self):
        pol = SkeletonPolicy.v4()
        self.assertEqual(POLICY_VERSION_V4, "skel-v4.0")
        self.assertEqual(pol.version, POLICY_VERSION_V4)
        self.assertTrue(pol.zh and pol.cite_set and pol.scenario)
        self.assertEqual(pol.tau_soft, 0.45)
        self.assertTrue(pol.hard_units and pol.temporal and pol.arg_slots and pol.dispatch)
        self.assertTrue(pol.v4_enabled and pol.v3_enabled)
        self.assertFalse(pol.v31_enabled)

    def test_default_policy_keeps_new_fields_off(self):
        pol = SkeletonPolicy()
        self.assertFalse(pol.zh)
        self.assertFalse(pol.cite_set)
        self.assertFalse(pol.scenario)
        self.assertFalse(pol.v4_enabled)
        self.assertEqual(pol.version, POLICY_VERSION)

    def test_describe_strings_unchanged_for_existing_versions(self):
        # describe() 只在 v4 开关开启时追加 v4(...) 段 → 既有版本字符串逐字不变
        self.assertEqual(
            SkeletonPolicy().describe(),
            "skel-v2.0(tau_soft=0.80, hard=1, idf=1, stem=1, stop=v2, temporal=1, session=1)",
        )
        self.assertEqual(
            SkeletonPolicy.v3().describe(),
            "skel-v3.0(tau_soft=0.80, hard=1, idf=1, stem=1, stop=v2, temporal=1, session=1, "
            "arg=1, dispatch=1)",
        )
        self.assertTrue(SkeletonPolicy.v3_c3().describe().startswith("skel-v3.1-C3("))
        self.assertNotIn("v4(", SkeletonPolicy.v3_c3().describe())
        self.assertEqual(POLICY_VERSION_V3, "skel-v3.0")
        self.assertEqual(POLICY_VERSION_V31, "skel-v3.1")
        self.assertEqual(POLICY_VERSION_V31_C3, "skel-v3.1-C3")

    def test_describe_reports_v4_flags(self):
        desc = SkeletonPolicy.v4().describe()
        self.assertTrue(desc.startswith("skel-v4.0("))
        self.assertTrue(desc.endswith("v4(zh=1, cite_set=1, scenario=1))"))

    def test_v4_is_only_reachable_by_opt_in(self):
        # 默认构造、v3、v3_c3、v3_1 都不得打开 v4 开关
        for pol in (SkeletonPolicy(), SkeletonPolicy.v3(), SkeletonPolicy.v3_c3(),
                    SkeletonPolicy.v3_1()):
            self.assertFalse(pol.zh or pol.cite_set or pol.scenario, pol.describe())

    def test_repeated_v4_construction_is_equal(self):
        self.assertEqual(SkeletonPolicy.v4(), SkeletonPolicy.v4())
        # 不可变 dataclass：同一组字段 → 同一个 hash（确定性；不涉及随机化）
        self.assertEqual(hash(SkeletonPolicy.v4()), hash(SkeletonPolicy.v4()))
        self.assertNotEqual(SkeletonPolicy.v4(), SkeletonPolicy())


# --------------------------------------------------------------------------- #
# 10. skeleton_check 的 zh 接入（Tier 1 从"空转"变成"可比较"）
# --------------------------------------------------------------------------- #

class SkeletonCheckZhTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.cn = self.ledger.append(
            "今天天气不错，我们上周三去了杭州西湖。", source_id="s", episode_id="e", speaker="用户")
        self.cn_other = self.ledger.append(
            "数据库迁移已经完成，下周上线，运维同事盯着。", source_id="s", episode_id="e",
            speaker="用户")

    def test_default_policy_is_vacuous_on_chinese(self):
        # 默认策略（zh=False）对中文仍是 0 token → 接地 vacuous 通过（历史行为，不许变）
        r = skeleton_check("我们上周三去了杭州西湖", [self.cn_other])
        self.assertEqual(r.n_soft_tokens, 0)
        self.assertTrue(r.ok)

    def test_zh_policy_grounds_chinese_claim(self):
        pol = SkeletonPolicy(zh=True, hard_units=False, temporal=False, tau_soft=0.45)
        r = skeleton_check("我们上周三去了杭州西湖", [self.cn], pol)
        self.assertGreater(r.n_soft_tokens, 0)
        self.assertTrue(r.ok, r.detail)

    def test_zh_policy_rejects_unrelated_chinese_evidence(self):
        pol = SkeletonPolicy(zh=True, hard_units=False, temporal=False, tau_soft=0.45)
        r = skeleton_check("我们上周三去了杭州西湖", [self.cn_other], pol)
        self.assertFalse(r.ok)
        self.assertEqual(r.rule, "soft_grounding")

    def test_zh_policy_result_is_deterministic(self):
        pol = SkeletonPolicy(zh=True, hard_units=False, temporal=False, tau_soft=0.45)
        first = skeleton_check("我们上周三去了杭州西湖", [self.cn], pol)
        for _ in range(3):
            self.assertEqual(skeleton_check("我们上周三去了杭州西湖", [self.cn], pol), first)

    def test_explicit_zh_false_equals_default(self):
        for claim, recs in (
            ("Alice bought a red car on March 5th.", [self.cn]),
            ("今天天气不错，我们上周三去了杭州西湖。", [self.cn]),
            ("数据库迁移已经完成。", [self.cn_other]),
        ):
            self.assertEqual(skeleton_check(claim, recs),
                             skeleton_check(claim, recs, SkeletonPolicy(zh=False)))


# --------------------------------------------------------------------------- #
# 11. obligations 的中文否定 / 时间路径（Tier 0）
# --------------------------------------------------------------------------- #

class ObligationsZhTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        # 中文语料里嵌了英文 SQL（含 NOT）——真实评测里 2/19 的 P_verbatim 假杀来源
        self.sql = self.ledger.append(
            "SELECT * FROM users WHERE NOT deleted = 0 我们用它查过数据",
            source_id="s", episode_id="e", speaker="用户")
        self.cn_neg = self.ledger.append(
            "我们没有保存那条记录，后来补上了。", source_id="s", episode_id="e", speaker="用户")
        self.cn_date = self.ledger.append(
            "会议在2026年3月5日举行，大家都到了。", source_id="s", episode_id="e", speaker="用户")
        self.cn_mixed = self.ledger.append(
            "他上周三去了杭州。 明天他回来。", source_id="s", episode_id="e", speaker="用户")
        self.en_sql = self.ledger.append(
            "SELECT * FROM users WHERE NOT deleted = 0", source_id="s", episode_id="e")

    @staticmethod
    def _by_name(results, name):
        return next(r for r in results if r.name == name)

    def test_sql_not_false_kill_is_fixed_by_zh_path(self):
        rec = self.sql
        quote = "我们用它查过数据"
        off = rec.content.find(quote)
        self.assertGreater(off, 0)
        # 改动前的英文逻辑：scope 里有英文 NOT、quote 里没有 → 判"丢掉否定"（假杀）
        legacy = self._by_name(apply_obligations(rec, quote, off, scope_mode="record"),
                               "preserve_negation")
        self.assertFalse(legacy.passed)
        # zh=True：任一侧含 CJK → 中文极性比较，不再用英文 NEGATION 判中文文本
        fixed = self._by_name(
            apply_obligations(rec, quote, off, scope_mode="record", zh=True),
            "preserve_negation")
        self.assertTrue(fixed.passed, fixed.detail)

    def test_chinese_negation_dropped_is_caught(self):
        rec = self.cn_neg
        quote = "那条记录，后来补上了。"
        off = rec.content.find(quote)
        self.assertGreater(off, 0)
        # zh=False：英文 NEGATION 看不见中文否定 → 放过（这就是中文否定 0/7 的缺口）
        legacy = self._by_name(apply_obligations(rec, quote, off, scope_mode="record"),
                               "preserve_negation")
        self.assertTrue(legacy.passed)
        # zh=True：scope 否定、quote 不否定 → 失败
        caught = self._by_name(
            apply_obligations(rec, quote, off, scope_mode="record", zh=True),
            "preserve_negation")
        self.assertFalse(caught.passed)
        self.assertIn("polarity", caught.detail)

    def test_chinese_negation_preserved_passes(self):
        rec = self.cn_neg
        res = apply_obligations(rec, rec.content, 0, scope_mode="record", zh=True)
        self.assertTrue(self._by_name(res, "preserve_negation").passed)

    def test_quote_spanning_negation_clause_passes(self):
        # v4.1 修复（Lead 裁决，见 docs/ALGO-V4-CONTRACT.md §3.2 修订）：
        # 极性比较改用**子句级**范围。引用是记录的连续子串 → 引用内部含否定时，
        # 它所在子句必然也含否定，"引用凭空加否定"这一方向不可能发生；
        # 真正的保护方向是「子句有否定、引用片段丢掉它」→ 见下一个用例。
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("甲已经好了。 但是乙没有去。", source_id="s", episode_id="e")
        quote = "甲已经好了。 但是乙没有去"
        self.assertGreater(rec.content.find(quote), -1)
        res = apply_obligations(rec, quote, rec.content.find(quote), scope_mode="clause", zh=True)
        neg = self._by_name(res, "preserve_negation")
        self.assertTrue(neg.passed, neg.detail)

    def test_same_clause_negation_dropped_by_fragment_is_caught(self):
        # 子句有否定、引用片段丢了它 → 必须失败（v4.1 保留的唯一方向）
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("甲已经好了。 但是乙没有去。", source_id="s", episode_id="e")
        quote = "但是乙"
        self.assertGreater(rec.content.find(quote), -1)
        res = apply_obligations(rec, quote, rec.content.find(quote), scope_mode="clause", zh=True)
        neg = self._by_name(res, "preserve_negation")
        self.assertFalse(neg.passed, neg.detail)
        self.assertIn("polarity", neg.detail)

    def test_both_non_cjk_keeps_english_logic(self):
        rec = self.en_sql
        quote = "SELECT * FROM users"
        off = rec.content.find(quote)
        for zh_flag in (False, True):
            res = apply_obligations(rec, quote, off, scope_mode="record", zh=zh_flag)
            self.assertFalse(self._by_name(res, "preserve_negation").passed, f"zh={zh_flag}")

    def test_zh_temporal_covered_by_more_precise_evidence(self):
        rec = self.cn_date
        quote = "3月5日举行，大家都到了。"
        off = rec.content.find(quote)
        self.assertGreater(off, 0)
        res = apply_obligations(rec, quote, off, scope_mode="record", zh=True)
        self.assertTrue(self._by_name(res, "preserve_temporal").passed)

    def test_zh_temporal_claim_more_precise_than_scope_fails(self):
        rec = self.cn_mixed
        quote = "他上周三去了杭州。 明天他回来"
        off = rec.content.find(quote)
        self.assertEqual(off, 0)
        # clause 范围 = 第一句「他上周三去了杭州。 」；quote 多了「明天」→ 未被证据覆盖
        res = apply_obligations(rec, quote, off, scope_mode="clause", zh=True)
        self.assertFalse(self._by_name(res, "preserve_temporal").passed)

    def test_zh_quote_with_no_temporal_is_not_blocked(self):
        # 契约 §3.2 字面口径：要求 quote 的时间项 ⊆ scope 的时间项；"无法提取则不阻断"。
        # 因此「quote 把时间整段丢掉」在 zh 路径**不**被标记（软谓词，方向为漏检）。
        # 这是已记录的契约口径，不是实现遗漏；若未来要收紧，须先改契约再改这里。
        rec = self.cn_mixed
        quote = "他上周三去了杭州。"
        res = apply_obligations(rec, quote, 0, scope_mode="record", zh=True)
        self.assertTrue(self._by_name(res, "preserve_temporal").passed)

    def test_zh_false_equals_default(self):
        # 不传 zh 与显式 zh=False 逐项相等（含 detail 文本）
        for rec in (self.sql, self.cn_neg, self.cn_date, self.cn_mixed, self.en_sql):
            for quote, off, mode in (
                (rec.content, 0, "record"),
                (rec.content[: max(1, len(rec.content) // 2)], 0, "clause"),
            ):
                self.assertEqual(
                    apply_obligations(rec, quote, off, scope_mode=mode),
                    apply_obligations(rec, quote, off, scope_mode=mode, zh=False),
                )

    def test_invalid_extra_still_raises_with_zh(self):
        from memory_kernel.obligations import TemplateError
        with self.assertRaises(TemplateError):
            apply_obligations(self.cn_neg, self.cn_neg.content, 0, extra=["nope"], zh=True)


# --------------------------------------------------------------------------- #
# 12. 回归门：zh=False 默认路径逐字节一致（硬编码期望）
# --------------------------------------------------------------------------- #

class ZhFalseRegressionTest(unittest.TestCase):
    """固定样例 + 硬编码期望：这些期望值来自改动前 HEAD 的实际输出，逐 token 抄录。"""

    def test_content_tokens_v2_fixed_samples(self):
        cases = {
            "": set(),
            "   ": set(),
            "!!!": set(),
            "Alice bought a red car on March 5th.": {"alice", "bought", "car", "march", "red"},
            "She wants his car and said it was fine": {"car", "fine"},
            "The quick brown fox jumps over the lazy dog":
                {"brown", "dog", "fox", "jumps", "lazy", "quick"},
            "Joanna's project don't work; running, ran, runs":
                {"joanna", "project", "ran", "running", "runs", "work"},
            "SELECT * FROM users WHERE NOT deleted = 0": {"deleted", "select", "users"},
            "It's 2026-03-05 and the ID is 42.": set(),
            # 中文在 zh=False 时**没有** token（v4 之前的真实缺口，回归门必须锁住）
            "今天天气不错，我们上周三去了杭州西湖。": set(),
            "从未见过这样的情况，并非所有人都能接受。": set(),
            "没有问题的方案我们已经确认了，不需要再改。": set(),
            "我们用 PostgreSQL 2023 年做过一个项目，但是后来没有继续维护。": {"postgresql"},
            "Mixed 中文 and English 混合文本 bigram 测试 2026年3月5日":
                {"bigram", "english", "mixed"},
        }
        for text, expected in cases.items():
            self.assertEqual(content_tokens(text), expected, text)
            self.assertEqual(content_tokens(text, zh=False), expected, text)

    def test_content_tokens_v1_fixed_samples(self):
        cases = {
            "She wants his car and said it was fine": {"car", "fine", "his", "said", "she", "wants"},
            "Joanna's project don't work; running, ran, runs":
                {"don't", "joanna's", "project", "ran", "running", "runs", "work"},
            "It's 2026-03-05 and the ID is 42.": {"it's"},
            "今天天气不错，我们上周三去了杭州西湖。": set(),
        }
        for text, expected in cases.items():
            self.assertEqual(content_tokens(text, stop_mode="v1"), expected, text)
            self.assertEqual(content_tokens(text, stop_mode="v1", zh=False), expected, text)

    def test_norm_tokens_fixed_samples(self):
        cases = {
            "Alice bought a red car on March 5th.": {"alic", "buy", "car", "march", "red"},
            "Joanna's project don't work; running, ran, runs": {"joanna", "project", "run", "work"},
            "The quick brown fox jumps over the lazy dog":
                {"brown", "dog", "fox", "jump", "lazy", "quick"},
            "今天天气不错，我们上周三去了杭州西湖。": set(),
        }
        for text, expected in cases.items():
            self.assertEqual(norm_tokens(text), expected, text)
            self.assertEqual(norm_tokens(text, zh=False), expected, text)

    def test_apply_obligations_fixed_samples(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("Alice bought a red car on March 5th, but she did not like the colour.",
                            source_id="s", episode_id="e", speaker="Alice")
        full = apply_obligations(rec, rec.content, 0, scope_mode="record")
        self.assertEqual([(r.name, r.passed, r.severity, r.detail) for r in full], [
            ("verbatim", True, "hard", ""),
            ("preserve_negation", True, "hard", ""),
            ("preserve_temporal", True, "soft", ""),
        ])
        partial = apply_obligations(rec, "Alice bought a red car", 0, scope_mode="record")
        self.assertEqual([(r.name, r.passed, r.severity, r.detail) for r in partial], [
            ("verbatim", True, "hard", ""),
            ("preserve_negation", False, "hard", "scope has negation ['not'], quote drops it"),
            ("preserve_temporal", False, "soft", "scope has temporal ['March 5th'], quote drops it"),
        ])
        # 非子串：verbatim 失败后立即返回（后续谓词不做）
        bad = apply_obligations(rec, "Alice bought a blue car", 0, scope_mode="record")
        self.assertEqual(len(bad), 1)
        self.assertFalse(bad[0].passed)

    def test_build_idf_fixed_sample(self):
        docs = [{"a", "b"}, {"b", "c"}, {"c"}]
        model = build_idf(docs)
        self.assertEqual(model.version, "idf-s2")
        self.assertEqual(model.n_docs, 3)
        self.assertEqual(model.to_json()["df"], {"a": 1, "b": 2, "c": 2})
        self.assertEqual(build_idf(docs, zh=False).to_json(), model.to_json())


class ZhFalseSnapshotTest(unittest.TestCase):
    """把固定样例的全部默认路径输出算 sha256，与改动前抓的快照比对（逐字节一致的机器证据）。"""

    def test_default_path_outputs_match_pre_change_snapshot(self):
        from memory_kernel.skeleton import stem

        texts = [
            "", "   ", "!!!",
            "Alice bought a red car on March 5th.",
            "She wants his car and said it was fine",
            "Joanna's project don't work; running, ran, runs",
            "The quick brown fox jumps over the lazy dog",
            "我们用 PostgreSQL 2023 年做过一个项目，但是后来没有继续维护。",
            "今天天气不错，我们上周三去了杭州西湖。",
            "没有问题的方案我们已经确认了，不需要再改。",
            "SELECT * FROM users WHERE NOT deleted = 0",
            "Mixed 中文 and English 混合文本 bigram 测试 2026年3月5日",
            "It's 2026-03-05 and the ID is 42.",
            "他说「这个功能已经上线了」，但其实是测试环境。",
            "从未见过这样的情况，并非所有人都能接受。",
        ]
        lex = {}
        for t in texts:
            lex[t] = {
                "content_v2": sorted(content_tokens(t)),
                "content_v1": sorted(content_tokens(t, stop_mode="v1")),
                "norm": sorted(norm_tokens(t)),
                "norm_uni": sorted(norm_tokens(t, stem_ok=False)),
            }
        stems = sorted(
            (w, stem(w)) for w in
            "wants want said says running ran studies studying information made making better worst "
            "car cares job transitioned transition created create".split()
        )
        ledger = AnchorLedger(None, KEY)
        recs = [
            ledger.append("Alice bought a red car on March 5th, but she did not like the colour.",
                          source_id="s1", episode_id="e1", speaker="Alice"),
            ledger.append("Bob moved to Berlin in June. He never visited Paris.",
                          source_id="s1", episode_id="e1", speaker="Bob"),
            ledger.append("我们没有保存那条记录，因为当时不需要它。",
                          source_id="s2", episode_id="e1", speaker="用户"),
            ledger.append("SELECT * FROM users WHERE NOT deleted = 0",
                          source_id="s2", episode_id="e1", speaker="用户"),
            ledger.append("上周三他去了杭州，明天可能还会去。",
                          source_id="s2", episode_id="e2", speaker="助手"),
        ]
        oblig = []
        for rec in recs:
            for (a, b) in [(0, min(len(rec.content), 20)), (0, len(rec.content))]:
                q = rec.content[a:b]
                for zmode in ("clause", "record"):
                    rs = apply_obligations(rec, q, a, scope_mode=zmode, asserted_speaker=None)
                    oblig.append([rec.id, a, b, zmode,
                                  [[r.name, r.passed, r.severity, r.detail] for r in rs]])
        idf_json = build_idf((norm_tokens(r.content) for r in recs)).to_json()
        idf_ledger = idf_for_ledger(ledger).to_json()
        policies = {"default": SkeletonPolicy(), "v3": SkeletonPolicy.v3(),
                    "v3c3": SkeletonPolicy.v3_c3()}
        claims = [
            ("Alice bought a red car on March 5th.", [recs[0]]),
            ("Alice bought a blue car.", [recs[0]]),
            ("Bob moved to Berlin in June.", [recs[1]]),
            ("我们没有保存那条记录。", [recs[2]]),
            ("我们保存了那条记录。", [recs[2]]),
            ("上周三他去了杭州。", [recs[4]]),
            ("今天他去了杭州。", [recs[4]]),
            ("SELECT * FROM users WHERE deleted = 0", [recs[3]]),
            ("Mixed 中文 and English 混合文本。", [recs[4]]),
        ]
        skel = []
        for pname, pol in policies.items():
            for claim, rs in claims:
                r = skeleton_check(claim, rs, pol)
                skel.append([pname, claim, r.ok, r.reason, round(r.grounding, 12), r.rule,
                             list(r.hard_missing), r.temporal_ok, r.n_soft_tokens,
                             r.arg_ok, r.n_units, round(r.unit_cov_min, 12), list(r.arg_missing),
                             r.plural_uncovered, r.version])
        payload = {"lex": lex, "stems": stems, "oblig": oblig, "idf": idf_json,
                   "idf_ledger": idf_ledger, "skel": skel}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        got = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        self.assertEqual(got, PRE_CHANGE_SNAPSHOT_SHA256,
                         "默认路径输出与改动前不一致（zh=False 回归门）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
