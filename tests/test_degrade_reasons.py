# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""v4.4 内核单测：`degrade_reason` 降级原因标签 + `scenario_superseded`「同一事实」判据。

覆盖任务裁决的两件事：
  A. `EnforcementResult.degrade_reason` / `.degrade_reasons`
     —— 只在 `verdict == "degraded"` 时填充；映射确定性（见 `enforce.DEGRADE_REASON_OF`）；
        多失败按固定谓词顺序取第一个为主原因。**不改判**：既有 verdict/cause/failed 不变。
  B. `scenario_superseded` 的「同一事实」判据
     —— 更正消息必须同时与 claim、与被引记录有内容词关联（wide >= 2，扣除更正标记自身
        token；或 narrow「否定标记后重复旧值」），否则 passed=True；正常样本仍要拦。

约束：不依赖 `internal corpus` 任何数据（离线可跑），只用 `memory_kernel` 公开 API；
判定路径不调模型、不联网、无时间/随机数。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import (  # noqa: E402
    AnchorLedger,
    Claim,
    Citation,
    EnforcementResult,
    ScenarioContext,
    SkeletonPolicy,
    enforce,
)
from memory_kernel.enforce import (  # noqa: E402
    DEGRADE_REASON_LABELS,
    DEGRADE_REASON_OF,
    DEGRADE_REASON_PREDICATE_ORDER,
    degrade_reason_for,
)
from memory_kernel.scenario import (  # noqa: E402
    SUPERSEDED_MARKER_TOKENS,
    SUPERSEDED_MIN_SHARED_TOKENS,
    SUPERSEDED_NEG_MARKERS,
    SUPERSEDED_NEG_REPETITION_WINDOW,
    check_scenario,
)

KEY = b"degrade-reasons-test-key-000001"   # 固定 32 字节，只为确定性

# 策略：与任务书/契约 §1 一致的能力层组合
ZH = SkeletonPolicy(zh=True)                                   # 只开中文词法
CITE = SkeletonPolicy(cite_set=True)                           # 引用集（英文用例）
SCEN_ZH = SkeletonPolicy(scenario=True, zh=True)               # 场景 + 中文词法
V4 = SkeletonPolicy.v4()                                       # 三个能力层全开


def soft_failed(result) -> list[str]:
    return [p.name for p in result.failed if not p.passed and p.severity == "soft"]


def hard_failed(result) -> list[str]:
    return [p.name for p in result.failed if not p.passed and p.severity == "hard"]


class _LedgerFixture(unittest.TestCase):
    """两张用途不同的账本：A 组造 degrade_reason 的各种软失败；B 组造 superseded 判据。"""

    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)

        def add(content, source_id="s0", episode_id="e0", turn_index=1, speaker="user"):
            return self.ledger.append(content, source_id=source_id, episode_id=episode_id,
                                      turn_index=turn_index, speaker=speaker)

        # ---- A 组：降级原因 ----
        # skeleton_grounding（部分证据）：断言内容在证据里只部分接地
        self.bakery = add("Gina told me she is opening a bakery next month.",
                          "sa", "ea", 1, "Jon")
        # preserve_temporal（时间项丢失）：scope 有 3月5日，引用片段没有
        self.meeting = add("The meeting is on March 5th about the budget.", "sa", "ea", 2, "Ann")
        # no_generalization（条件放大）：scope 有 if，引用片段丢掉条件
        self.picnic = add("If it rains, we will cancel the picnic.", "sa", "ea", 3, "Ann")
        # 引用集：主引用 + 无关 supporting
        self.car = add("Alice bought a red car on March 5th.", "sc", "ec", 1, "Alice")
        self.weather = add("The weather in Berlin was cold last week.", "sc", "ec", 2, "Bob")
        # skeleton_temporal（骨架族 fallback）：断言有时间、证据没有时间
        self.plain = add("Gina is opening a bakery.", "sa", "ea", 4, "Jon")

        # ---- B 组：superseded 同一事实判据（中文，zh=True）----
        # 正常样本：被引记录是"事实陈述"，后续用户更正（narrow「不是昨天」型 + 否定重复旧值）
        self.fact = add("我昨天和我妈去菜市场买了三斤苹果，花了大概二十块钱。",
                        "sb", "eb", 1, "user")
        self.fact_side = add("我记得那家菜市场的苹果很便宜。", "sb", "eb", 2, "user")
        self.fact_fix = add("不对，那是上周三的事，不是昨天，我们买的是橙子。",
                            "sb", "eb", 3, "user")
        # 误标形态 1（S_timeline_ctrl 同形）：更正消息与被引记录只共享 1 个内容词「皮肤」
        self.skin_q = add("有没有那种皮肤的传感器", "sd", "ed", 1, "user")
        self.skin_fix = add("机器人的触觉表皮我以为是不是只要压力、温度就足够了",
                            "sd", "ed", 2, "user")
        # 误标形态 2（S_superseded_ctrl 同形）：更正长消息与被引记录只共享「不是/格式」，
        # 其中「不是」是更正标记自身的 token，扣掉后只剩 1 个内容词
        self.fmt_q = add("这不是输出格式那处弄错了么？别动实现，只需说明输出参数的样式",
                         "se", "ee", 1, "user")
        self.fmt_fix = add("def main(arg1):\n    # 值不是最终版本\n    return 格式(dict)",
                           "se", "ee", 2, "user")
        # wide 命中：更正消息与被引记录共享 >= 2 个内容词
        self.stat_q = add("某地区按季统计工业增加值并持续近五年", "sf", "ef", 1, "user")
        self.stat_fix = add("季度和工业增加值的口径还是不对，重新算", "sf", "ef", 2, "user")
        # claim_link 守卫：更正消息与被引记录共享 >= 2 个内容词，但 claim 说的是另一回事
        self.price_q = add("我昨天去菜市场买苹果和香蕉，总共花了五十块钱。", "sg", "eg", 1, "user")
        self.price_fix = add("苹果不是昨天的，香蕉是前天买的。", "sg", "eg", 2, "user")

    def excerpt(self, rec, text=None, **kw):
        return Claim(Citation(rec.id, 0, rec.content),
                     text=(rec.content if text is None else text), **kw)

    def quote(self, rec, quote, text=None, **kw):
        return Claim(Citation(rec.id, rec.content.find(quote), quote),
                     text=(quote if text is None else text), **kw)


# --------------------------------------------------------------------------- #
# A. degrade_reason / degrade_reasons
# --------------------------------------------------------------------------- #

class DegradeReasonMappingTest(_LedgerFixture):
    """映射表本身的确定性（纯函数层）。"""

    def test_declared_mapping_matches_task_spec(self):
        self.assertEqual(DEGRADE_REASON_OF, {
            "skeleton_grounding": "partial_evidence",
            "skeleton_hard_unit_grounding": "partial_evidence",
            "skeleton_arg_object": "argument_mismatch",
            "skeleton_arg_subject": "argument_mismatch",
            "skeleton_arg_predicate": "argument_mismatch",
            "skeleton_arg_pred_family": "argument_mismatch",
            "skeleton_arg_unit_coverage": "argument_mismatch",
            "skeleton_plural_uncovered": "argument_mismatch",
            "scenario_cross_session": "cross_session",
            "scenario_superseded": "scenario_superseded",
            # Plan Z C2：第 5 谓词（当前 hard-only）；映射与 skeleton_grounding 同义，
            # 保留是为了契约枚举完整、且将来若改 soft 时措辞一致
            "scenario_evidence_incomplete": "partial_evidence",
            "over_citation": "over_citation",
            "preserve_temporal": "temporal",
            "no_generalization": "over_generalization",
            # internal-task-internal-task（F1 Phase 1，v4.7-meta，soft-only）：量词泛化复用
            # over_generalization 口径；来源归属单列封闭标签
            "quantifier_generalization": "over_generalization",
            "source_attribution": "source_attribution",
        })

    def test_mapping_is_total_and_deterministic(self):
        for predicate, reason in DEGRADE_REASON_OF.items():
            with self.subTest(predicate=predicate):
                self.assertEqual(degrade_reason_for(predicate), reason)
                self.assertIn(degrade_reason_for(predicate), DEGRADE_REASON_LABELS)
                for _ in range(5):
                    self.assertEqual(degrade_reason_for(predicate), reason)

    def test_other_skeleton_reasons_keep_their_name(self):
        # 任务书：其他 skeleton_* → skeleton_<reason>（grounding 是唯一被单独命名的）
        self.assertEqual(degrade_reason_for("skeleton_temporal"), "skeleton_temporal")
        self.assertEqual(degrade_reason_for("skeleton_whatever"), "skeleton_whatever")

    def test_unknown_predicate_falls_back_to_its_own_name(self):
        # 未收录的软谓词不允许"无原因"：回退成谓词名本身（字段始终可解释）
        self.assertEqual(degrade_reason_for("some_future_predicate"), "some_future_predicate")

    def test_fixed_predicate_order_is_declared_and_stable(self):
        self.assertEqual(DEGRADE_REASON_PREDICATE_ORDER,
                         ("preserve_temporal", "no_generalization",
                          # internal-task-internal-task（v4.7-meta，soft-only）：Tier 0 元数据谓词
                          "quantifier_generalization", "source_attribution",
                          "skeleton_grounding",
                          "skeleton_hard_unit_grounding",
                          "skeleton_arg_object", "skeleton_arg_subject", "skeleton_arg_predicate",
                          "skeleton_arg_pred_family", "skeleton_arg_unit_coverage",
                          "skeleton_plural_uncovered",
                          "over_citation", "scenario_cross_session", "scenario_superseded",
                          # Plan Z C2：第 5 谓词排在场景谓词末尾（追加，不改既有主次）
                          "scenario_evidence_incomplete"))
        # 顺序即降级原因的主次（与 enforce() 的 1→5 级执行顺序一致）
        self.assertEqual(tuple(DEGRADE_REASON_OF[p] for p in DEGRADE_REASON_PREDICATE_ORDER),
                         ("temporal", "over_generalization", "over_generalization",
                          "source_attribution", "partial_evidence", "partial_evidence",
                          "argument_mismatch", "argument_mismatch", "argument_mismatch",
                          "argument_mismatch", "argument_mismatch", "argument_mismatch",
                          "over_citation", "cross_session", "scenario_superseded",
                          "partial_evidence"))


class DegradeReasonEnforceTest(_LedgerFixture):
    """每种映射都有一个真实 `enforce()` 调用覆盖（不是只测纯函数）。"""

    def test_verified_has_no_degrade_reason(self):
        r = enforce(self.ledger, self.excerpt(self.bakery), policy=ZH)
        self.assertEqual(r.verdict, "verified")
        self.assertIsNone(r.degrade_reason)
        self.assertEqual(r.degrade_reasons, ())

    def test_skeleton_grounding_maps_to_partial_evidence(self):
        r = enforce(self.ledger,
                    Claim(Citation(self.bakery.id, 0, self.bakery.content),
                          text="Alice bought a red car on March 5th."),
                    policy=ZH)
        self.assertEqual(r.verdict, "degraded")
        self.assertIn("skeleton_grounding", soft_failed(r))
        self.assertEqual(r.degrade_reason, "partial_evidence")
        self.assertEqual(r.degrade_reasons, ("partial_evidence",))

    def test_over_citation_maps_to_over_citation(self):
        claim = Claim(Citation(self.car.id, 0, self.car.content),
                      text=self.car.content,
                      supporting=(Citation(self.weather.id, 0, self.weather.content),))
        r = enforce(self.ledger, claim, policy=CITE)
        self.assertEqual(r.verdict, "degraded")
        self.assertEqual(r.degrade_reason, "over_citation")
        self.assertEqual(r.degrade_reasons, ("over_citation",))

    def test_redundant_over_citation_reasons_are_deduplicated(self):
        # 两个 supporting 各触发一次 over_citation → reasons 只保留一个，不重复
        claim = Claim(Citation(self.car.id, 0, self.car.content),
                      text=self.car.content,
                      supporting=(Citation(self.weather.id, 0, self.weather.content),
                                  Citation(self.bakery.id, 0, self.bakery.content)))
        r = enforce(self.ledger, claim, policy=CITE)
        self.assertEqual(r.verdict, "degraded")
        self.assertEqual(r.degrade_reasons, ("over_citation",))
        self.assertEqual([p.name for p in r.failed if not p.passed and p.name == "over_citation"],
                         ["over_citation", "over_citation"])

    def test_preserve_temporal_maps_to_temporal(self):
        r = enforce(self.ledger, self.quote(self.meeting, "about the budget"), policy=ZH)
        self.assertEqual(r.verdict, "degraded")
        self.assertIn("preserve_temporal", soft_failed(r))
        self.assertEqual(r.degrade_reason, "temporal")

    def test_no_generalization_maps_to_over_generalization(self):
        r = enforce(self.ledger, self.quote(self.picnic, "we will cancel the picnic"),
                    policy=ZH, extra=["no_generalization"])
        self.assertEqual(r.verdict, "degraded")
        self.assertIn("no_generalization", soft_failed(r))
        self.assertEqual(r.degrade_reason, "over_generalization")

    def test_skeleton_temporal_uses_skeleton_fallback_reason(self):
        r = enforce(self.ledger,
                    Claim(Citation(self.plain.id, 0, self.plain.content),
                          text="Gina will open a bakery next month."),
                    policy=SkeletonPolicy(zh=True, tau_soft=0.0))
        self.assertEqual(r.verdict, "degraded")
        self.assertIn("skeleton_temporal", soft_failed(r))
        self.assertEqual(r.degrade_reason, "skeleton_temporal")

    def test_scenario_cross_session_maps_to_cross_session(self):
        rec = self.ledger.append("上次说好的方案我已经准备好了。", source_id="sh", episode_id="eh",
                                 turn_index=1, speaker="user")
        claim = Claim(Citation(rec.id, 0, rec.content), text="上次说好的方案")
        r = enforce(self.ledger, claim, policy=SCEN_ZH,
                    context=ScenarioContext(context_session_id="other"))
        self.assertEqual(r.verdict, "degraded")
        self.assertIn("scenario_cross_session", soft_failed(r))
        self.assertEqual(r.degrade_reason, "cross_session")
        self.assertEqual(r.degrade_reasons, ("cross_session",))

    def test_multi_failure_main_reason_follows_fixed_predicate_order(self):
        # 时间项丢失（Tier 0 软谓词）先于骨架（Tier 1）→ 主原因 temporal，全部原因按固定顺序
        claim = Claim(Citation(self.meeting.id, self.meeting.content.find("about the budget"),
                               "about the budget"),
                      text="Alice bought a red car on March 5th.")
        r = enforce(self.ledger, claim, policy=ZH)
        self.assertEqual(r.verdict, "degraded")
        self.assertEqual(r.degrade_reasons, ("temporal", "partial_evidence"))
        self.assertEqual(r.degrade_reason, "temporal")

    def test_main_reason_ignores_failed_list_order(self):
        # 同一批软失败下，主原因只由固定谓词顺序决定（与 failed 的拼接顺序无关）
        claim = Claim(Citation(self.car.id, 0, self.car.content),
                      text=self.car.content,
                      supporting=(Citation(self.weather.id, 0, self.weather.content),))
        r = enforce(self.ledger, claim, policy=SkeletonPolicy(cite_set=True, zh=True))
        self.assertEqual(r.verdict, "degraded")
        names = soft_failed(r)
        self.assertIn("over_citation", names)
        self.assertEqual(r.degrade_reason, r.degrade_reasons[0])
        self.assertEqual(set(r.degrade_reasons),
                         {degrade_reason_for(n) for n in names})

    def test_rejected_has_no_degrade_reason(self):
        claim = Claim(Citation(self.car.id, 0, "Alice bought a blue truck."),
                      text="Alice bought a blue truck.")
        r = enforce(self.ledger, claim, policy=CITE)
        self.assertEqual(r.verdict, "rejected")
        self.assertIsNone(r.degrade_reason)
        self.assertEqual(r.degrade_reasons, ())

    def test_coverage_full_false_has_no_degrade_reason(self):
        # coverage_full=False 只跑无条件全量项 → verified，不带降级原因
        claim = Claim(Citation(self.car.id, 0, self.car.content),
                      text=self.car.content,
                      supporting=(Citation(self.weather.id, 0, self.weather.content),))
        r = enforce(self.ledger, claim, policy=CITE, coverage_full=False)
        self.assertEqual(r.verdict, "verified")
        self.assertIsNone(r.degrade_reason)
        self.assertEqual(r.degrade_reasons, ())

    def test_verdict_cause_failed_are_not_changed_by_the_new_fields(self):
        # 新字段只加标签：既有判定三件套逐位与不含新字段时的口径一致
        claim = Claim(Citation(self.bakery.id, 0, self.bakery.content),
                      text="Alice bought a red car on March 5th.")
        r = enforce(self.ledger, claim, policy=ZH)
        self.assertEqual((r.verdict, r.rejected_cause), ("degraded", None))
        self.assertEqual([(p.name, p.passed, p.severity) for p in r.failed],
                         [("verbatim", True, "hard"), ("preserve_negation", True, "hard"),
                          ("preserve_temporal", True, "soft"),
                          ("skeleton_grounding", False, "soft")])
        self.assertEqual(hard_failed(r), [])

    def test_repeated_calls_are_identical(self):
        claim = Claim(Citation(self.car.id, 0, self.car.content),
                      text=self.car.content,
                      supporting=(Citation(self.weather.id, 0, self.weather.content),))
        first = enforce(self.ledger, claim, policy=V4)
        for _ in range(5):
            again = enforce(self.ledger, claim, policy=V4)
            self.assertEqual((again.verdict, again.rejected_cause),
                             (first.verdict, first.rejected_cause))
            self.assertEqual((again.degrade_reason, again.degrade_reasons),
                             (first.degrade_reason, first.degrade_reasons))

    def test_dataclass_defaults_are_none_and_empty(self):
        r = EnforcementResult(verdict="verified", coverage={})
        self.assertIsNone(r.degrade_reason)
        self.assertEqual(r.degrade_reasons, ())
        # 位置参数顺序不变（既有调用点不受影响）
        r2 = EnforcementResult("degraded", {}, None, [], 0.0, "a_1")
        self.assertEqual((r2.verdict, r2.anchor_id), ("degraded", "a_1"))
        self.assertIsNone(r2.degrade_reason)


# --------------------------------------------------------------------------- #
# B. scenario_superseded「同一事实」判据
# --------------------------------------------------------------------------- #

class SupersededSameFactTest(_LedgerFixture):
    def _superseded(self, claim, policy=SCEN_ZH):
        results = {p.name: p for p in check_scenario(self.ledger, claim, ScenarioContext(),
                                                     policy=policy)}
        return results["scenario_superseded"]

    def test_constants_are_declared(self):
        self.assertEqual(SUPERSEDED_MIN_SHARED_TOKENS, 2)
        self.assertEqual(SUPERSEDED_NEG_REPETITION_WINDOW, 4)
        # 更正标记自身 token 由 ZH_CORRECTED 派生（不是第二张手写表）
        self.assertEqual(SUPERSEDED_MARKER_TOKENS,
                         frozenset({"不是", "改成", "其实", "更正", "说错", "改口",
                                    "记错", "不对"}))
        # 单字否定词不进 narrow 判据的标记表（出现密度太高）
        self.assertEqual(SUPERSEDED_NEG_MARKERS,
                         ("没有", "从未", "不曾", "并非", "并不", "绝不", "不是", "不用", "毫无"))

    # -- 正常样本必须仍被标记（v4.6：soft → degraded） ---------------------- #
    def test_narrow_negated_repetition_still_flags(self):
        # fixture 形态：更正只重复了被替换的旧值「昨天」→ wide 不够 2 个词，narrow 命中
        p = self._superseded(self.excerpt(self.fact))
        self.assertEqual((p.passed, p.severity), (False, "soft"))
        self.assertIn(self.fact_fix.id, p.detail)

    def test_wide_content_overlap_flags(self):
        # 更正消息与被引记录共享 >= 2 个内容词（季度/工业/增加值…）→ wide 命中
        p = self._superseded(self.excerpt(self.stat_q))
        self.assertEqual((p.passed, p.severity), (False, "soft"))
        self.assertIn(self.stat_fix.id, p.detail)

    def test_enforce_superseded_degrades_with_scenario_reason(self):
        r = enforce(self.ledger, self.excerpt(self.fact), policy=SCEN_ZH,
                    context=ScenarioContext())
        self.assertEqual(r.verdict, "degraded")
        self.assertIsNone(r.rejected_cause)
        self.assertIn("scenario_superseded", soft_failed(r))
        self.assertEqual(r.degrade_reason, "scenario_superseded")

    # -- 误标形态必须被判 passed ------------------------------------------- #
    def test_single_shared_content_token_is_not_the_same_fact(self):
        # S_timeline_ctrl 同形：更正消息与被引记录只共享「皮肤」1 个内容词 → 不同事实
        p = self._superseded(self.excerpt(self.skin_q))
        self.assertTrue(p.passed)
        self.assertEqual(p.detail, "")

    def test_correction_marker_tokens_do_not_count_as_content_overlap(self):
        # S_superseded_ctrl 同形：原始交集 2 个词里「不是」是更正标记自身 → 只剩「格式」1 个
        p = self._superseded(self.excerpt(self.fmt_q))
        self.assertTrue(p.passed)
        self.assertEqual(p.detail, "")

    def test_claim_link_is_required(self):
        # 更正消息与被引记录共享 >= 2 个内容词（苹果/香蕉/昨天），但 claim 说的是价格 →
        # 更正的不是这条断言 → passed（否则会把无关更正算到这条引用头上）
        claim = self.quote(self.price_q, "总共花了五十块钱")
        self.assertTrue(self._superseded(claim).passed)

    def test_claim_must_overlap_the_cited_record(self):
        # 断言与被引记录毫无内容词交集 → 引用与断言无关，不判"已被更正"
        p = self._superseded(self.excerpt(self.fact, text="今天的天气不错"))
        self.assertTrue(p.passed)
        self.assertEqual(p.detail, "")

    def test_unrelated_correction_in_same_episode_does_not_flag(self):
        # 同一会话里另起话题的更正，与被引记录无内容词交集 → 不串味
        rec = self.ledger.append("我上个月搬到了新家，离公司近多了。",
                                 source_id="sj", episode_id="ej", turn_index=1, speaker="user")
        self.ledger.append("不是这样，搬家的日期记错了，是前年。",
                           source_id="sj", episode_id="ej", turn_index=2, speaker="user")
        p = self._superseded(self.excerpt(rec))
        self.assertTrue(p.passed)

    def test_assistant_correction_is_ignored(self):
        rec = self.ledger.append("我昨天去菜市场买了三斤苹果。",
                                 source_id="si", episode_id="ei", turn_index=1, speaker="user")
        self.ledger.append("不对，不是昨天，是前天。",
                           source_id="si", episode_id="ei", turn_index=2, speaker="assistant")
        p = self._superseded(self.excerpt(rec))
        self.assertTrue(p.passed)

    def test_zh_false_is_unassessable_and_passes(self):
        # 已文档化的单侧性：zh=False 时中文断言没有内容词 → 不可判 → passed（不编结论）
        p = self._superseded(self.excerpt(self.fact), policy=SkeletonPolicy(scenario=True))
        self.assertTrue(p.passed)
        self.assertIn("no assessable content token", p.detail)

    # -- 确定性 ------------------------------------------------------------ #
    def test_repeated_calls_are_identical(self):
        claim = self.excerpt(self.fact)
        first = check_scenario(self.ledger, claim, ScenarioContext(), policy=V4)
        for _ in range(10):
            self.assertEqual(check_scenario(self.ledger, claim, ScenarioContext(),
                                            policy=V4), first)


if __name__ == "__main__":
    unittest.main()
