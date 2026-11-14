# SPDX-License-Identifier: Apache-2.0
"""v4 能力层单测：引用集校验（`cite_set.py`）+ 场景谓词（`scenario.py`）+ `enforce()` 接入。

契约：`docs/ALGO-V4-CONTRACT.md` §4 / §5 / §6 / §7 / §8。

覆盖任务书要求的四类：
  * 过度引用：A 相关 + C 无关 → `degraded` 且 `over_citation` 在 failed；A + B 都相关 → `verified`；
    anchor 不存在 / quote 非连续子串 → hard（`rejected`，cause `citation_set`）；
  * 四个场景谓词各 1 正 1 反（timeline / cross_session / speaker_scope / superseded）；
  * `context=None` 时结果与 v3 逐字节一致；默认策略（`SkeletonPolicy()`）在所有新用例上行为不变；
  * 确定性：同一输入重复运行逐位一致（无时间/随机/模型/网络）。

测试不依赖 `eval/` 下任何数据（离线可跑），只使用 `memory_kernel` 的公开 API。
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import (  # noqa: E402
    Claim,
    Citation,
    SkeletonPolicy,
    enforce,
)
from memory_kernel.cite_set import OVER_CITATION_PREDICATES, check_citation_set  # noqa: E402
from memory_kernel.enforce import REJECT_CAUSES  # noqa: E402
from memory_kernel.ledger import AnchorLedger  # noqa: E402
from memory_kernel.scenario import (  # noqa: E402
    SCENARIO_PREDICATES,
    ScenarioContext,
    check_scenario,
)
from memory_kernel.skeleton import POLICY_VERSION, POLICY_VERSION_V4  # noqa: E402

KEY = hashlib.sha256(b"cite-scenario-test").digest()

# 与 `docs/ALGO-V4-CONTRACT.md` §4 的严重度约定一致
CITE = SkeletonPolicy(cite_set=True)                       # 引用集开、zh 关（英文用例）
CITE_ZH = SkeletonPolicy(cite_set=True, zh=True)           # 引用集 + 中文词法
SCEN = SkeletonPolicy(scenario=True)                       # 场景开、zh 关
SCEN_ZH = SkeletonPolicy(scenario=True, zh=True)           # 场景 + 中文词法
V4 = SkeletonPolicy.v4()                                   # 三个能力层全开（契约 §1）


def failed_pairs(result) -> list[tuple[str, bool]]:
    return [(p.name, p.passed) for p in result.failed]


def sup(rec) -> tuple[Citation, ...]:
    """把一条记录包成"逐字引用自身"的 supporting 元组。"""
    return (Citation(rec.id, 0, rec.content),)


def soft_failed(result) -> list[str]:
    return [p.name for p in result.failed if not p.passed and p.severity == "soft"]


def hard_failed(result) -> list[str]:
    return [p.name for p in result.failed if not p.passed and p.severity == "hard"]


class _Fixture(unittest.TestCase):
    """账本按固定顺序构建：前四条与「改动前基线快照」的输入完全一致（见下方冻结表）。"""

    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)

        def add(content, source_id, episode_id, turn_index, speaker):
            return self.ledger.append(content, source_id=source_id, episode_id=episode_id,
                                      turn_index=turn_index, speaker=speaker)

        # ---- 与改动前基线同一批输入（英文） ----
        self.aero = add("I finally got to see Aerosmith live last night, it was amazing.",
                        "s0", "e0", 1, "Dave")
        self.bakery = add("Gina told me she is opening a bakery next month.",
                          "s0", "e0", 2, "Jon")
        # ---- 与改动前基线同一批输入（中文） ----
        self.zh_market = add("我昨天和我妈去菜市场买了三斤苹果，花了大概二十块钱。",
                             "s1", "e1", 1, "user")
        # ---- 引用集相关性用例（英文，词面交集可手算） ----
        self.car = add("Alice bought a red car on March 5th.", "s2", "e2", 1, "Alice")
        self.car2 = add("Alice drove the red car to the market yesterday.",
                        "s2", "e2", 2, "Alice")
        self.weather = add("The weather in Berlin was cold last week.",
                           "s2", "e2", 3, "Bob")
        # ---- 场景谓词用例（中文） ----
        self.zh_fix = add("不对，那是上周三的事，不是昨天，我们买的是橙子。",
                          "s1", "e1", 3, "user")
        self.zh_market2 = add("我记得那家菜市场的苹果很便宜。", "s1", "e1", 2, "user")
        self.zh_work = add("我们在 2026年3月5日 上线了新的结算系统。",
                           "s3", "e3", 2, "user")
        self.zh_hedge = add("这个方案应该可以，可能下周就能上线。",
                            "s4", "e4", 1, "assistant")
        self.zh_plain = add("我来帮你查一下天气。", "s3", "e3", 3, "assistant")

    # -- 常用 claim 构造器 -------------------------------------------------- #
    def car_claim(self, support=(), text=None, anchor=None):
        return Claim(Citation((anchor or self.car).id, 0, (anchor or self.car).content),
                     text=self.car.content if text is None else text,
                     supporting=support)

    def excerpt(self, rec, **kw):
        return Claim(Citation(rec.id, 0, rec.content), **kw)


# --------------------------------------------------------------------------- #
# 1. 引用集校验（模块级，直接调 check_citation_set）
# --------------------------------------------------------------------------- #

class CiteSetUnitTest(_Fixture):
    def test_no_supporting_returns_empty_list(self):
        self.assertEqual(check_citation_set(self.ledger, self.car_claim(), zh=False, policy=CITE), [])

    def test_irrelevant_supporting_is_soft_over_citation(self):
        res = check_citation_set(self.ledger, self.car_claim(sup(self.weather)),
                                 zh=False, policy=CITE)
        self.assertEqual(len(res), 1)
        (r,) = res
        self.assertEqual(r.name, "over_citation")
        self.assertFalse(r.passed)
        self.assertEqual(r.severity, "soft")

    def test_redundant_supporting_is_over_citation(self):
        # v4.3：相关性判据改为「增量覆盖」。car2 与 claim 同题，但它的内容词已被主引用
        # （car 记录）全部覆盖 → 对断言不贡献新证据，属冗余引用（过度引用）。旧判据
        # （"词面有交集即通过"）对这类冗余是放行的，真实评测里 O_overcite 只抓到 3/19。
        res = check_citation_set(self.ledger, self.car_claim(sup(self.car2)),
                                 zh=False, policy=CITE)
        (r,) = res
        self.assertEqual((r.name, r.passed, r.severity), ("over_citation", False, "soft"))

    def test_supporting_with_new_evidence_passes(self):
        # 合法多引用：claim 需要两条证据，第二条贡献了主引用没有的 claim 内容词
        claim = Claim(
            Citation(self.car.id, 0, self.car.content),
            text="Alice bought a red car on March 5th and drove it to the market yesterday.",
            supporting=(Citation(self.car2.id, 0, self.car2.content),),
        )
        (r,) = check_citation_set(self.ledger, claim, zh=False, policy=CITE)
        self.assertEqual((r.name, r.passed, r.severity), ("over_citation", True, "soft"))

    def test_mixed_supporting_is_checked_one_by_one(self):
        # 第一条（car2）贡献新证据 → 通过；第二条（weather）无增量 → 过度引用
        claim = Claim(
            Citation(self.car.id, 0, self.car.content),
            text="Alice bought a red car on March 5th and drove it to the market yesterday.",
            supporting=(Citation(self.car2.id, 0, self.car2.content),
                        Citation(self.weather.id, 0, self.weather.content)),
        )
        res = check_citation_set(self.ledger, claim, zh=False, policy=CITE)
        self.assertEqual([(r.name, r.passed) for r in res],
                         [("over_citation", True), ("over_citation", False)])

    def test_missing_anchor_is_hard_over_citation_anchor(self):
        res = check_citation_set(
            self.ledger,
            Claim(Citation(self.car.id, 0, self.car.content), text=None,
                  supporting=(Citation("a_999999", 0, "x"),)),
            zh=False, policy=CITE)
        (r,) = res
        self.assertEqual((r.name, r.passed, r.severity),
                         ("over_citation_anchor", False, "hard"))
        self.assertIn("a_999999", r.detail)

    def test_noncontiguous_quote_is_hard_over_citation_verbatim(self):
        res = check_citation_set(
            self.ledger,
            self.car_claim((Citation(self.car2.id, 0, "Alice bought a blue truck."),)),
            zh=False, policy=CITE)
        (r,) = res
        self.assertEqual((r.name, r.passed, r.severity),
                         ("over_citation_verbatim", False, "hard"))

    def test_hard_failure_short_circuits_relevance_for_that_entry(self):
        res = check_citation_set(
            self.ledger,
            self.car_claim((Citation(self.weather.id, 0, "not the record text"),
                            Citation(self.weather.id, 0, self.weather.content))),
            zh=False, policy=CITE)
        self.assertEqual([(r.name, r.severity) for r in res],
                         [("over_citation_verbatim", "hard"), ("over_citation", "soft")])

    def test_claim_text_absent_marks_relevance_unassessable(self):
        # 摘录型断言不提供 text：两个 hard 检查照做，相关性不可判 → passed（不编结论）
        claim = Claim(Citation(self.car.id, 0, self.car.content), text=None,
                      supporting=sup(self.weather))
        res = check_citation_set(self.ledger, claim, zh=False, policy=CITE)
        (r,) = res
        self.assertTrue(r.passed)
        self.assertIn("not assessable", r.detail)

    def test_zh_false_chinese_claim_flags_every_supporting(self):
        # 已文档化的单侧性：zh=False 时中文断言文本 content_tokens 为空 ⇒ 相关性恒判"无交集"。
        # 该行为被锁在这里，避免把"打开 cite_set 却没开 zh"误当成可用配置。
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content, supporting=sup(self.zh_market2))
        (r,) = check_citation_set(self.ledger, claim, zh=False, policy=CITE)
        self.assertEqual((r.name, r.passed), ("over_citation", False))

    def test_zh_true_chinese_claim_relevance_is_incremental(self):
        # v4.3：中文同样按增量覆盖判。zh_market2 与 claim 同题（共享"菜市场/苹果"），
        # 但内容词已被主引用（zh_market）全覆盖 → 冗余引用（判失败）。
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content, supporting=sup(self.zh_market2))
        (r,) = check_citation_set(self.ledger, claim, zh=True, policy=CITE_ZH)
        self.assertEqual((r.name, r.passed), ("over_citation", False))

    def test_zh_true_supporting_with_new_evidence_passes(self):
        # 中文合法多引用：claim 含 zh_market2 独有的内容（便宜/菜市场评价）
        claim = Claim(
            Citation(self.zh_market.id, 0, self.zh_market.content),
            text="我昨天和我妈去菜市场买了三斤苹果，那家菜市场的苹果很便宜。",
            supporting=sup(self.zh_market2),
        )
        (r,) = check_citation_set(self.ledger, claim, zh=True, policy=CITE_ZH)
        self.assertEqual((r.name, r.passed), ("over_citation", True))

    def test_empty_quote_is_trivially_verbatim(self):
        # 空串在子串语义下平凡通过（documented）；相关性按增量覆盖判 → car2 无增量 → 失败。
        res = check_citation_set(
            self.ledger, self.car_claim((Citation(self.car2.id, 0, ""),)),
            zh=False, policy=CITE)
        (r,) = res
        self.assertEqual((r.name, r.passed), ("over_citation", False))

    def test_predicate_names_are_declared(self):
        self.assertEqual(OVER_CITATION_PREDICATES,
                         ("over_citation_anchor", "over_citation_verbatim", "over_citation"))

    def test_repeated_calls_are_identical(self):
        claim = self.car_claim((Citation(self.car2.id, 0, self.car2.content),
                                Citation(self.weather.id, 0, self.weather.content)))
        first = check_citation_set(self.ledger, claim, zh=False, policy=CITE)
        for _ in range(10):
            self.assertEqual(check_citation_set(self.ledger, claim, zh=False, policy=CITE), first)


# --------------------------------------------------------------------------- #
# 2. 引用集校验的 enforce() 接入（契约 §6 第 4 级）
# --------------------------------------------------------------------------- #

class CiteSetEnforceTest(_Fixture):
    def test_irrelevant_supporting_degrades(self):
        r = enforce(self.ledger, self.car_claim(sup(self.weather)), policy=CITE)
        self.assertEqual(r.verdict, "degraded")
        self.assertIsNone(r.rejected_cause)
        self.assertIn("over_citation", soft_failed(r))
        self.assertNotIn("over_citation", hard_failed(r))

    def test_redundant_supporting_degrades(self):
        # v4.3：car2 对"car 记录已完全覆盖的 claim"是冗余引用 → degraded（不是 verified）
        r = enforce(self.ledger, self.car_claim(sup(self.car2)), policy=CITE)
        self.assertEqual(r.verdict, "degraded")
        self.assertIn("over_citation", soft_failed(r))
        self.assertNotIn("over_citation", hard_failed(r))

    def test_supporting_with_new_evidence_stays_verified(self):
        claim = Claim(
            Citation(self.car.id, 0, self.car.content),
            text="Alice bought a red car on March 5th and drove it to the market yesterday.",
            supporting=(Citation(self.car2.id, 0, self.car2.content),),
        )
        r = enforce(self.ledger, claim, policy=CITE)
        self.assertEqual(r.verdict, "verified")
        self.assertTrue(all(p.passed for p in r.failed if p.name == "over_citation"))

    def test_noncontiguous_supporting_quote_rejects_with_citation_set(self):
        r = enforce(self.ledger,
                    self.car_claim((Citation(self.car2.id, 0, "Alice bought a blue truck."),)),
                    policy=CITE)
        self.assertEqual(r.verdict, "rejected")
        self.assertEqual(r.rejected_cause, "citation_set")
        self.assertIn("over_citation_verbatim", hard_failed(r))

    def test_missing_supporting_anchor_without_text_rejects_with_citation_set(self):
        claim = Claim(Citation(self.car.id, 0, self.car.content), text=None,
                      supporting=(Citation("a_999999", 0, "x"),))
        r = enforce(self.ledger, claim, policy=CITE)
        self.assertEqual(r.verdict, "rejected")
        self.assertEqual(r.rejected_cause, "citation_set")
        self.assertIn("over_citation_anchor", hard_failed(r))

    def test_missing_supporting_anchor_with_text_hits_tier1_precheck_first(self):
        # 契约 §6 的顺序：Tier 1（骨架）在前，其 supporting anchor 预检先命中 → invalid_anchor。
        # 这条断言锁住"顺序"这个契约事实，而不是把两种 cause 混为一谈。
        claim = Claim(Citation(self.car.id, 0, self.car.content), text=self.car.content,
                      supporting=(Citation("a_999999", 0, "x"),))
        r = enforce(self.ledger, claim, policy=CITE)
        self.assertEqual(r.verdict, "rejected")
        self.assertEqual(r.rejected_cause, "invalid_anchor")
        self.assertIn("anchor", hard_failed(r))

    def test_no_supporting_keeps_verified(self):
        r = enforce(self.ledger, self.car_claim(), policy=CITE)
        self.assertEqual(r.verdict, "verified")
        self.assertEqual([p.name for p in r.failed if p.name.startswith("over_citation")], [])

    def test_default_policy_ignores_supporting(self):
        # 默认策略（cite_set=False）下支持引用完全不参与判定 = 改动前行为
        r = enforce(self.ledger, self.car_claim(sup(self.weather)))
        self.assertEqual(r.verdict, "verified")
        self.assertEqual(soft_failed(r), [])

    def test_coverage_full_false_skips_citation_set(self):
        # coverage_full=False 的语义是"只跑无条件全量项"（v4 之前就是这样跳过 Tier 1 的）
        r = enforce(self.ledger, self.car_claim(sup(self.weather)), policy=CITE,
                    coverage_full=False)
        self.assertEqual(r.verdict, "verified")
        self.assertEqual([p.name for p in r.failed if p.name.startswith("over_citation")], [])

    def test_cite_set_hard_is_reachable_through_v4_policy(self):
        r = enforce(self.ledger,
                    self.car_claim((Citation(self.car2.id, 0, "Alice bought a blue truck."),)),
                    policy=V4)
        self.assertEqual(r.rejected_cause, "citation_set")

    def test_repeated_enforce_is_identical(self):
        claim = self.car_claim(sup(self.weather))
        first = enforce(self.ledger, claim, policy=CITE)
        for _ in range(5):
            again = enforce(self.ledger, claim, policy=CITE)
            self.assertEqual(again.verdict, first.verdict)
            self.assertEqual(again.rejected_cause, first.rejected_cause)
            self.assertEqual(failed_pairs(again), failed_pairs(first))


# --------------------------------------------------------------------------- #
# 3. 场景谓词（模块级，直接调 check_scenario）
# --------------------------------------------------------------------------- #

class ScenarioUnitTest(_Fixture):
    def _run(self, claim, ctx, policy=SCEN):
        return {p.name: p for p in check_scenario(self.ledger, claim, ctx, policy=policy)}

    # -- 门控 --------------------------------------------------------------- #
    def test_context_none_executes_nothing(self):
        self.assertEqual(check_scenario(self.ledger, self.excerpt(self.zh_work), None,
                                        policy=SCEN), [])

    def test_policy_scenario_false_executes_nothing(self):
        self.assertEqual(check_scenario(self.ledger, self.excerpt(self.zh_work),
                                        ScenarioContext(), policy=SkeletonPolicy()), [])

    def test_five_predicates_returned_in_fixed_order(self):
        # 契约 v1.1（Plan Z C2）：固定返回 5 条，顺序 = SCENARIO_PREDICATES
        res = check_scenario(self.ledger, self.excerpt(self.zh_work), ScenarioContext(),
                             policy=SCEN)
        self.assertEqual(tuple(p.name for p in res), SCENARIO_PREDICATES)
        self.assertEqual(SCENARIO_PREDICATES,
                         ("scenario_timeline", "scenario_cross_session",
                          "scenario_speaker_scope", "scenario_superseded",
                          "scenario_evidence_incomplete"))
        self.assertEqual([p.severity for p in res], ["hard", "soft", "hard", "hard", "hard"])

    # -- 1) scenario_timeline（hard） --------------------------------------- #
    def test_timeline_flags_later_evidence_for_earlier_assertion(self):
        res = self._run(self.excerpt(self.zh_work),
                        ScenarioContext(context_time=1000.0,
                                        record_times={self.zh_work.id: 2000.0}))
        p = res["scenario_timeline"]
        self.assertEqual((p.passed, p.severity), (False, "hard"))
        self.assertIn(self.zh_work.id, p.detail)

    def test_timeline_passes_when_evidence_is_earlier(self):
        p = self._run(self.excerpt(self.zh_work),
                      ScenarioContext(context_time=1000.0,
                                      record_times={self.zh_work.id: 500.0}))["scenario_timeline"]
        self.assertTrue(p.passed)

    def test_timeline_boundary_is_not_a_violation(self):
        # 严格 >：记录时点 == 断言时点不算"后事作前证"
        p = self._run(self.excerpt(self.zh_work),
                      ScenarioContext(context_time=1000.0,
                                      record_times={self.zh_work.id: 1000.0}))["scenario_timeline"]
        self.assertTrue(p.passed)

    def test_timeline_requires_context_time(self):
        p = self._run(self.excerpt(self.zh_work),
                      ScenarioContext(record_times={self.zh_work.id: 2000.0}))["scenario_timeline"]
        self.assertTrue(p.passed)
        self.assertIn("no context_time", p.detail)

    def test_timeline_skips_anchor_without_record_time(self):
        p = self._run(self.excerpt(self.zh_work),
                      ScenarioContext(context_time=1.0, record_times={}))["scenario_timeline"]
        self.assertTrue(p.passed)

    def test_timeline_checks_supporting_records_too(self):
        claim = self.car_claim(sup(self.car2))
        p = self._run(claim, ScenarioContext(context_time=5.0,
                                             record_times={self.car.id: 1.0,
                                                           self.car2.id: 9.0}))["scenario_timeline"]
        self.assertFalse(p.passed)
        self.assertIn(self.car2.id, p.detail)

    # -- 2) scenario_cross_session（soft） ---------------------------------- #
    def test_cross_session_flags_anaphor_across_sessions(self):
        claim = Claim(Citation(self.zh_work.id, 0, self.zh_work.content),
                      text="上次说好的方案呢？")
        p = self._run(claim, ScenarioContext(context_session_id="other"))["scenario_cross_session"]
        self.assertEqual((p.passed, p.severity), (False, "soft"))

    def test_cross_session_passes_within_the_same_session(self):
        claim = Claim(Citation(self.zh_work.id, 0, self.zh_work.content),
                      text="上次说好的方案呢？")
        p = self._run(claim, ScenarioContext(context_session_id="s3"))["scenario_cross_session"]
        self.assertTrue(p.passed)

    def test_cross_session_requires_context_session_id(self):
        claim = Claim(Citation(self.zh_work.id, 0, self.zh_work.content),
                      text="上次说好的方案呢？")
        p = self._run(claim, ScenarioContext())["scenario_cross_session"]
        self.assertTrue(p.passed)
        self.assertIn("no context_session_id", p.detail)

    def test_cross_session_requires_anaphor_in_claim_text(self):
        claim = Claim(Citation(self.zh_work.id, 0, self.zh_work.content), text="结算系统上线了。")
        p = self._run(claim, ScenarioContext(context_session_id="other"))["scenario_cross_session"]
        self.assertTrue(p.passed)
        self.assertIn("no anaphor", p.detail)

    # -- 3) scenario_speaker_scope（hard） ---------------------------------- #
    def test_speaker_scope_flags_assistant_hedge_as_user_fact(self):
        p = self._run(self.excerpt(self.zh_hedge),
                      ScenarioContext(asserted_scope="user_fact"))["scenario_speaker_scope"]
        self.assertEqual((p.passed, p.severity), (False, "hard"))
        self.assertIn(self.zh_hedge.id, p.detail)

    def test_speaker_scope_passes_when_scope_unspecified(self):
        p = self._run(self.excerpt(self.zh_hedge),
                      ScenarioContext())["scenario_speaker_scope"]
        self.assertTrue(p.passed)

    def test_speaker_scope_passes_for_user_records(self):
        p = self._run(self.excerpt(self.zh_work),
                      ScenarioContext(asserted_scope="user_fact"))["scenario_speaker_scope"]
        self.assertTrue(p.passed)

    def test_speaker_scope_passes_for_assistant_without_hedge(self):
        p = self._run(self.excerpt(self.zh_plain),
                      ScenarioContext(asserted_scope="user_fact"))["scenario_speaker_scope"]
        self.assertTrue(p.passed)

    def test_speaker_scope_passes_when_scope_is_assistant_claim(self):
        p = self._run(self.excerpt(self.zh_hedge),
                      ScenarioContext(asserted_scope="assistant_claim"))["scenario_speaker_scope"]
        self.assertTrue(p.passed)

    # -- 4) scenario_superseded（v4.6：soft → degraded） --------------------- #
    def test_superseded_flags_later_user_correction(self):
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content)
        p = self._run(claim, ScenarioContext(), policy=SCEN_ZH)["scenario_superseded"]
        self.assertEqual((p.passed, p.severity), (False, "soft"))
        self.assertIn(self.zh_fix.id, p.detail)

    def test_superseded_passes_without_later_correction(self):
        p = self._run(self.excerpt(self.zh_work), ScenarioContext(),
                      policy=SCEN_ZH)["scenario_superseded"]
        self.assertTrue(p.passed)

    def test_superseded_passes_for_unrelated_claim_text(self):
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text="今天的天气不错")
        p = self._run(claim, ScenarioContext(), policy=SCEN_ZH)["scenario_superseded"]
        self.assertTrue(p.passed)
        self.assertEqual(p.detail, "")  # 有 token 但与更正记录无交集 → 不触发

    def test_superseded_requires_zh_tokens(self):
        # 已文档化的单侧性：zh=False 时中文断言没有内容 token，交集无法计算 → 不可判 → passed。
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content)
        p = self._run(claim, ScenarioContext(), policy=SCEN)["scenario_superseded"]
        self.assertTrue(p.passed)
        self.assertIn("no assessable content token", p.detail)

    def test_superseded_ignores_assistant_corrections(self):
        # 更正只认 speaker == "user"；assistant 记录里的"不对"不构成用户更正
        claim = Claim(Citation(self.zh_plain.id, 0, self.zh_plain.content),
                      text=self.zh_plain.content)
        p = self._run(claim, ScenarioContext(), policy=SCEN_ZH)["scenario_superseded"]
        self.assertTrue(p.passed)

    # -- 其它 --------------------------------------------------------------- #
    def test_non_numeric_record_time_raises(self):
        # 显式失败，不做静默降级
        with self.assertRaises(TypeError):
            check_scenario(self.ledger, self.excerpt(self.car),
                           ScenarioContext(context_time=1.0,
                                           record_times={self.car.id: "not-a-number"}),
                           policy=SCEN)

    def test_repeated_calls_are_identical(self):
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content)
        ctx = ScenarioContext(context_time=1.0, context_session_id="s1",
                              asserted_scope="user_fact",
                              record_times={self.zh_market.id: 2.0})
        first = check_scenario(self.ledger, claim, ctx, policy=V4)
        for _ in range(10):
            self.assertEqual(check_scenario(self.ledger, claim, ctx, policy=V4), first)


# --------------------------------------------------------------------------- #
# 4. 场景谓词的 enforce() 接入（契约 §6 第 5 级）
# --------------------------------------------------------------------------- #

class ScenarioEnforceTest(_Fixture):
    def test_timeline_hard_rejects_with_scenario_cause(self):
        r = enforce(self.ledger, self.excerpt(self.zh_work), policy=SCEN,
                    context=ScenarioContext(context_time=1000.0,
                                            record_times={self.zh_work.id: 2000.0}))
        self.assertEqual(r.verdict, "rejected")
        self.assertEqual(r.rejected_cause, "scenario")
        self.assertIn("scenario_timeline", hard_failed(r))

    def test_timeline_pass_keeps_verified(self):
        r = enforce(self.ledger, self.excerpt(self.zh_work), policy=SCEN,
                    context=ScenarioContext(context_time=1000.0,
                                            record_times={self.zh_work.id: 5.0}))
        self.assertEqual(r.verdict, "verified")

    def test_cross_session_soft_degrades(self):
        claim = Claim(Citation(self.zh_work.id, 0, self.zh_work.content),
                      text="上次说好的方案呢？")
        r = enforce(self.ledger, claim, policy=SCEN,
                    context=ScenarioContext(context_session_id="other"))
        self.assertEqual(r.verdict, "degraded")
        self.assertIsNone(r.rejected_cause)
        self.assertIn("scenario_cross_session", soft_failed(r))

    def test_speaker_scope_hard_rejects_with_scenario_cause(self):
        r = enforce(self.ledger, self.excerpt(self.zh_hedge), policy=SCEN,
                    context=ScenarioContext(asserted_scope="user_fact"))
        self.assertEqual(r.verdict, "rejected")
        self.assertEqual(r.rejected_cause, "scenario")
        self.assertIn("scenario_speaker_scope", hard_failed(r))

    def test_superseded_soft_degrades_with_scenario_reason(self):
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content)
        r = enforce(self.ledger, claim, policy=SCEN_ZH, context=ScenarioContext())
        self.assertEqual(r.verdict, "degraded")
        self.assertIsNone(r.rejected_cause)
        self.assertIn("scenario_superseded", soft_failed(r))
        self.assertEqual(r.degrade_reason, "scenario_superseded")

    def test_superseded_is_reachable_through_v4_policy(self):
        claim = Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                      text=self.zh_market.content)
        r = enforce(self.ledger, claim, policy=V4, context=ScenarioContext())
        self.assertEqual((r.verdict, r.rejected_cause), ("degraded", None))
        self.assertEqual(r.degrade_reason, "scenario_superseded")

    def test_context_none_is_identical_to_omitted(self):
        claim = self.excerpt(self.zh_work)
        a = enforce(self.ledger, claim, policy=V4)
        b = enforce(self.ledger, claim, policy=V4, context=None)
        self.assertEqual(a.verdict, b.verdict)
        self.assertEqual(a.rejected_cause, b.rejected_cause)
        self.assertEqual(failed_pairs(a), failed_pairs(b))

    def test_citation_set_hard_precedes_scenario_hard(self):
        # 契约 §6 顺序：第 4 级（引用集）先于第 5 级（场景）
        claim = self.car_claim((Citation(self.car2.id, 0, "Alice bought a blue truck."),))
        r = enforce(self.ledger, claim, policy=SkeletonPolicy(cite_set=True, scenario=True),
                    context=ScenarioContext(context_time=1.0,
                                            record_times={self.car.id: 9.0}))
        self.assertEqual(r.rejected_cause, "citation_set")

    def test_repeated_enforce_is_identical(self):
        claim = self.excerpt(self.zh_work)
        ctx = ScenarioContext(context_time=1000.0, context_session_id="s3",
                              asserted_scope="user_fact",
                              record_times={self.zh_work.id: 5.0})
        first = enforce(self.ledger, claim, policy=V4, context=ctx)
        for _ in range(5):
            again = enforce(self.ledger, claim, policy=V4, context=ctx)
            self.assertEqual(failed_pairs(again), failed_pairs(first))
            self.assertEqual(again.verdict, first.verdict)


# --------------------------------------------------------------------------- #
# 5. 默认行为不变 / 契约面
# --------------------------------------------------------------------------- #

# 冻结表：**改动前**用同一批 ledger 输入跑 v2 默认策略得到的判定（任务书要求的显式断言）。
# 只要 `enforce()` 的默认路径有任何变化，这张表就会失败。
FROZEN_V2_DEFAULT = {
    "verified": ("verified", None,
                 (("verbatim", True), ("preserve_negation", True), ("preserve_temporal", True))),
    "degraded": ("degraded", None,
                 (("verbatim", True), ("preserve_negation", True),
                  ("preserve_temporal", True), ("skeleton_grounding", False))),
    "rejected_verbatim": ("rejected", "verbatim", (("verbatim", False),)),
    "rejected_speaker": ("rejected", "preserve_speaker",
                         (("verbatim", True), ("preserve_negation", True),
                          ("preserve_temporal", True), ("preserve_speaker", False))),
    "rejected_anchor": ("rejected", "search_empty", (("anchor", False),)),
    "zh_claim_default": ("verified", None,
                         (("verbatim", True), ("preserve_negation", True),
                          ("preserve_temporal", True))),
    "overcite_irrelevant": ("verified", None,
                            (("verbatim", True), ("preserve_negation", True),
                             ("preserve_temporal", True))),
}


class BackCompatTest(_Fixture):
    def _frozen_claims(self):
        return {
            "verified": Claim(Citation(self.aero.id, 0, self.aero.content), text=self.aero.content),
            "degraded": Claim(Citation(self.bakery.id, 0, self.bakery.content),
                              text="Alice bought a red car on March 5th."),
            "rejected_verbatim": Claim(Citation(self.aero.id, 0, "I saw Aerosmith in concert."),
                                       text="I saw Aerosmith in concert."),
            "rejected_speaker": Claim(Citation(self.aero.id, 0, self.aero.content),
                                      text=self.aero.content, asserted_speaker="Jon"),
            "rejected_anchor": Claim(Citation("a_999999", 0, "x"), text="x"),
            "zh_claim_default": Claim(Citation(self.zh_market.id, 0, self.zh_market.content),
                                      text=self.zh_market.content),
            "overcite_irrelevant": Claim(Citation(self.aero.id, 0, self.aero.content),
                                         text=self.aero.content,
                                         supporting=(Citation(self.car.id, 0, self.car.content),)),
        }

    def test_default_policy_frozen_verdicts(self):
        self.assertEqual(set(self._frozen_claims()), set(FROZEN_V2_DEFAULT))
        for name, claim in self._frozen_claims().items():
            with self.subTest(case=name):
                r = enforce(self.ledger, claim)
                want_verdict, want_cause, want_failed = FROZEN_V2_DEFAULT[name]
                self.assertEqual(r.verdict, want_verdict)
                self.assertEqual(r.rejected_cause, want_cause)
                self.assertEqual(tuple(failed_pairs(r)), want_failed)

    def test_default_policy_equals_explicit_v2_fields(self):
        explicit = SkeletonPolicy(version=POLICY_VERSION, arg_slots=False, dispatch=False,
                                  zh=False, cite_set=False, scenario=False)
        for name, claim in self._frozen_claims().items():
            with self.subTest(case=name):
                a = enforce(self.ledger, claim)
                b = enforce(self.ledger, claim, policy=explicit)
                self.assertEqual(a.verdict, b.verdict)
                self.assertEqual(a.rejected_cause, b.rejected_cause)
                self.assertEqual(failed_pairs(a), failed_pairs(b))
                self.assertEqual(a.hard_failed, b.hard_failed)

    def test_default_policy_ignores_context(self):
        ctx = ScenarioContext(context_session_id="nope", context_time=1.0,
                              asserted_scope="user_fact",
                              record_times={self.car.id: 9.0})
        for name, claim in self._frozen_claims().items():
            with self.subTest(case=name):
                a = enforce(self.ledger, claim)
                b = enforce(self.ledger, claim, context=ctx)
                self.assertEqual(a.verdict, b.verdict)
                self.assertEqual(a.rejected_cause, b.rejected_cause)
                self.assertEqual(failed_pairs(a), failed_pairs(b))

    def test_v3c3_with_context_matches_without_context(self):
        # 契约 §8：context=None 时结果与 v3 逐字节一致
        v3 = SkeletonPolicy.v3_c3()
        ctx = ScenarioContext(context_session_id="x", context_time=1.0,
                              asserted_scope="user_fact",
                              record_times={self.zh_market.id: 9.0})
        for name, claim in self._frozen_claims().items():
            with self.subTest(case=name):
                a = enforce(self.ledger, claim, policy=v3)
                b = enforce(self.ledger, claim, policy=v3, context=ctx)
                self.assertEqual(a.verdict, b.verdict)
                self.assertEqual(a.rejected_cause, b.rejected_cause)
                self.assertEqual(failed_pairs(a), failed_pairs(b))

    def test_reject_causes_extended(self):
        self.assertEqual(REJECT_CAUSES[-2:], ("citation_set", "scenario"))
        # 既有 cause 一个都没删、顺序也没动
        self.assertEqual(
            REJECT_CAUSES,
            ("invalid_anchor", "verbatim", "preserve_negation", "preserve_speaker",
             "skeleton", "budget_exhausted", "untrusted_writer", "search_empty",
             "citation_set", "scenario"))

    def test_scenario_context_defaults_and_frozen(self):
        ctx = ScenarioContext()
        self.assertIsNone(ctx.context_session_id)
        self.assertIsNone(ctx.context_time)
        self.assertEqual(ctx.asserted_scope, "unspecified")
        self.assertEqual(ctx.record_times, ())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.context_time = 1.0  # type: ignore[misc]

    def test_v4_policy_enables_all_three_capability_switches(self):
        self.assertEqual(POLICY_VERSION_V4, "skel-v4.0")
        self.assertEqual(V4.version, POLICY_VERSION_V4)
        self.assertTrue(V4.zh and V4.cite_set and V4.scenario)
        # 默认策略三个开关全关（默认行为不变的前提）
        default = SkeletonPolicy()
        self.assertFalse(default.zh or default.cite_set or default.scenario)

    def test_enforce_accepts_context_keyword(self):
        r = enforce(self.ledger, self.excerpt(self.car),
                    context=ScenarioContext(context_time=1.0,
                                            record_times={self.car.id: 2.0}))
        self.assertEqual(r.verdict, "verified")


if __name__ == "__main__":
    unittest.main()
