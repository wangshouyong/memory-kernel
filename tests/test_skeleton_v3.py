# SPDX-License-Identifier: Apache-2.0
"""skel-v3.0（论元槽对齐 + 多引用分派）的单测。

覆盖任务书要求的五类：
  * 并列主体（both X and Y / X and Y）与第一人称复数证据的豁免；
  * 客体缺失（客体专名未接地）；
  * 部分引用分派（多事件 claim 只引一条 / 事件核被"拼接"到两条引用）；
  * v2 向后兼容（默认 `SkeletonPolicy()` 行为逐位不变，版本号语义不变）；
  * 确定性 + 纯函数（无模型、无网络、无 I/O、重复调用逐位一致）。

只测 `memory_kernel` 的公开 API；不依赖 eval/ 下的任何数据（测试必须离线可跑）。
"""

from __future__ import annotations

import ast
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import Claim, Citation, SkeletonPolicy, enforce, skeleton_check
from memory_kernel.ledger import AnchorLedger
from memory_kernel.skeleton import (
    ARGUMENT_VERSION,
    POLICY_VERSION,
    POLICY_VERSION_V3,
    UNIFORM_IDF,
    arg_lemma,
    argument_grounding,
    claim_arguments,
    spans_of,
)

KEY = hashlib.sha256(b"skeleton-v3-test").digest()


def _ground(text, records, *, arg_slots=True, dispatch=True, tau=0.5):
    """直接测论元层（绕开硬单元/软词，规则名不被遮蔽）。"""
    return argument_grounding(
        claim_arguments(text), records, idf=UNIFORM_IDF, tau=tau,
        arg_slots=arg_slots, dispatch=dispatch,
    )


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.add = self.ledger.append

        def rec(content, speaker):
            return self.add(content, source_id="s", episode_id="e", speaker=speaker)

        self.alice_car = rec("Alice bought a red car on March 5th.", "Alice")
        self.alice_car_expensive = rec("It was really expensive.", "Alice")
        self.bob_berlin = rec("Bob moved to Berlin in June.", "Bob")
        self.market_red = rec("Alice went to the market with a red car.", "Alice")
        self.truck_blue = rec("Bob bought a blue truck.", "Bob")
        self.jon_job = rec(
            "Lost my job as a banker yesterday, so I am gonna take a shot at starting my own business.",
            "Jon",
        )
        self.gina_job = rec("Unfortunately, I also lost my job at Door Dash this month.", "Gina")
        self.both_jobs = rec("Jon and Gina both lost their jobs today.", "Jon")
        self.maria_we = rec("Remember when we volunteered together last year?", "Maria")
        self.maria_i = rec("I volunteered last year.", "Maria")
        self.caroline_car = rec("I donated my old car to a homeless shelter yesterday.", "Caroline")
        self.sam_checkup = rec("Appreciate the offer, but had a check-up with my doctor.", "Sam")


# --------------------------------------------------------------------------- #
# 1. claim → 论元
# --------------------------------------------------------------------------- #

class ArgumentExtractionTest(_Fixture):
    def test_coordinated_subject_yields_two_participants(self):
        units = claim_arguments("Both Jon and Gina recently lost their jobs.")
        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertEqual(tuple(u.subject_names), ("Jon", "Gina"))
        self.assertTrue(u.coordinated)
        self.assertEqual(u.main_predicate, arg_lemma("lost"))
        self.assertIn("jobs", [t.lower() for t in u.object_tokens])

    def test_coordinated_np_is_not_split_into_two_units(self):
        # "Andrew and his girlfriend …" 是并列主语，不是两个子句
        units = claim_arguments("Andrew and his girlfriend drove to the coast.")
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].coordinated)
        self.assertEqual(units[0].main_predicate, arg_lemma("drove"))

    def test_composite_claim_yields_one_unit_per_clause(self):
        units = claim_arguments("Alice bought a red car on March 5th. Bob moved to Berlin in June.")
        self.assertEqual(len(units), 2)
        self.assertEqual(tuple(units[0].subject_names), ("Alice",))
        self.assertEqual(tuple(units[1].subject_names), ("Bob",))
        self.assertEqual(units[1].main_predicate, arg_lemma("moved"))

    def test_object_slot_is_extracted(self):
        (u,) = claim_arguments("Caroline donated her car to David.")
        self.assertEqual(u.main_predicate, arg_lemma("donated"))
        self.assertEqual(tuple(u.object_names), ("David",))
        self.assertIn("car", [t.lower() for t in u.object_tokens])

    def test_attribution_verb_is_not_the_predicate(self):
        (u,) = claim_arguments("Jon said he lost his job.")
        self.assertEqual(u.main_predicate, arg_lemma("lost"))
        self.assertIn(arg_lemma("said"), u.predicates)

    def test_hyphenated_compound_is_a_noun_not_a_predicate(self):
        (u,) = claim_arguments("Sam had a check-up with his doctor.")
        self.assertIsNone(u.main_predicate, "check-up 是名词，不能被当成谓词 check")

    def test_time_argument_reuses_spans_of(self):
        (u,) = claim_arguments("Alice bought a red car on March 5th.")
        self.assertTrue(u.time_spans, "时间论元必须由 spans_of 抽出")
        self.assertEqual(tuple(u.time_spans), tuple(spans_of(u.segment)))

    def test_empty_claim_has_no_units(self):
        self.assertEqual(claim_arguments(""), ())
        self.assertEqual(claim_arguments("   "), ())


# --------------------------------------------------------------------------- #
# 2. 论元槽接地 + 分派
# --------------------------------------------------------------------------- #

class ArgumentGroundingTest(_Fixture):
    def test_participant_missing_from_citation_is_arg_subject(self):
        rep = _ground("Both Jon and Gina recently lost their jobs.", [self.jon_job])
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "arg_subject")
        self.assertIn("Gina", rep.detail)
        self.assertEqual(rep.participants, ("Jon", "Gina"))

    def test_both_citations_ground_both_participants(self):
        rep = _ground("Both Jon and Gina recently lost their jobs.", [self.jon_job, self.gina_job])
        self.assertTrue(rep.ok, rep.detail)

    def test_first_person_plural_evidence_licenses_coordinated_subject(self):
        # "we volunteered together" 为并列主体背书（Maria 是说话人，John 由 "我们" 覆盖）
        rep = _ground("Maria and John volunteered together last year", [self.maria_we])
        self.assertTrue(rep.ok, rep.detail)
        # 对照：证据是第一人称**单数**时不给豁免
        rep2 = _ground("Maria and John volunteered together last year", [self.maria_i])
        self.assertFalse(rep2.ok)
        self.assertEqual(rep2.rule, "arg_subject")

    def test_missing_object_is_arg_object(self):
        rep = _ground("Caroline donated her car to David.", [self.caroline_car])
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "arg_object")
        self.assertIn("David", rep.detail)
        self.assertIn("object:david", [m.lower() for m in rep.missing])

    def test_object_present_passes(self):
        rep = _ground("Caroline donated her old car to a homeless shelter.", [self.caroline_car])
        self.assertTrue(rep.ok, rep.detail)

    def test_predicate_missing_is_arg_predicate(self):
        rep = _ground("Bob drove to Berlin in June.", [self.bob_berlin])
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "arg_predicate")

    def test_second_event_needs_its_own_evidence_coverage(self):
        # 组合断言第二句在引用里没有覆盖 → 单条引用不够（v2 的总体覆盖率会把它平均掉）
        claim = "Alice bought a red car on March 5th. It was really expensive."
        rep = _ground(claim, [self.alice_car])
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "arg_unit_coverage")
        self.assertEqual(rep.n_units, 2)
        self.assertLess(rep.unit_coverage_min, 0.5)
        # 两条引用各覆盖一部分 → 通过（任务书要求的"分配"语义）
        rep2 = _ground(claim, [self.alice_car, self.alice_car_expensive])
        self.assertTrue(rep2.ok, rep2.detail)

    def test_dispatch_event_rejects_stapled_citations(self):
        # 谓词只在"Bob bought a blue truck"里、客体只在"Alice went to the market with a red car."里
        rep = _ground("Alice bought a red car.", [self.market_red, self.truck_blue])
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "dispatch_event")
        self.assertTrue(rep.dispatch_unassigned)
        # 单条同时含谓词与客体的引用 → 通过
        rep2 = _ground("Alice went to the market with a red car.", [self.market_red])
        self.assertTrue(rep2.ok, rep2.detail)

    def test_dispatch_plural_requires_a_second_citation(self):
        # 两个参与者都能在**同一条**引用文本里找到，但只有一条引用 → 并列断言缺第二条证据
        rep = _ground("Both Jon and Gina recently lost their jobs.", [self.both_jobs])
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "dispatch_plural")

    def test_arg_slots_and_dispatch_switches_are_independent(self):
        claim = "Both Jon and Gina recently lost their jobs."
        # 只开 dispatch：主体槽不查，但并列断言仍然需要第二条引用 → dispatch_plural
        rep = _ground(claim, [self.jon_job], arg_slots=False)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.rule, "dispatch_plural")
        # 只开 arg_slots：主体槽先报（Gina 不在引用里）
        self.assertEqual(_ground(claim, [self.jon_job], dispatch=False).rule, "arg_subject")

    def test_no_units_is_vacuously_ok(self):
        rep = _ground("", [self.alice_car])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.n_units, 0)


# --------------------------------------------------------------------------- #
# 3. 集成：skeleton_check / enforce
# --------------------------------------------------------------------------- #

class V3IntegrationTest(_Fixture):
    def test_v3_rejects_claim_whose_participant_is_uncovered(self):
        r = skeleton_check("Both Jon and Gina recently lost their jobs.", [self.jon_job],
                           SkeletonPolicy.v3(tau_soft=0.5))
        self.assertFalse(r.ok)
        self.assertIn(r.rule, {"hard_unit", "arg_subject"})
        self.assertFalse(r.arg_ok)
        self.assertEqual(r.n_units, 1)
        self.assertEqual(r.participants, ("Jon", "Gina"))
        self.assertEqual(r.version, POLICY_VERSION_V3)

    def test_v3_accepts_the_same_claim_with_two_citations(self):
        r = skeleton_check("Both Jon and Gina recently lost their jobs.",
                           [self.jon_job, self.gina_job], SkeletonPolicy.v3(tau_soft=0.5))
        self.assertTrue(r.ok, r.detail)
        self.assertTrue(r.arg_ok)

    def test_v3_only_rule_changes_the_verdict_where_v2_accepted(self):
        claim = "Alice bought a red car."
        v2 = skeleton_check(claim, [self.market_red, self.truck_blue], SkeletonPolicy(tau_soft=0.5))
        v3 = skeleton_check(claim, [self.market_red, self.truck_blue],
                            SkeletonPolicy.v3(tau_soft=0.5))
        self.assertTrue(v2.ok, "v2 的词袋度量看不见'谓词在 A、客体在 B'的拼接")
        self.assertFalse(v3.ok)
        self.assertEqual(v3.rule, "dispatch_event")

    def test_v3_tau_zero_disables_argument_rules(self):
        # τ=0 的语义（关闭整个接地检查）在 v3 下不变
        r = skeleton_check("Both Jon and Gina lost their jobs.", [self.jon_job],
                           SkeletonPolicy.v3(tau_soft=0.0))
        self.assertTrue(r.ok, r.detail)
        self.assertEqual(r.n_units, 0)

    def test_enforce_uses_v3_policy_and_stays_soft(self):
        claim = Claim(
            Citation(self.jon_job.id, 0, self.jon_job.content),
            text="Both Jon and Gina recently lost their jobs.",
        )
        res = enforce(self.ledger, claim, policy=SkeletonPolicy.v3(tau_soft=0.5))
        self.assertEqual(res.verdict, "degraded")       # Tier 1 仍然只降级，不硬拒
        self.assertIn("skeleton_grounding", [p.name for p in res.failed if not p.passed])

    def test_features_expose_argument_diagnostics(self):
        r = skeleton_check("Both Jon and Gina recently lost their jobs.", [self.jon_job],
                           SkeletonPolicy.v3(tau_soft=0.5))
        f = r.features()
        for key in ("arg_ok", "n_units", "unit_cov_min", "n_participants", "dispatch_unassigned",
                    "plural_uncovered"):
            self.assertIn(key, f)
        self.assertFalse(f["arg_ok"])
        self.assertEqual(f["n_participants"], 2)

    def test_dispatch_diagnostics_survive_earlier_slot_failures(self):
        """先报 arg_* 时，dispatch 的诊断字段仍必须是完整的（评测层依赖它做网格推导）。"""
        # 并列主体 + 谓词未接地：第一个违反是 arg_subject，但 dispatch 也已经判失败
        rep = _ground("Both Jon and Gina painted the fence.", [self.jon_job])
        self.assertEqual(rep.rule, "arg_subject")
        self.assertTrue(rep.dispatch_unassigned, "arg_* 先报时 dispatch_unassigned 不能丢")
        self.assertTrue(rep.plural_uncovered)
        r = skeleton_check("Both Jon and Gina painted the fence.", [self.jon_job],
                           SkeletonPolicy.v3(tau_soft=0.5, hard_units=False))
        self.assertEqual(r.rule, "arg_subject")
        self.assertTrue(r.dispatch_unassigned)
        self.assertTrue(r.plural_uncovered)


# --------------------------------------------------------------------------- #
# 4. v2 向后兼容
# --------------------------------------------------------------------------- #

class BackCompatTest(_Fixture):
    def test_default_policy_is_v2(self):
        pol = SkeletonPolicy()
        self.assertEqual(POLICY_VERSION, "skel-v2.0")
        self.assertEqual(pol.version, POLICY_VERSION)
        self.assertFalse(pol.arg_slots)
        self.assertFalse(pol.dispatch)
        self.assertFalse(pol.v3_enabled)
        self.assertEqual(
            pol.describe(),
            "skel-v2.0(tau_soft=0.80, hard=1, idf=1, stem=1, stop=v2, temporal=1, session=1)",
        )

    def test_v3_version_string_and_factory(self):
        pol = SkeletonPolicy.v3()
        self.assertEqual(POLICY_VERSION_V3, "skel-v3.0")
        self.assertEqual(pol.version, POLICY_VERSION_V3)
        self.assertTrue(pol.arg_slots and pol.dispatch and pol.v3_enabled)
        self.assertTrue(pol.describe().endswith("arg=1, dispatch=1)"))
        self.assertEqual(ARGUMENT_VERSION, "arg-s3")

    def test_default_policy_ignores_all_v3_code_paths(self):
        cases = [
            ("Both Jon and Gina recently lost their jobs.", [self.jon_job]),
            ("Alice bought a red car on March 5th. It was really expensive.", [self.alice_car]),
            ("Caroline donated her car to David.", [self.caroline_car]),
            ("Alice bought a red car.", [self.market_red, self.truck_blue]),
            ("Bob moved to Berlin in June.", [self.bob_berlin]),
        ]
        explicit_v2 = SkeletonPolicy(version=POLICY_VERSION, arg_slots=False, dispatch=False)
        for text, recs in cases:
            a = skeleton_check(text, recs)
            b = skeleton_check(text, recs, explicit_v2)
            self.assertEqual(a, b)
            self.assertEqual(a.n_units, 0)
            self.assertTrue(a.arg_ok)
            self.assertEqual(a.unit_cov_min, 1.0)

    def test_default_policy_result_unchanged_at_other_taus(self):
        text = "Alice bought a red car on March 5th."
        for tau in (0.0, 0.4, 0.5, 0.8, 0.95):
            a = skeleton_check(text, [self.alice_car], SkeletonPolicy(tau_soft=tau))
            b = skeleton_check(text, [self.alice_car],
                               SkeletonPolicy(tau_soft=tau, arg_slots=False, dispatch=False))
            self.assertEqual(a, b, f"tau={tau}")

    def test_enforce_default_verdict_unchanged(self):
        claim = Claim(Citation(self.bob_berlin.id, 0, self.bob_berlin.content),
                      text="Alice bought a red car.")
        a = enforce(self.ledger, claim)
        b = enforce(self.ledger, claim, policy=SkeletonPolicy(arg_slots=False, dispatch=False))
        self.assertEqual(a.verdict, b.verdict)
        self.assertEqual(a.rejected_cause, b.rejected_cause)
        self.assertEqual([p.name for p in a.failed], [p.name for p in b.failed])


# --------------------------------------------------------------------------- #
# 5. 确定性 + 纯函数
# --------------------------------------------------------------------------- #

class DeterminismPurityTest(_Fixture):
    def test_repeated_calls_are_bit_identical(self):
        pol = SkeletonPolicy.v3(tau_soft=0.5)
        text = "Alice bought a red car on March 5th. It was really expensive."
        first = skeleton_check(text, [self.alice_car], pol)
        for _ in range(25):
            self.assertEqual(skeleton_check(text, [self.alice_car], pol), first)

    def test_units_are_fresh_and_grounding_is_pure(self):
        text = "Both Jon and Gina recently lost their jobs."
        u1 = claim_arguments(text)
        u2 = claim_arguments(text)
        self.assertEqual(u1, u2)
        rep1 = _ground(text, [self.jon_job])
        rep2 = _ground(text, [self.jon_job])
        self.assertEqual(rep1, rep2)
        self.assertEqual(claim_arguments(text), u1, "grounding 不得改动 units")

    def test_record_order_does_not_change_argument_verdict(self):
        text = "Alice bought a red car."
        a = _ground(text, [self.market_red, self.truck_blue])
        b = _ground(text, [self.truck_blue, self.market_red])
        self.assertEqual(a.ok, b.ok)
        self.assertEqual(a.rule, b.rule)

    def test_enforcement_modules_import_only_stdlib_or_kernel(self):
        allowed = {
            "__future__", "math", "re", "weakref", "dataclasses", "datetime", "typing",
            "time",
            # 包内相对导入（v4 能力层同样是包内模块，不引入外部依赖）
            "ledger", "obligations", "skeleton",
            "zh", "cite_set", "scenario",
            # Plan Z C4：enforce 需要 `EvidenceUnavailable`（纯异常类型，无 I/O/时钟/网络）
            "scenario_context",
            # internal-task：跨请求缓存收敛到显式 EngineCache（纯 stdlib + 包内，无外部依赖）
            "engine_cache",
        }
        # v4 新增的包内模块也在扫描范围内：白名单只放宽"包内允许"，
        # 外部依赖仍然被禁止（见 test_no_model_or_network_dependencies_in_source）。
        for name in ("skeleton.py", "enforce.py", "zh.py", "cite_set.py", "scenario.py"):
            path = ROOT / "memory_kernel" / name
            if not path.exists():      # 模块尚未落地时跳过；一旦存在就必须过白名单
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            mods = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:        # from .ledger import …
                        if node.module:
                            mods.add(node.module.split(".")[0])
                    elif node.module:
                        mods.add(node.module.split(".")[0])
            self.assertTrue(mods <= allowed, f"{name} imports outside the allow-list: {mods - allowed}")

    def test_no_model_or_network_dependencies_in_source(self):
        forbidden = ("requests", "urllib", "httpx", "socket", "aiohttp", "openai", "anthropic",
                     "openai", "http.client", "subprocess")
        for name in ("skeleton.py", "enforce.py", "zh.py", "cite_set.py", "scenario.py"):
            path = ROOT / "memory_kernel" / name
            if not path.exists():      # 同上：未实现时跳过
                continue
            src = path.read_text(encoding="utf-8")
            for word in forbidden:
                self.assertNotIn(f"import {word}", src, f"{name} 不得依赖 {word}")
            self.assertNotIn("open(", src, f"{name} 不得做文件 I/O")


if __name__ == "__main__":
    unittest.main()
