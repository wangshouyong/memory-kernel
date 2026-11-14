# SPDX-License-Identifier: Apache-2.0
"""skel-v3.1（四个误杀修复开关）的单测。

覆盖任务书要求的六项：
  * 四个开关**各自**在 report7 §6 目标文本上的机制（nonevent / noun_guard /
    no_event_object / np_object）；
  * §6 的 7 条误杀（2 P_lab + 1 cases6 P1 + 4 C_ctrl）在四个开关全开时全部放行、
    在 v3.0（全关）时全部被拒；
  * **flags-off == v3.0 逐位一致**（含 `detail` / `version`，比的是整个 `SkeletonResult`）；
  * 四个开关在 v2 路径（`arg_slots=False, dispatch=False`）上完全惰性；
  * 版本字符串（`skel-v2.0` / `skel-v3.0` / `skel-v3.1`）与 `describe()` 的开关位；
  * 确定性 + 纯函数（重复调用逐位一致、不改入参）。

只测 `memory_kernel` 的公开 API；不依赖 eval/ 下任何数据（测试必须离线可跑）。
§6 的 7 条文本与它们的引用原文是**逐字**从冻结数据里抄来的，所以这些用例同时是
评测台账的单测版本。
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import SkeletonPolicy, skeleton_check  # noqa: E402
from memory_kernel.ledger import AnchorLedger  # noqa: E402
from memory_kernel.skeleton import (  # noqa: E402
    POLICY_VERSION,
    POLICY_VERSION_V3,
    POLICY_VERSION_V31,
    POLICY_VERSION_V31_C3,
    UNIFORM_IDF,
    argument_grounding,
    claim_arguments,
)

KEY = hashlib.sha256(b"skeleton-v31-test").digest()

V31_FLAGS = ("v31_nonevent", "v31_noun_guard", "v31_no_event_object", "v31_np_object")
ALL_ON = {name: True for name in V31_FLAGS}
ALL_OFF = {name: False for name in V31_FLAGS}


def _ground(text, records, *, flags=ALL_OFF, tau=0.0):
    """直接测论元层（tau=0 → 只留槽位/分派规则，屏蔽覆盖率与软词阈值）。"""
    return argument_grounding(
        claim_arguments(text, **{k: v for k, v in flags.items()
                                 if k in ("v31_nonevent", "v31_noun_guard", "v31_np_object")}),
        records, idf=UNIFORM_IDF, tau=tau, arg_slots=True, dispatch=True,
        v31_no_event_object=flags.get("v31_no_event_object", False),
        v31_np_object=flags.get("v31_np_object", False),
    )


class _Fixture(unittest.TestCase):
    """report7 §6 的 7 条误杀 + 它们的引用原文（逐字来自 bench6/cases6 冻结数据）。"""

    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)

        def rec(content, speaker):
            return self.ledger.append(content, source_id="s", episode_id="e", speaker=speaker)

        self.cases = {
            "P_lab/00886": (
                "Dave attended an event with many classic cars on show and was in awe.",
                [rec("They had so many classic cars on show, I was in awe.", "Dave")],
            ),
            "P_lab/00887": (
                "Dave finds classic cars charming and is amazed by the dedication people "
                "put into restoring them.",
                [rec("Classic cars are so charming and the dedication people put into "
                     "restoring them is amazing.", "Dave")],
            ),
            "cases6_P1/00020": (
                "Jon thinks the ideal dance studio should have great natural light.",
                [rec("I even found a place with great natural light!", "Jon")],
            ),
            "C_ctrl/01805": (
                "Hope you're doing fine",
                [rec("Hope you're doing good.", "Caroline")],
            ),
            "C_ctrl/01871": (
                "Sorry about your work Jon, but starting your own business sounds awesome!",
                [rec("Sorry about your job Jon, but starting your own business sounds "
                     "awesome!", "Gina")],
            ),
            "C_ctrl/01874": (
                "I rehearsed with a small circle of dancers after job",
                [rec("I rehearsed with a small group of dancers after work.", "Jon")],
            ),
            "C_ctrl/01891": (
                "Hey Gina, hope you're doing fantastic",
                [rec("Hey Gina, hope you're doing great!", "Jon")],
            ),
        }
        # 两条 v3.0 会**放行**的正样本：保证"flags-off == v3.0"的对比不是空集
        self.positives = {
            "pos/red_car": ("Alice bought a red car on March 5th.",
                            [rec("Alice bought a red car on March 5th.", "Alice")]),
            "pos/door_dash": (
                "Gina lost her job at Door Dash this month.",
                [rec("Unfortunately, I also lost my job at Door Dash this month.", "Gina")],
            ),
        }


# --------------------------------------------------------------------------- #
# 1. 四个开关各自的机制（单元级）
# --------------------------------------------------------------------------- #

class NoneventSwitchTest(_Fixture):
    def test_stance_predicate_is_not_an_event_predicate(self):
        text, recs = self.cases["P_lab/00887"]
        off = claim_arguments(text)
        on = claim_arguments(text, v31_nonevent=True)
        self.assertEqual(off[0].main_predicate, "find",
                         "v3.0 把私态谓词 find 当成必须接地的事件谓词（这就是误杀机制）")
        self.assertIsNone(on[0].main_predicate,
                          "v3.1 把 find 归入非事件谓词族 ⇒ 没有必须接地的谓词槽")

    def test_shell_noun_framework_makes_predicate_optional(self):
        text, recs = self.cases["P_lab/00886"]
        off = claim_arguments(text)
        on = claim_arguments(text, v31_nonevent=True)
        self.assertFalse(off[0].predicate_optional)
        self.assertTrue(on[0].predicate_optional,
                        "attended an event with … 是可核实内容在修饰语里的壳名词框架")
        # 客体区必须保留附加语（否则内容词被杀光）
        self.assertIn("classic", [t.lower() for t in on[0].object_tokens])

    def test_modals_and_copulars_are_nonevent(self):
        on = claim_arguments("Jon thinks the ideal studio should be quiet.",
                             v31_nonevent=True)
        self.assertIsNone(on[0].main_predicate,
                          "think/should/be 全在非事件族里 ⇒ 没有事件谓词")


class NounGuardTest(_Fixture):
    def test_noun_reading_of_dance_is_not_a_verb(self):
        text, recs = self.cases["cases6_P1/00020"]
        off = claim_arguments(text)
        on = claim_arguments(text, v31_noun_guard=True)
        self.assertIn("danc", off[0].predicates,
                      "v3.0 把 the ideal dance studio 里的 dance 当成限定动词")
        self.assertNotIn("danc", on[0].predicates)
        self.assertIsNone(on[0].main_predicate)

    def test_possessive_guards_work(self):
        text, recs = self.cases["C_ctrl/01871"]
        off = claim_arguments(text)
        on = claim_arguments(text, v31_noun_guard=True)
        self.assertIn("work", off[0].predicates, "your work 的 work 被当成谓词")
        self.assertNotIn("work", on[0].predicates)
        self.assertEqual(on[0].main_predicate, "start")

    def test_guard_does_not_swallow_real_verbs_after_subject(self):
        # 主体之后的限定动词不受影响（护栏只看"前一词/前两步是限定词"）
        (u,) = claim_arguments("Jon bought a new car.", v31_noun_guard=True)
        self.assertEqual(u.main_predicate, "buy")
        self.assertEqual(list(u.predicates)[:1], ["buy"])


class NoEventObjectTest(_Fixture):
    def test_object_slot_collapses_without_event_core(self):
        text, recs = self.cases["C_ctrl/01805"]
        off = _ground(text, recs)
        on = _ground(text, recs, flags={**ALL_OFF, "v31_no_event_object": True})
        self.assertFalse(off.ok)
        self.assertEqual(off.rule, "arg_object")
        self.assertTrue(on.ok, "没有事件性谓词 ⇒ 客体区退化，不构成可核实客体槽")

    def test_event_claims_still_check_objects(self):
        # 有事件谓词时客体槽照旧要接地（开关不能变成"一律不查客体"）
        recs = self.ledger
        record = recs.append("Alice bought a red car.", source_id="s", episode_id="e",
                             speaker="Alice")
        rep = _ground("Alice bought a blue truck.", [record],
                      flags={**ALL_OFF, "v31_no_event_object": True})
        self.assertFalse(rep.ok, "客体被换掉仍然必须被拒")
        self.assertEqual(rep.rule, "arg_object")


class NpObjectTest(_Fixture):
    def test_adjunct_after_preposition_is_not_the_object(self):
        text = "Alice donated her old car to a homeless shelter yesterday."
        off = claim_arguments(text)
        on = claim_arguments(text, v31_np_object=True)
        self.assertEqual(tuple(t.lower() for t in off[0].object_tokens),
                         ("old", "car", "homeless", "shelter"))
        self.assertEqual(tuple(t.lower() for t in on[0].object_tokens), ("old", "car"),
                         "介词后的附加语不是直接宾语")
        self.assertEqual(on[0].object_np_head, "car",
                         "客体 NP 的中心名词只作诊断，不参与判定")

    def test_adjunct_only_object_region_becomes_empty(self):
        text, recs = self.cases["C_ctrl/01874"]
        off = claim_arguments(text)
        on = claim_arguments(text, v31_np_object=True)
        self.assertIn("job", [t.lower() for t in off[0].object_tokens])
        self.assertEqual(tuple(on[0].object_tokens), ())


# --------------------------------------------------------------------------- #
# 2. §6 的 7 条误杀（回归台账）
# --------------------------------------------------------------------------- #

class Section6FalseKillsTest(_Fixture):
    def test_v30_rejects_all_seven(self):
        for name, (text, recs) in self.cases.items():
            with self.subTest(name):
                rep = _ground(text, recs)
                self.assertFalse(rep.ok, f"{name} 在 v3.0 下应当被拒（这是要修的误杀）")
                self.assertIn(rep.rule, ("arg_subject", "arg_predicate", "arg_object",
                                         "arg_unit_coverage"))

    def test_v31_accepts_all_seven(self):
        for name, (text, recs) in self.cases.items():
            with self.subTest(name):
                rep = _ground(text, recs, flags=ALL_ON)
                self.assertTrue(rep.ok, f"{name} 在 v3.1 下应当放行：{rep.rule} {rep.detail}")

    def test_attribution_of_each_fix(self):
        """每个开关单独开时，各自修好的是哪几条（与评测台账一致）。"""
        expect = {
            "v31_nonevent": {"P_lab/00886", "P_lab/00887"},
            "v31_noun_guard": {"cases6_P1/00020", "C_ctrl/01871", "C_ctrl/01874"},
            "v31_no_event_object": {"C_ctrl/01805", "C_ctrl/01891"},
            "v31_np_object": {"C_ctrl/01874"},
        }
        for flag, want in expect.items():
            got = set()
            for name, (text, recs) in self.cases.items():
                if _ground(text, recs, flags=ALL_OFF).ok:
                    continue                       # 已放行的不算"修好"
                if _ground(text, recs, flags={**ALL_OFF, flag: True}).ok:
                    got.add(name)
            with self.subTest(flag):
                self.assertEqual(got, want)

    def test_each_switch_fires_on_its_intended_text(self):
        """四个开关都必须在自己那条目标文本上留下可观察的机制痕迹。"""
        # nonevent：谓词槽消失
        t, r = self.cases["P_lab/00887"]
        self.assertIsNone(claim_arguments(t, v31_nonevent=True)[0].main_predicate)
        # noun_guard：名词位置上的候选动词被剔除
        t, r = self.cases["cases6_P1/00020"]
        self.assertNotIn("danc", claim_arguments(t, v31_noun_guard=True)[0].predicates)
        # no_event_object：客体槽不再判定
        t, r = self.cases["C_ctrl/01805"]
        self.assertTrue(_ground(t, r, flags={**ALL_OFF,
                                             "v31_no_event_object": True}).ok)
        # np_object：客体区截断到直接宾语 NP
        t, r = self.cases["C_ctrl/01874"]
        self.assertEqual(tuple(claim_arguments(t, v31_np_object=True)[0].object_tokens), ())


# --------------------------------------------------------------------------- #
# 3. 向后兼容：flags-off == v3.0；v2 路径完全惰性
# --------------------------------------------------------------------------- #

class BackCompatTest(_Fixture):
    def test_default_policy_is_v2_and_flags_are_false(self):
        pol = SkeletonPolicy()
        self.assertEqual(pol.version, POLICY_VERSION)
        for name in V31_FLAGS:
            self.assertIs(getattr(pol, name), False, f"默认策略的 {name} 必须关闭")
        self.assertEqual(pol.describe(), SkeletonPolicy().describe())

    def test_v3_policy_still_v30_and_flags_false(self):
        pol = SkeletonPolicy.v3()
        self.assertEqual(pol.version, POLICY_VERSION_V3)
        for name in V31_FLAGS:
            self.assertIs(getattr(pol, name), False)
        self.assertNotIn("v31(", pol.describe())

    def test_v3_1_policy_sets_version_and_all_flags(self):
        pol = SkeletonPolicy.v3_1()
        self.assertEqual(pol.version, POLICY_VERSION_V31)
        self.assertEqual(POLICY_VERSION_V31, "skel-v3.1")
        for name in V31_FLAGS:
            self.assertIs(getattr(pol, name), True)
        d = pol.describe()
        self.assertIn("skel-v3.1", d)
        self.assertIn("v31(nonevent=1, noun_guard=1, no_event_obj=1, np_obj=1)", d)

    def test_flags_off_is_bit_identical_to_v30(self):
        """显式把四个开关都关掉，必须与 v3.0 的全字段结果逐位相同。"""
        for tau in (0.2, 0.35, 0.5):
            pol30 = SkeletonPolicy(tau_soft=tau, arg_slots=True, dispatch=True,
                                   version=POLICY_VERSION_V3)
            polx = SkeletonPolicy(tau_soft=tau, arg_slots=True, dispatch=True,
                                  version=POLICY_VERSION_V3, **ALL_OFF)
            oks = 0
            for name, (text, recs) in {**self.cases, **self.positives}.items():
                a = skeleton_check(text, recs, pol30)
                b = skeleton_check(text, recs, polx)
                self.assertEqual(a, b, f"tau={tau} {name}: flags-off 改变了 v3.0 判定")
                oks += int(a.ok)
            self.assertGreater(oks, 0, "对比不能是空集（全部被拒时相等没有意义）")

    def test_v31_flags_are_inert_on_v2_path(self):
        """arg_slots/dispatch 都关时，四个开关不能改变任何判定（v2 逐位不变）。"""
        for name, (text, recs) in self.cases.items():
            old = skeleton_check(text, recs, SkeletonPolicy())
            new = skeleton_check(text, recs, SkeletonPolicy(**ALL_ON))
            self.assertEqual(old, new, f"{name}: v31 开关在 v2 路径上不惰性")

    def test_v3_1_superset_of_v3_rejections_on_fixtures(self):
        """v3.1 允许放松（所以不能断言 v3.0 拒 ⇒ v3.1 拒），但断言：
        v3.1 收下的样本里，v3.0 收下的必须仍被收下（单调性只在"v3.0 收"方向成立）。"""
        for name, (text, recs) in self.cases.items():
            a = skeleton_check(text, recs, SkeletonPolicy.v3())
            b = skeleton_check(text, recs, SkeletonPolicy.v3_1())
            if a.ok:
                self.assertTrue(b.ok, f"{name}: v3.0 放行的被判 v3.1 拒了")


# --------------------------------------------------------------------------- #
# 4. 确定性 + 纯函数
# --------------------------------------------------------------------------- #

class DeterminismTest(_Fixture):
    def test_repeated_calls_are_identical(self):
        pol = SkeletonPolicy.v3_1()
        for name, (text, recs) in self.cases.items():
            a = skeleton_check(text, recs, pol)
            b = skeleton_check(text, recs, pol)
            self.assertEqual(a, b, name)

    def test_claim_arguments_is_pure_and_stable(self):
        text, _ = self.cases["C_ctrl/01871"]
        one = claim_arguments(text, v31_nonevent=True, v31_noun_guard=True, v31_np_object=True)
        two = claim_arguments(text, v31_nonevent=True, v31_noun_guard=True, v31_np_object=True)
        self.assertEqual(one, two)
        self.assertEqual(text,
                         "Sorry about your work Jon, but starting your own business sounds "
                         "awesome!", "输入文本不应被修改")

    def test_grounding_is_pure(self):
        text, recs = self.cases["C_ctrl/01805"]
        before = [r.content for r in recs]
        a = _ground(text, recs, flags=ALL_ON)
        b = _ground(text, recs, flags=ALL_ON)
        self.assertEqual(a, b)
        self.assertEqual(before, [r.content for r in recs])


# --------------------------------------------------------------------------- #
# 5. v3.1-C3（用户裁决 C2）：正式入口 `SkeletonPolicy.v3_c3()`
#
# C3 = {nonevent, noun_guard, no_event_object} 三开，`np_object` 否决；
# 实测生产点 τ=0.45（`eval/out/results7b.json:selected_c3_candidate`）。
# 这一节钉住三件事：构造器形状（含 np_object=False）、7 条误杀在 C3 下放行、
# 以及"默认路径没被这次改动碰到"（v2.0 / v3.0 / v3.1 逐位不变）。
# --------------------------------------------------------------------------- #

C3_FLAGS_ON = ("v31_nonevent", "v31_noun_guard", "v31_no_event_object")


class V3C3PolicyTest(_Fixture):
    def test_defaults_are_the_measured_c3_production_point(self):
        pol = SkeletonPolicy.v3_c3()
        self.assertEqual(POLICY_VERSION_V31_C3, "skel-v3.1-C3")
        self.assertEqual(pol.version, POLICY_VERSION_V31_C3)
        self.assertAlmostEqual(pol.tau_soft, 0.45)
        for name in ("arg_slots", "dispatch", "hard_units", "temporal"):
            self.assertIs(getattr(pol, name), True, f"C3 生产点要求 {name}=True")
        for name in C3_FLAGS_ON:
            self.assertIs(getattr(pol, name), True, f"C3 要求 {name}=True")
        self.assertIs(pol.v31_np_object, False,
                      "np_object 已被数据否决（report7b §5/§9），C3 不得打开")

    def test_describe_shows_c3_version_and_np_obj_off(self):
        d = SkeletonPolicy.v3_c3().describe()
        self.assertIn("skel-v3.1-C3", d)
        self.assertIn("np_obj=0", d)
        self.assertIn("nonevent=1, noun_guard=1, no_event_obj=1", d)
        self.assertIn("tau_soft=0.45", d)

    def test_default_policy_is_still_v2_and_untouched(self):
        """默认路径：v2.0、四个开关全关、describe 逐字不变。"""
        pol = SkeletonPolicy()
        self.assertEqual(pol.version, POLICY_VERSION)
        for name in V31_FLAGS:
            self.assertIs(getattr(pol, name), False, f"默认策略的 {name} 必须关闭")
        self.assertEqual(
            pol.describe(),
            "skel-v2.0(tau_soft=0.80, hard=1, idf=1, stem=1, stop=v2, "
            "temporal=1, session=1)",
            "v2 默认策略的 describe 必须是冻结值（C3 的加入不许碰默认路径）")

    def test_v3_and_v3_1_constructors_are_unchanged(self):
        pol30 = SkeletonPolicy.v3()
        self.assertEqual(pol30.version, POLICY_VERSION_V3)
        for name in V31_FLAGS:
            self.assertIs(getattr(pol30, name), False)
        pol31 = SkeletonPolicy.v3_1()
        self.assertEqual(pol31.version, POLICY_VERSION_V31)
        for name in V31_FLAGS:
            self.assertIs(getattr(pol31, name), True, "v3_1() 仍是四开实验配置")
        # C3 与四开只差 np_object 一位
        c3 = SkeletonPolicy.v3_c3()
        self.assertIs(c3.v31_np_object, False)
        self.assertIs(pol31.v31_np_object, True)
        self.assertNotEqual(c3, pol31, "C3 不能与四开混为一谈")
        self.assertEqual(c3.tau_soft, 0.45)

    def test_version_constant_is_exported(self):
        from memory_kernel import POLICY_VERSION_V31_C3 as exported
        from memory_kernel import skeleton as skeleton_mod
        self.assertEqual(exported, POLICY_VERSION_V31_C3)
        self.assertIn("POLICY_VERSION_V31_C3", skeleton_mod.__all__)

    def test_c3_accepts_the_false_kills_it_was_adopted_for(self):
        """§6 的 7 条误杀：v3.0（同 τ=0.45）全拒、C3 全放行。

        至少覆盖任务书的三个代表项（00886/00887 P_lab、00020 P1、01805/01891 C_ctrl）；
        这里对 7 条逐条断言，是因为 report7b §6 就是 7/7。
        """
        representative = ("P_lab/00886", "P_lab/00887", "cases6_P1/00020",
                          "C_ctrl/01805", "C_ctrl/01891")
        pol_c3 = SkeletonPolicy.v3_c3()
        pol30 = SkeletonPolicy(tau_soft=0.45, hard_units=True, temporal=True,
                               arg_slots=True, dispatch=True,
                               version=POLICY_VERSION_V3)
        for name, (text, recs) in self.cases.items():
            with self.subTest(name):
                before = skeleton_check(text, recs, pol30)
                after = skeleton_check(text, recs, pol_c3)
                self.assertFalse(before.ok, f"{name}: v3.0@0.45 应当拒（这是被修的误杀）")
                self.assertTrue(after.ok,
                                f"{name}: C3 应当放行，实际 rule={after.rule} "
                                f"detail={after.detail}")
        for name in representative:
            with self.subTest(representative=name):
                text, recs = self.cases[name]
                self.assertTrue(skeleton_check(text, recs, pol_c3).ok)

    def test_c3_still_rejects_wrong_evidence_and_fabricated_objects(self):
        """放松不是"一律放行"：客体被换掉 / 谓词编造仍必须被拒。"""
        pol = SkeletonPolicy.v3_c3()
        record = self.ledger.append("Alice bought a red car on March 5th.",
                                    source_id="s", episode_id="e", speaker="Alice")
        # 对照：逐字支持的 claim 必须放行（否则下面的"被拒"没有意义）
        good = skeleton_check("Alice bought a red car on March 5th.", [record], pol)
        self.assertTrue(good.ok, f"支持 Claim 被判死：{good.rule} {good.detail}")
        for text, want_rule in (
            ("Alice bought a blue truck on March 5th.", "arg_object"),
            ("Alice donated her old car to a homeless shelter yesterday.", "arg_predicate"),
        ):
            with self.subTest(text[:32]):
                rep = skeleton_check(text, [record], pol)
                self.assertFalse(rep.ok, "编造/错证据的 claim 必须被拒")
                self.assertEqual(rep.rule, want_rule)

    def test_c3_is_deterministic(self):
        pol = SkeletonPolicy.v3_c3()
        for name, (text, recs) in self.cases.items():
            with self.subTest(name):
                a = skeleton_check(text, recs, pol)
                b = skeleton_check(text, recs, pol)
                self.assertEqual(a, b, "同一输入两次判定必须逐位一致")
                self.assertEqual(a.version, POLICY_VERSION_V31_C3)

    def test_c3_is_enforce_compatible(self):
        """`enforce(policy=SkeletonPolicy.v3_c3())` 能直接跑：支持→verified，错证据→degraded。"""
        from memory_kernel import Claim, Citation, enforce

        pol = SkeletonPolicy.v3_c3()
        # 独立小账本：IDF 由账本现算（生产路径），两条记录保证权重接近等权、结果稳定
        led = AnchorLedger(None, KEY)
        rec = led.append("Alice bought a red car on March 5th.", source_id="s",
                         episode_id="e", speaker="Alice")
        led.append("Bob painted the fence last week.", source_id="s", episode_id="e",
                   speaker="Bob")
        ok_res = enforce(led,
                         Claim(citation=Citation(rec.id, 0, rec.content),
                               text="Alice bought a red car on March 5th."),
                         policy=pol)
        self.assertEqual(ok_res.verdict, "verified", str(ok_res.failed))
        bad = enforce(led,
                      Claim(citation=Citation(rec.id, 0, rec.content),
                            text="Alice bought a blue truck on March 5th."),
                      policy=pol)
        self.assertEqual(bad.verdict, "degraded", "C3 是软失败（degraded），不是硬拒绝")
        self.assertIn("skeleton_grounding", bad.soft_failed)


if __name__ == "__main__":
    unittest.main()
