# SPDX-License-Identifier: Apache-2.0
"""Tier 1 骨架的单测：接地、时间、以及"骨架失败 = 降级而不是拒绝"。"""

from __future__ import annotations

import hashlib
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import Claim, Citation, SkeletonPolicy, enforce, skeleton_check
from memory_kernel.ledger import AnchorLedger
from memory_kernel.skeleton import (
    POLICY_VERSION,
    STOP_V1,
    build_idf,
    content_tokens,
    hard_units,
    stem,
)

KEY = hashlib.sha256(b"skeleton-test").digest()


class SkeletonUnitTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.a = self.ledger.append(
            "Alice bought a red car on March 5th.", source_id="s", episode_id="e", speaker="Alice"
        )
        self.b = self.ledger.append(
            "Bob moved to Berlin in June.", source_id="s", episode_id="e", speaker="Bob"
        )

    def test_grounded_claim_passes(self):
        r = skeleton_check("Alice bought a red car on March 5th.", [self.a])
        self.assertTrue(r.ok, r.detail)
        self.assertGreater(r.grounding, 0.99)

    def test_ungrounded_claim_fails_on_grounding(self):
        r = skeleton_check("Alice bought a red car on March 5th.", [self.b])
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "grounding")

    def test_temporal_mismatch_fails(self):
        r = skeleton_check("Bob moved to Berlin in September.", [self.b])
        self.assertFalse(r.ok)
        self.assertIn(r.reason, {"grounding", "temporal"})

    def test_temporal_grounding_requires_compatible_month(self):
        r = skeleton_check("Bob moved to Berlin in June.", [self.b])
        self.assertTrue(r.ok, r.detail)

    def test_tau_zero_disables_grounding(self):
        pol = SkeletonPolicy(tau=0.0)
        r = skeleton_check("Alice bought a red car on March 5th.", [self.b], pol)
        # 接地关闭后只剩时间规则，March 5th 在 b 里找不到 → 仍失败于 temporal
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "temporal")


class SkeletonEnforcementTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.a = self.ledger.append(
            "Alice bought a red car on March 5th.", source_id="s", episode_id="e", speaker="Alice"
        )
        self.b = self.ledger.append("Bob moved to Berlin in June.", source_id="s", episode_id="e", speaker="Bob")

    def test_wrong_evidence_is_degraded_not_rejected(self):
        # 引用 B 的原文（逐字通过），但断言说的是 A 的事
        res = enforce(
            self.ledger,
            Claim(Citation(self.b.id, 0, self.b.content), text="Alice bought a red car on March 5th."),
        )
        self.assertEqual(res.verdict, "degraded")
        self.assertIn("skeleton_grounding", [p.name for p in res.failed if not p.passed])

    def test_no_claim_text_means_tier0_only(self):
        res = enforce(self.ledger, Claim(Citation(self.b.id, 0, self.b.content)))
        self.assertEqual(res.verdict, "verified")

    def test_policy_can_escalate_to_rejected(self):
        pol = SkeletonPolicy(tau=0.8, failure_mode="rejected")
        res = enforce(
            self.ledger,
            Claim(Citation(self.b.id, 0, self.b.content), text="Alice bought a red car on March 5th."),
            policy=pol,
        )
        self.assertEqual(res.verdict, "rejected")
        self.assertEqual(res.rejected_cause, "skeleton")

    def test_grounded_composite_passes(self):
        res = enforce(
            self.ledger,
            Claim(
                Citation(self.a.id, 0, self.a.content),
                text="Alice bought a red car on March 5th.",
                supporting=(Citation(self.b.id, 0, self.b.content),),
            ),
        )
        self.assertEqual(res.verdict, "verified")

    def test_missing_supporting_anchor_is_hard_failure(self):
        res = enforce(
            self.ledger,
            Claim(
                Citation(self.a.id, 0, self.a.content),
                text="Alice bought a red car on March 5th.",
                supporting=(Citation("a_999999", 0, "nope"),),
            ),
        )
        self.assertEqual(res.verdict, "rejected")
        self.assertEqual(res.rejected_cause, "invalid_anchor")


class TokenizerV2Test(unittest.TestCase):
    """v2 词表/词干：这些是"算错了"的修复，不是"标准放松"。"""

    def test_stem_collapses_inflections(self):
        pairs = [
            ("wants", "want"), ("said", "say"), ("says", "say"),
            ("researching", "research"), ("transitioned", "transition"),
            ("created", "create"), ("making", "make"), ("using", "use"),
            ("studies", "study"), ("children", "child"), ("worked", "work"),
        ]
        for a, b in pairs:
            self.assertEqual(stem(a), stem(b), f"{a!r} / {b!r} 应落到同一个词干")

    def test_stem_is_versioned(self):
        with self.assertRaises(ValueError):
            stem("wants", version="nope")

    def test_stop_table_covers_pronouns_and_attribution_verbs(self):
        toks = content_tokens("She wants his car and said it was fine")
        for dropped in ("she", "his", "said", "wants", "want", "was"):
            self.assertNotIn(dropped, toks)
        self.assertIn("car", toks)
        self.assertIn("fine", toks)

    def test_possessive_is_stripped(self):
        self.assertIn("joanna", content_tokens("Joanna's project"))
        self.assertEqual(stem("Joanna's"), stem("joanna"))

    def test_v1_policy_reproduces_legacy_formula(self):
        """v1 策略必须逐位复现第五轮的等权覆盖率——否则跨轮数字不可比。"""
        claim = "She wants to start her own business and travel."
        rec = "I want to start my own business."
        legacy_re = re.compile(r"[a-zA-Z][a-zA-Z']{2,}")
        ct = {t.lower() for t in legacy_re.findall(claim)} - STOP_V1
        pt = {t.lower() for t in legacy_re.findall(rec)} - STOP_V1
        expected = len(ct & pt) / len(ct)
        ledger = AnchorLedger(None, KEY)
        r0 = ledger.append(rec, source_id="s", episode_id="e", speaker="Alice")
        pol = SkeletonPolicy(stop_mode="v1", stem=False, hard_units=False, idf=False,
                             temporal=False)
        got = skeleton_check(claim, [r0], pol).grounding
        self.assertAlmostEqual(got, expected, places=9)
        # v2 必须比 v1 高（代词/词形不再计入分母）
        v2 = skeleton_check(claim, [r0], SkeletonPolicy(temporal=False)).grounding
        self.assertGreater(v2, got)


class HardUnitTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.a = self.ledger.append("I ran a 10K race yesterday.", source_id="s",
                                    episode_id="session_1", speaker="Caroline")
        self.b = self.ledger.append("I painted a lake sunrise.", source_id="s",
                                    episode_id="session_1", speaker="Caroline")
        self.sd = {("s", "session_1"): "1:56 pm on 8 May, 2023"}

    def test_number_must_be_grounded(self):
        r = skeleton_check("Caroline ran a 5K race.", [self.a], SkeletonPolicy(),
                           session_dates=self.sd)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "grounding")
        self.assertEqual(r.rule, "hard_unit")
        self.assertIn("num:5", r.hard_missing)

    def test_grounded_number_passes(self):
        r = skeleton_check("Caroline ran a 10K race.", [self.a], SkeletonPolicy(),
                           session_dates=self.sd)
        self.assertTrue(r.ok, r.detail)

    def test_date_numbers_are_delegated_to_temporal(self):
        """7 May 2023 = 会话日期 8 May 减一天：数字不能当硬单元，否则答对也被杀。"""
        r = skeleton_check("Caroline ran a 10K race on 7 May 2023.", [self.a],
                           SkeletonPolicy(), session_dates=self.sd)
        # 硬单元只有 10 和 Caroline；7 / 2023 属于日期表达式，已屏蔽
        self.assertEqual(r.hard_total, 2)
        self.assertEqual(list(r.hard_missing), [])
        self.assertTrue(r.ok, r.detail)

    def test_wrong_absolute_date_fails_temporal(self):
        r = skeleton_check("Caroline ran a 10K race on 9 May 2023.", [self.a],
                           SkeletonPolicy(), session_dates=self.sd)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "temporal")

    def test_proper_noun_must_be_grounded(self):
        r = skeleton_check("Melanie painted a lake sunrise.", [self.b], SkeletonPolicy(),
                           session_dates=self.sd)
        self.assertFalse(r.ok)
        self.assertEqual(r.rule, "hard_unit")
        self.assertIn("proper:melanie", r.hard_missing)

    def test_speaker_name_grounds_proper_noun(self):
        r = skeleton_check("Caroline painted a lake sunrise.", [self.b], SkeletonPolicy(),
                           session_dates=self.sd)
        self.assertTrue(r.ok, r.detail)

    def test_participant_entity_is_not_a_fabrication(self):
        claim = "Gina told Jon to take breaks and destress."
        rec = self.ledger.append(
            "Don't forget to take breaks and dance it out when you need to destress!",
            source_id="s", episode_id="session_1", speaker="Gina")
        without = skeleton_check(claim, [rec], SkeletonPolicy(), session_dates=self.sd)
        self.assertIn("proper:jon", without.hard_missing)
        self.assertFalse(without.ok)
        with_ent = skeleton_check(claim, [rec], SkeletonPolicy(), session_dates=self.sd,
                                  entities={"jon", "gina"})
        self.assertTrue(with_ent.ok, with_ent.detail)

    def test_quoted_span_must_appear_in_evidence(self):
        r = skeleton_check('Caroline said "the group changed my life".', [self.b],
                           SkeletonPolicy(), session_dates=self.sd)
        self.assertFalse(r.ok)
        self.assertTrue(any(m.startswith("quote:") for m in r.hard_missing))

    def test_initial_only_proper_noun_can_be_switched_off(self):
        claim = "Melanie painted a lake sunrise."
        on = [u.value for u in hard_units(claim, include_initial=True)]
        off = [u.value for u in hard_units(claim, include_initial=False)]
        self.assertIn("melanie", on)
        self.assertNotIn("melanie", off)


class IDFWeightingTest(unittest.TestCase):
    def setUp(self):
        self.docs = [{"car", "red", "alice"}, {"car", "blue"}, {"car"}, {"boat", "sea"}]
        self.idf = build_idf(self.docs)

    def test_rare_token_weighs_more(self):
        self.assertGreater(self.idf.weight("sea"), self.idf.weight("car"))
        self.assertGreater(self.idf.weight("never_seen"), self.idf.weight("sea"))

    def test_idf_lowers_coverage_when_the_missing_word_is_rare(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("Alice bought a red car.", source_id="s", episode_id="e",
                            speaker="Alice")
        claim = "Alice bought a red kayak."
        pol_idf = SkeletonPolicy(idf=True, hard_units=False, temporal=False)
        pol_uni = SkeletonPolicy(idf=False, hard_units=False, temporal=False)
        a = skeleton_check(claim, [rec], pol_idf, idf_model=self.idf).soft
        b = skeleton_check(claim, [rec], pol_uni).soft
        self.assertLess(a, b)
        self.assertAlmostEqual(b, 3 / 4, places=9)


class TemporalV2Test(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.a = self.ledger.append("I went to a support group yesterday.", source_id="s",
                                    episode_id="session_1", speaker="Caroline")
        self.sd = {("s", "session_1"): "1:56 pm on 8 May, 2023"}

    def test_absolute_date_matching_session_date(self):
        r = skeleton_check("Caroline went to a support group on 8 May 2023.", [self.a],
                           SkeletonPolicy(), session_dates=self.sd)
        self.assertTrue(r.ok, r.detail)

    def test_session_date_resolves_yesterday(self):
        r = skeleton_check("Caroline went to a support group on 7 May 2023.", [self.a],
                           SkeletonPolicy(), session_dates=self.sd)
        self.assertTrue(r.ok, r.detail)

    def test_relative_phrase_resolves_against_session_date(self):
        r = skeleton_check("Caroline went to a support group the previous day.", [self.a],
                           SkeletonPolicy(), session_dates=self.sd)
        self.assertTrue(r.ok, r.detail)

    def test_without_session_dates_relative_is_unresolved(self):
        r = skeleton_check("Caroline went to a support group on 7 May 2023.", [self.a],
                           SkeletonPolicy())
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "temporal")

    def test_temporal_can_be_switched_off(self):
        pol = SkeletonPolicy(temporal=False)
        r = skeleton_check("Caroline went to a support group on 9 May 2023.", [self.a], pol,
                           session_dates=self.sd)
        self.assertTrue(r.ok, "时间规则关闭后只做接地")


class PolicyVersionTest(unittest.TestCase):
    def test_legacy_tau_alias_still_works(self):
        pol = SkeletonPolicy(tau=0.8)
        self.assertEqual(pol.tau_soft, 0.8)
        self.assertEqual(pol.tau, 0.8)
        self.assertEqual(SkeletonPolicy().tau_soft, 0.8)

    def test_result_carries_version_and_components(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("Alice bought a red car.", source_id="s", episode_id="e",
                            speaker="Alice")
        r = skeleton_check("Alice bought a red car.", [rec])
        self.assertEqual(r.version, POLICY_VERSION)
        self.assertTrue(r.ok)
        self.assertIn("skel-", r.version)
        self.assertIn("tau_soft", SkeletonPolicy().describe())

    def test_tau_zero_disables_grounding_including_hard_units(self):
        """τ_soft=0 表示关闭整个接地检查（含硬单元），与 v1 的 tau=0 语义一致。"""
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("Bob moved to Berlin.", source_id="s", episode_id="e",
                            speaker="Bob")
        r = skeleton_check("Melanie ran a 5K race in September.", [rec],
                           SkeletonPolicy(tau=0.0))
        self.assertNotEqual(r.rule, "hard_unit")
        self.assertFalse(r.ok)          # 仍会因时间论证失败
        self.assertEqual(r.reason, "temporal")


if __name__ == "__main__":
    unittest.main(verbosity=2)