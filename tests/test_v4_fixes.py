# SPDX-License-Identifier: Apache-2.0
"""v4.1 修复回归：中文否定「子句级范围」+ 断言文本级否定极性。

背景：v4 首版在真实聊天探针上 FPR_strict 7/71→14/71，新增误杀全部是 zh
preserve_negation 用**整句**范围比较片段引用（"我没去杭州，但我朋友去了"引用后半句）。
本文件锁住修复后的语义：

1. 子句级范围：引用后半句（与前句否定无关）→ 不误杀。
2. 断言文本级否定：claim 引入否定 / 丢掉否定 → 硬拒（N_negation 缺口的真修复）。
3. 默认策略（zh=False）逐字节不变：同样输入仍为 verified，且不产生新谓词。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel import AnchorLedger, Claim, Citation, enforce  # noqa: E402
from memory_kernel.skeleton import SkeletonPolicy  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="v41-")
        self.path = os.path.join(self.tmp, "ledger.jsonl")
        self.ledger = AnchorLedger(self.path, KEY)
        self.clause = self.ledger.append(
            "我没去杭州，但我朋友去了。", source_id="s1", episode_id="e1", turn_index=0, speaker="user")
        self.hz = self.ledger.append(
            "我上周三去了杭州，住在西湖边的一家民宿。", source_id="s1", episode_id="e1", turn_index=1, speaker="user")
        self.car_neg = self.ledger.append(
            "我没有把车卖掉。", source_id="s1", episode_id="e1", turn_index=2, speaker="user")
        self.other = self.ledger.append(
            "今天天气不错，适合出去玩。", source_id="s2", episode_id="e2", turn_index=0, speaker="user")

    def _run(self, rec, quote, text, policy, **kw):
        off = rec.content.find(quote)
        self.assertGreaterEqual(off, 0, "quote 必须来自记录")
        return enforce(self.ledger, Claim(Citation(rec.id, off, quote), text=text), policy=policy, **kw)


class ClauseScopeNegationTest(_Base):
    def test_fragment_quote_ignores_negation_in_other_clause(self):
        """引用后半句（无否定），前半句有否定 → 不应误杀。"""
        quote = "但我朋友去了"
        res = self._run(self.clause, quote, quote, SkeletonPolicy.v4())
        self.assertEqual(res.verdict, "verified", res.failed)
        self.assertEqual([p.name for p in res.failed if not p.passed], [])

    def test_same_clause_dropped_negation_still_fails(self):
        """同一子句里的否定落在引用片段之外 → 仍要判为丢否定。"""
        quote = "车卖掉"  # 证据同句是"我没有把车卖掉。"
        res = self._run(self.car_neg, quote, quote, SkeletonPolicy.v4())
        self.assertEqual(res.verdict, "rejected")
        self.assertEqual(res.rejected_cause, "preserve_negation")


class ClaimNegationPolarityTest(_Base):
    def test_claim_introduces_negation_is_rejected(self):
        """claim 在逐字引用前插入否定（N_negation 的真实构造）→ 硬拒。"""
        claim = "没有" + self.hz.content
        res = self._run(self.hz, self.hz.content, claim, SkeletonPolicy.v4())
        self.assertEqual(res.verdict, "rejected")
        self.assertEqual(res.rejected_cause, "preserve_negation_claim")
        self.assertIn("preserve_negation_claim", res.hard_failed)

    def test_free_paraphrase_is_not_negation_rejected(self):
        """自由转述（claim 与引用字数面重合低）不适用该硬规则，不能误杀。"""
        res = self._run(self.hz, self.hz.content, "用户从未到过杭州，也没住过西湖边。",
                        SkeletonPolicy.v4())
        self.assertNotIn("preserve_negation_claim", res.hard_failed)

    def test_claim_drops_negation_is_rejected(self):
        """claim 丢掉证据里的否定（摘录型）→ 硬拒。"""
        res = self._run(self.car_neg, self.car_neg.content, "我把车卖掉了。", SkeletonPolicy.v4())
        self.assertEqual(res.verdict, "rejected")
        self.assertEqual(res.rejected_cause, "preserve_negation_claim")

    def test_both_negative_is_not_negation_failure(self):
        """claim 与引用同为否定 → 否定谓词应通过（其余谓词另算）。"""
        res = self._run(self.car_neg, self.car_neg.content, "用户没有卖掉车。", SkeletonPolicy.v4())
        self.assertNotIn("preserve_negation_claim", res.hard_failed)


class DefaultPolicyUnchangedTest(_Base):
    def test_default_policy_has_no_v4_predicates(self):
        """zh=False（默认 v2）：同样输入仍是 verified，且不出现 v4 谓词。"""
        res = self._run(self.hz, self.hz.content, "用户没有去过杭州。", SkeletonPolicy())
        self.assertEqual(res.verdict, "verified")
        self.assertNotIn("preserve_negation_claim", [p.name for p in res.failed])
        self.assertNotIn("preserve_negation", [p.name for p in res.failed if not p.passed])


class OverCitationAndScenarioTest(_Base):
    def test_over_citation_flagged_under_v4_only(self):
        quote = "我上周三去了杭州"
        claim = Claim(
            Citation(self.hz.id, self.hz.content.find(quote), quote),
            text=quote,
            supporting=(Citation(self.other.id, 0, self.other.content),),
        )
        old = enforce(self.ledger, claim, policy=SkeletonPolicy())
        new = enforce(self.ledger, claim, policy=SkeletonPolicy.v4())
        self.assertEqual(old.verdict, "verified")
        self.assertEqual(new.verdict, "degraded")
        self.assertIn("over_citation", [p.name for p in new.failed if not p.passed])

    def test_scenario_timeline_rejects_late_evidence(self):
        from memory_kernel import ScenarioContext
        ctx = ScenarioContext(context_session_id="s1", context_time=0.0,
                              asserted_scope="unspecified", record_times={self.hz.id: 1e9})
        res = self._run(self.hz, self.hz.content, self.hz.content, SkeletonPolicy.v4(), context=ctx)
        self.assertEqual(res.verdict, "rejected")
        self.assertEqual(res.rejected_cause, "scenario")
        self.assertIn("scenario_timeline", res.hard_failed)

    def test_scenario_none_keeps_v3_behavior(self):
        a = enforce(self.ledger, Claim(Citation(self.hz.id, 0, self.hz.content), text=self.hz.content),
                    policy=SkeletonPolicy.v4())
        b = enforce(self.ledger, Claim(Citation(self.hz.id, 0, self.hz.content), text=self.hz.content),
                    policy=SkeletonPolicy.v4(), context=None)
        self.assertEqual((a.verdict, a.rejected_cause), (b.verdict, b.rejected_cause))


class DeterminismTest(_Base):
    def test_repeated_v4_verdicts_identical(self):
        pol = SkeletonPolicy.v4()
        first = self._run(self.hz, self.hz.content, "用户没有去过杭州。", pol)
        for _ in range(10):
            again = self._run(self.hz, self.hz.content, "用户没有去过杭州。", pol)
            self.assertEqual(first.verdict, again.verdict)
            self.assertEqual(first.rejected_cause, again.rejected_cause)
            self.assertEqual([p.name for p in first.failed], [p.name for p in again.failed])


if __name__ == "__main__":
    unittest.main()
