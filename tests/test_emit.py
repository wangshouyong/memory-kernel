# SPDX-License-Identifier: Apache-2.0
"""阶段三发射门控单测（契约 docs/EMISSION-CONTRACT.md §3）。"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel import AnchorLedger  # noqa: E402
from memory_kernel.emit import (  # noqa: E402
    FALLBACK_TEXT,
    SPECULATION_PREFIX,
    emit_answer,
)
from memory_kernel.scenario import ScenarioContext  # noqa: E402
from memory_kernel.skeleton import SkeletonPolicy  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"
CITE_SET = SkeletonPolicy(cite_set=True)


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AnchorLedger(None, KEY)
        self.car = self.ledger.append("Alice bought a red car on March 5th.", source_id="s1",
                                      episode_id="e1", turn_index=0, speaker="Alice")
        self.weather = self.ledger.append("The weather in Berlin was cold last week.", source_id="s1",
                                          episode_id="e1", turn_index=1, speaker="Bob")

    def _cite(self, rec):
        return {"anchor": rec.id, "quote": rec.content}


class VerifiedKeepTest(_Base):
    def test_verified_is_kept_verbatim(self):
        res = emit_answer([{"type": "claim", "text": self.car.content,
                            "cites": [self._cite(self.car)]}], self.ledger)
        self.assertEqual(res.text, self.car.content)
        self.assertEqual(res.audit[0]["action"], "keep")
        self.assertEqual(res.audit[0]["verdict"], "verified")


class DegradedHedgeTest(_Base):
    def test_over_citation_gets_own_short_wording(self):
        res = emit_answer([{"type": "claim", "text": self.car.content,
                            "cites": [self._cite(self.car), self._cite(self.weather)]}],
                          self.ledger, policy=CITE_SET)
        self.assertTrue(res.text.startswith(self.car.content.rstrip(".")), res.text)
        self.assertIn("（这条引用有冗余", res.text)
        self.assertIn("仅供参考", res.text)
        self.assertEqual(res.audit[0]["action"], "hedged")
        self.assertEqual(res.audit[0]["verdict"], "degraded")
        self.assertEqual(res.audit[0]["degrade_reason"], "over_citation")

    def test_reason_templates_are_distinct_and_short(self):
        # 用户裁决：B 情况按原因分别措辞，不用统一长前缀
        from memory_kernel.emit import HEDGED_BY_REASON, hedged_text

        keys = {"over_citation", "scenario_superseded", "argument_mismatch"}
        self.assertTrue(keys.issubset(HEDGED_BY_REASON))
        rendered = {k: hedged_text("这句话", k) for k in keys}
        self.assertEqual(len(set(rendered.values())), 3)
        for text in rendered.values():
            self.assertLessEqual(len(text), 40, text)  # 短说明，不是长前缀

    def test_scenario_superseded_uses_its_own_wording(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("我们下季度工业增加值预计增长5%。", source_id="sz",
                            episode_id="ez", turn_index=0, speaker="user")
        ledger.append("不是这样，工业增加值是增长3%。", source_id="sz",
                      episode_id="ez", turn_index=1, speaker="user")
        claim = {"type": "claim", "text": rec.content,
                 "cites": [{"anchor": rec.id, "quote": rec.content}]}
        res = emit_answer([claim], ledger,
                          policy=SkeletonPolicy(scenario=True, zh=True),
                          context=ScenarioContext())
        self.assertEqual(res.audit[0]["action"], "hedged")
        self.assertEqual(res.audit[0]["degrade_reason"], "scenario_superseded")
        self.assertIn("后来被更正过", res.text)

    def test_argument_mismatch_uses_its_own_wording(self):
        from memory_kernel.emit import hedged_text

        text = hedged_text("他昨天去了北京", "argument_mismatch")
        self.assertIn("对不上", text)
        self.assertNotIn("引用有冗余", text)


class RejectedDropTest(_Base):
    def test_fabricated_quote_is_deleted_and_falls_back(self):
        res = emit_answer([{"type": "claim", "text": "Alice bought a blue car.",
                            "cites": [{"anchor": self.car.id, "quote": "Alice bought a blue car."}]}],
                          self.ledger)
        self.assertEqual(res.text, FALLBACK_TEXT)
        self.assertEqual(res.audit[0]["action"], "drop")

    def test_rejected_claim_removed_but_narrative_kept(self):
        res = emit_answer([
            {"type": "narrative", "text": "先说结论。"},
            {"type": "claim", "text": "Alice bought a blue car.",
             "cites": [{"anchor": self.car.id, "quote": "not the record"}]},
        ], self.ledger)
        self.assertEqual(res.text, "先说结论。")
        self.assertEqual([a["action"] for a in res.audit], ["keep", "drop"])

    def test_middle_deletion_keeps_context_with_redaction_marker(self):
        # 删句会切断上下文：中间被删的 claim 在原句位留省略标记，前后句仍连贯
        res = emit_answer([
            {"type": "narrative", "text": "第一句。"},
            {"type": "claim", "text": self.car.content, "cites": [self._cite(self.car)]},
            {"type": "claim", "text": "Alice bought a blue car.",
             "cites": [{"anchor": self.car.id, "quote": "not the record"}]},
            {"type": "narrative", "text": "最后一句。"},
        ], self.ledger)
        lines = res.text.split("\n")
        self.assertEqual(lines[0], "第一句。")
        self.assertIn("此处省略一句缺少可核验依据的内容", lines[2])
        self.assertEqual(lines[3], "最后一句。")

    def test_redaction_marker_optional(self):
        res = emit_answer([
            {"type": "narrative", "text": "第一句。"},
            {"type": "claim", "text": "Alice bought a blue car.",
             "cites": [{"anchor": self.car.id, "quote": "not the record"}]},
            {"type": "narrative", "text": "最后一句。"},
        ], self.ledger, redact=False)
        self.assertEqual(res.text, "第一句。\n最后一句。")

    def test_consecutive_deletions_collapse_to_one_marker(self):
        bad = {"type": "claim", "text": "Alice bought a blue car.",
               "cites": [{"anchor": self.car.id, "quote": "not the record"}]}
        res = emit_answer([
            {"type": "narrative", "text": "前。"},
            bad, dict(bad),
            {"type": "narrative", "text": "后。"},
        ], self.ledger)
        self.assertEqual(res.text.count("此处省略一句缺少可核验依据的内容"), 1)
        self.assertLess(res.text.count("\n"), 4)


class SpeculationTest(_Base):
    def test_uncited_claim_gets_speculation_prefix(self):
        res = emit_answer([{"type": "claim", "text": "明天可能会下雨。", "cites": []}], self.ledger)
        self.assertTrue(res.text.startswith(SPECULATION_PREFIX), res.text)
        self.assertEqual(res.audit[0]["action"], "speculation")


class AssemblyTest(_Base):
    def test_hedged_template_does_not_double_punctuation(self):
        res = emit_answer([{"type": "claim", "text": self.car.content,
                            "cites": [self._cite(self.car), self._cite(self.weather)]}],
                          self.ledger, policy=CITE_SET)
        self.assertNotIn("。，", res.text, res.text)
        self.assertNotIn(".，", res.text, res.text)

    def test_mixed_segments_order_preserved(self):
        segs = [
            {"type": "narrative", "text": "背景。"},
            {"type": "claim", "text": self.car.content, "cites": [self._cite(self.car)]},
            {"type": "claim", "text": "我在猜。", "cites": []},
        ]
        res = emit_answer(segs, self.ledger)
        lines = res.text.split("\n")
        self.assertEqual(lines[0], "背景。")
        self.assertEqual(lines[1], self.car.content)
        self.assertTrue(lines[2].startswith(SPECULATION_PREFIX))

    def test_audit_contains_anchors_and_reasons(self):
        res = emit_answer([{"type": "claim", "text": self.car.content,
                            "cites": [self._cite(self.car)]}], self.ledger)
        self.assertEqual(res.audit[0]["anchors"], [self.car.id])
        self.assertIn("verdict", res.audit[0])

    def test_deterministic_repeat(self):
        segs = [
            {"type": "narrative", "text": "背景。"},
            {"type": "claim", "text": self.car.content, "cites": [self._cite(self.car)]},
            {"type": "claim", "text": "无引用猜测", "cites": []},
        ]
        first = emit_answer(segs, self.ledger)
        for _ in range(5):
            again = emit_answer(segs, self.ledger)
            self.assertEqual(first.text, again.text)
            self.assertEqual(first.audit, again.audit)


class EmptyInputTest(_Base):
    def test_empty_segments_falls_back(self):
        res = emit_answer([], self.ledger)
        self.assertEqual(res.text, FALLBACK_TEXT)

    def test_only_narrative_keeps_text(self):
        res = emit_answer([{"type": "narrative", "text": "只有叙述。"}], self.ledger)
        self.assertEqual(res.text, "只有叙述。")


class UncitedIsUnverifiedTest(_Base):
    """internal-task：无引用 = 未验证——调用方必须能从 audit 区分四种状态。"""

    def test_audit_state_four_ways(self):
        res = emit_answer([
            {"type": "claim", "text": self.car.content, "cites": [self._cite(self.car)]},
            {"type": "claim", "text": self.car.content,
             "cites": [self._cite(self.car), self._cite(self.weather)]},
            {"type": "claim", "text": "Alice bought a blue car.",
             "cites": [{"anchor": self.car.id, "quote": "not the record"}]},
            {"type": "claim", "text": "明天可能下雨。", "cites": []},
        ], self.ledger, policy=CITE_SET)
        states = [a.get("state") for a in res.audit]
        self.assertEqual(states, ["verified", "degraded", "rejected", "uncited"])

    def test_uncited_text_is_prefixed_not_bare(self):
        res = emit_answer([{"type": "claim", "text": "明天可能下雨。", "cites": []}], self.ledger)
        self.assertTrue(res.text.startswith(SPECULATION_PREFIX))
        self.assertNotEqual(res.text, "明天可能下雨。")


if __name__ == "__main__":
    unittest.main()
