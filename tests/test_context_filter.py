# SPDX-License-Identifier: Apache-2.0
"""阶段一 Prompt 前置过滤单测（契约 docs/EMISSION-CONTRACT.md §1 / §1.1.1）。"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel import AnchorLedger  # noqa: E402
from memory_kernel.context_filter import (  # noqa: E402
    DEFAULT_DROP_TAIL,
    build_prompt_context,
    detect_superseded_anchors,
)

KEY = b"0123456789abcdef0123456789abcdef"


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AnchorLedger(None, KEY)

        def add(content, speaker="user", turn=0, source="s1", episode="e1"):
            return self.ledger.append(content, source_id=source, episode_id=episode,
                                      turn_index=turn, speaker=speaker)

        self.a = add("我把车卖了，换了一辆电动车。", turn=0)
        self.c = add("今天天气不错，适合出去玩。", turn=1)
        self.other = add("我上周去杭州出差了。", turn=2, source="s2", episode="e2")

    def test_contaminated_source_is_dropped_and_audited(self):
        ctx = build_prompt_context([self.a, self.other], {self.a.id: 1.0, self.other.id: 1.0},
                                   contaminated_sources={"s2"})
        self.assertEqual([d.reason for d in ctx.dropped], ["contaminated"])
        self.assertEqual([e.record.id for e in ctx.records], [self.a.id])

    def test_future_record_is_dropped(self):
        ctx = build_prompt_context(
            [self.a, self.other],
            relevance={self.a.id: 2.0, self.other.id: 1.0},
            now=100.0,
            record_times={self.a.id: 50.0, self.other.id: 200.0},
        )
        self.assertEqual([d.anchor for d in ctx.dropped], [self.other.id])
        self.assertEqual(ctx.dropped[0].reason, "future")

    def test_superseded_detection_uses_same_fact_rule(self):
        later = self.ledger.append("其实我没有把车卖掉，只是换了电池。", source_id="s1",
                                   episode_id="e1", turn_index=3, speaker="user")
        superseded = detect_superseded_anchors(self.ledger)
        self.assertIn(self.a.id, superseded)
        self.assertNotIn(self.c.id, superseded)
        self.assertNotIn(later.id, superseded)

    def test_low_score_tail_is_dropped_by_fraction_not_by_zero(self):
        recs = [self.ledger.append(f"记录{i}", source_id="s3", episode_id="e3", turn_index=i)
                for i in range(8)]
        relevance = {r.id: float(8 - i) for i, r in enumerate(recs)}
        ctx = build_prompt_context(recs, relevance, drop_tail=DEFAULT_DROP_TAIL)
        low = [d for d in ctx.dropped if d.reason == "low_relevance"]
        self.assertEqual(len(low), int(8 * DEFAULT_DROP_TAIL))  # 2
        kept = {e.record.id for e in ctx.records}
        self.assertEqual(len(kept), 6)

    def test_zero_scores_are_not_dropped_when_drop_tail_zero(self):
        recs = [self.ledger.append(f"零分记录{i}", source_id="s4", episode_id="e4", turn_index=i)
                for i in range(5)]
        ctx = build_prompt_context(recs, {r.id: 0.0 for r in recs}, drop_tail=0.0)
        self.assertEqual(len(ctx.records), 5)
        self.assertEqual(ctx.dropped, [])

    def test_gray_zone_is_kept_but_marked_low_confidence(self):
        recs = [self.ledger.append(f"灰区{i}", source_id="s5", episode_id="e5", turn_index=i)
                for i in range(10)]
        relevance = {r.id: float(10 - i) for i, r in enumerate(recs)}
        ctx = build_prompt_context(recs, relevance, drop_tail=0.2, low_conf_band=0.2)
        # 10 条：删 2；灰区 2 条保留并标注
        self.assertEqual(len(ctx.records), 8)
        self.assertEqual(sum(1 for e in ctx.records if e.low_confidence), 2)
        self.assertTrue(all(e.relevance > 0 for e in ctx.records))

    def test_token_budget_truncates_lowest_relevance_first(self):
        recs = [self.ledger.append("很长" * 50, source_id="s6", episode_id="e6", turn_index=i)
                for i in range(6)]
        relevance = {r.id: float(i) for i, r in enumerate(recs)}
        ctx = build_prompt_context(recs, relevance, drop_tail=0.0, token_budget=120,
                                   chars_per_token=1.0)
        self.assertLessEqual(ctx.tokens_estimate, 120)
        self.assertTrue(any(d.reason == "budget" for d in ctx.dropped))

    def test_audit_is_complete(self):
        recs = [self.a, self.c, self.other]
        ctx = build_prompt_context(recs, {self.a.id: 3.0, self.c.id: 2.0, self.other.id: 1.0},
                                   drop_tail=0.34)
        kept = {e.record.id for e in ctx.records}
        dropped = {d.anchor for d in ctx.dropped}
        self.assertEqual(kept | dropped, {r.id for r in recs})
        self.assertEqual(kept & dropped, set())

    def test_deterministic_repeat(self):
        recs = [self.ledger.append(f"d{i}", source_id="s7", episode_id="e7", turn_index=i)
                for i in range(6)]
        relevance = {r.id: float(i % 3) for i, r in enumerate(recs)}
        first = build_prompt_context(recs, relevance, drop_tail=0.34)
        for _ in range(5):
            again = build_prompt_context(recs, relevance, drop_tail=0.34)
            self.assertEqual([e.record.id for e in first.records], [e.record.id for e in again.records])
            self.assertEqual([(d.anchor, d.reason) for d in first.dropped],
                             [(d.anchor, d.reason) for d in again.dropped])


if __name__ == "__main__":
    unittest.main()
