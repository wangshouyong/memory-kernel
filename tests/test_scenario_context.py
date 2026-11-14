# SPDX-License-Identifier: Apache-2.0
"""生产链路场景上下文注入（`memory_kernel/scenario_context.py`）单测。

契约：`docs/SCENARIO-CONTEXT-PRODUCTION.md`（internal-task）。
重点覆盖：侧表只追加语义、显式失败、覆盖率审计、与 `enforce()` 的真实接线
（注入的时间/会话/主体范围确实能驱动四个场景谓词），以及"不改账本哈希"。
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import (  # noqa: E402
    ALLOWED_SCOPES,
    Claim,
    Citation,
    ContextAudit,
    RecordTimeStore,
    ScenarioContext,
    SkeletonPolicy,
    build_scenario_context,
    claim_anchors,
    enforce,
)
from memory_kernel.ledger import AnchorLedger, payload_hash  # noqa: E402

KEY = hashlib.sha256(b"scenario-context-test").digest()
V4 = SkeletonPolicy.v4()


class TestRecordTimeStore(unittest.TestCase):
    def test_note_and_get(self):
        store = RecordTimeStore()
        self.assertEqual(store.note("a_1", 1000), 1000.0)
        self.assertEqual(store.get("a_1"), 1000.0)
        self.assertIsNone(store.get("missing"))
        self.assertEqual(len(store), 1)

    def test_idempotent_same_value(self):
        store = RecordTimeStore()
        store.note("a_1", 1000)
        self.assertEqual(store.note("a_1", 1000.0), 1000.0)

    def test_immutable_conflict(self):
        store = RecordTimeStore()
        store.note("a_1", 1000)
        with self.assertRaises(ValueError):
            store.note("a_1", 1001)

    def test_rejects_bad_timestamps(self):
        store = RecordTimeStore()
        for bad in ("1000", None, True):
            with self.assertRaises(TypeError):
                store.note("a_1", bad)  # type: ignore[arg-type]
        for bad in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                store.note("a_1", bad)

    def test_rejects_bad_anchor(self):
        store = RecordTimeStore()
        for bad in ("", "   ", None, 3):
            with self.assertRaises(ValueError):
                store.note(bad, 1)  # type: ignore[arg-type]

    def test_record_times_filters_unknown(self):
        store = RecordTimeStore({"a_1": 10, "a_2": 20})
        self.assertEqual(store.record_times(["a_1", "zzz", "a_2"]), {"a_1": 10.0, "a_2": 20.0})

    def test_coverage(self):
        store = RecordTimeStore({"a_1": 10})
        self.assertEqual(store.coverage(["a_1", "a_2"]), 0.5)
        self.assertEqual(store.coverage(["a_1", "a_1"]), 1.0)  # 去重
        self.assertIsNone(store.coverage([]))

    def test_from_mapping(self):
        store = RecordTimeStore.from_mapping({"a_1": 5})
        self.assertEqual(store.get("a_1"), 5.0)


class TestBuildScenarioContext(unittest.TestCase):
    def test_fields_and_audit(self):
        store = RecordTimeStore({"a_1": 100})
        ctx, audit = build_scenario_context(anchors=["a_1", "a_2"], store=store,
                                            session_id="s1", now=200, asserted_scope="user_fact")
        self.assertIsInstance(ctx, ScenarioContext)
        self.assertEqual(ctx.context_session_id, "s1")
        self.assertEqual(ctx.context_time, 200.0)
        self.assertEqual(ctx.asserted_scope, "user_fact")
        self.assertEqual(dict(ctx.record_times), {"a_1": 100.0})
        self.assertIsInstance(audit, ContextAudit)
        self.assertEqual(audit.missing, ("a_2",))
        self.assertEqual(audit.timed, ("a_1",))
        self.assertEqual(audit.coverage, 0.5)
        self.assertEqual(audit.session_id, "s1")
        self.assertEqual(audit.as_dict()["n_timed"], 1)

    def test_defaults_disable_predicates(self):
        ctx, audit = build_scenario_context(anchors=["a_1"], store=RecordTimeStore())
        self.assertIsNone(ctx.context_time)
        self.assertIsNone(ctx.context_session_id)
        self.assertEqual(ctx.asserted_scope, "unspecified")
        self.assertEqual(audit.missing, ("a_1",))
        self.assertEqual(audit.coverage, 0.0)

    def test_no_anchors_coverage_none(self):
        _, audit = build_scenario_context(anchors=[], store=RecordTimeStore())
        self.assertIsNone(audit.coverage)
        self.assertEqual(audit.anchors, ())

    def test_validation(self):
        store = RecordTimeStore()
        with self.assertRaises(TypeError):
            build_scenario_context(anchors=[], store={})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_scenario_context(anchors=[], store=store, asserted_scope="vibes")
        with self.assertRaises(ValueError):
            build_scenario_context(anchors=[], store=store, session_id="")
        for bad in (float("nan"), -5, True):
            with self.assertRaises((TypeError, ValueError)):
                build_scenario_context(anchors=[], store=store, now=bad)

    def test_deterministic(self):
        store = RecordTimeStore({"a_1": 1})
        one = build_scenario_context(anchors=["a_1"], store=store, session_id="s",
                                     now=2, asserted_scope="user_fact")
        two = build_scenario_context(anchors=["a_1"], store=store, session_id="s",
                                     now=2, asserted_scope="user_fact")
        self.assertEqual(one[0], two[0])
        self.assertEqual(one[1], two[1])

    def test_allowed_scopes_contract(self):
        self.assertEqual(ALLOWED_SCOPES, ("user_fact", "assistant_claim", "unspecified"))


class TestClaimAnchors(unittest.TestCase):
    def test_primary_then_supporting_dedup(self):
        claim = Claim(
            Citation("a_1", 0, "content"),
            text="content",
            supporting=(Citation("a_2", 0, "content2"), Citation("a_1", 0, "content")),
        )
        self.assertEqual(claim_anchors(claim), ("a_1", "a_2"))

    def test_no_supporting(self):
        self.assertEqual(claim_anchors(Claim(Citation("a_9", 0, "x"), text="x")), ("a_9",))


class TestLedgerHashUnchanged(unittest.TestCase):
    def test_side_table_does_not_touch_ledger(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("我们在 2026年3月5日 上线了新的结算系统。",
                            source_id="s3", episode_id="e3", turn_index=2, speaker="user")
        before = (rec.record_hash, rec.signature, payload_hash(rec))
        store = RecordTimeStore()
        store.note(rec.id, 1700000000)
        build_scenario_context(anchors=[rec.id], store=store, session_id="s3", now=1800000000)
        after = ledger.get(rec.id)
        self.assertEqual((after.record_hash, after.signature, payload_hash(after)), before)


class TestEnforceIntegration(unittest.TestCase):
    """注入的上下文必须真的驱动 enforce 的场景谓词（不是摆设）。"""

    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.work = self.ledger.append("我们在 2026年3月5日 上线了新的结算系统。",
                                       source_id="s3", episode_id="e3", turn_index=2, speaker="user")
        self.hedge = self.ledger.append("这个方案应该可以，可能下周就能上线。",
                                        source_id="s4", episode_id="e4", turn_index=1,
                                        speaker="assistant")

    def _claim(self, rec):
        return Claim(Citation(rec.id, 0, rec.content), text=rec.content)

    def test_timeline_rejects_future_evidence(self):
        store = RecordTimeStore({self.work.id: 2000})
        ctx, audit = build_scenario_context(anchors=[self.work.id], store=store, now=1000)
        self.assertEqual(audit.coverage, 1.0)
        r = enforce(self.ledger, self._claim(self.work), policy=V4, context=ctx)
        self.assertEqual(r.verdict, "rejected")
        self.assertIn("scenario_timeline", [p.name for p in r.failed if not p.passed])

    def test_timeline_passes_with_earlier_evidence(self):
        store = RecordTimeStore({self.work.id: 500})
        ctx, _ = build_scenario_context(anchors=[self.work.id], store=store, now=1000)
        r = enforce(self.ledger, self._claim(self.work), policy=V4, context=ctx)
        self.assertEqual(r.verdict, "verified")

    def test_missing_timestamp_degrades_to_not_judgeable(self):
        # 没有时点 → 不放进 record_times → 时间谓词"不可判"而不是误杀
        ctx, audit = build_scenario_context(anchors=[self.work.id], store=RecordTimeStore(),
                                            now=1000)
        self.assertEqual(audit.coverage, 0.0)
        self.assertEqual(dict(ctx.record_times), {})
        r = enforce(self.ledger, self._claim(self.work), policy=V4, context=ctx)
        self.assertEqual(r.verdict, "verified")

    def test_speaker_scope_fires_when_caller_marks_user_fact(self):
        store = RecordTimeStore({self.hedge.id: 100})
        ctx, _ = build_scenario_context(anchors=[self.hedge.id], store=store,
                                        asserted_scope="user_fact")
        r = enforce(self.ledger, self._claim(self.hedge), policy=V4, context=ctx)
        self.assertIn("scenario_speaker_scope", [p.name for p in r.failed if not p.passed])

    def test_speaker_scope_not_guessed_by_default(self):
        ctx, _ = build_scenario_context(anchors=[self.hedge.id], store=RecordTimeStore())
        self.assertEqual(ctx.asserted_scope, "unspecified")
        r = enforce(self.ledger, self._claim(self.hedge), policy=V4, context=ctx)
        self.assertNotIn("scenario_speaker_scope", [p.name for p in r.failed if not p.passed])

    def test_cross_session_uses_injected_session(self):
        claim = Claim(Citation(self.work.id, 0, self.work.content),
                      text="上次说的那个结算系统已经上线了。")
        ctx_ok, _ = build_scenario_context(anchors=[self.work.id], store=RecordTimeStore(),
                                           session_id="s3")
        r_ok = enforce(self.ledger, claim, policy=V4, context=ctx_ok)
        self.assertNotIn("scenario_cross_session", [p.name for p in r_ok.failed if not p.passed])
        ctx_other, _ = build_scenario_context(anchors=[self.work.id], store=RecordTimeStore(),
                                              session_id="other")
        r_other = enforce(self.ledger, claim, policy=V4, context=ctx_other)
        self.assertIn("scenario_cross_session", [p.name for p in r_other.failed if not p.passed])
        self.assertEqual(r_other.verdict, "degraded")  # soft


if __name__ == "__main__":
    unittest.main()
