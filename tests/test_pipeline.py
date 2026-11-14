# SPDX-License-Identifier: Apache-2.0
"""异步生成 + 门控管线（`memory_kernel/pipeline.py`）单测。

契约：`docs/PIPELINE-LATENCY-CONTRACT.md`（internal-task）。
全部离线：generator 是注入的 fake，不联网、不调模型；门控路径确定性。
"""

from __future__ import annotations

import ast
import hashlib
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import (  # noqa: E402
    FAILED_TEXT,
    NO_PREFETCH_TEXT,
    UNREFINED_TEXT,
    AnswerPipeline,
    PipelineBudget,
    SkeletonPolicy,
)
from memory_kernel.ledger import AnchorLedger  # noqa: E402

KEY = hashlib.sha256(b"pipeline-test").digest()
V4 = SkeletonPolicy.v4()


class _Clock:
    """可注入时钟：依次返回给定序列，耗尽后返回最后一个值。"""

    def __init__(self, values: list[float]):
        self._values = list(values)
        self._i = 0

    def __call__(self) -> float:
        if self._i < len(self._values):
            v = self._values[self._i]
            self._i += 1
            return v
        return self._values[-1]


def _wait_done(pipe: AnswerPipeline, key: str, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        with pipe._lock:  # noqa: SLF001  (测试专用：观察预取是否完成)
            entry = pipe._prefetches.get(key)  # noqa: SLF001
        if entry is None or entry.future.done():
            return
        time.sleep(0.005)
    raise AssertionError("generation did not finish in time")


class _Base(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.rec = self.ledger.append("我们在 2026年3月5日 上线了新的结算系统。",
                                      source_id="s1", episode_id="e1", turn_index=1,
                                      speaker="user")

    def _segments_verified(self):
        return [{"type": "claim", "text": self.rec.content,
                 "cites": [{"anchor": self.rec.id, "quote": self.rec.content}]}]


class TestTwoPhase(_Base):
    def test_unrefined_while_generation_pending(self):
        gate = threading.Event()

        def slow_generator(question, prompt_ctx):
            gate.wait(timeout=5)
            return self._segments_verified()

        pipe = AnswerPipeline(generator=slow_generator, ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="结算系统什么时候上线？", records=[self.rec])
        t0 = time.perf_counter()
        r = pipe.answer("k1")
        elapsed = time.perf_counter() - t0
        self.assertEqual(r.state, "unrefined")
        self.assertEqual(r.text, UNREFINED_TEXT)
        self.assertEqual(r.audit, ())            # 未就绪：不返回任何模型内容
        self.assertLess(elapsed, 0.5)            # 同步路径不阻塞等待生成
        self.assertTrue(r.budget_ok)
        gate.set()
        _wait_done(pipe, "k1")
        r2 = pipe.answer("k1")
        self.assertEqual(r2.state, "gated")
        self.assertIn("结算系统", r2.text)
        self.assertEqual([a["state"] for a in r2.audit], ["verified"])

    def test_gate_rejects_bad_quote_not_passthrough(self):
        def bad_generator(question, prompt_ctx):
            return [{"type": "claim", "text": "结算系统已经上线三个月了。",
                     "cites": [{"anchor": self.rec.id, "quote": "这句话不在证据里"}]}]

        pipe = AnswerPipeline(generator=bad_generator, ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec])
        _wait_done(pipe, "k1")
        r = pipe.answer("k1")
        self.assertEqual(r.state, "gated")
        self.assertNotIn("三个月", r.text)        # 门控删句，不透传模型原文
        self.assertEqual([a["state"] for a in r.audit], ["rejected"])

    def test_generation_failure_is_explicit(self):
        def broken(question, prompt_ctx):
            raise RuntimeError("gateway down")

        pipe = AnswerPipeline(generator=broken, ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec])
        _wait_done(pipe, "k1")
        r = pipe.answer("k1")
        self.assertEqual(r.state, "generation_failed")
        self.assertEqual(r.text, FAILED_TEXT)
        self.assertIn("gateway down", r.reason or "")

    def test_missing_prefetch(self):
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        r = pipe.answer("nope")
        self.assertEqual(r.state, "no_prefetch")
        self.assertEqual(r.text, NO_PREFETCH_TEXT)

    def test_prefetch_validates_key(self):
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        with self.assertRaises(ValueError):
            pipe.prefetch("", question="q", records=[])


class TestBudgetAndTtl(_Base):
    def test_budget_breach_flagged(self):
        # clock 序列：prefetch(prune,submitted)=0,0 → answer(t0,prune)=0,0 → elapsed=0.06s
        clock = _Clock([0.0, 0.0, 0.0, 0.0, 0.06, 0.06])
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4,
                              budget=PipelineBudget(sync_budget_ms=50), clock=clock)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec])
        r = pipe.answer("k1")
        self.assertFalse(r.budget_ok)
        self.assertAlmostEqual(r.sync_ms, 60.0, places=3)
        self.assertEqual(pipe.stats()["budget_breach"], 1)

    def test_ttl_eviction(self):
        # prefetch(prune=0, submitted=0) → answer t0=1000, prune=1000 → 超过 ttl=10 被清
        clock = _Clock([0.0, 0.0, 1000.0, 1000.0])
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4,
                              budget=PipelineBudget(prefetch_ttl_s=10), clock=clock)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec])
        r = pipe.answer("k1")
        self.assertEqual(r.state, "no_prefetch")
        self.assertEqual(pipe.stats()["evicted"], 1)

    def test_budget_validation(self):
        for kwargs in ({"sync_budget_ms": 0}, {"prefetch_ttl_s": 0}, {"max_workers": 0}):
            with self.assertRaises(ValueError):
                PipelineBudget(**kwargs)
        with self.assertRaises(TypeError):
            AnswerPipeline(generator="nope", ledger=self.ledger)


class TestAuditAndStats(_Base):
    def test_evidence_and_dropped_audit(self):
        future_rec = self.ledger.append("这条是未来记录", source_id="s1",
                                        episode_id="e1", turn_index=2, speaker="user")
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec, future_rec],
                      now=1000.0, record_times={future_rec.id: 2000.0})
        r = pipe.answer("k1")
        self.assertEqual(r.evidence, (self.rec.id,))
        self.assertIn({"anchor": future_rec.id, "reason": "future",
                       "detail": "time>1000.0"}, list(r.dropped))

    def test_stats_percentiles(self):
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        for i in range(3):
            pipe.prefetch(f"k{i}", question="q", records=[self.rec])
            _wait_done(pipe, f"k{i}")
            pipe.answer(f"k{i}")
        stats = pipe.stats()
        self.assertEqual(stats["prefetch"], 3)
        self.assertEqual(stats["gated"], 3)
        self.assertIsNotNone(stats["sync_p50_ms"])
        self.assertIsNotNone(stats["sync_p95_ms"])

    def test_forget(self):
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec])
        self.assertTrue(pipe.forget("k1"))
        self.assertFalse(pipe.forget("k1"))
        self.assertEqual(pipe.answer("k1").state, "no_prefetch")

    def test_deterministic_gate(self):
        def generator(q, c):
            return self._segments_verified()

        def run_once():
            pipe = AnswerPipeline(generator=generator, ledger=self.ledger, policy=V4)
            try:
                pipe.prefetch("k", question="q", records=[self.rec])
                _wait_done(pipe, "k")
                return pipe.answer("k")
            finally:
                pipe.close()

        a, b = run_once(), run_once()
        self.assertEqual((a.state, a.text, a.audit), (b.state, b.text, b.audit))


class TestPrewarm(_Base):
    def test_prewarm_is_idempotent_and_does_not_change_verdict(self):
        from memory_kernel.pipeline import prewarm_ledger

        claim = self._segments_verified()
        pipe = AnswerPipeline(generator=lambda q, c: claim, ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[self.rec])
        _wait_done(pipe, "k1")
        before = pipe.answer("k1")
        prewarm_ledger(self.ledger, V4)
        prewarm_ledger(self.ledger, V4)   # 幂等
        pipe.prefetch("k2", question="q", records=[self.rec])
        _wait_done(pipe, "k2")
        after = pipe.answer("k2")
        self.assertEqual((before.state, before.text), (after.state, after.text))

    def test_prewarm_method_uses_pipeline_policy(self):
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prewarm()   # 不抛错；后续门控仍正常
        pipe.prefetch("k1", question="q", records=[self.rec])
        _wait_done(pipe, "k1")
        self.assertEqual(pipe.answer("k1").state, "gated")


class TestNoModelInKernel(unittest.TestCase):
    def test_pipeline_module_has_no_network_imports(self):
        src = (ROOT / "memory_kernel" / "pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = {"urllib", "http", "socket", "requests", "openai", "anthropic", "aiohttp"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], banned)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], banned)


if __name__ == "__main__":
    unittest.main()
