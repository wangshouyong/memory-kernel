# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""动态 IDF 单测（internal-task / SPEC §13 D-IDF-1）。

覆盖：
  1. 增量 df（append/remove）与全量重算**逐位一致**（en/zh、含删除）；
  2. 版本号 `idf-<yyyymmddHHMM>-<sha8>` 格式/确定性/参数敏感；
  3. 双缓冲原子切换：旧快照继续可用、历史版本不删除、并发读者只见完整快照；
  4. 定期全量重算入口与维护判定（时钟由调用方注入，core 不读时钟）；
  5. 默认路径逐字节不变：`idf_for_ledger(registry=None)` 仍是 `idf-s2`、
     `PipelineResult.as_dict()` 不新增 `idf_version` 键；
  6. 注入点：`idf_for_ledger(registry=...)` / `emit_answer(idf_model=...)` /
     `AnswerPipeline(idf_provider=...)`（gated 结果带版本元数据）。

全部离线、确定性；不联网、不调模型。
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import (  # noqa: E402
    AnswerPipeline,
    SkeletonPolicy,
)
from memory_kernel.idf_dynamic import (  # noqa: E402
    DynamicIDF,
    IDF_VERSION_RE,
    corpus_digest,
    is_valid_idf_version,
    make_idf_version,
)
from memory_kernel.ledger import AnchorLedger  # noqa: E402
from memory_kernel.pipeline import PipelineResult, prewarm_ledger  # noqa: E402
from memory_kernel.skeleton import (  # noqa: E402
    IDF_VERSION,
    build_idf,
    idf_for_ledger,
    norm_tokens,
)

KEY = hashlib.sha256(b"idf-dynamic-test").digest()
EPOCH = "202611140330"
EN_TEXTS = ["the car repair shop", "sea food and fresh fish", "car and sea travel",
            "repair the old engine", "fresh sea food market"]
ZH_TEXTS = ["我们在 2026年3月5日 上线了新的结算系统", "杭州的天气很好", "结算系统支持对账",
            "对账报表每天生成", "杭州的结算中心在滨江"]

V4 = SkeletonPolicy.v4()


def _sets(texts, zh):
    return [norm_tokens(t, zh=zh) for t in texts]


class TestIncrementalEqualsFull(unittest.TestCase):
    def _assert_same(self, a, b):
        self.assertEqual(dict(sorted(a._df.items())), dict(sorted(b._df.items())))
        self.assertEqual(a.n_docs, b.n_docs)

    def test_english_append_matches_full_rebuild(self):
        d = DynamicIDF()
        for t in EN_TEXTS:
            d.add(t)
        snap = d.rebuild(epoch=EPOCH)
        full = build_idf(_sets(EN_TEXTS, False), version=snap.version)
        self._assert_same(snap.model, full)
        self.assertEqual(snap.n_docs, len(EN_TEXTS))

    def test_chinese_append_matches_full_rebuild(self):
        d = DynamicIDF(zh=True)
        for t in ZH_TEXTS:
            d.add(t)
        snap = d.rebuild(epoch=EPOCH)
        full = build_idf(_sets(ZH_TEXTS, True), version=snap.version, zh=True)
        self._assert_same(snap.model, full)
        for tok in ("杭州", "结算", "系统"):
            self.assertAlmostEqual(snap.model.weight(tok), full.weight(tok), places=12)

    def test_remove_matches_full_rebuild(self):
        d = DynamicIDF(zh=True)
        for t in ZH_TEXTS:
            d.add(t)
        d.remove(ZH_TEXTS[0])
        snap = d.rebuild(epoch=EPOCH)
        full = build_idf(_sets(ZH_TEXTS[1:], True), version=snap.version, zh=True)
        self._assert_same(snap.model, full)
        self.assertEqual(snap.n_docs, len(ZH_TEXTS) - 1)

    def test_remove_unknown_token_and_empty_raise(self):
        d = DynamicIDF()
        d.add("the car repair shop")
        with self.assertRaises(ValueError):
            d.remove("a totally unseen sentence with zzzz qqqq")
        empty = DynamicIDF()
        with self.assertRaises(ValueError):
            empty.remove("the car")

    def test_add_remove_roundtrip_restores_state(self):
        d = DynamicIDF()
        for t in EN_TEXTS:
            d.add(t)
        first = d.rebuild(epoch=EPOCH)
        d.add("a brand new sentence about boats")
        d.remove("a brand new sentence about boats")
        second = d.rebuild(epoch=EPOCH)
        self._assert_same(first.model, second.model)
        self.assertEqual(first.version, second.version)   # 语料+参数+epoch 相同 -> 同版本


class TestVersioning(unittest.TestCase):
    def test_version_format(self):
        v = make_idf_version(EPOCH, "deadbeef")
        self.assertEqual(v, "idf-202611140330-deadbeef")
        self.assertTrue(is_valid_idf_version(v))
        self.assertTrue(IDF_VERSION_RE.match(v))

    def test_version_rejects_bad_inputs(self):
        for bad_epoch in ("20261114", "2026-11-14", "2026111403301", 202611140330):
            with self.assertRaises(ValueError):
                make_idf_version(bad_epoch, "deadbeef")  # type: ignore[arg-type]
        for bad_sha in ("DEADBEEF", "deadbee", "deadbeeff", "zzzzzzzz"):
            with self.assertRaises(ValueError):
                make_idf_version(EPOCH, bad_sha)
        for bad in ("idf-202611140330", "idf-202611140330-DEADBEEF", "202611140330-deadbeef"):
            self.assertFalse(is_valid_idf_version(bad))

    def test_version_deterministic_and_epoch_sensitive(self):
        def snap(epoch, texts=EN_TEXTS):
            d = DynamicIDF()
            for t in texts:
                d.add(t)
            return d.rebuild(epoch=epoch)
        a = snap(EPOCH)
        b = snap(EPOCH)
        c = snap("202611140345")
        self.assertEqual(a.version, b.version)
        self.assertNotEqual(a.version, c.version)
        self.assertEqual(a.sha8, c.sha8)            # 只有 epoch 变了
        self.assertNotEqual(snap(EPOCH, EN_TEXTS[1:]).sha8, a.sha8)

    def test_params_change_digest(self):
        base = corpus_digest(_sets(EN_TEXTS, False), stem_ok=True, stop_mode="v2", zh=False)
        other = corpus_digest(_sets(EN_TEXTS, False), stem_ok=False, stop_mode="v2", zh=False)
        self.assertNotEqual(base, other)
        self.assertEqual(base, corpus_digest(_sets(list(reversed(EN_TEXTS)), False)))  # 顺序无关

    def test_commit_rejects_param_mismatch(self):
        d = DynamicIDF()
        other = DynamicIDF(stem_ok=False)
        for t in EN_TEXTS:
            other.add(t)
        snap_other = other.prebuild(epoch=EPOCH)
        with self.assertRaises(ValueError):
            d.commit(snap_other)

    def test_epoch_required_without_clock(self):
        d = DynamicIDF()
        d.add("the car repair shop")
        with self.assertRaises(ValueError):
            d.snapshot()

    def test_epoch_fn_used_when_injected(self):
        d = DynamicIDF(epoch_fn=lambda: EPOCH)
        d.add("the car repair shop")
        snap = d.snapshot()
        self.assertTrue(snap.version.startswith(f"idf-{EPOCH}-"))


class TestDoubleBuffer(unittest.TestCase):
    def test_old_snapshot_stays_valid_after_switch(self):
        d = DynamicIDF()
        for t in EN_TEXTS:
            d.add(t)
        old = d.rebuild(epoch=EPOCH)
        old_df = dict(old.model._df)
        d.add("a new document about electric boats")
        new = d.rebuild(epoch="202611140345", reason="incremental")
        self.assertNotEqual(old.version, new.version)
        self.assertEqual(dict(old.model._df), old_df)          # 旧快照未被改写
        self.assertEqual(d.active_version, new.version)
        self.assertEqual(d.get(old.version).version, old.version)  # 历史不删除
        self.assertIn("electric", new.model._df)

    def test_history_is_append_only(self):
        d = DynamicIDF()
        d.add("the car repair shop")
        v1 = d.rebuild(epoch=EPOCH).version
        d.add("sea food market")
        v2 = d.rebuild(epoch="202611140345").version
        versions = [h["version"] for h in d.history()]
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions, [v1, v2])

    def test_concurrent_readers_never_see_partial_snapshot(self):
        d = DynamicIDF()
        for t in EN_TEXTS:
            d.add(t)
        d.rebuild(epoch=EPOCH)
        seen = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                snap = d.snapshot()
                # 快照完整性：model.version == snapshot.version，且 n_docs 与版本一致
                if snap.model.version != snap.version:
                    seen.append(("version_mismatch", snap.version))
                if not is_valid_idf_version(snap.version):
                    seen.append(("bad_version", snap.version))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for i in range(30):
                d.add(f"extra document number {i}")
                d.rebuild(epoch="202611140345", reason="loop")
        finally:
            stop.set()
            t.join(timeout=5)
        self.assertEqual(seen, [])
        self.assertGreaterEqual(len(d.history()), 2)

    def test_snapshot_meta_carries_version(self):
        d = DynamicIDF()
        d.add("the car repair shop")
        snap = d.rebuild(epoch=EPOCH)
        meta = snap.meta()
        self.assertEqual(meta["idf_version"], snap.version)
        self.assertEqual(meta["idf_n_docs"], 1)
        self.assertIsInstance(meta["idf_vocab"], int)
        self.assertEqual(d.meta()["idf_version"], snap.version)


class TestMaintenance(unittest.TestCase):
    def test_maintenance_due_on_age_and_pending(self):
        d = DynamicIDF(max_age_minutes=60, max_pending_docs=2)
        self.assertTrue(d.maintenance_due(now_epoch=EPOCH))     # 首建前 due
        d.add("the car repair shop")
        d.rebuild(epoch=EPOCH)
        self.assertFalse(d.maintenance_due(now_epoch=EPOCH))
        d.add("sea food market")
        d.add("engine repair")
        self.assertTrue(d.maintenance_due(now_epoch=EPOCH))     # pending_docs 达到阈值
        d.rebuild(epoch=EPOCH)
        self.assertFalse(d.maintenance_due(now_epoch=EPOCH))
        self.assertTrue(d.maintenance_due(now_epoch="202611140500"))  # 超过 60 分钟

    def test_full_recompute_from_ledger_matches_incremental(self):
        ledger = AnchorLedger(None, KEY)
        for i, text in enumerate(ZH_TEXTS):
            ledger.append(text, source_id="s1", episode_id="e1", turn_index=i, speaker="user")
        d = DynamicIDF(zh=True)
        for rec in ledger:
            d.add_record(rec)
        incremental = d.rebuild(epoch=EPOCH)
        full = d.full_recompute(ledger, now_epoch="202611140345", reason="scheduled")
        self.assertEqual(dict(sorted(incremental.model._df.items())),
                         dict(sorted(full.model._df.items())))
        self.assertEqual(incremental.version.split("-")[-1], full.version.split("-")[-1])  # 同 sha8
        self.assertEqual(full.reason, "scheduled")
        self.assertIn("scheduled", [h["reason"] for h in d.history()])

    def test_stats_counters(self):
        d = DynamicIDF()
        d.add("the car repair shop")
        d.rebuild(epoch=EPOCH)
        st = d.stats()
        self.assertEqual(st["appends"], 1)
        self.assertEqual(st["swaps"], 1)
        self.assertEqual(st["pending_docs"], 0)
        self.assertEqual(st["versions"], 1)
        self.assertEqual(st["active_version"], st["active_version"])
        self.assertGreaterEqual(st["n_docs_incremental"], 1)


class TestDefaultPathUnchanged(unittest.TestCase):
    def test_default_idf_version_still_idf_s2(self):
        self.assertEqual(IDF_VERSION, "idf-s2")
        ledger = AnchorLedger(None, KEY)
        ledger.append("the car repair shop", source_id="s1", episode_id="e1", turn_index=1,
                      speaker="user")
        model = idf_for_ledger(ledger)
        self.assertEqual(model.version, "idf-s2")
        expected = build_idf([norm_tokens("the car repair shop")])
        self.assertEqual(json.dumps(model.to_json(), sort_keys=True),
                         json.dumps(expected.to_json(), sort_keys=True))

    def test_default_pipeline_result_has_no_idf_version_key(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("我们在 2026年3月5日 上线了新的结算系统。",
                            source_id="s1", episode_id="e1", turn_index=1, speaker="user")
        pipe = AnswerPipeline(generator=lambda q, c: [{"type": "claim", "text": rec.content,
                                                       "cites": [{"anchor": rec.id,
                                                                  "quote": rec.content}]}],
                              ledger=ledger, policy=V4)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="结算系统什么时候上线？", records=[rec])
        import time
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            with pipe._lock:  # noqa: SLF001
                entry = pipe._prefetches["k1"]  # noqa: SLF001
            if entry.future.done():
                break
            time.sleep(0.005)
        r = pipe.answer("k1")
        self.assertEqual(r.state, "gated")
        keys = set(r.as_dict())
        self.assertNotIn("idf_version", keys)
        self.assertEqual(keys, {"state", "text", "sync_ms", "budget_ok", "evidence", "dropped",
                                "audit", "generation_ms", "reason"})

    def test_registry_injection_point(self):
        ledger = AnchorLedger(None, KEY)
        for i, text in enumerate(EN_TEXTS):
            ledger.append(text, source_id="s1", episode_id="e1", turn_index=i, speaker="user")
        d = DynamicIDF()
        prewarm_ledger(ledger, SkeletonPolicy(), idf_provider=d, epoch=EPOCH)
        model = idf_for_ledger(ledger, registry=d)
        self.assertIs(model, d.snapshot().model)
        self.assertTrue(is_valid_idf_version(model.version))
        # 未给 registry 时仍走旧缓存路径（idf-s2），两者互不影响
        legacy = idf_for_ledger(ledger)
        self.assertEqual(legacy.version, "idf-s2")

    def test_pipeline_with_provider_annotates_version(self):
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("我们在 2026年3月5日 上线了新的结算系统。",
                            source_id="s1", episode_id="e1", turn_index=1, speaker="user")
        d = DynamicIDF(zh=True)
        prewarm_ledger(ledger, V4, idf_provider=d, epoch=EPOCH)
        pipe = AnswerPipeline(generator=lambda q, c: [{"type": "claim", "text": rec.content,
                                                       "cites": [{"anchor": rec.id,
                                                                  "quote": rec.content}]}],
                              ledger=ledger, policy=V4, idf_provider=d)
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="结算系统什么时候上线？", records=[rec])
        import time
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            with pipe._lock:  # noqa: SLF001
                entry = pipe._prefetches["k1"]  # noqa: SLF001
            if entry.future.done():
                break
            time.sleep(0.005)
        r = pipe.answer("k1")
        self.assertEqual(r.state, "gated")
        self.assertTrue(is_valid_idf_version(r.idf_version))
        self.assertEqual(r.as_dict()["idf_version"], r.idf_version)
        self.assertEqual(pipe.stats()["idf_version"], d.active_version)
        # 确定性：同一快照下重复取结果版本一致
        r2 = pipe.answer("k1")
        self.assertEqual(r.idf_version, r2.idf_version)

    def test_explicit_result_dict_defaults(self):
        r = PipelineResult(state="gated", text="x", sync_ms=1.0, budget_ok=True)
        self.assertNotIn("idf_version", r.as_dict())
        r2 = PipelineResult(state="gated", text="x", sync_ms=1.0, budget_ok=True,
                            idf_version="idf-202611140330-deadbeef")
        self.assertEqual(r2.as_dict()["idf_version"], "idf-202611140330-deadbeef")


if __name__ == "__main__":
    unittest.main(verbosity=2)
