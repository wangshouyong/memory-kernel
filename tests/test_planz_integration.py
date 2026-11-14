# SPDX-License-Identifier: Apache-2.0
"""Plan Z C1–C5 集成测试（契约：`docs/PLANZ-C1-C6-DECISIONS.md` / SPEC §14）。

覆盖：
  C1  AnchorRecord 完整性三元组（原子性 / payload 门控 / roundtrip / 篡改验签失败 / 旧记录逐字节不变）
  C2  第 5 谓词 scenario_evidence_incomplete（hard / pass / None 三态）
  C3  销毁 / 缺失证据的可见降级（evidence_destroyed / evidence_missing + emit 措辞）
  C4  注入 resolver（四类 reason → degrade；core 无 I/O/时钟）
  C5  slot 契约（纯 dataclass 确定性 + 未接入 enforce + core isolation）

C6 不在本批代码范围（「已送达」判定在代理层，core 只存字段）——这里只锁字段语义。
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel import (  # noqa: E402
    EVIDENCE_DEGRADE_REASON_OF,
    EVIDENCE_INCOMPLETE,
    EVIDENCE_RESOLUTION_PREDICATE,
    EVIDENCE_UNAVAILABLE_REASONS,
    SCENARIO_PREDICATES,
    AnchorLedger,
    AnchorRecord,
    ChainBroken,
    Claim,
    Citation,
    EvidenceUnavailable,
    RecordVerificationError,
    SkeletonPolicy,
    SlotEvent,
    StateSlot,
    check_scenario,
    enforce,
    make_slot_event,
    sha256_hex,
    slot_event_hash,
    state_slot_payload_hash,
)
from memory_kernel.emit import emit_answer  # noqa: E402
from memory_kernel.ledger import payload_hash, record_to_json  # noqa: E402
from memory_kernel.pipeline import AnswerPipeline  # noqa: E402
from memory_kernel.scenario import COMPLETENESS_UNKNOWN, ScenarioContext  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"
SCEN = SkeletonPolicy(scenario=True)

# A1 之前用同一 fixture 实测得到的金标（与 tests/test_ledger_attribution.py 一致）。
# C1 之后**默认记录**必须仍然是这三个值：任何字节级变化都会在这里失败。
GOLDEN_RECORD_HASH = "f3aff090068f9469f3bb40f6d9990de1b149b293de7a558c3c6d44ace5dce69f"
GOLDEN_SIGNATURE = "7cbb19487b786fa67a26a8a6790c2166e080eeab0fb6f4a97084b1b0fe878ada"
GOLDEN_JSON = (
    '{"char_span":[0,16],"content":"alpha beta gamma",'
    '"content_hash":"64989ccbf3efa9c84e2afe7cee9bc5828bf0fcb91e44f8c1e591638a2c2e90e3",'
    '"episode_id":"e1","id":"a_000000","modality":"text",'
    '"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000",'
    '"record_hash":"f3aff090068f9469f3bb40f6d9990de1b149b293de7a558c3c6d44ace5dce69f",'
    '"seq":0,"signature":"7cbb19487b786fa67a26a8a6790c2166e080eeab0fb6f4a97084b1b0fe878ada",'
    '"source_id":"s1","speaker":"Alice","turn_index":2}'
)


def _json_of(rec: AnchorRecord) -> str:
    return json.dumps(record_to_json(rec), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _append(ledger, content, **kw):
    return ledger.append(content, source_id=kw.pop("source_id", "s1"),
                         episode_id=kw.pop("episode_id", "e1"), **kw)


def _triple(is_complete, finish_reason="stop", text="delivered"):
    return {
        "is_complete": is_complete,
        "finish_reason": finish_reason,
        "delivered_sha256": sha256_hex(text.encode("utf-8")),
    }


def _excerpt(rec, **kw):
    return Claim(Citation(rec.id, 0, rec.content), text=rec.content, **kw)


# --------------------------------------------------------------------------- #
# C1：记录完整性三元组
# --------------------------------------------------------------------------- #
class C1SchemaTest(unittest.TestCase):
    def test_default_record_is_byte_identical_to_pre_c1_golden(self):
        rec = _append(AnchorLedger(None, KEY), "alpha beta gamma", turn_index=2, speaker="Alice")
        self.assertIsNone(rec.is_complete)
        self.assertIsNone(rec.finish_reason)
        self.assertIsNone(rec.delivered_sha256)
        self.assertFalse(rec.has_completeness)
        self.assertEqual(rec.record_hash, GOLDEN_RECORD_HASH)
        self.assertEqual(rec.signature, GOLDEN_SIGNATURE)
        self.assertEqual(_json_of(rec), GOLDEN_JSON)

    def test_default_json_has_no_c1_keys(self):
        rec = _append(AnchorLedger(None, KEY), "alpha beta gamma", turn_index=2, speaker="Alice")
        payload = record_to_json(rec)
        for key in ("is_complete", "finish_reason", "delivered_sha256"):
            self.assertNotIn(key, payload)

    def test_triple_roundtrip_persists_and_verifies(self):
        triple = _triple(True, "stop", "完整回答")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = AnchorLedger(path, KEY)
            rec = led.append("assistant 的完整回答", source_id="s1", episode_id="e1",
                             turn_index=0, speaker="assistant", **triple)
            again = AnchorLedger(path, KEY)          # 重新加载 = 校验整条链
            got = again.get(rec.id)
            self.assertEqual((got.is_complete, got.finish_reason, got.delivered_sha256),
                             (True, "stop", triple["delivered_sha256"]))
            self.assertTrue(again.verify_chain()[0])
            # 三元组存在时序列化才输出（并通过签名校验）
            self.assertTrue({"is_complete", "finish_reason", "delivered_sha256"}
                            <= set(record_to_json(got)))

    def test_append_batch_supports_triple(self):
        led = AnchorLedger(None, KEY)
        triple = _triple(False, "cancel", "部分回答")
        recs = led.append_batch([
            {"content": "普通记录", "source_id": "s1", "episode_id": "e1", "speaker": "user"},
            {"content": "被取消的流式回答", "source_id": "s1", "episode_id": "e1",
             "speaker": "assistant", **triple},
        ])
        self.assertIsNone(recs[0].is_complete)
        self.assertEqual(recs[1].is_complete, False)
        self.assertEqual(recs[1].finish_reason, "cancel")
        self.assertTrue(led.verify_chain()[0])

    def test_triple_enters_payload_hash(self):
        plain = _append(AnchorLedger(None, KEY), "same content", speaker="assistant")
        full = _append(AnchorLedger(None, KEY), "same content", speaker="assistant",
                       **_triple(True))
        self.assertNotEqual(payload_hash(plain), payload_hash(full))
        self.assertNotEqual(plain.record_hash, full.record_hash)
        self.assertNotEqual(plain.signature, full.signature)

    def test_tampering_any_triple_field_fails_verification(self):
        led = AnchorLedger(None, KEY)
        rec = led.append("回答正文", source_id="s1", episode_id="e1", speaker="assistant",
                         **_triple(True, "stop"))
        tampered = {
            "is_complete": dataclasses.replace(rec, is_complete=False),
            "finish_reason": dataclasses.replace(rec, finish_reason="cancel"),
            "delivered_sha256": dataclasses.replace(rec, delivered_sha256=sha256_hex(b"other")),
        }
        for field, bad in tampered.items():
            with self.subTest(field=field):
                with self.assertRaises(RecordVerificationError):
                    led.verify(bad)
        # 原记录仍然可验（篡改只影响被改对象）
        led.verify(rec)

    def test_tampered_triple_in_file_breaks_chain(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = AnchorLedger(path, KEY)
            led.append("回答正文", source_id="s1", episode_id="e1", speaker="assistant",
                       **_triple(True, "stop"))
            with open(path, "rb") as fh:
                lines = fh.read().splitlines()
            obj = json.loads(lines[0])
            obj["is_complete"] = False   # 篡改三元组 → 签名/record_hash 对不上
            with open(path, "wb") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8") + b"\n")
            with self.assertRaises(ChainBroken):
                AnchorLedger(path, KEY)

    def test_half_triple_is_rejected_and_writes_nothing(self):
        halfs = [
            {"is_complete": True},
            {"finish_reason": "stop"},
            {"delivered_sha256": sha256_hex(b"x")},
            {"is_complete": True, "finish_reason": "stop"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = AnchorLedger(path, KEY)
            for half in halfs:
                with self.subTest(half=sorted(half)):
                    with self.assertRaises(ValueError):
                        led.append("x", source_id="s", episode_id="e", **half)
            # 原子拒绝：失败路径不产生任何字节（也没有进内存链）
            self.assertEqual(os.path.getsize(path) if os.path.exists(path) else 0, 0)
            self.assertEqual(len(led), 0)
        with self.assertRaises(ValueError):
            AnchorLedger(None, KEY).append_batch([
                {"content": "x", "source_id": "s", "episode_id": "e", "is_complete": True},
            ])

    def test_bad_triple_types_are_rejected(self):
        led = AnchorLedger(None, KEY)
        bad_cases = [
            dict(is_complete=1, finish_reason="stop", delivered_sha256=sha256_hex(b"x")),
            dict(is_complete=True, finish_reason="", delivered_sha256=sha256_hex(b"x")),
            dict(is_complete=True, finish_reason="stop", delivered_sha256="abc"),
            dict(is_complete=True, finish_reason="stop", delivered_sha256=None),
        ]
        for kw in bad_cases:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    led.append("x", source_id="s", episode_id="e", **kw)

    def test_old_json_without_triple_still_loads_and_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            with open(path, "wb") as fh:
                fh.write(GOLDEN_JSON.encode("utf-8") + b"\n")
            led = AnchorLedger(path, KEY)
            rec = led.get("a_000000")
            self.assertEqual((rec.is_complete, rec.finish_reason, rec.delivered_sha256),
                             (None, None, None))
            self.assertEqual(rec.record_hash, GOLDEN_RECORD_HASH)
            self.assertTrue(led.verify_chain()[0])

    def test_repeated_construction_is_deterministic(self):
        triple = _triple(False, "cancel")
        hashes = []
        for _ in range(5):
            led = AnchorLedger(None, KEY)
            rec = led.append("deterministic", source_id="s", episode_id="e",
                             speaker="assistant", **triple)
            hashes.append((rec.record_hash, rec.signature, _json_of(rec)))
        self.assertEqual(len(set(hashes)), 1)


# --------------------------------------------------------------------------- #
# C2：第 5 谓词 scenario_evidence_incomplete
# --------------------------------------------------------------------------- #
class C2PredicateTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.complete = _append(self.ledger, "The deployment finished on March 5th.",
                                speaker="assistant", **_triple(True, "stop"))
        self.truncated = _append(self.ledger, "The deployment finished on March",
                                 speaker="assistant", **_triple(False, "cancel"))
        self.legacy = _append(self.ledger, "The rollout started yesterday.",
                              speaker="assistant")  # 旧记录：无完整性证据
        self.user_cancelled = _append(self.ledger, "User message body.",
                                      speaker="user", **_triple(False, "cancel"))

    def _by_name(self, claim):
        return {p.name: p for p in check_scenario(self.ledger, claim, ScenarioContext(),
                                                  policy=SCEN)}

    def test_five_predicates_fixed_order_and_severity(self):
        res = check_scenario(self.ledger, _excerpt(self.complete), ScenarioContext(), policy=SCEN)
        self.assertEqual(tuple(p.name for p in res), SCENARIO_PREDICATES)
        self.assertEqual(SCENARIO_PREDICATES[-1], EVIDENCE_INCOMPLETE)
        # 第 5 谓词是 hard（残缺回答不得作证据）；前四条的顺序/严重度由
        # tests/test_cite_scenario.py 的固定顺序用例锁住（那里用的是中文 fixture）。
        self.assertEqual(res[-1].severity, "hard")
        self.assertEqual(res[-1].name, EVIDENCE_INCOMPLETE)

    def test_truncated_assistant_record_hard_fails(self):
        p = self._by_name(_excerpt(self.truncated))[EVIDENCE_INCOMPLETE]
        self.assertEqual((p.passed, p.severity), (False, "hard"))
        self.assertIn(self.truncated.id, p.detail)

    def test_truncated_supporting_record_hard_fails(self):
        claim = Claim(Citation(self.complete.id, 0, self.complete.content),
                      text=self.complete.content,
                      supporting=(Citation(self.truncated.id, 0, self.truncated.content),))
        p = self._by_name(claim)[EVIDENCE_INCOMPLETE]
        self.assertFalse(p.passed)
        self.assertIn(self.truncated.id, p.detail)

    def test_legacy_record_passes_with_completeness_unknown_audit(self):
        p = self._by_name(_excerpt(self.legacy))[EVIDENCE_INCOMPLETE]
        self.assertTrue(p.passed)               # 不追溯改旧行为
        self.assertIn(COMPLETENESS_UNKNOWN, p.detail)   # 但如实标"不可判"
        self.assertIn(self.legacy.id, p.detail)

    def test_complete_assistant_record_passes_cleanly(self):
        p = self._by_name(_excerpt(self.complete))[EVIDENCE_INCOMPLETE]
        self.assertTrue(p.passed)
        self.assertEqual(p.detail, "")

    def test_user_record_is_not_scope_of_this_predicate(self):
        p = self._by_name(_excerpt(self.user_cancelled))[EVIDENCE_INCOMPLETE]
        self.assertTrue(p.passed)
        self.assertEqual(p.detail, "")

    def test_enforce_rejects_with_scenario_cause(self):
        r = enforce(self.ledger, _excerpt(self.truncated), policy=SCEN, context=ScenarioContext())
        self.assertEqual(r.verdict, "rejected")
        self.assertEqual(r.rejected_cause, "scenario")
        self.assertIn(EVIDENCE_INCOMPLETE, r.hard_failed)

    def test_enforce_degrades_not_rejects_for_legacy_record(self):
        r = enforce(self.ledger, _excerpt(self.legacy), policy=SCEN, context=ScenarioContext())
        self.assertNotEqual(r.verdict, "rejected")
        self.assertNotIn(EVIDENCE_INCOMPLETE, r.hard_failed)

    def test_default_policy_unchanged(self):
        # 默认策略（scenario=False）下第 5 谓词一次都不进：残缺回答在默认路径不会被它判死
        r = enforce(self.ledger, _excerpt(self.truncated))
        self.assertNotEqual(r.verdict, "rejected")
        self.assertIsNone(r.rejected_cause)
        self.assertNotIn(EVIDENCE_INCOMPLETE, [p.name for p in r.failed])

    def test_repeated_calls_identical(self):
        claim = _excerpt(self.legacy)
        first = check_scenario(self.ledger, claim, ScenarioContext(), policy=SCEN)
        for _ in range(10):
            again = check_scenario(self.ledger, claim, ScenarioContext(), policy=SCEN)
            self.assertEqual(again, first)


# --------------------------------------------------------------------------- #
# C3/C4：resolver 注入 + 证据不可用降级
# --------------------------------------------------------------------------- #
def _archive_record(content="Aerosmith played in Berlin.", speaker="assistant"):
    """另建一个账本拿到一条"本地账本里没有"的记录（同一密钥，因此可独立验签）。"""
    archive = AnchorLedger(None, KEY)
    rec = _append(archive, content, speaker=speaker)
    return archive, rec


class _BaseResolutionTest(unittest.TestCase):
    def setUp(self):
        self.archive, self.archived = _archive_record()
        self.local = AnchorLedger(None, KEY)          # 本地账本里没有这条记录
        self.claim = _excerpt(self.archived)

    def _resolver(self, rec):
        return lambda anchor_id: rec

    def _raise(self, reason, detail=""):
        def _resolver(anchor_id):
            raise EvidenceUnavailable(reason, detail)
        return _resolver


class C4ResolverPathTest(_BaseResolutionTest):
    def test_resolver_recovers_same_result_as_local_record(self):
        ctx = ScenarioContext(context_session_id="s1", asserted_scope="user_fact")
        want = enforce(self.archive, self.claim, policy=SCEN, context=ctx)
        got = enforce(self.local, self.claim, policy=SCEN, context=ctx,
                      resolver=self._resolver(self.archived))
        self.assertEqual(got.verdict, want.verdict)
        self.assertEqual(got.rejected_cause, want.rejected_cause)
        self.assertEqual([p.name for p in got.failed], [p.name for p in want.failed])
        self.assertEqual(got.degrade_reason, want.degrade_reason)

    def test_resolver_not_called_when_local_record_hits(self):
        def exploding(anchor_id):  # pragma: no cover - 被调用即失败
            raise AssertionError("resolver must not be called when local record verifies")

        r = enforce(self.archive, self.claim, policy=SCEN, resolver=exploding)
        self.assertEqual(r.verdict, "verified")

    def test_each_reason_maps_to_visible_degrade(self):
        for reason in EVIDENCE_UNAVAILABLE_REASONS:
            with self.subTest(reason=reason):
                r = enforce(self.local, self.claim,
                            resolver=self._raise(reason, "cold archive"))
                self.assertEqual(r.verdict, "degraded")          # 不得静默 pass
                self.assertIsNone(r.rejected_cause)
                self.assertEqual(r.degrade_reason, EVIDENCE_DEGRADE_REASON_OF[reason])
                self.assertEqual(r.degrade_reasons, (EVIDENCE_DEGRADE_REASON_OF[reason],))
                self.assertEqual([p.name for p in r.failed], [EVIDENCE_RESOLUTION_PREDICATE])
                self.assertIn(reason, r.failed[0].detail)
                self.assertEqual(r.failed[0].severity, "soft")
                self.assertFalse(r.coverage["full_checks_ran"])  # 后续检查一次都没跑

    def test_resolver_returning_none_is_missing(self):
        r = enforce(self.local, self.claim, resolver=lambda anchor_id: None)
        self.assertEqual((r.verdict, r.degrade_reason), ("degraded", "evidence_missing"))
        self.assertIn("returned no record", r.failed[0].detail)

    def test_resolver_returning_wrong_record_is_missing(self):
        other_ledger = AnchorLedger(None, KEY)
        _append(other_ledger, "filler", speaker="user")
        other = _append(other_ledger, "another record", speaker="user")  # id != a_000000
        self.assertNotEqual(other.id, self.claim.citation.anchor_id)
        r = enforce(self.local, self.claim, resolver=lambda anchor_id: other)
        self.assertEqual((r.verdict, r.degrade_reason), ("degraded", "evidence_missing"))

    def test_resolver_returning_unverifiable_record_degrades(self):
        tampered = dataclasses.replace(  # 内容被改但哈希/签名没跟着改 → 验签失败
            self.archived, content=self.archived.content + " tampered")
        r = enforce(self.local, self.claim, resolver=lambda anchor_id: tampered)
        self.assertEqual((r.verdict, r.degrade_reason),
                         ("degraded", "evidence_unreachable"))
        self.assertIn("failed verification", r.failed[0].detail)

    def test_resolver_returning_non_record_is_explicit_type_error(self):
        with self.assertRaises(TypeError):
            enforce(self.local, self.claim, resolver=lambda anchor_id: "not a record")

    def test_no_resolver_missing_anchor_is_not_a_silent_pass(self):
        # 兼容性说明：无 resolver 时保持 C4 之前的既有行为（rejected search_empty），不是 pass。
        r = enforce(self.local, self.claim)
        self.assertEqual((r.verdict, r.rejected_cause), ("rejected", "search_empty"))

    def test_repeated_enforce_with_resolver_is_identical(self):
        def resolver(anchor_id):
            raise EvidenceUnavailable("destroyed")

        first = enforce(self.local, self.claim, resolver=resolver)
        for _ in range(5):
            again = enforce(self.local, self.claim, resolver=resolver)
            self.assertEqual((again.verdict, again.degrade_reason, again.degrade_reasons),
                             (first.verdict, first.degrade_reason, first.degrade_reasons))
            self.assertEqual([p.detail for p in again.failed], [p.detail for p in first.failed])

    def test_evidence_unavailable_validates_reason(self):
        with self.assertRaises(ValueError):
            EvidenceUnavailable("some_random_reason")


class C3EmitWordingTest(_BaseResolutionTest):
    def test_evidence_destroyed_has_own_wording(self):
        res = emit_answer([{"type": "claim", "text": self.archived.content,
                            "cites": [{"anchor": self.archived.id,
                                       "quote": self.archived.content}]}],
                          self.local, resolver=self._raise("destroyed"))
        self.assertEqual(res.audit[0]["verdict"], "degraded")
        self.assertEqual(res.audit[0]["degrade_reason"], "evidence_destroyed")
        self.assertIn("已按删除权销毁", res.text)
        self.assertIn("无法核验", res.text)

    def test_evidence_missing_has_own_wording(self):
        res = emit_answer([{"type": "claim", "text": self.archived.content,
                            "cites": [{"anchor": self.archived.id,
                                       "quote": self.archived.content}]}],
                          self.local, resolver=self._raise("missing"))
        self.assertEqual(res.audit[0]["degrade_reason"], "evidence_missing")
        self.assertIn("在账本中不可达", res.text)

    def test_archived_reasons_also_get_wording(self):
        for reason, label in (("archived_timeout", "evidence_archived"),
                              ("archived_unreachable", "evidence_unreachable")):
            with self.subTest(reason=reason):
                res = emit_answer([{"type": "claim", "text": self.archived.content,
                                    "cites": [{"anchor": self.archived.id,
                                               "quote": self.archived.content}]}],
                                  self.local, resolver=self._raise(reason))
                self.assertEqual(res.audit[0]["degrade_reason"], label)
                self.assertIn("冷归档" if label == "evidence_archived" else "暂不可达", res.text)


class C4PipelinePassthroughTest(_BaseResolutionTest):
    def test_pipeline_forwards_resolver_to_gate(self):
        def generator(question, prompt_ctx):
            return [{"type": "claim", "text": self.archived.content,
                     "cites": [{"anchor": self.archived.id, "quote": self.archived.content}]}]

        pipe = AnswerPipeline(generator=generator, ledger=self.local,
                              resolver=self._raise("destroyed"))
        self.addCleanup(pipe.close)
        pipe.prefetch("k1", question="q", records=[])
        deadline = time.time() + 5.0
        result = pipe.answer("k1")
        while result.state != "gated" and time.time() < deadline:
            time.sleep(0.01)
            result = pipe.answer("k1")
        self.assertEqual(result.state, "gated")
        self.assertEqual(result.audit[0]["degrade_reason"], "evidence_destroyed")
        self.assertIn("已按删除权销毁", result.text)

    def test_pipeline_rejects_non_callable_resolver(self):
        with self.assertRaises(TypeError):
            AnswerPipeline(generator=lambda q, c: [], ledger=self.local, resolver="nope")


# --------------------------------------------------------------------------- #
# C5：slot 契约
# --------------------------------------------------------------------------- #
def _slot(**overrides):
    base = dict(
        slot_id="slot_0001",
        l0_id="a_000000",
        l0_hash=sha256_hex(b"L0 content"),
        char_span=(0, 40),
        subject_span=(0, 5),
        attribute="deploy_status",
        attribute_span=(6, 19),
        value_span=(23, 32),
        value_quote="已完成上线",
        polarity="positive",
        extractor_version="extract-v1",
        model_id="test-model",
        prompt_hash=sha256_hex(b"prompt"),
    )
    base.update(overrides)
    return StateSlot(**base)


class C5SlotContractTest(unittest.TestCase):
    def test_state_slot_roundtrip(self):
        slot = _slot()
        again = StateSlot.from_dict(slot.to_dict())
        self.assertEqual(again, slot)
        self.assertEqual(again.to_dict(), slot.to_dict())

    def test_state_slot_deterministic(self):
        keys = {json.dumps(_slot().to_dict(), ensure_ascii=False, sort_keys=True)
                for _ in range(10)}
        self.assertEqual(len(keys), 1)
        self.assertEqual(len({state_slot_payload_hash(_slot()) for _ in range(10)}), 1)
        # 往返后哈希不变（序列化口径稳定）
        self.assertEqual(state_slot_payload_hash(StateSlot.from_dict(_slot().to_dict())),
                         state_slot_payload_hash(_slot()))

    def test_state_slot_hash_changes_with_content(self):
        self.assertNotEqual(state_slot_payload_hash(_slot()),
                            state_slot_payload_hash(_slot(value_quote="已失败")))

    def test_state_slot_validates_spans_and_polarity(self):
        for bad in (dict(char_span=(5, 1)), dict(subject_span=(0, 99)),
                    dict(polarity="maybe"), dict(value_quote="")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _slot(**bad)

    def test_slot_event_hash_chain(self):
        payload = state_slot_payload_hash(_slot())
        first = make_slot_event(kind="proposed", slot_id=_slot().slot_id, l0_id="a_000000",
                                emitted_at="2026-11-13T23:00:00+08:00", payload_hash=payload)
        second = make_slot_event(kind="verified", slot_id=_slot().slot_id, l0_id="a_000000",
                                 emitted_at="2026-11-13T23:00:01+08:00", payload_hash=payload,
                                 prev_event_hash=first.event_hash)
        self.assertTrue(first.is_self_consistent())
        self.assertTrue(second.is_self_consistent())
        self.assertEqual(second.prev_event_hash, first.event_hash)
        self.assertEqual(second.event_hash,
                         slot_event_hash("verified", second.slot_id, second.l0_id,
                                         second.emitted_at, second.payload_hash,
                                         first.event_hash))

    def test_slot_event_tamper_is_detected(self):
        ev = make_slot_event(kind="proposed", slot_id="slot_0001", l0_id="a_000000",
                             emitted_at="2026-11-13T23:00:00+08:00",
                             payload_hash=sha256_hex(b"p"))
        for field, value in (("kind", "verified"), ("payload_hash", sha256_hex(b"q")),
                             ("emitted_at", "2026-11-14T00:00:00+08:00")):
            with self.subTest(field=field):
                bad = dataclasses.replace(ev, **{field: value})
                self.assertFalse(bad.is_self_consistent())

    def test_slot_event_roundtrip_and_validation(self):
        ev = make_slot_event(kind="rejected", slot_id="s1", l0_id="a_0",
                             emitted_at="2026-11-13T23:00:00+08:00",
                             payload_hash=sha256_hex(b"p"))
        self.assertEqual(SlotEvent.from_dict(ev.to_dict()), ev)
        with self.assertRaises(ValueError):
            make_slot_event(kind="bogus", slot_id="s1", l0_id="a_0",
                            emitted_at="2026-11-13T23:00:00+08:00",
                            payload_hash=sha256_hex(b"p"))

    def test_slot_not_wired_into_enforce(self):
        # 契约：本批不把 slot 接入 enforce 判定
        params = set(inspect.signature(enforce).parameters)
        self.assertFalse({"slot", "slots", "slot_store"} & params)
        self.assertTrue({"resolver"} <= params)

    def test_slot_contract_has_no_clock_or_network_imports(self):
        path = os.path.join(ROOT, "memory_kernel", "slot_contract.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        forbidden = {"time", "datetime", "random", "socket", "urllib", "http",
                     "requests", "httpx", "aiohttp", "openai", "anthropic",
                     "onnxruntime", "numpy", "torch", "subprocess"}
        tree = ast.parse(src, filename=path)
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        self.assertEqual(mods & forbidden, set())
        # 不读时钟：源码里不允许出现 sleep / perf_counter / time() 调用
        for needle in ("sleep(", "perf_counter", "time.time(", "datetime.now("):
            self.assertNotIn(needle, src)


if __name__ == "__main__":
    unittest.main()
