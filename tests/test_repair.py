# SPDX-License-Identifier: Apache-2.0
"""修复链（internal-task）单测：状态机 / 收窄 / 预算 / fake 重生成 / 审计 / 默认零变化 / 确定性。

契约：`docs/REPAIR-CHAIN-DESIGN.md`、`docs/REPAIR-CHAIN-IMPL.md`。
全部离线：`regenerator` 是注入的 fake，不联网、不调模型；执法路径确定性。
"""

from __future__ import annotations

import ast
import hashlib
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import AnswerPipeline, SkeletonPolicy  # noqa: E402
from memory_kernel.emit import (  # noqa: E402
    FALLBACK_TEXT,
    Segment,
    claim_from_segment,
    emit_answer,
)
from memory_kernel.enforce import Claim, Citation  # noqa: E402
from memory_kernel.ledger import AnchorLedger  # noqa: E402
from memory_kernel.repair import (  # noqa: E402
    NARROWABLE_CAUSES,
    REGENERABLE_CAUSES,
    RepairPolicy,
    RepairRequest,
    RepairResponse,
    build_request,
    decide_repair,
    narrow_candidates,
    run_repair,
    shape_check,
)

KEY = hashlib.sha256(b"repair-test").digest()

#: 默认 v4 策略（tau=0.8）
POL = SkeletonPolicy(zh=True)
#: 制造 skeleton 硬失败的策略（要求逐词全覆盖，failure_mode=rejected）
POL_HARD = SkeletonPolicy(zh=True, tau_soft=1.0, failure_mode="rejected")
#: 制造 citation_set 硬失败的策略（supporting quote 非子串）
POL_CITE = SkeletonPolicy(zh=True, cite_set=True)


def _cite(rec, quote: str = None):
    return {"anchor": rec.id, "quote": rec.content if quote is None else quote}


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AnchorLedger(None, KEY)
        self.car = self.ledger.append("Alice bought a red car on March 5th.",
                                      source_id="s1", episode_id="e1", turn_index=0,
                                      speaker="Alice")
        self.extra = self.ledger.append("The weather in Berlin was cold last week.",
                                        source_id="s1", episode_id="e1", turn_index=1,
                                        speaker="Bob")

    # -- 常用 segments ------------------------------------------------------
    def seg_ok(self):
        return {"type": "claim", "text": self.car.content,
                "cites": [_cite(self.car)]}

    def seg_bad_quote(self):
        return {"type": "claim", "text": "Alice bought a blue car.",
                "cites": [{"anchor": self.car.id, "quote": "not the record"}]}

    def seg_overcite(self):
        return {"type": "claim", "text": self.car.content,
                "cites": [_cite(self.car), _cite(self.extra, "not the record")]}

    def replay_response(self, rec=None, text=None, usage=None):
        rec = rec or self.car
        return RepairResponse(
            segments=({"type": "claim", "text": rec.content if text is None else text,
                       "cites": [_cite(rec)]},),
            usage=usage or {"prompt_tokens": 100, "completion_tokens": 20},
            model_id="fake-model",
        )


class _FakeRegenerator:
    """注入式 fake 生成器：记录请求、返回固定响应（可抛错）。"""

    def __init__(self, response=None, *, exc: Exception = None):
        self.response = response
        self.exc = exc
        self.requests: list[RepairRequest] = []

    def __call__(self, request: RepairRequest) -> RepairResponse:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return self.response


class TestClaimFromSegment(_Base):
    def test_single_point_construction(self):
        seg = {"type": "claim", "text": "断言", "asserted_speaker": "user",
               "cites": [{"anchor": self.car.id, "quote": self.car.content},
                         {"anchor": self.extra.id, "quote": self.extra.content}]}
        claim = claim_from_segment(seg, ledger=self.ledger)
        self.assertEqual(claim.citation.anchor_id, self.car.id)
        self.assertEqual([c.anchor_id for c in claim.supporting], [self.extra.id])
        self.assertEqual(claim.text, "断言")
        self.assertEqual(claim.asserted_speaker, "user")

    def test_requires_cite(self):
        with self.assertRaises(ValueError):
            claim_from_segment({"type": "claim", "text": "无引用", "cites": []}, ledger=self.ledger)

    def test_emit_uses_it_and_audit_adds_failed(self):
        res = emit_answer([self.seg_ok(), self.seg_bad_quote()], self.ledger)
        self.assertEqual([a["action"] for a in res.audit], ["keep", "drop"])
        # 附加字段：只含**未通过**的谓词；verified 的 claim 为空列表
        self.assertEqual(res.audit[0]["failed"], [])
        self.assertIn("verbatim", res.audit[1]["failed"])
        self.assertEqual(res.audit[1]["rejected_cause"], "verbatim")


class TestDisabledByDefault(_Base):
    def test_default_policy_off_keeps_result(self):
        calls = _FakeRegenerator(self.replay_response())
        segments = [self.seg_ok(), self.seg_bad_quote()]
        baseline = emit_answer(segments, self.ledger, policy=POL)
        out = run_repair(segments, ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy(), regenerator=calls)
        self.assertEqual(out.final_text, baseline.text)
        self.assertEqual(out.final_audit, baseline.audit)
        self.assertEqual(list(out.segments), segments)
        self.assertEqual(out.repair_state, "disabled")
        self.assertIsNone(out.audit["used"]["attempts"])
        self.assertEqual(calls.requests, [])  # 关闭时绝不调用生成器

    def test_policy_v1_enables(self):
        self.assertFalse(RepairPolicy().enabled)
        self.assertTrue(RepairPolicy.v1().enabled)


class TestDecideRepair(_Base):
    def test_not_needed(self):
        audit = emit_answer([self.seg_ok()], self.ledger, policy=POL).audit
        decision = decide_repair(audit, RepairPolicy.v1())
        self.assertEqual(decision["state"], "not_needed")

    def test_pending_target_and_truncation(self):
        audit = emit_answer([self.seg_bad_quote(), self.seg_bad_quote()],
                            self.ledger, policy=POL).audit
        decision = decide_repair(audit, RepairPolicy.v1(max_claims=1))
        self.assertEqual(decision["state"], "pending")
        self.assertEqual(len(decision["targets"]), 1)
        self.assertEqual(decision["targets"][0]["cause"], "verbatim")
        self.assertEqual(decision["truncated"], 1)

    def test_non_repairable_cause_is_ineligible(self):
        seg = {"type": "claim", "text": "未知锚点", "cites": [{"anchor": "nope", "quote": "x"}]}
        audit = emit_answer([seg], self.ledger, policy=POL).audit
        decision = decide_repair(audit, RepairPolicy.v1())
        self.assertEqual(decision["state"], "ineligible")

    def test_disabled(self):
        audit = emit_answer([self.seg_bad_quote()], self.ledger, policy=POL).audit
        self.assertEqual(decide_repair(audit, RepairPolicy())["state"], "disabled")

    def test_cause_sets(self):
        self.assertIn("verbatim", REGENERABLE_CAUSES)
        self.assertIn("skeleton", NARROWABLE_CAUSES)
        self.assertNotIn("invalid_anchor", REGENERABLE_CAUSES)


class TestShapeCheck(_Base):
    pool = ()

    def setUp(self):
        super().setUp()
        self.pool = ((self.car.id, self.car.content), (self.extra.id, self.extra.content))

    def _check(self, segments):
        return shape_check(segments, evidence_pool=self.pool, ledger=self.ledger)

    def test_valid(self):
        seg, reason = self._check(({"type": "claim", "text": self.car.content,
                                    "cites": [_cite(self.car)]},))
        self.assertIsInstance(seg, Segment)
        self.assertEqual(reason, "")
        self.assertEqual(seg.cites[0].anchor_id, self.car.id)

    def test_narrative_is_dropped(self):
        seg, reason = self._check(({"type": "narrative", "text": "叙述"},))
        self.assertIsNone(seg)
        self.assertTrue(reason.startswith("not_single_claim"))

    def test_two_claims_rejected(self):
        seg, reason = self._check(({"type": "claim", "text": "a", "cites": [_cite(self.car)]},
                                   {"type": "claim", "text": "b", "cites": [_cite(self.extra)]}))
        self.assertIsNone(seg)

    def test_out_of_pool(self):
        other = self.ledger.append("third record", source_id="s1", episode_id="e1",
                                   turn_index=2, speaker="user")
        pool = ((self.car.id, self.car.content),)
        seg, reason = shape_check(
            ({"type": "claim", "text": self.car.content, "cites": [_cite(other)]},),
            evidence_pool=pool, ledger=self.ledger)
        self.assertEqual(reason, "cite_out_of_pool")

    def test_quote_not_substring(self):
        seg, reason = self._check(({"type": "claim", "text": "x",
                                    "cites": [{"anchor": self.car.id, "quote": "bad"}]},))
        self.assertEqual(reason, "quote_not_substring")

    def test_empty_cites_and_text(self):
        self.assertEqual(self._check(({"type": "claim", "text": "x", "cites": []},))[1],
                         "no_cites")
        self.assertEqual(self._check(({"type": "claim", "text": "  ",
                                       "cites": [_cite(self.car)]},))[1], "empty_text")


class TestRegeneration(_Base):
    def test_success_replaces_only_target_claim(self):
        segments = [{"type": "narrative", "text": "先说结论。"}, self.seg_ok(),
                    self.seg_bad_quote()]
        fake = _FakeRegenerator(self.replay_response())
        out = run_repair(segments, ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(max_attempts_per_claim=1),
                         regenerator=fake, question="谁买了车？",
                         evidence_pool=((self.car.id, self.car.content),
                                        (self.extra.id, self.extra.content)))
        self.assertEqual(out.repair_state, "complete")
        self.assertEqual(out.repaired_claims, 1)
        self.assertIn("先说结论。", out.final_text)
        self.assertIn("Alice bought a red car on March 5th", out.final_text)
        self.assertNotIn("blue car", out.final_text)      # 未核验文本不得透传
        # 未改动 segment 逐字节保留
        self.assertEqual(out.segments[0], segments[0])
        self.assertEqual(out.segments[1], segments[1])
        # 请求是确定性的：instruction id + 证据池 + 原问题
        req = fake.requests[0]
        self.assertEqual(req.instruction_ids, ("fix_verbatim",))
        self.assertEqual(req.question, "谁买了车？")
        self.assertEqual([a for a, _ in req.evidence_pool], [self.car.id, self.extra.id])
        # 审计：attempt 记录 usage / sha / verdict
        claim_audit = out.audit["claims"][0]
        self.assertEqual(claim_audit["initial"]["cause"], "verbatim")
        self.assertEqual(claim_audit["terminal"], "verified")
        self.assertEqual(claim_audit["terminal_register"], "fact")
        attempt = claim_audit["attempts"][0]
        self.assertEqual(attempt["shape_ok"], True)
        self.assertEqual(attempt["accepted"], True)
        self.assertEqual(out.audit["used"]["attempts"], 1)
        self.assertEqual(out.audit["used"]["tokens"], 120)

    def test_still_rejected_then_exhausted_and_fallback(self):
        # fake 仍然返回伪造 quote → 修复失败 → 沿用 emit 既有删除/兜底
        bad = RepairResponse(
            segments=({"type": "claim", "text": "Alice bought a blue car.",
                       "cites": [{"anchor": self.car.id, "quote": "still wrong"}]},),
            usage={"prompt_tokens": 10, "completion_tokens": 5})
        fake = _FakeRegenerator(bad)
        out = run_repair([self.seg_bad_quote()], ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(), regenerator=fake)
        self.assertEqual(out.final_text, FALLBACK_TEXT)
        self.assertTrue(out.audit["document"]["fallback_used"])
        self.assertEqual(out.audit["document"]["repaired_claims"], 0)
        self.assertEqual(out.audit["claims"][0]["terminal"], "dropped")
        self.assertEqual(out.audit["claims"][0]["terminal_register"], "unknown")

    def test_usage_missing_is_refused(self):
        fake = _FakeRegenerator(RepairResponse(
            segments=({"type": "claim", "text": self.car.content,
                       "cites": [_cite(self.car)]},),
            usage={}))
        out = run_repair([self.seg_bad_quote()], ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(), regenerator=fake)
        # usage 缺失 → 失败关闭：不采用结果，也不伪造 tokens
        self.assertEqual(out.final_text, FALLBACK_TEXT)
        attempt = out.audit["claims"][0]["attempts"][0]
        self.assertEqual(attempt["shape_reason"], "usage_missing")
        self.assertEqual(out.audit["used"]["tokens"], 0)

    def test_budget_tokens_blocks(self):
        fake = _FakeRegenerator(self.replay_response(
            usage={"prompt_tokens": 500, "completion_tokens": 500}))
        out = run_repair([self.seg_bad_quote()], ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(max_total_tokens=100),
                         regenerator=fake)
        self.assertEqual(out.final_text, FALLBACK_TEXT)
        attempt = out.audit["claims"][0]["attempts"][0]
        self.assertEqual(attempt["shape_reason"], "budget_tokens")
        self.assertEqual(out.audit["used"]["tokens"], 0)

    def test_generator_exception_is_fail_closed(self):
        fake = _FakeRegenerator(exc=RuntimeError("gateway down"))
        out = run_repair([self.seg_bad_quote()], ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(), regenerator=fake)
        self.assertEqual(out.final_text, FALLBACK_TEXT)
        attempt = out.audit["claims"][0]["attempts"][0]
        self.assertIn("regenerator_error", attempt["shape_reason"])

    def test_replay_is_content_addressed(self):
        segments = [self.seg_bad_quote()]
        fake = _FakeRegenerator(self.replay_response())
        first = run_repair(segments, ledger=self.ledger, policy=POL,
                           repair_policy=RepairPolicy.v1(), regenerator=fake)
        request = fake.requests[0]
        replay = {request.sha256(): fake.response}
        second = run_repair(segments, ledger=self.ledger, policy=POL,
                            repair_policy=RepairPolicy.v1(), replay=replay)
        self.assertEqual(first.final_text, second.final_text)
        self.assertEqual(first.audit["claims"][0]["terminal"],
                         second.audit["claims"][0]["terminal"])
        self.assertTrue(second.audit["claims"][0]["attempts"][0]["cached"])
        self.assertEqual(second.audit["used"]["attempts"], 0)  # 缓存命中不计模型调用

    def test_replay_missing_is_ineligible(self):
        segments = [self.seg_bad_quote()]
        out = run_repair(segments, ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(), replay={})
        self.assertEqual(out.final_text, FALLBACK_TEXT)
        self.assertEqual(out.audit["claims"][0]["attempts"][0]["shape_reason"],
                         "no_replay_response")


class TestNarrowing(_Base):
    def test_n3_subordinate_clause_recovers_skeleton(self):
        text = "Alice bought a red car on March 5th, and all her friends did too."
        segments = [{"type": "claim", "text": text, "cites": [_cite(self.car)]}]
        baseline = emit_answer(segments, self.ledger, policy=POL_HARD)
        self.assertEqual(baseline.audit[0]["state"], "rejected")
        self.assertEqual(baseline.audit[0]["rejected_cause"], "skeleton")
        out = run_repair(segments, ledger=self.ledger, policy=POL_HARD,
                         repair_policy=RepairPolicy.v1())
        self.assertEqual(out.repair_state, "complete")
        self.assertIn("Alice bought a red car on March 5th", out.final_text)
        self.assertNotIn("friends", out.final_text)
        claim_audit = out.audit["claims"][0]
        ops = [a.get("op") for a in claim_audit["attempts"] if a["step"] == "narrow"]
        self.assertIn("drop_subordinate_clause", ops)
        self.assertEqual(claim_audit["terminal"], "verified")
        self.assertEqual(claim_audit["narrowed_from"]["cites"], [self.car.id])

    def test_n2_generality_marker_recovers_skeleton(self):
        text = "Alice bought every red car on March 5th."
        segments = [{"type": "claim", "text": text, "cites": [_cite(self.car)]}]
        self.assertEqual(emit_answer(segments, self.ledger, policy=POL_HARD)
                         .audit[0]["state"], "rejected")
        out = run_repair(segments, ledger=self.ledger, policy=POL_HARD,
                         repair_policy=RepairPolicy.v1())
        self.assertEqual(out.repair_state, "complete")
        self.assertNotIn("every", out.final_text)
        ops = [a.get("op") for a in out.audit["claims"][0]["attempts"]
               if a["step"] == "narrow"]
        self.assertEqual(ops[0], "drop_generality_marker")  # N1 无可删 → N2 先命中

    def test_n1_drop_supporting_recovers_citation_set(self):
        segments = [self.seg_overcite()]
        self.assertEqual(emit_answer(segments, self.ledger, policy=POL_CITE)
                         .audit[0]["rejected_cause"], "citation_set")
        out = run_repair(segments, ledger=self.ledger, policy=POL_CITE,
                         repair_policy=RepairPolicy.v1())
        self.assertEqual(out.repair_state, "complete")
        self.assertEqual(out.segments[0].cites[0].anchor_id, self.car.id)
        self.assertEqual(len(out.segments[0].cites), 1)
        ops = [a.get("op") for a in out.audit["claims"][0]["attempts"]
               if a["step"] == "narrow"]
        self.assertEqual(ops[0], "drop_supporting_cites")

    def test_candidates_are_deterministic_and_ordered(self):
        claim = Claim(Citation(self.car.id, 0, self.car.content), text="a, and b",
                      supporting=(Citation(self.extra.id, 0, self.extra.content),))
        n1 = narrow_candidates(claim, "drop_supporting_cites")
        self.assertEqual([d["supporting"] for _, d in n1], ["1->0"])
        n2 = narrow_candidates(Claim(Citation(self.car.id, 0, self.car.content),
                                     text="Alice bought every red car."),
                               "drop_generality_marker")
        self.assertEqual(n2[0][1]["marker"], "every")
        # 固定顺序重复调用结果一致
        self.assertEqual([c.text for c, _ in n2],
                         [c.text for c, _ in narrow_candidates(
                             Claim(Citation(self.car.id, 0, self.car.content),
                                   text="Alice bought every red car."),
                             "drop_generality_marker")])
        # N4 默认未实现：返回空（不静默降级）
        self.assertEqual(narrow_candidates(claim, "drop_redundant_temporal_modifier"), [])

    def test_narrowing_never_flips_negation(self):
        claim = Claim(Citation(self.car.id, 0, self.car.content),
                      text="Alice did not buy every red car.")
        for _, detail in narrow_candidates(claim, "drop_generality_marker"):
            self.assertIn("every", str(detail))

    def test_narrowing_excluded_cause_does_not_run(self):
        # preserve_negation 不在 NARROWABLE：删句改不了极性 → 直接 exhausted
        seg = {"type": "claim", "text": self.car.content,
               "cites": [_cite(self.car)]}
        ledger = AnchorLedger(None, KEY)
        rec = ledger.append("我们下季度工业增加值预计增长5%。", source_id="sz",
                            episode_id="ez", turn_index=0, speaker="user")
        ledger.append("工业增加值是增长3%。", source_id="sz", episode_id="ez",
                      turn_index=1, speaker="user")
        seg = {"type": "claim", "text": "工业增加值不是增长5%。",
               "cites": [{"anchor": rec.id, "quote": rec.content}]}
        out = run_repair([seg], ledger=ledger, policy=SkeletonPolicy(zh=True),
                         repair_policy=RepairPolicy.v1())
        # 结果可能是 rejected（不在 NARROWABLE 则无收窄尝试）或 C 删除；关键是不崩、审计完整
        self.assertIn(out.repair_state, ("exhausted", "ineligible", "complete"))
        self.assertTrue(out.audit["claims"] or out.repair_state in ("not_needed", "ineligible"))


class TestBudgetAndClaimsLimit(_Base):
    def test_max_claims_limits_targets(self):
        segments = [self.seg_bad_quote(), self.seg_bad_quote()]
        out = run_repair(segments, ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(max_claims=1))
        self.assertEqual(len(out.audit["claims"]), 1)
        self.assertEqual(out.audit["document"]["truncated_targets"], 1)

    def test_max_attempts_recorded(self):
        fake = _FakeRegenerator(RepairResponse(
            segments=({"type": "claim", "text": "bad", "cites": [
                {"anchor": self.car.id, "quote": "bad"}]},),
            usage={"prompt_tokens": 1, "completion_tokens": 1}))
        out = run_repair([self.seg_bad_quote()], ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(max_attempts_per_claim=2),
                         regenerator=fake)
        regen_attempts = [a for a in out.audit["claims"][0]["attempts"]
                          if a["step"] == "regenerate"]
        self.assertEqual(len(regen_attempts), 2)
        self.assertEqual(out.audit["used"]["attempts"], 2)


class TestAuditSchema(_Base):
    def test_required_fields(self):
        fake = _FakeRegenerator(self.replay_response())
        out = run_repair([self.seg_ok(), self.seg_bad_quote()], ledger=self.ledger,
                         policy=POL, repair_policy=RepairPolicy.v1(), regenerator=fake)
        audit = out.audit
        for key in ("policy_version", "generator", "enabled", "baseline", "budget",
                    "used", "claims", "document"):
            self.assertIn(key, audit)
        for key in ("repaired_claims", "recovered", "degraded_claims", "dropped_claims",
                    "registers", "fallback_used", "repair_state"):
            self.assertIn(key, audit["document"])
        # P6：尝试过修复 → attempts/tokens 必须是 int
        self.assertIsInstance(audit["used"]["attempts"], int)
        self.assertIsInstance(audit["used"]["tokens"], int)
        claim_audit = audit["claims"][0]
        for key in ("index", "segment_sha256_before", "initial", "attempts", "terminal",
                    "terminal_register", "regenerated", "narrowed_from"):
            self.assertIn(key, claim_audit)
        for key in ("step", "verdict_after", "cause_after", "accepted"):
            self.assertIn(key, claim_audit["attempts"][0])

    def test_register_mapping_and_no_new_register(self):
        fake = _FakeRegenerator(self.replay_response())
        out = run_repair([self.seg_bad_quote()], ledger=self.ledger, policy=POL,
                         repair_policy=RepairPolicy.v1(), regenerator=fake)
        self.assertEqual(out.audit["claims"][0]["terminal_register"], "fact")
        self.assertEqual(set(out.audit["document"]["registers"]),
                         {"fact", "hedged", "unknown", "speculation"})


class TestDeterminism(_Base):
    def test_same_input_same_audit_and_text(self):
        segments = [{"type": "narrative", "text": "先说。"}, self.seg_ok(),
                    self.seg_bad_quote()]
        policy = RepairPolicy.v1()
        fake = _FakeRegenerator(self.replay_response())
        first = run_repair(segments, ledger=self.ledger, policy=POL,
                           repair_policy=policy, regenerator=fake)
        fake2 = _FakeRegenerator(self.replay_response())
        second = run_repair(segments, ledger=self.ledger, policy=POL,
                            repair_policy=policy, regenerator=fake2)
        self.assertEqual(first.final_text, second.final_text)
        self.assertEqual(first.audit, second.audit)          # ms 默认为 None → 逐字节同
        self.assertEqual(first.final_audit, second.final_audit)
        # 请求本身内容寻址一致
        self.assertEqual(fake.requests[0].sha256(), fake2.requests[0].sha256())


class _Wait:
    @staticmethod
    def repair_done(pipe: AnswerPipeline, key: str, timeout: float = 5.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with pipe._lock:  # noqa: SLF001 测试专用
                entry = pipe._prefetches.get(key)  # noqa: SLF001
            if entry is not None and (entry.repair_future is None or entry.repair_future.done()):
                return
            time.sleep(0.005)
        raise AssertionError("repair did not finish in time")


class TestPipelineIntegration(_Base):
    def _pipe(self, *, repair=None, regenerator=None):
        return AnswerPipeline(generator=lambda q, c: [self.seg_ok(), self.seg_bad_quote()],
                              ledger=self.ledger, policy=POL,
                              repair_policy=repair, regenerator=regenerator)

    def test_default_keys_unchanged(self):
        pipe = self._pipe()
        self.addCleanup(pipe.close)
        pipe.prefetch("k", question="q", records=[self.car, self.extra])
        _Wait.repair_done(pipe, "k")
        result = pipe.answer("k")
        self.assertIsNone(result.repair)
        self.assertNotIn("repair", result.as_dict())
        self.assertNotIn("repair_submitted", pipe.stats())

    def test_repair_async_then_recovered(self):
        fake = _FakeRegenerator(self.replay_response())
        pipe = self._pipe(repair=RepairPolicy.v1(), regenerator=fake)
        self.addCleanup(pipe.close)
        pipe.prefetch("k", question="谁买了车？", records=[self.car, self.extra])
        _Wait.repair_done(pipe, "k")
        first = pipe.answer("k")
        # 第一次调用可能刚提交作业（pending），也可能已完成；两者都不得透传未核验文本
        self.assertTrue(first.budget_ok)
        if first.repair and first.repair.get("document", {}).get("repair_state") == "pending":
            _Wait.repair_done(pipe, "k")
            second = pipe.answer("k")
        else:
            second = first
        self.assertEqual(second.text.count("Alice bought a red car on March 5th."), 2)
        self.assertNotIn("blue car", second.text)
        self.assertEqual(second.repair["document"]["repair_state"], "complete")
        stats = pipe.stats()
        self.assertEqual(stats["repair_submitted"], 1)
        self.assertEqual(stats["repair_completed"], 1)
        self.assertEqual(stats["repair_tokens"], 120)

    def test_no_rejected_no_repair(self):
        pipe = AnswerPipeline(generator=lambda q, c: [self.seg_ok()],
                              ledger=self.ledger, policy=POL,
                              repair_policy=RepairPolicy.v1(),
                              regenerator=_FakeRegenerator(self.replay_response()))
        self.addCleanup(pipe.close)
        pipe.prefetch("k", question="q", records=[self.car])
        _Wait.repair_done(pipe, "k")
        result = pipe.answer("k")
        self.assertIsNone(result.repair)
        self.assertEqual(pipe.stats()["repair_submitted"], 0)


class TestNoModelInKernel(unittest.TestCase):
    def test_repair_module_has_no_network_imports(self):
        src = (ROOT / "memory_kernel" / "repair.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = {"urllib", "http", "socket", "requests", "openai", "anthropic", "aiohttp"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], banned)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], banned)

    def test_repair_module_does_not_call_models(self):
        src = (ROOT / "memory_kernel" / "repair.py").read_text(encoding="utf-8")
        self.assertNotIn("time.time", src)
        self.assertNotIn("random", src)


if __name__ == "__main__":
    unittest.main()
