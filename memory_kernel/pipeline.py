# SPDX-License-Identifier: Apache-2.0
"""生产集成：异步生成 + 门控管线的延迟预算（internal-task）。

背景
────
外部生成模型单条生成可能耗时数十秒（重推理；实测 153 条总耗时约 2 小时）。同步请求路径不能等它：
真正需要"同步 ≤50ms"的只有**确定性部分**——阶段一（`build_prompt_context`）与
阶段三（`emit_answer` 门控）。模型生成必须异步预取，未就绪时**绝不伪造回答**。

架构
────
```
请求 → 阶段一 build_prompt_context（同步，毫秒级）
      → 阶段二 generator(question, prompt_ctx)  ← 注入的生成器，后台线程执行（~43s）
      → 阶段三 emit_answer（同步，毫秒级）  ← 生成就绪后调用
调用方两阶段取结果：
  1) prefetch(key, ...)  → 立刻返回，后台生成；
  2) answer(key)         → 未就绪：{state:"unrefined"}（明确标"未精排"，不给断言）
                           已就绪：{state:"gated"}（走门控，A/B/C/D 重写后的文本）
```

红线
────
1. **本模块不直连模型/网络**：`generator` 是调用方注入的纯 callable（生产绑定具体生成模型与 prompt 版本，
   测试绑 fake）。模块源码不 import 任何网络库；内核仍满足"执法不调模型"（门控本身确定性）。
2. **未就绪不编造**：`unrefined` 只返回一句"未完成核验/未精排、暂不给出断言"，**不返回任何
   模型内容、不返回未经门控的文本**。
3. **失败显式**：生成异常 → `generation_failed`，不静默降级成"无依据"，也不重试掩盖。
4. **预算可审计**：每次 `answer` 记录 `sync_ms`（阶段一+阶段三实际耗时）、生成耗时、
   是否超过 `sync_budget_ms`；调用方必须把 `state` 呈现给用户（"未验证/未精排"不可隐藏）。

用法
────
```python
pipe = AnswerPipeline(generator=my_generator, ledger=ledger,
                      policy=SkeletonPolicy.v4(), budget=PipelineBudget(sync_budget_ms=50))
pipe.prefetch("req-1", question=q, records=recalled, relevance=rrf_scores, now=now)
r1 = pipe.answer("req-1")          # 可能 unrefined
r2 = pipe.answer("req-1")          # 生成就绪后 gated
```
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .context_filter import PromptContext, build_prompt_context
from .emit import EmitResult, emit_answer
from .engine_cache import EngineCache
from .repair import RepairPolicy, RepairResult, run_repair
from .scenario import CLAIM_MODES, UNSPECIFIED_CLAIM_MODE
from .skeleton import SkeletonPolicy, entities_for_ledger, idf_for_ledger

__all__ = [
    "FAILED_TEXT",
    "NO_PREFETCH_TEXT",
    "AnswerPipeline",
    "PipelineBudget",
    "PipelineResult",
    "UNREFINED_TEXT",
    "prewarm_ledger",
]

UNREFINED_TEXT = "本次回答未能完成完整核验与精排，暂不给出断言；请稍后重试。"
FAILED_TEXT = "本次回答生成失败，未给出任何断言。"
NO_PREFETCH_TEXT = "本次请求没有对应的生成任务，未给出任何断言。"


def prewarm_ledger(ledger, policy=None, idf_provider=None, epoch=None, cache=None) -> None:
    """把 IDF / 实体索引的首建成本移出请求路径（部署必需，见延迟契约 §3）。

    账本加载后调用一次：`idf_for_ledger` 与 `entities_for_ledger` 按策略参数建缓存。
    不做预热时，冷启动首个门控请求最坏 ~78.5ms（超 50ms 预算）；预热后落入热路径分布
    （153 条探针 p95 35.2ms / max 46.6ms）。幂等：可重复调用。

    `idf_provider`（internal-task，可选）：`DynamicIDF` 提供者。给出时用**全量账本**构建
    首个版本化快照并原子切换；`epoch`（`yyyymmddHHMM`，调用方时钟）用于版本号，
    首建时必填。默认 `None` → 旧路径逐字节不变。

    `cache`（internal-task，可选）：显式 `EngineCache`（多实例隔离用）；`None` → 进程级兼容默认。
    """
    pol = policy or SkeletonPolicy()
    if idf_provider is not None:
        idf_provider.rebuild_from_ledger(ledger, epoch=epoch, reason="prewarm")
    else:
        idf_for_ledger(ledger, stem_ok=pol.stem, stop_mode=pol.stop_mode, zh=pol.zh,
                       cache=cache)
    entities_for_ledger(ledger, cache=cache)


@dataclass(frozen=True)
class PipelineBudget:
    """延迟预算（毫秒）。默认同步 50ms 来自交接文档 §9 的生产要求。"""

    sync_budget_ms: float = 50.0
    prefetch_ttl_s: float = 300.0
    max_workers: int = 2

    def __post_init__(self) -> None:
        if self.sync_budget_ms <= 0:
            raise ValueError(f"sync_budget_ms must be > 0, got {self.sync_budget_ms!r}")
        if self.prefetch_ttl_s <= 0:
            raise ValueError(f"prefetch_ttl_s must be > 0, got {self.prefetch_ttl_s!r}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers!r}")


@dataclass
class PipelineResult:
    """一次 `answer()` 的结果。`state` 是调用方必须呈现给用户的核心字段。"""

    state: str                       # gated / unrefined / generation_failed / no_prefetch
    text: str
    sync_ms: float
    budget_ok: bool
    evidence: tuple[str, ...] = ()
    dropped: tuple[dict, ...] = ()
    audit: tuple[dict, ...] = ()
    generation_ms: Optional[float] = None
    reason: Optional[str] = None
    #: internal-task：本条结果实际使用的 IDF 版本（注入 DynamicIDF 才有；默认 None = 不出现该键）
    idf_version: Optional[str] = None
    #: internal-task：修复链审计（opt-in；默认 None = 不出现该键，键集合与改动前一致）
    repair: Optional[dict] = None

    def as_dict(self) -> dict:
        out = {
            "state": self.state,
            "text": self.text,
            "sync_ms": round(self.sync_ms, 4),
            "budget_ok": self.budget_ok,
            "evidence": list(self.evidence),
            "dropped": list(self.dropped),
            "audit": list(self.audit),
            "generation_ms": None if self.generation_ms is None else round(self.generation_ms, 4),
            "reason": self.reason,
        }
        # 默认路径（未注入 IDF provider）逐字节不变：不新增键
        if self.idf_version is not None:
            out["idf_version"] = self.idf_version
        # 默认路径（未启用修复链）逐字节不变：不新增键
        if self.repair is not None:
            out["repair"] = self.repair
        return out


@dataclass
class _Prefetch:
    future: Future
    prompt_ctx: PromptContext
    submitted_at: float
    question: str
    # internal-task：修复链异步作业（None = 未提交；仅 opt-in 且首轮存在 rejected 时才提交）
    repair_future: Optional[Future] = None
    repair_result: Optional[RepairResult] = None
    repair_error: Optional[str] = None


class AnswerPipeline:
    """两阶段（预取 + 门控）管线。线程安全由内部锁 + ThreadPoolExecutor 保证。"""

    #: 生成器签名：`generator(question, prompt_context) -> Sequence[segment dict]`
    GeneratorSignature = Callable[[str, PromptContext], Sequence[Mapping]]

    def __init__(
        self,
        *,
        generator: GeneratorSignature,
        ledger,
        policy=None,
        budget: Optional[PipelineBudget] = None,
        clock: Callable[[], float] = time.perf_counter,
        resolver: Optional[Callable[[str], object]] = None,
        claim_mode: Optional[str] = None,
        idf_provider=None,
        repair_policy: Optional[RepairPolicy] = None,
        regenerator: Optional[Callable] = None,
        engine_cache: Optional[EngineCache] = None,
    ) -> None:
        if not callable(generator):
            raise TypeError("generator must be callable(question, prompt_context) -> segments")
        if resolver is not None and not callable(resolver):
            raise TypeError("resolver must be callable(anchor_id) -> AnchorRecord, or None")
        if claim_mode is not None and claim_mode not in CLAIM_MODES:
            raise ValueError(f"claim_mode must be one of {CLAIM_MODES}, got {claim_mode!r}")
        if idf_provider is not None and not callable(getattr(idf_provider, "snapshot", None)):
            raise TypeError("idf_provider must expose snapshot() -> IDFSnapshot, or be None")
        if repair_policy is not None and not isinstance(repair_policy, RepairPolicy):
            raise TypeError("repair_policy must be a RepairPolicy, or None")
        if regenerator is not None and not callable(regenerator):
            raise TypeError("regenerator must be callable(RepairRequest) -> RepairResponse, or None")
        if engine_cache is not None and not isinstance(engine_cache, EngineCache):
            raise TypeError("engine_cache must be an EngineCache, or None")
        self._generator = generator
        self._ledger = ledger
        self._policy = policy
        self._budget = budget or PipelineBudget()
        self._clock = clock
        # Plan Z C4：冷归档证据解析器原样透传给 emit_answer()/enforce()；None = 与 C4 之前一致
        self._resolver = resolver
        # Plan Z 两态：管线级**默认**模式（None = 不指定，逐字节保持旧行为）。
        # 阶段一用它门控 superseded；阶段三透传给 enforce（context.claim_mode 非 unspecified 时优先）。
        self._claim_mode = claim_mode
        # internal-task：动态 IDF 提供者（None = 旧路径逐字节不变；给出时结果带 idf_version）
        self._idf_provider = idf_provider
        # internal-task：跨请求派生缓存**每实例一份**（或由调用方显式注入共享）。
        # None → 自建实例并拥有它（close() 时清空）；注入 → 仅借用，close() 不动调用方的缓存。
        if engine_cache is not None:
            self._engine_cache = engine_cache
            self._owns_engine_cache = False
        else:
            self._engine_cache = EngineCache(idf_provider=idf_provider)
            self._owns_engine_cache = True
        # internal-task：修复链（opt-in；None 或 enabled=False 时零行为变化、stats 不新增键）
        self._repair_policy = repair_policy
        self._regenerator = regenerator
        self._repair_enabled = bool(repair_policy is not None and repair_policy.enabled)
        self._executor = ThreadPoolExecutor(max_workers=self._budget.max_workers,
                                            thread_name_prefix="answer-gen")
        self._lock = threading.Lock()
        self._prefetches: dict[str, _Prefetch] = {}
        self._stats = {"prefetch": 0, "gated": 0, "unrefined": 0,
                       "generation_failed": 0, "no_prefetch": 0,
                       "evicted": 0, "budget_breach": 0}
        # 修复链计数只在 opt-in 时出现在 stats()；默认键集合不变
        self._repair_stats = {"repair_submitted": 0, "repair_completed": 0,
                              "repair_failed": 0, "repair_tokens": 0}
        self._sync_samples: list[float] = []

    # ---- 阶段一 + 阶段二（异步） ---------------------------------------- #
    def prefetch(
        self,
        key: str,
        *,
        question: str,
        records: Sequence,
        relevance: Optional[Mapping[str, float]] = None,
        now: Optional[float] = None,
        record_times: Optional[Mapping[str, float]] = None,
        contaminated_sources: Iterable[str] = (),
        superseded: Optional[Iterable[str]] = None,
        token_budget: Optional[int] = None,
    ) -> PromptContext:
        """同步跑阶段一，然后后台提交阶段二；返回阶段一结果（含 dropped 审计）。"""
        if not key:
            raise ValueError("key must be a non-empty string")
        prompt_ctx = build_prompt_context(
            records, relevance, now=now, record_times=record_times,
            contaminated_sources=contaminated_sources, superseded=superseded,
            token_budget=token_budget,
            claim_mode=self._claim_mode or UNSPECIFIED_CLAIM_MODE,
        )
        with self._lock:
            self._prune_locked()
            fut = self._executor.submit(self._generator, question, prompt_ctx)
            self._prefetches[key] = _Prefetch(future=fut, prompt_ctx=prompt_ctx,
                                             submitted_at=self._clock(), question=question)
            self._stats["prefetch"] += 1
        return prompt_ctx

    # ---- 阶段三（同步门控） --------------------------------------------- #
    def answer(self, key: str, *, context=None, timeout_ms: float = 0.0) -> PipelineResult:
        """取结果：就绪则门控重写（慢在后台，不在同步路径）；未就绪返回 `unrefined`。

        `timeout_ms` 仅用于诊断/离线场景；生产同步路径保持 0（不阻塞）。
        """
        t0 = self._clock()
        with self._lock:
            self._prune_locked()
            entry = self._prefetches.get(key)
        if entry is None:
            return self._finish("no_prefetch", NO_PREFETCH_TEXT, t0, reason="no such prefetch")

        if timeout_ms > 0 and not entry.future.done():
            entry.future.wait(timeout_ms / 1000.0)

        if not entry.future.done():
            result = self._finish("unrefined", UNREFINED_TEXT, t0, entry=entry,
                                  reason="generation not ready")
            with self._lock:
                self._stats["unrefined"] += 1
            return result

        exc = entry.future.exception()
        if exc is not None:
            result = self._finish("generation_failed", FAILED_TEXT, t0, entry=entry,
                                  reason=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._stats["generation_failed"] += 1
            return result

        segments = entry.future.result() or ()
        idf_model = None
        idf_version = None
        if self._idf_provider is not None:
            # internal-task：读路径只取当前生效快照（O(1)），并用同一版本号标注结果
            snap = self._idf_provider.snapshot()
            idf_model = getattr(snap, "model", None)
            idf_version = getattr(snap, "version", None)
        emitted: EmitResult = emit_answer(segments, self._ledger, policy=self._policy,
                                          context=context, resolver=self._resolver,
                                          claim_mode=self._claim_mode, idf_model=idf_model,
                                          cache=self._engine_cache)
        text, audit, repair_meta = emitted.text, tuple(emitted.audit), None
        # internal-task：修复链只对"首轮存在 rejected claim"的文档生效；异步作业不进同步路径。
        # 未启用 repair_policy 时整段不执行 → 结果/键集合逐字节不变。
        if self._repair_enabled and _has_rejected(emitted.audit):
            staged_result, repair_meta = self._repair_stage(entry, segments, context)
            if staged_result is not None:
                text = staged_result.final_text
                audit = tuple(staged_result.final_audit)
        result = self._finish("gated", text, t0, entry=entry,
                              audit=audit, idf_version=idf_version, repair=repair_meta)
        with self._lock:
            self._stats["gated"] += 1
        return result

    # ---- 修复链（internal-task，异步；仅 opt-in） --------------------------- #
    def _repair_stage(self, entry: _Prefetch, segments, context):
        """修复链两阶段：首次调用提交后台作业，后续调用回收结果。

        返回 `(RepairResult|None, repair_audit|None)`：
          * `(None, {"document": {"repair_state": "pending"}})` = 作业进行中（调用方仍用首轮文本）；
          * `(result, result.audit)` = 修复完成（调用方改用 `result.final_text/final_audit`）。
        """
        if entry.repair_result is not None:
            return entry.repair_result, entry.repair_result.audit
        if entry.repair_error is not None:
            return None, {"enabled": True,
                          "document": {"repair_state": "error", "reason": entry.repair_error}}
        if entry.repair_future is None:
            pool = tuple((e.record.id, e.record.content) for e in entry.prompt_ctx.records)
            with self._lock:
                if entry.repair_future is None:
                    entry.repair_future = self._executor.submit(
                        run_repair, tuple(segments),
                        ledger=self._ledger, policy=self._policy, context=context,
                        repair_policy=self._repair_policy, regenerator=self._regenerator,
                        question=entry.question, evidence_pool=pool,
                    )
                    self._repair_stats["repair_submitted"] += 1
            return None, {"enabled": True, "document": {"repair_state": "pending"}}
        if entry.repair_future.done():
            try:
                repaired = entry.repair_future.result()
            except Exception as exc:  # 失败关闭：保留首轮文本，审计显式
                entry.repair_error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._repair_stats["repair_failed"] += 1
                return None, {"enabled": True,
                              "document": {"repair_state": "error",
                                           "reason": entry.repair_error}}
            entry.repair_result = repaired
            with self._lock:
                self._repair_stats["repair_completed"] += 1
                used = repaired.audit.get("used") or {}
                self._repair_stats["repair_tokens"] += int(used.get("tokens") or 0)
            return repaired, repaired.audit
        return None, {"enabled": True, "document": {"repair_state": "pending"}}

    # ---- 收尾/统计 ------------------------------------------------------- #
    def _finish(self, state: str, text: str, t0: float, *, entry: Optional[_Prefetch] = None,
                audit: tuple[dict, ...] = (), reason: Optional[str] = None,
                idf_version: Optional[str] = None,
                repair: Optional[dict] = None) -> PipelineResult:
        elapsed_ms = (self._clock() - t0) * 1000.0
        budget_ok = elapsed_ms <= self._budget.sync_budget_ms
        evidence = tuple(e.record.id for e in (entry.prompt_ctx.records if entry else ()))
        dropped = tuple({"anchor": d.anchor, "reason": d.reason, "detail": d.detail}
                        for d in (entry.prompt_ctx.dropped if entry else ()))
        generation_ms = None
        if entry is not None and entry.future.done():
            generation_ms = (self._clock() - entry.submitted_at) * 1000.0
        with self._lock:
            self._sync_samples.append(elapsed_ms)
            if not budget_ok:
                self._stats["budget_breach"] += 1
        return PipelineResult(state=state, text=text, sync_ms=elapsed_ms, budget_ok=budget_ok,
                              evidence=evidence, dropped=dropped, audit=audit,
                              generation_ms=generation_ms, reason=reason,
                              idf_version=idf_version, repair=repair)

    def _prune_locked(self) -> None:
        now = self._clock()
        stale = [k for k, e in self._prefetches.items()
                 if now - e.submitted_at > self._budget.prefetch_ttl_s]
        for k in stale:
            self._prefetches.pop(k, None)
            self._stats["evicted"] += 1

    def forget(self, key: str) -> bool:
        """显式丢弃一个预取（例如请求已取消）。"""
        with self._lock:
            return self._prefetches.pop(key, None) is not None

    def prewarm(self, epoch: Optional[str] = None) -> None:
        """便捷入口：账本加载后调用，预热本管线所用策略的 IDF/实体缓存。

        `epoch` 仅在注入了 `idf_provider` 时使用（版本号里的 `yyyymmddHHMM`，调用方时钟）。
        缓存写进**本实例的** `EngineCache`（internal-task），不碰进程级默认缓存。
        """
        prewarm_ledger(self._ledger, self._policy,
                       idf_provider=self._idf_provider, epoch=epoch,
                       cache=self._engine_cache)

    def cache_stats(self) -> dict:
        """本实例派生缓存的计数器（internal-task；只是可观测性，不影响判定）。"""
        return self._engine_cache.stats().as_dict()

    def stats(self) -> dict:
        """运行统计（含同步路径 p50/p95，供延迟预算回归）。"""
        with self._lock:
            samples = sorted(self._sync_samples)
            stats = dict(self._stats)
            stats["pending"] = len(self._prefetches)
        stats["sync_p50_ms"] = _percentile(samples, 50)
        stats["sync_p95_ms"] = _percentile(samples, 95)
        if self._idf_provider is not None:
            stats["idf_version"] = getattr(self._idf_provider, "active_version", None)
        # 修复链计数只在 opt-in 时出现（默认 stats 键集合不变）
        if self._repair_enabled:
            stats.update(self._repair_stats)
        return stats

    def close(self) -> None:
        """停止线程池（生产进程退出/测试清理用）。不等待正在进行的生成。

        internal-task：只清空**本实例自建**的 `EngineCache`（注入的共享缓存归调用方管理），
        幂等；清空只丢派生数据，不影响任何判定（重建后结果相同）。
        """
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._owns_engine_cache:
            self._engine_cache.clear()


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    """最近秩百分位（ceil）；空样本返回 None。确定性，无插值。"""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return round(sorted_values[rank - 1], 4)


def _has_rejected(audit: Sequence[dict]) -> bool:
    """首轮 audit 是否存在被拒 claim（修复链唯一入口条件；纯函数）。"""
    return any(a.get("type") == "claim" and a.get("state") == "rejected"
               for a in (audit or ()))
