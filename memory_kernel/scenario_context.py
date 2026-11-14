# SPDX-License-Identifier: Apache-2.0
"""生产链路场景上下文注入（internal-task）。

问题
────
`memory_kernel/scenario.py` 的四个场景谓词需要 `ScenarioContext` 的四个字段
（`context_session_id` / `context_time` / `asserted_scope` / `record_times`），
但账本 `AnchorRecord` **不存时点**，评测侧是从探针 JSON 注入的。生产链路必须有
确定性的来源与回退，否则这些谓词在生产里永远不可用（上下文为空 = 谓词全部 passed）。

设计要点
────────
1. **不改账本契约**：`AnchorRecord` 的 payload 哈希参与签名链，加字段会改历史指纹
   （`7d9ea11b…`）与既有冻结产物的可比性。因此时点放**锚点侧表** `RecordTimeStore`：
   锚点入库时由调用方写入 `anchor_id -> epoch 秒`；侧表与账本一样**只追加、不可改**
   （同一锚点重复写相同值幂等，写不同值显式报错）。
2. **不可判就报不可判**：某锚点没有时点 → 不放进 `record_times`（`scenario_timeline`
   对该锚点"不可判"而不是"通过"），同时用 `ContextAudit.coverage` 把覆盖率报出来。
3. **上下文来源单一、显式**：`session_id`（当前会话）、`now`（当前断言时点，epoch 秒）、
   `asserted_scope`（断言主体范围）全部由调用方显式传入，模块不猜、不读环境变量、
   不做 I/O、无随机/时间调用（确定性）。
4. **失败显式**：非数字时点抛 `TypeError`、NaN/Inf/负数抛 `ValueError`、非法
   `asserted_scope` 抛 `ValueError`——不做静默降级（与 `scenario.py` 契约 §5 一致）。

生产接线
────────
```python
store = RecordTimeStore()
for batch in ingestion:                       # 入库时：账本 append 与侧表 note 成对写入
    for anchor_id, epoch in batch.stamps:
        store.note(anchor_id, epoch)
ctx, audit = build_scenario_context(
    anchors=claim_anchors(claim), store=store,
    session_id=request_session_id, now=request_time, asserted_scope="user_fact")
result = emit_answer(segments, ledger, context=ctx, policy=SkeletonPolicy.v4())
```
`audit.coverage` 必须随请求落日志（覆盖率不足时，场景谓词只能算部分生效）。

明确不声称
──────────
* 不解决 `asserted_scope` 的**自动判定**：从自由文本猜"这句是用户事实还是助手推测"
  需要语义判断（公理 B 禁止执法层调用模型）。默认 `unspecified`（谓词不启用），
  调用方若知道来源就显式传 `user_fact`；不认识就保持 `unspecified`，不得编造。
* 不负责侧表持久化：生产存储由调用方实现（要求与账本同样的只追加语义），本模块只提供
  内存实现与 `from_mapping` 装载入口。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .scenario import (
    CLAIM_MODES,
    CURRENT_BASIS_CLAIM_MODE,
    HISTORICAL_RECALL_CLAIM_MODE,
    UNSPECIFIED_CLAIM_MODE,
    ScenarioContext,
)

__all__ = [
    "ALLOWED_SCOPES",
    "ARCHIVED_TIMEOUT_REASON",
    "CURRENT_MODE_CUES",
    "EVIDENCE_UNAVAILABLE_REASONS",
    "HISTORICAL_MODE_CUES",
    "ContextAudit",
    "EvidenceUnavailable",
    "RecordTimeStore",
    "build_scenario_context",
    "claim_anchors",
    "is_claim_mode_ambiguous",
    "lexical_mode_cues",
    "resolve_claim_mode",
]

ALLOWED_SCOPES = ("user_fact", "assistant_claim", "unspecified")

# --------------------------------------------------------------------------- #
# Plan Z 两态记忆：模式只由调用方显式给出（SPEC §15 D-TS-2）
# --------------------------------------------------------------------------- #
# 「意图只认调用方显式 mode」：`enforce()` 只读 `ScenarioContext.claim_mode`，不读题面/
# answer_policy/模型输出。下面两张线索表**只用于歧义诊断**（`is_claim_mode_ambiguous`），
# 绝不作为推断来源——V3 反例：revoke 31 题里 18 题同时命中历史/当前线索，词面意图失效
# （`docs/review-fix-2-3.md` §4：V3 29.0% < V1 48.4%）。
HISTORICAL_MODE_CUES: tuple[str, ...] = (
    "之前", "以前", "当时", "原来", "早先", "曾经", "那次", "过去", "上次",
    "previously", "before", "at the time", "used to", "back then",
)
CURRENT_MODE_CUES: tuple[str, ...] = (
    "现在", "目前", "当前", "如今", "最新",
    "currently", "right now", "now", "latest", "as of",
)


def lexical_mode_cues(claim_text: str) -> tuple[str, ...]:
    """返回题面命中的线索族（`"historical"` / `"current"`，固定顺序、去重）。

    只做**歧义诊断**，不产生 mode；调用方不得把它当 mode 推断器（D-TS-2）。
    """
    text = (claim_text or "").lower()
    out: list[str] = []
    if any(cue in text for cue in HISTORICAL_MODE_CUES):
        out.append("historical")
    if any(cue in text for cue in CURRENT_MODE_CUES):
        out.append("current")
    return tuple(out)


def is_claim_mode_ambiguous(claim_text: str) -> bool:
    """题面**同时**命中历史/当前线索族 → True（词面意图不可判，只能落 unspecified）。"""
    return len(lexical_mode_cues(claim_text)) > 1


def resolve_claim_mode(explicit_mode: Optional[str], claim_text: str = "") -> str:
    """把"调用方给的显式 mode"规整成合法 mode（唯一允许的赋值路径）。

    * `explicit_mode` 非 None → 校验后原样返回（调用方显式意图优先，词面线索不覆盖它）；
    * `explicit_mode` 为 None → **返回 `unspecified`**，绝不用词面线索推断出模式
      （词面线索只在 `is_claim_mode_ambiguous` 里用于"歧义→unspecified"的审计说明）。

    `claim_text` 参数只为审计/日志方便而接受，**不改变返回值**：确定性、无随机、无模型。
    """
    if explicit_mode is None:
        # 无显式意图 = unspecified（歧义时更是 unspecified；这里返回值与 claim_text 无关）
        return UNSPECIFIED_CLAIM_MODE
    if explicit_mode not in CLAIM_MODES:
        raise ValueError(
            f"claim_mode must be one of {CLAIM_MODES}, got {explicit_mode!r}"
        )
    return explicit_mode

# --------------------------------------------------------------------------- #
# Plan Z C4：冷归档证据解析的失败原因（封闭集合）
# --------------------------------------------------------------------------- #
# `resolver` 是调用方注入的**纯 callable**：`anchor_id -> AnchorRecord`，不可解析时抛
# `EvidenceUnavailable(reason)`。core 内不 sleep / 不读时钟 / 不联网 / 不做 I/O——超时预算
# 与重试全部由调用方拥有（沿用"50ms 不是硬目标"的既有裁决）。
ARCHIVED_TIMEOUT_REASON = "archived_timeout"
EVIDENCE_UNAVAILABLE_REASONS: tuple[str, ...] = (
    "archived_timeout",      # 冷归档查询超时（预算由调用方拥有）
    "archived_unreachable",  # 冷归档暂时不可达（网络/服务错误、解出的记录验签失败）
    "destroyed",             # crypto-shred：payload 已按删除权销毁，不可解密
    "missing",               # manifest 都没有：数据事故（审计告警）
)


class EvidenceUnavailable(Exception):
    """证据无法解析（Plan Z C4）。`reason` 必须是 `EVIDENCE_UNAVAILABLE_REASONS` 之一。

    **不静默 pass**：`enforce()` 收到它会把判定降级（degrade）并记录 reason，
    而不是当作"没有引用"放行。
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in EVIDENCE_UNAVAILABLE_REASONS:
            raise ValueError(
                f"evidence unavailable reason must be one of "
                f"{EVIDENCE_UNAVAILABLE_REASONS}, got {reason!r}"
            )
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"evidence unavailable ({reason})" + (f": {detail}" if detail else "")
        )


def _check_epoch(value, *, what: str) -> float:
    """校验 epoch 秒：拒绝 bool（bool 是 int 子类）、非数字、NaN/Inf、负数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be int/float epoch seconds, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite, got {value!r}")
    if v < 0:
        raise ValueError(f"{what} must be >= 0 epoch seconds, got {value!r}")
    return v


class RecordTimeStore:
    """锚点 → 入库时点（epoch 秒）的**只追加侧表**。

    只追加语义（与账本一致）：
    * 同一锚点首次 `note` 生效；
    * 重复 `note` 相同值幂等（返回已存值）；
    * 重复 `note` 不同值 → `ValueError`（时点不可篡改）。
    """

    def __init__(self, stamps: Optional[Mapping[str, float]] = None) -> None:
        self._stamps: dict[str, float] = {}
        if stamps:
            for anchor_id, epoch in stamps.items():
                self.note(anchor_id, epoch)

    @classmethod
    def from_mapping(cls, stamps: Mapping[str, float]) -> "RecordTimeStore":
        return cls(stamps)

    def note(self, anchor_id: str, recorded_at: float) -> float:
        if not isinstance(anchor_id, str) or not anchor_id.strip():
            raise ValueError(f"anchor_id must be a non-empty str, got {anchor_id!r}")
        epoch = _check_epoch(recorded_at, what=f"recorded_at for {anchor_id!r}")
        if anchor_id in self._stamps:
            if self._stamps[anchor_id] != epoch:
                raise ValueError(
                    f"record time for anchor {anchor_id!r} is immutable "
                    f"(already {self._stamps[anchor_id]!r}, got {epoch!r})")
            return self._stamps[anchor_id]
        self._stamps[anchor_id] = epoch
        return epoch

    def get(self, anchor_id: str) -> Optional[float]:
        return self._stamps.get(anchor_id)

    def __contains__(self, anchor_id: object) -> bool:
        return anchor_id in self._stamps

    def __len__(self) -> int:
        return len(self._stamps)

    def items(self):
        return self._stamps.items()

    def record_times(self, anchors: Iterable[str]) -> dict[str, float]:
        """只保留已知锚点；未知锚点**不出现**在结果里（谓词对其"不可判"）。"""
        out: dict[str, float] = {}
        for anchor_id in anchors:
            epoch = self._stamps.get(anchor_id)
            if epoch is not None:
                out[anchor_id] = epoch
        return out

    def coverage(self, anchors: Iterable[str]) -> Optional[float]:
        """时点覆盖率 = 已知时点锚点数 / 去重锚点数；空输入返回 None（无可判对象）。"""
        unique = _dedupe(anchors)
        if not unique:
            return None
        timed = sum(1 for a in unique if a in self._stamps)
        return round(timed / len(unique), 4)


@dataclass(frozen=True)
class ContextAudit:
    """一次上下文注入的审计记录（必须随请求落日志，供报告计算覆盖率）。"""

    anchors: tuple[str, ...]
    timed: tuple[str, ...]
    missing: tuple[str, ...]
    coverage: Optional[float]
    context_time: Optional[float]
    session_id: Optional[str]
    asserted_scope: str
    # Plan Z 两态：默认 unspecified（既有构造点与读取点不受影响）
    claim_mode: str = UNSPECIFIED_CLAIM_MODE

    def as_dict(self) -> dict:
        return {
            "anchors": list(self.anchors),
            "n_anchors": len(self.anchors),
            "n_timed": len(self.timed),
            "missing": list(self.missing),
            "coverage": self.coverage,
            "context_time": self.context_time,
            "session_id": self.session_id,
            "asserted_scope": self.asserted_scope,
            "claim_mode": self.claim_mode,
        }


def claim_anchors(claim) -> tuple[str, ...]:
    """按"主引用 → supporting 声明顺序"取 anchor id（去重保序），与场景谓词遍历顺序一致。"""
    out: list[str] = []
    seen: set[str] = set()

    def push(anchor_id) -> None:
        aid = str(anchor_id)
        if aid not in seen:
            seen.add(aid)
            out.append(aid)

    primary = getattr(claim, "citation", None)
    if primary is not None:
        push(getattr(primary, "anchor_id", ""))
    for cite in getattr(claim, "supporting", ()) or ():
        push(getattr(cite, "anchor_id", ""))
    return tuple(out)


def _dedupe(anchors: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for a in anchors:
        aid = str(a)
        if aid and aid not in seen:
            seen.add(aid)
            out.append(aid)
    return tuple(out)


def build_scenario_context(
    *,
    anchors: Iterable[str],
    store: RecordTimeStore,
    session_id: Optional[str] = None,
    now: Optional[float] = None,
    asserted_scope: str = "unspecified",
    claim_mode: str = UNSPECIFIED_CLAIM_MODE,
) -> tuple[ScenarioContext, ContextAudit]:
    """把生产侧输入组装成 `ScenarioContext` + 审计。

    * `anchors`：本断言引用到的锚点（可用 `claim_anchors(claim)`）；只对这些锚点查时点。
    * `store`：入库时写好的锚点侧表。
    * `session_id`：当前会话（`AnchorRecord.source_id` 的空间）；None = 不判跨会话。
    * `now`：当前断言时点（epoch 秒）；None = 不判时间方向。
    * `asserted_scope`：`user_fact` / `assistant_claim` / `unspecified`；默认 unspecified
      （不启用 speaker_scope 谓词，不猜）。
    * `claim_mode`（Plan Z 两态）：`unspecified`（默认，既有行为逐字节不变）/
      `current_basis` / `historical_recall`。只接受调用方**显式**传入；本函数不做推断。
    """
    if not isinstance(store, RecordTimeStore):
        raise TypeError(f"store must be RecordTimeStore, got {type(store).__name__}")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise ValueError(f"session_id must be a non-empty str or None, got {session_id!r}")
    if asserted_scope not in ALLOWED_SCOPES:
        raise ValueError(f"asserted_scope must be one of {ALLOWED_SCOPES}, got {asserted_scope!r}")
    if claim_mode not in CLAIM_MODES:
        raise ValueError(f"claim_mode must be one of {CLAIM_MODES}, got {claim_mode!r}")
    ctx_time = None if now is None else _check_epoch(now, what="now")

    unique = _dedupe(anchors)
    times = store.record_times(unique)
    timed = tuple(a for a in unique if a in times)
    missing = tuple(a for a in unique if a not in times)
    ctx = ScenarioContext(
        context_session_id=session_id,
        context_time=ctx_time,
        asserted_scope=asserted_scope,
        record_times=times,
        claim_mode=claim_mode,
    )
    audit = ContextAudit(
        anchors=unique,
        timed=timed,
        missing=missing,
        coverage=store.coverage(unique),
        context_time=ctx_time,
        session_id=session_id,
        asserted_scope=asserted_scope,
        claim_mode=claim_mode,
    )
    return ctx, audit
