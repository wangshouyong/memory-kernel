# SPDX-License-Identifier: Apache-2.0
"""阶段一：Prompt 前置过滤（三阶段管线；契约 `docs/EMISSION-CONTRACT.md` §1 / §1.1.1）。

目的：在把证据塞进模型 Prompt 之前，做**确定性 + 相关性分档**的过滤：
  1) 硬规则：已被更正（superseded，复用 v4.4「同一事实」判据）、未来记录、污染源；
  2) 相关性分档：按调用方给的 relevance 分数（如 RRF 分数）删低分尾部（默认 25%），
     灰区保留但标 `low_confidence`；
  3) 预算溢出：超 token 预算时按 relevance 从低到高截断（单独记账）。

已校准的默认档（153 条真实聊天探针，`docs/EMISSION-CONTRACT.md` §1.1.1）：
  删低分尾部 25% → gold 保留率 98.7%、过滤后 R@10 96.1%（门禁：保留率 ≥98%、R@10 降幅 ≤2pp）。
**禁止**用「词面交集 == 0」当过滤判据：实测零分剔除只保住 92.2% 的 gold（转述类 claim 与
证据本来就没有词面交集），这条已被否决。

本模块**不调用模型、不联网、无随机**；所有丢弃都进 `dropped` 审计，调用方可回捞。

Plan Z 两态记忆（SPEC §15 / D-TS-2）
────────────────────────────────────
`claim_mode` 只门控 **"non-current"（已被更正/superseded）这一条轴**：
  * `unspecified`（默认）→ 传给 `superseded` 就丢（既有行为，逐字节不变）；
  * `current_basis` → 剔除 non-current（把"当前依据"语义显式化）；
  * `historical_recall` → 全留（回忆历史时，被更正的旧状态本身可能就是答案）。
`contaminated` / `future` / `low_relevance` / `budget` 四条丢弃轴**不受 mode 影响**；
mode 只由调用方显式传入，本模块不读题面/answer_policy/模型意图（V3 词面意图反例见
`docs/review-fix-2-3.md` §4）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Collection, Mapping, Optional, Sequence

from .ledger import AnchorRecord
from .scenario import (
    CLAIM_MODES,
    CURRENT_BASIS_CLAIM_MODE,
    HISTORICAL_RECALL_CLAIM_MODE,
    UNSPECIFIED_CLAIM_MODE,
)
from .skeleton import content_tokens
from .zh import zh_corrected

__all__ = [
    "DEFAULT_DROP_TAIL",
    "LOW_CONF_BAND",
    "SUPERSEDED_MIN_SHARED_TOKENS",
    "ContextEntry",
    "DroppedRecord",
    "PromptContext",
    "build_prompt_context",
    "detect_superseded_anchors",
]

#: 默认删低分尾部比例（校准推荐档，见模块 docstring）。
DEFAULT_DROP_TAIL = 0.25
#: 灰区宽度：在删除线之上再保留一段并标低置信（不删除，只标注）。
LOW_CONF_BAND = 0.20
#: 「同一事实」判据阈值：与被引记录共享的内容词数（口径对齐 scenario v4.4）。
SUPERSEDED_MIN_SHARED_TOKENS = 2

_STOP_LEVELS = ("contaminated", "future", "superseded", "low_relevance", "budget")


@dataclass(frozen=True)
class ContextEntry:
    """进入 Prompt 的一条记录。"""

    record: AnchorRecord
    relevance: float = 0.0
    low_confidence: bool = False


@dataclass(frozen=True)
class DroppedRecord:
    """被过滤掉的一条记录（审计用；`reason` 取值见 _STOP_LEVELS）。"""

    anchor: str
    reason: str
    detail: str = ""


@dataclass
class PromptContext:
    """过滤结果：保留记录 + 丢弃审计 + 估算信息。"""

    records: list[ContextEntry] = field(default_factory=list)
    dropped: list[DroppedRecord] = field(default_factory=list)
    tokens_estimate: int = 0
    chars_total: int = 0
    chars_kept: int = 0
    drop_tail: float = DEFAULT_DROP_TAIL

    @property
    def dropped_chars_ratio(self) -> float:
        if self.chars_total <= 0:
            return 0.0
        return (self.chars_total - self.chars_kept) / self.chars_total

    @property
    def dropped_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.dropped:
            out[d.reason] = out.get(d.reason, 0) + 1
        return out


def _tokens_of(text: str) -> set[str]:
    return content_tokens(text, zh=True)


def detect_superseded_anchors(ledger, *, min_shared: int = SUPERSEDED_MIN_SHARED_TOKENS) -> set[str]:
    """记录级 superseded 检测：同一 source/episode 里，后续用户更正消息「说的是同一件事」。

    口径对齐 v4.4 `scenario_superseded`：更正消息必须与被评记录共享 >= min_shared 个
    内容词（**在更正消息里移除更正标记词**，避免「不是」凭空凑词），且该记录更早。
    返回被更正记录的 anchor id 集合。确定性、O(n·k)，n 为同位事记录数。
    """
    records = list(ledger.records)
    by_episode: dict[tuple[str, str], list[AnchorRecord]] = defaultdict(list)
    for rec in records:
        by_episode[(rec.source_id, rec.episode_id)].append(rec)

    superseded: set[str] = set()
    for recs in by_episode.values():
        recs = sorted(recs, key=lambda r: (r.turn_index, r.seq))
        for later in recs:
            if (later.speaker or "").lower() != "user":
                continue
            if not zh_corrected(later.content):
                continue
            later_tokens = _tokens_of(_strip_correction_markers(later.content))
            if not later_tokens:
                continue
            for earlier in recs:
                if earlier.turn_index >= later.turn_index:
                    continue
                shared = _tokens_of(earlier.content) & later_tokens
                if len(shared) >= min_shared:
                    superseded.add(earlier.id)
    return superseded


#: 更正标记：与 zh.zh_corrected 的标记表保持同一批（此处仅用于预算/交集时剔除）。
_CORRECTION_MARKERS = ("不是", "改成", "其实", "更正", "说错", "改口", "记错", "不对")


def _strip_correction_markers(text: str) -> str:
    out = text
    for marker in _CORRECTION_MARKERS:
        out = out.replace(marker, " ")
    return out


def _estimate_tokens(chars: int, chars_per_token: float) -> int:
    if chars <= 0:
        return 0
    return int(math.ceil(chars / max(chars_per_token, 1e-6)))


def build_prompt_context(
    records: Sequence[AnchorRecord],
    relevance: Optional[Mapping[str, float]] = None,
    *,
    now: Optional[float] = None,
    record_times: Optional[Mapping[str, float]] = None,
    contaminated_sources: Collection[str] = (),
    superseded: Optional[Collection[str]] = None,
    token_budget: Optional[int] = None,
    chars_per_token: float = 1.0,
    drop_tail: float = DEFAULT_DROP_TAIL,
    low_conf_band: float = LOW_CONF_BAND,
    claim_mode: str = UNSPECIFIED_CLAIM_MODE,
) -> PromptContext:
    """构造进入 Prompt 的上下文（阶段一）。

    参数：
      records            候选记录（通常来自检索召回，**不再做语义硬判断**）。
      relevance          anchor id -> 相关性分数（如 RRF 分数）；缺省视为 0。
      now                当前时间（epoch 秒）；配合 record_times 过滤"未来记录"。
      record_times       anchor id -> 记录时间（epoch 秒）。
      contaminated_sources  被污染的 source id 集合（防御性过滤）。
      superseded         记录级「已被更正」的 anchor 集合；缺省不判（调用方可先调
                         ``detect_superseded_anchors``）。**不要传"词面无交集"结果**。
      token_budget       上下文 token 预算；超出后按 relevance 从低到高截断。
      chars_per_token    字符 -> token 估算比（默认 1.0，保守）。
      drop_tail          删低分尾部比例（默认 0.25，校准推荐档）。
      low_conf_band      灰区宽度（保留并标 low_confidence，不删除）。
      claim_mode         Plan Z 两态（SPEC §15 D-TS-2），**只门控** "non-current"（superseded）
                         这一条轴：
                           * ``unspecified``（默认）→ 现状：传了 superseded 就丢（逐字节不变）；
                           * ``current_basis`` → 剔除 non-current（= 丢 superseded，语义显式化）；
                           * ``historical_recall`` → **全留**（被更正的旧状态本身可能就是答案；
                             其中"未来记录"仍丢——那是时间方向问题，不是两态问题）。
                         contaminated / future / low_relevance / budget 四条丢弃轴不受 mode 影响；
                         mode 只由调用方显式给出，本函数不做意图推断。

    返回 ``PromptContext``；**丢弃项全部进 dropped 审计**，不静默丢。
    """
    if claim_mode not in CLAIM_MODES:
        raise ValueError(f"claim_mode must be one of {CLAIM_MODES}, got {claim_mode!r}")
    relevance = relevance or {}
    record_times = record_times or {}
    superseded = set(superseded or ())
    contaminated_sources = set(contaminated_sources or ())
    drop_tail = max(0.0, float(drop_tail))
    low_conf_band = max(0.0, float(low_conf_band))
    # 两态门控：只有 historical_recall 会改变 superseded 的处置（其它 = 现状）
    keep_non_current = claim_mode == HISTORICAL_RECALL_CLAIM_MODE

    dropped: list[DroppedRecord] = []
    survivors: list[AnchorRecord] = []
    for rec in records:
        if rec.source_id in contaminated_sources:
            dropped.append(DroppedRecord(rec.id, "contaminated", f"source={rec.source_id!r}"))
            continue
        if now is not None and rec.id in record_times and record_times[rec.id] > now:
            dropped.append(DroppedRecord(rec.id, "future", f"time>{now}"))
            continue
        if rec.id in superseded and not keep_non_current:
            detail = "later user correction about same fact"
            if claim_mode == CURRENT_BASIS_CLAIM_MODE:
                detail += " (current_basis: non-current evidence excluded)"
            dropped.append(DroppedRecord(rec.id, "superseded", detail))
            continue
        survivors.append(rec)

    # 稳定排序：分数降序，同分按 anchor id（确定性）
    ordered = sorted(survivors, key=lambda r: (-float(relevance.get(r.id, 0.0)), r.id))
    n = len(ordered)
    k_drop = int(math.floor(n * drop_tail)) if n >= 3 else 0
    k_low = int(math.floor(n * (drop_tail + low_conf_band))) if n >= 3 else 0
    k_low = min(k_low, n)

    kept_records = ordered[: n - k_drop]
    for idx in range(n - k_drop, n):
        rec = ordered[idx]
        dropped.append(DroppedRecord(rec.id, "low_relevance",
                                     f"relevance={float(relevance.get(rec.id, 0.0)):.6f} rank={idx + 1}/{n}"))

    entries: list[ContextEntry] = []
    for idx, rec in enumerate(kept_records):
        # 灰区：删除线之上的最低一段保留并标低置信（不删）
        low_conf = idx >= (len(kept_records) - max(0, k_low - k_drop))
        entries.append(ContextEntry(rec, float(relevance.get(rec.id, 0.0)), low_conf))

    # 预算截断：只按 relevance（与预算原因分开记账）
    chars_total = sum(len(r.content) for r in records)
    while token_budget is not None:
        chars_kept = sum(len(e.record.content) for e in entries)
        if _estimate_tokens(chars_kept, chars_per_token) <= token_budget or not entries:
            break
        victim = min(entries, key=lambda e: (e.relevance, e.record.id))
        entries = [e for e in entries if e.record.id != victim.record.id]
        dropped.append(DroppedRecord(victim.record.id, "budget",
                                     f"over token_budget={token_budget}"))

    chars_kept = sum(len(e.record.content) for e in entries)
    return PromptContext(
        records=entries,
        dropped=dropped,
        tokens_estimate=_estimate_tokens(chars_kept, chars_per_token),
        chars_total=chars_total,
        chars_kept=chars_kept,
        drop_tail=drop_tail,
    )
