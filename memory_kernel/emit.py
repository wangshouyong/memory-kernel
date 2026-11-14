# SPDX-License-Identifier: Apache-2.0
"""阶段三：发射门控与强制重写（三阶段管线；契约 `docs/EMISSION-CONTRACT.md` §3）。

**绝不把模型原始输出直接透传给用户**：每个 claim segment 先过 `enforce()`，
再按判定重写：

  A verified            → 原样保留（锚点标记保留在文本里）
  B degraded            → 加 hedged 前缀（按 degrade_reason 细分措辞）
  C rejected            → 删除该 segment；若最终无任何 claim 且无 narrative → 兜底句
  D cites == []（无引用）→ 加「以下是我的个人推测：」前缀

narrative segment 原样保留（不参与执法）；`audit` 返回每条 claim 的判定详情，
**不回显给用户**。本模块不调用模型、无随机、无 I/O。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .enforce import Claim, Citation, enforce
from .skeleton import SkeletonPolicy

__all__ = [
    "FALLBACK_TEXT",
    "REDACTION_TEXT",
    "HEDGED_BY_REASON",
    "DEFAULT_HEDGED",
    "SPECULATION_PREFIX",
    "EmitResult",
    "Segment",
    "claim_from_segment",
    "emit_answer",
    "hedged_text",
]

SPECULATION_PREFIX = "以下是我的个人推测："
FALLBACK_TEXT = "这件事我无法给出可核验的依据。"
#: 删除 claim 后用于保住上下文连贯性的省略标记（放在原句位置；首尾会被清理）。
REDACTION_TEXT = "（此处省略一句缺少可核验依据的内容）"

# 用户裁决（v4.6）：B（保留但降级）按**原因分别措辞**，不用统一长前缀。
# 句式：断言原句 + 括号内的短原因说明（按 degrade_reason 选）。
HEDGED_BY_REASON: dict[str, str] = {
    "over_citation": "{text}（这条引用有冗余/不直接相关，仅供参考）",
    "scenario_superseded": "{text}（这条后来被更正过，可能已过时）",
    "argument_mismatch": "{text}（主体/谓词/客体对不上，仅供参考）",
    "partial_evidence": "{text}（证据不完整，仅供参考）",
    "cross_session": "{text}（依据不在当前会话内，仅供参考）",
    "temporal": "{text}（时间上可能有出入，仅供参考）",
    "over_generalization": "{text}（说得过于笼统，仅供参考）",
    # Plan Z C3：证据被 crypto-shred 销毁 / manifest 缺失（可后续由产品文案微调）
    "evidence_destroyed": "{text}（这段记录已按删除权销毁，无法核验，仅供参考）",
    "evidence_missing": "{text}（这段记录在账本中不可达，无法核验）",
    # Plan Z C4：冷归档超时 / 不可达
    "evidence_archived": "{text}（依据在冷归档中暂不可达，无法即时核验，仅供参考）",
    "evidence_unreachable": "{text}（依据暂不可达，无法核验，仅供参考）",
    # Plan Z 两态（SPEC §15 D-TS-3）：historical_recall 下被更正的旧状态本身是答案，
    # 不能按"已过时"措辞；只换标签，不改判定（既有语域复用，不新增第 5 语域）。
    "historical_record": "{text}（这是历史记录，当前状态可能已不同，仅供参考）",
    # v4.7-meta（F1 Phase 1）：user_fact 断言引用了非用户来源（助手/外部）的内容
    "source_attribution": "{text}（这条依据不是用户本人说的，仅供参考）",
}
DEFAULT_HEDGED = "{text}（依据不足，仅供参考）"


@dataclass(frozen=True)
class Segment:
    """模型输出的一段：narrative 原样文本，或带引用的 claim。"""

    type: str
    text: str
    cites: tuple[Citation, ...] = ()
    asserted_speaker: Optional[str] = None


@dataclass
class EmitResult:
    text: str
    audit: list[dict] = field(default_factory=list)


def _coerce_citation(ledger, raw) -> Citation:
    if isinstance(raw, Citation):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"citation must be Citation or mapping, got {type(raw)!r}")
    anchor_id = str(raw.get("anchor") or raw.get("anchor_id") or "")
    quote = str(raw.get("quote") or "")
    offset = raw.get("offset")
    if offset is None:
        try:
            offset = ledger.get(anchor_id).content.find(quote)
        except Exception:
            offset = -1
        if offset is None or offset < 0:
            offset = 0
    return Citation(anchor_id, int(offset), quote)


def _coerce_segment(ledger, raw) -> Segment:
    if isinstance(raw, Segment):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"segment must be Segment or mapping, got {type(raw)!r}")
    stype = str(raw.get("type") or "claim")
    cites = tuple(_coerce_citation(ledger, c) for c in (raw.get("cites") or ()))
    return Segment(stype, str(raw.get("text") or ""), cites, raw.get("asserted_speaker"))


def claim_from_segment(seg, *, ledger=None) -> Claim:
    """由 claim segment 构造 `Claim`（**单点构造**，emit/repair 共用，避免双路径）。

    契约（internal-task）：`primary = seg.cites[0]`、`supporting = seg.cites[1:]`、
    `asserted_speaker`、`text` 原样透传。仅在 `seg.cites` 非空时允许调用
    （无引用 segment 走 speculation 路径，不构造 Claim）。

    `seg` 可为 `Segment` 或 `segments-v1` 的 mapping；mapping 的 cite 需要 `ledger`
    才能补 `offset`（与 emit 既有 `_coerce_citation` 同一条路径，不另写一套）。
    """
    if isinstance(seg, Mapping):
        if str(seg.get("type") or "claim") != "claim":
            raise ValueError("claim_from_segment only accepts claim segments")
        seg = Segment("claim", str(seg.get("text") or ""), tuple(seg.get("cites") or ()),
                      seg.get("asserted_speaker"))
    cites = tuple(seg.cites or ())
    if not cites:
        raise ValueError("claim_from_segment requires at least one cite")
    if any(not isinstance(c, Citation) for c in cites):
        if ledger is None:
            raise ValueError("claim_from_segment needs ledger to coerce raw citations")
        cites = tuple(_coerce_citation(ledger, c) for c in cites)
    return Claim(citation=cites[0], text=seg.text, asserted_speaker=seg.asserted_speaker,
                 supporting=tuple(cites[1:]))


def hedged_text(text: str, degrade_reason: Optional[str]) -> str:
    """B（保留但降级）的措辞：按 degrade_reason 选短原因说明，不用统一长前缀。"""
    tpl = HEDGED_BY_REASON.get(degrade_reason or "", DEFAULT_HEDGED)
    stripped = text.strip().rstrip("。．.!！?？；;，,、 　")
    return tpl.format(text=stripped)


def emit_answer(
    segments: Sequence,
    ledger,
    *,
    policy: Optional[SkeletonPolicy] = None,
    context=None,
    redact: bool = True,
    redaction_text: str = REDACTION_TEXT,
    resolver=None,
    claim_mode: Optional[str] = None,
    idf_model=None,
    cache=None,
) -> EmitResult:
    """阶段三入口：逐条执法 + 强制重写。

    `policy` 缺省为 `SkeletonPolicy()`（v2）；中文场景请显式传 `SkeletonPolicy.v4()`，
    否则中文否定/引用集/场景谓词不会生效。

    `redact`：C（rejected）删句时是否在原句位放省略标记（默认 True），避免上下文断层；
    标记出现在首/尾或连续出现时会被清理。

    `resolver`（Plan Z C4）：冷归档证据解析 callable，原样透传给 `enforce()`；`None`（默认）
    时行为与 C4 之前逐字节一致。解析失败时该 claim 记为 hedged（degrade），措辞见
    `HEDGED_BY_REASON` 的 `evidence_*` 条目。

    `claim_mode`（Plan Z 两态）：原样透传给 `enforce()`；`None`（默认）时由
    `context.claim_mode` 决定（再缺省 = `unspecified`，行为逐字节不变）。

    `idf_model`（internal-task）：动态 IDF 快照里的 `IDFModel`；`None`（默认）时
    `enforce()` 仍按既有路径从账本现算并缓存 → 默认路径逐字节不变。

    `cache`（internal-task）：显式 `EngineCache`，原样透传给 `enforce()`；本模块**不持有**
    任何跨请求状态/缓存。`None`（默认）→ 进程级兼容默认缓存（结果逐字节不变）。
    """
    pol = policy or SkeletonPolicy()
    pieces: list[str] = []
    audit: list[dict] = []
    kept_claims = 0
    has_narrative = False

    for idx, raw in enumerate(segments):
        seg = _coerce_segment(ledger, raw)
        if seg.type == "narrative":
            if seg.text.strip():
                pieces.append(seg.text.strip())
                has_narrative = True
            audit.append({"index": idx, "type": "narrative", "action": "keep"})
            continue

        # claim
        if not seg.cites:
            pieces.append(f"{SPECULATION_PREFIX}{seg.text.strip()}")
            kept_claims += 1
            audit.append({"index": idx, "type": "claim", "action": "speculation",
                          "verdict": None, "degrade_reason": None,
                          "rejected_cause": None, "anchors": [],
                          # 无引用 = 未验证（调用方契约：不得当事实展示）
                          "state": "uncited"})
            continue

        primary = seg.cites[0]
        claim = claim_from_segment(seg, ledger=ledger)
        result = enforce(ledger, claim, policy=pol, context=context, resolver=resolver,
                         claim_mode=claim_mode, idf_model=idf_model, cache=cache)
        anchors = [c.anchor_id for c in seg.cites]
        entry = {
            "index": idx, "type": "claim", "verdict": result.verdict,
            "degrade_reason": getattr(result, "degrade_reason", None),
            "rejected_cause": result.rejected_cause,
            "anchors": anchors,
            "state": result.verdict,  # verified / degraded / rejected
            # internal-task（附加字段）：失败谓词名（只含**未通过**的谓词；固定顺序由 enforce 保证）。
            # 既有字段含义不变；消费方不得把它当"全部谓词"用。
            "failed": [p.name for p in result.failed if not p.passed],
        }
        if result.verdict == "verified":
            pieces.append(seg.text.strip())
            kept_claims += 1
            entry["action"] = "keep"
        elif result.verdict == "degraded":
            pieces.append(hedged_text(seg.text.strip(), getattr(result, "degrade_reason", None)))
            kept_claims += 1
            entry["action"] = "hedged"
        else:  # rejected
            # 删句会切断上下文：在句位放一个省略标记，保留前后衔接（首尾/连续标记会被清理）
            if redact:
                pieces.append(redaction_text)
            entry["action"] = "drop"
        audit.append(entry)

    if kept_claims == 0 and not has_narrative:
        text = FALLBACK_TEXT
    else:
        cleaned = _clean_pieces(pieces, redaction_text)
        text = "\n".join(cleaned) if cleaned else FALLBACK_TEXT
    return EmitResult(text=text, audit=audit)


def _clean_pieces(pieces: list[str], redaction_text: str) -> list[str]:
    """清理省略标记：去掉首尾标记、合并连续标记；其余顺序不变。"""
    out: list[str] = []
    for piece in pieces:
        if piece == redaction_text:
            if not out or out[-1] == redaction_text:
                continue
            out.append(piece)
        else:
            if piece:
                out.append(piece)
    while out and out[-1] == redaction_text:
        out.pop()
    return out
