# SPDX-License-Identifier: Apache-2.0
"""修复链（internal-task · SPEC §3.4 前半段：重生成 → 收窄 → 重走执法 → 降级/放弃）。

设计契约：`docs/REPAIR-CHAIN-DESIGN.md`；实现映射：`docs/REPAIR-CHAIN-IMPL.md`。

边界（不可违反）
────────────────
1. **本模块是确定性纯函数层**：`decide_repair()` / `shape_check()` / 收窄算子 / 审计构造
   全部不读时钟、不联网、不调模型；唯一的"模型"是注入的 `regenerator(request) -> RepairResponse`，
   本模块源码不 import 任何模型/网络库（与 `pipeline.py` 同一红线）。
2. **不绕过 `enforce()`**：重生成/收窄的每个候选都必须走完整 `enforce()`（锚定→Tier0→Tier1→
   cite_set→scenario），并只以真实 verdict 接受；本模块不构造 `EnforcementResult`、不改默认策略。
3. **不新增语域**：修复只产出两种终态——`verified`（事实）、`degraded`（保留，真实 soft 判定）；
   救不回则沿用 `emit_answer` 既有 C 行为（删句/兜底，未知语域）。无判定依据的"保留"渲染
   本版不实现（见 IMPL 文档「未做项」，需 emit 受控渲染钩子 + 产品裁决）。
4. **失败关闭**：形状非法 / usage 缺失 / 预算触顶 / 回放缺失 / 生成器异常 → 不采用该次修复，
   回落首轮门控结果；绝不静默接受收窄结果。
5. **默认零行为变化**：`RepairPolicy.enabled=False` 或 `repair_policy=None` 时，
   本模块只读不写，`emit_answer`/`pipeline` 输出与改动前逐字节一致。
6. **不重复判定**：claim→`Claim` 的构造统一走 `emit.claim_from_segment()`（单点构造）。

状态机（claim 级）
──────────────────
`pending → regenerating → rechecking → (accepted | narrowing → rechecking) → accepted/exhausted`
终态：`verified` / `degraded` / `dropped`（= 沿用现状删除，审计记 `exhausted`）。

审计（P6）：`repair_attempts`（模型调用次数，缓存命中不计）+ `tokens_spent`（Σ usage）必填；
未尝试任何修复时二者写 `null` 并给 `repair_state` 与 reason。时间字段只作运行审计，
不参与确定性判据（回放时用 `clock=None`）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .emit import FALLBACK_TEXT, Segment, claim_from_segment, emit_answer
from .enforce import Claim, Citation, enforce
from .skeleton import SkeletonPolicy
from .zh import zh_negated

__all__ = [
    "DEFAULT_REPAIR_PROMPT_VERSION",
    "INSTRUCTION_IDS_OF_CAUSE",
    "NARROWABLE_CAUSES",
    "REGENERABLE_CAUSES",
    "REPAIR_POLICY_VERSION",
    "RepairPolicy",
    "RepairRequest",
    "RepairResponse",
    "RepairResult",
    "build_request",
    "decide_repair",
    "narrow_candidates",
    "run_repair",
    "shape_check",
]

REPAIR_POLICY_VERSION = "repair-v1"
DEFAULT_REPAIR_PROMPT_VERSION = "repair-v1"

#: 失败原因 → 封闭指令 ID（只传机器可读码，不传自由文本 detail；见设计 §4.2）
INSTRUCTION_IDS_OF_CAUSE: dict[str, tuple[str, ...]] = {
    "verbatim": ("fix_verbatim",),
    "preserve_negation": ("fix_negation",),
    "preserve_speaker": ("fix_speaker",),
    "skeleton": ("fix_scope",),
    "citation_set": ("fix_single_cite",),
    "scenario": ("fix_scenario",),
}

#: 允许送重生成的原因（无可核验对象的 cause 不送：invalid_anchor/search_empty/...）
REGENERABLE_CAUSES: frozenset[str] = frozenset(INSTRUCTION_IDS_OF_CAUSE)

#: 允许纯删除收窄的原因（删句改不了 polarity/speaker/场景时间论元 → 排除）
NARROWABLE_CAUSES: frozenset[str] = frozenset({"skeleton", "citation_set", "verbatim"})

#: 终态 → 四语域（不新增语域；`dropped` 走 emit 既有 C 行为 = 未知）
_TERMINAL_REGISTER = {"verified": "fact", "degraded": "hedged", "dropped": "unknown"}

#: 收窄算子固定顺序（N4 默认关；每个算子只删不改）
_NARROW_OP_ORDER: tuple[str, ...] = (
    "drop_supporting_cites",
    "drop_generality_marker",
    "drop_subordinate_clause",
    "drop_redundant_temporal_modifier",
)

#: N2 封闭泛化词表（不得含否定词；删后极性必须不变）
_GENERALITY_MARKERS: tuple[str, ...] = (
    "所有", "每个", "每一", "一定", "总是", "任何", "一律", "全部", "从来", "永远",
    "all", "every", "always", "any", "each",
)
#: N3 从属子句连接词（逗号/分号后的 clause 若以这些词开头才允许删）
_CONNECTIVE_HEADS: tuple[str, ...] = (
    "and", "but", "or", "which", "that", "who", "although", "because", "so",
    "而且", "但是", "但", "不过", "并且", "因为", "所以", "同时", "以及",
)
#: 单算子最多产出的候选数（确定性截断）
_MAX_CANDIDATES_PER_OP = 8

_EN_NEG = re.compile(r"\b(?:not|no|never|none|without|cannot)\b|n't", re.IGNORECASE)
_PAREN_RE = re.compile(r"[（(][^（）()]{1,200}[）)]")


# --------------------------------------------------------------------------- #
# 纯函数小工具
# --------------------------------------------------------------------------- #
def _canonical_json(obj) -> str:
    """规范 JSON（键排序、无空白）——用于 request/response 内容寻址与逐字节重放。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _negation_state(text: str) -> bool:
    """极性状态（中文否定表 + 英文否定词）。收窄前后必须相同，否则该候选作废。"""
    return bool(zh_negated(text)) or bool(_EN_NEG.search(text or ""))


def _tidy(text: str) -> str:
    """删除片段后的最小清洗：合并空白、清理悬挂标点。不改变词序、不生成新词。"""
    out = re.sub(r"[ \t　]+", " ", text or "").strip()
    out = re.sub(r"\s+([，,。.；;：:！!？?、])", r"\1", out)
    out = re.sub(r"([，,；;：:])\s*(?=[，,。.；;：:！!？?]|$)", "", out)
    return out.strip(" ，,；;：:、 \t　")


def _content_cites(claim: Claim) -> tuple[Citation, ...]:
    return (claim.citation, *claim.supporting)


def _segment_sha(seg) -> str:
    """segment 内容指纹（只覆盖 text+cites+speaker；用于审计 before/after 对照）。"""
    if isinstance(seg, Mapping):
        return _sha256(_canonical_json({
            "text": str(seg.get("text") or ""),
            "cites": [{"anchor": str(c.get("anchor") or c.get("anchor_id") or ""),
                       "quote": str(c.get("quote") or "")}
                      for c in (seg.get("cites") or ())],
            "speaker": seg.get("asserted_speaker"),
        }))
    return _sha256(_canonical_json({
        "text": seg.text,
        "cites": [{"anchor": c.anchor_id, "quote": c.quote} for c in seg.cites],
        "speaker": seg.asserted_speaker,
    }))


# --------------------------------------------------------------------------- #
# 策略（版本化；默认关闭 = 调用方 opt-in）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepairPolicy:
    """修复链策略（`repair-v1`）。默认 `enabled=False`：不启用时修复链不产生任何副作用。"""

    version: str = REPAIR_POLICY_VERSION
    enabled: bool = False
    max_claims: int = 1                  # 每篇最多修几个 claim
    max_attempts_per_claim: int = 1       # 每个 claim 最多几次重生成
    max_narrow_ops: int = 3               # 收窄链长度上限（按 _NARROW_OP_ORDER 计）
    max_total_tokens: int = 6000          # P6 tokens 预算（超即冻结）
    max_total_ms: Optional[float] = None  # 生产熔断；回放/确定性测试置 None（默认）
    narrow_ops: tuple[str, ...] = _NARROW_OP_ORDER[:3]  # N4 默认关
    terminal_degrade_causes: frozenset[str] = frozenset({"skeleton", "citation_set"})

    def __post_init__(self) -> None:
        if not isinstance(self.narrow_ops, tuple):
            object.__setattr__(self, "narrow_ops", tuple(self.narrow_ops))
        for name in self.narrow_ops:
            if name not in _NARROW_OP_ORDER:
                raise ValueError(f"unknown narrow op {name!r}")
        for name, value in (
            ("max_claims", self.max_claims),
            ("max_attempts_per_claim", self.max_attempts_per_claim),
            ("max_narrow_ops", self.max_narrow_ops),
            ("max_total_tokens", self.max_total_tokens),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.max_total_ms is not None and float(self.max_total_ms) <= 0:
            raise ValueError("max_total_ms must be > 0 or None")

    @classmethod
    def v1(cls, **overrides) -> "RepairPolicy":
        """正式 opt-in 构造器：`RepairPolicy.v1()` = enabled + repair-v1 默认预算。"""
        opts = dict(enabled=True)
        opts.update(overrides)
        return cls(**opts)


# --------------------------------------------------------------------------- #
# 请求 / 响应 / 结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepairRequest:
    """一次重生成请求（内容寻址：`sha256` 即回放键；见设计 §4.5）。"""

    claim_index: int
    question: str
    claim_text: str
    cites: tuple[dict, ...]                 # [{"anchor","quote"}]，声明顺序
    evidence_pool: tuple[tuple[str, str], ...]  # ((anchor_id, content))，冻结顺序
    rejected_cause: str
    failed_predicates: tuple[str, ...]
    instruction_ids: tuple[str, ...]
    attempt: int = 1
    repair_prompt_version: str = DEFAULT_REPAIR_PROMPT_VERSION

    def as_dict(self) -> dict:
        return {
            "claim_index": self.claim_index,
            "question": self.question,
            "claim": {"text": self.claim_text, "cites": list(self.cites)},
            "evidence_pool": [{"anchor": a, "content": c} for a, c in self.evidence_pool],
            "rejected_cause": self.rejected_cause,
            "failed_predicates": list(self.failed_predicates),
            "instruction_ids": list(self.instruction_ids),
            "attempt": self.attempt,
            "repair_prompt_version": self.repair_prompt_version,
        }

    def sha256(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


@dataclass(frozen=True)
class RepairResponse:
    """生成器返回（注入接口）。`usage` 必填（P6 `tokens_spent` 的唯一来源）。"""

    segments: tuple                 # 0/1 个 claim segment（mapping 或 emit.Segment）
    usage: Mapping[str, int] = field(default_factory=dict)
    model_id: str = ""

    def total_tokens(self) -> Optional[int]:
        try:
            prompt = int(self.usage.get("prompt_tokens", 0))
            completion = int(self.usage.get("completion_tokens", 0))
        except (TypeError, ValueError):
            return None
        total = prompt + completion
        if total <= 0:
            return None
        return total

    def sha256(self) -> str:
        segs = []
        for seg in self.segments or ():
            if isinstance(seg, Segment):
                segs.append({"type": seg.type, "text": seg.text,
                             "cites": [{"anchor": c.anchor_id, "quote": c.quote}
                                       for c in seg.cites]})
            elif isinstance(seg, Mapping):
                segs.append({"type": str(seg.get("type") or "claim"),
                             "text": str(seg.get("text") or ""),
                             "cites": [{"anchor": str(c.get("anchor") or c.get("anchor_id") or ""),
                                        "quote": str(c.get("quote") or "")}
                                       for c in (seg.get("cites") or ())]})
            else:
                segs.append({"type": "?invalid", "repr": repr(seg)})
        return _sha256(_canonical_json({
            "segments": segs,
            "usage": {str(k): int(v) for k, v in sorted((self.usage or {}).items())},
            "model_id": self.model_id,
        }))


@dataclass
class RepairResult:
    """`run_repair()` 返回：修复后的 segments + 最终门控结果 + P6 审计。"""

    segments: tuple
    audit: dict
    final_text: str
    final_audit: list[dict] = field(default_factory=list)

    @property
    def repair_state(self) -> str:
        return str((self.audit.get("document") or {}).get("repair_state"))

    @property
    def repaired_claims(self) -> int:
        return int((self.audit.get("document") or {}).get("repaired_claims", 0))


# --------------------------------------------------------------------------- #
# 决策层（纯函数）
# --------------------------------------------------------------------------- #
def decide_repair(baseline_audit: Sequence[Mapping], policy: RepairPolicy) -> dict:
    """由首轮 `emit_answer` 的 audit 决定"修不修、修哪几条"。

    纯函数；只依赖 audit 内容与 policy（不看时钟、不看模型）。
    返回 `{"state", "targets", "rejected_count", "truncated", "reason"}`。
    """
    if not policy.enabled:
        return {"state": "disabled", "targets": [], "rejected_count": 0,
                "truncated": 0, "reason": "policy_disabled"}
    rejected = [
        a for a in baseline_audit
        if a.get("type") == "claim" and a.get("state") == "rejected"
    ]
    if not rejected:
        return {"state": "not_needed", "targets": [], "rejected_count": 0,
                "truncated": 0, "reason": "no_rejected_claim"}
    targets = []
    for entry in rejected:
        cause = str(entry.get("rejected_cause") or "")
        if cause not in REGENERABLE_CAUSES and cause not in NARROWABLE_CAUSES:
            continue
        targets.append({
            "index": int(entry.get("index", -1)),
            "cause": cause,
            "failed": tuple(entry.get("failed") or ()),
        })
        if len(targets) >= policy.max_claims:
            break
    if not targets:
        return {"state": "ineligible", "targets": [], "rejected_count": len(rejected),
                "truncated": 0, "reason": "no_repairable_cause"}
    return {"state": "pending", "targets": targets, "rejected_count": len(rejected),
            "truncated": max(0, len(rejected) - len(targets)), "reason": ""}


def build_request(
    claim_index: int,
    claim: Claim,
    *,
    evidence_pool: Sequence[tuple[str, str]],
    cause: str,
    failed: Sequence[str],
    question: str = "",
    attempt: int = 1,
) -> RepairRequest:
    """构造确定性重生成请求（输入=原问题+冻结证据池+失败码指令 ID；不重检索）。"""
    instructions = INSTRUCTION_IDS_OF_CAUSE.get(cause, ())
    return RepairRequest(
        claim_index=int(claim_index),
        question=str(question or ""),
        claim_text=str(claim.text or ""),
        cites=tuple({"anchor": c.anchor_id, "quote": c.quote} for c in _content_cites(claim)),
        evidence_pool=tuple((str(a), str(c)) for a, c in evidence_pool),
        rejected_cause=str(cause),
        failed_predicates=tuple(str(p) for p in failed),
        instruction_ids=tuple(instructions),
        attempt=int(attempt),
    )


# --------------------------------------------------------------------------- #
# 形状预检（便宜、确定性；不能替代 enforce）
# --------------------------------------------------------------------------- #
def shape_check(
    response_segments: Sequence,
    *,
    evidence_pool: Sequence[tuple[str, str]],
    ledger,
) -> tuple[Optional[Segment], str]:
    """`RepairResponse.segments` → 单个合法 claim `Segment` 或 (None, reason)。

    规则（设计 §4.4 / §8.3）：
    * 只接受**恰好 1 个** claim；narrative/多余 segment 一律丢弃（由调用方审计）；
    * text 非空；cites 非空；anchor 必须在**冻结证据池**内；quote 必须是账本原文连续子串。
    """
    pool_ids = {a for a, _ in evidence_pool}
    claims = []
    for raw in response_segments or ():
        if isinstance(raw, Segment):
            if raw.type == "claim":
                claims.append(raw)
            continue
        if not isinstance(raw, Mapping):
            return None, "segment_not_mapping"
        if str(raw.get("type") or "claim") != "claim":
            continue
        claims.append(raw)
    if len(claims) != 1:
        return None, f"not_single_claim:{len(claims)}"
    seg = claims[0]
    if isinstance(seg, Mapping):
        text = str(seg.get("text") or "")
        speaker = seg.get("asserted_speaker")
        raw_cites = seg.get("cites") or ()
    else:
        text, speaker, raw_cites = seg.text, seg.asserted_speaker, seg.cites
    if not text.strip():
        return None, "empty_text"
    if not raw_cites:
        return None, "no_cites"
    cites: list[Citation] = []
    for raw in raw_cites:
        if isinstance(raw, Citation):
            anchor_id, quote, offset = raw.anchor_id, raw.quote, raw.offset
        elif isinstance(raw, Mapping):
            anchor_id = str(raw.get("anchor") or raw.get("anchor_id") or "")
            quote = str(raw.get("quote") or "")
            offset = raw.get("offset")
        else:
            return None, "cite_not_mapping"
        if anchor_id not in pool_ids:
            return None, "cite_out_of_pool"
        if not quote:
            return None, "empty_quote"
        try:
            rec = ledger.get(anchor_id)
        except Exception:
            return None, "cite_unresolved"
        if quote not in rec.content:
            return None, "quote_not_substring"
        if offset is None:
            offset = rec.content.find(quote)
        cites.append(Citation(anchor_id, int(offset), quote))
    return Segment("claim", text, tuple(cites), speaker), ""


# --------------------------------------------------------------------------- #
# 收窄算子（只删不改；每个候选都必须整套重执法）
# --------------------------------------------------------------------------- #
def _supporting_candidates(claim: Claim) -> list[tuple[Claim, dict]]:
    """N1：按固定顺序删 supporting（保 primary；先丢末位、再逐条向前）。"""
    out: list[tuple[Claim, dict]] = []
    n = len(claim.supporting)
    for k in range(n - 1, -1, -1):
        out.append((
            Claim(claim.citation, claim.asserted_speaker, claim.claim_type,
                  claim.text, tuple(claim.supporting[:k])),
            {"supporting": f"{n}->{k}"},
        ))
    return out


def _generality_candidates(claim: Claim) -> list[tuple[Claim, dict]]:
    """N2：删封闭泛化词（否定极性必须不变，否则该候选作废）。"""
    text = claim.text or ""
    out: list[tuple[Claim, dict]] = []
    n = len(_GENERALITY_MARKERS)
    for idx, marker in enumerate(_GENERALITY_MARKERS):
        if re.fullmatch(r"[A-Za-z][A-Za-z ]*", marker):
            m = re.search(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE)
            start, end = (m.start(), m.end()) if m else (-1, -1)
        else:
            start = text.find(marker)
            end = start + len(marker) if start >= 0 else -1
        if start < 0:
            continue
        removed = text[start:end]
        if _negation_state(removed):
            continue
        new_text = _tidy(text[:start] + " " + text[end:])
        if not new_text or new_text == text or _negation_state(new_text) != _negation_state(text):
            continue
        out.append((
            Claim(claim.citation, claim.asserted_speaker, claim.claim_type,
                  new_text, claim.supporting),
            {"marker": marker, "span": [start, end], "order": idx},
        ))
        if len(out) >= _MAX_CANDIDATES_PER_OP:
            break
    return out


def _subordinate_candidates(claim: Claim) -> list[tuple[Claim, dict]]:
    """N3：删括号/插入语、删逗号/分号后的连接词从属子句（主句必须保留）。"""
    text = claim.text or ""
    out: list[tuple[Claim, dict]] = []
    for m in _PAREN_RE.finditer(text):
        new_text = _tidy(text[:m.start()] + text[m.end():])
        if _valid_narrowed(text, new_text):
            out.append((
                Claim(claim.citation, claim.asserted_speaker, claim.claim_type,
                      new_text, claim.supporting),
                {"op": "parenthetical", "span": [m.start(), m.end()]},
            ))
        if len(out) >= _MAX_CANDIDATES_PER_OP:
            return out
    parts = re.split(r"([,，;；])", text)
    for i in range(2, len(parts), 2):
        clause = parts[i].strip()
        lowered = clause.lower()
        if not any(lowered.startswith(head) for head in _CONNECTIVE_HEADS):
            continue
        new_text = _tidy("".join(parts[: i - 1]) + "".join(parts[i + 1:]))
        if not _valid_narrowed(text, new_text):
            continue
        out.append((
            Claim(claim.citation, claim.asserted_speaker, claim.claim_type,
                  new_text, claim.supporting),
            {"op": "connective_clause", "clause": clause[:60]},
        ))
        if len(out) >= _MAX_CANDIDATES_PER_OP:
            break
    return out


def _valid_narrowed(old: str, new: str) -> bool:
    if not new or new == old or len(new) < 2:
        return False
    if _negation_state(new) != _negation_state(old):
        return False
    return True


def narrow_candidates(claim: Claim, op: str) -> list[tuple[Claim, dict]]:
    """按算子名生成**确定顺序**的收窄候选（纯函数；不做判定）。"""
    if op == "drop_supporting_cites":
        return _supporting_candidates(claim)
    if op == "drop_generality_marker":
        return _generality_candidates(claim)
    if op == "drop_subordinate_clause":
        return _subordinate_candidates(claim)
    if op == "drop_redundant_temporal_modifier":
        # N4 默认关闭（需产品裁决）；显式开启时也只认封闭表，本版返回空（未实现 → 不静默降级）
        return []
    raise ValueError(f"unknown narrow op {op!r}")


# --------------------------------------------------------------------------- #
# 编排层
# --------------------------------------------------------------------------- #
def _enforce(ledger, claim: Claim, policy: SkeletonPolicy, context) -> object:
    return enforce(ledger, claim, policy=policy, context=context)


def _failed_names(result) -> list[str]:
    return [p.name for p in result.failed if not p.passed]


def _baseline_counts(audit: Sequence[Mapping]) -> dict:
    counts = {"claims": 0, "verified": 0, "degraded": 0, "rejected": 0, "uncited": 0}
    for entry in audit:
        if entry.get("type") != "claim":
            continue
        counts["claims"] += 1
        state = str(entry.get("state") or "")
        if state in counts:
            counts[state] += 1
    return counts


def _default_evidence_pool(segments: Sequence, ledger) -> tuple[tuple[str, str], ...]:
    """缺省证据池 = 原文所有 cite 的 (anchor, content)，按首次出现顺序去重。"""
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    for seg in segments:
        cites = getattr(seg, "cites", None)
        if cites is None and isinstance(seg, Mapping):
            cites = tuple(
                (str(c.get("anchor") or c.get("anchor_id") or ""), str(c.get("quote") or ""))
                for c in (seg.get("cites") or ())
            )
        for cite in cites or ():
            anchor_id = cite.anchor_id if isinstance(cite, Citation) else str(cite[0])
            if anchor_id in seen:
                continue
            seen.add(anchor_id)
            try:
                pool.append((anchor_id, ledger.get(anchor_id).content))
            except Exception:
                continue
    return tuple(pool)


def _disabled_audit(final, baseline, reason: str) -> dict:
    return {
        "policy_version": REPAIR_POLICY_VERSION,
        "generator": {"injected": False, "model": None, "prompt": DEFAULT_REPAIR_PROMPT_VERSION},
        "enabled": False,
        "baseline": _baseline_counts(baseline.audit),
        "budget": {},
        "used": {"attempts": None, "tokens": None, "ms": None},
        "claims": [],
        "document": {"repaired_claims": 0, "recovered": 0, "degraded_claims": 0,
                     "dropped_claims": 0, "registers": _register_counts(final.audit),
                     "fallback_used": final.text == FALLBACK_TEXT,
                     "repair_state": "disabled", "reason": reason},
    }


def _register_counts(audit: Sequence[Mapping]) -> dict:
    out = {"fact": 0, "hedged": 0, "unknown": 0, "speculation": 0}
    for entry in audit:
        if entry.get("type") != "claim":
            continue
        state = str(entry.get("state") or "")
        if state == "verified":
            out["fact"] += 1
        elif state == "degraded":
            out["hedged"] += 1
        elif state == "rejected":
            out["unknown"] += 1
        elif state == "uncited":
            out["speculation"] += 1
    return out


def _repair_one_claim(
    *,
    index: int,
    segments: list,
    target: Mapping,
    ledger,
    policy: SkeletonPolicy,
    context,
    repair_policy: RepairPolicy,
    regenerator,
    evidence_pool: Sequence[tuple[str, str]],
    question: str,
    replay,
    budget: dict,
) -> tuple[Optional[Segment], dict]:
    """修一个 claim：重生成（有生成器时）→ 收窄 → 终态。返回 (replacement|None, audit entry)。"""
    seg = segments[index]
    claim = claim_from_segment(seg, ledger=ledger)
    initial = _enforce(ledger, claim, policy, context)
    entry: dict = {
        "index": index,
        "segment_sha256_before": _segment_sha(seg),
        "initial": {
            "verdict": initial.verdict,
            "cause": initial.rejected_cause,
            "failed": _failed_names(initial),
        },
        "attempts": [],
        "terminal": None,
        "terminal_register": None,
        "regenerated": False,
        "narrowed_from": None,
    }
    current = claim
    accepted_result = None

    # ---- 阶段 A：重生成 ----
    cause = str(target.get("cause") or "")
    if (regenerator is not None or replay is not None) and cause in REGENERABLE_CAUSES:
        for attempt in range(1, repair_policy.max_attempts_per_claim + 1):
            request = build_request(
                index, claim, evidence_pool=evidence_pool, cause=cause,
                failed=entry["initial"]["failed"], question=question, attempt=attempt,
            )
            attempt_rec: dict = {
                "step": "regenerate", "attempt": attempt,
                "request_sha256": request.sha256(),
                "instruction_ids": list(request.instruction_ids),
                "response_sha256": None, "usage": None, "ms": 0.0,
                "shape_ok": False, "shape_reason": None, "cached": False,
                "verdict_after": None, "cause_after": None, "failed_after": None,
                "accepted": False,
            }
            try:
                if replay is not None:
                    response = replay.get(request.sha256())
                    attempt_rec["cached"] = True
                    if response is None:
                        attempt_rec["shape_reason"] = "no_replay_response"
                        entry["attempts"].append(attempt_rec)
                        break
                else:
                    response = regenerator(request)
                if not isinstance(response, RepairResponse):
                    raise TypeError(f"regenerator must return RepairResponse, got {type(response)!r}")
            except Exception as exc:  # 生成器异常/超时：失败关闭
                attempt_rec["shape_reason"] = f"regenerator_error:{type(exc).__name__}"
                budget["attempts"] += 1
                entry["attempts"].append(attempt_rec)
                break
            tokens = response.total_tokens()
            attempt_rec["response_sha256"] = response.sha256()
            attempt_rec["usage"] = dict(response.usage or {})
            # P6：attempts = 实际模型调用次数（缓存命中不计）；形状失败照样计一次调用
            if not attempt_rec["cached"]:
                budget["attempts"] += 1
            if tokens is None:
                attempt_rec["shape_reason"] = "usage_missing"
                entry["attempts"].append(attempt_rec)
                break
            if budget["tokens"] + tokens > repair_policy.max_total_tokens:
                attempt_rec["shape_reason"] = "budget_tokens"
                entry["attempts"].append(attempt_rec)
                budget["budget_blocked"] = True
                break
            budget["tokens"] += tokens
            shape_seg, reason = shape_check(
                response.segments, evidence_pool=evidence_pool, ledger=ledger)
            attempt_rec["shape_ok"] = shape_seg is not None
            attempt_rec["shape_reason"] = reason or None
            if shape_seg is None:
                entry["attempts"].append(attempt_rec)
                continue
            candidate = claim_from_segment(shape_seg)
            result = _enforce(ledger, candidate, policy, context)
            attempt_rec["verdict_after"] = result.verdict
            attempt_rec["cause_after"] = result.rejected_cause
            attempt_rec["failed_after"] = _failed_names(result)
            entry["attempts"].append(attempt_rec)
            current = candidate
            entry["regenerated"] = True
            if result.verdict != "rejected":
                attempt_rec["accepted"] = True
                accepted_result = result
                break
            # 重生成仍 rejected → 用再生文本继续收窄（§2.1 的 `regenerating → rechecking`）

    # ---- 阶段 B：收窄（只删不改；每个候选整套重执法）----
    if accepted_result is None and cause in NARROWABLE_CAUSES:
        ops_done = 0
        for op in repair_policy.narrow_ops:
            if ops_done >= repair_policy.max_narrow_ops:
                break
            ops_done += 1
            for candidate, detail in narrow_candidates(current, op):
                result = _enforce(ledger, candidate, policy, context)
                attempt_rec = {
                    "step": "narrow", "op": op, "detail": detail,
                    "policy_version": str(result.coverage.get("policy_version")),
                    "skeleton_version": policy.version,
                    "full_checks_ran": bool(result.coverage.get("full_checks_ran")),
                    "verdict_after": result.verdict,
                    "cause_after": result.rejected_cause,
                    "failed_after": _failed_names(result),
                    "accepted": False,
                }
                entry["attempts"].append(attempt_rec)
                if result.verdict != "rejected":
                    attempt_rec["accepted"] = True
                    current = candidate
                    accepted_result = result
                    break
            if accepted_result is not None:
                break

    # ---- 终态 ----
    if accepted_result is not None:
        terminal = accepted_result.verdict
        entry["terminal"] = terminal
        entry["terminal_register"] = _TERMINAL_REGISTER[terminal]
        entry["narrowed_from"] = {
            "text_sha256": _sha256(claim.text or ""),
            "cites": [c.anchor_id for c in _content_cites(claim)],
        }
        replacement = Segment("claim", current.text, tuple(_content_cites(current)),
                              current.asserted_speaker)
        return replacement, entry

    entry["terminal"] = "dropped"
    entry["terminal_register"] = _TERMINAL_REGISTER["dropped"]
    return None, entry


def run_repair(
    segments: Sequence,
    *,
    ledger,
    policy: Optional[SkeletonPolicy] = None,
    context=None,
    repair_policy: Optional[RepairPolicy] = None,
    regenerator: Optional[Callable[[RepairRequest], RepairResponse]] = None,
    question: str = "",
    evidence_pool: Optional[Sequence[tuple[str, str]]] = None,
    replay: Optional[Mapping[str, RepairResponse]] = None,
    clock: Optional[Callable[[], float]] = None,
) -> RepairResult:
    """修复链入口（确定性编排；模型只能经注入 `regenerator` 出现）。

    `replay`：request.sha256 → RepairResponse 的冻结映射（评测/单测重放；缺失该请求 = 不修）。
    `clock`：只在给出时记录 `ms`（运行审计）；确定性重放请传 `None`（默认）。
    """
    pol = policy or SkeletonPolicy()
    rp = repair_policy or RepairPolicy()
    seg_list = list(segments)
    baseline = emit_answer(seg_list, ledger, policy=pol, context=context)

    if not rp.enabled:
        audit = _disabled_audit(baseline, baseline, "policy_disabled")
        return RepairResult(tuple(seg_list), audit, baseline.text, baseline.audit)

    started = clock() if clock is not None else None
    decision = decide_repair(baseline.audit, rp)
    if decision["state"] != "pending":
        audit = _disabled_audit(baseline, baseline, str(decision.get("reason") or decision["state"]))
        audit["enabled"] = True
        audit["document"].update({
            "repair_state": decision["state"],
            "rejected_count": decision["rejected_count"],
        })
        return RepairResult(tuple(seg_list), audit, baseline.text, baseline.audit)

    if evidence_pool is None:
        evidence_pool = _default_evidence_pool(seg_list, ledger)

    budget = {"attempts": 0, "tokens": 0, "budget_blocked": False}
    claim_audits: list[dict] = []
    coverage_incomplete = False

    for target in decision["targets"]:
        index = int(target["index"])
        if index < 0 or index >= len(seg_list):
            continue
        seg = seg_list[index]
        try:
            probe = claim_from_segment(seg, ledger=ledger)
            probe_result = _enforce(ledger, probe, pol, context)
        except Exception:
            claim_audits.append({
                "index": index, "segment_sha256_before": _segment_sha(seg),
                "initial": {"verdict": "rejected", "cause": target.get("cause"),
                            "failed": list(target.get("failed") or ())},
                "attempts": [], "terminal": "dropped", "terminal_register": "unknown",
                "regenerated": False, "narrowed_from": None,
                "skipped": "claim_unbuildable",
            })
            continue
        if not probe_result.coverage.get("full_checks_ran"):
            coverage_incomplete = True
            claim_audits.append({
                "index": index, "segment_sha256_before": _segment_sha(seg),
                "initial": {"verdict": probe_result.verdict,
                            "cause": probe_result.rejected_cause,
                            "failed": _failed_names(probe_result)},
                "attempts": [], "terminal": "dropped", "terminal_register": "unknown",
                "regenerated": False, "narrowed_from": None,
                "skipped": "coverage_incomplete",
            })
            continue
        if rp.max_total_ms is not None and started is not None and clock is not None:
            if (clock() - started) * 1000.0 > rp.max_total_ms:
                budget["budget_blocked"] = True
                break
        replacement, claim_audit = _repair_one_claim(
            index=index, segments=seg_list, target=target, ledger=ledger, policy=pol,
            context=context, repair_policy=rp, regenerator=regenerator,
            evidence_pool=evidence_pool, question=question, replay=replay, budget=budget,
        )
        claim_audits.append(claim_audit)
        if replacement is not None:
            seg_list[index] = replacement
        if budget.get("budget_blocked"):
            break

    final = emit_answer(seg_list, ledger, policy=pol, context=context)
    accepted = [c for c in claim_audits if c.get("terminal") in ("verified", "degraded")]
    dropped = [c for c in claim_audits if c.get("terminal") == "dropped"]
    if coverage_incomplete:
        state = "ineligible"
        reason = "coverage_incomplete"
    elif budget.get("budget_blocked"):
        state = "exhausted"
        reason = "budget_exhausted"
    elif not dropped:
        state = "complete"
        reason = ""
    elif accepted:
        state = "partial"
        reason = "some_claims_dropped"
    else:
        state = "exhausted"
        reason = "all_claims_exhausted"

    ms = round((clock() - started) * 1000.0, 4) if (started is not None and clock is not None) else None
    audit = {
        "policy_version": rp.version,
        "generator": {
            "injected": regenerator is not None,
            "model": None,
            "prompt": DEFAULT_REPAIR_PROMPT_VERSION,
        },
        "enabled": True,
        "baseline": _baseline_counts(baseline.audit),
        "budget": {
            "max_claims": rp.max_claims,
            "max_attempts_per_claim": rp.max_attempts_per_claim,
            "max_narrow_ops": rp.max_narrow_ops,
            "max_total_tokens": rp.max_total_tokens,
            "max_total_ms": rp.max_total_ms,
        },
        "used": {
            "attempts": int(budget["attempts"]),
            "tokens": int(budget["tokens"]),
            "ms": ms,
        },
        "claims": claim_audits,
        "document": {
            "repaired_claims": len(accepted),
            "recovered": len(accepted),
            "degraded_claims": sum(1 for c in accepted if c.get("terminal") == "degraded"),
            "dropped_claims": len(dropped),
            "registers": _register_counts(final.audit),
            "fallback_used": final.text == FALLBACK_TEXT,
            "repair_state": state,
            "reason": reason,
            "rejected_count": decision["rejected_count"],
            "truncated_targets": decision["truncated"],
        },
    }
    return RepairResult(tuple(seg_list), audit, final.text, final.audit)
