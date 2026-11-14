# SPDX-License-Identifier: Apache-2.0
"""Tier 0 谓词：零模型依赖、O(n)、逐谓词论证单侧性。

单侧性说明（见 SPEC §3.1）：
  verbatim        严格单侧：子串匹配不会把非子串判成子串。
  preserve_negation / preserve_temporal  条件单侧：词表漏检会产生假接受，依赖 gold set 度量 recall。
  preserve_speaker 严格单侧（元数据比较，不做语义推断）。

中文词法层（v4，契约 `docs/ALGO-V4-CONTRACT.md` §3.2）
────────────────────────────────────────────────────────────────────────────
`apply_obligations(..., zh=True)` 打开中文路径，**默认 `zh=False` → 本文件既有行为逐字节不变**：

  * `preserve_negation`：当 `zh=True` **且 scope/quote 任一侧含 CJK** 时，改为用
    `zh.zh_negated` 的**极性比较**（一侧否定、另一侧不否定 → 失败）；这一刻**不再**用英文
    `NEGATION` 正则去判中文文本——SQL/代码里的英文 `NOT` 出现在中文语料里时，会把一条
    逐字正确的引用判成"丢掉否定"（真实评测里的 2/19 假杀就是这个）。两侧都不含 CJK 时
    仍是原有英文逻辑（`zh=True` 也不动它）。
  * `preserve_temporal`：`zh=True` 时有 CJK 的一侧用 `zh.zh_temporal_terms` 归一，要求
    quote 的时间项能被 scope 的时间项**覆盖**（`zh.zh_temporal_subset`：证据更精确 OK、
    断言更精确不 OK）；两侧都是非 CJK 时仍是原有英文逻辑；一侧中文一侧英文（两套归一化
    不可比）按"无法提取"处理，不阻断——软谓词宁可漏也不编。
  * `preserve_speaker` / `verbatim` 不变（元数据比较与子串判定与语言无关）。

**方向性**（必须如实读）：中文否定/时间都是**标记表**检测，漏检 → 假接受（放过）；
误检 → 假杀。标记表只收封闭常量，无模型；`zh.py` 的模块 docstring 列了已抑制的同形词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from .ledger import AnchorRecord
from .zh import has_cjk, zh_hedge, zh_negated, zh_temporal_subset, zh_temporal_terms

NEGATION = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot|can't|won't|don't|"
    r"doesn't|didn't|isn't|wasn't|aren't|weren't|wouldn't|shouldn't|couldn't|"
    r"hardly|barely|without)\b",
    re.IGNORECASE,
)

_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
TEMPORAL = re.compile(
    rf"\b(?:"
    rf"{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\.?(?:\s*,?\s*\d{{4}})?"
    rf"|\d{{4}}"
    rf"|yesterday|today|tomorrow|tonight"
    rf"|last\s+(?:week|month|year|night|summer|winter|spring|fall)"
    rf"|next\s+(?:week|month|year)"
    rf"|this\s+(?:week|month|year|morning|afternoon|evening)"
    rf"|\d+\s+(?:day|week|month|year)s?\s+ago"
    rf")\b",
    re.IGNORECASE,
)

CONDITIONAL = re.compile(r"\b(?:if|when|unless|whenever|in case|as long as)\b", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# v4.7-meta（F1 Phase 1，`docs/F1-PHASE1.md`）：quantifier / source_type
# --------------------------------------------------------------------------- #
# 方向：两条都 **soft-only**（只产生 degraded）。quantifier 用"claim 有、证据没有"的单侧判据：
# 证据也有量词 → 不判（宁可漏，避免"单例证据 vs 全称断言"以外的误杀）。
QUANTIFIER_EN = re.compile(
    r"\b(?:all|every|always|any|each|everyone|everything|everybody)\b", re.IGNORECASE
)
QUANTIFIER_CN = ("所有", "每个", "每一", "任何", "总是", "一直", "全部", "一律", "永远", "从来")
#: 中文量词的否定前置守卫（"不是所有/并非每个"不是全称断言 → 不触发）
_QUANT_CN_NEG_WINDOW = 3
_QUANT_CN_NEG_MARKS = ("不", "没", "未", "非", "无")
#: 英文量词的否定前置守卫（"not all / no one always"）
_QUANT_EN_NEG = re.compile(r"\b(?:not|no|never|none)\b(?:\s+\w+){0,2}\s*$", re.IGNORECASE)

#: `AnchorRecord.speaker` → source_type 的封闭映射（未列出的 speaker 一律 unknown，不判）
SPEAKER_SOURCE_TYPE = {
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "system": "tool",
    "document": "document",
    "external": "document",
}
SOURCE_TYPES = ("user", "assistant", "tool", "document", "unknown")
#: 非 user 来源（`asserted_scope=user_fact` 时断言引它们 → `source_attribution` soft 失败）
NON_USER_SOURCES = frozenset({"assistant", "tool", "document"})

_SENT_END = re.compile(r'[.!?。！？]["\')\]]*\s')


@dataclass(frozen=True)
class PredicateResult:
    name: str
    passed: bool
    severity: str  # "hard" | "soft"
    detail: str = ""


class TemplateError(Exception):
    pass


#: 幽灵谓词清理（internal-task-internal-task）：`joint_consistent` 此前只在白名单里声明、**无任何实现**，
#: 已按裁决从白名单移除（写它现在显式 `TemplateError`，而不是"接受但静默不生效"）。
#: 若将来要做多证据一致性，须先给 gold 与实现，再重新加入本表。
ALLOWED_EXTRA = {"no_generalization"}
SEVERITY = {
    "verbatim": "hard",
    "preserve_negation": "hard",
    "preserve_speaker": "hard",
    "preserve_temporal": "soft",
    "no_generalization": "soft",
    # v4.7-meta（soft-only；不得产生 verified 资格）
    "quantifier_generalization": "soft",
    "source_attribution": "soft",
}


def derive_source_type(rec: AnchorRecord) -> str:
    """从既有记录元数据确定性派生来源类型（user/assistant/tool/document/unknown）。

    只用 `AnchorRecord.speaker`（封闭映射）；未列出的 speaker（含 `None`）→ `unknown`，
    谓词对其不判（宁可漏，不噪声）。无 I/O、无模型。
    """
    return SPEAKER_SOURCE_TYPE.get((rec.speaker or "").strip().lower(), "unknown")


def _has_negated_quantifier_prefix(text: str, start: int) -> bool:
    if start <= 0:
        return False
    window = text[max(0, start - _QUANT_CN_NEG_WINDOW):start]
    return any(mark in window for mark in _QUANT_CN_NEG_MARKS)


def quantifiers_in(text: str) -> set[str]:
    """抽取文本中的**未被否定**的量词标记（中文封闭表 + 英文词边界）。

    否定前置守卫：中文看前 3 字符是否含否定词；英文看前文 `not/no/never ...`。
    只做标记检测，不做语义强度比较（单侧：证据有量词就不判）。
    """
    found: set[str] = set()
    if not text:
        return found
    for marker in QUANTIFIER_CN:
        start = text.find(marker)
        while start != -1:
            if not _has_negated_quantifier_prefix(text, start):
                found.add(marker)
            start = text.find(marker, start + 1)
    for m in QUANTIFIER_EN.finditer(text):
        prefix = text[max(0, m.start() - 24):m.start()]
        if not _QUANT_EN_NEG.search(prefix):
            found.add(m.group(0).lower())
    return found


def quantifier_generalization(
    claim_text: str, evidence_texts: list[str] | tuple[str, ...]
) -> PredicateResult:
    """`quantifier` 元数据谓词（soft）：claim 出现量词而**全部证据**都没有 → 泛化降级。

    * 谓词名 `quantifier_generalization`，严重度 soft，映射 degrade_reason=`over_generalization`；
    * claim 无量词 / 证据也含量词 / 量词被否定 → passed（单侧：宁可漏，不假杀）；
    * 没有任何 verified 资格影响：`enforce()` 只在 soft 失败集合里使用它。
    """
    claim_q = quantifiers_in(claim_text or "")
    if not claim_q:
        return PredicateResult("quantifier_generalization", True, "soft", "no quantifier in claim")
    evidence_q: set[str] = set()
    for text in evidence_texts or ():
        evidence_q |= quantifiers_in(text or "")
    # 单侧判据：只有"claim 出现、而证据里**同一标记**没出现"的量词才算泛化；
    # 证据出现任一同名量词即视为已覆盖（宁可漏，不假杀；例："each time" 覆盖 "each"）。
    uncovered = claim_q - evidence_q
    if not uncovered:
        return PredicateResult("quantifier_generalization", True, "soft",
                               "quantifiers covered by evidence")
    return PredicateResult(
        "quantifier_generalization", False, "soft",
        f"claim has quantifier {sorted(uncovered)[:3]} but evidence lacks it",
    )


def sentence_bounds(text: str, offset: int) -> tuple[int, int]:
    """返回 offset 所在句子的 [start, end)。确定性、无模型。"""
    starts = [0] + [m.end() for m in _SENT_END.finditer(text)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        if start <= offset < end:
            return start, end
    return 0, len(text)


def obligation_scope(rec: AnchorRecord, offset: int, length: int, mode: str) -> str:
    """谓词的检查范围。

    mode="record"：整条锚定记录（最早的写法，粒度粗）
    mode="clause"：引用片段所在句子（SPEC 的结论：谓词范围应与引用粒度一致）
    """
    if mode == "record":
        return rec.content
    if mode == "clause":
        start, end = sentence_bounds(rec.content, offset)
        return rec.content[start:end]
    raise ValueError(f"unknown scope mode {mode!r}")


_ZH_CLAUSE_END = re.compile(r"[，,；;。！？!?\n]")


def zh_clause_scope(text: str, offset: int, length: int) -> str:
    """中文**子句级**范围：覆盖 [offset, offset+length) 的逗号/分号/句末片段。

    为什么不用整句：中文句子里常有"我没去杭州，但我朋友去了"这种结构，引用后半句时
    整句含否定会让极性比较假杀。子句级范围与引用粒度对齐后，只有"同一子句里被省略掉
    的否定"才算丢否定。
    """
    if not text:
        return ""
    start, end = 0, len(text)
    for m in _ZH_CLAUSE_END.finditer(text):
        if m.end() <= offset:
            start = m.end()
        else:
            break
    scan_from = min(len(text), offset + max(length, 0))
    for m in _ZH_CLAUSE_END.finditer(text, scan_from):
        end = m.start()
        break
    if end < scan_from:
        end = len(text)
    return text[start:end]


def _temporal_side(text: str, *, zh: bool) -> Tuple[str, Set[str]]:
    """单侧时间项抽取：含 CJK 的一侧走中文归一，否则走原有英文正则。

    返回 `(kind, terms)`；`kind` 用于判断两侧是否"同一套归一化"（不可比就不阻断）。
    """
    if zh and has_cjk(text):
        return "zh", zh_temporal_terms(text)
    return "en", set(TEMPORAL.findall(text))


def apply_obligations(
    rec: AnchorRecord,
    quote: str,
    offset: int,
    *,
    scope_mode: str = "clause",
    asserted_speaker: Optional[str] = None,
    extra: Optional[list[str]] = None,
    zh: bool = False,
    source_type: bool = False,
    asserted_scope: str = "unspecified",
) -> list[PredicateResult]:
    """Tier 0 谓词。`source_type`（v4.7-meta，默认 False）启用 `source_attribution` soft 谓词；
    `asserted_scope` 只在该开关打开时参与（缺省 `unspecified` → 谓词不判）。默认路径逐字节不变。
    """
    extra = list(extra or [])
    for name in extra:
        if name not in ALLOWED_EXTRA:
            raise TemplateError(f"unknown obligation {name!r}")

    results: list[PredicateResult] = []

    # 1) verbatim（硬）：引用必须是记录的连续子串，且落在 char_span 内
    end = offset + len(quote)
    in_bounds = 0 <= offset and end <= len(rec.content) and rec.char_span[0] <= offset
    exact = in_bounds and rec.content[offset:end] == quote
    results.append(
        PredicateResult(
            "verbatim",
            exact,
            "hard",
            "" if exact else f"quote is not the contiguous substring at [{offset},{end})",
        )
    )
    if not exact:
        return results  # 后面的谓词在非子串上无意义

    scope = obligation_scope(rec, offset, len(quote), scope_mode)

    # 2) preserve_negation（硬）
    #    zh 路径：任一侧含 CJK → 中文极性比较，**不用**英文 NEGATION 判中文文本
    #    （SQL 里的 NOT / if 出现在中文语料里时会把逐字引用误判成"丢否定"）。
    #    zh=False 或两侧都非 CJK → 与 v4 之前逐字节相同的英文逻辑。
    if zh and (has_cjk(scope) or has_cjk(quote)):
        # 极性比较用**子句级**范围（v4.1 修复）：整句范围会把"引用后半句、前半句有否定"
        # 误判成丢否定（真实评测里 FPR_strict 7/71→14/71 的主因）。
        neg_scope = zh_clause_scope(rec.content, offset, len(quote))
        scope_neg_zh = zh_negated(neg_scope)
        quote_neg_zh = zh_negated(quote)
        if scope_neg_zh != quote_neg_zh:
            results.append(
                PredicateResult(
                    "preserve_negation", False, "hard",
                    f"zh negation polarity mismatch (clause scope): "
                    f"scope negated={scope_neg_zh}, quote negated={quote_neg_zh}",
                )
            )
        else:
            results.append(PredicateResult("preserve_negation", True, "hard"))
    else:
        scope_neg = NEGATION.findall(scope)
        quote_neg = NEGATION.findall(quote)
        if scope_neg and not quote_neg:
            results.append(
                PredicateResult(
                    "preserve_negation", False, "hard", f"scope has negation {scope_neg}, quote drops it"
                )
            )
        else:
            results.append(PredicateResult("preserve_negation", True, "hard"))

    # 3) preserve_temporal（软）
    if zh:
        scope_kind, scope_t_zh = _temporal_side(scope, zh=True)
        quote_kind, quote_t_zh = _temporal_side(quote, zh=True)
        if scope_kind == "zh" and quote_kind == "zh":
            # quote 的时间项必须能被 scope 覆盖（证据更精确 OK，断言更精确不 OK）
            if scope_t_zh and not zh_temporal_subset(quote_t_zh, scope_t_zh):
                results.append(
                    PredicateResult(
                        "preserve_temporal", False, "soft",
                        f"zh temporal {sorted(quote_t_zh)[:2]} not covered by scope "
                        f"{sorted(scope_t_zh)[:2]}",
                    )
                )
            else:
                results.append(PredicateResult("preserve_temporal", True, "soft"))
        elif scope_kind == "zh" or quote_kind == "zh":
            # 一侧中文、一侧英文：两套归一化不可比 → 按"无法提取"处理，不阻断（软谓词）
            results.append(PredicateResult("preserve_temporal", True, "soft"))
        else:
            scope_t = TEMPORAL.findall(scope)
            quote_t = TEMPORAL.findall(quote)
            if scope_t and not quote_t:
                results.append(
                    PredicateResult(
                        "preserve_temporal", False, "soft",
                        f"scope has temporal {scope_t[:2]}, quote drops it",
                    )
                )
            else:
                results.append(PredicateResult("preserve_temporal", True, "soft"))
    else:
        scope_t = TEMPORAL.findall(scope)
        quote_t = TEMPORAL.findall(quote)
        if scope_t and not quote_t:
            results.append(
                PredicateResult(
                    "preserve_temporal", False, "soft", f"scope has temporal {scope_t[:2]}, quote drops it"
                )
            )
        else:
            results.append(PredicateResult("preserve_temporal", True, "soft"))

    # 4) preserve_speaker（硬，元数据比较）
    if asserted_speaker is not None:
        if rec.speaker is None or asserted_speaker != rec.speaker:
            results.append(
                PredicateResult(
                    "preserve_speaker",
                    False,
                    "hard",
                    f"asserted {asserted_speaker!r} but record speaker is {rec.speaker!r}",
                )
            )
        else:
            results.append(PredicateResult("preserve_speaker", True, "hard"))

    # 5) no_generalization（软，仅当模板显式追加，或表面存在条件标记）
    if "no_generalization" in extra or CONDITIONAL.search(scope):
        scope_c = CONDITIONAL.findall(scope)
        quote_c = CONDITIONAL.findall(quote)
        if scope_c and not quote_c:
            results.append(
                PredicateResult(
                    "no_generalization", False, "soft", f"scope has condition {scope_c}, quote drops it"
                )
            )
        else:
            results.append(PredicateResult("no_generalization", True, "soft"))

    # 6) source_attribution（软，v4.7-meta；默认关闭）
    #    条件：asserted_scope=="user_fact" ∧ 来源类型非 user ∧ 不落在 ass/hedge（后者由
    #    scenario_speaker_scope hard 负责，**同一缺陷不重复计罚**）。
    if source_type:
        source = derive_source_type(rec)
        if asserted_scope != "user_fact" or source not in NON_USER_SOURCES:
            results.append(PredicateResult("source_attribution", True, "soft",
                                           f"source={source}, asserted_scope={asserted_scope}"))
        elif source == "assistant" and zh_hedge(rec.content or ""):
            # assistant+hedge：交给 scenario_speaker_scope（hard）；此处只记录不重复失败
            results.append(PredicateResult(
                "source_attribution", True, "soft",
                "assistant+hedge is handled by scenario_speaker_scope (no double penalty)"))
        else:
            results.append(PredicateResult(
                "source_attribution", False, "soft",
                f"asserted as user_fact but evidence source_type={source}"))

    return results
