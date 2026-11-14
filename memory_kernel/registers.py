# SPDX-License-Identifier: Apache-2.0
"""输出层：4 语域 + 措辞变体，语域由执法结果确定性派生。

SPEC §4：emitted_register = min(register(verdict_semantic), register(coverage))。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .enforce import EnforcementResult

REGISTERS = ("fact", "hedged", "unknown", "speculation")
RANK = {"fact": 3, "hedged": 2, "unknown": 1, "speculation": 1}

TEMPLATES = {
    ("fact", "plain"): "根据记录，{text}",
    ("hedged", "degraded"): "我记得{text}，但这条在{missing}上可能有歧义。",
    ("hedged", "pending"): "我记得{text}，但这条我还没核实过。",
    ("unknown", "gap"): "这件事我不记得了。",
    ("unknown", "unverifiable"): "我给不出可核验的依据。",
    ("speculation", "experience"): "按我过去的经验推测，{text}（经验 {ref}）",
    ("speculation", "model"): "按我的知识推测，{text}（无记忆依据）",
}

_UNVERIFIABLE_CAUSES = {
    "verbatim",
    "invalid_anchor",
    "preserve_negation",
    "preserve_speaker",
    "skeleton",
    "budget_exhausted",
    "untrusted_writer",
}


@dataclass
class Rendered:
    register: str
    variant: str
    text: str
    reason: Optional[str] = None


def register_of(result: EnforcementResult) -> tuple[str, str]:
    """返回 (register, variant)。纯函数，不经过模型。"""
    full = bool(result.coverage.get("full_checks_ran"))

    if result.verdict == "verified":
        base = ("fact", "plain")
    elif result.verdict == "degraded":
        missing = "、".join(result.soft_failed) or "限定条件"
        base = ("hedged", "degraded")
        base = (base[0], base[1])
    else:  # rejected
        if result.rejected_cause in _UNVERIFIABLE_CAUSES:
            base = ("unknown", "unverifiable")
        else:
            base = ("unknown", "gap")

    if not full:
        # coverage 不完整 ⇒ 语域上限压到 hedged（§4.1 格序）
        if RANK.get(base[0], 0) > RANK["hedged"]:
            base = ("hedged", "pending")
        else:
            base = base
    return base


def render(result: EnforcementResult, text: str, *, missing: str = "限定条件") -> Rendered:
    register, variant = register_of(result)
    tpl = TEMPLATES[(register, variant)]
    rendered = tpl.format(text=text, missing=missing, ref="ref:experience/…")
    return Rendered(register=register, variant=variant, text=rendered, reason=result.rejected_cause)
