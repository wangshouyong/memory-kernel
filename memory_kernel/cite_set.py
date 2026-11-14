# SPDX-License-Identifier: Apache-2.0
r"""引用集校验（v4 能力层；契约 `docs/ALGO-V4-CONTRACT.md` §4）。

为什么需要这一层
────────────────────────────────────────────────────────────────────────────
Tier 0 的 `verbatim` 只校验**主引用**（`claim.citation`）。`claim.supporting` 里的每一条
附加引用在 v4 之前**从未被逐条校验**：它们可以是别的记录、可以是与断言无关的记录，
甚至可以是引错位置的片段，判定仍然 `verified`。真实评测里"过度引用 0/19 可见"就是这个缺口
（评测报告见 `docs/ALGO-V4-CONTRACT.md` §4 与 v4 评测记录）。本模块把 supporting 逐条拆成三个确定性检查：

| 谓词 | 条件 | 严重度 | 为什么是这个严重度 |
|---|---|---|---|
| `over_citation_anchor` | anchor 不存在或不可验证（记录被改/签名不符） | hard | 引用指向不存在的证据 = 伪造，与 Tier 0 的 `invalid_anchor` 同性质 |
| `over_citation_verbatim` | `quote` 不是该记录在 `offset` 处的连续子串 | hard | 引文对不上原文 = 伪造，与 Tier 0 的 `verbatim` 同性质 |
| `over_citation` | 该记录**没有为 claim 贡献任何「主引用/先前已接受支持」未覆盖的内容词**（增量覆盖为 0） | soft | 只说明"这条证据对断言是冗余/不支撑"（过度引用），不能证明引用伪造 |

`hard` 让 `enforce()` 直接 `rejected`（cause `citation_set`），`soft` 只让判定降级为 `degraded`。
这与 Tier 0/Tier 1 的严重度分工完全一致：**伪造是硬失败，不充分是软失败**。

v4.3 起相关性判据从"与 claim 有交集"升级为**增量覆盖**：主引用已经覆盖的 claim 内容词，
再挂同样的引用不产生任何新证据 → 判为过度引用。理由：旧判据下只要无关记录与 claim 共享一个
常见词就放行（真实评测 O_overcite 只抓到 3/19）；增量覆盖判据对"主引用已足够却额外挂一条"的
冗余引用是直接命中的，且对"需要两条证据"的合法多引用不吃亏（第二条会贡献新词）。

单侧性（必须如实读）
────────────────────────────────────────────────────────────────────────────
* 相关性检查是**词面/增量**检查，不做语义判等（公理 B：执法路径不调用模型）。它能抓
  "完全冗余"和"词面不交集"的过度引用，但**抓不到**"同话题、共享词、事实不同"的无关引用
  （假阴性方向）；报告里必须按此口径读召回率。
* 增量覆盖只看**claim.text 的内容词**：claim 文本为空（摘录型）→ 不可判，soft 记 passed。
* `zh=False` 时中文 claim 的 `content_tokens` 为空集 ⇒ 任何 supporting 都会被判"无增量"（soft）。
  这是 v4 开关组合的已知单侧性：`cite_set` 应与 `zh=True` 一起用（`SkeletonPolicy.v4()` 的组合）；
  单独打开 `cite_set=True` 只对英文断言有意义。行为已由测试锁住，不是隐藏缺陷。
* 主引用记录取不到（anchor 不可验证）时 `covered` 为空 → 支持引用更容易被判"有增量"（保守方向）；
  主引用本身已由 Tier 0 硬失败，不在这里重复报错。
* 空 `quote` 在子串语义下平凡通过（`content[o:o] == ""`），本模块不额外发明"空引用"规则。
* token 空间用 `content_tokens(..., stop_mode=policy.stop_mode, zh=zh)`（**不做词干化**）：
  契约 §4 字面就是 content tokens；词干化会把 `cars`/`car` 也算相关，等于进一步放松相关性。

确定性
────────────────────────────────────────────────────────────────────────────
按 `claim.supporting` 的**声明顺序**逐条处理，结果顺序固定；无时间/随机/模型/网络/I-O。
"""

from __future__ import annotations

from .ledger import RecordVerificationError
from .obligations import PredicateResult
from .skeleton import SkeletonPolicy, content_tokens

__all__ = ["OVER_CITATION_PREDICATES", "check_citation_set"]

# 本模块可能产出的谓词名（固定顺序；报告层的 failed_predicates 直接用这些名字）
OVER_CITATION_PREDICATES: tuple[str, ...] = (
    "over_citation_anchor",
    "over_citation_verbatim",
    "over_citation",
)


def _quote_at(rec, offset: int, quote: str) -> bool:
    """`quote` 是否是 `rec.content` 在 `offset` 处的连续子串（与 Tier 0 verbatim 同口径）。

    只做内容边界 + 相等判定；`char_span` 是**来源**区间，Tier 0 对主引用额外要求
    `char_span[0] <= offset`，这里不重复那一条（supporting 只回答"引文在这一页的哪个位置"）。
    """
    end = offset + len(quote)
    if offset < 0 or end > len(rec.content):
        return False
    return rec.content[offset:end] == quote


def _tokens(text: str, *, zh: bool, policy: SkeletonPolicy) -> set[str]:
    return content_tokens(text, stop_mode=policy.stop_mode, zh=zh)


def check_citation_set(
    ledger,
    claim,
    *,
    zh: bool = False,
    policy: SkeletonPolicy,
) -> list[PredicateResult]:
    """逐条校验 `claim.supporting`。无 supporting 时返回空列表（不改变现状）。

    每条 supporting 按固定顺序命中**第一个**决定性谓词（anchor → verbatim → 相关性），
    因此返回列表长度 ≤ len(claim.supporting)，顺序与声明顺序一致：

      * hard 失败（`over_citation_anchor` / `over_citation_verbatim`）→ `passed=False, severity="hard"`；
      * 相关性通过/不可判 → `over_citation` `passed=True, severity="soft"`；
      * 相关性交集为 0 → `over_citation` `passed=False, severity="soft"`。

    `zh` 决定 token 空间（中文 bigram），`policy.stop_mode` 决定停用词表；`enforce()` 传
    `zh=policy.zh` 保证与 Tier 1 使用同一套分词口径。
    """
    supporting = tuple(getattr(claim, "supporting", ()) or ())
    if not supporting:
        return []

    claim_text = claim.text or ""
    # None = 相关性不可判（不提供断言文本时不编结论）；set() = 有文本但无内容词
    claim_tokens: set[str] | None = (
        _tokens(claim_text, zh=zh, policy=policy) if claim_text else None
    )
    # 增量覆盖基线：主引用已经覆盖的 claim 内容词。取不到主引用时按空集处理（保守：
    # 让 supporting 更可能被当作"有增量"，不去重复 Tier 0 已经会报的 anchor 错误）。
    covered: set[str] = set()
    if claim_tokens:
        try:
            primary_rec = ledger.get(claim.citation.anchor_id)
        except RecordVerificationError:
            primary_rec = None
        if primary_rec is not None:
            covered = claim_tokens & _tokens(primary_rec.content, zh=zh, policy=policy)

    results: list[PredicateResult] = []
    for idx, sup in enumerate(supporting):
        try:
            rec = ledger.get(sup.anchor_id)
        except RecordVerificationError as exc:
            # 记录被改/签名不符/不存在：引用指向不存在的证据 → 硬失败（不是"跳过"）
            results.append(
                PredicateResult(
                    "over_citation_anchor",
                    False,
                    "hard",
                    f"supporting[{idx}] anchor {sup.anchor_id!r}: {exc.reason}",
                )
            )
            continue

        if not _quote_at(rec, sup.offset, sup.quote):
            results.append(
                PredicateResult(
                    "over_citation_verbatim",
                    False,
                    "hard",
                    f"supporting[{idx}] quote is not the contiguous substring of "
                    f"{rec.id!r} at [{sup.offset},{sup.offset + len(sup.quote)})",
                )
            )
            continue

        if claim_tokens is None:
            results.append(
                PredicateResult(
                    "over_citation", True, "soft",
                    f"supporting[{idx}]: claim text absent, relevance not assessable",
                )
            )
            continue

        sup_tokens = _tokens(rec.content, zh=zh, policy=policy)
        gain = (claim_tokens & sup_tokens) - covered
        if gain:
            # 贡献了主引用/先前支持未覆盖的 claim 内容词 → 不是冗余引用
            covered |= gain
            results.append(PredicateResult("over_citation", True, "soft"))
        else:
            results.append(
                PredicateResult(
                    "over_citation",
                    False,
                    "soft",
                    f"supporting[{idx}] record {rec.id!r} adds no claim content token "
                    f"beyond the primary citation and earlier supporting evidence",
                )
            )
    return results
