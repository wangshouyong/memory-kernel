# SPDX-License-Identifier: Apache-2.0
"""执法层：确定性判定，不经过生成模型。

判定顺序（`enforce()` 内固定，契约 `docs/ALGO-V4-CONTRACT.md` §6）：

  0. 证据解析（Plan Z C4，仅注入 `resolver` 时执行）——本地账本取不到的锚点交给注入的
     纯 callable 解析；解析失败 → **degrade**（`EVIDENCE_DEGRADE_REASON_OF`），不 hard、
     不静默 pass。`resolver=None` 时整段不执行 → 默认路径零行为变化；
  1. 锚定验证（哈希 + 全载荷签名）——无条件全量；
  2. Tier 0 谓词（verbatim / 否定 / 时间 / 说话人）——无条件全量；
  3. Tier 1 骨架——有断言文本且覆盖完整时执行（贵检查，可被采样跳过）；
  4. 引用集校验 `cite_set`——`policy.cite_set` 且 `claim.supporting` 非空时逐条校验 supporting
     （hard → rejected cause `citation_set`；soft → 至少 degraded）；
  5. 场景谓词 `scenario`——`policy.scenario` 且 `context` 非 None 时跑**五个**谓词
     （hard → rejected cause `scenario`；soft → 至少 degraded；v1.1 新增
     `scenario_evidence_incomplete`：残缺 assistant 回答不得作证据）。

3/4/5 都只在 `coverage_full=True` 时执行：`coverage_full=False` 的语义是"只跑无条件全量项"，
保持 v4 之前一样跳过全部后置检查。**默认策略（`SkeletonPolicy()`，无 context）行为逐字节
不变**：`policy.zh=False` / `cite_set=False` / `scenario=False` 时新增的代码路径一次都不进。

降级原因（v4.4，契约 §6）
────────────────────────────────────────────────────────────────────────────
`verdict == "degraded"` 时 `EnforcementResult.degrade_reason`（主原因）/ `.degrade_reasons`
（全部原因，固定顺序、去重）说明"为什么降级"：映射见 `DEGRADE_REASON_OF` / `degrade_reason_for`，
主原因取 `DEGRADE_REASON_PREDICATE_ORDER` 里第一个。**这一层只加语义标签，不改任何判定**：
`verified` / `rejected` 上两个字段分别是 `None` / `()`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from .cite_set import check_citation_set
from .ledger import AnchorLedger, AnchorRecord, RecordVerificationError
from .obligations import PredicateResult, apply_obligations, quantifier_generalization
from .scenario import (
    CLAIM_MODES,
    HISTORICAL_RECALL_CLAIM_MODE,
    UNSPECIFIED_CLAIM_MODE,
    ScenarioContext,
    check_scenario,
)
from .scenario_context import EVIDENCE_UNAVAILABLE_REASONS, EvidenceUnavailable
from .skeleton import (
    IDFModel,
    SkeletonPolicy,
    content_tokens,
    entities_for_ledger,
    idf_for_ledger,
    skeleton_check,
)
from .zh import has_cjk, zh_negated

# v4.2：断言级否定极性的适用范围。自由转述（assistant 用自己的话总结）不该被
# 「claim 与引用片段否定极性必须一致」这条硬规则判死；只有摘录型/近似摘录的 claim
# 才承担这个义务。实测分布：N_negation（插否定）覆盖率 0.82–0.93、P_verbatim/O_baseline
# 1.0；自由转述的 P_paraphrase 0.00–0.40，0.6 是清晰分界。
_EXCERPT_CONTAINMENT = 0.6

VERDICTS = ("verified", "degraded", "rejected")
REJECT_CAUSES = (
    "invalid_anchor",
    "verbatim",
    "preserve_negation",
    "preserve_speaker",
    "skeleton",
    "budget_exhausted",
    "untrusted_writer",
    "search_empty",
    # v4 能力层（契约 §6）：引用集 hard 失败 / 场景谓词 hard 失败
    "citation_set",
    "scenario",
)

POLICY_VERSION = "p1"

# --------------------------------------------------------------------------- #
# 降级原因标签（v4.4，契约 §6）：verdict == "degraded" 时说明"为什么降级"
# --------------------------------------------------------------------------- #
# 产品要求：`degraded` 必须能说清"为什么降级"，但**不改判**——软失败仍是 `degraded`，
# 不升 `rejected`，也不影响任何既有判定（这些字段只在最终 `degraded` 返回时填充）。
#
# 映射是**确定性、封闭**的（谓词名 → 原因标签）：
#   skeleton_grounding     → "partial_evidence"     证据只部分接地（断言证据不足）
#   scenario_cross_session → "cross_session"        引用了当前会话之外的记录（指代不可解析）
#   over_citation          → "over_citation"        过度引用（supporting 无增量证据）
#   preserve_temporal      → "temporal"             引用丢掉/改写时间项
#   no_generalization      → "over_generalization"  把条件句扩大成一般断言
#   其它 skeleton_*        → "skeleton_<reason>"    其余骨架软失败按原 reason 命名
#                                                  （如 skeleton_temporal）
#   （未收录的软谓词）      → 谓词名本身（保持字段总是可解释，不静默丢弃）
DEGRADE_REASON_OF: dict[str, str] = {
    "skeleton_grounding": "partial_evidence",
    "skeleton_hard_unit_grounding": "partial_evidence",
    # v4.6：论元级失败单独命名，供发射门控用不同措辞（用户裁决：B 情况要细分原因）
    "skeleton_arg_object": "argument_mismatch",
    "skeleton_arg_subject": "argument_mismatch",
    "skeleton_arg_predicate": "argument_mismatch",
    "skeleton_arg_pred_family": "argument_mismatch",
    "skeleton_arg_unit_coverage": "argument_mismatch",
    "skeleton_plural_uncovered": "argument_mismatch",
    "scenario_cross_session": "cross_session",
    # v4.6：superseded 由 hard 改 soft（产品口径：保留但加"后来被更正过"前缀）
    "scenario_superseded": "scenario_superseded",
    # Plan Z C2：第 5 谓词当前是 hard-only（不会走降级路径）；映射保留是为了契约枚举完整、
    # 且将来若裁决把严重度改为 soft，"残缺回答"的措辞与骨架 grounding 同义（证据不完整）。
    "scenario_evidence_incomplete": "partial_evidence",
    "over_citation": "over_citation",
    "preserve_temporal": "temporal",
    "no_generalization": "over_generalization",
    # v4.7-meta（F1 Phase 1）：量词泛化复用 over_generalization 口径；来源归属单列标签
    "quantifier_generalization": "over_generalization",
    "source_attribution": "source_attribution",
}

# 主原因取**固定谓词顺序**里的第一个。顺序 = `enforce()` 的执行顺序（本模块 docstring 的
# 1→5 级：Tier 0 软谓词 → Tier 1 骨架 → Tier 2 引用集 → Tier 3 场景），不依赖 `failed`
# 列表的拼接顺序，因此新增检查项不会悄悄改变既有降级原因。
DEGRADE_REASON_PREDICATE_ORDER: tuple[str, ...] = (
    "preserve_temporal",
    "no_generalization",
    # v4.7-meta：执行顺序在 Tier 0（source_attribution 在 apply_obligations 内，
    # quantifier 紧随其后追加），故排在 no_generalization 之后、骨架族之前。
    "quantifier_generalization",
    "source_attribution",
    "skeleton_grounding",
    "skeleton_hard_unit_grounding",
    "skeleton_arg_object",
    "skeleton_arg_subject",
    "skeleton_arg_predicate",
    "skeleton_arg_pred_family",
    "skeleton_arg_unit_coverage",
    "skeleton_plural_uncovered",
    "over_citation",
    "scenario_cross_session",
    "scenario_superseded",
    "scenario_evidence_incomplete",
)

# `degrade_reasons` 里可能出现的**封闭**标签集（骨架族的 `skeleton_<reason>` 由
# `DEGRADE_REASON_OF` 之外的 fallback 生成，见 `degrade_reason_for`）。
DEGRADE_REASON_LABELS: tuple[str, ...] = (
    "temporal",
    "over_generalization",
    "partial_evidence",
    "argument_mismatch",
    "over_citation",
    "cross_session",
    "scenario_superseded",
    # Plan Z C3/C4：证据不可解析时的可见降级标签（由 resolver 失败路径产生，不是谓词映射）
    "evidence_archived",
    "evidence_unreachable",
    "evidence_destroyed",
    "evidence_missing",
    # Plan Z 两态（SPEC §15 D-TS-3）：historical_recall 下 superseded 的"历史记录"措辞
    "historical_record",
    # v4.7-meta（F1 Phase 1）：来源归属（user_fact 断言引 assistant/tool/document）
    "source_attribution",
)

# --------------------------------------------------------------------------- #
# Plan Z 两态记忆：(谓词, claim_mode) → 降级原因（SPEC §15 D-TS-3）
# --------------------------------------------------------------------------- #
# 裁决：**不改** `scenario_superseded` 谓词语义（它仍按 v4.4 判"同一事实的显式更正"，
# 仍 soft）；两态只影响 `degrade_reason` 的措辞：
#   * unspecified / current_basis → "scenario_superseded"（既有措辞："这条后来被更正过"）；
#   * historical_recall         → "historical_record"（"这是历史记录，当前状态可能已不同"）——
#     回忆历史时，被更正的旧状态本身是答案，不该被说成"已过时"。
# 「不新增第 5 语域」：这里只复用既有语域 + 新降级标签，不改 registers。
HISTORICAL_RECORD_DEGRADE_REASON = "historical_record"
MODE_DEGRADE_REASON_OF: dict[tuple[str, str], str] = {
    (HISTORICAL_RECALL_CLAIM_MODE, "scenario_superseded"): HISTORICAL_RECORD_DEGRADE_REASON,
}

# --------------------------------------------------------------------------- #
# Plan Z C3/C4：证据解析失败 → 降级原因（确定性封闭映射）
# --------------------------------------------------------------------------- #
# 方向（用户裁决）：无法证伪 → 不 hard；静默 pass = 假接受 → 不 pass。故一律 degrade，
# 并带可审计的 `degrade_reason`。原因由 resolver 抛出的 `EvidenceUnavailable.reason` 给出；
# 生产代理负责把网络/超时/解密失败归到 `EVIDENCE_UNAVAILABLE_REASONS` 的封闭集合里。
EVIDENCE_DEGRADE_REASON_OF: dict[str, str] = {
    "archived_timeout": "evidence_archived",
    "archived_unreachable": "evidence_unreachable",
    "destroyed": "evidence_destroyed",
    "missing": "evidence_missing",
}
# 解析失败时 `failed` 里的谓词名（不是场景谓词；报告层据此区分"证据取不到"与"证据不成立"）
EVIDENCE_RESOLUTION_PREDICATE = "evidence_resolution"

assert set(EVIDENCE_DEGRADE_REASON_OF) == set(EVIDENCE_UNAVAILABLE_REASONS), (
    "EVIDENCE_DEGRADE_REASON_OF must be total over EVIDENCE_UNAVAILABLE_REASONS"
)


def degrade_reason_for(
    predicate_name: str, claim_mode: str = UNSPECIFIED_CLAIM_MODE
) -> str:
    """把 (失败谓词名, claim_mode) 映射成降级原因标签（纯函数、确定性）。

    `skeleton_grounding` 是唯一被单独命名的骨架失败（partial_evidence）；其余骨架
    软失败保留 reason（`skeleton_temporal` 等），既不改写语义也不丢信息。
    Plan Z 两态（SPEC §15 D-TS-3）：只有 `historical_recall` + `scenario_superseded`
    这一组合改变措辞（→ `historical_record`），其余组合与加两态之前逐字节一致。
    """
    mode_mapped = MODE_DEGRADE_REASON_OF.get((claim_mode, predicate_name))
    if mode_mapped is not None:
        return mode_mapped
    mapped = DEGRADE_REASON_OF.get(predicate_name)
    if mapped is not None:
        return mapped
    if predicate_name.startswith("skeleton_"):
        return "skeleton_" + predicate_name[len("skeleton_"):]
    return predicate_name


def _effective_claim_mode(
    context: Optional[ScenarioContext], claim_mode: Optional[str]
) -> str:
    """决定本次判定的两态模式（**只认显式输入**，不推断）。

    优先级：`context.claim_mode`（非 unspecified，最贴近这条断言）> `claim_mode` 显式参数
    （管线级默认）> `unspecified`。`claim_mode=None` 且 context 缺省时与加两态之前逐字节
    一致。非法值显式报错，不静默回落。
    """
    ctx_mode = getattr(context, "claim_mode", None)
    if ctx_mode is not None and ctx_mode != UNSPECIFIED_CLAIM_MODE:
        mode = ctx_mode
    elif claim_mode is not None:
        mode = claim_mode
    else:
        mode = UNSPECIFIED_CLAIM_MODE
    if mode not in CLAIM_MODES:
        raise ValueError(f"claim_mode must be one of {CLAIM_MODES}, got {mode!r}")
    return mode


def _degrade_reason_rank(predicate_name: str) -> int:
    """固定谓词顺序的排序键；未收录的骨架名并到骨架族，其余排到末尾。"""
    if predicate_name in DEGRADE_REASON_PREDICATE_ORDER:
        return DEGRADE_REASON_PREDICATE_ORDER.index(predicate_name)
    if predicate_name.startswith("skeleton_"):
        return DEGRADE_REASON_PREDICATE_ORDER.index("skeleton_grounding")
    return len(DEGRADE_REASON_PREDICATE_ORDER)


def _degrade_reasons(
    results: list[PredicateResult], claim_mode: str = UNSPECIFIED_CLAIM_MODE
) -> tuple[str, ...]:
    """从失败谓词里抽取降级原因（固定顺序、同因去重）。

    只取 `severity == "soft"` 且 `not passed` 的谓词：硬失败会直接 `rejected`
    （不走本函数），所以 `degraded` 的原因集合与软失败集合一一对应。
    重复的同一原因（如多条 supporting 各自触发 `over_citation`）只保留一次。
    `claim_mode` 只影响 (谓词, mode) 的措辞映射（historical_recall 的 superseded）。
    """
    soft_failed = [r for r in results if not r.passed and r.severity == "soft"]
    ordered = sorted(soft_failed, key=lambda r: _degrade_reason_rank(r.name))  # stable
    out: list[str] = []
    for r in ordered:
        reason = degrade_reason_for(r.name, claim_mode)
        if reason not in out:
            out.append(reason)
    return tuple(out)


@dataclass(frozen=True)
class Citation:
    anchor_id: str
    offset: int
    quote: str

    @property
    def length(self) -> int:
        return len(self.quote)


@dataclass(frozen=True)
class Claim:
    citation: Citation
    asserted_speaker: Optional[str] = None
    claim_type: str = "excerpt"
    # Tier 1 骨架需要断言文本；不提供时只走 Tier 0（摘录型断言的常态）
    text: Optional[str] = None
    supporting: tuple[Citation, ...] = ()


@dataclass
class EnforcementResult:
    verdict: str
    coverage: dict
    rejected_cause: Optional[str] = None
    failed: list[PredicateResult] = field(default_factory=list)
    elapsed_us: float = 0.0
    anchor_id: Optional[str] = None
    # v4.4 降级原因：**只在 verdict == "degraded" 时填充**，其余情形保持 None / ()。
    # 字段加在末尾且都有默认值 → 既有构造点与读取点（registers.py 等）行为不变。
    degrade_reason: Optional[str] = None
    degrade_reasons: tuple[str, ...] = ()

    @property
    def hard_failed(self) -> list[str]:
        return [p.name for p in self.failed if p.severity == "hard" and not p.passed]

    @property
    def soft_failed(self) -> list[str]:
        return [p.name for p in self.failed if p.severity == "soft" and not p.passed]


class _ResolvedLedgerView:
    """只读账本视图：本地记录优先，缺失锚点由注入 resolver 解析后覆盖（Plan Z C4）。

    目的：让 Tier 0 / Tier 1 / 引用集 / 场景谓词看到**同一份**证据视图，从而满足
    "resolver 正常 → 与本地记录同结果"。本类不做任何 I/O——解析在
    `_resolve_missing_evidence()` 里一次性完成，这里只做确定性的字典查找。
    """

    def __init__(self, ledger, resolved: Mapping[str, AnchorRecord]) -> None:
        self._ledger = ledger
        # dict 保留插入顺序（= claim 里锚点的声明顺序），因此 records 的扩展顺序是确定的
        self._resolved = dict(resolved)

    def __len__(self) -> int:
        return len(self._ledger) + len(self._resolved)

    def __iter__(self):
        return iter(self.records)

    @property
    def records(self) -> tuple[AnchorRecord, ...]:
        return tuple(self._ledger.records) + tuple(self._resolved.values())

    def get(self, anchor_id: str) -> AnchorRecord:
        rec = self._resolved.get(anchor_id)
        if rec is not None:
            return rec
        return self._ledger.get(anchor_id)

    def try_get(self, anchor_id: str) -> Optional[AnchorRecord]:
        rec = self._resolved.get(anchor_id)
        if rec is not None:
            return rec
        return self._ledger.try_get(anchor_id)


def _claim_anchor_ids(claim) -> list[str]:
    """claim 引用到的 anchor id（主引用 → supporting 声明顺序，去重保序）。"""
    out: list[str] = []
    seen: set[str] = set()

    def push(anchor_id) -> None:
        aid = str(anchor_id)
        if aid not in seen:
            seen.add(aid)
            out.append(aid)

    cit = getattr(claim, "citation", None)
    if cit is not None:
        push(cit.anchor_id)
    for sup in getattr(claim, "supporting", ()) or ():
        push(sup.anchor_id)
    return out


def _resolve_missing_evidence(
    ledger: AnchorLedger,
    claim,
    resolver: Callable[[str], AnchorRecord],
) -> tuple[Optional[object], Optional[tuple[str, str]], Optional[RecordVerificationError]]:
    """一次性解析本地账本里取不到的锚点（Plan Z C4）。

    返回 `(view, degrade, hard_failure)`：
      * `view`：可直接替代 `ledger` 的只读视图（全部本地命中时就是 `ledger` 本身）；
      * `degrade`：`(reason, detail)`，reason 属于 `EVIDENCE_UNAVAILABLE_REASONS`，
        调用方必须据此返回 degraded（**不得静默 pass**）；
      * `hard_failure`：本地有这条记录但验签失败 → 保持既有 hard 拒绝（篡改 = 攻击，不降级）。

    I/O 只可能发生在调用方注入的 resolver 里；core 不 sleep、不读时钟、不联网。
    """
    resolved: dict[str, AnchorRecord] = {}
    for anchor_id in _claim_anchor_ids(claim):
        try:
            ledger.get(anchor_id)
            continue
        except RecordVerificationError as exc:
            if "not found" not in exc.reason:
                # 本地记录在但被改/签名不符：既有 hard 拒绝路径，不因有 resolver 而降级
                return None, None, exc
        try:
            rec = resolver(anchor_id)
        except EvidenceUnavailable as exc:
            return None, (exc.reason, exc.detail or str(exc)), None
        if rec is None:
            return None, ("missing", f"anchor {anchor_id!r}: resolver returned no record"), None
        if not isinstance(rec, AnchorRecord):
            raise TypeError(
                "resolver must return AnchorRecord or raise EvidenceUnavailable, "
                f"got {type(rec).__name__}"
            )
        if rec.id != anchor_id:
            return None, (
                "missing",
                f"anchor {anchor_id!r}: resolver returned record {rec.id!r}",
            ), None
        try:
            # 冷归档记录必须过与本地记录完全相同的校验才允许进入执法路径
            ledger.verify(rec)
        except RecordVerificationError as exc:
            return None, (
                "archived_unreachable",
                f"anchor {anchor_id!r}: resolved record failed verification: {exc.reason}",
            ), None
        resolved[anchor_id] = rec
    if not resolved:
        return ledger, None, None
    return _ResolvedLedgerView(ledger, resolved), None, None


def _evidence_degraded_result(
    reason: str, detail: str, *, started_ns: int, anchor_id: Optional[str], coverage_full: bool
) -> EnforcementResult:
    """证据不可解析时的可见降级（Plan Z C3/C4）：不 hard、不静默 pass。"""
    degrade = EVIDENCE_DEGRADE_REASON_OF[reason]
    return EnforcementResult(
        verdict="degraded",
        # 证据取不到 → 后续全量检查一次都没跑，coverage 如实标 False（不是"跑过且通过"）
        coverage=_coverage(False),
        failed=[
            PredicateResult(
                EVIDENCE_RESOLUTION_PREDICATE,
                False,
                "soft",
                f"evidence unavailable ({reason}): {detail}",
            )
        ],
        elapsed_us=_elapsed(started_ns),
        anchor_id=anchor_id,
        degrade_reason=degrade,
        degrade_reasons=(degrade,),
    )


def enforce(
    ledger: AnchorLedger,
    claim: Claim,
    *,
    scope_mode: str = "clause",
    coverage_full: bool = True,
    extra: Optional[list[str]] = None,
    policy: Optional[SkeletonPolicy] = None,
    session_dates: Optional[Mapping] = None,
    idf_model: Optional[IDFModel] = None,
    context: Optional[ScenarioContext] = None,
    resolver: Optional[Callable[[str], AnchorRecord]] = None,
    claim_mode: Optional[str] = None,
    cache=None,
) -> EnforcementResult:
    """确定性执法。**执行路径不调用任何模型**（公理 B）。

    `session_dates` / `idf_model` 只影响 Tier 1：
      - `session_dates`: {(source_id, episode_id): 会话日期串}，把会话日期并进证据时间池；
      - `idf_model`: IDF 权重表；不给且策略要求 IDF 时，从 `ledger` 现算并缓存
        （纯计数，仍然是确定性的、不碰模型）。

    `context`（v4，契约 §6）只影响第 5 级场景谓词：`None` 或 `policy.scenario=False` 时
    场景谓词不执行。`policy` 在所有层级都参与：`policy.zh` 同时传给 Tier 0 与 IDF 构建，
    保证中文词法层在 Tier 0/Tier 1 里用的是同一套分词口径。

    `resolver`（Plan Z C4）：`None`（默认）时行为与 C4 之前**逐字节一致**；给出时，本地账本
    取不到的锚点交给它解析（纯 callable，`anchor_id -> AnchorRecord`，失败抛
    `EvidenceUnavailable(reason)`）。解析成功 → 解析结果进同一份只读视图，后续各层与"记录
    本来就在本地"同结果；解析失败 → **可见降级**（`degrade_reason` 见
    `EVIDENCE_DEGRADE_REASON_OF`），绝不静默 pass。core 内不 sleep / 不读时钟 / 不联网。

    `claim_mode`（Plan Z 两态，SPEC §15 D-TS-3）：`None`（默认）时读 `context.claim_mode`，
    两者都没有 → `unspecified`（**判定逐字节不变**）。只影响 `degrade_reason` 的措辞
    （`historical_recall` 下 `scenario_superseded` → `historical_record`），**不改任何判定、
    不改谓词语义、不新增语域**。意图只认显式输入：本函数不读题面/answer_policy/模型输出。

    `cache`（internal-task，可选）：`EngineCache` 实例。本函数**不持有跨请求状态**：IDF/实体
    派生缓存全部来自注入的这个对象；`None`（默认）→ 进程级兼容默认缓存（结果与历史逐字节
    一致）。缓存只做 memoization，清空/重建不改任何判定（`tests/test_stateless.py`）。
    """
    started = time.perf_counter_ns()
    cit = claim.citation
    # 策略提前构造：Tier 0 的 zh 开关与第 4/5 级都要用它；构造是纯函数，默认值 = 历史 v2 行为
    pol = policy or SkeletonPolicy()
    # Plan Z 两态：context.claim_mode（非 unspecified）> claim_mode 参数 > unspecified（无推断）
    mode = _effective_claim_mode(context, claim_mode)

    # 第 0 级（Plan Z C4，仅在注入 resolver 时启用）：解析本地账本缺失锚点。
    # resolver=None 时整段不执行 → 默认路径零行为变化。
    source = ledger
    if resolver is not None:
        view, degrade, hard_failure = _resolve_missing_evidence(ledger, claim, resolver)
        if hard_failure is not None:
            # 本地记录存在但验签失败：保持既有 hard 拒绝（篡改是攻击，不因有 resolver 而降级）
            return EnforcementResult(
                verdict="rejected",
                coverage=_coverage(coverage_full),
                rejected_cause="invalid_anchor",
                failed=[PredicateResult("anchor", False, "hard", hard_failure.reason)],
                elapsed_us=_elapsed(started),
                anchor_id=cit.anchor_id,
            )
        if degrade is not None:
            return _evidence_degraded_result(
                degrade[0], degrade[1], started_ns=started,
                anchor_id=cit.anchor_id, coverage_full=coverage_full,
            )
        source = view

    # 第一级：锚定验证（哈希 + 全载荷签名）——无条件全量
    try:
        rec = source.get(cit.anchor_id)
    except RecordVerificationError as exc:
        return EnforcementResult(
            verdict="rejected",
            coverage=_coverage(coverage_full),
            rejected_cause="invalid_anchor" if "not found" not in exc.reason else "search_empty",
            failed=[PredicateResult("anchor", False, "hard", exc.reason)],
            elapsed_us=_elapsed(started),
            anchor_id=cit.anchor_id,
        )

    # 第二级：verbatim + Tier 0 谓词——无条件全量
    results = apply_obligations(
        rec,
        cit.quote,
        cit.offset,
        scope_mode=scope_mode,
        asserted_speaker=claim.asserted_speaker,
        extra=extra,
        zh=pol.zh,
        # v4.7-meta（F1 Phase 1，默认关闭）：`source_attribution` soft 谓词 +
        # `asserted_scope` 只在该开关打开时参与（缺省 unspecified → 谓词不判）。
        source_type=pol.source_type,
        asserted_scope=(context.asserted_scope if context is not None else "unspecified"),
    )
    # 第二级补充（v4.1/v4.2）：断言文本级的中文否定极性。
    #   Tier 0 的 preserve_negation 只比较"引用片段 vs 证据子句"，看不到 claim 自己引入/丢掉的否定。
    #   中文真实语料里 N_negation 这一类（把"我去了"改写成"我没有去"）必须在这一层抓：
    #   claim 的否定极性 ≠ 所引片段的否定极性 → 硬失败。
    #   v4.2 收窄：只对**摘录型/近似摘录**的 claim 生效（token 覆盖率 ≥ _EXCERPT_CONTAINMENT）。
    #   自由转述（assistant 用自己的话总结、可能自己加否定）不适用这条硬规则，交给 Tier 1；
    #   否则会把合法的转述误杀（首轮实测 P_paraphrase strict 0.05→0.40 就是这里过报）。
    if pol.zh and claim.text and has_cjk(claim.text):
        claim_tokens = content_tokens(claim.text, zh=True)
        cite_tokens = content_tokens(cit.quote, zh=True)
        containment = (
            len(claim_tokens & cite_tokens) / len(claim_tokens) if claim_tokens else 0.0
        )
        if containment >= _EXCERPT_CONTAINMENT:
            claim_neg = zh_negated(claim.text)
            cite_neg = zh_negated(cit.quote)
            results.append(
                PredicateResult(
                    "preserve_negation_claim",
                    claim_neg == cite_neg,
                    "hard",
                    "" if claim_neg == cite_neg else
                    f"claim negation polarity mismatch: claim negated={claim_neg}, "
                    f"cited quote negated={cite_neg} (excerpt containment={containment:.2f})",
                )
            )
    # 第二级补充（v4.7-meta，F1 Phase 1）：量词泛化（soft-only，默认关闭）。
    # 判据单侧：claim 出现（未被否定的）量词，而全部证据都没有量词 → soft 失败。
    # 证据也有量词 / claim 无量词 → passed（宁可漏，不假杀）；不能产生 verified 资格。
    if pol.quantifier and claim.text:
        evidence_texts = [rec.content]
        for sup in claim.supporting:
            try:
                evidence_texts.append(source.get(sup.anchor_id).content)
            except Exception:
                continue  # 不可验证的 supporting 交给 Tier 1/cite_set 报，不在此重复
        results.append(quantifier_generalization(claim.text, evidence_texts))

    failed = [r for r in results if not r.passed]
    hard = [r for r in failed if r.severity == "hard"]
    soft = [r for r in failed if r.severity == "soft"]

    if hard:
        return EnforcementResult(
            verdict="rejected",
            coverage=_coverage(coverage_full),
            rejected_cause=hard[0].name,
            failed=results,
            elapsed_us=_elapsed(started),
            anchor_id=rec.id,
        )

    if not coverage_full:
        return EnforcementResult(
            verdict="verified",
            coverage=_coverage(False),
            failed=results,
            elapsed_us=_elapsed(started),
            anchor_id=rec.id,
        )

    # 第三级（Tier 1）：骨架——只在有断言文本且覆盖完整时执行（贵检查，可被采样跳过）
    if claim.text and pol.enabled:
        supporting_records = []
        for sup in claim.supporting:
            try:
                supporting_records.append(source.get(sup.anchor_id))
            except RecordVerificationError as exc:
                return EnforcementResult(
                    verdict="rejected",
                    coverage=_coverage(True),
                    rejected_cause="invalid_anchor",
                    failed=results + [PredicateResult("anchor", False, "hard", exc.reason)],
                    elapsed_us=_elapsed(started),
                    anchor_id=rec.id,
                )
        idf = idf_model
        if pol.idf and idf is None:
            idf = idf_for_ledger(source, stem_ok=pol.stem, stop_mode=pol.stop_mode,
                                 zh=pol.zh, cache=cache)
        sk = skeleton_check(claim.text, [rec, *supporting_records], pol,
                            session_dates=session_dates, idf_model=idf,
                            entities=entities_for_ledger(source, cache=cache).get(rec.source_id))
        if not sk.ok:
            results = results + [
                PredicateResult(f"skeleton_{sk.reason}", False, sk.severity, sk.detail)
            ]
            if pol.failure_mode == "rejected":
                return EnforcementResult(
                    verdict="rejected",
                    coverage=_coverage(True),
                    rejected_cause="skeleton",
                    failed=results,
                    elapsed_us=_elapsed(started),
                    anchor_id=rec.id,
                )

    # 第四级（v4）：引用集校验——只在 cite_set 打开且有 supporting 时逐条校验（契约 §4）
    # 注意顺序：Tier 1 的 supporting anchor 预检先于本层，因此"有断言文本 + supporting anchor
    # 不存在"由 Tier 1 报 invalid_anchor；没有断言文本时本层报 citation_set（契约 §6 的顺序）。
    if pol.cite_set and claim.supporting:
        cite_results = check_citation_set(source, claim, zh=pol.zh, policy=pol)
        results = results + cite_results
        if any(p.severity == "hard" and not p.passed for p in cite_results):
            return EnforcementResult(
                verdict="rejected",
                coverage=_coverage(True),
                rejected_cause="citation_set",
                failed=results,
                elapsed_us=_elapsed(started),
                anchor_id=rec.id,
            )

    # 第五级（v4）：场景谓词——只在 scenario 打开且给定 context 时执行（契约 §5）
    if pol.scenario and context is not None:
        scen_results = check_scenario(source, claim, context, policy=pol)
        results = results + scen_results
        if any(p.severity == "hard" and not p.passed for p in scen_results):
            return EnforcementResult(
                verdict="rejected",
                coverage=_coverage(True),
                rejected_cause="scenario",
                failed=results,
                elapsed_us=_elapsed(started),
                anchor_id=rec.id,
            )

    soft_all = [r for r in results if not r.passed and r.severity == "soft"]
    # v4.4：degraded 时给出"为什么降级"（主原因 = 固定谓词顺序的第一个；全部原因去重保序）。
    # verified（无软失败）时保持默认 None / ()：字段只在 degraded 上有意义。
    # Plan Z 两态：mode 只改 (谓词, mode) 的措辞映射，不改判定（`_degrade_reasons` 内映射）。
    reasons = _degrade_reasons(results, mode) if soft_all else ()
    return EnforcementResult(
        verdict="degraded" if soft_all else "verified",
        coverage=_coverage(True),
        failed=results,
        elapsed_us=_elapsed(started),
        anchor_id=rec.id,
        degrade_reason=reasons[0] if reasons else None,
        degrade_reasons=reasons,
    )


def _coverage(full: bool) -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "full_checks_ran": full,
        "sampling_rate": 1.0 if full else 0.0,
        "decision_seed": None,
        "stratified_bucket": None,
    }


def _elapsed(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1000.0
