# SPDX-License-Identifier: Apache-2.0
r"""场景谓词（v4 能力层；契约 `docs/ALGO-V4-CONTRACT.md` §5）。

为什么需要这一层
────────────────────────────────────────────────────────────────────────────
`verbatim` + 骨架能证明"引文是真的、词面是接地的"，但证明不了"这条证据在这个语境里还能不能用"：

  1. **时间方向**：断言发生在 T 时点，却引用 T **之后**才出现的记录（后事作前证）；
  2. **跨会话**：断言用指代（"上次""之前"）指向当前会话之外的记录，指代对象在会话内不可解析；
  3. **说话人范围**：把**助手**的推测/建议当成**用户事实**引用（`assistant_claim` ≠ `user_fact`）；
  4. **已被更正**：被引记录后来被同一会话里更新的用户更正消息推翻（同话题），仍被当作有效证据。

四个谓词全部只读账本元数据 + 中文结构标记（`zh.py` 的 `zh_anaphor` / `zh_hedge` /
`zh_corrected`），**不调用模型、不联网、无 I/O、无时间/随机数**，同一输入永远同一输出。
v1.1（Plan Z C2）新增第 5 个谓词 `scenario_evidence_incomplete`：只看被引记录的
`is_complete` 元数据（同样无 I/O、无时间依赖）。

| 谓词 | 条件（全部确定性） | 严重度 |
|---|---|---|
| `scenario_timeline` | `ctx.context_time` 有值且被引记录 `record_times[anchor] > context_time` | hard |
| `scenario_cross_session` | `claim.text` 含指代且 `rec.source_id != ctx.context_session_id` | soft |
| `scenario_speaker_scope` | `asserted_scope == "user_fact"` 且 `rec.speaker == "assistant"` 且 `zh_hedge(rec.content)` | hard |
| `scenario_superseded` | 同 `source_id`+`episode_id` 存在 `turn_index >` 该记录、`speaker == "user"`、`zh_corrected` 的**同一事实**更正：与 claim 有内容词交集 **且** 与被引记录有内容词交集（`>= SUPERSEDED_MIN_SHARED_TOKENS`，扣除更正标记自身 token）**或** 否定标记后重复了被引记录的内容词（narrow，见下） | soft |
| `scenario_evidence_incomplete` | 被引 assistant 记录 `is_complete is False`（残缺回答不得作证据） | hard |

严重度的理由（v4.6 用户裁决后）：**场景不适用**是"这条证据在当前语境下无效"的结构事实。
时间倒置、拿推测当事实仍是硬错误（`rejected`，cause `scenario`）；跨会话指代只是**指代对象
不在本会话内**、可能仍然对，所以是软失败（`degraded`）。**`scenario_superseded` 改为软失败**
（产品口径：已被更正的旧说法"保留但降级"，由 `emit.py` 加"这条后来被更正过"的前缀，而不是
直接删句）；代价是它从 strict 拦截移出、计入 loose 标记。

单侧性（必须如实读）
────────────────────────────────────────────────────────────────────────────
* 四个谓词都是**结构信号**，不是语义理解：`zh_anaphor`/`zh_hedge`/`zh_corrected` 是封闭
  标记表（见 `zh.py`），表外表达漏检 → 假接受（放过）；同形误检 → 假杀。方向和每张表的
  已知同形词都在 `zh.py` 的模块 docstring 里列了。
* `scenario_cross_session` 只在 `ctx.context_session_id` 有值时才判（否则"会话不等"恒真，
  会把所有指代断言一律标脏——那不是检查，是噪声）。
* `scenario_superseded` 依赖 `claim.text` 与被引记录/更正消息的内容词交集：没有断言文本 /
  断言无内容词时**不可判 → passed**（不编结论）。`zh=False` 时中文断言 token 为空，同样判不了
  ⇒ passed（这是 v4 开关组合的已知单侧性；`SkeletonPolicy.v4()` 是 `zh=True` 的组合）。
  判据细节（v4.4「同一事实」三段判据、阈值常量、`narrow` 否定重复路径的理由）见本文件
  下方 `SUPERSEDED_*` 常量区的注释。方向：**宁可漏也不假杀**——漏检（真被更正却放过）
  是软方向，误标（无关更正算到这条证据头上）会直接把正当引用判成 `rejected`。
* `record_times` 里缺某个 anchor 的时点 → 该 anchor 不参与时间方向检查（不可判，不是通过；
  报告层应把 `record_times` 的覆盖率一起报出来）。
* 被引 anchor 不可验证（记录被改/不存在）时本模块**跳过**它，不在这里重复报 anchor 错误：
  那由 Tier 0 的锚定校验与 `cite_set.py` 的 `over_citation_anchor` 负责。
* `record_times` 的值不是数字 → 比较时**直接抛 TypeError**（显式失败），不做静默降级。

确定性
────────────────────────────────────────────────────────────────────────────
被引记录按"主引用 → `claim.supporting` 声明顺序"去重后遍历；账本候选记录按
`ledger.records` 的追加顺序遍历（`AnchorLedger` 只追加、顺序固定）。四个谓词的返回顺序
固定为 `SCENARIO_PREDICATES`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .obligations import PredicateResult
from .skeleton import SkeletonPolicy, content_tokens
from .zh import (
    ZH_CORRECTED,
    ZH_NEGATION_MARKERS,
    zh_anaphor,
    zh_corrected,
    zh_hedge,
    zh_tokens,
)

__all__ = [
    "CLAIM_MODES",
    "COMPLETENESS_UNKNOWN",
    "EVIDENCE_INCOMPLETE",
    "SCENARIO_PREDICATES",
    "ScenarioContext",
    "check_scenario",
]

# Plan Z 两态记忆（SPEC §15 / D-TS-1）：调用方显式声明这条断言建立在
#   * `current_basis`    —— 以"当前状态"为依据（当前有效的记忆才可用）；
#   * `historical_recall` —— 回忆历史（被更正/被撤销的旧状态本身可能是答案）。
# 默认 `unspecified` = 不做两态区分（既有行为逐字节不变）。
# 判定层（`enforce`/场景谓词）**不推断**模式：模式只由调用方写入 `ScenarioContext`。
CLAIM_MODES: tuple[str, ...] = ("unspecified", "current_basis", "historical_recall")
UNSPECIFIED_CLAIM_MODE = "unspecified"
CURRENT_BASIS_CLAIM_MODE = "current_basis"
HISTORICAL_RECALL_CLAIM_MODE = "historical_recall"

# 五个谓词的固定返回顺序（v1.1：Plan Z C2 新增第 5 条 `scenario_evidence_incomplete`）
SCENARIO_PREDICATES: tuple[str, ...] = (
    "scenario_timeline",
    "scenario_cross_session",
    "scenario_speaker_scope",
    "scenario_superseded",
    "scenario_evidence_incomplete",
)

# --------------------------------------------------------------------------- #
# `scenario_evidence_incomplete` 的语义（Plan Z C2 / D-PZ-2）
# --------------------------------------------------------------------------- #
# 与 `_speaker_scope` **不合并**：那边判"把助手的推测当用户事实"（说话人范围错位），
# 这边判"这条回答本身没写完"（流式被取消/断连/写失败 → 残缺回答不得作证据），两者语义不同。
#
# 判定（确定性，只看账本元数据，不看内容）：
#   * 被引 assistant 记录 `is_complete is False` → **hard fail**（残缺回答不得作证据）；
#   * `is_complete is None`（旧记录 / 未带完整性证据）→ **pass**，但 detail 标
#     `completeness_unknown`——不追溯改旧行为，同时把"不可判"如实报出来（不是"已证明完整"）；
#   * `is_complete is True` 或 user 记录 → pass。
#
# 单侧性：`None` 走 pass 是刻意的（旧账本没有完整性证据，追溯判死会改写历史判定）；
# 因此"通过"在这一谓词上可能只是"不可判"，报告层必须读 detail 里的 `completeness_unknown`。
EVIDENCE_INCOMPLETE = "scenario_evidence_incomplete"
COMPLETENESS_UNKNOWN = "completeness_unknown"

# --------------------------------------------------------------------------- #
# `scenario_superseded` 的「同一事实」判据（v4.4）
# --------------------------------------------------------------------------- #
# 旧判据只看「后续有更正词 且 与 claim 有内容词交集」，**不验证更正的是不是同一件事**：
# 连续更正的会话里，任何一条别的更正消息只要与 claim 共享一个常见词就会被当成"这条证据
# 已被推翻"（真实评测实测：S_timeline_ctrl 2 条 + S_superseded_ctrl 1 条误标）。
#
# 新判据 = 三段全部满足（任一不满足 → passed=True，不编结论）：
#   1. `claim_link`：候选更正消息与 **claim** 有内容词交集（更正的是这条断言说的事）；
#   2. `claim↔record`：claim 与被引**记录**也有内容词交集（引用与断言无关时不判）；
#   3. `same_fact`：候选更正消息确实在更正**这条被引记录**，两条任一成立：
#      a. **wide**：候选消息与被引记录的内容词交集 >= `SUPERSEDED_MIN_SHARED_TOKENS`；
#         计数时扣掉「更正标记自身」的 token（理由见下）；
#      b. **narrow**：候选消息里某个否定标记之后 `SUPERSEDED_NEG_REPETITION_WINDOW` 个字符内
#         重复了被引记录的内容词 —— 即「不是昨天」这种**否定重复旧值**的最小更正形态。
#
# 为什么 wide 计数要扣掉更正标记自身的 token：
#   `ZH_CORRECTED` 的标记（不是/改成/其实/…）本身只证明"这是一条更正消息"，对"是不是同一件事"
#   不提供任何信息；而「不是」「不对」在中文里极其常见，一条无关的更正消息只要碰巧含「不是」
#   就能凭空凑出 1~2 个共享 token（实测 S_superseded_ctrl-13 的更正长消息与被引记录恰好只
#   共享「不是/格式」两个词）。扣掉标记 token 后，阈值 2 才真的等价于"至少 2 个**内容**词重合"
#   （≈ 一个 ≥3 字词，或两个词），与任务书口径一致。
#
# 为什么还需要 narrow（否定重复旧值）：
#   真实更正会**替换**旧值（苹果→橙子、昨天→上周三），与旧记录的交集天然很低——单测 fixture
#   「不是昨天」型更正与被引记录只有 1 个共享内容词。若只留 wide，这条正当拦截会丢；而只把阈值
#   降到 1 又会让上面那批误标全部回来（实测：阈值 1 时 ctrl 误标 2+1 全在）。narrow 用结构信号
#   （否定标记紧邻重复旧值）区分"更正这条记录"与"恰好共享一个词"，两个方向都保得住。
#   实测（v4.4 评测记录与只读诊断脚本，聚合数字见 `docs/ALGO-V4-CONTRACT.md` §5.1）：
#     仅 wide(>=2)：ctrl 误标 2+1（阈值 1）/ 0（阈值 2，但 S_superseded 正常样本 & 单测 fixture 丢拦截）
#     wide(>=2, 扣标记) + narrow：ctrl 误标 0，S_superseded 正常样本 1/1 仍拦，单测 fixture 仍拦
#
# 已知单侧性：narrow 只认**否定标记之后**的重复（"不是X"型）；"X 改成 Y"型（旧值在标记之前）
# 不参与 narrow——它在真实语料里既可能是更正事实、也可能只是改需求（实测 S_timeline_ctrl-04
# 的「季度改成月份」是改任务参数，不是更正被引记录），方向取漏检（当噪音处理）而不假杀。
SUPERSEDED_MIN_SHARED_TOKENS = 2
SUPERSEDED_NEG_REPETITION_WINDOW = 4
# 更正标记自身产生的 token（由 `ZH_CORRECTED` 派生，不另维护第二张表）
SUPERSEDED_MARKER_TOKENS: frozenset[str] = frozenset(
    t for marker in ZH_CORRECTED for t in zh_tokens(marker)
)
# narrow 判据只认**多字否定标记**：单字（没/未/无/别/甭）在中文里出现密度太高，
# 拿它做 4 字窗口会让"没有+任意内容词"变成噪音源。
SUPERSEDED_NEG_MARKERS: tuple[str, ...] = tuple(
    marker for marker in ZH_NEGATION_MARKERS if len(marker) >= 2
)


@dataclass(frozen=True)
class ScenarioContext:
    """断言所处的语境（全部可选；缺省字段 = 该谓词不可判 → passed）。

    `context_session_id`: 断言发生在哪个会话（`AnchorRecord.source_id` 的空间）。
    `context_time`:       断言的时点（epoch 秒，float）；被引记录更新于它之后 = 后事作前证。
    `asserted_scope`:     `"user_fact"` | `"assistant_claim"` | `"unspecified"`（缺省）。
    `record_times`:       anchor_id -> epoch 秒；缺项 = 该 anchor 不参与时间方向检查。
    `claim_mode`:         Plan Z 两态记忆（`docs/TWO-STATE-MEMORY.md` / SPEC §15 D-TS-1）：
                          `"unspecified"`（缺省）/ `"current_basis"` / `"historical_recall"`。
                          只由调用方**显式**给出；谓词本身**不读**它（D-TS-2），
                          它只影响 `context_filter` 的当前/历史门控与 `enforce` 的降级措辞。

    `record_times` 默认值是**不可变空元组**（契约 §5 字面），调用方传 Mapping 时按 Mapping
    读取；两种形态都不做原地修改（frozen dataclass 保证字段不可重绑）。
    """

    context_session_id: Optional[str] = None
    context_time: Optional[float] = None
    asserted_scope: str = "unspecified"
    record_times: Mapping[str, float] = ()
    # Plan Z 两态记忆：默认 unspecified → 判定与加该字段之前逐字节一致（同型先例 asserted_scope）
    claim_mode: str = "unspecified"

    def __post_init__(self) -> None:
        if self.claim_mode not in CLAIM_MODES:
            raise ValueError(
                f"claim_mode must be one of {CLAIM_MODES}, got {self.claim_mode!r}"
            )


def _times(ctx: ScenarioContext) -> Optional[Mapping[str, float]]:
    """取 record_times；空 / 非 Mapping（默认空元组）→ None（不可判）。"""
    rt = ctx.record_times
    if not rt or not hasattr(rt, "get"):
        return None
    return rt


def _cited_records(ledger, claim) -> list:
    """被引记录（主引用 → supporting 声明顺序，按 anchor_id 去重）。不可验证的跳过。"""
    anchors: list[str] = []
    cit = getattr(claim, "citation", None)
    if cit is not None:
        anchors.append(cit.anchor_id)
    for sup in getattr(claim, "supporting", ()) or ():
        anchors.append(sup.anchor_id)

    out: list = []
    seen: set[str] = set()
    for anchor_id in anchors:
        if anchor_id in seen:
            continue
        seen.add(anchor_id)
        rec = ledger.try_get(anchor_id)
        if rec is not None:
            out.append(rec)
    return out


def _claim_tokens(claim, policy: SkeletonPolicy) -> set[str]:
    text = claim.text or ""
    if not text:
        return set()
    return content_tokens(text, stop_mode=policy.stop_mode, zh=policy.zh)


def _rec_tokens(rec, policy: SkeletonPolicy) -> set[str]:
    return content_tokens(rec.content, stop_mode=policy.stop_mode, zh=policy.zh)


# --------------------------------------------------------------------------- #
# 四个谓词：条件不满足时一律 passed=True（不可判 ≠ 通过，但结果都是 passed）
# --------------------------------------------------------------------------- #


def _timeline(ledger, claim, ctx: ScenarioContext) -> PredicateResult:
    name = "scenario_timeline"
    if ctx.context_time is None:
        return PredicateResult(name, True, "hard", "no context_time")
    times = _times(ctx)
    if times is None:
        return PredicateResult(name, True, "hard", "no record_times")
    for rec in _cited_records(ledger, claim):
        ts = times.get(rec.id)
        if ts is None:
            continue
        if ts > ctx.context_time:
            return PredicateResult(
                name,
                False,
                "hard",
                f"cited record {rec.id!r} timestamp {ts!r} > context_time "
                f"{ctx.context_time!r} (later evidence used for an earlier assertion)",
            )
    return PredicateResult(name, True, "hard")


def _cross_session(ledger, claim, ctx: ScenarioContext) -> PredicateResult:
    name = "scenario_cross_session"
    text = claim.text or ""
    if not text or not zh_anaphor(text):
        return PredicateResult(name, True, "soft", "no anaphor in claim text")
    if ctx.context_session_id is None:
        # 会话未知时"不相等"恒真，判了只会把每条指代断言都标脏 —— 不可判 → passed
        return PredicateResult(name, True, "soft", "no context_session_id")
    for rec in _cited_records(ledger, claim):
        if rec.source_id != ctx.context_session_id:
            return PredicateResult(
                name,
                False,
                "soft",
                f"anaphoric claim cites {rec.id!r} from session {rec.source_id!r} "
                f"!= context session {ctx.context_session_id!r}",
            )
    return PredicateResult(name, True, "soft")


def _speaker_scope(ledger, claim, ctx: ScenarioContext) -> PredicateResult:
    name = "scenario_speaker_scope"
    if ctx.asserted_scope != "user_fact":
        return PredicateResult(name, True, "hard", "asserted_scope is not user_fact")
    for rec in _cited_records(ledger, claim):
        if rec.speaker == "assistant" and zh_hedge(rec.content):
            return PredicateResult(
                name,
                False,
                "hard",
                f"asserted as user_fact but cites assistant hedged record {rec.id!r}",
            )
    return PredicateResult(name, True, "hard")


def _negated_repetition(text: str, rec_tokens: set[str], policy: SkeletonPolicy) -> bool:
    """narrow 判据：否定标记后 `SUPERSEDED_NEG_REPETITION_WINDOW` 字符内重复了被引记录的内容词。

    确定性：按 `SUPERSEDED_NEG_MARKERS` 的固定顺序、每次从上次命中位置继续向后找；
    同一输入永远同一输出。窗口内 token 化与判定用的 `stop_mode` / `zh` 口径一致。
    """
    for marker in SUPERSEDED_NEG_MARKERS:
        start = text.find(marker)
        while start != -1:
            begin = start + len(marker)
            tail = text[begin:begin + SUPERSEDED_NEG_REPETITION_WINDOW]
            if tail and content_tokens(tail, stop_mode=policy.stop_mode, zh=policy.zh) & rec_tokens:
                return True
            start = text.find(marker, start + 1)
    return False


def _same_fact_correction(
    cand, cand_tokens: set[str], rec_tokens: set[str], policy: SkeletonPolicy
) -> bool:
    """wide / narrow 两条「同一事实」证据（任一成立即视为更正的就是这条被引记录）。"""
    shared = (cand_tokens - SUPERSEDED_MARKER_TOKENS) & rec_tokens
    if len(shared) >= SUPERSEDED_MIN_SHARED_TOKENS:
        return True
    return _negated_repetition(cand.content, rec_tokens, policy)


def _superseded(ledger, claim, ctx: ScenarioContext, policy: SkeletonPolicy) -> PredicateResult:
    name = "scenario_superseded"
    claim_tokens = _claim_tokens(claim, policy)
    if not claim_tokens:
        return PredicateResult(name, True, "hard", "claim has no assessable content token")
    records = ledger.records
    # 每条记录只 token 化一次（更正候选消息可能很长；重复 token 化既是浪费也让大消息拖慢判定）
    token_cache: dict[str, set[str]] = {}

    def tokens_of(rec) -> set[str]:
        got = token_cache.get(rec.id)
        if got is None:
            got = _rec_tokens(rec, policy)
            token_cache[rec.id] = got
        return got

    for rec in _cited_records(ledger, claim):
        rec_tokens = tokens_of(rec)
        # claim↔record：断言与被引记录毫无内容词交集 → 引用与断言无关，不判"已被更正"
        if not (claim_tokens & rec_tokens):
            continue
        for cand in records:
            if cand.source_id != rec.source_id or cand.episode_id != rec.episode_id:
                continue
            if cand.turn_index <= rec.turn_index:
                continue
            if cand.speaker != "user":
                continue
            if not zh_corrected(cand.content):
                continue
            cand_tokens = tokens_of(cand)
            # claim_link：更正消息与断言有内容词交集（更正的是这条断言说的事）
            if not (claim_tokens & cand_tokens):
                continue
            # 同一事实：更正消息确实在更正这条**被引记录**（wide 或 narrow）
            if not _same_fact_correction(cand, cand_tokens, rec_tokens, policy):
                continue
            return PredicateResult(
                name,
                False,
                "soft",
                f"cited record {rec.id!r} superseded by user correction "
                f"{cand.id!r} (turn_index {cand.turn_index} > {rec.turn_index})",
            )
    return PredicateResult(name, True, "soft")


def _evidence_incomplete(ledger, claim, ctx: ScenarioContext) -> PredicateResult:
    """Plan Z C2：被引 assistant 记录是残缺回答（`is_complete is False`）→ hard fail。

    确定性遍历顺序与其它谓词一致（主引用 → supporting 声明顺序、去重）。`is_complete is None`
    （旧记录 / 无完整性证据）→ passed 但 detail 标 `completeness_unknown`，不追溯改旧行为。
    `ctx` 不参与判定（完整性与语境无关），保留参数是为了与其它谓词同签名。
    """
    unknown: list[str] = []
    for rec in _cited_records(ledger, claim):
        if getattr(rec, "speaker", None) != "assistant":
            continue
        is_complete = getattr(rec, "is_complete", None)
        if is_complete is False:
            return PredicateResult(
                name=EVIDENCE_INCOMPLETE,
                passed=False,
                severity="hard",
                detail=(
                    f"cited assistant record {rec.id!r} is an incomplete/truncated answer "
                    f"(is_complete=False, finish_reason={getattr(rec, 'finish_reason', None)!r}); "
                    "a partial answer must not be used as evidence"
                ),
            )
        if is_complete is None:
            unknown.append(rec.id)
    if unknown:
        return PredicateResult(
            name=EVIDENCE_INCOMPLETE,
            passed=True,
            severity="hard",
            detail=(
                f"{COMPLETENESS_UNKNOWN}: cited assistant record(s) "
                f"{', '.join(repr(a) for a in unknown)} carry no completeness evidence "
                "(legacy records; not retroactively rejected)"
            ),
        )
    return PredicateResult(EVIDENCE_INCOMPLETE, True, "hard")


def check_scenario(
    ledger,
    claim,
    ctx: Optional[ScenarioContext],
    *,
    policy: SkeletonPolicy,
) -> list[PredicateResult]:
    """跑五个场景谓词，固定顺序返回五条 `PredicateResult`（契约 v1.1）。

    `ctx is None` 或 `policy.scenario is False` → 返回空列表（契约 §5「不执行」；
    `enforce()` 在 §6 的接入点也会先判一次，这里再兜一道，保证直接调用本函数时语义一致）。
    """
    if ctx is None:
        return []
    if not policy.scenario:
        return []
    return [
        _timeline(ledger, claim, ctx),
        _cross_session(ledger, claim, ctx),
        _speaker_scope(ledger, claim, ctx),
        _superseded(ledger, claim, ctx, policy),
        _evidence_incomplete(ledger, claim, ctx),
    ]
