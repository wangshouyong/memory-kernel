# ALGO-V4 契约（固定版）

> 背景：真实中文聊天域评测（`评测报告`）暴露出四个结构性缺口：
> ① 中文否定/时间谓词是英文词表 → 中文否定 0/7 被拦、SQL 里的英文 NOT 反而误杀；
> ② Tier 1 分词只认 `[A-Za-z]+` → 中文 0 token，判定空转；
> ③ `Claim.supporting` 不逐条校验 → 过度引用 0/19 可见；
> ④ 没有场景谓词（时间方向/跨会话/说话人范围/已被更正）→ 场景不适用不可见。
>
> 本契约定义 **v4 能力层**：新增中文词法层、引用集校验、场景谓词。**默认策略行为必须逐字节不变**，
> 全部新能力通过 `SkeletonPolicy` 新字段显式开启（opt-in），符合公理 B（执行路径不调用任何模型）。
>
> **版本记录（只增不删）**：v4.0（本契约主体）→ v4.1–v4.6（§3.2.1 / §4.1 / §5.1 / §6.1 修订）
> → **v4.7-meta（2026-11-14 增量补充，见 §9）**：F1 Phase 1 的 `quantifier_generalization` 与
> `source_attribution` 两条 soft-only 元数据谓词、`SkeletonPolicy.v4_meta()` 入口与封闭映射。
> 实现侧记录见 本文 §9；本文补的是**契约条目**，不改任何代码/测试/SPEC。

> **版本矩阵（契约，只增不删）**：`skel-v2.0`（默认）→ `skel-v3.0`（`.v3()`）→
> `skel-v3.1-C3`（`.v3_c3()`，C3 唯一正式入口；四开为实验配置、不得用于生产）→
> `skel-v4.0`（`.v4()`，中文词法/引用集/场景）→ v4.1–v4.6 修订 →
> `skel-v4.7-meta`（`.v4_meta()`，F1 Phase 1 元数据 soft 谓词）。

## 0. 不可违反

1. `enforce()` / `skeleton_check()` / 谓词实现内**不得引入任何模型调用或网络**；只用 stdlib。
2. **向后兼容**：所有新参数必须有默认值且默认行为与现状逐字节一致；`SkeletonPolicy()` 仍是 v2.0。
3. 现有测试必须全绿；不得改测试迁就实现（新增测试可以）。
4. 确定性：同一输入重复运行输出一致；不得使用时间、随机数、dict 迭代顺序以外的不可控状态。
5. 不改已固定的评测产物；新增评测产物只写私有目录（不发布）。

## 1. `SkeletonPolicy` 新增字段（`memory_kernel/skeleton.py`）

```python
zh: bool = False          # 中文词法层（分词/否定/时间）
cite_set: bool = False    # 引用集校验（过度引用）
scenario: bool = False    # 场景谓词
```

新增类方法：

```python
@classmethod
def v4(cls) -> "SkeletonPolicy":
    return cls(version="skel-v4.0", tau_soft=0.45, hard_units=True, temporal=True,
               zh=True, cite_set=True, scenario=True, arg_slots=True, dispatch=True)
```

### 1.1 v4.7-meta 追加字段与入口（增量补充，见 §9）

```python
# SkeletonPolicy 追加两个字段（默认 False → 既有版本逐字节不变）
quantifier: bool = False    # 量词泛化（soft，谓词 quantifier_generalization）
source_type: bool = False   # 来源归属（soft，谓词 source_attribution）

POLICY_VERSION_V4_META = "skel-v4.7-meta"

@classmethod
def v4_meta(cls) -> "SkeletonPolicy":
    # = v4() 全部开关 + quantifier=True + source_type=True
    return cls(version=POLICY_VERSION_V4_META, tau_soft=0.45, hard_units=True,
               temporal=True, zh=True, cite_set=True, scenario=True,
               arg_slots=True, dispatch=True, quantifier=True, source_type=True)
```

* **唯一正式入口**是 `SkeletonPolicy.v4_meta()`（或显式字段 opt-in）；`SkeletonPolicy()` 仍是
  v2.0、`.v4()` 仍是 v4.0 —— 两开关默认关，默认路径/固定产物逐字节不变。
* `meta_enabled` property = `enabled and (quantifier or source_type)`；`describe()` 追加
  `meta(quantifier=0|1, source_type=0|1)` 片段（只在 `meta_enabled` 时）。
* 版本号常量 `POLICY_VERSION_V4_META` 定义并列入 `memory_kernel/skeleton.py::__all__`；
  顶层 `memory_kernel/__init__.py` 目前只导出到 `POLICY_VERSION_V31_C3`（既有导出边界，
  不在本条目扩范围；调用方从 `memory_kernel.skeleton` 导入常量，行为入口用 `v4_meta()`）。

## 2. 中文词法层：新文件 `memory_kernel/zh.py`

全部纯函数 + 常量表，禁止外部依赖。必须提供：

| API | 语义 |
|---|---|
| `has_cjk(text) -> bool` | 是否含 CJK 字符 |
| `cjk_ratio(text) -> float` | CJK 字符 / 非空白字符（空串返回 0.0） |
| `zh_tokens(text) -> list[str]` | 确定性切分：连续汉字段 → **字符 bigram**（长度 1 的段取 unigram）；ASCII 字母数字段 → 小写原样。**顺序 = 原文出现顺序，允许重复** |
| `ZH_STOP: frozenset[str]` | 中文停用词（的/了/是/在/我/你/他/她/它/我们/这/那/和/与/就/也/都/很/还/又/会/能/要/有/一个/这个/那个…），只收高频虚词，**不得**含否定词 |
| `zh_negated(text) -> bool` | 是否含中文否定（没有/没/从未/不曾/并非/并不/绝不/不是/不用/未/无/别/甭/毫无） |
| `zh_temporal_terms(text) -> set[str]` | 归一化时间词：昨天/前天/今天/明天/后天、上周X、下周X、上个月/下个月、去年/今年/明年、`\d{4}年`、`\d{1,2}月\d{1,2}[日号]`、`\d{1,2}点`、`\d+天前`、`\d+个月后` → 统一成可比较的规范串（如 `2026-03-05`、`上周三`） |
| `zh_corrected(text) -> bool` | 更正标记：不是/改成/其实/更正/说错/改口/记错/不对 |
| `zh_anaphor(text) -> bool` | 指代：上次/上回/之前/刚才/前面说/我们说/你说过/刚才那 |
| `zh_hedge(text) -> bool` | 推测/建议：可能/也许/大概/或许/建议/应该/可以/不妨/大概率 |

## 3. 中文接入（`skeleton.py` / `obligations.py`）

### 3.1 分词
```python
def content_tokens(text: str, *, stop_mode: str = "v2", zh: bool = False) -> set[str]:
    # zh=False：现有行为逐字节不变
    # zh=True：现有英文 token ∪ (zh_tokens - ZH_STOP)
```
`idf` 构建/缓存键必须包含 `zh`（`idf_for_ledger(..., zh=False)` / `build_idf(..., zh=False)`）。
`skeleton_check` 用 `policy.zh` 选择；`claim_arguments`/`arg_lemma` 的中文扩展可选（不阻塞本轮）。

### 3.2 Tier 0 谓词（`apply_obligations(..., zh: bool = False)`）
- `preserve_negation`：
  - 当 `zh=True` 且 `scope`/`quote` 任一侧含 CJK → 用 `zh_negated` 极性比较（一侧否定、另一侧不否定 → 失败）；**不再**用英文 NEGATION 正则判中文文本（避免 SQL 里的 `NOT` 误杀）。
  - 当 `zh=True` 且两侧都不含 CJK → 保持现有英文逻辑。
  - 当 `zh=False` → 现有逻辑逐字节不变。

#### 3.2.1 v4.1 修订（首轮对照评测后的设计取舍）

首轮 v4 在真实探针上 FPR_strict 7/71→14/71，新增误杀**全部**是中文否定用**整句**范围比较片段引用
（"我没去杭州，但我朋友去了"引用后半句被判丢否定）。修订两点：

1. **子句级范围**：`preserve_negation` 的 zh 路径改用 `zh_clause_scope(rec.content, offset, len(quote))`
   （按 `，,；;。！？!?\n` 切分，覆盖引用所在子句），不再用整句 `obligation_scope`。
   语义：只判"同一子句内被引用片段省略掉的否定"；其他子句的否定与本次引用无关。
2. **断言级极性**（`enforce()` 内新增硬谓词 `preserve_negation_claim`）：`policy.zh=True` 且
   `claim.text` 含 CJK 时，比较 `zh_negated(claim.text)` 与 `zh_negated(cit.quote)`，极性不一致
   → 硬失败（cause `preserve_negation_claim`）。这是中文 N_negation（claim 自行引入/丢掉否定）
   的真正修复点——Tier 0 的片段级比较看不到 claim 自己的否定。
   **v4.2 适用范围收窄**：仅当 claim 与引用片段的 zh content-token 覆盖率
   `|claim∩quote| / |claim| >= 0.6`（摘录型/近似摘录）时执行；自由转述不承担此义务，交给 Tier 1。
   依据实测分布：N_negation（插否定）0.82–0.93、P_verbatim/O_baseline 1.0、自由转述 0.00–0.40。
   不收窄会把「assistant 自己加否定的合法转述」误杀（v4.1 实测 P_paraphrase strict 0.05→0.40）。

方向性不变：标记表漏检 → 假接受；子句切分把"跨子句否定"从假杀里剔除。
- `preserve_temporal`：`zh=True` 时有 CJK 的一侧用 `zh_temporal_terms`，要求 **quote 的时间项 ⊆ scope 的时间项**；无法提取则不阻断（保持软谓词语义）。
- `preserve_speaker` / `verbatim` 不变。

### 3.3 校验
`apply_obligations` 的新参数默认 `zh=False`；`enforce()` 传 `zh=policy.zh`。

## 4. 引用集校验：新文件 `memory_kernel/cite_set.py`

```python
def check_citation_set(ledger, claim, *, zh: bool = False,
                       policy: SkeletonPolicy) -> list[PredicateResult]
```
对 `claim.supporting` 逐条：
1. anchor 不存在 → **hard** `over_citation_anchor`；
2. `quote` 不是该记录在 `offset` 处的连续子串 → **hard** `over_citation_verbatim`；
3. 相关性（v4.3 起为**增量覆盖**）：该记录的 content tokens 与 claim 的 content tokens 有交集、
   且贡献了「主引用 + 先前已接受 supporting」未覆盖的 claim 内容词 → 通过；**增量覆盖为 0**
   （无论是否共享常见词）→ **soft** `over_citation`（冗余/无关引用）。
   无 supporting 时返回空列表（不改变现状）。

#### 4.1 v4.3 修订（真实评测中 O_overcite 只抓到 3/19 后的设计取舍）

旧判据"与 claim 有交集即通过"对真实语料几乎无效：无关记录只要共享一个常见词就放行（3/19）。
改为增量覆盖后，19 条 O_overcite 的 supporting 全部贡献 0 个新 claim 内容词（主引用已覆盖 100%），
可全部命中；合法的多证据引用不受影响（第二条会贡献新词）。代价：**同题 corroboration（两条
记录说同一件事）也会被判冗余** —— 这是本规则明确定义的"过度引用"，报告按此口径读。
仍然是词面检查：共享词但事实不同的无关引用抓不到（假阴性方向）。

## 5. 场景谓词：新文件 `memory_kernel/scenario.py`

```python
@dataclass(frozen=True)
class ScenarioContext:
    context_session_id: Optional[str] = None      # 断言所在会话
    context_time: Optional[float] = None          # 断言时点（epoch 秒）
    asserted_scope: str = "unspecified"            # "user_fact" | "assistant_claim" | "unspecified"
    record_times: Mapping[str, float] = ()         # anchor_id -> epoch 秒（可选）
```

```python
def check_scenario(ledger, claim, ctx: ScenarioContext, *,
                   policy: SkeletonPolicy) -> list[PredicateResult]
```

| 谓词 | 条件（全部确定性） | 严重度 |
|---|---|---|
| `scenario_timeline` | `ctx.context_time` 有值且被引记录 `record_times[anchor] > context_time`（后事作前证） | hard |
| `scenario_cross_session` | claim.text 含指代（`zh_anaphor`）且 `rec.source_id != ctx.context_session_id` | soft |
| `scenario_speaker_scope` | `ctx.asserted_scope == "user_fact"` 且 `rec.speaker == "assistant"` 且 `zh_hedge(rec.content)` | hard |
| `scenario_superseded` | 同 `source_id`+`episode_id` 中存在 `turn_index > rec.turn_index` 的更正消息，且**满足同一事实判据**（见 5.1） | soft（v4.6 起） |

全部谓词在条件不满足时返回 `passed=True`；`ctx` 为 None 或 `policy.scenario=False` 时不执行。

### 5.1 v4.4 修订：superseded 的「同一事实」判据（设计取舍）

旧判据只要求"存在含更正词的后续用户消息 + 与 claim 有内容词交集"，连续更正会话里会串味
（真实实测：S_timeline_ctrl 2 条 + S_superseded_ctrl 1 条误标）。新判据要求更正的是**同一件事**：

1. 候选更正消息 `t`：同 `source_id`+`episode_id`、`turn_index > rec.turn_index`、`speaker == "user"`、`zh_corrected(t)`；
2. 必要守卫：`claim ↔ rec` 与 `claim ↔ t` 都必须有内容词交集；
3. `same_fact` = **wide**（`t` 与 `rec` 的内容词交集（扣除更正标记自身 token）≥ `SUPERSEDED_MIN_SHARED_TOKENS=2`）
   **或 narrow**（否定标记后 4 字内重复了 `rec` 的内容词，覆盖「不是昨天」型最小更正）。

阈值实测（`评测产物`）：min_shared=1 → ctrl 误标 2+1 全在；
min_shared=2 不扣标记 token → 仍有 1 条误标；所选组合是唯一同时满足「ctrl 误标 0 / 正常样本仍拦 /
既有测试不破」的配置。方向性：词面判据对"真被更正但零词重合"的样本仍会漏（假阴性）。

## 6. `enforce()` 接入（`memory_kernel/enforce.py`）

新签名（只新增可选参数）：
```python
def enforce(ledger, claim, *, scope_mode="clause", coverage_full=True, extra=None,
            policy=None, session_dates=None, idf_model=None,
            context: Optional[ScenarioContext] = None) -> EnforcementResult
```
执行顺序：
1. anchor 校验（不变）
2. Tier 0（`apply_obligations(..., zh=pol.zh)`）
3. Tier 1 骨架（`skeleton_check(..., policy=pol)`；idf 带 `zh=pol.zh`）
4. **`policy.cite_set` 且 claim.supporting** → `check_citation_set`：hard → `rejected` cause `citation_set`；soft → 至少 `degraded`
5. **`policy.scenario` 且 context 非 None** → `check_scenario`：hard → `rejected` cause `scenario`；soft → 至少 `degraded`

`REJECT_CAUSES` 追加 `"citation_set"`, `"scenario"`。失败谓词保留在 `result.failed` 里，
报告层用 `failed_predicates` 区分具体原因。

### 6.1 v4.4：`degrade_reason`（产品口径，不改判定）

`degraded` 不是错误，是"保留但降级"。为了让调用方知道**为什么降级**，`EnforcementResult` 新增：

```python
degrade_reason: Optional[str] = None          # 主原因（仅 degraded 时非 None）
degrade_reasons: tuple[str, ...] = ()         # 全部原因（固定顺序、去重保序）
```

映射（封闭表，见 `enforce.degrade_reason_for()` / `DEGRADE_REASON_PREDICATE_ORDER`）：

| 失败谓词 | reason |
|---|---|
| `skeleton_grounding` / `skeleton_hard_unit_grounding` | `partial_evidence` |
| `skeleton_arg_object` / `_arg_subject` / `_arg_predicate` / `_arg_pred_family` / `_arg_unit_coverage` / `_plural_uncovered` | `argument_mismatch`（v4.6） |
| `scenario_cross_session` | `cross_session` |
| `scenario_superseded`（v4.6 起 soft） | `scenario_superseded` |
| `over_citation` | `over_citation` |
| `preserve_temporal` | `temporal` |
| `no_generalization` | `over_generalization` |
| `quantifier_generalization`（v4.7-meta） | `over_generalization`（复用同口径，见 §9.3） |
| `source_attribution`（v4.7-meta） | `source_attribution`（新增封闭标签，见 §9.4） |
| 其它 `skeleton_*` | `skeleton_<reason>` |

主原因 = 按 enforce 1→5 级执行顺序的第一个失败原因；`verified` / `rejected` / `coverage_full=False`
一律保持 `None` / `()`。**v4.6 起 `scenario_superseded` 由 hard 改 soft**：不再 rejected，而是
degraded + `degrade_reason="scenario_superseded"`，由 `emit.py` 渲染为「这条后来被更正过」；
实测 FPR_strict 3/71 → 0/71、strict/loose TPR 不变。

## 7. `__init__.py` 导出

新增：`ScenarioContext`、`zh` 模块（可选，不导出单个函数也行）。不得改动既有导出含义。

**v4.7-meta（§9）导出补充**：`POLICY_VERSION_V4_META` 已在 `memory_kernel/skeleton.py::__all__`；
`SkeletonPolicy.v4_meta()` 为公开入口。`quantifier` / `source_type` 字段与谓词函数
（`obligations.py::quantifier_generalization` / `apply_obligations(source_type=...)`）不强制
从 `memory_kernel` 顶层导出；顶层导出清单是否需要扩容留给后续任务，本文不改变既有导出边界。

## 8. 测试与验证（两个实现都必须做）

1. `tests/test_zh.py`：
   - `zh_tokens` 确定性、bigram 正确（如 `"我上周去了杭州"` → `我上/上周/周去/去了/了杭/杭州`）；
   - `ZH_STOP` 不含否定词；`zh_negated` 命中"没有/从未/并非"，不命中"没问题的"（可接受误报需注释说明）；
   - `zh_temporal_terms` 归一（`2026年3月5日` → `2026-03-05`；`上周三` → `上周三`）；
   - **回归**：`content_tokens(text, zh=False)` 与改动前实现逐 token 相等（用 git show HEAD:memory_kernel/skeleton.py 的等价性断言或固定样例）。
2. `tests/test_cite_scenario.py`：
   - 过度引用：A 相关 + C 无关 → `degraded` 且 `over_citation` 在 failed；A+B 都相关 → `verified`；
   - 4 个场景谓词各 1 正 1 反；`context=None` 时结果与 v3 逐字节一致；
   - 默认策略（`SkeletonPolicy()`）在所有新用例上行为与改动前一致。
3. `python -m unittest discover -s tests` 全绿（既有 + 新增）。
4. `python `实现脚本` 退出码 0。
5. 不得修改 `固定评测产物` 下任何文件。

## 9. v4.7-meta 补齐（2026-11-14；契约条目，仅文档增量）

> 背景：F1 Phase 1 已按取舍选项 C 实现并验收（新增测试、全量测试通过、
> 评测/数据产物零改动）；能力边界见 `docs/ALGO-CEILING.md` 与聚合评测结果。
> 本文只把实现已固定的条目补进契约；**本节不引入新行为、不改代码/测试/SPEC**。
> 对应源码（只读引用）：`memory_kernel/{skeleton,obligations,enforce,emit}.py`。

### 9.1 版本号与入口（固定）

| 项 | 固定值 |
|---|---|
| 版本号 | `POLICY_VERSION_V4_META = "skel-v4.7-meta"`（`memory_kernel/skeleton.py`） |
| 唯一正式入口 | `SkeletonPolicy.v4_meta()`：`v4()` 全部开关 + `quantifier=True` + `source_type=True`，`version="skel-v4.7-meta"` |
| 两开关默认 | `quantifier=False`、`source_type=False`（`SkeletonPolicy()` 仍是 v2.0、`.v4()` 仍是 v4.0） |
| 派生属性 | `meta_enabled = enabled and (quantifier or source_type)`；`describe()` 仅在 `meta_enabled` 时追加 `meta(quantifier=…, source_type=…)` |
| 兼容承诺 | 默认路径判定与全部既有固定产物**逐字节不变**；153 探针 v4 判定矩阵指纹与 `results_v4.json` 一致（诚实测见 本文 §9 §4） |
| 回退 | 不使用 `v4_meta()` 即回到 v4.0（所有新代码路径由两开关守卫） |

### 9.2 `quantifier_generalization`（soft-only 元数据谓词）

| 字段 | 契约 |
|---|---|
| 名称 / 严重度 | `quantifier_generalization` / **soft**（只能 `degraded`，永不 `rejected`、不产生 verified 资格） |
| 词表（封闭） | 中文 `QUANTIFIER_CN = 所有/每个/每一/任何/总是/一直/全部/一律/永远/从来`；英文 `QUANTIFIER_EN = all/every/always/any/each/everyone/everything/everybody`（词边界、忽略大小写） |
| 否定守卫 | 中文：量词前 3 字符含 `不/没/未/非/无` → 该量词不算；英文：前文 24 字符内 `not/no/never/none` 命中 → 不算（"不是所有/not all" 不是全称断言） |
| 抽取 | `quantifiers_in(text) -> set[str]`（确定性，仅标记检测，不做语义强度比较） |
| 判定（单侧） | `claim 量词 − 证据量词 ≠ ∅` → soft 失败；证据出现**任一同名量词**即 passed（宁可漏、不假杀；例："each time" 覆盖 "each"）；claim 无量词 → passed |
| 证据集 | primary 记录 content + `claim.supporting` 对应记录 content；不可验证的 supporting 交给 Tier 1 / cite_set 报，不在此重复计罚 |
| 接线位置 | `enforce()` 第二级 Tier 0 补充：`apply_obligations(...)` 之后、`pol.quantifier and claim.text` 时追加（默认关） |
| 降级标签 | `degrade_reason = "over_generalization"`（与 `no_generalization` **同口径复用**；不新增语义） |
| 固定顺序位置 | `DEGRADE_REASON_PREDICATE_ORDER`：`no_generalization` 之后、骨架族之前 |

### 9.3 `source_attribution`（soft-only 元数据谓词）

| 字段 | 契约 |
|---|---|
| 名称 / 严重度 | `source_attribution` / **soft**（只能 `degraded`） |
| 来源派生（封闭） | `SPEAKER_SOURCE_TYPE`：`user→user`、`assistant→assistant`、`tool→tool`、`system→tool`、`document→document`、`external→document`；其余（含 `None`）→ `unknown`。`SOURCE_TYPES = user/assistant/tool/document/unknown` |
| 判定条件 | `asserted_scope == "user_fact"` **且** 来源 ∈ `NON_USER_SOURCES = {assistant, tool, document}` → soft 失败；`asserted_scope` 缺省 `unspecified` → **不判**（不猜来源） |
| 不重复计罚 | `source == "assistant"` 且 `zh_hedge(rec.content)` → 谓词 passed（同一缺陷由 `scenario_speaker_scope`(hard) 负责） |
| 接线位置 | `apply_obligations(..., source_type=policy.source_type, asserted_scope=context.asserted_scope if context else "unspecified")`，即 Tier 0 内（默认关） |
| 降级标签 | 新增封闭标签 `source_attribution`（`DEGRADE_REASON_LABELS` 收录）；`emit.py::HEDGED_BY_REASON["source_attribution"] = "{text}（这条依据不是用户本人说的，仅供参考）"` |
| 不新增 reject cause | `REJECT_CAUSES` 不变（无 `source_attribution`；soft-only 不能改变 rejected 语义） |

### 9.4 固定谓词顺序与封闭映射（v4.7-meta 增量，其余一字未动）

* `DEGRADE_REASON_OF`（封闭映射）新增两条：
  `"quantifier_generalization" → "over_generalization"`、`"source_attribution" → "source_attribution"`。
* `DEGRADE_REASON_PREDICATE_ORDER`（主原因取第一个）新增两条，位置固定在
  `no_generalization` 之后、`skeleton_grounding` 之前；两条之间的固定顺序为
  **`quantifier_generalization` 在前、`source_attribution` 在后**（以源码 tuple 为准）。
* `DEGRADE_REASON_LABELS`（封闭标签集）新增 `"source_attribution"`；`"over_generalization"` 已存在。
* 未收录谓词的 fallback 规则不变：`skeleton_*` → `skeleton_<reason>`，其余 → 谓词名本身；
  `degrade_reason_for()` / `_degrade_reason_rank()` 语义不变，只是新增上述两条记录。
* `REJECT_CAUSES`、`SCENARIO_PREDICATES`（5 条固定顺序）、`severity` 表（`SEVERITY`）其余条目、
  既有 `MODE_DEGRADE_REASON_OF` / `EVIDENCE_DEGRADE_REASON_OF` 均未改动。
* 幽灵谓词清理：`ALLOWED_EXTRA` 收敛为 `{"no_generalization"}`；`joint_consistent` 无实现，
  已从白名单移除 —— 写它现在**显式** `TemplateError`（不再"接受但静默不生效"）。
* 既有 `tests/test_degrade_reasons.py` 的"封闭映射/固定顺序"断言同步收录两条新标签
  （是契约新增，不是放宽既有断言；其余断言逐字未动）。

### 9.5 soft-only 与"未覆盖"声明（不得弱化）

1. **soft-only 硬约束**：两条谓词只产生 `degraded`；不产生 verified 资格、不改变任何
   `rejected_cause`、不替代 Tier 0 hard 判定。单测矩阵 + 153 探针只读复算锁定：
   verdict 变化 0、`rejected→非 rejected` 0；153 条上新增失败谓词 1 行
   （`P_paraphrase-13-m0` 的 `quantifier_generalization`，该行本就因 `no_generalization` 降级，
   原因集合不变）；`source_attribution` 触发 0（探针未显式传 `asserted_scope="user_fact"`，按设计不判）。
2. **量词域内 recall 未由 gold 度量**：词表是封闭常量表，覆盖范围 = 表内标记；表外表达
   （"全部人/全员/无一例外/without exception" 等）会漏检（**假接受方向**）。不承诺"量词泛化已治理"。
3. **中文条件句（conditions→scope）仍未实现**：`CONDITIONAL` 仍只有 6 个英文词
   （if/when/unless/whenever/in case/as long as）；`如果/要是/只要/除非/当…时` 不在任何词表
   （单测锁定"不匹配"，防止把未覆盖说成已覆盖）。Phase 2 启动前置：中文条件句 gold ≥60–100 条、
   双标 kappa ≥0.7；条件对条件的结构比较在 Phase 2。
4. **`source_type` 只是最小派生版**：只映射 `AnchorRecord.speaker`；无 provenance 标签
   （tool/document 具体类型、外部文档来源链、模型生成内容标记）；未列出 speaker 一律 `unknown` 不判。
5. **英文 assistant hedge 不被识别**：`zh_hedge` 只有中文标记表，英文 "I think / maybe" 不触发
   `scenario_speaker_scope`；此时由 `source_attribution` 报降级（不重复计罚），但 hard 层对英文
   hedge 无防线 —— 既有缺口，本批不修，不承诺已覆盖。
6. **无 gold / 无 LLM 全量重跑**：本批实测只到"既有 153 探针上的触发与方向"，未重跑
   LLM-in-the-loop（成本 ~2h10m），因此**不得**声称引用精度/覆盖率有提升。
7. **F1 持久化未实现**：本批只做"按需确定性编译"，不建 F1 表（fact_id/status/supersedes 后置）。

### 9.6 验收口径（与 本文 §9 一致）

```bash
PY="python"
python -m unittest tests.test_f1_phase1        # 39 条（单元/集成/默认零变化/未覆盖/确定性/AST）
python -m unittest discover -s tests            # 全量
git status --porcelain -- eval data               # 必须为空（不碰评测/数据/固定产物）
```

* 默认零变化：`SkeletonPolicy()` / `.v4()` 不打开新字段 → 判定与固定指纹逐字节一致；
  `v4_meta` 在无元数据触发时与 `v4` 的 verdict/cause/degrade_reason/失败谓词一致
  （只允许新增**通过态** soft 审计项）。
* 确定性：同输入重复运行判定签名一致；新代码路径无网络/模型/时钟/随机依赖（AST + 源码扫描锁定）。
* 153 固定指纹：私有语料存在时执行；缺失时 skip（OSS CI 友好），不得以 skip 冒充通过。

### 9.7 与本文其它节的关系

* §1.1（字段/入口）、§6.1（降级映射表）、§7（导出）已同步补 v4.7-meta 条目；
  其余节（§2 中文词法、§3 Tier 0、§4 引用集、§5 场景谓词、§6 接入顺序、§8 测试）**一字未删**，
  本次只是在 §9 汇总 v4.7-meta 的差异，避免把"元数据层"混进 v4 能力层的原始条目。
* 方向不变：`enforce()` 零模型；元数据层**只允许产生降级，不允许产生事实资格**（SPEC §3.2）。
