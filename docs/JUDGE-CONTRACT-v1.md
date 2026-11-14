# Judge 契约 v1（语义正确性独立判定 · 评测专用）

> **公开版（public edition）**：仅保留对外可发布的契约与口径；不含私有路径、真实聊天片段、任务编号或逐题评测产物。

> 版本：`judge-prompt-v1` / `judge-label-v1`（2026-11-13，双模型路径：免费双模型 + 分歧人工复核）。
> 实现：`<评测脚本，不随仓库发布>`（脚本内 `JUDGE_SYSTEM` 常量必须与本文件 §2 逐字一致）。
> 上游：`docs/ALGO-CEILING.md`（语义鸿沟声明）、`docs/LLM-IN-THE-LOOP-REPORT.md` §5.3（未覆盖项）。
> **本文档定义的是"评测口径"，不是执法逻辑。**

## 0. 不变量（违反即作废）

1. **judge 不进执法层**：`memory_kernel/enforce.py`、`memory_kernel/emit.py` 的判定与输出
   **不依赖** judge 结果；judge 只产出评测数字。任何"用 judge 结果改写发射内容"的实现都
   违反公理 B（执法不调用模型），不得合入。
2. **judge 不看探针标注**：judge 输入只有「用户问题 / 断言 / 被引证据原文」，不提供
   `expect`/`cls`/gold 标注，避免标签泄漏。
3. **模型与 Prompt 成对记录**：生成模型 + Prompt 版本 + judge 模型 + judge Prompt 版本必须
   写进评测产物 `meta`；任一变更都要重跑并升版本。
4. **失败如实记账**：judge 调用失败/JSON 坏 → 记 `error`，不进分母、不静默当 supported。

## 1. 评测单位与输入

- 单位：模型生成输出里**带引用**（`cites` 非空）的一条 `claim`。
- 输入三件：探针问题原文（用户那句话）、claim 断言文本、**主引用**（`cites[0]`）指向的
  证据原文（截断 1200 字符；被引证据的池内来源 kind 一并给出）。多引用时 `n_cites` 记录，
  但判定只针对主引用——因为执法层以主引用为准（`emit_answer` 取 `cites[0]` 做 `enforce`）。
- 证据池构造与 <评测脚本，不随仓库发布>` **完全一致**（复用其 `build_pool`/`chat`/`parse_segments`），
  保证与全量 153 条评测同口径。

## 2. Judge Prompt v1（原样，不得改写措辞）

System：

```
你是独立的"引用语义适用性"判官。输入三部分：用户的问题、助手给出的一句断言、
助手为该断言引用的证据原文。你的唯一任务是判断：**该证据原文是否在语义上支持这句断言**。

判定标签（只能选一个）：
- supported：证据明确支持断言的核心内容（允许合理转述、概括、同义表达；不要求逐字相同）。
- partial：证据只支持断言的一部分，或断言明显超出证据的范围（主体/时间/数量/范围被放大）。
- unsupported：证据与断言话题相近但语义上不支持（例如同话题不同事实、与断言无关的内容、
  断言是证据没有的推断、或肯定/否定极性相反）。
- unclear：证据残缺/被截断/指代无法解析，无法判断语义关系。

纪律：
1. 只依据给出的证据原文判断，不得使用外部知识或自行补全证据。
2. 只判语义适用性，不判逐字一致性（逐字由确定性层负责）；措辞不同但语义相同 → supported。
3. 断言是常识/推测（没有证据也能成立）时，仍按"该证据是否支持它"判断，不要因为断言正确就给 supported。
4. 输出必须是合法 JSON（不要 Markdown、不要解释文字）：
{"support":"supported|partial|unsupported|unclear","reason":"不超过60字的中文理由"}
```

User 模板：

```
用户问题：{question}
断言：{claim}
引用证据原文（编号 {eid}，来源 {kind}）：
{evidence}

请输出 JSON。
```

调用参数：`temperature=0`、`response_format={"type":"json_object"}`、`max_tokens=4000`、
**不传 `reasoning`**（judge 模型 A 只声明 high 档，传 low 有兼容风险）。

## 3. 标签语义（判定边界）

| 标签 | 含义 | 典型例子 |
|---|---|---|
| `supported` | 证据在语义上支持断言核心内容 | 断言是证据的转述/概括/同义句 |
| `partial` | 只支持一部分，或断言超出证据范围 | 证据说"想买滴耳液"，断言说"已买"；证据说"某月"，断言说"每月" |
| `unsupported` | 话题相关但语义不支持 | 引用同一话题的另一条事实；断言是证据没有的推断；极性相反 |
| `unclear` | 证据残缺/截断/指代无法解析 | 引用的证据被截成半句、代词无先行词 |

- `unclear` **不计入** strict/lenient/残差分母，单列比率（`unclear_rate`）。
- 逐字性、否定、时间、说话人、场景、引用集等**机械性质不由 judge 判**（那是执法层的职责，
  judge 只判"语义是否支持"，避免与执法层重复或冲突）。

## 4. 抽样与生成（为什么不是直接判既有全量产物）

- **既有产物不可直接判**：`<私有产物，不随仓库发布>` 的 `audit` 只存
  `anchors/verdict/action`，不存 claim 原文与 quote，无法离线给出"断言 × 证据原文"。
- 因此按**同生成模型（`生成模型`）+ 同 Prompt 版本（`prompt-v1.2`）+ 同证据池构造**
  重生成抽样探针，取带引用 claim 判定。这是"取原文"，不改任何口径。
- 分层抽样（`DEFAULT_QUOTAS`，seed=20261113，确定性、与文件顺序无关）：

| 类 | P_verbatim | P_paraphrase | O_baseline | O_overcite | O_overcite_pair | N_negation | N_speaker | N_unrelated | N_partial | N_number | S_timeline | S_timeline_ctrl | S_cross_session_ctrl | S_speaker_scope | S_speaker_scope_ctrl | S_superseded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 探针数 | 5 | 5 | 4 | 4 | 4 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |

合计 35 探针。P_verbatim 权重最高（全量 strict 89.7%，最弱类）；`N_cross` 全量无带引用 claim，不取样。

## 5. 双 judge 与一致率

- judge A：`judge 模型 A`；judge B：`judge 模型 B`。二者均**不是**生成模型（生成用 生成模型），
  降低自我偏好（self-preference）偏差。
- **一致率**：两位 judge 都给出合法非 `unclear` 标签的 claim 上，
  `exact_agreement` = 完全相同比例；`cohen_kappa` = 多类别 Cohen's kappa（4 标签）。
- **共识**：两判一致 → 采用该标签；不一致 → `disputed`，**不计入**语义支持/残差率，
  进 `disagreements` 清单交**人工终审**。
- 只有一位 judge 给出结论（另一位失败/出错）→ `single_judge`，不计入共识指标，单列计数。

## 6. 指标定义（正式口径）

| 指标 | 定义 | 说明 |
|---|---|---|
| 语义精度 strict | `supported / (supported+partial+unsupported)` | 共识口径；越高越好 |
| 语义精度 lenient | `(supported+partial) / 同上` | partial 计通过 |
| **语义残差率** | `unsupported / 同上` | **本任务核心**：语义上不支持却带引用的比例 |
| 机械 verified 语义残差率 | 在 `machine_verdict == verified` 子集上算 `unsupported / n` | 关键交叉：机械判 verified 但语义不支持的比例 |
| 机械×语义交叉表 | 按 `machine_verdict` 分组统计四标签 | 用于报告 |
| 一致率 / kappa | 见 §5 | 判定可信度 |
| `unclear_rate` | `unclear / 全部判定` | 证据残缺导致的不可判 |

**报告纪律**：`disputed` 与 `unclear` 必须与主指标一起列出，不得只报有利数字；
样本量小（35 探针级）时必须给样本数与置信区间提示，不得外推为全量结论。

## 7. 版本与复跑

- 改 judge prompt / 换 judge 模型 / 换生成模型或 Prompt / 改抽样配额 → 必须升版本并重跑，旧结果归档不覆盖。
- **复跑脚本与评测语料不随仓库发布**：公开仓库只提供本契约（指标定义、判读规则）；
  私有语料、逐题判定与人工终审清单永不导出。
- 报告只放聚合数字与脱敏示例（片段泄漏检查）。

## 8. 已知限制（如实）

1. 抽样 35 探针（实际 51 条带引用 claim），CI 宽，只作"首次量化"，不外推全量。
2. judge 同为 LLM，存在系统性宽松/严格偏差；双判一致率高不代表 judge 本身正确，
   所以分歧项必须人工终审，且本指标**不得**用于改验收阈值。
3. 只判主引用；多引用 claim 的次要引用是否适用未单独判（`n_cites` 已记录）。
4. 证据截断 1200 字符；超长证据可能被 judge 记为 `unclear`（计入 unclear，不算支持）。
5. 重生成的输出分布与原全量跑并非逐位一致（服务端非确定性），结论按"同口径抽样"表述。
6. 分歧项需人工终审；终审记录只作校准依据，不改 judge 自动结果。

## 9. 负例对照校准（judge-calibration-v1，2026-11-13 校准）

> 目的：既有评测在真实样本上双判全部 `supported`（0/51）**不能证明** judge 有区分度。
> 本校准用"构造即定义标签"的负例检验 judge 能否识别引用不适用。
> 实现：`<评测脚本，不随仓库发布>`；测试：`tests/test_judge_calibration.py`（9 条）。

**对照集**（只读现有探针与语料，不调生成模型、不改 judge prompt）：

| 集合 | 来源 | 期望 |
|---|---|---|
| 正例（20 条） | `P_verbatim` / `P_paraphrase` 确定性抽样（各 10） | `supported` |
| 负例（32 条） | `N_unrelated`(6) / `N_cross`(6) / `N_negation`(7) / `N_number`(2) / `N_speaker`(6) / `N_partial`(5) | 非 `supported` |

**指标（口径写死，不按类调参）**：
- 正例**特异性** = 判 `supported` 的比例。
- 负例**敏感性 strict** = 判 `unsupported` 的比例（最严）。
- 负例**敏感性 lenient** = 判 `unsupported` 或 `partial` 的比例（"证据没有完整支持该断言"）。
- 双判一致率 + Cohen's kappa；分歧、`unclear`、调用失败分开报。

**判读规则（先定，防止事后解释）**：
- lenient 敏感性 ≥ 80% 且特异性 ≥ 90% → judge 对明显负例有效，该次评测的 0/51 可作为
  "上界 + 有限可信"结论；仍须注明 judge 敏感性不是 100%。
- lenient 敏感性 < 80% → judge 宽松偏差成立，该次评测语义残差率只能作**下界参考**，
  须另立任务换 judge/prompt（本契约内不得调参重跑到通过为止）。
- 本校准负例以"明显不适用"为主（无关/跨会话/极性/数值/说话人/片段），
  **不覆盖"同话题不同事实、措辞相近"的细微负例**——那类需要人工标注构造，另行立项。