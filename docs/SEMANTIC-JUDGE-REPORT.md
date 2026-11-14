# 语义正确性独立判定报告（公开脱敏版）

> **公开版（public edition）**：仅保留对外可发布的契约与口径；不含私有路径、真实聊天片段、任务编号或逐题评测产物。


> 日期：2026-11-13。写入方式：任务（设计取舍路径 A）。
> 实现：`<评测脚本，不随仓库发布>`；契约：`docs/JUDGE-CONTRACT-v1.md`；测试：`tests/test_semantic_judge.py`（17 条）。
> 数据：`<评测产物，不随仓库发布>` + `<人工终审清单，不发布>`（均 gitignored，含真实原文）。
> **judge 只用于评测，不进执法层**（公理 B：`memory_kernel/enforce.py`/`emit.py` 不依赖任何 judge 结果）。

## 0. 结论（先说结果，再说边界）

1. **51 条带引用 claim，双 judge 102 次判定全部 `supported`**：共识语义精度 **100%**
   （51/51，Wilson 95% CI 下界 93.0%），**语义残差率 0/51**（Wilson 95% CI 上界 **7.0%**）。
2. **机械 verified 的 48 条 claim 全部语义支持**（48/48），即本次抽样里
   **没有出现「机械 verified 但语义不支持」** 的案例。
3. **双判完全一致（51/51），Cohen's kappa 不可计算**（期望一致率 pe=1）——
   这**不等于** judge 可信度已被证明：一致率 100% + 全部同向，说明本样本**没有提供区分度信号**。
4. 因此正确表述是：**在该 35 探针样本上没有观察到语义残差，残差率上界 ≤7.0%（95% CI）；
   不能外推为「全量零语义残差」**。judge 在真实长证据上的宽松偏差无法用本样本排除，
   需负例对照校准（见 §6）。

## 1. 设置（可复跑）

| 项 | 取值 |
|---|---|
| 抽样 | 按 16 个类分层，seed=20261113，**35 探针**（确定性、与文件顺序无关） |
| 生成 | `生成模型` + `prompt-v1.2`（与全量 153 条评测同模型同 Prompt 同证据池构造） |
| 为什么重生成 | 既有全量产物 `<评测产物，不随仓库发布>` 的 audit 只存 anchors/verdict，**不存 claim 原文与 quote**，无法离线判定 |
| judge | `judge 模型 A`、`judge 模型 B`（免费、均非生成模型，降低自我偏好） |
| judge 输入 | 用户问题 + 断言 + **主引用证据原文**（截断 1200 字符）；**不看探针标注**（无标签泄漏） |
| 判定单位 | 模型输出里带引用（cites 非空）的 claim；多引用只判主引用（`n_cites` 记录） |
| 限速/续跑 | 顺序执行、每判间隔 6s、429/5xx 退避、progress 续跑 |
| 执行结果 | 35/35 生成成功、0 解析失败、102/102 judge 调用成功、0 错误 |

抽样分布（35 探针）：P_verbatim 5、P_paraphrase 5、O_baseline 4、O_overcite 4、O_overcite_pair 4、
N_negation 2、N_number 2、N_speaker 1、N_unrelated 1、N_partial 1、S_timeline 1、S_timeline_ctrl 1、
S_cross_session_ctrl 1、S_speaker_scope 1、S_speaker_scope_ctrl 1、S_superseded 1。
其中 3 个探针本次生成没有带引用 claim（N_partial-05、O_baseline-00、O_overcite_pair-05），
最终判定 **51 条 claim**（按类：O_baseline 14、P_verbatim 9、P_paraphrase 5、O_overcite 4、
O_overcite_pair 4、N_negation 2、N_number 2、N_speaker 2、S_superseded 2、S_timeline 2、
N_unrelated 1、S_cross_session_ctrl 1、S_speaker_scope 1、S_speaker_scope_ctrl 1、S_timeline_ctrl 1）。

## 2. 指标定义（与 `docs/JUDGE-CONTRACT-v1.md` §6 一致）

- 语义精度 strict = `supported / (supported+partial+unsupported)`（双判共识，分歧/unclear 不计）。
- 语义精度 lenient = `(supported+partial) / 同上`。
- **语义残差率** = `unsupported / 同上`（本任务核心：语义不支持却带引用）。
- 机械 verified 语义残差率 = 在 `machine_verdict == verified` 子集上算 unsupported 比例。
- 一致率 = 两位 judge 标签完全相同的比例；kappa = 多类别 Cohen's kappa。

## 3. 结果

| 指标 | 结果 | 说明 |
|---|---|---|
| 判定 claim 数 | **51** | 双判均成功，0 单判、0 错误 |
| 语义精度 strict（共识） | **100%**（51/51，Wilson 95% CI 93.0%–100%） | 全部 supported |
| 语义精度 lenient | 100% | partial 也为 0 |
| **语义残差率** | **0%（0/51）**，95% CI 上界 **7.0%** | 未观察到 unsupported |
| 机械 verified 语义残差率 | **0%（0/48）** | 48 条 verified 全部 supported |
| unclear | 0 | 证据截断未造成不可判 |
| 双判分歧 | 0 | 人工终审清单为空（0 条） |
| 一致率 | 100%（51/51） | exact agreement |
| Cohen's kappa | 不可计算（pe=1） | 全同标签时 kappa 无定义，如实记录 |
| judge 调用错误 | 0/102 | 无 JSON 失败、无限流失败 |

**机械 × 语义交叉表**（关键交叉）：

| 机械判定 | n | supported | partial | unsupported |
|---|---|---|---|---|
| verified | 48 | 48 | 0 | 0 |
| degraded | 2 | 2 | 0 | 0 |
| rejected | 1 | 1 | 0 | 0 |

**口径一致性核对**：本样本机械 strict 精度 = 48/51 = **94.1%**（Wilson 95% CI 84.1%–98.0%），
与全量 153 条的 94.58%（192/203）一致 → 抽样生成与门控链路行为没有漂移。

> 备注：机械 `rejected` 的 1 条被判 `supported`，与算法天花板声明一致——执法层拒绝它的原因
> 不是「语义不适用」，而是逐字/极性等**机械性质**硬错误（语义层看它是适用的）。
> 这正是两层分工的设计预期，不是矛盾。

## 4. judge 判定能力自检（构造用例，非真实数据）

在正式抽样前用 5 条**人工构造**用例（不含真实聊天原文）验证四标签可区分，两个 judge 均 **5/5** 正确：

| 用例 | 期望 | judge 模型 A | judge 模型 B |
|---|---|---|---|
| 转述支持 | supported | supported | supported |
| 断言超出证据（只支持一部分） | partial | partial | partial |
| 同话题不同事实（周五 vs 周三） | unsupported | unsupported | unsupported |
| 极性反转（想买 vs 不想买） | unsupported | unsupported | unsupported |
| 证据被截断 | unclear | unclear | unclear |

自检过程本身也暴露过一次口径执行问题：最初我把「每天三次 vs 证据每天两次」标成 partial，
按契约应判 **unsupported**（与证据数量矛盾，不是「超出范围」）；judge 判 unsupported 才是对的，
用例已按契约修正后重测。这说明四标签边界**需要按契约读**，不是随手打标签。
产物：`<评测产物，不随仓库发布>`。

## 5. 解释与限制（必须一起读）

1. **全 supported ≠ 证明零语义残差**：点估计 0%，但 95% CI 上界 7.0%；51 条样本量太小，
   且样本偏向 P_*/O_* 类（O_baseline 一条高噪声探针就贡献了 14 条 claim）。
2. **judge 宽松偏差无法排除**：一致率 100% 但全部同向 = 没有区分度信号。构造负例能判对，
   只说明"标签能力存在"，不说明 judge 在真实长证据上不会一律说 supported。
   **要证明 0% 可信，必须做负例对照校准（§6）。**
3. **人工终审清单是空的**（0 分歧、0 unclear）：路径 A 的"分歧人工复核"环节本次无对象可审。
   这是如实结果，不是跳过了人工环节。
4. **单主引用判定**：多引用 claim 的次要引用是否适用未单独判（本样本 `n_cites>1` 为 0 条，
   影响小）；证据截断 1200 字符。
5. **重生成与原全量跑非逐位一致**（服务端非确定性），结论按"同口径抽样"表述。
6. **judge 不进执法层**：本报告数字不改变 `enforce`/`emit` 任何行为；也不作为放宽/收紧验收阈值的依据。
7. 本报告与 `<私有产物，不随仓库发布>` 产物含聚合数字；分歧清单（含真实原文）只存 `<私有产物，不随仓库发布>`，禁止进 git/报告。

## 6. 下一步建议（需设计取舍）

1. **负例对照校准（建议优先）**：从真实会话构造"同话题不同事实 / 极性相反 / 范围放大"的引用负例
   （确定性脚本构造，类似既有探针的 `N_*` 类），让两个 judge 判；若 judge 能稳定识别，
   则本次 0/51 残差结论可信度显著提升；若识别不了，则 judge 指标只能作**下界参考**，
   需换 judge prompt 或换 judge 模型。
2. **扩样本**：按同 seed 追加第二批（如每类配额翻倍，70 探针）可把残差率 CI 上界压到 ~3.5%；
   `--resume` 支持增量，成本约 30 分钟（免费网关）。
3. **若负例校准通过且扩样本仍 0**：可把「语义残差 ≤7%（0/51）」写入说明文档作为已知上界，
   并在 任务 结论中标注"语义正确性已量化、残差在抽样上界内"。

## 7. 复跑（）

评测协议、指标定义与判读规则见 `docs/JUDGE-CONTRACT-v1.md`；复跑脚本、模型网关与
评测输入语料属私有资产，**不随仓库发布**（公开仓库不含真实语料与私有评测脚本）。

## 8. 负例对照校准结果（judge-calibration-v1，用户 2026-11-13 批准）

目的：任务 双判 0/51 unsupported **不能证明** judge 有区分度。用"构造即定义标签"的探针对照：
正例 = P_verbatim/P_paraphrase 确定性抽样各 10 条；负例 = N_* 六类共 32 条
（unrelated 6 / cross 6 / negation 7 / number 2 / speaker 6 / partial 5）。
判官与 任务 完全一致（`judge-prompt-v1`、judge 模型 A + judge 模型 B，不看标签），
52 对 × 2 = **104 次判定，0 错误**。指标口径与判读规则先写在 `docs/JUDGE-CONTRACT-v1.md` §9。

### 8.1 指标

| 指标 | judge 模型 A | judge 模型 B | 共识（双判一致） |
|---|---|---|---|
| 正例特异性（判 supported） | 60.0%（12/20） | 50.0%（10/20） | 58.8%（10/17） |
| 负例敏感性 strict（判 unsupported） | 50.0%（16/32） | 62.5%（20/32） | 69.6%（16/23） |
| 负例敏感性 lenient（unsupported 或 partial） | 68.8%（22/32） | **90.6%（29/32）** | **91.3%（21/23）** |
| 双判一致率 / kappa | — | — | 76.9%（40/52）/ **0.644** |
| 分歧数 | — | — | 12/52 |

### 8.2 分类型（共识 strict / lenient 命中）

| 负例类 | n | judge 模型 A strict/lenient | judge 模型 B strict/lenient | 观察 |
|---|---|---|---|---|
| `N_cross`（跨会话错配） | 6 | 6/6 | 6/6 | 两判全中 |
| `N_negation`（极性反转） | 7 | 5/7（2 条判 supported） | **7/7** | judge 模型 A 对插入式否定有 2 次漏检 |
| `N_number`（数值篡改） | 2 | 1/2 | 1/2 | 两条均判 partial（数值不符但其余支持）→ lenient 全中 |
| `N_speaker`（说话人错配） | 6 | 0/3 | 0/5 | 两判都倾向 partial（内容对、归属错）→ strict 全漏 |
| `N_unrelated`（无关引用） | 6 | 3/4 | 4/5 | 各有 1 条判 supported（漏检） |
| `N_partial`（只覆盖片段） | 5 | 1/2 | 2/4 | 含 2–3 条 `unclear`（证据过短） |

正例侧：`P_verbatim` 特异性 judge 模型 A 10/10、judge 模型 B 8/10（合计 **18/20 = 90%**）；
**`P_paraphrase` 两判都只有 2/10 判 supported**（其余 unsupported/partial）。

### 8.3 判读（按预先写死的规则，不给事后解释）

- **规则层结论：未完全通过。** 预定义门禁 = lenient 敏感性 ≥80% **且** 特异性 ≥90%；
  实测 lenient 敏感性 91.3% 达标，但总特异性 58.8% 未达标。
- **拆解：特异性缺口全部来自 `P_paraphrase`。** `P_verbatim` 特异性 90%（18/20，刚好达标）。
  `P_paraphrase` 的"正例标签"本身语义上不成立——它的证据是 user 的**短句**，而 claim 是 assistant
  的**长转述/扩写**，judge 按 prompt 规则 3（"断言的每一部分都要有证据覆盖"）判 unsupported/partial
  是**合理严格**，不是 judge 错误。但这条同时说明：**"探针构造标签 = 语义真值"不成立**，
  转述类不能用机械标签当语义正例。
- **judge 区分度已被证明存在**：共识 lenient 敏感性 91.3%，且对 `N_cross`/`N_negation` 基本全中；
  judge **不会**把明显负例一律判 supported。因此 任务 的 0/51 **不是"judge 一律宽松"的伪影**，
  可信度显著提升（但仍不是证明）。
- **仍有实质漏检**：`N_speaker` strict 0/6（两判都判 partial）、judge 模型 A 对极性反转漏 2 条、
  `N_unrelated` 各有 1 条漏检、`unclear` 若干。**结论：语义残差率 0/51 仍只能当作上界（7.0%）**，
  本版本不承诺"零语义残差"；且 任务 的 claim 多为逐字引用（相对容易），转述类未覆盖。
- **双判一致 76.9% / kappa 0.644**（中等）——任务 在 51 条上 100% 一致，说明那批样本本身较"容易"，
  不能反推双判在困难样本上也一致。

### 8.4 后续建议（需设计取舍）

1. 把"双判 + 负例校准"绑定为评测协议：**换 judge 模型或改 judge prompt 必须重跑校准**
   （校准数据与 任务 同口径，成本约 25 分钟免费网关）。
2. 若要更严格结论：人工标注 20–30 条"同话题不同事实、措辞相近"的**细微负例**（本校准未覆盖，
   探针构造无法自动保证标签），另立任务。
3. `N_speaker` 类判官倾向 partial → 若产品把"归属错误"当硬错误，需要在 judge prompt 里
   明确"主体/归属不符 = unsupported"（属 prompt 升版本，须升 `judge-prompt-v2` 并全量重跑）。