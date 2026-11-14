# SPDX-License-Identifier: Apache-2.0
"""Tier 1 骨架（v2）：内容词接地（IDF 加权 + 硬单元）+ 时间论证接地（会话日期）。

判定归属：骨架失败是**软失败**（degraded），不是硬拒绝——它证明"证据不足以支撑该断言"，
但不能证明"引用是伪造的"。伪造由 Tier 0 的 verbatim 负责。

────────────────────────────────────────────────────────────────────────────
v2 相对 v1 的改动（每一项都由 eval/out/report6.md 的数据支撑，逐项可消融）
────────────────────────────────────────────────────────────────────────────
1. **词干化 + 词表**（`stem=True`, `stop_mode="v2"`）
   v1 的 `content_tokens` 不做词干化，`wants`/`want`、`researching`/`research` 被算成
   未接地内容词；STOP 表里没有代词（his/her/she），claim 里"自然改写必带"的词被计入分母。
   第五轮量化：未接地 token 里 19.9% 是代词/序数、6.5% 是纯词形差异——都是**度量缺陷**，
   不是模型漂移。
2. **硬单元 / 软词分离**（`hard_units=True`）
   token 拆成两类：
     - 硬单元：数字（日期表达式以外的）、专有名词、引号内容 → 要求 **100% 接地**；
     - 软词：其余内容词 → 用 IDF 加权覆盖率 ≥ `tau_soft`。
   理由：硬单元是"具体到可核实"的论元（人名、数量、引语），错一个就是断言换了事实，
   不能用"平均覆盖率"把它摊平；软词承担自然改写的自由度。
3. **IDF 加权**（`idf=True`）
   v1 是等权覆盖率，一个稀有专名和 the 一样重。IDF 由**锚定库本身**统计（确定性、纯计数、
   无模型），稀有的未接地 token 权重大 → 判罚重；常见词的改写权重大 → 判罚轻。
4. **时间接地接会话日期**（`temporal_session=True`）
   v1 的 `evidence_times` 只看 `rec.content`，看不见 episode 的会话日期；LoCoMo 的时间答案
   常常是"会话日期 + yesterday"，于是一条**答对**的日期型 claim 会被判 temporal 失败。
   v2 把每条被引用记录所在会话的日期并进证据时间池，并把相对时间（yesterday / last week /
   the previous day / 3 days ago …）**先解析成绝对区间再比较**。
5. **版本化**：`SkeletonPolicy.version` + `SkeletonResult` 携带版本、规则名、各分量，
   报告里每个数字都能对应到一个版本。

不改的东西：Tier 0 的判定语义（verbatim / 否定 / 时间 / 说话人）一行没动；Tier 1 仍然只
产出软失败；`failure_mode="rejected"` 的升级开关语义不变。

────────────────────────────────────────────────────────────────────────────
v3 相对 v2 的改动（`PolicyVersion = "skel-v3.0"`，**只能由新字段启用**，默认策略与 v2 逐位一致）
────────────────────────────────────────────────────────────────────────────
v2 的软覆盖率是**词袋**度量：claim 与全部引用并成一个词池算 IDF 覆盖率。它有三种结构性盲区：

1. **论元槽对齐**（`arg_slots=True`）：claim 拆成论元单元 (主体, 谓词, 客体)，逐槽接地——
   - 主体专名必须出现在某条引用里（或就是该引用的说话人）；缺失 → `arg_subject`。
     这条直接对症 "'Both Jon and Gina …' 只引了 Jon 那句话"（`report6.md` §6 的第一类 beyond）。
   - 事件性谓词（去掉助动词/私态/言语行为/轻动词后的第一个动词）必须接地，允许词干、
     已观测同义组、同词族形态差异；缺失 → `arg_predicate`。
   - 客体专名必须接地（言语行为动词的受话人除外）；缺失 → `arg_object`。
   - **每个单元**的覆盖率都要 ≥ τ_soft（v2 只看总体覆盖率，一句无关的第二句会被
     平均掉）；缺失 → `arg_unit_coverage`。这条对症 N5a / N_B 的"拼接第二句"。
2. **多引用分派**（`dispatch=True`）：
   - 并列主体（both X and Y / X and Y）断言 ≥2 个参与者时，需要 ≥2 条引用（除非证据
     本身是第一人称复数 "we/us/and I"）；否则 → `dispatch_plural`。
   - 每个事件核必须能被**单条**引用覆盖（谓词与主体名/客体词在同一条引用里），
     不允许"谓词在 A、客体在 B"地拼出来；否则 → `dispatch_event`。
     这条对症 N3 的诱饵引用：谓词在无关引用里恰好出现，但那条引用讲的不是这件事。
3. **版本化**：`POLICY_VERSION_V3 = "skel-v3.0"`；`SkeletonPolicy.v3()` 一次性打开两条规则。
   `SkeletonResult` 增补 `arg_ok / n_units / unit_cov_min / participants / arg_missing /
   dispatch_unassigned`，规则的第一个违反项写进 `rule`（顺序固定，见 `skeleton_check`）。

代价（必须在报告里如实读）：论元槽是**文本级**证据，不是语义级证据。claim 里指向"提问语境的
指代表达"（"the check-up"、"New York City"、"Aerosmith"）在引用原文里找不到时会被判成未接地，
即使标注者认为它被支持。`eval/out/report7.md` §10 给出这类单侧性代价的完整清单。

────────────────────────────────────────────────────────────────────────────
v3.1-C3（`skel-v3.1-C3`，用户裁决 C2 采纳的正式实现，2026-11-12）
────────────────────────────────────────────────────────────────────────────
C3 = v3.1 的**三条修复开关**子集 `{v31_nonevent, v31_noun_guard, v31_no_event_object}`，
`v31_np_object` 永久否决（净有害：P_lab 误杀 7/162→16/162、新增 24 条正类误杀，
`eval/out/report7b.md` §5/§9）。四个全开的 `v3_1()` 只是实验配置，不是生产入口。

    pol = SkeletonPolicy.v3_c3()      # τ=0.45 + 三开关，版本 skel-v3.1-C3
    pol = SkeletonPolicy.v3_c3(tau_soft=0.35)   # 只调 τ，其余保持 C3 生产点

**默认路径不变**：`SkeletonPolicy()` 仍是 v2.0、`SkeletonPolicy.v3()` 仍是 v3.0，
`v3_c3()` 是显式 opt-in。代价（报告 §5/§9）：7/7 误杀修好，同点少抓 24 条负类
（N_C1 11、N3/N3e 各 1…）；τ=0.45 上与 v3.0 打平（P_lab 4.3%、TPR_hard 83.3%）。

────────────────────────────────────────────────────────────────────────────
v4 能力层（`skel-v4.0`；本文件只实现其中的**中文词法接入**这一半，见下）
────────────────────────────────────────────────────────────────────────────
真实中文聊天域评测暴露的结构性缺口之一是 **Tier 1 分词只认 `[A-Za-z]+` → 中文 0 token**，
判定空转。v4 的做法不是"上分词器/上模型"，而是给 `content_tokens` / `norm_tokens` /
IDF 加一个显式开关 `zh`：

    pol = SkeletonPolicy.v4()          # zh=True, cite_set=True, scenario=True,
                                       # τ_soft=0.45 + arg_slots/dispatch，版本 skel-v4.0
    content_tokens("我们用 PostgreSQL 做过项目", zh=True)

* `zh=True`：现有英文 token **并上** `zh.zh_tokens(text) - zh.ZH_STOP`
  （连续汉字段 → 字符 bigram，长度 1 取 unigram；ASCII 段小写原样）。
* `zh=False`（默认）：表达式与 v4 之前**逐字节相同**，默认路径 token 集合零变化
  （回归门：`tests/test_zh.py` 里对固定样例的逐 token 断言 + `git show HEAD` 等价性）。
* IDF 的缓存键与构建路径都带 `zh`（`(stem_ok, stop_mode, zh)` 三元组）：中文 bigram 与
  英文词 MUST NOT 共用一份 IDF，否则 df 统计混在一起、权重不可解释。
* 论元接地（v3）的接地面也用同一个 `zh` 开关构造（`_arg_pool` / `_arg_coverage`），
  保证 claim 侧与证据侧在同一个比较空间里；`claim_arguments` 本身仍是英文浅层规则
  （中文 claim 抽不出英文论元 ⇒ 论元槽空转，不误杀也不假装检查过）。

**不属于本文件的 v4 部分**（由 `cite_set.py` / `scenario.py` / `enforce.py` 负责）：
`cite_set` / `scenario` 两个字段只在这里定义与版本化，谓词实现与 `enforce()` 接入在
对应模块；本文件对它们的语义只有"开关默认关闭、不改变任何既有路径"。
    """

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Mapping, Optional, Sequence

from .zh import ZH_STOP, zh_tokens

POLICY_VERSION = "skel-v2.0"
# v3 只通过新字段启用（`arg_slots` / `dispatch`）；默认 `SkeletonPolicy()` 仍是 v2.0 行为
POLICY_VERSION_V3 = "skel-v3.0"
# v3.1 = v3.0 + 三类误杀修复（非事件谓词族扩展 / 名词位置护栏 / 无事件谓词时客体槽不成立）。
# 只通过 `v31_*` 字段启用；`SkeletonPolicy.v3()` 与默认策略都**不**打开它们，
# 因此 v3.0 路径逐位不变（回归门见 eval/out/results7b.json:backcompat）。
POLICY_VERSION_V31 = "skel-v3.1"
# v3.1-C3 = v3.1 的**可用子集**（用户裁决 C2，2026-11-12）：只要三条修复开关，
# `v31_np_object` 永久关闭（数据否决：P_lab 误杀 7/162→16/162、+24 条正类误杀，
# 见 eval/out/report7b.md §5/§9）。唯一正式入口是 `SkeletonPolicy.v3_c3()`；
# `SkeletonPolicy()` 仍是 v2.0、`SkeletonPolicy.v3()` 仍是 v3.0。
POLICY_VERSION_V31_C3 = "skel-v3.1-C3"
# v4 = v3.1-C3 的接地工作点 + 三个能力层开关（中文词法 / 引用集 / 场景谓词），
# 版本号 `skel-v4.0`。**只通过 `v4()` 或显式字段启用**：`SkeletonPolicy()` 仍是 v2.0。
POLICY_VERSION_V4 = "skel-v4.0"
# v4.7-meta（internal-task-internal-task，F1 Phase 1）：v4 + 两条**元数据 soft 谓词**开关
# （`quantifier` → 量词泛化检测并入 `no_generalization`；`source_type` → 新增 `source_attribution`）。
# **只通过 `v4_meta()` 或显式字段启用**：`SkeletonPolicy()` 仍是 v2.0、`.v4()` 仍是 v4.0，
# 默认路径与全部既有冻结产物逐字节不变（见 `docs/F1-PHASE1.md`）。
POLICY_VERSION_V4_META = "skel-v4.7-meta"
TOKENIZER_VERSION = "tok-v2"
STEM_VERSION = "stem-s2"
IDF_VERSION = "idf-s2"

# --------------------------------------------------------------------------- #
# 1. 分词 / 词表 / 词干
# --------------------------------------------------------------------------- #

# 只取字母串；撇号作为词内分隔符保留下来再统一处理（don't / Joanna's）
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# v1 停用词表：**原样保留**，用于复现第五轮（`stop_mode="v1"`）的数字
STOP_V1 = set(
    """the and for you your that with have has had this was were are been being
they them their there here what when where which who whom how why can could would should
just also very really actually about into over after before from out not but all any some
been will shall may might must does did doing get got make made like know think want
    """.split()
)

# v2：v1 + 代词/限定词 + 助动词与缩写残基 + attribution 动词
_STOP_V2_EXTRA = """
i me my mine you your yours he him his she her hers it its we us our ours they them their
theirs myself yourself himself herself itself ourselves themselves this that these those such
same own another other each every both either neither few several many much more most less
least some any all none one two three
am is are was were be been being do does did done doing have has had having
will would shall should can could may might must ought
and or but nor so yet for because if unless when whenever while whereas though although
than then thus hence therefore however moreover furthermore meanwhile
not no never none nobody nothing neither nor cannot
here there where when why how what which who whom whose
just also very really actually quite rather pretty fairly almost nearly only even still yet
again ever always usually often sometimes seldom rarely now soon later already
about above across after against along among around at before behind below beneath beside
between beyond by despite down during except for from in inside into near of off on onto
out outside over past per since through throughout till to toward towards under until up
upon via with within without
get gets got gotten make makes made doing done says say said says saying tell tells told
telling think thinks thought thinking know knows knew known knowing want wants wanted
wanting need needs needed seem seems seemed feel feels felt look looks looked
s ok okay yeah yep nope hey hi hello well oh ah um uh huh
dont didnt doesnt isnt wasnt arent werent wont wouldnt couldnt shouldnt cant cannot
im ive youre youve theyre theyve weve ill id thats theres whats hes shes lets aint
havent hasnt hadnt mustnt neednt shant
previous prior next last first second third
    """.split()

STOP = frozenset(STOP_V1 | set(_STOP_V2_EXTRA))

# 不规则动词/名词 → 原形（后缀规则拿不到，必须查表；确定性常量表）
IRREGULAR = {
    "said": "say", "says": "say", "told": "tell", "went": "go", "gone": "go", "bought": "buy",
    "made": "make", "took": "take", "taken": "take", "got": "get", "gotten": "get",
    "came": "come", "saw": "see", "seen": "see", "thought": "think", "felt": "feel",
    "found": "find", "gave": "give", "given": "give", "knew": "know", "known": "know",
    "left": "leave", "met": "meet", "ran": "run", "wrote": "write", "written": "write",
    "spoke": "speak", "spoken": "speak", "began": "begin", "begun": "begin",
    "became": "become", "brought": "bring", "paid": "pay", "sold": "sell", "spent": "spend",
    "taught": "teach", "understood": "understand", "won": "win", "wore": "wear",
    "worn": "wear", "ate": "eat", "eaten": "eat", "drank": "drink", "drove": "drive",
    "driven": "drive", "flew": "fly", "flown": "fly", "swam": "swim", "grew": "grow",
    "grown": "grow", "threw": "throw", "thrown": "throw", "caught": "catch",
    "fought": "fight", "built": "build", "sent": "send", "lost": "lose", "held": "hold",
    "kept": "keep", "meant": "mean", "sat": "sit", "stood": "stand", "slept": "sleep",
    "woke": "wake", "woken": "wake", "broke": "break", "broken": "break",
    "chose": "choose", "chosen": "choose", "fell": "fall", "fallen": "fall",
    "forgot": "forget", "forgotten": "forget", "hid": "hide", "hidden": "hide",
    "led": "lead", "rode": "ride", "risen": "rise", "rose": "rise", "sought": "seek",
    "shook": "shake", "shaken": "shake", "shot": "shoot", "shut": "shut", "sang": "sing",
    "sung": "sing", "sank": "sink", "stole": "steal", "stolen": "steal", "stuck": "stick",
    "struck": "strike", "swept": "sweep", "swore": "swear", "sworn": "swear",
    "tore": "tear", "torn": "tear", "withdrew": "withdraw", "drew": "draw",
    "drawn": "draw", "blew": "blow", "blown": "blow", "knees": "knee", "feet": "foot",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}

# 句首常见的"大写但其实是普通词/话语标记"——不作为硬单元
_SENT_START_COMMON = set(
    """yeah yes no so now then well oh ah ok okay hey hi hello maybe perhaps actually
sometimes recently currently today yesterday tomorrow tonight last next this that these
those there here what when where which who whom whose how why if after before during
once later finally however although because since while when my our your their his her its
one two three all some many much more most other another both each every few several
such same own very really just even still yet again always never often usually
i we they she he it you me us them him her
first second third next last previous prior instead meanwhile besides anyway anyways
researching working trying looking thinking going getting making doing being having
    """.split()
)

_VOWELS = frozenset("aeiou")


def _has_vowel(s: str) -> bool:
    return any(c in _VOWELS for c in s)


def _raw_words(text: str) -> list[str]:
    """小写化 + 去所有格（'s）与缩写撇号：'Joanna's'→'joanna'、'don't'→'dont'。"""
    out: list[str] = []
    for m in _WORD.finditer(text):
        w = m.group(0).lower()
        if w.endswith("'s"):
            w = w[:-2]
        else:
            w = w.replace("'", "")
        out.append(w)
    return out


def stem(word: str, version: str = STEM_VERSION) -> str:
    """确定性 Porter-lite 词干（只求两侧一致，不追求语言学正确）。

    规则顺序固定，是纯函数：同一输入永远同一输出。顺序：
      1) 查 IRREGULAR 表（不规则动词/名词）
      2) 复数 / 第三人称单数：sses→ss、ies→y、s→∅（ss/us/is 结尾不动）
      3) -ing / -ed：词干含元音才剥；running→run（双写辅音合并）；mak→make 不需要——
         因为第 5 步会统一去掉词尾 e，所以 "making"/"make" 都落到 "mak"
      4) 派生后缀：ly / ness / ment / tion / sion / est / er
      5) 词尾 e 去掉（长度 > 3）
    这样 "wants/want"、"said/says/say"、"transitioned/transition"、"created/create"
    两侧都会落到同一个键上。
    """
    if version != STEM_VERSION:
        raise ValueError(f"unknown stem version {version!r}")
    w = word.lower().replace("'", "")
    if len(w) <= 2:
        return w
    if w in IRREGULAR:
        w = IRREGULAR[w]
        # 不规则形也要过词尾 e（"give"/"gave" 都落到 "giv"）
        return w[:-1] if (len(w) >= 3 and w.endswith("e")) else w
    # 2) 复数 / 三单
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("es") and not w.endswith(("ses", "xes", "zes", "ches", "shes")) and _has_vowel(w[:-2]):
        w = w[:-1]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    # 3) 过去式 / 进行时
    for suf in ("ingly", "edly", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 2 and _has_vowel(w[: -len(suf)]):
            base = w[: -len(suf)]
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in _VOWELS:
                base = base[:-1]  # running → run
            w = base
            break
    # 4) 派生后缀
    for suf in ("fully", "ously", "ively", "ally", "ily"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + "y" if suf == "ily" else w[: -len(suf)]
            break
    else:
        for suf in ("ness", "ment", "tion", "sion", "est", "er", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
    # 5) 词尾 e（"make"/"making" 都到 "mak"）
    if len(w) >= 3 and w.endswith("e"):
        w = w[:-1]
    return w


def content_tokens(text: str, *, stop_mode: str = "v2", zh: bool = False) -> set[str]:
    """内容词集（小写、去停用词、最小长度 3）。不做词干化——词干由 `norm_tokens` 负责。

    `stop_mode="v1"` 完全复刻第五轮的 `content_tokens`（含旧 TOKEN_RE 的撇号处理差异），
    只用于消融对照，不用于执法默认路径。

    `zh=True` 时**并上**中文词法层：`zh_tokens(text) - ZH_STOP`（连续汉字段的字符 bigram，
    长度 1 取 unigram；ASCII 段小写原样）。`zh=False`（默认）走的是与 v4 之前**逐字节相同**
    的表达式，所以默认路径的 token 集合一个都没有变（回归门见 tests/test_zh.py）。
    """
    if stop_mode not in ("v1", "v2"):
        raise ValueError(f"unknown stop_mode {stop_mode!r}")
    if stop_mode == "v1":
        toks = {t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", text)}
        return (toks - STOP_V1) if not zh else ((toks - STOP_V1) | (set(zh_tokens(text)) - ZH_STOP))
    base = {w for w in _raw_words(text) if len(w) >= 3 and w not in STOP}
    if not zh:
        return base
    return base | (set(zh_tokens(text)) - ZH_STOP)


def norm_tokens(text: str, *, stem_ok: bool = True, stop_mode: str = "v2",
                zh: bool = False) -> set[str]:
    """接地的比较空间：content_tokens [+ 词干]。claim 与证据两侧必须用同一个函数。

    `zh=True` 时汉 token 是 1~2 字，`stem()` 对长度 ≤2 的词原样返回（纯汉 token 不做词干化），
    因此中英混排文本里英文照旧词干化、中文按 bigram 原样比较——两侧同函数，比较空间一致。
    """
    toks = content_tokens(text, stop_mode=stop_mode, zh=zh)
    if not stem_ok:
        return toks
    return {stem(t) for t in toks}


# --------------------------------------------------------------------------- #
# 2. IDF（由锚定库统计，纯计数、确定性、无模型）
# --------------------------------------------------------------------------- #


class IDFModel:
    """IDFModel"""

    __slots__ = ("version", "n_docs", "_df")

    def __init__(self, version: str, n_docs: int, df: Mapping[str, int]) -> None:
        self.version = version
        self.n_docs = max(1, int(n_docs))
        self._df = dict(df)

    def weight(self, token: str) -> float:
        df = self._df.get(token, 0)
        return math.log((self.n_docs + 1.0) / (df + 1.0)) + 1.0

    def mass(self, tokens: Iterable[str]) -> float:
        return sum(self.weight(t) for t in tokens)

    def to_json(self) -> dict:
        return {"version": self.version, "n_docs": self.n_docs, "df": dict(sorted(self._df.items()))}

    @classmethod
    def from_json(cls, obj: Mapping) -> "IDFModel":
        return cls(str(obj["version"]), int(obj["n_docs"]), obj["df"])

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return f"IDFModel(version={self.version!r}, n_docs={self.n_docs}, vocab={len(self._df)})"


class UniformIDF:
    """UniformIDF"""

    version = "uniform"
    n_docs = 1

    def weight(self, token: str) -> float:  # noqa: ARG002
        return 1.0

    def mass(self, tokens: Iterable[str]) -> float:
        return float(len(list(tokens)))


UNIFORM_IDF = UniformIDF()


def build_idf(token_sets: Iterable[set[str]], *, n_docs: Optional[int] = None,
              version: str = IDF_VERSION, zh: bool = False) -> IDFModel:
    """按文档频次建 IDF。`token_sets` 每条 = 一条锚定记录的 token 集。

    `zh` 只影响 **模型版本标签**（`idf-s2` → `idf-s2+zh`）：分词发生在上游
    （`norm_tokens(..., zh=zh)`），这里拿到的是已经切好的 token 集；标签的意义是
    「用中文词法层切出来的 IDF 不得与英文 IDF 混用/混读」，`to_json()` 里能自证来源。
    默认 `zh=False` → 版本号与 v4 之前逐字节一致。
    """
    df: dict[str, int] = {}
    n = 0
    for toks in token_sets:
        n += 1
        for t in toks:
            df[t] = df.get(t, 0) + 1
    return IDFModel(f"{version}+zh" if zh else version, n if n_docs is None else n_docs, df)


# internal-task：模块级 `_LEDGER_IDF` / `_LEDGER_ENTITIES` 缓存已移除，收敛到
# `memory_kernel.engine_cache.EngineCache`（显式、可注入、可 clear()、可重建）。
# 这里保留一个内部构建入口给 engine_cache 用（纯函数语义不变）。


def idf_for_ledger(ledger, *, stem_ok: bool = True, stop_mode: str = "v2",
                   zh: bool = False, registry=None, cache=None) -> IDFModel:
    """按需从账本算 IDF 并缓存（internal-task：缓存收敛到显式 `EngineCache`）。

    **缓存代际 = `len(ledger)`**：账本只追加、内容不可变，但每 append 一次 IDF 的 n/df
    都会变；旧实现只按 ledger 身份缓存，append 后仍返回旧模型（生产路径 IDF 滞后）。
    现在代数变化时整份缓存重建；同一代际内按 `(stem_ok, stop_mode, zh)` 分桶（中文词法层
    切出的 token 空间与英文空间不同，共用一份 IDF 会让中文 bigram 的 df 落进同一张表）。

    `registry`（internal-task，可选）：注入带 `snapshot()` 的 IDF 提供者
    （`memory_kernel.idf_dynamic.DynamicIDF`）时直接返回其当前生效快照的 model（版本化、
    可审计）。默认 `None` → 与旧路径**逐字节一致**（不 import 动态模块，默认零行为变化）。

    `cache`（internal-task，可选）：注入显式 `EngineCache`。默认 `None` → 使用进程级兼容默认
    缓存（`engine_cache.default_engine_cache()`，可 `clear()`/可替换），结果与历史逐字节一致；
    多实例/无状态场景请显式传各自 `EngineCache()`。
    """
    if registry is not None:
        snapshot = registry.snapshot()
        model = getattr(snapshot, "model", None)
        if model is None:
            raise TypeError("registry.snapshot() must expose a .model IDFModel")
        return model
    if cache is None:
        # 函数内 import：避免 skeleton ←→ engine_cache 的模块级循环 import；
        # 默认路径仍是"同一份进程级兼容缓存"，结果与历史实现逐字节一致。
        from .engine_cache import default_engine_cache

        cache = default_engine_cache()
    return cache.idf_for_ledger(ledger, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)


def idf_for_records(records: Sequence, *, stem_ok: bool = True, stop_mode: str = "v2",
                    zh: bool = False,
                    extra_docs: Optional[Iterable[str]] = None) -> IDFModel:
    """给一段显式记录集建 IDF（离线评测用；不用于执法路径）。"""
    def _docs():
        for rec in records:
            yield norm_tokens(rec.content, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
        for txt in extra_docs or ():
            yield norm_tokens(txt, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)

    return build_idf(_docs(), zh=zh)


def entities_for_ledger(ledger, cache=None) -> dict[str, frozenset]:
    """source_id -> 该 source 的参与者名字（小写）。

    claim 里出现**对话参与者**不算编造实体：说话人就在引用的元数据里（prompt 里也给了）。
    一次扫描 O(n)，按 `len(ledger)` 代际缓存；append 后代数变化会重建（旧缓存不得复用）。

    internal-task：缓存收敛到显式 `EngineCache`；`cache=None` → 进程级兼容默认缓存。
    """
    if cache is None:
        from .engine_cache import default_engine_cache

        cache = default_engine_cache()
    return cache.entities_for_ledger(ledger)


# --------------------------------------------------------------------------- #
# 3. 硬单元（必须 100% 接地）
# --------------------------------------------------------------------------- #

# 日期表达式：先整段遮掉，数字型硬单元不再从里面取（日期归 temporal 管，不双重判罚）
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_ALT = "|".join(_MONTH_NAMES)
_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
DATE_MD = re.compile(
    rf"\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH_ALT})\.?(?:\s*,?\s*\d{{4}})?\b"
    rf"|\b(?:{_MONTH_ALT})\.?\s*,?\s*\d{{4}}\b"
    rf"|\b(?:{_MONTH_ALT})\b"
    rf"|\b\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}\b",
    re.IGNORECASE,
)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
QUOTED_RE = re.compile(r"[\"“”「」]([^\"“”「」]{4,})[\"“”「」]|'([^']{4,})'")
YEAR_LIKE = re.compile(r"^(?:19|20)\d{2}$")

_SENT_END_CHARS = ".!?。！？:;,\n"


def _num_values(text: str) -> set[str]:
    return {_canon_num(m.group(0)) for m in NUM_RE.finditer(text)}


def _canon_num(raw: str) -> str:
    try:
        f = float(raw)
    except ValueError:  # pragma: no cover - 正则保证不会到这里
        return raw
    return str(int(f)) if f == int(f) else repr(round(f, 6))


def _is_sentence_initial(text: str, start: int) -> bool:
    j = start - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    return j < 0 or text[j] in _SENT_END_CHARS


@dataclass(frozen=True)
class HardUnit:
    kind: str   # "num" | "proper" | "quote"
    value: str
    initial: bool = False


def hard_units(text: str, *, include_initial: bool = True) -> tuple[HardUnit, ...]:
    """抽出 claim 里的硬单元：数字（日期/相对时间之外的）、专有名词、缩写词、引号内容。

    `include_initial=False` 时，**只出现在句首**的大写词不算硬单元（更保守，也更弱）。
    `_SENT_START_COMMON` 里的大写词（Yeah/Researching/…）任何位置都不算。

    刻意**不**把"4 位年份"和"日期表达式里的数字"当硬单元：它们归 temporal 管，
    否则"会话日期 − 1 天"这类正确推理会被判成数字未接地（数据里出现过这种假杀）。
    """
    out: list[HardUnit] = []
    seen: set[tuple[str, str]] = set()
    masked = DATE_MD.sub(lambda m: " " * len(m.group(0)), text)
    masked = REL_RE.sub(lambda m: " " * len(m.group(0)), masked)
    for m in NUM_RE.finditer(masked):
        v = _canon_num(m.group(0))
        if YEAR_LIKE.match(v):
            continue
        if ("num", v) in seen:
            continue
        seen.add(("num", v))
        out.append(HardUnit("num", v))
    initial_names: dict[str, bool] = {}
    for m in PROPER_RE.finditer(masked):
        raw = m.group(0)
        lw = raw.lower()
        if lw in _MONTH_NAMES or lw in _WEEKDAYS or lw in STOP or lw in _SENT_START_COMMON:
            continue
        init = _is_sentence_initial(masked, m.start())
        # 记录：这个词是否**只在**句首出现（出现一次非句首就视为普通专名）
        initial_names[lw] = initial_names.get(lw, True) and init
    for lw, only_initial in sorted(initial_names.items()):
        if only_initial and not include_initial:
            continue
        if ("proper", lw) in seen:
            continue
        seen.add(("proper", lw))
        out.append(HardUnit("proper", lw, initial=only_initial))
    for m in ACRONYM_RE.finditer(masked):
        lw = m.group(0).lower()
        if len(lw) < 2 or lw in STOP or lw == "i":
            continue
        if ("proper", lw) in seen:
            continue
        seen.add(("proper", lw))
        out.append(HardUnit("proper", lw))
    for m in QUOTED_RE.finditer(text):
        q = (m.group(1) or m.group(2) or "").strip()
        if len(q) < 4:
            continue
        key = ("quote", q.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(HardUnit("quote", q))
    return tuple(out)


def hard_unit_grounding(units: Sequence[HardUnit], *, pool_tokens: set[str],
                        pool_numbers: set[str], pool_text: str,
                        stem_ok: bool = True, stop_mode: str = "v2",
                        pool_words: Optional[set[str]] = None,
                        entities: Optional[set[str]] = None) -> tuple[int, list[str]]:
    """返回 (接地个数, 未接地描述)。硬单元必须 100% 接地才算通过。

    `pool_words` 是**不做最小长度过滤**的证据词集：专有名词/缩写常常只有 2 个字母
    （UK / VR / IBM），用软词那套 len>=3 的词表会把它们判成未接地（数据里出现过）。
    `entities` 是该 episode 的**参与者集合**：claim 里出现对话参与者（Jon/Nate/John）
    不算编造实体——那些名字就在引用元数据里。
    """
    if not units:
        return 0, []
    norm_pool_text = " ".join(pool_text.lower().split())
    words = pool_words if pool_words is not None else pool_tokens
    ents = entities or set()
    missing: list[str] = []
    grounded = 0
    for u in units:
        if u.kind == "num":
            ok = u.value in pool_numbers
        elif u.kind == "proper":
            key = stem(u.value) if stem_ok else u.value
            ok = key in words or u.value in ents
        else:  # quote
            q = " ".join(u.value.lower().split())
            ok = q in norm_pool_text
        if ok:
            grounded += 1
        else:
            missing.append(f"{u.kind}:{u.value}")
    return grounded, missing


# --------------------------------------------------------------------------- #
# 4. 时间论证（绝对 / 相对 → 区间）
# --------------------------------------------------------------------------- #

ABS_YMD = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")
_MD = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
_DM = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\.?(?:\s*,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
_MY = re.compile(rf"\b({_MONTH_ALT})\.?\s*,?\s*(\d{{4}})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_BARE_MONTH = re.compile(rf"\b({_MONTH_ALT})\b", re.IGNORECASE)

REL_RE = re.compile(
    r"\b(?:yesterday|today|tomorrow|tonight"
    r"|last\s+(?:night|week|month|year|summer|winter|spring|fall|autumn)"
    r"|this\s+(?:morning|afternoon|evening|week|month|year|spring|summer|fall|winter)"
    r"|next\s+(?:week|month|year|day)"
    r"|previous\s+(?:day|week|month|year)"
    r"|the\s+(?:day|week|month|year)\s+before"
    r"|(\d+)\s+(day|week|month|year)s?\s+ago)\b",
    re.IGNORECASE,
)

_REL_OFFSETS = {
    "yesterday": -1, "today": 0, "tomorrow": 1, "tonight": 0,
}
_MONTH_LEN = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


@dataclass(frozen=True)
class TSpan:
    """一个时间论证。

    kind:
      "point" —— 具体日期（lo == hi）
      "range" —— 区间（月/年/相对时间解析结果）
      "wild"  —— 缺年份：mo/d 里非 None 的分量必须一致
      "rel"   —— 未能解析的相对时间（没有参考日期），退化为字符串相等
    """

    kind: str
    lo: Optional[date] = None
    hi: Optional[date] = None
    mo: Optional[int] = None
    d: Optional[int] = None
    raw: str = ""

    def iso(self) -> str:
        if self.kind == "rel":
            return f"rel:{self.raw}"
        if self.kind == "wild":
            return f"wild:{self.mo}-{self.d}"
        if self.lo == self.hi:
            return self.lo.isoformat() if self.lo else "?"
        return f"{self.lo.isoformat() if self.lo else '?'}..{self.hi.isoformat() if self.hi else '?'}"


def _add_months(d: date, delta: int) -> date:
    y, m = d.year, d.month + delta
    while m <= 0:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    day = min(d.day, _MONTH_LEN.get(m, 30))
    if m == 2 and _is_leap(y):
        day = min(d.day, 29)
    return date(y, m, day)


def _is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def _month_range(y: int, m: int) -> tuple[date, date]:
    last = 29 if (m == 2 and _is_leap(y)) else _MONTH_LEN[m]
    return date(y, m, 1), date(y, m, last)


def _rel_span(raw: str, ref: Optional[date]) -> TSpan:
    low = raw.lower().strip()
    if ref is None:
        return TSpan("rel", raw=low)
    if low in _REL_OFFSETS:
        d = ref + timedelta(days=_REL_OFFSETS[low])
        return TSpan("point", lo=d, hi=d, raw=low)
    if low.startswith("last night"):
        d = ref - timedelta(days=1)
        return TSpan("point", lo=d, hi=d, raw=low)
    m = re.fullmatch(r"(\d+)\s+(day|week|month|year)s?\s+ago", low)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "day":
            lo = hi = ref - timedelta(days=n)
        elif unit == "week":
            lo, hi = ref - timedelta(days=7 * n + 6), ref - timedelta(days=7 * n)
        elif unit == "month":
            hi = _add_months(ref, -n)
            lo = _month_range(hi.year, hi.month)[0]
        else:
            lo, hi = date(ref.year - n, 1, 1), date(ref.year - n, 12, 31)
        return TSpan("range", lo=lo, hi=hi, raw=low)
    if low.startswith("last week") or low.startswith("this week"):
        start = ref - timedelta(days=ref.weekday())
        if low.startswith("last"):
            start -= timedelta(days=7)
        return TSpan("range", lo=start, hi=start + timedelta(days=6), raw=low)
    if low.startswith("next week"):
        start = ref - timedelta(days=ref.weekday()) + timedelta(days=7)
        return TSpan("range", lo=start, hi=start + timedelta(days=6), raw=low)
    for word, delta in (("last month", -1), ("this month", 0), ("next month", 1)):
        if low.startswith(word):
            lo, hi = _month_range(*_ym(_add_months(ref.replace(day=1), delta)))
            return TSpan("range", lo=lo, hi=hi, raw=low)
    for word, delta in (("last year", -1), ("this year", 0), ("next year", 1)):
        if low.startswith(word):
            y = ref.year + delta
            return TSpan("range", lo=date(y, 1, 1), hi=date(y, 12, 31), raw=low)
    if low.startswith("previous day") or low.startswith("the day before"):
        d = ref - timedelta(days=1)
        return TSpan("point", lo=d, hi=d, raw=low)
    if low.startswith("previous week") or low.startswith("the week before"):
        start = ref - timedelta(days=ref.weekday()) - timedelta(days=7)
        return TSpan("range", lo=start, hi=start + timedelta(days=6), raw=low)
    if low.startswith("previous month") or low.startswith("the month before"):
        lo, hi = _month_range(*_ym(_add_months(ref.replace(day=1), -1)))
        return TSpan("range", lo=lo, hi=hi, raw=low)
    if low.startswith("previous year") or low.startswith("the year before"):
        y = ref.year - 1
        return TSpan("range", lo=date(y, 1, 1), hi=date(y, 12, 31), raw=low)
    if low.startswith("this morning") or low.startswith("this afternoon") or low.startswith("this evening"):
        return TSpan("point", lo=ref, hi=ref, raw=low)
    if low.startswith("next day"):
        d = ref + timedelta(days=1)
        return TSpan("point", lo=d, hi=d, raw=low)
    return TSpan("rel", raw=low)


def _ym(d: date) -> tuple[int, int]:
    return d.year, d.month


def spans_of(text: str, ref: Optional[date] = None) -> list[TSpan]:
    """抽出一条文本里的全部时间论证（绝对优先；重叠的裸月份不重复计入）。"""
    out: list[TSpan] = []
    consumed: list[tuple[int, int]] = []

    def _free(start: int, end: int) -> bool:
        return not any(not (end <= s or start >= e) for s, e in consumed)

    for m in ABS_YMD.finditer(text):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        out.append(TSpan("point", lo=d, hi=d, raw=m.group(0)))
        consumed.append((m.start(), m.end()))
    for m in _MD.finditer(text):
        if not _free(m.start(), m.end()):
            continue
        mo, dd = _MONTH_NAMES[m.group(1).lower()], int(m.group(2))
        y = int(m.group(3)) if m.group(3) else None
        if y:
            try:
                d = date(y, mo, dd)
            except ValueError:
                continue
            out.append(TSpan("point", lo=d, hi=d, raw=m.group(0)))
        else:
            out.append(TSpan("wild", mo=mo, d=dd, raw=m.group(0)))
        consumed.append((m.start(), m.end()))
    for m in _DM.finditer(text):
        if not _free(m.start(), m.end()):
            continue
        mo, dd = _MONTH_NAMES[m.group(2).lower()], int(m.group(1))
        y = int(m.group(3)) if m.group(3) else None
        if y:
            try:
                d = date(y, mo, dd)
            except ValueError:
                continue
            out.append(TSpan("point", lo=d, hi=d, raw=m.group(0)))
        else:
            out.append(TSpan("wild", mo=mo, d=dd, raw=m.group(0)))
        consumed.append((m.start(), m.end()))
    for m in _MY.finditer(text):
        if not _free(m.start(), m.end()):
            continue
        lo, hi = _month_range(int(m.group(2)), _MONTH_NAMES[m.group(1).lower()])
        out.append(TSpan("range", lo=lo, hi=hi, raw=m.group(0)))
        consumed.append((m.start(), m.end()))
    for m in _YEAR.finditer(text):
        if not _free(m.start(), m.end()):
            continue
        y = int(m.group(0))
        out.append(TSpan("range", lo=date(y, 1, 1), hi=date(y, 12, 31), raw=m.group(0)))
        consumed.append((m.start(), m.end()))
    for m in _BARE_MONTH.finditer(text):
        if not _free(m.start(), m.end()):
            continue
        lw = m.group(1).lower()
        # "may" 是情态动词；裸月份只在首字母大写或前面有介词时算时间论证
        if lw in ("may", "march") and not (
            text[m.start(): m.start() + 1].isupper() or re.search(r"\bin\s+$", text[: m.start()], re.I)
        ):
            continue
        out.append(TSpan("wild", mo=_MONTH_NAMES[lw], raw=m.group(0)))
        consumed.append((m.start(), m.end()))
    for m in REL_RE.finditer(text):
        out.append(_rel_span(m.group(0), ref))
    return out


def _span_compatible(a: TSpan, b: TSpan, tol_days: int = 0) -> bool:
    if a.kind == "rel" or b.kind == "rel":
        return a.kind == "rel" and b.kind == "rel" and a.raw == b.raw
    if a.kind == "wild" and b.kind == "wild":
        if a.mo is not None and b.mo is not None and a.mo != b.mo:
            return False
        if a.d is not None and b.d is not None and a.d != b.d:
            return False
        return True
    if a.kind == "wild" or b.kind == "wild":
        w, r = (a, b) if a.kind == "wild" else (b, a)
        if r.lo is None or r.hi is None:
            return False
        return _wild_in_range(w, r.lo, r.hi)
    # 两边都是区间/点混合
    if a.lo is None or a.hi is None or b.lo is None or b.hi is None:
        return False
    return not (a.hi + timedelta(days=tol_days) < b.lo or b.hi + timedelta(days=tol_days) < a.lo)


def _wild_in_range(w: TSpan, lo: date, hi: date) -> bool:
    if (hi - lo).days > 800:  # 区间过长（>2 年）时按逐日做一次扫描上限保护
        hi = lo + timedelta(days=800)
    d = lo
    step = timedelta(days=1)
    while d <= hi:
        if (w.mo is None or d.month == w.mo) and (w.d is None or d.day == w.d):
            return True
        d += step
    return False


def _parse_session_date(raw: str) -> Optional[date]:
    """解析会话日期。支持 LoCoMo `1:56 pm on 8 May, 2023` 与 LME `2023/04/10 (Mon) 17:50`。"""
    if not raw:
        return None
    s = str(raw).strip()
    m = ABS_YMD.search(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _MD.search(s)
    if m:
        y = m.group(3)
        if y:
            try:
                return date(int(y), _MONTH_NAMES[m.group(1).lower()], int(m.group(2)))
            except ValueError:
                return None
    m = _DM.search(s)
    if m and m.group(3):
        try:
            return date(int(m.group(3)), _MONTH_NAMES[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None
    return None


def _session_dates_for(records: Sequence, session_dates: Optional[Mapping]) -> list[date]:
    """把 (source_id, episode_id) 的会话日期表翻译成这批记录的去重日期列表。"""
    if not session_dates:
        return []
    out: list[date] = []
    seen: set[date] = set()
    for rec in records:
        key = (getattr(rec, "source_id", None), getattr(rec, "episode_id", None))
        raw = session_dates.get(key)
        if raw is None:
            continue
        d = _parse_session_date(raw)
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return sorted(out)


# --------------------------------------------------------------------------- #
# 5. 论元结构对齐 + 多引用分派（v3）
# --------------------------------------------------------------------------- #
#
# v2 的度量是**词袋**：claim 与全部引用并成一个词池比覆盖率。它看不见三类结构事实：
#   1) claim 由**两个事件**拼成，只引了其中一个（N5a / N_B）；
#   2) claim 的**参与者**（"Both Jon and Gina" 里的 Gina）根本没出现在所引证据里（N5b）；
#   3) 谓词在**另一条引用**里恰好出现过，但那条引用讲的不是这件事（N3 的诱饵引用）。
# v3 把 claim 抽成论元单元 (主体, 谓词, 客体)，逐槽接地，并把每个事件**分派**到单条引用上。
# 全部是确定性的浅层规则：词表 + 词干 + 位置启发式，不调用模型、不做句法分析、无 I/O。
# --------------------------------------------------------------------------- #

ARGUMENT_VERSION = "arg-s3"

# 口语缩写/连写 → 原形（"wanna start" 与 "want to start" 必须同键）
_ARG_CLITIC = {
    "wanna": "want", "gonna": "go", "gotta": "get", "wanta": "want", "hafta": "have",
    "tryna": "try", "lemme": "let", "gimme": "give", "sorta": "sort", "outta": "out",
}


def arg_lemma(word: str) -> str:
    """谓词归一：缩写展开 → `stem`。claim 侧与证据侧必须用同一个函数。"""
    lw = word.lower().replace("'", "")
    return stem(_ARG_CLITIC.get(lw, lw))


# 可作谓词的动词原形（确定性常量表；只用于"这个词能不能算谓词"的判断，不做语义分类）
_ARG_VERB_SURFACE = frozenset(
    """accept add adopt agree allow answer arrive ask attend avoid bake beat begin believe
    belong break bring build buy call camp care carry catch change chat check choose come compete
    complete consider continue cook create cut dance decide describe die discover donate draw dream
    drink drive eat encourage end enjoy explain explore fall feel fight figure find finish fly focus
    follow forget forgive get give grow guess happen hate head hear help hold hope hurt imagine
    improve include join keep kick know land laugh launch learn leave like listen live look lose
    love make manage mean meet mention miss move need note notice offer open paint pass pay pick plan
    play prefer prepare provide publish push put quit raise reach read realize receive recommend refer
    relax remember remind rent reply report rescue research rest return ride ring rise run save say
    see seem sell send serve set share shop show sing sit sleep smile speak spend stand start stay
    stop study suggest support swim take talk teach tell tend think throw train transition travel
    treat try turn understand use visit volunteer wait walk want watch wear win wonder work worry
    write""".split()
)
_ARG_VERB_LEMMAS = frozenset(arg_lemma(v) for v in _ARG_VERB_SURFACE)

# 助动词：不承载断言内容，定位主动词时跳过
_ARG_AUX = frozenset(
    """be am is are was were been being have has had having do does did doing done
    will would shall should can could may might must ought let lets""".split()
)

# 私态谓词（want/hope/think/like…）：证据是**第一人称**记录，第三称 claim 的私态无法由引用
# 直接证实，因此不作"必须接地"的槽（否则会把"she wanted to…"整类忠实 claim 判死）。
_ARG_STANCE = frozenset(arg_lemma(v) for v in
                        """want hope wish plan try decide need love like enjoy prefer think
                        believe feel know seem tend care mind expect guess mean remember forget
                        wonder understand hate dream imagine miss worry smile laugh consider
                        suppose""".split())

# 言语行为谓词（said/told/asked/described…）：claim 在转述证据，本身不必出现在证据里
_ARG_ATTRIBUTION = frozenset(arg_lemma(v) for v in
                             """say tell ask mention explain add reply answer describe note
                             report recommend refer share talk speak chat encourage suggest
                             invite remind agree promise admit claim""".split())

# 轻动词：只提供时态/体，不提供可核实的论元（take a trip / have a look / go to）
_ARG_LIGHT = frozenset(arg_lemma(v) for v in
                       """be have do get make go let seem tend keep become turn put come take
                       give use happen occur""".split())

# 已观测到的同义谓词（对称组）。只在两组词**都出自本语料**的释义关系上手工建表；
# 不含反义/方向相反的对（buy≠sell），避免把 N3 谓词错配一起放行。
_ARG_ALTERNATION_GROUPS = (
    ("attend", "go"),
    ("start", "begin", "launch", "open"),
    ("buy", "purchase", "get"),
    ("own", "have"),
    ("love", "like", "enjoy", "include"),
    ("join", "enter"),
    ("die", "pass"),
    ("study", "research"),
    ("show", "display"),
    ("help", "support", "volunteer"),
    ("finish", "complete"),
    ("give", "provide"),
    ("find", "discover"),
    ("create", "make", "build"),
    ("travel", "go", "visit"),
    ("meet", "know"),
    ("win", "beat"),
    ("write", "author"),
    ("live", "stay"),
    ("look", "search", "seek"),
    ("move", "relocate"),
    ("adopt", "get", "have"),
    ("compete", "play"),
)

_ARG_ALTERNATION: dict[str, frozenset] = {}
for _group in _ARG_ALTERNATION_GROUPS:
    _lemmas = frozenset(arg_lemma(w) for w in _group)
    for _lem in _lemmas:
        _ARG_ALTERNATION[_lem] = _ARG_ALTERNATION.get(_lem, frozenset()) | _lemmas

# 派生后缀：谓词接地允许同一词族的形态差异（help / helpful / helping）
_ARG_DERIVATIONAL = ("fully", "lessly", "able", "ible", "ful", "less", "ive", "ous",
                     "ness", "ment", "tion", "sion", "ing", "ed", "ly", "al")


def arg_pred_family(lemma: str) -> str:
    """谓词的词族键：剥一个派生后缀（两侧同键才算同族）。"""
    for suf in _ARG_DERIVATIONAL:
        if lemma.endswith(suf) and len(lemma) - len(suf) >= 3:
            return lemma[: -len(suf)]
    return lemma


_ARG_PRONOUNS = frozenset("i we he she they you it".split())
_ARG_POSSESSIVES = frozenset("my our his her their your its".split())
_ARG_SUBJECT_STARTERS = _ARG_PRONOUNS | _ARG_POSSESSIVES
_ARG_PREPOSITIONS = frozenset(
    """of in at on with to from by for about as like toward towards into over under after
    before during through between against around across since until within without""".split()
)
# 专名槽里要排除的"大写但不是名字"的词（月份/星期/停用词/句首话语标记）
_ARG_EXCLUDED_NAMES = frozenset(_MONTH_NAMES) | frozenset(_WEEKDAYS) | STOP | frozenset(_SENT_START_COMMON)

# --------------------------------------------------------------------------- #
# 5b. v3.1 修复用的常量表（**v3.0 路径不读这里任何一张表**）
#
# report7 §6/§9.8 归出的三类误杀：
#   (1) 没有事件谓词的短断言（Hope you're doing fine）——客体区退化；
#   (2) 非事件谓词（find / think / attended an event / sounds …）被当成必须接地的谓词；
#   (3) 限定/不定 NP 客体（the check-up / a small circle of dancers）被当成可核实客体。
# 对应四张谓词表 + 一个位置护栏（见 _arg_in_noun_position）+ 一张壳名词表。
# --------------------------------------------------------------------------- #

ARGUMENT_VERSION_V31 = "arg-s31"

# 私态谓词扩展：report7 的 P_lab "Dave finds classic cars charming" 就是 find 被当成事件谓词。
# 归一用 arg_lemma，claim/证据两侧同键。
_V31_STANCE_EXTRA = frozenset(arg_lemma(w) for w in
                              """find wish appreciate adore dislike fear doubt suspect assume
                              notice recognize trust admire respect regret envy""".split())

# 系表谓词：be 类已在 _ARG_LIGHT，这里补齐"系动词 + 形容词"的常见成员。
# 它们的补语是形容词/小句，不是可核实客体，谓词本身也不承载事件内容。
_V31_COPULAR = frozenset(arg_lemma(w) for w in
                         """be am is are was were been being seem sound look appear feel
                         remain taste smell prove become turn stay""".split())

# 情态：本身不承载可核实内容（多数已在 _ARG_AUX 里被跳过；need/dare/ought 作主动词时在此补齐）。
_V31_MODAL = frozenset(arg_lemma(w) for w in
                       """can could may might must shall should will would ought need dare""".split())

# 言语行为谓词扩展：claim 在转述/致意，谓词不必出现在证据里（补语仍然要接地）。
_V31_ATTRIB_EXTRA = frozenset(arg_lemma(w) for w in
                              """complain announce warn insist shout whisper beg thank greet
                              clarify confirm deny argue brag joke tease scold praise""".split())

# v3.1 的非事件谓词全集（v3.0 用的是 stance ∪ attribution ∪ light 三张表）
_ARG_NONEVENT_BASE = _ARG_STANCE | _ARG_ATTRIBUTION | _ARG_LIGHT
_ARG_NONEVENT_V31 = (_ARG_NONEVENT_BASE | _V31_STANCE_EXTRA | _V31_COPULAR
                     | _V31_MODAL | _V31_ATTRIB_EXTRA)

# 名词位置护栏：限定词/所有格/量词之后不可能是限定动词（your work / the ideal dance）。
_ARG_DETERMINERS = frozenset(
    """the a an this that these those my your his her their our its some any every each
    another no both all many much few several""".split())
_ARG_PREP_GUARD = _ARG_PREPOSITIONS - {"to"}      # "to" + 动词原形是不定式，不能当名词

# 框架/轻动词 + 壳名词（attended an event with …）：谓词只提供框架，可核实内容在修饰语里，
# 因此谓词槽不成立（客体槽照旧要接地）。只收"事件框架"名词；time/moment/thing/stuff 这类
# 太泛的词会让 "make time to dance" 之类的真事件也失去事件核，因此**故意不收录**。
_ARG_SHELL_NOUNS = frozenset(
    """event events occasion occasions gathering gatherings""".split())

_ARG_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s+[—–]\s+")
_ARG_COORD_SPLIT = re.compile(r"\s*(?:,\s*)?\b(?:and|but|so|because|while|which|when)\b\s+", re.I)
# 第一人称复数证据："we/us/our"、"and I"、"& I" —— 它可以为**并列主体**背书
_ARG_PLURAL_EVIDENCE = re.compile(
    r"\b(?:we|us|our|ours)\b|\band i\b|\bi and\b|[&]\s*i\b|\bi\s*[&]", re.I)
# 并列主体："both X and Y" / "X and Y" / "X and his|her|their|my|our …"
_ARG_COORD_SUBJECT = re.compile(
    r"\bboth\b|[A-Z][a-zA-Z]+ and [A-Z][a-zA-Z]+|[A-Z][a-zA-Z]+ and (?:his|her|their|my|our) ")


@dataclass(frozen=True)
class ArgumentUnit:
    """claim 的一个论元单元（大致 = 一个子句 / 一个事件）。"""

    segment: str                    # 该单元的原文片段
    subject_names: tuple = ()       # 主体位置的专名（"Both Jon and Gina" → (Jon, Gina)）
    subject_pronouns: tuple = ()    # 主体位置的代词（含所有格）
    predicates: tuple = ()          # 全部动词 lemma（含助动词/言语行为动词）
    main_predicate: Optional[str] = None   # 第一个**事件性**谓词（stance/attribution/light 之后）
    object_tokens: tuple = ()       # 客体位置的内容词
    object_names: tuple = ()        # 客体位置的专名（言语行为动词的受话人除外）
    coordinated: bool = False       # 主体是否为并列（both X and Y / X and Y）
    time_spans: tuple = ()          # 时间论元（复用 `spans_of`；接地仍由 temporal 规则负责）
    # ---- v3.1 字段（v3.0 路径永远保持默认值，因此 v3.0 的单元逐位不变）----
    predicate_optional: bool = False   # 谓词槽不成立（壳名词框架：attended an event with …）
    object_np_head: str = ""           # 限定/不定 NP 客体的中心名词（v3.1 NP 规则用）

    def describe(self) -> str:
        extra = ", pred_opt=1" if self.predicate_optional else ""
        return (f"unit(pred={self.main_predicate}, subj={list(self.subject_names)}, "
                f"obj={list(self.object_tokens[:3])}, coord={int(self.coordinated)}{extra})")


def _arg_tokens(segment: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(segment)]


def _arg_is_verb(word: str) -> bool:
    """是不是动词候选。用**词面**判断，避免词干碰撞把名词当动词（car→care、job→?）。

    接受三种词面：不规则动词表里的形式、动词原形表里的形式、带屈折后缀（-ed/-ing/-s）
    且词干落在动词表里的形式。
    """
    lw = word.lower().replace("'", "")
    if lw in _ARG_AUX:
        return False
    if lw in STOP and lw not in IRREGULAR and lw not in _ARG_CLITIC:
        return False
    if lw in IRREGULAR or lw in _ARG_CLITIC:
        return arg_lemma(lw) in _ARG_VERB_LEMMAS
    if lw in _ARG_VERB_SURFACE:
        return True
    if lw.endswith(("ed", "ing", "s")):
        return arg_lemma(lw) in _ARG_VERB_LEMMAS
    return False


def _arg_is_content(word: str) -> bool:
    """内容词（可以当名词/形容词）：不是停用词/介词/代词/所有格/助动词/动词候选。"""
    lw = word.lower().replace("'", "")
    if not lw or lw in STOP or lw in _ARG_PREPOSITIONS or lw in _ARG_PRONOUNS:
        return False
    if lw in _ARG_POSSESSIVES or lw in _ARG_AUX or lw in _ARG_EXCLUDED_NAMES:
        return False
    return not _arg_is_verb(word)


def _arg_in_noun_position(toks: list, i: int) -> bool:
    """v3.1：第 i 个 token 是否处在**名词位置**——因此不可能是限定动词。

    两条判据（任一成立即判名词，确定性、只看相邻 token）：
      C1 紧邻前词是限定词/所有格/介词（"your work" / "of dancers" / "on show"）；"to" 除外；
      C2 前两步之内是限定词/量词、中间夹一个修饰词（形容词/名词修饰语），且候选词**后面紧跟
         一个内容词**（"the ideal dance studio" 里的 dance）。C2 的"后接内容词"是必要的护栏：
         否则 "This made me happy" 会因为 "This" 是限定词而把 made 判成名词。
    """
    if i <= 0:
        return False
    prev = toks[i - 1][2].lower().replace("'", "")
    if prev in _ARG_DETERMINERS or prev in _ARG_PREP_GUARD:
        return True
    if i >= 2:
        prev2 = toks[i - 2][2].lower().replace("'", "")
        nxt = toks[i + 1][2].lower().replace("'", "") if i + 1 < len(toks) else ""
        if (prev2 in _ARG_DETERMINERS and _arg_is_content(toks[i - 1][2])
                and _arg_is_content(nxt)):
            return True
    return False


def _arg_np_head(toks: list, verb_pos: Optional[int], obj_tokens: list) -> str:
    """v3.1：客体区里**直接宾语 NP** 的中心名词（诊断用；不参与判定）。

    判据：谓词之后第一个词是限定词/所有格 ⇒ 客体区是 NP；中心名词 = 宾语 NP 里第一个介词
    之前的最后一个内容词（"a small circle of dancers" → circle，"the check-up with …" → check）。
    """
    if verb_pos is None or not obj_tokens:
        return ""
    head = ""
    for _s, _e, w in toks[verb_pos + 1:]:
        lw = w.lower().replace("'", "")
        if lw in _ARG_PREPOSITIONS:
            break
        if (lw in STOP or lw in _ARG_PRONOUNS or lw in _ARG_POSSESSIVES
                or lw in _ARG_DETERMINERS or len(lw) < 3 or lw in _ARG_EXCLUDED_NAMES):
            continue
        head = lw
    return head



def _arg_clause_spans(text: str) -> list[tuple[int, int]]:
    """切子句：句末标点 / 破折号；再按协调连词切，但**左侧必须已有动词**
    （否则 "Andrew and his girlfriend …" 会在主语内部被切开）。"""
    spans, pos = [], 0
    for m in _ARG_SENT_SPLIT.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for a, b in spans:
        seg = text[a:b]
        if not seg.strip():
            continue
        last = 0
        for m in _ARG_COORD_SPLIT.finditer(seg):
            nxt = re.match(r"[A-Za-z']+", seg[m.end():])
            word = (nxt.group(0) if nxt else "").lower()
            left = seg[last:m.start()]
            if word in _ARG_SUBJECT_STARTERS and any(
                    _arg_is_verb(w) for _, _, w in _arg_tokens(left)):
                out.append((a + last, a + m.start()))
                last = m.end()
        out.append((a + last, a + b))
    return [(x, y) for x, y in out if x < y and text[x:y].strip()]


def claim_arguments(text: str, *, v31_nonevent: bool = False,
                    v31_noun_guard: bool = False, v31_np_object: bool = False) -> tuple:
    """把 claim 抽成论元单元序列（确定性浅层规则，纯函数）。

    v3.1 三个开关（默认全关 → 与 v3.0 逐位一致）：
      * `v31_nonevent` —— 非事件谓词族扩展（情态/系表/私态/言语行为）+ 壳名词框架谓词；
      * `v31_noun_guard` —— 名词位置护栏（限定词/所有格/介词之后的候选词判名词，不当谓词）；
      * `v31_np_object` —— 抽出限定/不定 NP 客体的中心名词（供客体槽规则用）。

    规则（全部可解释、可单测）：
      * 子句切分见 `_arg_clause_spans`；
      * 第一个动词之前的词是**主体区**，其中的专名（排除月份/星期/停用词/句首话语标记）
        是主体专名；代词/所有格是主体代词；
      * 谓词 = 子句内全部动词 lemma；**事件性谓词** = 去掉助动词/私态/言语行为/轻动词后的第一个，
        作为"必须接地"的主谓词；
      * 第一个动词之后的词是**客体区**，去掉介词/停用词/代词/短词后是客体词；
        其中紧跟非言语行为动词的专名是客体专名（"told Jon" 的 Jon 是受话人，不算客体论元）；
      * 并列主体只在前 5 个词里识别，避免把 "dogs are named Pepper, Precious, and Panda"
        里的并列客体误判成并列主体。
    """
    units: list = []
    for a, b in _arg_clause_spans(text):
        seg = text[a:b]
        toks = _arg_tokens(seg)
        if not toks:
            continue
        verb_pos = None
        verbs: list[str] = []
        for i, (s, e, w) in enumerate(toks):
            if seg[e:e + 1] == "-":          # 连字符复合词是名词（check-up / eye-opening）
                continue
            if not _arg_is_verb(w):
                continue
            if v31_noun_guard and _arg_in_noun_position(toks, i):
                continue                     # v3.1：名词位置上的词不是谓词
            if verb_pos is None:
                verb_pos = i
            lem = arg_lemma(w)
            if lem not in verbs:
                verbs.append(lem)
        subj_words = [w for _, _, w in (toks[:verb_pos] if verb_pos is not None else toks)]
        names: list[str] = []
        for w in subj_words:
            lw = w.lower().replace("'", "")
            if len(w) >= 3 and w[:1].isupper() and lw not in _ARG_EXCLUDED_NAMES and w not in names:
                names.append(w)
        pronouns: list[str] = []
        for w in subj_words:
            lw = w.lower().split("'")[0]
            if (lw in _ARG_PRONOUNS or lw in _ARG_POSSESSIVES) and lw not in pronouns:
                pronouns.append(lw)
        nonevent = _ARG_NONEVENT_V31 if v31_nonevent else _ARG_NONEVENT_BASE
        eventive = [v for v in verbs if v not in nonevent]
        main_pred = eventive[0] if eventive else None
        obj_tokens: list[str] = []
        obj_tokens_full: list[str] = []
        obj_names: list[str] = []
        if verb_pos is not None:
            recipient_ctx = bool(verbs) and verbs[0] in _ARG_ATTRIBUTION
            in_object_np = True
            for s, e, w in toks[verb_pos + 1:]:
                lw = w.lower().replace("'", "")
                if v31_np_object and lw in _ARG_PREPOSITIONS:
                    # v3.1：介词之后的成分是**附加语**（with my doctor / after work），
                    # 不是直接宾语——客体槽只看直接宾语 NP；附加语由逐单元覆盖率负责。
                    in_object_np = False
                if (lw in _ARG_PREPOSITIONS or lw in STOP or lw in _ARG_PRONOUNS
                        or lw in _ARG_POSSESSIVES):
                    continue
                if lw in _ARG_AUX or lw in _ARG_EXCLUDED_NAMES or len(lw) < 3:
                    continue
                if _arg_is_verb(w) and arg_lemma(lw) in _ARG_LIGHT:
                    continue
                obj_tokens_full.append(w)
                if in_object_np:
                    obj_tokens.append(w)
                if not recipient_ctx and len(w) >= 3 and w[:1].isupper() \
                        and lw not in _ARG_EXCLUDED_NAMES:
                    obj_names.append(w)
        coordinated = bool(_ARG_COORD_SUBJECT.search(" ".join(subj_words[:5])))
        predicate_optional = False
        object_np_head = ""
        if v31_nonevent and main_pred is not None and obj_tokens and \
                obj_tokens[0].lower().replace("'", "") in _ARG_SHELL_NOUNS:
            # 壳名词框架（attended an event with many classic cars）：
            # 谓词不承载可核实内容，**可核实内容在修饰语里**，所以客体区要保留附加语
            # （与逐单元覆盖率的分工一致：这里的客体是"事件的内容"，不是直接宾语）。
            predicate_optional = True
            obj_tokens = obj_tokens_full
        if v31_np_object and obj_tokens and not obj_names:
            object_np_head = _arg_np_head(toks, verb_pos, obj_tokens)
        units.append(ArgumentUnit(
            segment=seg,
            subject_names=tuple(names),
            subject_pronouns=tuple(pronouns),
            predicates=tuple(verbs),
            main_predicate=main_pred,
            object_tokens=tuple(obj_tokens),
            object_names=tuple(obj_names),
            coordinated=coordinated,
            time_spans=tuple(spans_of(seg)),
            predicate_optional=predicate_optional,
            object_np_head=object_np_head,
        ))
    return tuple(units)


@dataclass(frozen=True)
class _ArgPool:
    """单条引用（或全部引用的并集）的接地面：原词 / 词干 / 谓词 lemma / 说话人。"""

    raw: frozenset
    tokens: frozenset
    lemmas: frozenset
    families: frozenset
    speakers: frozenset
    plural_evidence: bool

    def name_ok(self, name: str) -> bool:
        lw = name.lower().replace("'s", "").replace("'", "")
        return lw in self.raw or lw in self.speakers or stem(lw) in self.tokens

    def predicate_ok(self, pred: Optional[str]) -> bool:
        if pred is None:
            return True
        cands = {pred} | set(_ARG_ALTERNATION.get(pred, frozenset()))
        if cands & self.lemmas:
            return True
        return bool({arg_pred_family(c) for c in cands} & self.families)

    def object_ok(self, tokens) -> bool:
        if not tokens:
            return True
        return any(stem(t.lower()) in self.tokens or t.lower() in self.raw for t in tokens)


def _arg_pool(records: Sequence, *, stem_ok: bool, stop_mode: str,
              zh: bool = False) -> _ArgPool:
    raw: set[str] = set()
    tokens: set[str] = set()
    lemmas: set[str] = set()
    speakers: set[str] = set()
    plural = False
    for rec in records:
        content = rec.content
        if _ARG_PLURAL_EVIDENCE.search(content):
            plural = True
        speaker = getattr(rec, "speaker", None)
        if speaker:
            speakers.update(_raw_words(speaker))
        for part in (content, speaker or ""):
            raw.update(_raw_words(part))
            for w in _raw_words(part):
                if w not in STOP:
                    lemmas.add(arg_lemma(w))
        tokens |= norm_tokens(content, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
        if speaker:
            tokens |= norm_tokens(speaker, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
    return _ArgPool(
        raw=frozenset(raw),
        tokens=frozenset(tokens),
        lemmas=frozenset(lemmas),
        families=frozenset(arg_pred_family(x) for x in lemmas),
        speakers=frozenset(speakers),
        plural_evidence=plural,
    )


def _arg_coverage(segment: str, pool_tokens: set, idf, *, stem_ok: bool,
                  stop_mode: str, zh: bool = False) -> float:
    toks = norm_tokens(segment, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
    total = idf.mass(toks)
    if total <= 0:
        return 1.0
    return idf.mass(toks & pool_tokens) / total


@dataclass(frozen=True)
class ArgumentGrounding:
    """v3 的论元接地结论。`rule` 取第一个违反的槽（规则名进 `SkeletonResult.rule`）。"""

    ok: bool = True
    rule: str = ""
    detail: str = ""
    n_units: int = 0
    unit_coverage_min: float = 1.0
    participants: tuple = ()           # 全部主体专名（去重保序）
    missing: tuple = ()                 # 未接地论元（槽:值）
    dispatch_unassigned: tuple = ()     # 事件核无法分派到单条引用的单元
    plural_uncovered: bool = False      # 并列主体断言了 >=2 参与者但只有一条引用

    def features(self) -> dict:
        return {
            "arg_ok": self.ok,
            "arg_rule": self.rule,
            "plural_uncovered": self.plural_uncovered,
            "n_units": self.n_units,
            "unit_cov_min": round(self.unit_coverage_min, 6),
            "n_participants": len(self.participants),
            "dispatch_unassigned": len(self.dispatch_unassigned),
        }


def argument_grounding(
    units: Sequence[ArgumentUnit],
    records: Sequence,
    *,
    idf,
    tau: float,
    stem_ok: bool = True,
    stop_mode: str = "v2",
    arg_slots: bool = True,
    dispatch: bool = True,
    v31_no_event_object: bool = False,
    v31_np_object: bool = False,
    zh: bool = False,
) -> ArgumentGrounding:
    """论元槽接地 + 多引用分派（确定性，纯函数）。

    两个开关分别对应 v3 的两条规则族：
      `arg_slots` —— 逐槽接地：主体专名必须出现在某条引用里（或就是它的说话人）、
          事件性谓词必须接地、客体专名必须接地、**每个单元**的 IDF 覆盖率 ≥ tau；
      `dispatch` —— 分派：并列主体（k≥2 参与者）需要 ≥2 条引用（除非证据是第一人称复数），
          且每个事件核必须能被**单条**引用覆盖（谓词 + 主体名或客体词同句），
          不能靠"谓词在 A、客体在 B"的两条引用拼出来。

    v3.1 修复开关（默认全关 → 与 v3.0 逐位一致）：
      `v31_no_event_object` —— 整个 claim 没有事件性谓词时，客体区不是可核实客体槽
          （"Hope you're doing fine" 的 fine / "a small circle of dancers" 的退化客体区），
          主体槽与逐单元覆盖率仍然照旧；
      `v31_np_object` —— 限定/不定 NP 客体的接地只看**中心名词**（专名仍走逐名严格路径）。
    """
    if not units:
        return ArgumentGrounding(ok=True, n_units=0)

    pool = _arg_pool(records, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
    rec_pools = [_arg_pool([r], stem_ok=stem_ok, stop_mode=stop_mode, zh=zh) for r in records]

    # v3.1：claim 里是否存在**事件核**（所有单元都没有事件性谓词 ⇒ 客体区退化）
    no_event_core = bool(v31_no_event_object
                         and not any(u.main_predicate is not None for u in units))

    missing_subject: list[str] = []
    missing_predicate: list[str] = []
    missing_object: list[str] = []
    low_coverage: list[str] = []
    unassigned: list[str] = []
    participants: list[str] = []
    coverage_min = 1.0

    for u in units:
        for n in u.subject_names:
            if n not in participants:
                participants.append(n)
        if arg_slots:
            for name in u.subject_names:
                # 并列主体 + 第一人称复数证据（"we volunteered together"）时，
                # 非说话人参与者允许由"我们"背书；否则每个名字都必须落在引用里。
                if not pool.name_ok(name) and not (u.coordinated and pool.plural_evidence):
                    missing_subject.append(name)
            if not pool.predicate_ok(u.main_predicate) and not u.predicate_optional:
                missing_predicate.append(u.main_predicate or "")
            if u.object_names:
                for name in u.object_names:
                    if not pool.name_ok(name):
                        missing_object.append(name)
            elif u.object_tokens and not no_event_core and not pool.object_ok(u.object_tokens):
                missing_object.append(u.object_tokens[0])
            if pool.tokens:
                cov = _arg_coverage(u.segment, set(pool.tokens), idf,
                                    stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
            else:
                cov = 0.0
            coverage_min = min(coverage_min, cov)
            if cov < tau:
                low_coverage.append(f"{cov:.3f}:{u.segment[:48]}")
        if dispatch and u.main_predicate is not None and not u.predicate_optional:
            attached = False
            for rp in rec_pools:
                if not rp.predicate_ok(u.main_predicate):
                    continue
                if u.subject_names and any(rp.name_ok(n) for n in u.subject_names):
                    attached = True
                    break
                if u.object_tokens and rp.object_ok(u.object_tokens):
                    attached = True
                    break
                if not u.subject_names and not u.object_tokens:
                    attached = True
                    break
            if not attached:
                unassigned.append(f"{u.main_predicate}:{u.segment[:48]}")

    plural_uncovered = bool(
        dispatch and len(records) < 2 and not pool.plural_evidence
        and any(u.coordinated for u in units))

    missing: list[str] = []
    if missing_subject:
        missing.extend(f"subject:{x}" for x in missing_subject[:4])
    if missing_predicate:
        missing.extend(f"predicate:{x}" for x in missing_predicate[:4])
    if missing_object:
        missing.extend(f"object:{x}" for x in missing_object[:4])

    # 违反顺序固定（报告里每个数字都能对应到第一个违反的槽）
    if missing_subject:
        return ArgumentGrounding(
            ok=False, rule="arg_subject",
            detail=f"subject argument(s) not grounded in any citation: {missing_subject[:3]}",
            n_units=len(units), unit_coverage_min=coverage_min,
            participants=tuple(participants), missing=tuple(missing),
            dispatch_unassigned=tuple(unassigned[:4]),
            plural_uncovered=plural_uncovered)
    if missing_predicate:
        return ArgumentGrounding(
            ok=False, rule="arg_predicate",
            detail=f"predicate not grounded in any citation: {missing_predicate[:3]}",
            n_units=len(units), unit_coverage_min=coverage_min,
            participants=tuple(participants), missing=tuple(missing),
            dispatch_unassigned=tuple(unassigned[:4]),
            plural_uncovered=plural_uncovered)
    if missing_object:
        return ArgumentGrounding(
            ok=False, rule="arg_object",
            detail=f"object argument(s) not grounded in any citation: {missing_object[:3]}",
            n_units=len(units), unit_coverage_min=coverage_min,
            participants=tuple(participants), missing=tuple(missing),
            dispatch_unassigned=tuple(unassigned[:4]),
            plural_uncovered=plural_uncovered)
    if low_coverage:
        return ArgumentGrounding(
            ok=False, rule="arg_unit_coverage",
            detail=f"per-unit grounding below tau: {low_coverage[:2]}",
            n_units=len(units), unit_coverage_min=coverage_min,
            participants=tuple(participants), missing=tuple(missing),
            dispatch_unassigned=tuple(unassigned[:4]),
            plural_uncovered=plural_uncovered)
    if dispatch:
        if plural_uncovered:
            plural_units = [u for u in units if u.coordinated]
            return ArgumentGrounding(
                ok=False, rule="dispatch_plural",
                detail=(f"coordinated subject {list(plural_units[0].subject_names) or 'both'} "
                        f"asserts >=2 participants but only {len(records)} citation(s)"),
                n_units=len(units), unit_coverage_min=coverage_min,
                participants=tuple(participants), missing=tuple(missing),
                plural_uncovered=True)
        if unassigned:
            return ArgumentGrounding(
                ok=False, rule="dispatch_event",
                detail=f"event core(s) not assignable to a single citation: {unassigned[:2]}",
                n_units=len(units), unit_coverage_min=coverage_min,
                participants=tuple(participants), missing=tuple(missing),
                dispatch_unassigned=tuple(unassigned[:4]))
    return ArgumentGrounding(
        ok=True, n_units=len(units), unit_coverage_min=coverage_min,
        participants=tuple(participants), missing=tuple(missing),
        dispatch_unassigned=tuple(unassigned[:4]),
        plural_uncovered=plural_uncovered)


# --------------------------------------------------------------------------- #
# 6. 策略
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SkeletonPolicy:
    """版本化的 Tier 1 策略。

    字段说明见模块 docstring。`tau` 是 v1 的旧字段名，保留为**兼容别名**
    （构造时 `tau=` 会覆盖 `tau_soft`，构造后两者永远相等），旧调用点无需改动。
    """

    version: str = POLICY_VERSION
    tau_soft: float = 0.8
    tau: Optional[float] = None
    enabled: bool = True
    failure_mode: str = "degraded"  # "degraded" | "rejected"
    # ---- v2 组件开关（网格维度）----
    hard_units: bool = True
    hard_units_initial: bool = True
    idf: bool = True
    stem: bool = True
    stop_mode: str = "v2"
    temporal: bool = True
    temporal_session: bool = True
    temporal_tolerance_days: int = 0
    use_speaker: bool = True
    # ---- v3 组件开关（默认关闭 → 默认策略与 v2 逐位一致）----
    arg_slots: bool = False        # 论元槽对齐（主体/谓词/客体/逐单元覆盖率）
    dispatch: bool = False         # 多引用分派（并列主体需要多条引用；事件核要能落到单条引用）
    # ---- v3.1 修复开关（默认关闭 → `SkeletonPolicy()` 是 v2.0、`SkeletonPolicy.v3()` 是 v3.0）----
    v31_nonevent: bool = False          # 非事件谓词族扩展（情态/系表/私态/言语行为）+ 壳名词框架
    v31_noun_guard: bool = False        # 名词位置护栏（限定词/所有格/介词之后不是谓词）
    v31_no_event_object: bool = False   # 没有事件性谓词 ⇒ 客体槽不成立（客体区退化）
    v31_np_object: bool = False         # 客体槽只看直接宾语 NP（介词后的附加语不算客体）
    # ---- v4 能力层开关（默认关闭 → `SkeletonPolicy()` 仍是 v2.0，v3/v3.1 及以下逐字节不变）----
    zh: bool = False               # 中文词法层：分词（bigram）/ 中文否定 / 中文时间归一
    cite_set: bool = False         # 引用集校验（`claim.supporting` 逐条核 anchor/quote/相关性）
    scenario: bool = False         # 场景谓词（时间方向/跨会话/说话人范围/已被更正）
    # ---- v4.7-meta 元数据层开关（默认关闭 → 全部既有版本逐字节不变）----
    # `quantifier`：中英文封闭量词表参与 `no_generalization` 判定（claim 有量词、证据无量词 → soft）。
    # `source_type`：新增 `source_attribution` soft 谓词（user_fact 断言引 assistant/tool/document → 降级）。
    # 两者都只能产生 degraded，**不得产生 verified 资格**（SPEC §3.2 元数据层）。
    quantifier: bool = False
    source_type: bool = False

    def __post_init__(self) -> None:
        t = self.tau if self.tau is not None else self.tau_soft
        object.__setattr__(self, "tau_soft", float(t))
        object.__setattr__(self, "tau", float(t))
        if self.stop_mode not in ("v1", "v2"):
            raise ValueError(f"unknown stop_mode {self.stop_mode!r}")

    @property
    def grounding_enabled(self) -> bool:
        """τ_soft=0 表示**关闭整个接地检查**（含硬单元与 v3 论元规则）——与 v1 的 tau=0 语义一致。"""
        return self.enabled and self.tau_soft > 0.0

    @property
    def v3_enabled(self) -> bool:
        """是否启用任一 v3 规则（与 τ 无关：报告里用它区分"策略是 v3"和"接地是否执行"）。"""
        return bool(self.enabled and (self.arg_slots or self.dispatch))

    @property
    def v31_enabled(self) -> bool:
        """是否启用任一 v3.1 修复（与 τ、与 arg_slots/dispatch 无关）。"""
        return bool(self.enabled and (self.v31_nonevent or self.v31_noun_guard
                                      or self.v31_no_event_object or self.v31_np_object))

    @property
    def v4_enabled(self) -> bool:
        """是否启用任一 v4 能力层开关（与 τ、与 v3/v3.1 开关无关）。"""
        return bool(self.enabled and (self.zh or self.cite_set or self.scenario))

    @property
    def meta_enabled(self) -> bool:
        """是否启用任一 v4.7-meta 元数据谓词（F1 Phase 1；默认两开关都关）。"""
        return bool(self.enabled and (self.quantifier or self.source_type))

    @classmethod
    def v3(cls, **overrides) -> "SkeletonPolicy":
        """构造 v3 策略：默认打开两条 v3 规则并把版本号写成 `skel-v3.0`。

        显式传 `arg_slots=False` / `dispatch=False` 可做单侧消融（版本号仍标 v3，
        因为"只用一条 v3 规则"也是 v3 行为，报告里必须能区分）。
        """
        opts = dict(arg_slots=True, dispatch=True, version=POLICY_VERSION_V3)
        opts.update(overrides)
        return cls(**opts)

    @classmethod
    def v3_1(cls, **overrides) -> "SkeletonPolicy":
        """构造 v3.1 策略：v3.0 + 四个修复开关全开，版本号 `skel-v3.1`。

        四个全开是实验配置，已被数据否决（report7b §9）；生产/正式使用 v3_c3()。

        每个 `v31_*` 都可以单独关掉做逐规则消融（关掉后该条行为退回 v3.0）。
        """
        opts = dict(arg_slots=True, dispatch=True, version=POLICY_VERSION_V31,
                    v31_nonevent=True, v31_noun_guard=True,
                    v31_no_event_object=True, v31_np_object=True)
        opts.update(overrides)
        return cls(**opts)

    @classmethod
    def v3_c3(cls, **overrides) -> "SkeletonPolicy":
        """构造 v3.1-C3 策略：**用户裁决的 C2 子集**，版本号 `skel-v3.1-C3`。

        默认值 = 实测的 C3 生产工作点（`eval/out/results7b.json:selected_c3_candidate`）：
        `arg_slots / dispatch / hard_units / temporal` 全开、`tau_soft=0.45`、三条修复开关
        `v31_nonevent / v31_noun_guard / v31_no_event_object` 打开。

        **`v31_np_object` 默认关闭，且不是可调项**：该项被数据否决（单开把 P_lab 误杀
        7/162→16/162、P1 5/84→9/84、新增 24 条正类误杀，唯一"修好"的一条已由 `noun_guard`
        覆盖；见 report7b §5/§9），不是可调参数。四个全开的 `v3_1()` 只是实验配置。

        这是 C3 的**显式 opt-in 入口**：`SkeletonPolicy()` 仍是 v2.0、`.v3()` 仍是 v3.0，
        默认路径不会因为本方法而改变。

        代价必须一起读（report7b §5/§9）：7/7 误杀修好，同点（τ=0.35）少抓 24 条负类
        （N_C1 11、N3/N3e 各 1…）；τ=0.45 上与 v3.0 打平（P_lab 4.3%、TPR_hard 83.3%、
        TPR_partial 94.7%、cases7 97.4%、C_ctrl 1.8%）。
        """
        opts = dict(arg_slots=True, dispatch=True, hard_units=True, temporal=True,
                    tau_soft=0.45, version=POLICY_VERSION_V31_C3,
                    v31_nonevent=True, v31_noun_guard=True,
                    v31_no_event_object=True, v31_np_object=False)
        opts.update(overrides)
        return cls(**opts)

    @classmethod
    def v4(cls) -> "SkeletonPolicy":
        """构造 v4 策略（契约 `docs/ALGO-V4-CONTRACT.md` §1，字面一致）。

        `τ_soft=0.45` + 硬单元 + 时间 + **中文词法层 / 引用集校验 / 场景谓词** 全开，
        再加 `arg_slots / dispatch`，版本号 `skel-v4.0`。

        这是 v4 能力层的**唯一正式入口**：`SkeletonPolicy()` 仍是 v2.0、`.v3()` 仍是 v3.0、
        `.v3_c3()` 仍是 v3.1-C3，默认路径与 v4 之前的所有版本都逐字节不变。

        读数字时必须一起读的代价：`zh=True` 只把"中文文本从 0 token 变成可比较的 bigram"，
        它不是分词器（词内/跨词 bigram 一视同仁，噪声靠 1~2 字停用词表压）；`cite_set` /
        `scenario` 的谓词严重度与单侧性见 `cite_set.py` / `scenario.py` 的模块 docstring。
        """
        return cls(version="skel-v4.0", tau_soft=0.45, hard_units=True, temporal=True,
                   zh=True, cite_set=True, scenario=True, arg_slots=True, dispatch=True)

    @classmethod
    def v4_meta(cls) -> "SkeletonPolicy":
        """构造 v4.7-meta 策略（F1 Phase 1，`docs/F1-PHASE1.md`）。

        = `v4()` 全部开关 + 两条**元数据 soft 谓词**：`quantifier=True`、`source_type=True`，
        版本号 `skel-v4.7-meta`。这是元数据层的**唯一正式入口**；`SkeletonPolicy()` / `.v4()`
        的判定与冻结产物逐字节不变。

        口径（必须一起读）：
        * 两条谓词都 **soft-only**：只产生 `degraded`，不产生 `verified` 资格、不改 hard 拒绝语义；
        * `quantifier` 只做"claim 出现量词而证据没有"的**单侧**判据（证据也有量词就不判，宁可漏）；
        * `source_type` 只扩到"非 hedge 的 assistant / tool / document"；assistant+hedge 仍由
          `scenario_speaker_scope`(hard) 负责，同一缺陷不重复计罚；
        * **中文条件句仍不覆盖**（`conditions → scope` 属 Phase 2，见 ALGO-CEILING §4）。
        """
        return cls(version=POLICY_VERSION_V4_META, tau_soft=0.45, hard_units=True,
                   temporal=True, zh=True, cite_set=True, scenario=True,
                   arg_slots=True, dispatch=True, quantifier=True, source_type=True)

    def describe(self) -> str:
        base = (f"{self.version}(tau_soft={self.tau_soft:.2f}, hard={int(self.hard_units)}, "
                f"idf={int(self.idf)}, stem={int(self.stem)}, stop={self.stop_mode}, "
                f"temporal={int(self.temporal)}, session={int(self.temporal_session)}")
        if self.arg_slots or self.dispatch:
            base += f", arg={int(self.arg_slots)}, dispatch={int(self.dispatch)}"
        if self.v31_enabled:
            base += (f", v31(nonevent={int(self.v31_nonevent)}, "
                     f"noun_guard={int(self.v31_noun_guard)}, "
                     f"no_event_obj={int(self.v31_no_event_object)}, "
                     f"np_obj={int(self.v31_np_object)})")
        if self.v4_enabled:
            base += (f", v4(zh={int(self.zh)}, cite_set={int(self.cite_set)}, "
                     f"scenario={int(self.scenario)})")
        if self.meta_enabled:
            base += (f", meta(quantifier={int(self.quantifier)}, "
                     f"source_type={int(self.source_type)})")
        return base + ")"


@dataclass(frozen=True)
class SkeletonResult:
    ok: bool
    reason: Optional[str]
    grounding: float
    detail: str = ""
    # v2 诊断字段（默认值保证旧调用点不受影响）
    version: str = POLICY_VERSION
    rule: str = ""                 # "" | "soft_grounding" | "hard_unit" | "temporal" | v3 规则名
    soft: float = 0.0              # IDF 加权软词覆盖率
    hard_total: int = 0
    hard_grounded: int = 0
    hard_missing: tuple = ()
    temporal_ok: bool = True
    n_soft_tokens: int = 0
    # v3 诊断字段（默认值保证旧调用点不受影响；arg_slots/dispatch 关闭时保持默认）
    arg_ok: bool = True
    n_units: int = 0
    unit_cov_min: float = 1.0
    participants: tuple = ()
    arg_missing: tuple = ()
    dispatch_unassigned: tuple = ()
    plural_uncovered: bool = False

    @property
    def severity(self) -> str:
        return "soft"

    def features(self) -> dict:
        return {
            "ok": self.ok,
            "rule": self.rule,
            "soft": round(self.soft, 6),
            "hard_total": self.hard_total,
            "hard_grounded": self.hard_grounded,
            "hard_cov": (self.hard_grounded / self.hard_total) if self.hard_total else 1.0,
            "temporal_ok": self.temporal_ok,
            "n_soft_tokens": self.n_soft_tokens,
            "arg_ok": self.arg_ok,
            "n_units": self.n_units,
            "unit_cov_min": round(self.unit_cov_min, 6),
            "n_participants": len(self.participants),
            "dispatch_unassigned": len(self.dispatch_unassigned),
            "plural_uncovered": self.plural_uncovered,
        }


# 兼容第五轮脚本的旧 API（`run_eval5` 用它做诊断）
def claim_times(text: str) -> list[tuple]:
    """旧格式时间列表：('abs', y|None, m|None) / ('rel', word)。保留给诊断脚本。"""
    out: list[tuple] = []
    for sp in spans_of(text):
        if sp.kind == "point" and sp.lo:
            out.append(("abs", sp.lo.year, sp.lo.month))
        elif sp.kind == "range" and sp.lo:
            out.append(("abs", sp.lo.year, sp.lo.month))
        elif sp.kind == "wild":
            out.append(("abs", None, sp.mo))
        elif sp.kind == "rel":
            out.append(("rel", sp.raw))
    return out


def evidence_times(texts: Sequence[str], ref_dates: Optional[Sequence[date]] = None) -> list[tuple]:
    """旧格式证据时间列表。`ref_dates` 是可选参考日期（会话日期）。"""
    out: list[tuple] = []
    for t in texts:
        out.extend(claim_times(t))
    for d in ref_dates or ():
        out.append(("abs", d.year, d.month))
    return out


# --------------------------------------------------------------------------- #
# 6. 主判定
# --------------------------------------------------------------------------- #


def skeleton_check(
    claim_text: str,
    records: Sequence,
    policy: Optional[SkeletonPolicy] = None,
    *,
    session_dates: Optional[Mapping] = None,
    idf_model: Optional[IDFModel] = None,
    entities: Optional[set[str]] = None,
) -> SkeletonResult:
    """Tier 1：claim 的硬单元必须 100% 接地、软词 IDF 覆盖率 ≥ τ_soft、时间论证兼容。

    v3（`policy.arg_slots` / `policy.dispatch`）额外做论元槽对齐与多引用分派，
    违反顺序固定为：hard_unit → arg_subject → arg_predicate → arg_object →
    arg_unit_coverage → dispatch_plural → dispatch_event → soft_grounding → temporal。

    参数
    ----
    records        : 被引用记录（主引用 + supporting），需有 .content / .speaker /
                     .source_id / .episode_id
    session_dates  : {(source_id, episode_id): 会话日期字符串}；给了才启用会话日期接地
    idf_model      : 由锚定库统计的 IDF；不给则退化为等权（v1 行为）
    entities       : 该 episode 参与者的小写名字集合（硬单元里的"人名"用它接地）

    `policy.zh=True` 时，claim 与证据两侧的 token 都走中文词法层（`zh=True`），
    IDF 也必须是用同一个 `zh` 建出来的（`idf_for_ledger(..., zh=True)`），否则两侧
    比较空间不一致。`policy.zh=False`（默认）时本函数行为与 v4 之前逐字节相同。
    """
    pol = policy or SkeletonPolicy()
    if not pol.enabled:
        return SkeletonResult(ok=True, reason=None, grounding=1.0, detail="disabled", version=pol.version)

    idf: object = (idf_model if (pol.idf and idf_model is not None) else UNIFORM_IDF)
    stem_ok = pol.stem

    claim_toks = norm_tokens(claim_text, stem_ok=stem_ok, stop_mode=pol.stop_mode, zh=pol.zh)
    pool_toks: set[str] = set()
    pool_words: set[str] = set()
    pool_text_parts: list[str] = []
    for rec in records:
        pool_toks |= norm_tokens(rec.content, stem_ok=stem_ok, stop_mode=pol.stop_mode, zh=pol.zh)
        pool_text_parts.append(rec.content)
        if pol.use_speaker and getattr(rec, "speaker", None):
            pool_toks |= norm_tokens(rec.speaker, stem_ok=stem_ok, stop_mode=pol.stop_mode,
                                     zh=pol.zh)
            pool_text_parts.append(rec.speaker)
    # 硬单元用**不做最小长度过滤**的词集（UK / VR / IBM 这类 2 字母缩写）
    for part in pool_text_parts:
        for w in _raw_words(part):
            if w in STOP:
                continue
            pool_words.add(stem(w) if stem_ok else w)

    total_mass = idf.mass(claim_toks)
    grounded_mass = idf.mass(claim_toks & pool_toks)
    soft = (grounded_mass / total_mass) if total_mass > 0 else 1.0

    hard_units_list = hard_units(claim_text, include_initial=pol.hard_units_initial) if pol.hard_units else ()
    if pol.hard_units:
        pool_numbers = _num_values(" ".join(pool_text_parts))
        for d in _session_dates_for(records, session_dates):
            pool_numbers.add(_canon_num(str(d.year)))
            pool_numbers.add(_canon_num(str(d.day)))
        hard_grounded, hard_missing = hard_unit_grounding(
            hard_units_list,
            pool_tokens=pool_toks,
            pool_numbers=pool_numbers,
            pool_text=" ".join(pool_text_parts),
            stem_ok=stem_ok,
            stop_mode=pol.stop_mode,
            pool_words=pool_words,
            entities=entities,
        )
    else:
        hard_grounded, hard_missing = 0, []

    # --- 时间论证 ---
    refs = _session_dates_for(records, session_dates) if pol.temporal_session else []
    temporal_ok = True
    temporal_detail = ""
    if pol.temporal:
        claim_spans = spans_of(claim_text)
        if claim_spans:
            ev_spans: list[TSpan] = []
            for rec in records:
                rec_ref = _parse_session_date(
                    (session_dates or {}).get(
                        (getattr(rec, "source_id", None), getattr(rec, "episode_id", None)), ""
                    )
                ) if pol.temporal_session else None
                ev_spans.extend(spans_of(rec.content, ref=rec_ref))
            if pol.temporal_session:
                for d in refs:
                    ev_spans.append(TSpan("point", lo=d, hi=d, raw="session_date"))
            if not ev_spans:
                temporal_ok = False
                temporal_detail = "claim asserts time but evidence has none"
            else:
                missing = []
                for cs in claim_spans:
                    candidates = [cs]
                    if cs.kind == "rel" and refs:
                        candidates = [_rel_span(cs.raw, r) for r in refs] or [cs]
                    if not any(
                        _span_compatible(c, e, pol.temporal_tolerance_days)
                        for c in candidates for e in ev_spans
                    ):
                        missing.append(cs.iso())
                if missing:
                    temporal_ok = False
                    temporal_detail = f"ungrounded temporal arguments: {missing[:2]}"

    # --- v3 论元结构（论元槽对齐 + 多引用分派；默认关闭 → 不做任何额外计算）---
    # τ_soft=0 时整条接地检查关闭（含 v3），与 v1/v2 的 tau=0 语义一致。
    arg_rep: Optional[ArgumentGrounding] = None
    if pol.grounding_enabled and (pol.arg_slots or pol.dispatch):
        arg_rep = argument_grounding(
            claim_arguments(claim_text, v31_nonevent=pol.v31_nonevent,
                            v31_noun_guard=pol.v31_noun_guard,
                            v31_np_object=pol.v31_np_object),
            records,
            idf=idf, tau=pol.tau_soft, stem_ok=stem_ok, stop_mode=pol.stop_mode,
            arg_slots=pol.arg_slots, dispatch=pol.dispatch,
            v31_no_event_object=pol.v31_no_event_object,
            v31_np_object=pol.v31_np_object,
            zh=pol.zh,
        )

    base = dict(
        version=pol.version,
        soft=soft,
        hard_total=len(hard_units_list),
        hard_grounded=hard_grounded,
        hard_missing=tuple(hard_missing[:4]),
        temporal_ok=temporal_ok,
        n_soft_tokens=len(claim_toks),
        arg_ok=(arg_rep.ok if arg_rep is not None else True),
        n_units=(arg_rep.n_units if arg_rep is not None else 0),
        unit_cov_min=(arg_rep.unit_coverage_min if arg_rep is not None else 1.0),
        participants=(arg_rep.participants if arg_rep is not None else ()),
        arg_missing=(arg_rep.missing if arg_rep is not None else ()),
        dispatch_unassigned=(arg_rep.dispatch_unassigned if arg_rep is not None else ()),
        plural_uncovered=(arg_rep.plural_uncovered if arg_rep is not None else False),
    )

    if pol.grounding_enabled:
        if pol.hard_units and hard_missing:
            return SkeletonResult(
                ok=False, reason="grounding", grounding=soft,
                detail=f"hard units ungrounded: {hard_missing[:3]}",
                rule="hard_unit", **base,
            )
        if arg_rep is not None and not arg_rep.ok:
            return SkeletonResult(
                ok=False, reason="grounding", grounding=soft,
                detail=arg_rep.detail, rule=arg_rep.rule, **base,
            )
        if soft < pol.tau_soft:
            return SkeletonResult(
                ok=False, reason="grounding", grounding=soft,
                detail=f"idf-weighted soft grounding {soft:.3f} < tau {pol.tau_soft:.2f}",
                rule="soft_grounding", **base,
            )

    if not temporal_ok:
        return SkeletonResult(
            ok=False, reason="temporal", grounding=soft, detail=temporal_detail,
            rule="temporal", **base,
        )

    return SkeletonResult(ok=True, reason=None, grounding=soft, **base)


__all__ = [
    "POLICY_VERSION",
    "POLICY_VERSION_V3",
    "POLICY_VERSION_V31",
    "POLICY_VERSION_V31_C3",
    "POLICY_VERSION_V4",
    "POLICY_VERSION_V4_META",
    "ARGUMENT_VERSION",
    "ARGUMENT_VERSION_V31",
    "STOP",
    "STOP_V1",
    "IDFModel",
    "UniformIDF",
    "UNIFORM_IDF",
    "HardUnit",
    "ArgumentUnit",
    "ArgumentGrounding",
    "TSpan",
    "SkeletonPolicy",
    "SkeletonResult",
    "arg_lemma",
    "arg_pred_family",
    "argument_grounding",
    "build_idf",
    "claim_arguments",
    "claim_times",
    "content_tokens",
    "evidence_times",
    "hard_unit_grounding",
    "hard_units",
    "idf_for_ledger",
    "idf_for_records",
    "norm_tokens",
    "skeleton_check",
    "spans_of",
    "stem",
]

