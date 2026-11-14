# SPDX-License-Identifier: Apache-2.0
r"""中文词法层（v4 能力层；契约 `docs/ALGO-V4-CONTRACT.md` §2）。

全部是**纯函数 + 封闭常量表**：只用 stdlib、不调用模型、不联网、无 I/O、无时间/随机数，
同一输入永远同一输出。本模块只提供"中文文本的确定性可比较量"，判定语义由调用方
（`skeleton.py` / `obligations.py` / `cite_set.py` / `scenario.py`）决定。

设计要点（为什么这样做，而不是别的做法）
────────────────────────────────────────────────────────────────────────────
1. **分词用字符 bigram，不用词典/分词器/模型**
   连续汉字段切字符 bigram（长度 1 的段取 unigram），ASCII 字母数字段小写原样。
   没有分词器就没有歧义，也就没有不可复现的行为。代价必须一起读：词内 bigram
   （「杭州」）与跨词 bigram（「去了」）同等对待，噪声只能靠停用词表按 1~2 字压。
   这不是"精确分词"，它是"确定性可比较的近似接地单位"。

2. **`ZH_STOP` 只收虚词，绝不收否定词**
   否定由 `zh_negated` 单独处理。否定词一旦混进停用词表，否定就会在分词阶段被**悄悄
   抹掉**——那正是"中文否定 0/7"这类缺口的一个来源。测试里用两道断言锁死：
   表内每一项都不得含否定标记字符，且 `zh_negated(entry)` 必须为 False。

3. **否定检测 = 标记表 + 有限豁免（方向安全的假阳性）**
   单字标记（没/未/无/别/非）会命中「未来/无论/特别」这类**词内成分**，因此对最常见的
   几类非否定同形词加**封闭豁免表**（`_BEI_EXCLUDE_*` / `_NEG_EXEMPT`）。未收录的同形词
   会产生假否定；方向的后果是：scope 侧假否定 → 要求 quote 也否定 → 更严（假杀），
   quote 侧假否定 → 可能放过（假接受）。两者都只在 `zh=True` 路径上，默认路径零影响。

4. **时间归一化按"精度可包含"设计**
   `zh_temporal_terms` 给**最小规范串**（`2026-03-05` / `2026-03` / `2026` / `03-05` /
   `上周三` / `昨天` / `09:00` / `3天前` …）；`zh_temporal_subset` 回答"quote 的时间项
   能否被 scope 的时间项覆盖"——**证据更精确 OK，断言更精确不 OK**。这样
   「claim 说 3月5日、证据给 2026年3月5日」不会被假杀，而「证据只有 3月5日、claim 说
   2026年3月5日」仍然失败（年份未被证据接地）。

已知限制（必须如实读）
────────────────────────────────────────────────────────────────────────────
* 只用阿拉伯数字：`三天前` / `二〇二六年` 这类中文数字不归一（`\d+天前` 契约口径）。
* 时间归一化不建日历：`今天`/`昨天` 只做**字符串级**规范（不换成绝对日期），跨"相对→绝对"
  的推理仍由 `skeleton.py` 的英文 `spans_of` + 会话日期负责。
* 停用词表只收 1~2 字项：`zh_tokens` 只产出 1~2 字汉 token，3 字以上的停用词永远不会命中，
  因此故意不收录（测试里断言所有项长度 ∈ {1,2}）。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Iterable

__all__ = [
    "ZH_STOP",
    "ZH_NEGATION_MARKERS",
    "ZH_CORRECTED",
    "ZH_ANAPHOR",
    "ZH_HEDGE",
    "has_cjk",
    "cjk_ratio",
    "zh_tokens",
    "zh_negated",
    "zh_temporal_terms",
    "zh_temporal_subset",
    "zh_corrected",
    "zh_anaphor",
    "zh_hedge",
]

# --------------------------------------------------------------------------- #
# 1. CJK 判定
# --------------------------------------------------------------------------- #
#
# 只认**汉字**（Han）区段。CJK 标点（U+3000..U+303F）、假名、全角字母都不算 CJK：
# 它们既不承载内容，也不承载否定/时间，算进来只会让 cjk_ratio 偏高、让"是否中文"判断变糊。
# Ext-B 以上（U+20000+）在 Python 里是单码点字符，ord() 直接可用。

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs（基本区）
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2EBEF),  # Extension C/D/E/F
    (0x30000, 0x323AF),  # Extension G/H
)


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def has_cjk(text: str) -> bool:
    """文本是否含至少一个汉字。空串 / 纯 ASCII / 纯标点 → False。"""
    return any(_is_cjk_char(c) for c in text)


def cjk_ratio(text: str) -> float:
    """汉字数 / 非空白字符数。空串或全空白 → 0.0；分母**不含空白**，标点算分母。"""
    total = 0
    cjk = 0
    for c in text:
        if c.isspace():
            continue
        total += 1
        if _is_cjk_char(c):
            cjk += 1
    return 0.0 if total == 0 else cjk / total


# --------------------------------------------------------------------------- #
# 2. 确定性切分：连续汉字段 → 字符 bigram（长度 1 取 unigram）；ASCII 段小写原样
# --------------------------------------------------------------------------- #

_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]+")


def zh_tokens(text: str) -> list[str]:
    """确定性切分：返回**原文出现顺序**的 token 列表（允许重复，不去停用词）。

    规则（逐条可单测）：
      * 连续汉字段 → 逐字符 bigram（`我上周去了杭州` → 我上/上周/周去/去了/了杭/杭州）；
        段长度 1 → unigram（`5 个` 的 `个`）；
      * 连续 ASCII 字母数字段 → 整段小写原样（`PostgreSQL` → `postgresql`、`2026` → `2026`）；
      * 其它字符（标点/空白/全角符号）一律是分隔符，不产出 token。

    不做停用词过滤（那是 `skeleton.content_tokens(zh=True)` 的活），也不去重
    （去重会丢掉"同一个词出现几次"的信息，集合由调用方自己取）。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if _is_cjk_char(ch):
            j = i + 1
            while j < n and _is_cjk_char(text[j]):
                j += 1
            run = text[i:j]
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
            continue
        m = _ASCII_ALNUM.match(text, i)
        if m is not None:
            out.append(m.group(0).lower())
            i = m.end()
            continue
        i += 1
    return out


# --------------------------------------------------------------------------- #
# 3. 中文停用词表（只收 1~2 字虚词；**绝不含否定词**）
# --------------------------------------------------------------------------- #
#
# 两条硬约束（测试逐条断言）：
#   1) 每一项长度 ∈ {1, 2}——`zh_tokens` 只产出 1~2 字汉 token，更长的项永远不会命中；
#   2) 任何一项都不得含否定标记字符（不/没/未/无/别/非/否/甭/莫/勿）；
#      这也排除了「不过」「无论」「别人」「非常」「分别」这类看似虚词、实则含否定字形的词。

_ZH_STOP_1 = """
的 了 是 在 我 你 他 她 它 这 那 和 与 就 也 都 很 还 又 会 能 要 有 吧 呢 啊 吗 着 过
而 且 或 之 但 却 被 把 给 让 使 由 从 向 对 为 以 及 等 个 些 您 其 此 亦 乃 则 于 啦
呀 哦 嗯 哈 嘛 呵 呜 哎 唉 耶 哇
"""

_ZH_STOP_2 = """
我们 你们 他们 她们 它们 咱们 大家 自己 我的 你的 他的 她的 它的 这个 那个 这些 那些
这里 那里 这样 那样 这种 那种 这么 那么 怎样 什么 怎么 为啥 因为 所以 但是 可是 然后
而且 或者 还是 就是 也是 都是 只是 而是 于是 因此 已经 正在 可以 应该 如果 虽然 即使
尽管 由于 对于 关于 按照 根据 一样 一起 一直 一般 时候 一个 一会 有点 有些 有的 每个 各个
之类 等等 上面 下面 里面 外面 前面 后面 中间 左右 附近 什么 这样
"""

ZH_STOP: frozenset[str] = frozenset((_ZH_STOP_1 + _ZH_STOP_2).split())

# --------------------------------------------------------------------------- #
# 4. 中文否定（标记表 + 有限豁免）
# --------------------------------------------------------------------------- #
#
# 契约要求的 14 个标记：没有/没/从未/不曾/并非/并不/绝不/不是/不用/未/无/别/甭/毫无。
# 单字标记（没/未/无/别）会产生词内假阳性，用**封闭豁免**压掉最常见的几类：
#   * 未 → 未来 / 未知（"未知"是否定其实也说得通，但"未来"一定不是；两个一起豁免，
#     代价是「未知」不再判否定，方向是更松，可接受）
#   * 无 → 无论 / 无线（"无线"是名词性"wireless"；"无论"是连词）
#   * 别 → 特别/区别/类别/性别/级别/告别/识别/差别/个别/派别/分别/话别/道别（前字判定）
#          别墅/别致/别扭/别处/别名/别样/别的/别国/别具（后字判定）
#   * 没 → 没问题 / 没毛病（"没问题"是约定俗成的正面评价，判否定会把"没问题的方案"整类判反）
#
# **方向说明**：豁免让"假否定→否定"减少（更准），但也让少数真否定被漏掉（更松）。
# 只影响 zh=True 路径；zh=False 路径一行代码都不经过这里。

_ZH_NEG_RE = re.compile(
    r"没有|从未|不曾|并非|并不|绝不|不是|不用|毫无"      # 多字标记（完整否定词）
    r"|没"
    r"|未(?!来|知)"
    r"|无(?!论|线)"
    r"|(?<![特区分性级别告派识差个类分话道])别(?!人|墅|致|扭|的|处|名|样|国|具)"
    r"|甭"
)

# 先遮掉的"含否定义但整体不是否定"的固定说法（顺序无关；确定性）
_NEG_EXEMPT: tuple[str, ...] = ("没问题", "没毛病", "没事儿")

ZH_NEGATION_MARKERS: tuple[str, ...] = (
    "没有", "从未", "不曾", "并非", "并不", "绝不", "不是", "不用", "毫无",
    "没", "未", "无", "别", "甭",
)


def zh_negated(text: str) -> bool:
    """是否含中文否定。确定性；命中任一标记即 True。

    会命中词内假阳性（如 `无线`/`未来` 之外的未收录同形词），说见模块 docstring：
    豁免表是封闭的，未收录的同形词按否定处理（方向更严）。
    """
    t = text
    for ex in _NEG_EXEMPT:
        if ex in t:
            t = t.replace(ex, "\u0000" * len(ex))
    return _ZH_NEG_RE.search(t) is not None


# --------------------------------------------------------------------------- #
# 5. 时间词归一化（精度可包含）
# --------------------------------------------------------------------------- #
#
# 规范串格式（**封闭**，测试逐项锁）：
#   绝对日期族： "YYYY" / "YYYY-MM" / "YYYY-MM-DD" / "MM" / "MM-DD"（两位补零）
#   时刻：       "HH:MM"（分缺省 00）
#   相对天：     "大前天/前天/昨天/今天/明天/后天/大后天/今晚/明晚/昨儿/今儿"
#   相对周：     "（上|下|本）周" / "（上|下|本）周X" / "周X"（星期/礼拜归一成 周；天→日）
#   相对月/年：  "上个月" / "下个月" / "本月" / "去年" / "今年" / "明年" / "前年" / "后年"
#   偏移量：     "N天前" / "N天后" / "N周前" / "N周后" / "N个月前" / "N个月后"
#   时段：       "早上/上午/中午/下午/晚上/夜里/凌晨/傍晚/半夜/清晨/深夜"
#
# 每一条规则匹配到的字符区间会被"消费"掉（`consumed`），避免 "2026年3月5日" 同时产出
# "2026年"、"3月"、"3月5日" 这类重叠项（重叠项会让 ⊆ 比较自相矛盾）。

_WEEKDAY_MAP = {"一": "一", "二": "二", "三": "三", "四": "四", "五": "五",
                "六": "六", "日": "日", "天": "日"}
_WEEKDAY_CLASS = "一二三四五六日天"


def _canon_point(y: int, mo: int, d: int) -> str | None:
    """校验并规范化绝对日期；非法日历日（2月30日）返回 None（宁可漏也不编）。"""
    try:
        date(y, mo, d)
    except ValueError:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _canon_month(y: int, mo: int) -> str | None:
    if not 1 <= mo <= 12:
        return None
    return f"{y:04d}-{mo:02d}"


def _canon_ymd_ascii(m: "re.Match[str]") -> set[str]:
    s = _canon_point(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return set() if s is None else {s}


def _canon_ymd_cn(m: "re.Match[str]") -> set[str]:
    s = _canon_point(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return set() if s is None else {s}


def _canon_ym_cn(m: "re.Match[str]") -> set[str]:
    s = _canon_month(int(m.group(1)), int(m.group(2)))
    return set() if s is None else {s}


def _canon_year(m: "re.Match[str]") -> set[str]:
    y = int(m.group(1))
    return {f"{y:04d}"}


def _canon_md_cn(m: "re.Match[str]") -> set[str]:
    mo, d = int(m.group(1)), int(m.group(2))
    # 缺年份的月日无法做日历校验（2月29日合法与否取决于年），只校验范围
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return set()
    return {f"{mo:02d}-{d:02d}"}


def _canon_month_only(m: "re.Match[str]") -> set[str]:
    mo = int(m.group(1))
    return {f"{mo:02d}"} if 1 <= mo <= 12 else set()


def _canon_offset_month(m: "re.Match[str]") -> set[str]:
    n = int(m.group(1))
    return {f"{n}个月{m.group(2)}"}


def _canon_offset_day(m: "re.Match[str]") -> set[str]:
    n = int(m.group(1))
    return {f"{n}天{m.group(2)}"}


def _canon_offset_week(m: "re.Match[str]") -> set[str]:
    n = int(m.group(1))
    return {f"{n}周{m.group(2)}"}


def _canon_hour(m: "re.Match[str]") -> set[str]:
    h = int(m.group(1))
    mi = int(m.group(2)) if m.group(2) else 0
    if not (0 <= h <= 24 and 0 <= mi <= 59):
        return set()
    return {f"{h:02d}:{mi:02d}"}


def _canon_rel_day(m: "re.Match[str]") -> set[str]:
    return {m.group(0)}


def _canon_week_prefix(m: "re.Match[str]") -> set[str]:
    prefix = {"这": "本"}.get(m.group(1), m.group(1))
    wd = m.group(2)
    if wd is None:
        return {f"{prefix}周"}
    return {f"{prefix}周{_WEEKDAY_MAP[wd]}"}


def _canon_week_bare(m: "re.Match[str]") -> set[str]:
    return {f"周{_WEEKDAY_MAP[m.group(1)]}"}


def _canon_month_rel(m: "re.Match[str]") -> set[str]:
    prefix = {"这": "本", "当": "本", "本": "本"}.get(m.group(1), m.group(1))
    if prefix == "本":
        return {"本月"}
    return {"上个月" if prefix == "上" else "下个月"}


def _canon_year_rel(m: "re.Match[str]") -> set[str]:
    return {m.group(0)}


def _canon_daypart(m: "re.Match[str]") -> set[str]:
    return {m.group(0)}


# 顺序即优先级（长/具体 → 短/泛化），与"消费区间"一起保证不重叠。
_ZH_TEMPORAL_RULES: tuple[tuple["re.Pattern[str]", object], ...] = (
    (re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"), _canon_ymd_ascii),
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"), _canon_ymd_cn),
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月"), _canon_ym_cn),
    (re.compile(r"(\d{4})\s*年"), _canon_year),
    (re.compile(r"(\d{1,3})\s*个?\s*月\s*([后前])"), _canon_offset_month),
    (re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"), _canon_md_cn),
    (re.compile(r"(\d{1,2})\s*月"), _canon_month_only),
    (re.compile(r"(\d{1,2})\s*天\s*([后前])"), _canon_offset_day),
    (re.compile(r"(\d{1,2})\s*个?\s*(?:周|星期|礼拜)\s*([后前])"), _canon_offset_week),
    (re.compile(r"(\d{1,2})\s*[点時时](?:\s*(\d{1,2})\s*分)?"), _canon_hour),
    (re.compile(r"大前天|大后天|前天|后天|昨天|今天|明天|今晚|明晚|昨儿|今儿"), _canon_rel_day),
    (re.compile(r"(上|下|本|这)\s*个?\s*(?:周|星期|礼拜)\s*([" + _WEEKDAY_CLASS + r"])?"),
     _canon_week_prefix),
    (re.compile(r"(?:周|星期|礼拜)\s*([" + _WEEKDAY_CLASS + r"])"), _canon_week_bare),
    (re.compile(r"(上|下|这|本|当)\s*个?\s*月"), _canon_month_rel),
    (re.compile(r"去年|今年|明年|前年|后年"), _canon_year_rel),
    (re.compile(r"早上|上午|中午|下午|晚上|夜里|凌晨|傍晚|半夜|清晨|深夜"), _canon_daypart),
)


def zh_temporal_terms(text: str) -> set[str]:
    """抽取并归一化时间词，返回**可比较的规范串**集合（空集 = 未提取到时间项）。

    确定性：规则顺序固定、区间消费固定；重复运行结果一致。示例：
        `2026年3月5日` → {"2026-03-05"}
        `上周三`      → {"上周三"}
        `3月5日`      → {"03-05"}
        `2026年`      → {"2026"}
        `昨天下午3点` → {"昨天", "下午", "03:00"}

    非法日历日期（`2026年2月30日`）与非法月份（`13月`）**整段消费且不产出任何项**：
    既不编一个"看起来像日期"的规范串，也不退回更粗的规则（否则 `2026年2月30日` 会退化成
    `2026-02`，把"日期非法"悄悄读成"只有月份"）。方向是漏检（软谓词语义），不假杀。
    """
    out: set[str] = set()
    consumed: list[tuple[int, int]] = []

    def _free(start: int, end: int) -> bool:
        return not any(not (end <= s or start >= e) for s, e in consumed)

    for pattern, builder in _ZH_TEMPORAL_RULES:
        for m in pattern.finditer(text):
            if not _free(m.start(), m.end()):
                continue
            terms = builder(m)  # type: ignore[operator]
            # 非法值：整段消费 + 不产出（宁可漏也不编，且不许退化到更粗的规则）
            out |= terms
            consumed.append((m.start(), m.end()))
    return out


# --- 精度包含（coverage） -------------------------------------------------- #
#
# 纯字符串 ⊆ 会把「claim 3月5日 / 证据 2026年3月5日」判成不兼容。这里给一条**只放宽、
# 不放宽判定方向**的覆盖关系：scope（证据）必须至少在断言指到的每一个分量上同样精确。
#   * 断言说 03-05、证据说 2026-03-05 → 覆盖（证据更精确）✓
#   * 断言说 2026-03-05、证据说 03-05     → 不覆盖（年份没有被证据接地）✗
#   * 断言说 上周三、证据说 上周          → 不覆盖（证据少一个"周几"分量）✗
#   * 断言说 上周、证据说 上周三          → 覆盖 ✓

_ABS_TERM = re.compile(r"^(?:(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?|(\d{2})(?:-(\d{2}))?)$")
_WEEK_TERM = re.compile(r"^(上|下|本)?周([" + _WEEKDAY_CLASS + r"])?$")


def _parse_abs(term: str) -> tuple[int | None, int | None, int | None] | None:
    m = _ABS_TERM.fullmatch(term)
    if m is None:
        return None
    if m.group(1) is not None:
        return (int(m.group(1)),
                int(m.group(2)) if m.group(2) else None,
                int(m.group(3)) if m.group(3) else None)
    return (None,
            int(m.group(4)),
            int(m.group(5)) if m.group(5) else None)


def _parse_week(term: str) -> tuple[str | None, str | None] | None:
    m = _WEEK_TERM.fullmatch(term)
    if m is None:
        return None
    return (m.group(1), m.group(2))


def _term_covered(quote: str, scope: str) -> bool:
    """scope 是否能覆盖 quote（只在同一族内比较；跨族一律不覆盖，除了字符串相等）。"""
    qa, sa = _parse_abs(quote), _parse_abs(scope)
    if qa is not None and sa is not None:
        for qv, sv in zip(qa, sa):
            if qv is None:
                continue
            if sv is None or sv != qv:
                return False
        return True
    qw, sw = _parse_week(quote), _parse_week(scope)
    if qw is not None and sw is not None:
        if qw[0] is not None and sw[0] != qw[0]:
            return False
        if qw[1] is not None and sw[1] != qw[1]:
            return False
        return True
    return False  # 其余族（相对天/时段/偏移量/时刻）只认字符串相等（已在调用方判过）


def zh_temporal_subset(quote_terms: Iterable[str], scope_terms: Iterable[str]) -> bool:
    """quote 的时间项是否全部被 scope 覆盖（⊆ 的精度包含加强版）。

    实现说明：先做普通集合 ⊆；不满足的项再试"精度包含"（`_term_covered`）。
    因此**普通 ⊆ 成立就一定成立**，本函数只会把"证据更精确"的合法情形放进来，
    不会反过来放宽"断言比证据更精确"的情形。
    """
    scope = set(scope_terms)
    for q in set(quote_terms):
        if q in scope:
            continue
        if not any(_term_covered(q, s) for s in scope):
            return False
    return True


# --------------------------------------------------------------------------- #
# 6. 更正 / 指代 / 推测（场景谓词用的结构信号）
# --------------------------------------------------------------------------- #

ZH_CORRECTED: frozenset[str] = frozenset(
    ("不是", "改成", "其实", "更正", "说错", "改口", "记错", "不对")
)

ZH_ANAPHOR: frozenset[str] = frozenset(
    ("上次", "上回", "之前", "刚才", "前面说", "我们说", "你说过", "刚才那")
)

ZH_HEDGE: frozenset[str] = frozenset(
    ("可能", "也许", "大概", "或许", "建议", "应该", "可以", "不妨", "大概率")
)


def _has_any(text: str, markers: Iterable[str]) -> bool:
    return any(m in text for m in markers)


def zh_corrected(text: str) -> bool:
    """是否含**更正标记**（"不是…是…" / 改成 / 其实 / 更正 / 说错 / 改口 / 记错 / 不对）。

    注意：这些标记只说明"文本在更正"，不判断更正了什么；具体是否与 claim 有关由
    `scenario.py` 的 content-token 交集负责。
    """
    return _has_any(text, ZH_CORRECTED)


def zh_anaphor(text: str) -> bool:
    """是否含**指代**（上次 / 上回 / 之前 / 刚才 / 前面说 / 我们说 / 你说过 / 刚才那）。"""
    return _has_any(text, ZH_ANAPHOR)


def zh_hedge(text: str) -> bool:
    """是否含**推测/建议**（可能 / 也许 / 大概 / 或许 / 建议 / 应该 / 可以 / 不妨 / 大概率）。"""
    return _has_any(text, ZH_HEDGE)
