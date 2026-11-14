# SPDX-License-Identifier: Apache-2.0
"""Plan Z C5：state slot 的**纯契约**（无 I/O、无模型、无时钟）。

边界（用户裁决 D-PZ-5 / 裁决记录 §C5）
────────────────────────────────────────────────────────────────────────────
本模块**只**定义数据结构与确定性序列化/哈希：

* `StateSlot`：从 L0 原文抽出的一个状态槽（谁/何时/什么属性/什么值，全部带字符区间，
  可回原文逐字核对）；
* `SlotEvent`：槽的生命周期事件（proposed / verified / rejected / invalidated），
  事件之间用 `prev_event_hash` 组成只追加 hash 链（与账本同款思路）。

**存储实现必须在 core 之外**：append-only JSONL + 同款 hash 链的生产实现放
`production package/`（顶层包，owner 待裁决）；私有参考实现在工具目录（未进 core）。
本批**不把 slot 接入 `enforce()` 判定**——槽精度未达 `current_basis` 门槛
（模型抽取 verified attr 92.27%，规则基线 0.212），接入需另立任务并通过 95% 门槛 + 人工冻结集。

红线
────
1. 不 import `openai` / `httpx` / `requests` / `onnxruntime` / 任何网络或模型库；
2. **不读时钟**：`emitted_at` 由调用方传入（时钟/时区归调用方拥有，core 只做确定性变换）；
3. 不做 I/O、无随机：同输入永远同一输出（`to_dict` / `from_dict` / 哈希都是纯函数）。

C6 附注
───────
「已送达」判定在代理层（SSE 写成功计数），不在 core；本模块与 `AnchorRecord` 的
`is_complete / finish_reason / delivered_sha256` 只**存字段**，不做判定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .canon import digest

__all__ = [
    "DOMAIN_SLOT_EVENT",
    "DOMAIN_SLOT_PAYLOAD",
    "POLARITIES",
    "SLOT_EVENT_KINDS",
    "SLOT_SCHEMA_VERSION",
    "ZERO_EVENT_HASH",
    "SlotEvent",
    "StateSlot",
    "make_slot_event",
    "slot_event_hash",
    "state_slot_payload_hash",
]

#: 域分隔标签（与 canon.py 的域名分开；这里不新增 canon 常量，避免动既有内核契约）
DOMAIN_SLOT_PAYLOAD = b"MEM/SLOT-PAYLOAD/v1"
DOMAIN_SLOT_EVENT = b"MEM/SLOT-EVENT/v1"

#: slot 契约版本（独立于账本 schema；槽表升级不触碰 L0 账本）
SLOT_SCHEMA_VERSION = "slot-v1"

#: 事件类型（封闭集合；顺序即语义上的生命周期顺序）
SLOT_EVENT_KINDS: tuple[str, ...] = ("proposed", "verified", "rejected", "invalidated")

#: 极性（封闭集合；neg/停止类的 value 语义仍有待裁决，暂用 unknown 兜底而不是猜）
POLARITIES: tuple[str, ...] = ("positive", "negative", "neutral", "unknown")

#: 事件链起点（与账本的 ZERO_HASH 同构）
ZERO_EVENT_HASH = "0" * 64


def _check_str(value: Any, *, what: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise ValueError(f"{what} must be a non-empty str")
    return value


def _check_span(value: Any, *, what: str) -> tuple[int, int]:
    """字符区间：两个非负整数且 start <= end（确定性，不自动纠正顺序）。"""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"{what} must be a 2-item sequence, got {value!r}")
    start, end = value[0], value[1]
    for idx, item in enumerate((start, end)):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{what}[{idx}] must be int, got {type(item).__name__}")
        if item < 0:
            raise ValueError(f"{what}[{idx}] must be >= 0, got {item!r}")
    if start > end:
        raise ValueError(f"{what} must satisfy start <= end, got {(start, end)!r}")
    return (start, end)


def _canonical_json(obj: Mapping[str, Any]) -> str:
    """确定性 JSON：键排序 + 紧凑分隔 + 非 ASCII 原样保留。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# SlotEvent：生命周期事件（只追加 hash 链）
# --------------------------------------------------------------------------- #
def slot_event_hash(
    kind: str,
    slot_id: str,
    l0_id: str,
    emitted_at: str,
    payload_hash: str,
    prev_event_hash: str,
) -> str:
    """事件的确定性哈希：覆盖事件全部字段 + 前驱哈希（篡改任一字段 → 链断裂）。"""
    return digest(
        DOMAIN_SLOT_EVENT,
        kind,
        slot_id,
        l0_id,
        emitted_at,
        payload_hash,
        prev_event_hash,
    )


@dataclass(frozen=True)
class SlotEvent:
    """一个只追加的槽事件（字段顺序即哈希口径，见 `slot_event_hash`）。

    * `emitted_at`：**调用方**提供的时点（ISO-8601 串或调用方约定的稳定格式）；
      core 不读时钟，因此同输入永远同输出。
    * `payload_hash`：该事件所涉槽内容/修改请求的哈希（由调用方用 `StateSlot` 或
      自己的 payload 算法算好传入）。
    * `prev_event_hash`：前一个事件的 `event_hash`；链首用 `ZERO_EVENT_HASH`。
    """

    kind: str
    slot_id: str
    l0_id: str
    emitted_at: str
    payload_hash: str
    prev_event_hash: str
    event_hash: str

    def __post_init__(self) -> None:
        if self.kind not in SLOT_EVENT_KINDS:
            raise ValueError(f"kind must be one of {SLOT_EVENT_KINDS}, got {self.kind!r}")
        _check_str(self.slot_id, what="slot_id")
        _check_str(self.l0_id, what="l0_id")
        _check_str(self.emitted_at, what="emitted_at")
        _check_str(self.payload_hash, what="payload_hash")
        _check_str(self.prev_event_hash, what="prev_event_hash")
        _check_str(self.event_hash, what="event_hash")

    def recompute_event_hash(self) -> str:
        """按契约字段重算 `event_hash`（校验用；不改对象）。"""
        return slot_event_hash(
            self.kind,
            self.slot_id,
            self.l0_id,
            self.emitted_at,
            self.payload_hash,
            self.prev_event_hash,
        )

    def is_self_consistent(self) -> bool:
        """`event_hash` 是否与字段一致（不验证前驱是否在链上——那需要存储实现）。"""
        return self.event_hash == self.recompute_event_hash()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "slot_id": self.slot_id,
            "l0_id": self.l0_id,
            "emitted_at": self.emitted_at,
            "payload_hash": self.payload_hash,
            "prev_event_hash": self.prev_event_hash,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "SlotEvent":
        if not isinstance(obj, Mapping):
            raise ValueError(f"SlotEvent.from_dict expects a mapping, got {type(obj).__name__}")
        return cls(
            kind=obj["kind"],
            slot_id=obj["slot_id"],
            l0_id=obj["l0_id"],
            emitted_at=obj["emitted_at"],
            payload_hash=obj["payload_hash"],
            prev_event_hash=obj["prev_event_hash"],
            event_hash=obj["event_hash"],
        )


def make_slot_event(
    *,
    kind: str,
    slot_id: str,
    l0_id: str,
    emitted_at: str,
    payload_hash: str,
    prev_event_hash: str = ZERO_EVENT_HASH,
) -> SlotEvent:
    """构造一个 hash 正确的事件（唯一的"造事件"入口，避免手填 `event_hash`）。"""
    return SlotEvent(
        kind=kind,
        slot_id=slot_id,
        l0_id=l0_id,
        emitted_at=emitted_at,
        payload_hash=payload_hash,
        prev_event_hash=prev_event_hash,
        event_hash=slot_event_hash(kind, slot_id, l0_id, emitted_at, payload_hash, prev_event_hash),
    )


# --------------------------------------------------------------------------- #
# StateSlot：从 L0 原文抽出的状态槽（全部带区间，可回原文逐字核对）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateSlot:
    """一个状态槽。所有区间都是**原文字符区间**（`char_span` 限定在 L0 记录内）。

    设计约束（来自用户裁决与参考实现）：
    * `value_quote` 必须是 L0 原文的连续子串——抽取结果必须能被逐字校验；
    * `l0_hash` 把槽钉死在某条具体 L0 内容上（L0 不可改，所以槽可审计）；
    * `model_id` / `prompt_hash` / `extractor_version` 让"同一个槽由谁、用什么版本、
      什么提示词抽出"可追溯（模型只提议，判定在 core 之外）。
    """

    slot_id: str
    l0_id: str
    l0_hash: str
    char_span: tuple[int, int]
    subject_span: tuple[int, int]
    attribute: str
    attribute_span: tuple[int, int]
    value_span: tuple[int, int]
    value_quote: str
    polarity: str
    extractor_version: str
    model_id: Optional[str]
    prompt_hash: Optional[str]
    schema_version: str = SLOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_str(self.slot_id, what="slot_id")
        _check_str(self.l0_id, what="l0_id")
        _check_str(self.l0_hash, what="l0_hash")
        _check_str(self.attribute, what="attribute")
        _check_str(self.value_quote, what="value_quote")
        _check_str(self.extractor_version, what="extractor_version")
        _check_str(self.schema_version, what="schema_version")
        _check_span(self.char_span, what="char_span")
        _check_span(self.subject_span, what="subject_span")
        _check_span(self.attribute_span, what="attribute_span")
        _check_span(self.value_span, what="value_span")
        if self.polarity not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES}, got {self.polarity!r}")
        if self.model_id is not None:
            _check_str(self.model_id, what="model_id")
        if self.prompt_hash is not None:
            _check_str(self.prompt_hash, what="prompt_hash")
        # 区间必须落在 char_span 内（槽是 L0 记录的一个子区间；不越界、不反向）
        _c0, _c1 = self.char_span
        for name in ("subject_span", "attribute_span", "value_span"):
            start, end = getattr(self, name)
            if start < _c0 or end > _c1:
                raise ValueError(
                    f"{name} {(start, end)!r} is outside char_span {self.char_span!r}"
                )

    def to_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "l0_id": self.l0_id,
            "l0_hash": self.l0_hash,
            "char_span": list(self.char_span),
            "subject_span": list(self.subject_span),
            "attribute": self.attribute,
            "attribute_span": list(self.attribute_span),
            "value_span": list(self.value_span),
            "value_quote": self.value_quote,
            "polarity": self.polarity,
            "extractor_version": self.extractor_version,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "StateSlot":
        if not isinstance(obj, Mapping):
            raise ValueError(f"StateSlot.from_dict expects a mapping, got {type(obj).__name__}")
        return cls(
            slot_id=obj["slot_id"],
            l0_id=obj["l0_id"],
            l0_hash=obj["l0_hash"],
            char_span=tuple(obj["char_span"]),
            subject_span=tuple(obj["subject_span"]),
            attribute=obj["attribute"],
            attribute_span=tuple(obj["attribute_span"]),
            value_span=tuple(obj["value_span"]),
            value_quote=obj["value_quote"],
            polarity=obj["polarity"],
            extractor_version=obj["extractor_version"],
            model_id=obj.get("model_id"),
            prompt_hash=obj.get("prompt_hash"),
            schema_version=obj.get("schema_version", SLOT_SCHEMA_VERSION),
        )

    def canonical_json(self) -> str:
        """确定性 JSON（键排序、紧凑分隔）——同一槽永远同一字节串。"""
        return _canonical_json(self.to_dict())


def state_slot_payload_hash(slot: StateSlot) -> str:
    """槽内容的确定性哈希（可作 `SlotEvent.payload_hash` 用）。"""
    return digest(DOMAIN_SLOT_PAYLOAD, slot.canonical_json())
