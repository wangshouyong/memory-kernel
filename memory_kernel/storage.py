# SPDX-License-Identifier: Apache-2.0
"""存储端口（internal-task；契约 `docs/internal-doc.md` §1、`docs/PRINCIPAL-SCOPE-DESIGN.md` §2）。

只定**接口签名**，不做具体数据库实现：
- `LedgerStore`：只追加账本的语义操作（append / get / try_get / 长度 / 迭代 / 验链）
- `RecordTimeStore`：锚点 -> 入库时点的只追加侧表（场景谓词需要）
- `AuditStore`：审计事件只追加
- `IndexStore`：检索索引（派生数据，必须可从账本重建）

本地适配器 `LocalLedgerStore` 只做委托，行为与 `AnchorLedger` 完全一致（默认路径零变化）。
SaaS 阶段换远端适配器即可，内核不感知存储形态。
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Protocol, runtime_checkable

from .identity import AuditEvent
from .ledger import AnchorLedger, AnchorRecord

__all__ = [
    "LedgerStore",
    "RecordTimeStore",
    "AuditStore",
    "IndexStore",
    "LocalLedgerStore",
    "InMemoryRecordTimeStore",
    "InMemoryAuditStore",
]


@runtime_checkable
class LedgerStore(Protocol):
    """只追加账本端口（语义操作，不暴露文件格式）。"""

    def append(self, content: str, **fields) -> AnchorRecord: ...

    def get(self, anchor_id: str) -> AnchorRecord: ...

    def try_get(self, anchor_id: str) -> Optional[AnchorRecord]: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[AnchorRecord]: ...

    def verify_chain(self) -> tuple[bool, Optional[int]]: ...


@runtime_checkable
class RecordTimeStore(Protocol):
    """锚点 -> 入库时点（epoch 秒）只追加侧表；缺时点返回 None（谓词不可判）。"""

    def record_time(self, anchor_id: str, ts: float) -> None: ...

    def get_time(self, anchor_id: str) -> Optional[float]: ...

    def items(self) -> Iterable[tuple[str, float]]: ...


@runtime_checkable
class AuditStore(Protocol):
    """审计事件只追加。"""

    def append_event(self, event: AuditEvent) -> None: ...

    def events(self) -> Iterable[AuditEvent]: ...


@runtime_checkable
class IndexStore(Protocol):
    """检索索引（派生数据）：可重建、可查询、不改变账本。"""

    def rebuild(self, records: Iterable[AnchorRecord]) -> None: ...

    def query(self, text: str, *, limit: int = 10) -> list[str]: ...

    def __len__(self) -> int: ...


class LocalLedgerStore:
    """本地 JSONL 账本适配器：纯委托，零行为变化。"""

    def __init__(self, ledger: AnchorLedger) -> None:
        self._ledger = ledger

    def append(self, content: str, **fields) -> AnchorRecord:
        return self._ledger.append(content, **fields)

    def get(self, anchor_id: str) -> AnchorRecord:
        return self._ledger.get(anchor_id)

    def try_get(self, anchor_id: str) -> Optional[AnchorRecord]:
        return self._ledger.try_get(anchor_id)

    def __len__(self) -> int:
        return len(self._ledger)

    def __iter__(self) -> Iterator[AnchorRecord]:
        return iter(self._ledger)

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        return self._ledger.verify_chain()


class InMemoryRecordTimeStore:
    """本地时点侧表：同锚点重复写相同值幂等；不同值显式失败（不静默覆盖）。"""

    def __init__(self) -> None:
        self._times: dict[str, float] = {}

    def record_time(self, anchor_id: str, ts: float) -> None:
        if not isinstance(ts, (int, float)) or ts < 0:
            raise ValueError("ts must be a non-negative number")
        old = self._times.get(anchor_id)
        if old is not None and old != ts:
            raise ValueError(f"record_time conflict for {anchor_id!r}: {old} != {ts}")
        self._times[anchor_id] = float(ts)

    def get_time(self, anchor_id: str) -> Optional[float]:
        return self._times.get(anchor_id)

    def items(self) -> Iterable[tuple[str, float]]:
        return tuple(self._times.items())


class InMemoryAuditStore:
    """本地审计事件存储（只追加）。"""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def append_event(self, event: AuditEvent) -> None:
        self._events.append(event)

    def events(self) -> Iterable[AuditEvent]:
        return tuple(self._events)
