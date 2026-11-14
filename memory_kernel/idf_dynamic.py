# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""动态 IDF（防静态陷阱）—— internal-task / SPEC §13 D-IDF-1。

设计要点（与 `docs/IDF-DYNAMIC-v1.md` 对应）：

1. **增量 df**：`add()/remove()` 直接维护 `df` 与 `n_docs`；增删与全量重算**同构**（纯整数计数，
   没有浮点累加，因此不存在浮点漂移；权重仍在读取时按同一公式现算）。
2. **版本号**：`idf-<yyyymmddHHMM>-<sha8>`，sha8 = 语料（排序 df + n_docs）+ 参数
   `(stem_ok, stop_mode, zh)` 的 sha256 前 8 位；同语料同参数同 epoch → 同版本号。
3. **双缓冲原子切换**：`prebuild()` 在锁外构建新快照，`commit()` 在锁内一次性替换；
   构建期间旧快照继续服务，切换后旧快照对象仍可被已持有它的调用方使用（不会被改写）。
4. **定期全量重算入口**：`maintenance_due()` + `full_recompute()`；时钟由调用方注入
   （core 不读时钟，沿用 D-PZ-4 的"预算/时钟归调用方"原则）。
5. **历史版本不静默删除**：所有提交过的快照进 `history()`，`get(version)` 可取回。
6. **默认零变化**：本模块不被 `skeleton/enforce/pipeline` 默认路径 import；只有显式注入
   `idf_provider` 才启用（见各注入点的 `None` 默认值）。

core 隔离：仅 stdlib + 包内导入（`test_core_isolation.py` 会检查）。
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .skeleton import IDFModel, norm_tokens

__all__ = [
    "DynamicIDF",
    "IDFSnapshot",
    "IDF_VERSION_PREFIX",
    "IDF_VERSION_RE",
    "corpus_digest",
    "is_valid_idf_version",
    "make_idf_version",
]

IDF_VERSION_PREFIX = "idf-"
IDF_VERSION_RE = re.compile(r"^idf-\d{12}-[0-9a-f]{8}$")
_EPOCH_RE = re.compile(r"^\d{12}$")
_DIGEST_LEN = 8


def _digest8(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_LEN]


def is_valid_idf_version(version: object) -> bool:
    return isinstance(version, str) and bool(IDF_VERSION_RE.match(version))


def make_idf_version(epoch: str, sha8: str) -> str:
    """`epoch` 必须是调用方给的 `yyyymmddHHMM`（12 位数字）；本函数不读时钟。"""
    if not isinstance(epoch, str) or not _EPOCH_RE.match(epoch):
        raise ValueError(f"epoch must be yyyyMMddHHMM (12 digits), got {epoch!r}")
    if not isinstance(sha8, str) or not re.match(r"^[0-9a-f]{8}$", sha8):
        raise ValueError(f"sha8 must be 8 lowercase hex chars, got {sha8!r}")
    return f"{IDF_VERSION_PREFIX}{epoch}-{sha8}"


def corpus_digest(token_sets: Iterable[set[str]], *, stem_ok: bool = True,
                  stop_mode: str = "v2", zh: bool = False) -> str:
    """语料 + 参数的确定性摘要（与文档顺序无关：df 按键排序后再哈希）。"""
    df: dict[str, int] = {}
    n_docs = 0
    for toks in token_sets:
        n_docs += 1
        for t in toks:
            df[t] = df.get(t, 0) + 1
    h = hashlib.sha256()
    h.update(f"idf-dynamic|{int(bool(stem_ok))}|{stop_mode}|{int(bool(zh))}|{n_docs}".encode("utf-8"))
    for token in sorted(df):
        h.update(f"|{token}:{df[token]}".encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class IDFSnapshot:
    """不可变 IDF 快照。`model` 可直接注入 `enforce(idf_model=...)` / `skeleton_check`。"""

    version: str
    model: IDFModel
    n_docs: int
    epoch: str
    sha8: str
    reason: str
    params: tuple[bool, str, bool]

    @property
    def vocab(self) -> int:
        return len(self.model._df)  # noqa: SLF001 - 快照自身审计用

    def meta(self) -> dict[str, Any]:
        """检索结果元数据（D-IDF-1 第 3 条：结果必须携带实际使用的版本）。"""
        return {
            "idf_version": self.version,
            "idf_n_docs": self.n_docs,
            "idf_vocab": self.vocab,
            "idf_epoch": self.epoch,
        }


class DynamicIDF:
    """增量维护 + 双缓冲切换的 IDF 提供者。

    线程模型：所有可变状态（`_df/_n_docs/_active/_history`）都在 `_lock` 内；快照是
    不可变对象，读路径无锁（`snapshot()` 只在需要构建时加锁）。构建（`prebuild()`）在锁外，
    因此不会阻塞读；`commit()` 只做指针替换。
    """

    def __init__(self, *, stem_ok: bool = True, stop_mode: str = "v2", zh: bool = False,
                 max_age_minutes: Optional[int] = None,
                 max_pending_docs: Optional[int] = None,
                 epoch_fn: Optional[Callable[[], str]] = None) -> None:
        self._stem_ok = bool(stem_ok)
        self._stop_mode = str(stop_mode)
        self._zh = bool(zh)
        self._max_age_minutes = max_age_minutes
        self._max_pending_docs = max_pending_docs
        self._epoch_fn = epoch_fn
        self._lock = threading.Lock()
        self._df: dict[str, int] = {}
        self._n_docs = 0
        self._revision = 0
        self._active: Optional[IDFSnapshot] = None
        self._history: dict[str, IDFSnapshot] = {}
        self._stats = {"appends": 0, "removals": 0, "swaps": 0, "pending_docs": 0,
                       "last_reason": None}
        self._last_epoch: Optional[str] = None

    # ---- 参数 ---------------------------------------------------------- #
    @property
    def params(self) -> tuple[bool, str, bool]:
        return (self._stem_ok, self._stop_mode, self._zh)

    def _tokens(self, text: str) -> set[str]:
        return norm_tokens(text or "", stem_ok=self._stem_ok,
                           stop_mode=self._stop_mode, zh=self._zh)

    # ---- 增量维护 ------------------------------------------------------ #
    def add(self, text: str) -> int:
        """追加一条文档：df+1 / n_docs+1。返回新 revision。"""
        toks = self._tokens(text)
        with self._lock:
            for t in toks:
                self._df[t] = self._df.get(t, 0) + 1
            self._n_docs += 1
            self._revision += 1
            self._stats["appends"] += 1
            self._stats["pending_docs"] += 1
            return self._revision

    def add_record(self, record) -> int:
        return self.add(getattr(record, "content", ""))

    def remove(self, text: str) -> int:
        """删除/遗忘一条文档：df-1 / n_docs-1；token 未出现是逻辑错误，直接报错（确定性）。"""
        toks = self._tokens(text)
        with self._lock:
            if self._n_docs <= 0:
                raise ValueError("remove() on empty DynamicIDF")
            missing = [t for t in toks if self._df.get(t, 0) <= 0]
            if missing:
                raise ValueError(f"remove() token(s) not present: {sorted(missing)[:5]}")
            for t in toks:
                self._df[t] -= 1
                if self._df[t] == 0:
                    del self._df[t]
            self._n_docs -= 1
            self._revision += 1
            self._stats["removals"] += 1
            if self._stats["pending_docs"] > 0:
                self._stats["pending_docs"] -= 1
            return self._revision

    def remove_record(self, record) -> int:
        return self.remove(getattr(record, "content", ""))

    # ---- 双缓冲：构建/提交 --------------------------------------------- #
    def _resolve_epoch(self, epoch: Optional[str]) -> str:
        if epoch is None:
            epoch = self._epoch_fn() if self._epoch_fn is not None else None
        if epoch is None:
            raise ValueError("epoch is required (caller owns the clock; no clock is read in core)")
        if not isinstance(epoch, str) or not _EPOCH_RE.match(epoch):
            raise ValueError(f"epoch must be yyyyMMddHHMM (12 digits), got {epoch!r}")
        return epoch

    def prebuild(self, *, epoch: Optional[str] = None, reason: str = "manual") -> IDFSnapshot:
        """锁外构建新快照（旧快照继续服务）。校验：快照参数必须与本实例一致。"""
        ep = self._resolve_epoch(epoch)
        with self._lock:
            df = dict(self._df)
            n_docs = self._n_docs
        sha8 = _digest8(f"idf-dynamic|{int(self._stem_ok)}|{self._stop_mode}|"
                        f"{int(self._zh)}|{n_docs}|" +
                        "|".join(f"{t}:{df[t]}" for t in sorted(df)))
        version = make_idf_version(ep, sha8)
        model = IDFModel(version=version, n_docs=max(1, n_docs), df=df)
        return IDFSnapshot(version=version, model=model, n_docs=n_docs, epoch=ep,
                           sha8=sha8, reason=reason, params=self.params)

    def commit(self, snapshot: IDFSnapshot) -> IDFSnapshot:
        """锁内一次性切换（双缓冲）。旧版本进 history，不删除。"""
        if not isinstance(snapshot, IDFSnapshot):
            raise TypeError("commit() expects IDFSnapshot")
        if snapshot.params != self.params:
            raise ValueError(f"snapshot params {snapshot.params} != provider params {self.params}")
        if not is_valid_idf_version(snapshot.version):
            raise ValueError(f"invalid idf version {snapshot.version!r}")
        with self._lock:
            previous = self._active
            self._active = snapshot
            self._history[snapshot.version] = snapshot
            self._stats["swaps"] += 1
            self._stats["pending_docs"] = 0
            self._stats["last_reason"] = snapshot.reason
            self._last_epoch = snapshot.epoch
        return previous if previous is not None else snapshot

    def rebuild(self, *, epoch: Optional[str] = None, reason: str = "manual") -> IDFSnapshot:
        """增量状态 → 新快照 → 原子切换。返回**当前生效**快照。"""
        snap = self.prebuild(epoch=epoch, reason=reason)
        self.commit(snap)
        return snap

    def rebuild_from_ledger(self, ledger, *, epoch: Optional[str] = None,
                            reason: str = "full_recompute") -> IDFSnapshot:
        """用全量账本做一次全量重算（与增量路径同构：同样按 token 集计 df）。"""
        counts: dict[str, int] = {}
        n_docs = 0
        for rec in ledger:
            toks = self._tokens(getattr(rec, "content", ""))
            n_docs += 1
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        with self._lock:
            self._df = counts
            self._n_docs = n_docs
            self._revision += 1
            self._stats["pending_docs"] = 0
        return self.rebuild(epoch=epoch, reason=reason)

    def load_token_sets(self, token_sets: Iterable[set[str]], *,
                        epoch: Optional[str] = None, reason: str = "load") -> IDFSnapshot:
        """直接装载（离线/重放用）：与增量维护后的状态必须逐位一致。"""
        counts: dict[str, int] = {}
        n_docs = 0
        for toks in token_sets:
            n_docs += 1
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        with self._lock:
            self._df = counts
            self._n_docs = n_docs
            self._revision += 1
        return self.rebuild(epoch=epoch, reason=reason)

    # ---- 读路径 -------------------------------------------------------- #
    def snapshot(self, *, epoch: Optional[str] = None) -> IDFSnapshot:
        """返回当前生效快照；尚未构建时按当前增量状态构建一次（仅首建）。"""
        with self._lock:
            active = self._active
        if active is not None:
            return active
        return self.rebuild(epoch=epoch, reason="initial")

    @property
    def active_version(self) -> Optional[str]:
        with self._lock:
            return self._active.version if self._active is not None else None

    def meta(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
        return active.meta() if active is not None else {"idf_version": None,
                                                         "idf_n_docs": 0, "idf_vocab": 0}

    def history(self) -> tuple[dict[str, Any], ...]:
        """已提交版本（只追加，不删除）——审计可回答"历史结果当时用的哪个版本"。"""
        with self._lock:
            items = list(self._history.values())
        return tuple({"version": s.version, "epoch": s.epoch, "sha8": s.sha8,
                      "n_docs": s.n_docs, "vocab": s.vocab, "reason": s.reason}
                     for s in items)

    def get(self, version: str) -> IDFSnapshot:
        with self._lock:
            snap = self._history.get(version)
        if snap is None:
            raise KeyError(f"unknown idf version {version!r}")
        return snap

    # ---- 定期全量重算 / 监控 ------------------------------------------- #
    def maintenance_due(self, *, now_epoch: str) -> bool:
        """低峰期触发全量重算：按版本年龄或未合并的增量文档数判定（时钟由调用方给）。"""
        ep = self._resolve_epoch(now_epoch)
        with self._lock:
            active = self._active
            pending = self._stats["pending_docs"]
            last = self._last_epoch
        if active is None:
            return True
        if self._max_pending_docs is not None and pending >= self._max_pending_docs:
            return True
        if self._max_age_minutes is not None and last is not None:
            if _epoch_minutes(ep) - _epoch_minutes(last) >= self._max_age_minutes:
                return True
        return False

    def full_recompute(self, ledger, *, now_epoch: str,
                       reason: str = "scheduled") -> IDFSnapshot:
        """定期全量重算入口（低峰期调用；调用方保证不阻塞查询）。"""
        return self.rebuild_from_ledger(ledger, epoch=now_epoch, reason=reason)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            snap = dict(self._stats)
            snap.update({
                "n_docs_incremental": self._n_docs,
                "vocab_incremental": len(self._df),
                "revision": self._revision,
                "versions": len(self._history),
                "active_version": active.version if active else None,
                "active_epoch": self._last_epoch,
                "active_n_docs": active.n_docs if active else 0,
            })
        return snap


def _epoch_minutes(epoch: str) -> int:
    """`yyyymmddHHMM` -> 绝对分钟数（仅用于年龄比较；不读时钟）。"""
    y, mo, d, h, mi = (int(epoch[0:4]), int(epoch[4:6]), int(epoch[6:8]),
                       int(epoch[8:10]), int(epoch[10:12]))
    return ((y * 32 + mo) * 32 + d) * 24 * 60 + h * 60 + mi
