# SPDX-License-Identifier: Apache-2.0
"""跨请求缓存（internal-task）：显式、可注入、可清空、可重建的 `EngineCache`。

为什么需要这一层
────────────────
`internal-task`（无状态化）审计发现 `enforce` 路径只剩两类**跨请求可变状态**：
`memory_kernel/skeleton.py` 的模块级弱引用缓存 `_LEDGER_IDF` / `_LEDGER_ENTITIES`
（按账本身份缓存 IDF 与实体索引）。它们本身是**纯派生数据**（可由账本确定性重建），
但作为"藏起来的模块全局"有三个问题：

1. **不可注入**：多实例/多租户想隔离缓存只能共用一个进程级字典；
2. **不可显式丢弃**：没有 `clear()`，测试/长驻进程无法主动释放或验证"丢了也一样"；
3. **隐式生命周期**：`WeakKeyDictionary` 的回收时机依赖 GC，不是可审计的契约。

本模块把这些缓存收敛成**一个显式对象**：

* 缓存内容 = 只由 `(ledger 身份, len(ledger) 代际, 分词参数)` 决定的派生值；
* `EngineCache()` 每个实例独立：两个实例互不影响（`AnswerPipeline` 每实例一份）；
* `clear()` 丢弃全部派生缓存，下一次读取**原地重建**（结果不变，只影响耗时）；
* 可选 `idf_provider`（`memory_kernel.idf_dynamic.DynamicIDF` 等带 `snapshot()` 的提供者）：
  给出时 IDF 直接读其当前生效快照（版本化、可审计），本层不落 IDF 缓存；
  **provider 自己的增量状态/历史语义不变**，`clear()` 也不重置 provider。

默认零行为变化
──────────────
`EngineCache` 只做 memoization：同一输入 → 同一输出。是否命中缓存、缓存是否被清空，
**不影响任何判定结果**（`tests/test_stateless.py` 断言 default / 注入 / clear 后三者结果一致）。
旧 API `skeleton.idf_for_ledger(ledger)` / `skeleton.entities_for_ledger(ledger)` 仍可用，
内部委托 `default_engine_cache()`（进程级兼容默认，可 `clear()` / 可替换 / 可重置）；
新代码（`AnswerPipeline` / `enforce(cache=...)`）应显式注入自己的 `EngineCache`。

确定性/隔离：只 import stdlib + 包内模块；无 I/O、无网络、无时钟；`IDFModel` 不可变。
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any, Optional

from .skeleton import IDFModel, STOP, _raw_words, build_idf, norm_tokens

__all__ = [
    "CacheStats",
    "EngineCache",
    "clear_default_cache",
    "default_engine_cache",
    "reset_default_engine_cache",
    "set_default_engine_cache",
]


@dataclass(frozen=True)
class CacheStats:
    """缓存计数器（只做可观测性；不影响判定）。"""

    idf_hits: int = 0
    idf_misses: int = 0
    idf_builds: int = 0
    entity_hits: int = 0
    entity_misses: int = 0
    entity_builds: int = 0
    clears: int = 0

    def as_dict(self) -> dict:
        return {
            "idf_hits": self.idf_hits,
            "idf_misses": self.idf_misses,
            "idf_builds": self.idf_builds,
            "entity_hits": self.entity_hits,
            "entity_misses": self.entity_misses,
            "entity_builds": self.entity_builds,
            "clears": self.clears,
        }


class EngineCache:
    """跨请求派生数据缓存：每实例独立、可 `clear()`、可重建、可选 IDF provider。

    线程安全：所有可变状态在 `_lock` 内；返回的 `IDFModel` 不可变，读者拿到后不受别人
    `clear()`/重建影响。弱引用：不持有账本强引用（账本可被回收，对应条目随之消失）。

    使用：

    ```python
    cache = EngineCache()                       # 或 EngineCache(idf_provider=DynamicIDF(...))
    enforce(ledger, claim, policy=pol, cache=cache)
    cache.clear()                               # 丢弃 → 下次读取原地重建
    ```
    """

    def __init__(self, *, idf_provider: Optional[Any] = None) -> None:
        self._lock = threading.RLock()
        # ledger -> (generation, {(stem_ok, stop_mode, zh): IDFModel})
        self._idf: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        # ledger -> (generation, {source_id: frozenset(names)})
        self._entities: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self._idf_provider = None
        self._counters: dict[str, int] = {
            "idf_hits": 0, "idf_misses": 0, "idf_builds": 0,
            "entity_hits": 0, "entity_misses": 0, "entity_builds": 0,
            "clears": 0,
        }
        if idf_provider is not None:
            self.bind_idf_provider(idf_provider)

    # ---- IDF provider（动态 IDF 接入点；不改 provider 语义） ------------- #
    def bind_idf_provider(self, provider: Optional[Any]) -> None:
        """绑定/解绑 IDF 提供者（`snapshot() -> 带 .model 的快照`）。`None` = 解绑。"""
        if provider is not None and not callable(getattr(provider, "snapshot", None)):
            raise TypeError("idf_provider must expose snapshot() -> IDFSnapshot, or be None")
        with self._lock:
            self._idf_provider = provider

    @property
    def idf_provider(self):
        with self._lock:
            return self._idf_provider

    # ---- IDF ------------------------------------------------------------ #
    def idf_for_ledger(
        self, ledger, *, stem_ok: bool = True, stop_mode: str = "v2", zh: bool = False
    ) -> IDFModel:
        """取账本代际 `len(ledger)` 对应的 IDF 模型；同代际同参数复用，跨代际重建。

        语义与 `skeleton.idf_for_ledger` 的历史实现**逐字节一致**（同样的 bucket 与
        `build_idf` 参数）；本方法只把缓存从模块全局搬到一个显式对象里。
        绑定 `idf_provider` 时直接返回其当前生效快照的 `model`（不落本地缓存）。
        """
        with self._lock:
            provider = self._idf_provider
        if provider is not None:
            snapshot = provider.snapshot()
            model = getattr(snapshot, "model", None)
            if model is None:
                raise TypeError("idf_provider.snapshot() must expose a .model IDFModel")
            return model

        key = (bool(stem_ok), str(stop_mode), bool(zh))
        generation = len(ledger)
        with self._lock:
            entry = self._idf.get(ledger)
            if entry is not None and entry[0] == generation:
                cached = entry[1]
                model = cached.get(key)
                if model is not None:
                    self._counters["idf_hits"] += 1
                    return model
            else:
                cached = {}
                self._idf[ledger] = (generation, cached)
            self._counters["idf_misses"] += 1
            # 在锁内构建：同一缓存实例的并发读者不会重复构建（代价是构建期间阻塞本实例
            # 的其他读者；单实例缓存语义下这是可接受且确定的）。构建本身是纯函数。
            model = build_idf(
                (norm_tokens(rec.content, stem_ok=stem_ok, stop_mode=stop_mode, zh=zh)
                 for rec in ledger),
                zh=zh,
            )
            cached[key] = model
            self._counters["idf_builds"] += 1
            return model

    # ---- 实体索引 -------------------------------------------------------- #
    def entities_for_ledger(self, ledger) -> dict[str, frozenset]:
        """`source_id -> 该 source 的参与者名字（小写）`；按 `len(ledger)` 代际缓存。"""
        generation = len(ledger)
        with self._lock:
            entry = self._entities.get(ledger)
            if entry is not None and entry[0] == generation:
                self._counters["entity_hits"] += 1
                return entry[1]
            self._counters["entity_misses"] += 1
            out: dict[str, set[str]] = {}
            for rec in ledger:
                spk = getattr(rec, "speaker", None)
                if spk:
                    out.setdefault(rec.source_id, set()).update(
                        w for w in _raw_words(spk) if w not in STOP
                    )
            frozen = {k: frozenset(v) for k, v in out.items()}
            self._entities[ledger] = (generation, frozen)
            self._counters["entity_builds"] += 1
            return frozen

    # ---- 生命周期 -------------------------------------------------------- #
    def clear(self) -> None:
        """丢弃全部派生缓存（不影响账本、不影响判定结果；下次读取原地重建）。

        绑定 `idf_provider` 时**不重置 provider**：其增量状态/版本历史由 provider 自己拥有
        （见 `docs/IDF-DYNAMIC-v1.md` §8）；本方法只清本层派生缓存。
        """
        with self._lock:
            self._idf = weakref.WeakKeyDictionary()
            self._entities = weakref.WeakKeyDictionary()
            self._counters["clears"] += 1

    def __len__(self) -> int:
        """已缓存 IDF 的账本数（诊断用）。"""
        with self._lock:
            return len(self._idf)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(**dict(self._counters))


# --------------------------------------------------------------------------- #
# 进程级兼容默认（仅给旧 API `skeleton.idf_for_ledger(ledger)` 用）
# --------------------------------------------------------------------------- #
# 新代码应显式持有 `EngineCache`（管线/多实例）；这里的单例只保证旧调用点"行为不变"，
# 同时仍然可 `clear()` / 可替换 / 可重置——不是一个藏起来的、无法丢弃的缓存。
_DEFAULT_CACHE: Optional[EngineCache] = None
_DEFAULT_LOCK = threading.Lock()


def default_engine_cache() -> EngineCache:
    """返回进程级兼容默认缓存（惰性创建）。只读使用；需要隔离请自建 `EngineCache()`。"""
    global _DEFAULT_CACHE
    with _DEFAULT_LOCK:
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = EngineCache()
        return _DEFAULT_CACHE


def set_default_engine_cache(cache: Optional[EngineCache]) -> None:
    """替换进程级默认缓存（测试/多租户切换用；`None` 等价于重置）。"""
    global _DEFAULT_CACHE
    if cache is not None and not isinstance(cache, EngineCache):
        raise TypeError(f"cache must be EngineCache or None, got {type(cache).__name__}")
    with _DEFAULT_LOCK:
        _DEFAULT_CACHE = cache


def reset_default_engine_cache() -> None:
    """丢弃并重置进程级默认缓存（下一次调用惰性重建）。"""
    set_default_engine_cache(None)


def clear_default_cache() -> None:
    """清空进程级默认缓存的派生数据（等价 `default_engine_cache().clear()`）。"""
    default_engine_cache().clear()
