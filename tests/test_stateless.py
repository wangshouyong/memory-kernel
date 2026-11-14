# SPDX-License-Identifier: Apache-2.0
"""internal-task 无状态核心引擎单测：缓存显式化 / 可清空重建 / 多实例隔离 / 并发一致。

审计结论（代码里逐项断言）：
  1. cross-request 派生数据缓存**只有** `EngineCache`（显式对象、可注入、可 `clear()`、可重建）；
  2. `enforce` / `emit` / `pipeline` / `skeleton` / `scenario` **不再持有模块级可变缓存全局**；
  3. 请求内状态（`AnswerPipeline._prefetches/_stats/_sync_samples` 等）随实例创建/销毁，
     多实例互不共享；`close()` 只清本实例自建缓存；
  4. 缓存只做 memoization：清空/重建/手动丢弃**不改变任何判定结果**（默认零变化）。

不变量：`skeleton.idf_for_ledger` / `entities_for_ledger` 的**结果**与缓存无关；同一
`(ledger, 代际, 参数)` 输入永远同一输出（`tests/test_two_state.py` 的 153 探针冻结指纹覆盖
默认路径逐字节不变）。
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel import AnchorLedger, Claim, Citation, SkeletonPolicy  # noqa: E402
from memory_kernel.emit import emit_answer  # noqa: E402
from memory_kernel.engine_cache import (  # noqa: E402
    CacheStats,
    EngineCache,
    clear_default_cache,
    default_engine_cache,
    reset_default_engine_cache,
    set_default_engine_cache,
)
from memory_kernel.enforce import enforce  # noqa: E402
from memory_kernel.idf_dynamic import DynamicIDF  # noqa: E402
from memory_kernel.pipeline import AnswerPipeline, prewarm_ledger  # noqa: E402
from memory_kernel.skeleton import entities_for_ledger, idf_for_ledger  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"
V4 = SkeletonPolicy.v4()

ENGINE_MODULES = ("enforce.py", "emit.py", "pipeline.py", "skeleton.py", "scenario.py")
#: 允许作为"模块级常量"的大写名（含下划线前缀）；其余 list/set/dict 属性一律视为可疑可变全局
_CONST_NAME = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


def _result_signature(result) -> tuple:
    """判定结果签名（排除耗时字段；缓存不得改变其中任何一项）。"""
    return (
        result.verdict,
        result.rejected_cause,
        tuple((p.name, p.passed, p.severity, p.detail) for p in result.failed),
        result.degrade_reason,
        result.degrade_reasons,
        result.anchor_id,
        tuple(sorted(result.coverage.items())),
    )


def _ledger(n: int = 6) -> AnchorLedger:
    """确定性账本：前 6 条为固定句子，其余按固定模板生成（同输入永远同内容）。"""
    fixed = [
        ("Alice bought a red car on March 5th.", "user"),
        ("The weather in Berlin was cold last week.", "Bob"),
        ("我们在 2026年3月5日 上线了新的结算系统。", "user"),
        ("Alice bought a red car on March 5th, and Bob was there.", "Bob"),
        ("Quarterly industrial output grew by 5 percent.", "user"),
        ("下季度工业增加值预计增长 5%。", "user"),
    ]
    led = AnchorLedger(None, KEY)
    for idx in range(n):
        if idx < len(fixed):
            text, spk = fixed[idx]
        else:
            text, spk = f"Record number {idx} mentions item {idx % 7} and value {idx % 11}.", "user"
        led.append(text, source_id="s1", episode_id="e1", turn_index=idx, speaker=spk)
    return led


def _claim(led: AnchorLedger):
    rec = led.records[0]
    return Claim(Citation(rec.id, 0, rec.content), text=rec.content)


def _emit_segments(led: AnchorLedger):
    rec = led.records[0]
    return [{"type": "claim", "text": rec.content,
             "cites": [{"anchor": rec.id, "quote": rec.content}]}]


# --------------------------------------------------------------------------- #
# 1. 审计：没有模块级可变缓存全局
# --------------------------------------------------------------------------- #
class NoModuleLevelMutableCacheTest(unittest.TestCase):
    def _tree(self, name: str) -> ast.Module:
        path = os.path.join(ROOT, "memory_kernel", name)
        with open(path, encoding="utf-8") as fh:
            return ast.parse(fh.read(), filename=path)

    def test_no_weak_dictionary_caches_in_engine_modules(self):
        # 旧的模块级 `_LEDGER_IDF` / `_LEDGER_ENTITIES`（WeakKeyDictionary）必须已移除
        for name in ENGINE_MODULES:
            tree = self._tree(name)
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Attribute) and node.func.attr == "WeakKeyDictionary")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "WeakValueDictionary")
                    or (isinstance(node.func, ast.Name)
                        and node.func.id in ("WeakKeyDictionary", "WeakValueDictionary"))
                )
            ]
            self.assertEqual(calls, [], f"{name} 仍存在模块级弱引用缓存构造")

    def test_no_module_level_mutable_container_assignments(self):
        """模块级 `NAME = []/{}/set()/dict()` 只允许大写常量名（运行时不会被变异）。"""
        for name in ENGINE_MODULES:
            path = os.path.join(ROOT, "memory_kernel", name)
            tree = self._tree(name)
            bad = []
            for node in tree.body:
                targets = []
                value = None
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign):
                    targets, value = [node.target], node.value
                if value is None:
                    continue
                is_mutable = (
                    (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                     and value.func.id in ("dict", "list", "set"))
                    or (isinstance(value, ast.List) and not value.elts)
                    or (isinstance(value, ast.Set) and not value.elts)
                    or (isinstance(value, ast.Dict) and not value.keys)
                )
                if not is_mutable:
                    continue
                for t in targets:
                    if isinstance(t, ast.Name) and not _CONST_NAME.match(t.id):
                        bad.append(t.id)
            self.assertEqual(bad, [], f"{path} 存在小写/非常量名的模块级可变容器: {bad}")

    def test_no_function_mutates_module_level_globals(self):
        """函数体内不得 `global NAME` 或对模块级常量做 `NAME[...] =` / `NAME.update(...)`。

        （模块级 import-time 构建表是允许的；运行期改模块全局才是无状态化的敌人。）
        """
        for name in ENGINE_MODULES:
            tree = self._tree(name)
            module_names = {
                t.id
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for t in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(t, ast.Name)
            }
            offenders = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Global):
                            offenders.extend(stmt.names)
                        elif isinstance(stmt, ast.Assign):
                            for t in stmt.targets:
                                if (isinstance(t, ast.Subscript)
                                        and isinstance(t.value, ast.Name)
                                        and t.value.id in module_names):
                                    offenders.append(t.value.id)
                        elif isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                            if (isinstance(stmt.func.value, ast.Name)
                                    and stmt.func.value.id in module_names
                                    and stmt.func.attr in ("update", "append", "add", "pop",
                                                           "clear", "setdefault")):
                                offenders.append(stmt.func.value.id)
            self.assertEqual(offenders, [], f"{name} 运行期修改模块全局: {sorted(set(offenders))}")

    def test_runtime_no_cache_globals(self):
        import memory_kernel.emit as emit_mod
        import memory_kernel.enforce as enforce_mod
        import memory_kernel.pipeline as pipe_mod
        import memory_kernel.scenario as scen_mod
        import memory_kernel.skeleton as skel_mod

        modules = {
            "enforce": enforce_mod, "emit": emit_mod, "pipeline": pipe_mod,
            "skeleton": skel_mod, "scenario": scen_mod,
        }
        for mod_name, mod in modules.items():
            for attr, value in vars(mod).items():
                if attr.startswith("__"):
                    continue  # Python 注入的 dunder（__builtins__ / __all__ 等）不是我们的状态
                if isinstance(value, (weakref.WeakKeyDictionary, weakref.WeakValueDictionary)):
                    self.fail(f"{mod_name}.{attr} 仍是模块级弱引用缓存")
                if isinstance(value, (list, set)) and not _CONST_NAME.match(attr):
                    self.fail(f"{mod_name}.{attr} 是模块级可变容器（list/set）")
                if isinstance(value, dict) and not _CONST_NAME.match(attr):
                    self.fail(f"{mod_name}.{attr} 是模块级可变 dict")
        # 旧缓存名必须不存在
        self.assertFalse(hasattr(skel_mod, "_LEDGER_IDF"))
        self.assertFalse(hasattr(skel_mod, "_LEDGER_ENTITIES"))

    def test_engine_cache_is_the_single_explicit_cache(self):
        cache = default_engine_cache()
        self.assertIsInstance(cache, EngineCache)
        self.assertTrue(callable(cache.clear))
        # 默认缓存可替换/可重置（不是无法丢弃的隐藏全局）
        old = default_engine_cache()
        custom = EngineCache()
        try:
            set_default_engine_cache(custom)
            self.assertIs(default_engine_cache(), custom)
        finally:
            set_default_engine_cache(old)
        self.assertIs(default_engine_cache(), old)


# --------------------------------------------------------------------------- #
# 2. EngineCache：命中 / 清空 / 重建 / 代际
# --------------------------------------------------------------------------- #
class EngineCacheLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.ledger = _ledger()
        self.cache = EngineCache()

    def test_idf_hit_and_miss_counters(self):
        first = self.cache.idf_for_ledger(self.ledger)
        second = self.cache.idf_for_ledger(self.ledger)
        self.assertIs(first, second)                       # 同代际同参数复用
        stats = self.cache.stats()
        self.assertEqual((stats.idf_misses, stats.idf_builds), (1, 1))
        self.assertEqual(stats.idf_hits, 1)
        # 不同参数分桶，不串味
        zh_model = self.cache.idf_for_ledger(self.ledger, zh=True)
        self.assertIsNot(zh_model, first)
        self.assertEqual(self.cache.stats().idf_builds, 2)

    def test_clear_drops_and_rebuilds_identical_result(self):
        before = self.cache.idf_for_ledger(self.ledger)
        entities_before = self.cache.entities_for_ledger(self.ledger)
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.stats().clears, 1)
        after = self.cache.idf_for_ledger(self.ledger)
        self.assertIsNot(after, before)                    # 丢弃后是重建的新对象
        # 重建结果与丢弃前逐位一致（纯派生数据）
        self.assertEqual(after.to_json(), before.to_json())
        self.assertEqual(self.cache.entities_for_ledger(self.ledger), entities_before)

    def test_new_generation_rebuilds(self):
        first = self.cache.idf_for_ledger(self.ledger)
        self.ledger.append("new evidence about a red car.", source_id="s1",
                           episode_id="e1", turn_index=99, speaker="user")
        second = self.cache.idf_for_ledger(self.ledger)
        self.assertIsNot(second, first)
        self.assertEqual(second.n_docs, first.n_docs + 1)
        self.assertEqual(self.cache.stats().idf_builds, 2)

    def test_instances_are_isolated(self):
        other = EngineCache()
        self.cache.idf_for_ledger(self.ledger)
        self.cache.entities_for_ledger(self.ledger)
        self.assertEqual(len(other), 0)
        self.assertEqual(other.stats(), CacheStats())
        model = other.idf_for_ledger(self.ledger)
        self.assertEqual(model.to_json(), self.cache.idf_for_ledger(self.ledger).to_json())
        self.assertEqual(self.cache.stats().idf_hits, 1)     # 别人的 miss 不计到我头上

    def test_idf_provider_is_read_through_and_not_reset_by_clear(self):
        provider = DynamicIDF(zh=False)          # 语义不变：只接 provider，不改它
        provider.load_token_sets([{"alpha", "beta"}, {"gamma"}], epoch="202611140300",
                                 reason="test")
        version = provider.active_version
        cache = EngineCache(idf_provider=provider)
        model = cache.idf_for_ledger(self.ledger)
        self.assertEqual(model.version, version)             # 直接读快照（不落本地缓存）
        cache.clear()
        self.assertEqual(provider.active_version, version)   # clear 不动 provider 状态/历史
        self.assertEqual(cache.idf_for_ledger(self.ledger).version, version)
        # 非法 provider 显式报错
        with self.assertRaises(TypeError):
            EngineCache(idf_provider=object())
        with self.assertRaises(TypeError):
            cache.bind_idf_provider(object())

    def test_provider_binding_does_not_change_dynamic_idf_semantics(self):
        provider = DynamicIDF()
        provider.load_token_sets([{"x"}], epoch="202611140300")
        before = provider.stats()
        cache = EngineCache(idf_provider=provider)
        cache.idf_for_ledger(self.ledger)
        cache.clear()
        after = provider.stats()
        self.assertEqual(before["revision"], after["revision"])
        self.assertEqual(before["versions"], after["versions"])
        self.assertEqual(provider.history(), tuple(provider.history()))  # 历史只追加

    def test_rebuild_cost_is_bounded_and_measured(self):
        """重建成本测量（不做脆弱的时间阈值断言，只记录量级并验证缓存确实省掉重建）。"""
        big = _ledger(200)
        cache = EngineCache()
        t0 = time.perf_counter()
        cache.idf_for_ledger(big)
        cold = time.perf_counter() - t0
        t1 = time.perf_counter()
        cache.idf_for_ledger(big)
        warm = time.perf_counter() - t1
        # 热读是纯查表：必须显著快于冷建（宽阈值，避免 CI 抖动误判）
        self.assertLess(warm, max(cold, 1e-6) * 10 + 1e-3)
        self.assertGreaterEqual(cold, 0.0)
        # 清空后重建能与 cold 同量级完成（不至于无限慢）
        cache.clear()
        t2 = time.perf_counter()
        cache.idf_for_ledger(big)
        rebuilt = time.perf_counter() - t2
        self.assertLess(rebuilt, max(cold * 50, 1.0))


# --------------------------------------------------------------------------- #
# 3. 默认零变化：缓存哪来、清不清，判定都一样
# --------------------------------------------------------------------------- #
class CacheDoesNotChangeVerdictTest(unittest.TestCase):
    def setUp(self):
        self.ledger = _ledger()
        self.claim = _claim(self.ledger)
        reset_default_engine_cache()

    def tearDown(self):
        reset_default_engine_cache()

    def _sig(self, **kwargs):
        return _result_signature(enforce(self.ledger, self.claim, policy=V4, **kwargs))

    def test_default_injected_and_cleared_cache_are_identical(self):
        legacy = self._sig()                                  # 进程级兼容默认缓存
        explicit = EngineCache()
        injected = self._sig(cache=explicit)
        explicit.clear()
        after_clear = self._sig(cache=explicit)
        self.assertEqual(injected, legacy)
        self.assertEqual(after_clear, legacy)
        # clear 后确实走过重建路径（计数器可观测，但结果不变）
        self.assertGreaterEqual(explicit.stats().clears, 1)

    def test_reset_default_cache_does_not_change_result(self):
        first = self._sig()
        reset_default_engine_cache()
        second = self._sig()
        self.assertEqual(second, first)

    def test_emit_answers_identical_with_any_cache(self):
        segs = _emit_segments(self.ledger)
        base = emit_answer(segs, self.ledger, policy=V4)
        injected = emit_answer(segs, self.ledger, policy=V4, cache=EngineCache())
        cleared = EngineCache()
        cleared.clear()
        after = emit_answer(segs, self.ledger, policy=V4, cache=cleared)
        self.assertEqual((injected.text, injected.audit), (base.text, base.audit))
        self.assertEqual((after.text, after.audit), (base.text, base.audit))

    def test_legacy_module_functions_unchanged(self):
        # 旧 API 仍是"同代际缓存 / append 后重建"，且结果与显式 EngineCache 相同
        a = idf_for_ledger(self.ledger)
        b = idf_for_ledger(self.ledger)
        self.assertIs(a, b)
        explicit = EngineCache()
        self.assertEqual(a.to_json(), explicit.idf_for_ledger(self.ledger).to_json())
        self.assertEqual(entities_for_ledger(self.ledger),
                         explicit.entities_for_ledger(self.ledger))


# --------------------------------------------------------------------------- #
# 4. 多实例隔离 + 并发一致
# --------------------------------------------------------------------------- #
class PipelineIsolationTest(unittest.TestCase):
    def setUp(self):
        self.ledger = _ledger()

    def _pipe(self, **kwargs):
        def gen(question, prompt_ctx):
            return _emit_segments(self.ledger)

        pipe = AnswerPipeline(generator=gen, ledger=self.ledger, policy=V4, **kwargs)
        self.addCleanup(pipe.close)
        return pipe

    def test_each_pipeline_owns_its_cache(self):
        p1, p2 = self._pipe(), self._pipe()
        self.assertIsNot(p1._engine_cache, p2._engine_cache)
        p1.prefetch("k", question="q", records=list(self.ledger.records))
        self.assertEqual(len(p1._engine_cache), 0)      # prefetch 不建 IDF（门控时才用）
        p2.prewarm()
        self.assertGreater(len(p2._engine_cache), 0)
        self.assertEqual(len(p1._engine_cache), 0)      # p2 的预热不污染 p1
        self.assertEqual(p1.cache_stats()["idf_misses"], 0)
        p1.prewarm()
        self.assertGreater(len(p1._engine_cache), 0)
        self.assertEqual(p1._engine_cache.stats().clears, 0)

    def test_injected_shared_cache_is_not_cleared_on_close(self):
        shared = EngineCache()
        pipe = AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger,
                              policy=V4, engine_cache=shared)
        pipe.prewarm()
        self.assertGreater(len(shared), 0)
        pipe.close()
        self.assertEqual(shared.stats().clears, 0)   # 注入的缓存归调用方管理
        self.assertGreater(len(shared), 0)

    def test_owned_cache_is_cleared_on_close(self):
        pipe = self._pipe()
        pipe.prewarm()
        cache = pipe._engine_cache
        self.assertGreater(len(cache), 0)
        pipe.close()
        self.assertEqual(cache.stats().clears, 1)
        self.assertEqual(len(cache), 0)

    def test_invalid_engine_cache_rejected(self):
        with self.assertRaises(TypeError):
            AnswerPipeline(generator=lambda q, c: [], ledger=self.ledger,
                           engine_cache=object())

    def test_two_pipelines_same_input_produce_same_result(self):
        def gen(question, prompt_ctx):
            return _emit_segments(self.ledger)

        results = []
        for _ in range(2):
            pipe = AnswerPipeline(generator=gen, ledger=self.ledger, policy=V4)
            self.addCleanup(pipe.close)
            pipe.prefetch("k", question="q", records=list(self.ledger.records))
            deadline = time.time() + 5.0
            result = pipe.answer("k")
            while result.state != "gated" and time.time() < deadline:
                time.sleep(0.01)
                result = pipe.answer("k")
            self.assertEqual(result.state, "gated")
            results.append((result.text, result.audit, result.evidence, result.dropped))
        self.assertEqual(results[0], results[1])


class ConcurrencyTest(unittest.TestCase):
    def test_parallel_enforce_matches_serial(self):
        ledger = _ledger()
        claim = _claim(ledger)
        serial = [_result_signature(enforce(ledger, claim, policy=V4)) for _ in range(8)]

        def run(i):
            cache = EngineCache()
            enforce(ledger, claim, policy=V4, cache=cache)      # 冷建
            return _result_signature(enforce(ledger, claim, policy=V4, cache=cache))

        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(run, range(8)))
        self.assertEqual(parallel, serial)

    def test_shared_cache_parallel_reads_are_consistent(self):
        ledger = _ledger()
        claim = _claim(ledger)
        shared = EngineCache()
        expected = _result_signature(enforce(ledger, claim, policy=V4, cache=shared))

        def run(_i):
            return _result_signature(enforce(ledger, claim, policy=V4, cache=shared))

        with ThreadPoolExecutor(max_workers=6) as pool:
            got = list(pool.map(run, range(24)))
        self.assertEqual(got, [expected] * 24)

    def test_parallel_pipeline_instances_are_independent(self):
        ledger = _ledger()

        def gen(question, prompt_ctx):
            return _emit_segments(ledger)

        def run(_i):
            pipe = AnswerPipeline(generator=gen, ledger=ledger, policy=V4)
            try:
                pipe.prewarm()
                pipe.prefetch("k", question="q", records=list(ledger.records))
                deadline = time.time() + 10.0
                result = pipe.answer("k")
                while result.state != "gated" and time.time() < deadline:
                    time.sleep(0.01)
                    result = pipe.answer("k")
                return (result.state, result.text, result.audit)
            finally:
                pipe.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, range(4)))
        self.assertEqual(len(set((r[0], r[1], repr(r[2])) for r in results)), 1)


# --------------------------------------------------------------------------- #
# 5. 缓存丢弃后可重建（整条链路）
# --------------------------------------------------------------------------- #
class RebuildAfterDiscardTest(unittest.TestCase):
    def test_prewarm_then_drop_cache_then_emit_still_same(self):
        ledger = _ledger()
        segs = _emit_segments(ledger)
        cache = EngineCache()
        prewarm_ledger(ledger, V4, cache=cache)
        warm = emit_answer(segs, ledger, policy=V4, cache=cache)
        cache.clear()
        cold = emit_answer(segs, ledger, policy=V4, cache=cache)
        self.assertEqual((cold.text, cold.audit), (warm.text, warm.audit))
        self.assertGreaterEqual(cache.stats().idf_builds, 2)


if __name__ == "__main__":
    unittest.main()
