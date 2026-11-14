# SPDX-License-Identifier: Apache-2.0
"""开源隔离门禁：核心 `memory_kernel/` 必须零私有依赖（AST 级）。

规则（docs/internal-doc.md §2）：
1. 只允许标准库 import（相对导入允许，但必须落在包内）；
2. 禁止网络/模型/评测/数据相关 import（urllib/http/socket/requests/onnxruntime/numpy/torch/openai…）；
3. 源码里不得出现私有路径、绝对路径、URL、内部脚本名（字符串字面量级检查）；
4. 不得 import `eval` / `data` / `scripts` 等仓库非核心顶层包。

本测试只读源码，不联网、不依赖私有数据，可在 CI 中运行。
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "memory_kernel")

ALLOWED_INTERNAL_TOPS = {"memory_kernel"}
FORBIDDEN_TOPS = {
    "urllib", "http", "socket", "requests", "httpx", "aiohttp",
    "onnxruntime", "numpy", "torch", "transformers", "openai", "anthropic",
    "eval", "data", "scripts",
}
FORBIDDEN_SUBSTRINGS = (
    "data" "/" "private", "data" "\\" "private",
    "C:" "\\", "C:" "/", "/home" "/", "/Users" "/",
    "http" "://", "https" "://",
    "probe_" "scenario_ab", "retrieval_" "hybrid.py", "llm_" "in_loop.py",
)


def _core_files() -> list[str]:
    return sorted(
        os.path.join(CORE, f) for f in os.listdir(CORE)
        if f.endswith(".py")
    )


def _imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append(node.module)
    return out


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class CoreIsolationTest(unittest.TestCase):
    def test_core_has_files(self):
        self.assertGreaterEqual(len(_core_files()), 10)

    def test_imports_are_stdlib_or_internal(self):
        stdlib = set(sys.stdlib_module_names)
        bad: list[str] = []
        for path in _core_files():
            tree = ast.parse(_read(path), filename=path)
            for mod in _imports(tree):
                top = mod.split(".")[0]
                if top in ALLOWED_INTERNAL_TOPS or top in stdlib:
                    continue
                bad.append(f"{os.path.basename(path)}: {mod}")
        self.assertEqual(bad, [], "核心出现非标准库/非包内导入: " + ", ".join(bad))

    def test_no_forbidden_tops(self):
        bad: list[str] = []
        for path in _core_files():
            tree = ast.parse(_read(path), filename=path)
            for mod in _imports(tree):
                top = mod.split(".")[0]
                if top in FORBIDDEN_TOPS:
                    bad.append(f"{os.path.basename(path)}: {mod}")
        self.assertEqual(bad, [], "核心出现禁用依赖: " + ", ".join(bad))

    def test_no_private_paths_or_internal_script_names(self):
        bad: list[str] = []
        for path in _core_files():
            src = _read(path)
            for needle in FORBIDDEN_SUBSTRINGS:
                if needle in src:
                    bad.append(f"{os.path.basename(path)}: {needle!r}")
        self.assertEqual(bad, [], "核心出现私有路径/URL/内部脚本名: " + ", ".join(bad))


if __name__ == "__main__":
    unittest.main()
