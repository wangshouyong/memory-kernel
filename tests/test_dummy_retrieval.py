# SPDX-License-Identifier: Apache-2.0
"""参考检索器单测（examples/dummy_retrieval）：接口、确定性、免责声明。"""

from __future__ import annotations

import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "dummy_retrieval")
sys.path.insert(0, EXAMPLE)

from dummy_retriever import BaseRetriever, DummyRetriever, char_ngrams, tokenize  # noqa: E402

DOCS = [
    ("a", "示例文档一：用于演示的检索样例，先做需求梳理。"),
    ("b", "示例文档二：记录一次虚构的周末出行，目的地是示例公园。"),
    ("c", "示例文档四：示例地区的天气不错，适合户外活动。"),
    ("d", "示例文档三：示例项目决定下个季度按月更新指标。"),
]


class InterfaceTest(unittest.TestCase):
    def test_base_retriever_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseRetriever()  # type: ignore[abstract]

    def test_dummy_is_a_base_retriever(self):
        self.assertIsInstance(DummyRetriever(), BaseRetriever)

    def test_stdlib_only(self):
        path = os.path.join(EXAMPLE, "dummy_retriever.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        stdlib = set(sys.stdlib_module_names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], stdlib)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                self.assertIn(node.module.split(".")[0], stdlib)


class TokenizeTest(unittest.TestCase):
    def test_cjk_bigram_and_ascii(self):
        toks = tokenize("工程机械 BM25 平台")
        self.assertIn("工程", toks)
        self.assertIn("bm25", toks)

    def test_char_ngrams(self):
        grams = char_ngrams("abcd")
        self.assertIn("ab", grams)
        self.assertIn("abc", grams)
        self.assertIn("abcd", grams)

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(char_ngrams(""), [])


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.r = DummyRetriever()
        self.r.index(DOCS)

    def test_exact_match_first(self):
        hits = self.r.search("示例文档一 演示检索样例", k=3)
        self.assertEqual(hits[0], "a")

    def test_respects_limit(self):
        self.assertLessEqual(len(self.r.search("示例文档", k=2)), 2)

    def test_deterministic(self):
        q = "示例项目 指标 月度更新"
        first = self.r.search(q, k=4)
        for _ in range(20):
            self.assertEqual(self.r.search(q, k=4), first)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.r.search("zzzzyyyyxxxx", k=3), [])

    def test_empty_index(self):
        r = DummyRetriever()
        self.assertEqual(r.search("anything", k=3), [])


class DisclaimerTest(unittest.TestCase):
    def test_header_has_iron_warning(self):
        with open(os.path.join(EXAMPLE, "dummy_retriever.py"), encoding="utf-8") as fh:
            head = fh.read(1200)
        self.assertIn("DO NOT USE IN PRODUCTION", head)
        self.assertIn("reference retriever", head)
        self.assertIn("LlamaIndex", head)

    def test_readme_has_swap_and_disclaimer(self):
        with open(os.path.join(EXAMPLE, "README.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("BaseRetriever", text)
        self.assertIn("Milvus", text)
        self.assertIn("wontfix", text)


if __name__ == "__main__":
    unittest.main()
