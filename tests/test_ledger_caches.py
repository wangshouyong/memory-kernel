# SPDX-License-Identifier: Apache-2.0
"""账本级缓存代际单测（internal-task：append 后 IDF / 实体缓存必须失效重建）。"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel import AnchorLedger  # noqa: E402
from memory_kernel.skeleton import entities_for_ledger, idf_for_ledger  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"


class LedgerCacheGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AnchorLedger(None, KEY)

    def test_idf_rebuilds_after_append(self):
        self.ledger.append("alpha beta gamma", source_id="s1", episode_id="e1",
                           turn_index=0, speaker="user")
        first = idf_for_ledger(self.ledger)
        self.assertEqual(first.n_docs, 1)
        w_alpha_before = first.weight("alpha")

        self.ledger.append("delta epsilon", source_id="s1", episode_id="e1",
                           turn_index=1, speaker="assistant")
        second = idf_for_ledger(self.ledger)
        self.assertEqual(second.n_docs, 2)                       # n 随账本增长
        self.assertNotEqual(second.weight("alpha"), w_alpha_before)  # df/n 变了 → 权重重算
        self.assertIsNot(second, first)                          # 旧模型不得复用

    def test_idf_same_generation_is_cached(self):
        self.ledger.append("alpha beta", source_id="s1", episode_id="e1",
                           turn_index=0, speaker="user")
        a = idf_for_ledger(self.ledger)
        b = idf_for_ledger(self.ledger)
        self.assertIs(a, b)                                      # 同代际内复用

    def test_entities_rebuild_after_append(self):
        self.ledger.append("hello", source_id="s1", episode_id="e1",
                           turn_index=0, speaker="Alice")
        before = entities_for_ledger(self.ledger)
        self.assertEqual(before["s1"], frozenset({"alice"}))

        self.ledger.append("world", source_id="s1", episode_id="e1",
                           turn_index=1, speaker="Bob")
        after = entities_for_ledger(self.ledger)
        self.assertEqual(after["s1"], frozenset({"alice", "bob"}))

    def test_independent_ledgers_do_not_share_cache(self):
        other = AnchorLedger(None, KEY)
        self.ledger.append("one", source_id="a", episode_id="e", turn_index=0, speaker="Ann")
        self.ledger.append("two", source_id="a", episode_id="e", turn_index=1, speaker="Ann")
        other.append("three", source_id="b", episode_id="e", turn_index=0, speaker="Ben")
        self.assertEqual(idf_for_ledger(self.ledger).n_docs, 2)
        self.assertEqual(idf_for_ledger(other).n_docs, 1)
        self.assertEqual(entities_for_ledger(self.ledger)["a"], frozenset({"ann"}))
        self.assertEqual(entities_for_ledger(other)["b"], frozenset({"ben"}))


if __name__ == "__main__":
    unittest.main()
