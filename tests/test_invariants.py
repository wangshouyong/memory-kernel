# SPDX-License-Identifier: Apache-2.0
"""T0 不可逆决策的回归测试。

这五条一旦破坏，历史记录全部失效，所以必须有测试守着，而不是靠文档约定。
运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import AnchorLedger, RecordVerificationError
from memory_kernel.canon import (
    DOMAIN_PAYLOAD,
    DOMAIN_SIG,
    digest,
    frame,
    normalize,
    sign,
)

KEY = hashlib.sha256(b"test-key").digest()


class FramingTest(unittest.TestCase):
    def test_length_prefix_prevents_separator_injection(self):
        # 字符串拼接 (f"{a}|{b}") 会让这两组产生同一个串；长度前缀不会
        self.assertNotEqual(frame("a|b", "c"), frame("a", "b|c"))
        self.assertNotEqual(frame("ab", "c"), frame("a", "bc"))

    def test_field_order_matters(self):
        self.assertNotEqual(frame("a", "b"), frame("b", "a"))


class DomainSeparationTest(unittest.TestCase):
    def test_same_payload_different_domain_different_signature(self):
        a = sign(KEY, DOMAIN_SIG, "payload")
        b = sign(KEY, DOMAIN_PAYLOAD, "payload")
        self.assertNotEqual(a, b)

    def test_domain_prefix_prevents_cross_protocol_replay(self):
        # 把 domain 当普通字段混进 payload，不能与真正的 domain 等价
        self.assertNotEqual(digest(DOMAIN_SIG, "x"), digest(DOMAIN_PAYLOAD, "x"))


class NormalizationTest(unittest.TestCase):
    def test_nfc_makes_equivalent_forms_hash_equal(self):
        composed = "caf\u00e9"          # café
        decomposed = "cafe\u0301"       # cafe + combining acute
        self.assertEqual(normalize(composed), normalize(decomposed))
        self.assertEqual(
            hashlib.sha256(normalize(composed).encode()).hexdigest(),
            hashlib.sha256(normalize(decomposed).encode()).hexdigest(),
        )


class SignatureCoverageTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.rec = self.ledger.append(
            "I went to the support group yesterday.",
            source_id="s0",
            episode_id="session_1",
            turn_index=0,
            speaker="Caroline",
        )

    def test_metadata_tampering_is_detected(self):
        tampered = replace(self.rec, speaker="Melanie")
        with self.assertRaises(RecordVerificationError):
            self.ledger._verify(tampered)

    def test_content_tampering_is_detected(self):
        tampered = replace(self.rec, content=self.rec.content + " (edited)")
        with self.assertRaises(RecordVerificationError):
            self.ledger._verify(tampered)

    def test_untampered_record_passes(self):
        self.assertEqual(self.ledger.get(self.rec.id).id, self.rec.id)

    def test_chain_link_is_committed(self):
        second = self.ledger.append("Second turn.", source_id="s0", episode_id="session_1", turn_index=1)
        self.assertEqual(second.prev_hash, self.rec.record_hash)
        ok, broken = self.ledger.verify_chain()
        self.assertTrue(ok)
        self.assertIsNone(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
