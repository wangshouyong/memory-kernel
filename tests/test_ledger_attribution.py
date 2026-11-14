# SPDX-License-Identifier: Apache-2.0
"""A1 归属槽位单测（internal-task）：默认行为逐字节不变 + 归属字段进 payload + 校验框架。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel.ledger import AnchorLedger, record_to_json  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"

# A1 之前（commit d4e798c 之前）用同一 fixture 实测得到的金标：默认记录不得变化。
GOLDEN_RECORD_HASH = "f3aff090068f9469f3bb40f6d9990de1b149b293de7a558c3c6d44ace5dce69f"
GOLDEN_SIGNATURE = "7cbb19487b786fa67a26a8a6790c2166e080eeab0fb6f4a97084b1b0fe878ada"
GOLDEN_JSON = (
    '{"char_span":[0,16],"content":"alpha beta gamma",'
    '"content_hash":"64989ccbf3efa9c84e2afe7cee9bc5828bf0fcb91e44f8c1e591638a2c2e90e3",'
    '"episode_id":"e1","id":"a_000000","modality":"text",'
    '"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000",'
    '"record_hash":"f3aff090068f9469f3bb40f6d9990de1b149b293de7a558c3c6d44ace5dce69f",'
    '"seq":0,"signature":"7cbb19487b786fa67a26a8a6790c2166e080eeab0fb6f4a97084b1b0fe878ada",'
    '"source_id":"s1","speaker":"Alice","turn_index":2}'
)


def _ledger() -> AnchorLedger:
    return AnchorLedger(None, KEY)


class DefaultUnchangedTest(unittest.TestCase):
    def test_default_record_hash_and_signature_unchanged(self):
        rec = _ledger().append("alpha beta gamma", source_id="s1", episode_id="e1",
                               turn_index=2, speaker="Alice")
        self.assertEqual(rec.writer_id, None)
        self.assertFalse(rec.attributed)
        self.assertEqual(rec.record_hash, GOLDEN_RECORD_HASH)
        self.assertEqual(rec.signature, GOLDEN_SIGNATURE)

    def test_default_json_serialization_unchanged(self):
        rec = _ledger().append("alpha beta gamma", source_id="s1", episode_id="e1",
                               turn_index=2, speaker="Alice")
        got = json.dumps(record_to_json(rec), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
        self.assertEqual(got, GOLDEN_JSON)


class AttributionTest(unittest.TestCase):
    def test_attribution_fields_change_payload_hash(self):
        a = _ledger().append("alpha beta gamma", source_id="s1", episode_id="e1",
                             turn_index=2, speaker="Alice")
        b = _ledger().append("alpha beta gamma", source_id="s1", episode_id="e1",
                             turn_index=2, speaker="Alice",
                             writer_id="agent-1", key_epoch=0, writer_seq=7,
                             submission_hash="deadbeef")
        self.assertNotEqual(a.record_hash, b.record_hash)  # 归属进 payload
        self.assertTrue(b.attributed)
        self.assertEqual(json.loads(json.dumps(record_to_json(b)))["writer_id"], "agent-1")

    def test_different_writer_different_hash(self):
        c = _ledger().append("x", source_id="s", episode_id="e",
                             writer_id="agent-1", writer_seq=1, submission_hash="h1")
        d = _ledger().append("x", source_id="s", episode_id="e",
                             writer_id="agent-2", writer_seq=1, submission_hash="h1")
        self.assertNotEqual(c.record_hash, d.record_hash)

    def test_half_attribution_rejected(self):
        with self.assertRaises(ValueError):
            _ledger().append("x", source_id="s", episode_id="e", writer_id="agent-1")
        with self.assertRaises(ValueError):
            _ledger().append("x", source_id="s", episode_id="e", writer_seq=1)
        with self.assertRaises(ValueError):
            _ledger().append("x", source_id="s", episode_id="e", key_epoch=3)

    def test_roundtrip_persists_attribution(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            led = AnchorLedger(path, KEY)
            led.append("hello", source_id="s", episode_id="e",
                       writer_id="agent-9", key_epoch=2, writer_seq=4, submission_hash="abc")
            again = AnchorLedger(path, KEY)
            rec = again.get("a_000000")
            self.assertEqual(
                (rec.writer_id, rec.key_epoch, rec.writer_seq, rec.submission_hash),
                ("agent-9", 2, 4, "abc"),
            )
            self.assertTrue(again.verify_chain()[0])

    def test_old_format_record_still_loads_and_verifies(self):
        # A1 之前的 JSON（无归属字段）必须原样可读可验（用金标那一行）
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            with open(path, "wb") as fh:
                fh.write(GOLDEN_JSON.encode("utf-8") + b"\n")
            led = AnchorLedger(path, KEY)
            rec = led.get("a_000000")
            self.assertEqual(rec.writer_id, None)
            self.assertEqual(rec.record_hash, GOLDEN_RECORD_HASH)
            self.assertTrue(led.verify_chain()[0])


if __name__ == "__main__":
    unittest.main()
