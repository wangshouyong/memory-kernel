# SPDX-License-Identifier: Apache-2.0
"""主体/作用域抽象与存储端口单测（internal-task / internal-task，A1 接口阶段）。"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory_kernel.identity import (  # noqa: E402
    AuditEvent,
    Principal,
    Scope,
    WriteContext,
    allows,
)
from memory_kernel.ledger import AnchorLedger  # noqa: E402
from memory_kernel.storage import (  # noqa: E402
    AuditStore,
    InMemoryAuditStore,
    InMemoryRecordTimeStore,
    LedgerStore,
    LocalLedgerStore,
    RecordTimeStore,
)

KEY = b"0123456789abcdef0123456789abcdef"


class PrincipalScopeTest(unittest.TestCase):
    def test_principal_validation(self):
        with self.assertRaises(ValueError):
            Principal(kind="robot")
        with self.assertRaises(ValueError):
            Principal(tenant_id="")
        with self.assertRaises(ValueError):
            Principal(principal_id="")

    def test_scope_validation(self):
        with self.assertRaises(ValueError):
            Scope(visibility="public")
        with self.assertRaises(ValueError):
            Scope(namespace="")
        with self.assertRaises(ValueError):
            Scope(acl={"a": ("read", "exec")})

    def test_cross_tenant_denied(self):
        p = Principal(tenant_id="t1")
        s = Scope(tenant_id="t2", visibility="tenant")
        for action in ("read", "write", "delete", "audit"):
            self.assertFalse(allows(p, s, action))

    def test_private_scope_only_owner(self):
        owner = Principal(principal_id="owner")
        other = Principal(principal_id="agent-1")
        s = Scope(visibility="private", owner_principal_id="owner")
        self.assertTrue(allows(owner, s, "read"))
        self.assertFalse(allows(other, s, "read"))
        self.assertTrue(allows(owner, s, "write"))
        self.assertFalse(allows(other, s, "write"))

    def test_shared_scope_acl(self):
        agent = Principal(kind="agent", principal_id="agent-1")
        s = Scope(visibility="shared", owner_principal_id="owner",
                  acl={"agent-1": ("read", "write")})
        self.assertTrue(allows(agent, s, "read"))
        self.assertTrue(allows(agent, s, "write"))
        self.assertFalse(allows(agent, s, "delete"))  # 删除仅 owner

    def test_tenant_visibility_reads_for_all_members(self):
        member = Principal(principal_id="agent-2")
        s = Scope(visibility="tenant", owner_principal_id="owner")
        self.assertTrue(allows(member, s, "read"))
        self.assertFalse(allows(member, s, "write"))

    def test_write_context_and_audit_validation(self):
        with self.assertRaises(ValueError):
            WriteContext(principal=Principal(), trust="maybe")
        with self.assertRaises(ValueError):
            WriteContext(principal=Principal(), ts=-1)
        with self.assertRaises(ValueError):
            AuditEvent(kind="", principal=Principal(), scope=Scope())
        ev = AuditEvent(kind="deny", principal=Principal(), scope=Scope(), record_id="a_1")
        self.assertEqual(ev.record_id, "a_1")


class StoragePortTest(unittest.TestCase):
    def setUp(self):
        self.ledger = AnchorLedger(None, KEY)
        self.store = LocalLedgerStore(self.ledger)

    def test_ledger_store_protocol(self):
        self.assertIsInstance(self.store, LedgerStore)
        rec = self.store.append("hello", source_id="s", episode_id="e")
        self.assertEqual(self.store.get(rec.id), rec)
        self.assertEqual(len(self.store), 1)
        self.assertTrue(self.store.verify_chain()[0])
        self.assertEqual(list(self.store), [rec])

    def test_record_time_store_idempotent_and_conflict(self):
        ts = InMemoryRecordTimeStore()
        self.assertIsInstance(ts, RecordTimeStore)
        ts.record_time("a_1", 100.0)
        ts.record_time("a_1", 100.0)  # 幂等
        self.assertEqual(ts.get_time("a_1"), 100.0)
        with self.assertRaises(ValueError):
            ts.record_time("a_1", 101.0)
        with self.assertRaises(ValueError):
            ts.record_time("a_2", -1)

    def test_audit_store_is_append_only(self):
        audit = InMemoryAuditStore()
        self.assertIsInstance(audit, AuditStore)
        ev = AuditEvent(kind="write", principal=Principal(), scope=Scope())
        audit.append_event(ev)
        self.assertEqual(list(audit.events()), [ev])


if __name__ == "__main__":
    unittest.main()
