# SPDX-License-Identifier: Apache-2.0
"""主体 / 作用域统一抽象（internal-task；契约 `docs/PRINCIPAL-SCOPE-DESIGN.md`）。

设计目标：**多租户与多 Agent 共用一套隔离底座**。
- 多租户只用 `tenant_id` 这一维；
- 多 Agent 在同一租户内再加 `principal_id` 与共享命名空间/ACL。

本模块只做**接口与校验**（A1 阶段），不接存储、不做鉴权服务：
ACL 判定是纯函数，供存储/服务入口调用；**绝不进入执法层**（公理 B）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional

__all__ = [
    "PRINCIPAL_KINDS",
    "VISIBILITIES",
    "TRUST_LEVELS",
    "ACTIONS",
    "Principal",
    "Scope",
    "WriteContext",
    "AuditEvent",
    "allows",
    "LOCAL_TENANT",
]

PRINCIPAL_KINDS = ("user", "agent", "system")
VISIBILITIES = ("private", "shared", "tenant")
TRUST_LEVELS = ("verified_source", "agent_claim", "unknown")
ACTIONS = ("read", "write", "delete", "audit")

LOCAL_TENANT = "local"


@dataclass(frozen=True)
class Principal:
    """操作主体。多租户只用 tenant_id；多 Agent 用 (tenant_id, principal_id)。"""

    kind: str = "user"
    tenant_id: str = LOCAL_TENANT
    principal_id: str = "owner"
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in PRINCIPAL_KINDS:
            raise ValueError(f"principal kind must be one of {PRINCIPAL_KINDS}, got {self.kind!r}")
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not self.principal_id:
            raise ValueError("principal_id must be non-empty")


@dataclass(frozen=True)
class Scope:
    """记录归属的命名空间与可见性。`acl` 形如 {"agent-1": ("read",)}。"""

    namespace: str = "default"
    visibility: str = "private"
    tenant_id: str = LOCAL_TENANT
    owner_principal_id: Optional[str] = None
    acl: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.visibility not in VISIBILITIES:
            raise ValueError(f"visibility must be one of {VISIBILITIES}, got {self.visibility!r}")
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        for pid, actions in self.acl.items():
            bad = set(actions) - set(ACTIONS)
            if bad:
                raise ValueError(f"acl for {pid!r} has unknown actions {sorted(bad)}")


@dataclass(frozen=True)
class WriteContext:
    """写入时随记录落账的上下文（只追加；A1 里暂不强制写入账本）。"""

    principal: Principal
    provenance: str = ""
    trust: str = "unknown"
    ts: Optional[float] = None

    def __post_init__(self) -> None:
        if self.trust not in TRUST_LEVELS:
            raise ValueError(f"trust must be one of {TRUST_LEVELS}, got {self.trust!r}")
        if self.ts is not None and (not isinstance(self.ts, (int, float)) or self.ts < 0):
            raise ValueError("ts must be a non-negative number or None")


@dataclass(frozen=True)
class AuditEvent:
    """读/写/删除/越权拒绝的审计事件（只追加）。"""

    kind: str
    principal: Principal
    scope: Scope
    record_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("audit kind must be non-empty")
        if not isinstance(self.ts, (int, float)) or self.ts < 0:
            raise ValueError("ts must be non-negative")


def allows(principal: Principal, scope: Scope, action: str) -> bool:
    """纯函数 ACL：fail-closed。

    规则（A1 槽位语义，可扩展）：
    1. 租户不同 → 一律拒绝（多租户隔离的硬边界）；
    2. `tenant` 可见性 → 同租户内 read 允许，write 仍限 owner/acl；
    3. `shared` → owner 或 acl 中显式授予；
    4. `private` → 仅 owner。
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    if principal.tenant_id != scope.tenant_id:
        return False  # 租户隔离硬边界：跨租户一律拒绝
    granted = set(scope.acl.get(principal.principal_id, ()))
    is_owner = scope.owner_principal_id is not None and principal.principal_id == scope.owner_principal_id
    if action == "write":
        return is_owner or "write" in granted
    if action == "delete":
        return is_owner
    if action == "audit":
        return is_owner or "audit" in granted
    # read
    if is_owner or "read" in granted:
        return True
    return scope.visibility == "tenant"
