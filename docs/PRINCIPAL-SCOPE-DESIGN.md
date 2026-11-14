# 主体 / 作用域统一抽象（多租户与多 Agent 共用的底座）

> 目的：**只做一套隔离抽象**。多租户用"租户"这一维；多 Agent 在同一个租户内再加"代理"这一维。
> 这样多 Agent 上线时只改 principal/ACL/冲突策略，不重写存储与账本。
> 关联设计线：多租户槽位、本抽象、归属、多 Agent。

## 1. 四个核心类型（先定签名，后写实现）

| 类型 | 字段 | 说明 |
|---|---|---|
| `Principal` | `kind`（user/agent/system）、`tenant_id`、`principal_id`、`roles` | **多租户只用到 tenant_id；多 Agent 用 (tenant_id, principal_id)** |
| `Scope` | `namespace`、`visibility`（private/shared/tenant）、`acl` | 记录归属的命名空间与可见性；共享命名空间是 Agent 协同的关键 |
| `WriteContext` | `principal`、`provenance`（source/model/prompt_version）、`trust`（verified_source/agent_claim/unknown）、`ts` | 写入时随记录落账（只追加）；`trust` 决定后续能否当事实引用 |
| `AuditEvent` | `kind`、`principal`、`scope`、`record_id`、`ts`、`reason` | 读/写/删除/越权拒绝都留事件 |

## 2. 存储与索引约定

- 存储键统一 `(tenant_id, namespace, record_seq)`；索引命名空间同样带 `(tenant_id, namespace)`。
- 单机版：`tenant_id` 固定 `"local"`（零成本）；`namespace` 默认 `"default"`。
- 索引是派生数据，必须能从账本重建；重建不得改变 `record_hash`。

## 3. ACL 边界（一条硬规则）

**ACL 只在存储/服务入口检查，绝不进入执法层。** 执法层只判"引用是否合法"，不判"你有没有权限看"。
越权访问在存储端口直接拒绝（fail-closed），并记 `AuditEvent`；越权不得表现为"记录不存在"而不留痕。

## 4. 多租户 vs 多 Agent 的差异（谁需要额外做什么）

| 维度 | 多租户 | 多 Agent |
|---|---|---|
| Principal | 只到 user/system | 到 agent（同租户内可区分） |
| Scope | 只有 tenant 级 | 增加 shared 命名空间 + 每 Agent 写作用域 |
| 归属 | 租户内可省略 | **必须**：谁写的（写入者标识；第三方可验证归属未实现） |
| 冲突 | 不需要 | **必须**：同事实多写入者的冲突/可信度策略（多 Agent 设计线） |
| 评测 | 隔离/越权 | 污染注入/越权/隔离/溯源 |

**结论**：隔离这件事"稍微改改"就够；**归属与冲突策略不会自动出现**，需要单独立项（归属与冲突策略需单独立项；A2 第三方可验证归属本版本不提供）。

## 5. 非目标（防止过度设计）

- 不做分布式事务、不做跨库两阶段提交；单机先简单正确。
- 不把租户/Agent 权限判定塞进 `enforce()`/`emit()`；它们保持纯函数。
- 不在单机版引入必须的租户管理服务/计费/多租户 UI。

## 6. 验收（落地时）

1. 单机：`tenant="local"` 下所有功能与现在一致（回归全绿）。
2. 越权测试：A 租户读/写/审计 B 租户 → 显式拒绝 + 审计事件（读/写/审计三条路径都覆盖）。
3. 多 Agent 预演（不实现业务逻辑）：同一租户两个 agent，按共享命名空间 ACL 可读/写/拒绝三类用例通过。
4. 内核无状态：两个 principal 并发调用互不影响；同输入重复调用逐字节一致。
