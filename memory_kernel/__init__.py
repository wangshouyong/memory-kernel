# SPDX-License-Identifier: Apache-2.0
"""memory_kernel — 不可变证据账本的确定性内核（最小闭环）。

对应 SPEC 的"核心"分级：L0/L1 链、Tier 0 谓词、Tier 1 极简骨架、verdict/coverage、
四语域派生、审计估计。合规/安全件（外锚、身份、可归因承诺、结算）不在本内核内。
"""

from .canon import NORMALIZATION, digest, find_all, frame, normalize, sha256_hex, sign, verify_sign
from .ledger import (
    AnchorLedger,
    AnchorRecord,
    ChainBroken,
    LedgerError,
    RecordVerificationError,
    check_completeness_triple,
)
from .obligations import PredicateResult, apply_obligations, obligation_scope, sentence_bounds
from .skeleton import (
    POLICY_VERSION,
    POLICY_VERSION_V3,
    POLICY_VERSION_V31_C3,
    IDFModel,
    ArgumentGrounding,
    ArgumentUnit,
    SkeletonPolicy,
    SkeletonResult,
    arg_lemma,
    argument_grounding,
    build_idf,
    claim_arguments,
    content_tokens,
    hard_units,
    idf_for_ledger,
    norm_tokens,
    skeleton_check,
    spans_of,
    stem,
)
from .enforce import (
    EVIDENCE_DEGRADE_REASON_OF,
    EVIDENCE_RESOLUTION_PREDICATE,
    Claim,
    Citation,
    EnforcementResult,
    enforce,
)
from .cite_set import OVER_CITATION_PREDICATES, check_citation_set
# internal-task：跨请求派生缓存收敛为显式、可注入、可 clear() 的 EngineCache
from .engine_cache import (
    CacheStats,
    EngineCache,
    clear_default_cache,
    default_engine_cache,
    reset_default_engine_cache,
    set_default_engine_cache,
)
from .scenario import (
    COMPLETENESS_UNKNOWN,
    EVIDENCE_INCOMPLETE,
    SCENARIO_PREDICATES,
    ScenarioContext,
    check_scenario,
)
from .scenario_context import (
    ALLOWED_SCOPES,
    ARCHIVED_TIMEOUT_REASON,
    EVIDENCE_UNAVAILABLE_REASONS,
    ContextAudit,
    EvidenceUnavailable,
    RecordTimeStore,
    build_scenario_context,
    claim_anchors,
)
from .slot_contract import (
    POLARITIES,
    SLOT_EVENT_KINDS,
    SLOT_SCHEMA_VERSION,
    SlotEvent,
    StateSlot,
    make_slot_event,
    slot_event_hash,
    state_slot_payload_hash,
)
from .pipeline import (
    FAILED_TEXT,
    NO_PREFETCH_TEXT,
    UNREFINED_TEXT,
    AnswerPipeline,
    PipelineBudget,
    PipelineResult,
    prewarm_ledger,
)
from .registers import REGISTERS, register_of, render
from .audit import ContaminationReport, required_sample, stratified_sample, wilson_ci
from .identity import (
    ACTIONS,
    LOCAL_TENANT,
    PRINCIPAL_KINDS,
    TRUST_LEVELS,
    VISIBILITIES,
    AuditEvent,
    Principal,
    Scope,
    WriteContext,
    allows,
)
from .storage import (
    AuditStore,
    IndexStore,
    InMemoryAuditStore,
    InMemoryRecordTimeStore,
    LedgerStore,
    LocalLedgerStore,
)
