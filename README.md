# memory-kernel — deterministic citation enforcement for long-term memory

> [简体中文](README.zh-CN.md) | English

> **What this is**: a small, dependency-free kernel that makes an AI assistant's *citations* checkable.
> It gives you (1) an append-only, tamper-evident evidence ledger and (2) a deterministic
> enforcement layer that judges every `(claim, citation)` pair — fabricated quotes, negation flips,
> time inversions, speaker mismatches, stale/superseded evidence, redundant citations.
> **What this is not**: a memory database, a retrieval system, or a hosted product.

- License: code **Apache-2.0**, docs **CC-BY-4.0**.
- Pure Python standard library. No network. No model inside the enforcement path.
- Version 0.x: contracts may evolve; every contract change ships with a version number.

## Why

LLM assistants cite "evidence". Most systems never verify it. This kernel makes the verification
mechanical and reproducible: the ledger says what evidence exists (and that it wasn't edited),
the enforcement layer says whether a citation is legal/relevant, and the emission gate rewrites
what the model produced (hedge / delete / mark as unverified) before the user sees it.

## How this differs from mainstream memory architectures

Most "memory" systems optimize for **context efficiency**: summarize old turns, extract facts into a graph/vector store, or compress and retrieve chunks so the model can fit more history. This kernel optimizes for **claim integrity**: whether each claim can be traced to immutable evidence and mechanically checked before the user sees it. The two are complementary, not competing.

| Dimension | Mainstream memory (summarize / extract / compress / RAG) | memory-kernel |
|---|---|---|
| Optimizes for | token efficiency, recall, persistence | auditability, refusal correctness, reproducibility |
| Store model | mutable summaries / extracted facts / vector chunks | append-only, hash-chained evidence ledger (L0); derived layers allowed but marked derived |
| Update semantics | overwrite / merge / last-write-wins | append + explicit correction/supersede events; historical facts and *current basis* are separate |
| Retrieval | built-in embeddings / BM25 / rerank | external and replaceable; the kernel is retriever-agnostic |
| Model output | free-form text | per-segment claim + verbatim quote; no evidence → explicitly unverified |
| Adjudication | none, or another LLM as judge | deterministic predicates only — no model in the enforcement path |
| Typical failure mode | fluent but unverifiable; silent evidence loss; stale facts served as current | hedges / refuses / rewrites; explicitly marks unverified |
| Cost profile | cheap context | more tokens and storage, plus millisecond-level gate; **no compression savings** |
| Scale | vector databases at billions of items | a kernel, not a database: 8k-message demo works; 100k needs index engineering |

**Advantages**

- **Verifiable, replayable claims**: fabricated quotes, negation flips, time inversions, speaker mismatches, incomplete answers and superseded evidence are caught deterministically.
- **No silent evidence loss**: originals are never rewritten; corrections are events, not mutations.
- **Model-agnostic and offline-capable**: pure standard library; the same gate works with any generator and any retriever.
- **Explicit uncertainty**: no citation means "unverified"; degraded claims are rendered with hedges instead of quietly passing.
- **Historical vs current separation**: "what was true then" and "what is the current basis" can both be answered without lying.

**Limitations (read before adopting)**

- **No efficiency gain**: this is not a summarizer or compressor. Expect more storage and more generation tokens, plus a small gate overhead.
- **Not a semantic judge**: a verbatim but irrelevant citation still passes mechanical checks; truth is not guaranteed. See `docs/ALGO-CEILING.md`.
- **Requires model cooperation**: structured per-segment output with verbatim quotes; free-form text cannot be enforced.
- **Not a memory database**: retrieval quality, indexing, scale and latency remain your system's responsibility. The reference retriever is lexical on purpose: cross-encoder reranking and all-cross-session dense fusion were negative on our data, and a **conditional dense routing** (preference-guarded, adopted internally after a one-shot test on n=150) is **not** part of the reference demo.
- **Integration and operations overhead**: ledger writes, citation anchors, key management and audit surfaces must be wired into your pipeline.

**When it fits**

- Regulated or enterprise knowledge assistants where answers must be auditable.
- High-stakes Q&A, compliance records, and post-incident debugging of hallucinated citations.
- Long-running assistants that must distinguish "what the user said / what changed / what is current now".
- Multi-agent or shared-memory setups that need **locally auditable** per-source provenance (writer/speaker metadata; **not** third-party-verifiable attribution — that capability is deliberately out of scope).
- Local-first / offline deployments (no network, no model in the trust path).

**When it does not fit**

- Consumer companion chat where fluency, latency and recall matter more than citation integrity.
- Large-scale semantic recall where the bottleneck is indexing and ANN, not verification.
- Workflows that cannot require citations or have no evidence to cite.
- Teams looking for a drop-in replacement for a memory API (Mem0, Zep, LangChain memory, vector DB memory) — this is a lower layer, not that product.

**Combining with mainstream memory**: you can keep a summarizer/extractor upstream — as long as derived outputs are marked derived and the originals stay in L0. Compression then cannot destroy evidence, and extracted facts may *propose* but never *adjudicate* (the enforcement layer remains deterministic).

## Quickstart (no dependencies)

```bash
python examples/dummy_retrieval/demo.py        # ledger -> retrieve -> enforce, end to end
python -m unittest discover -s tests           # run the test suite
```

## Retrieval is a commodity — enforcement is the asset

The demo ships a **reference retriever** (`examples/dummy_retrieval/`) so you can run the demo
without writing code. It is naive BM25 + char n-gram + RRF and is **not** for production.
Replace it with LangChain / LlamaIndex / Milvus / your own vector DB — the enforcement layer
does not care how evidence was retrieved:

```python
from dummy_retriever import BaseRetriever            # the only interface the demo needs
from langchain_community.vectorstores import Milvus  # any real retriever
retriever = Milvus(embeddings=embeddings, connection_args={"host": "localhost", "port": "19530"})
hits = retriever.similarity_search(claim_text, k=10)  # adapt to BaseRetriever: index()/search()
```

- Do **not** file issues complaining that the bundled BM25 is not smart enough. It exists so you can
  see the enforcement layer working in 30 seconds. Retrieval quality is your call.
- The bundled retriever's **cross-session** recall@10 is **77.8%** on our internal claim→evidence
  probes (8,219-candidate pool). The often-quoted 96.7% same-session number is **in-sample**
  (8 candidates on median) and is **not a capability claim**; neither number is a promise for
  your corpus. Cross-encoder reranking and all-cross-session dense fusion were negative on our
  data (a preference-guarded conditional dense routing was adopted internally after a one-shot
  n=150 test, but it is **not** part of the reference demo).

## What is deliberately NOT in this repo

- Private corpora, real chat data, evaluation probes and evaluation artifacts.
- Model weights (embedding / reranker ONNX files). If you want them, download them yourself.
- The hosted product, multi-tenant service code, UI, and connectors.
- Internal design/decision records (a sanitized copy of the design rationale is published separately).

## Reading order

1. `docs/EMISSION-CONTRACT.md` — the three stages (deterministic prefilter → model generation → emission gate).
2. `docs/ALGO-V4-CONTRACT.md` — what the enforcement predicates check.
3. `docs/ALGO-CEILING.md` — what enforcement can and cannot catch (read this before trusting it).
4. `docs/PROMPT-CONTRACT-v1.md` — the citation rules the model must follow.
5. `examples/dummy_retrieval/` — runnable demo + the retriever you should replace.
6. **[docs/EVAL-SUMMARY.md](docs/EVAL-SUMMARY.md)** — what it achieves today (aggregates only; failures included).

## Evaluation results (aggregated, no probes)

We publish **aggregate numbers only** — no per-probe rows, no real chat excerpts, no private corpora:

- **[docs/EVAL-SUMMARY.md](docs/EVAL-SUMMARY.md)** — capability overview: what the system can / cannot do today
  (headline results, the enforcement layer's role, failure scenes, negative results);
- **[docs/EVAL-RESULTS.md](docs/EVAL-RESULTS.md)** — every number with its dataset kind, n, caliber and limitations;
- [`results/aggregate/`](results/aggregate/) — machine-readable aggregates (numbers / short English labels only).

Internal probes, per-question artifacts and private corpora are **deliberately not released**
(evaluation assets + privacy). Public benchmarks here are retrieval/enforcement-layer only, not
official QA scores; same-session or in-sample numbers are explicitly marked as **not citable**, and
negative results are listed rather than hidden. Read the caliber notes before quoting any number.

## Security / ethics

- The enforcement layer is **not** a semantic judge: "verbatim but irrelevant" citations pass
  mechanical checks. See `docs/ALGO-CEILING.md`.
- Do not put secrets or real personal data into issues/PRs.
- No warranty. This is infrastructure for building auditability, not a compliance product.
