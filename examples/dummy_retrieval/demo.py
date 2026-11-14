# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: reference retrieval -> deterministic citation enforcement.

Run from the repository root:

    python examples/dummy_retrieval/demo.py

What it shows:
1. a tiny in-memory immutable ledger,
2. the reference retriever indexing ledger contents,
3. retrieval of candidate evidence for a claim,
4. `memory_kernel.enforce` judging the (claim, citation) pair — **retrieval-agnostic**.

Replace `DummyRetriever` with any `BaseRetriever` implementation (LangChain,
LlamaIndex, your vector DB) and this script works unchanged.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dummy_retriever import DummyRetriever  # noqa: E402
from memory_kernel import AnchorLedger, Citation, Claim, SkeletonPolicy, enforce  # noqa: E402

KEY = b"demo-key-0123456789abcdef"[:32].ljust(32, b"0")

DOCS = [
    "示例文档一：用于演示的检索样例，先做需求梳理与指标定义。",
    "示例文档二：记录一次虚构的周末出行，目的地是示例公园。",
    "示例文档三：示例项目决定下个季度按月更新指标。",
    "示例文档四：示例地区的天气不错，适合户外活动。",
]


def main() -> int:
    ledger = AnchorLedger(None, KEY)
    anchors = [
        ledger.append(text, source_id="demo", episode_id="demo", turn_index=i, speaker="user")
        for i, text in enumerate(DOCS)
    ]

    docs = [(rec.id, rec.content) for rec in anchors]
    retriever = DummyRetriever()          # <-- swap this line for a real retriever
    retriever.index(docs)

    claim_text = "示例文档一"
    hits = retriever.search(claim_text, k=3)
    print("retrieved:", hits)

    if not hits:
        print("no candidates; nothing to enforce")
        return 0

    rec = ledger.get(hits[0])
    offset = rec.content.find(claim_text)
    quote = rec.content if offset < 0 else claim_text
    offset = max(offset, 0)

    result = enforce(ledger, Claim(Citation(rec.id, offset, quote), text=claim_text),
                     policy=SkeletonPolicy())
    print(f"verdict={result.verdict} cause={result.rejected_cause} "
          f"failed={[p.name for p in result.failed if not p.passed]}")
    print("note: enforcement judges the citation, not how it was retrieved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
