# SPDX-License-Identifier: Apache-2.0
# ⚠️ WARNING: This is a zero-dependency reference retriever.
# It uses naive BM25 + char n-gram + RRF purely to make the end-to-end demo runnable.
# DO NOT USE IN PRODUCTION. Replace this with LlamaIndex, LangChain, Milvus,
# or your own Vector DB (see README-OSS.md for a 3-line swap example).
"""Reference-only retriever so the enforcement layer can be demoed end-to-end.

This module exists for ONE reason: let you run `python examples/dummy_retrieval/demo.py`
without installing anything. It is intentionally naive:

- lexical BM25 over CJK bigrams + ASCII words,
- BM25 over character 2/3/4-grams (helps a bit with paraphrases),
- reciprocal rank fusion (RRF, k=60).

Known limitations (measured on the project's own claim->evidence probe set, which is a
*claim->evidence* setup, NOT real question->evidence retrieval):

- recall@10 ~96.7%, recall@3 ~83.7% on that probe set; debiased subset much lower;
- dense embeddings and a cross-encoder reranker were tried and turned out to be
  negative contributors on this data, so they are deliberately NOT included here;
- no query understanding, no entity/time filters, no ANN index, no incremental updates.

If retrieval quality matters for your use case, swap this class out. The enforcement
layer (`memory_kernel.enforce`) does not know or care how evidence was retrieved --
it only judges the (claim, citation) pair it is given.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Sequence

__all__ = ["BaseRetriever", "DummyRetriever", "tokenize", "char_ngrams"]

_ASCII_WORD = re.compile(r"[a-z0-9]+")


class BaseRetriever(ABC):
    """Minimal retriever interface. Swap implementations without touching enforcement.

    The contract is deliberately tiny on purpose:
      - `index(docs)` takes `(doc_id, text)` pairs (idempotent rebuild is fine),
      - `search(query, k)` returns doc ids, best first.

    A production implementation can wrap LangChain/LLamaIndex/Milvus and still satisfy
    this interface; the demo pipeline only ever calls these two methods.
    """

    @abstractmethod
    def index(self, docs: Sequence[tuple[str, str]]) -> None:
        """(Re)build the index from `(doc_id, text)` pairs."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[str]:
        """Return up to `k` doc ids, most relevant first. Must be deterministic."""


def tokenize(text: str) -> list[str]:
    """CJK bigrams + lowercased ASCII words. Naive, deterministic, zero-dependency."""
    text = text.lower()
    out: list[str] = []
    for word in _ASCII_WORD.findall(text):
        if len(word) > 1:
            out.append(word)
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def char_ngrams(text: str, nmin: int = 2, nmax: int = 4) -> list[str]:
    """Character 2..4-grams over letters/digits/CJK; punctuation acts as separator."""
    text = text.lower()
    runs = re.findall(r"[0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", text)
    out: list[str] = []
    for run in runs:
        for n in range(nmin, nmax + 1):
            if len(run) < n:
                continue
            out.extend(run[i:i + n] for i in range(len(run) - n + 1))
    return out


class _BM25:
    """Textbook BM25 with binary-ish tf and smooth idf. No dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._n = 0
        self._avgdl = 0.0
        self._ids: list[str] = []

    def fit(self, docs: Sequence[tuple[str, str]], tokenize_fn=tokenize) -> None:
        self._tf = []
        self._len = []
        self._df = Counter()
        self._ids = []
        for doc_id, text in docs:
            toks = tokenize_fn(text)
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())
            self._ids.append(doc_id)
        self._n = len(self._ids)
        self._avgdl = (sum(self._len) / self._n) if self._n else 0.0

    def scores(self, query: str, tokenize_fn=tokenize) -> list[float]:
        q = Counter(tokenize_fn(query))
        out: list[float] = []
        for tf, dl in zip(self._tf, self._len):
            s = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl)) if self._avgdl else self.k1
            for term, qtf in q.items():
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
                s += idf * (tf.get(term, 0) * (self.k1 + 1.0)) / (tf.get(term, 0) + norm)
            out.append(s)
        return out


def _ranks(scores: Sequence[float]) -> list[int]:
    """1-based ranks, descending, stable by index for deterministic ties."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


class DummyRetriever(BaseRetriever):
    """Reference implementation: lexical BM25 + char n-gram BM25 + RRF. Not production."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60) -> None:
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self._lex = _BM25(k1=k1, b=b)
        self._ng = _BM25(k1=k1, b=b)
        self._ids: list[str] = []

    def index(self, docs: Sequence[tuple[str, str]]) -> None:
        self._ids = [doc_id for doc_id, _ in docs]
        self._lex.fit(docs, tokenize_fn=tokenize)
        self._ng.fit(docs, tokenize_fn=char_ngrams)

    def search(self, query: str, k: int = 10) -> list[str]:
        if not self._ids:
            return []
        lex = self._lex.scores(query, tokenize_fn=tokenize)
        ng = self._ng.scores(query, tokenize_fn=char_ngrams)
        r_lex, r_ng = _ranks(lex), _ranks(ng)
        fused = [
            (1.0 / (self.rrf_k + r_lex[i])) + (1.0 / (self.rrf_k + r_ng[i]))
            if (lex[i] > 0.0 or ng[i] > 0.0) else 0.0
            for i in range(len(self._ids))
        ]
        order = sorted(range(len(fused)), key=lambda i: (-fused[i], self._ids[i]))
        return [self._ids[i] for i in order if fused[i] > 0.0][:k]
