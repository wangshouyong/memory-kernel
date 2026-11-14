# `examples/dummy_retrieval/` — reference retriever (NOT the product)

> ⚠️ **This is a zero-dependency reference retriever.** It uses naive BM25 + char n-gram + RRF
> purely to make the end-to-end demo runnable. **DO NOT USE IN PRODUCTION.** Replace this with
> LlamaIndex, LangChain, Milvus, or your own Vector DB.

## Why does this exist?

The core asset of this project is **deterministic citation enforcement + an immutable evidence
ledger** (`memory_kernel/`). Retrieval is a commodity capability. This folder exists only so that
you can run an end-to-end demo **without installing anything and without writing retrieval code**.

- `dummy_retriever.py` — ~200 lines: CJK-bigram BM25 + character 2/3/4-gram BM25 + RRF (+ `BaseRetriever` ABC).
- `demo.py` — ledger → index → retrieve → `enforce()` verdict. One command, zero dependencies.

```bash
python examples/dummy_retrieval/demo.py
```

## The one rule: retrieval is swappable, enforcement is not

`BaseRetriever` has exactly two methods — `index(docs)` and `search(query, k)` — so the demo
pipeline never depends on this implementation. Replacing it takes three lines:

```python
# from dummy_retriever import DummyRetriever
from langchain_community.vectorstores import Milvus
retriever = Milvus(embeddings=embeddings, connection_args={"host": "localhost", "port": "19530"})
retriever.index([(doc_id, text) for doc_id, text in docs])   # or your adapter
hits = retriever.search(claim_text, k=10)
```

The enforcement layer (`memory_kernel.enforce`) receives only `(claim, citation)` — it **does not
know and does not care** how the evidence was retrieved. Swap in LangChain / LlamaIndex / your own
vector DB and the enforcement layer is completely unaffected.

## Please do not file retrieval-quality issues here

- This retriever is **not** tuned and is **not** a claim of best practice. It is the "no code
  required to see the demo" path.
- It was measured on the project's own **claim→evidence** probe set (not real question→evidence
  retrieval): recall@10 ≈ 96.7%, recall@3 ≈ 83.7%; the debiased subset is much lower. Those numbers
  are **not** a promise for your data.
- Dense embeddings and a cross-encoder reranker were tried and turned out to be **negative
  contributors** on our data (reranking lowered recall@3), so they are deliberately not shipped.
  The experiments are documented in the project's internal evaluation reports; do not assume
  "adding embeddings will help" — measure on your own corpus.
- Bugs in this folder that are clearly about naïve BM25 quality will be closed as `wontfix`
  with a pointer to this file. Issues about the **enforcement / ledger semantics** are very welcome.

## License & disclaimer

Apache-2.0, like the rest of the code. Provided **AS IS**, no warranty of retrieval quality,
fitness for production, or correctness on your data. You are responsible for validating any
retriever you put in front of the enforcement layer.
