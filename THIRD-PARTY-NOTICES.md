memory-kernel
Copyright 2026 The memory-kernel Authors

This product includes software developed by third parties. The core package
(`memory_kernel/`) uses only the Python standard library; the dependencies
listed below are used by optional evaluation tooling or local model inference
and are **not distributed** with this repository.

Third-party components and expected licenses (verify at the upstream repository
before any release; this file is not a legal opinion):

| Component | Use | Expected license | Distribution |
|---|---|---|---|
| Python standard library | core + tests | PSF License | runtime |
| `tokenizers` (Hugging Face, Rust) | optional eval tooling | Apache-2.0 | not distributed |
| `onnxruntime` | optional local inference | MIT | not distributed |
| `numpy` | optional eval tooling | BSD-3-Clause | not distributed |
| BGE embedding model weights (`BAAI/bge-small-zh-v1.5`, ONNX conversion) | optional retrieval experiments | check upstream (commonly MIT) | **not distributed** |
| BGE reranker weights (`BAAI/bge-reranker-base` / `Xenova` ONNX conversion) | optional rerank experiments (negative result, not adopted) | check upstream (commonly MIT) | **not distributed** |

Model weights are downloaded by the user, if at all; no model binaries are
included in this repository. If you redistribute a build that bundles any of
the above, you must include their license texts and attribution notices.
