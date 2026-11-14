# SPDX-License-Identifier: Apache-2.0
"""审计层：对照组污染率估计 + Wilson 置信区间。

SPEC §6.1 的硬规则：
  * 硬检查无条件全量，永不被采样；
  * 污染率报告必须带样本量与 CI；
  * 样本量不足时输出 unmeasured，禁止点估计。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def required_sample(p: float, rel_halfwidth: float = 0.2, z: float = 1.96) -> int:
    if p <= 0:
        return 0
    return math.ceil((z / rel_halfwidth) ** 2 * (1 - p) / p)


@dataclass
class Stratum:
    checked: int = 0
    dirty: int = 0


@dataclass
class ContaminationReport:
    strata: dict[str, Stratum]
    n_min: int = 100
    merged: list[str] = field(default_factory=list)

    def add(self, bucket: str, *, checked: bool, dirty: bool) -> None:
        if not checked:
            return
        s = self.strata.setdefault(bucket, Stratum())
        s.checked += 1
        s.dirty += int(dirty)

    def estimate(self) -> dict:
        out = {"overall": None, "by_stratum": {}, "unmeasured": [], "uncovered": 0}
        tot_n, tot_k = 0, 0
        for bucket, s in self.strata.items():
            if s.checked < self.n_min:
                out["unmeasured"].append(bucket)
                out["uncovered"] += s.checked
                continue
            lo, hi = wilson_ci(s.dirty, s.checked)
            out["by_stratum"][bucket] = {
                "n": s.checked,
                "dirty": s.dirty,
                "point": s.dirty / s.checked,
                "ci95": [lo, hi],
            }
            tot_n += s.checked
            tot_k += s.dirty
        if tot_n >= self.n_min:
            lo, hi = wilson_ci(tot_k, tot_n)
            out["overall"] = {
                "n": tot_n,
                "dirty": tot_k,
                "point": tot_k / tot_n,
                "ci95": [lo, hi],
            }
        else:
            out["unmeasured"].append("__overall__")
        return out


def stratified_sample(
    events: Iterable[dict], *, rate: float, seed: int = 0
) -> list[dict]:
    """确定性分层抽样：同一个 (event_id, seed) 永远得到同一个决策（可复放）。"""
    import hashlib

    picked = []
    for ev in events:
        bucket = ev.get("bucket", "default")
        h = hashlib.sha256(f"{seed}|{ev['event_id']}|{bucket}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        if u < rate:
            picked.append(ev)
    return picked
