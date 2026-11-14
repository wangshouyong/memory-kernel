# SPDX-License-Identifier: Apache-2.0
"""最小闭环端到端冒烟：写入 → 执法 → 语域渲染 → 篡改检测 → 审计估计器。

目的：给"最小闭环"一条**一键可重跑**的证据（此前只有各轮评测报告与单测，
没有单条命令把六段串起来跑一遍）。

覆盖链条（全部确定性、不触网、不调用模型）：
  1. L0/L1 只追加账本：content_hash + prev_hash 链 + 全载荷签名（append 时 fsync）
  2. Tier 0 + Tier 1 执法：合法引用 → verified；逐字但证据不支撑 → degraded；
     捏造/非逐字 → rejected(verbatim)；归属翻转 → rejected(preserve_speaker)
  3. 输出层：verdict/coverage → 四语域（fact/hedged/unknown）确定性派生 + 措辞模板
  4. 篡改检测：直接改磁盘内容 → 重载时报 ChainBroken（content_hash mismatch）
  5. 审计估计器：Wilson 区间（样本不足必须报 unmeasured 的口径依据）

用法：
  <python> eval/min_loop_demo.py        # 全 PASS 退出码 0；任一断言失败退出码 1
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_kernel import AnchorLedger, Claim, Citation, enforce  # noqa: E402
from memory_kernel import registers  # noqa: E402
from memory_kernel.audit import wilson_ci  # noqa: E402
from memory_kernel.skeleton import SkeletonPolicy  # noqa: E402

KEY = b"0123456789abcdef0123456789abcdef"
_V3 = dict(tau_soft=0.35, hard_units=True, temporal=True, idf=False,
           arg_slots=True, dispatch=True, version="skel-v3.0")


def main() -> int:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    tmp = Path(tempfile.mkdtemp(prefix="minloop-"))
    path = tmp / "ledger.jsonl"

    # ---- 1. 只追加账本 + 全载荷签名 ----
    led = AnchorLedger(path, KEY)
    r1 = led.append("I finally got to see Aerosmith live last night, it was amazing.",
                    source_id="s0", episode_id="e0", turn_index=1, speaker="Dave")
    r2 = led.append("Gina told me she is opening a bakery next month.",
                    source_id="s0", episode_id="e0", turn_index=2, speaker="Jon")
    check("1.1 账本写入并落盘", path.exists() and len(led) == 2, f"{len(led)} records")
    check("1.2 链校验通过", led.verify_chain() == (True, None))
    check("1.3 读取时全载荷签名校验通过", led.get(r1.id).id == r1.id)

    # ---- 2. 执法（Tier 0 + Tier 1）----
    pol = SkeletonPolicy(**_V3)
    verified = enforce(led, Claim(citation=Citation(r1.id, 0, r1.content), text=r1.content), policy=pol)
    check("2.1 合法引用 → verified", verified.verdict == "verified", verified.verdict)

    degraded = enforce(led, Claim(citation=Citation(r2.id, 0, r2.content), text=r1.content), policy=pol)
    check("2.2 逐字但证据不支撑 → degraded", degraded.verdict == "degraded", degraded.verdict)

    fab = enforce(led, Claim(citation=Citation(r1.id, 0, "I saw Aerosmith in concert."),
                             text="I saw Aerosmith in concert."), policy=pol)
    check("2.3 捏造/非逐字 → rejected(verbatim)",
          fab.verdict == "rejected" and fab.rejected_cause == "verbatim", str(fab.rejected_cause))

    flip = enforce(led, Claim(citation=Citation(r1.id, 0, r1.content), text=r1.content,
                              asserted_speaker="Jon"), policy=pol)
    check("2.4 归属翻转 → rejected(preserve_speaker)",
          flip.verdict == "rejected" and flip.rejected_cause == "preserve_speaker",
          str(flip.rejected_cause))

    # ---- 3. 四语域派生（模型不参与）----
    check("3.1 verified → fact/plain", registers.register_of(verified) == ("fact", "plain"))
    check("3.2 degraded → hedged/degraded", registers.register_of(degraded) == ("hedged", "degraded"))
    check("3.3 rejected → unknown/unverifiable",
          registers.register_of(fab) == ("unknown", "unverifiable"))
    check("3.4 措辞模板可渲染且不含模型自由发挥",
          registers.render(verified, "测试断言").text == "根据记录，测试断言")

    # ---- 4. 篡改检测（改磁盘内容后重载）----
    lines = path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["content"] = obj["content"].replace("Aerosmith", "Metallica")
    lines[0] = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        AnchorLedger(path, KEY)
        check("4.1 篡改被检出", False, "重载未报错")
    except Exception as exc:  # ChainBroken
        check("4.1 篡改被检出", type(exc).__name__ == "ChainBroken",
              f"{type(exc).__name__}: {str(exc)[:60]}")

    # ---- 5. 审计估计器 ----
    lo, hi = wilson_ci(3, 400)
    check("5.1 Wilson95 区间落在 [0,1] 且非空", 0.0 <= lo < hi <= 1.0, f"[{lo:.4f}, {hi:.4f}]")

    print("\n最小闭环：写入 → 执法 → 语域 → 篡改检测 → 审计 全部环节已贯通。" if ok else "\n有环节失败。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
