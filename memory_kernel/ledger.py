# SPDX-License-Identifier: Apache-2.0
"""L0/L1：只追加哈希链 + 全载荷签名。

与 SPEC 的对应：
  content_hash   —— 只回答"内容是否被改"
  record_hash    —— 回答"这条记录是否被改"（含 prev_hash / seq）
  signature      —— 覆盖**全载荷**，包括 speaker 等谓词会读取的字段（修正 N1）
唯一写接口是 append()；没有 update / delete。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional

from .canon import (
    DOMAIN_PAYLOAD,
    DOMAIN_RECORD,
    DOMAIN_SIG,
    NORMALIZATION,
    digest,
    normalize,
    sha256_hex,
    sign,
    verify_sign,
)

ZERO_HASH = "0" * 64


class LedgerError(Exception):
    pass


class RecordVerificationError(LedgerError):
    def __init__(self, anchor_id: str, reason: str) -> None:
        super().__init__(f"record {anchor_id!r} rejected: {reason}")
        self.anchor_id = anchor_id
        self.reason = reason


class ChainBroken(LedgerError):
    def __init__(self, seq: int, anchor_id: str, reason: str) -> None:
        super().__init__(f"hash chain broken at seq={seq} id={anchor_id!r}: {reason}")
        self.seq = seq
        self.anchor_id = anchor_id
        self.reason = reason


_HEX_DIGITS = frozenset("0123456789abcdef")


def _has_completeness(rec: "AnchorRecord") -> bool:
    """三元组是否被使用（存在性区分：全缺 = 旧形态，全有 = C1 新形态）。"""
    return (
        rec.is_complete is not None
        or rec.finish_reason is not None
        or rec.delivered_sha256 is not None
    )


def check_completeness_triple(
    is_complete: Optional[bool],
    finish_reason: Optional[str],
    delivered_sha256: Optional[str],
) -> None:
    """C1 原子性与类型校验：全缺或全有；半套 / 类型不符 → ValueError（拒绝写入）。

    单侧性：这里只拒绝"不可审计"的输入，不替调用方推断缺省值；缺省就是"没有完整性证据"。
    """
    present = (
        is_complete is not None,
        finish_reason is not None,
        delivered_sha256 is not None,
    )
    if any(present) and not all(present):
        raise ValueError(
            "completeness evidence is atomic: is_complete + finish_reason + "
            "delivered_sha256 must be given together (got "
            f"is_complete={is_complete!r}, finish_reason={finish_reason!r}, "
            f"delivered_sha256={delivered_sha256!r})"
        )
    if not any(present):
        return
    if not isinstance(is_complete, bool):
        raise ValueError(f"is_complete must be bool, got {type(is_complete).__name__}")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ValueError(f"finish_reason must be a non-empty str, got {finish_reason!r}")
    if not isinstance(delivered_sha256, str) or len(delivered_sha256) != 64 \
            or any(ch not in _HEX_DIGITS for ch in delivered_sha256.lower()):
        raise ValueError(
            "delivered_sha256 must be a 64-char hex sha256 digest, "
            f"got {delivered_sha256!r}"
        )


@dataclass(frozen=True)
class AnchorRecord:
    seq: int
    id: str
    content: str
    content_hash: str
    char_span: tuple[int, int]
    source_id: str
    episode_id: str
    turn_index: int
    speaker: Optional[str]
    modality: str
    record_hash: str
    prev_hash: str
    signature: str
    # ---- A1：写入者身份/签名的 schema 槽位（默认 None/0，不改变既有哈希与序列化） ----
    # A2 才接 Ed25519 双签名/密钥注册/外锚；A1 只保证"以后接不用改账本格式"。
    writer_id: Optional[str] = None
    key_epoch: int = 0
    writer_seq: Optional[int] = None
    submission_hash: Optional[str] = None
    # ---- Plan Z C1：streaming 完整性证据的三元组（默认全缺，不改变既有哈希与序列化） ----
    # 原子性：三者要么全缺（用户轮 / 旧记录 / 非流式记录），要么同时存在（代理写入的 assistant 轮）。
    # 存在时纳入 payload → 参与 payload_hash / signature / record_hash（篡改任一项 → 验签失败）；
    # 全缺时与 C1 之前逐字节一致。**不升全局 schema 版本**（存在性区分 v1/v2 形态）。
    # C6 语义：`is_complete=True` 仅当上游正常结束**且**全部 SSE 增量写成功；
    # `delivered_sha256` 是用户实际看到的**原始字节**（NFC 前）的哈希——「已送达」判定在代理层，
    # core 只存字段、不判定、不重算。
    is_complete: Optional[bool] = None
    finish_reason: Optional[str] = None
    delivered_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        # 原子性校验放在记录构造处：任何构造路径（append / append_batch / _load / 直接构造）
        # 都不可能产生"半套完整性证据"这种不可审计状态。
        check_completeness_triple(self.is_complete, self.finish_reason, self.delivered_sha256)

    @property
    def length(self) -> int:
        return len(self.content)

    @property
    def attributed(self) -> bool:
        """是否带写入者归属（A1 槽位是否被使用）。"""
        return self.writer_id is not None

    @property
    def has_completeness(self) -> bool:
        """是否带 Plan Z C1 完整性三元组（旧记录 / 用户轮为 False）。"""
        return _has_completeness(self)


def _has_attribution(rec: AnchorRecord) -> bool:
    return (
        rec.writer_id is not None
        or rec.key_epoch != 0
        or rec.writer_seq is not None
        or rec.submission_hash is not None
    )


def _payload_fields(rec: AnchorRecord) -> tuple[str, ...]:
    """参与 payload 哈希与签名的全部字段（顺序固定）。

    **默认记录（无归属、无完整性三元组）与 C1/A1 之前逐字节一致**：不附加任何新字段，
    保证已发布指纹与冻结产物不变。带归属的记录（A2 起启用）把归属字段纳入 payload——
    签名覆盖"谁写的"，符合 SPEC N1；带完整性三元组的记录（C1）把三元组纳入 payload——
    签名覆盖"这条回答是否完整送达"，篡改任一项都会验签失败。
    """
    base = (
        rec.id,
        str(rec.char_span[0]),
        str(rec.char_span[1]),
        rec.content_hash,
        rec.source_id,
        rec.episode_id,
        str(rec.turn_index),
        rec.speaker or "",
        rec.modality,
        rec.content,
    )
    if _has_attribution(rec):
        base = base + (
            rec.writer_id or "",
            str(rec.key_epoch),
            "" if rec.writer_seq is None else str(rec.writer_seq),
            rec.submission_hash or "",
        )
    if _has_completeness(rec):
        base = base + (
            "true" if rec.is_complete else "false",
            rec.finish_reason or "",
            rec.delivered_sha256 or "",
        )
    return base


def payload_hash(rec: AnchorRecord) -> str:
    return digest(DOMAIN_PAYLOAD, *_payload_fields(rec))


def record_hash(rec: AnchorRecord) -> str:
    return digest(DOMAIN_RECORD, rec.prev_hash, str(rec.seq), payload_hash(rec))


def signature(rec: AnchorRecord, key: bytes) -> str:
    return sign(key, DOMAIN_SIG, payload_hash(rec))


class AnchorLedger:
    """JSONL 只追加账本。构造时校验整条链。"""

    def __init__(self, path: os.PathLike | str | None, key: bytes, *, verify_chain: bool = True) -> None:
        if len(key) < 16:
            raise ValueError("store key must be >= 16 bytes")
        self._key = bytes(key)
        self._path = Path(path) if path is not None else None
        self._records: list[AnchorRecord] = []
        self._by_id: dict[str, AnchorRecord] = {}
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                self._load(verify_chain=verify_chain)

    # ---- 读 ----
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[AnchorRecord]:
        return iter(self._records)

    @property
    def records(self) -> tuple[AnchorRecord, ...]:
        return tuple(self._records)

    def get(self, anchor_id: str) -> AnchorRecord:
        rec = self._by_id.get(anchor_id)
        if rec is None:
            raise RecordVerificationError(anchor_id, "not found")
        self._verify(rec)
        return rec

    def try_get(self, anchor_id: str) -> Optional[AnchorRecord]:
        try:
            return self.get(anchor_id)
        except RecordVerificationError:
            return None

    # ---- 写（唯一写路径） ----
    def append(
        self,
        content: str,
        *,
        source_id: str,
        episode_id: str,
        turn_index: int = 0,
        speaker: Optional[str] = None,
        modality: str = "text",
        char_span: Optional[tuple[int, int]] = None,
        writer_id: Optional[str] = None,
        key_epoch: int = 0,
        writer_seq: Optional[int] = None,
        submission_hash: Optional[str] = None,
        is_complete: Optional[bool] = None,
        finish_reason: Optional[str] = None,
        delivered_sha256: Optional[str] = None,
    ) -> AnchorRecord:
        """追加一条记录。

        A1 归属槽位（均可省略，省略时行为与 A1 之前逐字节一致）：
        `writer_id / key_epoch / writer_seq / submission_hash`。
        校验框架：**要么全不填，要么 writer_id + writer_seq + submission_hash 同时给**
        （key_epoch 可有默认 0）——避免"半个归属"这种不可审计状态。A2 才校验签名本身。

        Plan Z C1 完整性三元组（均可省略，省略时与 C1 之前逐字节一致）：
        `is_complete / finish_reason / delivered_sha256`——**原子**（全缺或全有，半套 ValueError），
        存在时纳入 payload（参与 payload_hash / signature / record_hash）。
        """
        attribution = (writer_id, key_epoch, writer_seq, submission_hash)
        if any(v is not None for v in (writer_id, writer_seq, submission_hash)) or key_epoch != 0:
            if not (writer_id and writer_seq is not None and submission_hash):
                raise ValueError(
                    "attribution requires writer_id + writer_seq + submission_hash together "
                    "(key_epoch optional, default 0)"
                )
            if key_epoch < 0:
                raise ValueError("key_epoch must be >= 0")
        elif attribution != (None, 0, None, None):  # pragma: no cover - defensive
            raise ValueError("invalid attribution combination")
        # 写入前先校验三元组原子性：拒绝路径不产生任何副作用（不写文件、不进内存链）
        check_completeness_triple(is_complete, finish_reason, delivered_sha256)
        body = normalize(content)
        seq = len(self._records)
        anchor_id = f"a_{seq:06d}"
        prev = self._records[-1].record_hash if self._records else ZERO_HASH
        base = AnchorRecord(
            seq=seq,
            id=anchor_id,
            content=body,
            content_hash=sha256_hex(body.encode("utf-8")),
            char_span=(0, len(body)) if char_span is None else char_span,
            source_id=source_id,
            episode_id=episode_id,
            turn_index=turn_index,
            speaker=speaker,
            modality=modality,
            record_hash="",
            prev_hash=prev,
            signature="",
            writer_id=writer_id,
            key_epoch=key_epoch,
            writer_seq=writer_seq,
            submission_hash=submission_hash,
            is_complete=is_complete,
            finish_reason=finish_reason,
            delivered_sha256=delivered_sha256,
        )
        rec = replace(base, record_hash=record_hash(base), signature=signature(base, self._key))
        if self._path is not None:
            line = json.dumps(record_to_json(rec), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._path.open("ab") as fh:
                fh.write(line.encode("utf-8") + b"\n")
                fh.flush()
                os.fsync(fh.fileno())
        self._records.append(rec)
        self._by_id[rec.id] = rec
        return rec

    # ---- 批量写（group commit：一次 fsync 提交多条） ----
    def append_batch(self, items: list[dict]) -> list[AnchorRecord]:
        """批量追加，只做一次 fsync。items 的字段同 append()（含 C1 三元组，同样原子校验）。"""
        built: list[AnchorRecord] = []
        lines: list[bytes] = []
        for item in items:
            is_complete = item.get("is_complete")
            finish_reason = item.get("finish_reason")
            delivered_sha256 = item.get("delivered_sha256")
            # 先校验三元组：半套 → ValueError，且此时尚未写任何字节（原子拒绝）
            check_completeness_triple(is_complete, finish_reason, delivered_sha256)
            body = normalize(item["content"])
            seq = len(self._records) + len(built)
            prev = (built[-1].record_hash if built else (self._records[-1].record_hash if self._records else ZERO_HASH))
            base = AnchorRecord(
                seq=seq,
                id=f"a_{seq:06d}",
                content=body,
                content_hash=sha256_hex(body.encode("utf-8")),
                char_span=(0, len(body)),
                source_id=item["source_id"],
                episode_id=item["episode_id"],
                turn_index=int(item.get("turn_index", 0)),
                speaker=item.get("speaker"),
                modality=item.get("modality", "text"),
                record_hash="",
                prev_hash=prev,
                signature="",
                writer_id=item.get("writer_id"),
                key_epoch=int(item.get("key_epoch", 0)),
                writer_seq=item.get("writer_seq"),
                submission_hash=item.get("submission_hash"),
                is_complete=is_complete,
                finish_reason=finish_reason,
                delivered_sha256=delivered_sha256,
            )
            rec = replace(base, record_hash=record_hash(base), signature=signature(base, self._key))
            built.append(rec)
            lines.append(
                json.dumps(record_to_json(rec), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
        if self._path is not None and lines:
            with self._path.open("ab") as fh:
                fh.write(b"".join(lines))
                fh.flush()
                os.fsync(fh.fileno())
        self._records.extend(built)
        for rec in built:
            self._by_id[rec.id] = rec
        return built

    # ---- 校验 ----
    def verify(self, rec: AnchorRecord) -> None:
        """校验任意一条记录（含 C1 完整性三元组）的内容哈希 + 全载荷签名 + record_hash。

        公开入口给 Plan Z C4 的冷归档 resolver 用：外部解析回来的记录必须先过与本地记录
        完全相同的校验，才允许进入执法路径。**不读取/不修改账本状态**。
        """
        self._verify(rec)

    def _verify(self, rec: AnchorRecord) -> None:
        if sha256_hex(rec.content.encode("utf-8")) != rec.content_hash:
            raise RecordVerificationError(rec.id, "content_hash mismatch")
        if not verify_sign(self._key, rec.signature, DOMAIN_SIG, payload_hash(rec)):
            raise RecordVerificationError(rec.id, "signature mismatch")
        if record_hash(rec) != rec.record_hash:
            raise RecordVerificationError(rec.id, "record_hash mismatch")

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        prev = ZERO_HASH
        for idx, rec in enumerate(self._records):
            try:
                self._verify(rec)
            except RecordVerificationError:
                return False, rec.seq
            if rec.seq != idx or rec.prev_hash != prev:
                return False, rec.seq
            prev = rec.record_hash
        return True, None

    def _load(self, *, verify_chain: bool) -> None:
        prev = ZERO_HASH
        with self._path.open("rb") as fh:
            for idx, raw in enumerate(fh):
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                try:
                    rec = AnchorRecord(
                        seq=int(obj["seq"]),
                        id=str(obj["id"]),
                        content=str(obj["content"]),
                        content_hash=str(obj["content_hash"]),
                        char_span=(int(obj["char_span"][0]), int(obj["char_span"][1])),
                        source_id=str(obj["source_id"]),
                        episode_id=str(obj["episode_id"]),
                        turn_index=int(obj["turn_index"]),
                        speaker=obj.get("speaker"),
                        modality=str(obj["modality"]),
                        record_hash=str(obj["record_hash"]),
                        prev_hash=str(obj["prev_hash"]),
                        signature=str(obj["signature"]),
                        writer_id=obj.get("writer_id"),
                        key_epoch=int(obj.get("key_epoch", 0)),
                        writer_seq=obj.get("writer_seq"),
                        submission_hash=obj.get("submission_hash"),
                        # C1：旧文件缺这三个键 → None（旧账本可读、链校验不变）
                        is_complete=obj.get("is_complete"),
                        finish_reason=obj.get("finish_reason"),
                        delivered_sha256=obj.get("delivered_sha256"),
                    )
                except ValueError as exc:
                    # 文件里出现"半套完整性证据" = 账本损坏，按链断裂显式报错，不静默修补
                    raise ChainBroken(int(obj.get("seq", idx)), str(obj.get("id", "")),
                                      f"invalid completeness triple: {exc}") from exc
                if verify_chain:
                    if rec.seq != idx:
                        raise ChainBroken(rec.seq, rec.id, "non-contiguous seq")
                    try:
                        self._verify(rec)
                    except RecordVerificationError as exc:
                        raise ChainBroken(rec.seq, rec.id, exc.reason) from exc
                    if rec.prev_hash != prev:
                        raise ChainBroken(rec.seq, rec.id, "prev_hash mismatch")
                    prev = rec.record_hash
                self._records.append(rec)
                self._by_id[rec.id] = rec


def record_to_json(rec: AnchorRecord) -> dict:
    out = {
        "seq": rec.seq,
        "id": rec.id,
        "content": rec.content,
        "content_hash": rec.content_hash,
        "char_span": list(rec.char_span),
        "source_id": rec.source_id,
        "episode_id": rec.episode_id,
        "turn_index": rec.turn_index,
        "speaker": rec.speaker,
        "modality": rec.modality,
        "record_hash": rec.record_hash,
        "prev_hash": rec.prev_hash,
        "signature": rec.signature,
    }
    # 默认记录（无归属）不输出新字段：序列化与 A1 之前逐字节一致（旧文件/指纹不受影响）。
    if _has_attribution(rec):
        out["writer_id"] = rec.writer_id
        out["key_epoch"] = rec.key_epoch
        out["writer_seq"] = rec.writer_seq
        out["submission_hash"] = rec.submission_hash
    # Plan Z C1：三元组只在存在时输出（旧记录 / 用户轮 JSON 逐字节不变）
    if _has_completeness(rec):
        out["is_complete"] = rec.is_complete
        out["finish_reason"] = rec.finish_reason
        out["delivered_sha256"] = rec.delivered_sha256
    return out
