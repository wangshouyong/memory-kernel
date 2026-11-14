# SPDX-License-Identifier: Apache-2.0
"""T0 不可逆的编码决策：规范化、长度前缀封装、域分隔标签。

这三样在 SPEC 里被列为"写第一条记录前必须定死"，因为事后修改会让历史签名全部失效。
本模块就是那份决策的可执行形式。
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import unicodedata

NORMALIZATION = "NFC"

# 域分隔标签：每个签名/哈希用途一个，防止跨协议重放与签名混淆（§2.5 技术细节）
DOMAIN_PAYLOAD = b"MEM/PAYLOAD/v1"
DOMAIN_RECORD = b"MEM/RECORD/v1"
DOMAIN_SIG = b"MEM/SIGNATURE/v1"


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"content must be str, got {type(text)!r}")
    return unicodedata.normalize(NORMALIZATION, text)


def frame(*parts: bytes | str) -> bytes:
    """长度前缀封装：禁止 a|b|c 字符串拼接（分隔符注入）。"""
    out = bytearray()
    for part in parts:
        b = part.encode("utf-8") if isinstance(part, str) else bytes(part)
        out += struct.pack(">I", len(b))
        out += b
    return bytes(out)


def digest(domain: bytes, *parts: bytes | str) -> str:
    return hashlib.sha256(frame(domain, *parts)).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sign(key: bytes, domain: bytes, *parts: bytes | str) -> str:
    return hmac.new(key, frame(domain, *parts), hashlib.sha256).hexdigest()


def verify_sign(key: bytes, signature: str, domain: bytes, *parts: bytes | str) -> bool:
    return hmac.compare_digest(sign(key, domain, *parts), signature)


def find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    spans, start = [], 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return spans
        spans.append(idx)
        start = idx + 1
