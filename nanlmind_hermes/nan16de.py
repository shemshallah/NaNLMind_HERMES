"""NaN16DE: the raw NaN16 double-endian envelope.

A NaN16 word is 64 bits in this exact byte order:

    b0 b1    far head, big-endian
    b2..b5   payload, little-endian
    b6 b7    near head, little-endian

The far/near heads are the four legal 0xNaN heads:

    0x7FF1 +sNaN
    0xFFF1 -sNaN
    0x7FF8 +qNaN
    0xFFF8 -qNaN

The class nibble is (near_class << 2) | far_class, giving exactly 16 Boolean
truth tables. The truth table bit index is 2*a+b.

A NaN16DE envelope carries one word plus a logical NANNET route. No host JSON
numbers define the word; the raw bytes are authoritative.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Iterable

from .crypto_compat import Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature

MAGIC = b"N16DE"
VERSION = 1
HEAD_BY_TYPE = (0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8)
TYPE_BY_HEAD = {head: i for i, head in enumerate(HEAD_BY_TYPE)}
TYPE_NAMES = ("+sNaN", "-sNaN", "+qNaN", "-qNaN")
ROUTE_RE = re.compile(
    r"^nan://HW(\d{1,3}):BS(\d{1,3}):MC(\d{1,3}):"
    r"SC(\d{1,3}):GM(\d{1,3})(?::UM(\d{1,3}))?$")


@dataclass(frozen=True)
class Route:
    hw: int
    bs: int
    mc: int
    sc: int
    gm: int
    um: int | None = None

    def __post_init__(self):
        values = (self.hw, self.bs, self.mc, self.sc, self.gm)
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255
               for v in values):
            raise ValueError("route coordinates must be 0..255")
        if self.um is not None and (isinstance(self.um, bool) or
                                    not isinstance(self.um, int) or
                                    not 0 <= self.um < 255):
            raise ValueError("UM must be 0..254 or omitted")

    def bytes(self) -> bytes:
        return bytes((self.hw, self.bs, self.mc, self.sc, self.gm,
                      255 if self.um is None else self.um))

    def text(self) -> str:
        base = (f"nan://HW{self.hw}:BS{self.bs}:MC{self.mc}:"
                f"SC{self.sc}:GM{self.gm}")
        return base if self.um is None else base + f":UM{self.um}"

    @classmethod
    def parse(cls, value: str) -> "Route":
        match = ROUTE_RE.match(value or "")
        if not match:
            raise ValueError("route must be nan://HW:BS:MC:SC:GM:UM with 0..255")
        groups = match.groups()
        return cls(*[int(x) for x in groups[:5]], int(groups[5]) if groups[5] else None)


@dataclass(frozen=True)
class NanWord:
    far: int
    near: int
    payload: int

    def __post_init__(self):
        if self.far not in (0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8):
            raise ValueError("invalid far NaN head")
        if self.near not in (0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8):
            raise ValueError("invalid near NaN head")
        if isinstance(self.payload, bool) or not isinstance(self.payload, int) \
                or not 0 <= self.payload <= 0xFFFFFFFF:
            raise ValueError("payload must be 32 bits")

    @classmethod
    def from_bits(cls, far_type: int, near_type: int, payload: int) -> "NanWord":
        if far_type not in range(4) or near_type not in range(4):
            raise ValueError("NaN type must be 0..3")
        return cls(HEAD_BY_TYPE[far_type], HEAD_BY_TYPE[near_type], payload)

    @classmethod
    def unpack(cls, raw: bytes) -> "NanWord":
        if len(raw) != 8:
            raise ValueError("NaN16 word must be exactly 8 bytes")
        far = int.from_bytes(raw[0:2], "big")
        payload = int.from_bytes(raw[2:6], "little")
        near = int.from_bytes(raw[6:8], "little")
        return cls(far, near, payload)

    def pack(self) -> bytes:
        return (self.far.to_bytes(2, "big") +
                self.payload.to_bytes(4, "little") +
                self.near.to_bytes(2, "little"))

    @property
    def far_class(self) -> int:
        return TYPE_BY_HEAD[self.far]

    @property
    def near_class(self) -> int:
        return TYPE_BY_HEAD[self.near]

    @property
    def opcode(self) -> int:
        return (self.near_class << 2) | self.far_class

    @property
    def opcode_name(self) -> str:
        names = ("false", "and", "a_and_not_b", "a", "not_a_and_b", "b",
                 "xor", "or", "nor", "xnor", "not_b", "a_or_not_b",
                 "not_a", "not_a_or_b", "nand", "true")
        return names[self.opcode]

    def eval_bool(self, a: int, b: int) -> int:
        return (self.opcode >> (2 * (a & 1) + (b & 1))) & 1

    def as_dict(self) -> dict:
        return {
            "raw_hex": self.pack().hex(),
            "far": hex(self.far),
            "near": hex(self.near),
            "far_type": TYPE_NAMES[self.far_class],
            "near_type": TYPE_NAMES[self.near_class],
            "opcode": self.opcode,
            "opcode_name": self.opcode_name,
            "truth_table": format(self.opcode, "04b"),
            "payload": self.payload,
            "byte_order": "far:BE,payload:LE,near:LE",
        }


@dataclass
class Envelope:
    route: Route
    kind: str
    word: NanWord
    installation_id: str
    epoch: int = 0
    timestamp: int = 0
    nonce: str = ""
    signature: bytes = b""

    def body(self) -> bytes:
        kind = self.kind.encode("utf-8")
        if not kind or len(kind) > 64:
            raise ValueError("kind must be 1..64 UTF-8 bytes")
        if not isinstance(self.installation_id, str) or not self.installation_id:
            raise ValueError("installation_id is required")
        identity = self.installation_id.encode("utf-8")
        if len(identity) > 512:
            raise ValueError("installation_id is too long for NaN16DE")
        return (MAGIC + bytes([VERSION]) + self.route.bytes() +
                len(kind).to_bytes(2, "big") + kind +
                int(self.epoch).to_bytes(8, "big") +
                int(self.timestamp).to_bytes(8, "big") +
                bytes.fromhex(self.nonce) +
                len(identity).to_bytes(2, "big") + identity +
                self.word.pack())

    def signing_bytes(self) -> bytes:
        body = self.body()
        return hashlib.sha256(body).digest()

    def sign(self, private_key: Ed25519PrivateKey) -> "Envelope":
        self.signature = private_key.sign(self.signing_bytes())
        return self

    def verify(self, public_key: Ed25519PublicKey) -> None:
        if len(self.signature) != 64:
            raise InvalidSignature("NaN16DE signature must be 64 bytes")
        public_key.verify(self.signature, self.signing_bytes())

    def to_bytes(self) -> bytes:
        body = self.body()
        return body + self.signature

    def as_dict(self) -> dict:
        return {
            "schema": "NAN16DE/1",
            "route": self.route.text(),
            "route_bytes": self.route.bytes().hex(),
            "kind": self.kind,
            "word": self.word.as_dict(),
            "word_raw": self.word.pack().hex(),
            "installation_id": self.installation_id,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": base64.urlsafe_b64encode(self.signature).rstrip(b"=").decode(),
            "body_sha256": hashlib.sha256(self.body()).hexdigest(),
            "byte_order": "NaN16DE/1 double-endian",
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def make_envelope(route: Route | str, kind: str, word: NanWord,
                  private_key: Ed25519PrivateKey, installation_id: str,
                  epoch: int = 0, now: int | None = None) -> Envelope:
    env = Envelope(
        route=Route.parse(route) if isinstance(route, str) else route,
        kind=kind, word=word, installation_id=installation_id,
        epoch=int(epoch), timestamp=int(time.time()) if now is None else int(now),
        nonce=secrets.token_bytes(16).hex(),
    )
    return env.sign(private_key)


def decode_envelope(raw: bytes) -> Envelope:
    if len(raw) < 5 + 1 + 6 + 2 + 1 + 8 + 8 + 16 + 2 + 1 + 8 + 64:
        raise ValueError("NaN16DE envelope is too short")
    if raw[:5] != MAGIC or raw[5] != VERSION:
        raise ValueError("unsupported NaN16DE magic/version")
    p = 6
    route_values = list(raw[p:p + 6])
    route = Route(*route_values[:5], None if route_values[5] == 255 else route_values[5])
    p += 6
    kind_len = int.from_bytes(raw[p:p + 2], "big"); p += 2
    if kind_len < 1 or kind_len > 64 or p + kind_len > len(raw):
        raise ValueError("invalid NaN16DE kind length")
    kind = raw[p:p + kind_len].decode("utf-8"); p += kind_len
    epoch = int.from_bytes(raw[p:p + 8], "big"); p += 8
    timestamp = int.from_bytes(raw[p:p + 8], "big"); p += 8
    nonce = raw[p:p + 16].hex(); p += 16
    identity_len = int.from_bytes(raw[p:p + 2], "big"); p += 2
    if identity_len < 1 or identity_len > 512 or p + identity_len > len(raw):
        raise ValueError("invalid NaN16DE installation identity length")
    installation_id = raw[p:p + identity_len].decode("utf-8"); p += identity_len
    word = NanWord.unpack(raw[p:p + 8]); p += 8
    signature = raw[p:p + 64]; p += 64
    if p != len(raw):
        raise ValueError("trailing bytes in NaN16DE envelope")
    return Envelope(route, kind, word, installation_id, epoch, timestamp,
                    nonce, signature)


def send_envelope(portal_url: str, bearer_token: str, envelope: Envelope,
                  timeout: int = 30) -> dict:
    """POST one signed NaN16DE envelope to the NANNET portal route."""
    import urllib.request
    body = json.dumps({"envelope": envelope.to_bytes().hex()}).encode()
    request = urllib.request.Request(
        portal_url.rstrip("/") + "/v1/nan16de/route", data=body,
        headers={"Authorization": "Bearer " + bearer_token,
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
