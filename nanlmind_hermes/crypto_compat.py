"""Cryptography compatibility without the `cryptography` package.

When `cryptography` is installed, its primitives are used directly. On Termux or
other minimal images, the fallback provides:

* RFC 8032 Ed25519 signatures;
* AES-256-GCM authenticated encryption;
* HKDF-SHA256;
* the small serialization surface used by HERMES.

The fallback is intentionally local and dependency-free. It is not a substitute
for a platform HSM, but it keeps Ed25519 and authenticated encryption available.
"""
from __future__ import annotations

import hashlib
import hmac
import os

try:  # Prefer the audited package when the platform can install it.
    from cryptography.exceptions import InvalidSignature, InvalidTag
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    CRYPTOGRAPHY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on Termux/minimal images
    CRYPTOGRAPHY_AVAILABLE = False

    class InvalidSignature(Exception):
        pass

    class InvalidTag(Exception):
        pass

    # ------------------------------------------------------------------
    # RFC 8032 Ed25519, extended-coordinate implementation.
    # ------------------------------------------------------------------
    _P = 2 ** 255 - 19
    _L = 2 ** 252 + 27742317777372353535851937790883648493
    _D = (-121665 * pow(121666, _P - 2, _P)) % _P
    _I = pow(2, (_P - 1) // 4, _P)
    _B_Y = (4 * pow(5, _P - 2, _P)) % _P

    def _recover_x(y: int, sign: int) -> int:
        x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
        x = pow(x2, (_P + 3) // 8, _P)
        if (x * x - x2) % _P != 0:
            x = x * _I % _P
        if x & 1 != sign:
            x = _P - x
        return x

    _B = (_recover_x(_B_Y, 0), _B_Y, 1, (_recover_x(_B_Y, 0) * _B_Y) % _P)

    def _padd(p, q):
        x1, y1, z1, t1 = p
        x2, y2, z2, t2 = q
        a = (y1 - x1) * (y2 - x2) % _P
        b = (y1 + x1) * (y2 + x2) % _P
        c = 2 * t1 * t2 * _D % _P
        d = 2 * z1 * z2 % _P
        e, f, g, h = b - a, d - c, d + c, b + a
        return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)

    def _pmul(s: int, p):
        q = (0, 1, 1, 0)
        while s:
            if s & 1:
                q = _padd(q, p)
            p = _padd(p, p)
            s >>= 1
        return q

    def _compress(p) -> bytes:
        x, y, z, _ = p
        inv = pow(z, _P - 2, _P)
        x, y = x * inv % _P, y * inv % _P
        return (y | ((x & 1) << 255)).to_bytes(32, "little")

    def _decompress(raw: bytes):
        if len(raw) != 32:
            raise InvalidSignature("Ed25519 public key/signature has wrong length")
        y = int.from_bytes(raw, "little")
        sign = y >> 255
        y &= (1 << 255) - 1
        if y >= _P:
            raise InvalidSignature("invalid Ed25519 point")
        x = _recover_x(y, sign)
        return (x, y, 1, x * y % _P)

    def _sha512(data: bytes) -> bytes:
        return hashlib.sha512(data).digest()

    def _expand(seed: bytes):
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes")
        h = _sha512(seed)
        a = int.from_bytes(h[:32], "little")
        a &= (1 << 254) - 8
        a |= (1 << 254)
        return a, h[32:]

    class Ed25519PrivateKey:
        def __init__(self, seed: bytes):
            self._seed = bytes(seed)
            self._a, self._prefix = _expand(self._seed)
            self._public = _compress(_pmul(self._a, _B))

        @classmethod
        def generate(cls):
            return cls(os.urandom(32))

        @classmethod
        def from_private_bytes(cls, raw: bytes):
            return cls(raw)

        def private_bytes(self, encoding=None, format=None, encryption_algorithm=None):
            return self._seed

        def public_key(self):
            return Ed25519PublicKey(self._public)

        def sign(self, message: bytes) -> bytes:
            r = int.from_bytes(_sha512(self._prefix + message), "little") % _L
            R = _compress(_pmul(r, _B))
            k = int.from_bytes(_sha512(R + self._public + message), "little") % _L
            S = (r + k * self._a) % _L
            return R + S.to_bytes(32, "little")

    class Ed25519PublicKey:
        def __init__(self, raw: bytes):
            self._raw = bytes(raw)
            self._point = _decompress(self._raw)

        @classmethod
        def from_public_bytes(cls, raw: bytes):
            return cls(raw)

        def public_bytes(self, encoding=None, format=None):
            return self._raw

        def verify(self, signature: bytes, message: bytes) -> None:
            if len(signature) != 64:
                raise InvalidSignature("Ed25519 signature has wrong length")
            Rraw, Sraw = signature[:32], signature[32:]
            S = int.from_bytes(Sraw, "little")
            if S >= _L:
                raise InvalidSignature("Ed25519 signature scalar out of range")
            try:
                R, A = _decompress(Rraw), self._point
            except Exception as exc:
                raise InvalidSignature("invalid Ed25519 point") from exc
            k = int.from_bytes(_sha512(Rraw + self._raw + message), "little") % _L
            if _compress(_pmul(S, _B)) != _compress(_padd(R, _pmul(k, A))):
                raise InvalidSignature("Ed25519 signature mismatch")

    # ------------------------------------------------------------------
    # Minimal serialization surface.
    # ------------------------------------------------------------------
    class _Encoding:
        Raw = "raw"

    class _PrivateFormat:
        Raw = "raw"

    class _PublicFormat:
        Raw = "raw"

    class _NoEncryption:
        pass

    class _Serialization:
        Encoding = _Encoding
        PrivateFormat = _PrivateFormat
        PublicFormat = _PublicFormat
        NoEncryption = _NoEncryption

    serialization = _Serialization()

    # ------------------------------------------------------------------
    # AES-256-GCM.
    # ------------------------------------------------------------------
    _SBOX = (
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16)
    _RCON = (0,1,2,4,8,16,32,64,128,27,54,108,216,171,77)
    _MUL2 = tuple(((x << 1) ^ (0x11b if x & 0x80 else 0)) & 0xff for x in range(256))
    _MUL3 = tuple((_MUL2[x] ^ x) & 0xff for x in range(256))

    def _sub_word(w):
        return bytes(_SBOX[b] for b in w)

    def _rot_word(w):
        return w[1:] + w[:1]

    def _expand_aes_key(key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")
        words = [list(key[i:i+4]) for i in range(0, 32, 4)]
        for i in range(8, 60):
            temp = words[i-1][:]
            if i % 8 == 0:
                temp = list(_sub_word(bytes(_rot_word(bytes(temp)))))
                temp[0] ^= _RCON[i // 8]
            elif i % 8 == 4:
                temp = list(_sub_word(bytes(temp)))
            words.append([a ^ b for a, b in zip(words[i-8], temp)])
        return [bytes(b for w in words[4*i:4*i+4] for b in w) for i in range(15)]

    def _aes_block(key_schedule, block: bytes) -> bytes:
        state = list(block)
        def add(rk):
            for i, b in enumerate(rk):
                state[i] ^= b
        def sub():
            for i, b in enumerate(state):
                state[i] = _SBOX[b]
        def shift():
            state[1], state[5], state[9], state[13] = state[5], state[9], state[13], state[1]
            state[2], state[6], state[10], state[14] = state[10], state[14], state[2], state[6]
            state[3], state[7], state[11], state[15] = state[15], state[3], state[7], state[11]
        def mix():
            for c in range(4):
                i = 4 * c
                a0, a1, a2, a3 = state[i:i+4]
                state[i] = _MUL2[a0] ^ _MUL3[a1] ^ a2 ^ a3
                state[i+1] = a0 ^ _MUL2[a1] ^ _MUL3[a2] ^ a3
                state[i+2] = a0 ^ a1 ^ _MUL2[a2] ^ _MUL3[a3]
                state[i+3] = _MUL3[a0] ^ a1 ^ a2 ^ _MUL2[a3]
        add(key_schedule[0])
        for rnd in range(1, 14):
            sub(); shift(); mix(); add(key_schedule[rnd])
        sub(); shift(); add(key_schedule[14])
        return bytes(state)

    def _gf_mul(x: int, y: int) -> int:
        z, v = 0, y
        for i in range(128):
            if (x >> (127 - i)) & 1:
                z ^= v
            v = (v >> 1) ^ ((0xe1 << 120) if (v & 1) else 0)
        return z

    def _ghash(h: int, data: bytes) -> int:
        y = 0
        for i in range(0, len(data), 16):
            block = data[i:i+16].ljust(16, b"\0")
            y = _gf_mul(y ^ int.from_bytes(block, "big"), h)
        return y

    def _gcm_tag(key_schedule, nonce: bytes, aad: bytes, ciphertext: bytes) -> bytes:
        if len(nonce) != 12:
            raise ValueError("AES-GCM fallback requires a 12-byte nonce")
        h = int.from_bytes(_aes_block(key_schedule, b"\0" * 16), "big")
        j0 = nonce + b"\0\0\0\1"
        lengths = (len(aad) * 8).to_bytes(8, "big") + (len(ciphertext) * 8).to_bytes(8, "big")
        s = _ghash(h, aad + b"\0" * (-len(aad) % 16) + ciphertext +
                   b"\0" * (-len(ciphertext) % 16) + lengths)
        return (int.from_bytes(_aes_block(key_schedule, j0), "big") ^ s).to_bytes(16, "big")

    class AESGCM:
        def __init__(self, key: bytes):
            self._key = key
            self._schedule = _expand_aes_key(key)

        def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes:
            if len(nonce) != 12:
                raise ValueError("AES-GCM fallback requires a 12-byte nonce")
            j0 = nonce + b"\0\0\0\1"
            out = bytearray()
            counter = int.from_bytes(j0, "big") + 1
            for i in range(0, len(data), 16):
                block = data[i:i+16]
                stream = _aes_block(self._schedule, counter.to_bytes(16, "big"))
                out.extend(a ^ b for a, b in zip(block, stream))
                counter = (counter & ~0xffffffff) | ((counter + 1) & 0xffffffff)
            return bytes(out) + _gcm_tag(self._schedule, nonce, associated_data, bytes(out))

        def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes:
            if len(data) < 16:
                raise InvalidTag("AES-GCM payload is too short")
            ciphertext, tag = data[:-16], data[-16:]
            expected = _gcm_tag(self._schedule, nonce, associated_data, ciphertext)
            if not hmac.compare_digest(tag, expected):
                raise InvalidTag("AES-GCM authentication failed")
            j0 = nonce + b"\0\0\0\1"
            out = bytearray()
            counter = int.from_bytes(j0, "big") + 1
            for i in range(0, len(ciphertext), 16):
                block = ciphertext[i:i+16]
                stream = _aes_block(self._schedule, counter.to_bytes(16, "big"))
                out.extend(a ^ b for a, b in zip(block, stream))
                counter = (counter & ~0xffffffff) | ((counter + 1) & 0xffffffff)
            return bytes(out)

    # ------------------------------------------------------------------
    # HKDF-SHA256.
    # ------------------------------------------------------------------
    class _SHA256:
        pass

    class _Hashes:
        SHA256 = _SHA256

    hashes = _Hashes()

    class HKDF:
        def __init__(self, algorithm, length: int, salt: bytes, info: bytes):
            if not (algorithm is _SHA256 or isinstance(algorithm, _SHA256)):
                raise ValueError("only SHA256 is supported by the fallback")
            self.length = length
            self.salt = salt or b"\0" * 32
            self.info = info

        def derive(self, key_material: bytes) -> bytes:
            prk = hmac.new(self.salt, key_material, hashlib.sha256).digest()
            output = b""
            block = b""
            counter = 1
            while len(output) < self.length:
                block = hmac.new(prk, block + self.info + bytes([counter]),
                                 hashlib.sha256).digest()
                output += block
                counter += 1
            return output[:self.length]
