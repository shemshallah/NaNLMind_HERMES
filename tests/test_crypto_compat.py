import unittest

try:
    from cryptography.hazmat.primitives import serialization
except ImportError:  # optional dependency; the pure-Python fallback is supported
    serialization = None

from nanlmind_hermes.crypto_compat import (
    AESGCM, Ed25519PrivateKey, Ed25519PublicKey, HKDF, hashes,
)


class CryptoCompatTests(unittest.TestCase):
    def test_rfc8032_ed25519_vector(self):
        seed = bytes.fromhex(
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
        public = bytes.fromhex(
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        signature = bytes.fromhex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
        key = Ed25519PrivateKey.from_private_bytes(seed)
        public_key = key.public_key()
        if serialization is None:
            exported = public_key.public_bytes()
        else:
            exported = public_key.public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.assertEqual(exported, public)
        self.assertEqual(key.sign(b""), signature)
        Ed25519PublicKey.from_public_bytes(public).verify(signature, b"")

    def test_aes_256_gcm_vector(self):
        cipher = AESGCM(b"\0" * 32)
        encrypted = cipher.encrypt(b"\0" * 12, b"\0" * 16, b"")
        self.assertEqual(
            encrypted[:-16].hex(),
            "cea7403d4d606b6e074ec5d3baf39d18")
        self.assertEqual(
            encrypted[-16:].hex(),
            "d0d1c8a799996bf0265b98b5d48ab919")
        self.assertEqual(cipher.decrypt(b"\0" * 12, encrypted, b""), b"\0" * 16)

    def test_hkdf_rfc5869_case_1(self):
        ikm = bytes.fromhex("0b" * 22)
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        okm = HKDF(hashes.SHA256(), 42, salt, info).derive(ikm)
        self.assertEqual(
            okm.hex(),
            "3cb25f25faacd57a90434f64d0362f2a"
            "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
            "34007208d5b887185865")


if __name__ == "__main__":
    unittest.main()
