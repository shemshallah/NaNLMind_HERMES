import time
import unittest

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey
from nanlmind_hermes.nan16de import (
    Envelope, NanWord, Route, decode_envelope, make_envelope,
)


class Nan16DETests(unittest.TestCase):
    def test_double_endian_word_layout_and_boolean_table(self):
        word = NanWord.from_bits(2, 1, 0x12345678)
        self.assertEqual(word.pack().hex(), "7ff878563412f1ff")
        self.assertEqual(word.opcode, 6)
        self.assertEqual(word.opcode_name, "xor")
        self.assertEqual([word.eval_bool(a, b) for a in (0, 1) for b in (0, 1)],
                         [0, 1, 1, 0])
        restored = NanWord.unpack(word.pack())
        self.assertEqual(restored, word)

    def test_route_round_trip_and_um_reservation(self):
        route = Route.parse("nan://HW2:BS1:MC3:SC4:GM5:UM6")
        self.assertEqual(route.text(), "nan://HW2:BS1:MC3:SC4:GM5:UM6")
        self.assertEqual(route.bytes().hex(), "020103040506")
        with self.assertRaises(ValueError):
            Route.parse("nan://HW0:BS0:MC0:SC0:GM0:UM255")

    def test_envelope_sign_verify_and_decode(self):
        key = Ed25519PrivateKey.generate()
        word = NanWord.from_bits(0, 3, 42)
        envelope = make_envelope("nan://HW1:BS2:MC3:SC4:GM5:UM6", "chat",
                                 word, key, "installation-id", now=int(time.time()))
        decoded = decode_envelope(envelope.to_bytes())
        decoded.verify(key.public_key())
        self.assertEqual(decoded.route.text(), "nan://HW1:BS2:MC3:SC4:GM5:UM6")
        self.assertEqual(decoded.word, word)
        self.assertEqual(decoded.installation_id, "installation-id")


if __name__ == "__main__":
    unittest.main()
