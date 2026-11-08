"""Boolean-to-wave embedding and atomic word-leaf update tests."""
import unittest

from nanlmind_hermes.boolean_wave import (
    BOOL_MASKS,
    boolean,
    multilinear,
    negate,
    polarity,
    verify_exact,
)
from nanlmind_hermes.word_writer import (
    WordLeafWriter,
    pack_state,
    rank_of_pair,
    unpack_state,
    zeros,
)


class BooleanWaveTests(unittest.TestCase):
    def test_all_sixteen_functions_are_exact_on_booleans(self):
        result = verify_exact()
        self.assertTrue(result["exact"])
        self.assertEqual(result["correct"], 64)
        for mask in BOOL_MASKS:
            self.assertEqual(multilinear(mask, 1, 1), boolean(mask, 1, 1))
            self.assertEqual(multilinear(mask, 0, 1), boolean(mask, 0, 1))

    def test_fuzzy_extension_is_not_boolean_idempotent(self):
        # OR(a,a) = 2a - a^2; it equals a only on {0,1}.
        self.assertAlmostEqual(multilinear(14, 0.5, 0.5), 0.75)
        self.assertAlmostEqual(multilinear(8, 0.5, 0.5), 0.25)
        self.assertAlmostEqual(negate(0.25), 0.75)
        self.assertTrue(polarity(0.7)["yes"])
        self.assertTrue(polarity(-0.7)["no"])


class WordWriterTests(unittest.TestCase):
    def setUp(self):
        self.writer = WordLeafWriter(n=8, inner={0, 7},
                                     permutation=[7, 6, 5, 4, 3, 2, 1, 0])
        self.matrix = zeros(8)

    def test_letter_packets_cannot_occupy_the_void(self):
        with self.assertRaises(ValueError):
            self.writer.letter_packet([0], [1.0])

    def test_word_update_preserves_centrosymmetry_and_void(self):
        first = self.writer.letter_packet([2, 3], [1.0, 0.5])
        second = self.writer.letter_packet([4, 5], [0.75, 1.0])
        updated, record = self.writer.write_word(self.matrix, [first, second])
        self.assertEqual(record["kind"], "word")
        self.assertEqual(record["letters"], 2)
        self.assertTrue(record["centrosymmetric"])
        self.assertTrue(record["void_zero"])
        self.assertLess(self.writer._residual(updated), 1e-9)
        self.assertTrue(all(abs(updated[head][column]) < 1e-12
                            for head in (0, 7) for column in range(8)))

    def test_letter_update_is_centrosymmetric_and_void_safe(self):
        packet = self.writer.letter_packet([2, 5], [1.0, 0.75])
        updated, record = self.writer.write_letter(self.matrix, packet)
        self.assertEqual(record["kind"], "letter")
        self.assertLess(self.writer._residual(updated), 1e-9)
        self.assertTrue(all(abs(updated[head][column]) < 1e-12
                            for head in (0, 7) for column in range(8)))

    def test_pair_rank_and_nan_payload(self):
        self.assertEqual(rank_of_pair(0, 1), 0)
        self.assertEqual(rank_of_pair(127, 128), 24384)
        self.assertEqual(rank_of_pair(254, 255), 32639)
        payload = pack_state(24384, 128, 300)
        state = unpack_state(payload)
        self.assertEqual(state, {"rank": 24384, "amplitude": 128, "phase": 300})
        self.assertLess(payload, 1 << 32)


if __name__ == "__main__":
    unittest.main()
