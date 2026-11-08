"""Temporal spine tests: true rank mirror and local coupling convergence."""
import unittest

from nanlmind_hermes.temporal_spine import (
    PAIR_COUNT,
    Spine,
    mirror_pair,
    mirror_rank,
    pair_at,
    rank_of_pair,
)


class SpineTests(unittest.TestCase):
    def test_rank_mapping_is_a_bijection(self):
        self.assertEqual(PAIR_COUNT, 32640)
        self.assertEqual(rank_of_pair(0, 1), 0)
        self.assertEqual(rank_of_pair(127, 128), 24384)
        self.assertEqual(rank_of_pair(254, 255), 32639)
        self.assertEqual(pair_at(0), (0, 1))
        self.assertEqual(pair_at(32639), (254, 255))
        self.assertEqual(pair_at(24384), (127, 128))

    def test_true_rank_mirror_is_not_complement(self):
        rank = rank_of_pair(127, 128)
        self.assertEqual(mirror_rank(rank), 32639 - rank)
        self.assertEqual(mirror_pair(rank), pair_at(8255))
        self.assertNotEqual(mirror_pair(rank), (127, 128))
        for rank in (0, 1, 100, 24384, 32639):
            self.assertEqual(mirror_rank(mirror_rank(rank)), rank)

    def test_local_update_converges_and_preserves_mirror(self):
        ranks = list(range(0, 64))
        spine = Spine(ranks)
        past = {rank: complex(0.5 + (rank % 7) / 10, -0.2) for rank in ranks}
        truth = {rank: complex(0.3, 0.4 + rank / 1000) for rank in ranks}
        future = {rank: truth[rank] * past[rank] for rank in ranks}
        spine.load(past, future)
        before = spine.error()
        history = spine.train(epochs=120, eta=0.9)
        self.assertLess(history[-1], before)
        self.assertLess(spine.error(), 1e-3)
        self.assertLess(spine.mirror_residual(), 1e-12)


if __name__ == "__main__":
    unittest.main()
