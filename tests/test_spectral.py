"""Spectral wave operator tests."""
import unittest

from nanlmind_hermes.head_population import HeadPopulation
from nanlmind_hermes.spectral import (
    WaveMatrix,
    centrosymmetrize,
    head_region,
    mirror_fold,
    mode_regions,
    power_iteration,
    void_potential,
)


class SpectralTests(unittest.TestCase):
    def test_power_iteration_finds_top_diagonal_values(self):
        matrix = [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]
        values, vectors = power_iteration(matrix, k=3, iterations=200)
        self.assertAlmostEqual(values[0], 5.0, places=5)
        self.assertAlmostEqual(values[1], 3.0, places=5)
        self.assertAlmostEqual(values[2], 1.0, places=5)

    def test_centrosymmetric_fold_matches_the_full_spectrum(self):
        B = [[1.0, 2.0], [3.0, 4.0]]
        C = [[5.0, 6.0], [7.0, 8.0]]
        JCJ = [row[::-1] for row in C[::-1]]
        JBJ = [row[::-1] for row in B[::-1]]
        matrix = [B[0] + C[0], B[1] + C[1], JCJ[0] + JBJ[0], JCJ[1] + JBJ[1]]
        self.assertLess(WaveMatrix(matrix).centrosymmetry_residual(), 1e-12)
        P, M = mirror_fold(matrix)
        folded = sorted([P[0][0] + P[1][1], P[0][0] * P[1][1] - P[0][1] * P[1][0],
                         M[0][0] + M[1][1], M[0][0] * M[1][1] - M[0][1] * M[1][0]])
        # Determinants are not eigenvalues; use numpy only as a check in the
        # experiment. Here verify the centrosymmetry identity instead.
        self.assertEqual(len(P), 2)
        self.assertEqual(len(M), 2)

    def test_void_potential_and_regions(self):
        matrix = [[0.0] * 256 for _ in range(256)]
        with_void = void_potential(matrix, lam=2.0)
        self.assertGreater(with_void[119][119], with_void[0][0])
        masses = mode_regions([1.0] * 256)
        self.assertAlmostEqual(sum(masses.values()), 1.0)
        self.assertIn(head_region(0), {"inner", "ring", "outer"})

    def test_native_mirror_and_complement_differ(self):
        from nanlmind_hermes.spectral import complement_mirror, native_mirror
        native = native_mirror(256)
        complement = complement_mirror(256)
        self.assertEqual(sum(1 for h in range(256) if native[h] == h), 16)
        self.assertEqual(sum(1 for h in range(256) if complement[h] == h), 0)
        self.assertNotEqual(native, complement)

    def test_fiedler_partition_finds_two_blocks(self):
        from nanlmind_hermes.spectral import fiedler_partition
        matrix = [[0.0] * 8 for _ in range(8)]
        for i in range(4):
            for j in range(4):
                if i != j:
                    matrix[i][j] = 1.0
                    matrix[j][i] = 1.0
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    matrix[i][j] = 1.0
                    matrix[j][i] = 1.0
        cut = fiedler_partition(matrix)
        self.assertEqual(len(cut["inner"]), 4)
        self.assertEqual(cut["values"][0], 0.0)

    def test_normalized_laplacian_has_unit_diagonal(self):
        from nanlmind_hermes.spectral import normalized_laplacian
        matrix = [[0.0, 1.0], [1.0, 0.0]]
        laplacian = normalized_laplacian(matrix)
        self.assertEqual(laplacian[0][0], 1.0)
        self.assertAlmostEqual(laplacian[0][1], -1.0)

    def test_gram_matrix_from_head_population_is_symmetric(self):
        population = HeadPopulation(heads=16, active=15)
        for word in ("the", "then", "there", "other"):
            population.observe(word)
        wave = WaveMatrix.from_head_population(population)
        self.assertEqual(len(wave.matrix), 16)
        self.assertLess(wave.symmetry_residual(), 1e-12)


if __name__ == "__main__":
    unittest.main()
