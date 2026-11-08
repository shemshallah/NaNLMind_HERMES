"""Head-population next-letter model tests."""
import unittest

from nanlmind_hermes.head_population import BigramBaseline, HeadPopulation


class HeadPopulationTests(unittest.TestCase):
    def test_heads_hold_letter_superpositions_not_letters(self):
        population = HeadPopulation(heads=64, active=8)
        for word in ("the", "then", "there", "these", "other", "mother"):
            population.observe(word)
        distribution = population.distribution("th")
        self.assertTrue(distribution)
        self.assertEqual(next(iter(distribution)), "e")
        votes = population.head_votes("th")
        self.assertTrue(votes)
        self.assertLessEqual(sum(votes.values()), 9)
        self.assertTrue(any(len(head) > 1 for head in population.heads if head))
        status = population.status()
        self.assertFalse(status["letter_to_head_mapping"])
        self.assertEqual(status["head_0"], "NaNNet + polarity prior")

    def test_partial_words_train_and_no_dictionary_is_stored(self):
        population = HeadPopulation(heads=64, active=8)
        for prefix, nxt in (("th", "e"), ("the", "r"), ("ther", "e")):
            population.observe_partial(prefix, nxt)
        status = population.status()
        self.assertTrue(status["dictionary_in_superposition"])
        self.assertEqual(status["stored_words"], 0)
        self.assertTrue(population.superposition())
        self.assertEqual(population.distribution("th").most_common(1)[0][0]
                         if hasattr(population.distribution("th"), "most_common")
                         else next(iter(population.distribution("th"))), "e")

    def test_all_256_heads_activate_with_head_zero_as_prior(self):
        population = HeadPopulation(heads=256, active=255)
        active = population._active("th")
        self.assertEqual(len(active), 256)
        self.assertIn(0, active)
        self.assertEqual(active, list(range(256)))

    def test_diversity_reports_specialization(self):
        population = HeadPopulation(heads=32, active=8, train_head0=False)
        for word in ("the", "then", "there", "other", "mother"):
            population.observe(word)
        diversity = population.diversity()
        self.assertGreaterEqual(diversity["heads"], 1)
        self.assertGreaterEqual(diversity["effective_rank"], 1.0)
        self.assertGreaterEqual(diversity["mean_entropy"], 0.0)

    def test_longer_context_and_balancing_activate_unique_heads(self):
        population = HeadPopulation(heads=64, active=8, context_length=3,
                                    balance=True)
        for word in ("the", "then", "there"):
            population.observe_partial("the", "r")
        active = population._active("the")
        self.assertEqual(len(active), 9)
        self.assertEqual(len(set(active)), 9)
        population.decorrelate(0.05)
        self.assertTrue(any(head for head in population.heads))

    def test_bigram_baseline_trains(self):
        baseline = BigramBaseline()
        baseline.train(["the", "then", "there"])
        self.assertEqual(baseline.predict("th"), "e")
        result = baseline.evaluate(["these"])
        self.assertGreater(result["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
