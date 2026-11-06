import hashlib
import unittest

from nanlmind_hermes.cross_resolution import (
    RESOLUTIONS, analyze_cross_resolution, feedback_records,
)


class CrossResolutionTests(unittest.TestCase):
    def row(self, event, subject, resolution, polarity="supports", obj="same"):
        return {
            "event_id": event, "subject": subject, "polarity": polarity,
            "object": obj, "provenance": {"resolution": resolution},
        }

    def test_all_ten_resolutions_have_distinct_matrix_channels(self):
        rows = [self.row("e" + str(i), "concept", name) for i, name in enumerate(RESOLUTIONS)]
        result = analyze_cross_resolution(rows)
        self.assertEqual(result["resolution_order"], list(RESOLUTIONS))
        self.assertEqual(len(result["cooccurrence_matrix"]), 10)
        self.assertTrue(all(len(line) == 10 for line in result["cooccurrence_matrix"]))
        self.assertEqual(result["concepts_compared"], 1)
        self.assertEqual(result["feedback"][0]["consistency_score"], 1.0)
        self.assertEqual(result["feedback"][0]["pair_count"], 45)
        self.assertEqual(result["cooccurrence_matrix"][0][9], 1)
        self.assertEqual(result["cooccurrence_matrix"][9][0], 1)
        self.assertEqual(result["fine_to_coarse_feedback_matrix"][0][9], 1.0)
        self.assertEqual(result["coarse_to_fine_feedback_matrix"][9][0], 1.0)

    def test_cross_resolution_contradictions_are_not_averaged_away(self):
        rows = [
            self.row("a", "gravity", "bit", "supports"),
            self.row("b", "gravity", "narrative", "refutes"),
            self.row("c", "gravity", "sentence", "supports"),
        ]
        result = analyze_cross_resolution(rows)
        item = result["feedback"][0]
        self.assertEqual(item["contradicted_pairs"], 2)
        self.assertEqual(item["aligned_pairs"], 1)
        self.assertEqual(item["content_divergent_pairs"], 0)
        self.assertAlmostEqual(item["consistency_score"], -1 / 3)
        self.assertTrue(any(p["status"] == "contradiction" for p in item["pairs"]))

    def test_feedback_hyperedge_merges_into_tenth_review_channel(self):
        import tempfile
        from pathlib import Path
        from nanlmind_hermes.memory import MemoryStore
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "universe.sqlite3", layers=10)
            digest = hashlib.sha256(b"cross-resolution-test").hexdigest()
            left = store.ingest(
                {"kind": "hypothesis", "subject": "shared-concept",
                 "polarity": "supports", "provenance": {"resolution": "lexical",
                 "alignment_id": "concept:river"}},
                source_digest=digest,
            )
            right = store.ingest(
                {"kind": "hypothesis", "subject": "shared-concept",
                 "polarity": "supports", "provenance": {"resolution": "sentence",
                 "alignment_id": "concept:river"}},
                source_digest=hashlib.sha256(b"second").hexdigest(),
            )
            analysis = analyze_cross_resolution(store.query(limit=100))
            generated = feedback_records(analysis)
            merged = store.merge(generated)
            self.assertEqual(merged["accepted"], 1)
            store.reorganize()
            subject = generated[0]["item"]["subject"]
            route = store.query(subject=subject)
            self.assertEqual(len(route), 1)
            self.assertEqual(route[0]["layer"], 9)
            self.assertEqual(store.status()["hyperedges"], 1)

    def test_feedback_records_are_deterministic_hyperedges(self):
        rows = [
            self.row("a", "river", "lexical"),
            self.row("b", "river", "sentence"),
        ]
        analysis = analyze_cross_resolution(rows)
        first = feedback_records(analysis)
        second = feedback_records(analysis)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["item"]["kind"], "equation")
        self.assertEqual(first[0]["item"]["participants"], ["a", "b"])
        self.assertEqual(first[0]["layer"], 9)


if __name__ == "__main__":
    unittest.main()
