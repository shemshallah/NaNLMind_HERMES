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


    def test_numeric_relations_discovered_only_with_alignment_and_matching_units(self):
        from nanlmind_hermes.cross_resolution import discover_numeric_relations
        rows = [
            {"event_id": "bit-v", "object": 8, "units": "count",
             "provenance": {"resolution": "bit", "alignment_id": "count:octet"}},
            {"event_id": "byte-v", "expression": {"value": 8, "units": "count"},
             "provenance": {"resolution": "byte", "alignment_id": "count:octet"}},
            {"event_id": "sentence-v", "object": 4, "units": "seconds",
             "provenance": {"resolution": "sentence", "alignment_id": "count:octet"}},
            {"event_id": "unanchored", "object": 2,
             "provenance": {"resolution": "narrative"}},
        ]
        result = discover_numeric_relations(rows)
        self.assertEqual(result["relation_count"], 1)
        relation = result["relations"][0]
        self.assertEqual(relation["relation"], "equal")
        self.assertEqual(relation["left_resolution"], "bit")
        self.assertEqual(relation["right_resolution"], "byte")
        self.assertTrue(relation["verified"])

    def test_numeric_relation_finds_exact_ratio_without_executing_expression(self):
        from nanlmind_hermes.cross_resolution import discover_numeric_relations
        rows = [
            {"event_id": "fine", "expression": {"value": 12, "op": "untrusted"},
             "units": "meters", "provenance": {"resolution": "morpheme", "alignment_id": "length:x"}},
            {"event_id": "coarse", "object": 3, "units": "meters",
             "provenance": {"resolution": "paragraph", "alignment_id": "length:x"}},
        ]
        relation = discover_numeric_relations(rows)["relations"][0]
        self.assertEqual(relation["relation"], "ratio")
        self.assertEqual(relation["result"], 4.0)

    def test_permutation_negative_control_is_deterministic(self):
        from nanlmind_hermes.cross_resolution import permutation_control
        rows = [
            self.row("a", "concept-a", "bit", "supports"),
            self.row("b", "concept-a", "byte", "supports"),
            self.row("c", "concept-b", "sentence", "refutes"),
            self.row("d", "concept-b", "narrative", "refutes"),
        ]
        for row, label in zip(rows, ["A", "A", "B", "B"]):
            row["provenance"]["alignment_id"] = label
        first = permutation_control(rows, seed=19)
        second = permutation_control(rows, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 19)
        self.assertIn("not a p-value", first["interpretation"])

    def test_negative_control_has_no_false_links_when_labels_are_unique(self):
        from nanlmind_hermes.cross_resolution import analyze_cross_resolution
        rows = [self.row("e" + str(i), "subject" + str(i), name)
                for i, name in enumerate(RESOLUTIONS)]
        for i, row in enumerate(rows):
            row["provenance"]["alignment_id"] = "unique-" + str(i)
        result = analyze_cross_resolution(rows)
        self.assertEqual(result["concepts_compared"], 0)
        self.assertEqual(sum(map(sum, result["cooccurrence_matrix"])), 0)


    def test_numeric_equation_is_emitted_as_a_review_hyperedge(self):
        rows = [
            {"event_id": "e12", "object": 12, "units": "meters",
             "provenance": {"resolution": "bit", "alignment_id": "length:known"}},
            {"event_id": "e3", "object": 3, "units": "meters",
             "provenance": {"resolution": "narrative", "alignment_id": "length:known"}},
        ]
        analysis = analyze_cross_resolution(rows)
        self.assertEqual(analysis["numeric_relations"]["relation_count"], 1)
        records = feedback_records(analysis)
        numeric = [r for r in records if r["item"].get("predicate") == "verified_numeric_relation"]
        self.assertEqual(len(numeric), 1)
        self.assertEqual(numeric[0]["item"]["expression"]["result"], 4.0)
        self.assertEqual(numeric[0]["layer"], 9)

    def test_synthetic_baselines_have_known_answer_and_negative_control(self):
        from benchmarks.cross_resolution_baseline import run_baselines
        result = run_baselines()
        self.assertEqual(result["resolution_count"], 10)
        self.assertEqual(result["pairwise_level_pairs"], 45)
        cases = result["cases"]
        aligned = cases["perfect_alignment"]["feedback"][0]
        self.assertEqual(aligned["pair_count"], 45)
        self.assertEqual(aligned["consistency_score"], 1.0)
        negative = cases["unrelated_negative_control"]
        self.assertEqual(negative["concepts_compared"], 0)
        ratio = cases["numeric_ratio_control"]["relations"][0]
        self.assertEqual(ratio["relation"], "ratio")
        self.assertEqual(ratio["result"], 4.0)


if __name__ == "__main__":
    unittest.main()
