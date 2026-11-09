"""Tests for bounded mirror-pair retrieval over typed facts.

The tests exercise both a real :class:`ResidentCells` region and a small
duck-typed fake.  Loader tests use generated TSV files plus, when present, the
Round 032 sample facts/queries.  The mirror index is never described as faster
than the exact hash comparator; the comparator is asserted to contain the same
facts and an explicit no-performance-claim marker.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.cells import (
    REGION_BYTES,
    ResidentCells,
    canonical_address,
    mirror_address,
    pair_index,
    to_fixed,
)
from nanlmind_hermes.resident.mirror_retrieval import (
    DEFAULT_SNAPSHOT_LIMIT,
    MirrorBoundsError,
    MirrorConflictError,
    MirrorCorruptionError,
    MirrorRetrieval,
    MirrorRetrievalError,
    MirrorRowError,
    SNAPSHOT_SCHEMA,
    anchor_addresses,
    anchor_pair,
    dump_snapshot,
    hash64,
    iter_fact_rows,
    iter_query_rows,
    load_facts,
    load_queries,
    load_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
ROUND032 = ROOT / "experiments" / "round032"
SAMPLE_FACTS = ROUND032 / "data" / "facts.tsv"
SAMPLE_QUERIES = ROUND032 / "data" / "queries.tsv"
SAMPLE_RESULTS = ROUND032 / "results.json"
DIGEST = hashlib.sha256(b"mirror-retrieval-test-source").hexdigest()


def find_anchor_collision(span: int = 100_000) -> tuple[int, int, tuple[int, int]]:
    """Find two distinct concept ids that share an anchored pair."""
    seen: dict[tuple[int, int], int] = {}
    for concept in range(span):
        pair = anchor_pair(concept)
        previous = seen.get(pair)
        if previous is not None and previous != concept:
            return previous, concept, pair
        seen[pair] = concept
    raise AssertionError("no anchor collision found in %d concepts" % span)


class FakeCells:
    """Minimal cells adapter that records observations and uses real geometry."""

    def __init__(self):
        self.observed: list[tuple[int, int, float, int]] = []
        self.pair_calls = 0

    def pair_index(self, first: int, second: int) -> int:
        self.pair_calls += 1
        return pair_index(first, second)

    def observe_pair(self, first: int, second: int, weight: float = 1.0,
                     relation_id: int = 0):
        self.observed.append((first, second, float(weight), relation_id))
        return None


class BucketFakeCells:
    """A deliberately lossy pair adapter: all anchors share one bucket."""

    def __init__(self):
        self.observed = []

    def pair_index(self, first: int, second: int) -> int:
        return 7

    def observe_pair(self, first: int, second: int, weight: float = 1.0,
                     relation_id: int = 0):
        self.observed.append((first, second, float(weight), relation_id))


class AnchorTests(unittest.TestCase):
    def test_anchor_is_stable_in_range_and_uses_real_pair_geometry(self):
        first, second = anchor_pair(1)
        self.assertEqual((first, second), (96, 131))
        self.assertEqual(anchor_pair(1), (first, second))
        self.assertTrue(1 <= first <= 255)
        self.assertTrue(1 <= second <= 255)
        self.assertNotEqual(first, second)
        self.assertLess(first, second)
        addresses = anchor_addresses(1)
        self.assertEqual(addresses["first"], first)
        self.assertEqual(addresses["second"], second)
        self.assertEqual(addresses["pair_index"], pair_index(first, second))
        canonical = canonical_address(first, second)
        self.assertEqual(addresses["canonical_address"], canonical)
        self.assertEqual(addresses["mirror_address"], mirror_address(canonical))
        self.assertEqual(hash64(0), 0)

    def test_anchor_collision_is_resolved_by_exact_source_and_relation(self):
        left, right, pair = find_anchor_collision()
        self.assertNotEqual(left, right)
        self.assertEqual(anchor_pair(left), pair)
        self.assertEqual(anchor_pair(right), pair)

        retrieval = MirrorRetrieval(max_posting_scan=64)
        left_id = retrieval.add_fact(left, 1, 100, 1.0, DIGEST)
        right_id = retrieval.add_fact(right, 1, 200, 1.0, DIGEST)
        self.assertNotEqual(left_id, right_id)
        self.assertEqual(retrieval.get(left, 1)["destination"], 100)
        self.assertEqual(retrieval.get(right, 1)["destination"], 200)
        self.assertEqual(retrieval.get_hash(left, 1)["destination"], 100)
        self.assertEqual(retrieval.get_hash(right, 1)["destination"], 200)
        self.assertIsNone(retrieval.get(left, 2))
        self.assertIsNone(retrieval.get_hash(left, 2))
        self.assertGreaterEqual(retrieval.status()["collision"]["collision_lists"], 1)
        audit = retrieval.audit()
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertGreaterEqual(audit["collision_lists"], 1)


class RealCellsRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells.format_region(self.buffer)
        self.retrieval = MirrorRetrieval(self.cells, max_posting_scan=128)

    def add(self, source, relation, destination, weight=1.0):
        return self.retrieval.add_fact(
            source, relation, destination, weight, DIGEST)

    def test_add_get_and_resident_cell_observation(self):
        fact_id = self.add(1, 1, 2, 2.5)
        self.assertIsInstance(fact_id, int)
        self.assertEqual(fact_id, 1)
        self.assertEqual(len(self.retrieval), 1)
        record = self.retrieval.get(1, 1)
        self.assertEqual(record["fact_id"], fact_id)
        self.assertEqual(record["source"], 1)
        self.assertEqual(record["relation"], 1)
        self.assertEqual(record["destination"], 2)
        self.assertEqual(record["weight"], 2.5)
        self.assertEqual(record["source_digest"], DIGEST)
        self.assertEqual(record["anchor"], anchor_pair(1))
        self.assertEqual(record["pair_index"], pair_index(*anchor_pair(1)))
        self.assertEqual(self.retrieval.get_hash(1, 1), record)
        self.assertIsNone(self.retrieval.get(1, 2))

        state = self.cells.pair_state(*anchor_pair(1))
        self.assertEqual(state.total_weight, to_fixed(2.5))
        self.assertGreaterEqual(self.cells.generation, 1)
        self.assertTrue(self.retrieval.audit()["ok"])

    def test_duplicate_is_idempotent_and_conflict_is_not_merged(self):
        first = self.add(5, 7, 9, 1.25)
        before = self.retrieval.snapshot()
        duplicate = self.add(5, 7, 9, 1.25)
        self.assertEqual(duplicate, first)
        self.assertEqual(len(self.retrieval), 1)
        self.assertEqual(self.retrieval.status()["stats"]["duplicate_facts"], 1)

        with self.assertRaises(MirrorConflictError) as raised:
            self.add(5, 7, 10, 1.25)
        self.assertEqual(raised.exception.code, "fact_conflict")
        self.assertFalse(raised.exception.details["canonical_merge"])
        self.assertEqual(len(self.retrieval), 1)
        self.assertEqual(self.retrieval.get(5, 7)["destination"], 9)
        self.assertEqual(self.retrieval.snapshot()["facts"], before["facts"])
        self.assertEqual(self.retrieval.status()["stats"]["conflicts"], 1)

        with self.assertRaises(MirrorRetrievalError) as merged:
            self.retrieval.merge()
        self.assertEqual(merged.exception.code, "canonical_merge_disabled")
        self.assertFalse(self.retrieval.canonical_merge)

    def test_mirror_postings_and_hash_comparator_return_same_facts(self):
        left, right, _pair = find_anchor_collision()
        sources = [left, right, 11, 12, 13]
        for offset, source in enumerate(sources):
            self.add(source, 1, 1000 + offset, 1.0 + offset / 10)
            self.add(source, 2, 2000 + offset, 0.5)
        self.assertEqual(len(self.retrieval), len(sources) * 2)
        for fact_id in range(1, len(self.retrieval) + 1):
            fact = self.retrieval.snapshot(
                limit=DEFAULT_SNAPSHOT_LIMIT)["facts"][fact_id - 1]
            self.assertEqual(self.retrieval.get(fact["source"], fact["relation"]),
                             self.retrieval.get_hash(fact["source"],
                                                     fact["relation"]))
        audit = self.retrieval.audit()
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertTrue(audit["comparator"]["same_facts"])
        self.assertEqual(audit["comparator"]["performance_claim"], "none")
        self.assertIn("not measured or asserted",
                      audit["comparator"]["note"])

    def test_two_hop_requires_two_typed_edges_and_neighbors_are_bounded(self):
        # 10 -> 20 -> 30, 10 -> 40 -> 50, plus a different relation and a
        # direct one-edge link that must not satisfy two_hop by itself.
        self.add(10, 1, 20)
        self.add(20, 1, 30)
        self.add(10, 1, 40)
        self.add(40, 1, 50)
        self.add(10, 2, 60)
        self.add(60, 2, 70)
        self.add(10, 1, 80)  # direct only

        self.assertTrue(self.retrieval.two_hop(10, 30, 1))
        self.assertTrue(self.retrieval.two_hop(10, 50, 1))
        self.assertFalse(self.retrieval.two_hop(10, 80, 1))
        self.assertFalse(self.retrieval.two_hop(10, 20, 1))
        self.assertTrue(self.retrieval.two_hop(10, 70, 2))
        self.assertFalse(self.retrieval.two_hop(10, 70, 1))

        destinations = {row["destination"] for row in self.retrieval.neighbors(10)}
        self.assertEqual(destinations, {20, 40, 60, 80})
        typed = {row["destination"] for row in self.retrieval.neighbors(10, 1)}
        self.assertEqual(typed, {20, 40, 80})
        limited = self.retrieval.neighbors(10, limit=1)
        self.assertEqual(len(limited), 1)
        detail = self.retrieval.neighbors_detail(10, limit=2)
        self.assertEqual(detail["returned"], 2)
        self.assertTrue(detail["truncated"])
        with self.assertRaises(MirrorBoundsError):
            self.retrieval.neighbors(10, limit=10_000)
        with self.assertRaises(MirrorRetrievalError):
            self.retrieval.neighbors("bad")
        self.assertTrue(self.retrieval.audit()["ok"])

    def test_bounds_are_explicit(self):
        bounded = MirrorRetrieval(capacity=1)
        bounded.add_fact(1, 1, 2, 1.0, DIGEST)
        with self.assertRaises(MirrorBoundsError) as raised:
            bounded.add_fact(2, 1, 3, 1.0, DIGEST)
        self.assertEqual(raised.exception.code, "fact_capacity")
        self.assertEqual(len(bounded), 1)

        neighbor_bound = MirrorRetrieval(max_neighbor_scan=1)
        neighbor_bound.add_fact(1, 1, 2, 1.0, DIGEST)
        neighbor_bound.add_fact(1, 2, 3, 1.0, DIGEST)
        with self.assertRaises(MirrorBoundsError) as raised:
            neighbor_bound.neighbors(1)
        self.assertEqual(raised.exception.code, "neighbor_scan_limit")

        expansion_bound = MirrorRetrieval(max_expansions=1)
        expansion_bound.add_fact(1, 1, 2, 1.0, DIGEST)
        expansion_bound.add_fact(2, 1, 3, 1.0, DIGEST)
        with self.assertRaises(MirrorBoundsError) as raised:
            expansion_bound.two_hop(1, 3, 1)
        self.assertEqual(raised.exception.code, "two_hop_limit")

        posting_bound = MirrorRetrieval(max_posting_scan=1)
        left, right, _pair = find_anchor_collision()
        posting_bound.add_fact(left, 1, 100, 1.0, DIGEST)
        posting_bound.add_fact(right, 1, 200, 1.0, DIGEST)
        with self.assertRaises(MirrorBoundsError) as raised:
            posting_bound.get(left, 1)
        self.assertEqual(raised.exception.code, "posting_scan_limit")

    def test_snapshot_json_restore_and_atomic_adapter(self):
        left, right, _pair = find_anchor_collision()
        self.add(1, 1, 2, 3.0)
        self.add(2, 1, 3, 1.5)
        self.add(left, 1, 4, 0.5)
        self.add(right, 1, 5, 0.25)
        snapshot = self.retrieval.snapshot()
        self.assertEqual(snapshot["schema"], SNAPSHOT_SCHEMA)
        self.assertEqual(snapshot["returned"], 4)
        self.assertFalse(snapshot["truncated"])
        self.assertFalse(snapshot["canonical_merge"])
        self.assertTrue(snapshot["comparator"]["same_facts"])
        encoded = json.dumps(snapshot, allow_nan=False)
        self.assertIn("fact_count", encoded)

        restored = MirrorRetrieval.from_snapshot(json.loads(encoded))
        self.assertEqual(len(restored), len(self.retrieval))
        for source, relation in ((1, 1), (2, 1), (left, 1), (right, 1)):
            self.assertEqual(restored.get(source, relation),
                             self.retrieval.get(source, relation))
        self.assertTrue(restored.audit()["ok"])
        self.assertEqual(restored.last_load["restored_facts"], 4)
        self.assertTrue(snapshot["generation"] > 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mirror.json"
            receipt = dump_snapshot(path, self.retrieval)
            self.assertEqual(receipt["facts"], 4)
            self.assertEqual(receipt["bytes"], path.stat().st_size)
            from_file = load_snapshot(path)
            self.assertEqual(len(from_file), 4)
            self.assertTrue(from_file.audit()["ok"])

    def test_truncated_snapshot_is_rejected_but_can_be_explicit(self):
        self.add(1, 1, 2)
        self.add(2, 1, 3)
        snapshot = self.retrieval.snapshot(limit=1)
        self.assertTrue(snapshot["truncated"])
        with self.assertRaises(MirrorBoundsError) as raised:
            MirrorRetrieval.from_snapshot(snapshot)
        self.assertEqual(raised.exception.code, "truncated_snapshot")
        partial = MirrorRetrieval.from_snapshot(snapshot, allow_truncated=True)
        self.assertEqual(len(partial), 1)
        self.assertEqual(partial.get(1, 1)["destination"], 2)

        bad = dict(snapshot)
        bad["schema"] = "wrong"
        with self.assertRaises(MirrorCorruptionError):
            MirrorRetrieval.from_snapshot(bad)

    def test_projection_has_no_raw_pointers_and_audit_is_complete(self):
        self.add(1, 1, 2)
        snapshot = self.retrieval.snapshot()
        for fact in snapshot["facts"]:
            for value in fact.values():
                self.assertNotIsInstance(value, (bytes, bytearray, memoryview))
        json.dumps(snapshot, allow_nan=False)
        audit = self.retrieval.audit()
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["facts_checked"], 1)
        self.assertEqual(audit["exact_index_size"], 1)
        self.assertEqual(audit["posting_entries"], 1)
        self.assertFalse(audit["truncated"])
        self.assertFalse(audit["canonical_merge"])
        self.assertEqual(audit["comparator"]["indexed_facts"], 1)


class FakeCellsTests(unittest.TestCase):
    def test_fake_cells_receive_pair_observations(self):
        fake = FakeCells()
        retrieval = MirrorRetrieval(fake)
        fact_id = retrieval.add_fact(1, 1, 2, 1.75, DIGEST)
        self.assertEqual(fact_id, 1)
        self.assertGreater(fake.pair_calls, 0)
        self.assertEqual(len(fake.observed), 1)
        first, second, weight, relation_id = fake.observed[0]
        self.assertEqual((first, second), anchor_pair(1))
        self.assertEqual(weight, 1.75)
        self.assertEqual(relation_id, 1)
        self.assertEqual(retrieval.get(1, 1)["destination"], 2)
        self.assertTrue(retrieval.audit()["ok"])

    def test_lossy_pair_bucket_still_resolves_exact_keys(self):
        fake = BucketFakeCells()
        retrieval = MirrorRetrieval(fake, max_posting_scan=64)
        left, right, _pair = find_anchor_collision()
        retrieval.add_fact(left, 1, 10, 1.0, DIGEST)
        retrieval.add_fact(right, 1, 20, 1.0, DIGEST)
        retrieval.add_fact(left, 2, 30, 1.0, DIGEST)
        self.assertEqual(retrieval.get(left, 1)["destination"], 10)
        self.assertEqual(retrieval.get(right, 1)["destination"], 20)
        self.assertEqual(retrieval.get(left, 2)["destination"], 30)
        self.assertEqual(retrieval.get_hash(right, 1)["destination"], 20)
        audit = retrieval.audit()
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertEqual(audit["posting_lists"], 1)
        self.assertGreaterEqual(audit["collision_keys"], 3)


class LoaderTests(unittest.TestCase):
    def make_temp(self, content: str, name: str = "facts.tsv") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_facts_streams_with_limit_and_content_digest(self):
        content = "\n# sample facts\n1 1 2\n\n2 1 3\n3 1 4\n"
        path = self.make_temp(content)
        retrieval = load_facts(path, limit=2)
        self.assertEqual(len(retrieval), 2)
        self.assertEqual(retrieval.get(1, 1)["destination"], 2)
        self.assertEqual(retrieval.get(2, 1)["destination"], 3)
        self.assertIsNone(retrieval.get(3, 1))
        report = retrieval.last_load
        self.assertEqual(report["facts_added"], 2)
        self.assertEqual(report["rows_read"], 2)
        self.assertTrue(report["truncated"])
        self.assertEqual(report["source_digest"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(report["fact_ids"], [1, 2])
        self.assertFalse(report["canonical_merge"])

        complete = load_facts(path)
        self.assertEqual(len(complete), 3)
        self.assertFalse(complete.last_load["truncated"])
        self.assertEqual(complete.get(3, 1)["destination"], 4)

    def test_malformed_rows_raise_or_skip(self):
        cases = (
            ("1 1\n", "malformed_row"),
            ("1 1 2 3\n", "malformed_row"),
            ("a 1 2\n", "malformed_row"),
            ("-1 1 2\n", "integer_out_of_bounds"),
            ("1 1 999999999999999999999999999\n", "integer_out_of_bounds"),
        )
        for content, code in cases:
            path = self.make_temp(content)
            with self.assertRaises(MirrorRowError) as raised:
                load_facts(path)
            self.assertEqual(raised.exception.code, code, content)
            self.assertEqual(raised.exception.details["line"], 1)

        path = self.make_temp("1 1 2\nbad row here\n3 1 4\n")
        with self.assertRaises(MirrorRowError):
            load_facts(path)
        skipped = load_facts(path, skip_malformed=True)
        self.assertEqual(len(skipped), 2)
        self.assertEqual(skipped.last_load["skipped_malformed"], 1)
        self.assertIsNone(skipped.get(2, 1))

        queries_path = self.make_temp("1 1 2\n1 1\n3 1 4\n", "queries.tsv")
        with self.assertRaises(MirrorRowError):
            load_queries(queries_path)
        queries = load_queries(queries_path, skip_malformed=True)
        self.assertEqual([tuple(row) for row in queries],
                         [(1, 1, 1, 2), (3, 1, 3, 4)])

    def test_loader_bounds_and_query_limit(self):
        path = self.make_temp("1 1 2\n2 1 3\n3 1 4\n")
        with self.assertRaises(MirrorBoundsError) as raised:
            load_facts(path, max_line_bytes=3)
        self.assertEqual(raised.exception.code, "line_too_long")

        with self.assertRaises(MirrorBoundsError) as raised:
            load_facts(path, source_digest=DIGEST, max_rows=2)
        self.assertEqual(raised.exception.code, "row_limit")

        with self.assertRaises(MirrorBoundsError) as raised:
            load_facts(path, max_bytes=8)
        self.assertEqual(raised.exception.code, "file_size_limit")

        queries = load_queries(path, limit=2)
        self.assertEqual(len(queries), 2)
        self.assertTrue(queries.report["truncated"])
        self.assertEqual(queries[0], (1, 1, 1, 2))
        self.assertEqual(queries.report["queries"], 2)

        rows = list(iter_fact_rows(path, max_rows=3))
        self.assertEqual([row["destination"] for row in rows], [2, 3, 4])
        query_rows = list(iter_query_rows(path, max_rows=3))
        self.assertEqual([row.destination for row in query_rows], [2, 3, 4])

    def test_load_into_existing_retrieval_and_capacity(self):
        path = self.make_temp("1 1 2\n2 1 3\n")
        retrieval = MirrorRetrieval(capacity=2)
        loaded = load_facts(path, retrieval)
        self.assertIs(loaded, retrieval)
        self.assertEqual(len(loaded), 2)
        again = load_facts(path, retrieval)
        self.assertEqual(len(again), 2)
        self.assertEqual(again.last_load["duplicates"], 2)

        larger = self.make_temp("1 1 2\n2 1 3\n3 1 4\n")
        full = MirrorRetrieval(capacity=2)
        with self.assertRaises(MirrorBoundsError) as raised:
            load_facts(larger, full)
        self.assertEqual(raised.exception.code, "fact_capacity")
        self.assertEqual(len(full), 2)

    @unittest.skipUnless(SAMPLE_FACTS.is_file() and SAMPLE_QUERIES.is_file(),
                         "Round 032 sample TSV data is absent")
    def test_round032_sample_facts_and_two_hop_queries(self):
        retrieval = load_facts(SAMPLE_FACTS, source_digest="round032-sample")
        expected_facts = sum(
            1 for line in SAMPLE_FACTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#"))
        self.assertEqual(len(retrieval), expected_facts)
        self.assertEqual(retrieval.last_load["facts_added"], expected_facts)
        self.assertFalse(retrieval.last_load["truncated"])
        self.assertTrue(retrieval.audit()["ok"])

        first_line = next(line for line in SAMPLE_FACTS.read_text(
            encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#"))
        source, relation, destination = map(int, first_line.split())
        record = retrieval.get(source, relation)
        self.assertIsNotNone(record)
        self.assertEqual(record["destination"], destination)
        self.assertEqual(retrieval.get_hash(source, relation), record)

        queries = load_queries(SAMPLE_QUERIES)
        self.assertGreater(len(queries), 0)
        for query in queries:
            self.assertTrue(
                retrieval.two_hop(query.source, query.destination,
                                  query.relation),
                "two-hop query %r was not found" % (query,))

        for fact in retrieval.snapshot(limit=128)["facts"]:
            self.assertEqual(
                retrieval.get(fact["source"], fact["relation"]),
                retrieval.get_hash(fact["source"], fact["relation"]))

    @unittest.skipUnless(SAMPLE_RESULTS.is_file(), "Round 032 results are absent")
    def test_native_results_report_comparator_without_speed_claim(self):
        results = json.loads(SAMPLE_RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(results["posting_found"], results["queries"])
        self.assertEqual(results["hash_found"], results["queries"])
        self.assertGreater(results["posting_ms"], results["hash_ms"])
        retrieval = MirrorRetrieval()
        comparator = retrieval.audit()["comparator"]
        self.assertEqual(comparator["performance_claim"], "none")
        self.assertIn("not measured or asserted", comparator["note"])


if __name__ == "__main__":
    unittest.main()
