"""Round 032 total-integration contract tests.

Round 032 adds an exact sparse retrieval layer over the same 8 MiB mirror-pair
geometry used by Rounds 029-031: concept IDs hash to an anchored pair cell,
posting lists live outside the immutable-location root, and typed hypernym
edges are traversed directly or in two hops.

The suite checks the published archive, reproduces the deterministic retrieval
comparison in Python, verifies the long-context control, persists retrieval
hyperedges through a resident universe reopen, and connects the Round 032
anchor/cycle arithmetic to the resident cycle observable and head field.

Optional modules are imported lazily.  A genuinely absent module or artifact is
a skip; a present one that violates the contract is a failure.  The native
timings in ``results.json`` are never treated as portable or phone performance
claims.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Keep ``python3 -m unittest`` usable from root.
    sys.path.insert(0, str(ROOT))

ROUND032 = ROOT / "experiments" / "round032"
FACTS_PATH = ROUND032 / "data" / "facts.tsv"
QUERIES_PATH = ROUND032 / "data" / "queries.tsv"
RESULTS_PATH = ROUND032 / "results.json"
LONG_CONTEXT_PATH = ROUND032 / "long_context_results.json"
SHA256SUMS_PATH = ROUND032 / "SHA256SUMS"
README_PATH = ROUND032 / "README.md"
NATIVE_SOURCE = ROUND032 / "native_recursive.cpp"

ROUND031 = ROOT / "experiments" / "round031" / "nan_round031"
ROUND031_RESULTS = ROUND031 / "results.json"
ROUND031_README = ROUND031 / "README.md"
ROUND031_AUDIT = ROUND031 / "audit.json"

MIB = 1024 * 1024
UNIVERSE_BYTES = 16 * MIB
ROOT_BYTES = 8_388_608
HEAD_COUNT = 256
PAIR_COUNT = 32_640
ROOT_CYCLE_COUNT = 32_385
WORD_MASK = (1 << 32) - 1
HASH_MASK = (1 << 64) - 1
MIX_ADD = 0x9E3779B97F4A7C15
MIX_MUL_1 = 0xBF58476D1CE4E5B9
MIX_MUL_2 = 0x94D049BB133111EB

EXPECTED_SHA_NAMES = {
    "native_recursive.cpp",
    "prepare.py",
    "long_context.py",
    "data/facts.tsv",
    "data/queries.tsv",
    "results.json",
    "long_context_results.json",
}


def load_module(name: str):
    """Import ``name`` lazily; skip only when the module is genuinely absent."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        raise unittest.SkipTest("%s is not present" % name)
    return importlib.import_module(name)


def require_artifacts(*paths: Path):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise unittest.SkipTest("Round 032 artifacts are absent: %s"
                                % ", ".join(missing))


@lru_cache(maxsize=1)
def read_facts() -> tuple[tuple[int, int, int], ...]:
    if not FACTS_PATH.is_file():
        raise unittest.SkipTest("Round 032 facts.tsv is absent")
    rows = []
    for line_number, line in enumerate(
            FACTS_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise AssertionError("facts.tsv:%d is not a triple" % line_number)
        rows.append(tuple(int(field) for field in fields))
    return tuple(rows)


@lru_cache(maxsize=1)
def read_queries() -> tuple[tuple[int, int, int], ...]:
    if not QUERIES_PATH.is_file():
        raise unittest.SkipTest("Round 032 queries.tsv is absent")
    rows = []
    for line_number, line in enumerate(
            QUERIES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise AssertionError("queries.tsv:%d is not a triple"
                                 % line_number)
        rows.append(tuple(int(field) for field in fields))
    return tuple(rows)


@lru_cache(maxsize=1)
def read_results() -> dict:
    if not RESULTS_PATH.is_file():
        raise unittest.SkipTest("Round 032 results.json is absent")
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def read_long_context() -> dict:
    if not LONG_CONTEXT_PATH.is_file():
        raise unittest.SkipTest("Round 032 long_context_results.json is absent")
    return json.loads(LONG_CONTEXT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# exact Python replica of native_recursive.cpp's addressing and lookup
# ---------------------------------------------------------------------------
def hash64(value: int) -> int:
    value &= HASH_MASK
    value ^= value >> 30
    value = (value * MIX_MUL_1) & HASH_MASK
    value ^= value >> 27
    value = (value * MIX_MUL_2) & HASH_MASK
    return (value ^ (value >> 31)) & HASH_MASK


def anchor(concept_id: int) -> tuple[int, int]:
    """Return the canonical anchored head pair used by Round 032."""
    key = hash64(concept_id + MIX_ADD)
    first = 1 + key % 255
    second = 1 + ((key >> 15) % 255)
    if first == second:
        second = second % 255 + 1
    if first > second:
        first, second = second, first
    return first, second


def root_pair_index(first: int, second: int) -> int:
    """``Root::idx``: a dense index over the 32,640 unordered pairs."""
    if first > second:
        first, second = second, first
    return first * (511 - first) // 2 + second - first - 1


class PairIndexReplica:
    """The C++ posting/hash memory, with stable fact ids and adjacency."""

    def __init__(self, facts):
        self.facts = tuple(facts)
        self.postings: list[list[int]] = [[] for _ in range(PAIR_COUNT)]
        self.exact: dict[tuple[int, int], int] = {}
        self.adjacency: dict[int, list[int]] = defaultdict(list)
        for identifier, (source, relation, destination) in enumerate(
                self.facts, start=1):
            index = root_pair_index(*anchor(source))
            self.postings[index].append(identifier)
            self.exact.setdefault((source, relation), identifier)
            self.adjacency[source].append(identifier)

    def get_posting(self, source: int, relation: int):
        index = root_pair_index(*anchor(source))
        for identifier in reversed(self.postings[index]):
            fact = self.facts[identifier - 1]
            if fact[0] == source and fact[1] == relation:
                return fact
        return None

    def get_hash(self, source: int, relation: int):
        identifier = self.exact.get((source, relation))
        return None if identifier is None else self.facts[identifier - 1]

    def membership_count(self) -> int:
        return sum(len(posting) for posting in self.postings)

    def two_hop(self, source: int, goal: int, relation: int) -> bool:
        for identifier in self.adjacency.get(source, ()):
            middle_relation, middle = self.facts[identifier - 1][1:]
            if middle_relation != relation:
                continue
            for second in self.adjacency.get(middle, ()):
                second_relation, destination = self.facts[second - 1][1:]
                if second_relation == relation and destination == goal:
                    return True
        return False

    def direct(self, source: int, goal: int, relation: int) -> bool:
        return any(self.facts[identifier - 1][1:] == (relation, goal)
                   for identifier in self.adjacency.get(source, ()))


# ---------------------------------------------------------------------------
# archive integrity and published numbers
# ---------------------------------------------------------------------------
class Round032ArtifactTests(unittest.TestCase):
    def setUp(self):
        require_artifacts(FACTS_PATH, QUERIES_PATH, RESULTS_PATH,
                          LONG_CONTEXT_PATH, SHA256SUMS_PATH, README_PATH,
                          NATIVE_SOURCE)

    def test_sha256_manifest_covers_and_matches_the_archive(self):
        entries = {}
        for line in SHA256SUMS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(None, 1)
            name = name.strip().lstrip("*")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            entries[name] = digest
        self.assertEqual(set(entries), EXPECTED_SHA_NAMES)
        for name, expected in entries.items():
            payload = (ROUND032 / name).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected,
                             name)

    def test_root_record_geometry_is_exactly_eight_mib(self):
        head_bytes = HEAD_COUNT * 128
        pair_bytes = PAIR_COUNT * 256
        self.assertEqual(head_bytes, 32_768)
        self.assertEqual(pair_bytes, 8_355_840)
        self.assertEqual(head_bytes + pair_bytes, ROOT_BYTES)
        self.assertEqual(PAIR_COUNT, HEAD_COUNT * (HEAD_COUNT - 1) // 2)
        self.assertEqual(ROOT_CYCLE_COUNT, PAIR_COUNT - (HEAD_COUNT - 1))
        results = read_results()
        self.assertEqual(results["root_bytes"], ROOT_BYTES)
        self.assertIn(results["root_locked"], (0, 1))
        self.assertTrue(results["pair_map_pass"])

    def test_native_results_are_internally_consistent(self):
        results = read_results()
        self.assertEqual(results["facts"], 50_268)
        self.assertEqual(results["membership_count"], results["facts"])
        self.assertEqual(results["queries"], 250_000)
        self.assertEqual(results["posting_found"], results["queries"])
        self.assertEqual(results["hash_found"], results["queries"])
        self.assertGreaterEqual(results["posting_ms"], 0.0)
        self.assertGreaterEqual(results["hash_ms"], 0.0)
        self.assertGreater(results["posting_ms"], results["hash_ms"])
        self.assertEqual(results["twohop_queries"], 2_000)
        self.assertEqual(results["twohop_found"], results["twohop_queries"])
        self.assertEqual(results["direct_found"], 0)
        self.assertTrue(math.isfinite(results["ingest_ms"]))
        self.assertGreaterEqual(results["ingest_ms"], 0.0)
        # The demonstrative triangle learns one unit; the gauge must not undo it.
        self.assertAlmostEqual(results["cycle_after_update"]
                               - results["cycle_before"], 1.0, places=5)
        self.assertAlmostEqual(results["cycle_after_gauge"],
                               results["cycle_after_update"], places=6)

    def test_facts_and_queries_parse_as_typed_hypernym_edges(self):
        facts = read_facts()
        queries = read_queries()
        self.assertEqual(len(facts), 50_268)
        self.assertEqual(len(queries), 2_000)
        self.assertTrue(all(relation == 1 for _source, relation, _dest
                            in facts))
        self.assertTrue(all(source > 0 and destination > 0
                            for source, _relation, destination in facts))
        self.assertTrue(all(source != destination
                            for source, _relation, destination in facts))
        self.assertTrue(all(relation == 1 for _source, relation, _goal
                            in queries))
        sources = {source for source, _relation, _dest in facts}
        destinations = {destination for _source, _relation, destination
                        in facts}
        self.assertTrue(all(source in sources for source, _relation, _goal
                            in queries))
        self.assertTrue(all(goal in destinations for _source, _relation, goal
                            in queries))

    def test_readme_keeps_the_claims_bounded(self):
        text = README_PATH.read_text(encoding="utf-8").lower()
        for phrase in ("not yet a fully trained retrieval-augmented lm",
                       "is faster in this microbenchmark",
                       "no unique performance advantage",
                       "does not test any neural decoder",
                       "no gpu/quantum operation",
                       "not trained from next-token loss"):
            self.assertIn(phrase, text)
        self.assertNotIn("quantum advantage", text)
        self.assertNotIn("state-of-the-art", text)


# ---------------------------------------------------------------------------
# sparse retrieval, mirror addressing, and two-hop traversal
# ---------------------------------------------------------------------------
class Round032RetrievalTests(unittest.TestCase):
    def setUp(self):
        require_artifacts(FACTS_PATH, QUERIES_PATH, RESULTS_PATH)
        self.facts = read_facts()
        self.queries = read_queries()
        self.index = PairIndexReplica(self.facts)

    def test_mirror_map_is_complete_and_pair_symmetric(self):
        indexes = set()
        for first in range(1, HEAD_COUNT):
            for second in range(first + 1, HEAD_COUNT):
                index = root_pair_index(first, second)
                self.assertEqual(index, root_pair_index(second, first))
                indexes.add(index)
        self.assertEqual(len(indexes), ROOT_CYCLE_COUNT)
        self.assertTrue(indexes <= set(range(PAIR_COUNT)))
        for source in range(1, 50_000):
            first, second = anchor(source)
            self.assertTrue(1 <= first < second < HEAD_COUNT)

    def test_pair_postings_and_hash_index_find_the_same_members(self):
        self.assertEqual(self.index.membership_count(), len(self.facts))
        self.assertEqual(len(self.index.exact), 49_321)
        keys = Counter((source, relation)
                       for source, relation, _dest in self.facts)
        unique_keys = [key for key, count in keys.items() if count == 1]
        duplicate_keys = [key for key, count in keys.items() if count > 1]
        self.assertEqual(len(unique_keys), 49_321 - 937)
        self.assertEqual(len(duplicate_keys), 937)

        for key in unique_keys:
            posting = self.index.get_posting(*key)
            hashed = self.index.get_hash(*key)
            self.assertIsNotNone(posting)
            self.assertIsNotNone(hashed)
            self.assertEqual(posting, hashed)

        destinations = defaultdict(set)
        for source, relation, destination in self.facts:
            destinations[(source, relation)].add(destination)
        for key in duplicate_keys:
            posting = self.index.get_posting(*key)
            hashed = self.index.get_hash(*key)
            self.assertIsNotNone(posting)
            self.assertIsNotNone(hashed)
            self.assertIn(posting[2], destinations[key])
            self.assertIn(hashed[2], destinations[key])

    def test_250k_native_query_pattern_finds_every_record_both_ways(self):
        step = max(1, len(self.facts) // 5_000)
        sample = [(source, relation)
                  for source, relation, _destination
                  in self.facts[::step]]
        random.Random(2026).shuffle(sample)
        self.assertEqual(len(sample), 5_027)
        posting_hits = 0
        hash_hits = 0
        trials = read_results()["queries"]
        for trial in range(trials):
            source, relation = sample[trial % len(sample)]
            posting_hits += self.index.get_posting(source, relation) is not None
            hash_hits += self.index.get_hash(source, relation) is not None
        self.assertEqual(posting_hits, trials)
        self.assertEqual(hash_hits, trials)
        self.assertEqual(posting_hits, read_results()["posting_found"])
        self.assertEqual(hash_hits, read_results()["hash_found"])

    def test_two_hop_queries_are_real_typed_chains_not_direct_edges(self):
        found = 0
        direct = 0
        for source, relation, goal in self.queries:
            found += self.index.two_hop(source, goal, relation)
            direct += self.index.direct(source, goal, relation)
        self.assertEqual(found, 2_000)
        self.assertEqual(direct, 0)
        self.assertEqual(found, read_results()["twohop_found"])
        self.assertEqual(direct, read_results()["direct_found"])
        # Every query has at least one verified middle edge pair.
        for source, relation, goal in self.queries[:50]:
            middles = [self.facts[identifier - 1][2]
                       for identifier in self.index.adjacency[source]
                       if self.facts[identifier - 1][1] == relation]
            self.assertTrue(any(
                self.index.direct(middle, goal, relation)
                for middle in middles))


class Round032LongContextTests(unittest.TestCase):
    def setUp(self):
        require_artifacts(FACTS_PATH, LONG_CONTEXT_PATH)

    def test_published_long_context_result_is_reproducible(self):
        published = read_long_context()
        self.assertEqual(published["queries"], 1_000)
        self.assertEqual(published["distractor_facts"], 512)
        self.assertEqual(published["short_context_correct"], 0)
        self.assertEqual(published["pair_index_conceptual_correct"], 1_000)
        self.assertEqual(published["hash_index_correct"], 1_000)
        self.assertTrue(math.isfinite(published["evaluation_ms"]))
        self.assertIn("not demonstrate neural language generation",
                      published["note"].lower())

        facts = read_facts()
        parents = {}
        for source, _relation, destination in facts:
            parents.setdefault(source, set()).add(destination)
        valid = [(source, next(iter(destinations)))
                 for source, destinations in parents.items()
                 if len(destinations) == 1]
        rng = random.Random(2026)
        rng.shuffle(valid)
        valid = valid[:1_000]
        answerbank = dict(valid)
        short_hits = index_hits = hash_hits = 0
        for source, correct in valid:
            distractors = [(other, destination)
                           for other, destination
                           in rng.sample(valid, min(512, len(valid)))
                           if other != source]
            short_recent = dict(distractors[-32:])
            short_hits += int(short_recent.get(source) == correct)
            index_hits += int(answerbank.get(source) == correct)
            hash_hits += int(answerbank.get(source) == correct)
        self.assertEqual(short_hits, published["short_context_correct"])
        self.assertEqual(index_hits,
                         published["pair_index_conceptual_correct"])
        self.assertEqual(hash_hits, published["hash_index_correct"])


# ---------------------------------------------------------------------------
# persistent typed retrieval hyperedges
# ---------------------------------------------------------------------------
class Round032RetrievalPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.universe_module = load_module(
            "nanlmind_hermes.resident.universe")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "round032-retrieval.nlmres"

    @staticmethod
    def _sample(facts, limit=8):
        seen_sources = set()
        seen_destinations = set()
        sample = []
        for source, relation, destination in facts:
            if source == destination:
                continue
            if source in seen_sources or destination in seen_destinations:
                continue
            seen_sources.add(source)
            seen_destinations.add(destination)
            sample.append((source, relation, destination))
            if len(sample) >= limit:
                break
        return sample

    def test_typed_hyperedges_and_retrieval_postings_survive_reopen(self):
        facts = read_facts()
        sample = self._sample(facts)
        self.assertEqual(len(sample), 8)

        universe = self.universe_module.ResidentUniverse.create(
            self.path, size=UNIVERSE_BYTES)
        receipts = []
        subjects = {}
        try:
            for source, relation, destination in sample:
                subject = "synset:%d" % source
                predicate = "hypernym:%d" % relation
                object_text = "synset:%d" % destination
                subjects[(source, relation)] = (subject, predicate,
                                                object_text)
                receipt = universe.append_relation({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_text,
                    "confidence": 1.0,
                    "head": source % HEAD_COUNT,
                })
                self.assertTrue(receipt["created"])
                receipts.append(receipt)

            before = [universe.get_relation(receipt["relation_id"])
                      for receipt in receipts]
            before_hits = {}
            for key, (subject, predicate, object_text) in subjects.items():
                result = universe.lookup(subject=subject,
                                         predicate=predicate)
                before_hits[key] = {
                    relation["relation_id"]
                    for relation in result["relations"]
                    if relation.get("object_text") == object_text
                }
                self.assertEqual(len(before_hits[key]), 1)
            universe.flush()
            generation = universe.generation
            digest = universe.digest
        finally:
            universe.close()

        reopened = self.universe_module.ResidentUniverse.open(self.path)
        try:
            status = reopened.status()
            self.assertEqual(status["counts"]["relations"], len(sample))
            self.assertEqual(status["counts"]["hyperedges"], len(sample))
            self.assertEqual(status["counts"]["pairs"], PAIR_COUNT)
            self.assertEqual(status["counts"]["root_cycles"],
                             ROOT_CYCLE_COUNT)
            self.assertFalse(status["canonical_merge"]["enabled"])
            self.assertFalse(status["canonical_merge"]["attempted"])
            self.assertEqual(reopened.generation, generation)
            self.assertEqual(reopened.digest, digest)

            for receipt, relation_before, item in zip(receipts, before, sample):
                source, relation, destination = item
                relation_after = reopened.get_relation(
                    receipt["relation_id"])
                self.assertEqual(relation_after, relation_before)
                self.assertEqual(relation_after["object_text"],
                                 "synset:%d" % destination)
                subject, predicate, object_text = subjects[
                    (source, relation)]
                result = reopened.lookup(subject=subject,
                                         predicate=predicate)
                hit_ids = {
                    view["relation_id"]
                    for view in result["relations"]
                    if view.get("object_text") == object_text
                }
                self.assertEqual(hit_ids, before_hits[(source, relation)])

                edge = reopened.get_hyperedge(receipt["hyperedge_id"])
                self.assertIsNotNone(edge)
                kinds = [member["kind"] for member in edge["members"]]
                self.assertEqual(kinds.count("leaf"), 3)
                self.assertEqual(kinds.count("pair"), 1)
                self.assertEqual(kinds.count("mirror"), 1)
                self.assertEqual(kinds.count("relation"), 1)
                self.assertLessEqual(edge["arity"], 64)
                pair_id = int(relation_after["pair_id"], 16)
                mirror_id = int(relation_after["mirror_id"], 16)
                self.assertGreaterEqual(pair_id, 0)
                self.assertLess(pair_id, PAIR_COUNT)
                self.assertGreaterEqual(mirror_id, 0)
                self.assertLess(mirror_id, 1 << 16)
        finally:
            reopened.close()


# ---------------------------------------------------------------------------
# cycle observable, head field, and the full basis
# ---------------------------------------------------------------------------
class Round032CycleObservableIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.observable = load_module(
            "nanlmind_hermes.resident.cycle_observable")
        self.head_field_module = load_module(
            "nanlmind_hermes.resident.head_field")

    def test_retrieved_anchor_cycle_is_gauge_invariant_and_keeps_heads(self):
        facts = read_facts()
        replica = PairIndexReplica(facts)
        source, relation = facts[0][0], facts[0][1]
        fact = replica.get_posting(source, relation)
        self.assertIsNotNone(fact)
        self.assertEqual(replica.get_hash(source, relation), fact)
        first, second = anchor(source)

        buffer = bytearray(self.cells_module.REGION_BYTES)
        cells = self.cells_module.ResidentCells(buffer, initialize=True)
        field = self.observable.CycleField(cells)
        heads = self.head_field_module.ResidentHeadField(cells)
        semantic = 0x12345678
        field.set_oriented(0, first, 0)
        field.set_oriented(first, second, semantic)
        field.set_oriented(0, second, 0)
        for pair in ((0, first), (first, second), (0, second)):
            field.mirror(*pair)
        self.assertEqual(field.cycle(first, second), semantic)

        heads.activate(first, 1.25, reason="round032 retrieval")
        heads.activate(second, -0.5, reason="round032 control")
        head_before = heads.read(first)
        activation_before = heads.field_activation()
        head_bytes = bytes(
            buffer[:self.cells_module.HEAD_REGION_BYTES])
        cycles_before = field.batch_cycles(limit=64)

        gauge = tuple((index * 29 + 7) & WORD_MASK
                      for index in range(HEAD_COUNT))
        self.assertEqual(field.gauge_shift(gauge), 2 * PAIR_COUNT)

        self.assertEqual(field.cycle(first, second), semantic)
        self.assertEqual(field.batch_cycles(limit=64), cycles_before)
        self.assertEqual(field.gauge_residual(), semantic)
        self.assertEqual(bytes(
            buffer[:self.cells_module.HEAD_REGION_BYTES]), head_bytes)
        self.assertEqual(heads.read(first), head_before)
        self.assertEqual(heads.field_activation(), activation_before)
        self.assertEqual(cells.read_head(first).head_id, first)
        self.assertEqual(cells.read_head(first).identity,
                         self.cells_module.identity_address(first))
        self.assertEqual(len(field.root_cycle_pairs()), ROOT_CYCLE_COUNT)


class Round032FullBasisTests(unittest.TestCase):
    def setUp(self):
        self.full = load_module(
            "nanlmind_hermes.resident.full_cycle_training")
        self.observable = load_module(
            "nanlmind_hermes.resident.cycle_observable")

    def test_full_basis_is_32385_cycles_not_a_256_effective_window(self):
        full = self.full
        self.assertEqual(full.FULL_CYCLE_COUNT, ROOT_CYCLE_COUNT)
        self.assertEqual(full.cycle_count, ROOT_CYCLE_COUNT)
        memory = full.FullCycleMemory()
        self.assertEqual(len(memory), ROOT_CYCLE_COUNT)
        report = memory.validate_full_basis()
        self.assertEqual(report["readable"], ROOT_CYCLE_COUNT)
        self.assertTrue(report["full_basis"])
        self.assertFalse(report["truncated"])
        with self.assertRaises(full.FullCycleTruncationError):
            full.FullCycleMemory([0] * 256)
        trainer = full.FullCycleTrainer()
        with self.assertRaises(full.FullCycleTruncationError):
            trainer.validate_features([0] * 256)
        self.assertEqual(ROOT_CYCLE_COUNT - 256, 32_129)
        field = self.observable.CycleField.__new__(
            self.observable.CycleField)
        self.assertEqual(field.root_cycle_count, ROOT_CYCLE_COUNT)

    def test_full_cycle_training_gauge_and_snapshot_keep_all_dimensions(self):
        full = self.full
        fixed_one = full.FIXED_ONE
        memory = full.FullCycleMemory()
        trainer = full.FullCycleTrainer(memory)
        features = {0: fixed_one, 7: -fixed_one, 32_384: fixed_one}
        targets = {0: 3 * fixed_one, 7: -2 * fixed_one,
                   32_384: fixed_one}
        before = trainer.loss(features, targets)
        receipt = trainer.sgd_step(features, targets, fixed_one)
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["changed"])
        self.assertEqual(receipt["dimension"], ROOT_CYCLE_COUNT)
        self.assertFalse(receipt["truncated"])
        self.assertFalse(receipt["canonical_merge"])
        self.assertTrue(receipt["full_basis"])
        self.assertEqual(receipt["weights_updated"], 3)
        self.assertEqual(trainer.loss(features, targets), 0.0)
        self.assertLess(trainer.loss(features, targets), before)

        snapshot = trainer.snapshot()
        digest = trainer.weight_digest()
        gauge = tuple(((index * 13) % 17) - 8
                      for index in range(HEAD_COUNT))
        gauge_receipt = trainer.apply_root_star_gauge(gauge)
        self.assertTrue(gauge_receipt["cycle_readout_unchanged"])
        self.assertTrue(gauge_receipt["weights_unchanged"])
        self.assertFalse(gauge_receipt["canonical_merge"])
        self.assertEqual(gauge_receipt["dimension"], ROOT_CYCLE_COUNT)
        self.assertEqual(trainer.weight_digest(), digest)

        restored = full.FullCycleTrainer(full.FullCycleMemory())
        restored.restore(snapshot)
        self.assertEqual(restored.weight_digest(), digest)
        self.assertEqual(restored.weight(0), 3 * fixed_one)

    def test_round031_published_artifacts_show_the_256_weight_window(self):
        require_artifacts(ROUND031_README, ROUND031_RESULTS,
                          ROUND031_AUDIT)
        text = ROUND031_README.read_text(encoding="utf-8").lower()
        self.assertIn("only 256 effective triangle weights", text)
        self.assertIn("remaining 32,129 cycle degrees", text)
        results = json.loads(ROUND031_RESULTS.read_text(encoding="utf-8"))
        plain_17 = results["runs"]["plain_17"]
        native_17 = results["runs"]["native_17"]
        plain_29 = results["runs"]["plain_29"]
        native_29 = results["runs"]["native_29"]
        self.assertGreater(native_17["test"]["bits_per_byte"],
                           plain_17["test"]["bits_per_byte"])
        self.assertGreater(native_29["test"]["bits_per_byte"],
                           plain_29["test"]["bits_per_byte"])
        self.assertEqual(native_17["params"], 40_496)
        self.assertEqual(native_29["params"], 40_496)
        self.assertTrue(native_17["gauge_test"]["audit"])
        self.assertLess(native_17["gauge_test"]["max_weight_diff"], 1e-6)
        audit = json.loads(ROUND031_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["native"]["snapshot_max_diff"], 0.0)
        self.assertTrue(audit["native"]["mirror_audit"])
        self.assertEqual(audit["wordnet"]["direct_edge_only_hits"], 0)


# ---------------------------------------------------------------------------
# canonical merge stays disabled
# ---------------------------------------------------------------------------
class Round032CanonicalMergeTests(unittest.TestCase):
    def test_canonical_merge_is_false_across_the_cycle_layers(self):
        full = load_module("nanlmind_hermes.resident.full_cycle_training")
        snapshots = load_module("nanlmind_hermes.resident.cycle_snapshot")
        hypergraph = load_module(
            "nanlmind_hermes.resident.cycle_hypergraph")
        self.assertFalse(full.CANONICAL_MERGE)
        self.assertFalse(full.FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH)
        trainer = full.FullCycleTrainer()
        self.assertFalse(trainer.stats()["canonical_merge"])
        self.assertFalse(snapshots.PERFORMS_CANONICAL_MERGE)
        cycle_graph = hypergraph.CycleHypergraph()
        self.assertFalse(cycle_graph.canonical_merge)
        self.assertFalse(cycle_graph.snapshot()["canonical_merge"])
        with self.assertRaises(hypergraph.CycleDenied) as caught:
            cycle_graph.merge((0, 1, 2), (0, 2, 3))
        self.assertEqual(caught.exception.code, "canonical_merge_disabled")


if __name__ == "__main__":
    unittest.main()
