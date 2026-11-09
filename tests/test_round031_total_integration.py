"""Round 031 total-integration contract tests.

Round 031 trains a byte-level recurrent language model whose 16 x 16 readout
matrix is read from, and written back to, an 8 MiB native mirror-pair root.
The published run exposes only 256 effective triangle weights.  This suite
separates that measured artifact from the repository's full resident cycle
basis and checks both:

* the published Round 031 artifacts, snapshots and SHA-256 manifest;
* the exact 32,385-dimensional root-cycle mapping over 32,640 resident pairs;
* raw 32-bit word preservation, gauge cancellation and persistence/reopen;
* a full-basis fixed-point training step over all 32,385 root cycles;
* hypergraph incidence for every root cycle and a real store holding more
  than 256 cycle hyperedges;
* the explicit native-256 versus resident-32,385 distinction;
* optional full-cycle modules and canonical-merge-false authority flags.

Every optional module is imported lazily.  A genuinely absent module skips;
a present module or artifact that violates the contract fails.  No test
writes to the repository, and no native timing is treated as a portable or
phone performance claim.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Keep ``python3 -m unittest`` usable from root.
    sys.path.insert(0, str(ROOT))

ROUND031 = ROOT / "experiments" / "round031" / "nan_round031"
RESULTS = ROUND031 / "results.json"
AUDIT = ROUND031 / "audit.json"
TRANSFER = ROUND031 / "transfer_results.json"
SHA256SUMS = ROUND031 / "SHA256SUMS"
NATIVE_SOURCE = ROUND031 / "native_cycle.cpp"
README = ROUND031 / "README.md"

HEAD_COUNT = 256
PAIR_COUNT = 32_640
ROOT_CYCLE_COUNT = 32_385
NATIVE_DIMENSIONS = 16
NATIVE_CHANNELS = 16
NATIVE_EFFECTIVE_WEIGHTS = NATIVE_DIMENSIONS * NATIVE_CHANNELS
ROOT_BYTES = 8_388_608
SNAPSHOT_BYTES = ROOT_BYTES + 8
SNAPSHOT_MAGIC = b"NANCYC30"
HEAD_BYTES = 128
PAIR_BYTES = 256
DIRECTION_WORDS = 32
MIB = 1024 * 1024
UNIVERSE_BYTES = 16 * MIB


def load_module(name: str):
    """Import ``name`` lazily; skip only when it is genuinely absent."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        raise unittest.SkipTest("%s is not present" % name)
    return importlib.import_module(name)


def require_artifacts(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise unittest.SkipTest("Round 031 artifacts are absent: %s"
                                % ", ".join(missing))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in SHA256SUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.strip()
        if name in entries:
            raise AssertionError("duplicate checksum entry: %s" % name)
        entries[name] = digest
    return entries


def resident_field(testcase):
    """Build a fresh raw cycle field over a real resident cell region."""
    cells_module = load_module("nanlmind_hermes.resident.cells")
    observable = load_module("nanlmind_hermes.resident.cycle_observable")
    cells = cells_module.ResidentCells(
        bytearray(cells_module.REGION_BYTES), initialize=True)
    return cells_module, observable, cells, observable.CycleField(cells)


def fill_full_pair_field(field, observable) -> dict[tuple[int, int], int]:
    """Write one deterministic raw word to every canonical pair orientation."""
    words: dict[tuple[int, int], int] = {}
    for index in range(PAIR_COUNT):
        identity = observable.pair_identity(index)
        word = (index * 0x9E3779B1 + 0x13579BDF) & 0xFFFFFFFF
        field.set_oriented(identity.first, identity.second, word)
        words[(identity.first, identity.second)] = word
    return words


def snapshot_digest(snapshot) -> str:
    digest = hashlib.sha256()
    for pair in snapshot["pairs"]:
        digest.update(struct.pack("<III", pair.index, pair.forward,
                                  pair.reverse))
    for cycle in snapshot["cycles"]:
        digest.update(struct.pack("<III", cycle.left, cycle.right,
                                  cycle.word))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# full 32,385 mapping and storage
# ---------------------------------------------------------------------------
class FullCycleBasisTests(unittest.TestCase):
    def setUp(self):
        (self.cells_module, self.observable, self.cells,
         self.field) = resident_field(self)

    def test_all_32385_root_cycles_map_one_to_one(self):
        observable = self.observable
        self.assertEqual(observable.ROOT_CYCLE_COUNT, 32_385)
        self.assertEqual(observable.PAIR_CANONICALS.__len__(), 32_640)
        self.assertEqual(len(set(observable.PAIR_CANONICALS)), 32_640)
        self.assertEqual(PAIR_COUNT - (HEAD_COUNT - 1), ROOT_CYCLE_COUNT)

        expected = {(left, right)
                    for left in range(1, HEAD_COUNT)
                    for right in range(left + 1, HEAD_COUNT)}
        self.assertEqual(len(expected), ROOT_CYCLE_COUNT)
        pairs = self.field.root_cycle_pairs()
        self.assertEqual(len(pairs), ROOT_CYCLE_COUNT)
        self.assertEqual(set(pairs), expected)
        self.assertEqual(pairs[-1], (254, 255))

        samples = self.field.batch_cycles()
        self.assertEqual(len(samples), ROOT_CYCLE_COUNT)
        seen = {(sample.left, sample.right) for sample in samples}
        self.assertEqual(len(seen), ROOT_CYCLE_COUNT)
        self.assertTrue(all(sample.root == 0 for sample in samples))
        self.assertTrue(all(sample.left < sample.right for sample in samples))

        for index in (0, 1, 256, 32_639):
            identity = observable.pair_identity(index)
            self.assertEqual(identity.index, index)
            self.assertEqual(identity.canonical,
                             observable.PAIR_CANONICALS[index])
            self.assertEqual(identity.mirror,
                             (int("{0:016b}".format(identity.canonical)[::-1],
                                  2)))

    def test_all_pairs_and_all_cycle_words_are_stored_and_recomputable(self):
        words = fill_full_pair_field(self.field, self.observable)
        snapshot = self.field.snapshot()
        self.assertEqual(snapshot["mode"], "raw-u32")
        self.assertEqual(snapshot["pair_count"], 32_640)
        self.assertEqual(snapshot["root_cycle_count"], 32_385)
        self.assertEqual(len(snapshot["pairs"]), 32_640)
        self.assertEqual(len(snapshot["cycles"]), 32_385)

        for index, pair in enumerate(snapshot["pairs"]):
            self.assertEqual(pair.index, index)
            identity = self.observable.pair_identity(index)
            self.assertEqual(
                pair.forward,
                words[(identity.first, identity.second)])

        for sample in snapshot["cycles"]:
            expected = (words.get((0, sample.left), 0)
                        + words.get((sample.left, sample.right), 0)
                        - words.get((0, sample.right), 0)) & 0xFFFFFFFF
            self.assertEqual(sample.word, expected)
            self.assertEqual(
                sample.word,
                self.field.cycle(sample.left, sample.right))


class No256OnlyPathTests(unittest.TestCase):
    def setUp(self):
        (self.cells_module, self.observable, self.cells,
         self.field) = resident_field(self)

    def test_batches_and_snapshots_expose_more_than_256_cycles(self):
        first_256 = self.field.batch_cycles(limit=256)
        next_257 = self.field.batch_cycles(limit=257)
        self.assertEqual(len(first_256), 256)
        self.assertEqual(len(next_257), 257)
        self.assertEqual(next_257[:256], first_256)
        self.assertNotIn(next_257[256], first_256)

        self.assertEqual(self.field.root_cycle_count, 32_385)
        self.assertEqual(len(self.field.root_cycle_pairs()), 32_385)
        self.assertGreater(self.field.root_cycle_count, 256)
        self.assertNotEqual(self.field.root_cycle_count, 256)
        self.assertEqual(len(self.field.snapshot()["cycles"]), 32_385)

        with self.assertRaises(self.observable.CycleFieldRangeError):
            self.field.batch_cycles(limit=32_386)
        with self.assertRaises(self.observable.CycleFieldRangeError):
            self.field.batch_cycles(limit=-1)

    def test_fixed_point_trainer_accepts_batches_above_256(self):
        training = load_module("nanlmind_hermes.resident.cycle_training")
        field = training.IntegerLaneField()
        trainer = training.CycleTrainer(
            field, learning_rate=training.fixed_ratio(1, 256),
            normalize_gradient=True, max_batch_size=512,
            max_pairs_per_step=2048)
        root_pairs = [(left, right)
                      for left in range(1, 255)
                      for right in range(left + 1, 256)]
        observations = [(left, right, 3 * training.FIXED_ONE)
                        for left, right in root_pairs[:300]]
        before = trainer.loss_units(observations)
        receipt = trainer.step(observations)

        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["observations"], 300)
        self.assertGreater(receipt["pairs_updated"], 256)
        self.assertLess(trainer.loss_units(observations), before)
        self.assertEqual(field.generation, 1)


# ---------------------------------------------------------------------------
# raw words, gauge, full-basis training
# ---------------------------------------------------------------------------
class RawWordPreservationTests(unittest.TestCase):
    def setUp(self):
        (self.cells_module, self.observable, self.cells,
         self.field) = resident_field(self)

    def test_nan_bit_patterns_and_all_32_bit_words_survive_unchanged(self):
        words = (0x00000000, 0x00000001, 0x7FC00000, 0x7FC00001,
                 0x7F800001, 0xFFC00000, 0x80000000, 0xDEADBEEF,
                 0xFFFFFFFE, 0xFFFFFFFF)
        heads_before = bytes(self.cells.buffer[:
                                              self.cells_module.HEAD_REGION_BYTES])
        for first, second in ((1, 2), (200, 5), (17, 136), (254, 255)):
            for word in words:
                self.field.set_oriented(second, first, 0)
                self.assertEqual(
                    self.field.set_oriented(first, second, word), word)
                self.assertIsInstance(self.field.get_oriented(first, second),
                                      int)
                self.assertEqual(self.field.get_oriented(first, second), word)
                self.assertEqual(self.field.get_oriented(second, first), 0)
                state = self.cells.pair_state(first, second)
                chosen = (state.forward
                          if self.cells_module.address_of(first, second)
                          == state.canonical else state.reverse)
                self.assertEqual(chosen.weight & 0xFFFFFFFF, word)
                self.assertEqual(chosen.amplitude, 0)
                self.assertEqual(chosen.phase, 0)
                self.assertEqual(chosen.source, first)
                self.assertEqual(chosen.target, second)

                mirror = self.field.mirror(first, second)
                self.assertEqual(mirror, (-word) & 0xFFFFFFFF)
                self.assertEqual(self.field.get_oriented(second, first),
                                 mirror)
                self.assertEqual(self.field.get_oriented(first, second), word)
        self.assertEqual(
            bytes(self.cells.buffer[:self.cells_module.HEAD_REGION_BYTES]),
            heads_before)

    def test_adjacent_nan_payload_words_are_not_float_collapsed(self):
        self.field.set_oriented(3, 4, 0x7FC00000)
        self.field.set_oriented(5, 6, 0x7FC00001)
        self.assertNotEqual(self.field.get_oriented(3, 4),
                            self.field.get_oriented(5, 6))
        snapshot = self.field.snapshot()
        by_index = {pair.index: pair.forward for pair in snapshot["pairs"]}
        self.assertEqual(by_index[self.observable.pair_identity(
            self.cells_module.pair_index(3, 4)).index], 0x7FC00000)
        self.assertEqual(by_index[self.observable.pair_identity(
            self.cells_module.pair_index(5, 6)).index], 0x7FC00001)


class GaugeInvarianceTests(unittest.TestCase):
    def setUp(self):
        (self.cells_module, self.observable, self.cells,
         self.field) = resident_field(self)

    def test_full_basis_is_exactly_invariant_under_a_large_vertex_gauge(self):
        fill_full_pair_field(self.field, self.observable)
        before = self.field.batch_cycles()
        residual_before = self.field.gauge_residual()
        edge_before = self.field.get_oriented(0, 1)
        gauge = [((index * 0x01010101) ^ 0xA5A5A5A5) & 0xFFFFFFFF
                 for index in range(HEAD_COUNT)]

        moved = self.field.gauge_shift(gauge)
        self.assertEqual(moved, 2 * PAIR_COUNT)
        after = self.field.batch_cycles()
        self.assertEqual(after, before)
        self.assertEqual(self.field.gauge_residual(), residual_before)
        edge_after = self.field.get_oriented(0, 1)
        self.assertEqual((edge_after - edge_before
                          - (gauge[0] - gauge[1])) & 0xFFFFFFFF, 0)

        # The raw lane moved; every one of the 32,385 cycle words did not.
        self.assertNotEqual(edge_after, edge_before)
        self.assertTrue(all(left.word == right.word
                            for left, right in zip(before, after)))


class FullBasisTrainingTests(unittest.TestCase):
    def test_one_full_basis_step_reduces_loss_over_all_32385_cycles(self):
        training = load_module("nanlmind_hermes.resident.cycle_training")
        observations = [(left, right, 3 * training.FIXED_ONE)
                        for left in range(1, 255)
                        for right in range(left + 1, 256)]
        self.assertEqual(len(observations), 32_385)

        field = training.IntegerLaneField()
        trainer = training.CycleTrainer(
            field, learning_rate=training.fixed_ratio(1, 1024),
            normalize_gradient=True, max_batch_size=40_000,
            max_pairs_per_step=70_000)
        before = trainer.loss_units(observations)
        receipt = trainer.step(observations)
        after = trainer.loss_units(observations)

        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["observations"], 32_385)
        self.assertGreater(receipt["pairs_updated"], 32_000)
        self.assertLess(after, before)
        self.assertEqual(receipt["generation_after"], 1)
        self.assertIs(receipt["canonical_merge"], False)
        self.assertIs(receipt["semantic_truth"], False)


# ---------------------------------------------------------------------------
# persistence and hypergraph coverage
# ---------------------------------------------------------------------------
class PersistenceReopenTests(unittest.TestCase):
    def setUp(self):
        self.universe_module = load_module(
            "nanlmind_hermes.resident.universe")
        (self.cells_module, self.observable, _ignored,
         _ignored_field) = resident_field(self)
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.path = Path(self._temporary.name) / "round031-full-cycle.nlmres"
        self.universe = self.universe_module.ResidentUniverse.create(
            self.path, size=UNIVERSE_BYTES)

    def tearDown(self):
        universe = getattr(self, "universe", None)
        if universe is not None:
            try:
                universe.close()
            except Exception:
                pass

    def test_every_cycle_word_survives_close_and_reopen(self):
        field = self.observable.CycleField(self.universe.cells)
        fill_full_pair_field(field, self.observable)
        before = self.field_snapshot(field)
        before_digest = snapshot_digest(before)

        self.universe.flush()
        generation = self.universe.generation
        digest = self.universe.digest
        self.universe.close()

        self.universe = self.universe_module.ResidentUniverse.open(self.path)
        reopened = self.observable.CycleField(self.universe.cells)
        after = self.field_snapshot(reopened)
        self.assertEqual(snapshot_digest(after), before_digest)
        self.assertEqual(len(after["cycles"]), 32_385)
        self.assertEqual(after["cycles"], before["cycles"])
        self.assertEqual(after["pairs"], before["pairs"])
        self.assertEqual(self.universe.generation, generation)
        self.assertEqual(self.universe.digest, digest)
        self.assertFalse(
            self.universe.status()["canonical_merge"]["enabled"])

    @staticmethod
    def field_snapshot(field):
        return field.snapshot()


class CycleHypergraphTests(unittest.TestCase):
    def setUp(self):
        self.hypergraph = load_module(
            "nanlmind_hermes.resident.cycle_hypergraph")

    def test_symbolic_incidence_covers_all_32385_cycles_and_32640_pairs(self):
        module = self.hypergraph
        cycle_ids = set()
        leaf_ids = set()
        covered_pairs = set()
        for left in range(1, 255):
            for right in range(left + 1, 256):
                cycle = module.normalize_cycle((0, left, right))
                self.assertEqual(cycle, (0, left, right))
                edges = module.cycle_edges(cycle)
                self.assertEqual(len(edges), 3)
                self.assertEqual(len(set(edges)), 3)
                for first, second in edges:
                    reference = module.pair_reference(first, second)
                    self.assertGreaterEqual(reference, 0)
                    self.assertLess(reference, 32_640)
                    covered_pairs.add(tuple(sorted((first, second))))
                cycle_ids.add(module.cycle_hyperedge_id(cycle))
                leaf_ids.add(module.cycle_leaf_id(cycle))
        self.assertEqual(len(cycle_ids), 32_385)
        self.assertEqual(len(leaf_ids), 32_385)
        self.assertEqual(len(covered_pairs), 32_640)

    def test_real_store_holds_more_than_256_cycle_hyperedges(self):
        module = self.hypergraph
        relations = load_module("nanlmind_hermes.resident.relations")
        leaves = load_module("nanlmind_hermes.resident.leaves")
        capacity = 260
        index_capacity = 1
        while index_capacity < max(16, capacity * 2):
            index_capacity <<= 1
        leaf_buffer = bytearray(
            leaves.HEADER_BYTES + capacity * leaves.RECORD_BYTES
            + index_capacity * leaves.INDEX_ENTRY_BYTES)
        leaf_store = leaves.LeafStore.create(
            leaf_buffer, capacity=capacity)
        relation_store = relations.RelationStore.allocate(
            relation_capacity=capacity, hyperedge_capacity=capacity,
            arena_bytes=capacity * 9 * 40)
        hypergraph = module.CycleHypergraph(
            relations=relation_store, leaves=leaf_store)
        source = hashlib.sha256(b"round031-full-cycle-test").hexdigest()

        added = 0
        for left in range(1, 255):
            for right in range(left + 1, 256):
                if added >= capacity:
                    break
                receipt = hypergraph.add_cycle(
                    left, right, left * 1000 + right, source)
                self.assertTrue(receipt["created"])
                added += 1
            if added >= capacity:
                break

        self.assertEqual(added, capacity)
        self.assertEqual(len(hypergraph), capacity)
        self.assertGreater(len(hypergraph), 256)
        self.assertEqual(
            hypergraph.snapshot(limit=capacity)["cycle_count"], capacity)
        audit = hypergraph.audit()
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["cycles_checked"], capacity)
        self.assertEqual(hypergraph.cycle_members((0, 1, 255)).value, 1255)
        self.assertEqual(hypergraph.cycle_members((0, 2, 8)).value, 2008)
        self.assertFalse(audit["canonical_merge"])


# ---------------------------------------------------------------------------
# Round 031 artifacts
# ---------------------------------------------------------------------------
class Round031ArtifactTests(unittest.TestCase):
    def setUp(self):
        require_artifacts(RESULTS, AUDIT, TRANSFER, SHA256SUMS,
                          NATIVE_SOURCE, README)

    def test_results_show_conventional_model_win_and_memory_ablation(self):
        payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"],
                         {"train": 1800, "val": 1800, "test": 1800})
        runs = payload["runs"]
        self.assertEqual(set(runs), {"plain_17", "native_17",
                                     "plain_29", "native_29"})
        for seed in (17, 29):
            plain = runs["plain_%d" % seed]
            native = runs["native_%d" % seed]
            self.assertEqual(plain["best_step"], 100)
            self.assertEqual(native["best_step"], 100)
            self.assertEqual(plain["validation"][-1]["step"], 100)
            self.assertEqual(native["validation"][-1]["step"], 100)
            self.assertEqual(plain["test"]["n_bytes"], 13_200)
            self.assertEqual(native["test"]["n_bytes"], 13_200)
            self.assertLess(plain["test"]["bits_per_byte"],
                            native["test"]["bits_per_byte"])
            self.assertEqual(native["params"], plain["params"] - 256)

            gauge = native["gauge_test"]
            self.assertTrue(gauge["audit"])
            self.assertLessEqual(gauge["max_weight_diff"], 1e-6)
            self.assertEqual(gauge["before"]["accuracy"],
                             gauge["after"]["accuracy"])
            self.assertAlmostEqual(gauge["before"]["bits_per_byte"],
                                   gauge["after"]["bits_per_byte"],
                                   places=5)
            self.assertGreater(native["memory_disabled"]["bits_per_byte"],
                               native["test"]["bits_per_byte"])

    def test_native_audit_and_wordnet_chain_counts(self):
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        native = audit["native"]
        self.assertEqual(native["reads"], 10_000)
        self.assertGreater(native["mean_read_ms"], 0.0)
        self.assertLess(native["mean_read_ms"], 1.0)
        self.assertLess(native["gauge_max_diff"], 2e-6)
        self.assertEqual(native["snapshot_max_diff"], 0.0)
        self.assertIs(native["mirror_audit"], True)

        wordnet = audit["wordnet"]
        self.assertEqual(wordnet["synsets"], 117_570)
        self.assertEqual(wordnet["two_hop_candidates"], 87_655)
        self.assertEqual(wordnet["two_hop_proofs_checked"], 1_500)
        self.assertIs(wordnet["all_chain_edges_verified"], True)
        self.assertEqual(wordnet["direct_edge_only_hits"], 0)

    def test_transfer_records_are_bounded_and_improve_validation(self):
        transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
        self.assertEqual(set(transfer), {"plain_17", "native_17",
                                         "plain_29", "native_29"})
        for key, run in transfer.items():
            self.assertEqual(run["before"]["wordnet"]["n_bytes"], 7_061)
            self.assertEqual(run["after"]["wordnet"]["n_bytes"], 7_061)
            self.assertEqual(run["before"]["alice"]["n_bytes"], 324)
            self.assertEqual(run["after"]["alice"]["n_bytes"], 324)
            self.assertLess(run["after"]["alice"]["bits_per_byte"],
                            run["before"]["alice"]["bits_per_byte"], key)
            self.assertLess(run["after"]["wordnet"]["bits_per_byte"],
                            run["before"]["wordnet"]["bits_per_byte"], key)
            self.assertEqual(run["alice_test"]["n_bytes"], 288)
            self.assertTrue(math.isfinite(
                run["alice_test"]["bits_per_byte"]))
            if key.startswith("native"):
                self.assertEqual(run["alice_test_no_memory"]["n_bytes"], 288)

    def test_checksums_cover_every_published_artifact(self):
        entries = checksum_entries()
        required = {
            "native_cycle.cpp", "train.py", "audit.py", "transfer.py",
            "final_audit.py", "README.md", "results.json", "audit.json",
            "transfer_results.json", "train.log", "transfer.log",
            "plain_17.pt", "plain_29.pt", "native_17.pt", "native_29.pt",
            "transfer_plain_17.pt", "transfer_plain_29.pt",
            "transfer_native_17.pt", "transfer_native_29.pt",
            "root_17.bin", "root_29.bin", "transfer_root_17.bin",
            "transfer_root_29.bin",
        }
        self.assertEqual(set(entries), required)
        for name, digest in entries.items():
            path = ROUND031 / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(file_sha256(path), digest, name)

        for seed in (17, 29):
            raw = (ROUND031 / ("root_%d.bin" % seed)).read_bytes()
            self.assertEqual(len(raw), SNAPSHOT_BYTES)
            self.assertEqual(raw[:8], SNAPSHOT_MAGIC)
            self.assertEqual(len(raw[8:]), ROOT_BYTES)
            self.assertTrue(any(raw[8:]), "root has no learned lanes")

    def test_native_256_weights_are_not_the_full_resident_basis(self):
        source = NATIVE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("static constexpr int N=256, D=16, K=16;", source)
        self.assertIn("PAIRS=32640, ROOT=8388608", source)
        self.assertIn("struct alignas(64) Pair{float f[32],r[32];};", source)
        self.assertIn("void nan_get(void* ptr,float *dst)", source)
        self.assertIn("static_assert(sizeof(Memory)==ROOT);", source)
        self.assertNotIn("uint64_t", source)
        self.assertNotIn("payload", source.lower())
        self.assertEqual(NATIVE_EFFECTIVE_WEIGHTS, 256)
        self.assertEqual(NATIVE_EFFECTIVE_WEIGHTS + 32_129,
                         ROOT_CYCLE_COUNT)

        self.assertEqual(HEAD_COUNT * HEAD_BYTES, 32_768)
        self.assertEqual(PAIR_COUNT * PAIR_BYTES, 8_355_840)
        self.assertEqual(HEAD_COUNT * HEAD_BYTES + PAIR_COUNT * PAIR_BYTES,
                         ROOT_BYTES)
        self.assertEqual(2 * DIRECTION_WORDS * 4, PAIR_BYTES)

        raw = (ROUND031 / "root_17.bin").read_bytes()
        self.assertEqual(raw[:8], SNAPSHOT_MAGIC)
        self.assertEqual(len(raw), SNAPSHOT_BYTES)
        head_region = HEAD_COUNT * HEAD_BYTES
        words = struct.unpack_from("<64I", raw, 8 + head_region)
        self.assertEqual(len(words), 64)
        self.assertEqual(len(words[:32]), DIRECTION_WORDS)
        self.assertEqual(len(words[32:]), DIRECTION_WORDS)

        try:
            spec = importlib.util.find_spec(
                "nanlmind_hermes.resident.cycle_snapshot")
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            snapshot_module = importlib.import_module(
                "nanlmind_hermes.resident.cycle_snapshot")
            snapshot = snapshot_module.parse_snapshot(
                raw, source="round031/root_17.bin")
            self.assertEqual(snapshot.head_count, 256)
            self.assertEqual(snapshot.pair_count, 32_640)
            self.assertEqual(snapshot.root_bytes, ROOT_BYTES)
            self.assertEqual(len(snapshot.head_bytes(0)), HEAD_BYTES)
            self.assertEqual(len(snapshot.raw_words(1, 2)), DIRECTION_WORDS)
            self.assertIs(snapshot_module.CLAIMS_FLOAT_IDENTITY, False)
            self.assertIs(snapshot_module.PERFORMS_CANONICAL_MERGE, False)


# ---------------------------------------------------------------------------
# optional full-cycle modules
# ---------------------------------------------------------------------------
class OptionalFullCycleModulesTests(unittest.TestCase):
    def test_cycle_observable_surface(self):
        module = load_module("nanlmind_hermes.resident.cycle_observable")
        self.assertEqual(module.ROOT_CYCLE_COUNT, 32_385)
        self.assertEqual(module.WORD_BITS, 32)
        self.assertEqual(module.WORD_MASK, 0xFFFFFFFF)
        self.assertEqual(len(module.PAIR_CANONICALS), 32_640)
        self.assertTrue(callable(module.CycleField))

    def test_cycle_training_surface_and_synthetic_claims(self):
        module = load_module("nanlmind_hermes.resident.cycle_training")
        self.assertIs(module.CANONICAL_MERGE, False)
        self.assertIs(module.CYCLE_TRAINING_IS_SEMANTIC_TRUTH, False)
        self.assertEqual(module.FIXED_ONE, 65_536)
        self.assertEqual(module.HEAD_COUNT, 256)
        result = module.synthetic_gauge_shift_experiment(samples=3, seed=31)
        claims = result["claims"]
        self.assertIs(claims["canonical_merge"], False)
        self.assertIs(claims["semantic_truth"], False)
        self.assertIs(claims["language_model_win"], False)
        self.assertIs(claims["cycle_gauge_invariance"], True)
        self.assertEqual(result["cycle"]["test_exact"], 3)

    def test_cycle_persistence_surface(self):
        module = load_module("nanlmind_hermes.resident.cycle_persistence")
        self.assertEqual(module.WORD_MASK, (1 << 64) - 1)
        self.assertEqual(module.CYCLE_SCHEMA, "nanlm.resident-cycle/1")
        self.assertTrue(callable(module.CyclePersistence))
        self.assertTrue(callable(module.CycleReceipt))

    def test_cycle_hypergraph_surface(self):
        module = load_module("nanlmind_hermes.resident.cycle_hypergraph")
        self.assertEqual(module.DEFAULT_NODES, 256)
        self.assertEqual(module.MEMBER_PAIR, 3)
        self.assertEqual(module.MEMBER_CYCLE, 5)
        self.assertEqual(module.MEMBER_VALUE, 6)
        self.assertEqual(module.MEMBER_VERTEX, 7)
        self.assertEqual(module.MEMBER_PAYLOAD, 8)
        self.assertTrue(callable(module.CycleHypergraph))

    def test_cycle_snapshot_surface(self):
        module = load_module("nanlmind_hermes.resident.cycle_snapshot")
        self.assertEqual(module.MARKER, SNAPSHOT_MAGIC)
        self.assertEqual(module.SNAPSHOT_BYTES, SNAPSHOT_BYTES)
        self.assertEqual(module.PAIR_COUNT, 32_640)
        self.assertIs(module.CLAIMS_FLOAT_IDENTITY, False)
        self.assertIs(module.PERFORMS_CANONICAL_MERGE, False)

    def test_cycle_tools_surface(self):
        module = load_module("nanlmind_hermes.resident.cycle_tools")
        self.assertEqual(module.NODE_COUNT, 256)
        self.assertEqual(module.EDGE_COUNT, 32_640)
        self.assertEqual(module.CYCLE_DIMENSION, 32_385)
        flags = module._policy_flags()
        self.assertFalse(flags["canonical_merge"]["enabled"])
        self.assertFalse(flags["canonical_merge"]["attempted"])
        self.assertFalse(flags["canonical_merge_enabled"])
        self.assertFalse(flags["canonical_promotion"])
        self.assertFalse(flags["promotion_enabled"])

    def test_cycle_learning_bridge_surface_and_exact_flux(self):
        module = load_module(
            "nanlmind_hermes.resident.cycle_learning_bridge")
        self.assertIs(module.CANONICAL_MERGE, False)
        self.assertIs(module.CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH, False)
        self.assertIs(module.HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH, False)
        cycle_id = module.encode_cycle_id(1, 2, 3)
        self.assertEqual(module.decode_cycle_id(cycle_id), (1, 2, 3))
        edges = {
            module.encode_edge_id(1, 2): 10,
            module.encode_edge_id(2, 3): 20,
            module.encode_edge_id(1, 3): 5,
        }
        self.assertEqual(module.triangle_flux(edges, cycle_id), 25)
        delta = module.cycle_edge_delta(cycle_id, 7)
        self.assertEqual(module.gauge_residual(delta), 0)
        shifted = dict(edges)
        for edge_id, amount in delta.items():
            shifted[edge_id] = shifted.get(edge_id, 0) + amount
        self.assertEqual(module.triangle_flux(shifted, cycle_id), 46)

    def test_cycle_native_surface_stays_float32_and_noncanonical(self):
        module = load_module("nanlmind_hermes.resident.cycle_native")
        self.assertEqual(module.PAIRS, 32_640)
        self.assertEqual(module.ROOT_BYTES, ROOT_BYTES)
        self.assertEqual(module.SNAPSHOT_BYTES, SNAPSHOT_BYTES)
        self.assertEqual(module.SNAPSHOT_MAGIC, SNAPSHOT_MAGIC)
        self.assertIs(module.CLAIMS["canonical"], False)
        self.assertIs(module.CLAIMS["canonical_merge"], False)
        self.assertIs(module.CLAIMS["raw_nan_payload_arithmetic"], False)
        self.assertIs(module.CLAIMS["float32_boundary"], True)


# ---------------------------------------------------------------------------
# canonical merge false
# ---------------------------------------------------------------------------
class CanonicalMergeFalseTests(unittest.TestCase):
    def test_training_learning_and_tool_flags_stay_false(self):
        training = load_module("nanlmind_hermes.resident.cycle_training")
        bridge = load_module(
            "nanlmind_hermes.resident.cycle_learning_bridge")
        tools = load_module("nanlmind_hermes.resident.cycle_tools")
        self.assertIs(training.CANONICAL_MERGE, False)
        self.assertIs(training.CYCLE_TRAINING_IS_SEMANTIC_TRUTH, False)
        self.assertIs(bridge.CANONICAL_MERGE, False)
        self.assertIs(bridge.CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH, False)
        self.assertIs(bridge.HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH, False)
        flags = tools._policy_flags()
        self.assertFalse(flags["canonical_merge"]["enabled"])
        self.assertFalse(flags["canonical_merge"]["attempted"])
        self.assertFalse(flags["canonical_merge_enabled"])
        self.assertFalse(flags["canonical_promotion"])
        self.assertFalse(flags["canonical_promotion_attempted"])
        self.assertFalse(flags["promotion_enabled"])

    def test_hypergraph_and_snapshot_codec_refuse_promotion(self):
        module = load_module("nanlmind_hermes.resident.cycle_hypergraph")
        snapshot = load_module("nanlmind_hermes.resident.cycle_snapshot")
        native = load_module("nanlmind_hermes.resident.cycle_native")
        hypergraph = module.CycleHypergraph()
        self.assertIs(hypergraph.canonical_merge, False)
        self.assertIs(hypergraph.snapshot()["canonical_merge"], False)
        self.assertIs(hypergraph.audit()["canonical_merge"], False)
        with self.assertRaises(module.CycleDenied):
            hypergraph.merge()
        self.assertIs(snapshot.CLAIMS_FLOAT_IDENTITY, False)
        self.assertIs(snapshot.PERFORMS_CANONICAL_MERGE, False)
        self.assertIs(native.CLAIMS["canonical_merge"], False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
