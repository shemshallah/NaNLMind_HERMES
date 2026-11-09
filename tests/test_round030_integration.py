"""Round 030 integration contract tests.

Round 030 stores a small semantic classifier directly in one native 8 MiB
mirror-pair root.  The tests below check the published artifacts, then exercise
the same geometry through the resident Python stack and, when a C++20 compiler
is available, through the actual shared library.

Optional material is skipped only when it is genuinely absent.  A present
artifact or module that violates the contract is a failure.  The native timing
field is never turned into a phone or portable-performance claim.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Keep ``python3 -m unittest`` usable from root.
    sys.path.insert(0, str(ROOT))

ROUND030 = ROOT / "experiments" / "round030"
RESULTS = ROUND030 / "results.json"
TUNED_RESULTS = ROUND030 / "tuned_results.json"
NATIVE_AUDIT = ROUND030 / "native_audit.json"
NATIVE_SOURCE = ROUND030 / "native_cycle.cpp"
SHA256SUMS = ROUND030 / "SHA256SUMS"

ROOT_BYTES = 8_388_608
SNAPSHOT_BYTES = ROOT_BYTES + 8
PAIR_COUNT = 32_640
ROOT_CYCLE_COUNT = 32_385
SNAPSHOT_MAGIC = b"NANCYC30"

D = 24
K = 16
F32P = C.POINTER(C.c_float)
I32P = C.POINTER(C.c_int)


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
        raise unittest.SkipTest("Round 030 artifacts are absent: %s"
                                % ", ".join(missing))


def float_array(values):
    return (C.c_float * len(values))(*values)


def int_array(values):
    return (C.c_int * len(values))(*values)


def max_abs_difference(left, right) -> float:
    if len(left) != len(right):
        raise AssertionError("vectors have different lengths")
    if isinstance(left[0], list):
        return max(max_abs_difference(a, b) for a, b in zip(left, right))
    return max(abs(a - b) for a, b in zip(left, right))


class NativeBridge:
    """Small ctypes facade over ``native_cycle.cpp`` for the tests."""

    def __init__(self, library):
        self.lib = library
        lib = self.lib
        lib.nan_new.argtypes = [C.c_int]
        lib.nan_new.restype = C.c_void_p
        lib.nan_free.argtypes = [C.c_void_p]
        lib.nan_get.argtypes = [C.c_void_p, F32P]
        lib.nan_predict.argtypes = [C.c_void_p, F32P, F32P]
        lib.nan_raw_predict.argtypes = [C.c_void_p, F32P, F32P]
        lib.nan_epoch.argtypes = [C.c_void_p, F32P, I32P, C.c_int,
                                  C.c_float, C.c_uint]
        lib.nan_epoch.restype = C.c_double
        lib.nan_gauge.argtypes = [C.c_void_p, C.c_int, C.c_float]
        lib.nan_snapshot.argtypes = [C.c_void_p, C.c_char_p]
        lib.nan_snapshot.restype = C.c_int
        lib.nan_restore.argtypes = [C.c_void_p, C.c_char_p]
        lib.nan_restore.restype = C.c_int
        lib.nan_audit.argtypes = [C.c_void_p]
        lib.nan_audit.restype = C.c_int
        self.ptr = lib.nan_new(0)
        if not self.ptr:
            raise RuntimeError("native root allocation failed")

    def close(self):
        if self.ptr:
            self.lib.nan_free(self.ptr)
            self.ptr = None

    def weights(self):
        out = (C.c_float * (K * D))()
        self.lib.nan_get(self.ptr, out)
        return [[out[c * D + d] for d in range(D)] for c in range(K)]

    def predict(self, row, *, raw=False):
        x = float_array(row)
        out = (C.c_float * K)()
        method = self.lib.nan_raw_predict if raw else self.lib.nan_predict
        method(self.ptr, x, out)
        return [out[index] for index in range(K)]

    def train(self, features, labels, *, epochs=1, lr=0.05, seed=43):
        flat = [value for row in features for value in row]
        xs = float_array(flat)
        ys = int_array(labels)
        count = len(labels)
        return [self.lib.nan_epoch(self.ptr, xs, ys, count, lr, seed + index)
                for index in range(epochs)]

    def shift(self, seed, sigma):
        self.lib.nan_gauge(self.ptr, seed, sigma)

    def snapshot(self, path: Path) -> bool:
        return bool(self.lib.nan_snapshot(self.ptr, str(path).encode()))

    def restore(self, path: Path) -> bool:
        return bool(self.lib.nan_restore(self.ptr, str(path).encode()))

    def audit(self) -> bool:
        return bool(self.lib.nan_audit(self.ptr))


def argmax(values):
    return max(range(len(values)), key=lambda index: values[index])


def accuracy(bridge, features, labels, *, raw=False):
    correct = 0
    for row, label in zip(features, labels):
        if argmax(bridge.predict(row, raw=raw)) == label:
            correct += 1
    return correct / len(labels)


class Round030ArtifactTests(unittest.TestCase):
    """Published artifacts stay self-consistent and honestly bounded."""

    def setUp(self):
        require_artifacts(RESULTS, TUNED_RESULTS, NATIVE_AUDIT, NATIVE_SOURCE)

    def test_results_and_tuned_summaries(self):
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
        runs = results["runs"]
        self.assertEqual([run["seed"] for run in runs], [17, 29, 41])
        for run in runs:
            samples = run["samples"]
            self.assertEqual(len(samples["categories"]), K)
            self.assertEqual((samples["train"], samples["val"],
                              samples["test"]), (2400, 560, 560))
            for key in ("train_ids_sha256", "test_ids_sha256"):
                self.assertRegex(samples[key], r"^[0-9a-f]{64}$")
            self.assertNotEqual(samples["train_ids_sha256"],
                                samples["test_ids_sha256"])
            self.assertEqual(run["root_bytes"], ROOT_BYTES)
            self.assertTrue(run["mirror_audit"])
            self.assertTrue(math.isfinite(run["native_training_seconds"]))
            self.assertLessEqual(run["native_vs_real_linear_max_logit_error"],
                                 0.002)
            self.assertLessEqual(run["cycle_max_drift_from_gauge"], 1e-5)
            self.assertLessEqual(run["max_cycle_logits_change"], 3e-4)
            self.assertEqual(run["restore_max_logit_error"], 0.0)

            before = run["cycle_before"]
            shifted = run["cycle_after_large_gauge"]
            self.assertEqual(before["accuracy"], shifted["accuracy"])
            self.assertAlmostEqual(before["cross_entropy"],
                                   shifted["cross_entropy"], places=5)
            self.assertLessEqual(run["raw_after_large_gauge"]["accuracy"],
                                 run["raw_before"]["accuracy"] - 0.20)
            self.assertLessEqual(run["raw_after_large_gauge"]["accuracy"],
                                 shifted["accuracy"] - 0.20)

        tuned = json.loads(TUNED_RESULTS.read_text(encoding="utf-8"))
        mean = tuned["mean_accuracy"]
        self.assertAlmostEqual(mean["cycle_test"], 0.42083333333333334)
        self.assertEqual(mean["cycle_test"], mean["shift_test"])
        self.assertLess(mean["raw_shift_test"], mean["cycle_test"] - 0.20)
        self.assertLess(mean["raw_shift_test"], 0.09)
        for run in tuned["results"]:
            self.assertEqual(run["cycle_test"]["accuracy"],
                             run["shift_test"]["accuracy"])
            self.assertLessEqual(run["max_logit_delta_single_shift"], 3e-5)
            self.assertLessEqual(run["max_logit_delta_101_shifts"], 3e-4)
            self.assertEqual(run["native_root_bytes"], ROOT_BYTES)
            self.assertTrue(run["cycle_equals_same_real_linear"])
        limitations = " ".join(tuned["limitations"]).lower()
        for phrase in ("synthetic", "external", "conventional",
                       "not a generative"):
            self.assertIn(phrase, limitations)

    def test_native_audit_numbers(self):
        audit = json.loads(NATIVE_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["root_bytes"], ROOT_BYTES)
        self.assertAlmostEqual(audit["native_gradient_max_abs_error"],
                               1.862645149230957e-09, delta=1e-15)
        self.assertLess(audit["native_gradient_max_abs_error"], 1e-6)
        self.assertAlmostEqual(audit["gauge_weight_max_abs_error"],
                               4.1653402149677277e-07, delta=1e-12)
        self.assertLess(audit["gauge_weight_max_abs_error"], 1e-5)
        self.assertIs(audit["mirror_antisymmetry"], True)
        self.assertIs(audit["snapshot_bitwise_restore"], True)
        self.assertEqual(audit["predictions"], 1000)
        self.assertGreater(audit["ctypes_wall_ms"], 0.0)
        self.assertIs(audit["memory_wordnet_predictor_is_native"], True)
        self.assertIs(audit["text_encoder_is_external"], True)

    def test_root_snapshots_and_checksums(self):
        checksum_file = SHA256SUMS
        require_artifacts(checksum_file)
        entries = {}
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                entries[name.strip()] = digest
        for name in ("results.json", "tuned_results.json", "native_audit.json",
                     "root_17.bin", "root_29.bin", "root_41.bin"):
            self.assertIn(name, entries)
        for name, digest in entries.items():
            path = ROUND030 / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             digest, name)

        for seed in (17, 29, 41):
            raw = (ROUND030 / ("root_%d.bin" % seed)).read_bytes()
            self.assertEqual(len(raw), SNAPSHOT_BYTES)
            self.assertEqual(raw[:8], SNAPSHOT_MAGIC)
            self.assertEqual(len(raw[8:]), ROOT_BYTES)
            self.assertTrue(any(raw[8:]), "snapshot contains no learned lanes")

    def test_native_root_geometry_and_unique_pairs(self):
        ng = load_module("nanlmind_hermes.nan_geometry")
        self.assertEqual(ng.N_HEADS, 256)
        self.assertEqual(ng.HEAD_BYTES, 128)
        self.assertEqual(ng.LANE_BYTES, 128)
        self.assertEqual(ng.PAIR_BYTES, 256)
        self.assertEqual(ng.PAIR_COUNT, PAIR_COUNT)
        self.assertEqual(ng.ROOT_BYTES, ROOT_BYTES)
        self.assertEqual(ng.PAIR_COUNT * 2, 65_280)

        ids = []
        for first, second in ng.iter_pairs():
            pair = ng.pair_id(first, second)
            ids.append(pair)
            self.assertEqual(pair, ng.pair_id(second, first))
            self.assertEqual(ng.pair_heads(pair), (first, second))
            forward = ng.mirror_address(first, second)
            reverse = ng.mirror_address(second, first)
            self.assertEqual(forward.pair, reverse.pair)
            self.assertEqual(forward.cell_offset, reverse.cell_offset)
            self.assertEqual(abs(forward.lane_offset - reverse.lane_offset),
                             ng.LANE_BYTES)
        self.assertEqual(len(set(ids)), PAIR_COUNT)
        self.assertEqual(ids, list(range(PAIR_COUNT)))

        source = NATIVE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("static constexpr int N=256, D=24, K=16;", source)
        self.assertIn("PAIRS=32640, ROOT=8388608", source)
        self.assertIn("static_assert(sizeof(Memory)==ROOT);", source)
        self.assertIn("struct alignas(64) Pair{float f[32],r[32];};", source)
        self.assertIn("int idx(int i,int j)", source)
        self.assertIn("return i*(511-i)/2+j-i-1;", source)


class NativeSharedLibraryTests(unittest.TestCase):
    """Optional dynamic checks against the compiled C++20 root."""

    @classmethod
    def setUpClass(cls):
        if not NATIVE_SOURCE.is_file():
            raise unittest.SkipTest("native_cycle.cpp is absent")
        compiler = (shutil.which("c++") or shutil.which("g++")
                    or shutil.which("clang++"))
        if compiler is None:
            raise unittest.SkipTest("no C++20 compiler is available")
        cls._temporary = tempfile.TemporaryDirectory(prefix="round030-native-")
        library = Path(cls._temporary.name) / "libnative_cycle.so"
        build = subprocess.run(
            [compiler, "-std=c++20", "-O3", "-shared", "-fPIC",
             str(NATIVE_SOURCE), "-o", str(library)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        if build.returncode != 0:
            raise AssertionError("native compile failed:\n%s"
                                 % build.stderr[-2000:])
        cls.library = C.CDLL(str(library))

    @classmethod
    def tearDownClass(cls):
        temporary = getattr(cls, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()

    def bridge(self):
        bridge = NativeBridge(self.library)
        self.addCleanup(bridge.close)
        return bridge

    def test_root_snapshot_and_bitwise_restore(self):
        bridge = self.bridge()
        self.assertTrue(bridge.audit())
        bridge.train([[(index - 12) / 64.0 for index in range(D)]], [4],
                     epochs=1, lr=0.05, seed=43)
        before_weights = bridge.weights()
        probes = [[(index * 7 - 3) / 100.0 for index in range(D)]]
        before_logits = bridge.predict(probes[0])
        self.assertTrue(any(value != 0.0 for row in before_weights
                            for value in row))

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "root.bin"
            self.assertTrue(bridge.snapshot(snapshot))
            self.assertEqual(snapshot.stat().st_size, SNAPSHOT_BYTES)
            raw = snapshot.read_bytes()
            self.assertEqual(raw[:8], SNAPSHOT_MAGIC)
            self.assertEqual(len(raw[8:]), ROOT_BYTES)

            bridge.shift(111, 7.0)
            self.assertNotEqual(bridge.weights(), before_weights)
            self.assertTrue(bridge.restore(snapshot))
            self.assertEqual(bridge.weights(), before_weights)
            self.assertEqual(bridge.predict(probes[0]), before_logits)
            self.assertTrue(bridge.audit())

    def test_gradient_matches_analytic_float32_update(self):
        bridge = self.bridge()
        row = [(index - 12) / 64.0 for index in range(D)]
        probability = 1.0 / K
        expected = [
            [-0.05 * (probability - (1.0 if label == 4 else 0.0)) * value
             for value in row]
            for label in range(K)
        ]
        loss = bridge.train([row], [4], epochs=1, lr=0.05, seed=43)[0]
        self.assertAlmostEqual(loss, math.log(K), places=6)
        error = max_abs_difference(bridge.weights(), expected)
        self.assertLess(error, 1e-6)
        self.assertTrue(bridge.audit())

    def test_gauge_preserves_cycle_and_shifts_raw_edges(self):
        bridge = self.bridge()
        features = [[(index * 13 + 5) / 64.0 for index in range(D)]]
        bridge.train(features, [7], epochs=1, lr=0.05, seed=43)
        weights_before = bridge.weights()
        cycle_before = bridge.predict(features[0])
        raw_before = bridge.predict(features[0], raw=True)

        bridge.shift(110, 1.5)
        drift = max_abs_difference(weights_before, bridge.weights())
        self.assertLess(drift, 1e-5)
        cycle_after = bridge.predict(features[0])
        raw_after = bridge.predict(features[0], raw=True)
        self.assertLess(max_abs_difference(cycle_before, cycle_after), 1e-4)
        self.assertGreater(max_abs_difference(raw_before, raw_after), 1e-2)
        self.assertTrue(bridge.audit())

    def test_raw_accuracy_degrades_after_large_gauge(self):
        bridge = self.bridge()
        rng = random.Random(17)
        features = [[rng.gauss(0.0, 0.25) for _ in range(D)]
                    for _ in range(240)]
        labels = [rng.randrange(K) for _ in features]
        bridge.train(features, labels, epochs=10, lr=0.05, seed=17)

        cycle_before = accuracy(bridge, features, labels)
        raw_before = accuracy(bridge, features, labels, raw=True)
        bridge.shift(1017, 3.0)
        cycle_after = accuracy(bridge, features, labels)
        raw_after = accuracy(bridge, features, labels, raw=True)

        self.assertEqual(cycle_before, cycle_after)
        self.assertGreater(cycle_after, 0.20)
        self.assertLess(raw_after, raw_before - 0.15)
        self.assertGreater(cycle_after - raw_after, 0.15)


class ExactGaugeInvarianceTests(unittest.TestCase):
    """The integer readout is exactly invariant, not merely float-close."""

    def test_gauge_store_full_graph_is_exact_after_huge_shifts(self):
        ng = load_module("nanlmind_hermes.nan_geometry")
        store = ng.GaugeStore(ng.N_HEADS)
        rng = random.Random(30)
        gauge = [rng.randint(-(1 << 40), 1 << 40)
                 for _ in range(ng.N_HEADS)]
        store.set_gauge(gauge)
        cycles = {}
        for first in range(1, ng.N_HEADS):
            for second in range(first + 1, ng.N_HEADS):
                value = rng.randint(-(1 << 20), 1 << 20)
                cycles[(first, second)] = value
                store.set_cycle(first, second, value)
        self.assertEqual(len(cycles), ROOT_CYCLE_COUNT)

        shift = [rng.randint(-(1 << 50), 1 << 50)
                 for _ in range(ng.N_HEADS)]
        store.shift_gauge(shift)
        moved = store.gauge
        for (first, second), value in cycles.items():
            self.assertEqual(store.cycle_readout(first, second), value)
            self.assertEqual(store.canonical_flux(first, second), value)
            self.assertEqual(store.canonical_flux(second, first), -value)
            self.assertEqual(store.edge(first, second),
                             value + moved[first] - moved[second])
            self.assertEqual(store.edge(second, first),
                             -store.edge(first, second))
            self.assertEqual(store.edge(0, first), moved[0] - moved[first])
            self.assertEqual(store.edge(first, 0), moved[first] - moved[0])

    def test_resident_cycle_field_full_basis_is_exact(self):
        cells_module = load_module("nanlmind_hermes.resident.cells")
        observable = load_module("nanlmind_hermes.resident.cycle_observable")
        cells = cells_module.ResidentCells(
            bytearray(cells_module.REGION_BYTES), initialize=True)
        field = observable.CycleField(cells)
        heads_before = bytes(cells.buffer[:cells_module.HEAD_REGION_BYTES])

        rng = random.Random(31)
        for index in range(48):
            first = 1 + 2 * index
            second = first + 1
            word = rng.randrange(1 << 32)
            field.set_oriented(0, first, word)
            field.set_oriented(first, second, word ^ 0x5A5A5A5A)
            field.set_oriented(0, second, word // 3)
            field.mirror(0, first)
            field.mirror(first, second)
            field.mirror(0, second)
        before = field.batch_cycles()
        self.assertEqual(len(before), ROOT_CYCLE_COUNT)
        generation = field.generation

        gauge = [rng.randrange(1 << 32) for _ in range(cells_module.HEAD_COUNT)]
        self.assertEqual(field.gauge_shift(gauge), 2 * PAIR_COUNT)
        after = field.batch_cycles()
        self.assertEqual(before, after)
        self.assertGreater(field.generation, generation)
        self.assertEqual(bytes(cells.buffer[:cells_module.HEAD_REGION_BYTES]),
                         heads_before)

        first, second = 1, 2
        forward = field.get_oriented(first, second)
        reverse = field.get_oriented(second, first)
        self.assertEqual((forward + reverse) & observable.WORD_MASK, 0)
        self.assertEqual((forward
                          - field.get_oriented(first, second)) & 0xFFFFFFFF, 0)


class ResidentCycleIntegrationTests(unittest.TestCase):
    """Training and observation share the resident pair rails."""

    def test_cycle_trainer_writes_resident_mirror_lanes(self):
        cells_module = load_module("nanlmind_hermes.resident.cells")
        training = load_module("nanlmind_hermes.resident.cycle_training")
        observable = load_module("nanlmind_hermes.resident.cycle_observable")
        cells = cells_module.ResidentCells(
            bytearray(cells_module.REGION_BYTES), initialize=True)
        lane_field = training.CellsLaneField(cells)
        trainer = training.CycleTrainer(
            lane_field, learning_rate=training.FIXED_ONE,
            normalize_gradient=True, max_step_units=1 << 26)
        target = 6 * training.FIXED_ONE
        receipt = trainer.step((3, 4, target))

        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["changed"])
        self.assertEqual(receipt["pairs_updated"], 3)
        self.assertIs(receipt["canonical_merge"], False)
        self.assertIs(receipt["semantic_truth"], False)
        self.assertEqual(trainer.predict(3, 4), target)
        self.assertEqual(observable.CycleField(cells).cycle(3, 4), target)
        for source, destination in ((0, 3), (3, 4), (4, 0)):
            self.assertEqual(lane_field.read_lane(source, destination),
                             -lane_field.read_lane(destination, source))
        self.assertTrue(trainer.finite_difference_check(
            (3, 4, target), delta=17)["exact"])
        self.assertEqual(cells.generation, receipt["generation_after"])

    def test_cycle_field_snapshot_is_bounded_and_detached(self):
        cells_module = load_module("nanlmind_hermes.resident.cells")
        observable = load_module("nanlmind_hermes.resident.cycle_observable")
        cells = cells_module.ResidentCells(
            bytearray(cells_module.REGION_BYTES), initialize=True)
        field = observable.CycleField(cells)
        field.set_oriented(0, 1, 123)
        field.set_oriented(1, 2, 456)
        field.set_oriented(0, 2, 100)
        snapshot = field.snapshot()
        self.assertEqual(snapshot["mode"], "raw-u32")
        self.assertEqual(snapshot["pair_count"], PAIR_COUNT)
        self.assertEqual(snapshot["root_cycle_count"], ROOT_CYCLE_COUNT)
        self.assertEqual(len(snapshot["pairs"]), PAIR_COUNT)
        self.assertEqual(len(snapshot["cycles"]), ROOT_CYCLE_COUNT)
        original = next(sample.word for sample in snapshot["cycles"]
                        if sample.left == 1 and sample.right == 2)
        self.assertEqual(original, 123 + 456 - 100)
        self.assertEqual(field.cycle(1, 2), original)
        field.set_oriented(1, 2, 0)
        self.assertEqual(field.cycle(1, 2), 123 - 100)
        self.assertEqual(snapshot["cycles"][
            next(index for index, sample in enumerate(snapshot["cycles"])
                 if sample.left == 1 and sample.right == 2)].word, original)


class DistinctPayloadBridgeTests(unittest.TestCase):
    """Raw NaN payload modules are not the C++ float32 training bridge."""

    def test_raw_lane_layer_remains_bit_preserving_and_separate(self):
        ng = load_module("nanlmind_hermes.nan_geometry")
        nan16 = load_module("nanlmind_hermes.nan16de")
        self.assertEqual(ng.PAYLOAD_BITS, 52)
        self.assertEqual(ng.VALUE_BITS, 48)
        self.assertEqual(ng.LANE_WORD_COUNT, 14)
        self.assertEqual(ng.LANE_HEADER_BYTES + 8 * ng.LANE_WORD_COUNT,
                         ng.LANE_BYTES)
        values = [ng.VALUE_MIN, -1, 0, 1, ng.VALUE_MAX, 0x123456789ABC]
        words = tuple(ng.encode_fixed_units(value) for value in values)
        words += (0,) * (ng.LANE_WORD_COUNT - len(words))
        lane = ng.Lane(tag=ng.LANE_TAG_NAN_WORDS, generation=7, words=words)
        restored = ng.unpack_lane(ng.pack_lane(lane))
        self.assertEqual(restored, lane)
        for word, value in zip(restored.words[:len(values)], values):
            self.assertTrue(ng.is_quiet_nan(word))
            self.assertEqual(ng.decode_fixed_units(word), value)

        word = nan16.NanWord.from_bits(0, 3, 0xFEEDFACE)
        self.assertEqual(nan16.NanWord.unpack(word.pack()), word)
        self.assertEqual(nan16.NanWord.unpack(word.pack()).payload,
                         0xFEEDFACE)

        source = NATIVE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("Pair{float f[32],r[32];}", source)
        self.assertNotIn("uint64_t", source)
        self.assertNotIn("payload", source.lower())
        self.assertEqual(32 * 4, ng.LANE_BYTES)
        self.assertNotEqual(ng.LANE_WORD_COUNT, 32)
        geometry_source = Path(ng.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ctypes", geometry_source)
        self.assertNotIn("native_cycle", geometry_source)


class CanonicalMergeFalseTests(unittest.TestCase):
    """Cycle training and its synthetic experiment never promote truth."""

    def test_all_cycle_training_authority_flags_stay_false(self):
        training = load_module("nanlmind_hermes.resident.cycle_training")
        self.assertIs(training.CANONICAL_MERGE, False)
        self.assertIs(training.CYCLE_TRAINING_IS_SEMANTIC_TRUTH, False)

        field = training.IntegerLaneField()
        trainer = training.CycleTrainer(
            field, learning_rate=training.FIXED_ONE,
            normalize_gradient=True)
        receipt = trainer.step((1, 2, 3 * training.FIXED_ONE))
        self.assertIs(receipt["canonical_merge"], False)
        self.assertIs(receipt["semantic_truth"], False)
        stats = trainer.stats()
        self.assertIs(stats["canonical_merge"], False)
        self.assertIs(stats["semantic_truth"], False)
        denied = trainer.try_step((3, 3, 1))
        self.assertFalse(denied["accepted"])
        self.assertIs(denied["canonical_merge"], False)
        self.assertIs(denied["semantic_truth"], False)

        experiment = training.synthetic_gauge_shift_experiment(
            samples=3, seed=2029)
        claims = experiment["claims"]
        self.assertIs(claims["canonical_merge"], False)
        self.assertIs(claims["semantic_truth"], False)
        self.assertIs(claims["language_model_win"], False)
        self.assertIs(claims["held_out_samples"], False)
        self.assertIs(claims["cycle_gauge_invariance"], True)
        self.assertTrue(claims["cycle_exact_reconstruction"])
        self.assertEqual(experiment["cycle"]["test_exact"], 3)
        self.assertGreater(experiment["raw_edge"]["test_error_units"], 0)
        self.assertTrue(experiment["raw_edge"]["predictions_changed"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
