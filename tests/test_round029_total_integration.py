"""Round 029 total-integration contract tests.

Round 029 demonstrated a controlled gauge-invariance result on WordNet gloss
classification.  This suite checks that the *same exact edge arithmetic* is
represented correctly by the resident stack:

* the raw NaN16 payload/word codec;
* the 32,640 canonical/mirror pair map and the independent rank involution;
* exact triangle-flux gauge cancellation and the cycle-vs-raw comparison;
* cycle fields stored on real resident cell rails;
* pair/mirror membership in resident hyperedges;
* cycle-state survival through a resident-universe reopen;
* a small temporal + independent-head-field + cycle bridge;
* tool approval/bounds and canonical-merge false;
* optional native C++ comparison (skip when no toolchain; fail on mismatch);
* deterministic head-field benchmark output.

Optional modules are imported lazily.  A genuinely absent module is a skip; a
module that exists but violates the contract is a failure.  The native C++
test compiles and runs the Round 029 proof-of-implementation only when a C++20
compiler is available.  Its timing is never asserted and is not an Android or
portable-performance claim.
"""
from __future__ import annotations

import importlib
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

ROUND029_DIR = ROOT / "experiments" / "round029" / "nan_round029"
ROUND029_SUMMARY = ROUND029_DIR / "summary.json"
NATIVE_SOURCE = ROUND029_DIR / "native_mirror.cpp"

MIB = 1024 * 1024
UNIVERSE_BYTES = 16 * MIB


def load_module(name: str):
    """Import ``name`` lazily; skip only when the module is genuinely absent."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        raise unittest.SkipTest("%s is not present" % name)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# resident rail math used by several tests
# ---------------------------------------------------------------------------
def directed_weight(cells_module, cells, first: int, second: int) -> float:
    """Signed Q16.16 weight in the ``first -> second`` rail direction."""
    state = cells.pair_state(first, second)
    rail = (state.forward
            if cells_module.address_of(first, second) == state.canonical
            else state.reverse)
    return rail.weight / float(cells_module.FIXED_ONE)


def antisymmetric(cells_module, cells, first: int, second: int) -> float:
    return 0.5 * (directed_weight(cells_module, cells, first, second)
                  - directed_weight(cells_module, cells, second, first))


def resident_triangle_flux(cells_module, cells, first: int, second: int) -> float:
    """Round 029 fundamental flux ``anti(0,a)+anti(a,b)-anti(0,b)``."""
    return (antisymmetric(cells_module, cells, 0, first)
            + antisymmetric(cells_module, cells, first, second)
            - antisymmetric(cells_module, cells, 0, second))


def store_skew(cells, first: int, second: int, value: float) -> None:
    """Store a skew edge field: +value one way, -value the other way."""
    cells.observe_pair(first, second, weight=value)
    cells.observe_pair(second, first, weight=-value)


def cycle_triangle_rows(vector, gauge, dimensions: int):
    """Round 029 readout rows: ``(g0-ga, z+ga-gb, g0-gb)``."""
    zero_a, a_b, zero_b = [], [], []
    for index in range(dimensions):
        a = 1 + 2 * index
        b = a + 1
        zero_a.append(gauge[0] - gauge[a])
        a_b.append(vector[index] + gauge[a] - gauge[b])
        zero_b.append(gauge[0] - gauge[b])
    return zero_a, a_b, zero_b


# ---------------------------------------------------------------------------
# raw NaN payload codec
# ---------------------------------------------------------------------------
class RawNanPayloadCodecTests(unittest.TestCase):
    def setUp(self):
        self.nan16de = load_module("nanlmind_hermes.nan16de")

    def test_all_16_truth_tables_round_trip_with_exact_payload(self):
        codec = self.nan16de
        payloads = (0, 1, 0x00C0FFEE, 0xDEADBEEF, 0xFFFFFFFF)
        seen = set()
        for far_type in range(4):
            for near_type in range(4):
                for payload in payloads:
                    word = codec.NanWord.from_bits(
                        far_type, near_type, payload)
                    self.assertEqual(word.payload, payload)
                    self.assertEqual(
                        word.opcode, (near_type << 2) | far_type)
                    raw = word.pack()
                    self.assertEqual(len(raw), 8)
                    decoded = codec.NanWord.unpack(raw)
                    self.assertEqual(decoded, word)
                    self.assertEqual(decoded.pack(), raw)
                    for a in (0, 1):
                        for b in (0, 1):
                            expected = (word.opcode
                                        >> (2 * a + b)) & 1
                            self.assertEqual(word.eval_bool(a, b), expected)
                    seen.add(word.opcode)
        self.assertEqual(seen, set(range(16)))

    def test_nanvm_word_codec_agrees_on_payload_and_heads(self):
        nanvm = load_module("nanlmind_hermes.nanvm")
        for selector in (0, 1, 6, 15):
            for payload in (0, 7, 0xDEADBEEF, 0xFFFFFFFF):
                word = nanvm.make_word(payload, selector)
                self.assertEqual(nanvm.word_payload(word), payload)
                self.assertEqual(nanvm.far_head(word), nanvm.NAN_HEADS[selector & 3])
                self.assertEqual(
                    nanvm.near_head(word), nanvm.NAN_HEADS[(selector >> 2) & 3])

    def test_packed_pair_state_payload_round_trips_rank_amplitude_phase(self):
        writer = load_module("nanlmind_hermes.word_writer")
        spine = load_module("nanlmind_hermes.temporal_spine")
        for rank in (0, 1, 24_384, 32_639):
            pair = spine.pair_at(rank)
            self.assertEqual(writer.rank_of_pair(*pair), rank)
            for amplitude in (0, 1, 128, 255):
                for phase in (0, 1, 300, 511):
                    payload = writer.pack_state(rank, amplitude, phase)
                    self.assertLess(payload, 1 << 32)
                    self.assertEqual(
                        writer.unpack_state(payload),
                        {"rank": rank, "amplitude": amplitude,
                         "phase": phase})


# ---------------------------------------------------------------------------
# 32,640 pair mapping
# ---------------------------------------------------------------------------
class MirrorPairMappingTests(unittest.TestCase):
    def setUp(self):
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.spine = load_module("nanlmind_hermes.temporal_spine")

    def test_all_32640_pairs_have_unique_mirror_cell_and_rank_involution(self):
        cells = self.cells_module
        spine = self.spine
        self.assertEqual(cells.PAIR_COUNT, 32_640)
        self.assertEqual(spine.PAIR_COUNT, 32_640)

        indexes = set()
        canonicals = set()
        ranks = set()
        sample_pair = None
        for first in range(256):
            for second in range(first + 1, 256):
                forward = cells.address_of(first, second)
                reverse = cells.address_of(second, first)
                canonical = cells.canonical_address(first, second)
                mirror = cells.mirror_address(canonical)

                self.assertEqual(canonical, min(forward, cells.reverse16(forward)))
                self.assertEqual(canonical, min(reverse, cells.reverse16(reverse)))
                self.assertEqual(canonical, cells.canonical_address(second, first))
                self.assertEqual(mirror, cells.reverse16(canonical))

                index = cells.pair_index(first, second)
                self.assertEqual(index, cells.pair_index(second, first))
                indexes.add(index)
                canonicals.add(canonical)

                rank = spine.rank_of_pair(first, second)
                self.assertEqual(spine.pair_at(rank), (first, second))
                self.assertEqual(spine.mirror_rank(spine.mirror_rank(rank)), rank)
                self.assertEqual(spine.pair_at(spine.mirror_rank(rank)),
                                 spine.mirror_pair(rank))
                ranks.add(rank)
                sample_pair = (first, second, rank, canonical, mirror)

        self.assertEqual(len(indexes), 32_640)
        self.assertEqual(indexes, set(range(32_640)))
        self.assertEqual(len(canonicals), 32_640)
        self.assertEqual(len(ranks), 32_640)
        self.assertIsNotNone(sample_pair)

    def test_address_mirror_and_rank_mirror_are_distinct_involutions(self):
        cells = self.cells_module
        spine = self.spine
        pair = (127, 128)
        rank = spine.rank_of_pair(*pair)
        rank_mirror_pair = spine.mirror_pair(rank)
        address_mirror = cells.mirror_address(
            cells.canonical_address(*pair))
        address_mirror_heads = (
            address_mirror >> 8, cells.reverse8(address_mirror & 0xFF))
        self.assertEqual(spine.mirror_rank(rank), 32_639 - rank)
        self.assertNotEqual(rank_mirror_pair, pair)
        self.assertNotEqual(address_mirror_heads, pair)
        # The two involutions are different maps and must never be conflated.
        self.assertNotEqual(rank_mirror_pair, address_mirror_heads)


# ---------------------------------------------------------------------------
# exact gauge geometry
# ---------------------------------------------------------------------------
class GaugeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cycle_space = load_module("nanlmind_hermes.cycle_space")

    def test_exact_triangle_flux_cancels_arbitrary_vertex_gauge(self):
        rng = random.Random(2029)
        dimensions = 48
        z = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        for _trial in range(12):
            gauge = [rng.uniform(-5.0, 5.0) for _ in range(256)]
            values = {}
            for index, semantic in enumerate(z):
                a = 1 + 2 * index
                b = a + 1
                values[(0, a)] = gauge[0] - gauge[a]
                values[(a, b)] = semantic + gauge[a] - gauge[b]
                values[(0, b)] = gauge[0] - gauge[b]
            for index, semantic in enumerate(z):
                a = 1 + 2 * index
                b = a + 1
                flux = self.cycle_space.triangle_flux(values, 0, a, b)
                self.assertLess(abs(flux - semantic), 1e-12)
        self.assertEqual(self.cycle_space.cycle_dimension(256), 32_385)
        self.assertEqual(self.cycle_space.edge_count(256), 32_640)

    def test_cycle_readout_cancels_gauge_while_raw_lane_does_not(self):
        rng = random.Random(41)
        dimensions = 48
        vectors = []
        gauges = []
        for _row in range(64):
            vectors.append([rng.uniform(-1.0, 1.0) for _ in range(dimensions)])
            gauges.append([rng.gauss(0.0, 3.0) for _ in range(256)])

        max_cycle_error = 0.0
        raw_error_sum = 0.0
        raw_error_count = 0
        for vector, gauge in zip(vectors, gauges):
            zero_a, a_b, zero_b = cycle_triangle_rows(
                vector, gauge, dimensions)
            cycle = [first + middle - last
                     for first, middle, last in zip(zero_a, a_b, zero_b)]
            conventional = [first + middle - last
                            for first, middle, last in zip(zero_a, a_b, zero_b)]
            wrong = [first + middle - last
                     for first, middle, last in zip(
                         zero_a, a_b, zero_b[1:] + zero_b[:1])]
            raw = list(a_b)
            for index, semantic in enumerate(vector):
                max_cycle_error = max(
                    max_cycle_error, abs(cycle[index] - semantic))
                raw_error_sum += abs(raw[index] - semantic)
                raw_error_count += 1
            self.assertEqual(conventional, cycle)
            self.assertNotEqual(wrong, cycle)
        self.assertLess(max_cycle_error, 1e-12)
        self.assertGreater(raw_error_sum / raw_error_count, 1.0)


# ---------------------------------------------------------------------------
# resident cycle field and persistence
# ---------------------------------------------------------------------------
class ResidentCycleFieldTests(unittest.TestCase):
    def setUp(self):
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.cells = self.cells_module.ResidentCells(
            bytearray(self.cells_module.REGION_BYTES), initialize=True)

    def test_48_triangles_cancel_gauge_on_real_rails(self):
        cells = self.cells
        module = self.cells_module
        rng = random.Random(2029)
        dimensions = 48
        semantics = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        gauge = [rng.gauss(0.0, 3.0) for _ in range(256)]
        for index, semantic in enumerate(semantics):
            a = 1 + 2 * index
            b = a + 1
            store_skew(cells, 0, a, gauge[0] - gauge[a])
            store_skew(cells, a, b, semantic + gauge[a] - gauge[b])
            store_skew(cells, 0, b, gauge[0] - gauge[b])

        errors = []
        for index, semantic in enumerate(semantics):
            a = 1 + 2 * index
            b = a + 1
            errors.append(abs(resident_triangle_flux(module, cells, a, b)
                              - semantic))
        self.assertLess(max(errors), 1e-4)

        # Reverse antisymmetry is exact in the two independent rails.
        for index in range(dimensions):
            a = 1 + 2 * index
            b = a + 1
            for first, second in ((0, a), (a, b), (0, b)):
                self.assertEqual(
                    directed_weight(module, cells, first, second)
                    + directed_weight(module, cells, second, first),
                    0.0)

        # A second independent vertex gauge must not change any flux.
        second_gauge = [rng.gauss(0.0, 3.0) for _ in range(256)]
        for index in range(dimensions):
            a = 1 + 2 * index
            b = a + 1
            store_skew(cells, 0, a, second_gauge[0] - second_gauge[a])
            store_skew(cells, a, b, second_gauge[a] - second_gauge[b])
            store_skew(cells, 0, b, second_gauge[0] - second_gauge[b])
        shifted_errors = []
        for index, semantic in enumerate(semantics):
            a = 1 + 2 * index
            b = a + 1
            shifted_errors.append(
                abs(resident_triangle_flux(module, cells, a, b) - semantic))
        self.assertLess(max(shifted_errors), 1e-4)

        for head in (0, 1, 48, 97, 255):
            record = cells.read_head(head)
            self.assertEqual(record.head_id, head)
            self.assertEqual(record.identity, module.identity_address(head))


class CycleHyperedgeMembershipTests(unittest.TestCase):
    def setUp(self):
        self.universe_module = load_module(
            "nanlmind_hermes.resident.universe")
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "cycle-hyperedge.nlmres"
        self.universe = self.universe_module.ResidentUniverse.create(
            self.path, size=UNIVERSE_BYTES)

    def tearDown(self):
        try:
            self.universe.close()
        except Exception:
            pass

    def test_relation_hyperedge_binds_pair_and_mirror_members(self):
        receipt = self.universe.append_relation({
            "subject": "penguin",
            "predicate": "is_a",
            "object": "bird",
            "confidence": 0.8,
            "polarity": "supports",
            "head": 17,
        })
        self.assertTrue(receipt["created"])
        relation = self.universe.get_relation(receipt["relation_id"])
        self.assertIsNotNone(relation)
        self.assertIsNotNone(relation["pair_id"])
        self.assertIsNotNone(relation["mirror_id"])
        pair_id = int(relation["pair_id"], 16)
        mirror_id = int(relation["mirror_id"], 16)
        self.assertGreaterEqual(pair_id, 0)
        self.assertLess(pair_id, self.cells_module.PAIR_COUNT)
        self.assertGreaterEqual(mirror_id, 0)
        self.assertLess(mirror_id, 1 << 16)

        edge = self.universe.get_hyperedge(receipt["hyperedge_id"])
        self.assertIsNotNone(edge)
        self.assertGreaterEqual(edge["arity"], 6)
        kinds = [member["kind"] for member in edge["members"]]
        for kind in ("leaf", "pair", "mirror", "relation"):
            self.assertEqual(kinds.count(kind), 1 if kind != "leaf" else 3,
                             kind)
        by_kind = {member["kind"]: member for member in edge["members"]}
        self.assertEqual(int(by_kind["pair"]["id"], 16), pair_id)
        self.assertEqual(int(by_kind["mirror"]["id"], 16), mirror_id)
        self.assertEqual(int(by_kind["relation"]["id"], 16),
                         int(receipt["relation_id"], 16))

        traversal = self.universe.hyperedge_members(receipt["hyperedge_id"])
        self.assertFalse(traversal["truncated"])
        self.assertEqual(traversal["arity"], edge["arity"])
        self.assertEqual(traversal["members"], edge["members"])


class CyclePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.universe_module = load_module(
            "nanlmind_hermes.resident.universe")
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "cycle-persistence.nlmres"
        self.universe = self.universe_module.ResidentUniverse.create(
            self.path, size=UNIVERSE_BYTES)

    def tearDown(self):
        try:
            self.universe.close()
        except Exception:
            pass

    def test_cycle_state_and_hyperedge_survive_universe_reopen(self):
        module = self.cells_module
        rng = random.Random(29029)
        dimensions = 6
        semantics = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        gauge = [rng.gauss(0.0, 2.0) for _ in range(256)]
        for index, semantic in enumerate(semantics):
            a = 1 + 2 * index
            b = a + 1
            store_skew(self.universe.cells, 0, a, gauge[0] - gauge[a])
            store_skew(self.universe.cells, a, b,
                       semantic + gauge[a] - gauge[b])
            store_skew(self.universe.cells, 0, b, gauge[0] - gauge[b])

        receipt = self.universe.append_relation({
            "subject": "river",
            "predicate": "carries",
            "object": "water",
            "confidence": 0.7,
            "head": 23,
        })
        before = [resident_triangle_flux(module, self.universe.cells,
                                         1 + 2 * index, 2 + 2 * index)
                  for index in range(dimensions)]
        before_rails = []
        for index in range(dimensions):
            a = 1 + 2 * index
            b = a + 1
            for first, second in ((0, a), (a, b), (0, b)):
                state = self.universe.cells.pair_state(first, second)
                before_rails.append((first, second,
                                     state.forward.weight,
                                     state.reverse.weight))
        self.universe.flush()
        generation = self.universe.generation
        digest = self.universe.digest
        self.universe.close()

        reopened = self.universe_module.ResidentUniverse.open(self.path)
        try:
            after = [resident_triangle_flux(module, reopened.cells,
                                            1 + 2 * index, 2 + 2 * index)
                     for index in range(dimensions)]
            for expected, actual in zip(before, after):
                self.assertEqual(actual, expected)
            for index, semantic in enumerate(semantics):
                self.assertLess(abs(after[index] - semantic), 1e-4)
            for first, second, forward, reverse in before_rails:
                state = reopened.cells.pair_state(first, second)
                self.assertEqual(state.forward.weight, forward)
                self.assertEqual(state.reverse.weight, reverse)
            self.assertEqual(reopened.generation, generation)
            self.assertEqual(reopened.digest, digest)
            self.assertIsNotNone(
                reopened.get_hyperedge(receipt["hyperedge_id"]))
            self.assertFalse(
                reopened.status()["canonical_merge"]["enabled"])
        finally:
            reopened.close()


# ---------------------------------------------------------------------------
# temporal + field + cycle bridge
# ---------------------------------------------------------------------------
class _CycleLearningBridge:
    """Test-local bridge: temporal LM + independent field + cycle state.

    The production field learner already owns the temporal and head-field
    channels.  This small adapter only turns its committed coupling deltas into
    a divergence-free cycle state, which is the missing wiring tested here.
    """

    def __init__(self, temporal, field, cycle):
        self.temporal = temporal
        self.field = field
        self.cycle = cycle

    def observe(self, text: str):
        result = self.field.observe(text)
        edges = {
            (int(item["left"]), int(item["right"])): float(item["delta"])
            for item in result.get("coupling_updates", [])
        }
        if edges:
            self.cycle.observe(edges, rate=1.0)
        return result


class CycleLearningBridgeTests(unittest.TestCase):
    def setUp(self):
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.field_learning = load_module(
            "nanlmind_hermes.resident.field_learning")
        self.temporal_module = load_module("nanlmind_hermes.temporal_lm")
        self.cycle_state_module = load_module("nanlmind_hermes.cycle_state")
        cells = self.cells_module.ResidentCells(
            bytearray(self.cells_module.REGION_BYTES), initialize=True)
        temporal = self.temporal_module.TemporalMirrorLM()
        self.cells = cells
        self.temporal = temporal
        self.learner = self.field_learning.FieldLearner.from_cells(
            temporal, cells)
        self.cycle = self.cycle_state_module.CycleState(256)
        self.bridge = _CycleLearningBridge(temporal, self.learner, self.cycle)

    def test_temporal_and_field_and_cycle_update_together(self):
        vocab_before = len(self.temporal.vocab)
        result = self.bridge.observe(
            "A penguin swims in cold water because it is a bird.")
        self.assertTrue(result["learned"])
        self.assertTrue(result["accepted"])
        self.assertTrue(result["temporal_learned"])
        self.assertGreater(len(self.temporal.vocab), vocab_before)
        self.assertGreater(result["active_heads"], 1)
        self.assertGreater(len(result["head_updates"]), 1)
        self.assertGreater(len(result["coupling_updates"]), 0)
        self.assertGreater(result["writes"], 0)
        self.assertGreater(result["entropy"], 0.0)
        self.assertFalse(result["canonical_merge"])
        self.assertFalse(result["semantic_truth"])

        self.assertGreater(len(self.cycle.values), 0)
        self.assertGreater(self.cycle.energy(), 0.0)
        snapshot = self.cycle.snapshot()
        restored = self.cycle_state_module.CycleState(256)
        restored.restore(snapshot)
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.energy(), self.cycle.energy())

        first_left, first_right, first_delta = (
            result["coupling_updates"][0]["left"],
            result["coupling_updates"][0]["right"],
            result["coupling_updates"][0]["delta"],
        )
        state = self.cells.pair_state(first_left, first_right)
        rail = (state.forward
                if self.cells_module.address_of(first_left, first_right)
                == state.canonical else state.reverse)
        self.assertGreater(rail.weight, 0)
        self.assertGreater(first_delta, 0.0)

        stats = self.learner.stats()
        self.assertGreaterEqual(stats["temporal"]["observations"], 1)
        self.assertFalse(stats["canonical_merge"])
        self.assertFalse(stats["semantic_truth"])
        self.assertFalse(hasattr(self.learner, "merge"))

        evaluation = self.learner.evaluate([{
            "text": "A bird can fly over the river."}])
        self.assertTrue(evaluation["temporal"]["available"])
        self.assertIsNotNone(
            evaluation["temporal"]["mean_negative_log_likelihood"])
        self.assertFalse(evaluation["canonical_merge"])


# ---------------------------------------------------------------------------
# tools: approval and bounds
# ---------------------------------------------------------------------------
class _FieldToolBackend:
    """Minimal real-state backend satisfying the field_tools contract."""

    def __init__(self, field):
        self.field = field
        self.generation = 0
        self.proposals = {}
        self.sequence = 0

    def snapshot(self):
        activation = self.field.field_activation()
        return {
            "opened": True,
            "generation": self.generation,
            "heads": self.field.read_all(),
            "active_heads": activation["active_indices"],
            "budget": {
                "bytes": UNIVERSE_BYTES,
                "used_bytes": 0,
                "max_bytes": 1024 * MIB,
            },
        }

    def activate(self, head, delta):
        record = self.field.activate(head, delta)
        self.generation += 1
        return {
            "accepted": True,
            "generation": self.generation,
            "score": record["score"],
            "state": record["state"],
            "active": record["score"] > 0.0,
        }

    def propose(self, head, delta, reason):
        self.sequence += 1
        token = "round029-%04d" % self.sequence
        self.proposals[token] = (head, delta, reason)
        return {"token": token, "reasons": ["scratch_only"]}

    def commit(self, token, expected_generation):
        if expected_generation != self.generation:
            return {"accepted": False, "reasons": ["stale_generation"],
                    "generation": self.generation}
        staged = self.proposals.pop(token, None)
        if staged is None:
            return {"accepted": False, "reasons": ["unknown_token"],
                    "generation": self.generation}
        head, delta, _reason = staged
        self.field.activate(head, delta)
        self.generation += 1
        return {"accepted": True, "generation": self.generation,
                "head": head, "delta": delta}

    def head_metrics(self, head):
        return self.field.read(head)

    def field_metrics(self):
        activation = self.field.field_activation()
        return {
            "head_count": 256,
            "active_heads": activation["active_indices"],
            "active_count": activation["active_heads"],
            "coverage": activation["active_fraction"],
            "generation": self.generation,
        }

    def resonance(self, text, top_k):
        seed = sum(text.encode("utf-8")) % 256
        return [{"head": (seed + index) % 256,
                 "score": round(1.0 / (index + 1), 8)}
                for index in range(min(top_k, 3))]


class FieldToolApprovalBoundsTests(unittest.TestCase):
    def setUp(self):
        self.tools = load_module("nanlmind_hermes.resident.field_tools")
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.field_module = load_module(
            "nanlmind_hermes.resident.head_field")
        cells = self.cells_module.ResidentCells(
            bytearray(self.cells_module.REGION_BYTES), initialize=True)
        self.backend = _FieldToolBackend(
            self.field_module.ResidentHeadField(cells))

    def test_missing_approval_bounds_and_failures_never_mutate(self):
        status = self.tools.status({}, backend=self.backend)
        self.assertTrue(status["read_only"])
        self.assertFalse(status["writes_main"])
        self.assertFalse(status["canonical_merge"]["enabled"])
        self.assertFalse(status["canonical_merge"]["attempted"])
        self.assertFalse(status["canonical_promotion"])

        before = self.backend.field.read(3)
        with self.assertRaises(self.tools.FieldApprovalError):
            self.tools.activate({"head": 3, "delta": 0.5},
                                backend=self.backend)
        for params in (
            {"approved": True, "head": 256, "delta": 0.5},
            {"approved": True, "head": -1, "delta": 0.5},
            {"approved": True, "head": 3, "delta": 1.5},
            {"approved": True, "head": 3, "delta": float("nan")},
            {"approved": True, "head": 3, "delta": 0.5, "extra": 1},
        ):
            with self.assertRaises(self.tools.FieldValidationError):
                self.tools.activate(params, backend=self.backend)
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)

        proposal = self.tools.propose(
            {"head": 3, "delta": 0.5, "reason": "round029"},
            backend=self.backend)
        self.assertTrue(proposal["scratch_only"])
        self.assertFalse(proposal["writes_main"])
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)
        token = proposal["token"]

        with self.assertRaises(self.tools.FieldApprovalError):
            self.tools.commit(
                {"token": token, "expected_generation": 0},
                backend=self.backend)
        stale = self.tools.commit(
            {"approved": True, "token": token, "expected_generation": 7},
            backend=self.backend)
        self.assertFalse(stale["accepted"])
        self.assertTrue(stale["denied"])
        self.assertFalse(stale["writes_main"])
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)

        committed = self.tools.commit(
            {"approved": True, "token": token, "expected_generation": 0},
            backend=self.backend)
        self.assertTrue(committed["accepted"])
        self.assertEqual(committed["generation"], 1)
        self.assertAlmostEqual(self.backend.field.read(3)["score"], 0.5,
                               places=6)
        self.assertFalse(committed["canonical_merge"]["enabled"])
        self.assertFalse(committed["canonical_promotion"])

        generation = self.backend.generation
        query = self.tools.query({"head": 3}, backend=self.backend)
        self.assertTrue(query["read_only"])
        self.assertEqual(query["mode"], "head")
        self.assertEqual(self.backend.generation, generation)
        with self.assertRaises(self.tools.FieldValidationError):
            self.tools.query({"head": 3, "text": "penguin"},
                             backend=self.backend)


# ---------------------------------------------------------------------------
# native C++ comparison: skip or fail honestly
# ---------------------------------------------------------------------------
REQUIRED_NATIVE_KEYS = (
    "root_bytes", "mirror_pairs", "max_gauge_error", "max_reverse_error",
    "finite_difference_error", "loss_before_step", "loss_after_step",
    "reads", "read_milliseconds", "mlock_status", "passed",
)


def compare_native_payload(payload, expected):
    """Compare a native mirror JSON payload with the recorded summary.

    ``expected is None`` is an explicit skip status.  Any structural or
    numerical mismatch raises ``AssertionError``; it is never silently treated
    as a pass.
    """
    if expected is None:
        return {"status": "skipped", "reason": "recorded summary unavailable"}
    if not isinstance(payload, dict):
        raise AssertionError("native payload must be a JSON object")
    missing = [key for key in REQUIRED_NATIVE_KEYS if key not in payload]
    if missing:
        raise AssertionError("native payload missing keys: %s" % missing)
    if payload["root_bytes"] != expected["root_bytes"]:
        raise AssertionError("native root_bytes differs from the summary")
    if payload["mirror_pairs"] != expected["mirror_pairs"]:
        raise AssertionError("native mirror_pairs differs from the summary")
    if payload["mirror_pairs"] != 32_640:
        raise AssertionError("native mirror mapping is not 32,640")
    if float(payload["max_gauge_error"]) > 3e-6:
        raise AssertionError("native max_gauge_error exceeds the gate")
    if float(payload["max_reverse_error"]) != 0.0:
        raise AssertionError("native reverse antisymmetry is not exact")
    if float(payload["finite_difference_error"]) >= 2e-3:
        raise AssertionError("native finite-difference error exceeds the gate")
    if float(payload["loss_after_step"]) >= float(payload["loss_before_step"]):
        raise AssertionError("native gradient step did not reduce the loss")
    if payload["reads"] != 500_000:
        raise AssertionError("native read count differs from 500,000")
    if payload["passed"] is not True:
        raise AssertionError("native proof-of-implementation did not pass")
    return {
        "status": "ok",
        "root_bytes": payload["root_bytes"],
        "mirror_pairs": payload["mirror_pairs"],
        "max_gauge_error": float(payload["max_gauge_error"]),
        "max_reverse_error": float(payload["max_reverse_error"]),
    }


class NativeCompareTests(unittest.TestCase):
    def _expected(self):
        if not ROUND029_SUMMARY.is_file():
            self.skipTest("Round 029 summary.json is absent")
        summary = json.loads(ROUND029_SUMMARY.read_text(encoding="utf-8"))
        native = summary.get("native")
        if not isinstance(native, dict):
            self.skipTest("Round 029 summary has no native result")
        return native

    def test_compare_helper_skips_without_expected_and_fails_on_mismatch(self):
        expected = self._expected()
        self.assertEqual(
            compare_native_payload({"unused": True}, None)["status"],
            "skipped")
        bad = dict(expected)
        bad["mirror_pairs"] = 1
        with self.assertRaises(AssertionError):
            compare_native_payload(bad, expected)
        bad = dict(expected)
        bad["max_gauge_error"] = 1.0
        with self.assertRaises(AssertionError):
            compare_native_payload(bad, expected)

    def test_native_mirror_matches_summary_when_toolchain_is_available(self):
        expected = self._expected()
        if not NATIVE_SOURCE.is_file():
            self.skipTest("native_mirror.cpp is absent")
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("no C++20 compiler is available")
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "native_mirror"
            build = subprocess.run(
                [compiler, "-std=c++20", "-O3", "-Wall", "-Wextra",
                 str(NATIVE_SOURCE), "-o", str(binary)],
                cwd=str(ROOT), capture_output=True, text=True, timeout=120)
            if build.returncode != 0:
                self.fail("native compile failed:\n%s" % build.stderr[-2000:])
            run = subprocess.run(
                [str(binary)], cwd=str(ROOT), capture_output=True,
                text=True, timeout=120)
            if run.returncode != 0:
                self.fail("native compare failed with code %d:\n%s"
                          % (run.returncode, run.stderr[-2000:]))
        lines = [line for line in run.stdout.splitlines() if line.strip()]
        if not lines:
            self.fail("native program emitted no JSON")
        payload = json.loads(lines[-1])
        result = compare_native_payload(payload, expected)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mirror_pairs"], 32_640)
        self.assertEqual(result["max_reverse_error"], 0.0)


# ---------------------------------------------------------------------------
# benchmark and canonical-merge policy
# ---------------------------------------------------------------------------
class BenchmarkDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.bench = load_module("scripts.head_field_benchmark")
        if not self.bench.cells_available():
            self.skipTest("resident cells are unavailable for the benchmark")

    @staticmethod
    def _without_timing(value):
        if isinstance(value, dict):
            return {
                key: BenchmarkDeterminismTests._without_timing(item)
                for key, item in value.items()
                if key not in {"timing", "rates"}
            }
        if isinstance(value, list):
            return [BenchmarkDeterminismTests._without_timing(item)
                    for item in value]
        return value

    def test_fixed_seed_is_deterministic_and_merge_stays_false(self):
        first = self.bench.run_benchmark(8, 4, seed=4242)
        second = self.bench.run_benchmark(8, 4, seed=4242)
        self.assertEqual(first["deterministic_fingerprint"],
                         second["deterministic_fingerprint"])
        self.assertEqual(self._without_timing(first),
                         self._without_timing(second))
        self.assertEqual(json.dumps(self._without_timing(first),
                                    sort_keys=True),
                         json.dumps(self._without_timing(second),
                                    sort_keys=True))
        self.assertFalse(first["scope"]["canonical_merge_performed"])
        self.assertFalse(
            first["merge_contract"]["canonical_merge_performed"])
        self.assertFalse(first["scope"]["native_code_used"])
        self.assertFalse(first["scope"]["phone_performance_claimed"])
        self.assertTrue(first["verdict"]["passed"])

        different = self.bench.run_benchmark(8, 4, seed=4243)
        self.assertNotEqual(first["deterministic_fingerprint"],
                            different["deterministic_fingerprint"])


class CanonicalMergeFalseTests(unittest.TestCase):
    def test_every_reported_authority_flag_stays_false(self):
        field_learning = load_module(
            "nanlmind_hermes.resident.field_learning")
        field_persistence = load_module(
            "nanlmind_hermes.resident.field_persistence")
        self.assertIs(field_learning.CANONICAL_MERGE, False)
        self.assertIs(
            field_learning.HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH, False)

        class EmptyUniverse:
            def ingest(self, record):
                return {"generation": 0,
                        "leaf_id": record.get("leaf_id", "")}

            def get_leaf(self, leaf_id):
                return None

            def snapshot(self):
                return {"generation": 0, "digest": "0" * 64}

        persistence = field_persistence.FieldPersistence(EmptyUniverse())
        status = persistence.status()
        self.assertFalse(status["canonical_commit"])
        self.assertFalse(status["canonical_state_modified"])
        denial = persistence.save_field({"cycle": [1.0, -1.0]},
                                        approved=False)
        self.assertFalse(denial)
        self.assertEqual(denial["status"], "approval_required")
        self.assertFalse(denial["canonical_commit"])
        self.assertFalse(denial["canonical_state_modified"])

        cells_module = load_module("nanlmind_hermes.resident.cells")
        head_field = load_module("nanlmind_hermes.resident.head_field")
        tools = load_module("nanlmind_hermes.resident.field_tools")
        cells = cells_module.ResidentCells(
            bytearray(cells_module.REGION_BYTES), initialize=True)
        backend = _FieldToolBackend(head_field.ResidentHeadField(cells))
        result = tools.status({}, backend=backend)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_merge"]["attempted"])
        self.assertFalse(result["canonical_merge_enabled"])
        self.assertFalse(result["canonical_promotion"])
        self.assertFalse(result["canonical_promotion_enabled"])
        self.assertFalse(result["promotion_enabled"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
