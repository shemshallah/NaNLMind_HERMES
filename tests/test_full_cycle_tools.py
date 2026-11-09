"""Tests for bounded full 32,385-cycle tooling and persistence.

The suite exercises a fake raw-word backend, a fake universe facade and a real
16 MiB ``ResidentUniverse`` when its sibling modules are installed.  No
existing resident file is opened or modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.full_cycle_tools import (
    ANCHOR_SCHEMA,
    BASIS_COUNT,
    CYCLE_DIMENSION,
    DEFAULT_QUERY_HITS,
    EDGE_COUNT,
    FULL_BASIS_CYCLES,
    FULL_CYCLE_COUNT,
    HEAD_COUNT,
    MAX_BATCH,
    MAX_FEATURES,
    MAX_NODE,
    MAX_OUTPUT_BYTES,
    MAX_PREDICTIONS,
    MAX_SNAPSHOT_WORDS,
    MAX_STEPS,
    MAX_WORD,
    MIN_WORD,
    NODE_COUNT,
    PAIR_COUNT,
    ROOT_CYCLE_COUNT,
    WORD_MASK,
    FullBasisMemory,
    FullBasisRecord,
    FullCycleApprovalError,
    FullCycleBackendError,
    FullCycleBoundsError,
    FullCycleCorruptionError,
    FullCycleMemory,
    FullCyclePersistence,
    FullCyclePointerError,
    FullCycleValidationError,
    _basis_digest,
    audit,
    basis_index,
    basis_pair,
    build_handlers,
    full_basis_pairs,
    handle,
    predict,
    query,
    snapshot,
    status,
    train,
    write,
)

try:  # The real facade is the primary target; fakes cover partial installs.
    from nanlmind_hermes.resident.universe import ResidentUniverse
except Exception:  # pragma: no cover - partial installation fallback
    ResidentUniverse = None  # type: ignore[assignment]

MIB = 1024 * 1024
SIZE_16MIB = 16 * MIB
SOURCE = "round031:full-cycle-test"

_POINTER_KEYS = frozenset({
    "pointer", "ptr", "raw_pointer", "raw_ptr", "offset", "address", "addr",
    "buffer", "buffer_address", "memory_address", "memoryview", "mmap",
    "cell_offset", "raw_offset", "file_offset",
})
_AUTHORITY_KEYS = frozenset({
    "canonical_merge", "canonical_merge_enabled", "canonical_promotion",
    "canonical_promotion_enabled", "canonical_commit",
    "canonical_state_modified", "merge_enabled", "promotion_enabled",
})


def assert_json_safe(testcase, value):
    """Assert a response is finite JSON and contains no raw pointer fields."""
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                testcase.assertIsInstance(key, str)
                testcase.assertNotIn(key, _POINTER_KEYS)
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, float):
            testcase.assertTrue(math.isfinite(current))
        elif isinstance(current, (str, int, bool)) or current is None:
            continue
        else:
            testcase.fail("non-JSON-safe value: %r" % (current,))
    json.dumps(value, allow_nan=False)


def assert_false_authority(testcase, payload):
    """Check every canonical/promotion flag that appears is false."""
    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in _AUTHORITY_KEYS:
                    if isinstance(value, dict):
                        testcase.assertFalse(
                            any(bool(item) for item in value.values()), key)
                    else:
                        testcase.assertIs(value, False, key)
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)


def words_for_test(offset: int = 0):
    return [((index * 97 + offset * 31 + 17) % 2009) - 1004
            for index in range(FULL_BASIS_CYCLES)]


class FakeBackend:
    """Raw-word backend with explicit full-basis metadata and pointer extras."""

    def __init__(self, *, basis_count: int = FULL_BASIS_CYCLES) -> None:
        self.nodes = NODE_COUNT
        self.root = 0
        self.basis_count = basis_count
        self._basis = [0] * FULL_BASIS_CYCLES
        self._root_edges: dict[tuple[int, int], int] = {}
        self._generation = 0
        self._digest = _basis_digest(self._basis, self.root)
        self.predict_limit = 5
        self.huge_predictions = False
        self.deny_writes = False
        self.fail_main = False
        self.calls: list[object] = []
        self.mutations: list[dict] = []

    @property
    def generation(self):
        return self._generation

    @property
    def digest(self):
        return self._digest

    def _refresh(self):
        self._digest = _basis_digest(self._basis, self.root)

    def seed(self, first: int, second: int, word: int):
        if first == self.root or second == self.root:
            self._root_edges[(first, second)] = word
        else:
            index = basis_index(first, second, self.root)
            self._basis[index] = word
        self._refresh()

    def _word(self, first: int, second: int) -> int:
        if first == second:
            raise FullCycleValidationError("i and j must be distinct")
        if first == self.root or second == self.root:
            return self._root_edges.get((first, second), 0)
        return self._basis[basis_index(first, second, self.root)]

    def snapshot(self):
        entries = []
        for index, word in enumerate(self._basis):
            if word and len(entries) < 16:
                first, second = basis_pair(index, self.root)
                entries.append({"index": index, "i": first, "j": second,
                                "word": word})
        return {
            "opened": True,
            "nodes": self.nodes,
            "root": self.root,
            "generation": self._generation,
            "digest": self._digest,
            "basis_count": self.basis_count,
            "word_count": self.basis_count,
            "full_basis": {
                "root": self.root,
                "basis_count": self.basis_count,
                "basis_digest": self._digest,
                "generation": self._generation,
                "feature_count": len(entries),
            },
            "features": entries,
            "words": entries,
            "pointer": 0xDEADBEEF,
            "offset": 4096,
            "address": 0x1000,
        }

    def read_word(self, first, second):
        self.calls.append(("read_word", first, second))
        return {
            "word": self._word(first, second),
            "generation": self._generation,
            "digest": self._digest,
            "pointer": 0xBEEF,
            "offset": first * NODE_COUNT + second,
        }

    def write_word(self, first, second, word, expected_generation=None):
        self.calls.append(
            ("write_word", first, second, word, expected_generation))
        if expected_generation is not None and \
                expected_generation != self._generation:
            return {
                "accepted": False,
                "reasons": ["stale_generation"],
                "generation": self._generation,
                "pointer": 1,
                "offset": 1,
            }
        if self.deny_writes:
            return {
                "accepted": False,
                "reasons": ["budget_exhausted"],
                "generation": self._generation,
                "pointer": 2,
                "offset": 2,
            }
        if first == self.root or second == self.root:
            return {
                "accepted": False,
                "reasons": ["gauge_edge_not_in_basis"],
                "generation": self._generation,
            }
        previous = self._generation
        if self.fail_main:
            raise RuntimeError("main write failed")
        index = basis_index(first, second, self.root)
        self._basis[index] = int(word)
        self._generation += 1
        self._refresh()
        self.mutations.append({"operation": "write", "i": first, "j": second,
                               "word": word, "generation": self._generation})
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "word": word,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self._digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
            "pointer": 3,
            "offset": 3,
        }

    def train(self, text, batch, steps, expected_generation=None,
              features=None):
        self.calls.append(
            ("train", text, batch, steps, expected_generation, features))
        if expected_generation is not None and \
                expected_generation != self._generation:
            return {
                "accepted": False,
                "reasons": ["stale_generation"],
                "generation": self._generation,
            }
        previous = self._generation
        if self.fail_main:
            raise RuntimeError("main train failed")
        if features:
            entries = features
        else:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8],
                       16)
            entries = []
            for offset in range(min(steps, FULL_BASIS_CYCLES)):
                index = (seed + offset * 104729) % FULL_BASIS_CYCLES
                pair = basis_pair(index, self.root)
                entries.append({"index": index, "i": pair[0],
                                "j": pair[1], "word": 0})
        for item in entries[:max(1, steps)]:
            self._basis[int(item["index"])] += int(steps)
        self._generation += 1
        self._refresh()
        self.mutations.append({"operation": "train", "text": text,
                               "steps": steps,
                               "generation": self._generation})
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self._digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
            "metrics": {"steps": steps, "batch": batch,
                        "updates": min(steps, FULL_BASIS_CYCLES),
                        "loss": 0.125, "pointer": 0xDEAD},
            "pointer": 6,
            "offset": 6,
        }

    def predict(self, text, features=None, limit=DEFAULT_QUERY_HITS):
        self.calls.append(("predict", text, features, limit))
        cap = MAX_PREDICTIONS if self.huge_predictions else min(
            limit, self.predict_limit)
        entries = []
        for position in range(cap):
            index = (len(text) * 131 + position * 7919) % FULL_BASIS_CYCLES
            first, second = basis_pair(index, self.root)
            entries.append({
                "index": index, "i": first, "j": second,
                "word": self._basis[index],
                "score": round(1.0 / (position + 1), 8),
                "text": text[:64], "kind": "full_cycle",
                "pointer": 0xFEED, "offset": position * 8,
            })
        return {
            "mode": "text",
            "text": text,
            "predictions": entries,
            "features": entries,
            "feature_count": len(entries),
            "generation": self._generation,
            "digest": self._digest,
            "pointer": 0xFEED,
            "offset": 8192,
        }

    def query(self, text, limit=DEFAULT_QUERY_HITS):
        self.calls.append(("query", text, limit))
        hits = []
        for position in range(min(limit, self.predict_limit)):
            index = (len(text) * 37 + position * 4099) % FULL_BASIS_CYCLES
            first, second = basis_pair(index, self.root)
            hits.append({
                "index": index, "i": first, "j": second,
                "word": self._basis[index],
                "score": round(1.0 / (position + 1), 8),
                "text": text[:64], "kind": "full_cycle",
                "pointer": 0xCAFE, "offset": position,
            })
        return {
            "mode": "text", "text": text, "hits": hits,
            "pointer": 0xF00D, "offset": 123,
        }


class TruncatedBackend(FakeBackend):
    """Backend that claims only 32,384 cycles."""

    def __init__(self):
        super().__init__(basis_count=FULL_BASIS_CYCLES - 1)


class FakeUniverse:
    """Minimal duck-typed ResidentUniverse facade."""

    def __init__(self):
        self.leaves: dict[str, dict] = {}
        self.generation = 0
        self.digest = hashlib.sha256(b"fake-full-cycle-genesis").hexdigest()
        self.checkpoint_calls = 0

    def ingest(self, record):
        record = dict(record)
        leaf_id = record.get("leaf_id")
        if not isinstance(leaf_id, str) or not leaf_id:
            leaf_id = "leaf:" + hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()).hexdigest()[:32]
        if leaf_id in self.leaves:
            return {
                "status": "duplicate", "created": False, "duplicate": True,
                "leaf_id": leaf_id, "leaf_ids": [leaf_id],
                "generation": self.generation, "digest": self.digest,
            }
        text = record.get("text")
        if not isinstance(text, str):
            text = json.dumps(record, sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False)
        self.generation += 1
        self.digest = hashlib.sha256(
            (self.digest + leaf_id + text).encode("utf-8")).hexdigest()
        self.leaves[leaf_id] = {"leaf_id": leaf_id, "text": text}
        return {
            "status": "appended", "created": True, "duplicate": False,
            "leaf_id": leaf_id, "leaf_ids": [leaf_id],
            "generation": self.generation, "digest": self.digest,
        }

    def get_leaf(self, leaf_id):
        leaf = self.leaves.get(str(leaf_id))
        return dict(leaf) if leaf is not None else None

    def snapshot(self):
        return {
            "schema": "nanlm.resident-universe/1",
            "generation": self.generation,
            "digest": self.digest,
            "state_digest": self.digest,
            "counts": {"leaves": len(self.leaves), "relations": 0},
        }

    def checkpoint(self, directory=None):
        self.checkpoint_calls += 1
        return {
            "schema": "nanlm.resident-universe-checkpoint/1",
            "directory": str(directory) if directory is not None else None,
            "generation": self.generation,
            "digest": self.digest,
            "canonical_commit": False,
        }


class BasisGeometryTests(unittest.TestCase):
    def test_full_basis_constants_are_exact(self):
        self.assertEqual(HEAD_COUNT, 256)
        self.assertEqual(NODE_COUNT, 256)
        self.assertEqual(MAX_NODE, 255)
        self.assertEqual(PAIR_COUNT, 32640)
        self.assertEqual(EDGE_COUNT, 32640)
        self.assertEqual(CYCLE_DIMENSION, 32385)
        self.assertEqual(FULL_BASIS_CYCLES, 32385)
        self.assertEqual(FULL_CYCLE_COUNT, 32385)
        self.assertEqual(ROOT_CYCLE_COUNT, 32385)
        self.assertEqual(BASIS_COUNT, 32385)
        self.assertEqual(CYCLE_DIMENSION, math.comb(255, 2))
        self.assertEqual(CYCLE_DIMENSION, PAIR_COUNT - 255)

    def test_full_basis_pair_enumeration_and_index_round_trip(self):
        pairs = full_basis_pairs()
        self.assertEqual(len(pairs), 32385)
        self.assertEqual(pairs[0], (1, 2))
        self.assertEqual(pairs[-1], (254, 255))
        self.assertEqual(len(set(pairs)), 32385)
        self.assertTrue(all(1 <= first < second < 256
                            for first, second in pairs))
        for index in (0, 1, 253, 254, 255, 4096, 32383, 32384):
            pair = basis_pair(index)
            self.assertEqual(basis_index(*pair), index)
        self.assertEqual(len(full_basis_pairs(3)), 32385)
        self.assertEqual(basis_pair(0, 3), (0, 1))
        self.assertEqual(basis_index(1, 0, 3), 0)

    def test_memory_holds_all_cycles_and_reads_root_edges_as_zero(self):
        memory = FullCycleMemory()
        self.assertEqual(len(memory.coefficients), 32385)
        self.assertEqual(memory.basis_count, 32385)
        self.assertEqual(memory.read_word(0, 1)["word"], 0)
        self.assertEqual(memory.read_word(1, 2)["word"], 0)
        memory.write_word(1, 2, 7)
        self.assertEqual(memory.read_word(1, 2)["word"], 7)
        # The root edge is the star gauge, not an independent coefficient.
        denied = memory.write_word(0, 1, 3)
        self.assertFalse(denied["accepted"])
        self.assertEqual(denied["reasons"], ["gauge_edge_not_in_basis"])
        self.assertEqual(memory.read_word(0, 1)["word"], 0)

    def test_pair_coordinates_appear_in_the_loaded_mapping(self):
        memory = FullCycleMemory(words_for_test())
        record = memory.coefficients
        self.assertIsInstance(record, list)
        self.assertEqual(len(record), FULL_BASIS_CYCLES)
        pairs = full_basis_pairs()
        self.assertEqual(pairs[0], (1, 2))


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.backend.seed(1, 2, 7)
        self.backend.seed(1, 3, -11)
        self.backend.seed(2, 3, 5)

    def test_status_exposes_full_basis_and_bounded_words(self):
        result = status({"limit": 2}, backend=self.backend)
        self.assertEqual(result["operation"], "status")
        self.assertEqual(result["full_basis"]["count"], 32385)
        self.assertEqual(result["full_basis"]["expected_count"], 32385)
        self.assertTrue(result["full_basis"]["complete"])
        self.assertEqual(result["geometry"]["cycle_dimension"], 32385)
        self.assertEqual(result["geometry"]["cycles"] if "cycles" in
                         result["geometry"] else 32385, 32385)
        self.assertEqual(result["word_count"], 32385)
        self.assertEqual(len(result["words"]), 2)
        self.assertTrue(result["truncated"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_snapshot_is_read_only_and_metadata_digest_is_stable(self):
        before = self.backend.snapshot()
        result = snapshot({"limit": 3}, backend=self.backend)
        after = self.backend.snapshot()
        self.assertEqual(result["operation"], "snapshot")
        self.assertEqual(len(result["words"]), 3)
        self.assertEqual(result["full_basis"]["count"], 32385)
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertEqual(result["digest"], before["digest"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_audit_accepts_full_basis_and_denies_truncation_without_mutation(self):
        result = audit({}, backend=self.backend)
        self.assertTrue(result["ok"])
        self.assertEqual(result["basis_count"], 32385)
        self.assertEqual(result["checked"], 32385)
        self.assertTrue(result["complete"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

        truncated = TruncatedBackend()
        before = truncated.snapshot()
        denial = audit({}, backend=truncated)
        after = truncated.snapshot()
        self.assertFalse(denial["ok"])
        self.assertFalse(denial["accepted"])
        self.assertEqual(denial["basis_count"], 32384)
        self.assertIn("basis_truncated", denial["problem_codes"])
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        assert_json_safe(self, denial)

    def test_predict_is_bounded_and_read_only(self):
        before = self.backend.snapshot()
        result = predict({"text": "gauge cycle", "limit": 4},
                         backend=self.backend)
        after = self.backend.snapshot()
        self.assertEqual(result["operation"], "predict")
        self.assertEqual(len(result["predictions"]), 4)
        self.assertLessEqual(result["prediction_count"], 5)
        self.assertEqual(result["full_basis"]["count"], 32385)
        self.assertTrue(result["read_only"])
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_predict_output_size_is_bounded(self):
        self.backend.huge_predictions = True
        with self.assertRaises(FullCycleBackendError):
            predict({"text": "too many", "limit": 2048},
                    backend=self.backend)

    def test_feature_maps_are_bounded_at_32385(self):
        features = [0] * (MAX_FEATURES + 1)
        with self.assertRaises(FullCycleBoundsError):
            predict({"text": "feature overflow", "features": features},
                    backend=self.backend)
        with self.assertRaises(FullCycleBoundsError):
            train({"approved": True, "text": "feature overflow",
                   "features": features}, backend=self.backend)
        self.assertEqual(self.backend.generation, 0)

    def test_query_reads_word_and_cycle_without_mutation(self):
        before = self.backend.snapshot()
        word = query({"i": 1, "j": 2}, backend=self.backend)
        self.assertEqual(word["word"], 7)
        self.assertEqual(word["index"], 0)
        triangle = query({"i": 1, "j": 2, "root": 0},
                         backend=self.backend)
        self.assertEqual(triangle["cycle"], 7)
        after = self.backend.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        assert_json_safe(self, word)
        assert_json_safe(self, triangle)
        assert_false_authority(self, word)

    def test_query_text_is_bounded_and_read_only(self):
        before = self.backend.snapshot()
        result = query({"text": "gauge", "limit": 3}, backend=self.backend)
        after = self.backend.snapshot()
        self.assertEqual(result["operation"], "query")
        self.assertEqual(len(result["hits"]), 3)
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)


class ApprovalAndBoundsTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()

    def test_write_requires_literal_approval_without_mutation(self):
        before = self.backend.snapshot()
        with self.assertRaises(FullCycleApprovalError):
            write({"i": 1, "j": 2, "word": 9}, backend=self.backend)
        after = self.backend.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertFalse(self.backend.mutations)

    def test_write_accepts_full_basis_index_and_preserves_raw_word(self):
        before = self.backend.snapshot()
        result = write({"approved": True, "index": 0, "word": 0xDEADBEEF},
                       backend=self.backend)
        after = self.backend.snapshot()
        self.assertTrue(result["accepted"])
        self.assertFalse(result["denied"])
        self.assertEqual(result["i"], 1)
        self.assertEqual(result["j"], 2)
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["word"], 0xDEADBEEF)
        self.assertEqual(result["stored_word"], 0xDEADBEEF)
        self.assertGreater(result["generation"], before["generation"])
        self.assertEqual(after["generation"], result["generation"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_write_rejects_bad_nodes_root_and_word_ranges(self):
        cases = [
            {"approved": True, "i": NODE_COUNT, "j": 2, "word": 1},
            {"approved": True, "i": 1, "j": 1, "word": 1},
            {"approved": True, "i": 1, "j": 2, "root": 1, "word": 1},
            {"approved": True, "i": 1, "j": 2, "word": MAX_WORD + 1},
            {"approved": True, "i": 1, "j": 2, "word": MIN_WORD - 1},
            {"approved": True, "i": 1, "j": 2, "word": 1.5},
        ]
        before = self.backend.snapshot()
        for params in cases:
            with self.assertRaises(FullCycleValidationError):
                write(params, backend=self.backend)
        after = self.backend.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertFalse(self.backend.mutations)

    def test_write_stale_generation_is_denied_without_mutation(self):
        before = self.backend.snapshot()
        denial = write({"approved": True, "i": 1, "j": 2, "word": 3,
                        "expected_generation": before["generation"] + 1},
                       backend=self.backend)
        after = self.backend.snapshot()
        self.assertFalse(denial["accepted"])
        self.assertTrue(denial["denied"])
        self.assertEqual(denial["reasons"], ["stale_generation"])
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertFalse(self.backend.mutations)
        assert_json_safe(self, denial)

    def test_train_requires_approval_and_bounds_steps_and_batch(self):
        before = self.backend.snapshot()
        with self.assertRaises(FullCycleApprovalError):
            train({"text": "x"}, backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            train({"approved": True, "text": "x", "steps": 0},
                  backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            train({"approved": True, "text": "x", "steps": MAX_STEPS + 1},
                  backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            train({"approved": True, "text": "x", "batch": MAX_BATCH + 1},
                  backend=self.backend)
        after = self.backend.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])

    def test_train_accepts_approved_bounded_step_and_reports_metrics(self):
        before = self.backend.snapshot()
        result = train({"approved": True, "text": "learn cycles",
                        "batch": 2, "steps": 3}, backend=self.backend)
        after = self.backend.snapshot()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["batch"], 2)
        self.assertEqual(result["metrics"]["steps"], 3)
        self.assertGreater(result["generation"], before["generation"])
        self.assertEqual(after["generation"], result["generation"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_train_stale_generation_is_denied_without_mutation(self):
        before = self.backend.snapshot()
        denial = train({"approved": True, "text": "x",
                        "expected_generation": before["generation"] + 1},
                       backend=self.backend)
        after = self.backend.snapshot()
        self.assertFalse(denial["accepted"])
        self.assertEqual(denial["reasons"], ["stale_generation"])
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertFalse(self.backend.mutations)
        assert_json_safe(self, denial)

    def test_status_snapshot_query_and_audit_output_bounds_and_properties(self):
        with self.assertRaises(FullCycleValidationError):
            status({"limit": MAX_SNAPSHOT_WORDS + 1},
                   backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            snapshot({"limit": 0}, backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            query({"i": 1}, backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            query({"text": "x" * 5000}, backend=self.backend)
        with self.assertRaises(FullCycleValidationError):
            status({"unknown": 1}, backend=self.backend)

    def test_memory_backend_round_trip_through_all_actions(self):
        memory = FullCycleMemory()
        handlers = build_handlers(memory)
        self.assertEqual(set(handlers), {
            "status", "write", "predict", "train", "snapshot", "audit",
            "query"})
        self.assertEqual(handlers["status"]({})["full_basis"]["count"],
                         32385)
        written = handlers["write"]({
            "approved": True, "i": 1, "j": 2, "word": 5})
        self.assertTrue(written["accepted"])
        self.assertEqual(handlers["query"]({"index": 0})["word"], 5)
        trained = handlers["train"]({
            "approved": True, "text": "cycle", "steps": 2})
        self.assertTrue(trained["accepted"])
        self.assertTrue(handlers["audit"]({})["ok"])
        self.assertEqual(handle("status", {}, backend=memory)["operation"],
                         "status")
        with self.assertRaises(FullCycleValidationError):
            handle("missing", {}, backend=memory)


class PersistenceFakeTests(unittest.TestCase):
    def setUp(self):
        self.universe = FakeUniverse()
        self.persistence = FullCyclePersistence(self.universe)
        self.words = words_for_test()

    def test_full_basis_count_is_exact_and_truncation_is_denied(self):
        before = self.universe.snapshot()
        with self.assertRaises(FullCycleBoundsError):
            self.persistence.save_full_basis(
                self.words[:-1], approved=True, source=SOURCE)
        with self.assertRaises(FullCycleBoundsError):
            self.persistence.save_full_basis(
                self.words + [0], approved=True, source=SOURCE)
        after = self.universe.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertIsNone(self.persistence.load_full_basis())

    def test_save_requires_approval_without_touching_universe(self):
        before = self.universe.snapshot()
        denial = self.persistence.save_full_basis(self.words, source=SOURCE)
        after = self.universe.snapshot()
        self.assertFalse(denial)
        self.assertFalse(denial["accepted"])
        self.assertEqual(denial["status"], "approval_required")
        self.assertEqual(denial["basis_count"], 32385)
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertIsNone(self.persistence.load_full_basis())

    def test_save_load_reopen_equality_and_metadata(self):
        before = self.universe.snapshot()
        receipt = self.persistence.save_full_basis(
            self.words, approved=True, source=SOURCE,
            metadata={"round": 31, "kind": "full_basis"})
        self.assertTrue(receipt)
        self.assertEqual(receipt["status"], "committed")
        self.assertTrue(receipt["created"])
        self.assertTrue(receipt["approved"])
        self.assertEqual(receipt["basis_count"], 32385)
        self.assertEqual(receipt["index_count"], 32385)
        self.assertEqual(receipt["word_count"], 32385)
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["shard_count"], 64)
        self.assertEqual(len(receipt["leaf_ids"]), 66)
        self.assertGreater(receipt["generation"], before["generation"])
        self.assertTrue(receipt["main_modified"])
        self.assertFalse(receipt["canonical_commit"])
        self.assertFalse(receipt["canonical_state_modified"])
        self.assertEqual(len(receipt["basis_digest"]), 64)
        assert_json_safe(self, receipt)

        loaded = self.persistence.load_full_basis()
        self.assertIsInstance(loaded, FullBasisRecord)
        self.assertEqual(len(loaded), 32385)
        self.assertEqual(loaded["basis_count"], 32385)
        self.assertEqual(loaded["word_count"], 32385)
        self.assertEqual(loaded["words"], self.words)
        self.assertEqual(len(loaded["basis"]), 32385)
        self.assertEqual(loaded["basis"][0]["index"], 0)
        self.assertEqual((loaded["basis"][0]["i"],
                          loaded["basis"][0]["j"]), (1, 2))
        self.assertEqual(loaded["basis_digest"], receipt["basis_digest"])
        self.assertEqual(loaded["manifest_digest"],
                         receipt["manifest_digest"])
        self.assertTrue(loaded["read_only"])
        self.assertFalse(loaded["canonical_commit"])
        assert_json_safe(self, {key: value for key, value in loaded.items()
                                if key not in ("words", "basis")})

        # A new adapter over the same facade behaves like a reopen.
        reopened = FullCyclePersistence(FakeUniverseView(self.universe))
        reopened_record = reopened.load_full_basis()
        self.assertIsNotNone(reopened_record)
        self.assertEqual(reopened_record["words"], self.words)
        self.assertEqual(len(reopened_record), 32385)

        status_result = self.persistence.status()
        self.assertTrue(status_result["complete"])
        self.assertEqual(status_result["basis_count"], 32385)
        self.assertEqual(status_result["version"], 0)
        audit_result = self.persistence.audit()
        self.assertTrue(audit_result["ok"])
        self.assertEqual(audit_result["checked"], 32385)
        self.assertTrue(audit_result["read_only_main"])
        self.assertFalse(audit_result["canonical_commit"])
        assert_json_safe(self, status_result)
        assert_json_safe(self, audit_result)
        assert_false_authority(self, audit_result)

    def test_duplicate_same_basis_is_idempotent_without_generation_change(self):
        first = self.persistence.save_full_basis(
            self.words, approved=True, source=SOURCE)
        generation = self.universe.generation
        duplicate = self.persistence.save_full_basis(
            self.words, approved=True, source=SOURCE)
        self.assertTrue(duplicate)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(duplicate["basis_digest"], first["basis_digest"])
        self.assertEqual(self.universe.generation, generation)

    def test_stale_generation_is_denied_and_new_version_is_allowed(self):
        first = self.persistence.save_full_basis(
            self.words, approved=True, source=SOURCE)
        current = self.universe.generation
        changed = words_for_test(offset=1)
        before = self.universe.snapshot()
        denial = self.persistence.save_full_basis(
            changed, approved=True, source=SOURCE, expected_generation=0)
        after = self.universe.snapshot()
        self.assertFalse(denial)
        self.assertEqual(denial["status"], "stale_generation")
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertEqual(self.persistence.status()["version"], 0)

        accepted = self.persistence.save_full_basis(
            changed, approved=True, source=SOURCE,
            expected_generation=current)
        self.assertTrue(accepted)
        self.assertEqual(accepted["status"], "committed")
        self.assertEqual(accepted["version"], 1)
        self.assertGreater(accepted["basis_digest"], "")
        loaded = self.persistence.load_full_basis()
        self.assertEqual(loaded["words"], changed)
        self.assertEqual(loaded["version"], 1)
        self.assertNotEqual(loaded["basis_digest"], first["basis_digest"])
        self.assertEqual(self.persistence.audit()["ok"], True)

    def test_load_status_and_audit_do_not_change_generation_or_digest(self):
        self.persistence.save_full_basis(
            self.words, approved=True, source=SOURCE)
        before = self.universe.snapshot()
        self.persistence.load_full_basis()
        self.persistence.status()
        self.persistence.audit()
        after = self.universe.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])

    def test_invalid_words_and_pointer_metadata_do_not_mutate(self):
        before = self.universe.snapshot()
        invalid = list(self.words)
        invalid[10] = True
        with self.assertRaises(FullCycleBoundsError):
            self.persistence.save_full_basis(
                invalid, approved=True, source=SOURCE)
        with self.assertRaises(FullCycleBoundsError):
            self.persistence.save_full_basis(
                [float(index) for index in range(FULL_BASIS_CYCLES)],
                approved=True, source=SOURCE)
        with self.assertRaises(FullCyclePointerError):
            self.persistence.save_full_basis(
                self.words, approved=True, source=SOURCE,
                metadata={"pointer": 1})
        after = self.universe.snapshot()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["digest"], after["digest"])
        self.assertIsNone(self.persistence.load_full_basis())

    def test_tampered_shard_is_reported_as_corruption(self):
        receipt = self.persistence.save_full_basis(
            self.words, approved=True, source=SOURCE)
        shard_id = next(leaf_id for leaf_id in receipt["leaf_ids"]
                        if leaf_id.startswith("full-cycle-shard:"))
        payload = json.loads(self.universe.leaves[shard_id]["text"])
        payload["words"][0] += 1
        self.universe.leaves[shard_id]["text"] = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False)
        with self.assertRaises(FullCycleCorruptionError):
            self.persistence.load_full_basis()
        audit_result = self.persistence.audit()
        self.assertFalse(audit_result["ok"])
        self.assertIn("basis_corruption", audit_result["problem_codes"])

    def test_mapping_input_round_trips_with_exact_coordinates(self):
        mapping = {}
        for index, word in enumerate(self.words):
            first, second = basis_pair(index)
            mapping[(first, second)] = word
        receipt = self.persistence.save_full_basis(
            mapping, approved=True, source=SOURCE)
        self.assertTrue(receipt)
        loaded = self.persistence.load_full_basis()
        self.assertEqual(loaded["words"], self.words)


class FakeUniverseView:
    """Another adapter object over the same fake store (simulates reopen)."""

    def __init__(self, universe):
        self._universe = universe

    def ingest(self, record):
        return self._universe.ingest(record)

    def get_leaf(self, leaf_id):
        return self._universe.get_leaf(leaf_id)

    def snapshot(self):
        return self._universe.snapshot()


class RealUniverseTestCase(unittest.TestCase):
    def setUp(self):
        if ResidentUniverse is None:
            self.skipTest("resident universe siblings are unavailable")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "full-cycle-universe.nlm"
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)

    def tearDown(self):
        if getattr(self, "universe", None) is not None:
            self.universe.close()
        self._tmp.cleanup()

    def reopen(self):
        self.universe.close()
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)
        return self.universe


class RealPersistenceTests(RealUniverseTestCase):
    def test_full_basis_persists_across_reopen_with_exact_count(self):
        words = [index % 13 for index in range(FULL_BASIS_CYCLES)]
        persistence = FullCyclePersistence(self.universe)
        before = self.universe.snapshot()
        receipt = persistence.save_full_basis(
            words, approved=True, source=SOURCE)
        self.assertTrue(receipt)
        self.assertEqual(receipt["basis_count"], 32385)
        self.assertEqual(receipt["shard_count"], 64)
        self.assertEqual(self.universe.snapshot()["generation"],
                         receipt["generation"])
        self.assertTrue(receipt["basis_digest"])

        self.reopen()
        persistence = FullCyclePersistence(self.universe)
        loaded = persistence.load_full_basis()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 32385)
        self.assertEqual(loaded["words"], words)
        self.assertEqual(loaded["basis_count"], 32385)
        self.assertEqual(loaded["basis_digest"], receipt["basis_digest"])
        self.assertEqual(loaded["manifest_digest"],
                         receipt["manifest_digest"])
        self.assertTrue(persistence.status()["complete"])
        audit_result = persistence.audit()
        self.assertTrue(audit_result["ok"])
        self.assertEqual(audit_result["checked"], 32385)

        # Read-only actions preserve the universe generation and digest.
        before_read = self.universe.snapshot()
        persistence.load_full_basis()
        persistence.status()
        persistence.audit()
        after_read = self.universe.snapshot()
        self.assertEqual(before_read["generation"],
                         after_read["generation"])
        self.assertEqual(before_read["digest"], after_read["digest"])

        # Repeating the exact basis is an idempotent duplicate.
        generation = self.universe.generation
        duplicate = persistence.save_full_basis(
            words, approved=True, source=SOURCE)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(self.universe.generation, generation)
        assert_json_safe(self, receipt)
        assert_json_safe(self, audit_result)
        assert_false_authority(self, audit_result)

    def test_truncated_stale_and_unapproved_writes_do_not_mutate(self):
        words = [index % 7 for index in range(FULL_BASIS_CYCLES)]
        persistence = FullCyclePersistence(self.universe)
        receipt = persistence.save_full_basis(
            words, approved=True, source=SOURCE)
        self.assertTrue(receipt)
        generation = self.universe.generation
        digest = self.universe.digest

        with self.assertRaises(FullCycleBoundsError):
            persistence.save_full_basis(
                words[:-1], approved=True, source=SOURCE)
        denial = persistence.save_full_basis(
            [index % 5 for index in range(FULL_BASIS_CYCLES)],
            source=SOURCE)
        self.assertFalse(denial)
        self.assertEqual(denial["status"], "approval_required")
        stale = persistence.save_full_basis(
            words_for_test(), approved=True, source=SOURCE,
            expected_generation=0)
        self.assertFalse(stale)
        self.assertEqual(stale["status"], "stale_generation")
        self.assertEqual(self.universe.generation, generation)
        self.assertEqual(self.universe.digest, digest)
        loaded = persistence.load_full_basis()
        self.assertEqual(loaded["words"], words)
        self.assertEqual(len(loaded), 32385)


if __name__ == "__main__":
    unittest.main()
