"""Tests for bounded, dependency-injected resident cycle tool handlers."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import unittest

from nanlmind_hermes.resident import cycle_tools
from nanlmind_hermes.resident.cycle_tools import (
    DEFAULT_QUERY_HITS,
    DEFAULT_SNAPSHOT_WORDS,
    DEFAULT_STATUS_WORDS,
    EDGE_COUNT,
    MAX_BATCH,
    MAX_OUTPUT_BYTES,
    MAX_QUERY_HITS,
    MAX_SNAPSHOT_WORDS,
    MAX_STEPS,
    MAX_TEXT_BYTES,
    MAX_WORD,
    MIN_WORD,
    NODE_COUNT,
    CycleApprovalError,
    CycleBackendError,
    CycleToolError,
    CycleValidationError,
    cycle,
    gauge,
    query,
    set as set_edge,
    snapshot,
    status,
    train,
)

_POINTER_KEYS = frozenset({
    "pointer", "ptr", "raw_pointer", "offset", "address", "buffer",
    "memoryview", "cell_offset", "raw_offset", "file_offset", "mmap",
})

_AUTHORITY_KEYS = frozenset({
    "canonical", "canonical_merge", "canonical_merge_enabled",
    "canonical_authority", "canonical_commit", "canonical_state_modified",
    "canonical_promotion", "canonical_promotion_enabled", "merge_enabled",
    "promotion_allowed", "promotion_enabled",
})


def assert_json_safe(testcase, value):
    """Assert a response is JSON-safe and contains no raw pointer fields."""
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


def assert_false_authority(testcase, payload, path=""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            here = "%s.%s" % (path, key) if path else str(key)
            if key in _AUTHORITY_KEYS:
                if isinstance(value, dict):
                    testcase.assertFalse(
                        any(bool(item) for item in value.values()), here)
                else:
                    testcase.assertIs(value, False, here)
            assert_false_authority(testcase, value, here)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_false_authority(
                testcase, value, "%s[%d]" % (path, index))


def words_digest(words):
    material = json.dumps(
        sorted([[key[0], key[1], word] for key, word in words.items()]),
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class CycleToolsBackend:
    """Fake raw integer field used by the handler tests.

    Every result deliberately carries ``pointer``/``offset`` extras so the
    handlers' projection and sanitisation can be checked.  Writes are staged
    in ``scratch`` before the main word map is changed.
    """

    def __init__(self, *, nodes=NODE_COUNT, root=0):
        self.nodes = nodes
        self.root = root
        self._generation = 0
        self._words = {}
        self._scratch = {}
        self._errors = []
        self.calls = []
        self.events = []
        self.mutations = []
        self.fail_main = False
        self.deny_writes = False
        self.deny_reason = "budget_exhausted"
        self.corrupt_cycle = False

    # -- introspection --------------------------------------------------
    @property
    def generation(self):
        return self._generation

    @property
    def digest(self):
        return words_digest(self._words)

    def main_state(self):
        return {
            "generation": self._generation,
            "digest": self.digest,
            "words": {
                "%d,%d" % key: value
                for key, value in sorted(self._words.items())
            },
        }

    def seed(self, first, second, word):
        """Place a main word without going through the handler."""
        self._words[(first, second)] = word

    def snapshot(self):
        self.calls.append("snapshot")
        return {
            "opened": True,
            "nodes": self.nodes,
            "root": self.root,
            "generation": self._generation,
            "digest": self.digest,
            "word_count": len(self._words),
            "words": {
                "%d,%d" % key: value
                for key, value in sorted(self._words.items())
            },
            "pointer": 0xDEADBEEF,
            "offset": 4096,
            "address": 0x1000,
        }

    def read_word(self, first, second):
        self.calls.append(("read_word", first, second))
        return {
            "word": self._words.get((first, second), 0),
            "generation": self._generation,
            "digest": self.digest,
            "pointer": 0xBEEF,
            "offset": first * 256 + second,
        }

    def cycle(self, first, second, root):
        self.calls.append(("cycle", first, second, root))
        value = (self._words.get((root, first), 0)
                 + self._words.get((first, second), 0)
                 - self._words.get((root, second), 0))
        if self.corrupt_cycle:
            value += 1
        return {"value": value, "pointer": 0xCAFE, "offset": 512}

    # -- writes ---------------------------------------------------------
    def _stale(self, expected_generation):
        return expected_generation is not None and \
            expected_generation != self._generation

    def set_word(self, first, second, word, expected_generation=None):
        self.calls.append(
            ("set_word", first, second, word, expected_generation))
        if self._stale(expected_generation):
            return {
                "accepted": False,
                "reasons": ["stale_generation"],
                "generation": self._generation,
                "pointer": 0x1,
                "offset": 1,
            }
        if self.deny_writes:
            return {
                "accepted": False,
                "reasons": [self.deny_reason],
                "generation": self._generation,
                "pointer": 0x2,
                "offset": 2,
            }
        previous = self._generation
        self.events.append("scratch")
        self._scratch[(first, second)] = word
        if self.fail_main:
            raise RuntimeError("main write failed")
        self._words[(first, second)] = self._scratch.pop((first, second))
        self._generation += 1
        self.events.append("main")
        self.mutations.append({
            "operation": "set",
            "i": first,
            "j": second,
            "word": word,
            "generation": self._generation,
        })
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "word": word,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self.digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
            "pointer": 0x3,
            "offset": 3,
        }

    def train(self, text, batch, steps, expected_generation=None):
        self.calls.append(
            ("train", text, batch, steps, expected_generation))
        if self._stale(expected_generation):
            return {
                "accepted": False,
                "reasons": ["stale_generation"],
                "generation": self._generation,
                "pointer": 0x4,
                "offset": 4,
            }
        if self.deny_writes:
            return {
                "accepted": False,
                "reasons": [self.deny_reason],
                "generation": self._generation,
                "pointer": 0x5,
                "offset": 5,
            }
        previous = self._generation
        self.events.append("scratch")
        self._scratch[("train", text)] = steps
        if self.fail_main:
            raise RuntimeError("main train failed")
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        first = seed % (NODE_COUNT - 1) + 1
        second = (first + 1 + seed % 17) % NODE_COUNT
        if second == first:
            second = (second + 1) % NODE_COUNT
        self._words[(first, second)] = (
            self._words.get((first, second), 0) + steps)
        self._scratch.pop(("train", text), None)
        self._generation += 1
        self.events.append("main")
        self.mutations.append({
            "operation": "train",
            "text": text,
            "batch": batch,
            "steps": steps,
            "generation": self._generation,
        })
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self.digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
            "metrics": {
                "steps": steps,
                "batch": batch,
                "updates": steps,
                "loss": 0.125,
                "pointer": 0xDEAD,
                "offset": 123,
            },
            "pointer": 0x6,
            "offset": 6,
        }

    def apply_gauge(self, root, potential, *, edges=None):
        """Apply an additive vertex gauge directly to main raw words."""
        keys = set(edges or ())
        keys.update(self._words)
        for key in list(keys):
            first, second = key
            keys.add((second, first))
        for first, second in keys:
            self._words[(first, second)] = (
                self._words.get((first, second), 0)
                + potential[second] - potential[first])
        self._generation += 1
        self.mutations.append({
            "operation": "gauge",
            "root": root,
            "generation": self._generation,
        })

    # -- queries --------------------------------------------------------
    def query(self, text, limit=DEFAULT_QUERY_HITS):
        self.calls.append(("query", text, limit))
        seed = sum(ord(character) for character in text)
        hits = []
        for index in range(limit):
            first = (seed + index * 37) % NODE_COUNT
            second = (first + 1 + index) % NODE_COUNT
            if second == first:
                second = (second + 1) % NODE_COUNT
            hits.append({
                "i": first,
                "j": second,
                "word": self._words.get((first, second), index),
                "score": round(1.0 / (index + 1), 8),
                "text": text[:64],
                "kind": "raw_word",
                "pointer": 0xFEED,
                "offset": index * 8,
            })
        return {
            "mode": "text",
            "text": text,
            "hits": hits,
            "pointer": 0xFEED,
            "offset": 8192,
        }


class ResidentCycleFieldBackend:
    """Adapter over the real resident pair-rail raw words.

    Each resident pair rail stores a signed Q16.16 ``weight`` i32.  This test
    adapter reads and writes that word directly through the public rail offset
    helpers, so the real ``ResidentCells`` layout exercises the handlers.
    """

    def __init__(self):
        from nanlmind_hermes.resident import cells
        self.cells_module = cells
        self.cells = cells.ResidentCells(
            bytearray(cells.REGION_BYTES), initialize=True)
        self._generation = 0
        self._tracked = {}
        self._scratch = {}
        self.calls = []
        self.events = []
        self.mutations = []

    @property
    def generation(self):
        return self._generation

    @property
    def digest(self):
        return words_digest(self._tracked)

    def main_state(self):
        return {
            "generation": self._generation,
            "digest": self.digest,
            "words": {
                "%d,%d" % key: value
                for key, value in sorted(self._tracked.items())
            },
        }

    def _rail(self, first, second):
        canonical = self.cells_module.canonical_address(first, second)
        if self.cells_module.address_of(first, second) == canonical:
            return self.cells_module.RAIL_S_PLUS
        return self.cells_module.RAIL_S_MINUS

    def _write_raw(self, first, second, word):
        rail = self._rail(first, second)
        record = self.cells.read_rail(first, second, rail)
        updated = record._replace(
            weight=word,
            generation=(record.generation + 1) & 0xFFFFFFFF,
            flags=record.flags | self.cells_module.RAIL_FLAG_INITIALIZED)
        offset = self.cells.rail_record_offset(first, second, rail)
        self.cells_module.RAIL_STRUCT.pack_into(
            self.cells.buffer, offset, *updated)
        self._tracked[(first, second)] = word

    def snapshot(self):
        self.calls.append("snapshot")
        return {
            "opened": True,
            "nodes": NODE_COUNT,
            "root": 0,
            "generation": self._generation,
            "digest": self.digest,
            "word_count": len(self._tracked),
            "words": {
                "%d,%d" % key: value
                for key, value in sorted(self._tracked.items())
            },
            "pointer": 0xDEAD,
            "offset": 64,
        }

    def read_word(self, first, second):
        self.calls.append(("read_word", first, second))
        record = self.cells.read_rail(first, second, self._rail(first, second))
        return {
            "word": record.weight,
            "generation": self._generation,
            "digest": self.digest,
        }

    def set_word(self, first, second, word, expected_generation=None):
        self.calls.append(
            ("set_word", first, second, word, expected_generation))
        if expected_generation is not None and \
                expected_generation != self._generation:
            return {"accepted": False, "reasons": ["stale_generation"],
                    "generation": self._generation}
        previous = self._generation
        self.events.append("scratch")
        self._scratch[(first, second)] = word
        self._write_raw(first, second, word)
        self._scratch.pop((first, second), None)
        self._generation += 1
        self.events.append("main")
        self.mutations.append({
            "operation": "set",
            "i": first,
            "j": second,
            "word": word,
            "generation": self._generation,
        })
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "word": word,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self.digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
        }

    def query(self, text, limit=DEFAULT_QUERY_HITS):
        self.calls.append(("query", text, limit))
        return {"mode": "text", "text": text, "hits": []}

    def apply_gauge(self, root, potential, edges):
        keys = set(edges)
        for key in list(keys):
            keys.add((key[1], key[0]))
        for first, second in sorted(keys):
            self._write_raw(
                first, second,
                self._tracked.get((first, second), 0)
                + potential[second] - potential[first])
        self._generation += 1
        self.mutations.append({"operation": "gauge", "root": root,
                               "generation": self._generation})


class CycleStatusSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.backend = CycleToolsBackend()

    def test_status_reports_bounded_geometry_and_disables_canonical(self):
        state = self.backend.main_state()
        result = status({}, backend=self.backend)
        self.assertEqual(result["operation"], "status")
        self.assertEqual(result["nodes"], NODE_COUNT)
        self.assertEqual(result["root"], 0)
        self.assertEqual(result["generation"], 0)
        self.assertEqual(result["digest"], self.backend.digest)
        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["words"], [])
        self.assertEqual(result["geometry"]["edges"], EDGE_COUNT)
        self.assertEqual(result["geometry"]["cycle_dimension"],
                         EDGE_COUNT - (NODE_COUNT - 1))
        self.assertEqual(result["geometry"]["max_node"], NODE_COUNT - 1)
        self.assertTrue(result["raw_words"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        self.assertEqual(result["canonical_merge"],
                         {"enabled": False, "attempted": False})
        self.assertFalse(result["canonical_promotion"])
        self.assertFalse(result["canonical_promotion_enabled"])
        self.assertFalse(result["promotion_enabled"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.mutations, [])

    def test_status_and_snapshot_limits_are_bounded_and_sorted(self):
        self.backend.seed(9, 3, 30)
        self.backend.seed(1, 2, 20)
        self.backend.seed(0, 7, 10)
        limited = status({"limit": 2}, backend=self.backend)
        self.assertEqual([(item["i"], item["j"]) for item in limited["words"]],
                         [(0, 7), (1, 2)])
        self.assertEqual(limited["word_count"], 3)
        self.assertTrue(limited["truncated"])
        self.assertEqual(len(status({"limit": DEFAULT_STATUS_WORDS},
                                    backend=self.backend)["words"]), 3)

        snap = snapshot({"limit": 2}, backend=self.backend)
        self.assertEqual(snap["operation"], "snapshot")
        self.assertEqual(len(snap["words"]), 2)
        self.assertEqual(snap["word_count"], 3)
        self.assertTrue(snap["truncated"])
        self.assertEqual(snap["digest"], self.backend.digest)
        self.assertEqual(len(snapshot({"limit": MAX_SNAPSHOT_WORDS},
                                      backend=self.backend)["words"]), 3)
        assert_json_safe(self, snap)
        assert_false_authority(self, snap)

    def test_snapshot_sanitises_backend_extras_and_rejects_bad_limits(self):
        self.backend.seed(4, 5, -7)
        result = snapshot({}, backend=self.backend)
        self.assertEqual(result["words"], [{"i": 4, "j": 5, "word": -7}])
        assert_json_safe(self, result)
        for value in (0, MAX_SNAPSHOT_WORDS + 1, True, "2", 1.5, None):
            with self.subTest(value=value):
                with self.assertRaises(CycleToolError):
                    status({"limit": value}, backend=self.backend)
                with self.assertRaises(CycleToolError):
                    snapshot({"limit": value}, backend=self.backend)

    def test_status_and_snapshot_reject_unknown_properties(self):
        state = self.backend.main_state()
        for params in ({"pointer": 1}, {"words": []}, {"i": 0},
                       {"address": 1}):
            with self.subTest(params=params):
                with self.assertRaises(CycleValidationError):
                    status(params, backend=self.backend)
                with self.assertRaises(CycleValidationError):
                    snapshot(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, [])


class CycleSetTests(unittest.TestCase):
    def setUp(self):
        self.backend = CycleToolsBackend()

    def test_set_requires_literal_approved_true(self):
        state = self.backend.main_state()
        for params in (
            {"i": 1, "j": 2, "word": 10},
            {"approved": False, "i": 1, "j": 2, "word": 10},
            {"approved": 1, "i": 1, "j": 2, "word": 10},
            {"approved": "true", "i": 1, "j": 2, "word": 10},
            {"approved": None, "i": 1, "j": 2, "word": 10},
        ):
            with self.subTest(params=params):
                with self.assertRaises(CycleApprovalError):
                    set_edge(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, [])
        self.assertNotIn("set_word", [call[0] if isinstance(call, tuple)
                                      else call
                                      for call in self.backend.calls])

    def test_set_bounds_and_unknown_properties_do_not_mutate(self):
        state = self.backend.main_state()
        mutations = list(self.backend.mutations)
        cases = (
            {"approved": True, "j": 2, "word": 10},
            {"approved": True, "i": 1, "word": 10},
            {"approved": True, "i": 1, "j": 2},
            {"approved": True, "i": -1, "j": 2, "word": 10},
            {"approved": True, "i": NODE_COUNT, "j": 2, "word": 10},
            {"approved": True, "i": True, "j": 2, "word": 10},
            {"approved": True, "i": 1.0, "j": 2, "word": 10},
            {"approved": True, "i": "1", "j": 2, "word": 10},
            {"approved": True, "i": 1, "j": -1, "word": 10},
            {"approved": True, "i": 1, "j": NODE_COUNT, "word": 10},
            {"approved": True, "i": 1, "j": 1, "word": 10},
            {"approved": True, "i": 1, "j": 2, "word": True},
            {"approved": True, "i": 1, "j": 2, "word": 1.5},
            {"approved": True, "i": 1, "j": 2, "word": str(10)},
            {"approved": True, "i": 1, "j": 2, "word": MIN_WORD - 1},
            {"approved": True, "i": 1, "j": 2, "word": MAX_WORD + 1},
            {"approved": True, "i": 1, "j": 2, "word": 10,
             "expected_generation": -1},
            {"approved": True, "i": 1, "j": 2, "word": 10,
             "expected_generation": True},
            {"approved": True, "i": 1, "j": 2, "word": 10,
             "expected_generation": "0"},
            {"approved": True, "i": 1, "j": 2, "word": 10, "pointer": 1},
            {"approved": True, "i": 1, "j": 2, "word": 10, "root": 0},
            {"approved": True, "i": 1, "j": 2, "word": 10, "text": "x"},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(CycleToolError):
                    set_edge(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, mutations)
        self.assertNotIn("set_word", [call[0] if isinstance(call, tuple)
                                      else call
                                      for call in self.backend.calls])

    def test_set_stages_scratch_before_main_and_sanitises_receipt(self):
        result = set_edge({"approved": True, "i": 1, "j": 2, "word": 12345},
                          backend=self.backend)
        self.assertEqual(result["operation"], "set")
        self.assertTrue(result["accepted"])
        self.assertFalse(result["denied"])
        self.assertTrue(result["writes_main"])
        self.assertFalse(result["read_only"])
        self.assertEqual(result["i"], 1)
        self.assertEqual(result["j"], 2)
        self.assertEqual(result["word"], 12345)
        self.assertEqual(result["stored_word"], 12345)
        self.assertEqual(result["previous_generation"], 0)
        self.assertEqual(result["generation"], 1)
        self.assertTrue(result["scratch"]["written"])
        self.assertTrue(result["main"]["written"])
        self.assertEqual(result["events"], ["scratch", "main"])
        self.assertTrue(result["scratch_before_main"])
        self.assertEqual(result["receipt"]["word"], 12345)
        self.assertEqual(self.backend.events, ["scratch", "main"])
        self.assertEqual(self.backend.generation, 1)
        self.assertEqual(self.backend.read_word(1, 2)["word"], 12345)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_set_expected_generation_is_check_before_backend_write(self):
        state = self.backend.main_state()
        result = set_edge({"approved": True, "i": 1, "j": 2, "word": 10,
                           "expected_generation": 4}, backend=self.backend)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["denied"])
        self.assertEqual(result["reasons"], ["stale_generation"])
        self.assertEqual(result["generation"], 0)
        self.assertEqual(result["previous_generation"], 0)
        self.assertFalse(result["writes_main"])
        self.assertTrue(result["read_only"])
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.mutations, [])
        self.assertNotIn("set_word", [call[0] if isinstance(call, tuple)
                                      else call
                                      for call in self.backend.calls])
        assert_json_safe(self, result)

    def test_backend_stale_or_denied_write_never_touches_main(self):
        state = self.backend.main_state()
        for mode in ("stale", "denied"):
            self.backend.calls.clear()
            if mode == "stale":
                result = set_edge(
                    {"approved": True, "i": 1, "j": 2, "word": 10,
                     "expected_generation": 1}, backend=self.backend)
                # Handler-level CAS catches this before the backend call.
                self.assertEqual(result["reasons"], ["stale_generation"])
                self.assertNotIn(
                    "set_word", [call[0] if isinstance(call, tuple) else call
                                 for call in self.backend.calls])
            else:
                self.backend.deny_writes = True
                result = set_edge(
                    {"approved": True, "i": 1, "j": 2, "word": 10},
                    backend=self.backend)
                self.backend.deny_writes = False
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reasons"],
                                 ["budget_exhausted"])
                self.assertIn("set_word",
                              [call[0] if isinstance(call, tuple)
                               else call for call in self.backend.calls])
            self.assertEqual(self.backend.main_state(), state)
            self.assertEqual(self.backend.mutations, [])
            assert_json_safe(self, result)

    def test_failed_main_write_leaves_main_state_and_digest_unchanged(self):
        state = self.backend.main_state()
        self.backend.fail_main = True
        with self.assertRaises(CycleBackendError):
            set_edge({"approved": True, "i": 3, "j": 4, "word": 99},
                     backend=self.backend)
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.digest, state["digest"])
        self.assertEqual(self.backend.mutations, [])

    def test_scratch_main_order_violation_is_rejected(self):
        class ReorderedBackend(CycleToolsBackend):
            def set_word(self, first, second, word,
                         expected_generation=None):
                self._words[(first, second)] = word
                self._generation += 1
                return {
                    "accepted": True,
                    "generation": self._generation,
                    "events": ["main", "scratch"],
                    "scratch": {"written": True,
                                "generation": self._generation},
                    "main": {"written": True,
                             "generation": self._generation},
                }

        backend = ReorderedBackend()
        with self.assertRaises(cycle_tools.CycleInvariantError):
            set_edge({"approved": True, "i": 1, "j": 2, "word": 1},
                     backend=backend)


class CycleGeometryTests(unittest.TestCase):
    def setUp(self):
        self.backend = CycleToolsBackend()
        self.backend.seed(0, 1, 100)
        self.backend.seed(1, 2, 50)
        self.backend.seed(0, 2, 30)

    def test_cycle_is_exact_integer_geometry_and_read_only(self):
        state = self.backend.main_state()
        result = cycle({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.assertEqual(result["operation"], "cycle")
        self.assertEqual(result["i"], 1)
        self.assertEqual(result["j"], 2)
        self.assertEqual(result["root"], 0)
        self.assertEqual(result["value"], 120)
        self.assertEqual(result["word"], 120)
        self.assertEqual(result["raw_edges"],
                         {"root_i": 100, "i_j": 50, "root_j": 30})
        self.assertTrue(result["exact"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        self.assertEqual(self.backend.main_state(), state)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_cycle_handles_negative_and_mixed_words_exactly(self):
        self.backend.seed(0, 1, -100)
        self.backend.seed(1, 2, 25)
        self.backend.seed(0, 2, -200)
        result = cycle({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.assertEqual(result["value"], 125)

    def test_cycle_uses_backend_default_root(self):
        self.backend.root = 3
        self.backend.seed(3, 1, 40)
        self.backend.seed(3, 2, 10)
        result = cycle({"i": 1, "j": 2}, backend=self.backend)
        self.assertEqual(result["root"], 3)
        # x(3,1) + x(1,2) - x(3,2) = 40 + 50 - 10
        self.assertEqual(result["value"], 80)

    def test_cycle_rejects_backend_disagreement(self):
        self.backend.corrupt_cycle = True
        with self.assertRaises(cycle_tools.CycleInvariantError):
            cycle({"i": 1, "j": 2, "root": 0}, backend=self.backend)

    def test_cycle_and_gauge_bounds_and_unknown_properties_do_not_mutate(self):
        state = self.backend.main_state()
        mutations = list(self.backend.mutations)
        cases = (
            {"j": 2, "root": 0},
            {"i": 1, "root": 0},
            {"i": -1, "j": 2, "root": 0},
            {"i": NODE_COUNT, "j": 2, "root": 0},
            {"i": 1, "j": -1, "root": 0},
            {"i": 1, "j": NODE_COUNT, "root": 0},
            {"i": 1, "j": 2, "root": -1},
            {"i": 1, "j": 2, "root": NODE_COUNT},
            {"i": 1, "j": 2, "root": True},
            {"i": 1, "j": 1, "root": 0},
            {"i": 1, "j": 2, "root": 1},
            {"i": True, "j": 2, "root": 0},
            {"i": 1.0, "j": 2, "root": 0},
            {"i": 1, "j": 2, "root": 0, "pointer": 1},
            {"i": 1, "j": 2, "root": 0, "word": 5},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(CycleToolError):
                    cycle(params, backend=self.backend)
                with self.assertRaises(CycleToolError):
                    gauge(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, mutations)

    def test_gauge_cancellation_is_exact(self):
        result = gauge({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.assertEqual(result["operation"], "gauge")
        self.assertEqual(result["potentials"],
                         {"root": 0, "i": 100, "j": 30})
        self.assertEqual(result["gauge_edges"],
                         {"root_i": 100, "i_j": -70, "root_j": 30})
        self.assertEqual(result["gauge_cycle"], 0)
        self.assertEqual(result["cycle"], 120)
        self.assertEqual(result["invariant_cycle"], 120)
        self.assertEqual(result["cancellation_residual"], 0)
        self.assertTrue(result["cancelled"])
        self.assertTrue(result["read_only"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_arbitrary_vertex_gauge_leaves_cycle_bit_exact(self):
        before = cycle({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.backend.apply_gauge(
            0, [0, 41, -17, 500], edges={(0, 1), (1, 2), (0, 2)})
        after = cycle({"i": 1, "j": 2, "root": 0},
                      backend=self.backend)
        self.assertEqual(after["value"], before["value"])
        self.assertEqual(after["raw_edges"]["root_i"], 141)
        self.assertEqual(after["raw_edges"]["i_j"], 50 - 17 - 41)
        self.assertEqual(after["raw_edges"]["root_j"], 30 - 17)
        shifted = gauge({"i": 1, "j": 2, "root": 0},
                        backend=self.backend)
        self.assertEqual(shifted["cycle"], before["value"])
        self.assertEqual(shifted["gauge_cycle"], 0)
        self.assertEqual(shifted["cancellation_residual"], 0)
        self.assertTrue(shifted["cancelled"])

    def test_read_only_digest_is_stable_across_every_readout(self):
        state = self.backend.main_state()
        readouts = (
            status({}, backend=self.backend),
            snapshot({}, backend=self.backend),
            cycle({"i": 1, "j": 2, "root": 0}, backend=self.backend),
            gauge({"i": 1, "j": 2, "root": 0}, backend=self.backend),
            query({"i": 1, "j": 2}, backend=self.backend),
            query({"text": "cycle geometry"}, backend=self.backend),
        )
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.generation, 0)
        for result in readouts:
            self.assertEqual(result["digest"], state["digest"])
            self.assertTrue(result["read_only"])
            assert_json_safe(self, result)

    def test_gauge_backend_fallback_without_raw_read(self):
        class GaugeOnlyBackend:
            def snapshot(self):
                return {"nodes": NODE_COUNT, "root": 0, "generation": 0,
                        "digest": "0" * 64, "words": []}

            def gauge(self, first, second, root):
                return {"cycle": 77, "gauge_cycle": 0,
                        "cancellation_residual": 0,
                        "pointer": 0x1, "offset": 2}

            def cycle(self, first, second, root):
                return 77

        result = gauge({"i": 1, "j": 2, "root": 0},
                       backend=GaugeOnlyBackend())
        self.assertEqual(result["cycle"], 77)
        self.assertTrue(result["cancelled"])
        assert_json_safe(self, result)


class CycleTrainTests(unittest.TestCase):
    def setUp(self):
        self.backend = CycleToolsBackend()

    def test_train_requires_literal_approved_true(self):
        state = self.backend.main_state()
        for params in (
            {"text": "learn cycle"},
            {"approved": False, "text": "learn cycle"},
            {"approved": 1, "text": "learn cycle"},
            {"approved": "true", "text": "learn cycle"},
        ):
            with self.subTest(params=params):
                with self.assertRaises(CycleApprovalError):
                    train(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, [])
        self.assertNotIn("train", [call[0] if isinstance(call, tuple)
                                   else call
                                   for call in self.backend.calls])

    def test_train_bounds_text_batch_steps_and_unknown_properties(self):
        state = self.backend.main_state()
        cases = (
            {"approved": True},
            {"approved": True, "text": ""},
            {"approved": True, "text": "   "},
            {"approved": True, "text": 3},
            {"approved": True, "text": "x" * (MAX_TEXT_BYTES + 1)},
            {"approved": True, "text": "x", "batch": 0},
            {"approved": True, "text": "x", "batch": MAX_BATCH + 1},
            {"approved": True, "text": "x", "batch": True},
            {"approved": True, "text": "x", "batch": "2"},
            {"approved": True, "text": "x", "steps": 0},
            {"approved": True, "text": "x", "steps": MAX_STEPS + 1},
            {"approved": True, "text": "x", "steps": True},
            {"approved": True, "text": "x", "steps": "2"},
            {"approved": True, "text": "x", "expected_generation": -1},
            {"approved": True, "text": "x", "expected_generation": True},
            {"approved": True, "text": "x", "pointer": 1},
            {"approved": True, "text": "x", "i": 1},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(CycleToolError):
                    train(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, [])
        self.assertNotIn("train", [call[0] if isinstance(call, tuple)
                                   else call
                                   for call in self.backend.calls])

    def test_train_success_is_bounded_staged_and_sanitised(self):
        result = train({"approved": True, "text": "penguin cycle",
                        "batch": MAX_BATCH, "steps": MAX_STEPS},
                       backend=self.backend)
        self.assertEqual(result["operation"], "train")
        self.assertTrue(result["accepted"])
        self.assertTrue(result["applied"])
        self.assertTrue(result["writes_main"])
        self.assertEqual(result["text"], "penguin cycle")
        self.assertEqual(result["batch"], MAX_BATCH)
        self.assertEqual(result["steps"], MAX_STEPS)
        self.assertEqual(result["previous_generation"], 0)
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["events"], ["scratch", "main"])
        self.assertTrue(result["scratch_before_main"])
        self.assertEqual(result["metrics"]["steps"], MAX_STEPS)
        self.assertEqual(result["metrics"]["batch"], MAX_BATCH)
        self.assertNotIn("pointer", result["metrics"])
        self.assertNotIn("offset", result["metrics"])
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_json_safe(self, result)
        assert_false_authority(self, result)
        self.assertEqual(self.backend.generation, 1)

    def test_train_defaults_are_one_batch_one_step(self):
        result = train({"approved": True, "text": "one step"},
                       backend=self.backend)
        self.assertEqual(result["batch"], 1)
        self.assertEqual(result["steps"], 1)

    def test_train_accepts_batch_steps_text_backend_order(self):
        events = []

        class BatchFirstBackend(CycleToolsBackend):
            def train(self, batch, steps, text, expected_generation=None):
                events.append((batch, steps, text, expected_generation))
                return super().train(
                    text, batch, steps,
                    expected_generation=expected_generation)

        backend = BatchFirstBackend()
        result = train({"approved": True, "text": "ordered",
                        "batch": 3, "steps": 2,
                        "expected_generation": 0}, backend=backend)
        self.assertTrue(result["accepted"])
        self.assertEqual(events, [(3, 2, "ordered", 0)])
        self.assertEqual(result["metrics"]["batch"], 3)
        self.assertEqual(result["metrics"]["steps"], 2)

    def test_train_accepts_exact_text_bound(self):
        text = "x" * MAX_TEXT_BYTES
        result = train({"approved": True, "text": text},
                       backend=self.backend)
        self.assertEqual(result["text"], text)
        self.assertEqual(len(result["text"].encode("utf-8")),
                         MAX_TEXT_BYTES)

    def test_train_stale_generation_is_denied_before_backend_call(self):
        state = self.backend.main_state()
        result = train({"approved": True, "text": "stale",
                        "expected_generation": 2}, backend=self.backend)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["stale_generation"])
        self.assertEqual(result["generation"], 0)
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.mutations, [])
        self.assertNotIn("train", [call[0] if isinstance(call, tuple)
                                   else call
                                   for call in self.backend.calls])

    def test_train_backend_denial_is_read_only(self):
        self.backend.deny_writes = True
        state = self.backend.main_state()
        result = train({"approved": True, "text": "denied"},
                       backend=self.backend)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["denied"])
        self.assertEqual(result["reasons"], ["budget_exhausted"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.mutations, [])
        assert_json_safe(self, result)

    def test_train_failure_leaves_main_unchanged(self):
        state = self.backend.main_state()
        self.backend.fail_main = True
        with self.assertRaises(CycleBackendError):
            train({"approved": True, "text": "fails"},
                  backend=self.backend)
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.mutations, [])


class CycleQueryTests(unittest.TestCase):
    def setUp(self):
        self.backend = CycleToolsBackend()
        self.backend.seed(0, 1, 100)
        self.backend.seed(1, 2, 50)
        self.backend.seed(0, 2, 30)

    def test_query_word_is_read_only_and_sanitised(self):
        state = self.backend.main_state()
        result = query({"i": 1, "j": 2}, backend=self.backend)
        self.assertEqual(result["mode"], "word")
        self.assertEqual(result["i"], 1)
        self.assertEqual(result["j"], 2)
        self.assertEqual(result["word"], 50)
        self.assertEqual(result["value"], 50)
        self.assertEqual(result["digest"], state["digest"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        self.assertEqual(self.backend.main_state(), state)
        assert_json_safe(self, result)

    def test_query_word_with_root_includes_exact_cycle(self):
        result = query({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.assertEqual(result["word"], 50)
        self.assertEqual(result["cycle"], 120)
        self.assertEqual(result["raw_edges"]["root_i"], 100)
        assert_json_safe(self, result)

    def test_text_query_is_bounded_and_sanitised(self):
        state = self.backend.main_state()
        result = query({"text": "penguin river", "limit": 3},
                       backend=self.backend)
        self.assertEqual(result["mode"], "text")
        self.assertEqual(result["text"], "penguin river")
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(len(result["hits"]), 3)
        for hit in result["hits"]:
            self.assertIn("word", hit)
            self.assertGreaterEqual(hit["i"], 0)
            self.assertLess(hit["i"], NODE_COUNT)
            self.assertNotIn("pointer", hit)
            self.assertNotIn("offset", hit)
        self.assertEqual(self.backend.main_state(), state)
        assert_json_safe(self, result)
        assert_false_authority(self, result)

    def test_text_query_default_and_maximum_limits(self):
        default = query({"text": "river"}, backend=self.backend)
        self.assertEqual(default["top_k"], DEFAULT_QUERY_HITS)
        self.assertEqual(len(default["hits"]), DEFAULT_QUERY_HITS)
        maximum = query({"text": "river", "limit": MAX_QUERY_HITS},
                        backend=self.backend)
        self.assertEqual(maximum["top_k"], MAX_QUERY_HITS)
        self.assertEqual(len(maximum["hits"]), MAX_QUERY_HITS)
        assert_json_safe(self, maximum)

    def test_query_bounds_modes_and_unknown_properties_do_not_mutate(self):
        state = self.backend.main_state()
        cases = (
            {},
            {"i": 1},
            {"j": 2},
            {"i": -1, "j": 2},
            {"i": NODE_COUNT, "j": 2},
            {"i": 1, "j": 1},
            {"i": True, "j": 2},
            {"i": 1, "j": 2, "root": -1},
            {"i": 1, "j": 2, "root": 1},
            {"i": 1, "j": 2, "text": "river"},
            {"root": 0},
            {"text": ""},
            {"text": 3},
            {"text": "x" * (MAX_TEXT_BYTES + 1)},
            {"text": "river", "limit": 0},
            {"text": "river", "limit": MAX_QUERY_HITS + 1},
            {"text": "river", "limit": True},
            {"text": "river", "limit": "2"},
            {"text": "river", "root": 0},
            {"limit": 2},
            {"i": 1, "j": 2, "limit": 2},
            {"pointer": 1},
            {"text": "river", "address": 1},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(CycleToolError):
                    query(params, backend=self.backend)
                self.assertEqual(self.backend.main_state(), state)
                self.assertEqual(self.backend.mutations, [])


class CycleSurfaceTests(unittest.TestCase):
    def test_build_handlers_binds_every_action(self):
        backend = CycleToolsBackend()
        handlers = cycle_tools.build_handlers(backend)
        self.assertEqual(set(handlers),
                         {"status", "set", "cycle", "gauge", "train",
                          "snapshot", "query"})
        self.assertEqual(handlers["status"]({})["operation"], "status")
        written = handlers["set"]({
            "approved": True, "i": 0, "j": 1, "word": 9})
        self.assertTrue(written["accepted"])
        self.assertEqual(handlers["cycle"]({
            "i": 0, "j": 1, "root": 2})["value"], 9)
        self.assertTrue(handlers["gauge"]({
            "i": 0, "j": 1, "root": 2})["cancelled"])
        trained = handlers["train"]({
            "approved": True, "text": "surface"})
        self.assertTrue(trained["accepted"])
        self.assertEqual(handlers["snapshot"]({})["word_count"], 2)
        self.assertEqual(handlers["query"]({"i": 0, "j": 1})["word"], 9)
        assert_json_safe(self, written)
        assert_json_safe(self, trained)

    def test_module_exports_all_actions_and_imports_no_network_libraries(self):
        for name in ("status", "set", "set_edge", "cycle", "gauge", "train",
                     "snapshot", "query", "HANDLERS", "build_handlers"):
            self.assertIn(name, cycle_tools.__all__)
        source = inspect.getsource(cycle_tools)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0]
                                for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {
            "socket", "ssl", "urllib", "http", "requests", "httpx",
            "aiohttp", "ftplib", "telnetlib", "subprocess",
        })

    def test_output_bound_is_declared(self):
        self.assertGreaterEqual(MAX_OUTPUT_BYTES, 64 * 1024)
        self.assertEqual(DEFAULT_SNAPSHOT_WORDS, 64)


class ResidentCycleFieldIntegrationTests(unittest.TestCase):
    """Exercise the same handlers over real ResidentCells pair rails."""

    def setUp(self):
        try:
            self.backend = ResidentCycleFieldBackend()
        except (ImportError, ValueError) as exc:  # pragma: no cover
            self.skipTest("resident cells unavailable: %s" % exc)

    def test_real_resident_words_round_trip_and_cycle_exactly(self):
        for first, second, word in ((0, 1, 700), (1, 2, 300), (0, 2, 100)):
            result = set_edge({"approved": True, "i": first, "j": second,
                               "word": word}, backend=self.backend)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["stored_word"], word)
        self.assertEqual(self.backend.read_word(0, 1)["word"], 700)
        self.assertEqual(self.backend.read_word(1, 2)["word"], 300)
        result = cycle({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.assertEqual(result["value"], 900)
        self.assertEqual(result["raw_edges"],
                         {"root_i": 700, "i_j": 300, "root_j": 100})
        self.assertEqual(self.backend.generation, 3)
        assert_json_safe(self, result)

    def test_real_resident_gauge_shift_cancels_exactly(self):
        for first, second, word in ((0, 1, 700), (1, 2, 300), (0, 2, 100)):
            set_edge({"approved": True, "i": first, "j": second,
                      "word": word}, backend=self.backend)
        before = cycle({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.backend.apply_gauge(
            0, [0, 41, -17, 500],
            edges={(0, 1), (1, 2), (0, 2)})
        after = cycle({"i": 1, "j": 2, "root": 0},
                      backend=self.backend)
        self.assertEqual(after["value"], before["value"])
        result = gauge({"i": 1, "j": 2, "root": 0},
                       backend=self.backend)
        self.assertEqual(result["gauge_cycle"], 0)
        self.assertEqual(result["cancellation_residual"], 0)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["cycle"], before["value"])

    def test_real_resident_read_only_digest_stays_stable(self):
        set_edge({"approved": True, "i": 3, "j": 4, "word": -123},
                 backend=self.backend)
        state = self.backend.main_state()
        status({}, backend=self.backend)
        snapshot({}, backend=self.backend)
        cycle({"i": 3, "j": 4, "root": 0}, backend=self.backend)
        gauge({"i": 3, "j": 4, "root": 0}, backend=self.backend)
        query({"i": 3, "j": 4}, backend=self.backend)
        query({"text": "resident"}, backend=self.backend)
        self.assertEqual(self.backend.main_state(), state)
        self.assertEqual(self.backend.generation, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
