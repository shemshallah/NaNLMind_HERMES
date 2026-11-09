"""Tests for the bounded, dependency-injected resident field tool handlers."""
from __future__ import annotations

import ast
import inspect
import json
import math
import unittest

from nanlmind_hermes.resident import field_tools
from nanlmind_hermes.resident.field_tools import (
    DEFAULT_BUDGET_BYTES,
    HEAD_COUNT,
    MAX_BUDGET_BYTES,
    MAX_DELTA,
    MAX_STATUS_LIMIT,
    MAX_TEXT_BYTES,
    MAX_TOKEN_BYTES,
    MAX_TOP_K,
    FieldApprovalError,
    FieldToolError,
    activate,
    commit,
    propose,
    query,
    status,
)


_POINTER_KEYS = frozenset({
    "pointer", "ptr", "raw_pointer", "offset", "address", "buffer",
    "memoryview", "cell_offset", "raw_offset",
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


def assert_unchanged(testcase, backend, snapshot, mutations):
    testcase.assertEqual(backend.snapshot(), snapshot)
    testcase.assertEqual(backend.mutations, mutations)


class InMemoryFieldBackend:
    """Small backend used only by these tests.

    It deliberately returns extra ``pointer``/``offset`` fields from every
    mutation and query so the handlers' sanitisation can be verified.
    """

    def __init__(self, *, budget_bytes=DEFAULT_BUDGET_BYTES):
        self._generation = 0
        self._heads = [
            {"head": head, "score": 0.0, "state": 0,
             "generation": 0, "active": False}
            for head in range(HEAD_COUNT)
        ]
        self._proposals = {}
        self._token_seq = 0
        self.budget_bytes = budget_bytes
        self.used_bytes = 0
        self.calls = []
        self.mutations = []

    @property
    def generation(self):
        return self._generation

    @property
    def heads(self):
        return self._heads

    def _called(self, name):
        self.calls.append(name)

    def snapshot(self):
        self._called("snapshot")
        return {
            "opened": True,
            "generation": self._generation,
            "budget": {
                "bytes": self.budget_bytes,
                "max_bytes": MAX_BUDGET_BYTES,
                "used_bytes": self.used_bytes,
            },
            "heads": [dict(record) for record in self._heads],
            "active_heads": [
                record["head"] for record in self._heads if record["active"]
            ],
            "pointer": 0xDEADBEEF,
            "offset": 4096,
        }

    def _apply(self, operation, head, delta):
        self._generation += 1
        record = self._heads[head]
        record["score"] = round(
            max(-1.0, min(1.0, record["score"] + delta)), 8)
        record["state"] = (record["state"] + 1) & 0xFFFFFFFF
        record["generation"] = self._generation
        record["active"] = True
        self.mutations.append({
            "operation": operation,
            "head": head,
            "delta": delta,
            "generation": self._generation,
        })

    def activate(self, head, delta):
        self._called("activate")
        self._apply("activate", head, delta)
        record = self._heads[head]
        return {
            "generation": self._generation,
            "score": record["score"],
            "state": record["state"],
            "active": True,
            "pointer": 0xBEEF,
            "offset": 128,
        }

    def propose(self, head, delta, reason):
        self._called("propose")
        self._token_seq += 1
        token = "proposal-%04d" % self._token_seq
        self._proposals[token] = {
            "head": head,
            "delta": delta,
            "reason": reason,
            "generation": self._generation,
            "committed": False,
        }
        return {
            "token": token,
            "reasons": ["scratch_only",
                        "reason_recorded" if reason else "no_reason"],
            "generation": self._generation,
            "scratch": {"head": head, "delta": delta, "token": token},
            "pointer": 0xCAFE,
            "offset": 512,
        }

    def commit(self, token, expected_generation):
        self._called("commit")
        proposal = self._proposals.get(token)
        if proposal is None:
            return {"accepted": False, "reasons": ["unknown_token"],
                    "generation": self._generation, "pointer": 0x1}
        if proposal["committed"]:
            return {"accepted": False, "reasons": ["already_committed"],
                    "generation": self._generation, "pointer": 0x2}
        if proposal["generation"] != expected_generation or \
                self._generation != expected_generation:
            return {"accepted": False, "reasons": ["generation_mismatch"],
                    "generation": self._generation, "pointer": 0x3}
        proposal["committed"] = True
        self._apply("commit", proposal["head"], proposal["delta"])
        return {
            "accepted": True,
            "reasons": [],
            "generation": self._generation,
            "head": proposal["head"],
            "delta": proposal["delta"],
            "receipt": {"head": proposal["head"],
                        "delta": proposal["delta"]},
            "pointer": 0x4,
            "offset": 1024,
        }

    def head_metrics(self, head):
        self._called("head_metrics")
        record = self._heads[head]
        return {
            **record,
            "weight": record["score"],
            "phase": 0.25,
            "amplitude": abs(record["score"]),
            "incoming": 0,
            "outgoing": 1,
            "pair_count": 2,
            "leaf_count": 1,
            "relation_count": 0,
            "pointer": 0xDEAD,
            "offset": head * 32,
            "address": head,
        }

    def field_metrics(self):
        self._called("field_metrics")
        active = [record["head"] for record in self._heads
                  if record["active"]]
        return {
            "head_count": HEAD_COUNT,
            "active_heads": active,
            "active_count": len(active),
            "coverage": len(active) / float(HEAD_COUNT),
            "mean_weight": 0.5,
            "max_share": 1.0,
            "pair_cells": 32640,
            "occupied_pairs": len(active),
            "leaves": 2,
            "relations": 1,
            "features": 3,
            "generation": self._generation,
            "pointer": 0xDEAD,
            "offset": 8192,
        }

    def resonance(self, text, top_k):
        self._called("resonance")
        seed = sum(ord(character) for character in text)
        hits = []
        for index in range(top_k):
            head = (seed + index * 37) % HEAD_COUNT
            hits.append({
                "head": head,
                "score": round(1.0 / (index + 1), 6),
                "weight": 0.5,
                "text": text[:64],
                "kind": "observation",
                "leaf_id": "leaf-%d" % head,
                "feature_id": "feature-%d" % head,
                "offset": head * 4,
                "pointer": 0xFEED,
            })
        return {"hits": hits, "generation": self._generation,
                "pointer": 0xFEED}


class FieldStatusTests(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryFieldBackend()

    def test_status_is_bounded_json_safe_and_disables_canonical(self):
        result = status({}, backend=self.backend)
        self.assertEqual(result["generation"], 0)
        self.assertTrue(result["opened"])
        self.assertEqual(result["field"]["head_count"], HEAD_COUNT)
        self.assertEqual(result["field"]["active_count"], 0)
        self.assertEqual(result["field"]["active_heads"], [])
        self.assertEqual(result["active_heads"], [])
        self.assertEqual(len(result["heads"]), HEAD_COUNT)
        self.assertEqual(result["budget"]["bytes"],
                         DEFAULT_BUDGET_BYTES)
        self.assertEqual(result["budget"]["max_bytes"], MAX_BUDGET_BYTES)
        self.assertEqual(result["budget"]["hard_ceiling_bytes"],
                         MAX_BUDGET_BYTES)
        self.assertTrue(result["budget"]["bounded"])
        self.assertFalse(result["budget"]["exceeded"])
        self.assertFalse(result["budget_exceeded"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        self.assertEqual(result["canonical_merge"],
                         {"enabled": False, "attempted": False})
        self.assertFalse(result["canonical_merge_enabled"])
        self.assertFalse(result["canonical_promotion"])
        self.assertFalse(result["canonical_promotion_enabled"])
        self.assertFalse(result["promotion_enabled"])
        assert_json_safe(self, result)
        self.assertEqual(self.backend.mutations, [])

    def test_status_reports_active_heads_with_a_strict_limit(self):
        self.backend.activate(9, -0.25)
        self.backend.activate(3, 0.5)
        result = status({}, backend=self.backend)
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["field"]["active_count"], 2)
        self.assertEqual(result["active_heads"], [3, 9])
        self.assertEqual(result["field"]["active_heads"], [3, 9])
        self.assertAlmostEqual(result["field"]["coverage"], 2 / 256.0)
        limited = status({"limit": 1}, backend=self.backend)
        self.assertEqual(limited["active_heads"], [3])
        self.assertEqual(limited["field"]["active_count"], 2)
        self.assertTrue(limited["field"]["active_truncated"])
        self.assertEqual(limited["field"]["top"][0]["head"], 3)
        self.assertEqual(len(limited["field"]["top"]), 1)
        self.assertEqual(len(status({"limit": MAX_STATUS_LIMIT},
                                    backend=self.backend)["heads"]),
                         HEAD_COUNT)
        assert_json_safe(self, limited)

    def test_status_rejects_unknown_properties_and_bad_limits(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        for params in (
            {"limit": 0},
            {"limit": MAX_STATUS_LIMIT + 1},
            {"limit": True},
            {"limit": "2"},
            {"limit": 1.5},
            {"limit": None},
            {"pointer": 1},
            {"include_heads": True},
        ):
            with self.subTest(params=params):
                with self.assertRaises(FieldToolError):
                    status(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)


class FieldActivateTests(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryFieldBackend()

    def test_activate_requires_explicit_approval_without_mutating(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        for params in (
            {"head": 3, "delta": 0.5},
            {"approved": False, "head": 3, "delta": 0.5},
            {"approved": 1, "head": 3, "delta": 0.5},
            {"approved": "true", "head": 3, "delta": 0.5},
            {"approved": None, "head": 3, "delta": 0.5},
        ):
            with self.subTest(params=params):
                with self.assertRaises(FieldApprovalError):
                    activate(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)

    def test_activate_bounds_and_unknown_properties_do_not_mutate(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        cases = (
            {"approved": True, "delta": 0.5},                    # missing head
            {"approved": True, "head": -1, "delta": 0.5},
            {"approved": True, "head": HEAD_COUNT, "delta": 0.5},
            {"approved": True, "head": True, "delta": 0.5},
            {"approved": True, "head": 3.0, "delta": 0.5},
            {"approved": True, "head": [3], "delta": 0.5},
            {"approved": True, "head": 3, "delta": MAX_DELTA + 0.0001},
            {"approved": True, "head": 3, "delta": -MAX_DELTA - 0.0001},
            {"approved": True, "head": 3, "delta": float("nan")},
            {"approved": True, "head": 3, "delta": float("inf")},
            {"approved": True, "head": 3, "delta": True},
            {"approved": True, "head": 3, "delta": "0.5"},
            {"approved": True, "head": 3, "delta": 0.5, "pointer": 1},
            {"approved": True, "head": 3, "delta": 0.5, "reason": "x"},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(FieldToolError):
                    activate(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)

    def test_activate_returns_a_sanitized_receipt_for_head_255(self):
        result = activate({"approved": True, "head": HEAD_COUNT - 1,
                           "delta": -0.5}, backend=self.backend)
        self.assertEqual(result["operation"], "activate")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["head"], HEAD_COUNT - 1)
        self.assertEqual(result["delta"], -0.5)
        self.assertEqual(result["previous_generation"], 0)
        self.assertEqual(result["generation"], 1)
        self.assertTrue(result["writes_main"])
        self.assertEqual(result["receipt"]["head"], HEAD_COUNT - 1)
        self.assertEqual(result["receipt"]["delta"], -0.5)
        self.assertEqual(result["receipt"]["score"], -0.5)
        self.assertEqual(result["receipt"]["generation"], 1)
        self.assertEqual(self.backend.heads[HEAD_COUNT - 1]["score"], -0.5)
        self.assertTrue(self.backend.heads[HEAD_COUNT - 1]["active"])
        self.assertEqual(len(self.backend.mutations), 1)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_merge"]["attempted"])
        self.assertFalse(result["canonical_promotion"])
        assert_json_safe(self, result)

    def test_activate_defaults_to_one_and_accepts_exact_delta_bounds(self):
        low = activate({"approved": True, "head": 0,
                        "delta": -MAX_DELTA}, backend=self.backend)
        high = activate({"approved": True, "head": 1,
                         "delta": MAX_DELTA}, backend=self.backend)
        default = activate({"approved": True, "head": 2},
                           backend=self.backend)
        self.assertEqual(low["delta"], -1.0)
        self.assertEqual(high["delta"], 1.0)
        self.assertEqual(default["delta"], 1.0)
        self.assertEqual(self.backend.generation, 3)
        self.assertEqual(
            [mutation["head"] for mutation in self.backend.mutations],
            [0, 1, 2])
        assert_json_safe(self, high)

    def test_activation_backend_denial_is_a_receipt_without_mutation(self):
        class DenyingBackend(InMemoryFieldBackend):
            def activate(self, head, delta):
                self._called("activate")
                return {"accepted": False,
                        "reasons": ["budget_exhausted"]}

        backend = DenyingBackend()
        before = backend.snapshot()
        mutations = list(backend.mutations)
        result = activate({"approved": True, "head": 3, "delta": 0.5},
                          backend=backend)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["writes_main"])
        self.assertEqual(result["reasons"], ["budget_exhausted"])
        self.assertEqual(result["generation"], 0)
        self.assertEqual(result["previous_generation"], 0)
        self.assertFalse(result["receipt"]["accepted"])
        self.assertFalse(result["receipt"]["applied"])
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_unchanged(self, backend, before, mutations)
        assert_json_safe(self, result)


class FieldProposeCommitTests(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryFieldBackend()

    def test_propose_stages_scratch_only_before_main_commit(self):
        before = self.backend.snapshot()
        result = propose({"head": 11, "delta": 0.75,
                          "reason": "approved plan"}, backend=self.backend)
        self.assertEqual(result["operation"], "propose")
        self.assertTrue(result["staged"])
        self.assertTrue(result["accepted"])
        self.assertTrue(result["scratch_only"])
        self.assertFalse(result["writes_main"])
        self.assertTrue(result["main_unchanged"])
        self.assertEqual(result["head"], 11)
        self.assertEqual(result["delta"], 0.75)
        self.assertEqual(result["generation"], 0)
        self.assertEqual(result["main_generation"], 0)
        self.assertIn("proposal-0001", result["token"])
        self.assertTrue(result["reasons"])
        self.assertEqual(result["receipt"]["token"], result["token"])
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.mutations, [])
        self.assertEqual(self.backend.snapshot(), before)
        self.assertEqual(self.backend.heads[11]["score"], 0.0)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_json_safe(self, result)

    def test_propose_bounds_unknown_properties_and_reasons(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        cases = (
            {"delta": 0.5},                                  # missing head
            {"head": -1, "delta": 0.5},
            {"head": HEAD_COUNT, "delta": 0.5},
            {"head": 3, "delta": MAX_DELTA + 0.1},
            {"head": 3, "delta": -MAX_DELTA - 0.1},
            {"head": 3, "delta": 0.5, "reason": "x" * 241},
            {"head": 3, "delta": 0.5, "reason": 7},
            {"head": 3, "delta": 0.5, "pointer": 1},
            {"head": 3, "delta": 0.5, "token": "smuggled"},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(FieldToolError):
                    propose(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)

    def test_commit_requires_approval_token_and_expected_generation(self):
        proposal = propose({"head": 7, "delta": 0.5},
                           backend=self.backend)
        token = proposal["token"]
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        approval_cases = (
            {"token": token, "expected_generation": 0},
            {"approved": False, "token": token, "expected_generation": 0},
            {"approved": 1, "token": token, "expected_generation": 0},
        )
        for params in approval_cases:
            with self.subTest(params=params):
                with self.assertRaises(FieldApprovalError):
                    commit(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)
        validation_cases = (
            {"approved": True, "expected_generation": 0},
            {"approved": True, "token": "", "expected_generation": 0},
            {"approved": True, "token": 123, "expected_generation": 0},
            {"approved": True, "token": "bad token", "expected_generation": 0},
            {"approved": True, "token": "x" * (MAX_TOKEN_BYTES + 1),
             "expected_generation": 0},
            {"approved": True, "token": token},
            {"approved": True, "token": token, "expected_generation": -1},
            {"approved": True, "token": token, "expected_generation": True},
            {"approved": True, "token": token, "expected_generation": 0,
             "pointer": 1},
            {"approved": True, "token": token, "expected_generation": 0,
             "head": 7},
        )
        for params in validation_cases:
            with self.subTest(params=params):
                with self.assertRaises(FieldToolError):
                    commit(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)

    def test_commit_wrong_generation_is_denied_without_mutation(self):
        proposal = propose({"head": 7, "delta": 0.5},
                           backend=self.backend)
        before = self.backend.snapshot()
        result = commit({"approved": True, "token": proposal["token"],
                         "expected_generation": 99}, backend=self.backend)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["denied"])
        self.assertEqual(result["reasons"], ["generation_mismatch"])
        self.assertEqual(result["generation"], 0)
        self.assertEqual(result["previous_generation"], 0)
        self.assertFalse(result["writes_main"])
        self.assertTrue(result["read_only"])
        self.assertEqual(result["receipt"]["accepted"], False)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_unchanged(self, self.backend, before,
                         list(self.backend.mutations))
        assert_json_safe(self, result)

    def test_commit_backend_denial_is_returned_without_mutation(self):
        before = self.backend.snapshot()
        result = commit({"approved": True, "token": "unknown-1",
                         "expected_generation": 0}, backend=self.backend)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["denied"])
        self.assertEqual(result["reasons"], ["unknown_token"])
        self.assertEqual(self.backend.generation, 0)
        assert_unchanged(self, self.backend, before,
                         list(self.backend.mutations))
        assert_json_safe(self, result)

    def test_commit_accepts_once_and_consumes_the_token(self):
        proposal = propose({"head": 13, "delta": 0.5,
                            "reason": "checked"}, backend=self.backend)
        result = commit({"approved": True, "token": proposal["token"],
                         "expected_generation": 0}, backend=self.backend)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["denied"])
        self.assertEqual(result["previous_generation"], 0)
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["head"], 13)
        self.assertEqual(result["delta"], 0.5)
        self.assertTrue(result["writes_main"])
        self.assertEqual(result["receipt"]["head"], 13)
        self.assertEqual(self.backend.heads[13]["score"], 0.5)
        self.assertEqual(len(self.backend.mutations), 1)
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        assert_json_safe(self, result)

        replay = commit({"approved": True, "token": proposal["token"],
                         "expected_generation": 1}, backend=self.backend)
        self.assertFalse(replay["accepted"])
        self.assertEqual(replay["reasons"], ["already_committed"])
        self.assertEqual(self.backend.generation, 1)
        self.assertEqual(len(self.backend.mutations), 1)

    def test_stale_proposal_is_denied_after_main_advances(self):
        proposal = propose({"head": 21, "delta": 0.5},
                           backend=self.backend)
        self.backend.activate(2, 0.25)
        self.assertEqual(self.backend.generation, 1)
        result = commit({"approved": True, "token": proposal["token"],
                         "expected_generation": 1}, backend=self.backend)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["generation_mismatch"])
        self.assertEqual(self.backend.heads[21]["score"], 0.0)
        self.assertEqual(self.backend.generation, 1)
        self.assertEqual(len(self.backend.mutations), 1)
        self.assertEqual(self.backend.mutations[0]["operation"], "activate")
        self.assertEqual(self.backend.mutations[0]["head"], 2)
        assert_json_safe(self, result)


class FieldQueryTests(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryFieldBackend()
        self.backend.activate(7, 0.5)
        self.backend.activate(42, -0.25)
        self.backend.calls.clear()

    def test_query_head_metrics_is_read_only_and_sanitized(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        result = query({"head": 7}, backend=self.backend)
        self.assertEqual(result["mode"], "head")
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["head"], 7)
        self.assertEqual(result["metrics"]["head"], 7)
        self.assertEqual(result["metrics"]["score"], 0.5)
        self.assertEqual(result["metrics"]["phase"], 0.25)
        self.assertTrue(result["metrics"]["active"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["writes_main"])
        self.assertFalse(result["canonical_merge"]["enabled"])
        self.assertFalse(result["canonical_promotion"])
        self.assertEqual(self.backend.generation, 2)
        assert_unchanged(self, self.backend, before, mutations)
        self.assertEqual(set(self.backend.calls),
                         {"snapshot", "head_metrics"})
        assert_json_safe(self, result)

    def test_query_field_metrics_is_independent_and_read_only(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        result = query({}, backend=self.backend)
        self.assertEqual(result["mode"], "field")
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["field"]["head_count"], HEAD_COUNT)
        self.assertEqual(result["field"]["active_count"], 2)
        self.assertEqual(result["field"]["active_heads"], [7, 42])
        self.assertEqual(result["field"]["generation"], 2)
        self.assertEqual(result["field"]["pair_cells"], 32640)
        self.assertTrue(result["read_only"])
        assert_unchanged(self, self.backend, before, mutations)
        self.assertEqual(set(self.backend.calls),
                         {"snapshot", "field_metrics"})
        assert_json_safe(self, result)

    def test_query_resonance_is_bounded_and_sanitized(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        result = query({"text": "penguin river", "top_k": 3},
                       backend=self.backend)
        self.assertEqual(result["mode"], "resonance")
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["text"], "penguin river")
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(len(result["hits"]), 3)
        for hit in result["hits"]:
            self.assertGreaterEqual(hit["head"], 0)
            self.assertLess(hit["head"], HEAD_COUNT)
            self.assertIn("score", hit)
            self.assertNotIn("pointer", hit)
            self.assertNotIn("offset", hit)
        self.assertTrue(result["read_only"])
        assert_unchanged(self, self.backend, before, mutations)
        self.assertEqual(set(self.backend.calls),
                         {"snapshot", "resonance"})
        assert_json_safe(self, result)

    def test_query_defaults_and_maximum_top_k_are_bounded(self):
        default = query({"text": "river"}, backend=self.backend)
        self.assertEqual(default["top_k"], field_tools.DEFAULT_TOP_K)
        self.assertEqual(len(default["hits"]), field_tools.DEFAULT_TOP_K)
        maximum = query({"text": "river", "top_k": MAX_TOP_K},
                        backend=self.backend)
        self.assertEqual(maximum["top_k"], MAX_TOP_K)
        self.assertEqual(len(maximum["hits"]), MAX_TOP_K)
        assert_json_safe(self, maximum)

    def test_query_rejects_bounds_mixed_modes_and_unknown_properties(self):
        before = self.backend.snapshot()
        mutations = list(self.backend.mutations)
        cases = (
            {"head": -1},
            {"head": HEAD_COUNT},
            {"head": True},
            {"head": 7.0},
            {"text": ""},
            {"text": 123},
            {"text": "x" * (MAX_TEXT_BYTES + 1)},
            {"head": 7, "text": "river"},
            {"top_k": 2},
            {"head": 7, "top_k": 2},
            {"text": "river", "top_k": 0},
            {"text": "river", "top_k": MAX_TOP_K + 1},
            {"text": "river", "top_k": True},
            {"text": "river", "top_k": "2"},
            {"pointer": 1},
            {"limit": 2},
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(FieldToolError):
                    query(params, backend=self.backend)
                assert_unchanged(self, self.backend, before, mutations)


class FieldSurfaceTests(unittest.TestCase):
    def test_build_handlers_binds_every_handler_to_the_backend(self):
        backend = InMemoryFieldBackend()
        handlers = field_tools.build_handlers(backend)
        self.assertEqual(set(handlers),
                         {"status", "activate", "propose", "commit", "query"})
        self.assertEqual(handlers["status"]({})["generation"], 0)
        receipt = handlers["activate"]({
            "approved": True, "head": 4, "delta": 0.25})
        self.assertTrue(receipt["accepted"])
        proposal = handlers["propose"]({"head": 4, "delta": 0.25})
        committed = handlers["commit"]({
            "approved": True,
            "token": proposal["token"],
            "expected_generation": 1,
        })
        self.assertTrue(committed["accepted"])
        self.assertEqual(handlers["query"]({"head": 4})["head"], 4)
        self.assertEqual(handlers["query"]({})["mode"], "field")
        assert_json_safe(self, receipt)
        assert_json_safe(self, committed)

    def test_module_imports_no_network_libraries(self):
        source = inspect.getsource(field_tools)
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
            "aiohttp", "ftplib", "telnetlib",
        })


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
