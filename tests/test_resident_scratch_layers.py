"""Tests for per-head L0-L9 scratch layers and staged CAS commits."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from nanlmind_hermes.resident.scratch_layers import (
    HEAD_COUNT,
    LAYER_COUNT,
    LAYER_NAMES,
    MAX_STAGED_TOKENS,
    MAX_VALUE_BYTES,
    PAGES_PER_LAYER,
    RECORDS_PER_PAGE,
    SLOTS_PER_LAYER,
    MemoryMainBackend,
    ScratchLayerBoundsError,
    ScratchLayerError,
    ScratchLayers,
    ScratchRecord,
    StagedToken,
)


class TrackingBackend:
    """Memory backend that records reads and compare-and-swap attempts."""

    def __init__(self):
        self.inner = MemoryMainBackend()
        self.reads = []
        self.cas_calls = []

    def read_main(self, head, layer):
        self.reads.append((head, layer))
        return self.inner.read_main(head, layer)

    def compare_and_swap_main(self, head, layer, *, expected_generation,
                              value=None, value_bytes=None, digest=None,
                              token_id=None):
        self.cas_calls.append({
            "head": head, "layer": layer,
            "expected_generation": expected_generation,
            "value": value, "value_bytes": value_bytes,
            "digest": digest, "token_id": token_id,
        })
        return self.inner.compare_and_swap_main(
            head, layer, expected_generation=expected_generation,
            value=value, value_bytes=value_bytes, digest=digest,
            token_id=token_id)

    def stats(self):
        return self.inner.stats()


class ScratchLayersTests(unittest.TestCase):
    def setUp(self):
        self.field = ScratchLayers()

    # ------------------------------------------------------------- geometry
    def test_geometry_is_fixed_and_bounded(self):
        self.assertEqual(HEAD_COUNT, 256)
        self.assertEqual(LAYER_COUNT, 10)
        self.assertEqual(LAYER_NAMES, tuple("L%d" % i for i in range(10)))
        self.assertEqual(SLOTS_PER_LAYER,
                         PAGES_PER_LAYER * RECORDS_PER_PAGE)
        stats = self.field.stats()
        self.assertEqual(stats["heads"], 256)
        self.assertEqual(stats["layers"], 10)
        self.assertEqual(stats["slots_per_layer"], SLOTS_PER_LAYER)
        self.assertEqual(stats["capacity_records"],
                         256 * 10 * SLOTS_PER_LAYER)
        self.assertEqual(stats["scratch_records"], 0)
        self.assertEqual(stats["allocated_layers"], 0)

    def test_bounds_are_rejected(self):
        for head in (-1, 256, True, "3", 1.5):
            with self.assertRaises(ScratchLayerBoundsError):
                self.field.write_layer(head, 0, "x")
        for layer in (-1, 10, True, "L10", "nope"):
            with self.assertRaises(ScratchLayerBoundsError):
                self.field.write_layer(0, layer, "x")
        with self.assertRaises(ScratchLayerBoundsError):
            self.field.write_layer(0, 0, "x" * (MAX_VALUE_BYTES + 1))
        with self.assertRaises(ScratchLayerError):
            self.field.write_layer(0, 0, float("nan"))
        with self.assertRaises(ScratchLayerBoundsError):
            self.field.field_snapshot([0, 256])
        with self.assertRaises(ScratchLayerBoundsError):
            self.field.stage_commit(256, 0, "x")
        with self.assertRaises(ScratchLayerBoundsError):
            self.field.stage_commit(0, 10, "x")

    # --------------------------------------------------------- independence
    def test_layers_are_independent_per_head_and_layer(self):
        self.field.write_layer(3, 0, "head3-L0")
        self.field.write_layer(3, 1, "head3-L1")
        self.field.write_layer(4, 0, "head4-L0")
        self.field.write_layer(4, 1, "head4-L1")

        self.assertEqual(self.field.read_value(3, 0), "head3-L0")
        self.assertEqual(self.field.read_value(3, 1), "head3-L1")
        self.assertEqual(self.field.read_value(4, 0), "head4-L0")
        self.assertEqual(self.field.read_value(4, 1), "head4-L1")

        # A write to one head/layer must not change any sibling state.
        before = self.field.read_layer(4, 0)
        self.field.write_layer(3, 0, "changed")
        after = self.field.read_layer(4, 0)
        self.assertEqual(before, after)
        self.assertEqual(after["revision"], 1)
        self.assertEqual(after["generation"], 3)

    def test_values_are_copied_and_views_are_immutable_snapshots(self):
        original = {"nested": [1, 2]}
        self.field.write_layer(9, 5, original)
        original["nested"].append(3)
        self.assertEqual(self.field.read_value(9, 5), {"nested": [1, 2]})

        view = self.field.read_layer(9, 5)
        view["value"]["nested"].append(99)
        self.assertEqual(self.field.read_value(9, 5), {"nested": [1, 2]})
        self.assertEqual(view["scope"], "scratch")

        record = self.field.main_record(0, 0)
        self.assertIsNone(record)

    # ------------------------------------------------------------- field read
    def test_field_snapshot_is_deterministic_and_bounded(self):
        self.field.write_layer(7, 2, {"answer": 42})
        self.field.write_layer(5, 9, "five-nine")
        self.field.write_layer(6, 0, [1, 2, 3])

        first = self.field.field_snapshot([7, 5, 6, 5])
        second = self.field.field_snapshot([5, 6, 7])
        self.assertEqual(first, second)
        self.assertEqual(first["heads"], (5, 6, 7))
        self.assertEqual(first["layers"], LAYER_NAMES)
        self.assertEqual(first["entries"][7][2]["value"], {"answer": 42})
        self.assertEqual(first["entries"][5][9]["value"], "five-nine")
        self.assertEqual(first["values"][6][0], [1, 2, 3])
        self.assertIsNone(first["values"][5][0])

        everything = self.field.field_snapshot()
        self.assertEqual(len(everything["values"]), 256)
        self.assertTrue(all(len(layers) == 10
                            for layers in everything["values"].values()))
        self.assertEqual(everything["capacity"],
                         256 * 10 * SLOTS_PER_LAYER)

    # ---------------------------------------------------------- scratch only
    def test_scratch_writes_never_touch_main(self):
        backend = TrackingBackend()
        field = ScratchLayers(backend=backend)
        for index in range(20):
            receipt = field.write_layer(index % 8, index % 10, index)
            self.assertEqual(receipt["scope"], "scratch")
            self.assertFalse(receipt["main_touched"])
            self.assertTrue(receipt["stored"])

        self.assertEqual(backend.reads, [])
        self.assertEqual(backend.cas_calls, [])
        self.assertEqual(field.main_generation(0, 0), 0)
        self.assertIsNone(field.main_record(0, 0))
        self.assertEqual(field.read_main(0, 0)["status"], "empty")
        self.assertEqual(field.stats()["main_commits"], 0)

        # Staging reads the base generation but still must not mutate main.
        token = field.stage_commit(0, 0, "proposal")
        self.assertEqual(token.expected_generation, 0)
        self.assertEqual(backend.cas_calls, [])
        self.assertIsNone(field.main_record(0, 0))

    # ---------------------------------------------------------- commit recipe
    def test_commit_recipe_requires_approval_and_uses_cas(self):
        field = self.field
        field.write_layer(1, 4, "scratch-before")
        token = field.stage_commit(1, 4, "committed-value")
        self.assertIsInstance(token, StagedToken)
        self.assertEqual(token.expected_generation, 0)
        self.assertEqual(field.main_generation(1, 4), 0)

        denied = field.commit_staged(
            token, approved=False, expected_generation=token.expected_generation)
        self.assertFalse(denied["committed"])
        self.assertFalse(denied["accepted"])
        self.assertEqual(denied["status"], "approval_required")
        self.assertEqual(denied["reasons"], ["approval required"])
        self.assertEqual(field.main_generation(1, 4), 0)
        self.assertIsNone(field.main_record(1, 4))

        # The same token remains valid after an approval-only denial.
        committed = field.commit_staged(
            token, approved=True, expected_generation=token.expected_generation)
        self.assertTrue(committed["committed"])
        self.assertTrue(committed["accepted"])
        self.assertTrue(committed["main_updated"])
        self.assertEqual(committed["status"], "committed")
        self.assertEqual(committed["generation"], 1)
        self.assertEqual(committed["value"], "committed-value")

        main = field.read_main(1, 4)
        self.assertEqual(main["status"], "committed")
        self.assertEqual(main["value"], "committed-value")
        self.assertEqual(main["generation"], 1)
        # A commit changes main only; the scratch layer is untouched.
        self.assertEqual(field.read_value(1, 4), "scratch-before")

        # Tokens are single-use.
        replay = field.commit_staged(
            token, approved=True, expected_generation=1)
        self.assertFalse(replay["committed"])
        self.assertEqual(replay["status"], "token_consumed")
        self.assertEqual(field.main_generation(1, 4), 1)

        # A second recipe commits the next generation.
        second = field.stage_commit(1, 4, "second-value")
        self.assertEqual(second.expected_generation, 1)
        result = field.commit_staged(
            second, approved=True, expected_generation=second.expected_generation)
        self.assertTrue(result["committed"])
        self.assertEqual(result["generation"], 2)
        self.assertEqual(field.read_main(1, 4)["value"], "second-value")
        self.assertEqual(field.main_generation(1, 4), 2)

    def test_approval_must_be_exactly_true_and_stale_generation_is_denied(self):
        token_a = self.field.stage_commit(2, 3, "a")
        token_b = self.field.stage_commit(2, 3, "b")

        self.assertEqual(
            self.field.commit_staged(
                token_a, approved=1,
                expected_generation=token_a.expected_generation)["status"],
            "approval_required")
        self.assertEqual(self.field.main_generation(2, 3), 0)

        first = self.field.commit_staged(
            token_a, approved=True,
            expected_generation=token_a.expected_generation)
        self.assertTrue(first["committed"])
        self.assertEqual(first["generation"], 1)

        before = self.field.read_main(2, 3)
        stale = self.field.commit_staged(
            token_b, approved=True,
            expected_generation=token_b.expected_generation)
        after = self.field.read_main(2, 3)
        self.assertFalse(stale["committed"])
        self.assertFalse(stale["applied"])
        self.assertEqual(stale["status"], "stale_generation")
        self.assertEqual(stale["expected_generation"], 0)
        self.assertEqual(stale["generation"], 1)
        self.assertEqual(before, after)
        self.assertEqual(self.field.main_generation(2, 3), 1)

        # Even a caller-supplied later generation cannot reuse the old base.
        mismatch = self.field.commit_staged(
            token_b, approved=True, expected_generation=1)
        self.assertEqual(mismatch["status"], "stale_generation")
        self.assertEqual(self.field.read_main(2, 3), after)

    def test_backend_cas_denial_after_stage_leaves_main_unchanged(self):
        token = self.field.stage_commit(8, 8, "staged")
        self.assertEqual(token.expected_generation, 0)
        # An external writer advances the main slot after staging.
        external = self.field.backend.compare_and_swap_main(
            8, 8, expected_generation=0, value="external")
        self.assertTrue(external["applied"])
        before = self.field.read_main(8, 8)
        self.assertEqual(before["generation"], 1)

        denied = self.field.commit_staged(
            token, approved=True, expected_generation=token.expected_generation)
        self.assertEqual(denied["status"], "stale_generation")
        self.assertFalse(denied["main_updated"])
        self.assertEqual(self.field.read_main(8, 8), before)
        self.assertEqual(self.field.read_main(8, 8)["value"], "external")

    # ---------------------------------------------------------------- bounds
    def test_slots_are_bounded_and_recycled(self):
        field = ScratchLayers(pages_per_layer=1, records_per_page=2)
        self.assertEqual(field.capacity_per_layer, 2)
        for value in range(5):
            receipt = field.write_layer(12, 9, value)
            self.assertEqual(receipt["evicted"], value >= 2)

        view = field.read_layer(12, 9)
        self.assertEqual(view["value"], 4)
        self.assertEqual(view["slots_used"], 2)
        self.assertEqual(view["capacity"], 2)
        self.assertEqual([record["value"] for record in view["records"]],
                         [3, 4])

        stats = field.stats()
        self.assertEqual(stats["scratch_records"], 2)
        self.assertEqual(stats["scratch_writes"], 5)
        self.assertEqual(stats["scratch_evictions"], 3)
        self.assertEqual(stats["max_revision"], 5)
        self.assertEqual(stats["scratch_generation"], 5)
        self.assertEqual(stats["allocated_layers"], 1)
        self.assertEqual(stats["pages_allocated"], 1)
        self.assertLessEqual(stats["scratch_records"],
                             stats["capacity_records"])

        # Other heads/layers remain empty and unallocated.
        self.assertEqual(field.read_layer(12, 8)["status"], "empty")
        self.assertEqual(field.stats()["allocated_layers"], 1)

    def test_staging_and_record_storage_are_bounded(self):
        field = ScratchLayers(max_staged_tokens=3)
        tokens = [field.stage_commit(0, 0, index) for index in range(5)]
        stats = field.stats()
        self.assertEqual(stats["staged_tokens"], 3)
        self.assertLessEqual(stats["staged_tokens"],
                             stats["max_staged_tokens"])
        self.assertGreater(MAX_STAGED_TOKENS, 3)

        expired = field.commit_staged(
            tokens[0], approved=True,
            expected_generation=tokens[0].expected_generation)
        self.assertEqual(expired["status"], "unknown_token")
        self.assertEqual(field.main_generation(0, 0), 0)

        newest = field.commit_staged(
            tokens[-1], approved=True,
            expected_generation=tokens[-1].expected_generation)
        self.assertTrue(newest["committed"])
        self.assertEqual(field.stats()["consumed_tokens"], 1)

    def test_generations_are_monotonic_and_records_are_frozen(self):
        receipts = [self.field.write_layer(0, 0, index) for index in range(4)]
        self.assertEqual([item["generation"] for item in receipts],
                         [1, 2, 3, 4])
        self.assertEqual([item["revision"] for item in receipts],
                         [1, 2, 3, 4])

        state = self.field._state(0, 0, create=False)
        self.assertIsNotNone(state)
        record = state.slots[0]
        self.assertIsInstance(record, ScratchRecord)
        with self.assertRaises(FrozenInstanceError):
            record.revision = 99

        main = self.field.main_record(0, 0)
        self.assertIsNone(main)
        token = self.field.stage_commit(0, 0, "main")
        self.field.commit_staged(
            token, approved=True, expected_generation=token.expected_generation)
        main = self.field.main_record(0, 0)
        self.assertIsNotNone(main)
        with self.assertRaises(FrozenInstanceError):
            main.generation = 99

        # Failed commits never lower the main generation.
        stale = self.field.commit_staged(
            token, approved=True, expected_generation=0)
        self.assertFalse(stale["committed"])
        self.assertEqual(self.field.main_generation(0, 0), 1)

    # ------------------------------------------------------------- injection
    def test_injected_backend_is_used_for_stage_and_cas(self):
        backend = TrackingBackend()
        field = ScratchLayers(backend=backend)
        field.write_layer(6, 6, "scratch")
        token = field.stage_commit(6, 6, {"main": True})
        self.assertEqual(backend.reads, [(6, 6)])
        self.assertEqual(backend.cas_calls, [])

        result = field.commit_staged(
            token, approved=True, expected_generation=token.expected_generation)
        self.assertTrue(result["committed"])
        self.assertEqual(len(backend.cas_calls), 1)
        call = backend.cas_calls[0]
        self.assertEqual(call["head"], 6)
        self.assertEqual(call["layer"], 6)
        self.assertEqual(call["expected_generation"], 0)
        self.assertEqual(call["token_id"], token.token_id)
        self.assertEqual(call["value"], {"main": True})
        self.assertEqual(call["value_bytes"], token.payload)
        self.assertEqual(field.read_main(6, 6)["value"], {"main": True})

    def test_callback_backend_can_be_used_before_resident_integration(self):
        state = {"generation": 4, "value": "seed"}
        calls = []

        def read_main(head, layer):
            self.assertEqual((head, layer), (10, 2))
            return {"generation": state["generation"],
                    "value": state["value"]}

        def commit_main(head, layer, *, expected_generation, value,
                        value_bytes, digest, token_id):
            calls.append({
                "head": head, "layer": layer,
                "expected_generation": expected_generation,
                "value": value, "value_bytes": value_bytes,
                "digest": digest, "token_id": token_id,
            })
            if expected_generation != state["generation"]:
                return {"applied": False, "status": "stale_generation",
                        "generation": state["generation"]}
            state["generation"] += 1
            state["value"] = value
            return {"applied": True, "status": "committed",
                    "generation": state["generation"]}

        field = ScratchLayers(
            read_callback=read_main, commit_callback=commit_main)
        token = field.stage_commit(10, 2, "next")
        self.assertEqual(token.expected_generation, 4)
        result = field.commit_staged(
            token, approved=True, expected_generation=4)
        self.assertTrue(result["committed"])
        self.assertEqual(result["generation"], 5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["value"], "next")
        self.assertEqual(calls[0]["token_id"], token.token_id)
        self.assertEqual(state["value"], "next")
        self.assertEqual(field.main_generation(10, 2), 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
