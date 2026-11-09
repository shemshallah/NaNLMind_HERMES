"""Tests for the bounded per-head MTH Boolean bridge.

The runtime is faked for the full 256-head matrix so the tests remain fast and
deterministic.  One optional integration test exercises the real bundled
NaN0S/N16I toolchain when it is available.
"""
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

from nanlmind_hermes.resident import cells
from nanlmind_hermes.resident import mth_head_bridge as module
from nanlmind_hermes.resident.mth_head_bridge import (
    FIELD_SCHEMA,
    HEAD_COUNT,
    MAX_AGGREGATE_BYTES,
    MAX_FIELD_SECONDS,
    MAX_PLAN_BYTES,
    MAX_STEPS,
    MAX_TIMEOUT,
    MTHHeadBridge,
    MTHHeadBridgeError,
    OP_NAMES,
)


def truth(mask, a, b):
    return (mask >> ((a << 1) | b)) & 1


def head_case(head):
    """A varied but fully deterministic operation per head."""
    return OP_NAMES[head % len(OP_NAMES)], (head >> 1) & 1, (head >> 3) & 1


def field_plan():
    return {head: head_case(head) for head in range(HEAD_COUNT)}


class FakeRuntime:
    name = "fake_runtime"

    def __init__(self, *, available=True, mode="interpreted",
                 fail_head=None, raise_head=None, override=None):
        self._available = available
        self.mode = mode
        self.fail_head = fail_head
        self.raise_head = raise_head
        self.override = override or {}
        self.calls = []

    def available(self):
        return {
            "available": self._available,
            "name": self.name,
            "reason": None if self._available else "fake_unavailable",
            "detail": None if self._available else "fake runtime is disabled",
        }

    def dispatch_boolean(self, mask, a, b, *, head, steps, timeout):
        self.calls.append({
            "mask": mask, "a": a, "b": b, "head": head,
            "steps": steps, "timeout": timeout,
        })
        if not self._available:
            return {"available": False, "reason": "fake_unavailable",
                    "detail": "fake runtime is disabled"}
        if self.raise_head == head:
            raise RuntimeError("fake runtime exploded")
        if self.fail_head == head:
            return {"available": True, "execution_mode": "unavailable",
                    "trap": "fake trap"}
        output = self.override.get(head, truth(mask, a, b))
        return {
            "available": True,
            "output": output,
            "execution_mode": self.mode,
            "steps": 3,
            "generation": 999,
            "detail": "fake runtime decision",
        }


def _real_toolchain_ready():
    try:
        from nanlmind_hermes import nan0s_runtime
        if not nan0s_runtime.compiler_available():
            return False
        if not (shutil.which("gawk") or shutil.which("mawk")):
            return False
        core = nan0s_runtime.discover_core()
        root = Path(str(core.get("root") or ""))
        return (root / "src" / "tools" / "n16i.nas").is_file()
    except Exception:
        return False


class HeadSelectionTests(unittest.TestCase):
    def test_selects_all_256_permanent_head_identities(self):
        bridge = MTHHeadBridge(FakeRuntime())
        identities = []
        for head in range(HEAD_COUNT):
            result = bridge.select_head(head)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["selected"])
            self.assertEqual(result["head"], head)
            self.assertEqual(result["head_id"], head)
            self.assertEqual(result["identity"], cells.identity_address(head))
            self.assertTrue(result["identity_valid"])
            self.assertEqual(result["generation"], 0)
            self.assertIs(result["canonical_merge"], False)
            identities.append(result["identity"])
        self.assertEqual(len(set(identities)), HEAD_COUNT)
        self.assertEqual(identities[0], 0x0000)
        self.assertEqual(identities[255], 0xFFFF)

    def test_rejects_invalid_heads_without_dispatch(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime)
        for head in (-1, HEAD_COUNT, HEAD_COUNT + 1, True, 1.5, "0", None, []):
            result = bridge.select_head(head)
            self.assertTrue(result["denied"], head)
            self.assertFalse(result["selected"])
            self.assertIsNone(result["identity"])
            self.assertEqual(result["reason"], "invalid_head")
            self.assertIs(result["canonical_merge"], False)
        self.assertEqual(runtime.calls, [])

    def test_supported_ops_cover_all_sixteen_truth_tables(self):
        operations = MTHHeadBridge.supported_ops()
        self.assertEqual(len(operations), 16)
        self.assertEqual({item["mask"] for item in operations}, set(range(16)))
        self.assertEqual({item["op"] for item in operations}, set(OP_NAMES))
        for item in operations:
            self.assertEqual(len(item["truth_table"]), 4)
            self.assertTrue(set(item["truth_table"]) <= {0, 1})


class SingleHeadOperationTests(unittest.TestCase):
    def test_boolean_op_covers_all_256_heads(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime)
        for head in range(HEAD_COUNT):
            op, a, b = head_case(head)
            receipt = bridge.boolean_op(head, a, b, op)
            expected = truth(OP_NAMES.index(op), a, b)
            self.assertEqual(receipt["status"], "ok", (head, receipt))
            self.assertEqual(receipt["head"], head)
            self.assertEqual(receipt["op"], op)
            self.assertEqual(receipt["inputs"], {"a": a, "b": b})
            self.assertEqual(receipt["outputs"], [expected])
            self.assertEqual(receipt["output"], expected)
            self.assertEqual(receipt["execution_mode"], "interpreted")
            self.assertEqual(receipt["generation"], 1)
            self.assertEqual(receipt["runtime_generation"], 999)
            self.assertIs(receipt["canonical_merge"], False)
            self.assertIs(receipt["canonical_commit"], False)
            self.assertIs(receipt["native_pointer"], False)
            self.assertIs(receipt["command_execution"], False)
            self.assertEqual(receipt["identity"], cells.identity_address(head))
        self.assertEqual(len(runtime.calls), HEAD_COUNT)
        self.assertEqual({call["head"] for call in runtime.calls}, set(range(HEAD_COUNT)))

    def test_evaluate_and_dispatch_aliases_share_the_bounded_path(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime)
        first = bridge.evaluate(4, 1, 0, "OR")
        second = bridge.dispatch_boolean(4, 1, 0, 14)
        third = bridge.boolean_op(4, 1, 0, "or")
        self.assertEqual(first["outputs"], [1])
        self.assertEqual(second["outputs"], [1])
        self.assertEqual(third["outputs"], [1])
        self.assertEqual([first["generation"], second["generation"],
                          third["generation"]], [1, 2, 3])
        self.assertEqual(len(runtime.calls), 3)

    def test_native_execution_mode_is_preserved(self):
        bridge = MTHHeadBridge(FakeRuntime(mode="native"))
        receipt = bridge.boolean_op(9, 0, 1, "AND")
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["outputs"], [0])
        self.assertEqual(receipt["execution_mode"], "native")

    def test_invalid_requests_are_denied_before_dispatch(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime)
        cases = [
            ((-1, 1, 1, "AND"), "invalid_head"),
            ((HEAD_COUNT, 1, 1, "AND"), "invalid_head"),
            ((True, 1, 1, "AND"), "invalid_head"),
            ((0, 2, 1, "AND"), "invalid_operand"),
            ((0, 1, -1, "AND"), "invalid_operand"),
            ((0, "1", 1, "AND"), "invalid_operand"),
            ((0, 1, 1, "NOPE"), "invalid_op"),
            ((0, 1, 1, 16), "invalid_op"),
            ((0, 1, 1, True), "invalid_op"),
        ]
        for args, reason in cases:
            receipt = bridge.boolean_op(*args)
            self.assertEqual(receipt["status"], "denied", args)
            self.assertEqual(receipt["reason"], reason, args)
            self.assertEqual(receipt["outputs"], [])
            self.assertIsNone(receipt["output"])
            self.assertEqual(receipt["execution_mode"], "unavailable")
            self.assertEqual(receipt["generation"], 0)
        self.assertEqual(runtime.calls, [])

    def test_runtime_output_mismatch_is_denied_not_replaced(self):
        bridge = MTHHeadBridge(FakeRuntime(override={2: 0}))
        receipt = bridge.boolean_op(2, 1, 1, "AND")
        self.assertEqual(receipt["status"], "denied")
        self.assertEqual(receipt["reason"], "runtime_output_mismatch")
        self.assertEqual(receipt["outputs"], [0])
        self.assertEqual(receipt["expected_outputs"], [1])
        self.assertEqual(receipt["execution_mode"], "interpreted")
        self.assertEqual(receipt["generation"], 0)
        self.assertEqual(bridge.select_head(2)["generation"], 0)


class UnavailableRuntimeTests(unittest.TestCase):
    def test_unavailable_runtime_returns_structured_denial(self):
        runtime = FakeRuntime(available=False)
        bridge = MTHHeadBridge(runtime)
        receipt = bridge.boolean_op(3, 1, 1, "AND")
        self.assertEqual(receipt["status"], "denied")
        self.assertEqual(receipt["reason"], "runtime_unavailable")
        self.assertEqual(receipt["execution_mode"], "unavailable")
        self.assertEqual(receipt["outputs"], [])
        self.assertIsNone(receipt["output"])
        self.assertIs(receipt["available"], False)
        self.assertEqual(receipt["generation"], 0)
        self.assertIs(receipt["canonical_merge"], False)
        self.assertEqual(receipt["runtime_unavailable_reason"], "fake_unavailable")
        self.assertEqual(runtime.calls, [])

    def test_missing_runtime_interface_is_a_denial(self):
        bridge = MTHHeadBridge(object())
        receipt = bridge.boolean_op(0, 1, 1, "AND")
        self.assertEqual(receipt["status"], "denied")
        self.assertEqual(receipt["reason"], "runtime_unavailable")
        self.assertEqual(receipt["outputs"], [])
        self.assertIn("no dispatch_boolean", receipt["runtime_detail"])

    def test_runtime_exception_is_not_converted_to_a_result(self):
        bridge = MTHHeadBridge(FakeRuntime(raise_head=5))
        receipt = bridge.boolean_op(5, 1, 0, "XOR")
        self.assertEqual(receipt["status"], "denied")
        self.assertEqual(receipt["reason"], "runtime_unavailable")
        self.assertEqual(receipt["outputs"], [])
        self.assertEqual(receipt["runtime_error_type"], "RuntimeError")
        self.assertEqual(receipt["generation"], 0)

    def test_default_bridge_prefers_optional_mth_runtime_module(self):
        calls = []

        def dispatch_boolean(mask, a, b, *, head, steps, timeout):
            calls.append((mask, a, b, head, steps, timeout))
            return {"available": True, "output": truth(mask, a, b),
                    "execution_mode": "interpreted", "steps": 1}

        fake_module = types.SimpleNamespace(
            name="future_mth_runtime", dispatch_boolean=dispatch_boolean)
        with mock.patch.object(module, "_load_optional_mth_runtime",
                               return_value=fake_module):
            bridge = MTHHeadBridge()
            self.assertEqual(bridge.runtime_origin, "mth_runtime")
            self.assertEqual(bridge.runtime_name, "future_mth_runtime")
            receipt = bridge.boolean_op(6, 1, 0, "AND")
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["outputs"], [0])
        self.assertEqual(len(calls), 1)

    def test_closed_bridge_fails_closed(self):
        bridge = MTHHeadBridge(FakeRuntime())
        bridge.close()
        selection = bridge.select_head(0)
        receipt = bridge.boolean_op(0, 1, 1, "AND")
        field = bridge.field_aggregate(field_plan())
        self.assertEqual(selection["reason"], "bridge_closed")
        self.assertEqual(receipt["reason"], "bridge_closed")
        self.assertEqual(receipt["outputs"], [])
        self.assertEqual(field["reason"], "bridge_closed")
        self.assertIs(receipt["canonical_merge"], False)


class FieldAggregateTests(unittest.TestCase):
    def test_field_covers_all_256_heads_and_matches_isolated_decisions(self):
        plan = field_plan()
        field_bridge = MTHHeadBridge(FakeRuntime())
        aggregate = field_bridge.field_aggregate(plan)
        self.assertEqual(aggregate["status"], "ok", aggregate.get("detail"))
        self.assertEqual(aggregate["schema"], FIELD_SCHEMA)
        self.assertEqual(aggregate["head_count"], HEAD_COUNT)
        self.assertEqual(aggregate["selected_heads"], HEAD_COUNT)
        self.assertTrue(aggregate["identities_valid"])
        self.assertEqual(aggregate["identity_count"], HEAD_COUNT)
        self.assertTrue(aggregate["independent"])
        self.assertEqual(aggregate["execution_mode"], "interpreted")
        self.assertEqual(aggregate["generation"], 1)
        self.assertEqual(aggregate["generation_before"], 0)
        self.assertIs(aggregate["canonical_merge"], False)
        self.assertIs(aggregate["canonical_commit"], False)
        self.assertEqual(len(aggregate["receipts"]), HEAD_COUNT)
        self.assertEqual(len(aggregate["outputs"]), HEAD_COUNT)
        self.assertEqual({item["head"] for item in aggregate["receipts"]},
                         set(range(HEAD_COUNT)))
        expected = [truth(OP_NAMES.index(op), a, b)
                    for op, a, b in (head_case(head)
                                     for head in range(HEAD_COUNT))]
        self.assertEqual(aggregate["outputs"], expected)
        self.assertEqual(aggregate["true_heads"], sum(expected))
        self.assertEqual(aggregate["false_heads"], HEAD_COUNT - sum(expected))

        isolated = MTHHeadBridge(FakeRuntime())
        for head in range(HEAD_COUNT):
            op, a, b = plan[head]
            receipt = isolated.boolean_op(head, a, b, op)
            field_receipt = aggregate["receipts"][head]
            for key in ("head", "op", "op_mask", "inputs", "outputs",
                        "execution_mode", "generation", "identity",
                        "canonical_merge"):
                self.assertEqual(field_receipt[key], receipt[key],
                                 (head, key))

    def test_field_is_deterministic_across_bridges_and_repeats(self):
        plan = field_plan()
        first = MTHHeadBridge(FakeRuntime()).field_aggregate(plan)
        second = MTHHeadBridge(FakeRuntime()).field_aggregate(plan)
        self.assertEqual(first["output_digest"], second["output_digest"])
        self.assertEqual(first["outputs"], second["outputs"])
        self.assertEqual(first["identity_digest"], second["identity_digest"])
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))

        bridge = MTHHeadBridge(FakeRuntime())
        initial = bridge.field_aggregate(plan)
        repeated = bridge.field_aggregate(plan)
        self.assertEqual(initial["output_digest"], repeated["output_digest"])
        self.assertEqual(initial["outputs"], repeated["outputs"])
        self.assertEqual(initial["true_heads"], repeated["true_heads"])
        self.assertEqual(repeated["generation"], 2)
        self.assertEqual(repeated["receipts"][0]["generation"], 2)

    def test_field_rejects_incomplete_or_extra_heads_before_dispatch(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime)
        valid = field_plan()

        missing = {head: valid[head] for head in range(255)}
        extra = dict(valid)
        extra[HEAD_COUNT] = ("AND", 1, 1)
        out_of_range = {head: ("AND", 1, 1)
                        for head in range(1, HEAD_COUNT + 1)}
        duplicate = [((head if head != 1 else 0), "AND", 1, 1)
                     for head in range(HEAD_COUNT)]
        extra_key = dict(valid)
        extra_key[0] = {"op": "AND", "a": 1, "b": 1, "path": "/bin/sh"}
        bad_op = dict(valid)
        bad_op[5] = ("NOPE", 1, 1)
        bad_a = dict(valid)
        bad_a[6] = ("AND", 2, 1)
        bad_b = dict(valid)
        bad_b[7] = ("AND", 1, -1)

        for plan in (missing, extra, out_of_range, duplicate,
                     extra_key, bad_op, bad_a, bad_b):
            result = bridge.field_aggregate(plan)
            self.assertEqual(result["status"], "denied", plan)
            self.assertEqual(result["reason"], "plan_invalid")
            self.assertFalse(result["identities_valid"])
            self.assertEqual(result["outputs"], [])
            self.assertEqual(result["receipts"], [])
            self.assertIs(result["canonical_merge"], False)
        self.assertEqual(runtime.calls, [])

    def test_field_unavailable_denies_all_256_heads_without_fake_outputs(self):
        runtime = FakeRuntime(available=False)
        bridge = MTHHeadBridge(runtime)
        result = bridge.field_aggregate(field_plan())
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["reason"], "runtime_unavailable")
        self.assertEqual(result["execution_mode"], "unavailable")
        self.assertEqual(len(result["receipts"]), HEAD_COUNT)
        self.assertEqual(result["unavailable_heads"], HEAD_COUNT)
        self.assertEqual(result["failed_heads"], list(range(HEAD_COUNT)))
        self.assertEqual(result["outputs"], [None] * HEAD_COUNT)
        self.assertEqual(result["true_heads"], 0)
        self.assertEqual(result["false_heads"], 0)
        self.assertTrue(result["identities_valid"])
        self.assertEqual(result["generation"], 0)
        self.assertIs(result["canonical_merge"], False)
        for receipt in result["receipts"]:
            self.assertEqual(receipt["status"], "denied")
            self.assertEqual(receipt["execution_mode"], "unavailable")
            self.assertEqual(receipt["outputs"], [])
            self.assertEqual(receipt["generation"], 0)
        self.assertEqual(runtime.calls, [])

    def test_one_failed_head_does_not_change_other_heads_or_commit(self):
        runtime = FakeRuntime(fail_head=17)
        bridge = MTHHeadBridge(runtime)
        result = bridge.field_aggregate(field_plan())
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["reason"], "head_execution_failed")
        self.assertEqual(result["failed_heads"], [17])
        self.assertEqual(len(result["receipts"]), HEAD_COUNT)
        self.assertEqual(result["receipts"][17]["status"], "denied")
        self.assertEqual(result["receipts"][17]["execution_mode"], "unavailable")
        for head in (0, 1, 16, 18, 255):
            self.assertEqual(result["receipts"][head]["status"], "ok")
        self.assertEqual(result["generation"], 0)
        for head in range(HEAD_COUNT):
            self.assertEqual(bridge.select_head(head)["generation"], 0)

    def test_uniform_field_applies_same_operation_per_head(self):
        bridge = MTHHeadBridge(FakeRuntime())
        result = bridge.uniform_field("XOR", 1, 0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["outputs"], [1] * HEAD_COUNT)
        self.assertEqual(result["true_heads"], HEAD_COUNT)
        self.assertEqual(result["false_heads"], 0)
        self.assertEqual(len(result["receipts"]), HEAD_COUNT)
        self.assertEqual(len(bridge._runtime.calls), HEAD_COUNT)


class BoundsTests(unittest.TestCase):
    def test_constructor_cannot_exceed_fixed_bounds(self):
        cases = [
            {"max_steps": MAX_STEPS + 1},
            {"max_timeout": MAX_TIMEOUT + 1},
            {"max_plan_bytes": MAX_PLAN_BYTES + 1},
            {"max_aggregate_bytes": MAX_AGGREGATE_BYTES + 1},
            {"max_field_seconds": MAX_FIELD_SECONDS + 1},
            {"max_steps": 0},
            {"max_timeout": -1},
            {"max_plan_bytes": True},
            {"max_aggregate_bytes": "1024"},
            {"max_field_seconds": 0},
        ]
        for kwargs in cases:
            with self.assertRaises(MTHHeadBridgeError, msg=str(kwargs)):
                MTHHeadBridge(FakeRuntime(), **kwargs)
        bridge = MTHHeadBridge(
            FakeRuntime(), max_steps=100, max_timeout=2,
            max_plan_bytes=4096, max_aggregate_bytes=65536,
            max_field_seconds=5)
        self.assertEqual(bridge.bounds["max_steps"], 100)
        self.assertEqual(bridge.bounds["max_timeout"], 2)
        self.assertEqual(bridge.bounds["max_plan_bytes"], 4096)
        self.assertEqual(bridge.bounds["max_aggregate_bytes"], 65536)
        self.assertEqual(bridge.bounds["max_field_seconds"], 5)
        self.assertEqual(bridge.bounds["heads"], HEAD_COUNT)
        self.assertEqual(bridge.bounds["max_image_bytes"],
                         module.MAX_IMAGE_BYTES)
        self.assertEqual(bridge.bounds["max_guest_words"],
                         module.MAX_GUEST_WORDS)

    def test_per_call_steps_and_timeout_are_bounded(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime, max_steps=10, max_timeout=2)
        for steps in (0, -1, 11, True, "5"):
            receipt = bridge.boolean_op(0, 1, 1, "AND", steps=steps)
            self.assertEqual(receipt["status"], "denied")
            self.assertEqual(receipt["reason"], "steps_out_of_range")
            self.assertEqual(receipt["outputs"], [])
        for timeout in (0, -1, 3, True, "2"):
            receipt = bridge.boolean_op(0, 1, 1, "AND", timeout=timeout)
            self.assertEqual(receipt["status"], "denied")
            self.assertEqual(receipt["reason"], "timeout_out_of_range")
            self.assertEqual(receipt["outputs"], [])
        self.assertEqual(runtime.calls, [])
        receipt = bridge.boolean_op(0, 1, 1, "AND", steps=10, timeout=2)
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["generation"], 1)

    def test_plan_and_aggregate_bytes_are_bounded(self):
        plan = field_plan()
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime, max_plan_bytes=512)
        result = bridge.field_aggregate(plan)
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["reason"], "plan_invalid")
        self.assertIn("bound", result["detail"])
        self.assertGreater(result["plan"]["bytes"], 512)
        self.assertEqual(runtime.calls, [])

        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime, max_aggregate_bytes=4096)
        result = bridge.field_aggregate(plan)
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["reason"], "aggregate_bytes_exceeded")
        self.assertEqual(result["head_count"], HEAD_COUNT)
        self.assertGreater(result["aggregate_bytes"], 4096)
        self.assertEqual(len(runtime.calls), HEAD_COUNT)
        # A report that violates the byte ceiling does not commit a generation.
        self.assertEqual(bridge.select_head(0)["generation"], 0)

    def test_no_pointer_or_command_surface(self):
        runtime = FakeRuntime()
        bridge = MTHHeadBridge(runtime)
        with self.assertRaises(TypeError):
            bridge.boolean_op(0, 1, 1, "AND", pointer=0x1000)
        with self.assertRaises(TypeError):
            bridge.boolean_op(0, 1, 1, "AND", command="true")
        with self.assertRaises(TypeError):
            bridge.field_aggregate(field_plan(), command="true")
        receipt = bridge.boolean_op(0, 1, 1, "AND")
        self.assertIs(receipt["native_pointer"], False)
        self.assertIs(receipt["command_execution"], False)
        self.assertIs(receipt["canonical_merge"], False)
        self.assertEqual(set(runtime.calls[-1]),
                         {"mask", "a", "b", "head", "steps", "timeout"})


class ToolContext:
    def __init__(self, root):
        self.config = {
            "state_dir": str(root),
            "nan0s_prefer_bundled": True,
        }

    def get_config(self, key, default=None):
        return self.config.get(key, default)


@unittest.skipUnless(_real_toolchain_ready(),
                     "real bundled NaN0S/N16I toolchain is not available")
class RealRuntimeIntegrationTests(unittest.TestCase):
    def test_real_bundled_boolean_kernel_executes_one_head_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = MTHHeadBridge(ctx=ToolContext(Path(tmp) / "state"))
            try:
                self.assertEqual(bridge.runtime_origin, "nan0s")
                receipt = bridge.boolean_op(17, 1, 0, "OR")
                self.assertEqual(receipt["status"], "ok",
                                 receipt.get("runtime_detail"))
                self.assertEqual(receipt["outputs"], [1])
                self.assertEqual(receipt["execution_mode"], "interpreted")
                self.assertEqual(receipt["generation"], 1)
                self.assertEqual(receipt["head"], 17)
                self.assertEqual(receipt["op"], "OR")
                self.assertIs(receipt["canonical_merge"], False)
                selection = bridge.select_head(17)
                self.assertEqual(selection["identity"],
                                 cells.identity_address(17))
            finally:
                bridge.close()


if __name__ == "__main__":
    unittest.main()
