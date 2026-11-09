"""Acceptance tests for the unified N16I path over the admitted vector kernels.

The four admitted vector kernels (dot, cosine, add, sub) must share one
general operand path.  ``execute_vector_kernel_native`` stays as the direct
native diagnostic, while ``execute_vector_kernel`` -- and therefore the
``nanlm_mth_execute`` tool and MTH leaf math pairing -- must run the same
assembled guest image under the bundled N16I interpreter.

These tests need a host C compiler and a NaN16-emitter-compatible awk (gawk or
mawk).  The runtime itself rejects BusyBox awk, so only the compiler is probed
here.  No merge, promotion, lease or NANNET route is exercised.
"""
import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import mth_leaves, nan0s_runtime, register


COMPILER_AVAILABLE = nan0s_runtime.compiler_available()
SKIP_REASON = (
    "C compiler and compatible awk are required for the bundled NaN0S core"
)

X = [2.0] * 256
Y = [3.0] * 256


class Context:
    """Minimal OpenCode/Hermes-style context for local registration."""

    def __init__(self, root):
        self.config = {
            "state_dir": str(root),
            "nan0s_prefer_bundled": True,
            "chat_mode": "provider",
            "chat_learning": False,
        }
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


def _assert_vector(test, values, expected, message=""):
    test.assertEqual(len(values), 256, message)
    for index, value in enumerate(values):
        test.assertAlmostEqual(
            float(value), expected, delta=1e-5,
            msg="%s element %d: %r != %r" % (message, index, value, expected))


def _assert_scalar(test, value, expected, message=""):
    test.assertAlmostEqual(
        float(value), expected, delta=1e-5,
        msg="%s: %r != %r" % (message, value, expected))


def _assert_result(test, value, expected, vector, message=""):
    if vector:
        _assert_vector(test, value, expected, message)
    else:
        _assert_scalar(test, value, expected, message)


def _assert_results_agree(test, first, second, vector, message=""):
    if vector:
        test.assertEqual(len(first), len(second), message)
        test.assertEqual(len(first), 256, message)
        for index, (left, right) in enumerate(zip(first, second)):
            test.assertAlmostEqual(
                float(left), float(right), delta=1e-5,
                msg="%s element %d differs: %r vs %r"
                    % (message, index, left, right))
    else:
        test.assertAlmostEqual(
            float(first), float(second), delta=1e-5,
            msg="%s scalar differs: %r vs %r" % (message, first, second))


class VectorKernelCase(unittest.TestCase):
    """Shared native/N16I contract for one admitted vector kernel."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ctx = Context(Path(cls._tmp.name) / "state")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _check_kernel(self, kernel_id, expected, vector):
        native = nan0s_runtime.execute_vector_kernel_native(
            self.ctx, kernel_id, list(X), list(Y))
        self.assertTrue(native["ok"], native)
        self.assertEqual(native["status"], 1)
        self.assertEqual(native["execution_mode"], "native_nan0s")
        self.assertFalse(native["n16i_used"])
        _assert_result(self, native["result"], expected, vector, kernel_id)

        unified = nan0s_runtime.execute_vector_kernel(
            self.ctx, kernel_id, list(X), list(Y))
        self.assertTrue(unified["ok"], unified)
        self.assertEqual(unified["status"], 1)
        self.assertEqual(unified["execution_mode"], "n16i_guest_interpreter")
        self.assertTrue(unified["n16i_used"])
        self.assertFalse(unified["native"])
        _assert_result(self, unified["result"], expected, vector, kernel_id)

        _assert_results_agree(
            self, native["result"], unified["result"], vector,
            kernel_id + " native/N16I")


@unittest.skipUnless(COMPILER_AVAILABLE, SKIP_REASON)
class UnifiedAddTests(VectorKernelCase):
    def test_add_is_unified(self):
        self._check_kernel("mth.vector.add256.f32", 5.0, True)


@unittest.skipUnless(COMPILER_AVAILABLE, SKIP_REASON)
class UnifiedSubTests(VectorKernelCase):
    def test_sub_is_unified(self):
        self._check_kernel("mth.vector.sub256.f32", -1.0, True)


@unittest.skipUnless(COMPILER_AVAILABLE, SKIP_REASON)
class UnifiedDotTests(VectorKernelCase):
    def test_dot_is_unified(self):
        self._check_kernel("mth.vector.dot256.f32", 1536.0, False)


@unittest.skipUnless(COMPILER_AVAILABLE, SKIP_REASON)
class UnifiedCosineTests(VectorKernelCase):
    def test_cosine_is_unified(self):
        self._check_kernel("mth.vector.cosine256.f32", 1.0, False)


@unittest.skipUnless(COMPILER_AVAILABLE, SKIP_REASON)
class PluginToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ctx = Context(Path(cls._tmp.name) / "state")
        register(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_nanlm_mth_execute_reports_unified_n16i_mode(self):
        executed = json.loads(self.ctx.tools["nanlm_mth_execute"]({
            "kernel_id": "mth.vector.add256.f32",
            "x": list(X),
            "y": list(Y),
        }))
        self.assertTrue(executed["ok"], executed)
        self.assertEqual(executed["status"], 1)
        self.assertEqual(
            executed["execution_mode"], "n16i_guest_interpreter")
        self.assertTrue(executed["n16i_used"])
        self.assertFalse(executed["native"])
        _assert_vector(
            self, executed["result"], 5.0, "nanlm_mth_execute add")
        self.assertEqual(executed["authorization"]["mode"], "local-unverified")


@unittest.skipUnless(COMPILER_AVAILABLE, SKIP_REASON)
class LeafMathTests(unittest.TestCase):
    def setUp(self):
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = Context(Path(self._tmp.name) / "state")
        self.store = mth_leaves.store_for_context(self.ctx)

    def tearDown(self):
        mth_leaves.reset_cache()
        self._tmp.cleanup()

    def test_add_pairing_records_unified_n16i_mode_and_preserves_inputs(self):
        left = self.store.create(
            kind="vector", payload={"fill": 1.0}, alias="n16i-left")
        right = self.store.create(
            kind="vector", payload={"fill": 2.0}, alias="n16i-right")
        left_before = self.store.get("n16i-left")
        right_before = self.store.get("n16i-right")

        paired = self.store.pair(
            "n16i-left", "n16i-right", operator="add", alias="n16i-sum")
        self.assertEqual(paired.status, "ok", paired.error)
        result_leaf = paired.leaf
        self.assertEqual(result_leaf["kind"], "vector")
        _assert_vector(
            self, result_leaf["payload"]["values"], 3.0, "paired add")

        execution = result_leaf["function_definition"]["execution"]
        self.assertEqual(execution["mode"], "n16i_guest_interpreter")
        self.assertTrue(execution["n16i_used"])

        left_after = self.store.get("n16i-left")
        right_after = self.store.get("n16i-right")
        self.assertEqual(left_after["leaf_id"], left["leaf_id"])
        self.assertEqual(right_after["leaf_id"], right["leaf_id"])
        self.assertEqual(
            left_after["content_sha256"], left_before["content_sha256"])
        self.assertEqual(
            right_after["content_sha256"], right_before["content_sha256"])
        self.assertEqual(left_after["payload"], left_before["payload"])
        self.assertEqual(right_after["payload"], right_before["payload"])
        self.assertEqual(
            left_after["leaf_id"], left_before["leaf_id"])
        self.assertEqual(
            right_after["leaf_id"], right_before["leaf_id"])


if __name__ == "__main__":
    unittest.main()
