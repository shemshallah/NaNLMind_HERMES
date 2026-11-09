import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import mth_catalog, nan0s_runtime, register


class ToolContext:
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
        self.tools[kwargs["name"]] = kwargs


class MthCatalogTests(unittest.TestCase):
    def test_bundle_has_all_ninety_one_kernels_and_sources(self):
        health = mth_catalog.health()
        self.assertEqual(health["total"], 91)
        self.assertEqual(health["ci_verified"], 91)
        self.assertEqual(health["native_dispatch_count"], 4)
        self.assertIn("graph", health["families"])
        self.assertIn("vector", health["families"])
        for kernel in mth_catalog.kernels():
            self.assertEqual(len(kernel.source_digest), 64)
            self.assertEqual(len(kernel.fixture_digest), 64)
            self.assertTrue(mth_catalog.source_path(kernel.kernel_id).is_file())
            self.assertTrue(mth_catalog.fixture_path(kernel.kernel_id).is_file())

    def test_catalog_pages_and_inspects(self):
        page = mth_catalog.list_kernels(family="vector", limit=100)
        ids = {item["kernel_id"] for item in page["kernels"]}
        self.assertIn("mth.vector.dot256.f32", ids)
        self.assertIn("mth.vector.add256.f32", ids)
        self.assertIn("mth.vector.cosine256.f32", ids)
        self.assertIn("mth.vector.sub256.f32", ids)
        dot = mth_catalog.inspect("mth.vector.dot256.f32")["kernel"]
        self.assertEqual(dot["execution_mode"], "native_dispatch_abi")
        self.assertTrue(dot["dynamic_execution_available"])
        graph = mth_catalog.inspect("mth.graph.wave_step.f32")["kernel"]
        self.assertEqual(graph["execution_mode"], "native_fixture_only")
        with self.assertRaises(mth_catalog.MthCatalogError):
            mth_catalog.inspect("not-a-kernel")
        with self.assertRaises(mth_catalog.MthCatalogError):
            mth_catalog.list_kernels(limit=101)


@unittest.skipUnless(
    nan0s_runtime.compiler_available(),
    "C compiler required to build the bundled NaN0S portable core")
class MthRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "state"
        cls.ctx = ToolContext(cls.root)
        register(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_architecture_autodetect_and_bundled_build(self):
        arch = nan0s_runtime.host_architecture()
        self.assertIn(arch["target"], {"aarch64", "x86_64", "x86", "riscv64",
                                       "ppc64le", "s390x", "portable-c"})
        core = nan0s_runtime.discover_core(self.ctx)
        self.assertEqual(core["layout"], "bundled")
        status = nan0s_runtime.runtime_status(self.ctx, build=True)
        self.assertTrue(status["native_available"])
        self.assertEqual(status["core"]["layout"], "bundled")
        self.assertTrue(Path(status["cli"]["cli"]).is_file())
        self.assertIn("n16i", status)
        self.assertTrue(status["n16i"]["guest_fallback_wired"])
        self.assertTrue(status["execution_modes"]["n16i_arbitrary_kernel_fallback"])

    def test_native_fixture_verification(self):
        result = nan0s_runtime.verify_kernel(
            self.ctx, "mth.k256.laplacian.u16")
        self.assertTrue(result["verified"], result["run"]["stderr"][-1000:])
        self.assertTrue(result["marker_found"])
        self.assertEqual(result["execution_mode"], "native_nan0s")
        self.assertTrue(result["run"]["halted"])
        self.assertEqual(result["run"]["trap"], "")
        self.assertFalse(result["run"]["n16i_used"])

    def assertResultsAgree(self, left, right):
        if isinstance(left, list):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, delta=1e-5)
        else:
            self.assertAlmostEqual(left, right, delta=1e-5)

    def test_unified_vector_execution_uses_n16i_and_matches_native(self):
        x = [2.0] * 256
        y = [3.0] * 256
        cases = {
            "mth.vector.add256.f32": [5.0] * 256,
            "mth.vector.sub256.f32": [-1.0] * 256,
            "mth.vector.dot256.f32": 1536.0,
            "mth.vector.cosine256.f32": 1.0,
        }
        for kernel_id, expected in cases.items():
            unified = nan0s_runtime.execute_vector_kernel(
                self.ctx, kernel_id, x, y)
            self.assertTrue(unified["ok"], kernel_id)
            self.assertEqual(unified["status"], 1)
            self.assertEqual(unified["execution_mode"],
                             "n16i_guest_interpreter")
            self.assertIs(unified["n16i_used"], True)
            self.assertIs(unified["native"], False)
            self.assertGreater(unified["guest_base"], 0)
            self.assertEqual(unified["dispatch"]["execution_mode"],
                             "n16i_guest_interpreter")
            self.assertIs(unified["dispatch"]["n16i_used"], True)
            self.assertResultsAgree(unified["result"], expected)

            native = nan0s_runtime.execute_vector_kernel_native(
                self.ctx, kernel_id, x, y)
            self.assertTrue(native["ok"], kernel_id)
            self.assertEqual(native["execution_mode"], "native_nan0s")
            self.assertIs(native["n16i_used"], False)
            self.assertResultsAgree(unified["result"], native["result"])
            # Both paths assemble the same driver image and guest readback.
            self.assertEqual(unified["image"]["image"], native["image"]["image"])
            guest_base = unified["guest_base"]
            guest_mask = unified["dispatch"]["guest_mask"]
            output_words = 1 if isinstance(expected, float) else 256
            self.assertEqual(
                unified["dispatch"]["peek"],
                "0x%X:%d,0x%X:1" % (
                    guest_base + (0x4200 & guest_mask), output_words,
                    guest_base + (0x4304 & guest_mask)))
        with self.assertRaises(nan0s_runtime.NativeRuntimeError):
            nan0s_runtime.execute_vector_kernel(
                self.ctx, "mth.graph.wave.f32", x, y)

    def test_arbitrary_kernel_image_dispatches_under_n16i(self):
        native = nan0s_runtime.verify_kernel(
            self.ctx, "mth.k256.laplacian.u16")
        dispatched = nan0s_runtime.verify_kernel_n16i(
            self.ctx, "mth.k256.laplacian.u16", max_steps=5_000_000)
        self.assertTrue(dispatched["verified"], dispatched["dispatch"]["run"]["stderr"][-1000:])
        self.assertTrue(dispatched["n16i_used"])
        self.assertTrue(dispatched["marker_found"])
        self.assertEqual(dispatched["execution_mode"], "n16i_guest_interpreter")
        self.assertGreater(dispatched["dispatch"]["guest_words"], 500)
        self.assertGreater(dispatched["dispatch"]["run"]["steps"],
                           native["run"]["steps"])
        self.assertEqual(dispatched["dispatch"]["stdout"].strip(),
                         native["run"]["stdout"].strip())

    def test_n16i_differential_self_test(self):
        result = nan0s_runtime.n16i_self_test(self.ctx)
        self.assertTrue(result["identical"], result["native"]["stderr"][-1000:])
        self.assertTrue(result["n16i_used"])
        self.assertGreater(result["interpreted_steps"], result["native_steps"])
        self.assertGreater(result["output_bytes"], 0)
        self.assertEqual(result["execution_mode"], "n16i_guest_interpreter")

    def test_plugin_tools_expose_catalog_runtime_verify_and_execute(self):
        catalog = json.loads(self.ctx.tools["nanlm_mth_catalog"]["handler"](
            {"limit": 3}))
        self.assertEqual(catalog["total"], 91)
        self.assertEqual(len(catalog["kernels"]), 3)
        inspected = json.loads(self.ctx.tools["nanlm_mth_inspect"]["handler"](
            {"kernel_id": "mth.vector.add256.f32"}))
        self.assertEqual(inspected["kernel"]["family"], "vector")
        runtime = json.loads(self.ctx.tools["nanlm_mth_runtime"]["handler"](
            {"build": True}))
        self.assertTrue(runtime["native_available"])
        executed = json.loads(self.ctx.tools["nanlm_mth_execute"]["handler"](
            {"kernel_id": "mth.vector.sub256.f32",
             "x": [2.0] * 256, "y": [3.0] * 256}))
        self.assertTrue(executed["ok"])
        self.assertEqual(executed["result"][0], -1.0)
        self.assertEqual(executed["execution_mode"], "n16i_guest_interpreter")
        self.assertIs(executed["n16i_used"], True)
        self.assertIs(executed["native"], False)
        self.assertEqual(executed["authorization"]["mode"], "local-unverified")
        n16i = json.loads(self.ctx.tools["nanlm_mth_n16i_selftest"]["handler"]({}))
        self.assertTrue(n16i["identical"])
        self.assertTrue(n16i["n16i_used"])
        dispatched = json.loads(self.ctx.tools["nanlm_mth_dispatch"]["handler"](
            {"kernel_id": "mth.k256.laplacian.u16", "max_steps": 5_000_000}))
        self.assertTrue(dispatched["verified"])
        self.assertTrue(dispatched["n16i_used"])
        self.assertEqual(dispatched["authorization"]["mode"], "local-unverified")


class MthRuntimeUnavailableTests(unittest.TestCase):
    def test_missing_compiler_reports_actionable_fail_closed_status(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(Path(tmp) / "state")
            with patch.object(nan0s_runtime, "_compiler", return_value=None):
                status = nan0s_runtime.runtime_status(ctx, build=True)
            self.assertFalse(status["native_available"])
            self.assertTrue(status["native_build_required"])
            self.assertIn("no C compiler found", status["error"])
            self.assertEqual(status["core"]["layout"], "bundled")


if __name__ == "__main__":
    unittest.main()
