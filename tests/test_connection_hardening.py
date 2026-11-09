"""Connection/authority edge hardening tests.

These tests stay local: no production NANNET, no canonical merge, no lease and
no provider network call. They exercise the registered handlers against a
temporary state directory and a loopback pairing authority.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import secrets
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanlmind_hermes import (
    auth_tools,
    mth_leaves,
    register,
    session_context,
    tools,
)
from nanlmind_hermes.http_api import make_server as make_authority
from nanlmind_hermes.layered_learning import (
    coordinator_for_context,
    reset_coordinator_cache,
)
from nanlmind_hermes.pairing import PairingAuthority
from nanlmind_hermes.phone_call import PhoneCall
from scripts import gemma_tool_learning_session as gemma_session
from scripts import gemma_training_series as gemma_series

SCOPES = ["universe:read", "training:execute"]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, messages=None, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if kwargs.get("purpose") in {
                "nanlmind-hermes-chat", "nanlm-phone-hermes"}:
            return SimpleNamespace(text="A river is a body of water.",
                                   provider="edge-fake", model="chat")
        return SimpleNamespace(
            text=json.dumps({"discoveries": [{
                "kind": "relation",
                "subject": "river",
                "predicate": "is_a",
                "object": "water-body",
                "resolution": "lexical",
                "polarity": "supports",
                "confidence": 0.9,
                "evidence": {"turns": 1},
                "provenance": {"test": "connection-hardening"},
            }]}),
            provider="edge-fake",
            model="reflection",
        )


class Context:
    def __init__(self, root, config=None, llm=None):
        self.config = {"state_dir": str(root), **(config or {})}
        self.llm = llm or FakeLLM()
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


class ScopeHardeningTests(unittest.TestCase):
    def setUp(self):
        session_context.reset_cache()
        reset_coordinator_cache()
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(mth_leaves.reset_cache)
        self.addCleanup(reset_coordinator_cache)
        self.addCleanup(session_context.reset_cache)

        root = Path(self._tmp.name)
        master = secrets.token_bytes(32)
        master_b64 = base64.urlsafe_b64encode(master).rstrip(b"=").decode()
        self.authority = PairingAuthority(root / "authority.sqlite", master)
        self.server = make_authority(self.authority, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_authority)
        self.authority_url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.secrets = {"NANNET_MASTER_KEY": master_b64}
        self._patcher = patch.object(
            auth_tools, "get_secret",
            side_effect=lambda name: self.secrets.get(name, ""))
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.ctx = Context(root / "state", {
            "chat_mode": "provider",
            "chat_learning": False,
            "nannet_auth_url": self.authority_url,
            "nannet_issuer_public_key": self.authority.issuer_public_key(),
        })
        register(self.ctx)

    def _stop_authority(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _call(self, name, params=None):
        raw = self.ctx.tools[name](params or {})
        return json.loads(raw) if isinstance(raw, str) else raw

    def _pair(self, scopes):
        self._call("nanlm_auth_init")
        code = self.authority.create_pairing_code(
            scopes, ttl_seconds=600)["pairing_code"]
        paired = self._call("nanlm_auth_pair", {"pairing_code": code})
        self.assertTrue(paired["paired"])
        return paired

    def test_curriculum_and_generate_handlers_fail_closed_without_pairing(self):
        cases = (
            ("nanlm_curriculum_experiment", {"topic": "logic"}),
            ("nanlm_generate_curriculum", {"topic": "logic"}),
            ("nanlm_generate_corpus", {"topic": "logic"}),
        )
        for name, params in cases:
            with self.subTest(tool=name):
                with self.assertRaisesRegex(ValueError, "pair with NANNET"):
                    self._call(name, params)
        self.assertEqual(self.ctx.llm.calls, [])

    def test_scopes_are_checked_by_name_with_a_limited_token(self):
        self._pair(["universe:read"])
        training_cases = (
            ("nanlm_curriculum_experiment", {"topic": "logic"}),
            ("nanlm_generate_curriculum", {"topic": "logic"}),
        )
        for name, params in training_cases:
            with self.subTest(tool=name):
                with self.assertRaisesRegex(
                        ValueError, "token lacks required scope: training:execute"):
                    self._call(name, params)
        with self.assertRaisesRegex(
                ValueError, "token lacks required scope: corpus:write"):
            self._call("nanlm_generate_corpus", {"topic": "logic"})
        self.assertEqual(self.ctx.llm.calls, [])


class MergeGateTests(unittest.TestCase):
    def setUp(self):
        session_context.reset_cache()
        reset_coordinator_cache()
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(mth_leaves.reset_cache)
        self.addCleanup(reset_coordinator_cache)
        self.addCleanup(session_context.reset_cache)
        self.ctx = Context(Path(self._tmp.name) / "state", {
            "chat_mode": "native",
            "chat_learning": False,
        })
        register(self.ctx)

    def _call(self, name, params=None):
        raw = self.ctx.tools[name](params or {})
        return json.loads(raw) if isinstance(raw, str) else raw

    def _cycle(self):
        return self._call("nanlm_learning_cycle", {
            "message": "Copper is a metal.",
            "source": "merge-gate-test",
            "relations": [{
                "subject": "copper",
                "predicate": "is",
                "object": "metal",
            }],
            "approved": True,
        })

    def test_status_exposes_merge_gate_disabled_by_default(self):
        status = self._call("nanlm_status")
        self.assertFalse(status["merge_gate"]["canonical_merge_enabled"])
        self.assertFalse(status["merge_gate"]["merge_enabled"])
        self.assertFalse(status["merge_gate"]["local_promotion_enabled"])

    def test_learning_sync_and_cycle_cannot_promote_when_disabled(self):
        for _ in range(3):
            cycle = self._cycle()
            self.assertEqual(cycle["sync"]["status"], "merge_disabled")
        sync = self._call("nanlm_learning_sync", {"approved": True})
        self.assertEqual(sync["status"], "merge_disabled")
        self.assertFalse(sync["approved"])
        self.assertEqual(sync["promoted"], 0)
        status = self._call("nanlm_learning_status")
        self.assertEqual(status["promoted"], 0)

    def test_wordnet_ingest_never_synchronizes_when_disabled(self):
        coordinator = coordinator_for_context(self.ctx)
        with patch.object(coordinator, "ingest_word",
                          return_value={"schema": "fake-wordnet/1", "word": "copper"}), \
             patch.object(coordinator, "synchronize",
                          side_effect=AssertionError("promotion must not run")):
            result = self._call("nanlm_wordnet_ingest", {
                "word": "copper", "depth": 2, "approved": True,
            })
        self.assertEqual(result["sync"]["status"], "merge_disabled")

    def test_memory_merge_is_disabled_without_touching_the_ledger(self):
        with patch.object(tools, "memory_merge",
                          side_effect=AssertionError("merge must not run")):
            result = self._call("nanlm_memory_merge", {
                "records": [{
                    "item": {"kind": "observation", "subject": "x"},
                    "source_digest": "a" * 64,
                }],
            })
        self.assertEqual(result["status"], "merge_disabled")
        self.assertEqual(result["merge"], {"accepted": 0, "duplicates": 0})


class MergeEnabledTests(unittest.TestCase):
    def setUp(self):
        session_context.reset_cache()
        reset_coordinator_cache()
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(mth_leaves.reset_cache)
        self.addCleanup(reset_coordinator_cache)
        self.addCleanup(session_context.reset_cache)
        self.ctx = Context(Path(self._tmp.name) / "state", {
            "chat_mode": "native",
            "chat_learning": False,
            "learning_require_approval": True,
            "learning_auto_sync": False,
            "merge_enabled": True,
        })
        register(self.ctx)

    def _call(self, name, params=None):
        raw = self.ctx.tools[name](params or {})
        return json.loads(raw) if isinstance(raw, str) else raw

    def test_operator_enabled_gate_preserves_local_promotion(self):
        for _ in range(3):
            self._call("nanlm_learning_cycle", {
                "message": "Copper is a metal.",
                "source": "merge-enabled-test",
                "relations": [{
                    "subject": "copper",
                    "predicate": "is",
                    "object": "metal",
                }],
                "approved": False,
            })
        sync = self._call("nanlm_learning_sync", {"approved": True})
        self.assertEqual(sync["status"], "promoted")
        self.assertGreaterEqual(sync["promoted"], 1)
        status = self._call("nanlm_status")
        self.assertTrue(status["merge_gate"]["merge_enabled"])
        self.assertFalse(status["merge_gate"]["canonical_merge_enabled"])
        self.assertTrue(status["merge_gate"]["local_promotion_enabled"])


class MthEdgeTests(unittest.TestCase):
    def setUp(self):
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = Context(Path(self._tmp.name) / "state", {
            "nan0s_prefer_bundled": True,
            "chat_learning": False,
        })
        self.store = mth_leaves.store_for_context(self.ctx)

    def tearDown(self):
        mth_leaves.reset_cache()
        self._tmp.cleanup()

    def test_unknown_left_or_right_leaf_pairing_is_denied_not_raised(self):
        self.store.create(kind="vector", payload={"fill": 1.0}, alias="v1")
        before = self.store.status()["leaves"]
        for left, right in (("missing-leaf", "v1"), ("v1", "missing-leaf")):
            with self.subTest(left=left, right=right):
                result = self.store.pair(left, right, operator="connect")
                self.assertEqual(result.status, "denied")
                self.assertEqual(result.error["code"], "unknown_leaf")
                self.assertIsNone(result.hashleaf)
                self.assertIsNone(result.leaf)
        self.assertEqual(self.store.status()["leaves"], before)

    def test_dangling_kernel_alias_makes_function_tree_audit_incomplete(self):
        mapped = self.store.ensure_function_tree()
        self.assertTrue(mapped["complete"])
        self.assertEqual(mapped["dangling_aliases"], [])
        kernel_id = "mth.vector.dot256.f32"
        self.store._data["aliases"]["kernel:" + kernel_id] = "leaf:" + "0" * 32
        audit = self.store.function_tree_audit(verify_images=False)
        self.assertEqual(audit["dangling_aliases"], [kernel_id])
        self.assertFalse(audit["complete"])

    def test_mth_hub_returns_a_bounded_hub_tree_summary(self):
        self.store.ensure_function_tree()
        register(self.ctx)
        raw = self.ctx.tools["nanlm_mth_hub"]({})
        hub = json.loads(raw) if isinstance(raw, str) else raw
        tree = hub["hub_tree"]
        self.assertEqual(tree["schema"], mth_leaves.HUB_SCHEMA)
        self.assertLessEqual(tree["limit"], mth_leaves.HUB_TREE_LIMIT)
        self.assertLessEqual(len(tree["hubs"]), tree["limit"])
        self.assertEqual(tree["truncated"], tree["total"] > len(tree["hubs"]))
        self.assertGreaterEqual(tree["total"], 9)
        for item in tree["hubs"]:
            self.assertTrue(item["leaf_id"].startswith("hub:"))
            self.assertEqual(item["kind"], "hub")
            self.assertTrue(item["parent"] is None
                            or str(item["parent"]).startswith("hub:"))
            self.assertIsInstance(item["child_count"], int)
            self.assertGreaterEqual(item["child_count"], 0)
        self.assertTrue(any(item["child_count"] > 0 for item in tree["hubs"]))


class PhoneCallContextTests(unittest.TestCase):
    def setUp(self):
        session_context.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(session_context.reset_cache)
        self.ctx = Context(Path(self._tmp.name) / "state", {
            "chat_mode": "provider",
        })

    def test_hermes_provider_receives_compact_permanent_instructions(self):
        call = PhoneCall(self.ctx, "edge-call")
        text = call._hermes_message("Relay this.", [], "fallback")
        self.assertEqual(text, "A river is a body of water.")
        self.assertEqual(len(self.ctx.llm.calls), 1)
        system = self.ctx.llm.calls[0]["messages"][0]["content"]
        self.assertIn("PERMANENT PLUGIN SESSION RULES", system)
        self.assertIn("Permanent routing anchors", system)
        self.assertNotIn("PERMANENT PLUGIN SESSION CONTEXT", system)
        self.assertNotIn("## 1. Permanent plugin-duration instructions", system)


class GemmaSafeSurfaceTests(unittest.TestCase):
    def _context(self):
        received = []

        def handler(name):
            def call(params):
                received.append((name, dict(params)))
                return {"ok": True}
            return call

        class Ctx:
            tools = {
                "nanlm_learning_cycle": {
                    "handler": handler("nanlm_learning_cycle"),
                    "schema": {
                        "name": "nanlm_learning_cycle",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "approved": {"type": "boolean"},
                            },
                            "required": ["message"],
                        },
                    },
                },
                "nanlm_wordnet_ingest": {
                    "handler": handler("nanlm_wordnet_ingest"),
                    "schema": {
                        "name": "nanlm_wordnet_ingest",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "approved": {"type": "boolean"},
                            },
                            "required": ["word"],
                        },
                    },
                },
                "nanlm_learning_sync": {
                    "handler": handler("nanlm_learning_sync"),
                },
                "nanlm_memory_merge": {
                    "handler": handler("nanlm_memory_merge"),
                },
            }

        return Ctx(), received

    def test_safe_surfaces_exclude_merge_and_promotion_tools(self):
        for module in (gemma_session, gemma_series):
            with self.subTest(module=module.__name__):
                self.assertNotIn("nanlm_learning_sync", module.SAFE_TOOL_SET)
                self.assertNotIn("nanlm_memory_merge", module.SAFE_TOOL_SET)
                self.assertIn("nanlm_learning_sync", module.DENIED_TOOLS)
                self.assertIn("nanlm_memory_merge", module.DENIED_TOOLS)
                for name in module.SAFE_TOOLS:
                    lowered = name.lower()
                    self.assertNotIn("merge", lowered)
                    self.assertNotIn("promot", lowered)

    def test_approved_true_is_stripped_before_promotion_capable_handlers(self):
        for module in (gemma_session, gemma_series):
            with self.subTest(module=module.__name__):
                ctx, received = self._context()
                result = module._execute(ctx, "nanlm_learning_cycle", {
                    "message": "Copper is a metal.", "approved": True,
                })
                self.assertEqual(result, {"ok": True})
                self.assertEqual(received[-1][0], "nanlm_learning_cycle")
                self.assertNotIn("approved", received[-1][1])
                result = module._execute(ctx, "nanlm_wordnet_ingest", {
                    "word": "copper", "approved": True,
                })
                self.assertEqual(result, {"ok": True})
                self.assertEqual(received[-1][0], "nanlm_wordnet_ingest")
                self.assertNotIn("approved", received[-1][1])

    def test_denied_tools_never_reach_a_handler(self):
        for module in (gemma_session, gemma_series):
            with self.subTest(module=module.__name__):
                ctx, received = self._context()
                for name in ("nanlm_learning_sync", "nanlm_memory_merge"):
                    result = module._execute(ctx, name, {"approved": True})
                    self.assertIn("error", result)
                self.assertEqual(received, [])

    def test_tool_schemas_hide_the_approval_argument(self):
        for module in (gemma_session, gemma_series):
            with self.subTest(module=module.__name__):
                ctx, _received = self._context()
                schemas = module._tool_schemas(ctx)
                for schema in schemas:
                    function = schema["function"]
                    properties = function["parameters"]["properties"]
                    self.assertNotIn("approved", properties)
                    self.assertNotIn("approved", function["parameters"].get(
                        "required", []))


if __name__ == "__main__":
    unittest.main()
