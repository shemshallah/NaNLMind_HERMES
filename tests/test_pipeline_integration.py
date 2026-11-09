"""End-to-end integration test for the local HERMES/NANNET/MTH pipeline.

This module deliberately stays on loopback: it starts the plugin's real
stdlib NANNET pairing authority on 127.0.0.1, pairs a temporary installation
with ``universe:read`` and ``training:execute``, and drives the registered
tool handlers against a temporary state directory. It does not contact a
production NANNET/CLOCKED/Koyeb endpoint, does not call merge or promotion
tools, and does not claim a canonical commit.
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

from nanlmind_hermes import auth_tools, mth_leaves, register, session_context
from nanlmind_hermes import nan0s_runtime
from nanlmind_hermes.http_api import make_server as make_authority
from nanlmind_hermes.layered_learning import reset_coordinator_cache
from nanlmind_hermes.pairing import PairingAuthority

SCOPES = ["universe:read", "training:execute"]
REQUIRED_SESSION_DOCUMENTS = (
    "nannet-routing",
    "wordnet-hw4",
    "mth-universal-runtime",
    "mth-data-leaves",
)
LAYER_NAMES = (
    "bit", "nibble", "byte", "phonetic", "morpheme",
    "lexical", "word_completion", "sentence", "paragraph", "narrative",
)


class FakeLLM:
    """Deterministic local provider; no network calls leave the process."""

    def __init__(self):
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if kwargs.get("purpose") == "nanlmind-hermes-chat":
            return SimpleNamespace(
                text="A river is a flowing body of water.",
                provider="pipeline-integration-fake",
                model="chat",
            )
        return SimpleNamespace(
            text=json.dumps({"discoveries": [{
                "kind": "relation",
                "subject": "river",
                "predicate": "is_a",
                "object": "flowing-water-body",
                "resolution": "lexical",
                "polarity": "supports",
                "confidence": 0.9,
                "evidence": {"turns": 1},
                "provenance": {"test": "pipeline-integration"},
            }]}),
            provider="pipeline-integration-fake",
            model="reflection",
        )


class Context:
    def __init__(self, root, config):
        self.config = {"state_dir": str(root), **config}
        self.llm = FakeLLM()
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Caches are process-wide; each test gets a fresh state directory.
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

        self.ctx = Context(root / "state", {
            "chat_mode": "provider",
            "chat_learning": True,
            "learning_require_approval": True,
            "learning_auto_sync": False,
            "nannet_auth_url": self.authority_url,
            "nannet_issuer_public_key": self.authority.issuer_public_key(),
        })
        self._patcher = patch.object(
            auth_tools, "get_secret",
            side_effect=lambda name: self.secrets.get(name, ""))
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        register(self.ctx)

    def _stop_authority(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _call(self, name, params=None):
        raw = self.ctx.tools[name]["handler"](params or {})
        return json.loads(raw) if isinstance(raw, str) else raw

    def _pair(self):
        self._call("nanlm_auth_init")
        code = self.authority.create_pairing_code(
            SCOPES, ttl_seconds=600)["pairing_code"]
        paired = self._call("nanlm_auth_pair", {"pairing_code": code})
        self.assertTrue(paired["paired"])
        self.assertTrue(paired["issuer_key_pinned"])
        status = self._call("nanlm_auth_status")
        self.assertTrue(status["token_valid"])
        self.assertEqual(sorted(status["token_scopes"]), sorted(SCOPES))
        return status

    def _create_vector(self, alias, fill):
        result = self._call("nanlm_mth_leaf_create", {
            "kind": "vector",
            "payload": {"fill": fill},
            "alias": alias,
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["leaf"]["payload"]["values"]), 256)
        return result["leaf"]

    def test_status_keeps_canonical_merge_closed_while_nannet_is_paired(self):
        paired = self._pair()
        self.assertTrue(paired["paired_token_present"])

        status = self._call("nanlm_status")
        self.assertTrue(status["session_context"]["loaded"])
        self.assertFalse(status["merge_gate"]["canonical_merge_enabled"])
        self.assertFalse(status["merge_gate"]["merge_enabled"])
        self.assertFalse(status["merge_gate"]["local_auto_sync"])

    def test_session_context_documents_are_readable_without_auth(self):
        before = self._call("nanlm_auth_status")
        self.assertFalse(before["paired_token_present"])

        result = self._call("nanlm_session_context", {
            "include_text": True, "max_bytes": 16384,
        })
        self.assertTrue(result["loaded"])
        self.assertEqual(tuple(result["document_ids"]),
                         session_context.DOCUMENT_IDS)
        documents = {item["id"]: item for item in result["documents_text"]}
        self.assertEqual(set(documents), set(session_context.DOCUMENT_IDS))
        for document_id in REQUIRED_SESSION_DOCUMENTS:
            self.assertIn(document_id, documents)
            self.assertTrue(documents[document_id]["text"].strip())
            self.assertEqual(len(documents[document_id]["sha256"]), 64)
        self.assertIn("nan://", documents["nannet-routing"]["text"])
        self.assertIn("nan://GM0:SC0:MC0:BS0:0:HW4:0",
                      documents["wordnet-hw4"]["text"])
        self.assertIn("91-kernel MTH manifest",
                      documents["mth-universal-runtime"]["text"])
        self.assertIn("content-addressed immutable object",
                      documents["mth-data-leaves"]["text"])

    def test_nannet_chat_feeds_ten_layers_and_grows_learning_status(self):
        self._pair()
        before = self._call("nanlm_learning_status")
        self.assertEqual(before["turns"], 0)
        self.assertEqual(before["rows"], 0)
        self.assertEqual(before["scratch"]["records"], 0)

        answer = self._call("nanlm_chat", {
            "message": "A river is a flowing body of water.",
            "session_id": "pipeline-integration",
        })
        self.assertEqual(answer["authorization"]["read"]["mode"], "nannet")
        self.assertEqual(answer["authorization"]["learning"]["mode"], "nannet")
        self.assertFalse(answer["authorization"]["local_only"])
        receipt = answer["layered_learning"]
        self.assertEqual(receipt["layers"], 10)
        self.assertEqual(tuple(receipt["layer_names"]), LAYER_NAMES)
        self.assertFalse(receipt["canonical_commit"])
        self.assertTrue(answer["response"].strip())

        after = self._call("nanlm_learning_status")
        self.assertGreater(after["turns"], before["turns"])
        self.assertGreater(after["rows"], before["rows"])
        self.assertGreater(after["scratch"]["records"],
                           before["scratch"]["records"])
        self.assertGreater(after["scratch"]["used_bytes"],
                           before["scratch"]["used_bytes"])

    def test_mth_function_tree_maps_and_audits_all_kernel_leaves(self):
        self._pair()
        mapped = self._call("nanlm_mth_tree", {"action": "map"})
        self.assertEqual(mapped["authorization"]["mode"], "nannet")
        self.assertTrue(mapped["complete"])

        audit = self._call("nanlm_mth_tree", {"action": "audit"})
        self.assertEqual(audit["authorization"]["mode"], "nannet")
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["expected"], 91)
        self.assertEqual(audit["linked"], 91)
        self.assertEqual(audit["images_verified"], 91)
        self.assertEqual(audit["missing"], [])
        self.assertEqual(audit["image_failures"], [])

    def test_connect_pairing_creates_relation_hash_leaf_and_preserves_inputs(self):
        self._pair()
        left = self._create_vector("it-connect-v1", 1.0)
        right = self._create_vector("it-connect-v2", 2.0)
        before = self._call("nanlm_mth_hub")["leaves"]

        result = self._call("nanlm_mth_pair", {
            "left": "it-connect-v1",
            "right": "it-connect-v2",
            "operator": "connect",
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["authorization"]["mode"], "nannet")
        self.assertTrue(result["hashleaf"].startswith("leaf:"))
        self.assertEqual(result["hashleaf"], result["leaf"]["leaf_id"])
        self.assertEqual(result["leaf"]["kind"], "relation")
        self.assertEqual(result["leaf"]["function_definition"]["operator"],
                         "connect")
        self.assertEqual(result["leaf"]["payload"]["left"], left["leaf_id"])
        self.assertEqual(result["leaf"]["payload"]["right"], right["leaf_id"])
        self.assertNotIn(result["hashleaf"], {left["leaf_id"], right["leaf_id"]})
        self.assertEqual(self._call("nanlm_mth_hub")["leaves"], before + 1)

        for reference, original in (("it-connect-v1", left),
                                    ("it-connect-v2", right)):
            current = self._call("nanlm_mth_leaf_get", {"leaf_id": reference})
            self.assertEqual(current["leaf_id"], original["leaf_id"])
            self.assertEqual(current["content_sha256"],
                             original["content_sha256"])

    @unittest.skipUnless(
        nan0s_runtime.compiler_available(),
        "C compiler required for native vector add pairing")
    def test_add_pairing_creates_new_hash_leaf_and_preserves_inputs(self):
        self._pair()
        left = self._create_vector("it-add-v1", 1.0)
        right = self._create_vector("it-add-v2", 2.0)
        before = self._call("nanlm_mth_hub")["leaves"]

        result = self._call("nanlm_mth_pair", {
            "left": "it-add-v1",
            "right": "it-add-v2",
            "operator": "add",
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["authorization"]["mode"], "nannet")
        self.assertTrue(result["hashleaf"].startswith("leaf:"))
        self.assertEqual(result["hashleaf"], result["leaf"]["leaf_id"])
        self.assertNotIn(result["hashleaf"], {left["leaf_id"], right["leaf_id"]})
        self.assertEqual(result["leaf"]["kind"], "vector")
        self.assertEqual(set(result["leaf"]["payload"]["values"]), {3.0})
        self.assertEqual(result["leaf"]["function_definition"]["kernel_id"],
                         "mth.vector.add256.f32")
        self.assertEqual(self._call("nanlm_mth_hub")["leaves"], before + 1)

        for reference, original in (("it-add-v1", left), ("it-add-v2", right)):
            current = self._call("nanlm_mth_leaf_get", {"leaf_id": reference})
            self.assertEqual(current["content_sha256"],
                             original["content_sha256"])

    def test_leaf_verify_executes_the_nan_image(self):
        self._pair()
        leaf = self._create_vector("it-verify", 0.5)
        verified = self._call("nanlm_mth_leaf_verify", {
            "leaf_id": leaf["leaf_id"],
        })
        self.assertTrue(verified["nan_executable"])
        self.assertEqual(verified["output"], "L")
        self.assertEqual(verified["leaf_id"], leaf["leaf_id"])

    def test_denied_pairings_create_no_leaf(self):
        self._pair()
        self._create_vector("it-denied-v1", 1.0)
        scalar = self._call("nanlm_mth_leaf_create", {
            "kind": "scalar", "payload": {"value": 2.0}, "alias": "it-denied-s1",
        })
        self.assertEqual(scalar["status"], "ok")
        before = self._call("nanlm_mth_hub")["leaves"]

        illegal = self._call("nanlm_mth_pair", {
            "left": "it-denied-v1",
            "right": "it-denied-s1",
            "operator": "add",
        })
        self.assertEqual(illegal["status"], "denied")
        self.assertEqual(illegal["error"]["code"], "illegal_pairing")
        self.assertIsNone(illegal["hashleaf"])

        unknown = self._call("nanlm_mth_pair", {
            "left": "it-denied-v1",
            "right": "it-denied-v1",
            "operator": "not-an-operator",
        })
        self.assertEqual(unknown["status"], "denied")
        self.assertEqual(unknown["error"]["code"], "unknown_operator")
        self.assertIsNone(unknown["hashleaf"])

        self.assertEqual(self._call("nanlm_mth_hub")["leaves"], before)


if __name__ == "__main__":
    unittest.main()
