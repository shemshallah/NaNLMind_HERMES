"""Live plugin-tool tests for the resident universe integration."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class _FakeLLM:
    def complete(self, **kwargs):
        return SimpleNamespace(text="The river carries water.",
                               provider="resident-tools-fake",
                               model="resident-tools")


class _Context:
    def __init__(self, root: Path):
        self.root = root
        self.tools = {}
        self.llm = _FakeLLM()

    def get_config(self, key, default=None):
        if key == "state_dir":
            return str(self.root)
        if key == "resident_size_bytes":
            return 16 * 1024 * 1024
        if key in {"chat_mode", "chat_learning", "resident_learning_enabled"}:
            return "provider" if key == "chat_mode" else True
        return default if default is not None else ""

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


class ResidentToolTests(unittest.TestCase):
    def setUp(self):
        import nanlmind_hermes
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ctx = _Context(Path(self._tmp.name) / "state")
        nanlmind_hermes.register(self.ctx)
        self._call = lambda name, params=None: json.loads(
            self.ctx.tools[name](params or {}))

    def test_resident_tools_are_registered_and_read_only_status(self):
        for name in ("nanlm_resident_status", "nanlm_resident_ingest",
                     "nanlm_resident_query", "nanlm_resident_evidence"):
            self.assertIn(name, self.ctx.tools)
        status = self._call("nanlm_resident_status")
        self.assertFalse(status["opened"])
        self.assertFalse(status["canonical_merge"]["enabled"])

    def test_resident_evidence_requires_approval_and_preserves_four_states(self):
        with self.assertRaises(RuntimeError):
            self._call("nanlm_resident_evidence", {
                "action": "tell", "statement": "quokka is a kind of animal"})
        first = self._call("nanlm_resident_evidence", {
            "action": "tell", "approved": True,
            "statement": "quokka is a kind of animal"})
        self.assertTrue(first["opened"])
        self.assertEqual(first["polarity"], "supports")
        second = self._call("nanlm_resident_evidence", {
            "action": "tell", "approved": True,
            "statement": "quokka is not a kind of animal"})
        self.assertEqual(second["polarity"], "refutes")
        answer = self._call("nanlm_resident_evidence", {
            "action": "ask", "question": "is quokka a kind of animal?"})
        self.assertEqual(answer["status"], "both")
        state = self._call("nanlm_resident_evidence", {
            "action": "state", "subject": "quokka",
            "predicate": "is a kind of", "object": "animal"})
        self.assertEqual(state["status"], "both")
        snapshot = self._call("nanlm_resident_evidence", {"action": "save"})
        self.assertEqual(len(snapshot["records"]), 2)
        deleted = self._call("nanlm_resident_evidence", {
            "action": "delete", "approved": True,
            "subject": "quokka", "predicate": "is a kind of",
            "object": "animal", "polarity": "supports",
            "expected_generation": second["lane"]["generation"]})
        self.assertEqual(deleted["deleted"], 1)
        self.assertEqual(self._call("nanlm_resident_evidence", {
            "action": "ask", "question": "is quokka a kind of animal?"
        })["status"], "refutes")
        with self.assertRaises(Exception):
            self._call("nanlm_resident_evidence", {
                "action": "delete", "approved": True,
                "subject": "quokka", "predicate": "is a kind of",
                "object": "animal", "polarity": "refutes",
                "expected_generation": second["lane"]["generation"] - 1})
        loaded = self._call("nanlm_resident_evidence", {
            "action": "load", "approved": True, "snapshot": snapshot,
            "expected_generation": deleted["generation"]})
        self.assertEqual(loaded["restored"], 1)
        self.assertEqual(self._call("nanlm_resident_evidence", {
            "action": "ask", "question": "is quokka a kind of animal?"
        })["status"], "both")
        status = self._call("nanlm_resident_evidence", {"action": "status"})
        self.assertEqual(status["supports"], 1)
        self.assertEqual(status["refutes"], 1)
        self.assertFalse(status["canonical_merge"]["enabled"])

    def test_ingest_requires_approval_and_query_returns_exact_hit(self):
        with self.assertRaises(RuntimeError):
            self._call("nanlm_resident_ingest", {
                "message": "A penguin is a flightless bird."})
        receipt = self._call("nanlm_resident_ingest", {
            "approved": True,
            "message": "A penguin is a flightless bird.",
            "head": 3,
        })
        self.assertGreaterEqual(receipt["accepted"], 1)
        self.assertFalse(receipt["canonical_merge"]["enabled"])
        result = self._call("nanlm_resident_query", {
            "text": "penguin flightless", "top_k": 4})
        self.assertTrue(result["hits"])
        self.assertTrue(any("penguin" in (hit.get("text") or "").lower()
                            for hit in result["hits"]))
        self.assertIn("lookup", result)

    def test_head_field_status_activate_propose_and_commit(self):
        status = self._call("nanlm_head_field",
                            {"action": "status", "limit": 8})
        self.assertTrue(status["opened"])
        self.assertEqual(status["field"]["head_count"], 256)
        before = status["generation"]
        with self.assertRaises(Exception):
            self._call("nanlm_head_field", {
                "action": "activate", "head": 3, "delta": 0.1})
        activated = self._call("nanlm_head_field", {
            "action": "activate", "approved": True,
            "head": 3, "delta": 0.1})
        self.assertTrue(activated["accepted"])
        self.assertGreater(activated["generation"], before)

        staged = self._call("nanlm_head_field", {
            "action": "propose", "head": 4, "delta": 0.2,
            "reason": "test"})
        self.assertTrue(staged["scratch_only"])
        self.assertEqual(staged["generation"], activated["generation"])
        committed = self._call("nanlm_head_field", {
            "action": "commit", "approved": True,
            "token": staged["token"],
            "expected_generation": staged["generation"]})
        self.assertTrue(committed["accepted"])
        self.assertGreater(committed["generation"], activated["generation"])
        query = self._call("nanlm_head_field", {
            "action": "query", "head": 4})
        self.assertEqual(query["mode"], "head")
        self.assertGreater(query["metrics"]["score"], 0)

    def test_probe_chat_does_not_mutate_but_normal_chat_learns(self):
        before = self._call("nanlm_resident_status")
        self.assertFalse(before["opened"])
        self._call("nanlm_chat", {
            "message": "A whale is a mammal.", "probe": True})
        after_probe = self._call("nanlm_resident_status")
        self.assertFalse(after_probe["opened"])

        normal = self._call("nanlm_chat", {
            "message": "A river carries water."})
        self.assertIn("resident_learning", normal)
        self.assertFalse(normal["resident_learning"].get("canonical_commit", False))
        field = normal["resident_learning"].get("field_learning", {})
        self.assertGreaterEqual(field.get("active_head_count", 0), 2)
        self.assertGreater(field.get("scratch", {}).get("records", 0), 0)
        after = self._call("nanlm_resident_status")
        self.assertTrue(after["opened"])
        self.assertGreaterEqual(after["counts"]["leaves"], 2)
        self.assertGreater(
            after.get("scratch_layers", {}).get("scratch_records", 0), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
