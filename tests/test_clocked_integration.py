import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.scratch import Scratch, RevisionConflict, ScratchError, ZERO
from nanlmind_hermes.evidence import build_evidence_package
from nanlmind_hermes.clocked_mode import make_task_plan, validate_task_plan, propose_mode
from nanlmind_hermes import clocked_tools
from unittest.mock import patch


class ScratchTests(unittest.TestCase):
    def test_hash_chain_and_expected_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Scratch(tmp, "turn-1")
            first = scratch.append("user", "user", {"text": "hello"}, ZERO)
            second = scratch.append("hermes_plan", "hermes", {"mode": "LOCAL_CHAT"},
                                    first["revision"])
            self.assertEqual(scratch.snapshot()["revision"], second["revision"])
            with self.assertRaises(RevisionConflict):
                scratch.append("final", "hermes", {"text": "stale"},
                               first["revision"])

    def test_lane_writer_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Scratch(tmp, "turn")
            with self.assertRaises(ScratchError):
                scratch.append("clocked_result", "hermes", {})
            with self.assertRaises(ScratchError):
                scratch.append("unknown", "clocked", {})

    def test_evidence_package_preserves_conflicts_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Scratch(tmp, "turn")
            scratch.append("user", "user", {"text": "Q"})
            scratch.append("clocked_result", "clocked", {"endpoint": "extract"})
            snap = scratch.snapshot()
            package = build_evidence_package(snap, "Q")
            self.assertEqual(package["truth_states"], ["TRUE", "FALSE", "UNKNOWN", "CONFLICT"])
            self.assertTrue(package["policy"]["clocked_outputs_are_not_ground_truth"])
            self.assertEqual(len(package["evidence_entries"]), 2)

    def test_multi_source_plan_is_bounded_and_dependency_ordered(self):
        plan = make_task_plan([{"id": "a", "text": "A"}, {"id": "b", "text": "B"}], "compare")
        result = validate_task_plan(plan)
        self.assertEqual(result["task_count"], 6)
        self.assertEqual(propose_mode("compare", [{"text": "A"}, {"text": "B"}])["mode"],
                         "CLOCKED_MULTI_SOURCE")

    def test_clocked_execution_requires_exact_reviewed_plan(self):
        class Context:
            def __init__(self, root):
                self.root = root
            def get_config(self, key, default=None):
                return str(self.root) if key == "state_dir" else default

        class Client:
            def execute(self, plan, idempotency_key=None):
                return {"protocol": "nanlm.clocked-task-results", "version": 1,
                        "results": [
                            {"task_id": task["id"], "source_id": task.get("source_id"),
                             "endpoint": task["endpoint"], "result_type": task["endpoint"],
                             "status": "ok",
                             "result": {"fields": {"claims": ["Claim from source"]}}
                                        if task["endpoint"] == "extract" else {"verdict": "unknown"},
                             "provenance": {"input_sha256": "a" * 64}}
                            for task in plan["tasks"]]}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            question = "What does the source claim?"
            sources = [{"id": "source-a", "text": "The river flows."}]
            preview = clocked_tools.clocked_plan(ctx, question, sources, "approved-turn")
            digest = preview["approval_digest"]
            with patch.dict("os.environ", {
                "CLOCKED_GATEWAY_URL": "https://clocked.example",
                "CLOCKED_GATEWAY_TOKEN": "scoped-participant-token",
            }), patch("nanlmind_hermes.clocked_tools.ClockedGatewayClient", return_value=Client()):
                with self.assertRaises(PermissionError):
                    clocked_tools.clocked_execute(ctx, question, sources, "approved-turn",
                                                   False, digest)
                with self.assertRaises(PermissionError):
                    clocked_tools.clocked_execute(ctx, question, sources, "approved-turn",
                                                   True, "0" * 64)
                result = clocked_tools.clocked_execute(ctx, question, sources,
                    "approved-turn", True, digest)
            self.assertEqual(result["result_count"], 3)
            self.assertGreater(result["scratch"]["entry_count"], preview["scratch"]["entry_count"])
            atoms = [row["payload"] for row in result["scratch"]["entries"]
                     if row["lane"] == "evidence_atoms" and
                     row["payload"].get("predicate") == "states_claim"]
            self.assertEqual(atoms[0]["truth_state"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
