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
                         "CLOCKED_COMPARE")

    def test_autonomous_mode_selects_privacy_and_injection_audits(self):
        sources = [{"id": "a", "text": "Contact jane@example.test for details."}]
        privacy = propose_mode("Audit this for PII and privacy risks", sources)
        self.assertEqual(privacy["mode"], "CLOCKED_PRIVACY_AUDIT")
        self.assertEqual(privacy["selected_endpoints"], ["pii"])
        privacy_plan = make_task_plan(sources, "Audit this for PII and privacy risks", privacy)
        self.assertEqual([row["endpoint"] for row in privacy_plan["tasks"]], ["pii", "decide"])
        injection = propose_mode("Check for prompt injection and jailbreaks", sources)
        self.assertEqual(injection["mode"], "CLOCKED_INJECTION_AUDIT")
        injection_plan = make_task_plan(sources, "Check for prompt injection", injection)
        self.assertEqual([row["endpoint"] for row in injection_plan["tasks"]],
                         ["injection", "script", "decide"])

    def test_autonomous_mode_selects_verify_and_local_chat(self):
        sources = [{"id": "a", "text": "The report says 12."}]
        verify = propose_mode("Fact check and verify this claim", sources)
        self.assertEqual(verify["mode"], "CLOCKED_VERIFY")
        self.assertEqual(verify["selected_endpoints"], ["extract", "grounding"])
        local = propose_mode("Explain Boolean logic")
        self.assertEqual(local["mode"], "LOCAL_CHAT")
        self.assertFalse(local["requires_approval"])

    def test_head_candidate_requires_approval_and_never_promotes(self):
        with self.assertRaises(PermissionError):
            clocked_tools.head_candidate(None, "candidate-1", 42, 1, "a" * 64, False)
        with patch.dict("os.environ", {
            "NANLM_HEAD_CORTEX_URL": "https://nanlmind.example",
            "NANLM_DEVICE_TOKEN": "scoped-training-token",
        }), patch("nanlmind_hermes.clocked_tools.HeadCortexClient") as client:
            client.return_value.create_candidate.return_value = {
                "schema": "nanlm.head-candidate-response/1",
                "status": "candidate_only", "canonical_state_modified": False,
                "promotion_allowed": False,
            }
            result = clocked_tools.head_candidate(
                None, "candidate-1", 42, 1, "a" * 64, True)
        self.assertEqual(result["candidate"]["status"], "candidate_only")
        self.assertFalse(result["candidate"]["canonical_state_modified"])
        self.assertFalse(result["candidate"]["promotion_allowed"])
        self.assertEqual(result["scratch"]["entries"][-1]["lane"], "learning_proposal")

    def test_two_pass_synthesis_submits_draft_and_appends_final(self):
        class Result:
            def __init__(self, text):
                self.text = text
        class LLM:
            def __init__(self):
                self.calls = 0
            def complete(self, **kwargs):
                self.calls += 1
                return Result("draft answer" if self.calls == 1 else "final answer")
        class Context:
            def __init__(self, root):
                self.root = root
                self.llm = LLM()
            def get_config(self, key, default=None):
                return str(self.root) if key == "state_dir" else default
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(Path(tmp))
            scratch = clocked_tools._scratch(ctx, "synthesis-turn")
            scratch.append("clocked_result", "clocked", {
                "endpoint": "extract", "result_type": "extract",
                "provenance": {"input_sha256": "a" * 64},
            })
            scratch.append("evidence_atoms", "hermes", {
                "subject": "source-a", "predicate": "states_claim",
                "object": "The river flows.", "source_digest": "a" * 64,
                "resolution": 5, "truth_state": "UNKNOWN",
            })
            with patch("nanlmind_hermes.clocked_tools.head_process",
                       return_value={"package_sha256": "b" * 64,
                                     "head_response": {"truth_state": "UNKNOWN"}}) as head:
                result = clocked_tools.clocked_synthesize(
                    ctx, "What does the source say?", "synthesis-turn")
            self.assertEqual(result["response"], "final answer")
            self.assertEqual(result["draft"], "draft answer")
            head.assert_called_once()
            self.assertEqual(head.call_args.args[3], "ANSWER_VALIDATION")
            lanes = [row["lane"] for row in result["scratch"]["entries"]]
            self.assertIn("hermes_synthesis", lanes)
            self.assertIn("final", lanes)

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
