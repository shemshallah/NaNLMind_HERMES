import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.scratch import Scratch, RevisionConflict, ScratchError, ZERO
from nanlmind_hermes.evidence import build_evidence_package
from nanlmind_hermes.clocked_mode import make_task_plan, validate_task_plan, propose_mode
from nanlmind_hermes import clocked_tools
from unittest.mock import patch


class AutonomousModeTests(unittest.TestCase):
    def test_llm_selects_policy_constrained_mode_without_receiving_raw_sources(self):
        class Provider:
            def __init__(self):
                self.calls = []
            def complete(self, **kwargs):
                self.calls.append(kwargs)
                from types import SimpleNamespace
                return SimpleNamespace(text='{"mode":"CLOCKED_COMPARE","reason":"The user asks to reconcile multiple sources."}')
        class Context:
            def __init__(self):
                self.llm = Provider()
        ctx = Context()
        sources = [{"id": "source-a", "text": "private text A"},
                   {"id": "source-b", "text": "private text B"}]
        result = clocked_tools.choose_clocked_mode(ctx, "Compare these sources", sources)
        self.assertEqual(result["mode"], "CLOCKED_COMPARE")
        self.assertEqual(result["selected_endpoints"], ["extract", "grounding", "compare"])
        self.assertEqual(result["decision_method"], "llm_policy_constrained")
        sent = ctx.llm.calls[0]["messages"][1]["content"]
        self.assertNotIn("private text A", sent)
        self.assertNotIn("private text B", sent)

    def test_invalid_llm_mode_falls_back_to_deterministic_policy(self):
        class Provider:
            def complete(self, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(text='{"mode":"arbitrary_endpoint","reason":"bad"}')
        class Context:
            llm = Provider()
        result = clocked_tools.choose_clocked_mode(
            Context(), "Compare these sources",
            [{"text": "A"}, {"text": "B"}])
        self.assertEqual(result["mode"], "CLOCKED_COMPARE")
        self.assertEqual(result["decision_method"], "deterministic_fallback")


class TaskPlanValidationTests(unittest.TestCase):
    def plan(self, tasks=None):
        return {
            "schema": "nanlm.clocked-task-plan/1",
            "version": 1,
            "plan_id": "plan-123456",
            "tasks": tasks if tasks is not None else [
                {"id": "task-a", "endpoint": "extract", "text": "source text",
                 "depends_on": [], "options": {}}
            ],
        }

    def test_rejects_self_and_forward_dependencies(self):
        with self.assertRaisesRegex(ValueError, "earlier tasks"):
            validate_task_plan(self.plan([
                {"id": "task-a", "endpoint": "extract", "text": "A",
                 "depends_on": ["task-a"]}
            ]))
        with self.assertRaisesRegex(ValueError, "earlier tasks"):
            validate_task_plan(self.plan([
                {"id": "task-a", "endpoint": "extract", "text": "A",
                 "depends_on": ["task-b"]}
            ]))

    def test_rejects_duplicate_or_non_string_dependencies(self):
        tasks = [
            {"id": "task-a", "endpoint": "extract", "text": "A", "depends_on": []},
            {"id": "task-b", "endpoint": "grounding", "text": "B",
             "depends_on": ["task-a", "task-a"]},
        ]
        with self.assertRaisesRegex(ValueError, "uniquely reference"):
            validate_task_plan(self.plan(tasks))
        tasks[1]["depends_on"] = [{"id": "task-a"}]
        with self.assertRaisesRegex(ValueError, "uniquely reference"):
            validate_task_plan(self.plan(tasks))

    def test_rejects_invalid_versions_options_and_budget_types(self):
        plan = self.plan()
        plan["version"] = True
        with self.assertRaisesRegex(ValueError, "version"):
            validate_task_plan(plan)
        plan = self.plan()
        plan["tasks"][0]["options"] = {"large": "x" * 9000}
        with self.assertRaisesRegex(ValueError, "options"):
            validate_task_plan(plan)
        with self.assertRaisesRegex(ValueError, "max_requests"):
            validate_task_plan(self.plan(), {"max_requests": True})



    def test_rejects_serialized_envelope_over_gateway_limit(self):
        tasks = [
            {"id": "task-%02d" % index, "endpoint": "extract",
             "text": "x" * 65536, "depends_on": []}
            for index in range(16)
        ]
        with self.assertRaisesRegex(ValueError, "serialized CLOCKED request"):
            validate_task_plan(self.plan(tasks))


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

    def test_clocked_scratch_builds_weak_supervision_candidate_without_raw_pii(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Context:
                def get_config(self, key, default=None):
                    return tmp if key == "state_dir" else default
            ctx = Context()
            scratch = Scratch(Path(tmp) / "clocked_scratch", "learn-turn")
            scratch.append("clocked_result", "clocked", {
                "task_id": "extract-a", "endpoint": "extract", "source_id": "source-a",
                "status": "ok", "provenance": {"input_sha256": "a" * 64},
                "result": {"fields": {"claims": ["The river flows."]}},
            })
            scratch.append("clocked_result", "clocked", {
                "task_id": "grounding-a", "endpoint": "grounding", "source_id": "source-a",
                "status": "ok", "provenance": {"input_sha256": "b" * 64},
                "result": {"fields": {"verdict": "supported"}},
            })
            scratch.append("clocked_result", "clocked", {
                "task_id": "pii-a", "endpoint": "pii", "source_id": "source-a",
                "status": "ok", "provenance": {"input_sha256": "c" * 64},
                "result": {"fields": {"detected": True, "values": ["PRIVATE-SSN-DO-NOT-LEARN"]}},
            })
            scratch.append("head_response", "nanlmind", {
                "provenance": {"state_digest": "d" * 64},
            })
            with patch.dict("os.environ", {
                "NANLM_HEAD_CORTEX_URL": "https://nanlmind.example",
                "NANLM_DEVICE_TOKEN": "scoped-training-token",
            }), patch("nanlmind_hermes.clocked_tools.HeadCortexClient") as client:
                client.return_value.create_learning_candidate.return_value = {
                    "schema": "nanlm.head-learning-candidate-response/1",
                    "status": "candidate_only", "teacher_outputs_are_ground_truth": False,
                    "promotion_allowed": False, "candidate_digest": "e" * 64,
                    "canonical_state_modified": False,
                }
                result = clocked_tools.head_learn_candidate(
                    ctx, "learned-heads-1", None, True, "learn-turn", 1)
            request = client.return_value.create_learning_candidate.call_args.args[0]
            self.assertEqual(request["base_topology_digest"], "d" * 64)
            self.assertEqual(request["split"], "train")
            self.assertEqual(request["teacher"], "CLOCKED")
            self.assertFalse(request["teacher_outputs_are_ground_truth"])
            self.assertGreaterEqual(len(request["records"]), 5)
            self.assertFalse(any("PRIVATE-SSN-DO-NOT-LEARN" in row["object"]
                                 for row in request["records"]))
            self.assertEqual(result["scratch"]["entries"][-1]["lane"], "learning_proposal")
        with self.assertRaises(PermissionError):
            clocked_tools.head_learn_candidate(None, "candidate", "d" * 64, False)

    def test_head_candidate_evaluation_requires_approval_and_records_metrics(self):
        records = [{"resolution": 5, "polarity": "+", "subject": "s",
                    "predicate": "p", "object": "o", "source_digest": "a" * 64}]
        with self.assertRaises(PermissionError):
            clocked_tools.head_candidate_evaluate(
                None, "candidate-1", records, False)
        with patch.dict("os.environ", {
            "NANLM_HEAD_CORTEX_URL": "https://nanlmind.example",
            "NANLM_DEVICE_TOKEN": "scoped-training-token",
        }), patch("nanlmind_hermes.clocked_tools.HeadCortexClient") as client:
            client.return_value.evaluate_candidate.return_value = {
                "schema": "nanlm.head-candidate-evaluation-response/1",
                "status": "evaluated_candidate", "utility_delta": 0.25,
                "recommendation": "review_candidate",
                "promotion_allowed": False, "canonical_state_modified": False,
                "baseline": {"accuracy": 0.5}, "candidate": {"accuracy": 0.75},
            }
            with tempfile.TemporaryDirectory() as tmp:
                class Context:
                    def get_config(self, key, default=None):
                        return tmp if key == "state_dir" else default
                result = clocked_tools.head_candidate_evaluate(
                    Context(), "candidate-1", records, True, "candidate-eval")
        self.assertEqual(result["evaluation"]["recommendation"], "review_candidate")
        self.assertFalse(result["evaluation"]["promotion_allowed"])
        self.assertEqual(result["scratch"]["entries"][-1]["lane"], "learning_proposal")
        sent = client.return_value.evaluate_candidate.call_args.args[0]
        self.assertEqual(sent["split"], "test")
        self.assertEqual(sent["dataset_sha256"], __import__("hashlib").sha256(
            __import__("json").dumps(records, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest())

    def test_candidate_evaluation_records_metrics_without_promotion(self):
        class Context:
            def __init__(self, root):
                self.root = root
            def get_config(self, key, default=None):
                return str(self.root) if key == "state_dir" else default
        records = [{
            "resolution": 5, "polarity": "+", "subject": "s",
            "predicate": "p", "object": "o", "source_digest": "a" * 64,
        }]
        evaluation = {
            "schema": "nanlm.head-candidate-evaluation-response/1",
            "status": "evaluated_candidate", "utility_delta": 0.1,
            "recommendation": "review_candidate", "promotion_allowed": False,
            "canonical_state_modified": False,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {
            "NANLM_HEAD_CORTEX_URL": "https://nanlmind.example",
            "NANLM_DEVICE_TOKEN": "scoped-training-token",
        }), patch("nanlmind_hermes.clocked_tools.HeadCortexClient") as client:
            client.return_value.evaluate_candidate.return_value = evaluation
            ctx = Context(Path(tmp))
            with self.assertRaises(PermissionError):
                clocked_tools.head_candidate_evaluate(ctx, "candidate-1", records, False)
            result = clocked_tools.head_candidate_evaluate(
                ctx, "candidate-1", records, True, "candidate-review")
        self.assertEqual(result["evaluation"]["recommendation"], "review_candidate")
        self.assertFalse(result["evaluation"]["promotion_allowed"])
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
