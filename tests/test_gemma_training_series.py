import json
import tempfile
import unittest
from pathlib import Path

from scripts import gemma_training_series as series
from nanlmind_hermes import register
from nanlmind_hermes.layered_learning import reset_coordinator_cache
from nanlmind_hermes.mth_leaves import reset_cache as reset_leaf_cache


def _perfect_answers():
    answers = {}
    for question in series.QUIZ_QUESTIONS:
        keywords = " ".join(
            alternative
            for group in question["keyword_groups"]
            for alternative in group
        )
        answers[question["id"]] = {
            key: keywords for key in question["keys"]
        }
    return answers


def _call(name, arguments, *, failed=False, turn=0):
    return {
        "turn": turn,
        "name": name,
        "arguments": arguments,
        "failed": failed,
        "invalid_arguments": False,
    }


class QuizTests(unittest.TestCase):
    def test_perfect_answers_score_full_marks(self):
        score = series.score_tool_use_quiz(_perfect_answers())
        self.assertEqual(score["percent"], 100.0)
        self.assertEqual(score["total_points"], score["max_points"])
        self.assertEqual(score["answered_questions"],
                         len(series.QUIZ_QUESTIONS))

    def test_missing_keys_and_keywords_reduce_score(self):
        answers = _perfect_answers()
        first = series.QUIZ_QUESTIONS[0]["id"]
        answers[first] = {"tool": "nanlm_session_context"}
        score = series.score_tool_use_quiz(answers)
        self.assertLess(score["percent"], 100.0)
        self.assertGreater(score["percent"], 0.0)
        result = next(item for item in score["questions"]
                      if item["id"] == first)
        self.assertIn("argument", result["keys_missing"])
        self.assertTrue(result["keyword_groups_missing"])

    def test_invalid_json_scores_zero_with_parse_error(self):
        score = series.score_tool_use_quiz("this is not json")
        self.assertEqual(score["percent"], 0.0)
        self.assertIn("parse_error", score)

    def test_fenced_answers_object_parses_and_scores(self):
        text = "```json\n" + json.dumps({"answers": _perfect_answers()}) + "\n```"
        answers = series.parse_quiz_answers(text)
        score = series.score_tool_use_quiz(answers)
        self.assertEqual(score["percent"], 100.0)

    def test_public_quiz_does_not_leak_keyword_rubric(self):
        quiz = series.build_tool_use_quiz()
        for question in quiz["questions"]:
            self.assertNotIn("keyword_groups", question)


class MetricTests(unittest.TestCase):
    def test_task_coverage_and_recovery_metrics(self):
        calls = [
            _call("nanlm_session_context", {}, turn=0),
            _call("nanlm_mth_leaf_create",
                  {"kind": "vector", "payload": {"fill": 2.0},
                   "alias": "v1"}, turn=1),
            _call("nanlm_mth_leaf_create",
                  {"kind": "vector", "payload": {"fill": 3.0},
                   "alias": "v2"}, turn=2),
            _call("nanlm_mth_pair",
                  {"left": "v1", "right": "v2", "operator": "connect"},
                  turn=3),
            {
                "turn": 4,
                "name": "nanlm_mth_leaf_verify",
                "arguments": {},
                "failed": True,
                "invalid_arguments": True,
                "argument_problems": ["missing required argument: leaf_id"],
            },
            _call("nanlm_mth_leaf_verify", {"leaf_id": "v1"}, turn=5),
        ]
        metrics = series.compute_session_metrics(
            calls,
            ["session_context", "vector_leaf_one", "vector_leaf_two",
             "pair_connect", "leaf_verify"],
            turns_used=6,
        )
        self.assertEqual(metrics["tool_calls"], 6)
        self.assertEqual(metrics["failed_calls"], 1)
        self.assertEqual(metrics["invalid_argument_calls"], 1)
        self.assertEqual(metrics["required_task_coverage"], 1.0)
        self.assertEqual(metrics["recovery"]["failures"], 1)
        self.assertEqual(metrics["recovery"]["recovered"], 1)
        self.assertEqual(metrics["recovery"]["unrecovered"], 0)

    def test_extract_system_metrics_and_deltas(self):
        before = series.extract_system_metrics(
            {"turns": 1, "rows": 2, "scratch": {"records": 20},
             "concept_saturation": {"statements": 3, "concepts": 4,
                                    "occupied_pairs": 5, "density": 0.1}},
            {"mathematical_memory": {"records": 7}},
            {"linked": 91, "expected": 91, "complete": True},
            {"leaves": 12},
        )
        after = series.extract_system_metrics(
            {"turns": 4, "rows": 8, "scratch": {"records": 55},
             "concept_saturation": {"statements": 9, "concepts": 10,
                                    "occupied_pairs": 11, "density": 0.2}},
            {"mathematical_memory": {"records": 9}},
            {"linked": 91, "expected": 91, "complete": True},
            {"leaves": 18},
        )
        self.assertEqual(before["leaves"], 12)
        self.assertEqual(after["linked_kernels"], 91)
        deltas = series.system_metric_deltas(before, after)
        self.assertEqual(deltas["turns"], 3)
        self.assertEqual(deltas["rows"], 6)
        self.assertEqual(deltas["scratch_records"], 35)
        self.assertEqual(deltas["linked_kernels"], 0)
        self.assertEqual(deltas["leaves"], 6)

    def test_summarize_series_aggregates_coverage_quiz_and_system_deltas(self):
        session_calls = {
            1: [
                _call("nanlm_session_context", {}, turn=0),
                _call("nanlm_mth_tree", {"action": "map"}, turn=1),
                _call("nanlm_mth_catalog", {}, turn=2),
                _call("nanlm_mth_runtime", {}, turn=3),
            ],
            2: [
                _call("nanlm_mth_leaf_create",
                      {"kind": "vector", "payload": {"fill": 2.0}}, turn=0),
                _call("nanlm_mth_leaf_create",
                      {"kind": "vector", "payload": {"fill": 3.0}}, turn=1),
                _call("nanlm_mth_pair",
                      {"left": "v1", "right": "v2", "operator": "connect"},
                      turn=2),
                _call("nanlm_mth_pair",
                      {"left": "v1", "right": "v2", "operator": "add"},
                      turn=3),
                _call("nanlm_mth_leaf_verify", {"leaf_id": "v1"}, turn=4),
            ],
            3: [
                _call("nanlm_chat", {"message": "A river is water."}, turn=0),
                _call("nanlm_chat", {"message": "What is a river?"}, turn=1),
                _call("nanlm_learning_status", {}, turn=2),
                _call("nanlm_scratch_register", {}, turn=3),
            ],
        }
        sessions = []
        for index, calls in session_calls.items():
            stage = series.stage_for_session(index)
            sessions.append({
                "session_index": index,
                "stage": stage,
                "tool_calls": calls,
                "metrics": series.compute_session_metrics(
                    calls, series.STAGE_TASKS[stage], turns_used=len(calls)),
            })
        before_metrics = series.extract_system_metrics(
            {"turns": 0, "rows": 0, "scratch": {"records": 0},
             "concept_saturation": {"statements": 0, "concepts": 0,
                                    "occupied_pairs": 0, "density": 0.0}},
            {}, {"linked": 91, "expected": 91, "complete": True},
            {"leaves": 9})
        after_metrics = series.extract_system_metrics(
            {"turns": 3, "rows": 5, "scratch": {"records": 30},
             "concept_saturation": {"statements": 2, "concepts": 3,
                                    "occupied_pairs": 4, "density": 0.2}},
            {}, {"linked": 91, "expected": 91, "complete": True},
            {"leaves": 17})
        summary = series.summarize_series(
            sessions,
            {"metrics": before_metrics},
            {"metrics": after_metrics},
            {"score": {"percent": 50.0}},
            {"score": {"percent": 83.3}},
        )
        self.assertEqual(summary["sessions"], 3)
        self.assertEqual(summary["tool_calls"], 13)
        self.assertEqual(summary["failed_calls"], 0)
        self.assertEqual(summary["distinct_tools"], 10)
        self.assertEqual(summary["required_task_coverage"], 1.0)
        self.assertTrue(summary["all_required_tasks_met"])
        self.assertEqual(summary["stage_coverage"]["2"]["coverage"], 1.0)
        self.assertEqual(summary["quiz"]["delta_percent"], 33.3)
        self.assertEqual(summary["system"]["deltas"]["turns"], 3)
        self.assertEqual(summary["system"]["deltas"]["leaves"], 8)
        self.assertEqual(len(summary["per_session"]), 3)


class SafetyTests(unittest.TestCase):
    def test_safe_tools_exclude_merge_and_promotion_paths(self):
        self.assertNotIn("nanlm_learning_sync", series.SAFE_TOOL_SET)
        self.assertNotIn("nanlm_memory_merge", series.SAFE_TOOL_SET)
        for name in series.SAFE_TOOLS:
            lowered = name.lower()
            self.assertNotIn("merge", lowered)
            self.assertNotIn("promot", lowered)
        self.assertIn("nanlm_learning_sync", series.DENIED_TOOLS)
        self.assertIn("nanlm_memory_merge", series.DENIED_TOOLS)

    def test_execute_rejects_denied_tool_even_if_registered(self):
        class Ctx:
            def __init__(self):
                self.tools = {
                    "nanlm_memory_merge": {
                        "handler": lambda params: {"merged": True},
                    },
                    "nanlm_session_context": {
                        "handler": lambda params: {"ok": True},
                    },
                }

        ctx = Ctx()
        denied = series._execute(ctx, "nanlm_memory_merge", {})
        self.assertIn("error", denied)
        allowed = series._execute(ctx, "nanlm_session_context", {})
        self.assertEqual(allowed, {"ok": True})


class ScriptedProvider:
    def __init__(self, turns):
        self.url = "http://fake.invalid/v1/chat/completions"
        self.turns = list(turns)
        self.index = 0
        self.summary_requests = 0

    def request(self, messages, *, tools=None, temperature=0.2,
                max_tokens=1200, timeout=300):
        if tools is None:
            self.summary_requests += 1
            return {"model": "fake",
                    "choices": [{"message": {"content": "Fake final summary."}}]}
        calls = self.turns[self.index] if self.index < len(self.turns) else []
        self.index += 1
        return {"model": "fake",
                "choices": [{"message": {"content": "", "tool_calls": calls}}]}


def _tool_call(name, arguments, index):
    return {
        "id": "call-%d" % index,
        "type": "function",
        "function": {
            "name": name,
            "arguments": (arguments if isinstance(arguments, str)
                          else json.dumps(arguments)),
        },
    }


class FakeProviderSessionTests(unittest.TestCase):
    def setUp(self):
        reset_coordinator_cache()
        reset_leaf_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "state"

    def tearDown(self):
        reset_coordinator_cache()
        reset_leaf_cache()
        self._tmp.cleanup()

    def test_session_loop_records_metrics_failures_and_recovery(self):
        provider = ScriptedProvider([
            [_tool_call("nanlm_session_context", {}, 1)],
            [_tool_call("nanlm_mth_leaf_create",
                        {"kind": "vector", "payload": {"fill": 2.0},
                         "alias": "fake-v1"}, 2)],
            [_tool_call("nanlm_mth_leaf_create",
                        {"kind": "vector", "payload": {"fill": 3.0},
                         "alias": "fake-v2"}, 3)],
            [_tool_call("nanlm_mth_pair",
                        {"left": "fake-v1", "right": "fake-v2",
                         "operator": "connect", "alias": "fake-link"}, 4)],
            [_tool_call("nanlm_mth_leaf_verify", "{}", 5)],
            [_tool_call("nanlm_mth_leaf_verify", {"leaf_id": "fake-v1"}, 6)],
        ])
        ctx = series.Context(self.root, provider)
        register(ctx)
        tools = series._tool_schemas(ctx)
        record = series.run_session(
            ctx, provider, tools, session_index=2, stage=2, turns=6,
            model="fake-model", state_dir=self.root,
            required_task_ids=[
                "session_context", "vector_leaf_one", "vector_leaf_two",
                "pair_connect", "leaf_verify",
            ])

        metrics = record["metrics"]
        self.assertEqual(metrics["tool_calls"], 6)
        self.assertEqual(metrics["failed_calls"], 1)
        self.assertEqual(metrics["invalid_argument_calls"], 1)
        self.assertEqual(metrics["distinct_tools"], 4)
        self.assertEqual(metrics["required_task_coverage"], 1.0)
        self.assertEqual(metrics["recovery"]["failures"], 1)
        self.assertEqual(metrics["recovery"]["recovered"], 1)
        self.assertEqual(record["final_summary"], "Fake final summary.")
        self.assertEqual(provider.summary_requests, 1)
        for item in record["transcript"]:
            if item.get("role") == "tool":
                self.assertLessEqual(
                    len(item["content"]), series.MAX_TOOL_RESULT_CHARS + 16)


if __name__ == "__main__":
    unittest.main()
