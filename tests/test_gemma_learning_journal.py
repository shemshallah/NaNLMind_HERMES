import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import gemma_learning_journal as journal
from scripts import gemma_series_diagnostics as diagnostics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "GEMMA_TRAINING_SERIES_RESULTS.json"


def _call(name, arguments=None, *, failed=False, invalid=False, turn=0,
          result=None, result_summary=None):
    return {
        "turn": turn,
        "name": name,
        "arguments": arguments or {},
        "failed": failed,
        "invalid_arguments": invalid,
        "argument_problems": (["missing required argument: leaf_id"]
                              if invalid else []),
        "result": result,
        "result_summary": result_summary,
    }


def _series_result():
    return {
        "schema": "nanlm.gemma-training-series/1",
        "system_learning": {
            "before": {"metrics": {
                "turns": 0, "rows": 0, "scratch_records": 0,
                "concept_statements": 0, "concept_concepts": 0,
                "concept_occupied_pairs": 0, "linked_kernels": 91,
                "expected_kernels": 91, "leaves": 144,
            }},
            "after": {"metrics": {
                "turns": 2, "rows": 1, "scratch_records": 20,
                "scratch_pages_allocated": 20, "concept_statements": 3,
                "concept_concepts": 34, "concept_occupied_pairs": 436,
                "concept_density": 0.013358, "linked_kernels": 91,
                "expected_kernels": 91, "images_verified": 91,
                "leaves": 148,
            }},
        },
        "sessions": [
            {"session_index": 1, "stage": 1, "tool_calls": [
                _call("nanlm_mth_tree", result={
                    "linked": 91, "expected": 91, "images_verified": 91,
                    "complete": True, "image_failures": []}),
                _call("nanlm_mth_verify", result={
                    "execution_mode": "native_nan0s", "n16i_used": False,
                    "marker_found": True,
                    "kernel_id": "mth.k256.laplacian.u16"}),
            ]},
            {"session_index": 2, "stage": 2, "tool_calls": [
                _call("nanlm_mth_pair", result={
                    "leaf": {"function_definition": {"execution": {
                        "mode": "n16i_guest_interpreter",
                        "n16i_used": True}}}}),
                _call("nanlm_mth_leaf_verify", result={
                    "nan_executable": True}),
            ]},
            {"session_index": 3, "stage": 3, "tool_calls": [
                _call("nanlm_learning_status", result={
                    "turns": 2, "rows": 1,
                    "scratch": {"records": 20, "allocated_pages": 20},
                    "concept_saturation": {
                        "statements": 3, "concepts": 34,
                        "occupied_pairs": 436, "density": 0.013358}}),
            ]},
        ],
    }


class SessionJournalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "state"

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_load_and_tail_are_bounded(self):
        store = journal.SessionJournal(self.root, max_entries=3)
        for index in range(1, 6):
            store.append_session(
                index, ((index - 1) % 3) + 1,
                [_call("nanlm_status", turn=0)],
                {"tool_calls": 1, "successful_calls": 1,
                 "failed_calls": 0, "invalid_argument_calls": 0,
                 "turns_used": 1})
        entries = store.load()
        self.assertEqual([entry["session_index"] for entry in entries],
                         [3, 4, 5])
        self.assertEqual(
            [entry["session_index"] for entry in store.tail(2)], [4, 5])
        self.assertEqual(store.tail(0), [])
        self.assertEqual(
            [entry["session_index"] for entry in store.tail(100)], [3, 4, 5])
        lines = store.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertLessEqual(len(line), journal.MAX_LINE_CHARS)

    def test_large_payloads_and_secrets_are_redacted(self):
        long_text = "A" * 20_000
        secret = "Bearer abcdefghijklmnop"
        store = journal.SessionJournal(self.root)
        entry = store.append_session(
            1, 1,
            [_call(
                "nanlm_mth_leaf_create",
                result_summary=json.dumps({
                    "payload": long_text,
                    "execution_mode": "n16i_guest_interpreter",
                    "n16i_used": True,
                }) + secret)],
            {"tool_calls": 1, "successful_calls": 1, "failed_calls": 0,
             "invalid_argument_calls": 0, "turns_used": 1},
            {
                "execution_mode": "n16i_guest_interpreter",
                "n16i_used": True,
                "leaf_count": 148,
                "note": long_text,
                "api_key": "sk-abcdefghijk",
                "token": "hunter2",
                "nested": {"payload": long_text},
            })
        raw = store.path.read_text(encoding="utf-8")
        self.assertNotIn("A" * 1000, raw)
        self.assertNotIn("hunter2", raw)
        self.assertNotIn("sk-abcdefghijk", raw)
        self.assertNotIn(secret, raw)
        self.assertLess(len(raw), 20_000)
        self.assertIn("nanlm_mth_leaf_create", entry["tools_used"])
        observations = entry["observations"]
        self.assertEqual(observations["execution_mode"],
                         "n16i_guest_interpreter")
        self.assertTrue(observations["n16i_used"])
        self.assertEqual(observations["leaf_count"], 148)
        self.assertEqual(observations["api_key"], "[redacted]")
        self.assertEqual(observations["token"], "[redacted]")
        self.assertLessEqual(len(observations["note"]),
                             journal.MAX_TEXT_CHARS + len("...[truncated]"))

    def test_experience_block_has_exact_tools_and_observations(self):
        store = journal.SessionJournal(self.root)
        store.append_session(
            1, 1,
            [_call("nanlm_mth_tree"), _call("nanlm_mth_verify")],
            {"tool_calls": 2, "successful_calls": 2, "failed_calls": 0,
             "invalid_argument_calls": 0, "turns_used": 2},
            {"execution_mode": "native_nan0s", "images_verified": 91,
             "linked_kernels": 91})
        store.append_session(
            2, 2,
            [_call("nanlm_mth_leaf_create"), _call("nanlm_mth_pair")],
            {"tool_calls": 2, "successful_calls": 2, "failed_calls": 0,
             "invalid_argument_calls": 0, "turns_used": 2},
            {"leaf_execution_mode": "n16i_guest_interpreter",
             "leaf_n16i_used": True, "leaves": 148})
        block = store.build_experience_block(limit=3)
        self.assertIn("nanlm_mth_tree", block)
        self.assertIn("nanlm_mth_verify", block)
        self.assertIn("nanlm_mth_leaf_create", block)
        self.assertIn("nanlm_mth_pair", block)
        self.assertIn("execution_mode=native_nan0s", block)
        self.assertIn("images_verified=91", block)
        self.assertIn("leaves=148", block)
        self.assertLessEqual(len(block), journal.MAX_BLOCK_CHARS)
        latest = store.build_experience_block(limit=1)
        self.assertIn("nanlm_mth_leaf_create", latest)
        self.assertNotIn("nanlm_mth_tree", latest)
        self.assertEqual(journal.SessionJournal(
            Path(self._tmp.name) / "empty").build_experience_block(), "")

    def test_extract_observations_prefers_full_results_and_reads_summaries(self):
        observations = journal.extract_observations([
            _call("nanlm_mth_tree", result={
                "linked": 91, "expected": 91, "images_verified": 91,
                "complete": True, "image_failures": []}),
            _call("nanlm_mth_verify", result={
                "execution_mode": "native_nan0s", "n16i_used": False,
                "marker_found": True}),
            _call("nanlm_mth_pair", result={
                "leaf": {"function_definition": {"execution": {
                    "mode": "n16i_guest_interpreter", "n16i_used": True}}}}),
            _call("nanlm_mth_leaf_verify", result={"nan_executable": True}),
            _call("nanlm_mth_hub", result={"leaves": 148}),
            _call("nanlm_learning_status", result={
                "turns": 2, "rows": 1,
                "scratch": {"records": 20, "allocated_pages": 20},
                "concept_saturation": {
                    "statements": 3, "concepts": 34,
                    "occupied_pairs": 436, "density": 0.013358}}),
        ])
        self.assertEqual(observations["linked_kernels"], 91)
        self.assertEqual(observations["expected_kernels"], 91)
        self.assertEqual(observations["images_verified"], 91)
        self.assertTrue(observations["tree_complete"])
        self.assertEqual(observations["execution_mode"], "native_nan0s")
        self.assertFalse(observations["n16i_used"])
        self.assertEqual(observations["leaf_execution_mode"],
                         "n16i_guest_interpreter")
        self.assertTrue(observations["leaf_n16i_used"])
        self.assertTrue(observations["nan_executable"])
        self.assertEqual(observations["leaves"], 148)
        self.assertEqual(observations["turns"], 2)
        self.assertEqual(observations["rows"], 1)
        self.assertEqual(observations["scratch_records"], 20)
        self.assertEqual(observations["concept_occupied_pairs"], 436)

        truncated = journal.extract_observations([
            _call("nanlm_mth_tree", result_summary=(
                '{"linked": 91, "expected": 91, "images_verified": 91, '
                '"complete": true')),
            _call("nanlm_mth_verify", result_summary=(
                '{"execution_mode": "native_nan0s", "marker_found": true')),
        ])
        self.assertEqual(truncated["linked_kernels"], 91)
        self.assertEqual(truncated["images_verified"], 91)
        self.assertEqual(truncated["execution_mode"], "native_nan0s")
        self.assertTrue(truncated["marker_found"])

    def test_dynamic_expected_derives_metrics_and_observations(self):
        expected = journal.dynamic_expected(_series_result())
        self.assertEqual(expected["turns"], 2)
        self.assertEqual(expected["rows"], 1)
        self.assertEqual(expected["scratch_records"], 20)
        self.assertEqual(expected["scratch_delta"], 20)
        self.assertEqual(expected["concept_concepts"], 34)
        self.assertEqual(expected["concept_occupied_pairs"], 436)
        self.assertEqual(expected["images_verified"], 91)
        self.assertEqual(expected["leaves_before"], 144)
        self.assertEqual(expected["leaves_after"], 148)
        self.assertEqual(expected["leaf_delta"], 4)
        self.assertEqual(expected["execution_mode"], "native_nan0s")
        self.assertFalse(expected["n16i_used"])
        self.assertEqual(expected["leaf_execution_mode"],
                         "n16i_guest_interpreter")
        self.assertTrue(expected["leaf_n16i_used"])
        self.assertTrue(expected["nan_executable"])

        direct = journal.dynamic_expected({
            "metrics": {"leaves": 148, "turns": 2,
                        "execution_mode": "native_nan0s",
                        "n16i_used": False}})
        self.assertEqual(direct["leaves_after"], 148)
        self.assertEqual(direct["execution_mode"], "native_nan0s")
        self.assertFalse(direct["n16i_used"])
        self.assertNotIn("leaf_delta", direct)


class DiscoveryQuizTests(unittest.TestCase):
    def test_static_questions_do_not_expose_expected_values(self):
        questions = journal.static_discovery_questions()
        self.assertGreaterEqual(len(questions), 6)
        ids = [question["id"] for question in questions]
        self.assertEqual(len(ids), len(set(ids)))
        for question in questions:
            self.assertTrue(question["prompt"])
            self.assertTrue(question["keys"])
            self.assertNotIn("expected", question)

    def test_scoring_perfect_partial_and_missing_answers(self):
        expected = journal.dynamic_expected(_series_result())
        perfect = {}
        for question in journal.static_discovery_questions():
            perfect[question["id"]] = {
                key: expected[key] for key in question["keys"]
                if key in expected
            }
        score = journal.score_discovery_answers(perfect, expected)
        self.assertEqual(score["percent"], 100.0)
        self.assertEqual(score["available_questions"],
                         len(journal.static_discovery_questions()))
        self.assertEqual(score["total_points"], score["max_points"])

        partial = json.loads(json.dumps(perfect))
        partial["leaf_growth"]["leaf_delta"] = 999
        partial["scratch_growth"] = {}
        score = journal.score_discovery_answers(
            json.dumps({"answers": partial}), expected)
        self.assertLess(score["percent"], 100.0)
        self.assertGreater(score["percent"], 0.0)
        leaf = next(item for item in score["questions"]
                    if item["id"] == "leaf_growth")
        self.assertIn("leaf_delta", leaf["missing_keys"])

        empty = journal.score_discovery_answers({}, expected)
        self.assertEqual(empty["percent"], 0.0)

    def test_question_keyed_expected_shape_is_accepted(self):
        expected = journal.dynamic_expected(_series_result())
        nested = {
            question["id"]: {
                key: expected[key] for key in question["keys"]
                if key in expected
            }
            for question in journal.static_discovery_questions()
        }
        answers = {
            question_id: values for question_id, values in nested.items()
        }
        score = journal.score_discovery_answers(answers, nested)
        self.assertEqual(score["percent"], 100.0)


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_explains_isolated_written_quiz(self):
        series = _series_result()
        series["quiz"] = {
            "before": {
                "label": "before",
                "prompt": [{"role": "user", "content": "same quiz"}],
                "answers": {"mth_tree": {"tool": "function_tree"}},
                "score": {"percent": 50.0},
            },
            "after": {
                "label": "after",
                "prompt": [{"role": "user", "content": "same quiz"}],
                "answers": {"mth_tree": {"tool": "function_tree"}},
                "score": {"percent": 50.0},
            },
            "delta_percent": 0.0,
        }
        report = diagnostics.diagnose(series, input_path="synthetic.json")
        written = report["written_quiz"]
        self.assertEqual(written["delta_percent"], 0.0)
        self.assertTrue(written["isolated_after_quiz"])
        self.assertTrue(written["answers_identical"])
        self.assertEqual(written["after_exact_tool_questions"], 0)
        self.assertFalse(written["after_prompt_has_journal_block"])
        self.assertIn("isolated", written["cause"].lower())
        discovery = report["discovery_quiz"]
        self.assertIsNone(discovery["answers"])
        self.assertFalse(discovery["model_call"])
        expected = discovery["expected"]
        self.assertEqual(expected["leaf_delta"], 4)
        self.assertEqual(expected["scratch_records"], 20)
        self.assertEqual(expected["concept_occupied_pairs"], 436)
        self.assertFalse(report["boundaries"]["model_called"])

    @unittest.skipUnless(RESULTS.exists(), "series results artifact missing")
    def test_diagnostics_handles_existing_results_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "series-diagnostic.json"
            self.assertEqual(diagnostics.main([
                "--input", str(RESULTS), "--output", str(output)]), 0)
            self.assertTrue(output.exists())
            markdown = output.with_suffix(".md")
            self.assertTrue(markdown.exists())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"],
                             diagnostics.DIAGNOSTIC_SCHEMA)
            written = report["written_quiz"]
            self.assertEqual(written["delta_percent"], 0.0)
            self.assertTrue(written["isolated_after_quiz"])
            self.assertEqual(written["after_exact_tool_questions"], 0)
            self.assertEqual(len(written["generic_aliases"]["after"]), 6)
            discovery = report["discovery_quiz"]
            expected = discovery["expected"]
            self.assertEqual(expected["leaf_delta"], 4)
            self.assertEqual(expected["scratch_records"], 20)
            self.assertEqual(expected["concept_occupied_pairs"], 436)
            self.assertEqual(expected["images_verified"], 91)
            self.assertEqual(expected["execution_mode"], "native_nan0s")
            self.assertIsNone(discovery["answers"])
            self.assertFalse(discovery["model_call"])
            self.assertIn("Discovery quiz", markdown.read_text(
                encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
