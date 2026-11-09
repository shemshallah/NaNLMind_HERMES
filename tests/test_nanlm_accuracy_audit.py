import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import nanlm_accuracy_audit as audit


def _row(question, expected, hit, response=None, structured=None):
    return {
        "question": question,
        "expected": list(expected),
        "response": response if response is not None
        else (expected[0] if hit else "unrelated words"),
        "hit": bool(hit),
        "structured": structured,
    }


def _record(vectors, responses=None, structured=None):
    """Build a synthetic artifact with one probe per position."""
    if responses is None:
        responses = [None] * len(vectors)
    if structured is None:
        structured = [None] * len(vectors)
    width = len(vectors[0])
    probes = [
        {"question": "q%d?" % index, "expected": ["term%d" % index]}
        for index in range(width)
    ]
    rounds = []
    for round_index, vector in enumerate(vectors, start=1):
        rows = []
        for index, bit in enumerate(vector):
            rows.append(_row(
                probes[index]["question"],
                probes[index]["expected"],
                bit == "1",
                responses[index] if index < len(responses) else None,
                structured[index] if index < len(structured) else None,
            ))
        rounds.append({
            "round": round_index,
            "hit_rate": vector.count("1") / width,
            "probes": rows,
        })
    baseline_rows = []
    for index in range(width):
        baseline_rows.append(_row(
            probes[index]["question"], probes[index]["expected"], False,
            "unrelated words",
            structured[index] if index < len(structured) else None,
        ))
    return {
        "schema": "nanlm.gemma-learning-over-time/1",
        "state_dir": "/tmp/synthetic-nanlm",
        "probes": probes,
        "baseline": {
            "hit_rate": 0.0,
            "probes": baseline_rows,
        },
        "rounds": rounds,
        "rounds_requested": len(vectors),
    }


class VectorTests(unittest.TestCase):
    def test_recomputes_vectors_and_integer_forms(self):
        vectors = ["000000", "101100", "011100", "111000", "110110"]
        report = audit.build_report(_record(vectors))
        # Synthetic rounds are numbered 1..5; baseline is stage 0.
        stages = report["stages"]
        self.assertEqual(stages[0]["stage"], "baseline")
        self.assertEqual(stages[1]["recorded"]["vector"], "000000")
        self.assertEqual(stages[2]["recorded"]["vector"], "101100")
        self.assertEqual(stages[2]["recorded"]["integer"], int("101100", 2))
        self.assertEqual(stages[2]["denominator"], 6)
        self.assertEqual(stages[2]["recorded"]["hits"], 3)
        self.assertTrue(stages[2]["recorded"]["matches_recomputed_keyword"])

    def test_recorded_hit_mismatch_is_visible(self):
        record = _record(["10"])
        record["rounds"][0]["probes"][0]["hit"] = False
        report = audit.build_report(record)
        stage = report["stages"][1]
        self.assertEqual(stage["recorded"]["vector"], "00")
        self.assertEqual(stage["metrics"]["keyword_hit"]["vector"], "10")
        self.assertFalse(stage["recorded"]["matches_recomputed_keyword"])


class FlipTests(unittest.TestCase):
    def setUp(self):
        # Round 3 -> 4: same count, dogs/cats-style swap at indices 0 and 1.
        # Round 4 -> 5: two hits added, one removed (three flips, net +1).
        # Round 5 -> 6: two hits added, one removed (three flips, net +1).
        self.vectors = [
            "000000",  # round 1
            "101100",  # round 2
            "101100",  # round 3
            "011100",  # round 4: same count as round 3, different vector
            "111000",  # round 5
            "110110",  # round 6: three flips, net +1
        ]
        self.report = audit.build_report(_record(self.vectors))

    def test_count_equal_but_vector_changed(self):
        transition = audit._find_transition(self.report["transitions"], 3, 4)
        self.assertIsNotNone(transition)
        self.assertTrue(transition["same_count"])
        self.assertFalse(transition["same_vector"])
        self.assertTrue(transition["same_count_but_vector_changed"])
        self.assertEqual(transition["flip_count"], 2)
        self.assertEqual(transition["net_delta"], 0)
        self.assertEqual(transition["xor"], "110000")
        self.assertEqual(transition["xor_integer"], int("110000", 2))
        self.assertEqual(transition["hamming_distance"], 2)
        self.assertEqual(
            transition["flipped_questions"], ["q0?", "q1?"])
        self.assertEqual(
            [item["status"] for item in transition["flips"]],
            ["hit->miss", "miss->hit"])

    def test_no_left_shift_and_net_plus_one_from_three_flips(self):
        transition = audit._find_transition(self.report["transitions"], 5, 6)
        self.assertIsNotNone(transition)
        self.assertEqual(transition["flip_count"], 3)
        self.assertEqual(transition["miss_to_hit"], 2)
        self.assertEqual(transition["hit_to_miss"], 1)
        self.assertEqual(transition["net_delta"], 1)
        self.assertEqual(transition["net_delta_from_flips"], 1)
        self.assertEqual(transition["xor"], "001110")
        self.assertEqual(
            [item["index"] for item in transition["flips"]], [2, 3, 4])
        self.assertNotEqual(
            audit._left_shift(transition["from_vector"]),
            transition["to_vector"])
        self.assertEqual(
            [item["status"] for item in transition["flips"]],
            ["hit->miss", "miss->hit", "miss->hit"])

    def test_transition_records_same_and_flip_statuses(self):
        transition = audit._find_transition(self.report["transitions"], 3, 4)
        self.assertEqual(len(transition["probes"]), 6)
        self.assertEqual(
            [item["status"] for item in transition["probes"]],
            ["hit->miss", "miss->hit", "same", "same", "same", "same"])
        self.assertEqual(
            [item["status"] for item in transition["flips"]],
            ["hit->miss", "miss->hit"])

    def test_observation_objects_report_verdicts(self):
        count_obs = self.report["observations"]["round_3_to_4"]
        shift_obs = self.report["observations"]["round_5_to_6"]
        self.assertTrue(count_obs["same_hit_count"])
        self.assertFalse(count_obs["same_hit_vector"])
        self.assertTrue(count_obs["vector_changed"])
        self.assertEqual(count_obs["hit_count_before"], 3)
        self.assertEqual(count_obs["hit_count_after"], 3)
        self.assertTrue(shift_obs["available"])
        self.assertEqual(shift_obs["net_delta"], 1)
        self.assertEqual(shift_obs["flip_count"], 3)
        self.assertFalse(shift_obs["left_shift_matches_to_vector"])
        self.assertEqual(
            shift_obs["left_shift_of_from_vector"], "110000")


class MetricTests(unittest.TestCase):
    def test_keyword_hit_accepts_incidental_clause(self):
        response = (
            "Yellow mix to make purple comes from red and blue create green.")
        self.assertTrue(audit.keyword_hit(response, ["green"]))
        self.assertFalse(audit.direct_exact_hit(response, ["green"]))

    def test_keyword_hit_uses_substring_not_word_boundary(self):
        self.assertTrue(audit.keyword_hit("That is nice.", ["ice"]))
        self.assertFalse(audit.direct_exact_hit("That is nice.", ["ice"]))

    def test_direct_hit_accepts_clean_answer(self):
        self.assertTrue(audit.direct_exact_hit(
            "Yellow and blue make green.", ["green"]))
        self.assertTrue(audit.direct_exact_hit(
            "A square has four sides.", ["square"]))
        self.assertTrue(audit.direct_exact_hit(
            "Water.", ["water"]))
        self.assertTrue(audit.direct_exact_hit(
            "Ice is water that froze due to cold.", ["frozen", "water"]))

    def test_direct_hit_rejects_unrelated_first_clause(self):
        self.assertFalse(audit.direct_exact_hit(
            "Water that froze due to cold turns water into ice. "
            "Happens blue and yellow mix to make purple.",
            ["ice", "frozen"]))
        self.assertFalse(audit.direct_exact_hit(
            "Ice water has the sky possesses a blue color.", ["water"]))

    def test_structured_hit_requires_flag_and_keyword(self):
        response = "Yellow and blue make green."
        self.assertTrue(audit.keyword_hit(response, ["green"]))
        self.assertTrue(audit.direct_exact_hit(response, ["green"]))

        def one_stage(structured):
            row = _row("clean?", ["green"], True, response,
                       structured=structured)
            return {
                "probes": [{"question": "clean?", "expected": ["green"]}],
                "baseline": {"hit_rate": 0.0, "probes": [
                    _row("clean?", ["green"], False, "unrelated words")]},
                "rounds": [{"round": 1, "hit_rate": 1.0, "probes": [row]}],
            }

        # Absence of the structured flag is not a structured hit.
        no_flag_report = audit.build_report(one_stage(None))
        self.assertEqual(
            no_flag_report["stages"][1]["metrics"]["structured_hit"]["hits"], 0)
        # A true flag on the same text is a structured hit.
        flagged_report = audit.build_report(one_stage(True))
        self.assertEqual(
            flagged_report["stages"][1]["metrics"]["structured_hit"]["hits"], 1)

    def test_corrected_vectors_and_denominator(self):
        probes = [
            {"question": "yellow?", "expected": ["green"]},
            {"question": "clean?", "expected": ["green"]},
            {"question": "unused 3?", "expected": ["term3"]},
            {"question": "unused 4?", "expected": ["term4"]},
            {"question": "unused 5?", "expected": ["term5"]},
            {"question": "unused 6?", "expected": ["term6"]},
        ]
        rows = [
            _row("yellow?", ["green"], True,
                 "Yellow mix to make purple comes from red and blue create green."),
            _row("clean?", ["green"], True,
                 "Yellow and blue make green.", structured=False),
            _row("unused 3?", ["term3"], False, "unrelated words"),
            _row("unused 4?", ["term4"], False, "unrelated words"),
            _row("unused 5?", ["term5"], False, "unrelated words"),
            _row("unused 6?", ["term6"], False, "unrelated words"),
        ]
        record = {
            "schema": "nanlm.gemma-learning-over-time/1",
            "probes": probes,
            "baseline": {"hit_rate": 0.0, "probes": [
                _row(item["question"], item["expected"], False, "unrelated words")
                for item in probes
            ]},
            "rounds": [{"round": 1, "hit_rate": 2 / 6, "probes": rows}],
        }
        report = audit.build_report(record)
        stage = report["stages"][1]
        self.assertEqual(stage["metrics"]["keyword_hit"]["vector"], "110000")
        self.assertEqual(stage["metrics"]["structured_hit"]["vector"], "000000")
        self.assertEqual(
            stage["metrics"]["direct_exact_hit"]["vector"], "010000")
        self.assertEqual(stage["denominator"], 6)

    def test_missing_structured_values_are_not_measured_misses(self):
        record = _record(["100000"])
        report = audit.build_report(record)
        stage = report["stages"][1]
        self.assertEqual(stage["structured_measured"], 0)
        self.assertEqual(stage["structured_missing"], 6)
        self.assertEqual(stage["metrics"]["structured_hit"]["hits"], 0)
        defect = next(
            item for item in report["measurement_defects"]
            if item["id"] == "implicit_denominator_and_missing_structured_values")
        self.assertEqual(defect["evidence"]["structured_missing_rows"], 12)


class RenderTests(unittest.TestCase):
    def test_markdown_contains_observations_and_metrics(self):
        report = audit.build_report(_record([
            "000000", "101100", "101100", "011100", "111000", "110110",
        ]))
        markdown = audit.render_markdown(report)
        self.assertIn("Observation 1: rounds 3 -> 4", markdown)
        self.assertIn("Observation 2: rounds 5 -> 6", markdown)
        self.assertIn("same vector: False", markdown)
        self.assertIn("direct_exact_hit", markdown)
        self.assertIn("not the earlier vector shifted by one bit", markdown)

    def test_summary_is_bounded(self):
        report = audit.build_report(_record([
            "000000", "101100", "101100", "011100", "111000", "110110",
        ]))
        lines = audit.summary_lines(report)
        text = audit._bounded(lines, max_lines=5, max_chars=300)
        self.assertIn("summary truncated", text)
        self.assertLessEqual(len(text.splitlines()), 6)


class CliTests(unittest.TestCase):
    def test_cli_writes_json_and_markdown(self):
        record = _record([
            "000000", "101100", "101100", "011100", "111000", "110110",
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "run.json"
            source.write_text(json.dumps(record), encoding="utf-8")
            json_path = root / "audit.json"
            markdown_path = root / "audit.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = audit.main([
                    "--input", str(source),
                    "--json", str(json_path),
                    "--markdown", str(markdown_path),
                ])
            self.assertEqual(code, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], audit.REPORT_SCHEMA)
            self.assertTrue(markdown_path.read_text(encoding="utf-8").startswith(
                "# NaNLM accuracy audit"))
            self.assertIn("observation 3->4", stdout.getvalue())

    def test_cli_summary_respects_bound(self):
        record = _record([
            "000000", "101100", "101100", "011100", "111000", "110110",
        ])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "run.json"
            source.write_text(json.dumps(record), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = audit.main(["--input", str(source)])
            self.assertEqual(code, 0)
            lines = stdout.getvalue().splitlines()
            self.assertLessEqual(len(lines), audit.MAX_SUMMARY_LINES + 1)


if __name__ == "__main__":
    unittest.main()
