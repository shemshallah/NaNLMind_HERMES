import contextlib
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts import nanlm_plateau_analysis as plateau


def _relation(subject, predicate, obj, *, confidence=0.55,
              polarity="supports", source_sentence=None, source="native_chat_parser"):
    row = {
        "kind": "relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "polarity": polarity,
        "confidence": confidence,
        "source": source,
    }
    if source_sentence:
        row["source_sentence"] = source_sentence
    return row


def _probe(question, expected, response, hit):
    return {
        "question": question,
        "expected": list(expected),
        "response": response,
        "hit": bool(hit),
    }


class ClassificationTests(unittest.TestCase):
    def test_categories_are_deterministic(self):
        cases = [
            (
                "no_structured_intent",
                {"question": "Tell me the color of the sky.", "expected": ["blue"]},
                "Sky is colored.",
                [{"round": 1, "statement": "The sky is blue."}],
                [_relation("The sky", "is", "blue",
                           source_sentence="The sky is blue.")],
                False,
            ),
            (
                "wrong_subject_interference",
                {"question": "What do red and blue make?", "expected": ["purple"]},
                "Red and blue make green.",
                [
                    {"round": 1, "statement": "Red and blue create purple."},
                    {"round": 1, "statement": "Yellow and blue make green."},
                ],
                [],
                False,
            ),
            (
                "reverse_lookup_missing",
                {"question": "What shape has four sides?", "expected": ["square"]},
                "Shape with four sides.",
                [{"round": 1, "statement": "A square has four sides."}],
                [_relation("A square", "has", "four sides",
                           source_sentence="A square has four sides.")],
                True,
            ),
            (
                "causal_question_missing",
                {"question": "What happens when water freezes?",
                 "expected": ["ice", "frozen"]},
                "Water has the ability to flow.",
                [{"round": 1, "statement": "Frozen water is ice."}],
                [_relation("Frozen water", "is", "ice",
                           source_sentence="Frozen water is ice.")],
                True,
            ),
            (
                "vocabulary/generation",
                {"question": "What is a cat?", "expected": ["animal"]},
                "Cat.",
                [{"round": 1, "statement": "A cat is an animal."}],
                [_relation("A cat", "is", "an animal",
                           source_sentence="A cat is an animal.")],
                True,
            ),
            (
                "teacher_curriculum_missing",
                {"question": "What is a circle?", "expected": ["round"]},
                "Circle.",
                [],
                [],
                False,
            ),
        ]
        for expected_cause, probe, response, statements, rows, candidate in cases:
            with self.subTest(cause=expected_cause):
                result = plateau.classify_miss(
                    probe, response, statements=statements, relation_rows=rows)
                self.assertEqual(result["cause"], expected_cause)
                self.assertEqual(result["structured_candidate_possible"], candidate)
                self.assertIn(result["question_kind"],
                              {"other", "what_make", "reverse", "causal",
                               "what_is"})
                self.assertIn("evidence", result)

    def test_unknown_for_malformed_probe(self):
        result = plateau.classify_miss(
            {"question": "", "expected": []}, "", statements=[], relation_rows=[])
        self.assertEqual(result["cause"], "unknown")
        self.assertFalse(result["structured_candidate_possible"])

    def test_subject_object_generic_words_do_not_create_interference(self):
        probe = {"question": "What do dogs do?", "expected": ["bark"]}
        result = plateau.classify_miss(
            probe,
            "Dogs, and birds are animals.",
            statements=[{"round": 1, "statement": "Dogs are animals that bark."}],
            relation_rows=[_relation("Dogs", "is", "animals that bark",
                                     source_sentence="Dogs are animals that bark.")],
        )
        self.assertNotEqual(result["cause"], "wrong_subject_interference")
        self.assertFalse(result["sibling_evidence"])


class DatabaseReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "universe" / "mathematical_memory.sqlite3"
        self.path.parent.mkdir(parents=True)
        self._make_db()

    def _make_db(self):
        connection = sqlite3.connect(self.path)
        connection.executescript("""
            CREATE TABLE events(
                event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                source_digest TEXT NOT NULL, created_ns INTEGER NOT NULL
            );
            CREATE TABLE projections(
                projection_id TEXT PRIMARY KEY, event_id TEXT NOT NULL,
                head INTEGER NOT NULL, layer INTEGER NOT NULL,
                role TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE edges(
                edge_id TEXT PRIMARY KEY, src TEXT NOT NULL,
                relation TEXT NOT NULL, dst TEXT NOT NULL, event_id TEXT NOT NULL
            );
            CREATE TABLE hyperedges(
                hyperedge_id TEXT PRIMARY KEY, relation TEXT NOT NULL,
                members TEXT NOT NULL, event_id TEXT NOT NULL
            );
            CREATE TABLE assessments(
                assessment_id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                status TEXT NOT NULL, support_weight REAL NOT NULL,
                refute_weight REAL NOT NULL, evidence_count INTEGER NOT NULL,
                source_event_ids TEXT NOT NULL, assessor_version TEXT NOT NULL
            );
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        payload = {
            "kind": "relation", "subject": "A cat", "predicate": "is",
            "object": "an animal", "polarity": "supports", "confidence": 0.55,
            "evidence": {"sentence": "A cat is an animal."},
            "provenance": {"source_type": "native_chat_parser"},
        }
        observation = {
            "kind": "observation", "subject": "conversation:test",
            "predicate": "said", "object": "A cat is an animal.",
            "polarity": "neutral", "confidence": 0.5,
        }
        connection.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            ("e1", json.dumps(payload, sort_keys=True), "d1", 1),
        )
        connection.execute(
            "INSERT INTO events VALUES(?,?,?,?)",
            ("e2", json.dumps(observation, sort_keys=True), "d2", 2),
        )
        connection.execute(
            "INSERT INTO projections VALUES(?,?,?,?,?,1)",
            ("p1", "e1", 7, 5, "discovery"),
        )
        connection.execute(
            "INSERT INTO projections VALUES(?,?,?,?,?,1)",
            ("p2", "e2", 8, 0, "discovery"),
        )
        connection.execute(
            "INSERT INTO meta VALUES('schema','nanlm-hermes-memory/2')")
        connection.commit()
        connection.close()

    def test_inspection_is_read_only_and_reports_rows(self):
        before_bytes = self.path.read_bytes()
        before_stat = self.path.stat()
        before_sidecars = plateau._sidecar_paths(self.path)
        result = plateau.inspect_memory_db(self.path)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["read_only"])
        self.assertTrue(result["unchanged"])
        self.assertEqual(result["relation_rows_total"], 1)
        self.assertEqual(result["observations_total"], 1)
        self.assertEqual(result["relation_rows"][0]["subject"], "A cat")
        self.assertEqual(result["relation_rows"][0]["predicate"], "is")
        self.assertEqual(result["relation_rows"][0]["object"], "an animal")
        self.assertEqual(result["relation_rows"][0]["polarity"], "supports")
        self.assertEqual(result["relation_rows"][0]["confidence"], 0.55)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(self.path.stat().st_mtime_ns, before_stat.st_mtime_ns)
        self.assertEqual(plateau._sidecar_paths(self.path), before_sidecars)

    def test_missing_database_is_graceful(self):
        result = plateau.inspect_memory_db(Path(self.tmp.name) / "missing.sqlite3")
        self.assertEqual(result["status"], "missing")
        self.assertTrue(result["read_only"])
        self.assertEqual(result["relation_rows"], [])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "universe" / "mathematical_memory.sqlite3"
        self.db_path.parent.mkdir(parents=True)
        self._make_db()
        self.run_path = self.root / "run.json"
        self._make_run()

    def _make_db(self):
        connection = sqlite3.connect(self.db_path)
        connection.executescript("""
            CREATE TABLE events(
                event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                source_digest TEXT NOT NULL, created_ns INTEGER NOT NULL
            );
            CREATE TABLE projections(
                projection_id TEXT PRIMARY KEY, event_id TEXT NOT NULL,
                head INTEGER NOT NULL, layer INTEGER NOT NULL,
                role TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        rows = [
            ("A cat", "is", "an animal", "A cat is an animal."),
            ("Grass", "is", "green", "Grass is green."),
            ("Ice", "is", "frozen water", "Ice is frozen water."),
            ("The sun", "is", "hot", "The sun is hot."),
            ("Frozen water", "is", "ice", "Frozen water is ice."),
        ]
        for index, (subject, predicate, obj, sentence) in enumerate(rows):
            payload = {
                "kind": "relation", "subject": subject,
                "predicate": predicate, "object": obj,
                "polarity": "supports", "confidence": 0.55,
                "evidence": {"sentence": sentence},
                "provenance": {"source_type": "native_chat_parser"},
            }
            event_id = "e%d" % index
            connection.execute(
                "INSERT INTO events VALUES(?,?,?,?)",
                (event_id, json.dumps(payload, sort_keys=True), "d%d" % index, index),
            )
            connection.execute(
                "INSERT INTO projections VALUES(?,?,?,?,?,1)",
                ("p%d" % index, event_id, index, 5, "discovery"),
            )
        connection.execute(
            "INSERT INTO meta VALUES('schema','nanlm-hermes-memory/2')")
        connection.commit()
        connection.close()

    def _make_run(self):
        probes = [
            {"question": "What is a cat?", "expected": ["animal"]},
            {"question": "What color is grass?", "expected": ["green"]},
            {"question": "What is ice?", "expected": ["frozen", "water"]},
            {"question": "What is the sun?", "expected": ["hot"]},
            {"question": "What is a river?", "expected": ["water"]},
            {"question": "What is a circle?", "expected": ["round"]},
            {"question": "What happens when water freezes?",
             "expected": ["ice", "frozen"]},
        ]
        answers = {
            "What is a cat?": "A cat is an animal.",
            "What color is grass?": "Grass is green.",
            "What is ice?": "Ice is frozen water.",
            "What is the sun?": "The sun is hot.",
            "What is a river?": "A river carries water.",
            "What is a circle?": "Circle.",
            "What happens when water freezes?": "Water has the ability to flow.",
        }
        hit = {
            "What is a cat?": True,
            "What color is grass?": True,
            "What is ice?": True,
            "What is the sun?": True,
            "What is a river?": True,
            "What is a circle?": False,
            "What happens when water freezes?": False,
        }
        round_probes = [
            _probe(probe["question"], probe["expected"],
                   answers[probe["question"]], hit[probe["question"]])
            for probe in probes
        ]
        baseline_probes = [
            _probe(probe["question"], probe["expected"], "unknown", False)
            for probe in probes
        ]
        statements = [
            "A cat is an animal.",
            "Grass is green.",
            "Ice is frozen water.",
            "The sun is hot.",
            "Frozen water is ice.",
        ]
        record = {
            "schema": "nanlm.gemma-learning-over-time/1",
            "model": "test-teacher",
            "state_dir": str(self.root),
            "rounds_requested": 2,
            "target_facts": [
                "A cat is an animal.",
                "Grass is green.",
                "Ice is frozen water. Heat causes melting.",
                "The sun is hot.",
                "A circle is round.",
                "A river is water.",
            ],
            "probes": probes,
            "baseline": {
                "probes": baseline_probes,
                "hit_rate": 0.0,
                "metrics": {},
            },
            "rounds": [
                {
                    "round": 1,
                    "statements": statements,
                    "probes": round_probes,
                    "hit_rate": 5 / 7,
                    "metrics": {},
                },
                {
                    "round": 2,
                    "statements": ["A circle is round."],
                    "probes": round_probes,
                    "hit_rate": 5 / 7,
                    "metrics": {},
                },
            ],
            "boundaries": {"proof_claim": False},
        }
        self.run_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def test_analyze_detects_71_4_plateau_and_read_only_db(self):
        before = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        report = plateau.analyze_run(
            self.run_path, state_dir=self.root, db_path=self.db_path)
        after = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertTrue(report["db"]["read_only"])
        self.assertTrue(report["db"]["unchanged"])
        self.assertEqual(report["schema"], plateau.REPORT_SCHEMA)
        plateau_row = report["summary"]["plateaus"][0]
        self.assertEqual(plateau_row["stages"], ["round 1", "round 2"])
        self.assertAlmostEqual(plateau_row["rate"], 5 / 7, places=6)
        self.assertIn("What is a circle?", plateau_row["shared_misses"])
        self.assertIn("What happens when water freezes?",
                      plateau_row["shared_misses"])
        causes = report["summary"]["cause_counts"]
        self.assertGreaterEqual(causes["causal_question_missing"], 1)
        self.assertGreaterEqual(causes["teacher_curriculum_missing"], 1)
        self.assertTrue(report["coverage"]["rows"])
        self.assertTrue(report["probe_db_findings"])
        self.assertEqual(len(report["ablation_plan"]), 5)
        self.assertTrue(report["ablation_plan_leak_check"]["passed"])

    def test_live_snapshot_is_marked_in_progress(self):
        record = json.loads(self.run_path.read_text(encoding="utf-8"))
        record["rounds_requested"] = 3
        partial = self.root / "partial.json"
        partial.write_text(json.dumps(record), encoding="utf-8")
        report = plateau.analyze_run(
            partial, state_dir=self.root, db_path=self.db_path)
        self.assertEqual(report["run"]["run_status"], "in_progress")
        text = plateau.render_text(report, max_lines=32)
        self.assertIn("status: in_progress", text)

    def test_text_summary_is_bounded_and_reports_plateau(self):
        report = plateau.analyze_run(
            self.run_path, state_dir=self.root, db_path=self.db_path)
        text = plateau.render_text(report, max_lines=24, max_chars=2000)
        self.assertIn("71.4%", text)
        self.assertLessEqual(len(text.splitlines()), 25)
        self.assertLessEqual(len(text), 2200)

    def test_markdown_and_json_reports_are_written(self):
        report = plateau.analyze_run(
            self.run_path, state_dir=self.root, db_path=self.db_path)
        markdown = plateau.render_markdown(report)
        self.assertIn("## Coverage matrix", markdown)
        self.assertIn("## Ablation plan", markdown)
        self.assertIn("71.4%", markdown)
        self.assertIn("**planned, not run**", markdown)
        json_path = self.root / "out" / "report.json"
        md_path = self.root / "out" / "report.md"
        written = plateau.write_reports(report, json_path, md_path)
        self.assertEqual(Path(written["json"]), json_path)
        self.assertEqual(Path(written["markdown"]), md_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], plateau.REPORT_SCHEMA)
        self.assertTrue(md_path.read_text(encoding="utf-8").startswith(
            "# NaNLM plateau diagnostic"))

        cli_json = self.root / "cli" / "report.json"
        cli_md = self.root / "cli" / "report.md"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = plateau.main([
                "--run-json", str(self.run_path),
                "--db", str(self.db_path),
                "--json", str(cli_json),
                "--markdown", str(cli_md),
                "--format", "text",
            ])
        self.assertEqual(code, 0)
        self.assertTrue(cli_json.is_file())
        self.assertTrue(cli_md.is_file())
        self.assertIn("NaNLM plateau diagnostic", output.getvalue())

    def test_ablation_plan_covers_required_interventions(self):
        plan = plateau.ablation_plan()
        ids = {item["id"] for item in plan}
        self.assertIn("A1_more_repetitions", ids)
        self.assertIn("A2_explicit_qa_teaching_without_exact_probe", ids)
        self.assertIn("A3_subject_balanced_color_facts", ids)
        self.assertIn("A4_reverse_lookup", ids)
        self.assertIn("A5_causal_paraphrase", ids)
        self.assertTrue(all(item["status"] == "planned_not_run" for item in plan))
        self.assertTrue(all(item["does_not_leak_probe"] for item in plan))


if __name__ == "__main__":
    unittest.main()
