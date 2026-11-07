"""Regression coverage for HERMES candidate head evaluation."""
import hashlib
import io
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes.head_cortex import HeadCortexClient, HeadCortexError
from nanlmind_hermes import clocked_tools


class FakeResponse:
    def __init__(self, payload):
        self.raw = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.raw.read(size)


class FakeScratch:
    def __init__(self):
        self.rows = []

    def append(self, lane, writer, payload, *args):
        self.rows.append({"lane": lane, "writer": writer, "payload": payload})
        return {"revision": len(self.rows)}

    def snapshot(self):
        return {"entry_count": len(self.rows), "entries": self.rows}


class HeadCortexEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.client = HeadCortexClient("https://head.example", "scoped-token")

    @patch("nanlmind_hermes.head_cortex.urlopen")
    def test_requires_explicit_nonpromotion_receipt(self, open_url):
        open_url.return_value = FakeResponse({
            "schema": "nanlm.head-candidate-evaluation-response/1",
            "promotion_allowed": False,
            "canonical_state_modified": False,
            "metrics": {"accuracy": 0.75},
        })
        result = self.client.evaluate_candidate({
            "schema": "nanlm.head-candidate-evaluation-request/1",
            "mutation_id": "eval-1", "split": "test",
            "dataset_sha256": "a" * 64, "records": [],
        })
        self.assertIs(result["promotion_allowed"], False)
        self.assertIs(result["canonical_state_modified"], False)

    @patch("nanlmind_hermes.head_cortex.urlopen")
    def test_rejects_promotion_and_canonical_mutation(self, open_url):
        for field in ("promotion_allowed", "canonical_state_modified"):
            response = {
                "schema": "nanlm.head-candidate-evaluation-response/1",
                "promotion_allowed": False,
                "canonical_state_modified": False,
            }
            response[field] = True
            open_url.return_value = FakeResponse(response)
            with self.subTest(field=field), self.assertRaises(HeadCortexError):
                self.client.evaluate_candidate({
                    "schema": "nanlm.head-candidate-evaluation-request/1",
                    "mutation_id": "eval-1", "split": "test",
                    "dataset_sha256": "a" * 64, "records": [],
                })

    @patch("nanlmind_hermes.head_cortex.urlopen")
    def test_rejects_wrong_schema_and_oversized_response(self, open_url):
        open_url.return_value = FakeResponse({"schema": "wrong"})
        with self.assertRaises(HeadCortexError):
            self.client.evaluate_candidate({
                "schema": "nanlm.head-candidate-evaluation-request/1"
            })
        open_url.return_value = FakeResponse({"schema": "x" * 1_048_600})
        with self.assertRaises(HeadCortexError):
            self.client.evaluate_candidate({
                "schema": "nanlm.head-candidate-evaluation-request/1"
            })

    def test_rejects_oversized_request_before_network(self):
        with patch("nanlmind_hermes.head_cortex.urlopen") as open_url:
            with self.assertRaises(ValueError):
                self.client.evaluate_candidate({
                    "schema": "nanlm.head-candidate-evaluation-request/1",
                    "payload": "x" * 1_100_000,
                })
            open_url.assert_not_called()


class CandidateInputTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "resolution": 5, "polarity": "+",
            "subject": "source", "predicate": "supports", "object": "claim",
            "source_digest": "a" * 64,
        }

    def test_requires_explicit_approval(self):
        with self.assertRaises(PermissionError):
            clocked_tools.head_candidate_evaluate(None, "eval-1", [self.row])

    def test_rejects_bad_relation_fields_and_digest(self):
        for update, message in [
            ({"source_digest": "bad"}, "source_digest"),
            ({"resolution": "mystery"}, "N16REL resolution"),
            ({"subject": ""}, "subject"),
        ]:
            row = dict(self.row, **update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                clocked_tools.head_candidate_evaluate(None, "eval-1", [row], approved=True)

    @patch.dict("os.environ", {
        "NANLM_HEAD_CORTEX_URL": "https://head.example",
        "NANLM_DEVICE_TOKEN": "scoped-token",
    })
    @patch("nanlmind_hermes.clocked_tools._scratch")
    @patch("nanlmind_hermes.clocked_tools.HeadCortexClient")
    def test_binds_dataset_digest_to_test_split(self, client_cls, scratch_factory):
        scratch = FakeScratch()
        scratch_factory.return_value = scratch
        client_cls.return_value.evaluate_candidate.return_value = {
            "schema": "nanlm.head-candidate-evaluation-response/1",
            "promotion_allowed": False,
            "canonical_state_modified": False,
            "metrics": {},
        }
        result = clocked_tools.head_candidate_evaluate(
            object(), "eval-1", [self.row], approved=True
        )
        request = client_cls.return_value.evaluate_candidate.call_args.args[0]
        expected = hashlib.sha256(json.dumps(
            [self.row], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        self.assertEqual(request["split"], "test")
        self.assertEqual(request["dataset_sha256"], expected)
        self.assertFalse(result["evaluation"]["promotion_allowed"])

    def test_plugin_registers_one_evaluation_handler_and_tool(self):
        entrypoint = Path(__file__).parents[1] / "nanlmind_hermes" / "__init__.py"
        source = entrypoint.read_text(encoding="utf-8")
        handlers = re.findall(r"^    def handle_head_candidate_evaluate\(", source, re.MULTILINE)
        specs = re.findall(r'^\s*\("nanlm_head_candidate_evaluate",', source, re.MULTILINE)
        self.assertEqual(len(handlers), 1)
        self.assertEqual(len(specs), 1)


if __name__ == "__main__":
    unittest.main()
