"""Regression tests for the HERMES head-cortex evaluation boundary."""
import io
import json
import unittest
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


class HeadCortexClientTests(unittest.TestCase):
    def setUp(self):
        self.client = HeadCortexClient("https://head.example", "scoped-token")

    @patch("nanlmind_hermes.head_cortex.urlopen")
    def test_candidate_evaluation_requires_non_promotion_receipt(self, open_url):
        open_url.return_value = FakeResponse({
            "schema": "nanlm.head-candidate-evaluation-response/1",
            "promotion_allowed": False,
            "canonical_state_modified": False,
            "metrics": {},
        })
        result = self.client.evaluate_candidate({
            "schema": "nanlm.head-candidate-evaluation-request/1",
            "mutation_id": "eval-1",
            "records": [],
        })
        self.assertIs(result["promotion_allowed"], False)
        self.assertIs(result["canonical_state_modified"], False)

    @patch("nanlmind_hermes.head_cortex.urlopen")
    def test_candidate_evaluation_rejects_promotion_or_canonical_mutation(self, open_url):
        for field in ("promotion_allowed", "canonical_state_modified"):
            payload = {
                "schema": "nanlm.head-candidate-evaluation-response/1",
                "promotion_allowed": False,
                "canonical_state_modified": False,
            }
            payload[field] = True
            open_url.return_value = FakeResponse(payload)
            with self.subTest(field=field), self.assertRaises(HeadCortexError):
                self.client.evaluate_candidate({
                    "schema": "nanlm.head-candidate-evaluation-request/1",
                    "mutation_id": "eval-1",
                    "records": [],
                })

    @patch("nanlmind_hermes.head_cortex.urlopen")
    def test_candidate_evaluation_rejects_wrong_response_schema(self, open_url):
        open_url.return_value = FakeResponse({"schema": "wrong"})
        with self.assertRaises(HeadCortexError):
            self.client.evaluate_candidate({
                "schema": "nanlm.head-candidate-evaluation-request/1",
                "mutation_id": "eval-1",
                "records": [],
            })

    def test_candidate_evaluation_rejects_oversized_request(self):
        with self.assertRaises(ValueError):
            self.client.evaluate_candidate({
                "schema": "nanlm.head-candidate-evaluation-request/1",
                "mutation_id": "eval-1",
                "records": [{"payload": "x" * 1_100_000}],
            })

    def test_invalid_request_schema_is_rejected_before_network(self):
        with patch("nanlmind_hermes.head_cortex.urlopen") as open_url:
            with self.assertRaises(ValueError):
                self.client.evaluate_candidate({"schema": "wrong"})
            open_url.assert_not_called()


class CandidateInputTests(unittest.TestCase):
    def test_held_out_rows_require_sha256_source_digest(self):
        row = {
            "resolution": "lexical",
            "polarity": "TRUE",
            "subject": "source",
            "predicate": "supports",
            "object": "claim",
            "source_digest": "not-a-digest",
        }
        with patch.dict("os.environ", {
            "NANLM_HEAD_CORTEX_URL": "https://head.example",
            "NANLM_DEVICE_TOKEN": "scoped-token",
        }):
            with self.assertRaisesRegex(ValueError, "source_digest"):
                clocked_tools.head_candidate_evaluate(
                    None, "eval-1", [row], approved=True
                )

    def test_candidate_evaluation_requires_explicit_approval(self):
        with self.assertRaises(PermissionError):
            clocked_tools.head_candidate_evaluate(
                None, "eval-1", [], approved=False
            )


    def test_plugin_registers_head_evaluation_handler_once(self):
        from pathlib import Path
        import re
        entrypoint = Path(__file__).parents[1] / "nanlmind_hermes" / "__init__.py"
        source = entrypoint.read_text(encoding="utf-8")
        handler_defs = re.findall(r"^    def handle_head_candidate_evaluate\\(", source, re.MULTILINE)
        tool_specs = re.findall(r'^\\s*\\("nanlm_head_candidate_evaluate",', source, re.MULTILINE)
        self.assertEqual(len(handler_defs), 1)
        self.assertEqual(len(tool_specs), 1)



if __name__ == "__main__":
    unittest.main()
