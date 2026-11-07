import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes import clocked_tools
from nanlmind_hermes.evidence import build_evidence_package


class FakeContext:
    def __init__(self, root):
        self.root = root
    def get_config(self, key, default=None):
        return str(self.root) if key == "state_dir" else default


class FakeHeadClient:
    def process(self, package):
        return {
            "schema": "nanlm.head-cortex-response",
            "request_sha256": package["package_sha256"],
            "truth_state": "UNKNOWN",
            "relations": [],
            "abstentions": [{"reason": "no_valid_relation_atoms"}],
            "provenance": {"read_only": True, "source_entry_ids": []},
        }


class TransformIntegrationTests(unittest.TestCase):
    def test_head_process_executes_full_m0_m9_chain_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            with patch("nanlmind_hermes.clocked_tools._head_client",
                       return_value=FakeHeadClient()):
                result = clocked_tools.head_process(
                    ctx, "What is established?", "transform-integration")

            trace = result["transformation_trace"]
            self.assertEqual([x["pass"] for x in trace["passes"]],
                             ["M0", "M1", "M2", "M3", "M4",
                              "M5", "M6", "M7", "M8", "M9"])
            self.assertEqual(trace["truth_state"], "UNKNOWN")
            self.assertEqual(trace["decision"], "abstain")
            self.assertFalse(trace["canonical_mutation_allowed"])
            self.assertFalse(trace["promotion_allowed"])

            scratch = result["scratch"]
            trace_rows = [x for x in scratch["entries"]
                          if x["lane"] == "transform_trace"]
            self.assertEqual(len(trace_rows), 2)
            rebuilt = build_evidence_package(
                scratch, "What is established?")
            self.assertEqual(rebuilt["package_sha256"], result["package_sha256"])
            self.assertFalse(any(x["lane"] == "transform_trace"
                                 for x in rebuilt["evidence_entries"]))


if __name__ == "__main__":
    unittest.main()
