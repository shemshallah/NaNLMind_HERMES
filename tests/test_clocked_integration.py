import tempfile
import unittest

from nanlmind_hermes.scratch import Scratch, RevisionConflict, ScratchError, ZERO
from nanlmind_hermes.evidence import build_evidence_package
from nanlmind_hermes.clocked_mode import make_task_plan, validate_task_plan, propose_mode


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


if __name__ == "__main__":
    unittest.main()
