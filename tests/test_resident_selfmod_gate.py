"""Tests for the conservative resident self-modification gate."""
from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes.resident.selfmod_gate import (
    MAX_RULES,
    MAX_VALUE_BYTES,
    SelfModificationGate,
)
from nanlmind_hermes.scratch_register import ScratchRegisterSpace

AUTHORITY_KEYS = (
    "canonical", "canonical_merge", "canonical_merge_enabled",
    "canonical_authority", "canonical_commit", "canonical_state_modified",
    "merge_enabled", "promotion_allowed",
)


class RecordingScratch:
    """Small scratch backend with an explicit ordering log."""

    def __init__(self, events: list[str]):
        self.events = events
        self.entries: list[dict] = []

    def append(self, *, layer, head, leaf_id, payload):
        self.events.append("scratch")
        entry = deepcopy(payload)
        entry.update({"layer": layer, "head": head, "leaf_id": leaf_id})
        self.entries.append(entry)
        return {"ok": True, "layer": layer, "head": head, "leaf_id": leaf_id}


class RecordingMain:
    """Main backend with a generation and an explicit ordering log."""

    def __init__(self, events: list[str], generation: int = 0,
                 advance: bool = True):
        self.events = events
        self.generation = generation
        self.advance = advance
        self.records: list[dict] = []

    def write(self, record):
        self.events.append("main")
        self.records.append(deepcopy(record))
        if self.advance:
            self.generation = record["generation"]
        return {"accepted": True, "generation": self.generation}


class ContradictingLogic:
    def evaluate(self, proposal):
        return {
            "accepted": False,
            "contradiction": True,
            "reasons": ["head %s contradicts the staged invariant" %
                        proposal["head"]],
        }


class SelfModificationGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _gate(self, *, logic=None, main=True):
        events: list[str] = []
        scratch = RecordingScratch(events)
        main_backend = RecordingMain(events) if main else None
        gate = SelfModificationGate(scratch, main_backend, logic)
        return gate, scratch, main_backend, events

    def _stage(self, gate):
        token = gate.propose(
            7, 5, "set", {"weight": 0.25},
            reason="bounded resident adjustment")
        self.assertIsInstance(token, str)
        evaluation = gate.evaluate(token)
        self.assertTrue(evaluation["accepted"], evaluation)
        return token, evaluation

    def assert_no_authority(self, payload):
        """No authority flag anywhere in a result may be true."""
        def walk(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    here = "%s.%s" % (path, key) if path else key
                    if key in AUTHORITY_KEYS:
                        self.assertIs(value, False, here)
                    if key == "six_runner_barrier":
                        self.assertEqual(value, "absent", here)
                    walk(value, here)
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, "%s[%d]" % (path, index))

        walk(payload)

    # --------------------------------------------------------------- happy path
    def test_happy_path_stages_evaluates_and_commits_once(self):
        gate, scratch, main, _events = self._gate()
        token = gate.propose(
            255, 9, "set", {"gain": 0.5},
            reason="bounded head adjustment")
        self.assertTrue(token.startswith("smod_"))
        self.assertEqual(len(scratch.entries), 1)
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)

        evaluation = gate.evaluate(token)
        self.assertEqual(set(evaluation), {"accepted", "reasons", "metrics"})
        self.assertTrue(evaluation["accepted"])
        self.assertEqual(evaluation["reasons"], [])
        self.assertEqual(evaluation["metrics"]["rules_failed"], 0)
        self.assertEqual(evaluation["metrics"]["main_writes"], 0)
        self.assertEqual(len(scratch.entries), 1)
        self.assertEqual(main.records, [])

        receipt = gate.commit(
            token, approved=True, expected_generation=0)
        self.assertEqual(receipt["status"], "committed")
        self.assertTrue(receipt["committed"])
        self.assertEqual(receipt["generation"], 1)
        self.assertEqual(receipt["previous_generation"], 0)
        self.assertTrue(receipt["generation_cas"]["ok"])
        self.assertEqual(receipt["main_writes"], 1)
        self.assertEqual(len(main.records), 1)
        self.assertEqual(main.generation, 1)
        self.assertEqual(main.records[0]["token"], token)
        self.assertEqual(main.records[0]["head"], 255)
        self.assertEqual(main.records[0]["layer"], 9)
        self.assertEqual(main.records[0]["generation"], 1)
        self.assert_no_authority(receipt)
        self.assert_no_authority(scratch.entries[0])
        self.assert_no_authority(main.records[0])

    def test_scratch_is_written_before_main_and_evaluate_is_read_only(self):
        gate, _scratch, _main, events = self._gate()
        token = gate.propose(
            3, 0, "update", 1.0, reason="ordering check")
        self.assertEqual(events, ["scratch"])
        gate.evaluate(token)
        self.assertEqual(events, ["scratch"])
        gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(events, ["scratch", "main"])

    # ---------------------------------------------------------------- denials
    def test_unknown_heads_are_denied_without_mutation(self):
        for head in (-1, 256, True, 1.5, "3", None):
            gate, scratch, main, events = self._gate()
            denial = gate.propose(
                head, 0, "set", 1, reason="invalid head")
            self.assertTrue(denial["denied"])
            self.assertEqual(denial["code"], "unknown_head")
            self.assertFalse(denial["accepted"])
            self.assertEqual(events, [])
            self.assertEqual(scratch.entries, [])
            self.assertEqual(main.records, [])
            self.assert_no_authority(denial)

    def test_unknown_layers_are_denied_without_mutation(self):
        for layer in (-1, 10, True, "L10", "hex"):
            gate, scratch, main, events = self._gate()
            denial = gate.propose(
                1, layer, "set", 1, reason="invalid layer")
            self.assertEqual(denial["code"], "unknown_layer")
            self.assertEqual(events, [])
            self.assertEqual(scratch.entries, [])
            self.assertEqual(main.records, [])

    def test_layer_names_are_normalised(self):
        gate, _scratch, _main, _events = self._gate()
        token = gate.propose(
            1, "lexical", "set", 1, reason="named layer")
        self.assertEqual(gate.evaluate(token)["accepted"], True)
        self.assertEqual(gate._proposals[token].layer, 5)

    def test_unknown_operations_are_denied_without_mutation(self):
        for operation in ("drop_everything", "delete_all", "", None, 7):
            gate, scratch, main, events = self._gate()
            denial = gate.propose(
                1, 0, operation, 1, reason="invalid operation")
            self.assertEqual(denial["code"], "unknown_operation")
            self.assertEqual(events, [])
            self.assertEqual(scratch.entries, [])
            self.assertEqual(main.records, [])

    def test_oversized_and_non_finite_values_are_denied(self):
        values = (
            ("x" * (MAX_VALUE_BYTES + 1), "oversized_value"),
            (10 ** 12, "oversized_value"),
            ([0] * 600, "oversized_value"),
            (float("nan"), "invalid_value"),
            (float("inf"), "invalid_value"),
        )
        for value, expected in values:
            gate, scratch, main, events = self._gate()
            denial = gate.propose(
                1, 0, "set", value, reason="bounded value check")
            self.assertEqual(denial["code"], expected)
            self.assertEqual(events, [])
            self.assertEqual(scratch.entries, [])
            self.assertEqual(main.records, [])

    def test_canonical_merge_attempts_are_denied(self):
        cases = (
            ("canonical_merge", 1),
            ("promote", 1),
            ("set", {"canonical_merge": True}),
            ("set", {"promotion_allowed": True}),
            ("set", ["merge"]),
        )
        for operation, value in cases:
            gate, scratch, main, events = self._gate()
            denial = gate.propose(
                1, 0, operation, value, reason="stage a change")
            self.assertEqual(denial["code"], "canonical_merge_attempt")
            self.assertEqual(events, [])
            self.assertEqual(scratch.entries, [])
            self.assertEqual(main.records, [])

        gate, scratch, main, events = self._gate()
        denial = gate.propose(
            1, 0, "set", 1, reason="canonical merge request")
        self.assertEqual(denial["code"], "canonical_merge_attempt")
        self.assertEqual(events, [])
        self.assertEqual(scratch.entries, [])
        self.assertEqual(main.records, [])

    def test_logic_contradiction_blocks_evaluation_and_commit(self):
        gate, scratch, main, events = self._gate(
            logic=ContradictingLogic())
        token = gate.propose(
            4, 2, "set", {"value": 3}, reason="logic will reject")
        self.assertEqual(events, ["scratch"])
        evaluation = gate.evaluate(token)
        self.assertFalse(evaluation["accepted"])
        self.assertEqual(evaluation["metrics"]["denial"], "logic_contradiction")
        self.assertTrue(evaluation["reasons"])
        self.assertEqual(events, ["scratch"])
        self.assertEqual(len(scratch.entries), 1)

        denial = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(denial["code"], "logic_contradiction")
        self.assertEqual(events, ["scratch"])
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)

    def test_callable_logic_false_is_a_contradiction(self):
        gate, _scratch, main, _events = self._gate(
            logic=lambda proposal: False)
        token = gate.propose(
            4, 2, "set", 3, reason="callable logic rejects")
        evaluation = gate.evaluate(token)
        self.assertFalse(evaluation["accepted"])
        self.assertEqual(evaluation["metrics"]["denial"], "logic_contradiction")
        self.assertEqual(main.records, [])

    def test_commit_requires_prior_accepted_evaluation(self):
        gate, _scratch, main, events = self._gate()
        token = gate.propose(
            1, 1, "set", 1, reason="not evaluated yet")
        self.assertEqual(events, ["scratch"])
        denial = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(denial["code"], "not_evaluated")
        self.assertEqual(events, ["scratch"])
        self.assertEqual(main.records, [])

    def test_unapproved_commit_is_denied_literally(self):
        for approval in (False, 1, "yes", None):
            gate, scratch, main, events = self._gate()
            token, _evaluation = self._stage(gate)
            denial = gate.commit(
                token, approved=approval, expected_generation=0)
            self.assertEqual(denial["code"], "unapproved")
            self.assertEqual(events, ["scratch"])
            self.assertEqual(len(scratch.entries), 1)
            self.assertEqual(main.records, [])

    def test_stale_generation_is_denied_and_retryable(self):
        gate, scratch, main, events = self._gate()
        token, _evaluation = self._stage(gate)
        denial = gate.commit(token, approved=True, expected_generation=7)
        self.assertEqual(denial["code"], "stale_generation")
        self.assertEqual(denial["details"]["observed_generation"], 0)
        self.assertEqual(events, ["scratch"])
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)

        receipt = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(receipt["status"], "committed")
        self.assertEqual(events, ["scratch", "main"])
        self.assertEqual(len(main.records), 1)

    def test_generation_cas_and_monotonic_second_commit(self):
        gate, _scratch, main, _events = self._gate()
        first, _ = self._stage(gate)
        gate.commit(first, approved=True, expected_generation=0)

        second = gate.propose(
            8, 3, "set", 2, reason="second bounded adjustment")
        self.assertTrue(gate.evaluate(second)["accepted"])
        stale = gate.commit(second, approved=True, expected_generation=0)
        self.assertEqual(stale["code"], "stale_generation")
        self.assertEqual(len(main.records), 1)
        receipt = gate.commit(second, approved=True, expected_generation=1)
        self.assertEqual(receipt["generation"], 2)
        self.assertEqual(main.generation, 2)
        self.assertEqual([record["generation"] for record in main.records],
                         [1, 2])

    def test_duplicate_token_is_denied_on_commit_replay(self):
        gate, scratch, main, events = self._gate()
        token, _evaluation = self._stage(gate)
        gate.commit(token, approved=True, expected_generation=0)
        denial = gate.commit(token, approved=True, expected_generation=1)
        self.assertEqual(denial["code"], "duplicate_token")
        self.assertEqual(events, ["scratch", "main"])
        self.assertEqual(len(scratch.entries), 1)
        self.assertEqual(len(main.records), 1)
        self.assertEqual(main.generation, 1)

    def test_duplicate_token_is_denied_when_proposed_twice(self):
        gate, scratch, main, events = self._gate()
        with patch.object(gate, "_new_token", return_value="smod_fixed"):
            token = gate.propose(
                1, 0, "set", 1, reason="first fixed token")
        self.assertEqual(token, "smod_fixed")
        with patch.object(gate, "_new_token", return_value="smod_fixed"):
            denial = gate.propose(
                2, 0, "set", 2, reason="second fixed token")
        self.assertEqual(denial["code"], "duplicate_token")
        self.assertEqual(events, ["scratch"])
        self.assertEqual(len(scratch.entries), 1)
        self.assertEqual(main.records, [])

    def test_unknown_token_evaluation_is_a_structured_denial(self):
        gate, _scratch, main, events = self._gate()
        evaluation = gate.evaluate("smod_missing")
        self.assertFalse(evaluation["accepted"])
        self.assertEqual(evaluation["metrics"]["denial"], "unknown_token")
        self.assertEqual(events, [])
        self.assertEqual(main.records, [])

    def test_commit_without_main_backend_is_denied(self):
        gate, _scratch, _main, events = self._gate(main=False)
        token, _evaluation = self._stage(gate)
        denial = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(denial["code"], "no_main_backend")
        self.assertEqual(events, ["scratch"])

    def test_scratch_write_failure_never_reaches_main(self):
        class BrokenScratch:
            def append(self, **kwargs):
                raise OSError("disk unavailable")

        events: list[str] = []
        main = RecordingMain(events)
        gate = SelfModificationGate(BrokenScratch(), main)
        denial = gate.propose(
            1, 0, "set", 1, reason="scratch unavailable")
        self.assertEqual(denial["code"], "scratch_write_failed")
        self.assertEqual(events, [])
        self.assertEqual(main.records, [])
        self.assertEqual(gate.status()["proposals"], 0)

    def test_main_rejection_does_not_consume_the_token(self):
        class RejectingMain:
            generation = 0

            def write(self, record):
                return {"accepted": False, "reason": "main refused"}

        scratch = RecordingScratch([])
        main = RejectingMain()
        gate = SelfModificationGate(scratch, main)
        token, _evaluation = self._stage(gate)
        denial = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(denial["code"], "main_write_rejected")
        self.assertEqual(main.generation, 0)
        self.assertEqual(gate.commit(
            token, approved=False, expected_generation=0)["code"], "unapproved")

    # ------------------------------------------------------------- adapters
    def test_mapping_backends_are_supported(self):
        scratch = {}
        main = {"generation": 0}
        gate = SelfModificationGate(scratch, main)
        token = gate.propose(
            10, 4, "set", {"x": 1}, reason="mapping backends")
        self.assertEqual(len(scratch["entries"]), 1)
        self.assertTrue(gate.evaluate(token)["accepted"])
        receipt = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(receipt["generation"], 1)
        self.assertEqual(main["generation"], 1)
        self.assertEqual(len(main["records"]), 1)
        self.assertEqual(main["records"][0]["token"], token)

    def test_plain_object_backend_generation_is_advanced_once(self):
        events: list[str] = []
        main = RecordingMain(events, advance=False)
        gate = SelfModificationGate(RecordingScratch(events), main)
        token, _evaluation = self._stage(gate)
        receipt = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(receipt["generation"], 1)
        self.assertEqual(main.generation, 1)
        self.assertEqual(len(main.records), 1)

    def test_real_scratch_register_integration(self):
        scratch = ScratchRegisterSpace(self.root / "scratch")
        events: list[str] = []
        main = RecordingMain(events)
        gate = SelfModificationGate(scratch, main)
        token = gate.propose(
            17, 5, "set", {"weight": 0.75},
            reason="real scratch register")
        records = list(scratch.records(layer=5, head=17))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["leaf_id"], token)
        self.assertEqual(records[0]["kind"], "selfmod-proposal")
        self.assertEqual(records[0]["payload"]["token"], token)
        self.assertEqual(records[0]["payload"]["layer"], 5)
        self.assertTrue(gate.evaluate(token)["accepted"])
        receipt = gate.commit(token, approved=True, expected_generation=0)
        self.assertEqual(receipt["status"], "committed")
        self.assertEqual(events, ["main"])
        self.assertEqual(len(main.records), 1)
        self.assertEqual(len(list(scratch.records(layer=5, head=17))), 1)

    def test_rule_table_is_bounded_and_status_is_non_authoritative(self):
        gate, _scratch, _main, _events = self._gate()
        self.assertLessEqual(len(gate.rules), MAX_RULES)
        self.assertEqual(len(gate.rule_table), len(gate.rules))
        status = gate.status()
        self.assertEqual(status["rule_count"], len(gate.rules))
        self.assertLessEqual(status["rule_count"], status["max_rules"])
        self.assertFalse(status["canonical_merge_enabled"])
        self.assertFalse(status["promotion_allowed"])
        self.assertEqual(status["six_runner_barrier"], "absent")
        self.assert_no_authority(status)

        too_many = {str(index): True for index in range(MAX_RULES + 1)}
        with self.assertRaises(ValueError):
            SelfModificationGate(RecordingScratch([]), None, too_many)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
