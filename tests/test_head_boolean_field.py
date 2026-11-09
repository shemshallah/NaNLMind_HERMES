"""Tests for the 256-head Boolean self-modification field."""
import dataclasses
import unittest

from nanlmind_hermes.boolean_wave import verify_exact as legacy_verify_exact
from nanlmind_hermes.head_boolean_field import (
    AND,
    AND_MASK,
    BOOLEAN_FUNCTION_COUNT,
    BOOLEAN_FUNCTION_NAMES,
    BOOLEAN_FUNCTIONS,
    BOOLEAN_MASK_ALL,
    BOOL_MASKS,
    HEAD_COUNT,
    OR,
    OR_MASK,
    XOR,
    XOR_MASK,
    ApprovalDenied,
    ApprovalRequired,
    BooleanField,
    BooleanFieldError,
    HeadIdentity,
    MaskError,
    MaskOverflowError,
    NonFiniteError,
    ScratchViolation,
    StaleProposalError,
    UnknownHeadError,
    UnknownProposalError,
    boolean,
    logic_base_preserved,
    table,
    truth_table,
    verify_exact,
    verify_logic_base,
)

XOR_SAMPLES = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
AND_SAMPLES = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1))
OR_SAMPLES = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))


def samples_for(mask):
    return tuple((a, b, (mask >> ((a << 1) | b)) & 1)
                 for a in (0, 1) for b in (0, 1))


class FoundationTests(unittest.TestCase):
    def test_constants_and_all_256_heads_exist(self):
        self.assertEqual(HEAD_COUNT, 256)
        self.assertEqual(BOOLEAN_FUNCTION_COUNT, 16)
        self.assertEqual(len(BOOLEAN_FUNCTIONS), 16)
        self.assertEqual(BOOL_MASKS, tuple(range(16)))
        self.assertEqual(len(BOOLEAN_FUNCTION_NAMES), 16)
        self.assertEqual((XOR, AND, OR), (6, 8, 14))
        self.assertEqual((XOR_MASK, AND_MASK, OR_MASK), (64, 256, 16384))

        field = BooleanField()
        self.assertEqual(len(field), 256)
        self.assertEqual(len(field.heads), 256)
        self.assertEqual([head.index for head in field.heads],
                         list(range(256)))
        self.assertEqual(len({head.identity.address for head in field.heads}),
                         256)
        self.assertTrue(field.foundation_status()["intact"])
        self.assertTrue(field.foundation_status()["unique_addresses"])
        self.assertEqual(field.heads[0].state.ops_mask, BOOLEAN_MASK_ALL)
        self.assertEqual(field.heads[255].identity.route, "HW0:255")

    def test_identity_is_immutable_but_state_is_mutable(self):
        field = BooleanField()
        head = field.heads[7]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            head.identity.index = 8
        head.state.scores[XOR] = 11
        head.state.rule = XOR
        head.state.ops_mask = 1 << XOR
        self.assertEqual(head.scores[XOR], 11)
        self.assertEqual(head.rule, XOR)
        self.assertEqual(head.ops_mask, 1 << XOR)
        self.assertIsNot(head.state.scores, field.heads[8].state.scores)

        identity = HeadIdentity(3)
        self.assertEqual(identity.index, 3)
        self.assertEqual(identity.head, 3)
        self.assertEqual(identity.address, (3 << 8) | 0xC0)


class BooleanLearningTests(unittest.TestCase):
    def test_four_of_four_exact_learning_for_all_16_functions(self):
        field = BooleanField()
        for target in range(BOOLEAN_FUNCTION_COUNT):
            samples = samples_for(target)
            report = field.train_boolean(target, samples, epochs=1)
            self.assertEqual(report["rule"], target)
            self.assertEqual(report["ops_mask"], 1 << target)
            self.assertEqual(report["correct"], 4)
            self.assertEqual(report["total"], 4)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertTrue(report["exact"])
            result = field.evaluate_boolean(target, samples)
            self.assertEqual(result["correct"], 4)
            self.assertEqual(result["total"], 4)
            self.assertEqual(result["accuracy"], 1.0)
            self.assertTrue(result["exact"])
            for a in (0, 1):
                for b in (0, 1):
                    self.assertEqual(
                        head_output(field, target, a, b),
                        boolean(target, a, b))

    def test_xor_and_or_remain_the_legacy_exact_base(self):
        field = BooleanField()
        for head, samples, target in (
                (0, XOR_SAMPLES, XOR),
                (1, AND_SAMPLES, AND),
                (2, OR_SAMPLES, OR)):
            field.train_boolean(head, samples, epochs=1)
            self.assertEqual(field.rule(head), target)
            self.assertEqual(field.evaluate_boolean(head, samples)["correct"], 4)

    def test_heads_diverge_independently(self):
        field = BooleanField()
        field.train_boolean(0, XOR_SAMPLES, epochs=2)
        field.train_boolean(1, AND_SAMPLES, epochs=2)
        field.train_boolean(2, OR_SAMPLES, epochs=2)
        self.assertEqual([field.rule(h) for h in (0, 1, 2)], [XOR, AND, OR])
        self.assertEqual(field.evaluate_boolean(0, AND_SAMPLES)["accuracy"], 0.25)
        self.assertEqual(field.evaluate_boolean(1, AND_SAMPLES)["accuracy"], 1.0)

        # Changing one head's score table never changes another head.
        before = list(field.heads[1].scores)
        field.heads[0].scores[AND] = 999
        field.heads[0].state.rule = XOR
        self.assertEqual(field.heads[1].scores, before)

        result = field.field_step([0, 1, 2], (0, 1))
        self.assertEqual(result["outputs"], [1, 0, 1])
        self.assertEqual(len(set(result["outputs"])), 2)
        self.assertFalse(result["collapsed_to_global_decision"])


class FieldCouplingTests(unittest.TestCase):
    def setUp(self):
        self.field = BooleanField()
        self.field.train_boolean(0, XOR_SAMPLES, epochs=2)
        self.field.train_boolean(1, AND_SAMPLES, epochs=2)
        self.field.train_boolean(2, OR_SAMPLES, epochs=2)

    def test_independent_outputs_and_bounded_coupling(self):
        result = self.field.field_step([0, 1, 2], [(0, 1), (0, 1), (0, 1)])
        self.assertEqual(result["outputs"], [1, 0, 1])
        self.assertEqual(result["independent_outputs"], [1, 0, 1])
        self.assertEqual(len(result["per_head"]), 3)
        self.assertTrue(-1.0 <= result["field_coupling"] <= 1.0)
        self.assertTrue(result["coupling_bounded"])
        self.assertTrue(0.0 <= result["field_activation"] <= 1.0)
        self.assertEqual(len(result["coupled_activations"]), 3)
        for head, output in zip(result["heads"], result["outputs"]):
            self.assertEqual(
                output,
                boolean(self.field.rule(head), 0, 1))
            self.assertEqual(result["output_by_head"][head], output)
            self.assertGreaterEqual(result["coupled_activation_by_head"][head], 0.0)
            self.assertLessEqual(result["coupled_activation_by_head"][head], 1.0)
        # The field activation is a scalar diagnostic, not a replacement for
        # the three independent per-head outputs.
        self.assertIsInstance(result["field_activation"], float)
        self.assertTrue(self.field.heads[0].activation >= 0.0)
        self.assertTrue(0.0 <= self.field.heads[0].coupled_activation <= 1.0)

    def test_coupling_is_positive_for_agreement_and_negative_for_disagreement(self):
        agree = self.field.field_step([1, 2], (1, 1))
        self.assertEqual(agree["outputs"], [1, 1])
        self.assertGreater(agree["field_coupling"], 0.0)
        self.assertTrue(agree["coupling_bounded"])

        disagree = self.field.field_step([0, 1], (1, 1))
        self.assertEqual(disagree["outputs"], [0, 1])
        self.assertLess(disagree["field_coupling"], 0.0)
        self.assertGreaterEqual(disagree["field_coupling"], -1.0)

        single = self.field.field_step([0], (1, 1))
        self.assertEqual(single["field_coupling"], 0.0)
        self.assertEqual(single["outputs"], [0])

    def test_one_pair_per_head_and_mapping_inputs(self):
        per_head = self.field.field_step([0, 1], [(0, 1), (1, 1)])
        self.assertEqual(per_head["inputs"], [(0, 1), (1, 1)])
        self.assertEqual(per_head["outputs"], [1, 1])
        mapped = self.field.field_step([0, 1], {0: (0, 1), 1: (1, 0)})
        self.assertEqual(mapped["inputs"], [(0, 1), (1, 0)])
        self.assertEqual(mapped["outputs"], [1, 0])
        flat = self.field.field_step([0, 1], [0, 1, 1, 0])
        self.assertEqual(flat["inputs"], [(0, 1), (1, 0)])


class SelfModificationTests(unittest.TestCase):
    def setUp(self):
        self.field = BooleanField()
        self.field.train_boolean(0, XOR_SAMPLES, epochs=2)

    def test_scratch_first_proposal_never_mutates_main_rule(self):
        before_mask = self.field.ops_mask(0)
        before_rule = self.field.rule(0)
        proposal = self.field.self_modify(0, 0x000F)
        self.assertEqual(proposal["status"], "proposed")
        self.assertFalse(proposal["applied"])
        self.assertFalse(proposal["canonical_state_modified"])
        self.assertTrue(proposal["requires_approval"])
        self.assertEqual(proposal["before_mask"], before_mask)
        self.assertEqual(proposal["after_mask"], before_mask ^ 0x000F)
        self.assertEqual(proposal["proposed_mask"], before_mask ^ 0x000F)
        self.assertEqual(self.field.ops_mask(0), before_mask)
        self.assertEqual(self.field.rule(0), before_rule)
        self.assertEqual(len(self.field.scratch[0]), 1)
        self.assertEqual(self.field.scratch[0][0]["id"], proposal["id"])
        self.assertEqual(len(self.field.pending_proposals(0)), 1)

        # A commit without an explicit token is rejected and leaves state alone.
        with self.assertRaises(ApprovalRequired):
            self.field.commit_proposal(proposal["id"])
        self.assertEqual(self.field.ops_mask(0), before_mask)
        self.assertEqual(self.field.rule(0), before_rule)

        committed = self.field.commit_proposal(
            proposal["id"], commit_token="operator-approved")
        self.assertEqual(committed["status"], "applied")
        self.assertTrue(committed["applied"])
        self.assertTrue(committed["main_rule_modified"])
        self.assertFalse(committed["canonical_state_modified"])
        self.assertEqual(self.field.ops_mask(0), before_mask ^ 0x000F)
        self.assertEqual(self.field.rule(0), proposal["after_rule"])
        self.assertEqual(self.field.pending_proposals(0), [])

    def test_non_scratch_write_requires_approval(self):
        with self.assertRaises(ScratchViolation):
            self.field.self_modify(0, 0x0001, scratch_only=False)
        self.assertEqual(self.field.ops_mask(0), 1 << XOR)
        direct = self.field.self_modify(
            0, 0x0001, scratch_only=False, approval=True)
        self.assertTrue(direct["applied"])
        self.assertTrue(direct["main_rule_modified"])
        self.assertFalse(direct["canonical_state_modified"])
        self.assertEqual(self.field.ops_mask(0), (1 << XOR) ^ 0x0001)

    def test_stale_proposal_fails_closed(self):
        proposal = self.field.self_modify(0, 0x0003)
        self.field.train_boolean(0, XOR_SAMPLES, epochs=1)
        with self.assertRaises(StaleProposalError):
            self.field.commit_proposal(proposal["id"], commit_token="ok")

    def test_configured_approval_token_must_match(self):
        field = BooleanField(approval_token="correct-token")
        field.train_boolean(0, XOR_SAMPLES)
        proposal = field.self_modify(0, 0x0001)
        with self.assertRaises(ApprovalDenied):
            field.commit_proposal(proposal["id"], commit_token="wrong-token")
        committed = field.commit_proposal(
            proposal["id"], commit_token="correct-token")
        self.assertTrue(committed["applied"])

    def test_replace_mode_is_explicit(self):
        proposal = self.field.self_modify(0, AND_MASK, mode="replace")
        self.assertEqual(proposal["after_mask"], AND_MASK)
        self.assertEqual(proposal["after_rule"], AND)
        self.assertTrue(proposal["applied"] is False)
        self.field.commit_proposal(proposal, approval=True)
        self.assertEqual(self.field.ops_mask(0), AND_MASK)
        self.assertEqual(self.field.rule(0), AND)


class DenialTests(unittest.TestCase):
    def setUp(self):
        self.field = BooleanField()

    def test_unknown_heads_are_denied_everywhere(self):
        for bad in (256, -1, 1.5, "1", True):
            with self.assertRaises(UnknownHeadError):
                self.field.train_boolean(bad, XOR_SAMPLES)
            with self.assertRaises(UnknownHeadError):
                self.field.evaluate_boolean(bad, XOR_SAMPLES)
            with self.assertRaises(UnknownHeadError):
                self.field.self_modify(bad, 0x0001)
        with self.assertRaises(UnknownHeadError):
            self.field.field_step([0, 256], (0, 1))
        with self.assertRaises(UnknownHeadError):
            HeadIdentity(256)

    def test_non_finite_values_are_denied(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(NonFiniteError):
                self.field.train_boolean(0, [(0, 0, bad)])
            with self.assertRaises(NonFiniteError):
                self.field.evaluate_boolean(0, [(0, 0, bad)])
            with self.assertRaises(NonFiniteError):
                self.field.field_step([0], (bad, 0))
            with self.assertRaises(NonFiniteError):
                self.field.self_modify(0, bad)
        with self.assertRaises(NonFiniteError):
            BooleanField(coupling_gain=float("nan"))

    def test_unsigned_and_overflowing_masks_are_denied(self):
        for bad in (-1, 1 << 16, 0x1FFFF):
            with self.assertRaises(MaskError):
                self.field.self_modify(0, bad)
        with self.assertRaises(MaskOverflowError):
            self.field.self_modify(0, 0x10000)
        for bad in (True, False, 1.0, "15", None):
            with self.assertRaises(MaskError):
                self.field.self_modify(0, bad)

    def test_scratch_only_must_be_bool_and_writes_outside_are_denied(self):
        with self.assertRaises(ScratchViolation):
            self.field.self_modify(0, 0x0001, scratch_only="yes")
        with self.assertRaises(ScratchViolation):
            self.field.self_modify(0, 0x0001, scratch_only=False)
        self.assertEqual(self.field.heads[0].ops_mask, BOOLEAN_MASK_ALL)
        self.assertEqual(self.field.scratch[0], [])

    def test_unknown_proposals_and_bad_tokens_are_denied(self):
        with self.assertRaises(UnknownProposalError):
            self.field.commit_proposal("does-not-exist", commit_token="ok")
        proposal = self.field.self_modify(0, 0x0001)
        with self.assertRaises(ApprovalDenied):
            self.field.commit_proposal(proposal["id"], commit_token="")
        with self.assertRaises(ApprovalDenied):
            self.field.commit_proposal(proposal["id"], approval=0)
        with self.assertRaises(ApprovalRequired):
            self.field.commit_proposal(proposal["id"], approval=False)

    def test_empty_training_batch_is_denied(self):
        with self.assertRaises(BooleanFieldError):
            self.field.train_boolean(0, [])


class PreservationTests(unittest.TestCase):
    def test_xor_and_or_and_full_legacy_truth_table_are_exact(self):
        expected = {
            XOR: (0, 1, 1, 0),
            AND: (0, 0, 0, 1),
            OR: (0, 1, 1, 1),
        }
        for mask, truth in expected.items():
            self.assertEqual(table(mask), truth)
            self.assertEqual(truth_table(mask), truth)
            for a in (0, 1):
                for b in (0, 1):
                    self.assertEqual(boolean(mask, a, b),
                                     truth[(a << 1) | b])

        result = verify_logic_base()
        self.assertTrue(result["preserved"])
        self.assertTrue(result["exact"])
        self.assertTrue(result["xor_exact"])
        self.assertTrue(result["and_exact"])
        self.assertTrue(result["or_exact"])
        self.assertTrue(result["functions_exact"])
        self.assertEqual(result["all_functions"]["correct"], 64)
        self.assertEqual(result["all_functions"]["total"], 64)
        self.assertTrue(result["legacy"]["matches"])
        self.assertEqual(result["legacy"]["reported_correct"], 64)
        self.assertTrue(logic_base_preserved())
        self.assertTrue(BooleanField.logic_base_preserved())
        self.assertTrue(BooleanField().verify_logic_base()["preserved"])

    def test_verify_exact_and_legacy_module_agree_64_of_64(self):
        ours = verify_exact()
        legacy = legacy_verify_exact()
        self.assertTrue(ours["exact"])
        self.assertTrue(legacy["exact"])
        self.assertEqual((ours["correct"], ours["total"]), (64, 64))
        self.assertEqual((legacy["correct"], legacy["total"]), (64, 64))
        for mask in BOOL_MASKS:
            for a in (0, 1):
                for b in (0, 1):
                    self.assertEqual(boolean(mask, a, b),
                                     (mask >> ((a << 1) | b)) & 1)


def head_output(field, head, a, b):
    return boolean(field.rule(head), a, b)


if __name__ == "__main__":
    unittest.main()
