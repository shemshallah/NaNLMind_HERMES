"""Resident 256-head activation field tests.

The tests exercise the real ``ResidentCells`` region in a 2 MiB bytearray:
independent per-head activation, normalized field metrics, pair coupling
through the existing rails, scratch-first proposals and permanent identity
preservation.  A small duck-typed fake and a wrapping host are also checked
because the adapter must not require the full universe object.
"""
import json
import math
import unittest

from nanlmind_hermes.resident.cells import (
    FIXED_ONE,
    HEAD_COUNT,
    REGION_BYTES,
    ResidentCells,
    address_of,
    canonical_address,
    identity_address,
    to_fixed,
    to_phase,
)
from nanlmind_hermes.resident.head_field import (
    MAX_DELTA,
    SNAPSHOT_EVENTS,
    SNAPSHOT_HEADS,
    SNAPSHOT_PROPOSALS,
    HeadFieldDenied,
    ResidentHeadField,
)


def forward_rail(cells, left, right):
    """Return the rail that ``observe_pair(left, right)`` updates."""
    state = cells.pair_state(left, right)
    if address_of(left, right) == canonical_address(left, right):
        return state.forward
    return state.reverse


class HeadFieldTestCase(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(2 * 1024 * 1024)
        self.assertEqual(len(self.buffer), REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)
        self.field = ResidentHeadField(self.cells)


class IndependentActivationTests(HeadFieldTestCase):
    def test_activation_is_additive_and_head_local(self):
        record = self.field.activate(7, 2.5, reason="seed")
        self.assertEqual(record["status"], "activated")
        self.assertEqual(record["reason"], "seed")

        head_seven = self.field.read(7)
        self.assertEqual(head_seven["head"], 7)
        self.assertEqual(head_seven["head_id"], 7)
        self.assertEqual(head_seven["identity"], identity_address(7))
        self.assertEqual(head_seven["score"], 2.5)
        self.assertEqual(head_seven["score_word"], to_fixed(2.5))
        self.assertEqual(head_seven["state"], 0)
        self.assertEqual(head_seven["generation"], 1)

        head_eight = self.field.read(8)
        self.assertEqual(head_eight["score"], 0.0)
        self.assertEqual(head_eight["state"], 0)
        self.assertEqual(head_eight["generation"], 0)
        self.assertEqual(head_eight["identity"], identity_address(8))

        self.field.activate(8, -1.25)
        self.assertEqual(self.field.read(8)["score"], -1.25)
        self.assertEqual(self.field.read(7)["score"], 2.5)

        self.field.activate(7, 0.5)
        self.assertEqual(self.field.read(7)["score"], 3.0)
        self.assertEqual(self.field.read(7)["generation"], 2)

        offset = self.cells.head_record_offset(8)
        before_bytes = bytes(self.buffer[offset:offset + 32])
        before = self.field.read(8)
        self.field.activate(7, 1.0)
        after_bytes = bytes(self.buffer[offset:offset + 32])
        self.assertEqual(before_bytes, after_bytes)
        self.assertEqual(before, self.field.read(8))

    def test_field_generation_is_monotonic(self):
        start = self.field.generation
        self.field.activate(0, 1.0)
        first = self.field.generation
        self.field.activate(1, 1.0)
        second = self.field.generation
        self.assertGreater(first, start)
        self.assertGreater(second, first)

    def test_activation_denials_do_not_mutate(self):
        before = self.field.read(3)
        for head in (-1, HEAD_COUNT, 3.0, True, "3", None):
            with self.assertRaises(HeadFieldDenied) as caught:
                self.field.activate(head, 1.0)
            self.assertEqual(caught.exception.code, "unknown_head")
        for delta in (float("nan"), float("inf"), float("-inf"),
                      10 ** 9, -10 ** 9, MAX_DELTA + 1.0):
            with self.assertRaises(HeadFieldDenied):
                self.field.activate(3, delta)
        self.assertEqual(before, self.field.read(3))


class FieldActivationTests(HeadFieldTestCase):
    def test_empty_field_has_zero_vector_and_metrics(self):
        activation = self.field.field_activation()
        self.assertEqual(len(activation["vector"]), HEAD_COUNT)
        self.assertEqual(len(activation["activation"]), HEAD_COUNT)
        self.assertEqual(math.fsum(activation["vector"]), 0.0)
        self.assertEqual(activation["active_heads"], 0)
        self.assertEqual(activation["active_indices"], [])
        self.assertEqual(activation["entropy"], 0.0)
        self.assertEqual(activation["entropy_bits"], 0.0)
        self.assertIsNone(activation["peak_head"])
        self.assertEqual(activation["metrics"]["active_heads"], 0)

    def test_normalized_vector_entropy_and_active_heads(self):
        self.field.activate(3, 3.0)
        self.field.activate(200, 1.0)
        self.field.activate(5, -2.0)  # negative scores carry no mass
        activation = self.field.field_activation()

        self.assertEqual(len(activation["vector"]), HEAD_COUNT)
        self.assertAlmostEqual(math.fsum(activation["vector"]), 1.0, places=12)
        self.assertEqual(activation["active_heads"], 2)
        self.assertEqual(activation["active_head_count"], 2)
        self.assertEqual(activation["active"], 2)
        self.assertEqual(activation["active_indices"], [3, 200])
        self.assertAlmostEqual(activation["vector"][3], 0.75, places=12)
        self.assertAlmostEqual(activation["vector"][200], 0.25, places=12)
        self.assertEqual(activation["vector"][5], 0.0)
        self.assertEqual(activation["peak_head"], 3)
        self.assertAlmostEqual(activation["peak"], 0.75, places=12)
        self.assertEqual(activation["total"], 4.0)

        expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        self.assertAlmostEqual(activation["entropy"], expected, places=10)
        self.assertAlmostEqual(
            activation["entropy_bits"], expected / math.log(2.0), places=10)
        self.assertAlmostEqual(
            activation["entropy_ratio"], expected / math.log(2.0), places=10)
        self.assertAlmostEqual(
            activation["metrics"]["entropy"], expected, places=10)

    def test_single_active_head_has_zero_entropy(self):
        self.field.activate(12, 9.0)
        activation = self.field.field_activation()
        self.assertEqual(activation["active_heads"], 1)
        self.assertEqual(activation["peak_head"], 12)
        self.assertEqual(activation["vector"][12], 1.0)
        self.assertEqual(activation["entropy"], 0.0)
        self.assertEqual(activation["entropy_ratio"], 0.0)


class PairCouplingTests(HeadFieldTestCase):
    def test_couple_uses_observe_pair_and_preserves_head_identity(self):
        left, right = 17, 136
        before_left = self.field.read(left)
        before_right = self.field.read(right)
        before_other = self.field.read(9)

        result = self.field.couple(left, right, weight=1.5, phase=0.25)
        self.assertEqual(result["status"], "coupled")
        self.assertTrue(result["identity_preserved"])
        self.assertEqual(result["left"], left)
        self.assertEqual(result["right"], right)
        self.assertEqual(result["weight"], 1.5)
        self.assertEqual(result["phase"], 0.25)
        self.assertIn("canonical", result)
        self.assertIn("mirror", result)

        rail = forward_rail(self.cells, left, right)
        self.assertEqual(rail.weight, to_fixed(1.5))
        self.assertEqual(rail.amplitude, to_fixed(1.5))
        self.assertEqual(rail.phase, to_phase(0.25))

        after_left = self.field.read(left)
        after_right = self.field.read(right)
        self.assertEqual(after_left["identity"], identity_address(left))
        self.assertEqual(after_right["identity"], identity_address(right))
        self.assertEqual(after_left["head_id"], left)
        self.assertEqual(after_right["head_id"], right)
        self.assertEqual(after_left["score"], before_left["score"])
        self.assertEqual(after_right["score"], before_right["score"])
        self.assertEqual(self.field.read(9), before_other)

        self.field.couple(left, right, weight=1.5, phase=0.25)
        self.assertEqual(forward_rail(self.cells, left, right).weight,
                         to_fixed(3.0))

    def test_coupling_denials_do_not_touch_pair_or_heads(self):
        left, right = 5, 200
        before_left = self.field.read(left)
        before_right = self.field.read(right)

        with self.assertRaises(HeadFieldDenied) as caught:
            self.field.couple(left, left, 1.0)
        self.assertEqual(caught.exception.code, "same_head")
        for bad_head in (-1, HEAD_COUNT, True, 1.5):
            with self.assertRaises(HeadFieldDenied):
                self.field.couple(bad_head, right, 1.0)
        for bad_weight in (float("nan"), float("inf"), 10 ** 9,
                           MAX_DELTA + 1.0):
            with self.assertRaises(HeadFieldDenied):
                self.field.couple(left, right, bad_weight)
        for bad_phase in (float("nan"), float("inf")):
            with self.assertRaises(HeadFieldDenied):
                self.field.couple(left, right, 1.0, phase=bad_phase)

        state = self.cells.pair_state(left, right)
        self.assertEqual(state.forward.weight, 0)
        self.assertEqual(state.reverse.weight, 0)
        self.assertEqual(self.field.read(left), before_left)
        self.assertEqual(self.field.read(right), before_right)


class ProposalTests(HeadFieldTestCase):
    def test_proposal_is_scratch_until_explicit_approved_apply(self):
        head = 42
        before = self.field.read(head)
        proposal = self.field.propose(head, 2.0)
        self.assertEqual(proposal["status"], "proposed")
        self.assertFalse(proposal["applied"])
        self.assertIsInstance(proposal["token"], str)
        self.assertEqual(proposal["proposed_score"], 2.0)
        self.assertEqual(proposal["previous_score"], 0.0)
        self.assertEqual(self.field.read(head), before)

        with self.assertRaises(HeadFieldDenied) as caught:
            self.field.apply_proposal(proposal["token"])
        self.assertEqual(caught.exception.code, "unapproved_apply")
        self.assertEqual(self.field.read(head), before)

        applied = self.field.apply_proposal(
            proposal["token"], approved=True)
        self.assertEqual(applied["status"], "applied")
        self.assertTrue(applied["applied"])
        self.assertEqual(applied["score"], 2.0)
        self.assertEqual(applied["identity"], identity_address(head))
        self.assertEqual(self.field.read(head)["score"], 2.0)

        # A consumed token never applies its delta twice.
        again = self.field.apply_proposal(
            proposal["token"], approved=True)
        self.assertEqual(again["status"], "already_applied")
        self.assertFalse(again["applied"])
        self.assertEqual(self.field.read(head)["score"], 2.0)

        # The whole proposal mapping is accepted as well as its token string.
        second = self.field.propose(43, 1.0)
        self.field.apply_proposal(second, approved=True)
        self.assertEqual(self.field.read(43)["score"], 1.0)

    def test_approved_propose_commits_explicitly(self):
        proposal = self.field.propose(44, 0.5, approved=True)
        self.assertEqual(proposal["status"], "applied")
        self.assertTrue(proposal["applied"])
        self.assertEqual(self.field.read(44)["score"], 0.5)

    def test_proposal_denials_never_stage_or_mutate(self):
        before = self.field.read(10)
        for head in (-1, HEAD_COUNT, True, "10"):
            with self.assertRaises(HeadFieldDenied) as caught:
                self.field.propose(head, 1.0)
            self.assertEqual(caught.exception.code, "unknown_head")
        for delta in (float("nan"), float("inf"), float("-inf"),
                      10 ** 9, MAX_DELTA + 1.0):
            with self.assertRaises(HeadFieldDenied):
                self.field.propose(10, delta)
        self.assertEqual(self.field.read(10), before)
        self.assertEqual(self.field.snapshot()["pending_proposals"], 0)

        with self.assertRaises(HeadFieldDenied) as caught:
            self.field.apply_proposal("missing-token", approved=True)
        self.assertEqual(caught.exception.code, "unknown_proposal")


class SnapshotAndIdentityTests(HeadFieldTestCase):
    def test_snapshot_is_bounded_json_and_identity_is_permanent(self):
        mask = []
        for head in range(HEAD_COUNT):
            offset = self.cells.head_record_offset(head)
            mask.append(bytes(self.buffer[offset:offset + 4]))
            self.assertEqual(self.field.read(head)["identity"],
                             identity_address(head))
        digest = self.field.snapshot()["identity_digest"]

        self.field.activate(3, 2.0)
        self.field.activate(200, 0.5)
        self.field.couple(3, 200, weight=1.25, phase=0.5)
        proposal = self.field.propose(77, 0.75)
        self.field.apply_proposal(proposal, approved=True)
        self.field.propose(78, 0.25)  # left pending

        for head in range(HEAD_COUNT):
            offset = self.cells.head_record_offset(head)
            self.assertEqual(bytes(self.buffer[offset:offset + 4]), mask[head])
            self.assertEqual(self.field.read(head)["head_id"], head)
            self.assertEqual(self.field.read(head)["identity"],
                             identity_address(head))

        snapshot = self.field.snapshot()
        encoded = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        self.assertLess(len(encoded), 65_536)
        self.assertEqual(len(snapshot["vector"]), HEAD_COUNT)
        self.assertEqual(len(snapshot["activation"]["vector"]), HEAD_COUNT)
        self.assertLessEqual(len(snapshot["heads"]), SNAPSHOT_HEADS)
        self.assertLessEqual(len(snapshot["events"]), SNAPSHOT_EVENTS)
        self.assertLessEqual(len(snapshot["proposals"]),
                             SNAPSHOT_PROPOSALS)
        self.assertEqual(snapshot["pending_proposals"], 1)
        self.assertEqual(snapshot["applied_proposals"], 1)
        self.assertEqual(snapshot["coupling_count"], 1)
        self.assertEqual(snapshot["identity_digest"], digest)
        self.assertEqual(snapshot["active_heads"], 3)
        self.assertEqual(snapshot["heads"][0]["head"], 3)
        self.assertEqual(snapshot["heads"][0]["score"], 2.0)
        self.assertEqual(snapshot["schema"],
                         "nanlm.resident-head-field/1")


class AdapterCompatibilityTests(HeadFieldTestCase):
    def test_wrapping_host_commit_flush_and_generation(self):
        class Host:
            def __init__(self, cells):
                self.cells = cells
                self.generation = 0
                self.commits = []
                self.flushes = 0

            def _commit(self, kind, identity):
                self.commits.append((kind, str(identity)))
                self.generation += 1

            def flush(self):
                self.flushes += 1

        host = Host(self.cells)
        field = ResidentHeadField(host)
        self.assertIs(field.cells, self.cells)

        field.activate(5, 1.5)
        self.assertEqual(host.commits[-1], ("activate", "head:5"))
        self.assertEqual(host.flushes, 1)
        self.assertEqual(field.read(5)["score"], 1.5)

        proposal = field.propose(6, 0.5)
        self.assertEqual(len(host.commits), 1)  # scratch: no commit
        self.assertEqual(host.flushes, 1)
        field.apply_proposal(proposal["token"], approved=True)
        self.assertEqual(host.commits[-1],
                         ("apply", proposal["token"]))
        self.assertEqual(host.flushes, 2)

        field.couple(5, 6, weight=0.5)
        self.assertEqual(host.commits[-1], ("pair", "5:6"))
        self.assertEqual(host.flushes, 3)
        self.assertGreaterEqual(field.generation, host.generation)

    def test_zero_argument_commit_hook_and_explicit_hooks(self):
        class Host:
            def __init__(self, cells):
                self.cells = cells
                self.generation = 0
                self.commits = 0
                self.flushes = 0

            def _commit(self):
                self.commits += 1
                self.generation += 1

            def flush(self):
                self.flushes += 1

        host = Host(self.cells)
        field = ResidentHeadField(host)
        field.activate(1, 1.0)
        self.assertEqual(host.commits, 1)
        self.assertEqual(host.flushes, 1)

        commits = []
        flushes = []
        direct = ResidentHeadField(
            self.cells, generation=7,
            commit=lambda: commits.append(True),
            flush=lambda: flushes.append(True))
        self.assertEqual(direct.generation, 7)
        direct.activate(2, 1.0)
        self.assertEqual(commits, [True])
        self.assertEqual(flushes, [True])

    def test_duck_typed_fake_store(self):
        class FakeCells:
            def __init__(self):
                self.records = [
                    {
                        "head_id": head,
                        "identity": identity_address(head),
                        "score": 0,
                        "state": 0,
                        "generation": 0,
                    }
                    for head in range(HEAD_COUNT)
                ]
                self.generation = 0
                self.pairs = {}

            def read_head(self, head):
                return dict(self.records[head])

            def update_head(self, head, score_delta=0.0,
                            state_delta=0, **kwargs):
                record = self.records[head]
                record["score"] += int(round(score_delta * FIXED_ONE))
                record["generation"] += 1
                self.generation += 1
                return dict(record)

            def observe_pair(self, left, right, weight=1.0, phase=0.0,
                             **kwargs):
                self.pairs[(left, right)] = (weight, phase)
                return None

        fake = FakeCells()
        field = ResidentHeadField(fake, generation=4)
        self.assertEqual(field.generation, 4)
        field.activate(11, 1.25)
        self.assertEqual(field.read(11)["score"], 1.25)
        self.assertEqual(field.read(11)["identity"],
                         identity_address(11))
        field.couple(11, 12, weight=0.5, phase=0.125)
        self.assertEqual(fake.pairs[(11, 12)], (0.5, 0.125))


if __name__ == "__main__":
    unittest.main()
