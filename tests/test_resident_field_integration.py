"""End-to-end merge tests for the resident head-field modules.

The head-field work lands as several independent modules.  This suite imports
them lazily so a genuinely absent module is reported as a skip, while a module
that exists but no longer matches the common integration contract fails loudly.

Covered contracts:

* real ``ResidentCells`` plus ``ResidentHeadField``: 256 independent heads,
  field activation, pair coupling and permanent identity;
* L0-L9 scratch layers -> Boolean evaluation -> approved generation-CAS commit
  to a main register, with denied/scratch-only paths leaving main untouched;
* ``FieldLearner`` over real resident cells: multiple active heads, entropy and
  collision metrics, no canonical merge/truth claim;
* the 16-function Boolean foundation and no shared state between heads;
* ``FieldPersistence`` round-trip through a real ``ResidentUniverse``;
* ``field_tools`` approval, bounds, read-only queries and no mutation on
  failure.

No network or cloud service is used.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

MIB = 1024 * 1024
UNIVERSE_BYTES = 16 * MIB

AUTHORITY_KEYS = (
    "canonical", "canonical_merge", "canonical_merge_enabled",
    "canonical_authority", "canonical_commit", "canonical_state_modified",
    "canonical_promotion", "canonical_promotion_enabled", "merge_enabled",
    "promotion_allowed", "promotion_enabled",
)


def load_module(name: str):
    """Import ``name`` lazily; skip only when the module is genuinely absent."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        raise unittest.SkipTest("%s is not present" % name)
    return importlib.import_module(name)


def assert_false_authority(testcase: unittest.TestCase, payload, path="") -> None:
    """Assert that no nested authority/canonical flag claims permission."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            here = "%s.%s" % (path, key) if path else str(key)
            if key in AUTHORITY_KEYS:
                if isinstance(value, dict):
                    testcase.assertFalse(
                        any(bool(item) for item in value.values()), here)
                else:
                    testcase.assertIs(value, False, here)
            if key == "six_runner_barrier":
                testcase.assertEqual(value, "absent", here)
            assert_false_authority(testcase, value, here)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_false_authority(testcase, value, "%s[%d]" % (path, index))


class ResidentHeadFieldIntegrationTests(unittest.TestCase):
    """Area 1: real cells plus the independent 256-head field."""

    def setUp(self):
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.head_field = load_module("nanlmind_hermes.resident.head_field")
        self.cells = self.cells_module.ResidentCells(
            bytearray(self.cells_module.REGION_BYTES), initialize=True)
        self.field = self.head_field.ResidentHeadField(self.cells)

    def test_all_256_heads_activate_independently_and_keep_identity(self):
        expected = {}
        for head in range(256):
            delta = 0.25 * ((head % 4) + 1)
            receipt = self.field.activate(head, delta, reason="integration")
            self.assertEqual(receipt["status"], "activated")
            self.assertEqual(receipt["identity"],
                             self.cells_module.identity_address(head))
            expected[head] = delta

        records = self.field.read_all()
        self.assertEqual(len(records), 256)
        identities = set()
        for head, delta in expected.items():
            record = records[head]
            self.assertEqual(record["head"], head)
            self.assertEqual(record["head_id"], head)
            self.assertEqual(record["identity"],
                             self.cells_module.identity_address(head))
            self.assertAlmostEqual(record["score"], delta, places=6)
            identities.add(record["identity"])
        self.assertEqual(len(identities), 256)

        # Activating one head must not alter any sibling's record bytes.
        before = {head: dict(self.field.read(head)) for head in range(256)}
        self.field.activate(42, 1.0)
        after = {head: dict(self.field.read(head)) for head in range(256)}
        for head in range(256):
            if head == 42:
                self.assertNotEqual(before[head], after[head])
            else:
                self.assertEqual(before[head], after[head])

        activation = self.field.field_activation()
        self.assertEqual(len(activation["vector"]), 256)
        self.assertAlmostEqual(math.fsum(activation["vector"]), 1.0,
                               places=10)
        self.assertGreater(activation["active_heads"], 1)
        self.assertGreater(activation["entropy"], 0.0)
        self.assertGreater(activation["entropy_bits"], 0.0)

        for head in range(256):
            record = self.cells.read_head(head)
            self.assertEqual(record.head_id, head)
            self.assertEqual(record.identity,
                             self.cells_module.identity_address(head))

    def test_pair_coupling_updates_rails_without_rewriting_heads(self):
        left, right, other = 17, 136, 9
        before_left = dict(self.field.read(left))
        before_right = dict(self.field.read(right))
        before_other = dict(self.field.read(other))

        result = self.field.couple(left, right, weight=1.25, phase=0.5)
        self.assertEqual(result["status"], "coupled")
        self.assertTrue(result["identity_preserved"])

        state = self.cells.pair_state(left, right)
        rail = (state.forward
                if self.cells_module.address_of(left, right)
                == state.canonical else state.reverse)
        self.assertEqual(rail.weight, self.cells_module.to_fixed(1.25))
        self.assertEqual(rail.phase, self.cells_module.to_phase(0.5))

        self.assertEqual(self.field.read(left)["score"],
                         before_left["score"])
        self.assertEqual(self.field.read(right)["score"],
                         before_right["score"])
        self.assertEqual(self.field.read(other), before_other)
        for head in (left, right):
            self.assertEqual(self.field.read(head)["identity"],
                             self.cells_module.identity_address(head))

    def test_proposal_is_scratch_only_until_explicit_approval(self):
        head = 77
        before = dict(self.field.read(head))
        proposal = self.field.propose(head, 0.75)
        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(self.field.read(head), before)

        with self.assertRaises(self.head_field.HeadFieldDenied):
            self.field.apply_proposal(proposal["token"])
        self.assertEqual(self.field.read(head), before)

        applied = self.field.apply_proposal(proposal["token"], approved=True)
        self.assertTrue(applied["applied"])
        self.assertAlmostEqual(applied["score"], 0.75, places=6)
        self.assertEqual(applied["identity"],
                         self.cells_module.identity_address(head))


class ScratchBooleanCommitIntegrationTests(unittest.TestCase):
    """Area 2: L0-L9 scratch -> Boolean evaluation -> approved CAS commit."""

    def setUp(self):
        self.scratch_module = load_module(
            "nanlmind_hermes.resident.scratch_layers")
        self.gate_module = load_module("nanlmind_hermes.resident.selfmod_gate")
        self.boolean_wave = load_module("nanlmind_hermes.boolean_wave")

    def test_l0_l9_scratch_and_generation_cas_commit_exactly_once(self):
        scratch = self.scratch_module.ScratchLayers()
        payloads = {}
        for layer in range(10):
            payload = {"layer": layer, "mask": 6, "truth": [0, 1, 1, 0]}
            receipt = scratch.write_layer(7, layer, payload)
            self.assertEqual(receipt["scope"], "scratch")
            self.assertFalse(receipt["main_touched"])
            payloads[layer] = payload
            self.assertEqual(scratch.read_value(7, layer), payload)
            self.assertEqual(scratch.main_generation(7, layer), 0)

        # The legacy Boolean base is the evaluation oracle for the commit.
        report = self.boolean_wave.verify_exact()
        self.assertEqual(report["correct"], 64)
        self.assertEqual(report["total"], 64)
        self.assertTrue(report["exact"])
        truth = [self.boolean_wave.boolean(6, a, b)
                 for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
        self.assertEqual(truth, [0, 1, 1, 0])

        proposal = {"mask": 6, "truth": truth, "layers": list(range(10))}
        token = scratch.stage_commit(7, 9, proposal, expected_generation=0)
        self.assertEqual(token.expected_generation, 0)
        self.assertEqual(scratch.main_generation(7, 9), 0)

        denied = scratch.commit_staged(
            token, approved=False, expected_generation=0)
        self.assertFalse(denied["committed"])
        self.assertEqual(denied["status"], "approval_required")
        self.assertEqual(scratch.main_generation(7, 9), 0)

        stale = scratch.commit_staged(
            token, approved=True, expected_generation=1)
        self.assertFalse(stale["committed"])
        self.assertEqual(stale["status"], "stale_generation")
        self.assertEqual(scratch.main_generation(7, 9), 0)

        committed = scratch.commit_staged(
            token, approved=True, expected_generation=0)
        self.assertTrue(committed["committed"])
        self.assertTrue(committed["main_updated"])
        self.assertEqual(committed["generation"], 1)
        self.assertEqual(scratch.main_generation(7, 9), 1)
        self.assertEqual(scratch.read_main(7, 9)["value"], proposal)
        self.assertEqual(scratch.stats()["main_commits"], 1)

        replay = scratch.commit_staged(
            token, approved=True, expected_generation=1)
        self.assertFalse(replay["committed"])
        self.assertEqual(replay["status"], "token_consumed")
        self.assertEqual(scratch.main_generation(7, 9), 1)
        self.assertEqual(scratch.stats()["main_commits"], 1)

        # The scratch layer itself was not replaced by the main commit.
        for layer, payload in payloads.items():
            self.assertEqual(scratch.read_value(7, layer), payload)

    def _gate(self):
        scratch_layers = self.scratch_module.ScratchLayers()

        class ScratchBridge:
            def __init__(self, layers):
                self.layers = layers
                self.writes = 0

            def append(self, *, layer, head, leaf_id, payload):
                self.writes += 1
                receipt = self.layers.write_layer(head, layer, payload)
                return {
                    "accepted": True,
                    "generation": receipt["generation"],
                    "scope": "scratch",
                }

        class CasMain:
            def __init__(self):
                self.generation = 0
                self.records = []
                self.fail = False

            def commit(self, record, expected_generation=None,
                       generation=None):
                if self.fail:
                    raise RuntimeError("simulated main write failure")
                expected = record["expected_generation"]
                if expected != self.generation:
                    return {"accepted": False,
                            "reasons": ["stale_generation"],
                            "generation": self.generation}
                self.records.append(dict(record))
                self.generation += 1
                return {"accepted": True, "generation": self.generation}

        def boolean_logic(proposal):
            report = self.boolean_wave.verify_exact()
            value = proposal.get("value") or {}
            mask = value.get("mask", 6)
            submitted = value.get("truth", [])
            expected = [self.boolean_wave.boolean(mask, a, b)
                        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
            checks = {"legacy_exact": report["exact"],
                      "truth_matches": submitted == expected}
            return {
                "accepted": all(checks.values()),
                "invariants": checks,
                "reason": None if all(checks.values())
                else "Boolean truth-table mismatch",
            }

        scratch = ScratchBridge(scratch_layers)
        main = CasMain()
        gate = self.gate_module.SelfModificationGate(
            scratch, main, logic=boolean_logic)
        return gate, scratch_layers, scratch, main

    def test_boolean_evaluation_precedes_approved_generation_cas(self):
        gate, scratch_layers, bridge, main = self._gate()
        for layer in range(9):
            scratch_layers.write_layer(7, layer, {"layer": layer})
        token = gate.propose(
            7, 9, "set", {"mask": 6, "truth": [0, 1, 1, 0]},
            reason="integration head update")
        self.assertIsInstance(token, str)
        self.assertEqual(bridge.writes, 1)
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)
        self.assertIsNotNone(scratch_layers.read_value(7, 9))

        evaluation = gate.evaluate(token)
        self.assertTrue(evaluation["accepted"])
        self.assertEqual(evaluation["metrics"]["main_writes"], 0)
        self.assertEqual(main.records, [])

        denied = gate.commit(token, approved=False, expected_generation=0)
        self.assertEqual(denied["code"], "unapproved")
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)

        stale = gate.commit(token, approved=True, expected_generation=1)
        self.assertEqual(stale["code"], "stale_generation")
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)

        committed = gate.commit(token, approved=True, expected_generation=0)
        self.assertTrue(committed["committed"])
        self.assertEqual(committed["generation"], 1)
        self.assertEqual(committed["main_writes"], 1)
        self.assertEqual(len(main.records), 1)
        self.assertEqual(main.generation, 1)
        assert_false_authority(self, committed)

        replay = gate.commit(token, approved=True, expected_generation=1)
        self.assertFalse(replay["committed"])
        self.assertEqual(replay["code"], "duplicate_token")
        self.assertEqual(len(main.records), 1)
        self.assertEqual(main.generation, 1)

    def test_boolean_contradiction_and_main_failure_leave_main_unchanged(self):
        gate, _layers, _bridge, main = self._gate()
        bad = gate.propose(
            4, 2, "set", {"mask": 6, "truth": [1, 1, 1, 1]},
            reason="wrong truth table")
        evaluation = gate.evaluate(bad)
        self.assertFalse(evaluation["accepted"])
        denied = gate.commit(bad, approved=True, expected_generation=0)
        self.assertEqual(denied["code"], "logic_contradiction")
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)

        good = gate.propose(
            4, 2, "set", {"mask": 8, "truth": [0, 0, 0, 1]},
            reason="valid truth table")
        self.assertTrue(gate.evaluate(good)["accepted"])
        main.fail = True
        failed = gate.commit(good, approved=True, expected_generation=0)
        self.assertEqual(failed["code"], "main_write_failed")
        self.assertFalse(failed["committed"])
        self.assertEqual(failed["main_writes"], 0)
        self.assertEqual(main.records, [])
        self.assertEqual(main.generation, 0)


class FieldLearningIntegrationTests(unittest.TestCase):
    """Area 3: corpus/free-chat field learning over real resident cells."""

    def setUp(self):
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.learning_module = load_module(
            "nanlmind_hermes.resident.field_learning")
        self.temporal_module = load_module("nanlmind_hermes.temporal_lm")

    def _learner(self):
        cells = self.cells_module.ResidentCells(
            bytearray(self.cells_module.REGION_BYTES), initialize=True)
        temporal = self.temporal_module.TemporalMirrorLM()
        learner = self.learning_module.FieldLearner.from_cells(
            temporal, cells)
        return cells, learner

    def test_real_cells_activate_more_than_one_head_with_metrics(self):
        cells, learner = self._learner()
        result = learner.observe(
            "A penguin swims in cold water because it is a bird.")

        self.assertTrue(result["learned"])
        self.assertTrue(result["accepted"])
        self.assertGreater(result["active_heads"], 1)
        self.assertGreater(len(result["head_updates"]), 1)
        self.assertGreater(result["entropy"], 0.0)
        self.assertIsInstance(result["collisions"], int)
        self.assertGreaterEqual(result["collisions"], 0)
        self.assertFalse(result["canonical_merge"])
        self.assertFalse(result["semantic_truth"])
        self.assertFalse(
            result["head_activations_are_semantic_truth"])
        self.assertLessEqual(result["writes"],
                             learner.max_writes_per_sentence)

        stats = learner.stats()
        self.assertGreater(stats["active_heads"], 1)
        self.assertGreater(stats["ledger_active_heads"], 1)
        self.assertGreater(stats["entropy"], 0.0)
        self.assertIsInstance(stats["collisions"], int)
        self.assertGreaterEqual(stats["accepted_proposals"], 1)
        self.assertFalse(stats["canonical_merge"])
        self.assertFalse(stats["semantic_truth"])
        self.assertFalse(hasattr(learner, "merge"))

        for head in range(256):
            record = cells.read_head(head)
            self.assertEqual(record.head_id, head)
            self.assertEqual(record.identity,
                             self.cells_module.identity_address(head))

    def test_corpus_batch_reports_active_entropy_and_collisions(self):
        _cells, learner = self._learner()
        records = [
            {"text": "A river carries water to the sea.",
             "source": "corpus"},
            {"text": "A bird can fly over the river.", "source": "corpus"},
            {"text": "Copper is a metal because it conducts electricity.",
             "source": "corpus"},
            {"text": "A penguin is a bird that swims.", "source": "corpus"},
        ]
        report = learner.observe_batch(records)

        self.assertEqual(report["records"], len(records))
        self.assertGreaterEqual(report["learned"], 3)
        self.assertGreater(report["active_heads"], 1)
        self.assertGreater(report["mean_active_heads"], 1.0)
        self.assertGreater(report["entropy"], 0.0)
        self.assertIsInstance(report["collisions"], int)
        self.assertGreaterEqual(report["collisions"], 0)
        self.assertGreater(report["accepted_proposals"], 0)
        self.assertIsNotNone(report["temporal_nll"])
        self.assertFalse(report["canonical_merge"])
        self.assertFalse(report["semantic_truth"])
        self.assertFalse(report["head_activations_are_semantic_truth"])

        evaluation = learner.evaluate(records)
        self.assertEqual(evaluation["evaluated"], len(records))
        self.assertGreater(evaluation["active_heads"], 1)
        self.assertGreater(evaluation["independent"]["entropy_bits"], 0.0)
        self.assertIsInstance(evaluation["collisions"], int)
        self.assertFalse(evaluation["canonical_merge"])
        self.assertFalse(evaluation["semantic_truth"])


class BooleanFoundationPreservationTests(unittest.TestCase):
    """Area 4: the 16-function Boolean base stays exact and head-local."""

    def setUp(self):
        self.boolean_wave = load_module("nanlmind_hermes.boolean_wave")

    def test_legacy_boolean_wave_16_functions_are_exact(self):
        report = self.boolean_wave.verify_exact()
        self.assertEqual(report["correct"], 64)
        self.assertEqual(report["total"], 64)
        self.assertTrue(report["exact"])

        for mask in range(16):
            expected = tuple((mask >> index) & 1 for index in range(4))
            self.assertEqual(self.boolean_wave.table(mask), expected)
            for a in (0, 1):
                for b in (0, 1):
                    wanted = expected[(a << 1) | b]
                    self.assertEqual(
                        self.boolean_wave.boolean(mask, a, b), wanted)
                    self.assertAlmostEqual(
                        self.boolean_wave.multilinear(mask, a, b), wanted,
                        places=12)

        self.assertEqual(
            [self.boolean_wave.boolean(6, a, b)
             for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))],
            [0, 1, 1, 0])
        self.assertEqual(
            [self.boolean_wave.boolean(8, a, b)
             for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))],
            [0, 0, 0, 1])
        self.assertEqual(
            [self.boolean_wave.boolean(14, a, b)
             for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))],
            [0, 1, 1, 1])

    def test_mth_root_boolean_table_is_exact(self):
        mth_layers = load_module("nanlmind_hermes.mth_layers")
        self.assertEqual(tuple(mth_layers.ROOT_MASKS), tuple(range(16)))
        for mask in range(16):
            for a in (0, 1):
                for b in (0, 1):
                    self.assertEqual(
                        mth_layers.bool_class(mask, a, b),
                        (mask >> ((a << 1) | b)) & 1)
            for left in range(16):
                for right in range(16):
                    expected = 0
                    for bit in range(4):
                        expected |= mth_layers.bool_class(
                            mask, (left >> bit) & 1,
                            (right >> bit) & 1) << bit
                    self.assertEqual(
                        mth_layers.apply_bits(mask, left, right, 4),
                        expected)

    def test_head_boolean_field_preserves_base_and_isolates_heads(self):
        head_field = load_module("nanlmind_hermes.head_boolean_field")
        field = head_field.BooleanField()
        self.assertEqual(field.head_count, 256)
        self.assertTrue(field.foundation_status()["intact"])

        preservation = field.verify_logic_base()
        self.assertTrue(preservation["preserved"])
        self.assertTrue(preservation["all_functions"]["exact"])
        self.assertTrue(preservation["legacy_preserved"])
        self.assertTrue(preservation["xor_exact"])
        self.assertTrue(preservation["and_exact"])
        self.assertTrue(preservation["or_exact"])

        xor_samples = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        and_samples = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        trained_xor = field.train_boolean(3, xor_samples, epochs=2)
        self.assertTrue(trained_xor["exact"])
        self.assertEqual(field.rule(3), head_field.XOR)

        head_three_before = field.head(3).snapshot()
        trained_and = field.train_boolean(4, and_samples, epochs=2)
        self.assertTrue(trained_and["exact"])
        self.assertEqual(field.rule(4), head_field.AND)
        self.assertEqual(field.rule(3), head_field.XOR)
        self.assertEqual(field.head(3).snapshot(), head_three_before)

        head_four_before = field.head(4).snapshot()
        field.train_boolean(3, xor_samples, epochs=1)
        self.assertEqual(field.head(4).snapshot(), head_four_before)

        legacy_before = field.verify_logic_base()
        proposal = field.self_modify(5, 1 << head_field.XOR,
                                     scratch_only=True)
        self.assertFalse(proposal["applied"])
        self.assertFalse(proposal["committed"])
        self.assertEqual(field.rule(5), 0)
        self.assertEqual(field.verify_logic_base(), legacy_before)
        self.assertEqual(
            [head_field.boolean(head_field.XOR, a, b)
             for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))],
            [0, 1, 1, 0])

        with self.assertRaises(head_field.ScratchViolation):
            field.self_modify(5, 1 << head_field.AND, scratch_only=False)
        self.assertEqual(field.rule(5), 0)

        committed = field.commit_proposal(
            proposal["proposal_id"], approval=True)
        self.assertTrue(committed["applied"])
        self.assertTrue(field.verify_logic_base()["preserved"])
        self.assertEqual(
            [head_field.boolean(head_field.XOR, a, b)
             for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))],
            [0, 1, 1, 0])

        # The independent AND head is still AND after another head changed.
        self.assertEqual(field.rule(4), head_field.AND)
        self.assertEqual(field.head(4).snapshot(), head_four_before)


class FieldPersistenceIntegrationTests(unittest.TestCase):
    """Area 5: field and scratch metadata survive a real universe reopen."""

    def setUp(self):
        self.universe_module = load_module(
            "nanlmind_hermes.resident.universe")
        self.field_module = load_module(
            "nanlmind_hermes.resident.head_field")
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.persistence_module = load_module(
            "nanlmind_hermes.resident.field_persistence")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "field-universe.nlmres"

    def test_field_and_scratch_round_trip_through_resident_universe(self):
        universe = self.universe_module.ResidentUniverse.create(
            self.path, size=UNIVERSE_BYTES)
        try:
            field = self.field_module.ResidentHeadField(universe)
            field.activate(3, 1.5, reason="persist")
            field.activate(200, 0.5, reason="persist")
            field.couple(3, 200, weight=1.0, phase=0.25)
            snapshot = field.snapshot()
            snapshot_json = json.dumps(snapshot, sort_keys=True,
                                       allow_nan=False)
            self.assertLess(len(snapshot_json), 1024 * 1024)

            persistence = self.persistence_module.FieldPersistence(universe)
            self.assertIsNone(persistence.load_field())

            denial = persistence.save_field(snapshot)
            self.assertFalse(denial)
            self.assertFalse(denial["accepted"])
            self.assertEqual(denial["status"], "approval_required")
            self.assertIsNone(persistence.load_field())

            scratch_payload = {"layer": 0, "activation": 0.25,
                               "mask": 6}
            scratch = persistence.save_scratch(5, 0, scratch_payload)
            self.assertTrue(scratch)
            self.assertTrue(scratch["accepted"])
            self.assertFalse(scratch["main_field_modified"])
            self.assertIsNone(persistence.load_field())
            self.assertEqual(persistence.load_scratch(5, 0),
                             scratch_payload)

            receipt = persistence.save_field(snapshot, approved=True)
            self.assertTrue(receipt)
            self.assertTrue(receipt["accepted"])
            self.assertTrue(receipt["created"])
            self.assertTrue(receipt["main_field_modified"])
            self.assertFalse(receipt["canonical_commit"])
            self.assertFalse(receipt["canonical_state_modified"])
            self.assertEqual(persistence.load_field(), snapshot)
            self.assertEqual(persistence.load_scratch(5, 0),
                             scratch_payload)

            duplicate = persistence.save_field(snapshot, approved=True)
            self.assertTrue(duplicate)
            self.assertEqual(duplicate["status"], "duplicate")
            self.assertEqual(duplicate["content_digest"],
                             receipt["content_digest"])
            universe.flush()
            generation = universe.generation
        finally:
            universe.close()

        reopened = self.universe_module.ResidentUniverse.open(self.path)
        try:
            reopened_field = self.field_module.ResidentHeadField(reopened)
            self.assertAlmostEqual(reopened_field.read(3)["score"], 1.5,
                                   places=6)
            self.assertAlmostEqual(reopened_field.read(200)["score"], 0.5,
                                   places=6)
            self.assertEqual(
                reopened_field.read(3)["identity"],
                self.cells_module.identity_address(3))
            self.assertGreaterEqual(reopened.generation, generation)

            restored = self.persistence_module.FieldPersistence(reopened)
            self.assertEqual(restored.load_field(), snapshot)
            self.assertEqual(restored.load_scratch(5, 0), scratch_payload)
            self.assertIsNone(restored.load_scratch(5, 9))
            self.assertFalse(restored.status()["canonical_commit"])
        finally:
            reopened.close()


class _ResidentToolBackend:
    """Small real-state backend implementing the field_tools contract."""

    def __init__(self, field, scratch):
        self.field = field
        self.scratch = scratch
        self.main_generation = 0
        self.proposals = {}
        self.sequence = 0
        self.fail_next_commit = False

    @property
    def generation(self):
        return self.main_generation

    def snapshot(self):
        activation = self.field.field_activation()
        return {
            "opened": True,
            "generation": self.main_generation,
            "heads": self.field.read_all(),
            "active_heads": activation["active_indices"],
            "budget": {
                "bytes": UNIVERSE_BYTES,
                "used_bytes": 0,
                "max_bytes": 1024 * MIB,
            },
        }

    def activate(self, head, delta):
        record = self.field.activate(head, delta)
        self.main_generation += 1
        return {
            "accepted": True,
            "generation": self.main_generation,
            "score": record["score"],
            "state": record["state"],
            "active": record["score"] > 0,
        }

    def propose(self, head, delta, reason):
        self.sequence += 1
        token = "proposal-%04d" % self.sequence
        self.scratch.write_layer(
            head, 9, {"delta": delta, "reason": reason,
                      "proposal": token})
        self.proposals[token] = (head, delta)
        return {"token": token, "reasons": ["scratch_only"]}

    def commit(self, token, expected_generation):
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise RuntimeError("simulated main commit failure")
        if expected_generation != self.main_generation:
            return {"accepted": False, "reasons": ["stale_generation"],
                    "generation": self.main_generation}
        staged = self.proposals.pop(token, None)
        if staged is None:
            return {"accepted": False, "reasons": ["unknown_token"],
                    "generation": self.main_generation}
        head, delta = staged
        self.field.activate(head, delta)
        self.main_generation += 1
        return {
            "accepted": True,
            "generation": self.main_generation,
            "head": head,
            "delta": delta,
        }

    def head_metrics(self, head):
        return self.field.read(head)

    def field_metrics(self):
        activation = self.field.field_activation()
        return {
            "head_count": 256,
            "active_heads": activation["active_indices"],
            "active_count": activation["active_heads"],
            "coverage": activation["active_fraction"],
            "generation": self.main_generation,
        }

    def resonance(self, text, top_k):
        seed = sum(text.encode("utf-8")) % 256
        return [
            {"head": (seed + index) % 256,
             "score": round(1.0 / (index + 1), 8),
             "text": text[:32]}
            for index in range(min(top_k, 4))
        ]


class FieldToolApprovalIntegrationTests(unittest.TestCase):
    """Area 6: tool handlers deny missing approval and fail closed."""

    def setUp(self):
        self.tools = load_module("nanlmind_hermes.resident.field_tools")
        self.cells_module = load_module("nanlmind_hermes.resident.cells")
        self.field_module = load_module(
            "nanlmind_hermes.resident.head_field")
        self.scratch_module = load_module(
            "nanlmind_hermes.resident.scratch_layers")
        cells = self.cells_module.ResidentCells(
            bytearray(self.cells_module.REGION_BYTES), initialize=True)
        field = self.field_module.ResidentHeadField(cells)
        scratch = self.scratch_module.ScratchLayers()
        self.backend = _ResidentToolBackend(field, scratch)

    def test_status_and_query_are_read_only_and_non_authoritative(self):
        status = self.tools.status({}, backend=self.backend)
        self.assertTrue(status["read_only"])
        self.assertFalse(status["writes_main"])
        self.assertEqual(status["field"]["head_count"], 256)
        self.assertEqual(status["generation"], 0)
        assert_false_authority(self, status)

        generation = self.backend.generation
        head_query = self.tools.query({"head": 3}, backend=self.backend)
        self.assertEqual(head_query["mode"], "head")
        self.assertEqual(head_query["metrics"]["head"], 3)
        self.assertTrue(head_query["read_only"])
        self.assertEqual(self.backend.generation, generation)
        assert_false_authority(self, head_query)

        text_query = self.tools.query(
            {"text": "penguin", "top_k": 2}, backend=self.backend)
        self.assertEqual(text_query["mode"], "resonance")
        self.assertLessEqual(len(text_query["hits"]), 2)
        self.assertTrue(text_query["read_only"])
        self.assertEqual(self.backend.generation, generation)
        assert_false_authority(self, text_query)

        field_query = self.tools.query({}, backend=self.backend)
        self.assertEqual(field_query["mode"], "field")
        self.assertEqual(field_query["field"]["head_count"], 256)
        self.assertEqual(self.backend.generation, generation)
        assert_false_authority(self, field_query)

        with self.assertRaises(self.tools.FieldValidationError):
            self.tools.query({"head": 3, "text": "penguin"},
                             backend=self.backend)
        with self.assertRaises(self.tools.FieldValidationError):
            self.tools.query({"top_k": 3}, backend=self.backend)
        with self.assertRaises(self.tools.FieldValidationError):
            self.tools.status({"unknown": 1}, backend=self.backend)

    def test_missing_approval_bounds_and_failure_never_mutate(self):
        before = self.backend.field.read(3)
        with self.assertRaises(self.tools.FieldApprovalError):
            self.tools.activate({"head": 3, "delta": 0.5},
                                backend=self.backend)
        with self.assertRaises(self.tools.FieldApprovalError):
            self.tools.activate({"approved": False, "head": 3,
                                 "delta": 0.5}, backend=self.backend)
        for params in (
            {"approved": True, "head": 256, "delta": 0.5},
            {"approved": True, "head": -1, "delta": 0.5},
            {"approved": True, "head": 3, "delta": 2.0},
            {"approved": True, "head": 3, "delta": float("nan")},
            {"approved": True, "head": 3, "delta": 0.5, "extra": 1},
        ):
            with self.assertRaises(self.tools.FieldValidationError):
                self.tools.activate(params, backend=self.backend)
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)

        proposed = self.tools.propose(
            {"head": 3, "delta": 0.5, "reason": "integration"},
            backend=self.backend)
        self.assertTrue(proposed["scratch_only"])
        self.assertFalse(proposed["writes_main"])
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)
        token = proposed["token"]
        self.assertIsNotNone(self.backend.scratch.read_value(3, 9))

        with self.assertRaises(self.tools.FieldApprovalError):
            self.tools.commit({"token": token, "expected_generation": 0},
                              backend=self.backend)
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)

        stale = self.tools.commit(
            {"approved": True, "token": token, "expected_generation": 7},
            backend=self.backend)
        self.assertFalse(stale["accepted"])
        self.assertTrue(stale["denied"])
        self.assertFalse(stale["writes_main"])
        self.assertEqual(self.backend.generation, 0)
        self.assertEqual(self.backend.field.read(3), before)

        committed = self.tools.commit(
            {"approved": True, "token": token, "expected_generation": 0},
            backend=self.backend)
        self.assertTrue(committed["accepted"])
        self.assertTrue(committed["writes_main"])
        self.assertEqual(committed["generation"], 1)
        self.assertAlmostEqual(self.backend.field.read(3)["score"], 0.5,
                               places=6)
        assert_false_authority(self, committed)

        replay = self.tools.commit(
            {"approved": True, "token": token, "expected_generation": 1},
            backend=self.backend)
        self.assertFalse(replay["accepted"])
        self.assertEqual(self.backend.generation, 1)
        self.assertAlmostEqual(self.backend.field.read(3)["score"], 0.5,
                               places=6)

        second = self.tools.propose(
            {"head": 3, "delta": 0.25, "reason": "fail closed"},
            backend=self.backend)
        self.backend.fail_next_commit = True
        score_before = self.backend.field.read(3)["score"]
        with self.assertRaises(self.tools.FieldBackendError):
            self.tools.commit(
                {"approved": True, "token": second["token"],
                 "expected_generation": 1}, backend=self.backend)
        self.assertEqual(self.backend.generation, 1)
        self.assertEqual(self.backend.field.read(3)["score"], score_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
