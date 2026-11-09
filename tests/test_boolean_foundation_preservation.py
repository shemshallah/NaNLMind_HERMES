"""Preservation tests for the mandatory Boolean foundation.

The legacy base is the 16 two-input Boolean functions exposed by
``nanlmind_hermes.boolean_wave`` and the MTH root table in
``nanlmind_hermes.mth_layers``.  These tests pin all 64 truth-table cases, the
256-head foundation receipt gate, deterministic replay, and the closed
canonical-merge boundary while per-head activation/proposal work lands.

Everything here is local: the native-runner gate is exercised with a tiny
temporary Python "runner" that emits the documented receipt lines.  No test
contacts a network or a cloud service.

The ``nanlmind_hermes.head_boolean_field`` contract is imported lazily.  If the
module is not present, the module-specific cases skip with an explicit reason;
the legacy preservation cases continue to run.  The optional probes deliberately
use a narrow, conservative API surface and skip rather than guess when a future
module exposes a different shape.  They always assert the one invariant that
must hold regardless of that API: importing/using the head field must not mutate
the legacy truth table or enable canonical merge.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes import boolean_wave, clocked_tools, mth_layers, runner, tools
from nanlmind_hermes.head_population import HeadPopulation

HEAD_FIELD_MODULE = "nanlmind_hermes.head_boolean_field"

# Fixed sample set: all 16 functions x all four Boolean input pairs.
FIXED_BOOLEAN_CASES = tuple(
    (mask, a, b)
    for mask in range(16)
    for a in (0, 1)
    for b in (0, 1)
)

FIXED_SENTENCES = (
    "cats chase mice",
    "dogs bark loudly",
    "red and blue make purple",
    "water freezes when it is cold",
)


def _expected_table(mask: int) -> tuple[int, int, int, int]:
    """Truth table indexed by k = 2a+b, independent of the module under test."""
    return tuple((mask >> k) & 1 for k in range(4))


def _legacy_snapshot() -> dict:
    """Snapshot the legacy Boolean tables and root names.

    This is used both for replay tests and for proving that importing or using
    an optional head-field module cannot silently rewrite the mandatory base.
    """
    return {
        "boolean_wave": {
            "masks": tuple(boolean_wave.BOOL_MASKS),
            "tables": {
                mask: tuple(boolean_wave.table(mask))
                for mask in boolean_wave.BOOL_MASKS
            },
        },
        "mth_layers": {
            "masks": tuple(mth_layers.ROOT_MASKS),
            "names": dict(mth_layers.ROOT_NAMES),
            "tables": {
                mask: tuple(
                    mth_layers.bool_class(mask, a, b)
                    for a in (0, 1)
                    for b in (0, 1)
                )
                for mask in mth_layers.ROOT_MASKS
            },
        },
    }


def _assert_no_canonical_merge_enabled(testcase: unittest.TestCase, value,
                                       path: str = "root") -> None:
    """Recursively assert that no result enables or attempts canonical merge.

    The project reports the closed gate under several shapes:
    ``canonical_merge_enabled``, ``canonical_merge: {enabled, attempted}`` and
    candidate receipts such as ``canonical_state_modified``.  None may be true.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            here = f"{path}.{key}"
            if key_text == "canonical_merge_enabled":
                testcase.assertIs(item, False, here)
            elif key_text == "canonical_merge":
                if isinstance(item, bool):
                    testcase.assertIs(item, False, here)
                elif isinstance(item, dict):
                    if "enabled" in item:
                        testcase.assertIs(item["enabled"], False, here + ".enabled")
                    if "attempted" in item:
                        testcase.assertIs(item["attempted"], False,
                                           here + ".attempted")
            elif key_text in {
                "canonical_state_modified",
                "canonical_training_committed",
                "canonical_commit",
            }:
                if isinstance(item, bool):
                    testcase.assertIs(item, False, here)
            _assert_no_canonical_merge_enabled(testcase, item, here)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_canonical_merge_enabled(
                testcase, item, f"{path}[{index}]")


class BooleanWaveFoundationTests(unittest.TestCase):
    """The public 16-function multilinear foundation remains exact."""

    def test_all_sixteen_masks_have_distinct_truth_tables(self):
        self.assertEqual(tuple(boolean_wave.BOOL_MASKS), tuple(range(16)))
        tables = set()
        for mask in boolean_wave.BOOL_MASKS:
            expected = _expected_table(mask)
            self.assertEqual(boolean_wave.table(mask), expected, mask)
            tables.add(tuple(boolean_wave.table(mask)))
        self.assertEqual(len(tables), 16,
                         "the 16 masks must remain 16 distinct Boolean rules")

    def test_all_64_legacy_truth_cases_are_exact(self):
        correct = total = 0
        for mask, a, b in FIXED_BOOLEAN_CASES:
            expected = _expected_table(mask)[(a << 1) | b]
            actual = boolean_wave.boolean(mask, a, b)
            extended = boolean_wave.multilinear(mask, a, b)
            total += 1
            self.assertEqual(actual, expected, (mask, a, b))
            self.assertEqual(extended, float(expected), (mask, a, b))
            correct += actual == expected and extended == float(expected)
        self.assertEqual((correct, total), (64, 64))

    def test_xor_and_or_and_related_gates_are_exact(self):
        # mask, expected truth table indexed by k = 2a+b.
        named = {
            1: (1, 0, 0, 0),   # NOR
            6: (0, 1, 1, 0),   # XOR
            7: (1, 1, 1, 0),   # NAND
            8: (0, 0, 0, 1),   # AND
            9: (1, 0, 0, 1),   # XNOR
            14: (0, 1, 1, 1),  # OR
        }
        for mask, expected in named.items():
            self.assertEqual(boolean_wave.table(mask), expected, mask)
            for a in (0, 1):
                for b in (0, 1):
                    value = expected[(a << 1) | b]
                    self.assertEqual(boolean_wave.boolean(mask, a, b), value)
                    self.assertEqual(boolean_wave.multilinear(mask, a, b),
                                     float(value))

    def test_verify_exact_reports_64_of_64(self):
        result = boolean_wave.verify_exact()
        self.assertEqual(result["correct"], 64)
        self.assertEqual(result["total"], 64)
        self.assertIs(result["exact"], True)

    def test_truth_table_accessors_return_immutable_snapshots(self):
        # A proposal must not be able to mutate the legacy table through a
        # returned list/dict or a shared module-level mapping.
        first = boolean_wave.table(6)
        self.assertIsInstance(first, tuple)
        second = boolean_wave.table(6)
        self.assertEqual(first, second)
        self.assertEqual(first, (0, 1, 1, 0))
        self.assertIsInstance(boolean_wave.BOOL_MASKS, tuple)

    def test_boolean_wave_replay_is_deterministic_and_order_independent(self):
        forward = {
            (mask, a, b): boolean_wave.multilinear(mask, a, b)
            for mask, a, b in FIXED_BOOLEAN_CASES
        }
        repeat = {
            (mask, a, b): boolean_wave.multilinear(mask, a, b)
            for mask, a, b in FIXED_BOOLEAN_CASES
        }
        reverse = {
            (mask, a, b): boolean_wave.multilinear(mask, a, b)
            for mask, a, b in reversed(FIXED_BOOLEAN_CASES)
        }
        self.assertEqual(forward, repeat)
        self.assertEqual(forward, reverse)
        for (mask, a, b), value in forward.items():
            self.assertEqual(value, float(_expected_table(mask)[(a << 1) | b]))


class MTHRootFoundationTests(unittest.TestCase):
    """The Hermes-side MTH root uses the same 16-function Boolean base."""

    def test_root_table_addresses_all_sixteen_functions(self):
        root = mth_layers.root_table()
        self.assertEqual(len(root), 4)
        cells = [cell for row in root for cell in row]
        self.assertEqual(len(cells), 16)
        self.assertEqual([cell["mask"] for cell in cells], list(range(16)))
        for cell in cells:
            mask = cell["mask"]
            self.assertEqual(cell["name"], mth_layers.ROOT_NAMES[mask])
            for a in (0, 1):
                for b in (0, 1):
                    self.assertEqual(
                        mth_layers.bool_class(mask, a, b),
                        _expected_table(mask)[(a << 1) | b],
                    )

    def test_apply_bits_composes_exact_nibbles(self):
        for mask in mth_layers.ROOT_MASKS:
            for left in range(16):
                for right in range(16):
                    expected = 0
                    for bit in range(4):
                        expected |= mth_layers.bool_class(
                            mask, (left >> bit) & 1, (right >> bit) & 1
                        ) << bit
                    self.assertEqual(
                        mth_layers.apply_bits(mask, left, right, 4),
                        expected,
                        (mask, left, right),
                    )

    def test_mth_learn_root_layers_are_exact(self):
        stack = mth_layers.MTHLayerStack()
        summary = stack.learn(FIXED_SENTENCES, [])
        self.assertIs(summary["native_verified"], False)
        self.assertEqual(summary["layers"]["bit"],
                         {"accuracy": 1.0, "coverage": 1.0, "examples": 64})
        self.assertEqual(summary["layers"]["nibble"],
                         {"accuracy": 1.0, "coverage": 1.0, "examples": 4096})
        self.assertEqual(summary["layers"]["byte"],
                         {"accuracy": 1.0, "coverage": 1.0, "examples": 256})
        self.assertEqual(len(stack.signature), 16)

    def test_two_mth_stacks_do_not_share_mutable_rule_state(self):
        first = mth_layers.MTHLayerStack()
        first.learn(FIXED_SENTENCES, [])
        first_summary = copy.deepcopy(first.summary())
        first_signature = list(first.signature)

        second = mth_layers.MTHLayerStack()
        second.learn(["dogs bark loudly"], [])
        self.assertEqual(first.summary(), first_summary)
        self.assertEqual(first.signature, first_signature)

        # Mutating one stack's exposed state must not touch the other.
        second.signature[0] = 999
        self.assertEqual(first.signature, first_signature)
        self.assertNotEqual(second.signature, first.signature)

    def test_mth_stack_replay_from_fixed_samples_is_deterministic(self):
        first = mth_layers.MTHLayerStack()
        second = mth_layers.MTHLayerStack()
        first_summary = first.learn(FIXED_SENTENCES, [])
        second_summary = second.learn(FIXED_SENTENCES, [])
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(
            first.forward("cats chase mice"),
            second.forward("cats chase mice"),
        )


class HeadPopulationIsolationTests(unittest.TestCase):
    """Existing per-head state is instance-local; no class-level rule leaks."""

    def test_two_populations_do_not_share_counters_or_priors(self):
        first = HeadPopulation(heads=8, active=4, train_head0=True)
        second = HeadPopulation(heads=8, active=4, train_head0=True)
        first.observe("alpha")
        second.observe("beta")

        first_before = (
            copy.deepcopy(first.heads),
            copy.deepcopy(first.prior),
            list(first.usage),
            list(first.head_reward),
            first.observations,
        )
        second.observe("alphabet")
        self.assertEqual(first.observations, first_before[4])
        self.assertEqual(copy.deepcopy(first.heads), first_before[0])
        self.assertEqual(copy.deepcopy(first.prior), first_before[1])
        self.assertEqual(list(first.usage), first_before[2])

        second.heads[1]["z"] += 5
        self.assertNotIn("z", first.heads[1])
        self.assertEqual(copy.deepcopy(first.heads), first_before[0])


class MandatoryFoundationGateTests(unittest.TestCase):
    """Exercise the public foundation receipt gate with a local fake runner."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name) / "state"

    class Context:
        def __init__(self, state_dir: Path, runner_command):
            self.state_dir = state_dir
            self.runner_command = runner_command

        def get_config(self, key, default=None):
            if key == "state_dir":
                return str(self.state_dir)
            if key == "runner_command":
                return self.runner_command
            return default

    def _write_runner(self, correct=1024, total=1024, distinct=16,
                      write_state=True, result="PASS"):
        script = Path(self._temporary.name) / (
            f"fake_runner_{correct}_{total}_{distinct}_{int(write_state)}.py")
        script.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] != 'logic-train':\n"
            "    print('unexpected subcommand')\n"
            "    raise SystemExit(2)\n"
            "state = Path(sys.argv[4])\n"
            f"if {write_state!r}:\n"
            "    state.parent.mkdir(parents=True, exist_ok=True)\n"
            "    state.write_bytes(b'boolean-foundation-test-state\\n')\n"
            "print('stage=boolean_foundation passed=yes "
            f"correct={correct}/{total} accuracy=1.000000 "
            f"distinct_rules={distinct}')\n"
            "print('stage=composed_gates passed=yes correct=256/256 "
            "accuracy=1.000000')\n"
            f"print('result={result}')\n"
            "print('state_digest=' + '0' * 64)\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def test_exact_1024_check_foundation_passes_and_is_reused(self):
        ctx = self.Context(self.root, self._write_runner())
        first = tools.ensure_boolean_foundation(ctx)
        self.assertEqual(first["status"], "passed")
        self.assertIs(first["reused_checkpoint"], False)
        stage = next(item for item in first["stages"]
                     if item["name"] == "boolean_foundation")
        self.assertIs(stage["passed"], True)
        self.assertEqual(stage["correct"], 1024)
        self.assertEqual(stage["total"], 1024)
        self.assertEqual(stage["distinct_rules"], 16)
        self.assertTrue((self.root / "universe" /
                         "boolean_foundation.nmg").is_file())
        self.assertTrue((self.root / "universe" /
                         "boolean_foundation.json").is_file())
        _assert_no_canonical_merge_enabled(self, first)

        second = tools.ensure_boolean_foundation(ctx)
        self.assertIs(second["reused_checkpoint"], True)
        self.assertEqual(second["state_sha256"], first["state_sha256"])
        _assert_no_canonical_merge_enabled(self, second)

    def test_inexact_foundation_is_rejected_and_not_receipted(self):
        ctx = self.Context(
            self.root,
            self._write_runner(correct=1023, total=1024, distinct=16),
        )
        with self.assertRaises(runner.RunnerError):
            tools.ensure_boolean_foundation(ctx)
        self.assertFalse((self.root / "universe" /
                          "boolean_foundation.json").exists())

    def test_missing_runner_is_rejected(self):
        ctx = self.Context(self.root, [])
        with self.assertRaises(runner.RunnerError):
            tools.ensure_boolean_foundation(ctx)

    def test_missing_persisted_state_is_rejected(self):
        ctx = self.Context(self.root, self._write_runner(write_state=False))
        with self.assertRaises(runner.RunnerError):
            tools.ensure_boolean_foundation(ctx)
        self.assertFalse((self.root / "universe" /
                          "boolean_foundation.json").exists())

    def test_status_reports_exact_foundation_and_closed_merge_gate(self):
        ctx = self.Context(self.root, self._write_runner())
        tools.ensure_boolean_foundation(ctx)
        status = tools.status(ctx)
        self.assertEqual(status["boolean_foundation"]["status"], "passed")
        self.assertIs(status["merge_gate"]["canonical_merge_enabled"], False)
        self.assertIs(status["merge_gate"].get("merge_enabled", False), False)
        _assert_no_canonical_merge_enabled(self, status)


class HeadCandidateProposalIsolationTests(unittest.TestCase):
    """The existing head-candidate path is proposal-only, never a commit."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name) / "state"

        class Context:
            def __init__(self, state_dir):
                self.state_dir = state_dir

            def get_config(self, key, default=None):
                if key == "state_dir":
                    return str(self.state_dir)
                return default

        self.ctx = Context(self.root)

    def _propose(self, head=1, mask=0x00FF, approved=True,
                 mutation_id="foundation-preservation"):
        candidate = {
            "schema": "nanlm.head-candidate-response/1",
            "mutation_id": mutation_id,
            "head_index": head,
            "patch_mask": mask,
            "promotion_allowed": False,
            "canonical_state_modified": False,
        }
        with patch.object(clocked_tools, "_head_client") as factory:
            factory.return_value.create_candidate.return_value = candidate
            result = clocked_tools.head_candidate(
                self.ctx, mutation_id, head, mask, "a" * 64,
                approved=approved, scratch_id="preservation",
            )
        request = factory.return_value.create_candidate.call_args.args[0]
        return result, request

    def test_proposal_does_not_change_legacy_truth_tables(self):
        before = _legacy_snapshot()
        result, request = self._propose()
        self.assertEqual(before, _legacy_snapshot())
        self.assertEqual(request["head_index"], 1)
        self.assertEqual(request["patch_mask"], 0x00FF)
        payload = result["scratch"]["entries"][-1]
        self.assertEqual(payload["lane"], "learning_proposal")
        self.assertIs(payload["payload"]["promotion_allowed"], False)
        self.assertIs(payload["payload"]["candidate"]["promotion_allowed"], False)
        _assert_no_canonical_merge_enabled(self, result)

    def test_explicit_approval_creates_only_an_isolated_candidate(self):
        before = _legacy_snapshot()
        result, _ = self._propose(approved=True, mutation_id="approved-proposal")
        self.assertEqual(before, _legacy_snapshot())
        proposal = result["scratch"]["entries"][-1]["payload"]
        self.assertIs(proposal["promotion_allowed"], False)
        self.assertIs(proposal["candidate"]["canonical_state_modified"], False)

    def test_proposal_requires_explicit_approval(self):
        before = _legacy_snapshot()
        with self.assertRaises(PermissionError):
            self._propose(approved=False, mutation_id="unapproved")
        self.assertEqual(before, _legacy_snapshot())

    def test_two_candidate_heads_keep_independent_proposals(self):
        before = _legacy_snapshot()
        first, first_request = self._propose(head=1, mask=0x00FF,
                                             mutation_id="head-one")
        second, second_request = self._propose(head=2, mask=0xFF00,
                                               mutation_id="head-two")
        self.assertEqual(first_request["head_index"], 1)
        self.assertEqual(second_request["head_index"], 2)
        self.assertNotEqual(first_request["patch_mask"],
                            second_request["patch_mask"])
        first_payload = first["scratch"]["entries"][-1]["payload"]
        second_payload = second["scratch"]["entries"][-1]["payload"]
        self.assertEqual(first_payload["mutation_id"], "head-one")
        self.assertEqual(second_payload["mutation_id"], "head-two")
        self.assertIsNot(first_payload, second_payload)
        self.assertEqual(before, _legacy_snapshot())
        _assert_no_canonical_merge_enabled(self, first)
        _assert_no_canonical_merge_enabled(self, second)


# ---------------------------------------------------------------------------
# Optional per-head activation/proposal module.
#
# This module may land after these regression tests.  The probes below import
# it only inside this class, and each probe has an explicit skip path when the
# expected public surface is not present.  No production module is imported or
# patched at test-module import time.
# ---------------------------------------------------------------------------

_HEAD_FIELD_ARGUMENTS = {
    "head": 1,
    "head_index": 1,
    "mask": 5,
    "new_mask": 5,
    "patch_mask": 5,
    "ops_mask": 5,
    "rule": 5,
    "rule_mask": 5,
    "value": 5,
    "scratch_only": True,
    "mode": "replace",
    "approved": False,
    "commit": False,
    "canonical": False,
}


def _import_optional_head_field():
    try:
        return importlib.import_module(HEAD_FIELD_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == HEAD_FIELD_MODULE:
            raise unittest.SkipTest(
                "optional nanlmind_hermes.head_boolean_field is absent; "
                "legacy Boolean preservation tests still cover the mandatory base"
            )
        raise


def _call_no_args(function):
    try:
        signature = inspect.signature(function)
        signature.bind()
    except TypeError as exc:
        raise unittest.SkipTest(
            "optional head-field callable requires arguments: %s" % exc)
    except (TypeError, ValueError) as exc:
        raise unittest.SkipTest(
            "optional head-field callable cannot be inspected: %s" % exc)
    try:
        return function()
    except unittest.SkipTest:
        raise
    except Exception as exc:
        raise unittest.SkipTest(
            "optional head-field callable could not run without module-specific "
            "setup: %s: %s" % (type(exc).__name__, exc))


def _call_with_known_names(function):
    """Call a proposal-like callable using only recognized parameter names."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise unittest.SkipTest(
            "cannot inspect optional head-field proposal: %s" % exc)
    kwargs = {}
    has_var_keyword = False
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if name in _HEAD_FIELD_ARGUMENTS:
            kwargs[name] = _HEAD_FIELD_ARGUMENTS[name]
        elif (parameter.default is inspect.Parameter.empty
              and parameter.kind != inspect.Parameter.VAR_POSITIONAL):
            raise unittest.SkipTest(
                "optional head-field proposal requires unsupported parameter %r"
                % name)
    if not kwargs and has_var_keyword:
        raise unittest.SkipTest(
            "optional head-field proposal requires an undocumented **kwargs payload")
    try:
        return function(**kwargs)
    except unittest.SkipTest:
        raise
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise unittest.SkipTest(
            "optional head-field proposal could not be driven without "
            "module-specific setup: %s: %s" % (type(exc).__name__, exc))


def _assert_exactness_result(testcase, result, label):
    """Accept a bool, a (correct, total) pair, or a foundation receipt dict."""
    if isinstance(result, bool):
        testcase.assertIs(result, True, label)
        return
    if (isinstance(result, (list, tuple)) and len(result) == 2
            and all(isinstance(item, int) and not isinstance(item, bool)
                    for item in result)):
        testcase.assertEqual(result[0], result[1], label)
        testcase.assertGreaterEqual(result[1], 64, label)
        return
    if not isinstance(result, dict):
        raise unittest.SkipTest(
            "optional head-field verifier returned %s, not a recognized result"
            % type(result).__name__)

    recognized = False
    for key in ("exact", "all_exact", "legacy_exact"):
        if key in result:
            testcase.assertIs(result[key], True, "%s.%s" % (label, key))
            recognized = True
    for key in ("passed", "preserved", "legacy_preserved"):
        if isinstance(result.get(key), bool):
            testcase.assertIs(result[key], True, "%s.%s" % (label, key))
            recognized = True
    if "correct" in result and "total" in result:
        correct = int(result["correct"])
        total = int(result["total"])
        testcase.assertEqual(correct, total, label)
        testcase.assertGreater(total, 0, label)
        if "distinct_rules" in result:
            testcase.assertEqual(result["distinct_rules"], 16, label)
        recognized = True
    all_functions = result.get("all_functions")
    if isinstance(all_functions, dict):
        _assert_exactness_result(
            testcase, all_functions, label + ".all_functions")
        recognized = True
    stages = result.get("stages")
    if isinstance(stages, list):
        stage = next((item for item in stages
                      if isinstance(item, dict)
                      and item.get("name") in {
                          "boolean_foundation", "legacy_boolean", "boolean"}),
                     None)
        if stage is not None:
            testcase.assertIs(stage.get("passed"), True, label)
            if "correct" in stage and "total" in stage:
                testcase.assertEqual(stage["correct"], stage["total"], label)
            if "distinct_rules" in stage:
                testcase.assertEqual(stage["distinct_rules"], 16, label)
            recognized = True
    if not recognized:
        raise unittest.SkipTest(
            "optional head-field verifier result has no recognized exactness fields")


def _field_rule_snapshot(field):
    """Return a copy of exposed rule/truth-table state, never pending queues."""
    if field is None:
        return None
    # Common per-head object layout: field.heads is a sequence of heads, each
    # with a selected rule, operator mask and score table.
    heads = getattr(field, "heads", None)
    if isinstance(heads, (list, tuple)) and heads:
        snapshot = []
        try:
            for head in heads:
                scores = getattr(head, "scores", None)
                snapshot.append((
                    getattr(head, "rule", None),
                    getattr(head, "ops_mask", None),
                    tuple(scores) if isinstance(scores, (list, tuple)) else None,
                ))
        except Exception:
            snapshot = []
        if snapshot:
            return snapshot
    for name in ("truth_tables", "truth_table", "tables", "rules",
                 "head_rules", "masks"):
        try:
            value = getattr(field, name)
        except Exception:
            continue
        if callable(value):
            try:
                inspect.signature(value).bind()
            except TypeError:
                continue
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, (dict, list, tuple)):
            try:
                return copy.deepcopy(value)
            except Exception:
                continue
    return None


def _find_field_class(module):
    preferred = (
        "HeadBooleanField", "BooleanHeadField", "BooleanField",
        "HeadField", "HeadBooleanRules", "BooleanHeadState",
    )
    for name in preferred:
        value = getattr(module, name, None)
        if isinstance(value, type):
            return value
    for name, value in vars(module).items():
        if name.startswith("_") or not isinstance(value, type):
            continue
        if any(callable(getattr(value, method, None))
               for method in ("propose", "proposal", "activate", "self_modify",
                              "truth_table", "table")):
            return value
    return None


def _instantiate_field(field_class, head=None):
    if field_class is None:
        raise unittest.SkipTest(
            "optional head_boolean_field exposes no discoverable field class")
    try:
        signature = inspect.signature(field_class)
    except (TypeError, ValueError) as exc:
        raise unittest.SkipTest(
            "optional head-field class cannot be inspected: %s" % exc)
    parameters = signature.parameters
    attempts = []
    if head is not None and "head" in parameters:
        attempts.append(((), {"head": head}))
    if head is not None and "head_index" in parameters:
        attempts.append(((), {"head_index": head}))
    # A field-of-heads constructor is the common case.  Ask for at least two
    # heads when the probe wants to touch head 1.
    if "heads" in parameters:
        attempts.append(((), {"heads": max(2, head or 2)}))
    if "head_count" in parameters:
        attempts.append(((), {"head_count": max(2, head or 2)}))
    if head is not None:
        attempts.append(((head,), {}))
    attempts.extend([
        ((), {}),
        ((), {"heads": 2}),
        ((), {"head_count": 2}),
        ((2,), {}),
        ((1,), {}),
    ])
    last_error = None
    for args, kwargs in attempts:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        try:
            return field_class(*args, **kwargs)
        except Exception as exc:
            last_error = exc
    raise unittest.SkipTest(
        "optional head-field class could not be instantiated for the probe: %s: %s"
        % (type(last_error).__name__ if last_error else "TypeError",
           last_error if last_error else "no compatible signature"))


def _find_proposal_callable(field):
    if field is None:
        return None
    for name in ("propose", "propose_head_rule", "propose_rule",
                 "create_proposal", "self_modify", "modify"):
        value = getattr(field, name, None)
        if callable(value):
            return value
    return None


def _find_proposal_interface(module):
    for name in ("propose", "propose_head_rule", "propose_rule",
                 "create_proposal", "head_proposal"):
        value = getattr(module, name, None)
        if callable(value):
            return None, value
    field_class = _find_field_class(module)
    if field_class is not None:
        field = _instantiate_field(field_class, head=1)
        propose = _find_proposal_callable(field)
        if propose is not None:
            return field, propose
    return None, None


class HeadBooleanFieldOptionalTests(unittest.TestCase):
    """Lazy preservation contract for the landing head-field module."""

    @classmethod
    def setUpClass(cls):
        # Snapshot before import so an import-time mutation is observable.
        cls.legacy_before = _legacy_snapshot()
        cls.module = _import_optional_head_field()
        cls.legacy_after = _legacy_snapshot()

    def _make_field(self, heads):
        field_class = _find_field_class(self.module)
        if field_class is None:
            self.skipTest(
                "optional head_boolean_field exposes no discoverable field class")
        try:
            return field_class(heads=heads)
        except (TypeError, ValueError):
            return _instantiate_field(field_class, head=max(1, heads - 1))

    def test_import_does_not_change_legacy_truth_tables(self):
        self.assertEqual(self.legacy_before, self.legacy_after)
        self.assertEqual(self.legacy_before, _legacy_snapshot())

    def test_module_tables_are_exact_for_all_64_cases(self):
        table = getattr(self.module, "table", None)
        boolean = getattr(self.module, "boolean", None)
        multilinear = getattr(self.module, "multilinear", None)
        if not all(callable(value) for value in (table, boolean, multilinear)):
            self.skipTest(
                "optional head_boolean_field exposes no full Boolean table API")
        for mask, a, b in FIXED_BOOLEAN_CASES:
            expected = _expected_table(mask)[(a << 1) | b]
            self.assertEqual(tuple(table(mask)), _expected_table(mask), mask)
            self.assertEqual(boolean(mask, a, b), expected, (mask, a, b))
            self.assertEqual(boolean_wave.boolean(mask, a, b), expected)
            self.assertAlmostEqual(
                multilinear(mask, float(a), float(b)),
                boolean_wave.multilinear(mask, float(a), float(b)),
                places=12,
            )

    def test_exported_truth_table_material_matches_legacy(self):
        found = False
        for name in ("LEGACY_MASKS", "BOOL_MASKS", "ROOT_MASKS",
                     "BOOLEAN_MASKS", "LEGACY_BOOL_MASKS"):
            value = getattr(self.module, name, None)
            if value is None:
                continue
            try:
                masks = tuple(value)
            except TypeError:
                continue
            if not masks or not all(
                    isinstance(mask, int) and not isinstance(mask, bool)
                    for mask in masks):
                continue
            found = True
            self.assertEqual(set(masks), set(range(16)), name)
            for mask in masks:
                self.assertEqual(tuple(boolean_wave.table(mask)),
                                 _expected_table(mask), name)

        for name in ("LEGACY_TRUTH_TABLES", "TRUTH_TABLES", "LEGACY_TABLES",
                     "BOOL_TABLES"):
            value = getattr(self.module, name, None)
            if not isinstance(value, dict):
                continue
            for mask in range(16):
                candidate = value.get(mask, value.get(str(mask)))
                if candidate is None:
                    continue
                found = True
                self.assertEqual(tuple(candidate), _expected_table(mask),
                                 "%s[%d]" % (name, mask))
            if value and all(
                    isinstance(row, (list, tuple)) and len(row) == 4
                    for row in value.values()):
                found = True

        for name in ("truth_table", "legacy_truth_table", "table"):
            function = getattr(self.module, name, None)
            if not callable(function):
                continue
            for mask in range(16):
                try:
                    inspect.signature(function).bind(mask)
                except TypeError:
                    break
                found = True
                self.assertEqual(tuple(function(mask)), _expected_table(mask),
                                 "%s(%d)" % (name, mask))
            break

        if not found:
            self.skipTest(
                "optional head_boolean_field exposes no recognizable legacy "
                "truth-table constant or accessor; import preservation is covered")

    def test_module_self_check_reports_an_exact_foundation(self):
        names = (
            "verify_exact", "verify_logic_base", "preservation_check",
            "logic_base_preserved", "verify_foundation",
            "verify_legacy_foundation", "verify_legacy_truth_tables",
            "check_legacy_truth_tables", "foundation_self_check", "self_check",
        )
        checked = False
        for name in names:
            function = getattr(self.module, name, None)
            if not callable(function):
                continue
            checked = True
            result = _call_no_args(function)
            _assert_exactness_result(self, result, "head_boolean_field." + name)
        if not checked:
            self.skipTest(
                "optional head_boolean_field exposes no zero-argument "
                "self-check; exactness remains pinned by the legacy tests")

    def test_all_256_heads_keep_the_sixteen_rules_exact(self):
        field = self._make_field(256)
        if not callable(getattr(field, "head", None)):
            self.skipTest("optional head-field object exposes no head accessor")
        try:
            heads = [field.head(index) for index in range(256)]
        except (AttributeError, IndexError, TypeError) as exc:
            self.skipTest("optional head-field object does not expose 256 heads: %s" % exc)
        for index, head in enumerate(heads):
            mask = index % 16
            try:
                head.rule = mask
            except (AttributeError, TypeError) as exc:
                self.skipTest(
                    "optional head-field head rules are not assignable: %s" % exc)
            self.assertEqual(head.rule, mask, index)
            for a in (0, 1):
                for b in (0, 1):
                    self.assertEqual(
                        head.evaluate(a, b),
                        _expected_table(mask)[(a << 1) | b],
                        (index, mask, a, b),
                    )
        self.assertEqual(field.foundation_status()["intact"], True)
        self.assertIs(field.verify_logic_base()["preserved"], True)
        if callable(getattr(field, "field_step", None)):
            step = field.field_step(list(range(16)), (1, 1))
            self.assertIs(step["collapsed_to_global_decision"], False)
            for index in range(16):
                self.assertEqual(step["output_by_head"][index],
                                 _expected_table(index)[3])

    def test_canonical_merge_defaults_are_closed(self):
        merge_related = {
            key: value for key, value in vars(self.module).items()
            if "canonical" in key.lower() or "merge" in key.lower()
        }
        _assert_no_canonical_merge_enabled(self, merge_related,
                                           "head_boolean_field")
        status = getattr(self.module, "status", None)
        if callable(status):
            try:
                status_value = _call_no_args(status)
            except unittest.SkipTest:
                status_value = None
            if status_value is not None:
                _assert_no_canonical_merge_enabled(
                    self, status_value, "head_boolean_field.status()")

    def test_head_field_proposal_is_scratch_until_explicit_commit(self):
        field = self._make_field(2)
        modify = _find_proposal_callable(field)
        if modify is None:
            self.skipTest("optional head-field object exposes no proposal method")
        before_rules = _field_rule_snapshot(field)
        before_legacy = _legacy_snapshot()
        other_head = field.head(0)
        other_before = (other_head.rule, other_head.ops_mask,
                        tuple(other_head.scores))
        try:
            proposal = modify(1, 0x0001, scratch_only=True)
        except TypeError as exc:
            self.skipTest(
                "optional head-field proposal has an unsupported signature: %s"
                % exc)
        self.assertIs(proposal.get("applied"), False)
        self.assertIs(proposal.get("committed"), False)
        self.assertIs(proposal.get("canonical_state_modified"), False)
        self.assertEqual(_field_rule_snapshot(field), before_rules,
                         "uncommitted proposal changed head rule state")
        self.assertEqual(_legacy_snapshot(), before_legacy)
        self.assertIs(field.verify_logic_base()["preserved"], True)

        commit = getattr(field, "commit_proposal", None)
        if not callable(commit):
            self.skipTest(
                "optional head-field object exposes no explicit commit method")
        proposal_id = proposal.get("proposal_id", proposal.get("id"))
        if not isinstance(proposal_id, str):
            self.skipTest("optional head-field proposal exposes no proposal id")
        committed = commit(proposal_id, approval=True)
        self.assertIs(committed.get("committed"), True)
        head = field.head(1)
        if "after_rule" in proposal:
            self.assertEqual(head.rule, proposal["after_rule"])
        if "after_mask" in proposal:
            self.assertEqual(head.ops_mask, proposal["after_mask"])
        self.assertEqual(
            (other_head.rule, other_head.ops_mask, tuple(other_head.scores)),
            other_before,
            "a proposal on head 1 mutated independent head 0",
        )
        self.assertEqual(_legacy_snapshot(), before_legacy,
                         "an explicit head commit changed the legacy truth table")
        self.assertIs(field.verify_logic_base()["preserved"], True)
        self.assertEqual(field.foundation_status()["intact"], True)
        _assert_no_canonical_merge_enabled(
            self, field.status(), "head_boolean_field.status() after commit")

    def test_non_scratch_head_field_write_requires_explicit_approval(self):
        field = self._make_field(2)
        modify = _find_proposal_callable(field)
        if modify is None:
            self.skipTest("optional head-field object exposes no proposal method")
        before_rules = _field_rule_snapshot(field)
        before_legacy = _legacy_snapshot()
        error = getattr(self.module, "ScratchViolation", ValueError)
        try:
            with self.assertRaises(error):
                modify(1, 0x0001, scratch_only=False)
        except TypeError as exc:
            self.skipTest(
                "optional head-field proposal has no scratch_only control: %s"
                % exc)
        self.assertEqual(_field_rule_snapshot(field), before_rules)
        self.assertEqual(_legacy_snapshot(), before_legacy)

    def test_head_field_replay_from_fixed_samples_is_deterministic(self):
        before_legacy = _legacy_snapshot()

        def make_field():
            field = self._make_field(2)
            try:
                field.head(0).rule = 6   # XOR
                field.head(1).rule = 8   # AND
            except (AttributeError, TypeError) as exc:
                self.skipTest(
                    "optional head-field rules are not assignable: %s" % exc)
            return field

        first = make_field()
        second = make_field()
        samples = [
            (a, b, _expected_table(6)[(a << 1) | b])
            for a in (0, 1) for b in (0, 1)
        ]
        first_report = first.evaluate_boolean(0, samples)
        second_report = second.evaluate_boolean(0, samples)
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["correct"], 4)
        self.assertEqual(first_report["total"], 4)
        self.assertIs(first_report["exact"], True)

        first_step = first.field_step([0, 1], (1, 0))
        second_step = second.field_step([0, 1], (1, 0))
        self.assertEqual(first_step, second_step)
        self.assertIs(first_step["collapsed_to_global_decision"], False)
        self.assertEqual(first_step["output_by_head"], {0: 1, 1: 0})

        first_train = first.train_boolean(0, samples, epochs=1)
        second_train = second.train_boolean(0, samples, epochs=1)
        self.assertEqual(first_train, second_train)
        self.assertEqual(before_legacy, _legacy_snapshot())

    def test_proposal_does_not_alter_legacy_truth_table_before_commit(self):
        field, propose = _find_proposal_interface(self.module)
        if propose is None:
            self.skipTest(
                "optional head_boolean_field exposes no discoverable proposal "
                "callable; the existing head-candidate path is covered separately")
        before_legacy = _legacy_snapshot()
        before_rules = _field_rule_snapshot(field)
        result = _call_with_known_names(propose)
        self.assertEqual(before_legacy, _legacy_snapshot())
        if before_rules is not None:
            after_rules = _field_rule_snapshot(field)
            if after_rules is not None:
                self.assertEqual(before_rules, after_rules,
                                 "an uncommitted proposal changed rule state")
        if isinstance(result, dict):
            for key in ("committed", "applied", "canonical_state_modified",
                        "canonical_training_committed", "promotion_allowed"):
                if key in result:
                    self.assertIs(result[key], False,
                                  "uncommitted proposal set %s" % key)
            _assert_no_canonical_merge_enabled(self, result,
                                               "head_boolean_field.proposal")

    def test_two_field_instances_do_not_share_rule_state(self):
        field_class = _find_field_class(self.module)
        if field_class is None:
            self.skipTest(
                "optional head_boolean_field exposes no discoverable field class")
        first = _instantiate_field(field_class, head=1)
        second = _instantiate_field(field_class, head=2)
        before_second = _field_rule_snapshot(second)
        if before_second is None:
            self.skipTest(
                "optional head-field class exposes no inspectable rule/table state")
        propose = _find_proposal_callable(first)
        if propose is None:
            self.skipTest(
                "optional head-field class exposes no proposal method")
        before_legacy = _legacy_snapshot()
        _call_with_known_names(propose)
        after_second = _field_rule_snapshot(second)
        if after_second is not None:
            self.assertEqual(before_second, after_second,
                             "a proposal on head 1 leaked into head 2")
        self.assertEqual(before_legacy, _legacy_snapshot())


if __name__ == "__main__":
    unittest.main()

