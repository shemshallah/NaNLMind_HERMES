"""256-head Boolean self-modification field.

This module is a small, dependency-free Python companion to the native
NaNLMind head layer.  It preserves the mandatory Boolean foundation:

* exactly 256 head identities (``0..255``);
* exactly 16 two-input Boolean functions (truth-table masks ``0..15``);
* each head owns an immutable identity and an independent mutable score table
  plus a selected rule and 16-bit operator mask;
* training is integer, winner-take-all score learning over the 16 functions;
* a field step reports every head's independent output *and* a separate,
  bounded field-coupling term.  Heads are never collapsed into one global
  decision;
* self-modification always creates a scratch proposal first.  The canonical
  rule/mask is changed only by an explicit commit token or approval.

The semantics of the legacy ``boolean_wave`` base are kept exact: mask 6 is
XOR, mask 8 is AND and mask 14 is OR.  ``verify_logic_base`` re-checks the
legacy module as well as this module's own truth tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Foundation constants
# ---------------------------------------------------------------------------

HEAD_COUNT = 256
HEADS = HEAD_COUNT
NUM_HEADS = HEAD_COUNT
N_HEADS = HEAD_COUNT
MAX_HEADS = HEAD_COUNT
HEAD_BITS = 8

BOOLEAN_FUNCTION_COUNT = 16
BOOL_FUNCTION_COUNT = BOOLEAN_FUNCTION_COUNT
FUNCTION_COUNT = BOOLEAN_FUNCTION_COUNT
NUM_FUNCTIONS = BOOLEAN_FUNCTION_COUNT
BOOLEAN_MASK_BITS = 16
BOOLEAN_MASK_MAX = (1 << BOOLEAN_MASK_BITS) - 1
BOOLEAN_MASK_ALL = BOOLEAN_MASK_MAX
OPS_MASK_ALL = BOOLEAN_MASK_ALL

# Truth-table masks, indexed by k = 2a + b exactly like the native NaN16 code.
BOOL_MASKS = tuple(range(BOOLEAN_FUNCTION_COUNT))
BOOLEAN_FUNCTIONS = BOOL_MASKS
BOOL_FUNCTIONS = BOOL_MASKS

# Names follow NaNLMind's bool_op_name().
BOOLEAN_FUNCTION_NAMES = (
    "CONST0", "NOR", "ANDN", "NOTA",
    "NADN", "NOTB", "XOR", "NAND",
    "AND", "XNOR", "B", "IMPLA",
    "A", "IMPLB", "OR", "CONST1",
)
BOOLEAN_NAMES = BOOLEAN_FUNCTION_NAMES
FUNCTION_NAMES = BOOLEAN_FUNCTION_NAMES
BOOL_NAMES = BOOLEAN_FUNCTION_NAMES
BOOL_FUNCTION_NAMES = BOOLEAN_FUNCTION_NAMES
BOOL_FUNCTION_MASKS = BOOL_MASKS
FUNCTIONS_BY_NAME = {
    name: mask for mask, name in enumerate(BOOLEAN_FUNCTION_NAMES)
}
FUNCTIONS_BY_NAME.update({
    "ZERO": 0,
    "NOT_A_AND_B": 2,
    "NOT_A": 3,
    "A_AND_NOT_B": 4,
    "NOT_B": 5,
    "A_IMPLIES_B": 11,
    "B_IMPLIES_A": 13,
    "ONE": 15,
})
BOOLEAN_FUNCTIONS_BY_NAME = FUNCTIONS_BY_NAME

# The legacy logic base used by the native curriculum.
XOR = 6
AND = 8
OR = 14
XOR_MASK = 1 << XOR
AND_MASK = 1 << AND
OR_MASK = 1 << OR

SCORE_MIN = -32768
SCORE_MAX = 32767
PROPOSAL_SCHEMA = "nanlm.hermes-head-boolean-proposal/1"


# ---------------------------------------------------------------------------
# Errors.  All are ValueError subclasses so callers can fail closed with the
# same broad exception they use for ordinary argument validation.
# ---------------------------------------------------------------------------

class BooleanFieldError(ValueError):
    """Base error for the Boolean field."""


class UnknownHeadError(BooleanFieldError):
    """A head identifier is not one of the configured 0..255 heads."""


class NonFiniteError(BooleanFieldError):
    """NaN/infinity is never a valid score, sample, operand or mask."""


class MaskError(BooleanFieldError):
    """An operator mask is not an unsigned 16-bit integer."""


class UnsignedMaskError(MaskError):
    """A negative operator mask was supplied."""


class MaskOverflowError(MaskError):
    """An operator mask does not fit in 16 bits."""


class SampleError(BooleanFieldError):
    """A Boolean training/evaluation sample is malformed."""


class InputError(BooleanFieldError):
    """A field-step input is malformed."""


class ScratchViolation(BooleanFieldError):
    """A mutation attempted to leave the scratch area without approval."""


class ApprovalRequired(ScratchViolation):
    """A commit was attempted without a token or approval."""


class ApprovalDenied(ScratchViolation):
    """The supplied approval token was malformed or did not match."""


class UnknownProposalError(BooleanFieldError):
    """The proposal identifier is not in scratch."""


class StaleProposalError(BooleanFieldError):
    """The head changed after the proposal was created."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _Report(dict):
    """A dict whose entries are also readable as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _CallableFloat(float):
    """A float that can also be called, for attribute/method API tolerance."""

    def __call__(self) -> float:
        return float(self)


def _reverse8(value: int) -> int:
    value &= 0xFF
    value = ((value & 0xF0) >> 4) | ((value & 0x0F) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    return ((value & 0xAA) >> 1) | ((value & 0x55) << 1)


def head_address(index: int) -> int:
    """The canonical palindromic 16-bit address for a head index."""
    if type(index) is not int or not 0 <= index < HEAD_COUNT:
        raise UnknownHeadError("head must be an integer 0..255")
    return (index << 8) | _reverse8(index)


def _uint16_mask(mask: Any, name: str = "ops_mask") -> int:
    if isinstance(mask, bool):
        raise MaskError(f"{name} must be an unsigned 16-bit integer")
    if isinstance(mask, float):
        if not math.isfinite(mask):
            raise NonFiniteError(f"{name} must be finite")
        raise MaskError(f"{name} must be an unsigned 16-bit integer")
    if type(mask) is not int:
        raise MaskError(f"{name} must be an unsigned 16-bit integer")
    if mask < 0:
        raise UnsignedMaskError(f"{name} must be unsigned")
    if mask > BOOLEAN_MASK_MAX:
        raise MaskOverflowError(f"{name} overflows 16 bits")
    return mask


def _rule_index(rule: Any, name: str = "rule") -> int:
    if isinstance(rule, bool) or type(rule) is not int:
        raise MaskError(f"{name} must be an integer 0..15")
    if not 0 <= rule < BOOLEAN_FUNCTION_COUNT:
        raise MaskError(f"{name} must be an integer 0..15")
    return rule


def _rule_from_mask(mask: int) -> int:
    """Lowest enabled function, matching native set_boolean_mask()."""
    mask = _uint16_mask(mask)
    if mask == 0:
        return 0
    return (mask & -mask).bit_length() - 1


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if type(value) not in (int, float):
        raise NonFiniteError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteError(f"{name} must be finite")
    return number


def _sample_bit(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if type(value) is int:
        # Native Boolean evaluation consumes the low bit of an integer.
        return value & 1
    if type(value) is float:
        if not math.isfinite(value):
            raise NonFiniteError(f"{name} must be finite")
        if value in (0.0, 1.0):
            return int(value)
        raise SampleError(f"{name} must be Boolean (0 or 1)")
    raise SampleError(f"{name} must be Boolean (0 or 1)")


def _operand(value: Any, name: str = "input") -> int | float:
    """Normalize one field operand; finite fuzzy values remain floats."""
    if isinstance(value, bool):
        return int(value)
    if type(value) is int:
        return value & 1
    if type(value) is float:
        if not math.isfinite(value):
            raise NonFiniteError(f"{name} must be finite")
        if value in (0.0, 1.0):
            return int(value)
        if 0.0 <= value <= 1.0:
            return value
        raise InputError(f"{name} must be in [0, 1]")
    raise InputError(f"{name} must be a finite Boolean or fuzzy value")


def _operand_pair(value: Any, name: str = "input") -> tuple[int | float, int | float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise InputError(f"{name} must be a two-element pair")
    try:
        pair = list(value)
    except TypeError as exc:
        raise InputError(f"{name} must be a two-element pair") from exc
    if len(pair) != 2:
        raise InputError(f"{name} must be exactly two operands")
    return _operand(pair[0], name), _operand(pair[1], name)


def _is_pair(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        return len(value) == 2
    except TypeError:
        return False


def _sample_from(value: Any) -> tuple[int, int, int]:
    """Accept a triple, a nested ((a, b), target), a mapping or an object."""
    if isinstance(value, Mapping):
        if "input" in value:
            a, b = _operand_pair(value["input"], "sample input")
        elif all(key in value for key in ("a", "b")):
            a = _sample_bit(value["a"], "sample a")
            b = _sample_bit(value["b"], "sample b")
        elif all(key in value for key in ("x", "y")):
            a = _sample_bit(value["x"], "sample x")
            b = _sample_bit(value["y"], "sample y")
        else:
            raise SampleError("sample mapping needs a/b or input")
        if type(a) is not int or type(b) is not int:
            raise SampleError("training samples must be Boolean")
        for key in ("target", "expected", "output", "label"):
            if key in value:
                target = _sample_bit(value[key], f"sample {key}")
                break
        else:
            raise SampleError("sample mapping needs target/expected/output")
        return int(a), int(b), target

    if hasattr(value, "a") and hasattr(value, "b"):
        target_value = getattr(value, "target", None)
        if target_value is None:
            target_value = getattr(value, "expected", None)
        if target_value is None:
            raise SampleError("sample object needs target")
        return (_sample_bit(value.a, "sample a"),
                _sample_bit(value.b, "sample b"),
                _sample_bit(target_value, "sample target"))

    if isinstance(value, (str, bytes)):
        raise SampleError("sample must be a triple or mapping")
    try:
        parts = list(value)
    except TypeError as exc:
        raise SampleError("sample must be a triple or mapping") from exc

    if len(parts) == 3:
        return (_sample_bit(parts[0], "sample a"),
                _sample_bit(parts[1], "sample b"),
                _sample_bit(parts[2], "sample target"))
    if len(parts) == 2 and _is_pair(parts[0]):
        a, b = _operand_pair(parts[0], "sample input")
        if type(a) is not int or type(b) is not int:
            raise SampleError("training samples must be Boolean")
        return int(a), int(b), _sample_bit(parts[1], "sample target")
    raise SampleError("sample must be (a, b, target)")


def _is_scalar_sample_value(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _is_sample_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        return True
    if hasattr(value, "a") and hasattr(value, "b"):
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        length = len(value)
    except TypeError:
        return False
    if length == 3:
        try:
            return all(_is_scalar_sample_value(item) for item in value)
        except TypeError:
            return False
    if length == 2:
        try:
            first = value[0]
            return (_is_pair(first)
                    and all(_is_scalar_sample_value(item) for item in first)
                    and _is_scalar_sample_value(value[1]))
        except (TypeError, IndexError):
            return False
    return False


def _normalize_samples(samples: Any, allow_empty: bool = False) -> list[tuple[int, int, int]]:
    if _is_sample_like(samples):
        items: list[Any] = [samples]
    else:
        if isinstance(samples, (str, bytes, Mapping)):
            raise SampleError("samples must be an iterable of Boolean samples")
        try:
            items = list(samples)
        except TypeError as exc:
            raise SampleError("samples must be an iterable of Boolean samples") from exc
    if not items and not allow_empty:
        raise SampleError("at least one Boolean sample is required")
    return [_sample_from(item) for item in items]


def _epochs(value: Any) -> int:
    if isinstance(value, bool):
        raise SampleError("epochs must be a positive integer")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteError("epochs must be finite")
        if not value.is_integer():
            raise SampleError("epochs must be a positive integer")
        value = int(value)
    if type(value) is not int:
        raise SampleError("epochs must be a positive integer")
    if value < 1:
        raise SampleError("epochs must be at least 1")
    if value > 1_000_000:
        raise SampleError("epochs exceeds the bounded maximum")
    return value


# ---------------------------------------------------------------------------
# Exact Boolean base
# ---------------------------------------------------------------------------

def _function_mask(mask: Any) -> int:
    if isinstance(mask, bool) or type(mask) is not int:
        raise MaskError("Boolean function mask must be an integer 0..15")
    if not 0 <= mask < BOOLEAN_FUNCTION_COUNT:
        raise MaskError("Boolean function mask must be an integer 0..15")
    return mask


def table(mask: int) -> tuple[int, int, int, int]:
    """Truth table indexed by k = 2a + b."""
    mask = _function_mask(mask)
    return tuple((mask >> k) & 1 for k in range(4))


truth_table = table


def boolean(mask: int, a: int, b: int) -> int:
    """Exact two-input Boolean function; a and b use their low bits."""
    mask = _function_mask(mask)
    return (mask >> ((int(a) & 1) << 1 | (int(b) & 1))) & 1


eval_boolean = boolean


def multilinear(mask: int, a: float, b: float) -> float:
    """Exact on {0,1}; a fuzzy-logic extension elsewhere in [0,1]."""
    mask = _function_mask(mask)
    a = _finite_number(a, "a")
    b = _finite_number(b, "b")
    f00, f01, f10, f11 = table(mask)
    return (f00 * (1 - a) * (1 - b) + f01 * (1 - a) * b
            + f10 * a * (1 - b) + f11 * a * b)


def function_name(mask: int) -> str:
    return BOOLEAN_FUNCTION_NAMES[_function_mask(mask)]


bool_op_name = function_name


def verify_exact() -> _Report:
    """Check all 16 functions on all four Boolean inputs (64/64)."""
    correct = total = 0
    per_function: dict[str, bool] = {}
    for mask in BOOL_MASKS:
        exact = True
        for a in (0, 1):
            for b in (0, 1):
                total += 1
                expected = (mask >> ((a << 1) | b)) & 1
                got = boolean(mask, a, b)
                if got != expected:
                    exact = False
                else:
                    correct += 1
        per_function[BOOLEAN_FUNCTION_NAMES[mask]] = exact
    return _Report(correct=correct, total=total,
                   exact=correct == total, functions=per_function)


def _legacy_base() -> dict[str, Any] | None:
    """Compare against the untouched legacy boolean_wave module when present."""
    try:
        from . import boolean_wave as legacy
    except Exception:
        return None
    result: dict[str, Any] = {}
    try:
        legacy_exact = legacy.verify_exact()
        result["module"] = "nanlmind_hermes.boolean_wave"
        result["reported_exact"] = bool(legacy_exact.get("exact"))
        result["reported_correct"] = int(legacy_exact.get("correct", 0))
        result["reported_total"] = int(legacy_exact.get("total", 0))
        matches = True
        for mask in BOOL_MASKS:
            for a in (0, 1):
                for b in (0, 1):
                    if legacy.boolean(mask, a, b) != boolean(mask, a, b):
                        matches = False
                    if abs(legacy.multilinear(mask, a, b) - multilinear(mask, a, b)) > 1e-12:
                        matches = False
        result["matches"] = matches
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["matches"] = False
    return result


def verify_logic_base() -> _Report:
    """Preservation check for XOR/AND/OR and the full legacy truth table."""
    expected = {
        "XOR": (XOR, (0, 1, 1, 0)),
        "AND": (AND, (0, 0, 0, 1)),
        "OR": (OR, (0, 1, 1, 1)),
    }
    checks: dict[str, bool] = {}
    correct = total = 0
    for name, (mask, truth) in expected.items():
        exact = True
        for a in (0, 1):
            for b in (0, 1):
                total += 1
                wanted = truth[(a << 1) | b]
                got = boolean(mask, a, b)
                if got != wanted:
                    exact = False
                else:
                    correct += 1
        checks[name] = exact

    all_functions = verify_exact()
    legacy = _legacy_base()
    legacy_ok = bool(legacy and legacy.get("matches"))
    return _Report(
        schema="nanlm.hermes-boolean-preservation/1",
        correct=correct,
        total=total,
        exact=all(checks.values()),
        checks=checks,
        xor_exact=checks["XOR"],
        and_exact=checks["AND"],
        or_exact=checks["OR"],
        all_functions=all_functions,
        functions_exact=bool(all_functions["exact"]),
        legacy=legacy,
        legacy_preserved=legacy_ok,
        preserved=bool(all(checks.values()) and all_functions["exact"] and legacy_ok),
    )


def logic_base_preserved() -> bool:
    return bool(verify_logic_base()["preserved"])


preservation_check = verify_logic_base
LEGACY_MASKS = BOOL_MASKS
BOOLEAN_MASKS = BOOL_MASKS
LEGACY_BOOL_MASKS = BOOL_MASKS
ROOT_MASKS = BOOL_MASKS
TRUTH_TABLES = {mask: table(mask) for mask in BOOL_MASKS}
BOOL_TABLES = TRUTH_TABLES
LEGACY_TRUTH_TABLES = TRUTH_TABLES
LEGACY_TABLES = TRUTH_TABLES
legacy_truth_table = table
verify_foundation = verify_exact
verify_legacy_foundation = verify_logic_base
verify_legacy_truth_tables = verify_logic_base
check_legacy_truth_tables = verify_logic_base
foundation_self_check = verify_exact
self_check = verify_exact


# ---------------------------------------------------------------------------
# Head identity and mutable state
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HeadIdentity:
    """Immutable per-head identity; never rewritten by learning."""

    index: int
    address: int | None = None
    head_id: str | None = None
    implementation: str = "nanlm.boolean-head/1"

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index < HEAD_COUNT:
            raise UnknownHeadError("head index must be an integer 0..255")
        address = head_address(self.index) if self.address is None else self.address
        if type(address) is not int or not 0 <= address <= 0xFFFF:
            raise BooleanFieldError("head address must be a 16-bit integer")
        identity = (f"{self.implementation}:{self.index}:{address}").encode("utf-8")
        head_id = (hashlib.sha256(identity).hexdigest()
                   if self.head_id is None else self.head_id)
        if not isinstance(head_id, str) or not head_id:
            raise BooleanFieldError("head_id must be a non-empty string")
        object.__setattr__(self, "address", address)
        object.__setattr__(self, "head_id", head_id)

    @property
    def head(self) -> int:
        return self.index

    @property
    def id(self) -> str:
        return self.head_id  # type: ignore[return-value]

    @property
    def route(self) -> str:
        return f"HW0:{self.index}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "head": self.index,
            "address": self.address,
            "head_id": self.head_id,
            "implementation": self.implementation,
            "route": self.route,
        }


HeadID = HeadIdentity


@dataclass
class HeadState:
    """Mutable per-head Boolean scores, selected rule and scratch state."""

    scores: list[int] = dataclass_field(
        default_factory=lambda: [0] * BOOLEAN_FUNCTION_COUNT)
    ops_mask: int = BOOLEAN_MASK_ALL
    rule: int = 0
    generation: int = 0
    scratch: dict[str, Any] = dataclass_field(default_factory=dict)
    activation: float = 0.0
    coupled_activation: float = 0.0

    def __post_init__(self) -> None:
        self.scores = self._scores(self.scores)
        self.ops_mask = _uint16_mask(self.ops_mask)
        self.rule = _rule_index(self.rule)
        if isinstance(self.generation, bool) or type(self.generation) is not int:
            raise BooleanFieldError("generation must be a non-negative integer")
        if self.generation < 0:
            raise BooleanFieldError("generation must be a non-negative integer")
        if not isinstance(self.scratch, dict):
            raise ScratchViolation("head scratch must be a mapping")
        self.activation = _finite_number(self.activation, "activation")
        self.coupled_activation = _finite_number(
            self.coupled_activation, "coupled_activation")

    @staticmethod
    def _scores(value: Any) -> list[int]:
        if isinstance(value, (str, bytes, Mapping)):
            raise SampleError("scores must contain 16 integers")
        try:
            scores = list(value)
        except TypeError as exc:
            raise SampleError("scores must contain 16 integers") from exc
        if len(scores) != BOOLEAN_FUNCTION_COUNT:
            raise SampleError("scores must contain exactly 16 integers")
        result: list[int] = []
        for index, score in enumerate(scores):
            if isinstance(score, bool) or type(score) is not int:
                raise NonFiniteError(f"score {index} must be a finite integer")
            if not SCORE_MIN <= score <= SCORE_MAX:
                raise SampleError(f"score {index} is outside int16 range")
            result.append(score)
        return result

    @property
    def learned_mask(self) -> int:
        mask = 0
        for op, score in enumerate(self.scores):
            if score > 0:
                mask |= 1 << op
        return mask

    @property
    def selected_score(self) -> int:
        return self.scores[self.rule]

    def copy(self) -> "HeadState":
        return HeadState(
            scores=list(self.scores),
            ops_mask=self.ops_mask,
            rule=self.rule,
            generation=self.generation,
            scratch=json.loads(json.dumps(self.scratch, default=str)),
            activation=self.activation,
            coupled_activation=self.coupled_activation,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": list(self.scores),
            "ops_mask": self.ops_mask,
            "rule": self.rule,
            "generation": self.generation,
            "scratch": dict(self.scratch),
            "activation": self.activation,
            "coupled_activation": self.coupled_activation,
        }


BooleanHeadState = HeadState


class BooleanHead:
    """One head: immutable identity plus mutable Boolean score/rule state."""

    __slots__ = ("identity", "state")

    def __init__(self, index: int, state: HeadState | None = None):
        self.identity = HeadIdentity(index)
        self.state = state if state is not None else HeadState()

    @property
    def index(self) -> int:
        return self.identity.index

    @property
    def head(self) -> int:
        return self.identity.index

    @property
    def head_id(self) -> str:
        return self.identity.head_id  # type: ignore[return-value]

    @property
    def address(self) -> int:
        return self.identity.address  # type: ignore[return-value]

    @property
    def rule(self) -> int:
        return self.state.rule

    @rule.setter
    def rule(self, value: int) -> None:
        self.state.rule = _rule_index(value)
        self.state.ops_mask = 1 << self.state.rule
        self.state.generation += 1

    @property
    def ops_mask(self) -> int:
        return self.state.ops_mask

    @ops_mask.setter
    def ops_mask(self, value: int) -> None:
        self.state.ops_mask = _uint16_mask(value)
        self.state.rule = _rule_from_mask(self.state.ops_mask)
        self.state.generation += 1

    @property
    def mask(self) -> int:
        return self.state.ops_mask

    @mask.setter
    def mask(self, value: int) -> None:
        self.ops_mask = value

    @property
    def scores(self) -> list[int]:
        return self.state.scores

    @scores.setter
    def scores(self, value: Any) -> None:
        self.state.scores = HeadState._scores(value)

    @property
    def learned_mask(self) -> int:
        return self.state.learned_mask

    @property
    def truth_table(self) -> tuple[int, int, int, int]:
        return table(self.state.rule)

    @property
    def generation(self) -> int:
        return self.state.generation

    @property
    def activation(self) -> float:
        return self.state.activation

    @activation.setter
    def activation(self, value: Any) -> None:
        self.state.activation = _finite_number(value, "activation")

    @property
    def coupled_activation(self) -> float:
        return self.state.coupled_activation

    @coupled_activation.setter
    def coupled_activation(self, value: Any) -> None:
        self.state.coupled_activation = _finite_number(
            value, "coupled_activation")

    def score(self, op: int) -> int:
        return self.state.scores[_rule_index(op, "op")]

    def evaluate(self, a: Any, b: Any) -> int | float:
        return _evaluate_rule(self.state.rule, a, b)

    def snapshot(self) -> _Report:
        return _Report(head=self.index, identity=self.identity.as_dict(),
                       state=self.state.as_dict())

    def __repr__(self) -> str:
        return (f"BooleanHead(index={self.index}, rule={self.rule}, "
                f"ops_mask=0x{self.ops_mask:04X})")


Head = BooleanHead


def _evaluate_rule(rule: int, a: Any, b: Any) -> int | float:
    rule = _rule_index(rule)
    left = _operand(a, "a")
    right = _operand(b, "b")
    if type(left) is int and type(right) is int:
        return boolean(rule, left, right)
    return multilinear(rule, float(left), float(right))


# ---------------------------------------------------------------------------
# The 256-head field
# ---------------------------------------------------------------------------

def _normalize_active_heads(active_heads: Any, head_count: int) -> list[int]:
    if active_heads is None:
        return list(range(head_count))
    if isinstance(active_heads, bool):
        raise UnknownHeadError("head must be an integer 0..255")
    if type(active_heads) is int:
        raw = [active_heads]
    else:
        if isinstance(active_heads, (str, bytes, Mapping)):
            raise UnknownHeadError("active_heads must be an iterable of head indices")
        try:
            raw = list(active_heads)
        except TypeError as exc:
            raise UnknownHeadError(
                "active_heads must be an iterable of head indices") from exc
        if isinstance(active_heads, (set, frozenset)):
            raw = sorted(raw)
    seen: set[int] = set()
    result: list[int] = []
    for item in raw:
        if type(item) is not int or not 0 <= item < head_count:
            raise UnknownHeadError(
                f"unknown head {item!r}; expected an integer 0..{head_count - 1}")
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_field_inputs(heads: Sequence[int], inputs: Any) -> list[tuple[Any, Any]]:
    if not heads:
        return []
    if isinstance(inputs, Mapping):
        result = []
        for head in heads:
            if head in inputs:
                key = head
            elif str(head) in inputs:
                key = str(head)
            else:
                raise InputError(f"missing input for head {head}")
            result.append(_operand_pair(inputs[key], f"input for head {head}"))
        return result

    if inputs is None or isinstance(inputs, (str, bytes)):
        raise InputError("inputs must be a pair, a list of pairs, or a head mapping")

    if _is_pair(inputs) and not any(_is_pair(item) for item in inputs):
        pair = _operand_pair(inputs)
        return [pair for _ in heads]

    try:
        items = list(inputs)
    except TypeError as exc:
        raise InputError("inputs must be a pair or a sequence of pairs") from exc

    if not items:
        raise InputError("inputs must not be empty")
    if len(items) == 1 and _is_pair(items[0]):
        pair = _operand_pair(items[0])
        return [pair for _ in heads]
    if all(_is_pair(item) for item in items):
        if len(items) != len(heads):
            raise InputError("one input pair is required per active head")
        return [_operand_pair(item, f"input for head {head}")
                for head, item in zip(heads, items)]
    if len(items) == 2 * len(heads):
        flat = [_operand(item, "input") for item in items]
        return [(flat[index], flat[index + 1])
                for index in range(0, len(flat), 2)]
    raise InputError("inputs must be one pair, one pair per head, or a flat pair list")


def _coupling_weight(left: int, right: int) -> float:
    """Resonance-inspired, deterministic, positive coupling in (0, 1]."""
    if left == right:
        return 0.0
    row_left, col_left = (left >> 4) & 0x0F, left & 0x0F
    row_right, col_right = (right >> 4) & 0x0F, right & 0x0F
    manhattan = abs(row_left - row_right) + abs(col_left - col_right)
    local = {1: 0.80, 2: 0.50, 3: 0.25, 4: 0.125}.get(manhattan, 0.0)
    hamming = (left ^ right).bit_count()
    remote = 1.0 / (1.0 + hamming)
    return max(local, remote)


def _bounded(value: float, limit: float = 1.0) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


class BooleanField:
    """A field of 256 independently learning Boolean heads."""

    HEAD_COUNT = HEAD_COUNT
    BOOLEAN_FUNCTION_COUNT = BOOLEAN_FUNCTION_COUNT
    BOOLEAN_FUNCTIONS = BOOLEAN_FUNCTIONS
    BOOLEAN_FUNCTION_NAMES = BOOLEAN_FUNCTION_NAMES
    BOOL_MASKS = BOOL_MASKS

    def __init__(self, heads: int = HEAD_COUNT, *,
                 head_count: int | None = None,
                 coupling_gain: float = 0.25,
                 approval_token: str | None = None):
        if head_count is not None:
            heads = head_count
        if isinstance(heads, bool) or type(heads) is not int:
            raise BooleanFieldError("heads must be an integer 1..256")
        if not 1 <= heads <= HEAD_COUNT:
            raise BooleanFieldError("heads must be an integer 1..256")
        gain = _finite_number(coupling_gain, "coupling_gain")
        if not 0.0 <= gain <= 1.0:
            raise BooleanFieldError("coupling_gain must be in [0, 1]")
        if approval_token is not None and (
                not isinstance(approval_token, str) or not approval_token):
            raise BooleanFieldError("approval_token must be a non-empty string")
        self.head_count = heads
        self.n_heads = heads
        self.coupling_gain = gain
        self.approval_token = approval_token
        self.heads: list[BooleanHead] = [BooleanHead(index) for index in range(heads)]
        self.scratch: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(heads)}
        self.scratch_entries: list[dict[str, Any]] = []
        self._proposals: dict[str, _Report] = {}
        self.last_step: _Report | None = None

    # -- basic access ------------------------------------------------------

    def __len__(self) -> int:
        return self.head_count

    def __iter__(self):
        return iter(self.heads)

    def __getitem__(self, head: int) -> BooleanHead:
        return self.head(head)

    def _head(self, head: Any) -> BooleanHead:
        if isinstance(head, bool) or type(head) is not int:
            raise UnknownHeadError("head must be an integer 0..255")
        if not 0 <= head < self.head_count:
            raise UnknownHeadError(
                f"unknown head {head}; expected an integer 0..{self.head_count - 1}")
        return self.heads[head]

    def head(self, head: int) -> BooleanHead:
        return self._head(head)

    get_head = head

    def rule(self, head: int) -> int:
        return self._head(head).state.rule

    def ops_mask(self, head: int) -> int:
        return self._head(head).state.ops_mask

    def scores(self, head: int) -> list[int]:
        return list(self._head(head).state.scores)

    @property
    def activations(self) -> list[float]:
        return [head.state.activation for head in self.heads]

    @property
    def coupled_activations(self) -> list[float]:
        return [head.state.coupled_activation for head in self.heads]

    @property
    def field_activation(self) -> _CallableFloat:
        if self.last_step is None:
            return _CallableFloat(0.0)
        return _CallableFloat(self.last_step["field_activation"])

    @property
    def field_coupling(self) -> _CallableFloat:
        if self.last_step is None:
            return _CallableFloat(0.0)
        return _CallableFloat(self.last_step["field_coupling"])

    def activation(self, head: int) -> float:
        return self._head(head).state.activation

    def coupled_activation(self, head: int) -> float:
        return self._head(head).state.coupled_activation

    def score(self, head: int, op: int) -> int:
        return self._head(head).score(op)

    def mask(self, head: int) -> int:
        return self._head(head).state.ops_mask

    def learned_mask(self, head: int) -> int:
        return self._head(head).state.learned_mask

    def truth_table(self, head: int) -> tuple[int, int, int, int]:
        return table(self._head(head).state.rule)

    def reset_boolean(self, head: int) -> _Report:
        target = self._head(head)
        target.state.scores = [0] * BOOLEAN_FUNCTION_COUNT
        target.state.rule = 0
        target.state.ops_mask = BOOLEAN_MASK_ALL
        target.state.generation += 1
        return _Report(head=target.index, rule=0,
                       ops_mask=BOOLEAN_MASK_ALL,
                       scores=list(target.state.scores))

    reset_learner = reset_boolean

    # -- learning ----------------------------------------------------------

    @staticmethod
    def _train_sample(state: HeadState, a: int, b: int, target: int) -> None:
        index = (a << 1) | b
        best = 0
        best_score = SCORE_MIN - 1
        for op in range(BOOLEAN_FUNCTION_COUNT):
            prediction = (op >> index) & 1
            score = state.scores[op] + (1 if prediction == target else -1)
            score = max(SCORE_MIN, min(SCORE_MAX, score))
            state.scores[op] = score
            if score > best_score:
                best_score = score
                best = op
        state.rule = best
        state.ops_mask = 1 << best

    def train_boolean(self, head: int, samples: Any, epochs: int = 1) -> _Report:
        """Train one head on Boolean samples for ``epochs`` passes.

        A sample is ``(a, b, target)``, ``((a, b), target)`` or a mapping with
        ``a``/``b``/``target``.  Scores are independent per head and update by
        the native integer rule: +1 for a matching function, -1 otherwise.
        """
        target = self._head(head)
        normalized = _normalize_samples(samples)
        passes = _epochs(epochs)
        before_rule = target.state.rule
        before_mask = target.state.ops_mask
        for _ in range(passes):
            for a, b, wanted in normalized:
                self._train_sample(target.state, a, b, wanted)
        target.state.generation += 1
        report = self.evaluate_boolean(target.index, normalized)
        return _Report(
            head=target.index,
            epochs=passes,
            samples=len(normalized),
            updates=passes * len(normalized),
            before_rule=before_rule,
            before_mask=before_mask,
            rule=target.state.rule,
            ops_mask=target.state.ops_mask,
            scores=list(target.state.scores),
            correct=report["correct"],
            total=report["total"],
            accuracy=report["accuracy"],
            exact=report["exact"],
            generation=target.state.generation,
        )

    def evaluate_boolean(self, head: int, samples: Any) -> _Report:
        target = self._head(head)
        normalized = _normalize_samples(samples, allow_empty=True)
        predictions = []
        correct = 0
        for a, b, wanted in normalized:
            got = boolean(target.state.rule, a, b)
            predictions.append(got)
            correct += got == wanted
        total = len(normalized)
        accuracy = (correct / total) if total else 0.0
        return _Report(
            head=target.index,
            rule=target.state.rule,
            ops_mask=target.state.ops_mask,
            total=total,
            correct=correct,
            wrong=total - correct,
            accuracy=accuracy,
            exact=bool(total) and correct == total,
            predictions=predictions,
        )

    train = train_boolean
    evaluate = evaluate_boolean

    # -- field step --------------------------------------------------------

    def field_step(self, active_heads: Any, inputs: Any) -> _Report:
        """Evaluate independent head outputs plus a bounded field coupling.

        ``active_heads`` may be one index, an iterable of indices or ``None``
        for all heads.  ``inputs`` may be one ``(a, b)`` pair broadcast to
        every head, one pair per active head, a mapping ``head -> pair`` or a
        flat sequence of operands.  Each head's independent output is kept;
        the coupling term is an additional bounded diagnostic, never a single
        global Boolean decision.
        """
        heads = _normalize_active_heads(active_heads, self.head_count)
        pairs = _normalize_field_inputs(heads, inputs)

        per_head: list[_Report] = []
        outputs: list[int | float] = []
        activations: list[int | float] = []
        signed: list[float] = []
        for head, (a, b) in zip(heads, pairs):
            target = self._head(head)
            output = _evaluate_rule(target.state.rule, a, b)
            signed_value = _bounded(2.0 * float(output) - 1.0)
            outputs.append(output)
            activations.append(output)
            signed.append(signed_value)
            per_head.append(_Report(
                head=head,
                input=(a, b),
                output=output,
                prediction=output,
                activation=output,
                signed_activation=signed_value,
                rule=target.state.rule,
                ops_mask=target.state.ops_mask,
                score=target.state.scores[target.state.rule],
                head_id=target.identity.head_id,
            ))

        weight_total = 0.0
        agreement = 0.0
        for left_index, left_head in enumerate(heads):
            for right_index in range(left_index + 1, len(heads)):
                right_head = heads[right_index]
                weight = _coupling_weight(left_head, right_head)
                if weight <= 0.0:
                    continue
                weight_total += weight
                agreement += weight * signed[left_index] * signed[right_index]
        coupling = _bounded(agreement / weight_total) if weight_total else 0.0

        coupled_signed = [
            _bounded(value + self.coupling_gain * coupling) for value in signed]
        coupled = [(value + 1.0) / 2.0 for value in coupled_signed]
        field_activation = (sum(coupled) / len(coupled)) if coupled else 0.0
        field_signed = (sum(coupled_signed) / len(coupled_signed)
                        if coupled_signed else 0.0)
        for head, activation, coupled_value, item in zip(
                heads, activations, coupled, per_head):
            target = self._head(head)
            target.state.activation = float(activation)
            target.state.coupled_activation = float(coupled_value)
            item["coupled_activation"] = coupled_value

        report = _Report(
            schema="nanlm.hermes-boolean-field-step/1",
            heads=list(heads),
            active_heads=list(heads),
            inputs=[tuple(pair) for pair in pairs],
            per_head=per_head,
            outputs=outputs,
            independent=list(outputs),
            independent_outputs=list(outputs),
            per_head_outputs=list(outputs),
            output_by_head={head: value for head, value in zip(heads, outputs)},
            activations=activations,
            head_activations=list(activations),
            per_head_activation=list(activations),
            activation_by_head={head: value for head, value in zip(heads, activations)},
            signed_activations=signed,
            field_coupling=coupling,
            coupling=coupling,
            coupling_term=coupling,
            field_coupling_magnitude=abs(coupling),
            coupling_magnitude=abs(coupling),
            coupling_bounded=abs(coupling) <= 1.0,
            bounded=abs(coupling) <= 1.0,
            field_activation=field_activation,
            field_signed_activation=field_signed,
            coupled_activation=field_activation,
            coupled_field_activation=field_activation,
            field_activation_vector=coupled,
            coupled_activations=coupled,
            coupled_activation_by_head={
                head: value for head, value in zip(heads, coupled)},
            coupling_gain=self.coupling_gain,
            collapsed_to_global_decision=False,
        )
        self.last_step = report
        return report

    step = field_step

    # -- scratch-first self-modification ----------------------------------

    def _approval(self, commit_token: Any = None,
                  approval: Any = None) -> bool:
        tokens: list[Any] = []
        if commit_token is not None:
            tokens.append(commit_token)
        if approval is not None:
            if approval is True:
                return True
            if isinstance(approval, Mapping):
                if approval.get("approved") is True:
                    return True
                if "token" in approval:
                    tokens.append(approval["token"])
                elif approval.get("approved") is not False:
                    raise ApprovalDenied("approval mapping is not an approval")
            elif isinstance(approval, str):
                tokens.append(approval)
            elif approval is not False:
                raise ApprovalDenied("approval must be True or a token string")
        if not tokens:
            return False
        if self.approval_token is None:
            for token in tokens:
                if not isinstance(token, str) or not token:
                    raise ApprovalDenied("commit token must be a non-empty string")
            return True
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ApprovalDenied("commit token must be a non-empty string")
            if token != self.approval_token:
                raise ApprovalDenied("commit token does not match this field")
        return True

    def self_modify(self, head: int, ops_mask: int, scratch_only: bool = True,
                    commit_token: Any = None, approval: Any = None, *,
                    token: Any = None, mode: str = "xor",
                    approved: Any = None, commit: Any = None,
                    canonical: Any = None) -> _Report:
        """Propose a bounded operator-mask self-modification.

        Default ``mode='xor'`` matches the native ``self_modify_boolean``:
        the 16-bit mask is XOR-patched.  ``mode='replace'``/``'set'`` sets the
        mask directly.  The canonical rule is left untouched unless a valid
        commit token/approval is supplied.  A non-scratch request without
        approval is rejected before anything is written.
        """
        target = self._head(head)
        requested = _uint16_mask(ops_mask)
        if canonical is not None and canonical is not False:
            raise ScratchViolation("canonical writes are not permitted here")
        if approved is not None:
            if approval is not None and approval != approved:
                raise ApprovalDenied("conflicting approval values")
            approval = approved
        if commit is not None:
            if isinstance(commit, str) and commit:
                if commit_token is not None and commit_token != commit:
                    raise ApprovalDenied("conflicting commit tokens")
                commit_token = commit
            elif commit is not False:
                if approval not in (None, False, True) and approval != commit:
                    raise ApprovalDenied("conflicting commit approval")
                approval = True if approval is None else approval
        if type(scratch_only) is not bool:
            raise ScratchViolation("scratch_only must be a bool")
        if mode not in ("xor", "patch", "replace", "set"):
            raise BooleanFieldError("mode must be xor/patch or replace/set")
        if token is not None:
            if commit_token is not None and commit_token != token:
                raise ApprovalDenied("conflicting commit tokens")
            commit_token = token
        try:
            approved = self._approval(commit_token, approval)
        except ApprovalDenied:
            raise
        if not scratch_only and not approved:
            raise ScratchViolation(
                "writes outside scratch require a commit token or approval")

        before_mask = target.state.ops_mask
        before_rule = target.state.rule
        if mode in ("replace", "set"):
            after_mask = requested
            effective_mode = "replace"
        else:
            after_mask = before_mask ^ requested
            effective_mode = "xor"
        after_rule = _rule_from_mask(after_mask)

        body = {
            "schema": PROPOSAL_SCHEMA,
            "head": target.index,
            "mode": effective_mode,
            "requested_mask": requested,
            "before_mask": before_mask,
            "after_mask": after_mask,
            "before_rule": before_rule,
            "after_rule": after_rule,
            "base_generation": target.state.generation,
            "scratch_only": scratch_only,
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        proposal = _Report(
            **body,
            proposal_id=digest,
            id=digest,
            digest=digest,
            candidate_digest=digest,
            scratch_id="head-boolean-scratch-%d" % target.index,
            head_index=target.index,
            head_id=target.identity.head_id,
            ops_mask=requested,
            patch_mask=requested,
            mask=requested,
            proposed_mask=after_mask,
            result_mask=after_mask,
            mask_before=before_mask,
            mask_after=after_mask,
            boolean_mask_before=before_mask,
            boolean_mask_after=after_mask,
            boolean_rule_before=before_rule,
            boolean_rule_after=after_rule,
            rule=after_rule,
            generation=target.state.generation + 1,
            status="proposed",
            state="STAGED",
            applied=False,
            committed=False,
            main_rule_modified=False,
            canonical_state_modified=False,
            scratch_modified=True,
            requires_approval=True,
            approval_required=True,
        )
        target.state.scratch.setdefault("proposals", []).append(digest)
        self.scratch[target.index].append(proposal)
        self.scratch_entries.append(proposal)
        self._proposals[digest] = proposal
        if approved:
            self._apply_proposal(proposal)
        return proposal

    modify = self_modify

    def _apply_proposal(self, proposal: dict[str, Any]) -> _Report:
        target = self._head(proposal["head"])
        if proposal.get("applied"):
            return _Report(proposal)
        if (target.state.ops_mask != proposal["before_mask"]
                or target.state.generation != proposal["base_generation"]):
            raise StaleProposalError(
                "head changed after proposal creation")
        target.state.ops_mask = proposal["after_mask"]
        target.state.rule = proposal["after_rule"]
        target.state.generation += 1
        proposal.update(
            status="applied",
            state="COMMITTED",
            applied=True,
            committed=True,
            main_rule_modified=True,
            canonical_state_modified=False,
            scratch_modified=True,
            requires_approval=False,
            approval_required=False,
            generation=target.state.generation,
        )
        return _Report(proposal)

    def commit_proposal(self, proposal_id: Any, commit_token: Any = None, *,
                        token: Any = None, approval: Any = None,
                        approved: Any = None, commit: Any = None,
                        canonical: Any = None) -> _Report:
        """Commit one scratch proposal after explicit approval."""
        if canonical is not None and canonical is not False:
            raise ScratchViolation("canonical writes are not permitted here")
        if approved is not None:
            if approval is not None and approval != approved:
                raise ApprovalDenied("conflicting approval values")
            approval = approved
        if commit is not None:
            if isinstance(commit, str) and commit:
                if commit_token is not None and commit_token != commit:
                    raise ApprovalDenied("conflicting commit tokens")
                commit_token = commit
            elif commit is not False:
                if approval not in (None, False, True) and approval != commit:
                    raise ApprovalDenied("conflicting commit approval")
                approval = True if approval is None else approval
        if isinstance(proposal_id, Mapping):
            proposal_id = proposal_id.get("id", proposal_id.get("proposal_id"))
        if not isinstance(proposal_id, str) or not proposal_id:
            raise UnknownProposalError("proposal_id must be a non-empty string")
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise UnknownProposalError(f"unknown proposal {proposal_id!r}")
        if token is not None:
            if commit_token is not None and commit_token != token:
                raise ApprovalDenied("conflicting commit tokens")
            commit_token = token
        if not self._approval(commit_token, approval):
            raise ApprovalRequired(
                "commit_proposal requires a commit token or approval")
        return self._apply_proposal(proposal)

    commit = commit_proposal
    approve = commit_proposal

    def proposals(self, head: int | None = None) -> list[_Report]:
        if head is None:
            return list(self._proposals.values())
        target = self._head(head)
        return list(self.scratch[target.index])

    list_proposals = proposals

    def pending(self, head: int | None = None) -> list[_Report]:
        return [proposal for proposal in self.proposals(head)
                if not proposal.get("applied")]

    def pending_proposals(self, head: int | None = None) -> list[_Report]:
        return self.pending(head)

    def get_proposal(self, proposal_id: str) -> _Report:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise UnknownProposalError(f"unknown proposal {proposal_id!r}")
        return _Report(proposal)

    # -- preservation / status --------------------------------------------

    @staticmethod
    def verify_logic_base() -> _Report:
        return verify_logic_base()

    @staticmethod
    def preservation_check() -> _Report:
        return verify_logic_base()

    @staticmethod
    def logic_base_preserved() -> bool:
        return logic_base_preserved()

    def foundation_status(self) -> _Report:
        identities = [head.identity.index for head in self.heads]
        unique_addresses = {head.identity.address for head in self.heads}
        scores_ok = all(
            len(head.state.scores) == BOOLEAN_FUNCTION_COUNT
            and all(SCORE_MIN <= score <= SCORE_MAX
                    for score in head.state.scores)
            for head in self.heads
        )
        rules_ok = all(0 <= head.state.rule < BOOLEAN_FUNCTION_COUNT
                       for head in self.heads)
        masks_ok = all(0 <= head.state.ops_mask <= BOOLEAN_MASK_MAX
                       for head in self.heads)
        identities_ok = identities == list(range(self.head_count))
        intact = bool(identities_ok and scores_ok and rules_ok and masks_ok)
        return _Report(
            schema="nanlm.hermes-head-boolean-foundation/1",
            head_count=self.head_count,
            heads=self.head_count,
            expected_heads=HEAD_COUNT,
            boolean_functions=BOOLEAN_FUNCTION_COUNT,
            functions=BOOLEAN_FUNCTION_COUNT,
            identities=identities_ok,
            unique_addresses=len(unique_addresses) == self.head_count,
            scores=scores_ok,
            rules=rules_ok,
            masks=masks_ok,
            intact=intact,
            exact=intact,
            passed=intact,
        )

    def status(self) -> _Report:
        foundation = self.foundation_status()
        return _Report(
            schema="nanlm.hermes-head-boolean-field/1",
            head_count=self.head_count,
            heads=self.head_count,
            boolean_functions=list(BOOLEAN_FUNCTIONS),
            boolean_function_names=list(BOOLEAN_FUNCTION_NAMES),
            coupling_gain=self.coupling_gain,
            scratch_proposals=len(self.scratch_entries),
            pending_proposals=len(self.pending()),
            foundation=foundation,
            foundation_intact=foundation["intact"],
            main_rule_modified=any(
                proposal.get("applied")
                for proposal in self._proposals.values()),
            canonical_state_modified=False,
            canonical_training_committed=False,
            canonical_merge_enabled=False,
            legacy_logic=verify_logic_base(),
        )


HeadBooleanField = BooleanField
BooleanHeadField = BooleanField
HeadField = BooleanField
HeadBooleanRules = BooleanField
BooleanHeadRules = BooleanField


# ---------------------------------------------------------------------------
# Module-level convenience wrappers.  Both ``train_boolean(field, ...)`` and
# ``train_boolean(head, ...)`` are accepted; the latter uses a lazy default
# 256-head field for small scripts and probes.
# ---------------------------------------------------------------------------

_DEFAULT_FIELD: BooleanField | None = None


def default_field() -> BooleanField:
    """Lazily create the process-local default 256-head field."""
    global _DEFAULT_FIELD
    if _DEFAULT_FIELD is None:
        _DEFAULT_FIELD = BooleanField()
    return _DEFAULT_FIELD


def reset_default_field() -> BooleanField:
    """Replace the lazy default field; useful for isolated tests."""
    global _DEFAULT_FIELD
    _DEFAULT_FIELD = BooleanField()
    return _DEFAULT_FIELD


def _wrapper_field(args: tuple[Any, ...], kwargs: dict[str, Any]):
    field = kwargs.pop("field", None)
    args = list(args)
    if args and isinstance(args[0], BooleanField):
        if field is not None:
            raise BooleanFieldError("field supplied twice")
        field = args.pop(0)
    if field is None:
        field = default_field()
    if not isinstance(field, BooleanField):
        raise BooleanFieldError("field must be a BooleanField")
    return field, tuple(args), kwargs


def train_boolean(*args: Any, **kwargs: Any) -> _Report:
    field, args, kwargs = _wrapper_field(args, dict(kwargs))
    return field.train_boolean(*args, **kwargs)


def evaluate_boolean(*args: Any, **kwargs: Any) -> _Report:
    field, args, kwargs = _wrapper_field(args, dict(kwargs))
    return field.evaluate_boolean(*args, **kwargs)


def field_step(*args: Any, **kwargs: Any) -> _Report:
    field, args, kwargs = _wrapper_field(args, dict(kwargs))
    return field.field_step(*args, **kwargs)


def self_modify(*args: Any, **kwargs: Any) -> _Report:
    field, args, kwargs = _wrapper_field(args, dict(kwargs))
    return field.self_modify(*args, **kwargs)


def status() -> _Report:
    """Status of a fresh full-width field (read-only convenience)."""
    return BooleanField().status()


def foundation_status() -> _Report:
    """Foundation receipt for a fresh full-width field."""
    return BooleanField().foundation_status()


__all__ = [
    "HEAD_COUNT", "HEADS", "NUM_HEADS", "N_HEADS", "MAX_HEADS", "HEAD_BITS",
    "BOOLEAN_FUNCTION_COUNT", "BOOL_FUNCTION_COUNT", "FUNCTION_COUNT",
    "NUM_FUNCTIONS", "BOOLEAN_MASK_BITS", "BOOLEAN_MASK_MAX",
    "BOOLEAN_MASK_ALL", "OPS_MASK_ALL", "BOOL_MASKS", "BOOLEAN_FUNCTIONS",
    "BOOL_FUNCTIONS", "BOOLEAN_FUNCTION_NAMES", "BOOLEAN_NAMES",
    "FUNCTION_NAMES", "BOOL_NAMES", "BOOL_FUNCTION_NAMES",
    "BOOL_FUNCTION_MASKS", "FUNCTIONS_BY_NAME", "BOOLEAN_FUNCTIONS_BY_NAME",
    "XOR", "AND", "OR", "XOR_MASK", "AND_MASK", "OR_MASK",
    "SCORE_MIN", "SCORE_MAX", "PROPOSAL_SCHEMA",
    "LEGACY_MASKS", "BOOLEAN_MASKS", "LEGACY_BOOL_MASKS", "ROOT_MASKS",
    "TRUTH_TABLES", "BOOL_TABLES", "LEGACY_TRUTH_TABLES", "LEGACY_TABLES",
    "BooleanFieldError", "UnknownHeadError", "NonFiniteError", "MaskError",
    "UnsignedMaskError", "MaskOverflowError", "SampleError", "InputError",
    "ScratchViolation", "ApprovalRequired", "ApprovalDenied",
    "UnknownProposalError", "StaleProposalError",
    "head_address", "table", "truth_table", "boolean", "eval_boolean",
    "multilinear", "function_name", "bool_op_name", "legacy_truth_table",
    "verify_exact", "verify_foundation", "verify_legacy_foundation",
    "verify_legacy_truth_tables", "check_legacy_truth_tables",
    "foundation_self_check", "self_check", "verify_logic_base",
    "logic_base_preserved", "preservation_check",
    "HeadIdentity", "HeadID", "HeadState", "BooleanHeadState",
    "BooleanHead", "Head", "BooleanField", "HeadBooleanField",
    "BooleanHeadField", "HeadField", "HeadBooleanRules",
    "BooleanHeadRules", "train_boolean", "evaluate_boolean", "field_step",
    "self_modify", "default_field", "reset_default_field", "status",
    "foundation_status",
]
