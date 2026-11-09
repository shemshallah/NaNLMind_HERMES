"""Raw IEEE-754 binary64 NaN payload geometry.

This module treats a 64-bit IEEE-754 double as an integer bit pattern.  The
identity, mirror-pair, gauge and cycle operations never call the host floating
point unit and never convert a value to a Python ``float``.  In particular, a
NaN payload is *storage*, not an arithmetic value, and every semantic quantity
in this module is an ``int`` payload unit.

Layout of one binary64 word (bit 63 is the most significant bit)::

    63       sign
    62..52   exponent (11 bits, 0x7ff denotes NaN/infinity)
    51       quiet/signaling bit (1 = quiet, 0 = signaling)
    50..0    remaining fraction bits

The 52-bit payload is therefore ``bit 51 .. bit 0``.  We reserve its high
four bits as a tag and the low 48 bits as a two's-complement fixed-point
value::

    51..48   tag nibble (tag bit 3 is the quiet/signaling bit)
    47..0    signed fixed-point units, two's complement

Tags ``0x8..0xf`` are quiet, tags ``0x0..0x7`` are signaling.  Tag ``0xa`` is
the canonical semantic fixed-point tag.  The IEEE sign bit is kept separate
from the two's-complement value; semantic fixed-point words require sign 0 so
that decoding is unambiguous.

The 256-head / 32,640-pair addressing below is the Python counterpart of
``experiments/round029/nan_round029/native_mirror.cpp``:

* 256 head records of 128 bytes,
* 32,640 unordered pair records of 256 bytes (forward + reverse lanes),
* one contiguous root of 8,388,608 bytes.

The gauge store keeps ``x(i,j) = z(i,j) + g[i] - g[j]`` in integer units.
For a triangle whose only semantic edge is ``(i,j)`` the cycle readout is
exactly ``z``, independent of every gauge ``g``.  This is integer arithmetic;
no floating-point cancellation is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator, Sequence


__all__ = [
    "NanGeometryError",
    "NonNanError",
    "NaNFieldError",
    "FixedPointOverflow",
    "HeadIndexError",
    "LaneError",
    "NanFields",
    "ReservedFields",
    "MirrorAddress",
    "Lane",
    "GaugeStore",
    "SIGN_SHIFT",
    "EXPONENT_BITS",
    "EXPONENT_MASK",
    "EXPONENT_SHIFT",
    "EXPONENT_ALL_ONES",
    "PAYLOAD_BITS",
    "PAYLOAD_MASK",
    "QUIET_BIT",
    "TAG_BITS",
    "TAG_SHIFT",
    "TAG_MASK",
    "VALUE_BITS",
    "VALUE_MASK",
    "VALUE_SIGN_BIT",
    "VALUE_MIN",
    "VALUE_MAX",
    "TAG_FIXED",
    "QUIET_TAGS",
    "SIGNALING_TAGS",
    "OVERFLOW_ERROR",
    "OVERFLOW_SATURATE",
    "N_HEADS",
    "HEAD_BYTES",
    "LANE_BYTES",
    "PAIR_BYTES",
    "PAIR_COUNT",
    "ROOT_BYTES",
    "LANE_TAG_OFFSET",
    "LANE_FLAGS_OFFSET",
    "LANE_GENERATION_OFFSET",
    "LANE_RESERVED_OFFSET",
    "LANE_WORDS_OFFSET",
    "LANE_WORD_COUNT",
    "LANE_HEADER_BYTES",
    "LANE_TAG_RAW",
    "LANE_TAG_NAN_WORDS",
    "LANE_TAG_HEAD",
    "LANE_TAG_FORWARD",
    "LANE_TAG_REVERSE",
    "unpack_bits",
    "pack_bits",
    "extract_sign",
    "extract_exponent",
    "extract_payload",
    "extract_quiet_bit",
    "extract_fraction",
    "is_nan",
    "is_finite",
    "is_infinity",
    "is_quiet_nan",
    "is_signaling_nan",
    "validate_nan",
    "validate_finite_control",
    "canonicalize_nan",
    "canonical_quiet_nan",
    "pack_nan",
    "split_reserved_payload",
    "join_reserved_payload",
    "encode_reserved",
    "decode_reserved",
    "encode_fixed_units",
    "decode_fixed_units",
    "encode_fixed_fraction",
    "decode_fixed_fraction",
    "fixed_add",
    "fixed_sub",
    "fixed_negate",
    "fixed_add_many",
    "checked_add_units",
    "checked_sub_units",
    "checked_negate_units",
    "saturating_add_units",
    "saturating_sub_units",
    "saturating_negate_units",
    "pair_id",
    "pair_heads",
    "direction_lane",
    "head_offset",
    "pair_cell_offset",
    "mirror_address",
    "iter_pairs",
    "iter_directed",
    "layout_report",
    "pack_lane",
    "unpack_lane",
]


# ---------------------------------------------------------------------------
# Exceptions


class NanGeometryError(ValueError):
    """Base class for malformed NaN geometry."""


class NonNanError(NanGeometryError):
    """A bit pattern is not a NaN (or is not finite when finite was required)."""


class NaNFieldError(NanGeometryError):
    """A NaN field, payload or tag is malformed."""


class FixedPointOverflow(NanGeometryError, ArithmeticError):
    """A signed 48-bit fixed-point operation left its representable range."""


class HeadIndexError(NanGeometryError, IndexError):
    """A head or pair index is outside the 256-head geometry."""


class LaneError(NanGeometryError):
    """A 128-byte lane is malformed."""


# ---------------------------------------------------------------------------
# IEEE-754 binary64 fields

SIGN_SHIFT = 63
EXPONENT_BITS = 11
EXPONENT_MASK = (1 << EXPONENT_BITS) - 1
EXPONENT_SHIFT = 52
EXPONENT_ALL_ONES = EXPONENT_MASK
PAYLOAD_BITS = 52
PAYLOAD_MASK = (1 << PAYLOAD_BITS) - 1
QUIET_BIT = 1 << (PAYLOAD_BITS - 1)
FRACTION_MASK = PAYLOAD_MASK
UINT64_MASK = (1 << 64) - 1


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    return value


def _bounded(value: object, name: str, low: int, high: int,
             error: type[NanGeometryError] = NaNFieldError) -> int:
    value = _require_int(value, name)
    if not low <= value <= high:
        raise error(f"{name} must be in {low}..{high}, got {value}")
    return value


@dataclass(frozen=True)
class NanFields:
    """Decoded 64-bit binary64 fields, with no float interpretation."""

    sign: int
    exponent: int
    payload: int

    def __post_init__(self) -> None:
        _bounded(self.sign, "sign", 0, 1)
        _bounded(self.exponent, "exponent", 0, EXPONENT_ALL_ONES)
        _bounded(self.payload, "payload", 0, PAYLOAD_MASK)

    @property
    def quiet(self) -> bool:
        return bool(self.payload & QUIET_BIT)

    @property
    def signaling(self) -> bool:
        return self.is_nan and not self.quiet

    @property
    def is_nan(self) -> bool:
        return self.exponent == EXPONENT_ALL_ONES and self.payload != 0

    @property
    def is_infinity(self) -> bool:
        return self.exponent == EXPONENT_ALL_ONES and self.payload == 0

    @property
    def is_finite(self) -> bool:
        return self.exponent != EXPONENT_ALL_ONES

    def pack(self) -> int:
        return pack_bits(self.sign, self.exponent, self.payload)


def unpack_bits(bits: int) -> NanFields:
    """Extract sign, exponent and 52-bit payload from a raw 64-bit word."""
    bits = _bounded(bits, "bits", 0, UINT64_MASK)
    return NanFields((bits >> SIGN_SHIFT) & 1,
                     (bits >> EXPONENT_SHIFT) & EXPONENT_MASK,
                     bits & PAYLOAD_MASK)


def pack_bits(sign: int, exponent: int, payload: int) -> int:
    """Pack raw binary64 fields without involving floating-point hardware."""
    sign = _bounded(sign, "sign", 0, 1)
    exponent = _bounded(exponent, "exponent", 0, EXPONENT_ALL_ONES)
    payload = _bounded(payload, "payload", 0, PAYLOAD_MASK)
    return ((sign << SIGN_SHIFT) | (exponent << EXPONENT_SHIFT) | payload)


def extract_sign(bits: int) -> int:
    return (unpack_bits(bits).sign)


def extract_exponent(bits: int) -> int:
    return unpack_bits(bits).exponent


def extract_payload(bits: int) -> int:
    return unpack_bits(bits).payload


def extract_fraction(bits: int) -> int:
    """Alias for the 52-bit fraction/payload field."""
    return unpack_bits(bits).payload


def extract_quiet_bit(bits: int) -> int:
    return 1 if unpack_bits(bits).quiet else 0


def is_nan(bits: int) -> bool:
    return unpack_bits(bits).is_nan


def is_finite(bits: int) -> bool:
    return unpack_bits(bits).is_finite


def is_infinity(bits: int) -> bool:
    return unpack_bits(bits).is_infinity


def is_quiet_nan(bits: int) -> bool:
    fields = unpack_bits(bits)
    return fields.is_nan and fields.quiet


def is_signaling_nan(bits: int) -> bool:
    return unpack_bits(bits).signaling


def validate_nan(bits: int, *, allow_signaling: bool = True) -> NanFields:
    """Return fields for a NaN, or raise :class:`NonNanError`.

    Infinity (payload zero) is not a NaN.  Signaling NaNs are permitted by
    default because they are a valid storage pattern; arithmetic helpers force
    quiet output by using a quiet tag.
    """
    fields = unpack_bits(bits)
    if not fields.is_nan:
        raise NonNanError("bit pattern is not a NaN")
    if fields.signaling and not allow_signaling:
        raise NaNFieldError("signaling NaN is not allowed here")
    return fields


def validate_finite_control(bits: int) -> NanFields:
    """Explicit finite control: accept only a finite binary64 pattern."""
    fields = unpack_bits(bits)
    if not fields.is_finite:
        raise NonNanError("finite control received a NaN or infinity")
    return fields


def pack_nan(sign: int, payload: int, *, quiet: bool | None = None) -> int:
    """Pack a NaN from a sign and 52-bit payload.

    ``payload`` may not be zero: that would describe infinity, not NaN.  When
    ``quiet`` is supplied it must agree with payload bit 51.
    """
    sign = _bounded(sign, "sign", 0, 1)
    payload = _bounded(payload, "payload", 0, PAYLOAD_MASK)
    if payload == 0:
        raise NonNanError("payload zero is infinity, not NaN")
    if quiet is not None:
        if not isinstance(quiet, bool):
            raise TypeError("quiet must be bool or None")
        if quiet != bool(payload & QUIET_BIT):
            raise NaNFieldError("quiet flag disagrees with payload bit 51")
    return pack_bits(sign, EXPONENT_ALL_ONES, payload)


def canonicalize_nan(bits: int, *, quiet: bool = True) -> int:
    """Canonicalize a NaN's quiet bit while preserving sign and payload bits.

    Clearing the quiet bit from the smallest quiet payload (payload == 2**51)
    would produce infinity, so that case becomes the smallest signaling payload
    (payload == 1).
    """
    fields = validate_nan(bits)
    payload = fields.payload
    if quiet:
        payload |= QUIET_BIT
    else:
        payload &= ~QUIET_BIT
        if payload == 0:
            payload = 1
    return pack_nan(fields.sign, payload)


def canonical_quiet_nan(sign: int = 0) -> int:
    """Return the canonical quiet NaN ``sign:0x7ff8:000...``."""
    return pack_nan(sign, QUIET_BIT)


# ---------------------------------------------------------------------------
# Reservation policy: 4-bit tag + 48-bit two's-complement fixed point

TAG_BITS = 4
TAG_SHIFT = PAYLOAD_BITS - TAG_BITS
TAG_MASK = (1 << TAG_BITS) - 1
VALUE_BITS = TAG_SHIFT
VALUE_MASK = (1 << VALUE_BITS) - 1
VALUE_SIGN_BIT = 1 << (VALUE_BITS - 1)
VALUE_MIN = -VALUE_SIGN_BIT
VALUE_MAX = VALUE_SIGN_BIT - 1
TAG_FIXED = 0xA  # 1010: quiet, semantic fixed-point units
QUIET_TAGS = frozenset(range(0x8, 0x10))
SIGNALING_TAGS = frozenset(range(0x0, 0x8))

OVERFLOW_ERROR = "error"
OVERFLOW_SATURATE = "saturate"


@dataclass(frozen=True)
class ReservedFields:
    """Decoded reservation fields inside a NaN payload."""

    sign: int
    tag: int
    value: int

    def __post_init__(self) -> None:
        _bounded(self.sign, "sign", 0, 1)
        _bounded(self.tag, "tag", 0, TAG_MASK)
        _bounded(self.value, "value", 0, VALUE_MASK)

    @property
    def quiet(self) -> bool:
        return bool(self.tag & 0x8)

    @property
    def signaling(self) -> bool:
        return not self.quiet

    @property
    def units(self) -> int:
        """Interpret the 48 raw bits as signed two's-complement units."""
        if self.value & VALUE_SIGN_BIT:
            return self.value - (1 << VALUE_BITS)
        return self.value

    def pack(self) -> int:
        return encode_reserved(self.tag, self.value, self.sign)


def split_reserved_payload(payload: int) -> tuple[int, int]:
    """Split a 52-bit payload into ``(tag, raw 48-bit value)``."""
    payload = _bounded(payload, "payload", 0, PAYLOAD_MASK)
    return payload >> TAG_SHIFT, payload & VALUE_MASK


def join_reserved_payload(tag: int, value: int) -> int:
    """Join a tag and 48 raw value bits, rejecting the infinity pattern."""
    tag = _bounded(tag, "tag", 0, TAG_MASK)
    value = _bounded(value, "value", 0, VALUE_MASK)
    payload = (tag << TAG_SHIFT) | value
    if payload == 0:
        raise NonNanError("tag 0 with value 0 is infinity, not NaN")
    return payload


def encode_reserved(tag: int, value: int, sign: int = 0,
                    *, quiet: bool | None = None) -> int:
    """Encode a tagged NaN, allowing both quiet and signaling tags."""
    tag = _bounded(tag, "tag", 0, TAG_MASK)
    value = _bounded(value, "value", 0, VALUE_MASK)
    sign = _bounded(sign, "sign", 0, 1)
    if quiet is not None:
        if not isinstance(quiet, bool):
            raise TypeError("quiet must be bool or None")
        if quiet != bool(tag & 0x8):
            raise NaNFieldError("quiet flag disagrees with the tag's quiet bit")
    return pack_nan(sign, join_reserved_payload(tag, value))


def decode_reserved(bits: int, *, expected_tag: int | None = None
                    ) -> ReservedFields:
    """Decode a tagged NaN, optionally requiring an exact tag."""
    fields = validate_nan(bits)
    tag, value = split_reserved_payload(fields.payload)
    if expected_tag is not None:
        expected_tag = _bounded(expected_tag, "expected_tag", 0, TAG_MASK)
        if tag != expected_tag:
            raise NaNFieldError(
                f"expected tag {expected_tag:#x}, got {tag:#x}")
    return ReservedFields(fields.sign, tag, value)


# ---------------------------------------------------------------------------
# Signed fixed-point encoding and integer-only arithmetic


def _check_units(value: int, name: str = "units") -> int:
    value = _require_int(value, name)
    if not VALUE_MIN <= value <= VALUE_MAX:
        raise FixedPointOverflow(
            f"{name} must be in {VALUE_MIN}..{VALUE_MAX}, got {value}")
    return value


def encode_fixed_units(units: int, *, tag: int = TAG_FIXED,
                       sign: int = 0) -> int:
    """Encode signed fixed-point units into a quiet tagged NaN payload.

    The value is two's complement in 48 bits.  The IEEE sign bit must remain 0
    for semantic words; use the two's-complement value for negative numbers.
    """
    units = _check_units(units)
    tag = _bounded(tag, "tag", 0, TAG_MASK)
    sign = _bounded(sign, "sign", 0, 1)
    if sign != 0:
        raise NaNFieldError(
            "fixed-point semantics use two's complement, not the IEEE sign bit")
    if tag not in QUIET_TAGS:
        raise NaNFieldError("fixed-point values require a quiet tag")
    return encode_reserved(tag, units & VALUE_MASK, sign=0)


def decode_fixed_units(bits: int, *, tag: int = TAG_FIXED) -> int:
    """Decode exact signed fixed-point units from a tagged NaN payload."""
    reserved = decode_reserved(bits, expected_tag=tag)
    if reserved.sign != 0:
        raise NaNFieldError(
            "fixed-point semantics require the IEEE sign bit to be zero")
    return reserved.units


def encode_fixed_fraction(value: Fraction | int, *, scale_bits: int = 0,
                          tag: int = TAG_FIXED) -> int:
    """Encode an exact rational value as units of ``2**-scale_bits``.

    Python ``float`` inputs are rejected.  Use ``int`` or ``Fraction`` so the
    scaling remains exact.
    """
    scale_bits = _bounded(scale_bits, "scale_bits", 0, 255)
    if isinstance(value, Fraction):
        scaled = value
    elif type(value) is int:
        scaled = Fraction(value)
    else:
        raise TypeError("fixed-point fraction must be Fraction or int, not "
                        f"{type(value).__name__}")
    scaled *= (1 << scale_bits)
    if scaled.denominator != 1:
        raise NaNFieldError(
            f"value {value!r} is not representable at scale 2**-{scale_bits}")
    return encode_fixed_units(scaled.numerator, tag=tag)


def decode_fixed_fraction(bits: int, *, scale_bits: int = 0,
                          tag: int = TAG_FIXED) -> Fraction:
    """Decode fixed-point units as an exact :class:`Fraction`."""
    scale_bits = _bounded(scale_bits, "scale_bits", 0, 255)
    units = decode_fixed_units(bits, tag=tag)
    return Fraction(units, 1 << scale_bits)


def _overflow_policy(overflow: str) -> str:
    if overflow not in (OVERFLOW_ERROR, OVERFLOW_SATURATE):
        raise ValueError("overflow must be 'error' or 'saturate'")
    return overflow


def checked_add_units(first: int, second: int) -> int:
    total = _check_units(first, "first") + _check_units(second, "second")
    return _check_units(total, "sum")


def checked_sub_units(first: int, second: int) -> int:
    difference = _check_units(first, "first") - _check_units(second, "second")
    return _check_units(difference, "difference")


def checked_negate_units(value: int) -> int:
    return _check_units(-_check_units(value, "value"), "negation")


def _saturate(value: int) -> int:
    return VALUE_MIN if value < VALUE_MIN else VALUE_MAX if value > VALUE_MAX else value


def saturating_add_units(first: int, second: int) -> int:
    return _saturate(_check_units(first, "first") +
                     _check_units(second, "second"))


def saturating_sub_units(first: int, second: int) -> int:
    return _saturate(_check_units(first, "first") -
                     _check_units(second, "second"))


def saturating_negate_units(value: int) -> int:
    return _saturate(-_check_units(value, "value"))


def fixed_add(first_bits: int, second_bits: int, *,
              overflow: str = OVERFLOW_ERROR, tag: int = TAG_FIXED) -> int:
    """Add two encoded fixed-point NaNs using integer payload arithmetic."""
    policy = _overflow_policy(overflow)
    first = decode_fixed_units(first_bits, tag=tag)
    second = decode_fixed_units(second_bits, tag=tag)
    result = (checked_add_units(first, second)
              if policy == OVERFLOW_ERROR
              else saturating_add_units(first, second))
    return encode_fixed_units(result, tag=tag)


def fixed_sub(first_bits: int, second_bits: int, *,
              overflow: str = OVERFLOW_ERROR, tag: int = TAG_FIXED) -> int:
    """Subtract two encoded fixed-point NaNs using integer arithmetic."""
    policy = _overflow_policy(overflow)
    first = decode_fixed_units(first_bits, tag=tag)
    second = decode_fixed_units(second_bits, tag=tag)
    result = (checked_sub_units(first, second)
              if policy == OVERFLOW_ERROR
              else saturating_sub_units(first, second))
    return encode_fixed_units(result, tag=tag)


def fixed_negate(bits: int, *, overflow: str = OVERFLOW_ERROR,
                 tag: int = TAG_FIXED) -> int:
    """Negate an encoded fixed-point NaN using integer arithmetic."""
    policy = _overflow_policy(overflow)
    value = decode_fixed_units(bits, tag=tag)
    result = (checked_negate_units(value)
              if policy == OVERFLOW_ERROR
              else saturating_negate_units(value))
    return encode_fixed_units(result, tag=tag)


def fixed_add_many(values: Iterable[int], *, overflow: str = OVERFLOW_ERROR,
                   tag: int = TAG_FIXED) -> int:
    """Sum encoded fixed-point NaNs left-to-right in integer units."""
    policy = _overflow_policy(overflow)
    total = 0
    for bits in values:
        value = decode_fixed_units(bits, tag=tag)
        total = (checked_add_units(total, value)
                 if policy == OVERFLOW_ERROR
                 else saturating_add_units(total, value))
    return encode_fixed_units(total, tag=tag)


# ---------------------------------------------------------------------------
# 256-head / 32,640-pair mirror geometry (native_mirror.cpp counterpart)

N_HEADS = 256
HEAD_BYTES = 128
LANE_BYTES = 128
PAIR_BYTES = 2 * LANE_BYTES
PAIR_COUNT = N_HEADS * (N_HEADS - 1) // 2
HEAD_REGION_BYTES = N_HEADS * HEAD_BYTES
ROOT_BYTES = HEAD_REGION_BYTES + PAIR_COUNT * PAIR_BYTES


def _require_head(head: int) -> int:
    return _bounded(head, "head", 0, N_HEADS - 1, HeadIndexError)


def _require_pair(pair: int) -> int:
    return _bounded(pair, "pair", 0, PAIR_COUNT - 1, HeadIndexError)


def _row_start(first: int) -> int:
    return first * (2 * N_HEADS - first - 1) // 2


def pair_id(first: int, second: int) -> int:
    """Canonical unordered-pair id, identical to ``Root::id`` in C++."""
    first = _require_head(first)
    second = _require_head(second)
    if first == second:
        raise HeadIndexError("a mirror pair requires two distinct heads")
    if first > second:
        first, second = second, first
    return _row_start(first) + (second - first - 1)


def pair_heads(pair: int) -> tuple[int, int]:
    """Invert :func:`pair_id`, returning heads in increasing order."""
    pair = _require_pair(pair)
    low, high = 0, N_HEADS - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _row_start(middle) <= pair:
            low = middle
        else:
            high = middle - 1
    first = low
    second = pair - _row_start(first) + first + 1
    if not (0 <= first < second < N_HEADS):
        raise HeadIndexError("pair id does not map to two heads")
    return first, second


def direction_lane(first: int, second: int) -> int:
    """Return 0 for the forward lane (i<j), 1 for the reverse lane (i>j)."""
    first = _require_head(first)
    second = _require_head(second)
    if first == second:
        raise HeadIndexError("a direction requires two distinct heads")
    return 0 if first < second else 1


@dataclass(frozen=True)
class MirrorAddress:
    """Address of one directed lane inside the contiguous 8 MiB root."""

    pair: int
    lane: int
    cell_offset: int
    lane_offset: int
    heads: tuple[int, int]

    @property
    def direction_offset(self) -> int:
        return self.lane_offset


def head_offset(head: int) -> int:
    """Byte offset of one 128-byte head record."""
    return _require_head(head) * HEAD_BYTES


def pair_cell_offset(first: int, second: int) -> int:
    """Byte offset of the 256-byte pair record containing both directions."""
    pair = pair_id(first, second)
    return HEAD_REGION_BYTES + pair * PAIR_BYTES


def mirror_address(first: int, second: int) -> MirrorAddress:
    """Return the shared pair cell and the addressed forward/reverse lane."""
    pair = pair_id(first, second)
    cell = HEAD_REGION_BYTES + pair * PAIR_BYTES
    lane = direction_lane(first, second)
    return MirrorAddress(pair, lane, cell, cell + lane * LANE_BYTES,
                         (first, second))


def iter_pairs() -> Iterator[tuple[int, int]]:
    """Yield all 32,640 unordered head pairs in native address order."""
    for first in range(N_HEADS):
        for second in range(first + 1, N_HEADS):
            yield first, second


def iter_directed() -> Iterator[tuple[int, int]]:
    """Yield all 65,280 directed head mappings."""
    for first in range(N_HEADS):
        for second in range(N_HEADS):
            if first != second:
                yield first, second


def layout_report() -> dict:
    """Return the integer layout invariants proved by the native test."""
    return {
        "heads": N_HEADS,
        "head_bytes": HEAD_BYTES,
        "pairs": PAIR_COUNT,
        "pair_bytes": PAIR_BYTES,
        "lanes_per_pair": 2,
        "lane_bytes": LANE_BYTES,
        "directed_mappings": N_HEADS * (N_HEADS - 1),
        "root_bytes": ROOT_BYTES,
    }


# ---------------------------------------------------------------------------
# 128-byte bit-preserving lane codec


LANE_TAG_OFFSET = 0
LANE_FLAGS_OFFSET = 2
LANE_GENERATION_OFFSET = 4
LANE_RESERVED_OFFSET = 12
LANE_WORDS_OFFSET = 16
LANE_WORD_COUNT = 14
LANE_HEADER_BYTES = 16

LANE_TAG_RAW = 0x0000
LANE_TAG_NAN_WORDS = 0x4E41  # "NA"
LANE_TAG_HEAD = 0x4844       # "HD"
LANE_TAG_FORWARD = 0x4644    # "FD"
LANE_TAG_REVERSE = 0x5245    # "RE"


@dataclass(frozen=True)
class Lane:
    """Typed 128-byte lane with little-endian fields.

    Layout::

        0..1    tag, uint16 little-endian
        2..3    flags, uint16 little-endian
        4..11   generation, uint64 little-endian
        12..15  reserved, uint32 little-endian
        16..127 14 payload words, uint64 little-endian each

    Every bit of every field is preserved by :meth:`pack`/:meth:`unpack`.
    Payload words may hold raw binary64 NaN words, but this codec never
    interprets them as floats.
    """

    tag: int
    generation: int
    words: tuple[int, ...]
    flags: int = 0
    reserved: int = 0

    def __post_init__(self) -> None:
        _bounded(self.tag, "tag", 0, 0xFFFF)
        _bounded(self.flags, "flags", 0, 0xFFFF)
        _bounded(self.generation, "generation", 0, UINT64_MASK)
        _bounded(self.reserved, "reserved", 0, 0xFFFFFFFF)
        try:
            words = tuple(self.words)
        except TypeError as exc:
            raise LaneError("words must be an iterable of 64-bit ints") from exc
        if len(words) != LANE_WORD_COUNT:
            raise LaneError(
                f"lane requires {LANE_WORD_COUNT} payload words, got {len(words)}")
        for index, word in enumerate(words):
            _bounded(word, f"words[{index}]", 0, UINT64_MASK)
        object.__setattr__(self, "words", words)

    def word(self, index: int) -> int:
        index = _bounded(index, "word index", 0, LANE_WORD_COUNT - 1,
                         LaneError)
        return self.words[index]

    def with_word(self, index: int, value: int) -> "Lane":
        index = _bounded(index, "word index", 0, LANE_WORD_COUNT - 1,
                         LaneError)
        value = _bounded(value, "word", 0, UINT64_MASK)
        words = list(self.words)
        words[index] = value
        return Lane(self.tag, self.generation, tuple(words),
                    self.flags, self.reserved)

    def pack(self) -> bytes:
        parts = [
            self.tag.to_bytes(2, "little"),
            self.flags.to_bytes(2, "little"),
            self.generation.to_bytes(8, "little"),
            self.reserved.to_bytes(4, "little"),
        ]
        parts.extend(word.to_bytes(8, "little") for word in self.words)
        raw = b"".join(parts)
        if len(raw) != LANE_BYTES:
            raise LaneError("internal lane layout is not 128 bytes")
        return raw

    @classmethod
    def unpack(cls, raw: bytes | bytearray | memoryview) -> "Lane":
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError("lane data must be bytes-like")
        data = bytes(raw)
        if len(data) != LANE_BYTES:
            raise LaneError(f"lane must be exactly {LANE_BYTES} bytes")
        words = tuple(
            int.from_bytes(data[LANE_WORDS_OFFSET + 8 * index:
                                LANE_WORDS_OFFSET + 8 * (index + 1)],
                           "little")
            for index in range(LANE_WORD_COUNT))
        return cls(
            int.from_bytes(data[LANE_TAG_OFFSET:LANE_FLAGS_OFFSET], "little"),
            int.from_bytes(data[LANE_GENERATION_OFFSET:LANE_RESERVED_OFFSET],
                           "little"),
            words,
            int.from_bytes(data[LANE_FLAGS_OFFSET:LANE_GENERATION_OFFSET],
                           "little"),
            int.from_bytes(data[LANE_RESERVED_OFFSET:LANE_WORDS_OFFSET],
                           "little"),
        )

    def as_dict(self) -> dict:
        return {
            "tag": self.tag,
            "flags": self.flags,
            "generation": self.generation,
            "reserved": self.reserved,
            "words": list(self.words),
            "raw_hex": self.pack().hex(),
            "byte_order": "little",
        }


def pack_lane(lane: Lane) -> bytes:
    if not isinstance(lane, Lane):
        raise TypeError("pack_lane expects a Lane")
    return lane.pack()


def unpack_lane(raw: bytes | bytearray | memoryview) -> Lane:
    return Lane.unpack(raw)


# ---------------------------------------------------------------------------
# Integer gauge transform store


class GaugeStore:
    """Complete-graph gauge store in integer payload units.

    For an unordered pair ``{i,j}`` a signed cycle value ``z`` is stored.  The
    directed field is::

        x(i,j) = z(i,j) + g[i] - g[j],   z(j,i) = -z(i,j)

    and the native three-edge circulation is::

        x(0,i) + x(i,j) + x(j,0) = z(i,j)

    for distinct ``0,i,j``.  Equivalently, with the canonical pair value
    ``x(0,j)``, ``x(0,i) + x(i,j) - x(0,j) = z(i,j)``.  Both forms are exposed
    below; all values are Python ints.
    """

    def __init__(self, heads: int = N_HEADS):
        heads = _require_int(heads, "heads")
        if heads < 3:
            raise HeadIndexError("gauge store needs at least three heads")
        self._heads = heads
        self._gauge = [0] * heads
        self._cycles: dict[tuple[int, int], int] = {}

    @property
    def heads(self) -> int:
        return self._heads

    @property
    def gauge(self) -> tuple[int, ...]:
        return tuple(self._gauge)

    def _head(self, value: int, name: str = "head") -> int:
        return _bounded(value, name, 0, self._heads - 1, HeadIndexError)

    def _pair(self, first: int, second: int) -> tuple[int, int]:
        first = self._head(first, "first")
        second = self._head(second, "second")
        if first == second:
            raise HeadIndexError("a pair requires two distinct heads")
        return (first, second) if first < second else (second, first)

    def set_gauge(self, values: Sequence[int]) -> "GaugeStore":
        values = list(values)
        if len(values) != self._heads:
            raise HeadIndexError(
                f"gauge requires {self._heads} values, got {len(values)}")
        self._gauge = [_require_int(value, "gauge value") for value in values]
        return self

    def shift_gauge(self, values: Sequence[int]) -> "GaugeStore":
        values = list(values)
        if len(values) != self._heads:
            raise HeadIndexError(
                f"gauge shift requires {self._heads} values, got {len(values)}")
        self._gauge = [current + _require_int(value, "gauge shift")
                       for current, value in zip(self._gauge, values)]
        return self

    def reset_gauge(self) -> "GaugeStore":
        self._gauge = [0] * self._heads
        return self

    def set_cycle(self, first: int, second: int, value: int) -> "GaugeStore":
        """Store a signed semantic cycle value on one unordered pair."""
        key = self._pair(first, second)
        value = _require_int(value, "cycle value")
        if first > second:
            value = -value
        if value:
            self._cycles[key] = value
        else:
            self._cycles.pop(key, None)
        return self

    def store_skew(self, first: int, second: int, value: int) -> "GaugeStore":
        """Native-compatible alias for setting an antisymmetric pair value."""
        return self.set_cycle(first, second, value)

    def cycle_value(self, first: int, second: int) -> int:
        """Return the signed cycle value oriented as ``first -> second``."""
        key = self._pair(first, second)
        value = self._cycles.get(key, 0)
        return value if first < second else -value

    def edge(self, first: int, second: int) -> int:
        """Return ``x(first,second) = z + g[first] - g[second]``."""
        first = self._head(first, "first")
        second = self._head(second, "second")
        if first == second:
            raise HeadIndexError("an edge requires two distinct heads")
        return (self.cycle_value(first, second)
                + self._gauge[first] - self._gauge[second])

    def canonical_edge(self, first: int, second: int) -> int:
        """Return the value oriented from the lower head to the higher head."""
        first = self._head(first, "first")
        second = self._head(second, "second")
        if first == second:
            raise HeadIndexError("an edge requires two distinct heads")
        value = self.edge(first, second)
        return value if first < second else -value

    def circulation(self, first: int, second: int,
                    anchor: int = 0) -> int:
        """Native readout ``x(anchor,first)+x(first,second)+x(second,anchor)``."""
        anchor = self._head(anchor, "anchor")
        first = self._head(first, "first")
        second = self._head(second, "second")
        if len({anchor, first, second}) != 3:
            raise HeadIndexError("a circulation needs three distinct heads")
        return (self.edge(anchor, first) + self.edge(first, second)
                + self.edge(second, anchor))

    def cycle_readout(self, first: int, second: int,
                      anchor: int = 0) -> int:
        """Alias for :meth:`circulation`; equals the stored ``z`` exactly."""
        return self.circulation(first, second, anchor)

    def canonical_flux(self, first: int, second: int,
                       anchor: int = 0) -> int:
        """Canonical formula ``x(a,i) + x(i,j) - x(a,j)``.

        The middle edge is directed ``i -> j``.  When ``i < j`` this is the
        README/Round-029 triangle flux and equals the stored ``z``; reversing
        the arguments negates it.  The anchor edge values use the directed
        field, so the identity also holds when ``anchor`` is not head 0.
        """
        anchor = self._head(anchor, "anchor")
        first = self._head(first, "first")
        second = self._head(second, "second")
        if len({anchor, first, second}) != 3:
            raise HeadIndexError("a flux needs three distinct heads")
        return (self.edge(anchor, first)
                + self.edge(first, second)
                - self.edge(anchor, second))

    def validate_payload_bounds(self, *, tag: int = TAG_FIXED) -> int:
        """Prove every directed edge fits in a 48-bit fixed-point payload.

        Returns the number of checked directed edges (65,280 for 256 heads).
        Raises :class:`FixedPointOverflow` for the first edge that cannot be
        represented, which is the correct failure mode for a fixed-width NaN
        payload even when a cycle would cancel the large gauge.
        """
        checked = 0
        for first in range(self._heads):
            for second in range(first + 1, self._heads):
                encode_fixed_units(self.edge(first, second), tag=tag)
                encode_fixed_units(self.edge(second, first), tag=tag)
                checked += 2
        return checked
