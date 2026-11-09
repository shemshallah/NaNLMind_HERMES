"""Raw NaN-payload lane ABI for the 128-byte native lane and 256-byte pair.

The native Round 23/29 mirror cell has a 128-byte forward lane and a 128-byte
reverse lane.  This module gives those lanes a strict, portable ABI without ever
treating a word as a Python or IEEE float.  A lane is exactly 32 little-endian
unsigned 32-bit words; every word is kept as an integer in ``[0, 2**32)`` and is
re-emitted byte-for-byte.

Word layout
-----------
======  ==========================================================
word    region
======  ==========================================================
0       tag/version: low 16 bits version, high 16 bits tag
1..28   28 channel words (raw NaN payload data)
29      generation, exact modulo ``2**32``
30..31  reserved
======  ==========================================================

The tag word is the lane identity.  It is written once by :meth:`Lane.new` and
an attempt to replace it with a different value is rejected.  Three tags are
defined:

``TAG_RAW``
    Any 32-bit word is accepted in every channel.  This is the ABI guard for
    raw NaN payloads: quiet NaN, signalling NaN, infinities, subnormals and
    ordinary bit patterns all round-trip unchanged.

``TAG_QUIET``
    Every channel word must be a quiet NaN pattern (all exponent bits set,
    non-zero mantissa, quiet bit set).  The sign bit is part of the payload and
    is preserved.

``TAG_SIGNALING``
    Every channel word must be a signalling NaN pattern (all exponent bits set,
    non-zero mantissa, quiet bit clear).  The sign bit is preserved.

Arithmetic in this module is integer arithmetic modulo ``2**32``.  A cycle is
the exact raw-word analogue of the Round 029 triangle observable::

    cycle_word(edge_0a, edge_ab, edge_0b, channel)
        == (edge_0a + edge_ab - edge_0b) & 0xFFFFFFFF

Because no float conversion happens, a gauge construction
``g[i] - g[j]`` cancels bit-exactly for arbitrary 32-bit gauges.  This is a
storage/ABI primitive, not a claim of quantum behaviour or native speed.

The compact :class:`Pair` is the native ``Pair`` layout::

    forward[32] (128 bytes) followed by reverse[32] (128 bytes)

Only raw 32-bit words are stored.  The reverse lane is an independent word
lane; :meth:`Pair.deposit_oriented` keeps a pair exactly antisymmetric in
modular arithmetic when that invariant is required.
"""
from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# sizes and typed word regions
# ---------------------------------------------------------------------------
LANE_WORDS = 32
WORD_BYTES = 4
LANE_BYTES = LANE_WORDS * WORD_BYTES          # 128
WORD_BITS = 32
WORD_MODULUS = 1 << WORD_BITS
WORD_MASK = WORD_MODULUS - 1

LANE_WORD_COUNT = LANE_WORDS
PAIR_LANES = 2
PAIR_WORDS = PAIR_LANES * LANE_WORDS          # 64
PAIR_BYTES = PAIR_LANES * LANE_BYTES          # 256
NATIVE_LANE_BYTES = LANE_BYTES
NATIVE_PAIR_BYTES = PAIR_BYTES

TAG_VERSION_WORD = 0
TAG_WORD = TAG_VERSION_WORD
FIRST_CHANNEL_WORD = 1
CHANNEL_COUNT = 28
LAST_CHANNEL_WORD = FIRST_CHANNEL_WORD + CHANNEL_COUNT - 1   # 28
GENERATION_WORD = 29
FIRST_RESERVED_WORD = 30
RESERVED_COUNT = 2
LAST_RESERVED_WORD = FIRST_RESERVED_WORD + RESERVED_COUNT - 1  # 31

VERSION_BITS = 16
VERSION_MASK = (1 << VERSION_BITS) - 1
TAG_SHIFT = VERSION_BITS
TAG_MASK = (1 << (WORD_BITS - VERSION_BITS)) - 1
LANE_VERSION = 1

TAG_RAW = 0x0000
TAG_QUIET = 0x0001
TAG_SIGNALING = 0x0002
TAG_NAMES = {
    TAG_RAW: "raw",
    TAG_QUIET: "quiet",
    TAG_SIGNALING: "signaling",
}
KNOWN_TAGS = frozenset(TAG_NAMES)

# Descriptive aliases used by callers that name the payload class explicitly.
TAG_RAW_WORDS = TAG_RAW
TAG_QUIET_NAN = TAG_QUIET
TAG_SIGNALING_NAN = TAG_SIGNALING

NAN_SIGN_BIT = 0x80000000
NAN_EXPONENT_MASK = 0x7F800000
NAN_MANTISSA_MASK = 0x007FFFFF
NAN_QUIET_BIT = 0x00400000
QUIET_NAN_WORD = 0x7FC00000
SIGNALING_NAN_WORD = 0x7F800001

ORIENT_FORWARD = 1
ORIENT_REVERSE = -1
ORIENT_NAMES = {
    "forward": ORIENT_FORWARD,
    "f": ORIENT_FORWARD,
    "s+": ORIENT_FORWARD,
    "+": ORIENT_FORWARD,
    "reverse": ORIENT_REVERSE,
    "r": ORIENT_REVERSE,
    "s-": ORIENT_REVERSE,
    "-": ORIENT_REVERSE,
}

WORD_STRUCT = struct.Struct("<I")
LANE_STRUCT = struct.Struct("<32I")
PAIR_STRUCT = struct.Struct("<64I")

# These assertions are part of the ABI.  A platform that cannot provide the
# fixed little-endian layouts must fail at import rather than corrupt a lane.
assert WORD_STRUCT.size == WORD_BYTES
assert LANE_STRUCT.size == LANE_BYTES
assert PAIR_STRUCT.size == PAIR_BYTES
assert FIRST_CHANNEL_WORD + CHANNEL_COUNT == GENERATION_WORD
assert GENERATION_WORD + 1 == FIRST_RESERVED_WORD
assert FIRST_RESERVED_WORD + RESERVED_COUNT == LANE_WORDS
assert LANE_BYTES == 128 and PAIR_BYTES == 256


# ---------------------------------------------------------------------------
# errors and scalar validation
# ---------------------------------------------------------------------------
class LaneError(ValueError):
    """Base class for all lane validation failures."""


class LaneBoundsError(LaneError):
    """A word index, channel index, reserved slot or lane size is out of range."""


class LaneTagError(LaneError):
    """The tag/version word or a channel word violates the lane's tag."""


class IdentityRewriteError(LaneError):
    """An attempt was made to replace a lane's tag/version identity."""


def _require_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaneError("%s must be an integer: %r" % (name, value))
    return value


def _check_word(value, name: str = "word") -> int:
    number = _require_int(value, name)
    if not 0 <= number <= WORD_MASK:
        raise LaneBoundsError(
            "%s must be an unsigned 32-bit word in [0, %d]: %r"
            % (name, WORD_MASK, value))
    return number


def _check_word_index(index, name: str = "index") -> int:
    number = _require_int(index, name)
    if not 0 <= number < LANE_WORDS:
        raise LaneBoundsError(
            "%s must be in [0, %d): %r" % (name, LANE_WORDS, index))
    return number


def _check_channel(channel) -> int:
    number = _require_int(channel, "channel")
    if not 0 <= number < CHANNEL_COUNT:
        raise LaneBoundsError(
            "channel must be in [0, %d): %r" % (CHANNEL_COUNT, channel))
    return number


def _check_reserved_index(index) -> int:
    number = _require_int(index, "reserved index")
    if not 0 <= number < RESERVED_COUNT:
        raise LaneBoundsError(
            "reserved index must be in [0, %d): %r"
            % (RESERVED_COUNT, index))
    return number


def _check_generation(generation) -> int:
    return _check_word(generation, "generation")


def _check_tag(tag) -> int:
    number = _require_int(tag, "tag")
    if number not in KNOWN_TAGS:
        raise LaneTagError(
            "tag must be one of %s: %r"
            % (sorted(TAG_NAMES), tag))
    return number


def _check_version(version) -> int:
    number = _require_int(version, "version")
    if number != LANE_VERSION:
        raise LaneTagError(
            "unsupported lane version %r (expected %d)"
            % (version, LANE_VERSION))
    return number


def _coerce_orientation(orientation) -> int:
    if isinstance(orientation, str):
        try:
            return ORIENT_NAMES[orientation.strip().lower()]
        except KeyError as error:
            raise LaneError(
                "orientation must be forward/+1 or reverse/-1: %r"
                % (orientation,)) from error
    number = _require_int(orientation, "orientation")
    if number not in (ORIENT_FORWARD, ORIENT_REVERSE):
        raise LaneError(
            "orientation must be forward/+1 or reverse/-1: %r"
            % (orientation,))
    return number


# ---------------------------------------------------------------------------
# tag/version and NaN payload helpers
# ---------------------------------------------------------------------------
def tag_version_word(tag, version: int = LANE_VERSION) -> int:
    """Return the packed tag/version identity word."""
    return (_check_tag(tag) << TAG_SHIFT) | _check_version(version)


def split_tag_version(word) -> tuple[int, int]:
    """Split a raw word into ``(tag, version)`` without validating the tag."""
    value = _check_word(word, "tag/version word")
    return value >> TAG_SHIFT, value & VERSION_MASK


def is_nan(word) -> bool:
    """True when ``word`` has IEEE-754 NaN exponent/mantissa bits."""
    value = _check_word(word)
    return ((value & NAN_EXPONENT_MASK) == NAN_EXPONENT_MASK
            and (value & NAN_MANTISSA_MASK) != 0)


def is_quiet_nan(word) -> bool:
    """True for an IEEE-754 quiet-NaN bit pattern (either sign)."""
    value = _check_word(word)
    return is_nan(value) and bool(value & NAN_QUIET_BIT)


def is_signaling_nan(word) -> bool:
    """True for an IEEE-754 signalling-NaN bit pattern (either sign)."""
    value = _check_word(word)
    return is_nan(value) and not (value & NAN_QUIET_BIT)


def nan_class(word) -> str | None:
    """Return ``"quiet"``, ``"signaling"`` or ``None`` for a raw word."""
    value = _check_word(word)
    if is_quiet_nan(value):
        return "quiet"
    if is_signaling_nan(value):
        return "signaling"
    return None


def _validate_channel_word(tag: int, word: int, channel: int) -> None:
    if tag == TAG_RAW:
        return
    if tag == TAG_QUIET and not is_quiet_nan(word):
        raise LaneTagError(
            "quiet lane channel %d requires a quiet NaN word: 0x%08X"
            % (channel, word))
    if tag == TAG_SIGNALING and not is_signaling_nan(word):
        raise LaneTagError(
            "signaling lane channel %d requires a signalling NaN word: 0x%08X"
            % (channel, word))


def _coerce_words(words) -> tuple[int, ...]:
    if isinstance(words, Lane):
        return words.words
    if isinstance(words, str):
        raise TypeError("lane words must be an iterable of 32 integers")
    if isinstance(words, (bytes, bytearray, memoryview)):
        raise TypeError("packed lane bytes must be passed to unpack_lane()")
    try:
        values = tuple(words)
    except TypeError as error:
        raise TypeError("lane words must be an iterable of 32 integers") from error
    if len(values) != LANE_WORDS:
        raise LaneBoundsError(
            "a lane has exactly %d words, got %d" % (LANE_WORDS, len(values)))
    return tuple(_check_word(value, "word[%d]" % index)
                 for index, value in enumerate(values))


class Lane:
    """Immutable 128-byte raw word lane.

    The tag/version word is an identity, not ordinary payload.  All operations
    that would change a word return a new :class:`Lane`; the receiver is never
    mutated.  Writing the current tag/version value back is an idempotent
    no-op, while writing a different identity raises
    :class:`IdentityRewriteError`.
    """

    __slots__ = ("_words",)

    def __init__(self, words):
        values = _coerce_words(words)
        tag, version = split_tag_version(values[TAG_VERSION_WORD])
        _check_tag(tag)
        _check_version(version)
        for channel in range(CHANNEL_COUNT):
            _validate_channel_word(
                tag, values[FIRST_CHANNEL_WORD + channel], channel)
        self._words = values

    # -- construction ---------------------------------------------------
    @classmethod
    def new(cls, tag=TAG_RAW, *, version: int = LANE_VERSION,
            generation: int = 0, channels=None, reserved=None) -> "Lane":
        """Create a fresh lane with a fixed identity.

        ``channels`` defaults to zero words for a raw lane and to a canonical
        quiet/signalling NaN word for the corresponding tag.  ``reserved``
        defaults to two zero words.
        """
        tag = _check_tag(tag)
        identity = tag_version_word(tag, version)
        generation = _check_generation(generation)
        if channels is None:
            default = {
                TAG_RAW: 0,
                TAG_QUIET: QUIET_NAN_WORD,
                TAG_SIGNALING: SIGNALING_NAN_WORD,
            }[tag]
            channel_words = (default,) * CHANNEL_COUNT
        else:
            if isinstance(channels, (bytes, bytearray, memoryview, str)):
                raise TypeError("channels must be an iterable of 28 integers")
            try:
                channel_words = tuple(channels)
            except TypeError as error:
                raise TypeError(
                    "channels must be an iterable of 28 integers") from error
            if len(channel_words) != CHANNEL_COUNT:
                raise LaneBoundsError(
                    "a lane has exactly %d channels, got %d"
                    % (CHANNEL_COUNT, len(channel_words)))
            channel_words = tuple(
                _check_word(word, "channel[%d]" % index)
                for index, word in enumerate(channel_words))
        if reserved is None:
            reserved_words = (0,) * RESERVED_COUNT
        else:
            if isinstance(reserved, (bytes, bytearray, memoryview, str)):
                raise TypeError("reserved must be an iterable of 2 integers")
            try:
                reserved_words = tuple(reserved)
            except TypeError as error:
                raise TypeError(
                    "reserved must be an iterable of 2 integers") from error
            if len(reserved_words) != RESERVED_COUNT:
                raise LaneBoundsError(
                    "a lane has exactly %d reserved words, got %d"
                    % (RESERVED_COUNT, len(reserved_words)))
            reserved_words = tuple(
                _check_word(word, "reserved[%d]" % index)
                for index, word in enumerate(reserved_words))
        words = ((identity,) + channel_words + (generation,) + reserved_words)
        return cls(words)

    # -- metadata -------------------------------------------------------
    @property
    def words(self) -> tuple[int, ...]:
        return self._words

    @property
    def tag(self) -> int:
        return self._words[TAG_VERSION_WORD] >> TAG_SHIFT

    @property
    def version(self) -> int:
        return self._words[TAG_VERSION_WORD] & VERSION_MASK

    @property
    def identity(self) -> int:
        return self._words[TAG_VERSION_WORD]

    @property
    def generation(self) -> int:
        return self._words[GENERATION_WORD]

    @property
    def channels(self) -> tuple[int, ...]:
        return self._words[FIRST_CHANNEL_WORD:GENERATION_WORD]

    @property
    def reserved(self) -> tuple[int, ...]:
        return self._words[FIRST_RESERVED_WORD:]

    # -- word access ----------------------------------------------------
    def read_word(self, index) -> int:
        return self._words[_check_word_index(index)]

    def write_word(self, index, word) -> "Lane":
        index = _check_word_index(index)
        word = _check_word(word)
        current = self._words[index]
        if index == TAG_VERSION_WORD:
            if word != current:
                raise IdentityRewriteError(
                    "cannot rewrite tag/version identity 0x%08X as 0x%08X"
                    % (current, word))
            return self
        if word == current:
            return self
        values = list(self._words)
        values[index] = word
        return Lane(values)

    def get_channel(self, channel) -> int:
        return self._words[FIRST_CHANNEL_WORD + _check_channel(channel)]

    def set_channel(self, channel, word) -> "Lane":
        channel = _check_channel(channel)
        return self.write_word(FIRST_CHANNEL_WORD + channel, word)

    def get_generation(self) -> int:
        return self._words[GENERATION_WORD]

    def set_generation(self, generation) -> "Lane":
        return self.write_word(GENERATION_WORD, _check_generation(generation))

    def bump_generation(self, delta: int = 1) -> "Lane":
        delta = _require_int(delta, "delta")
        return self.set_generation((self.generation + delta) & WORD_MASK)

    def get_reserved(self, index) -> int:
        return self._words[FIRST_RESERVED_WORD + _check_reserved_index(index)]

    def set_reserved(self, index, word) -> "Lane":
        index = _check_reserved_index(index)
        return self.write_word(FIRST_RESERVED_WORD + index, word)

    # -- orientation ----------------------------------------------------
    def mirror(self) -> "Lane":
        """Return the reverse-orientation lane.

        The tag/version identity, generation and reserved words stay in their
        fixed positions; only the channel order is reversed
        (``channel c <-> CHANNEL_COUNT - 1 - c``).  This is an involution and
        preserves every 32-bit payload exactly, including quiet/signalling NaN
        class and sign bits.
        """
        words = (
            (self._words[TAG_VERSION_WORD],)
            + tuple(reversed(self.channels))
            + (self._words[GENERATION_WORD],)
            + self.reserved
        )
        return Lane(words)

    def deposit_oriented(self, channel, word, orientation=ORIENT_FORWARD) -> "Lane":
        """Add ``word`` to a channel with an orientation sign, modulo 2**32."""
        channel = _check_channel(channel)
        word = _check_word(word)
        direction = _coerce_orientation(orientation)
        delta = word if direction == ORIENT_FORWARD else (-word) & WORD_MASK
        return self.set_channel(
            channel, (self.get_channel(channel) + delta) & WORD_MASK)

    # -- byte encoding --------------------------------------------------
    def to_bytes(self) -> bytes:
        return LANE_STRUCT.pack(*self._words)

    def pack(self) -> bytes:
        return self.to_bytes()

    # -- value protocol -------------------------------------------------
    def __len__(self) -> int:
        return LANE_WORDS

    def __iter__(self):
        return iter(self._words)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._words[index]
        return self.read_word(index)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Lane):
            return NotImplemented
        return self._words == other._words

    def __hash__(self) -> int:
        return hash(self._words)

    def __repr__(self) -> str:
        return ("Lane(tag=%s, version=%d, generation=%d, channels=%r)"
                % (TAG_NAMES.get(self.tag, self.tag), self.version,
                   self.generation, self.channels))


class Pair:
    """Compact native pair: ``forward[32]`` then ``reverse[32]``.

    A pair is exactly 256 bytes / 64 little-endian 32-bit words.  The two lanes
    keep independent identities (tag/version) and generations.
    """

    __slots__ = ("forward", "reverse")

    def __init__(self, forward, reverse):
        self.forward = _coerce_lane(forward)
        self.reverse = _coerce_lane(reverse)

    @classmethod
    def new(cls, forward_tag=TAG_RAW, reverse_tag=None, *,
            version: int = LANE_VERSION, generation: int = 0) -> "Pair":
        if reverse_tag is None:
            reverse_tag = forward_tag
        return cls(
            Lane.new(forward_tag, version=version, generation=generation),
            Lane.new(reverse_tag, version=version, generation=generation),
        )

    @classmethod
    def from_bytes(cls, data) -> "Pair":
        return unpack_pair(data)

    @property
    def s_plus(self) -> Lane:
        return self.forward

    @property
    def s_minus(self) -> Lane:
        return self.reverse

    @property
    def forward_lane(self) -> Lane:
        return self.forward

    @property
    def reverse_lane(self) -> Lane:
        return self.reverse

    @property
    def words(self) -> tuple[int, ...]:
        return self.forward.words + self.reverse.words

    def lane(self, orientation=ORIENT_FORWARD) -> Lane:
        direction = _coerce_orientation(orientation)
        return self.forward if direction == ORIENT_FORWARD else self.reverse

    def with_lane(self, orientation, lane) -> "Pair":
        direction = _coerce_orientation(orientation)
        lane = _coerce_lane(lane)
        if direction == ORIENT_FORWARD:
            return Pair(lane, self.reverse)
        return Pair(self.forward, lane)

    def opposite(self) -> "Pair":
        """Swap forward and reverse orientation without changing lane contents."""
        return Pair(self.reverse, self.forward)

    def mirror(self) -> "Pair":
        """Reverse the channel order of both lanes."""
        return Pair(self.forward.mirror(), self.reverse.mirror())

    def deposit_oriented(self, channel, word, orientation=ORIENT_FORWARD) -> "Pair":
        """Deposit a directed word while keeping the pair exactly antisymmetric.

        A forward deposit adds ``word`` to the forward lane and subtracts it
        from the reverse lane; a reverse deposit does the converse.  Both
        updates are exact modulo ``2**32``.  If the pair starts antisymmetric
        (``forward + reverse == 0`` per channel), it remains so.
        """
        channel = _check_channel(channel)
        word = _check_word(word)
        direction = _coerce_orientation(orientation)
        delta = word if direction == ORIENT_FORWARD else (-word) & WORD_MASK
        forward = self.forward.set_channel(
            channel,
            (self.forward.get_channel(channel) + delta) & WORD_MASK)
        reverse = self.reverse.set_channel(
            channel,
            (self.reverse.get_channel(channel) - delta) & WORD_MASK)
        return Pair(forward, reverse)

    def to_bytes(self) -> bytes:
        return self.forward.to_bytes() + self.reverse.to_bytes()

    def pack(self) -> bytes:
        return self.to_bytes()

    def __iter__(self):
        return iter((self.forward, self.reverse))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Pair):
            return NotImplemented
        return self.forward == other.forward and self.reverse == other.reverse

    def __hash__(self) -> int:
        return hash((self.forward, self.reverse))

    def __repr__(self) -> str:
        return "Pair(forward=%r, reverse=%r)" % (self.forward, self.reverse)


# ---------------------------------------------------------------------------
# byte and lane coercion
# ---------------------------------------------------------------------------
def _as_bytes(data, size: int, name: str) -> bytes:
    if isinstance(data, str):
        raise TypeError("%s data must be bytes-like, not str" % name)
    try:
        view = memoryview(data)
    except TypeError as error:
        raise TypeError("%s data must be a bytes-like object" % name) from error
    try:
        view = view.cast("B")
    except TypeError:
        view = memoryview(view.tobytes())
    if view.nbytes != size:
        raise LaneBoundsError(
            "%s must be exactly %d bytes, got %d"
            % (name, size, view.nbytes))
    return view.tobytes()


def _coerce_lane(value) -> Lane:
    if isinstance(value, Lane):
        return value
    if isinstance(value, Pair):
        raise TypeError(
            "a Pair holds two lanes; select .forward or .reverse explicitly")
    if isinstance(value, str):
        raise TypeError("value is not a lane")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return unpack_lane(value)
    try:
        return Lane(value)
    except TypeError as error:
        raise TypeError("value is not a lane") from error


def _coerce_pair(value) -> Pair:
    if isinstance(value, Pair):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return unpack_pair(value)
    if isinstance(value, (Lane, str)):
        raise TypeError("value is not a pair")
    try:
        items = tuple(value)
    except TypeError as error:
        raise TypeError("value is not a pair") from error
    if len(items) != PAIR_LANES:
        raise LaneBoundsError(
            "a pair has exactly two lanes, got %d" % len(items))
    return Pair(items[0], items[1])


def pack_lane(lane) -> bytes:
    """Encode a lane (or 32 words) as exactly 128 little-endian bytes."""
    return _coerce_lane(lane).to_bytes()


def unpack_lane(data) -> Lane:
    """Decode exactly 128 little-endian bytes into a validated :class:`Lane`."""
    return Lane(LANE_STRUCT.unpack(_as_bytes(data, LANE_BYTES, "lane")))


def pack_pair(pair) -> bytes:
    """Encode a pair as ``forward[32]`` followed by ``reverse[32]`` (256 bytes)."""
    return _coerce_pair(pair).to_bytes()


def unpack_pair(data) -> Pair:
    """Decode exactly 256 bytes into a :class:`Pair`."""
    raw = _as_bytes(data, PAIR_BYTES, "pair")
    return Pair(Lane(LANE_STRUCT.unpack(raw[:LANE_BYTES])),
                Lane(LANE_STRUCT.unpack(raw[LANE_BYTES:])))


# ---------------------------------------------------------------------------
# functional word/channel API
# ---------------------------------------------------------------------------
def read_word(lane, index) -> int:
    return _coerce_lane(lane).read_word(index)


def write_word(lane, index, word) -> Lane:
    return _coerce_lane(lane).write_word(index, word)


def get_channel(lane, channel) -> int:
    return _coerce_lane(lane).get_channel(channel)


def set_channel(lane, channel, word) -> Lane:
    return _coerce_lane(lane).set_channel(channel, word)


def get_generation(lane) -> int:
    return _coerce_lane(lane).get_generation()


def set_generation(lane, generation) -> Lane:
    return _coerce_lane(lane).set_generation(generation)


def bump_generation(lane, delta: int = 1) -> Lane:
    return _coerce_lane(lane).bump_generation(delta)


def get_reserved(lane, index) -> int:
    return _coerce_lane(lane).get_reserved(index)


def set_reserved(lane, index, word) -> Lane:
    return _coerce_lane(lane).set_reserved(index, word)


def mirror_lane(lane) -> Lane:
    """Reverse channel orientation while preserving identity and payload bits."""
    return _coerce_lane(lane).mirror()


def deposit_oriented(lane, channel, word, orientation=ORIENT_FORWARD):
    """Deposit ``word`` additively, exactly modulo ``2**32``.

    ``lane`` may be a :class:`Lane` or a :class:`Pair`.  A lane receives a
    signed modular addition.  A pair updates both rails antisymmetrically.
    """
    if isinstance(lane, Pair):
        return lane.deposit_oriented(channel, word, orientation)
    return _coerce_lane(lane).deposit_oriented(channel, word, orientation)


def _cycle_operand(value, channel: int) -> int:
    if isinstance(value, Pair):
        value = value.forward
    if isinstance(value, Lane):
        return value.get_channel(channel)
    if isinstance(value, bool):
        raise LaneError("cycle operand must be a word or lane, not bool")
    if isinstance(value, int):
        return _check_word(value, "cycle operand")
    return _coerce_lane(value).get_channel(channel)


def cycle_word(a, b=None, c=None, channel=0) -> int:
    """Return ``(a + b - c)`` for one channel, modulo ``2**32``.

    Operands may be lanes, pairs (forward lane), 32-word iterables, byte
    strings or raw 32-bit integers.  Passing a single sequence of three
    operands is also accepted.  The result is the exact raw-word triangle
    observable used by the Round 029 gauge construction.
    """
    if b is None and c is None:
        if isinstance(a, (Lane, Pair, int, str)):
            raise TypeError(
                "cycle_word needs three operands or one sequence of three")
        try:
            items = tuple(a)
        except TypeError as error:
            raise TypeError(
                "cycle_word needs three operands or one sequence of three"
            ) from error
        if len(items) != 3:
            raise LaneBoundsError(
                "cycle_word needs exactly three operands, got %d" % len(items))
        a, b, c = items
    elif b is None or c is None:
        raise TypeError("cycle_word needs all three operands")
    channel = _check_channel(channel)
    first = _cycle_operand(a, channel)
    second = _cycle_operand(b, channel)
    third = _cycle_operand(c, channel)
    return (first + second - third) & WORD_MASK


__all__ = (
    "LANE_WORDS", "LANE_WORD_COUNT", "WORD_BYTES", "WORD_BITS",
    "WORD_MODULUS", "WORD_MASK", "LANE_BYTES", "NATIVE_LANE_BYTES",
    "PAIR_LANES", "PAIR_WORDS", "PAIR_BYTES", "NATIVE_PAIR_BYTES",
    "TAG_VERSION_WORD", "TAG_WORD", "FIRST_CHANNEL_WORD", "CHANNEL_COUNT",
    "LAST_CHANNEL_WORD", "GENERATION_WORD", "FIRST_RESERVED_WORD",
    "RESERVED_COUNT", "LAST_RESERVED_WORD", "VERSION_BITS", "VERSION_MASK",
    "TAG_SHIFT", "TAG_MASK", "LANE_VERSION", "TAG_RAW", "TAG_QUIET",
    "TAG_SIGNALING", "TAG_NAMES", "KNOWN_TAGS", "TAG_RAW_WORDS",
    "TAG_QUIET_NAN", "TAG_SIGNALING_NAN", "NAN_SIGN_BIT",
    "NAN_EXPONENT_MASK", "NAN_MANTISSA_MASK", "NAN_QUIET_BIT",
    "QUIET_NAN_WORD", "SIGNALING_NAN_WORD", "ORIENT_FORWARD",
    "ORIENT_REVERSE", "ORIENT_NAMES", "WORD_STRUCT", "LANE_STRUCT",
    "PAIR_STRUCT", "LaneError", "LaneBoundsError", "LaneTagError",
    "IdentityRewriteError", "Lane", "Pair", "tag_version_word",
    "split_tag_version", "is_nan", "is_quiet_nan", "is_signaling_nan",
    "nan_class", "pack_lane", "unpack_lane", "pack_pair", "unpack_pair",
    "read_word", "write_word", "get_channel", "set_channel",
    "get_generation", "set_generation", "bump_generation", "get_reserved",
    "set_reserved", "mirror_lane", "deposit_oriented", "cycle_word",
)
