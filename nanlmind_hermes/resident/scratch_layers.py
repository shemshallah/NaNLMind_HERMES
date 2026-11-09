"""Per-head L0-L9 scratch layers with a bounded field view.

The module provides a small, dependency-free staging surface that can be used
before a full resident-universe integration exists:

* every one of the 256 heads owns ten independent L0-L9 scratch layers;
* each layer has a fixed number of pages and fixed record slots per page;
* scratch writes are canonical-JSON records and never touch a main register;
* :meth:`ScratchLayers.stage_commit` creates an immutable staged token;
* :meth:`ScratchLayers.commit_staged` is the only path that can change a main
  value, requires explicit approval, and uses a generation compare-and-swap.

The default main store is an in-memory :class:`MemoryMainBackend`.  A caller
can inject a different backend object (``read_main`` /
``compare_and_swap_main``) or a pair of callbacks, so this module is useful
before the resident universe owns the main register.

All records are immutable frozen dataclasses over canonical bytes.  Ring
allocation recycles the fixed slots per head/layer when a layer is full, so
memory use is bounded by construction and never grows with the number of
writes.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import threading
from typing import Any, Callable

HEAD_COUNT = 256
LAYER_COUNT = 10
LAYERS = LAYER_COUNT
LAYER_NAMES = tuple("L%d" % index for index in range(LAYER_COUNT))
L0, L1, L2, L3, L4, L5, L6, L7, L8, L9 = range(LAYER_COUNT)

# Fixed geometry per head/layer.  The defaults keep a fully populated field
# comfortably inside a phone-sized budget while still giving each layer a
# useful ring of scratch records.
PAGES_PER_LAYER = 2
RECORDS_PER_PAGE = 8
SLOTS_PER_PAGE = RECORDS_PER_PAGE
SLOTS_PER_LAYER = PAGES_PER_LAYER * RECORDS_PER_PAGE
RECORD_SLOTS = SLOTS_PER_LAYER

MAX_VALUE_BYTES = 1024
MAX_STAGED_TOKENS = 256
MAX_SLOTS_PER_LAYER = 4096
MAX_VALUE_LIMIT = 64 * 1024

SCRATCH_SCHEMA = "nanlm.resident-scratch-layer/1"
FIELD_SCHEMA = "nanlm.resident-scratch-field/1"
TOKEN_SCHEMA = "nanlm.resident-scratch-token/1"
COMMIT_SCHEMA = "nanlm.resident-scratch-commit/1"
MAIN_SCHEMA = "nanlm.resident-main-register/1"

SCRATCH_SCOPE = "scratch"
MAIN_SCOPE = "main"


class ScratchLayerError(ValueError):
    """Base error for invalid scratch-layer operations."""


class ScratchLayerBoundsError(ScratchLayerError):
    """Raised when a head, layer, slot geometry or value is out of bounds."""


class ScratchLayerBackendError(ScratchLayerError):
    """Raised when an injected main-store backend violates the protocol."""


# ---------------------------------------------------------------------------
# Validation and canonical values
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScratchLayerBoundsError("%s must be an integer: %r" % (name, value))
    return value


def check_head(head: Any) -> int:
    """Validate and return a head id in ``0..255``."""
    value = _require_int(head, "head")
    if not 0 <= value < HEAD_COUNT:
        raise ScratchLayerBoundsError(
            "head must be in [0, %d): %r" % (HEAD_COUNT, head))
    return value


def layer_index(layer: Any) -> int:
    """Validate a layer id, accepting ``5``, ``"L5"`` or ``"l5"``."""
    if isinstance(layer, str):
        text = layer.strip().upper()
        if text.startswith("L"):
            text = text[1:]
        try:
            layer = int(text, 10)
        except ValueError as exc:
            raise ScratchLayerBoundsError(
                "layer must identify L0..L9: %r" % (layer,)) from exc
    value = _require_int(layer, "layer")
    if not 0 <= value < LAYER_COUNT:
        raise ScratchLayerBoundsError(
            "layer must be in [0, %d): %r" % (LAYER_COUNT, layer))
    return value


def layer_name(layer: Any) -> str:
    """Return the canonical ``L0``..``L9`` name for a layer."""
    return LAYER_NAMES[layer_index(layer)]


def encode_value(value: Any, *, limit: int = MAX_VALUE_BYTES) -> bytes:
    """Encode a value as deterministic, bounded canonical JSON bytes."""
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScratchLayerError("value is not canonical-JSON encodable") from exc
    if len(raw) > limit:
        raise ScratchLayerBoundsError(
            "value is %d bytes; the bound is %d" % (len(raw), limit))
    return raw


def decode_value(raw: bytes) -> Any:
    """Decode canonical JSON bytes into a fresh Python value."""
    return json.loads(bytes(raw).decode("utf-8"))


def value_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _check_generation(value: Any, name: str = "generation") -> int:
    number = _require_int(value, name)
    if number < 0:
        raise ScratchLayerBoundsError("%s must be non-negative" % name)
    return number


def _check_geometry(value: Any, name: str) -> int:
    number = _require_int(value, name)
    if number <= 0 or number > MAX_SLOTS_PER_LAYER:
        raise ScratchLayerBoundsError(
            "%s must be in [1, %d]: %r" % (name, MAX_SLOTS_PER_LAYER, value))
    return number


def _check_value_limit(value: Any) -> int:
    number = _require_int(value, "max_value_bytes")
    if number <= 0 or number > MAX_VALUE_LIMIT:
        raise ScratchLayerBoundsError(
            "max_value_bytes must be in [1, %d]: %r"
            % (MAX_VALUE_LIMIT, value))
    return number


# ---------------------------------------------------------------------------
# Immutable records and views
# ---------------------------------------------------------------------------
class LayerView(dict):
    """A plain-dict compatible layer reading with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __eq__(self, other: Any) -> bool:
        # Keep exact dict comparison, but also allow a caller to compare the
        # view directly with the latest value (``read_layer(...) == value``).
        if isinstance(other, Mapping):
            return dict.__eq__(self, other)
        return self.get("value") == other

    __hash__ = None


class FieldView(dict):
    """A plain-dict compatible field snapshot with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class ScratchRecord:
    """One immutable scratch record in a fixed head/layer slot."""

    head: int
    layer: int
    page: int
    slot: int
    revision: int
    generation: int
    payload: bytes
    digest: str
    approved: bool = False

    @property
    def value(self) -> Any:
        return decode_value(self.payload)

    @property
    def layer_name(self) -> str:
        return LAYER_NAMES[self.layer]

    def as_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "layer": self.layer,
            "layer_name": self.layer_name,
            "page": self.page,
            "slot": self.slot,
            "revision": self.revision,
            "generation": self.generation,
            "value": self.value,
            "digest": self.digest,
            "approved": self.approved,
            "scope": SCRATCH_SCOPE,
        }


@dataclass(frozen=True)
class MainRecord:
    """One immutable main-register value owned by a backend slot."""

    head: int
    layer: int
    generation: int
    payload: bytes | None
    digest: str | None
    token_id: str | None = None

    @property
    def value(self) -> Any:
        if self.payload is None:
            return None
        return decode_value(self.payload)

    @property
    def layer_name(self) -> str:
        return LAYER_NAMES[self.layer]

    def as_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "layer": self.layer,
            "layer_name": self.layer_name,
            "generation": self.generation,
            "value": self.value,
            "digest": self.digest,
            "token_id": self.token_id,
            "scope": MAIN_SCOPE,
        }


@dataclass(frozen=True)
class StagedToken(Mapping[str, Any]):
    """Immutable token produced by :meth:`ScratchLayers.stage_commit`."""

    token_id: str
    head: int
    layer: int
    payload: bytes
    digest: str
    expected_generation: int
    staged_generation: int
    sequence: int

    @property
    def proposal(self) -> Any:
        return decode_value(self.payload)

    @property
    def value(self) -> Any:
        return self.proposal

    @property
    def layer_name(self) -> str:
        return LAYER_NAMES[self.layer]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TOKEN_SCHEMA,
            "token_id": self.token_id,
            "head": self.head,
            "layer": self.layer,
            "layer_name": self.layer_name,
            "proposal": self.proposal,
            "digest": self.digest,
            "expected_generation": self.expected_generation,
            "staged_generation": self.staged_generation,
            "sequence": self.sequence,
            "scope": MAIN_SCOPE,
        }

    def __getitem__(self, key: str) -> Any:
        try:
            return self.as_dict()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __iter__(self):
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


class _LayerState:
    """Fixed ring of scratch slots for exactly one head/layer pair."""

    __slots__ = ("slots", "cursor", "revision", "writes", "evictions")

    def __init__(self, capacity: int):
        self.slots: list[ScratchRecord | None] = [None] * capacity
        self.cursor = 0
        self.revision = 0
        self.writes = 0
        self.evictions = 0


# ---------------------------------------------------------------------------
# Main-store backends
# ---------------------------------------------------------------------------
class MemoryMainBackend:
    """Bounded in-memory main register with generation compare-and-swap.

    The backend is deliberately tiny and complete: it is the default used
    before the resident universe exposes a comparable method.  Each of the
    256 x 10 main slots has a generation that starts at zero and increases by
    exactly one on every successful commit.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._slots: list[list[MainRecord | None]] = [
            [None] * LAYER_COUNT for _ in range(HEAD_COUNT)
        ]
        self._commits = 0

    def read_main(self, head: Any, layer: Any) -> MainRecord | None:
        head = check_head(head)
        layer = layer_index(layer)
        with self._lock:
            return self._slots[head][layer]

    def compare_and_swap_main(
        self,
        head: Any,
        layer: Any,
        *,
        expected_generation: int,
        value: Any = None,
        value_bytes: bytes | None = None,
        digest: str | None = None,
        token_id: str | None = None,
    ) -> dict[str, Any]:
        head = check_head(head)
        layer = layer_index(layer)
        expected = _check_generation(expected_generation, "expected_generation")
        if value_bytes is None:
            value_bytes = encode_value(value)
            value_bytes = bytes(value_bytes)
        else:
            value_bytes = bytes(value_bytes)
        expected_digest = str(digest or value_digest(value_bytes))
        with self._lock:
            current = self._slots[head][layer]
            current_generation = current.generation if current is not None else 0
            if expected != current_generation:
                return {
                    "applied": False,
                    "accepted": False,
                    "status": "stale_generation",
                    "generation": current_generation,
                    "expected_generation": expected,
                    "reasons": ["generation compare-and-swap failed"],
                }
            new_generation = current_generation + 1
            record = MainRecord(
                head=head, layer=layer, generation=new_generation,
                payload=value_bytes, digest=expected_digest,
                token_id=None if token_id is None else str(token_id),
            )
            self._slots[head][layer] = record
            self._commits += 1
            return {
                "applied": True,
                "accepted": True,
                "status": "committed",
                "generation": new_generation,
                "expected_generation": expected,
                "record": record,
            }

    def main_generation(self, head: Any, layer: Any) -> int:
        record = self.read_main(head, layer)
        return record.generation if record is not None else 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            used = sum(
                1 for row in self._slots for record in row
                if record is not None)
            generation = max(
                (record.generation for row in self._slots for record in row
                 if record is not None), default=0)
        return {
            "backend": "memory",
            "slots": HEAD_COUNT * LAYER_COUNT,
            "committed_slots": used,
            "commits": self._commits,
            "max_generation": generation,
        }


def _call_filtered(method: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Call ``method`` with only the keyword arguments it accepts."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**dict(kwargs))
    parameters = signature.parameters
    if any(parameter.kind == parameter.VAR_KEYWORD
           for parameter in parameters.values()):
        return method(**dict(kwargs))
    accepted = {
        name: value for name, value in kwargs.items()
        if name in parameters
        and parameters[name].kind in (
            parameters[name].POSITIONAL_OR_KEYWORD,
            parameters[name].KEYWORD_ONLY,
        )
    }
    return method(**accepted)


class _CallbackMainBackend:
    """Adapter for integrations that only have read/commit callbacks."""

    def __init__(self, read_callback: Callable[..., Any] | None = None,
                 commit_callback: Callable[..., Any] | None = None):
        self.read_callback = read_callback
        self.commit_callback = commit_callback

    def read_main(self, head: int, layer: int) -> Any:
        if self.read_callback is None:
            return None
        return _call_filtered(
            self.read_callback, {"head": head, "layer": layer})

    def compare_and_swap_main(self, head: int, layer: int, *,
                              expected_generation: int,
                              value: Any = None,
                              value_bytes: bytes | None = None,
                              digest: str | None = None,
                              token_id: str | None = None) -> Any:
        if self.commit_callback is None:
            return {
                "applied": False,
                "status": "backend_unavailable",
                "generation": expected_generation,
                "reasons": ["no commit callback configured"],
            }
        return _call_filtered(self.commit_callback, {
            "head": head,
            "layer": layer,
            "expected_generation": expected_generation,
            "value": value,
            "value_bytes": value_bytes,
            "proposal": value,
            "digest": digest,
            "token_id": token_id,
        })


def _normalise_main_record(raw: Any, head: int, layer: int) -> MainRecord | None:
    if raw is None:
        return None
    if isinstance(raw, MainRecord):
        return raw
    if isinstance(raw, bool):
        raise ScratchLayerBackendError("main read returned a boolean")
    if isinstance(raw, int):
        return MainRecord(
            head=head, layer=layer,
            generation=_check_generation(raw, "generation"),
            payload=None, digest=None, token_id=None)
    if isinstance(raw, Mapping):
        generation = raw.get(
            "generation", raw.get("main_generation", raw.get("revision", 0)))
        generation = _check_generation(generation, "generation")
        payload = raw.get("value_bytes", raw.get("payload", raw.get("raw")))
        if payload is not None:
            payload = bytes(payload)
        elif "value" in raw:
            payload = encode_value(raw.get("value"), limit=MAX_VALUE_LIMIT)
        digest = raw.get("digest")
        if digest is not None:
            digest = str(digest)
        token_id = raw.get("token_id")
        if token_id is not None:
            token_id = str(token_id)
        return MainRecord(
            head=head, layer=layer, generation=generation,
            payload=payload, digest=digest, token_id=token_id)
    generation = getattr(raw, "generation", None)
    if generation is None:
        raise ScratchLayerBackendError(
            "main read must return a record, mapping, integer or None")
    payload = getattr(raw, "payload", None)
    if payload is not None:
        payload = bytes(payload)
    digest = getattr(raw, "digest", None)
    token_id = getattr(raw, "token_id", None)
    return MainRecord(
        head=head, layer=layer,
        generation=_check_generation(generation, "generation"),
        payload=payload,
        digest=None if digest is None else str(digest),
        token_id=None if token_id is None else str(token_id))


def _normalise_cas_result(result: Any, *, expected_generation: int,
                          head: int, layer: int) -> dict[str, Any]:
    if isinstance(result, MainRecord):
        return {
            "applied": True,
            "accepted": True,
            "status": "committed",
            "generation": result.generation,
            "expected_generation": expected_generation,
            "record": result,
        }
    if isinstance(result, bool):
        applied = result
        return {
            "applied": applied,
            "accepted": applied,
            "status": "committed" if applied else "denied",
            "generation": expected_generation + 1 if applied else expected_generation,
            "expected_generation": expected_generation,
            "reasons": [] if applied else ["backend denied commit"],
        }
    if isinstance(result, Mapping):
        status = str(result.get("status") or "")
        applied_raw = result.get(
            "applied", result.get("committed", result.get("accepted")))
        applied = applied_raw is True or status == "committed"
        if applied_raw is False and not status:
            status = "denied"
        generation = result.get(
            "generation", result.get("new_generation",
                                     result.get("main_generation")))
        if generation is None:
            generation = expected_generation + 1 if applied \
                else expected_generation
        generation = _check_generation(generation, "generation")
        record = result.get("record")
        if record is not None and not isinstance(record, MainRecord):
            record = _normalise_main_record(record, head, layer)
        reasons = result.get("reasons") or result.get("reason")
        if reasons is None:
            reasons = []
        elif not isinstance(reasons, (list, tuple)):
            reasons = [str(reasons)]
        return {
            "applied": applied,
            "accepted": applied,
            "status": status or ("committed" if applied else "denied"),
            "generation": generation,
            "expected_generation": expected_generation,
            "record": record,
            "reasons": list(reasons),
        }
    return {
        "applied": False,
        "accepted": False,
        "status": "backend_protocol_error",
        "generation": expected_generation,
        "expected_generation": expected_generation,
        "reasons": ["backend did not return a commit result"],
    }


# ---------------------------------------------------------------------------
# Scratch layer field
# ---------------------------------------------------------------------------
class ScratchLayers:
    """A bounded field of independent per-head L0-L9 scratch layers."""

    def __init__(
        self,
        *,
        backend: Any = None,
        read_callback: Callable[..., Any] | None = None,
        commit_callback: Callable[..., Any] | None = None,
        pages_per_layer: int = PAGES_PER_LAYER,
        records_per_page: int = RECORDS_PER_PAGE,
        max_value_bytes: int = MAX_VALUE_BYTES,
        max_staged_tokens: int = MAX_STAGED_TOKENS,
    ):
        if backend is not None and (
                read_callback is not None or commit_callback is not None):
            raise ScratchLayerError(
                "pass either backend or callbacks, not both")
        self.pages_per_layer = _check_geometry(
            pages_per_layer, "pages_per_layer")
        self.records_per_page = _check_geometry(
            records_per_page, "records_per_page")
        self.slots_per_layer = self.pages_per_layer * self.records_per_page
        if self.slots_per_layer > MAX_SLOTS_PER_LAYER:
            raise ScratchLayerBoundsError(
                "slots_per_layer exceeds %d" % MAX_SLOTS_PER_LAYER)
        self.max_value_bytes = _check_value_limit(max_value_bytes)
        self.max_staged_tokens = _check_geometry(
            max_staged_tokens, "max_staged_tokens")
        if backend is None:
            if read_callback is not None or commit_callback is not None:
                backend = _CallbackMainBackend(read_callback, commit_callback)
            else:
                backend = MemoryMainBackend()
        self.backend = backend
        self._lock = threading.RLock()
        self._scratch: list[list[_LayerState | None]] = [
            [None] * LAYER_COUNT for _ in range(HEAD_COUNT)
        ]
        self._generation = 0
        self._writes = 0
        self._evictions = 0
        self._token_sequence = 0
        self._commit_sequence = 0
        self._staged: OrderedDict[str, StagedToken] = OrderedDict()
        self._consumed: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------- geometry
    @property
    def generation(self) -> int:
        """Monotonic scratch-write generation for the whole field."""
        return self._generation

    @property
    def scratch_generation(self) -> int:
        return self._generation

    @property
    def capacity_per_layer(self) -> int:
        return self.slots_per_layer

    @property
    def capacity(self) -> int:
        return HEAD_COUNT * LAYER_COUNT * self.slots_per_layer

    def _state(self, head: int, layer: int, *, create: bool) -> _LayerState | None:
        state = self._scratch[head][layer]
        if state is None and create:
            state = _LayerState(self.slots_per_layer)
            self._scratch[head][layer] = state
        return state

    # ---------------------------------------------------------------- scratch
    def write_layer(self, head: Any, layer: Any, value: Any, *,
                    approved: bool = False) -> dict[str, Any]:
        """Write a scratch record only.

        The value is canonical JSON, bounded by ``max_value_bytes``.  When all
        fixed slots for the layer are full the oldest slot is recycled.  No
        main register or backend is ever consulted here.
        """
        head = check_head(head)
        layer = layer_index(layer)
        payload = encode_value(value, limit=self.max_value_bytes)
        digest = value_digest(payload)
        with self._lock:
            state = self._state(head, layer, create=True)
            assert state is not None
            index = state.cursor
            page = index // self.records_per_page
            slot = index % self.records_per_page
            evicted = state.slots[index] is not None
            generation = self._generation + 1
            revision = state.revision + 1
            record = ScratchRecord(
                head=head, layer=layer, page=page, slot=slot,
                revision=revision, generation=generation,
                payload=payload, digest=digest, approved=approved is True,
            )
            state.slots[index] = record
            state.cursor = (index + 1) % self.slots_per_layer
            state.revision = revision
            state.writes += 1
            if evicted:
                state.evictions += 1
                self._evictions += 1
            self._generation = generation
            self._writes += 1
        return {
            "schema": SCRATCH_SCHEMA,
            "status": "stored",
            "stored": True,
            "scope": SCRATCH_SCOPE,
            "main_touched": False,
            "head": head,
            "layer": layer,
            "layer_name": LAYER_NAMES[layer],
            "page": page,
            "slot": slot,
            "slot_index": index,
            "revision": revision,
            "generation": generation,
            "value": decode_value(payload),
            "digest": digest,
            "approved": approved is True,
            "evicted": evicted,
            "capacity": self.slots_per_layer,
        }

    def _read_layer_unlocked(self, head: int, layer: int) -> LayerView:
        state = self._state(head, layer, create=False)
        if state is None or state.revision == 0:
            return LayerView({
                "schema": SCRATCH_SCHEMA,
                "status": "empty",
                "scope": SCRATCH_SCOPE,
                "head": head,
                "layer": layer,
                "layer_name": LAYER_NAMES[layer],
                "value": None,
                "revision": 0,
                "generation": 0,
                "digest": None,
                "page": None,
                "slot": None,
                "slot_index": None,
                "approved": False,
                "slots_used": 0,
                "capacity": self.slots_per_layer,
                "pages": self.pages_per_layer,
                "records_per_page": self.records_per_page,
                "records": (),
            })
        index = (state.revision - 1) % self.slots_per_layer
        latest = state.slots[index]
        if latest is None:  # Defensive: a ring invariant was broken.
            raise ScratchLayerError(
                "scratch layer %d/%d lost its latest record" % (head, layer))
        records = sorted(
            (record for record in state.slots if record is not None),
            key=lambda record: record.revision)
        return LayerView({
            "schema": SCRATCH_SCHEMA,
            "status": "present",
            "scope": SCRATCH_SCOPE,
            "head": head,
            "layer": layer,
            "layer_name": LAYER_NAMES[layer],
            "value": latest.value,
            "revision": latest.revision,
            "generation": latest.generation,
            "digest": latest.digest,
            "page": latest.page,
            "slot": latest.slot,
            "slot_index": index,
            "approved": latest.approved,
            "slots_used": len(records),
            "capacity": self.slots_per_layer,
            "pages": self.pages_per_layer,
            "records_per_page": self.records_per_page,
            "records": tuple(record.as_dict() for record in records),
        })

    def read_layer(self, head: Any, layer: Any) -> LayerView:
        """Return a deterministic, bounded view of one scratch layer."""
        head = check_head(head)
        layer = layer_index(layer)
        with self._lock:
            return self._read_layer_unlocked(head, layer)

    def read_value(self, head: Any, layer: Any) -> Any:
        """Return only the latest scratch value (or ``None`` when empty)."""
        return self.read_layer(head, layer)["value"]

    def _select_heads(self, heads: Any) -> tuple[int, ...]:
        if heads is None:
            return tuple(range(HEAD_COUNT))
        if isinstance(heads, int) and not isinstance(heads, bool):
            return (check_head(heads),)
        try:
            values = list(heads)
        except TypeError as exc:
            raise ScratchLayerBoundsError(
                "heads must be an iterable of head ids") from exc
        selected = sorted({check_head(head) for head in values})
        return tuple(selected)

    def field_snapshot(self, heads: Iterable[int] | int | None = None) -> FieldView:
        """Return latest scratch values for selected heads as one field.

        Heads are de-duplicated and sorted; layers always run L0..L9.  The
        result contains at most ``len(heads) * 10`` layer views and is cheap to
        serialize or compare across calls.
        """
        selected = self._select_heads(heads)
        with self._lock:
            entries: dict[int, tuple[LayerView, ...]] = {}
            values: dict[int, dict[int, Any]] = {}
            for head in selected:
                views = tuple(
                    self._read_layer_unlocked(head, layer)
                    for layer in range(LAYER_COUNT))
                entries[head] = views
                values[head] = {
                    layer: views[layer]["value"]
                    for layer in range(LAYER_COUNT)
                }
            generation = self._generation
        return FieldView({
            "schema": FIELD_SCHEMA,
            "status": "present",
            "scope": SCRATCH_SCOPE,
            "generation": generation,
            "heads": selected,
            "layers": LAYER_NAMES,
            "entries": entries,
            "values": values,
            "capacity": self.capacity,
        })

    # ----------------------------------------------------------- main reads
    def _read_backend(self, head: int, layer: int) -> MainRecord | None:
        method = None
        for name in ("read_main", "read"):
            candidate = getattr(self.backend, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None:
            raise ScratchLayerBackendError(
                "backend must provide read_main(head, layer)")
        raw = _call_filtered(method, {"head": head, "layer": layer})
        return _normalise_main_record(raw, head, layer)

    def main_record(self, head: Any, layer: Any) -> MainRecord | None:
        """Return the immutable main record for one slot, if any."""
        head = check_head(head)
        layer = layer_index(layer)
        with self._lock:
            return self._read_backend(head, layer)

    def main_generation(self, head: Any, layer: Any) -> int:
        """Return the current main generation (zero when the slot is empty)."""
        record = self.main_record(head, layer)
        return record.generation if record is not None else 0

    def read_main(self, head: Any, layer: Any) -> LayerView:
        """Return a deterministic view of one main-register slot."""
        head = check_head(head)
        layer = layer_index(layer)
        record = self.main_record(head, layer)
        if record is None:
            return LayerView({
                "schema": MAIN_SCHEMA,
                "status": "empty",
                "scope": MAIN_SCOPE,
                "head": head,
                "layer": layer,
                "layer_name": LAYER_NAMES[layer],
                "value": None,
                "generation": 0,
                "digest": None,
                "token_id": None,
            })
        return LayerView(dict(record.as_dict(), status="committed"))

    # -------------------------------------------------------------- staging
    def stage_commit(self, head: Any, layer: Any, proposal: Any, *,
                     expected_generation: int | None = None) -> StagedToken:
        """Create an immutable staged token without touching main.

        The token captures the main generation observed at stage time.  A
        caller may pass ``expected_generation`` explicitly when it already
        knows that base; otherwise the backend is read.
        """
        head = check_head(head)
        layer = layer_index(layer)
        payload = encode_value(proposal, limit=self.max_value_bytes)
        digest = value_digest(payload)
        with self._lock:
            if expected_generation is None:
                current = self._read_backend(head, layer)
                expected = current.generation if current is not None else 0
            else:
                expected = _check_generation(
                    expected_generation, "expected_generation")
            self._token_sequence += 1
            sequence = self._token_sequence
            material = json.dumps(
                [head, layer, digest, expected, sequence],
                sort_keys=True, separators=(",", ":")).encode("utf-8")
            token_id = hashlib.sha256(material).hexdigest()[:32]
            token = StagedToken(
                token_id=token_id, head=head, layer=layer,
                payload=payload, digest=digest,
                expected_generation=expected,
                staged_generation=self._generation,
                sequence=sequence,
            )
            self._staged[token_id] = token
            while len(self._staged) > self.max_staged_tokens:
                self._staged.popitem(last=False)
        return token

    def _coerce_token(self, token: Any) -> StagedToken | None:
        if isinstance(token, StagedToken):
            return token
        if isinstance(token, Mapping):
            token_id = token.get("token_id")
            if token_id is None:
                return None
            return self._staged.get(str(token_id))
        return None

    def _denial(self, token: StagedToken | None, status: str,
                reasons: Iterable[str], *, expected_generation: int | None,
                head: int | None = None, layer: int | None = None,
                generation: int | None = None) -> dict[str, Any]:
        if token is not None:
            head = token.head if head is None else head
            layer = token.layer if layer is None else layer
            token_id = token.token_id
            if expected_generation is None:
                expected_generation = token.expected_generation
        else:
            token_id = None
        if head is not None:
            layer_name = LAYER_NAMES[layer]
        else:
            layer_name = None
        return {
            "schema": COMMIT_SCHEMA,
            "status": status,
            "committed": False,
            "accepted": False,
            "applied": False,
            "main_updated": False,
            "scope": MAIN_SCOPE,
            "head": head,
            "layer": layer,
            "layer_name": layer_name,
            "token_id": token_id,
            "expected_generation": expected_generation,
            "generation": generation,
            "reasons": list(reasons),
        }

    def _commit_backend(self, token: StagedToken, expected: int) -> dict[str, Any]:
        method = None
        for name in ("compare_and_swap_main", "compare_and_swap",
                     "commit_main", "commit"):
            candidate = getattr(self.backend, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None:
            return {
                "applied": False,
                "accepted": False,
                "status": "backend_unavailable",
                "generation": expected,
                "expected_generation": expected,
                "reasons": ["backend must provide compare_and_swap_main"],
            }
        raw = _call_filtered(method, {
            "head": token.head,
            "layer": token.layer,
            "expected_generation": expected,
            "value": token.value,
            "proposal": token.value,
            "value_bytes": token.payload,
            "payload": token.payload,
            "digest": token.digest,
            "token_id": token.token_id,
        })
        return _normalise_cas_result(
            raw, expected_generation=expected,
            head=token.head, layer=token.layer)

    def commit_staged(self, token: Any, *, approved: bool = False,
                      expected_generation: int | None = None) -> dict[str, Any]:
        """Commit one staged token to main with approval and CAS.

        Denials return a structured mapping with ``committed=False`` and never
        mutate a main value.  A successful commit consumes the token; replay is
        denied.
        """
        coerced = self._coerce_token(token)
        with self._lock:
            if approved is not True:
                current = None
                if coerced is not None:
                    try:
                        current = self._read_backend(coerced.head, coerced.layer)
                    except Exception:
                        current = None
                generation = current.generation if current is not None else 0
                return self._denial(
                    coerced, "approval_required", ["approval required"],
                    expected_generation=expected_generation,
                    generation=generation)
            if coerced is None:
                return self._denial(
                    None, "unknown_token", ["token was not staged by this field"],
                    expected_generation=expected_generation)
            token_id = coerced.token_id
            if token_id in self._consumed:
                generation = None
                try:
                    generation = self.main_generation(
                        coerced.head, coerced.layer)
                except Exception:
                    pass
                return self._denial(
                    coerced, "token_consumed",
                    ["staged token was already committed"],
                    expected_generation=expected_generation,
                    generation=generation)
            staged = self._staged.get(token_id)
            if staged is None:
                generation = None
                try:
                    generation = self.main_generation(
                        coerced.head, coerced.layer)
                except Exception:
                    pass
                return self._denial(
                    coerced, "unknown_token",
                    ["token is no longer in the bounded staging ring"],
                    expected_generation=expected_generation,
                    generation=generation)
            expected = staged.expected_generation \
                if expected_generation is None \
                else _check_generation(expected_generation, "expected_generation")
            current_generation = None
            try:
                current_generation = self.main_generation(
                    staged.head, staged.layer)
            except Exception:
                current_generation = None
            if expected != staged.expected_generation:
                return self._denial(
                    staged, "stale_generation",
                    ["expected_generation does not match the staged base"],
                    expected_generation=expected,
                    generation=current_generation)
            result = self._commit_backend(staged, expected)
            if not result.get("applied"):
                status = str(result.get("status") or "denied")
                if status == "committed":
                    status = "denied"
                return self._denial(
                    staged, status,
                    result.get("reasons") or ["main commit denied"],
                    expected_generation=expected,
                    generation=result.get("generation",
                                          current_generation))
            generation = _check_generation(
                result.get("generation", expected + 1), "generation")
            if generation <= expected:
                return self._denial(
                    staged, "backend_generation_violation",
                    ["backend commit did not advance the generation"],
                    expected_generation=expected, generation=generation)
            del self._staged[token_id]
            self._consumed[token_id] = "committed"
            while len(self._consumed) > self.max_staged_tokens:
                self._consumed.popitem(last=False)
            self._commit_sequence += 1
            record = result.get("record")
            if isinstance(record, MainRecord):
                value = record.value
                digest = record.digest or staged.digest
            else:
                value = staged.value
                digest = staged.digest
            return {
                "schema": COMMIT_SCHEMA,
                "status": "committed",
                "committed": True,
                "accepted": True,
                "applied": True,
                "main_updated": True,
                "scope": MAIN_SCOPE,
                "head": staged.head,
                "layer": staged.layer,
                "layer_name": LAYER_NAMES[staged.layer],
                "token_id": staged.token_id,
                "expected_generation": expected,
                "generation": generation,
                "value": value,
                "digest": digest,
                "sequence": self._commit_sequence,
                "reasons": [],
            }

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        """Return bounded-capacity and usage counters."""
        with self._lock:
            allocated_layers = 0
            records = 0
            pages_allocated = 0
            max_revision = 0
            for row in self._scratch:
                for state in row:
                    if state is None:
                        continue
                    allocated_layers += 1
                    occupied = [record for record in state.slots
                                if record is not None]
                    records += len(occupied)
                    pages_allocated += len({record.page for record in occupied})
                    max_revision = max(max_revision, state.revision)
            result = {
                "schema": SCRATCH_SCHEMA,
                "heads": HEAD_COUNT,
                "layers": LAYER_COUNT,
                "layer_names": LAYER_NAMES,
                "pages_per_layer": self.pages_per_layer,
                "records_per_page": self.records_per_page,
                "slots_per_layer": self.slots_per_layer,
                "capacity_records": self.capacity,
                "allocated_layers": allocated_layers,
                "pages_allocated": pages_allocated,
                "scratch_records": records,
                "scratch_writes": self._writes,
                "scratch_evictions": self._evictions,
                "scratch_generation": self._generation,
                "max_revision": max_revision,
                "staged_tokens": len(self._staged),
                "consumed_tokens": len(self._consumed),
                "max_staged_tokens": self.max_staged_tokens,
                "main_commits": self._commit_sequence,
                "max_value_bytes": self.max_value_bytes,
            }
            backend_stats = getattr(self.backend, "stats", None)
            if callable(backend_stats):
                try:
                    result["backend"] = backend_stats()
                except Exception:
                    result["backend"] = {"status": "unavailable"}
            return result


# Friendly aliases for integrations that name the field explicitly.
ScratchLayerField = ScratchLayers
ResidentScratchLayers = ScratchLayers
ScratchRegisterLayers = ScratchLayers


# Module-level conveniences backed by a lazily created default field.  They
# make the small API usable before a caller has a field object; production
# integrations should pass their own field/backend explicitly.
_DEFAULT_FIELD: ScratchLayers | None = None


def default_field() -> ScratchLayers:
    """Return the lazily created process-local field used by the helpers."""
    global _DEFAULT_FIELD
    if _DEFAULT_FIELD is None:
        _DEFAULT_FIELD = ScratchLayers()
    return _DEFAULT_FIELD


def reset_default_field(field: ScratchLayers | None = None) -> ScratchLayers:
    """Replace (or clear) the helper field, primarily for tests."""
    global _DEFAULT_FIELD
    if field is None:
        field = ScratchLayers()
    if not isinstance(field, ScratchLayers):
        raise ScratchLayerError("default field must be a ScratchLayers instance")
    _DEFAULT_FIELD = field
    return _DEFAULT_FIELD


def write_layer(head: Any, layer: Any, value: Any, *,
                approved: bool = False) -> dict[str, Any]:
    return default_field().write_layer(head, layer, value, approved=approved)


def read_layer(head: Any, layer: Any) -> LayerView:
    return default_field().read_layer(head, layer)


def read_value(head: Any, layer: Any) -> Any:
    return default_field().read_value(head, layer)


def field_snapshot(heads: Iterable[int] | int | None = None) -> FieldView:
    return default_field().field_snapshot(heads)


def stage_commit(head: Any, layer: Any, proposal: Any, *,
                 expected_generation: int | None = None) -> StagedToken:
    return default_field().stage_commit(
        head, layer, proposal, expected_generation=expected_generation)


def commit_staged(token: Any, *, approved: bool = False,
                  expected_generation: int | None = None) -> dict[str, Any]:
    return default_field().commit_staged(
        token, approved=approved, expected_generation=expected_generation)


def main_record(head: Any, layer: Any) -> MainRecord | None:
    return default_field().main_record(head, layer)


def main_generation(head: Any, layer: Any) -> int:
    return default_field().main_generation(head, layer)


__all__ = (
    "HEAD_COUNT", "LAYER_COUNT", "LAYERS", "LAYER_NAMES",
    "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9",
    "PAGES_PER_LAYER", "RECORDS_PER_PAGE", "SLOTS_PER_PAGE",
    "SLOTS_PER_LAYER", "RECORD_SLOTS", "MAX_VALUE_BYTES",
    "MAX_STAGED_TOKENS", "MAX_SLOTS_PER_LAYER", "MAX_VALUE_LIMIT",
    "SCRATCH_SCHEMA", "FIELD_SCHEMA", "TOKEN_SCHEMA", "COMMIT_SCHEMA",
    "MAIN_SCHEMA", "SCRATCH_SCOPE", "MAIN_SCOPE",
    "ScratchLayerError", "ScratchLayerBoundsError",
    "ScratchLayerBackendError", "ScratchRecord", "MainRecord",
    "StagedToken", "LayerView", "FieldView", "MemoryMainBackend",
    "ScratchLayers", "ScratchLayerField", "ResidentScratchLayers",
    "ScratchRegisterLayers", "check_head", "layer_index", "layer_name",
    "encode_value", "decode_value", "value_digest",
    "default_field", "reset_default_field", "write_layer", "read_layer",
    "read_value", "field_snapshot", "stage_commit", "commit_staged",
    "main_record", "main_generation",
)
