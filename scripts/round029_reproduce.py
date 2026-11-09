#!/usr/bin/env python3
"""Round 029 reproduction: exact gauge cancellation in NaN payload lanes.

The Round 029 experiment encodes a 48-dimensional semantic vector ``z`` into
three triangle lanes per dimension::

    (0, a) : g[0] - g[a]
    (a, b) : z + g[a] - g[b]
    (0, b) : g[0] - g[b]

with ``a = 2*i + 1`` and ``b = a + 1``.  The cycle readout

    (0, a) + (a, b) - (0, b)

is the ordinary exact integer identity ``z`` and is independent of the
arbitrary vertex gauge ``g``.  The raw ``(a, b)`` lane and a deliberately
mismatched triangle retain the gauge nuisance.

This script is a dependency-light reproduction of that geometric claim.  It
uses only the Python standard library.  Semantic values are fixed-point
integers; each lane value is stored in the 48-bit payload of a quiet NaN
double.  Arithmetic is performed on the extracted payload integers, *not* by
IEEE-754 NaN addition.  A conventional integer-lane implementation is run in
parallel and ties the NaN-carrier implementation bit-for-bit.

Boundaries:

* The exact cancellation is ordinary exact integer algebra.
* A conventional graph implementation ties the NaN-payload implementation.
* There is no NaN-unique advantage and no IEEE-NaN arithmetic claim.
* The default dataset is deterministic synthetic fixed-point data.  An
  optional local JSONL can be loaded with ``--wordnet`` (each line needs a
  48-dimensional ``vector`` and a ``class``/``label``; exactly 16 classes);
  no WordNet benchmark result is claimed in either case.
* No quantum or hardware claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "nanlmind-hermes.round029-reproduce/1"

DIMS = 48
CLASS_COUNT = 16
VERTEX_COUNT = 256
FIXED_POINT_SCALE = 256  # Q8.8
CLASS_CENTER_SPAN = 512
SAMPLE_NOISE_SPAN = 500
DEFAULT_SEED = 29
DEFAULT_SAMPLES = 40
DEFAULT_TRAIN_GAUGE = 2
DEFAULT_TEST_GAUGE = 1 << 20

# A quiet NaN has exponent 0x7ff and the quiet mantissa bit set.  We keep the
# top three mantissa payload bits zero and use the low 48 bits as a signed
# two's-complement lane value.  This leaves the carrier bits stable across
# struct.pack/struct.unpack round trips.
PAYLOAD_BITS = 48
PAYLOAD_MODULUS = 1 << PAYLOAD_BITS
PAYLOAD_MASK = PAYLOAD_MODULUS - 1
PAYLOAD_SIGN = 1 << (PAYLOAD_BITS - 1)
NAN_BASE_BITS = 0x7FF8000000000000
NAN_HIGH_PATTERN = NAN_BASE_BITS >> 48  # 0x7ff8

READOUTS = ("cycle", "raw", "wrong_triangle", "conventional")


# ---------------------------------------------------------------------------
# NaN payload lane codec
# ---------------------------------------------------------------------------


def pack_payload(value: int) -> float:
    """Return a quiet-NaN double carrying ``value`` in its low 48 payload bits.

    ``value`` must fit signed 48-bit two's complement.  The payload is a
    storage slot: the returned float is not used for arithmetic.
    """
    value = int(value)
    if not -PAYLOAD_SIGN <= value < PAYLOAD_SIGN:
        raise ValueError(
            "payload value %d is outside the signed %d-bit lane range"
            % (value, PAYLOAD_BITS)
        )
    bits = NAN_BASE_BITS | (value & PAYLOAD_MASK)
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def unpack_payload(carrier: float) -> int:
    """Extract the signed 48-bit payload from a quiet-NaN carrier."""
    bits = struct.unpack(">Q", struct.pack(">d", float(carrier)))[0]
    if bits >> 48 != NAN_HIGH_PATTERN:
        raise ValueError("carrier is not a quiet 48-bit NaN payload lane")
    raw = bits & PAYLOAD_MASK
    return raw - PAYLOAD_MODULUS if raw & PAYLOAD_SIGN else raw


def payload_bits(carrier: float) -> int:
    """Return the raw 64-bit IEEE-754 representation of ``carrier``."""
    return struct.unpack(">Q", struct.pack(">d", float(carrier)))[0]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def make_synthetic_dataset(
    seed: int = DEFAULT_SEED,
    samples_per_class: int = DEFAULT_SAMPLES,
    *,
    dims: int = DIMS,
    classes: int = CLASS_COUNT,
    fixed_point_scale: int = FIXED_POINT_SCALE,
    center_span: int = CLASS_CENTER_SPAN,
    noise_span: int = SAMPLE_NOISE_SPAN,
) -> dict[str, Any]:
    """Build a deterministic controlled fixed-point semantic dataset.

    Each class has a random integer prototype in ``[-center_span, center_span]``
    per dimension.  Samples add independent integer noise in
    ``[-noise_span, noise_span]``.  The values are fixed-point integers at
    ``fixed_point_scale`` (Q8.8 by default).  No WordNet data is used.
    """
    if dims != DIMS or classes != CLASS_COUNT:
        raise ValueError("the Round 029 reproduction uses 48 dims and 16 classes")
    if samples_per_class < 5:
        raise ValueError("samples_per_class must be at least 5")

    rng = random.Random(seed * 1000003 + 17)
    centers = [
        [rng.randint(-center_span, center_span) for _ in range(dims)]
        for _ in range(classes)
    ]
    if len({tuple(center) for center in centers}) != classes:
        raise RuntimeError("synthetic class centers collided; choose another seed")

    features: list[list[int]] = []
    labels: list[int] = []
    for label, center in enumerate(centers):
        for _ in range(samples_per_class):
            row = [
                center[dimension] + rng.randint(-noise_span, noise_span)
                for dimension in range(dims)
            ]
            features.append(row)
            labels.append(label)

    # Shuffle the rows deterministically so class order is not the data order.
    order = list(range(len(features)))
    rng.shuffle(order)
    features = [features[index] for index in order]
    labels = [labels[index] for index in order]

    return {
        "features": features,
        "labels": labels,
        "class_names": ["semantic_class_%02d" % label for label in range(classes)],
        "dims": dims,
        "classes": classes,
        "fixed_point_scale": fixed_point_scale,
        "source": "deterministic_synthetic_q8_8",
        "wordnet_used": False,
        "claims_wordnet_result": False,
        "samples_per_class": samples_per_class,
        "center_span": center_span,
        "noise_span": noise_span,
    }


def _label_sort_key(label: Any) -> tuple[int, int, str]:
    try:
        return (0, int(label), str(label))
    except (TypeError, ValueError):
        return (1, 0, str(label))


def _coerce_vector(value: Any, dims: int, scale: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != dims:
        raise ValueError("each record needs a %d-dimensional vector" % dims)
    converted: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("boolean vector elements are not features")
        if isinstance(item, int):
            converted.append(item)
        elif isinstance(item, float):
            converted.append(int(round(item * scale)))
        else:
            raise ValueError("vector elements must be ints or floats")
    if any(not -PAYLOAD_SIGN <= item < PAYLOAD_SIGN for item in converted):
        raise ValueError("a vector element does not fit the 48-bit payload lane")
    return converted


def load_wordnet_dataset(
    path: Path | str,
    *,
    dims: int = DIMS,
    classes: int = CLASS_COUNT,
    fixed_point_scale: int = FIXED_POINT_SCALE,
) -> dict[str, Any]:
    """Load an optional local JSONL feature file.

    Each line must contain a 48-dimensional ``vector`` (or ``features`` or
    ``embedding``) and a class label in ``class``/``label``/``lexfile``.  Integer
    elements are treated as already-fixed-point; float elements are multiplied
    by ``fixed_point_scale`` and rounded.  Exactly 16 classes are required.

    This is a caller-supplied feature source, not a WordNet benchmark.  The
    returned metadata always keeps ``claims_wordnet_result`` false.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("WordNet JSONL not found: %s" % source)

    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSON on line %d of %s: %s" % (line_number, source, error)
                ) from error
            if not isinstance(item, dict):
                raise ValueError("line %d is not a JSON object" % line_number)
            vector = item.get("vector", item.get("features", item.get("embedding")))
            if vector is None:
                raise ValueError("line %d has no vector/features/embedding" % line_number)
            label = item.get("class", item.get("label", item.get("lexfile")))
            if label is None:
                raise ValueError("line %d has no class/label/lexfile" % line_number)
            if isinstance(label, (dict, list)):
                raise ValueError("line %d has an unhashable class label" % line_number)
            records.append(
                {
                    "id": str(item.get("id", "line-%d" % line_number)),
                    "label": label,
                    "vector": _coerce_vector(vector, dims, fixed_point_scale),
                    "gloss": str(item.get("gloss", "")),
                }
            )

    if not records:
        raise ValueError("the WordNet JSONL contains no records")
    unique_labels = sorted({record["label"] for record in records}, key=_label_sort_key)
    if len(unique_labels) != classes:
        raise ValueError(
            "the Round 029 reproduction requires exactly %d classes; %s has %d"
            % (classes, source, len(unique_labels))
        )

    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    features = [record["vector"] for record in records]
    labels = [label_to_index[record["label"]] for record in records]
    return {
        "features": features,
        "labels": labels,
        "class_names": [str(label) for label in unique_labels],
        "dims": dims,
        "classes": classes,
        "fixed_point_scale": fixed_point_scale,
        "source": "local_wordnet_jsonl",
        "wordnet_used": True,
        "wordnet_path": str(source),
        "claims_wordnet_result": False,
        "wordnet_note": (
            "Features were read from a caller-supplied local JSONL. This is "
            "not a WordNet benchmark and no WordNet accuracy claim is made."
        ),
        "record_ids": [record["id"] for record in records[:8]],
        "record_count": len(records),
    }


def stratified_split(labels: Sequence[int], seed: int = DEFAULT_SEED) -> dict[str, list[int]]:
    """Deterministically split each class 60/20/20 into train/val/test."""
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)

    rng = random.Random(seed * 1000003 + 29)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(by_class):
        indices = list(by_class[label])
        if len(indices) < 3:
            raise ValueError("each class needs at least 3 samples for a split")
        rng.shuffle(indices)
        count = len(indices)
        train_count = max(1, (count * 3) // 5)
        val_count = max(1, count // 5)
        while train_count + val_count >= count:
            if val_count > 1:
                val_count -= 1
            elif train_count > 1:
                train_count -= 1
            else:
                break
        train.extend(indices[:train_count])
        val.extend(indices[train_count:train_count + val_count])
        test.extend(indices[train_count + val_count:])

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Lanes and readouts
# ---------------------------------------------------------------------------


def triangle_indices(index: int) -> tuple[int, int]:
    """Return the (a, b) vertices for semantic dimension ``index``."""
    if not 0 <= index < DIMS:
        raise ValueError("semantic dimension out of range")
    a = 2 * index + 1
    return a, a + 1


def make_gauge(rng: random.Random, magnitude: int) -> list[int]:
    """Draw one arbitrary K256 vertex gauge with entries in [-m, m]."""
    if magnitude < 0:
        raise ValueError("gauge magnitude must be non-negative")
    if magnitude == 0:
        return [0] * VERTEX_COUNT
    return [rng.randint(-magnitude, magnitude) for _ in range(VERTEX_COUNT)]


def encode_payload_lanes(
    values: Sequence[int], gauge: Sequence[int]
) -> dict[str, list[float]]:
    """Build (0,a), (a,b), (0,b) lanes and store each integer in a NaN payload."""
    if len(values) != DIMS:
        raise ValueError("expected %d semantic values" % DIMS)
    if len(gauge) < 2 * DIMS:
        raise ValueError("expected at least %d vertex gauges" % (2 * DIMS))
    oa: list[float] = []
    ab: list[float] = []
    ob: list[float] = []
    for index, value in enumerate(values):
        a, b = triangle_indices(index)
        oa.append(pack_payload(gauge[0] - gauge[a]))
        ab.append(pack_payload(int(value) + gauge[a] - gauge[b]))
        ob.append(pack_payload(gauge[0] - gauge[b]))
    return {"oa": oa, "ab": ab, "ob": ob}


def encode_integer_lanes(
    values: Sequence[int], gauge: Sequence[int]
) -> dict[str, list[int]]:
    """Conventional integer-lane control, with no NaN carriers at all."""
    if len(values) != DIMS:
        raise ValueError("expected %d semantic values" % DIMS)
    if len(gauge) < 2 * DIMS:
        raise ValueError("expected at least %d vertex gauges" % (2 * DIMS))
    oa: list[int] = []
    ab: list[int] = []
    ob: list[int] = []
    for index, value in enumerate(values):
        a, b = triangle_indices(index)
        oa.append(gauge[0] - gauge[a])
        ab.append(int(value) + gauge[a] - gauge[b])
        ob.append(gauge[0] - gauge[b])
    return {"oa": oa, "ab": ab, "ob": ob}


def _cycle(oa: Sequence[int], ab: Sequence[int], ob: Sequence[int]) -> list[int]:
    return [left + middle - right for left, middle, right in zip(oa, ab, ob)]


def _wrong_triangle(oa: Sequence[int], ab: Sequence[int], ob: Sequence[int]) -> list[int]:
    # Match Round 029: ob is rolled right by one, so the b vertex is mismatched.
    rolled_ob = list(ob[-1:]) + list(ob[:-1])
    return [left + middle - wrong for left, middle, wrong in zip(oa, ab, rolled_ob)]


def decode_payload_lanes(lanes: dict[str, list[float]], kind: str) -> list[int]:
    """Extract payload integers and apply a conventional graph readout."""
    oa = [unpack_payload(value) for value in lanes["oa"]]
    ab = [unpack_payload(value) for value in lanes["ab"]]
    ob = [unpack_payload(value) for value in lanes["ob"]]
    if kind == "raw":
        return ab
    if kind in ("cycle", "conventional"):
        return _cycle(oa, ab, ob)
    if kind == "wrong_triangle":
        return _wrong_triangle(oa, ab, ob)
    raise ValueError("unknown readout: %s" % kind)


def readout_payload_lanes(
    lanes: dict[str, list[float]], kind: str
) -> list[float]:
    """Compute a readout and re-pack each result into a quiet-NaN carrier.

    This is the lane-native representation of a readout.  The arithmetic is
    still exact integer arithmetic on extracted payloads; the result is simply
    stored back into the same 48-bit payload carrier.
    """
    return [pack_payload(value) for value in decode_payload_lanes(lanes, kind)]


def decode_integer_lanes(lanes: dict[str, list[int]], kind: str) -> list[int]:
    """Apply the same readouts to conventional integer lanes."""
    oa = list(lanes["oa"])
    ab = list(lanes["ab"])
    ob = list(lanes["ob"])
    if kind == "raw":
        return ab
    if kind in ("cycle", "conventional"):
        return _cycle(oa, ab, ob)
    if kind == "wrong_triangle":
        return _wrong_triangle(oa, ab, ob)
    raise ValueError("unknown readout: %s" % kind)


# ---------------------------------------------------------------------------
# Deterministic fixed-point nearest-centroid classifier
# ---------------------------------------------------------------------------


def _round_div(numerator: int, denominator: int) -> int:
    """Round an integer division to nearest, symmetrically for negatives."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def fit_centroids(
    rows: Sequence[Sequence[int]],
    labels: Sequence[int],
    classes: int = CLASS_COUNT,
) -> list[list[int]]:
    """Fit integer class centroids; this is the entire training procedure."""
    if len(rows) != len(labels):
        raise ValueError("rows and labels differ in length")
    sums = [[0] * DIMS for _ in range(classes)]
    counts = [0] * classes
    for row, label in zip(rows, labels):
        if len(row) != DIMS:
            raise ValueError("expected %d-dimensional rows" % DIMS)
        label = int(label)
        if not 0 <= label < classes:
            raise ValueError("label out of range")
        counts[label] += 1
        target = sums[label]
        for dimension, value in enumerate(row):
            target[dimension] += int(value)
    for label, count in enumerate(counts):
        if count == 0:
            raise ValueError("class %d has no training rows" % label)
    return [
        [_round_div(sums[label][dimension], counts[label]) for dimension in range(DIMS)]
        for label in range(classes)
    ]


def _squared_distance(row: Sequence[int], centroid: Sequence[int]) -> int:
    total = 0
    for value, center in zip(row, centroid):
        difference = int(value) - int(center)
        total += difference * difference
    return total


def _ce_like(distances: Sequence[int], label: int, temperature: float) -> float:
    """A stable cross-entropy-like score over negative squared distances."""
    best = min(distances)
    scaled = [-(distance - best) / temperature for distance in distances]
    log_normalizer = math.log(sum(math.exp(value) for value in scaled))
    return (distances[label] - best) / temperature + log_normalizer


def _accuracy_and_ce(
    centroids: Sequence[Sequence[int]],
    rows: Sequence[Sequence[int]],
    labels: Sequence[int],
    temperature: float,
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot evaluate an empty split")
    correct = 0
    ce_total = 0.0
    classes = len(centroids)
    for row, label in zip(rows, labels):
        distances = [_squared_distance(row, centroid) for centroid in centroids]
        prediction = min(range(classes), key=lambda item: (distances[item], item))
        correct += int(prediction == int(label))
        ce_total += _ce_like(distances, int(label), temperature)
    return {
        "accuracy": correct / len(rows),
        "ce_like": ce_total / len(rows),
    }


def evaluate_readout(
    train_rows: Sequence[Sequence[int]],
    train_labels: Sequence[int],
    val_rows: Sequence[Sequence[int]],
    val_labels: Sequence[int],
    test_rows: Sequence[Sequence[int]],
    test_labels: Sequence[int],
    classes: int = CLASS_COUNT,
) -> dict[str, Any]:
    """Train integer centroids and report held-out accuracy and CE-like loss."""
    centroids = fit_centroids(train_rows, train_labels, classes)
    within_class_total = sum(
        _squared_distance(row, centroids[int(label)])
        for row, label in zip(train_rows, train_labels)
    )
    temperature = max(1.0, (within_class_total / max(1, len(train_rows))) / 2.0)
    val = _accuracy_and_ce(centroids, val_rows, val_labels, temperature)
    test = _accuracy_and_ce(centroids, test_rows, test_labels, temperature)
    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "val_accuracy": val["accuracy"],
        "val_ce_like": val["ce_like"],
        "test_accuracy": test["accuracy"],
        "test_ce_like": test["ce_like"],
        "temperature": temperature,
        "parameters": classes * DIMS,
        "classifier": "nearest_centroid_integer_fixed_point",
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _decode_many(
    lane_rows: Sequence[dict[str, list[float]]], kind: str
) -> list[list[int]]:
    return [decode_payload_lanes(lanes, kind) for lanes in lane_rows]


def _max_abs_delta(first: Sequence[int], second: Sequence[int]) -> int:
    if len(first) != len(second):
        raise ValueError("readouts differ in length")
    return max((abs(a - b) for a, b in zip(first, second)), default=0)


def _mismatch_count(first: Sequence[int], second: Sequence[int]) -> int:
    return sum(1 for a, b in zip(first, second) if a != b)


def _packed_equal(first: int, second: int) -> bool:
    return payload_bits(pack_payload(first)) == payload_bits(pack_payload(second))


def _all_packed_equal(first: Sequence[int], second: Sequence[int]) -> bool:
    return all(_packed_equal(a, b) for a, b in zip(first, second))


def _gauges_for(count: int, magnitude: int, seed: int, salt: int) -> list[list[int]]:
    rng = random.Random(seed * 1000003 + salt)
    return [make_gauge(rng, magnitude) for _ in range(count)]


def run(
    seed: int = DEFAULT_SEED,
    samples_per_class: int = DEFAULT_SAMPLES,
    *,
    wordnet_path: Path | str | None = None,
    train_gauge: int = DEFAULT_TRAIN_GAUGE,
    test_gauge: int = DEFAULT_TEST_GAUGE,
) -> dict[str, Any]:
    """Run the deterministic Round 029 reproduction and return its report."""
    if train_gauge < 0 or test_gauge < 0:
        raise ValueError("gauge magnitudes must be non-negative")

    if wordnet_path is None:
        dataset = make_synthetic_dataset(seed, samples_per_class)
    else:
        dataset = load_wordnet_dataset(wordnet_path)

    features = dataset["features"]
    labels = dataset["labels"]
    split = stratified_split(labels, seed)
    zero_gauge = [0] * VERTEX_COUNT

    train_gauges = _gauges_for(len(split["train"]), train_gauge, seed, 101)
    val_gauges = _gauges_for(len(split["val"]), train_gauge, seed, 202)
    test_gauges = _gauges_for(len(split["test"]), test_gauge, seed, 303)

    train_z = [features[index] for index in split["train"]]
    val_z = [features[index] for index in split["val"]]
    test_z = [features[index] for index in split["test"]]
    train_y = [labels[index] for index in split["train"]]
    val_y = [labels[index] for index in split["val"]]
    test_y = [labels[index] for index in split["test"]]

    train_lanes = [
        encode_payload_lanes(values, gauge)
        for values, gauge in zip(train_z, train_gauges)
    ]
    val_lanes = [
        encode_payload_lanes(values, gauge)
        for values, gauge in zip(val_z, val_gauges)
    ]
    test_lanes = [
        encode_payload_lanes(values, gauge)
        for values, gauge in zip(test_z, test_gauges)
    ]

    results: dict[str, Any] = {}
    for kind in ("cycle", "raw", "wrong_triangle"):
        results[kind] = evaluate_readout(
            _decode_many(train_lanes, kind),
            train_y,
            _decode_many(val_lanes, kind),
            val_y,
            _decode_many(test_lanes, kind),
            test_y,
        )

    # A genuinely conventional implementation: the same lanes and readouts,
    # held as ordinary Python integers with no NaN carriers at all.
    integer_train_lanes = [
        encode_integer_lanes(values, gauge)
        for values, gauge in zip(train_z, train_gauges)
    ]
    integer_val_lanes = [
        encode_integer_lanes(values, gauge)
        for values, gauge in zip(val_z, val_gauges)
    ]
    integer_test_lanes = [
        encode_integer_lanes(values, gauge)
        for values, gauge in zip(test_z, test_gauges)
    ]
    results["conventional"] = evaluate_readout(
        [decode_integer_lanes(lanes, "conventional") for lanes in integer_train_lanes],
        train_y,
        [decode_integer_lanes(lanes, "conventional") for lanes in integer_val_lanes],
        val_y,
        [decode_integer_lanes(lanes, "conventional") for lanes in integer_test_lanes],
        test_y,
    )

    # Zero-gauge controls: raw and the mismatched triangle are not intrinsically
    # broken; the large gauge nuisance is what destroys them.
    zero_train_lanes = [
        encode_payload_lanes(values, zero_gauge) for values in train_z
    ]
    zero_val_lanes = [
        encode_payload_lanes(values, zero_gauge) for values in val_z
    ]
    zero_test_lanes = [
        encode_payload_lanes(values, zero_gauge) for values in test_z
    ]
    for kind in ("raw", "wrong_triangle"):
        results[kind + "_zero_gauge"] = evaluate_readout(
            _decode_many(zero_train_lanes, kind),
            train_y,
            _decode_many(zero_val_lanes, kind),
            val_y,
            _decode_many(zero_test_lanes, kind),
            test_y,
        )

    # Exact cancellation under two independent, fresh large gauges.
    alt_magnitude = max(test_gauge * 2 + 12345, test_gauge + 1)
    alt_gauges = _gauges_for(len(test_z), alt_magnitude, seed, 404)
    alt_lanes = [
        encode_payload_lanes(values, gauge)
        for values, gauge in zip(test_z, alt_gauges)
    ]

    cycle_first = _decode_many(test_lanes, "cycle")
    cycle_second = _decode_many(alt_lanes, "cycle")
    cycle_carriers = [
        readout_payload_lanes(lanes, "cycle") for lanes in test_lanes
    ]
    raw_first = _decode_many(test_lanes, "raw")
    raw_second = _decode_many(alt_lanes, "raw")
    wrong_first = _decode_many(test_lanes, "wrong_triangle")
    wrong_second = _decode_many(alt_lanes, "wrong_triangle")
    zero_raw = _decode_many(zero_test_lanes, "raw")
    zero_wrong = _decode_many(zero_test_lanes, "wrong_triangle")

    cycle_equal_semantics = all(
        _all_packed_equal(readout, semantics)
        for readout, semantics in zip(cycle_first, test_z)
    )
    cycle_gauge_invariant = all(
        _all_packed_equal(first, second)
        for first, second in zip(cycle_first, cycle_second)
    )
    raw_gauge_invariant = all(
        _all_packed_equal(first, second)
        for first, second in zip(raw_first, raw_second)
    )
    wrong_gauge_invariant = all(
        _all_packed_equal(first, second)
        for first, second in zip(wrong_first, wrong_second)
    )
    raw_zero_equal_semantics = all(
        _all_packed_equal(readout, semantics)
        for readout, semantics in zip(zero_raw, test_z)
    )
    wrong_zero_equal_semantics = all(
        _all_packed_equal(readout, semantics)
        for readout, semantics in zip(zero_wrong, test_z)
    )

    cycle_carrier_bits_equal_semantics = all(
        payload_bits(carrier) == payload_bits(pack_payload(semantic))
        for carrier_row, semantic_row in zip(cycle_carriers, test_z)
        for carrier, semantic in zip(carrier_row, semantic_row)
    )
    cycle_carriers_are_nan = all(
        math.isnan(carrier) for row in cycle_carriers for carrier in row
    )

    exactness = {
        "test_samples": len(test_z),
        "cycle_payload_bitwise_equals_semantics": cycle_equal_semantics,
        "cycle_readout_carriers_are_nan": cycle_carriers_are_nan,
        "cycle_readout_carrier_bits_equal_semantics": (
            cycle_carrier_bits_equal_semantics
        ),
        "cycle_bitwise_gauge_invariant": cycle_gauge_invariant,
        "max_cycle_abs_error": max(
            (_max_abs_delta(readout, semantics)
             for readout, semantics in zip(cycle_first, test_z)),
            default=0,
        ),
        "cycle_mismatch_count": sum(
            _mismatch_count(readout, semantics)
            for readout, semantics in zip(cycle_first, test_z)
        ),
        "raw_bitwise_gauge_invariant": raw_gauge_invariant,
        "raw_mismatch_count": sum(
            _mismatch_count(first, second)
            for first, second in zip(raw_first, raw_second)
        ),
        "raw_max_gauge_delta": max(
            (_max_abs_delta(first, second) for first, second in zip(raw_first, raw_second)),
            default=0,
        ),
        "wrong_triangle_bitwise_gauge_invariant": wrong_gauge_invariant,
        "wrong_triangle_mismatch_count": sum(
            _mismatch_count(first, second)
            for first, second in zip(wrong_first, wrong_second)
        ),
        "wrong_triangle_max_gauge_delta": max(
            (_max_abs_delta(first, second) for first, second in zip(wrong_first, wrong_second)),
            default=0,
        ),
        "raw_zero_gauge_bitwise_equals_semantics": raw_zero_equal_semantics,
        "wrong_triangle_zero_gauge_bitwise_equals_semantics": wrong_zero_equal_semantics,
        "alt_gauge_magnitude": alt_magnitude,
    }

    # Conventional integer implementation control.  It must tie every payload
    # readout bit-for-bit on every split.
    conventional_ties: dict[str, bool] = {}
    for kind in READOUTS:
        split_ties = []
        for payload_split, integer_split in (
            (train_lanes, integer_train_lanes),
            (val_lanes, integer_val_lanes),
            (test_lanes, integer_test_lanes),
        ):
            payload_rows = _decode_many(payload_split, kind)
            integer_rows = [
                decode_integer_lanes(lanes, kind) for lanes in integer_split
            ]
            split_ties.append(
                all(
                    _all_packed_equal(payload, integer)
                    for payload, integer in zip(payload_rows, integer_rows)
                )
            )
        conventional_ties[kind] = all(split_ties)
    conventional_ties["splits_checked"] = ["train", "val", "test"]
    conventional_ties["conventional_graph_implementation_ties"] = all(
        conventional_ties[kind] for kind in READOUTS
    )

    target = {
        "cycle_test_accuracy_min": 0.90,
        "raw_test_accuracy_max": 0.20,
        "wrong_triangle_test_accuracy_max": 0.20,
        "raw_zero_gauge_accuracy_min": 0.90,
        "wrong_triangle_zero_gauge_accuracy_min": 0.90,
        "cycle_max_abs_error": 0,
        "raw_max_gauge_delta_min": 1,
        "wrong_triangle_max_gauge_delta_min": 1,
    }
    measured = {
        "cycle_test_accuracy": results["cycle"]["test_accuracy"],
        "raw_test_accuracy": results["raw"]["test_accuracy"],
        "wrong_triangle_test_accuracy": results["wrong_triangle"]["test_accuracy"],
        "conventional_test_accuracy": results["conventional"]["test_accuracy"],
        "raw_zero_gauge_accuracy": results["raw_zero_gauge"]["test_accuracy"],
        "wrong_triangle_zero_gauge_accuracy": results[
            "wrong_triangle_zero_gauge"
        ]["test_accuracy"],
        "cycle_max_abs_error": exactness["max_cycle_abs_error"],
        "raw_max_gauge_delta": exactness["raw_max_gauge_delta"],
        "wrong_triangle_max_gauge_delta": exactness[
            "wrong_triangle_max_gauge_delta"
        ],
        "cycle_test_ce_like": results["cycle"]["test_ce_like"],
        "raw_test_ce_like": results["raw"]["test_ce_like"],
        "wrong_triangle_test_ce_like": results["wrong_triangle"]["test_ce_like"],
    }

    checks = {
        "cycle_exact_semantic_recovery": (
            cycle_equal_semantics and exactness["max_cycle_abs_error"] == 0
        ),
        "cycle_readout_carriers_are_nan": cycle_carriers_are_nan,
        "cycle_readout_carrier_bits_equal_semantics": (
            cycle_carrier_bits_equal_semantics
        ),
        "cycle_gauge_invariant": cycle_gauge_invariant,
        "cycle_high_accuracy": measured["cycle_test_accuracy"]
        >= target["cycle_test_accuracy_min"],
        "raw_fails_under_large_gauge": measured["raw_test_accuracy"]
        <= target["raw_test_accuracy_max"],
        "wrong_triangle_fails_under_large_gauge": measured[
            "wrong_triangle_test_accuracy"
        ]
        <= target["wrong_triangle_test_accuracy_max"],
        "raw_works_without_gauge": measured["raw_zero_gauge_accuracy"]
        >= target["raw_zero_gauge_accuracy_min"],
        "wrong_triangle_works_without_gauge": measured[
            "wrong_triangle_zero_gauge_accuracy"
        ]
        >= target["wrong_triangle_zero_gauge_accuracy_min"],
        "raw_changes_under_gauge": exactness["raw_max_gauge_delta"]
        >= target["raw_max_gauge_delta_min"],
        "wrong_triangle_changes_under_gauge": exactness[
            "wrong_triangle_max_gauge_delta"
        ]
        >= target["wrong_triangle_max_gauge_delta_min"],
        "conventional_ties_all_readouts": conventional_ties[
            "conventional_graph_implementation_ties"
        ],
    }
    checks["all_targets_met"] = all(checks.values())

    dataset_report = {
        "source": dataset["source"],
        "wordnet_used": bool(dataset.get("wordnet_used", False)),
        "claims_wordnet_result": bool(dataset.get("claims_wordnet_result", False)),
        "wordnet_note": dataset.get(
            "wordnet_note",
            "No WordNet JSONL was supplied; all samples are deterministic "
            "synthetic fixed-point vectors. No WordNet result is claimed.",
        ),
        "dims": dataset["dims"],
        "classes": dataset["classes"],
        "fixed_point_scale": dataset["fixed_point_scale"],
        "class_names": dataset["class_names"],
        "total_samples": len(features),
        "split": {name: len(indices) for name, indices in split.items()},
    }
    if "wordnet_path" in dataset:
        dataset_report["wordnet_path"] = dataset["wordnet_path"]
        dataset_report["record_ids"] = dataset["record_ids"]
        dataset_report["record_count"] = dataset["record_count"]
    if "samples_per_class" in dataset:
        dataset_report["samples_per_class"] = dataset["samples_per_class"]

    return {
        "schema": SCHEMA,
        "seed": seed,
        "samples_per_class": samples_per_class,
        "dataset": dataset_report,
        "encoding": {
            "carrier": "quiet_NaN_payload",
            "nan_base_bits": "0x7ff8000000000000",
            "payload_bits": PAYLOAD_BITS,
            "payload_signed_range": [-PAYLOAD_SIGN, PAYLOAD_SIGN - 1],
            "lanes": ["(0,a)", "(a,b)", "(0,b)"],
            "triangles": DIMS,
            "vertices": VERTEX_COUNT,
            "train_gauge_magnitude": train_gauge,
            "test_gauge_magnitude": test_gauge,
            "arithmetic": (
                "exact integer arithmetic after extracting the 48-bit payload; "
                "IEEE-754 NaN arithmetic is not used"
            ),
        },
        "results": results,
        "exactness": exactness,
        "conventional_ties": conventional_ties,
        "measured": measured,
        "target": target,
        "checks": checks,
        "conventional_graph_implementation_ties": conventional_ties[
            "conventional_graph_implementation_ties"
        ],
        "nan_unique_advantage": False,
        "no_nan_unique_advantage": True,
        "boundaries": {
            "conventional_graph_implementation_ties_nan_lanes": True,
            "no_nan_unique_advantage": True,
            "nan_unique_advantage": False,
            "ieee_nan_arithmetic_used": False,
            "payloads_are_storage_not_compute": True,
            "wordnet_result_claimed": False,
            "quantum_claimed": False,
            "note": (
                "The cycle identity is ordinary exact integer algebra. A "
                "conventional graph implementation using the same lanes produces "
                "identical bits. NaN payloads are a storage/addressing carrier "
                "only; IEEE-754 NaN arithmetic is not used and gives no "
                "NaN-unique advantage. The default dataset is synthetic, and no "
                "WordNet benchmark result is claimed."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Output rendering and CLI
# ---------------------------------------------------------------------------


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a deterministic markdown summary of ``report``."""
    dataset = report["dataset"]
    exactness = report["exactness"]
    measured = report["measured"]
    target = report["target"]
    results = report["results"]

    wordnet_line = (
        "optional local JSONL (`%s`)" % dataset.get("wordnet_path")
        if dataset["wordnet_used"]
        else "deterministic synthetic fixed-point data"
    )
    lines = [
        "# Round 029 reproduction: exact gauge cancellation",
        "",
        "Dependency-light reproduction of the Round 029 geometric cycle claim. "
        "Semantic values are fixed-point integers stored in quiet-NaN payload "
        "lanes; readout arithmetic is exact integer arithmetic on the extracted "
        "payloads, not IEEE-754 NaN arithmetic.",
        "",
        "## Dataset",
        "",
        "- Source: %s" % wordnet_line,
        "- WordNet used: %s" % _format_metric(dataset["wordnet_used"]),
        "- WordNet result claimed: %s" % _format_metric(dataset["claims_wordnet_result"]),
        "- Shape: %d classes x %d dimensions, fixed-point scale %d"
        % (dataset["classes"], dataset["dims"], dataset["fixed_point_scale"]),
        "- Split: train %d / val %d / test %d"
        % (
            dataset["split"]["train"],
            dataset["split"]["val"],
            dataset["split"]["test"],
        ),
        "- Gauge magnitudes: train +/-%d, test +/-%d"
        % (
            report["encoding"]["train_gauge_magnitude"],
            report["encoding"]["test_gauge_magnitude"],
        ),
        "",
        "## Held-out readouts",
        "",
        "| readout | val accuracy | test accuracy | test CE-like |",
        "|:--|---:|---:|---:|",
    ]
    for name in READOUTS:
        row = results[name]
        lines.append(
            "| %s | %.4f | %.4f | %.6g |"
            % (
                name,
                row["val_accuracy"],
                row["test_accuracy"],
                row["test_ce_like"],
            )
        )
    lines.extend(
        [
            "| raw, zero gauge | - | %.4f | %.6g |"
            % (
                results["raw_zero_gauge"]["test_accuracy"],
                results["raw_zero_gauge"]["test_ce_like"],
            ),
            "| wrong triangle, zero gauge | - | %.4f | %.6g |"
            % (
                results["wrong_triangle_zero_gauge"]["test_accuracy"],
                results["wrong_triangle_zero_gauge"]["test_ce_like"],
            ),
            "",
            "## Exact cancellation",
            "",
            "| property | measured |",
            "|:--|:--|",
            "| cycle payload bits equal semantics | %s |"
            % _format_metric(exactness["cycle_payload_bitwise_equals_semantics"]),
            "| cycle readout carriers are quiet NaNs | %s |"
            % _format_metric(exactness["cycle_readout_carriers_are_nan"]),
            "| cycle readout carrier bits equal semantics | %s |"
            % _format_metric(
                exactness["cycle_readout_carrier_bits_equal_semantics"]
            ),
            "| cycle gauge-invariant across two fresh gauges | %s |"
            % _format_metric(exactness["cycle_bitwise_gauge_invariant"]),
            "| max cycle absolute error | %s |"
            % _format_metric(exactness["max_cycle_abs_error"]),
            "| raw gauge-invariant | %s |"
            % _format_metric(exactness["raw_bitwise_gauge_invariant"]),
            "| raw max gauge delta | %s |"
            % _format_metric(exactness["raw_max_gauge_delta"]),
            "| wrong triangle gauge-invariant | %s |"
            % _format_metric(exactness["wrong_triangle_bitwise_gauge_invariant"]),
            "| wrong triangle max gauge delta | %s |"
            % _format_metric(exactness["wrong_triangle_max_gauge_delta"]),
            "| raw equals semantics at zero gauge | %s |"
            % _format_metric(exactness["raw_zero_gauge_bitwise_equals_semantics"]),
            "| wrong triangle equals semantics at zero gauge | %s |"
            % _format_metric(
                exactness["wrong_triangle_zero_gauge_bitwise_equals_semantics"]
            ),
            "",
            "## Measured vs target",
            "",
            "| metric | measured | target |",
            "|:--|---:|:--|",
            "| cycle test accuracy | %.4f | >= %.2f |"
            % (measured["cycle_test_accuracy"], target["cycle_test_accuracy_min"]),
            "| raw test accuracy | %.4f | <= %.2f |"
            % (measured["raw_test_accuracy"], target["raw_test_accuracy_max"]),
            "| wrong-triangle test accuracy | %.4f | <= %.2f |"
            % (
                measured["wrong_triangle_test_accuracy"],
                target["wrong_triangle_test_accuracy_max"],
            ),
            "| raw zero-gauge accuracy | %.4f | >= %.2f |"
            % (
                measured["raw_zero_gauge_accuracy"],
                target["raw_zero_gauge_accuracy_min"],
            ),
            "| wrong-triangle zero-gauge accuracy | %.4f | >= %.2f |"
            % (
                measured["wrong_triangle_zero_gauge_accuracy"],
                target["wrong_triangle_zero_gauge_accuracy_min"],
            ),
            "| cycle max absolute error | %d | exactly 0 |"
            % measured["cycle_max_abs_error"],
            "",
            "All target checks: **%s**." % _format_metric(report["checks"]["all_targets_met"]),
            "",
            "## Boundaries",
            "",
            "- The cycle identity is ordinary exact integer algebra. A conventional "
            "integer graph implementation ties the NaN-carrier implementation "
            "bit-for-bit for every readout "
            "(`conventional_graph_implementation_ties_nan_lanes: %s`)."
            % _format_metric(
                report["boundaries"][
                    "conventional_graph_implementation_ties_nan_lanes"
                ]
            ),
            "- NaN payloads are used only as a storage/addressing carrier. "
            "IEEE-754 NaN arithmetic is not used, and there is **no NaN-unique "
            "advantage** (`nan_unique_advantage: %s`)."
            % _format_metric(report["boundaries"]["nan_unique_advantage"]),
            "- The default dataset is synthetic and controlled. No WordNet "
            "benchmark result is claimed (`wordnet_result_claimed: %s`)."
            % _format_metric(report["boundaries"]["wordnet_result_claimed"]),
            "- No quantum or hardware claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dependency-light exact reproduction of the Round 029 gauge-cycle "
            "claim using NaN payload lanes."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="synthetic samples per class (16 classes; default 40)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the JSON report to this path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="write the markdown summary to this path",
    )
    parser.add_argument(
        "--wordnet",
        type=Path,
        default=None,
        help=(
            "optional local JSONL with 48-dimensional vectors and exactly 16 "
            "class labels; no WordNet benchmark claim is made"
        ),
    )
    parser.add_argument(
        "--train-gauge",
        type=int,
        default=DEFAULT_TRAIN_GAUGE,
        help="vertex gauge magnitude for train/val lanes",
    )
    parser.add_argument(
        "--test-gauge",
        type=int,
        default=DEFAULT_TEST_GAUGE,
        help="large vertex gauge magnitude for held-out test lanes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        seed=args.seed,
        samples_per_class=args.samples,
        wordnet_path=args.wordnet,
        train_gauge=args.train_gauge,
        test_gauge=args.test_gauge,
    )
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report_text, encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if args.json is None and args.markdown is None:
        print(report_text, end="")
    else:
        print(
            json.dumps(
                {
                    "json": str(args.json) if args.json is not None else None,
                    "markdown": (
                        str(args.markdown) if args.markdown is not None else None
                    ),
                    "checks": report["checks"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if report["checks"]["all_targets_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
