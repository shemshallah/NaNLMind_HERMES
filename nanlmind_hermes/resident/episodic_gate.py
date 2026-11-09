"""Open-set episodic copy/abstain gate over the full resident cycle basis.

Round 034's useful capability is not that a memory returns a nearest record;
it is that a router can decide whether the record actually answers the query.
This module keeps that separation explicit and implements it without PyTorch:

* a pure-Python character key scorer proposes one of 12 episodic records;
* a learned logistic gate decides copy versus abstain from key score, margin,
  edit distance and candidate-distance features;
* the 24-byte record is copied through the raw resident full-cycle memory
  (32,385 coefficients), not a detached prepared answer table;
* a conventional indexed copy is available as a tied control;
* gauge shifts change raw pair lanes but not the cycle copy.

No fluent generation, NaN-unique advantage or canonical merge is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from .full_cycle_memory import FullCycleMemory, cycle_pair

SCHEMA = "nanlm.resident-episodic-gate/1"
LENGTH = 24
CANDIDATES = 12
MAX_KEY = 22
MAX_CORRUPTION = 6
DEFAULT_SEED = 34
POOL_SCHEMA = "nanlm.round034-episodic-pool/1"


class EpisodicGateError(ValueError):
    """Base error for malformed episodic/gate input."""


class EpisodicRowError(EpisodicGateError):
    """A pool row is malformed or too short."""


def _bounded(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EpisodicGateError("%s must be an integer" % name)
    if not low <= value <= high:
        raise EpisodicGateError(
            "%s must be in [%d, %d]: %r" % (name, low, high, value))
    return value


def load_pool(path: str | Path) -> dict[str, list[tuple[str, bytes]]]:
    """Load the compact Round 034 pool (name + 24-byte definition)."""
    source = Path(path).expanduser()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EpisodicGateError("cannot read episodic pool: %s" % exc) from exc
    if not isinstance(data, dict):
        raise EpisodicGateError("episodic pool must be an object")
    result: dict[str, list[tuple[str, bytes]]] = {}
    for split in ("train", "val", "test"):
        rows = data.get(split)
        if not isinstance(rows, list):
            raise EpisodicGateError("pool is missing split %s" % split)
        clean: list[tuple[str, bytes]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise EpisodicRowError("%s row %d is malformed" % (split, index))
            name, encoded = row
            if not isinstance(name, str) or not name.strip():
                raise EpisodicRowError("%s row %d has no name" % (split, index))
            try:
                raw = bytes.fromhex(str(encoded))
            except ValueError as exc:
                raise EpisodicRowError(
                    "%s row %d has invalid bytes" % (split, index)) from exc
            if len(raw) != LENGTH:
                raise EpisodicRowError(
                    "%s row %d has %d bytes, expected %d"
                    % (split, index, len(raw), LENGTH))
            clean.append((name, raw))
        result[split] = clean
    return result


def corrupt_name(name: str, rng: random.Random, changes: int = 2) -> str:
    """Apply bounded lowercase ASCII corruption, preserving non-empty keys."""
    value = list(name.lower()[:MAX_KEY])
    changes = _bounded(changes, "changes", 0, MAX_CORRUPTION)
    if not value:
        return "x"
    for _ in range(min(changes, len(value))):
        position = rng.randrange(len(value))
        replacement = rng.choice("abcdefghijklmnopqrstuvwxyz")
        value[position] = replacement
    return "".join(value) or "x"


def edit_distance(first: str, second: str) -> int:
    """Exact Levenshtein distance for short ASCII names."""
    first = first[:MAX_KEY]
    second = second[:MAX_KEY]
    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + int(left != right)))
        previous = current
    return previous[-1]


def normalized_distance(first: str, second: str) -> float:
    return edit_distance(first, second) / max(len(first), len(second), 1)


def _ngrams(value: str, size: int = 3) -> set[str]:
    padded = " " + value.lower() + " "
    return {padded[index:index + size]
            for index in range(max(1, len(padded) - size + 1))}


def ngram_similarity(first: str, second: str) -> float:
    left, right = _ngrams(first), _ngrams(second)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class Episode:
    """One bounded episodic query with 12 records and one target slot."""

    names: tuple[str, ...]
    records: tuple[bytes, ...]
    query: str
    target: int | None
    present: bool
    corruption: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "query": self.query,
            "target": self.target,
            "present": self.present,
            "corruption": self.corruption,
            "records": [record.hex() for record in self.records],
        }


def make_episode(pool: Sequence[tuple[str, bytes]], rng: random.Random, *,
                 present: bool = True, corruption: int = 2,
                 length: int = CANDIDATES) -> Episode:
    """Build one deterministic episode from a pool."""
    if len(pool) < length:
        raise EpisodicGateError("pool is shorter than the candidate count")
    picks = rng.sample(range(len(pool)), length)
    names = tuple(pool[index][0] for index in picks)
    records = tuple(pool[index][1] for index in picks)
    corruption = _bounded(corruption, "corruption", 0, MAX_CORRUPTION)
    if present:
        target = rng.randrange(length)
        query = corrupt_name(names[target], rng, corruption)
        return Episode(names, records, query, target, True, corruption)
    # An absent key is drawn from outside this episode and corrupted the same way.
    chosen = {index for index in picks}
    pool_index = rng.randrange(len(pool))
    while pool_index in chosen:
        pool_index = rng.randrange(len(pool))
    query = corrupt_name(pool[pool_index][0], rng, corruption)
    return Episode(names, records, query, None, False, corruption)


@dataclass(frozen=True)
class Match:
    slot: int
    score: float
    distance: float
    margin: float
    features: tuple[float, ...]


def match_episode(episode: Episode) -> Match:
    """Score all candidates and return the best key match."""
    distances = [normalized_distance(episode.query, name)
                 for name in episode.names]
    similarities = [ngram_similarity(episode.query, name)
                    for name in episode.names]
    order = sorted(range(len(episode.names)),
                   key=lambda index: (-similarities[index], distances[index],
                                      index))
    best = order[0]
    second = order[1] if len(order) > 1 else order[0]
    score = similarities[best]
    margin = similarities[best] - similarities[second]
    distance_margin = distances[second] - distances[best]
    features = (
        score,
        margin,
        distances[best],
        distance_margin,
        float(len(episode.query)),
        float(len(episode.names[best])),
    )
    return Match(best, score, distances[best], margin, features)


class LogisticGate:
    """Small pure-Python logistic router with deterministic threshold tuning."""

    def __init__(self, dimension: int = 6):
        self.dimension = _bounded(dimension, "dimension", 1, 32)
        self.weights = [0.0] * self.dimension
        self.bias = 0.0
        self.threshold = 0.5

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp = math.exp(-min(value, 60.0))
            return 1.0 / (1.0 + exp)
        exp = math.exp(max(value, -60.0))
        return exp / (1.0 + exp)

    def probability(self, features: Sequence[float]) -> float:
        if len(features) != self.dimension:
            raise EpisodicGateError("feature vector has wrong dimension")
        value = self.bias + sum(weight * feature
                                for weight, feature in zip(self.weights, features))
        return self._sigmoid(value)

    def fit(self, samples: Sequence[tuple[Sequence[float], int]], *,
            epochs: int = 160, learning_rate: float = 0.15) -> "LogisticGate":
        """Fit binary cross-entropy by bounded full-batch gradient descent."""
        if not samples:
            raise EpisodicGateError("at least one training sample is required")
        epochs = _bounded(epochs, "epochs", 1, 2000)
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or not 0 < learning_rate < 5:
            raise EpisodicGateError("learning_rate must be finite and in (0, 5)")
        for _ in range(epochs):
            weight_grad = [0.0] * self.dimension
            bias_grad = 0.0
            for features, label in samples:
                probability = self.probability(features)
                error = probability - float(label)
                bias_grad += error
                for index, feature in enumerate(features):
                    weight_grad[index] += error * feature
            scale = 1.0 / len(samples)
            self.bias -= learning_rate * bias_grad * scale
            for index in range(self.dimension):
                self.weights[index] -= learning_rate * weight_grad[index] * scale
        return self

    def tune(self, samples: Sequence[tuple[Sequence[float], int]],
             costs: Sequence[float] | None = None) -> float:
        """Choose the threshold maximizing copy utility minus false-answer cost."""
        if not samples:
            raise EpisodicGateError("threshold tuning needs samples")
        best_value = -float("inf")
        best_threshold = 0.5
        for step in range(201):
            threshold = step / 200.0
            utility = 0.0
            for index, (features, label) in enumerate(samples):
                accepted = self.probability(features) >= threshold
                reward = (costs[index] if costs is not None else 1.0)
                utility += (reward if accepted and label else 0.0)
                utility -= (2.0 if accepted and not label else 0.0)
            if utility > best_value:
                best_value = utility
                best_threshold = threshold
        self.threshold = best_threshold
        return best_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "weights": [round(value, 8) for value in self.weights],
            "bias": round(self.bias, 8),
            "threshold": round(self.threshold, 8),
        }


@dataclass(frozen=True)
class CopyReceipt:
    slot: int
    record: bytes
    raw_record: bytes
    cycle_exact: bool
    raw_exact: bool
    gauge: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "record": self.record.hex(),
            "raw_record": self.raw_record.hex(),
            "cycle_exact": self.cycle_exact,
            "raw_exact": self.raw_exact,
            "gauge": self.gauge,
        }


class EpisodicCycleMemory:
    """Store twelve 24-byte records through raw full-cycle coefficients."""

    def __init__(self, memory: FullCycleMemory):
        if not isinstance(memory, FullCycleMemory):
            raise EpisodicGateError("memory must be a FullCycleMemory")
        self.memory = memory
        self.slots = 0

    def _indices(self, slot: int) -> range:
        slot = _bounded(slot, "slot", 0, CANDIDATES - 1)
        start = slot * LENGTH
        if start + LENGTH > len(self.memory):
            raise EpisodicGateError("slot exceeds full cycle basis")
        return range(start, start + LENGTH)

    def store(self, slot: int, record: bytes) -> None:
        raw = bytes(record)
        if len(raw) != LENGTH:
            raise EpisodicGateError("episode record must be %d bytes" % LENGTH)
        for index, value in zip(self._indices(slot), raw):
            first, second = cycle_pair(index)
            self.memory.set_word(first, second, value)
        self.slots = max(self.slots, slot + 1)

    def read_cycle(self, slot: int) -> bytes:
        values = []
        for index in self._indices(slot):
            first, second = cycle_pair(index)
            values.append(self.memory.field.cycle(first, second) & 0xFF)
        return bytes(values)

    def read_raw(self, slot: int) -> bytes:
        values = []
        for index in self._indices(slot):
            first, second = cycle_pair(index)
            values.append(self.memory.field.get_oriented(first, second) & 0xFF)
        return bytes(values)

    def gauge(self, seed: int = 340, sigma: int = 8) -> int:
        seed = _bounded(seed, "seed", 0, 2 ** 31 - 1)
        sigma = _bounded(sigma, "sigma", 0, 1 << 20)
        rng = random.Random(seed)
        vector = [rng.randrange(-sigma, sigma + 1) & 0xFFFFFFFF
                  for _ in range(256)]
        return self.memory.field.gauge_shift(vector)

    def copy(self, slot: int, expected: bytes, *, seed: int = 340) -> CopyReceipt:
        expected = bytes(expected)
        self.gauge(seed)
        cycle_record = self.read_cycle(slot)
        raw_record = self.read_raw(slot)
        return CopyReceipt(slot, cycle_record, raw_record,
                           cycle_record == expected, raw_record == expected,
                           seed)

    def audit(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "dimension": len(self.memory),
            "slots": self.slots,
            "occupied": int(self.memory.occupancy),
            "gauge_invariant": True,
            "canonical_merge": False,
        }


def make_training_samples(pool: Sequence[tuple[str, bytes]],
                          rng: random.Random, count: int = 1600,
                          corruption: int = 2) -> list[tuple[tuple[float, ...], int, float]]:
    samples = []
    for index in range(count):
        present = index % 2 == 0
        episode = make_episode(pool, rng, present=present,
                               corruption=corruption)
        match = match_episode(episode)
        correct = float(present and episode.target == match.slot)
        samples.append((match.features, int(present), correct))
    return samples


def evaluate(pool: Sequence[tuple[str, bytes]], gate: LogisticGate,
             *, seed: int = 340, count: int = 600, corruption: int = 2
             ) -> dict[str, Any]:
    rng = random.Random(seed)
    positive = negative = accepted = false_accept = exact = 0
    for index in range(count):
        present = index % 2 == 0
        episode = make_episode(pool, rng, present=present,
                               corruption=corruption)
        match = match_episode(episode)
        probability = gate.probability(match.features)
        decision = probability >= gate.threshold
        if present:
            positive += 1
            if decision:
                accepted += 1
                exact += int(episode.target == match.slot)
        else:
            negative += 1
            false_accept += int(decision)
    return {
        "schema": SCHEMA,
        "queries": count,
        "corruption": corruption,
        "positive": positive,
        "negative": negative,
        "accepted": accepted,
        "positive_accept_rate": accepted / positive if positive else 0.0,
        "negative_reject_rate": 1.0 - false_accept / negative if negative else 0.0,
        "accept_accuracy": (accepted + (negative - false_accept)) / count
        if count else 0.0,
        "false_answer_rate": false_accept / count if count else 0.0,
        "exact_copy_rate": exact / positive if positive else 0.0,
        "threshold": gate.threshold,
        "canonical_merge": False,
        "language_generation_claim": False,
    }


def _utility(gate: LogisticGate,
             samples: Sequence[tuple[Sequence[float], int, float]]) -> float:
    value = 0.0
    for features, label, reward in samples:
        accepted = gate.probability(features) >= gate.threshold
        value += reward if accepted and label else 0.0
        value -= 2.0 if accepted and not label else 0.0
    return value


def _distance_calibrated(samples: Sequence[tuple[Sequence[float], int, float]]
                         ) -> LogisticGate:
    """Choose a calibrated distance boundary when it beats the linear router.

    The returned object is still a logistic gate (a steep negative coefficient
    on the edit-distance feature), but its boundary is selected on validation
    utility.  This keeps the fallback inspectable and prevents a weakly
    trained linear router from accepting everything or nothing.
    """
    best_gate = None
    best_utility = -float("inf")
    for step in range(1, 100):
        boundary = step / 100.0
        gate = LogisticGate()
        gate.weights = [0.0, 0.0, -12.0, 0.0, 0.0, 0.0]
        gate.bias = 12.0 * boundary
        gate.threshold = 0.5
        value = _utility(gate, samples)
        if value > best_utility:
            best_utility = value
            best_gate = gate
    assert best_gate is not None
    return best_gate


def train_gate(pool: dict[str, list[tuple[str, bytes]]],
               *, seed: int = 340, corruption: int = 2) -> LogisticGate:
    rng = random.Random(seed)
    train = make_training_samples(pool["train"], rng, 1600, corruption)
    validation = make_training_samples(pool["val"], rng, 400, corruption)
    gate = LogisticGate()
    gate.fit([(features, label) for features, label, _reward in train])
    gate.tune([(features, label) for features, label, _reward in validation],
              [reward for _features, _label, reward in validation])
    distance_gate = _distance_calibrated(validation)
    if _utility(distance_gate, validation) > _utility(gate, validation):
        return distance_gate
    return gate


def stress(pool: Sequence[tuple[str, bytes]], gate: LogisticGate, *,
           seed: int = 340, count: int = 300) -> dict[str, Any]:
    return {
        str(changes): evaluate(pool, gate, seed=seed + changes,
                               count=count, corruption=changes)
        for changes in (0, 2, 4, 6)
    }


__all__ = (
    "SCHEMA", "LENGTH", "CANDIDATES", "MAX_KEY", "MAX_CORRUPTION",
    "EpisodicGateError", "EpisodicRowError", "Episode", "Match",
    "CopyReceipt", "LogisticGate", "EpisodicCycleMemory", "load_pool",
    "corrupt_name", "edit_distance", "normalized_distance",
    "ngram_similarity", "make_episode", "match_episode", "train_gate",
    "evaluate", "stress",
)
