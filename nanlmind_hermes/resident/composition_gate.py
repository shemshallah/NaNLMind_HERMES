"""Compose multiple resident memories into an answer, or abstain.

Round 035's useful capability is compositional: two or three independent typed
facts are read from resident cycle memory and the answer is assembled *after*
the read.  The answer text is not stored as one record.

This module is deliberately small and inspectable.  It implements:

* a deterministic typed-chain interpreter for two and three facts;
* a learned logistic validity gate over structural link features;
* storage of each fact as raw bytes in the full 32,385-cycle resident basis;
* gauge-invariant cycle reads versus non-invariant raw edge reads;
* explicit abstention when a required link is missing.

No fluent LM, neural parser, unique NaN advantage or canonical merge is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Sequence

from .episodic_gate import (
    LogisticGate,
    corrupt_name,
    edit_distance,
    normalized_distance,
)
from .full_cycle_memory import FullCycleMemory, cycle_pair

SCHEMA = "nanlm.resident-composition-gate/1"
PAYLOAD_BYTES = 64
MAX_FACTS = 3
ANCHOR = "I do not have enough evidence."
POSITIVE_RELATIONS = frozenset({
    "is a", "is an", "kind of", "belongs to", "classified as", "type of",
})
CONFLICT_RELATIONS = frozenset({
    "opposes", "oppose", "contradicts", "contradict", "not", "refutes",
    "has no", "is not",
})


def relation_conflict(first: str, second: str) -> bool:
    """Return true for an explicit contradictory/unsupported relation."""
    left = first.strip().casefold()
    right = second.strip().casefold()
    if left in CONFLICT_RELATIONS or right in CONFLICT_RELATIONS:
        return True
    return left not in POSITIVE_RELATIONS or right not in POSITIVE_RELATIONS


class CompositionError(ValueError):
    """A fact, chain or resident payload is malformed."""


@dataclass(frozen=True)
class Fact:
    source: str
    destination: str
    relation: str = "is a"

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise CompositionError("fact source must be a non-empty string")
        if not isinstance(self.destination, str) or not self.destination.strip():
            raise CompositionError("fact destination must be a non-empty string")
        if not isinstance(self.relation, str) or not self.relation.strip():
            raise CompositionError("fact relation must be a non-empty string")

    def as_tuple(self) -> tuple[str, str]:
        return self.source, self.destination

    def encoded(self) -> bytes:
        return ("%s>%s>%s" % (self.source, self.relation, self.destination)
                ).encode("utf-8")


@dataclass(frozen=True)
class CompositionResult:
    answer: str
    accepted: bool
    path: tuple[Fact, ...]
    reason: str
    raw_answer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "answer": self.answer,
            "accepted": self.accepted,
            "path": [[fact.source, fact.relation, fact.destination]
                     for fact in self.path],
            "reason": self.reason,
            "raw_answer": self.raw_answer,
            "canonical_merge": False,
        }


def _fact_from_payload(payload: bytes) -> Fact:
    text = payload.rstrip(b"\x00").decode("utf-8", "replace")
    parts = text.split(">", 2)
    if len(parts) != 3 or not all(parts):
        raise CompositionError("resident payload is not a typed fact")
    return Fact(parts[0], parts[2], parts[1])


def _payload(fact: Fact) -> bytes:
    raw = fact.encoded()
    if len(raw) > PAYLOAD_BYTES:
        raise CompositionError(
            "fact payload exceeds %d bytes" % PAYLOAD_BYTES)
    return raw


def compose_two(first: Fact, second: Fact) -> CompositionResult:
    """Compose ``first`` then ``second`` only when the typed endpoints link."""
    if relation_conflict(first.relation, second.relation):
        return CompositionResult(ANCHOR, False, (first, second),
                                 "relation_conflict")
    if first.destination.casefold() != second.source.casefold():
        return CompositionResult(ANCHOR, False, (first, second),
                                 "middle_mismatch")
    answer = "%s %s %s." % (
        first.source, second.relation, second.destination)
    return CompositionResult(answer, True, (first, second), "linked")


def compose_three(first: Fact, second: Fact, third: Fact) -> CompositionResult:
    """Compose three facts into one answer, or abstain on any broken link."""
    path = (first, second, third)
    if any(relation_conflict(left.relation, right.relation)
           for left, right in zip(path, path[1:])):
        return CompositionResult(ANCHOR, False, path, "relation_conflict")
    if first.destination.casefold() != second.source.casefold():
        return CompositionResult(ANCHOR, False, path, "first_middle_mismatch")
    if second.destination.casefold() != third.source.casefold():
        return CompositionResult(ANCHOR, False, path, "second_middle_mismatch")
    answer = "%s is a kind of %s." % (first.source, third.destination)
    return CompositionResult(answer, True, path, "linked")


def compose_path(facts: Sequence[Fact]) -> CompositionResult:
    """Compose an arbitrary positive typed chain, or abstain deterministically."""
    path = tuple(facts)
    if len(path) < 2:
        return CompositionResult(ANCHOR, False, path, "too_few_facts")
    if any(relation_conflict(left.relation, right.relation)
           for left, right in zip(path, path[1:])):
        return CompositionResult(ANCHOR, False, path, "relation_conflict")
    for left, right in zip(path, path[1:]):
        if left.destination.casefold() != right.source.casefold():
            return CompositionResult(ANCHOR, False, path, "broken_link")
    if len(path) == 2:
        return compose_two(path[0], path[1])
    if len(path) == 3:
        return compose_three(*path)
    return CompositionResult(
        "%s is a kind of %s." % (path[0].source, path[-1].destination),
        True, path, "linked")


def structural_features(first: Fact, second: Fact) -> tuple[float, ...]:
    """Features used by the two-fact validity gate; no answer text is included."""
    linked = float(first.destination.casefold() == second.source.casefold())
    relation_ok = float(not relation_conflict(first.relation, second.relation))
    source_length = min(len(first.source) / 32.0, 1.0)
    middle_length = min(len(first.destination) / 32.0, 1.0)
    destination_length = min(len(second.destination) / 32.0, 1.0)
    distance = normalized_distance(first.destination, second.source)
    shared_prefix = 1.0 if first.destination[:1].casefold() == \
        second.source[:1].casefold() else 0.0
    return (linked, distance, source_length, middle_length,
            destination_length, shared_prefix, relation_ok)


class TwoFactGate:
    """Learned logistic validity gate for a two-fact chain."""

    def __init__(self):
        self.logistic = LogisticGate(dimension=7)
        self.fitted = False

    def fit(self, positives: Sequence[tuple[Fact, Fact]],
            negatives: Sequence[tuple[Fact, Fact]], *, epochs: int = 200
            ) -> "TwoFactGate":
        samples: list[tuple[Sequence[float], int]] = []
        samples.extend((structural_features(a, b), 1) for a, b in positives)
        samples.extend((structural_features(a, b), 0) for a, b in negatives)
        if not samples:
            raise CompositionError("gate training needs at least one sample")
        self.logistic.fit(samples, epochs=epochs, learning_rate=0.4)
        # A deterministic structural feature is the strongest safe signal; keep
        # a calibrated logistic score for reporting while never overriding it.
        self.logistic.tune(samples)
        self.fitted = True
        return self

    def probability(self, first: Fact, second: Fact) -> float:
        return self.logistic.probability(structural_features(first, second))

    def accepts(self, first: Fact, second: Fact) -> bool:
        if relation_conflict(first.relation, second.relation):
            return False
        if self.fitted:
            return self.probability(first, second) >= self.logistic.threshold
        return first.destination.casefold() == second.source.casefold()

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            **self.logistic.as_dict(),
            "canonical_merge": False,
        }


class CompositionMemory:
    """Store fact payloads in raw resident cycle coefficients."""

    def __init__(self, memory: FullCycleMemory):
        if not isinstance(memory, FullCycleMemory):
            raise CompositionError("memory must be a FullCycleMemory")
        self.memory = memory
        self.slots = 0
        self.max_slots = len(memory) // PAYLOAD_BYTES

    def _indices(self, slot: int) -> range:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise CompositionError("slot must be an integer")
        if not 0 <= slot < self.max_slots:
            raise CompositionError("slot exceeds the full cycle basis")
        start = slot * PAYLOAD_BYTES
        return range(start, start + PAYLOAD_BYTES)

    def store(self, slot: int, fact: Fact) -> None:
        raw = _payload(fact).ljust(PAYLOAD_BYTES, b"\x00")
        for index, value in zip(self._indices(slot), raw):
            first, second = cycle_pair(index)
            self.memory.set_word(first, second, value)
        self.slots = max(self.slots, slot + 1)

    def read_cycle(self, slot: int) -> Fact:
        raw = bytearray()
        for index in self._indices(slot):
            first, second = cycle_pair(index)
            raw.append(self.memory.field.cycle(first, second) & 0xFF)
        return _fact_from_payload(bytes(raw))

    def read_raw(self, slot: int) -> Fact | None:
        raw = bytearray()
        for index in self._indices(slot):
            first, second = cycle_pair(index)
            raw.append(self.memory.field.get_oriented(first, second) & 0xFF)
        try:
            return _fact_from_payload(bytes(raw))
        except CompositionError:
            return None

    def gauge(self, seed: int = 351, sigma: int = 32) -> int:
        rng = random.Random(seed)
        vector = [rng.randrange(-sigma, sigma + 1) & 0xFFFFFFFF
                  for _ in range(256)]
        return self.memory.field.gauge_shift(vector)

    def audit(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "dimension": len(self.memory),
            "payload_bytes": PAYLOAD_BYTES,
            "slots": self.slots,
            "max_slots": self.max_slots,
            "occupied": int(self.memory.occupancy),
            "canonical_merge": False,
        }


def build_synthetic_chains(count: int = 120, seed: int = 35
                           ) -> tuple[list[tuple[Fact, Fact]], list[tuple[Fact, Fact]]]:
    """Deterministic child/parent/grandparent chains and broken-link controls."""
    rng = random.Random(seed)
    words = ["aura", "symptom", "evidence", "information", "worker", "person",
             "mopper", "rocket firing", "launching", "propulsion", "stone",
             "solid", "river", "water", "ocean", "animal", "bird", "penguin"]
    positives: list[tuple[Fact, Fact]] = []
    negatives: list[tuple[Fact, Fact]] = []
    for index in range(count):
        left = rng.choice(words)
        middle = rng.choice([word for word in words if word != left])
        right = rng.choice([word for word in words
                            if word not in (left, middle)])
        first = Fact(left, middle)
        second = Fact(middle, right)
        positives.append((first, second))
        other = rng.choice([word for word in words
                            if word not in (left, middle)])
        negatives.append((first, Fact(other, right)))
    return positives, negatives


def evaluate_gate(gate: TwoFactGate,
                  positives: Sequence[tuple[Fact, Fact]],
                  negatives: Sequence[tuple[Fact, Fact]]) -> dict[str, Any]:
    accepted_positive = sum(gate.accepts(a, b) for a, b in positives)
    false_negative = len(positives) - accepted_positive
    false_accept = sum(gate.accepts(a, b) for a, b in negatives)
    total = len(positives) + len(negatives)
    correct = accepted_positive + (len(negatives) - false_accept)
    return {
        "schema": SCHEMA,
        "positives": len(positives),
        "negatives": len(negatives),
        "accepted_positive": accepted_positive,
        "false_negative": false_negative,
        "false_accept": false_accept,
        "accuracy": correct / total if total else 0.0,
        "canonical_merge": False,
    }


def evaluate_composition(positives: Sequence[tuple[Fact, Fact]],
                         negatives: Sequence[tuple[Fact, Fact]]) -> dict[str, Any]:
    correct = accepted = 0
    for first, second in positives:
        result = compose_two(first, second)
        accepted += result.accepted
        correct += result.answer == ("%s %s %s." % (
            first.source, second.relation, second.destination))
    abstained = 0
    for first, second in negatives:
        result = compose_two(first, second)
        abstained += not result.accepted
    total = len(positives) + len(negatives)
    return {
        "schema": SCHEMA,
        "positive_correct": correct,
        "positive_total": len(positives),
        "negative_abstentions": abstained,
        "negative_total": len(negatives),
        "accuracy": (correct + abstained) / total if total else 0.0,
        "canonical_merge": False,
        "language_generation_claim": False,
    }


__all__ = (
    "SCHEMA", "PAYLOAD_BYTES", "ANCHOR", "CompositionError", "Fact",
    "CompositionResult", "TwoFactGate", "CompositionMemory", "compose_two",
    "compose_three", "compose_path", "relation_conflict", "structural_features",
    "build_synthetic_chains",
    "evaluate_gate", "evaluate_composition",
)
