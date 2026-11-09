"""Bounded add/ask session over the raw support/refute evidence lanes.

The session is deliberately small.  It recognizes a finite set of taxonomic
phrasings, writes the parsed record through :class:`EvidenceLanes`, and answers
only from the recorded evidence.  It is not an open-domain parser or a fluent
generator; unrecognized input is rejected instead of guessed.

Save/load is a JSON-safe lane snapshot; reload restores the same records into
raw cycle memory and preserves the four-state result.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .evidence_lanes import (
    EvidenceError,
    EvidenceLanes,
    EvidenceRecord,
    EvidenceState,
)

SCHEMA = "nanlm.resident-evidence-session/1"
MAX_STATEMENT = 512

_SUPPORT_PATTERNS = (
    (re.compile(r"^(?P<subject>.+?) is a kind of (?P<destination>.+)$",
                re.IGNORECASE), "is a kind of"),
    (re.compile(r"^(?P<subject>.+?) is a type of (?P<destination>.+)$",
                re.IGNORECASE), "is a type of"),
    (re.compile(r"^(?P<subject>.+?) is an? (?P<destination>.+)$",
                re.IGNORECASE), "is a"),
    (re.compile(r"^(?P<subject>.+?) are a kind of (?P<destination>.+)$",
                re.IGNORECASE), "is a kind of"),
    (re.compile(r"^(?P<subject>.+?) are an? (?P<destination>.+)$",
                re.IGNORECASE), "is a"),
)
_REFUTE_PATTERNS = (
    (re.compile(r"^(?P<subject>.+?) is not a kind of (?P<destination>.+)$",
                re.IGNORECASE), "is a kind of"),
    (re.compile(r"^(?P<subject>.+?) is not a type of (?P<destination>.+)$",
                re.IGNORECASE), "is a type of"),
    (re.compile(r"^(?P<subject>.+?) is not an? (?P<destination>.+)$",
                re.IGNORECASE), "is a"),
    (re.compile(r"^(?P<subject>.+?) are not a kind of (?P<destination>.+)$",
                re.IGNORECASE), "is a kind of"),
    (re.compile(r"^(?P<subject>.+?) are not an? (?P<destination>.+)$",
                re.IGNORECASE), "is a"),
)
_ASK_PATTERNS = (
    (re.compile(r"^is (?P<subject>.+?) a kind of (?P<destination>.+)$",
                re.IGNORECASE), "is a kind of"),
    (re.compile(r"^is (?P<subject>.+?) a type of (?P<destination>.+)$",
                re.IGNORECASE), "is a type of"),
    (re.compile(r"^is (?P<subject>.+?) an? (?P<destination>.+)$",
                re.IGNORECASE), "is a"),
)


class SessionError(EvidenceError):
    """A statement or question is outside the bounded grammar."""


def _clean(value: str) -> str:
    text = " ".join(value.strip().split())
    return text.rstrip(" .?!").strip()


def _bounded(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise SessionError("%s must be a string" % name)
    text = _clean(value)
    if not text or len(text) > MAX_STATEMENT:
        raise SessionError("%s must contain 1..%d characters"
                           % (name, MAX_STATEMENT))
    return text


def _match(patterns, text):
    for pattern, relation in patterns:
        match = pattern.match(text)
        if match:
            subject = _clean(match.group("subject"))
            destination = _clean(match.group("destination"))
            if subject and destination:
                return subject, relation, destination
    return None


@dataclass(frozen=True)
class TellResult:
    statement: str
    polarity: str
    subject: str
    relation: str
    destination: str
    lane: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "action": "tell",
            "statement": self.statement,
            "polarity": self.polarity,
            "subject": self.subject,
            "relation": self.relation,
            "destination": self.destination,
            "lane": self.lane,
            "canonical_merge": False,
        }


@dataclass(frozen=True)
class AskResult:
    question: str
    subject: str
    relation: str
    destination: str
    status: str
    answer: str
    state: EvidenceState

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "action": "ask",
            "question": self.question,
            "subject": self.subject,
            "relation": self.relation,
            "destination": self.destination,
            "status": self.status,
            "answer": self.answer,
            "state": self.state.as_dict(),
            "canonical_merge": False,
        }


class EvidenceSession:
    """Parse bounded taxonomic statements into independent evidence lanes."""

    def __init__(self, lanes: EvidenceLanes):
        if not isinstance(lanes, EvidenceLanes):
            raise SessionError("lanes must be an EvidenceLanes")
        self.lanes = lanes

    def tell(self, statement: str, polarity: str | None = None) -> TellResult:
        original = _bounded(statement, "statement")
        parsed = _match(_REFUTE_PATTERNS, original)
        inferred = "refutes" if parsed is not None else None
        if parsed is None:
            parsed = _match(_SUPPORT_PATTERNS, original)
            inferred = "supports" if parsed is not None else None
        if parsed is None:
            raise SessionError(
                "statement is outside the bounded 'X is [not] a [kind of] Y' grammar")
        if polarity is None:
            polarity = inferred
        if polarity not in ("supports", "refutes"):
            raise SessionError("polarity must be 'supports' or 'refutes'")
        subject, relation, destination = parsed
        receipt = self.lanes.add(subject, relation, destination, polarity)
        return TellResult(original, polarity, subject, relation, destination,
                          receipt.as_dict())

    def ask(self, question: str) -> AskResult:
        original = _bounded(question, "question")
        parsed = _match(_ASK_PATTERNS, original)
        if parsed is None:
            raise SessionError(
                "question is outside the bounded 'is X a [kind of] Y?' grammar")
        subject, relation, destination = parsed
        state = self.lanes.state(subject, relation, destination)
        if relation == "is a":
            article = "an" if destination[:1].lower() in "aeiou" else "a"
            positive = "%s is %s %s" % (subject, article, destination)
            negative = "%s is not %s %s" % (subject, article, destination)
        else:
            positive = "%s %s %s" % (subject, relation, destination)
            negative = "%s %s %s" % (
                subject, relation.replace("is a ", "is not a ", 1),
                destination)
        if state.status == "supports":
            answer = "Yes, %s." % positive
        elif state.status == "refutes":
            answer = "No, %s." % negative
        elif state.status == "both":
            answer = ("The resident evidence both supports and refutes: %s."
                      % positive)
        else:
            answer = ("I do not have enough resident evidence to decide "
                      "whether %s." % positive)
        return AskResult(original, subject, relation, destination,
                         state.status, answer, state)

    def save(self) -> dict[str, Any]:
        return self.lanes.snapshot()

    def load(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return self.lanes.restore(snapshot).as_dict()

    def status(self) -> dict[str, Any]:
        records = self.lanes.records()
        return {
            "schema": SCHEMA,
            "records": len(records),
            "supports": sum(record.polarity == "supports"
                            for record in records),
            "refutes": sum(record.polarity == "refutes"
                           for record in records),
            "generation": self.lanes.generation,
            "dimension": len(self.lanes.memory),
            "canonical_merge": False,
        }


__all__ = (
    "SCHEMA", "MAX_STATEMENT", "SessionError", "TellResult", "AskResult",
    "EvidenceSession",
)
