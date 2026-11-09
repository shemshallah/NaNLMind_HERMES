"""HW4 WordNet mirror reader for the Round 042 TRIGON experiment.

The canonical route is read-only evidence.  This module reads a compact local
mirror and builds the WordNet relation matrix used to initialize and probe the
cycle field.  It does not open a live NANNET route and never enables canonical
merge.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from trigon import CYCLE_COUNT, HEAD_COUNT, CycleField, head_for_name

HW4_ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW0:0::HW4:0"
ARITY = 4
RELATION_LABELS = {"none": 0, "hypernym": 1, "hyponym": 2, "opposes": 3}
PHASE = {
    "none": 0.0,
    "hypernym": math.pi / 2.0,
    "hyponym": -math.pi / 2.0,
    "opposes": math.pi,
}


@dataclass(frozen=True)
class Record:
    id: str
    split: str
    lemmas: tuple[str, ...]
    gloss: str
    pointers: tuple[tuple[str, str, str], ...]
    lm: bool = True
    route: str = HW4_ROUTE

    @property
    def lemma(self) -> str:
        return self.lemmas[0] if self.lemmas else self.id


@dataclass(frozen=True)
class Relation:
    left: str
    right: str
    label: int
    split: str
    provenance: str

    def as_tuple(self) -> tuple[str, str, int]:
        return self.left, self.right, self.label


def load_records(path: str | Path) -> list[Record]:
    records: list[Record] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records.append(Record(
                id=str(row["id"]),
                split=str(row.get("split", "train")),
                lemmas=tuple(str(item) for item in row.get("lemmas", ())),
                gloss=str(row.get("gloss", "")).strip(),
                pointers=tuple(
                    (str(kind), str(target), str(pos))
                    for kind, target, pos in row.get("pointers", ())
                ),
                lm=bool(row.get("lm", True)),
                route=str(row.get("route", HW4_ROUTE)),
            ))
    return records


def build_documents(records: list[Record]) -> dict[str, list[bytes]]:
    docs: dict[str, list[bytes]] = {"train": [], "val": [], "test": []}
    for record in records:
        if not record.lm:
            continue
        split = record.split if record.split in docs else "train"
        if not record.gloss:
            continue
        text = f"{record.lemma}: {record.gloss}".encode("utf-8", errors="replace")
        if 12 <= len(text) <= 360:
            docs[split].append(text)
    return docs


def _lemma_map(records: list[Record]) -> dict[str, str]:
    return {record.id: record.lemma for record in records}


def build_relations(records: list[Record], max_per_label: int = 700
                    ) -> dict[str, list[Relation]]:
    """Build balanced typed-evidence examples from the full WordNet matrix."""
    by_id = {record.id: record for record in records}
    buckets: dict[str, dict[int, list[Relation]]] = {
        split: {label: [] for label in RELATION_LABELS.values()}
        for split in ("train", "val", "test")
    }
    seen: set[tuple[str, str, str, int]] = set()

    def add(split: str, left: str, right: str, label: int, provenance: str) -> None:
        key = (split, left.casefold(), right.casefold(), label)
        if (not left or not right or left.casefold() == right.casefold()
                or key in seen or len(buckets[split][label]) >= max_per_label):
            return
        seen.add(key)
        buckets[split][label].append(Relation(left, right, label, split, provenance))

    for record in records:
        split = record.split if record.split in buckets else "train"
        child = record.lemma
        for kind, target, _pos in record.pointers:
            target_id = target if target.startswith("n") else f"n{target}"
            parent_record = by_id.get(target_id)
            if parent_record is None:
                continue
            parent = parent_record.lemma
            if kind in ("@", "@i"):
                add(split, child, parent, 1, f"{kind}:{record.id}->{target}")
                add(split, parent, child, 2, f"reverse:{kind}:{target}->{record.id}")
            elif kind == "!":
                add(split, child, parent, 3, f"!:{record.id}->{target}")

    # Balanced deterministic controls: unrelated lemmas from the same split.
    for split, labels in buckets.items():
        positives = max(len(labels[1]), len(labels[2]), 1)
        pool = [record for record in records if record.split == split and record.lm]
        if not pool:
            continue
        target = min(max_per_label, positives)
        cursor = 0
        while len(labels[0]) < target and cursor < len(pool) * 8:
            left = pool[cursor % len(pool)]
            right = pool[(cursor * 7919 + 17) % len(pool)]
            cursor += 1
            if left.lemma[:1].casefold() == right.lemma[:1].casefold():
                continue
            add(split, left.lemma, right.lemma, 0, "control:unrelated")
    return {split: [row for label in sorted(labels) for row in labels[label]]
            for split, labels in buckets.items()}


def apply_wordnet_prior(cycle: CycleField, relations: list[Relation],
                        scale: float = 0.35, amp: float = 0.55) -> dict:
    """Write the WordNet matrix into the cycle field's base edge coordinates."""
    relation_by_label = {value: key for key, value in RELATION_LABELS.items()}
    touched = 0
    with torch.no_grad():
        for relation in relations:
            if relation.label == 0:
                continue
            name = relation_by_label[relation.label]
            theta = PHASE[name] * scale
            logamp = math.log(max(amp, 1e-3)) * scale
            i = head_for_name(relation.left)
            j = head_for_name(relation.right)
            if i == j:
                continue
            # Minimum-norm three-edge distribution of a cycle value.
            cycle.edge_phase[0, i] += theta / 3.0
            cycle.edge_phase[i, j] += theta / 3.0
            cycle.edge_phase[0, j] -= theta / 3.0
            cycle.edge_logamp[0, i] += logamp / 3.0
            cycle.edge_logamp[i, j] += logamp / 3.0
            cycle.edge_logamp[0, j] -= logamp / 3.0
            touched += 1
    return {"relations_written": touched, "prior_scale": scale,
            "amp": amp, "route": HW4_ROUTE}


def matrix_digest(records: list[Record], relations: dict[str, list[Relation]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.id.encode())
        digest.update(b"\0")
        digest.update(record.gloss.encode())
        digest.update(b"\0")
    for split in sorted(relations):
        for relation in relations[split]:
            digest.update(repr(relation.as_tuple()).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def describe(path: str | Path) -> dict:
    records = load_records(path)
    relations = build_relations(records)
    return {
        "route": HW4_ROUTE,
        "source": str(path),
        "records": len(records),
        "splits": {split: sum(1 for record in records if record.split == split)
                   for split in ("train", "val", "test")},
        "relations": {split: len(rows) for split, rows in relations.items()},
        "digest": matrix_digest(records, relations),
        "live_nannet_route": False,
        "canonical_merge": False,
    }


__all__ = (
    "HW4_ROUTE", "RELATION_LABELS", "PHASE", "Record", "Relation",
    "load_records", "build_documents", "build_relations", "apply_wordnet_prior",
    "matrix_digest", "describe",
)
