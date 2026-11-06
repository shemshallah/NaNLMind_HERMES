"""Conservative corpus-to-memory extraction.

Only explicit surface forms are converted. This module never executes text or
infers unstated relations; every record retains exact source-line provenance.
"""
from __future__ import annotations

import re
from pathlib import Path

ABILITY = re.compile(
    r"\b(?P<entity>[a-z][a-z0-9_-]{0,50})\s+"
    r"(?P<polarity>cannot|can)\s+(?P<ability>[a-z][a-z0-9_-]{0,40})\b",
    re.IGNORECASE,
)
IS_A = re.compile(
    r"\b(?P<entity>[a-z][a-z0-9_-]{0,50})\s+is\s+"
    r"(?P<negative>not\s+)?(?:a|an)\s+(?P<class>[a-z][a-z0-9_-]{0,50})\b",
    re.IGNORECASE,
)
EQUALITY = re.compile(
    r"\b(?P<left>[a-z][a-z0-9_]{0,40})\s*(?:==|=)\s*"
    r"(?P<right>[a-z0-9][a-z0-9_.-]{0,40})\b",
    re.IGNORECASE,
)
IMPLICATION = re.compile(
    r"\bif\s+(?P<left>[a-z][a-z0-9_-]{0,40})\s+then\s+"
    r"(?P<right>[a-z][a-z0-9_-]{0,40})\b",
    re.IGNORECASE,
)


def extract_discoveries(corpus_path: str | Path, source_digest: str) -> list[dict]:
    """Extract literal relations/equations with line-accurate evidence."""
    path = Path(corpus_path)
    records: list[dict] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        sentences = re.split(r"(?<=[.!?])\s+", raw_line.strip())
        for sentence_index, sentence in enumerate(sentences):
            text = sentence.strip()
            if not text:
                continue
            evidence = {
                "source_line": line_number,
                "sentence_index": sentence_index,
                "text": text[:1000],
                "samples": 1,
            }
            common = {
                "source_digest": source_digest,
                "head": None,
                "layer": 0,
            }
            for match in ABILITY.finditer(text):
                entity = match.group("entity").lower()
                ability = match.group("ability").lower()
                negative = match.group("polarity").lower() == "cannot"
                item = {
                    "kind": "relation",
                    "subject": f"ability:{entity}:{ability}",
                    "predicate": "ability",
                    "object": ability,
                    "polarity": "refutes" if negative else "supports",
                    "confidence": 1.0,
                    "evidence": evidence,
                    "provenance": {"extractor": "surface-relations/1", "source_line": line_number},
                    "tags": ["corpus-extract", "ability"],
                }
                records.append({**common, "item": item})
            for match in IS_A.finditer(text):
                entity = match.group("entity").lower()
                category = match.group("class").lower()
                negative = bool(match.group("negative"))
                item = {
                    "kind": "relation",
                    "subject": f"entity:{entity}:is_a:{category}",
                    "predicate": "is_a",
                    "object": category,
                    "polarity": "refutes" if negative else "supports",
                    "confidence": 1.0,
                    "evidence": evidence,
                    "provenance": {"extractor": "surface-relations/1", "source_line": line_number},
                    "tags": ["corpus-extract", "taxonomy"],
                }
                records.append({**common, "item": item})
            for match in EQUALITY.finditer(text):
                left, right = match.group("left").lower(), match.group("right").lower()
                item = {
                    "kind": "equation",
                    "subject": f"equation:{left}={right}",
                    "expression": {"op": "equals", "left": left, "right": right},
                    "polarity": "neutral",
                    "confidence": 1.0,
                    "evidence": evidence,
                    "provenance": {"extractor": "surface-relations/1", "source_line": line_number},
                    "tags": ["corpus-extract", "equation"],
                }
                records.append({**common, "item": item})
            for match in IMPLICATION.finditer(text):
                left, right = match.group("left").lower(), match.group("right").lower()
                item = {
                    "kind": "constraint",
                    "subject": f"implication:{left}:{right}",
                    "predicate": "implies",
                    "object": right,
                    "conditions": [left],
                    "polarity": "supports",
                    "confidence": 1.0,
                    "evidence": evidence,
                    "provenance": {"extractor": "surface-relations/1", "source_line": line_number},
                    "tags": ["corpus-extract", "implication"],
                }
                records.append({**common, "item": item})
    return records
