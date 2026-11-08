"""Concept matrix: emergent leaves over the C(256,2) relation space.

A word is not assigned to a cell. Every statement contributes all of its
concept pairs in superposition; repeated exposure accumulates weights and
typed roles in the 32,640 unordered pair cells. A concept's leaf identity is
the set of pairs it participates in (its relation signature), not its spelling.
The 256-head coordinate is only a coordinate; meaning lives in the links.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in",
    "on", "for", "with", "about", "as", "at", "by", "from", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "i", "you", "we", "they", "he", "she", "me", "my",
    "your", "our", "do", "does", "did", "can", "could", "would", "should",
    "what", "how", "where", "when", "why", "who", "which",
})
PAIR_COUNT = 256 * 255 // 2  # 32,640


def content_words(text: str) -> list[str]:
    return [word for word in TOKEN_RE.findall(str(text).casefold())
            if word not in STOP and len(word) > 1]


class ConceptMatrix:
    SCHEMA = "nanlm.concept-matrix/1"

    def __init__(self):
        self.pairs: dict[str, dict] = {}
        self.concepts: dict[str, dict] = {}
        self.statements = 0

    @staticmethod
    def _key(first: str, second: str) -> str:
        a, b = sorted((first, second))
        return a + "\u0000" + b

    def observe(self, statement: str, relation: dict | None = None) -> None:
        words = []
        for word in content_words(statement):
            if word not in words:
                words.append(word)
        if len(words) < 1:
            return
        self.statements += 1
        for index, word in enumerate(words):
            if word not in self.concepts:
                self.concepts[word] = {"count": 0, "links": []}
            self.concepts[word]["count"] += 1
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                key = self._key(words[i], words[j])
                cell = self.pairs.setdefault(key, {"weight": 0.0, "count": 0,
                                                   "roles": {}})
                cell["weight"] += 1.0 / (1 + abs(i - j))
                cell["count"] += 1
                for word in (words[i], words[j]):
                    if key not in self.concepts[word]["links"]:
                        self.concepts[word]["links"].append(key)
        if relation:
            subject = str(relation.get("subject") or "").strip().casefold()
            obj = str(relation.get("object") or "").strip().casefold()
            role = str(relation.get("role") or relation.get("predicate") or "").strip()
            if subject and obj and role:
                key = self._key(subject, obj)
                cell = self.pairs.setdefault(key, {"weight": 0.0, "count": 0,
                                                   "roles": {}})
                cell["roles"][role] = cell["roles"].get(role, 0) + 1
                a, b = key.split("\u0000")
                cell["rail"] = "s+" if subject <= obj else "s-"

    def leaf(self, concept: str) -> dict:
        data = self.concepts.get(concept, {"count": 0, "links": []})
        links = sorted(data["links"])
        signature = hashlib.sha256(
            ("|".join(links) + "|" + str(data["count"])).encode("utf-8")).hexdigest()
        return {"concept": concept, "count": data["count"],
                "degree": len(links), "links": links[:64],
                "leaf_id": signature[:32]}

    def saturation(self) -> dict:
        occupied = len(self.pairs)
        degrees = [len(data["links"]) for data in self.concepts.values()]
        roles: dict[str, int] = {}
        for cell in self.pairs.values():
            for role, count in cell["roles"].items():
                roles[role] = roles.get(role, 0) + count
        top_concepts = sorted(self.concepts.items(),
                              key=lambda item: -len(item[1]["links"]))[:12]
        top_pairs = sorted(self.pairs.items(),
                           key=lambda item: -item[1]["weight"])[:12]
        return {
            "schema": self.SCHEMA,
            "statements": self.statements,
            "concepts": len(self.concepts),
            "pair_cells": PAIR_COUNT,
            "occupied_pairs": occupied,
            "density": round(occupied / PAIR_COUNT, 6),
            "average_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
            "roles": roles,
            "top_concepts": [{"concept": name, "degree": len(data["links"])}
                             for name, data in top_concepts],
            "top_pairs": [{"pair": key.split("\u0000"), "weight": round(cell["weight"], 2),
                           "roles": cell["roles"]} for key, cell in top_pairs],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps({
            "schema": self.SCHEMA, "statements": self.statements,
            "pairs": self.pairs, "concepts": self.concepts,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
