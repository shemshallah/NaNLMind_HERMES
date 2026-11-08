"""Structured definition readout over the typed WordNet universe.

This is a retrieval/graph decoder, not a generative language model. It returns
raw definitions, typed relations and ancestry paths; it does not synthesize
prose from templates.
"""
from __future__ import annotations

import math
import re

from nanlmind_hermes.dictionary_universe import DictionaryUniverse
from nanlmind_hermes.wordnet_reasoning import ancestors, parents

TOKEN_RE = re.compile(r"[a-z]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "with", "which", "who", "whose", "their", "this", "these", "those",
}


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.casefold())
            if token not in STOPWORDS and len(token) > 1]


class DefinitionAnswerer:
    def __init__(self, synsets: dict, index: dict):
        self.synsets = synsets
        self.index = index
        self.universe = DictionaryUniverse(synsets, index)
        self._idf = None

    def _documents(self):
        for offset, record in self.synsets.items():
            text = " ".join([*record.get("lemmas", []),
                             record.get("definition", ""),
                             record.get("gloss", "")])
            yield offset, set(tokens(text))

    def idf(self) -> dict[str, float]:
        if self._idf is None:
            frequency = {}
            count = 0
            for _offset, document in self._documents():
                count += 1
                for token in document:
                    frequency[token] = frequency.get(token, 0) + 1
            self._idf = {token: math.log((count + 1) / (value + 1)) + 1.0
                         for token, value in frequency.items()}
        return self._idf

    def search(self, definition: str, candidates: list[int] | None = None,
               limit: int = 5) -> list[dict]:
        query = set(tokens(definition))
        if not query:
            return []
        if candidates is None:
            candidates = list(self.synsets)
        idf = self.idf()
        query_weight = sum(idf.get(token, 1.0) for token in query)
        results = []
        for offset in candidates:
            record = self.synsets.get(offset, {})
            text = " ".join([*record.get("lemmas", []),
                             record.get("definition", ""),
                             record.get("gloss", "")])
            document = set(tokens(text))
            overlap = sum(idf.get(token, 1.0) for token in query & document)
            score = overlap / max(query_weight, 1e-12)
            if score > 0:
                results.append({"offset": offset, "score": score,
                                "definition": record.get("definition", ""),
                                "lemmas": record.get("lemmas", [])})
        results.sort(key=lambda item: (-item["score"], item["offset"]))
        return results[:limit]

    def answer_what(self, word: str, sense: int = 0) -> dict | None:
        offsets = self.index.get(word.casefold(), [])
        if not offsets:
            return None
        offset = offsets[min(max(sense, 0), len(offsets) - 1)]
        return self.universe.define(offset)

    def answer_is_a(self, child: str, parent: str) -> dict:
        child_offsets = self.index.get(child.casefold(), [])
        parent_offsets = set(self.index.get(parent.casefold(), []))
        for offset in child_offsets:
            if parent_offsets & set(ancestors(self.synsets, offset, max_depth=8)):
                return {"answer": True, "child_offset": offset,
                        "parent_offsets": sorted(parent_offsets),
                        "path": self.path(offset, parent_offsets)}
        return {"answer": False, "child_offsets": child_offsets,
                "parent_offsets": sorted(parent_offsets), "path": []}

    def path(self, start: int, targets: set[int], max_depth: int = 8) -> list[int]:
        frontier = [(start, [start])]
        seen = {start}
        for _ in range(max_depth):
            next_frontier = []
            for node, chain in frontier:
                for parent in parents(self.synsets.get(node, {})):
                    if parent in seen:
                        continue
                    next_chain = chain + [parent]
                    if parent in targets:
                        return next_chain
                    seen.add(parent)
                    next_frontier.append((parent, next_chain))
            frontier = next_frontier
            if not frontier:
                break
        return []

    def answer_relation(self, word: str, relation: str) -> list[dict]:
        offsets = self.index.get(word.casefold(), [])
        results = []
        for offset in offsets:
            for edge in self.universe.relations(offset, relation):
                results.append(edge)
        return results
