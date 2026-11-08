"""Conservative definition acquisition and inherited-category inference.

Supported grammar is deliberately narrow:
    "a kind/type/species/form/variety of X that ..."
    "is a X", "means a X"

A head is resolved only when exactly one WordNet noun sense exists. An unknown
category returns "unknown", never an invented "no". This is symbol grounding by
an explicit parent link, not unrestricted language understanding.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from nanlmind_hermes.wordnet_reasoning import ancestors, parents

HEAD_PATTERNS = [
    r"\b(?:a|an) (?:kind|type|species|form|variety) of "
    r"([a-z][a-z -]{1,45}?)(?:\s+(?:that|which|with|having|used|found|"
    r"living|capable)\b|[.,;]|$)",
    r"\b(?:is a|is an|means a|means an) "
    r"([a-z][a-z -]{1,45}?)(?:\s+(?:that|which|with|having|used|found|"
    r"living|capable)\b|[.,;]|$)",
]


def possible_heads(definition: str) -> list[str]:
    text = " ".join(str(definition).casefold().split())
    heads = []
    for pattern in HEAD_PATTERNS:
        match = re.search(pattern, text)
        if match:
            head = match.group(1).strip(" .,;")
            if head and head not in heads:
                heads.append(head)
    return heads


class SemanticAcquisition:
    def __init__(self, nodes: dict, lemma_index: dict, store=None):
        self.nodes = nodes
        self.lemma_index = lemma_index
        self.store = Path(store) if store else None
        self.custom: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.store or not self.store.exists():
            return
        try:
            data = json.loads(self.store.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.custom = {str(key): value for key, value in data.items()
                           if isinstance(value, dict)}

    def _save(self) -> None:
        if not self.store:
            return
        self.store.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.store.with_name(self.store.name + ".tmp")
        temporary.write_text(json.dumps(self.custom, indent=2, sort_keys=True,
                                        ensure_ascii=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, self.store)

    def teach(self, term: str, definition: str) -> dict:
        key = " ".join(str(term).casefold().split())
        if not key or not str(definition).strip():
            raise ValueError("term and definition are required")
        candidates = []
        for head in possible_heads(definition):
            for offset in self.lemma_index.get(head, []):
                candidates.append({
                    "head": head,
                    "synset": offset,
                    "gloss": self.nodes.get(offset, {}).get("definition",
                                                            self.nodes.get(offset, {}).get("gloss", "")),
                })
        # Deduplicate by sense; any ambiguity remains unresolved.
        unique = {}
        for candidate in candidates:
            unique[candidate["synset"]] = candidate
        candidates = list(unique.values())
        resolved = candidates[0]["synset"] if len(candidates) == 1 else None
        self.custom[key] = {
            "term": str(term).strip(),
            "definition": str(definition).strip(),
            "head_candidates": candidates,
            "resolved_synset": resolved,
        }
        self._save()
        return {
            "term": key,
            "head_candidates": len(candidates),
            "resolved": resolved is not None,
            "requires_disambiguation": len(candidates) > 1,
        }

    def ask(self, term: str, category: str) -> dict:
        key = " ".join(str(term).casefold().split())
        category = " ".join(str(category).casefold().split())
        item = self.custom.get(key)
        if item is None:
            return {"answer": "unknown", "reason": "term not taught"}
        start = item.get("resolved_synset")
        if start is None:
            return {"answer": "unknown",
                    "reason": "definition head ambiguous or unresolved",
                    "candidates": len(item.get("head_candidates", []))}
        targets = self.lemma_index.get(category, [])
        if not targets:
            return {"answer": "unknown", "reason": "category absent from noun lexicon"}
        closure = ancestors(self.nodes, start, max_depth=8)
        overlap = [(depth, target) for target in targets
                   for depth in [closure.get(target)] if depth is not None]
        if not overlap:
            return {"answer": "unknown",
                    "reason": "no known hypernym chain; not evidence of falsehood"}
        depth, target = min(overlap)
        parent_record = self.nodes.get(start, {})
        return {
            "answer": "yes",
            "distance": depth + 1,
            "path_target": self._label(target),
            "evidence": {
                "taught_definition": item["definition"],
                "head": self._label(start),
                "head_definition": parent_record.get("definition",
                                                     parent_record.get("gloss", "")),
            },
        }

    def explain(self, term: str) -> str:
        key = " ".join(str(term).casefold().split())
        item = self.custom.get(key)
        if item is None:
            return "I have not been taught that word."
        if item.get("resolved_synset") is None:
            return (f"{item['term']}: {item['definition']} "
                    "The parent sense is ambiguous or unresolved; no taxonomy claim is made.")
        parent = self.nodes[item["resolved_synset"]]
        return (f"{item['term']}: {item['definition']} "
                f"Resolved parent: {self._label(item['resolved_synset'])} — "
                f"{parent.get('definition', parent.get('gloss', ''))}")

    def _label(self, offset: int) -> str:
        record = self.nodes.get(offset, {})
        lemmas = record.get("lemmas", [])
        return lemmas[0] if lemmas else str(offset)
