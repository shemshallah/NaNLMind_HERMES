"""Persistent user-supplied lexical definitions.

This is an episodic glossary, not a learned semantic model: a term and its
definition are stored verbatim and can be recalled from another process. It
does not claim inference from the wording.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class LexicalMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        self._load()

    @staticmethod
    def key(term: str) -> str:
        return " ".join(str(term).casefold().split())

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            self.entries = {str(key): value for key, value in data.items()
                            if isinstance(value, dict)}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.entries, indent=2,
                                        sort_keys=True, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, self.path)

    def teach(self, term: str, definition: str, synset_address: str | None = None,
              related: list[dict] | None = None) -> dict:
        key = self.key(term)
        if not key or not definition.strip():
            raise ValueError("term and definition are required")
        record = {
            "term": str(term).strip(),
            "definition": str(definition).strip(),
            "synset_address": synset_address,
            "related": list(related or []),
        }
        replaced = key in self.entries
        self.entries[key] = record
        self._save()
        return {"stored": key, "replaced_existing": replaced,
                "memory_count": len(self.entries)}

    def recall(self, term: str) -> dict:
        self._load()
        key = self.key(term)
        return {"term": key, "found": key in self.entries,
                "result": self.entries.get(key)}

    def all_terms(self) -> list[str]:
        self._load()
        return sorted(self.entries)
