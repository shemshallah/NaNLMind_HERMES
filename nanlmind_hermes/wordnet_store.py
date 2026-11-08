"""Sparse HW4 WordNet universe loader.

The full WordNet 3.1 archive is streamed into a sparse synset dictionary and a
lemma -> offset index.  It is not converted to a dense per-word matrix.  The
result is the typed graph consumed by :class:`DictionaryUniverse`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dictionary_universe import DictionaryUniverse
from .wordnet_typed import iter_synsets

STORE_SCHEMA = "nanlm.wordnet-store/1"
DEFAULT_CANDIDATES = (
    "/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip",
    str(Path.home() / "wn3.1.dict.zip"),
    "/data/wn3.1.dict.zip",
)


class WordNetStoreError(ValueError):
    pass


class WordNetStore:
    def __init__(self, synsets: dict[int, dict], index: dict[str, list[int]]):
        if not isinstance(synsets, dict) or not isinstance(index, dict):
            raise WordNetStoreError("synsets and index must be dictionaries")
        self.synsets = synsets
        self.index = index
        self.universe = DictionaryUniverse(synsets, index)

    @classmethod
    def from_archive(
        cls,
        path: str | Path,
        *,
        pos: set[str] | None = None,
        max_synsets: int | None = None,
    ) -> "WordNetStore":
        target = Path(path)
        if not target.is_file():
            raise WordNetStoreError("WordNet archive does not exist")
        if max_synsets is not None and (isinstance(max_synsets, bool)
                                        or not isinstance(max_synsets, int)
                                        or max_synsets < 1):
            raise WordNetStoreError("max_synsets must be a positive integer")
        synsets: dict[int, dict] = {}
        index: dict[str, list[int]] = {}
        for record in iter_synsets(target):
            if pos is not None and record.get("pos") not in pos:
                continue
            offset = int(record["offset"])
            synsets[offset] = record
            for lemma in record.get("lemmas", []):
                offsets = index.setdefault(str(lemma).casefold(), [])
                if offset not in offsets:
                    offsets.append(offset)
            if max_synsets is not None and len(synsets) >= max_synsets:
                break
        if not synsets:
            raise WordNetStoreError("WordNet archive produced no synsets")
        return cls(synsets, index)

    @classmethod
    def from_synsets(cls, records: list[dict]) -> "WordNetStore":
        synsets: dict[int, dict] = {}
        index: dict[str, list[int]] = {}
        for record in records:
            if not isinstance(record, dict) or "offset" not in record:
                raise WordNetStoreError("synset record is missing its offset")
            offset = int(record["offset"])
            synsets[offset] = record
            for lemma in record.get("lemmas", []):
                offsets = index.setdefault(str(lemma).casefold(), [])
                if offset not in offsets:
                    offsets.append(offset)
        return cls(synsets, index)

    @classmethod
    def discover(cls, extra: str | None = None) -> "WordNetStore":
        candidates = (extra, *DEFAULT_CANDIDATES)
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.is_file():
                return cls.from_archive(path)
        raise WordNetStoreError("no WordNet 3.1 archive found; set NANLM_WORDNET_ARCHIVE")

    def stats(self) -> dict[str, Any]:
        positions: dict[str, int] = {}
        pointers = 0
        for record in self.synsets.values():
            pos = str(record.get("pos", "?"))
            positions[pos] = positions.get(pos, 0) + 1
            pointers += len(record.get("pointers", []))
        return {
            "schema": STORE_SCHEMA,
            "synsets": len(self.synsets),
            "lemmas": len(self.index),
            "pointers": pointers,
            "positions": dict(sorted(positions.items())),
            "interface": self.universe.interface(),
        }
