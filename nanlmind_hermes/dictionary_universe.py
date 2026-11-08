"""Dedicated dictionary universe at HW4, interfaced through NaNNet.

The WordNet graph is hosted as a sparse concept/synset universe at HW4. Hermes
(HW2) reaches it through NaNNet via HW4:0 and receives responses the same way.
No dense 256x256 matrix is allocated per word.
"""
from __future__ import annotations

from urllib.parse import quote

HW4_ROUTE = "nan://HW4:BS0:MC0:SC0:GM0"
HW4_LINK = "HW4:0"
UPSTREAM_LINK = "HW2:0"
BINARY_TRANSLATION = "NaN16DE"


class DictionaryUniverse:
    def __init__(self, synsets: dict, index: dict, properties: dict | None = None):
        self.synsets = synsets
        self.index = index
        self.properties = properties or {}

    def synset_address(self, offset: int) -> str:
        return f"{HW4_ROUTE}::synset/{offset:08d}"

    def word_address(self, word: str, offset: int) -> str:
        return f"{HW4_ROUTE}::word/{quote(str(word).casefold())}/{offset:08d}"

    def relation_address(self, source_offset: int, target_offset: int,
                         relation: str = "hypernym") -> str:
        return (f"{HW4_ROUTE}::relation/{quote(str(relation).casefold())}/"
                f"{source_offset:08d}/{target_offset:08d}")

    def ancestors(self, offset: int, depth: int = 8) -> set[int]:
        found = set()
        frontier = [offset]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for parent in self.synsets.get(node, {}).get("hypernyms", []):
                    if parent not in found:
                        found.add(parent)
                        next_frontier.append(parent)
            frontier = next_frontier
            if not frontier:
                break
        return found

    def inherit(self, offset: int) -> list[str]:
        closure = self.ancestors(offset)
        return sorted({prop for parent, prop in self.properties.items()
                       if parent in closure})

    def query(self, word: str) -> dict:
        offsets = self.index.get(str(word).casefold(), [])
        results = []
        for offset in offsets:
            results.append({
                "offset": offset,
                "address": self.synset_address(offset),
                "lemmas": self.synsets.get(offset, {}).get("lemmas", []),
                "inherited": self.inherit(offset),
            })
        return {"word": word, "synsets": results}

    def interface(self) -> dict:
        return {
            "route": HW4_ROUTE,
            "link": HW4_LINK,
            "upstream": UPSTREAM_LINK,
            "binary_translation": BINARY_TRANSLATION,
            "bidirectional": True,
            "note": ("Hermes HW2 -> NaNNet -> HW4:0 dictionary universe and back; "
                     "sparse synset/word addressing, no dense per-word matrix"),
        }
