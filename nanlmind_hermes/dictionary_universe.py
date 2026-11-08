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
                data = self.synsets.get(node, {})
                parents = list(data.get("hypernyms", []))
                parents += [pointer.get("target_offset")
                            for pointer in data.get("pointers", [])
                            if pointer.get("relation") in {"hypernym", "instance_hypernym"}]
                for parent in parents:
                    if parent is not None and parent not in found:
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

    def relations(self, offset: int, relation: str | None = None) -> list[dict]:
        """Typed outgoing edges; hypernym-only records are upgraded on read."""
        data = self.synsets.get(offset, {})
        pointers = list(data.get("pointers", []))
        if not pointers and data.get("hypernyms"):
            pointers = [{"relation": "hypernym", "target_pos": data.get("pos", "n"),
                         "target_offset": target}
                        for target in data["hypernyms"]]
        result = []
        for pointer in pointers:
            name = pointer.get("relation", "unknown")
            if relation is not None and name != relation:
                continue
            target_offset = pointer.get("target_offset")
            target = self.synsets.get(target_offset, {})
            result.append({
                "relation": name,
                "source": self.synset_address(offset),
                "target": self.synset_address(target_offset),
                "target_offset": target_offset,
                "target_pos": pointer.get("target_pos", target.get("pos")),
                "target_lemmas": target.get("lemmas", []),
            })
        return result

    def define(self, offset: int) -> dict:
        """Structured definition record; this is a graph readout, not generated prose."""
        data = self.synsets.get(offset, {})
        gloss = data.get("gloss", "")
        definition = data.get("definition") or gloss.split('; "', 1)[0].strip()
        return {
            "offset": offset,
            "address": self.synset_address(offset),
            "pos": data.get("pos"),
            "lemmas": data.get("lemmas", []),
            "lexname": data.get("lexname"),
            "definition": definition,
            "examples": data.get("examples", []),
            "inherited": self.inherit(offset),
            "hypernyms": self.relations(offset, "hypernym")
                          + self.relations(offset, "instance_hypernym"),
            "relations": self.relations(offset),
        }

    def hypernym_chain(self, offset: int, depth: int = 8) -> list[dict]:
        chain = []
        current = offset
        for _ in range(depth):
            parents = self.relations(current, "hypernym")
            parents += self.relations(current, "instance_hypernym")
            if not parents:
                break
            current = parents[0]["target_offset"]
            chain.append(self.define(current))
        return chain

    def learn_word(self, word: str, offset: int) -> str:
        """Attach a new word leaf to an existing synset without a dense matrix."""
        key = str(word).casefold()
        offsets = self.index.setdefault(key, [])
        if offset not in offsets:
            offsets.append(offset)
        return self.word_address(word, offset)

    def query(self, word: str) -> dict:
        offsets = self.index.get(str(word).casefold(), [])
        results = []
        for offset in offsets:
            data = self.synsets.get(offset, {})
            results.append({
                "offset": offset,
                "address": self.synset_address(offset),
                "lemmas": data.get("lemmas", []),
                "definition": data.get("definition", data.get("gloss", "")),
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
