#!/usr/bin/env python3
"""Controlled new-word acquisition through the typed WordNet graph.

This is deliberately a graph operation, not a prose parser or a trained
language model: a new word leaf is attached to an existing synset, then its
inherited properties are queried through hypernym pointers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.dictionary_universe import DictionaryUniverse  # noqa: E402
from nanlmind_hermes.wordnet_typed import iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/DEFINITION_ACQUISITION.md"
    synsets = {}
    index = {}
    for record in iter_synsets(ARCHIVE):
        if record["pos"] != "n":
            continue
        offset = record["offset"]
        synsets[offset] = record
        for lemma in record["lemmas"]:
            index.setdefault(lemma, []).append(offset)

    def first(word):
        return index.get(word, [None])[0]

    selected = {word: first(word) for word in
                ("animal", "bird", "penguin", "vehicle", "car")}
    properties = {}
    if selected["animal"]:
        properties[selected["animal"]] = "living"
    if selected["bird"]:
        properties[selected["bird"]] = "winged"
    universe = DictionaryUniverse(synsets, index, properties)

    new_word = "sparklebeak"
    target = selected["penguin"]
    address = universe.learn_word(new_word, target)
    query = universe.query(new_word)
    definition = universe.define(target)
    chain = universe.hypernym_chain(target, depth=5)
    report = {
        "schema": "nanlmind-hermes-definition-acquisition/1",
        "new_word": new_word,
        "word_address": address,
        "linked_synset": universe.synset_address(target),
        "query": query,
        "definition": definition,
        "hypernym_chain": chain,
        "properties": {str(key): value for key, value in properties.items()},
        "note": ("graph-level acquisition only: the new word leaf is attached to "
                 "an existing synset; prose definitions are not parsed yet"),
    }
    json_path = output.with_name("definition-acquisition.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# New word from a definition link",
        "",
        "This is the controlled graph step, not a learned prose parser: the new",
        "word is attached to an existing synset, then queried through typed",
        "hypernym pointers.",
        "",
        f"- new word: **{new_word}**",
        f"- word address: `{address}`",
        f"- linked synset: `{universe.synset_address(target)}`",
        f"- source lemmas: {', '.join(definition['lemmas'][:4])}",
        f"- source definition: {definition['definition']}",
        f"- inherited properties: **{', '.join(query['synsets'][0]['inherited'])}**",
        "",
        "## Hypernym chain",
        "",
    ]
    for node in chain:
        lines.append(f"- `{node['address']}` {', '.join(node['lemmas'][:4])} — "
                     f"{node['definition']}")
    lines += [
        "",
        "## Boundaries",
        "",
        "- The new word does not get a dense per-word matrix; it is a sparse leaf.",
        "- Linking a word to a synset is not parsing arbitrary natural-language",
        "  definitions. A prose parser and a learned definition decoder remain open.",
        "- Inheritance is graph traversal; it does not establish language understanding.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "new_word": new_word,
                      "inherited": query["synsets"][0]["inherited"]}, indent=2))


if __name__ == "__main__":
    main()
