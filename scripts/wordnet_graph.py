#!/usr/bin/env python3
"""Extract WordNet 3.1 into a concept/synset graph and test definition inheritance."""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.dictionary_universe import DictionaryUniverse  # noqa: E402
ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")
LINE_RE = re.compile(r"^(\d{8}) ")


def parse_synsets(archive: zipfile.ZipFile) -> dict:
    synsets = {}
    with archive.open("wn3.1.dict/dict/data.noun") as handle:
        for raw in handle:
            line = raw.decode("latin-1").strip()
            if "|" not in line:
                continue
            body, gloss = line.split("|", 1)
            parts = body.split()
            if len(parts) < 5 or not LINE_RE.match(body):
                continue
            offset = int(parts[0])
            pos = parts[2]
            word_count = int(parts[3], 16)
            index = 4
            words = []
            for _ in range(word_count):
                words.append(parts[index].replace("_", " "))
                index += 2
            if index >= len(parts):
                continue
            pointer_count = int(parts[index], 16)
            index += 1
            hypernyms = []
            for _ in range(pointer_count):
                if index + 3 >= len(parts):
                    break
                symbol = parts[index]
                target = int(parts[index + 1])
                index += 4
                if symbol in {"@", "@i"}:
                    hypernyms.append(target)
            synsets[offset] = {"offset": offset, "pos": pos, "lemmas": words,
                               "gloss": gloss.strip(), "hypernyms": hypernyms}
    return synsets


def parse_index(archive: zipfile.ZipFile, synsets: dict) -> dict:
    index = {}
    with archive.open("wn3.1.dict/dict/index.noun") as handle:
        for raw in handle:
            line = raw.decode("latin-1").strip()
            if not line or line.startswith(" "):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            lemma = parts[0].replace("_", " ").lower()
            try:
                synset_count = int(parts[2])
                pointer_count = int(parts[3])
            except ValueError:
                continue
            start = 6 + pointer_count
            offsets = [int(value) for value in parts[start:start + synset_count]]
            offsets = [offset for offset in offsets if offset in synsets]
            if offsets:
                index.setdefault(lemma, []).extend(offsets)
    return index


def ancestors(synsets: dict, offset: int, depth: int = 8) -> set[int]:
    found = set()
    frontier = [offset]
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            for parent in synsets.get(node, {}).get("hypernyms", []):
                if parent not in found:
                    found.add(parent)
                    next_frontier.append(parent)
        frontier = next_frontier
        if not frontier:
            break
    return found


def first_synset(index: dict, word: str):
    offsets = index.get(word.casefold(), [])
    return offsets[0] if offsets else None


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/WORDNET_GRAPH.md"
    archive = zipfile.ZipFile(ARCHIVE)
    synsets = parse_synsets(archive)
    index = parse_index(archive, synsets)
    lemma_count = len(index)
    edge_count = sum(len(data["hypernyms"]) for data in synsets.values())

    words = ["animal", "bird", "penguin", "vehicle", "car", "entity"]
    selected = {word: first_synset(index, word) for word in words}
    paths = {}
    for word, offset in selected.items():
        if offset is None:
            continue
        chain = [offset]
        current = offset
        for _ in range(8):
            parents = synsets.get(current, {}).get("hypernyms", [])
            if not parents:
                break
            current = parents[0]
            chain.append(current)
        paths[word] = [
            {"offset": node,
             "lemmas": synsets.get(node, {}).get("lemmas", [])[:4],
             "gloss": synsets.get(node, {}).get("gloss", "")[:120]}
            for node in chain
        ]

    # Controlled definition inheritance: attach properties to a few high synsets,
    # then test words defined only by a link to their synset.
    properties = {}
    if selected.get("animal"):
        properties[selected["animal"]] = "living"
    if selected.get("vehicle"):
        properties[selected["vehicle"]] = "artifact"
    if selected.get("bird"):
        properties[selected["bird"]] = "winged"
    universe = DictionaryUniverse(synsets, index, properties)
    probes = []
    inherited = 0
    sample_words = [word for word in index
                    if word not in {"animal", "vehicle", "bird"}]
    for word in sample_words[:5000]:
        offset = first_synset(index, word)
        if offset is None:
            continue
        hits = universe.inherit(offset)
        if hits:
            probes.append({"word": word, "definition_synset": offset,
                           "address": universe.synset_address(offset),
                           "inherited": hits})
            inherited += 1
        if len(probes) >= 32:
            break

    report = {
        "schema": "nanlmind-hermes-wordnet-graph/1",
        "archive": str(ARCHIVE),
        "synsets": len(synsets),
        "lemmas": lemma_count,
        "hypernym_edges": edge_count,
        "sample_paths": paths,
        "definition_inheritance_probes": probes,
        "definition_inheritance_count": inherited,
        "properties": {str(key): value for key, value in properties.items()},
        "interface": universe.interface(),
        "note": ("definitions are supplied graph edges; this is compositional "
                 "graph retrieval, not learned parsing of prose"),
    }
    json_path = output.with_name("wordnet-graph.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# WordNet 3.1 concept/synset graph",
        "",
        "WordNet is extracted as synset nodes, lemma edges and hypernym edges.",
        "A word is not assigned to a head: a word leaf points to its synset;",
        "the synset participates in the concept graph.",
        "",
        f"- synsets: **{len(synsets):,}**",
        f"- lemmas: **{lemma_count:,}**",
        f"- hypernym edges: **{edge_count:,}**",
        f"- definition-inheritance probes: **{inherited}**",
        f"- HW4 route: `{universe.interface()['route']}` via `{universe.interface()['link']}` "
        f"(upstream `{universe.interface()['upstream']}`, {universe.interface()['binary_translation']})",
        "",
        "## Hypernym paths (sample)",
        "",
    ]
    for word, path in paths.items():
        lines.append(f"### {word}")
        for node in path:
            lines.append(f"- {node['offset']}: {', '.join(node['lemmas'])} — {node['gloss']}")
    lines += [
        "",
        "## Definition inheritance",
        "",
        "A synthetic property is attached to a high-level synset; a word defined",
        "only by a synset link inherits it through the hypernym closure.",
        "",
        "```json",
        json.dumps(probes[:12], indent=2)[:5000],
        "```",
        "",
        "## Hosting and encoding",
        "",
        "- Do not allocate one 256×256 matrix per word. Use sparse synset/lemma",
        "  addresses and a dedicated HW4 universe/chart hierarchy behind NaNROUTER.",
        "- A word leaf is a hard-mapped letter hyperedge plus its synset membership;",
        "  polysemy is a superposition of synset leaves.",
        "- Hypernym/meronym/similar edges are typed; order-sensitive relations use",
        "  non-Abelian operators, not scalar phases.",
        "- Cycle fields and gauge-aware retrieval enforce consistency; definitions",
        "  are graph edges, not yet parsed prose.",
        "",
        "## Honest boundary",
        "",
        "This is a real dictionary graph and compositional inheritance test, not a",
        "trained language model. It supplies the lexical/concept substrate the",
        "cycle memory and head population need.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "synsets": len(synsets), "lemmas": lemma_count,
                      "hypernym_edges": edge_count,
                      "definition_inheritance": inherited}, indent=2))


if __name__ == "__main__":
    main()
