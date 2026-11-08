#!/usr/bin/env python3
"""Extract typed WordNet edges and write the inventory report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.dictionary_universe import (  # noqa: E402
    BINARY_TRANSLATION,
    HW4_LINK,
    HW4_ROUTE,
    UPSTREAM_LINK,
)
from nanlmind_hermes.wordnet_typed import audit, iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/WORDNET_TYPED.md"
    report = audit(ARCHIVE)
    report["schema"] = "nanlmind-hermes-wordnet-typed/1"
    report["archive"] = str(ARCHIVE)
    report["interface"] = {
        "route": HW4_ROUTE,
        "link": HW4_LINK,
        "upstream": UPSTREAM_LINK,
        "binary_translation": BINARY_TRANSLATION,
    }

    by_lemma = {}
    noun_synsets = {}
    for record in iter_synsets(ARCHIVE):
        noun_synsets[(record["pos"], record["offset"])] = record
        for lemma in record["lemmas"]:
            by_lemma.setdefault(lemma, []).append((record["pos"], record["offset"]))
    samples = {}
    for word in ("entity", "animal", "bird", "penguin", "vehicle", "car"):
        candidates = by_lemma.get(word, [])
        if not candidates:
            continue
        key = next((item for item in candidates if item[0] == "n"), candidates[0])
        record = noun_synsets[key]
        chain = [key]
        current = key
        for _ in range(8):
            parents = [pointer for pointer in noun_synsets[current]["pointers"]
                       if pointer["relation"] in {"hypernym", "instance_hypernym"}]
            if not parents:
                break
            parent = (parents[0]["target_pos"], parents[0]["target_offset"])
            if parent not in noun_synsets:
                break
            chain.append(parent)
            current = parent
        samples[word] = {
            "synset": f"{key[0]}:{key[1]:08d}",
            "lexname": record["lexname"],
            "definition": record["definition"],
            "examples": record["examples"][:2],
            "typed_relations": record["pointers"][:12],
            "hypernym_chain": [
                {"synset": f"{node[0]}:{node[1]:08d}",
                 "lemmas": noun_synsets[node]["lemmas"][:4],
                 "definition": noun_synsets[node]["definition"][:160]}
                for node in chain if node in noun_synsets
            ],
        }
    report["samples"] = samples

    json_path = output.with_name("wordnet-typed.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Typed WordNet 3.1 graph",
        "",
        "Every WordNet pointer is retained as a typed directed edge. The graph is",
        "addressed sparsely at HW4; it is not a dense matrix per word.",
        "",
        f"- synsets: **{report['synsets']:,}**",
        f"- lemma occurrences: **{report['lemmas']:,}**",
        f"- typed pointer edges: **{report['typed_edges']:,}**",
        f"- hypernym/instance edges: **{report['hypernym_edges']:,}**",
        f"- gloss characters: **{report['gloss_chars']:,}**",
        f"- route: `{HW4_ROUTE}` via `{HW4_LINK}` "
        f"(upstream `{UPSTREAM_LINK}`, {BINARY_TRANSLATION})",
        "",
        "## Positions",
        "",
        "| POS | synsets | lemma occurrences | typed pointers | gloss chars |",
        "|---|---:|---:|---:|---:|",
    ]
    for pos, data in report["positions"].items():
        lines.append(f"| {pos} | {data['synsets']:,} | {data['lemmas']:,} | "
                     f"{data['pointers']:,} | {data['gloss_chars']:,} |")
    lines += [
        "",
        "## Relation inventory",
        "",
        "| relation | edges |",
        "|---|---:|",
    ]
    for relation, count in report["relations"].items():
        lines.append(f"| {relation} | {count:,} |")
    lines += [
        "",
        "## Sample concept records",
        "",
    ]
    for word, sample in samples.items():
        lines += [
            f"### {word}",
            "",
            f"- synset: `{sample['synset']}`",
            f"- lexname: `{sample['lexname']}`",
            f"- definition: {sample['definition']}",
            f"- examples: {json.dumps(sample['examples'])}",
            "- hypernym chain:",
        ]
        for node in sample["hypernym_chain"]:
            lines.append(f"  - `{node['synset']}` {', '.join(node['lemmas'])} — "
                         f"{node['definition']}")
        lines.append("- typed relations: `" + json.dumps(sample["typed_relations"]) + "`")
        lines.append("")
    lines += [
        "## Boundaries",
        "",
        "- This is a typed lexical graph, not a learned language model.",
        "- Directed graph edges are not a quantum resource; gauge loops are classical",
        "  matrix algebra.",
        "- The next experiment trains operators on these typed edges and enforces",
        "  composition only where the relation semantics justify it.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "json": str(json_path),
        "synsets": report["synsets"],
        "typed_edges": report["typed_edges"],
        "hypernym_edges": report["hypernym_edges"],
        "relations": len(report["relations"]),
    }, indent=2))


if __name__ == "__main__":
    main()
