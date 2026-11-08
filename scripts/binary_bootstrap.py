#!/usr/bin/env python3
"""Absolute-basics binary bootstrap: bits -> functions -> bytes -> letters ->
words -> vectors, with bytewise completion and head self-modification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.leaf_algebra import (  # noqa: E402
    SEED_WORDS,
    ByteCompletion,
    LeafAlgebra,
    build_mapping_tree,
)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/BINARY_BOOTSTRAP.md"
    algebra = LeafAlgebra()

    # 1. Verify every root Boolean function against its own truth table.
    verified = 0
    total = 0
    for mask in range(16):
        for a in (0, 1):
            for b in (0, 1):
                left = algebra.value_leaf("BOOLEAN", a)
                right = algebra.value_leaf("BOOLEAN", b)
                result = algebra.apply(f"BOOL_{mask}", left, right)
                total += 1
                verified += int(result.value) == ((mask >> ((a << 1) | b)) & 1)

    # 2. Build the initial mapping matrix tree.
    tree = build_mapping_tree(algebra)

    # 3. Bytewise completion: train on 0..191, test on 192..255.
    learner = ByteCompletion()
    learner.train(list(range(0, 192)))
    completion = learner.evaluate(list(range(192, 256)))
    interleaved = ByteCompletion()
    interleaved.train([value for value in range(256) if value % 4 != 0])
    completion_interleaved = interleaved.evaluate(
        [value for value in range(256) if value % 4 == 0])

    # 4. Word leaves and typed vectors.
    words = {}
    for word in ("NO", "YES", "ON", "NOT", "ONE", "ZERO", "BOTH"):
        leaf = algebra.word(word)
        words[word] = {"leaf": leaf.leaf_id, "address": leaf.address,
                       "members": [algebra.leaves[member].value
                                   for member in leaf.provenance["members"]],
                       "hyperedge": leaf.provenance["hyperedge"]}
    letter = algebra.letter("A")
    letter_superposition = algebra.head_superposition(letter)
    vector = algebra.vector((1, 0, 1), dtype="BOOLEAN")
    matrix = algebra.matrix(((1, 0), (0, 1)))

    report = {
        "schema": "nanlmind-hermes-binary-bootstrap/1",
        "root_functions": {"verified": verified, "total": total},
        "tree": {level: len(items) for level, items in tree.items()},
        "byte_completion": completion,
        "byte_completion_interleaved": completion_interleaved,
        "letters": len([leaf for leaf in algebra.leaves.values()
                        if leaf.kind == "LETTER"]),
        "words": words,
        "letter_example": {"letter": "A", "canonical_head": letter.tags["canonical_head"],
                           "superposition": letter_superposition,
                           "derivatives": letter.tags["derivatives"]},
        "dictionary": {"seed_spellings": len(SEED_WORDS),
                        "semantics_preloaded": False,
                        "note": "orthography only; definitions and relations are learned"},
        "vector": {"rank": vector.rank, "shape": vector.shape,
                   "dtype": vector.dtype, "value": vector.value,
                   "address": vector.address},
        "matrix": {"rank": matrix.rank, "shape": matrix.shape,
                   "dtype": matrix.dtype},
        "heads": {"used": len(algebra.heads),
                  "usable": len([head for head in algebra.heads if head != 0]),
                  "nannet_head": 0,
                  "nannet_head_role": "NaNNet + polarity",
                  "leaves": len(algebra.leaves),
                  "admissions": sum(1 for event in algebra.ledger
                                    if event["event"] == "admit"),
                  "transforms": sum(1 for event in algebra.ledger
                                    if event["event"] == "transform")},
    }
    json_path = output.with_name("binary-bootstrap.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Binary bootstrap: bits to words",
        "",
        "Everything is a typed leaf with provenance. 0 and 1 are value leaves and",
        "the 16 Boolean functions are function leaves. Letters are byte patterns;",
        "words are ordered hyperedges over letters; vectors are constructed,",
        "dimension-aware leaves. No free-floating linguistic labels.",
        "",
        "## Root Boolean table",
        "",
        f"- verified: **{verified}/{total}** function/input combinations",
        "- functions: `BOOL_0` … `BOOL_15`, plus `NEGATE`, `AFFIRM`, `CONCAT`",
        "",
        "## Initial mapping matrix tree",
        "",
        "| level | contents | count |",
        "|---|---|---:|",
        f"| L0 | bit value leaves | {report['tree']['L0_bits']} |",
        f"| L1 | functions | {report['tree']['L1_functions']} |",
        f"| L2 | byte leaves (0x00 anchors head 0; 0x01–0xFF use heads 1–255) | {report['tree']['L2_bytes']} |",
        f"| L3 | printable letter leaves | {report['tree']['L3_letters']} |",
        f"| L4 | word hyperedges | {report['tree']['L4_words']} |",
        f"| L5 | constructed vector leaves | {report['tree']['L5_vectors']} |",
        "",
        "## Bytewise completion",
        "",
        "Contiguous split (train 0–191, test 192–255):",
        "",
        "```json",
        json.dumps(completion, indent=2, sort_keys=True),
        "```",
        "",
        "Interleaved split (train values not divisible by 4):",
        "",
        "```json",
        json.dumps(completion_interleaved, indent=2, sort_keys=True),
        "```",
        "",
        "## Hard-mapped letters with transform superposition",
        "",
        "A is fixed at ASCII 65; the leaf carries the transform family and",
        "derivatives. The identity transform is dominant; no letter mapping is",
        "learned.",
        "",
        "```json",
        json.dumps(report["letter_example"], indent=2, sort_keys=True),
        "```",
        "",
        "Structural dictionary only: " + str(report["dictionary"]["seed_spellings"])
        + " seed spellings; no definitions or semantic relations are preloaded.",
        "",
        "## Word leaves as hyperedges",
        "",
        "| word | letters | address |",
        "|---|---|---|",
    ]
    for word, data in words.items():
        lines.append(f"| {word} | {''.join(data['members'])} | `{data['address']}` |")
    lines += [
        "",
        "## Typed vector and matrix leaves",
        "",
        "```json",
        json.dumps({"vector": report["vector"], "matrix": report["matrix"]},
                   indent=2, sort_keys=True),
        "```",
        "",
        "## Heads and self-modification",
        "",
        "```json",
        json.dumps(report["heads"], indent=2, sort_keys=True),
        "```",
        "",
        "## What this establishes",
        "",
        "- 1 and 0 are both values and Boolean-function leaves; NEGATE is a",
        "  registered typed transform, not a language operator.",
        "- Letters are grounded in byte patterns; NO is a hyperedge over N and O.",
        "- Vectors carry rank, shape, dtype, value and provenance; they are built",
        "  by functions, not given as free embeddings.",
        "- The mapping tree is structural: bits -> functions -> bytes -> letters",
        "  -> words -> vectors. The heads admit every leaf and record transforms.",
        "- Bytewise completion is the first sequential skill; it is the same",
        "  prefix-to-next prediction later used on word/temporal rails.",
        "",
        "Honest limit: completion accuracy is a frequency baseline over held-out",
        "bytes; it is not yet a learned Boolean circuit per byte. The next stage",
        "is learned completion (Boolean circuits over bit prefixes), then letters",
        "and word hyperedges on the temporal rails.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "json": str(json_path),
        "root": f"{verified}/{total}",
        "completion_accuracy": completion["accuracy"],
        "completion_interleaved": completion_interleaved["accuracy"],
        "heads_used": report["heads"]["used"],
        "usable_heads": report["heads"]["usable"],
        "leaves": report["heads"]["leaves"],
    }, indent=2))


if __name__ == "__main__":
    main()
