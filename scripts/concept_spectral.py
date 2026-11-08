#!/usr/bin/env python3
"""Build the coupling matrix from the concept relation graph and cut it.

The letter Gram graph is disconnected (Fiedler value ~0). The concept matrix
provides typed relation weights, and a weak ring backbone keeps the graph
connected so the spectral cut is meaningful.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nanlmind_hermes.concept_matrix import ConceptMatrix  # noqa: E402
from nanlmind_hermes.mth_layers import MTHHeadMapper, MTHLayerStack  # noqa: E402
from nanlmind_hermes.spectral import (  # noqa: E402
    cut_potential,
    native_mirror,
    normalized_laplacian,
    smallest_eigen,
)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/CONCEPT_SPECTRAL.md"
    corpus = json.loads((ROOT / "docs/gemma-corpus.json").read_text(encoding="utf-8"))
    statements = []
    for round_name in ("round_1", "round_2"):
        plan = corpus[round_name]["plan"]
        statements.extend(plan.get("statements", []))
        statements.extend(plan.get("uncertainty", []))
    matrix = ConceptMatrix()
    for statement in statements:
        matrix.observe(statement)

    mapper = MTHHeadMapper(MTHLayerStack())

    def concept_heads(concept: str) -> list[tuple[int, float]]:
        # A concept is a superposition of four root-transform head addresses.
        base = (sum(concept.encode("utf-8")) + len(concept)) % 16
        weights = (1.0, 0.5, 0.25, 0.125)
        return [(mapper.transform(concept, (base + step * 5) % 16) & 0xFF,
                 weight) for step, weight in enumerate(weights)]

    heads = {concept: concept_heads(concept) for concept in matrix.concepts}
    coupling = np.zeros((256, 256))
    for key, cell in matrix.pairs.items():
        first, second = key.split("\u0000")
        if first not in heads or second not in heads:
            continue
        for one, one_weight in heads[first]:
            for two, two_weight in heads[second]:
                weight = cell["weight"] * one_weight * two_weight
                if one == two:
                    coupling[one][one] += weight
                else:
                    coupling[one][two] += weight
                    coupling[two][one] += weight
    # Weak ring backbone: connected graph without washing out relation weights.
    for head in range(256):
        other = (head + 1) % 256
        coupling[head][other] += 0.001
        coupling[other][head] += 0.001

    normalized = normalized_laplacian(coupling.tolist())
    values, vectors = smallest_eigen(normalized, k=2, iterations=120)
    vector = vectors[1]
    positive = {head for head, value in enumerate(vector) if value >= 0}
    negative = set(range(256)) - positive
    inner = negative if len(negative) <= len(positive) else positive
    boundary = set()
    for head in range(256):
        for other in range(head + 1, 256):
            if coupling[head][other] > 1e-12 and \
                    (head in positive) != (other in positive):
                boundary.add(head)
                boundary.add(other)
    rank = int(np.linalg.matrix_rank(coupling))
    report = {
        "schema": "nanlmind-hermes-concept-spectral/1",
        "statements": len(statements),
        "concepts": len(matrix.concepts),
        "pair_cells": len(matrix.pairs),
        "heads_used": len({head for values in heads.values()
                           for head, _weight in values}),
        "rank": rank,
        "fiedler_values": [round(value, 6) for value in values],
        "inner_size": len(inner),
        "positive_size": len(positive),
        "negative_size": len(negative),
        "boundary_size": len(boundary),
        "inner": sorted(inner),
        "boundary": sorted(boundary),
        "native_mirror_fixed_points": sum(1 for head in range(256)
                                          if native_mirror(256)[head] == head),
        "note": ("weak ring backbone keeps the concept graph connected; the "
                 "cut is the normalized-Laplacian Fiedler partition"),
    }
    json_path = output.with_name("concept-spectral.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Concept-graph spectral cut",
        "",
        "The letter Gram graph was disconnected (Fiedler value ~0, inner 1,",
        "boundary 0). This matrix is built from the concept relation graph with",
        "a weak ring backbone to keep it connected.",
        "",
        f"- statements: {len(statements)}",
        f"- concepts: {len(matrix.concepts)}; pair cells: {len(matrix.pairs)}",
        f"- heads used: {report['heads_used']}; matrix rank: **{rank}**",
        f"- normalized Laplacian Fiedler values: {report['fiedler_values']}",
        f"- inner/positive/negative: {len(inner)}/{len(positive)}/{len(negative)}",
        f"- boundary: **{len(boundary)}**",
        f"- native mirror fixed points: {report['native_mirror_fixed_points']}",
        "",
        "A nondegenerate second eigenvalue and a boundary > 0 mean the cut is",
        "meaningful. The next step is to write word leaves into this connected",
        "matrix and measure whether their couplings land on the boundary modes.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("concepts", "pair_cells", "heads_used", "rank",
                       "fiedler_values", "inner_size", "boundary_size")},
                     indent=2))


if __name__ == "__main__":
    main()
