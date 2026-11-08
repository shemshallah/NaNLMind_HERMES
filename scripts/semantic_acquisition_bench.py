#!/usr/bin/env python3
"""Round 019-style controlled definition acquisition and cycle audit."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.semantic_acquisition import SemanticAcquisition  # noqa: E402
from nanlmind_hermes.wordnet_reasoning import ancestors, parents  # noqa: E402
from nanlmind_hermes.wordnet_typed import iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")
GENERIC = {"entity", "physical entity", "object", "whole"}


def cycle_audit(seed: int = 19) -> dict:
    rng = np.random.default_rng(seed)
    n = 256
    components = rng.normal(size=96)
    field = np.zeros((n, n), dtype=float)
    index = 0
    for first in range(1, n):
        for second in range(first + 1, n):
            if index < len(components):
                field[first, second] = components[index]
                field[second, first] = -components[index]
            index += 1

    def fluxes(matrix):
        return np.array([matrix[0, first] + matrix[first, second]
                         - matrix[0, second]
                         for first in range(1, n)
                         for second in range(first + 1, n)])

    def gauge(matrix, potential):
        return matrix + potential[None, :] - potential[:, None]

    def project(matrix):
        divergence = matrix.sum(axis=1)
        return matrix - (divergence[:, None] - divergence[None, :]) / n

    original_flux = fluxes(field)
    potential = rng.normal(size=n)
    shifted_flux = fluxes(gauge(field, potential))
    projected = project(field)
    projected_shifted = project(gauge(field, potential))
    return {
        "vertices": n,
        "pairs": n * (n - 1) // 2,
        "independent_triangles": (n - 1) * (n - 2) // 2,
        "encoded_features": len(components),
        "max_gauge_flux_error": float(np.max(np.abs(original_flux - shifted_flux))),
        "max_encoded_recovery_error": float(
            np.max(np.abs(original_flux[:len(components)] - components))),
        "max_cycle_divergence": float(np.max(np.abs(projected.sum(axis=1)))),
        "max_projected_gauge_error": float(
            np.max(np.abs(projected - projected_shifted))),
        "max_projected_flux_error": float(
            np.max(np.abs(fluxes(projected) - original_flux))),
    }


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/SEMANTIC_ACQUISITION.md"
    nodes = {}
    lemma_index = {}
    for record in iter_synsets(ARCHIVE):
        if record["pos"] != "n":
            continue
        nodes[record["offset"]] = record
        for lemma in record["lemmas"]:
            lemma_index.setdefault(lemma, []).append(record["offset"])

    rng = random.Random(19019)
    eligible = []
    for offset, record in nodes.items():
        parent_offsets = parents(record)
        if len(parent_offsets) != 1 or parent_offsets[0] not in nodes:
            continue
        parent = parent_offsets[0]
        parent_label = nodes[parent]["lemmas"][0]
        if len(lemma_index.get(parent_label, [])) != 1:
            continue
        ancestor_map = ancestors(nodes, parent, max_depth=4)
        targets = [(depth, target) for target, depth in ancestor_map.items()
                   if 1 <= depth <= 2
                   and nodes[target]["lemmas"][0] not in GENERIC
                   and len(lemma_index.get(nodes[target]["lemmas"][0], [])) == 1]
        if targets:
            eligible.append((parent, targets))
    rng.shuffle(eligible)
    eligible = eligible[:400]
    engine = SemanticAcquisition(nodes, lemma_index)
    results = []
    abstentions = 0
    abstention_trials = 0
    for index, (parent, targets) in enumerate(eligible):
        term = f"novelterm{index}"
        definition = (f"a kind of {nodes[parent]['lemmas'][0]} "
                      "that has an unusual property")
        taught = engine.teach(term, definition)
        depth, target = targets[rng.randrange(len(targets))]
        answer = engine.ask(term, nodes[target]["lemmas"][0])
        distractor_parent, _ = eligible[(index + max(1, len(eligible) // 2))
                                        % len(eligible)]
        distractor = nodes[distractor_parent]["lemmas"][0]
        ancestor_labels = {nodes[item]["lemmas"][0]
                           for item in ancestors(nodes, parent, max_depth=8)}
        if distractor != nodes[parent]["lemmas"][0] and distractor not in ancestor_labels:
            abstention_trials += 1
            if engine.ask(term, distractor)["answer"] == "unknown":
                abstentions += 1
        results.append({
            "resolved": taught["resolved"],
            "correct": answer["answer"] == "yes",
            "hop": answer.get("distance"),
            "parent": nodes[parent]["lemmas"][0],
            "target": nodes[target]["lemmas"][0],
        })

    report = {
        "schema": "nanlmind-hermes-semantic-acquisition/1",
        "noun_synsets": sum(1 for record in nodes.values() if record["pos"] == "n"),
        "unique_noun_labels": len(lemma_index),
        "eligible": len(eligible),
        "new_terms_taught": len(results),
        "head_resolution_rate": float(np.mean([item["resolved"] for item in results])),
        "multi_hop_correct_rate": float(np.mean([item["correct"] for item in results])),
        "unrelated_category_abstentions": abstentions,
        "unrelated_category_trials": abstention_trials,
        "mean_inferred_hops": float(np.mean([item["hop"] for item in results
                                             if item["hop"]])),
        "examples": results[:10],
        "cycle_audit": cycle_audit(),
        "note": ("controlled grammar: the definition names one unambiguous WordNet "
                 "parent; this is symbolic composition, not unrestricted NLP"),
    }
    json_path = output.with_name("semantic-acquisition.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Controlled definition acquisition",
        "",
        "A new term is learned only from the conservative forms `a kind of X that`",
        "or `a type of X that`. The head must resolve to exactly one WordNet noun",
        "sense; otherwise the engine abstains. This is symbolic composition, not",
        "unrestricted definition understanding.",
        "",
        f"- noun synsets: **{report['noun_synsets']:,}**",
        f"- unique noun labels: **{report['unique_noun_labels']:,}**",
        f"- eligible new terms: **{report['eligible']:,}**",
        f"- head resolution: **{report['head_resolution_rate']:.4f}**",
        f"- multi-hop inference: **{report['multi_hop_correct_rate']:.4f}**",
        f"- unrelated-category abstentions: "
        f"**{report['unrelated_category_abstentions']}/{report['unrelated_category_trials']}**",
        f"- mean inferred hops: **{report['mean_inferred_hops']:.2f}**",
        "",
        "## Cycle audit",
        "",
        f"- encoded coordinates: **{report['cycle_audit']['encoded_features']}**",
        f"- max gauge flux error: "
        f"**{report['cycle_audit']['max_gauge_flux_error']:.3e}**",
        f"- max encoded recovery error: "
        f"**{report['cycle_audit']['max_encoded_recovery_error']:.3e}**",
        f"- max cycle divergence: "
        f"**{report['cycle_audit']['max_cycle_divergence']:.3e}**",
        f"- max projected gauge error: "
        f"**{report['cycle_audit']['max_projected_gauge_error']:.3e}**",
        "",
        "## Boundary",
        "",
        "- The parent sense is supplied by a narrow grammar and must be unique.",
        "- The target ancestor is known in WordNet; this is not open-domain prose QA.",
        "- Abstention is not a verified falsehood decision.",
        "- Cycle projection is classical linear algebra and is not yet in the",
        "  trainable language path.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
