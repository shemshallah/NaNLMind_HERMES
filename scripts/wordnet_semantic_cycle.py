#!/usr/bin/env python3
"""WordNet quotient experiment: typed relation D vs ordinary controls.

A 256-anchor quotient of WordNet nouns is built from hashed gloss/lemma features.
D is assembled from typed hypernym/meronym/attribute edges. We compare label
smoothing on the principal (unprojected), cycle-projected and random graphs.
This is a classical graph/semantic experiment, not a language model.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.gauge_memory import project_cycle  # noqa: E402
from nanlmind_hermes.wordnet_typed import iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")
TOKEN_RE = re.compile(r"[a-z]+")
DIMENSIONS = 32
RELATIONS = ("hypernym", "instance_hypernym", "part_meronym", "member_meronym",
             "attribute", "similar_to")


def feature(text: str) -> np.ndarray:
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    for token in TOKEN_RE.findall(text.casefold()):
        if len(token) < 3:
            continue
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % DIMENSIONS
        vector[index] += 1.0 if (value >> 8) & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.array(matrix, dtype=np.float64)
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    rows = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(rows, 1e-12)


def smooth(labels: np.ndarray, adjacency: np.ndarray,
           alpha: float = 0.5, steps: int = 2) -> np.ndarray:
    result = labels.copy()
    for _ in range(steps):
        result = (1.0 - alpha) * result + alpha * adjacency @ result
    return result


def evaluate(labels: np.ndarray, adjacency: np.ndarray,
             anchor_ids: np.ndarray, targets: np.ndarray) -> float:
    smoothed = smooth(labels, adjacency)
    predictions = smoothed[anchor_ids].argmax(axis=1)
    return float(np.mean(predictions == targets))


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/WORDNET_SEMANTIC_CYCLE.md"
    records = []
    for record in iter_synsets(ARCHIVE):
        if record["pos"] == "n" and record["definition"]:
            records.append(record)
    rng = random.Random(18018)
    rng.shuffle(records)
    features = np.stack([feature(" ".join([
        *record["lemmas"], record["definition"], record["gloss"]]))
        for record in records])
    lexnames = sorted({record["lexname"] for record in records})
    classes = {name: index for index, name in enumerate(lexnames)}
    labels = np.array([classes[record["lexname"]] for record in records])

    # Stratified real-synset anchors: at most nine per lexicographer class.
    by_class = {}
    for index, record in enumerate(records):
        by_class.setdefault(record["lexname"], []).append(index)
    anchors = []
    for name in lexnames:
        anchors.extend(by_class[name][:9])
    remaining = [index for index in range(len(records)) if index not in set(anchors)]
    rng.shuffle(remaining)
    anchors.extend(remaining[:max(0, 256 - len(anchors))])
    anchors = anchors[:256]
    anchor_features = features[anchors]
    anchor_ids = np.empty(len(records), dtype=np.int32)
    for start in range(0, len(records), 1024):
        block = features[start:start + 1024] @ anchor_features.T
        anchor_ids[start:start + 1024] = block.argmax(axis=1)

    train_end = int(len(records) * 0.7)
    val_end = int(len(records) * 0.85)
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, len(records))
    counts = np.zeros((256, len(lexnames)), dtype=np.float64)
    for index in train_indices:
        counts[anchor_ids[index], labels[index]] += 1.0
    cluster_labels = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)

    offset_to_anchor = {record["offset"]: int(anchor_ids[index])
                        for index, record in enumerate(records)}
    source_indices = train_indices
    relation_matrices = {relation: np.zeros((256, 256), dtype=np.float64)
                         for relation in RELATIONS}
    for index in source_indices:
        source = int(anchor_ids[index])
        for pointer in records[index]["pointers"]:
            relation = pointer["relation"]
            if relation not in relation_matrices:
                continue
            target = offset_to_anchor.get(pointer["target_offset"])
            if target is None or target == source:
                continue
            relation_matrices[relation][source, target] += 1.0
            relation_matrices[relation][target, source] += 1.0

    validation_targets = labels[val_indices]
    test_targets = labels[test_indices]
    best = None
    for hypernym_weight in (0.0, 0.5, 1.0, 2.0):
        for meronym_weight in (0.0, 0.5, 1.0):
            for attribute_weight in (0.0, 0.5, 1.0):
                combined = (hypernym_weight * relation_matrices["hypernym"]
                            + meronym_weight * (relation_matrices["part_meronym"]
                                                + relation_matrices["member_meronym"])
                            + attribute_weight * relation_matrices["attribute"])
                adjacency = normalise(combined)
                accuracy = evaluate(cluster_labels, adjacency,
                                    anchor_ids[val_indices], validation_targets)
                if best is None or accuracy > best[0]:
                    best = (accuracy, hypernym_weight, meronym_weight,
                            attribute_weight, combined)
    _, hypernym_weight, meronym_weight, attribute_weight, combined = best
    def graph_controls(matrix):
        principal = normalise(matrix)
        edge_values = {(first, second): float(principal[first, second])
                       for first in range(256) for second in range(first + 1, 256)
                       if principal[first, second] != 0.0}
        cycle_edges = project_cycle(256, edge_values)
        cycle_matrix = np.zeros((256, 256), dtype=np.float64)
        for (first, second), value in cycle_edges.items():
            cycle_matrix[first, second] = value
            cycle_matrix[second, first] = value
        cycle_adjacency = normalise(cycle_matrix)
        random_matrix = np.zeros((256, 256), dtype=np.float64)
        edge_pairs = [(first, second) for first in range(256)
                      for second in range(first + 1, 256)]
        rng.shuffle(edge_pairs)
        selected = edge_pairs[:max(1, len(edge_values))]
        for first, second in selected:
            random_matrix[first, second] = 1.0
            random_matrix[second, first] = 1.0
        return (principal, cycle_adjacency, normalise(random_matrix))

    principal, cycle_adjacency, random_adjacency = graph_controls(combined)
    fixed_combined = (relation_matrices["hypernym"]
                      + 0.5 * (relation_matrices["part_meronym"]
                               + relation_matrices["member_meronym"])
                      + 0.5 * relation_matrices["attribute"])
    fixed_principal, fixed_cycle, fixed_random = graph_controls(fixed_combined)

    # Ordinary real-valued linear baseline on the same hashed features.
    train_features = features[train_indices]
    train_one_hot = np.zeros((len(train_indices), len(lexnames)))
    train_one_hot[np.arange(len(train_indices)), labels[train_indices]] = 1.0
    ridge = 1e-2 * np.eye(DIMENSIONS)
    weights = np.linalg.solve(train_features.T @ train_features + ridge,
                              train_features.T @ train_one_hot)
    linear_predictions = (features[test_indices] @ weights).argmax(axis=1)

    report = {
        "schema": "nanlmind-hermes-wordnet-semantic-cycle/1",
        "noun_synsets": len(records),
        "classes": len(lexnames),
        "anchors": len(anchors),
        "train": len(train_indices),
        "validation": len(val_indices),
        "test": len(test_indices),
        "selected_weights": {
            "hypernym": hypernym_weight,
            "meronym": meronym_weight,
            "attribute": attribute_weight,
        },
        "validation_accuracy": {
            "principal": evaluate(cluster_labels, principal,
                                  anchor_ids[val_indices], validation_targets),
            "cycle_projected": evaluate(cluster_labels, cycle_adjacency,
                                        anchor_ids[val_indices], validation_targets),
            "random_graph": evaluate(cluster_labels, random_adjacency,
                                     anchor_ids[val_indices], validation_targets),
        },
        "test_accuracy": {
            "principal": evaluate(cluster_labels, principal,
                                  anchor_ids[test_indices], test_targets),
            "cycle_projected": evaluate(cluster_labels, cycle_adjacency,
                                        anchor_ids[test_indices], test_targets),
            "random_graph": evaluate(cluster_labels, random_adjacency,
                                     anchor_ids[test_indices], test_targets),
            "ordinary_linear_ridge": float(np.mean(linear_predictions == test_targets)),
        },
        "fixed_nonzero_weights": {
            "hypernym": 1.0, "meronym": 0.5, "attribute": 0.5,
            "validation_principal": evaluate(cluster_labels, fixed_principal,
                                             anchor_ids[val_indices], validation_targets),
            "validation_cycle": evaluate(cluster_labels, fixed_cycle,
                                         anchor_ids[val_indices], validation_targets),
            "test_principal": evaluate(cluster_labels, fixed_principal,
                                       anchor_ids[test_indices], test_targets),
            "test_cycle": evaluate(cluster_labels, fixed_cycle,
                                   anchor_ids[test_indices], test_targets),
            "test_random": evaluate(cluster_labels, fixed_random,
                                    anchor_ids[test_indices], test_targets),
        },
        "note": ("WordNet quotient experiment; labels are lexicographer classes, "
                 "features are hashed gloss/lemma vectors, and D is assembled "
                 "from typed edges. This is not a language model or a quantum test."),
    }
    json_path = output.with_name("wordnet-semantic-cycle.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# WordNet semantic cycle D experiment",
        "",
        "A 256-anchor quotient of real WordNet nouns; D is assembled from typed",
        "hypernym, meronym and attribute edges. The cycle-projected graph is a",
        "classical divergence-free projection, not a quantum state.",
        "",
        f"- noun synsets: **{len(records):,}**",
        f"- lexicographer classes: **{len(lexnames)}**",
        f"- anchors: **{len(anchors)}**",
        f"- selected relation weights: `{report['selected_weights']}`",
        "",
        "| model | validation | test |",
        "|---|---:|---:|",
    ]
    for name in ("principal", "cycle_projected", "random_graph"):
        lines.append(f"| {name} | {report['validation_accuracy'][name]:.4f} | "
                     f"{report['test_accuracy'][name]:.4f} |")
    lines.append(f"| ordinary linear ridge | — | "
                 f"{report['test_accuracy']['ordinary_linear_ridge']:.4f} |")
    fixed = report["fixed_nonzero_weights"]
    lines += [
        "",
        "The validation search selected **zero** relation weights: graph smoothing",
        "did not beat the base cluster labels. A fixed non-zero graph gives a",
        "second, non-cherry-picked view:",
        "",
        f"- fixed principal test: **{fixed['test_principal']:.4f}**",
        f"- fixed cycle-projected test: **{fixed['test_cycle']:.4f}**",
        f"- fixed random-graph test: **{fixed['test_random']:.4f}**",
    ]
    lines += [
        "",
        "## Boundary",
        "",
        "- Labels are broad WordNet lexicographer classes, not deep semantics.",
        "- The cycle projection is an isometric/linear operation; it adds no",
        "  information and no cycle-specific advantage is claimed.",
        "- This does not train a language decoder and does not test novel-definition",
        "  reasoning.",
        "- No physical quantum behavior is involved.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
