#!/usr/bin/env python3
"""Train a small state-conditioned answer decoder on controlled WordNet QA.

The decoder answers whether a newly taught term (linked to a unique WordNet
parent) is a queried category. Features are a definition hash, an ancestry
state and a query hash. Ordinary, cycle-projected and no-state controls use
the same decoder family and feature width. This is not a language model.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.semantic_state import (  # noqa: E402
    cycle_state,
    hash_features,
    ordinary_state,
)
from nanlmind_hermes.wordnet_reasoning import ancestors, parents  # noqa: E402
from nanlmind_hermes.wordnet_typed import iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")
GENERIC = {"entity", "physical entity", "object", "whole"}


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge: float):
    augmented = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    normal = augmented.T @ augmented
    normal[:-1, :-1] += ridge * np.eye(normal.shape[0] - 1)
    weights = np.linalg.solve(normal, augmented.T @ targets)
    return weights


def scores(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    return augmented @ weights


def predict(features: np.ndarray, weights: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return (scores(features, weights) >= threshold).astype(int)


def balanced_accuracy(labels, predictions):
    positive = np.mean(predictions[labels == 1] == 1)
    negative = np.mean(predictions[labels == 0] == 0)
    return float((positive + negative) / 2.0)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/JOINT_SEMANTIC_DECODER.md"
    nodes = {}
    lemma_index = {}
    for record in iter_synsets(ARCHIVE):
        if record["pos"] != "n":
            continue
        nodes[record["offset"]] = record
        for lemma in record["lemmas"]:
            lemma_index.setdefault(lemma, []).append(record["offset"])

    rng = random.Random(20020)
    eligible = []
    for offset, record in nodes.items():
        parent_offsets = parents(record)
        if len(parent_offsets) != 1 or parent_offsets[0] not in nodes:
            continue
        parent = parent_offsets[0]
        parent_label = nodes[parent]["lemmas"][0]
        if len(lemma_index.get(parent_label, [])) != 1:
            continue
        ancestor_map = ancestors(nodes, parent, max_depth=3)
        targets = [(depth, target) for target, depth in ancestor_map.items()
                   if 1 <= depth <= 3
                   and nodes[target]["lemmas"][0] not in GENERIC
                   and len(lemma_index.get(nodes[target]["lemmas"][0], [])) == 1]
        if targets:
            eligible.append((parent, targets))
    rng.shuffle(eligible)
    eligible = eligible[:5000]
    split = int(len(eligible) * 0.7)
    validation_split = int(len(eligible) * 0.85)
    splits = {
        "train": eligible[:split],
        "validation": eligible[split:validation_split],
        "test": eligible[validation_split:],
    }

    def examples(pairs):
        data = {"ordinary": [], "cycle": [], "none": [], "labels": [],
                "terms": []}
        for parent, targets in pairs:
            parent_label = nodes[parent]["lemmas"][0]
            definition = f"a kind of {parent_label} that has an unusual property"
            definition_feature = np.array(hash_features(definition, 64))
            ordinary = np.array(ordinary_state(nodes, parent, depth=3))
            cycle = np.array(cycle_state(nodes, parent, depth=3))
            target_offset = targets[rng.randrange(len(targets))][1]
            target_label = nodes[target_offset]["lemmas"][0]
            query_feature = np.array(hash_features(target_label, 64))
            # One positive and one unrelated same-schema negative.
            negative = rng.choice(eligible)[0]
            negative_label = nodes[negative]["lemmas"][0]
            ancestor_labels = {nodes[item]["lemmas"][0]
                               for item in ancestors(nodes, parent, max_depth=8)}
            if negative_label in ancestor_labels or negative_label == parent_label:
                negative_label = "entity"
            for label, query_label in ((1, target_label), (0, negative_label)):
                query = np.array(hash_features(query_label, 64))
                data["ordinary"].append(np.concatenate([definition_feature,
                                                         ordinary, query]))
                data["cycle"].append(np.concatenate([definition_feature,
                                                      cycle, query]))
                data["none"].append(np.concatenate([definition_feature,
                                                     np.zeros(256), query]))
                data["labels"].append(label)
                data["terms"].append((parent_label, query_label))
        return {key: np.asarray(value, dtype=np.float64)
                for key, value in data.items() if key != "terms"}

    train = examples(splits["train"])
    validation = examples(splits["validation"])
    test = examples(splits["test"])
    results = {}
    for variant in ("ordinary", "cycle", "none"):
        best = None
        for ridge in (0.01, 0.1, 1.0, 10.0):
            weights = fit_ridge(train[variant], train["labels"], ridge)
            validation_scores = scores(validation[variant], weights)
            candidates = np.quantile(validation_scores, np.linspace(0.05, 0.95, 37))
            for threshold in candidates:
                accuracy = balanced_accuracy(
                    validation["labels"],
                    (validation_scores >= threshold).astype(int))
                if best is None or accuracy > best[0]:
                    best = (accuracy, ridge, weights, float(threshold))
        _, ridge, weights, threshold = best
        predictions = predict(test[variant], weights, threshold)
        labels = test["labels"]
        results[variant] = {
            "ridge": ridge,
            "threshold": threshold,
            "accuracy": float(np.mean(predictions == labels)),
            "positive_accuracy": float(np.mean(predictions[labels == 1] == 1)),
            "negative_accuracy": float(np.mean(predictions[labels == 0] == 0)),
            "hallucination_rate": float(np.mean(predictions[labels == 0] == 1)),
            "n": len(labels),
        }

    report = {
        "schema": "nanlmind-hermes-joint-semantic-decoder/1",
        "noun_synsets": len(nodes),
        "eligible_parents": len(eligible),
        "splits": {name: len(value) for name, value in splits.items()},
        "feature_width": 64 + 256 + 64,
        "decoder": "ridge-linear, one shared family per variant",
        "seed": 20020,
        "maximum_parents": 5000,
        "results": results,
        "note": ("Novel parent labels are disjoint across splits. One deterministic "
                 "split and target sample is used. The task is a controlled yes/no "
                 "category question; the symbolic graph can answer it exactly, while "
                 "this decoder tests a trainable state path."),
    }
    json_path = output.with_name("joint-semantic-decoder.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# State-conditioned answer decoder",
        "",
        "A controlled yes/no question is asked about a novel term linked to a",
        "unique WordNet parent. Train/validation/test parents are disjoint.",
        "All variants use the same linear ridge decoder and feature width.",
        "",
        f"- noun synsets: **{len(nodes):,}**",
        f"- eligible parents: **{len(eligible):,}**",
        f"- test questions: **{results['ordinary']['n']:,}**",
        "",
        "| state | threshold | accuracy | positive | negative | hallucination |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("ordinary", "cycle", "none"):
        row = results[variant]
        lines.append(f"| {variant} | {row['threshold']:.3f} | {row['accuracy']:.4f} | "
                     f"{row['positive_accuracy']:.4f} | "
                     f"{row['negative_accuracy']:.4f} | "
                     f"{row['hallucination_rate']:.4f} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- The symbolic acquisition engine can answer this controlled task exactly;",
        "  this decoder is a learned control, not a language generator.",
        "- The cycle state is a projection of the same ancestry edges, not an",
        "  end-to-end trained semantic field.",
        "- Only one deterministic split and target sample was evaluated; the",
        "  differences are not accompanied by a multi-seed confidence interval.",
        "- No state/Cycle-specific advantage is assumed; the ordinary and no-state",
        "  rows are controls.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
