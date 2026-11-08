#!/usr/bin/env python3
"""Head-population diagnostics: routing sparsity, void head, specialization."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from nanlmind_hermes.head_population import HeadPopulation, fnv  # noqa: E402


def main():
    from dictionary_completion import load_lemmas  # noqa: E402

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/HEAD_POPULATION.md"
    words, _frequency = load_lemmas()
    train = [word for word in words if fnv(word) % 5 != 0][:1200]
    test = [word for word in words if fnv(word) % 5 == 0][:400]

    configs = [
        {"active": 8, "context_length": 2, "balance": False, "decorrelate": 0.0},
        {"active": 8, "context_length": 3, "balance": False, "decorrelate": 0.0},
        {"active": 8, "context_length": 3, "balance": True, "decorrelate": 0.0},
        {"active": 8, "context_length": 4, "balance": True, "decorrelate": 0.0},
        {"active": 16, "context_length": 3, "balance": True, "decorrelate": 0.0},
        {"active": 8, "context_length": 3, "balance": True, "decorrelate": 0.1},
        {"active": 255, "context_length": 2, "balance": False, "decorrelate": 0.0},
    ]
    rows = []
    for config in configs:
        population = HeadPopulation(
            heads=256, active=config["active"], prior_weight=0.0,
            train_head0=False, context_length=config["context_length"],
            balance=config["balance"])
        for word in train:
            population.observe(word)
        if config["decorrelate"]:
            population.decorrelate(config["decorrelate"])
        result = population.evaluate(test)
        vectors = np.array(population.vectors())
        norms = np.linalg.norm(vectors, axis=1)
        valid = vectors[norms > 0]
        if len(valid):
            normalized = valid / np.maximum(vectors[norms > 0].sum(axis=1, keepdims=True), 1e-12)
            gram = normalized @ normalized.T
            eigenvalues = np.linalg.eigvalsh(gram)
            eigenvalues = eigenvalues[eigenvalues > 1e-8]
            rank = int(len(eigenvalues))
            probabilities = eigenvalues / eigenvalues.sum()
            effective = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
            mean_cosine = float((gram.sum() - np.trace(gram)) /
                                max(1, len(gram) * (len(gram) - 1)))
        else:
            rank = 0
            effective = 0.0
            mean_cosine = 0.0
        diversity = population.diversity()
        rows.append({
            **config,
            "accuracy": result["accuracy"],
            "top3": result["top3"],
            "heads_with_distribution": sum(1 for head in population.heads if head),
            "rank": rank,
            "effective_rank": round(effective, 4),
            "mean_cosine": round(mean_cosine, 6),
            "mean_entropy": diversity["mean_entropy"],
        })

    report = {"schema": "nanlmind-hermes-head-diagnostics/1",
              "train": len(train), "test": len(test), "configs": rows}
    json_path = output.with_name("head-population.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Head-population diagnostics: routing, void head, specialization",
        "",
        "The head Gram rank is the key measure: rank 1 means every head carries",
        "the same distribution, so the spectral operator has only one mode.",
        "",
        "| active | context | balance | decorrelate | accuracy | top-3 | rank | effective rank | mean cosine |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['active']} | {row['context_length']} | {row['balance']} | "
                     f"{row['decorrelate']} | {row['accuracy']} | {row['top3']} | "
                     f"{row['rank']} | {row['effective_rank']} | {row['mean_cosine']} |")
    lines += [
        "",
        "## Reading",
        "",
        "- `active=255` trains every head on every context, so all heads converge",
        "  to the global letter distribution (rank 1). That was the bottleneck.",
        "- Sparse routing keeps all 256 heads available over the corpus while only",
        "  a subset fires per context, which is what creates specialization.",
        "- Head 0 should be the void/prior; training it or letting it dominate",
        "  the collapse destroys the distinction between void and content.",
        "- Higher rank/effective rank and lower mean cosine are the targets.",
        "- Longer contexts and load-balanced routing are expected to raise rank and",
        "  accuracy; decorrelation removes the shared global component.",
        "",
        "## What to do next",
        "",
        "1. Keep all 256 heads, but choose active heads sparsely per context.",
        "2. Freeze head 0 as the void/prior; do not train it on the corpus.",
        "3. Add load-balancing and a decorrelation term to the head updates.",
        "4. Extend context length to 3-4 letters for more distinct routes.",
        "5. Rebuild the spectral A from a diversified population and re-measure",
        "   the low-mode/cut leakage.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "configs": rows}, indent=2))


if __name__ == "__main__":
    main()
