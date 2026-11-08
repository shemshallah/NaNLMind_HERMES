#!/usr/bin/env python3
"""Build the 256x256 wave operator from the head population and measure it.

- A is the head-to-head cosine Gram matrix of letter distributions.
- Centrosymmetrize via the reversal J, then fold to two 128x128 blocks.
- Add the inverse-square void potential on a 16x16 head grid.
- Compare full/folded eigenvalues, eigenvector regions, and spectral leakage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402  (experiment-only reference)

from nanlmind_hermes.head_population import HeadPopulation  # noqa: E402
from nanlmind_hermes.spectral import (  # noqa: E402
    WaveMatrix,
    graph_laplacian,
    head_region,
    mode_regions,
    power_iteration,
    smallest_eigen,
    void_potential,
)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/SPECTRAL_WAVE.md"
    sys.path.insert(0, str(ROOT / "scripts"))
    from dictionary_completion import load_lemmas  # noqa: E402

    words, _frequency = load_lemmas()
    train = words[:1500]
    population = HeadPopulation(heads=256, active=8, prior_weight=0.0,
                                train_head0=False)
    for word in train:
        population.observe(word)

    wave = WaveMatrix.from_head_population(population).symmetrize().centrosymmetrize()
    base_values, base_vectors = power_iteration(wave.matrix, k=4, iterations=80)
    reference = np.linalg.eigvalsh(np.array(wave.matrix))[-4:][::-1].tolist()

    P, M = wave.fold()
    p_values, _ = power_iteration(P.matrix, k=4, iterations=80)
    m_values, _ = power_iteration(M.matrix, k=4, iterations=80)
    folded_values = sorted(p_values + m_values, reverse=True)[:4]

    base_laplacian = graph_laplacian(wave.matrix)
    low_values, low_vectors = smallest_eigen(base_laplacian, k=4, iterations=80)
    void_laplacian = void_potential(base_laplacian, lam=1.5)
    void_low_values, void_low_vectors = smallest_eigen(void_laplacian, k=4, iterations=80)
    void = wave.apply_void(lam=1.5).centrosymmetrize()
    void_values, void_vectors = power_iteration(void.matrix, k=4, iterations=80)

    def regions(vectors):
        rows = [mode_regions(vector) for vector in vectors]
        average = {key: round(sum(row[key] for row in rows) / len(rows), 6)
                   for key in ("inner", "ring", "outer")}
        return {"modes": rows, "average": average}

    base_regions = regions(base_vectors)
    void_regions = regions(void_vectors)
    low_regions = regions(low_vectors)
    void_low_regions = regions(void_low_vectors)
    low_nonconstant = {key: round(sum(row[key] for row in low_regions["modes"][1:])
                                  / max(1, len(low_regions["modes"]) - 1), 6)
                       for key in ("inner", "ring", "outer")}
    void_low_nonconstant = {key: round(sum(row[key] for row in void_low_regions["modes"][1:])
                                       / max(1, len(void_low_regions["modes"]) - 1), 6)
                            for key in ("inner", "ring", "outer")}
    rank = int(np.linalg.matrix_rank(np.array(wave.matrix)))

    def top_heads(vector, count=6):
        ranked = sorted(range(len(vector)), key=lambda head: -abs(vector[head]))[:count]
        result = []
        for head in ranked:
            letters = population.heads[head].most_common(3)
            result.append({
                "head": head,
                "amplitude": round(vector[head], 6),
                "region": head_region(head),
                "letters": [[letter, count] for letter, count in letters],
            })
        return result

    report = {
        "schema": "nanlmind-hermes-spectral-wave/1",
        "training_words": len(train),
        "heads": population.head_count,
        "status": wave.status(),
        "base_eigenvalues": [round(value, 6) for value in base_values],
        "numpy_reference": [round(value, 6) for value in reference],
        "eigen_reference_error": round(max(abs(a - b) for a, b in
                                           zip(sorted(base_values, reverse=True),
                                               sorted(reference, reverse=True))), 8),
        "folded_eigenvalues": [round(value, 6) for value in folded_values],
        "fold_error": round(max(abs(a - b) for a, b in
                                zip(sorted(base_values, reverse=True),
                                    sorted(folded_values, reverse=True))), 8),
        "base_regions": base_regions,
        "void_eigenvalues": [round(value, 6) for value in void_values],
        "void_regions": void_regions,
        "void_shift": [round(a - b, 6) for a, b in zip(base_values, void_values)],
        "rank": rank,
        "laplacian_low_eigenvalues": [round(value, 6) for value in low_values],
        "void_laplacian_low_eigenvalues": [round(value, 6) for value in void_low_values],
        "laplacian_low_regions": low_regions,
        "void_laplacian_low_regions": void_low_regions,
        "laplacian_low_nonconstant": low_nonconstant,
        "void_laplacian_low_nonconstant": void_low_nonconstant,
        "tunneling_reduction": round(1.0 - void_low_nonconstant["inner"]
                                     / low_nonconstant["inner"], 4)
        if low_nonconstant["inner"] else 0.0,
        "top_modes": [{"mode": index, "heads": top_heads(vector)}
                      for index, vector in enumerate(base_vectors[:3])],
        "centrosymmetry_residual": wave.centrosymmetry_residual(),
        "symmetry_residual": wave.symmetry_residual(),
        "note": ("tunneling is spectral leakage: the inner-region mass of an "
                 "eigenmode; it is not a quantum claim"),
    }
    json_path = output.with_name("spectral-wave.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Spectral wave operator over the 256x256 address space",
        "",
        "Built from the head population's letter distributions: A[h1][h2] is",
        "the cosine Gram weight. The matrix is symmetrized, centrosymmetrized",
        "with the reversal J, folded into two 128x128 blocks, and given the",
        "inverse-square void potential. Head 0 is the void/NaNNet anchor.",
        "",
        f"- training words: {len(train):,}",
        f"- matrix: {report['status']['size']}×{report['status']['size']}",
        f"- symmetric: {report['status']['symmetric']}",
        f"- centrosymmetric: {report['status']['centrosymmetric']}",
        f"- eigen reference error: {report['eigen_reference_error']}",
        f"- fold error: {report['fold_error']}",
        "",
        "## Top eigenvalues",
        "",
        "```json",
        json.dumps({"base": report["base_eigenvalues"],
                    "folded": report["folded_eigenvalues"],
                    "numpy": report["numpy_reference"],
                    "void": report["void_eigenvalues"]}, indent=2),
        "```",
        "",
        "## Regions and spectral leakage",
        "",
        "```json",
        json.dumps({"base_coupling": base_regions["average"],
                    "voided_coupling": void_regions["average"],
                    "laplacian_low": low_regions["average"],
                    "void_laplacian_low": void_low_regions["average"],
                    "rank": rank,
                    "void_shift": report["void_shift"]}, indent=2),
        "```",
        "",
        "Tunneling = inner-region mass. A nonzero value is spectral leakage",
        "through the void barrier, not a quantum measurement.",
        "",
        "## Principal modes (top heads)",
        "",
    ]
    for mode in report["top_modes"]:
        lines.append(f"### mode {mode['mode']}")
        lines.append("```json")
        lines.append(json.dumps(mode["heads"], indent=2)[:3000])
        lines.append("```")
    lines += [
        "",
        "## Reading",
        "",
        "- The full 256×256 spectrum and the two 128×128 folded spectra agree,",
        "  confirming the centrosymmetric/palindrome structure.",
        "- The coupling matrix is rank-limited; the graph Laplacian low modes",
        "  are the correct wave operator. The void potential on the Laplacian",
        "  should suppress inner-region mass; the remaining inner mass is",
        "  spectral leakage (tunneling).",
        "- A positive diagonal on the coupling matrix attracts the top mode; it",
        "  is not the barrier. Use L = D - A plus V and low modes.",
        "- The head population is the current coarse A; the next step is to use",
        "  the eigenmodes P and strengths Λ as the MTH L7 relation basis.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "json": str(json_path),
        "base": report["base_eigenvalues"], "folded": report["folded_eigenvalues"],
        "fold_error": report["fold_error"],
        "base_regions": base_regions["average"],
        "void_regions": void_regions["average"],
    }, indent=2))


if __name__ == "__main__":
    main()
