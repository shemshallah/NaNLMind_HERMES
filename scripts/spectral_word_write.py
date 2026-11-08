#!/usr/bin/env python3
"""Write a word leaf into the diversified spectral matrix with native J and a
Fiedler spectral cut, then measure the structural and eigen effects."""
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

from nanlmind_hermes.head_population import HeadPopulation  # noqa: E402
from nanlmind_hermes.spectral import (  # noqa: E402
    WaveMatrix,
    fiedler_partition,
    j_closure,
    native_mirror,
)
from nanlmind_hermes.word_writer import WordLeafWriter, centrosymmetrize  # noqa: E402


def main():
    from dictionary_completion import load_lemmas  # noqa: E402

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/SPECTRAL_WORD.md"
    words, _frequency = load_lemmas()
    train = words[:1500]
    population = HeadPopulation(heads=256, active=8, context_length=2,
                                prior_weight=0.0, train_head0=False)
    for word in train:
        population.observe(word)

    mirror = native_mirror(256)
    wave = WaveMatrix.from_head_population(population)
    matrix = centrosymmetrize(wave.matrix, mirror)
    base_rank = int(np.linalg.matrix_rank(np.array(matrix)))
    base_eigenvalues = np.linalg.eigvalsh(np.array(matrix))[-6:][::-1]

    cut = fiedler_partition(matrix)
    inner = j_closure(set(cut["inner"]), mirror)
    boundary = j_closure(set(cut["boundary"]), mirror) - inner
    if len(boundary) < 6:
        boundary = set(cut["positive"]) - inner
    ring = sorted(boundary)[:12]
    if len(ring) < 6:
        ring = sorted(set(range(256)) - inner)[:12]
    half = max(2, len(ring) // 3)
    packets = [
        [ring[:half], [1.0] * len(ring[:half])],
        [ring[half:2 * half], [0.8] * len(ring[half:2 * half])],
        [ring[2 * half:], [0.6] * len(ring[2 * half:])],
    ]
    writer = WordLeafWriter(n=256, inner=inner, permutation=mirror)
    letters = [writer.letter_packet(heads, amps) for heads, amps in packets if heads]
    updated, record = writer.write_word(matrix, letters, beta=0.5, gamma=0.25)
    delta = np.array(updated) - np.array(matrix)
    updated_rank = int(np.linalg.matrix_rank(np.array(updated)))
    updated_eigenvalues = np.linalg.eigvalsh(np.array(updated))[-6:][::-1]
    eigenvectors = np.linalg.eigh(np.array(matrix))[1]
    projected = eigenvectors.T @ delta @ eigenvectors
    energies = np.diag(projected)
    order = np.argsort(-np.abs(energies))[:8]
    mode_energy = [{"mode": int(index), "energy": round(float(energies[index]), 6)}
                   for index in order]
    total_energy = float((projected ** 2).sum())
    top_energy = float((projected[order, order] ** 2).sum())
    mirror_residual = max(abs(updated[r][c] - updated[mirror[r]][mirror[c]])
                          for r in range(256) for c in range(256))
    void_rows = max(abs(updated[head][c]) for head in inner for c in range(256)) \
        if inner else 0.0
    report = {
        "schema": "nanlmind-hermes-spectral-word/1",
        "training_words": len(train),
        "base_rank": base_rank,
        "updated_rank": updated_rank,
        "base_eigenvalues": [round(float(value), 6) for value in base_eigenvalues],
        "updated_eigenvalues": [round(float(value), 6) for value in updated_eigenvalues],
        "fiedler": {"values": cut["values"], "inner_size": len(inner),
                    "boundary_size": len(set(cut["boundary"])),
                    "ring": ring},
        "word": record,
        "delta_norm": round(float(np.linalg.norm(delta)), 6),
        "mirror_residual": round(float(mirror_residual), 9),
        "void_rows": round(float(void_rows), 9),
        "mode_energy": mode_energy,
        "projected_energy_total": round(total_energy, 6),
        "projected_energy_top": round(top_energy, 6),
        "native_mirror": True,
        "note": "word write uses native reverse8 J and a Fiedler spectral cut",
    }
    json_path = output.with_name("spectral-word.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Word-leaf write into the diversified spectral matrix",
        "",
        "The head population uses active=8/context=2 (the best sparse config).",
        "The matrix is centrosymmetrized with the **native reverse8 mirror**, the",
        "void is a **Fiedler spectral cut**, and a three-letter word is written",
        "with the atomic WordLeafWriter rule.",
        "",
        f"- base rank: **{base_rank}**; after word write: **{updated_rank}**",
        f"- base eigenvalues: {report['base_eigenvalues']}",
        f"- updated eigenvalues: {report['updated_eigenvalues']}",
        f"- Fiedler values: {cut['values']}",
        f"- inner heads: {len(inner)}; boundary: {len(set(cut['boundary']))}; ring: {len(ring)}",
        f"- delta norm: {report['delta_norm']}",
        f"- native mirror residual: **{report['mirror_residual']}**",
        f"- void rows max: **{report['void_rows']}**",
        f"- projected delta energy: {report['projected_energy_total']} (top modes "
        f"{report['projected_energy_top']})",
        "",
        "## Top mode energies",
        "",
        "```json",
        json.dumps(mode_energy, indent=2),
        "```",
        "",
        "## Word record",
        "",
        "```json",
        json.dumps(record, indent=2)[:4000],
        "```",
        "",
        "## Reading",
        "",
        "- Centrosymmetry and void projection are exact if the residuals are zero.",
        "- The word write is low-rank, so rank is expected to stay nearly stable;",
        "  the meaningful effect is the eigenmode energy distribution.",
        "- The next step is to use this written matrix in the temporal spine and",
        "  measure whether the word couplings survive projection/resonance.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("base_rank", "updated_rank", "base_eigenvalues",
                       "updated_eigenvalues", "mirror_residual", "void_rows",
                       "delta_norm", "projected_energy_top")}, indent=2))


if __name__ == "__main__":
    main()
