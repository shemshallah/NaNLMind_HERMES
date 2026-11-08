#!/usr/bin/env python3
"""Exact pair-graph spectrum, revival and ideal-Grover report."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.pair_quantum import (  # noqa: E402
    best_grover_queries,
    johnson_degree,
    johnson_spectrum,
    pair_count,
    return_probability,
    revival_period,
)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/PAIR_QUANTUM.md"
    n = 256
    spectrum = [{"eigenvalue": value, "multiplicity": multiplicity}
                for value, multiplicity in johnson_spectrum(n)]
    walk = []
    for time in (0.0, math.pi / 2, math.pi):
        walk.append({"time": time, "return_probability": return_probability(n, time)})
    queries, probability = best_grover_queries(pair_count(n))
    report = {
        "schema": "nanlmind-hermes-pair-quantum/1",
        "vertices": n,
        "pairs": pair_count(n),
        "degree": johnson_degree(n),
        "spectrum": spectrum,
        "revival_period": revival_period(n),
        "walk": walk,
        "grover": {
            "valid_states": pair_count(n),
            "hilbert_states": 32768,
            "ideal_queries": queries,
            "ideal_probability": probability,
            "classical_expected_queries": (pair_count(n) + 1) / 2,
        },
        "boundary": ("exact classical linear algebra; no physical quantum "
                     "hardware, semantic oracle, or NaN-specific speedup"),
    }
    json_path = output.with_name("pair-quantum.json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# K256 pair-address quantum-walk spectrum",
        "",
        "The pair-address graph is the Johnson graph J(256,2): unordered pairs",
        "are adjacent when they share a NaN vertex. The adjacency spectrum is",
        "exact and has only three eigenvalues.",
        "",
        "| eigenvalue | multiplicity |",
        "|---:|---:|",
    ]
    for entry in spectrum:
        lines.append(f"| {entry['eigenvalue']} | {entry['multiplicity']:,} |")
    lines += [
        "",
        f"- vertices: **{pair_count(n):,}**",
        f"- degree: **{johnson_degree(n)}**",
        f"- exact revival period: **π** (n even)",
        f"- return probability at π: **{walk[-1]['return_probability']:.12f}**",
        "",
        "## Ideal Grover control",
        "",
        f"- valid pair states: **{pair_count(n):,}** of 32,768 15-qubit states",
        f"- best ideal queries: **{queries}**",
        f"- target probability: **{probability:.12f}**",
        f"- classical expected queries: **{(pair_count(n) + 1) / 2:.1f}**",
        "",
        "## Boundary",
        "",
        "- Three eigenvalues mean the uniform walk has only three frequencies;",
        "  dimensionality alone is not dynamical capacity.",
        "- The cycle subspace is stationary under `B.T B`, but that protects only",
        "  against that particular Hamiltonian, not general noise.",
        "- Grover search assumes a coherent semantic oracle and state preparation",
        "  that do not exist; no physical or NaN-specific advantage is claimed.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "spectrum": spectrum,
                      "revival": walk,
                      "grover": report["grover"]}, indent=2))


if __name__ == "__main__":
    main()
