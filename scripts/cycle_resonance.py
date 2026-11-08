#!/usr/bin/env python3
"""Cycle space + nonuniform mirror-pair resonance experiment (numpy)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nanlmind_hermes.cycle_space import (  # noqa: E402
    cycle_dimension,
    edge_count,
    fundamental_fluxes,
    triangle_count,
)


def reverse8(value: int) -> int:
    return int(format(value & 0xFF, "08b")[::-1], 2)


def coherent_field(n=256):
    mirror = np.array([reverse8(index) for index in range(n)])
    amplitude = np.zeros((n, n))
    phase = np.zeros((n, n))
    for first in range(n):
        for second in range(first + 1, n):
            amplitude[first, second] = (1.0 + 0.5 * np.sin(2 * np.pi * (first + second) / n)
                                        + 0.3 * np.cos(2 * np.pi * first * second / (n * n)))
            phase[first, second] = 1.1 * np.sin(2 * np.pi * first / n + 3 * second / n)
    field = np.eye(n, dtype=complex)
    for first in range(n):
        for second in range(first + 1, n):
            value = amplitude[first, second] * np.exp(1j * phase[first, second])
            field[first, second] = value
            field[second, first] = np.conj(value)
    # Exact mirror projection: Z[i,j] = conj(Z[rev(i),rev(j)]).
    field = 0.5 * (field + np.conj(field[mirror][:, mirror]))
    field = 0.5 * (field + field.conj().T)
    return field, mirror

def uniform_field(n=256):
    field = np.ones((n, n), dtype=complex)
    np.fill_diagonal(field, 1.0)
    return field


def analyse(field, mirror):
    n = len(field)
    upper = [(first, second) for first in range(n)
             for second in range(first + 1, n)]
    values = {(first, second): field[first, second] for first, second in upper}
    # Orthogonal gradient/cycle split.
    divergence = np.zeros(n, dtype=complex)
    for (first, second), value in values.items():
        divergence[first] -= value
        divergence[second] += value
    potential = divergence / n
    gradient = np.zeros_like(field)
    cycle = np.zeros_like(field)
    for first, second in upper:
        grad = potential[second] - potential[first]
        gradient[first, second] = grad
        gradient[second, first] = -grad
        cyc = values[(first, second)] - grad
        cycle[first, second] = cyc
        cycle[second, first] = -cyc
    cycle_divergence = np.zeros(n, dtype=complex)
    for first, second in upper:
        value = cycle[first, second]
        cycle_divergence[first] -= value
        cycle_divergence[second] += value
    orthogonality = abs(sum((gradient[first, second]
                             * np.conj(cycle[first, second])).real
                            for first, second in upper))
    reconstruction = max(abs((gradient[first, second] + cycle[first, second])
                             - values[(first, second)])
                         for first, second in upper)

    fundamental = fundamental_fluxes(n, values)
    fluxes = np.array(list(fundamental.values()))
    magnitudes = np.abs(fluxes)
    coherence = np.abs(fluxes.real) / np.maximum(magnitudes, 1e-12)
    eigenvalues = np.linalg.eigvalsh(field)
    order = np.argsort(-eigenvalues)[:8]
    eigenvectors = np.linalg.eigh(field)[1][:, order]
    ipr = []
    for column in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, column]
        power = np.abs(vector) ** 2
        ipr.append(float((power ** 2).sum() / max(1e-12, power.sum() ** 2)))
    phase_order = float(abs(np.mean(np.exp(1j * np.angle(fluxes)))))
    mirror_residual = max(abs(field[first, second]
                              - np.conj(field[mirror[first], mirror[second]]))
                          for first, second in upper)
    return {
        "max_cycle_divergence": float(np.max(np.abs(cycle_divergence))),
        "gradient_cycle_orthogonality": float(orthogonality),
        "reconstruction_error": float(reconstruction),
        "fundamental_fluxes": len(fundamental),
        "triangle_coherence_mean": float(coherence.mean()),
        "triangle_coherence_p90": float(np.percentile(coherence, 90)),
        "triangle_coherence_fraction_above_half": float((coherence > 0.5).mean()),
        "top_eigenvalues": [float(value) for value in eigenvalues[order]],
        "spectral_gap": float(eigenvalues[order[0]] - eigenvalues[order[1]]),
        "top_ipr": [round(value, 6) for value in ipr[:4]],
        "phase_locking_order": phase_order,
        "mirror_residual": float(mirror_residual),
        "cycle_norm": float(np.linalg.norm(cycle)),
        "gradient_norm": float(np.linalg.norm(gradient)),
    }


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/CYCLE_SPACE.md"
    n = 256
    coherent, mirror = coherent_field(n)
    uniform = uniform_field(n)
    nonuniform = analyse(coherent, mirror)
    baseline = analyse(uniform, np.arange(n))

    # Energy conservation of the two-sector standing-wave model.
    upper = [(first, second) for first in range(n)
             for second in range(first + 1, n)]
    values = {(first, second): coherent[first, second] for first, second in upper}
    divergence = np.zeros(n, dtype=complex)
    for (first, second), value in values.items():
        divergence[first] -= value
        divergence[second] += value
    potential = divergence / n
    gradient = np.array([[potential[second] - potential[first]
                          if first < second else
                          (-(potential[first] - potential[second]) if first > second else 0)
                          for second in range(n)] for first in range(n)])
    cycle = coherent - gradient
    energies = []
    for time in (0.0, 0.2, 0.7, 1.3, 2.5, 5.0):
        grad = gradient * np.cos(2.0 * time)
        cyc = cycle * np.cos(3.0 * time)
        grad_v = -2.0 * gradient * np.sin(2.0 * time)
        cyc_v = -3.0 * cycle * np.sin(3.0 * time)
        kinetic = 0.5 * (np.linalg.norm(grad_v) ** 2 + np.linalg.norm(cyc_v) ** 2)
        potential_energy = 0.5 * (4.0 * np.linalg.norm(grad) ** 2
                                  + 9.0 * np.linalg.norm(cyc) ** 2)
        energies.append(float(kinetic + potential_energy))
    energy_error = max(abs(value - energies[0]) / max(1e-12, energies[0])
                       for value in energies)

    report = {
        "schema": "nanlmind-hermes-cycle-space/1",
        "n": n,
        "edges": edge_count(n),
        "cycle_dimension": cycle_dimension(n),
        "triangles": triangle_count(n),
        "fundamental_cycles": (n - 1) * (n - 2) // 2,
        "nonuniform": nonuniform,
        "uniform": baseline,
        "energy_relative_error": energy_error,
        "note": ("classical cycle/Hodge and resonance math; not quantum hardware"),
    }
    json_path = output.with_name("cycle-resonance.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Cycle space and nonuniform mirror-pair resonance",
        "",
        "Complete 256-node graph: 32,640 edges; gradient dimension 255; cycle",
        "dimension 32,385; triangles 2,763,520; independent fundamental fluxes",
        "32,385. Classically verified, not quantum hardware.",
        "",
        "| measure | uniform | nonuniform mirror |",
        "|---|---:|---:|",
    ]
    for key in ("max_cycle_divergence", "gradient_cycle_orthogonality",
                "reconstruction_error", "triangle_coherence_mean",
                "triangle_coherence_fraction_above_half", "phase_locking_order",
                "spectral_gap", "mirror_residual", "cycle_norm"):
        lines.append(f"| {key} | {baseline[key]} | {nonuniform[key]} |")
    lines += [
        "",
        f"- nonuniform top eigenvalues: {nonuniform['top_eigenvalues']}",
        f"- uniform top eigenvalues: {baseline['top_eigenvalues']}",
        f"- nonuniform top IPR: {nonuniform['top_ipr']}",
        f"- uniform top IPR: {baseline['top_ipr']}",
        f"- energy relative error: {energy_error}",
        "",
        "## Reading",
        "",
        "- Gradient/cycle orthogonality and reconstruction check the Hodge split.",
        "- Nonuniform mirror couplings change the resonance spectrum, localization",
        "  (IPR), phase-locking order, and triangle coherence relative to uniform.",
        "- Magnitudes alone do not determine triangle coherence; the phase field",
        "  does. This is a relational-phase measurement, not quantum evidence.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "nonuniform": nonuniform, "uniform": baseline,
                      "energy_relative_error": energy_error}, indent=2))


if __name__ == "__main__":
    main()
