#!/usr/bin/env python3
"""Small deterministic gauge-corrected cycle-code sweep (classical linear code)."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.cycle_codec import (  # noqa: E402
    decode_cycle,
    encode,
    gauge_values,
    joint_decode,
    random_cycle_basis,
)


def cosine(first, second):
    return (sum(a * b for a, b in zip(first, second))
            / (math.sqrt(sum(a * a for a in first)
                         * sum(b * b for b in second)) + 1e-12))


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/CYCLE_CODEC.md"
    n = 32
    dimensions = 4
    queries = 6
    columns, rows = random_cycle_basis(n, dimensions, seed=13013)
    edge_count = len(rows)
    rng = random.Random(13)
    results = []
    for erasure in (0.0, 0.5, 0.9):
        for spike in (0.0, 0.1, 0.4):
            gauge_cosines = []
            raw_cosines = []
            joint_cosines = []
            for _ in range(queries):
                z = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
                cycle = encode(rows, z)
                gauge = gauge_values(n, [rng.gauss(0.0, 2.0) for _ in range(n)])
                observed = [a + b for a, b in zip(cycle, gauge)]
                mask = [rng.random() >= erasure for _ in range(edge_count)]
                for edge in range(edge_count):
                    if mask[edge] and rng.random() < spike:
                        observed[edge] += rng.gauss(0.0, 4.0)
                decoded = decode_cycle(n, columns, observed, mask)
                observed_fraction = sum(mask) / edge_count
                uncorrected = [
                    sum(column[edge] * observed[edge]
                        for edge in range(edge_count) if mask[edge])
                    / observed_fraction
                    for column in columns
                ]
                gauge_cosines.append(cosine(decoded, z))
                raw_cosines.append(cosine(uncorrected, z))
                joint, _potential = joint_decode(
                    n, columns, observed, mask, iterations=5)
                joint_cosines.append(cosine(joint, z))
            results.append({
                "nodes": n,
                "embedding_dim": dimensions,
                "erasure": erasure,
                "spike_rate": spike,
                "queries": queries,
                "gauge_fit_cosine": sum(gauge_cosines) / queries,
                "joint_robust_cosine": sum(joint_cosines) / queries,
                "uncorrected_cosine": sum(raw_cosines) / queries,
            })
    report = {
        "schema": "nanlmind-hermes-cycle-codec/1",
        "note": ("random classical cycle code over K64; semantic vectors are "
                 "synthetic, gauge fitting is least squares and not robust to "
                 "large spikes"),
        "results": results,
    }
    json_path = output.with_name("cycle-codec.json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gauge-corrected cycle code",
        "",
        "A synthetic semantic vector is encoded into an orthonormal divergence-free",
        "cycle basis, then an unknown vertex gauge, edge erasure, and large spikes",
        "are applied. This is a classical linear code.",
        "",
        "| erasure | spikes | gauge cosine | joint robust | uncorrected |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(f"| {row['erasure']:.0%} | {row['spike_rate']:.0%} | "
                     f"{row['gauge_fit_cosine']:.4f} | "
                     f"{row['joint_robust_cosine']:.4f} | "
                     f"{row['uncorrected_cosine']:.4f} |")
    lines += [
        "",
        "## Boundaries",
        "",
        "- Gauge fitting recovers the encoded vector under erasure, but the",
        "  plain least-squares fit degrades under large independent spikes.",
        "- The joint robust decoder models cycle coefficients and gauge together",
        "  with Huber reweighting and recovers much more under spikes in this",
        "  synthetic test; this is still not a learned semantic encoder.",
        "- The random codebook is not a learned semantic encoder and adds no",
        "  information; equivalent real linear algebra is sufficient.",
        "- No quantum hardware or physical quantum effect is involved.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
