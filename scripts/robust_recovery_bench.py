#!/usr/bin/env python3
"""Reproduce the Round 012 gauge-flat robust recovery audit (classical)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def trial(n: int, fraction: float, seed: int, noise_sigma: float = 0.0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-2.0, 2.0, n)
    truth = theta[None, :] - theta[:, None]
    ii, jj = np.triu_indices(n, 1)
    edge_count = len(ii)
    observed = truth.copy()
    corrupted = rng.choice(edge_count, round(edge_count * fraction), replace=False)
    amplitude = rng.choice([-1.0, 1.0], len(corrupted)) * rng.uniform(1.0, 4.0, len(corrupted))
    observed[ii[corrupted], jj[corrupted]] += amplitude
    observed[jj[corrupted], ii[corrupted]] -= amplitude
    if noise_sigma:
        noise = rng.normal(0.0, noise_sigma, edge_count)
        observed[ii, jj] += noise
        observed[jj, ii] -= noise

    median_field = np.empty((n, n))
    mean_field = np.empty((n, n))
    for first in range(n):
        paths = observed[first, :, None] + observed
        median_field[first, :] = np.median(paths, axis=0)
        mean_field[first, :] = np.mean(paths, axis=0)
    median_field = (median_field - median_field.T) / 2.0
    mean_field = (mean_field - mean_field.T) / 2.0

    true_edges = truth[ii, jj]
    return {
        "n": n,
        "fraction": fraction,
        "seed": seed,
        "noise_sigma": noise_sigma,
        "raw_mae": float(np.mean(np.abs(observed[ii, jj] - true_edges))),
        "mean_mae": float(np.mean(np.abs(mean_field[ii, jj] - true_edges))),
        "median_mae": float(np.mean(np.abs(median_field[ii, jj] - true_edges))),
    }


def nonflat(n: int, seed: int):
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(n, 1)
    values = rng.normal(0.0, 1.0, len(ii))
    field = np.zeros((n, n))
    field[ii, jj] = values
    field[jj, ii] = -values
    median_field = np.empty((n, n))
    for first in range(n):
        median_field[first, :] = np.median(field[first, :, None] + field, axis=0)
    median_field = (median_field - median_field.T) / 2.0
    return {
        "n": n,
        "original_std": float(np.std(values)),
        "median_mae": float(np.mean(np.abs(median_field[ii, jj] - values))),
    }


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/ROBUST_RECOVERY.md"
    fractions = (0.1, 0.2, 0.4, 0.6, 0.8)
    rows = []
    for fraction in fractions:
        runs = [trial(256, fraction, 12000 + int(fraction * 1000) + seed)
                for seed in range(3)]
        rows.append({
            "fraction": fraction,
            "raw_mae": float(np.mean([run["raw_mae"] for run in runs])),
            "mean_mae": float(np.mean([run["mean_mae"] for run in runs])),
            "median_mae": float(np.mean([run["median_mae"] for run in runs])),
        })
    noisy = [trial(256, 0.1, 8700 + seed, noise_sigma=0.05) for seed in range(3)]
    report = {
        "schema": "nanlmind-hermes-robust-recovery/1",
        "method": ("gauge-flat 256-node field; two-hop median vs mean and raw; "
                   "classical robust estimation, not quantum error correction"),
        "aggregate": rows,
        "noise_sigma_0_05_fraction_0_1": {
            "raw_mae": float(np.mean([run["raw_mae"] for run in noisy])),
            "mean_mae": float(np.mean([run["mean_mae"] for run in noisy])),
            "median_mae": float(np.mean([run["median_mae"] for run in noisy])),
        },
        "nonflat_control": nonflat(256, 411),
        "limitations": [
            "A gauge-flat field has only 255 independent dimensions.",
            "The median decoder destroys genuinely non-flat cycle fields.",
            "No quantum hardware or physical quantum effect is involved.",
            "Detection/precision at a known corruption budget would be an oracle-budget experiment, not blind localization.",
        ],
    }
    json_path = output.with_name("robust-recovery.json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Robust gauge-flat recovery",
        "",
        "Two-hop median reconstruction of a 256-node gauge-flat field under",
        "independently corrupted edges. This is classical robust estimation.",
        "",
        "| corrupted edges | raw MAE | mean MAE | median MAE |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['fraction']:.0%} | {row['raw_mae']:.4f} | "
                     f"{row['mean_mae']:.4f} | {row['median_mae']:.4f} |")
    lines += [
        "",
        f"With Gaussian noise σ=0.05 and 10% corruption: raw "
        f"{report['noise_sigma_0_05_fraction_0_1']['raw_mae']:.4f}, mean "
        f"{report['noise_sigma_0_05_fraction_0_1']['mean_mae']:.4f}, median "
        f"{report['noise_sigma_0_05_fraction_0_1']['median_mae']:.4f}.",
        "",
        f"Non-flat control at n=256: median MAE "
        f"{report['nonflat_control']['median_mae']:.4f} — the filter fails, as it must.",
        "",
        "## Boundary",
        "",
        "- The flat field has 255 independent gradient dimensions; it is not the",
        "  32,385-dimensional cycle memory.",
        "- The decoder assumes zero curvature. It does not recover learned semantic",
        "  cycle states.",
        "- No physical quantum effect is claimed.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "json": str(json_path),
                      "aggregate": rows,
                      "noise": report["noise_sigma_0_05_fraction_0_1"],
                      "nonflat": report["nonflat_control"]}, indent=2))


if __name__ == "__main__":
    main()
