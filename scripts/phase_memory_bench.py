#!/usr/bin/env python3
"""Small deterministic phase-memory noise sweep for the Hermes docs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.phase_memory import evaluate  # noqa: E402


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/phase-memory.json"
    rows = []
    for dimensions in (16, 32, 64):
        for noise in (0.25, 1.0, 2.0):
            runs = [evaluate(dimensions, 64, 60, seed=seed, noise_std=noise)
                    for seed in (1, 2, 3)]
            rows.append({
                "dimensions": dimensions,
                "patterns": 64,
                "queries_per_seed": 60,
                "seeds": [1, 2, 3],
                "noise_std_radians": noise,
                "gauge_accuracy_mean": sum(run["gauge_accuracy"] for run in runs) / 3,
                "naive_accuracy_mean": sum(run["naive_accuracy"] for run in runs) / 3,
                "real_channel_agreement_mean":
                    sum(run["real_channel_agreement"] for run in runs) / 3,
            })
    output.write_text(json.dumps({"schema": "hermes-phase-memory/1",
                                  "rows": rows}, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
