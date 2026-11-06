"""Reproducible synthetic baselines for ten-resolution discovery.

Run: python -m benchmarks.cross_resolution_baseline
The fixtures are controls, not claims about real-world language performance.
"""
from __future__ import annotations

import json
from nanlmind_hermes.cross_resolution import (
    RESOLUTIONS, analyze_cross_resolution, discover_numeric_relations,
    permutation_control,
)


def row(event_id, concept, resolution, value, polarity="supports"):
    return {
        "event_id": event_id,
        "subject": concept,
        "object": value,
        "units": "count",
        "polarity": polarity,
        "provenance": {
            "resolution": resolution,
            "alignment_id": concept,
        },
    }


def run_baselines():
    aligned = [row(f"aligned-{i}", "known-count", name, 8)
               for i, name in enumerate(RESOLUTIONS)]
    contradiction = list(aligned)
    contradiction[-1] = row("contradiction", "known-count", "narrative", 8, "refutes")
    unrelated = [row(f"unrelated-{i}", f"unique-{i}", name, 8)
                 for i, name in enumerate(RESOLUTIONS)]
    numeric = [
        row("numeric-2", "known-ratio", "bit", 12),
        row("numeric-1", "known-ratio", "paragraph", 3),
    ]
    cases = {
        "perfect_alignment": analyze_cross_resolution(aligned),
        "one_level_contradiction": analyze_cross_resolution(contradiction),
        "unrelated_negative_control": analyze_cross_resolution(unrelated),
        "numeric_ratio_control": discover_numeric_relations(numeric),
        "permutation_negative_control": permutation_control(aligned, seed=1729),
    }
    return {
        "schema": "nanlm-cross-resolution-baselines/1",
        "fixture": "synthetic-known-answer-v1",
        "resolution_count": len(RESOLUTIONS),
        "pairwise_level_pairs": len(RESOLUTIONS) * (len(RESOLUTIONS) - 1) // 2,
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(run_baselines(), indent=2, sort_keys=True))
