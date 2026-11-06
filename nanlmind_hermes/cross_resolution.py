"""Mathematical feedback between the ten linguistic resolutions.

This module never executes model-supplied expressions. It computes deterministic
co-occurrence, alignment, contradiction, coverage and entropy summaries over
immutable memory events, then emits idempotent hyperedge/equation discoveries.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

RESOLUTIONS = (
    "bit", "nibble", "byte", "phonetic", "morpheme",
    "lexical", "word_completion", "sentence", "paragraph", "narrative",
)
INDEX = {name: i for i, name in enumerate(RESOLUTIONS)}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def analyze_cross_resolution(rows: list[dict]) -> dict:
    """Compute a symmetric 10x10 interaction matrix and per-concept feedback."""
    matrix = [[0 for _ in RESOLUTIONS] for _ in RESOLUTIONS]
    fine_to_coarse = [[0.0 for _ in RESOLUTIONS] for _ in RESOLUTIONS]
    coarse_to_fine = [[0.0 for _ in RESOLUTIONS] for _ in RESOLUTIONS]
    by_subject: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    coverage = Counter()
    for row in rows:
        provenance = row.get("provenance") or {}
        resolution = provenance.get("resolution")
        if resolution not in INDEX:
            continue
        coverage[resolution] += 1
        provenance = row.get("provenance") or {}
        alignment = provenance.get("alignment_id") or provenance.get("concept_id") or row.get("subject")
        if isinstance(alignment, str) and alignment:
            by_subject[alignment][resolution].append(row)

    feedback = []
    for subject in sorted(by_subject):
        groups = by_subject[subject]
        present = sorted(groups, key=INDEX.__getitem__)
        if len(present) < 2:
            continue
        pairs = []
        aligned = contradicted = comparable = 0
        content_aligned = content_divergent = 0
        event_ids = []
        for resolution in present:
            event_ids.extend(row["event_id"] for row in groups[resolution])
        for i, left_name in enumerate(present):
            for right_name in present[i + 1:]:
                left_i, right_i = INDEX[left_name], INDEX[right_name]
                left_rows, right_rows = groups[left_name], groups[right_name]
                matrix[left_i][right_i] += len(left_rows) * len(right_rows)
                matrix[right_i][left_i] += len(left_rows) * len(right_rows)
                pair_aligned = pair_contradicted = pair_comparable = 0
                for left in left_rows:
                    for right in right_rows:
                        lp, rp = left.get("polarity", "neutral"), right.get("polarity", "neutral")
                        status = "unresolved"
                        if {lp, rp} == {"supports", "refutes"}:
                            status = "contradiction"
                            contradicted += 1
                            comparable += 1
                            pair_contradicted += 1
                            pair_comparable += 1
                        elif lp == rp and lp in {"supports", "refutes"}:
                            status = "polarity-aligned"
                            aligned += 1
                            comparable += 1
                            pair_aligned += 1
                            pair_comparable += 1
                        elif lp == rp == "neutral":
                            status = "neutral-aligned"
                        lo, ro = left.get("object"), right.get("object")
                        if lo is not None and ro is not None:
                            if _canonical(lo) == _canonical(ro):
                                content_aligned += 1
                                status = "content-aligned" if status != "contradiction" else status
                            else:
                                content_divergent += 1
                        pairs.append({
                            "left_resolution": left_name,
                            "right_resolution": right_name,
                            "left_event": left["event_id"],
                            "right_event": right["event_id"],
                            "status": status,
                        })
                pair_score = ((pair_aligned - pair_contradicted) / pair_comparable
                              if pair_comparable else 0.0)
                fine_to_coarse[left_i][right_i] += pair_score
                coarse_to_fine[right_i][left_i] += pair_score
        denominator = aligned + contradicted
        score = ((aligned - contradicted) / denominator) if denominator else 0.0
        polarity_counts = Counter(
            row.get("polarity", "neutral")
            for resolution in present for row in groups[resolution]
        )
        total = sum(polarity_counts.values())
        entropy = (-sum((n / total) * math.log2(n / total)
                        for n in polarity_counts.values()) if total else 0.0)
        feedback.append({
            "subject": subject,
            "resolutions": present,
            "resolution_indices": [INDEX[name] for name in present],
            "event_ids": sorted(set(event_ids)),
            "pair_count": len(pairs),
            "aligned_pairs": aligned,
            "contradicted_pairs": contradicted,
            "comparable_pairs": comparable,
            "content_aligned_pairs": content_aligned,
            "content_divergent_pairs": content_divergent,
            "consistency_score": round(score, 8),
            "polarity_entropy_bits": round(entropy, 8),
            "pairs": pairs[:256],
        })

    return {
        "schema": "nanlm-cross-resolution-analysis/1",
        "resolution_order": list(RESOLUTIONS),
        "coverage": {name: coverage.get(name, 0) for name in RESOLUTIONS},
        "cooccurrence_matrix": matrix,
        "fine_to_coarse_feedback_matrix": fine_to_coarse,
        "coarse_to_fine_feedback_matrix": coarse_to_fine,
        "concepts_compared": len(feedback),
        "feedback": feedback,
    }


def feedback_records(analysis: dict) -> list[dict]:
    """Convert cross-level comparisons into deterministic, mergeable hyperedges."""
    records = []
    for item in analysis.get("feedback", []):
        participants = item.get("event_ids", [])
        if len(participants) < 2:
            continue
        identity = _digest({
            "subject": item["subject"],
            "participants": participants,
            "analysis": analysis["schema"],
        })
        source_digest = _digest({"cross_resolution_source": identity})
        payload = {
            "kind": "equation",
            "subject": "cross-resolution:" + identity[:24],
            "predicate": "cross_resolution_feedback",
            "object": item["consistency_score"],
            "expression": {
                "op": "resolution_feedback",
                "resolution_indices": item["resolution_indices"],
                "pair_count": item["pair_count"],
                "aligned_pairs": item["aligned_pairs"],
                "contradicted_pairs": item["contradicted_pairs"],
                "content_aligned_pairs": item["content_aligned_pairs"],
                "content_divergent_pairs": item["content_divergent_pairs"],
                "consistency_score": item["consistency_score"],
                "polarity_entropy_bits": item["polarity_entropy_bits"],
                "pair_evidence": item["pairs"],
                "score_definition": "(aligned_pairs - contradicted_pairs) / "
                                    "(aligned_pairs + contradicted_pairs), or 0 if none",
            },
            "participants": participants,
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence": {"source_subject": item["subject"], "event_count": len(participants)},
            "provenance": {
                "cross_resolution": True,
                "analysis_schema": analysis["schema"],
                "source_resolution_count": len(item["resolutions"]),
            },
            "tags": ["cross-resolution", "derived-mathematics", "non-destructive"],
        }
        records.append({"item": payload, "source_digest": source_digest, "layer": 9})
    return records
