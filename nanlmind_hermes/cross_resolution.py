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
        "numeric_relations": discover_numeric_relations(rows),
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
    for relation in analysis.get("numeric_relations", {}).get("relations", []):
        identity = _digest({"numeric_relation": relation})
        payload = {
            "kind": "equation",
            "subject": "cross-resolution-numeric:" + identity[:24],
            "predicate": "verified_numeric_relation",
            "object": relation["result"],
            "expression": {
                "op": relation["relation"],
                "left_value": relation["left_value"],
                "right_value": relation["right_value"],
                "result": relation["result"],
                "units": relation["units"],
                "left_resolution": relation["left_resolution"],
                "right_resolution": relation["right_resolution"],
                "verified": True,
            },
            "participants": [relation["left_event_id"], relation["right_event_id"]],
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence": {"alignment_id": relation["alignment_id"]},
            "provenance": {
                "cross_resolution": True,
                "cross_resolution_relation": True,
            },
            "tags": ["cross-resolution", "verified-numeric-relation", "non-destructive"],
        }
        records.append({
            "item": payload,
            "source_digest": _digest({"numeric_relation_source": identity}),
            "layer": 9,
        })
    return records


def discover_numeric_relations(rows: list[dict]) -> dict:
    """Discover only unit-compatible numeric relations explicitly grounded in events.

    Each row may expose expression.value (or a numeric object), units, and a
    stable alignment_id/concept_id. No model expression is evaluated.
    """
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        provenance = row.get("provenance") or {}
        resolution = provenance.get("resolution")
        if resolution not in INDEX:
            continue
        alignment = provenance.get("alignment_id") or provenance.get("concept_id")
        if not isinstance(alignment, str) or not alignment:
            continue
        expression = row.get("expression")
        value = expression.get("value") if isinstance(expression, dict) else row.get("object")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        units = row.get("units")
        if units is None and isinstance(expression, dict):
            units = expression.get("units")
        grouped[alignment][resolution].append({
            "event_id": row.get("event_id"),
            "value": float(value),
            "units": units,
        })

    relations = []
    for alignment in sorted(grouped):
        levels = grouped[alignment]
        names = sorted(levels, key=INDEX.__getitem__)
        for i, left_name in enumerate(names):
            for right_name in names[i + 1:]:
                for left in levels[left_name]:
                    for right in levels[right_name]:
                        if left["units"] is None or left["units"] != right["units"]:
                            continue
                        a, b = left["value"], right["value"]
                        if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                            relation, result = "equal", 1.0
                        elif not math.isclose(b, 0.0, abs_tol=1e-15):
                            relation, result = "ratio", a / b
                        else:
                            relation, result = "difference", a - b
                        relations.append({
                            "alignment_id": alignment,
                            "left_resolution": left_name,
                            "right_resolution": right_name,
                            "left_event_id": left["event_id"],
                            "right_event_id": right["event_id"],
                            "relation": relation,
                            "left_value": a,
                            "right_value": b,
                            "result": result,
                            "units": left["units"],
                            "verified": True,
                        })
    return {
        "schema": "nanlm-numeric-relations/1",
        "relations": relations,
        "relation_count": len(relations),
        "distinct_alignments": len(grouped),
        "policy": "explicit-alignment-and-unit-compatible; inert numeric fields only",
    }


def permutation_control(rows: list[dict], seed: int = 1729) -> dict:
    """Deterministic negative control: permute alignment labels, preserving coverage."""
    import random

    labels = []
    indices = []
    for i, row in enumerate(rows):
        provenance = row.get("provenance") or {}
        label = provenance.get("alignment_id") or provenance.get("concept_id")
        if isinstance(label, str) and label:
            labels.append(label)
            indices.append(i)
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    control_rows = [dict(row) for row in rows]
    for index, label in zip(indices, shuffled):
        provenance = dict(control_rows[index].get("provenance") or {})
        provenance["alignment_id"] = label
        provenance.pop("concept_id", None)
        control_rows[index]["provenance"] = provenance
    observed = analyze_cross_resolution(rows)
    control = analyze_cross_resolution(control_rows)
    return {
        "schema": "nanlm-cross-resolution-permutation-control/1",
        "seed": seed,
        "observed_concepts_compared": observed["concepts_compared"],
        "control_concepts_compared": control["concepts_compared"],
        "observed_comparable_pairs": sum(x["comparable_pairs"] for x in observed["feedback"]),
        "control_comparable_pairs": sum(x["comparable_pairs"] for x in control["feedback"]),
        "observed_abs_consistency": sum(abs(x["consistency_score"]) for x in observed["feedback"]),
        "control_abs_consistency": sum(abs(x["consistency_score"]) for x in control["feedback"]),
        "interpretation": "negative-control diagnostic; not a p-value",
    }
