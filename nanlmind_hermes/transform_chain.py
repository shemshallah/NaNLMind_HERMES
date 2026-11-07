"""Deterministic M0-M9 evidence-processing chain for CLOCKED/NaNLMind.

This is an evidence adapter, not a trainer. M6 delegates to the existing
read-only native relation-eval service; no pass mutates canonical model state.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRUTH_STATES = ("TRUE", "FALSE", "UNKNOWN", "CONFLICT")
LANE_FAMILY = {
    "user": "intent",
    "hermes_plan": "control",
    "clocked_result": "observation",
    "evidence_atoms": "relation",
    "conflict_map": "consensus",
    "hermes_synthesis": "answer",
    "head_response": "inference",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _pass(pass_id: str, operation: str, input_digest: str,
          output: Any, summary: dict | None = None) -> dict:
    return {
        "pass": pass_id,
        "operation": operation,
        "status": "pass",
        "input_sha256": input_digest,
        "output_sha256": _digest(output),
        "summary": summary or {},
    }


def preflight(package: dict) -> dict:
    """Run M0-M5 before the existing native inference endpoint."""
    package_digest = package.get("package_sha256")
    if not isinstance(package_digest, str) or not HEX64.fullmatch(package_digest):
        raise ValueError("M0 requires a valid evidence-package digest")
    unsigned = dict(package)
    unsigned.pop("package_sha256", None)
    if _digest(unsigned) != package_digest:
        raise ValueError("M0 evidence-package digest mismatch")

    states = package.get("truth_states")
    policy = package.get("policy")
    if (states != list(TRUTH_STATES) or not isinstance(policy, dict)
            or policy.get("unknown_is_not_false") is not True
            or policy.get("conflict_is_not_averaged") is not True
            or policy.get("clocked_outputs_are_not_ground_truth") is not True):
        raise ValueError("M0 truth-state policy mismatch")
    trace = [_pass("M0", "truth_gate", package_digest, states,
                   {"states": list(TRUTH_STATES), "unknown_preserved": True,
                    "conflict_preserved": True})]

    entries = package.get("evidence_entries")
    if not isinstance(entries, list) or len(entries) > 10000:
        raise ValueError("M1 evidence-entry bounds invalid")
    span_count = 0
    byte_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("M1 evidence entry must be an object")
        raw = _canonical(entry)
        byte_count += len(raw)
        if byte_count > 4 * 1024 * 1024:
            raise ValueError("M1 evidence byte budget exceeded")
        payload = entry.get("payload")
        if isinstance(payload, dict):
            spans = payload.get("spans", [])
            if spans is not None:
                if not isinstance(spans, list):
                    raise ValueError("M1 source spans must be a list")
                for span in spans:
                    if not isinstance(span, dict):
                        raise ValueError("M1 source span must be an object")
                    start, end = span.get("start"), span.get("end")
                    if (isinstance(start, bool) or not isinstance(start, int)
                            or isinstance(end, bool) or not isinstance(end, int)
                            or start < 0 or end < start):
                        raise ValueError("M1 source span bounds invalid")
                    span_count += 1
    trace.append(_pass("M1", "byte_span_mapper", package_digest,
                       {"entry_count": len(entries), "byte_count": byte_count,
                        "span_count": span_count},
                       {"entries": len(entries), "canonical_bytes": byte_count,
                        "validated_spans": span_count}))

    routes = []
    feature_rows = []
    for index, entry in enumerate(entries):
        lane = entry.get("lane")
        family = LANE_FAMILY.get(lane, "unclassified")
        entry_id = str(entry.get("entry_id") or index)
        hint = int(hashlib.sha256((family + ":" + entry_id).encode("utf-8")).hexdigest()[:8], 16) % 256
        routes.append({"entry_id": entry_id, "family": family,
                       "logical_head_hint": hint})
        payload_bytes = _canonical(entry.get("payload"))
        text = payload_bytes.decode("utf-8", errors="strict")
        feature_rows.append({
            "entry_id": entry_id,
            "byte_count": len(payload_bytes),
            "digit_count": sum(ch.isdigit() for ch in text),
            "whitespace_count": sum(ch.isspace() for ch in text),
            "has_source_digest": bool(
                isinstance(entry.get("payload"), dict)
                and isinstance(entry["payload"].get("source_digest"), str)
            ),
        })
    trace.append(_pass("M2", "type_head_router", package_digest, routes,
                       {"routed_entries": len(routes),
                        "logical_heads": sorted({r["logical_head_hint"] for r in routes})}))
    trace.append(_pass("M3", "feature_projector", package_digest, feature_rows,
                       {"projected_entries": len(feature_rows),
                        "feature_schema": "surface-byte-counts.v1"}))

    revisions = [entry.get("revision") for entry in entries]
    valid_revisions = [x for x in revisions if isinstance(x, int) and not isinstance(x, bool)]
    if valid_revisions != sorted(valid_revisions):
        raise ValueError("M4 scratch revisions are not monotonic")
    temporal = [{"entry_id": str(entry.get("entry_id") or i),
                 "revision": entry.get("revision", i)}
                for i, entry in enumerate(entries)]
    trace.append(_pass("M4", "temporal_mapper", package_digest, temporal,
                       {"ordered_entries": len(temporal),
                        "revision_monotonic": True}))

    counts = {
        "evidence_atoms": sum(e.get("lane") == "evidence_atoms" for e in entries),
        "conflict_maps": sum(e.get("lane") == "conflict_map" for e in entries),
        "clocked_observations": sum(e.get("lane") == "clocked_result" for e in entries),
    }
    trace.append(_pass("M5", "relation_hypergraph_builder", package_digest, counts,
                       counts))
    return {"package_sha256": package_digest, "passes": trace,
            "routes": routes, "features": feature_rows, "temporal": temporal,
            "relation_counts": counts}


def postflight(package: dict, response: dict, prepared: dict) -> dict:
    """Run M7-M9 after M6 has delegated to native relation-eval."""
    package_digest = package.get("package_sha256")
    if not isinstance(response, dict) or response.get("request_sha256") != package_digest:
        raise ValueError("M7 head-cortex response is not bound to this evidence package")
    provenance = response.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("read_only") is not True:
        raise ValueError("M7 head-cortex response is not read-only")
    relations = response.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("M7 relation result must be a list")
    reported_state = response.get("truth_state")
    if reported_state not in TRUTH_STATES:
        raise ValueError("M7 response truth state is invalid")
    raw_source_ids = provenance.get("source_entry_ids", [])
    if (not isinstance(raw_source_ids, list)
            or any(not isinstance(x, str) for x in raw_source_ids)):
        raise ValueError("M7 source-entry provenance is malformed")
    state_digest = provenance.get("state_digest")
    if state_digest is not None and (
            not isinstance(state_digest, str) or not HEX64.fullmatch(state_digest)):
        raise ValueError("M7 native state digest is invalid")
    states = set()
    source_ids = set(raw_source_ids)
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError("M7 relation must be an object")
        state = relation.get("truth_state")
        if state not in TRUTH_STATES:
            raise ValueError("M7 relation has an invalid truth state")
        states.add(state)
        head = relation.get("head_id")
        if head is not None and (isinstance(head, bool) or not isinstance(head, int)
                                 or not 0 <= head <= 255):
            raise ValueError("M7 relation head is outside the 256-head topology")
        premises = relation.get("premises", [])
        if premises and (not isinstance(premises, list)
                         or any(not isinstance(x, str) for x in premises)):
            raise ValueError("M7 relation premises are malformed")
        if source_ids and any(x not in source_ids for x in premises):
            raise ValueError("M7 relation premise is not in the response provenance")
    trace = list(prepared["passes"])
    trace.append(_pass("M6", "native_boolean_inference",
                       package_digest, response,
                       {"generation": response.get("generation"),
                        "relation_count": len(relations),
                        "native_read_only": True}))

    receipt = {
        "request_sha256": package_digest,
        "state_digest": provenance.get("state_digest"),
        "source_entry_ids": sorted(source_ids),
        "relation_count": len(relations),
    }
    trace.append(_pass("M7", "provenance_route_validator",
                       _digest(response), receipt,
                       {"read_only": True, "provenance_bound": True,
                        "validated_relations": len(relations)}))

    if "CONFLICT" in states or ("TRUE" in states and "FALSE" in states):
        consensus = "CONFLICT"
    elif not states or "UNKNOWN" in states:
        consensus = "UNKNOWN"
    else:
        consensus = next(iter(states))
    if reported_state != consensus:
        raise ValueError("M8 native and reconstructed consensus disagree")
    trace.append(_pass("M8", "cross_source_consensus",
                       _digest(relations), {"truth_state": consensus},
                       {"observed_states": sorted(states),
                        "truth_state": consensus}))

    abstentions = response.get("abstentions", [])
    if not isinstance(abstentions, list):
        abstentions = [{"reason": "malformed_abstention_receipt"}]
    decision = ("abstain" if consensus in ("UNKNOWN", "CONFLICT") or abstentions
                else "review_only")
    control = {
        "decision": decision,
        "truth_state": consensus,
        "canonical_mutation_allowed": False,
        "promotion_allowed": False,
        "clocked_outputs_are_ground_truth": False,
    }
    trace.append(_pass("M9", "metacontrol_calibration",
                       _digest({"truth_state": consensus, "abstentions": abstentions}),
                       control, control))
    return {
        "package_sha256": package_digest,
        "passes": trace,
        "truth_state": consensus,
        "decision": decision,
        "canonical_mutation_allowed": False,
        "promotion_allowed": False,
    }
