"""Typed evidence package codec for the NaNLMind receiving interface."""
from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA = "nanlm.hermes-evidence-package"
VERSION = 1
MAX_PACKAGE_BYTES = 4 * 1024 * 1024


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def build_evidence_package(scratch_snapshot: dict, question: str,
                           purpose: str = "EVIDENCE_COGNITION") -> dict:
    if purpose not in {"EVIDENCE_COGNITION", "ANSWER_VALIDATION"}:
        raise ValueError("unsupported evidence package purpose")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    entries = scratch_snapshot.get("entries")
    if not isinstance(entries, list):
        raise ValueError("scratch snapshot entries are required")
    selected = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("invalid scratch entry")
        lane = entry.get("lane")
        if lane not in {
            "user", "hermes_plan", "clocked_result", "evidence_atoms",
            "conflict_map", "hermes_synthesis", "head_response",
        }:
            continue
        selected.append({
            "entry_id": entry.get("entry_id"),
            "lane": lane,
            "writer": entry.get("writer"),
            "sequence": entry.get("sequence") if entry.get("sequence") is not None else index + 1,
            "revision": entry.get("revision"),
            "payload": entry.get("payload"),
        })
    package = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": purpose,
        "scratch_id": scratch_snapshot.get("scratch_id"),
        "scratch_revision": scratch_snapshot.get("revision"),
        "question": question,
        "truth_states": ["TRUE", "FALSE", "UNKNOWN", "CONFLICT"],
        "evidence_entries": selected,
        "result_type_semantics": {
            "decide": "decision probability/labels",
            "extract": "source spans and fields",
            "grounding": "claim grounding verdict",
            "compare": "pairwise relation/conflict signal",
            "rerank": "relevance ordering",
        },
        "policy": {
            "clocked_outputs_are_teacher_observations": True,
            "clocked_outputs_are_not_ground_truth": True,
            "unknown_is_not_false": True,
            "conflict_is_not_averaged": True,
        },
    }
    raw = canonical(package)
    if len(raw) > MAX_PACKAGE_BYTES:
        raise ValueError("evidence package exceeds 4 MiB")
    package["package_sha256"] = hashlib.sha256(raw).hexdigest()
    return package
