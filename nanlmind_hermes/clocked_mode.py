"""Autonomous CLOCKED mode planning and multi-source evidence assembly."""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from .clocked_client import ClockedGatewayClient
from .scratch import Scratch

ALLOWED_ENDPOINTS = {
    "decide", "extract", "fill", "pii", "injection", "script",
    "grounding", "compare", "rerank", "pipeline",
}
MAX_TASKS = 16
MAX_TEXT_BYTES = 64 * 1024


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def propose_mode(question: str, source_texts: list[str] | None = None) -> dict:
    """Select a bounded CLOCKED mode from intent and available source material."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    sources = source_texts or []
    normalized = []
    for source in sources:
        text = source.get("text") if isinstance(source, dict) else source
        if not isinstance(text, str) or not text.strip():
            raise ValueError("sources must be non-empty text strings or objects with text")
        normalized.append(text)
    if len(normalized) > 7 or any(len(s.encode("utf-8")) > MAX_TEXT_BYTES for s in normalized):
        raise ValueError("at most seven sources, each no larger than 64 KiB")
    q = question.casefold()
    if not sources:
        mode, endpoints, reason = "LOCAL_CHAT", [], "no source material was supplied"
    elif any(term in q for term in ("prompt injection", "jailbreak", "malicious instruction", "instruction attack")):
        mode, endpoints, reason = "CLOCKED_INJECTION_AUDIT", ["injection", "script"], "question requests an instruction or injection safety audit"
    elif any(term in q for term in ("pii", "personal information", "privacy", "sensitive data", "redact")):
        mode, endpoints, reason = "CLOCKED_PRIVACY_AUDIT", ["pii"], "question requests a privacy or sensitive-data audit"
    elif len(sources) > 1 and any(term in q for term in ("compare", "difference", "conflict", "contradiction", "agree", "disagree")):
        mode, endpoints, reason = "CLOCKED_COMPARE", ["extract", "grounding", "compare"], "multi-source comparison or conflict analysis requested"
    elif any(term in q for term in ("verify", "validate", "is it true", "fact check", "ground this")):
        mode, endpoints, reason = "CLOCKED_VERIFY", ["extract", "grounding"], "question requests claim verification or grounding"
    elif len(sources) > 1:
        mode, endpoints, reason = "CLOCKED_MULTI_SOURCE", ["extract", "grounding", "compare"], "multiple independent sources require structured extraction and reconciliation"
    else:
        mode, endpoints, reason = "CLOCKED_EXTRACT", ["extract", "grounding"], "one source requires structured extraction and grounding"
    return {
        "schema": "nanlm.hermes-mode-plan/1",
        "mode": mode,
        "reason": reason,
        "selected_endpoints": endpoints,
        "requires_approval": bool(sources),
        "source_count": len(sources),
        "policy": "approval" if sources else "local",
        "autonomous": True,
    }

def validate_task_plan(plan: dict, budget: dict | None = None) -> dict:
    if not isinstance(plan, dict) or plan.get("schema") != "nanlm.clocked-task-plan/1":
        raise ValueError("invalid CLOCKED task plan schema")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
        raise ValueError("task plan must contain 1..16 tasks")
    ids = set()
    total_bytes = 0
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be an object")
        task_id, endpoint, text = task.get("id"), task.get("endpoint"), task.get("text")
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError("task IDs must be unique non-empty strings")
        ids.add(task_id)
        if endpoint not in ALLOWED_ENDPOINTS:
            raise ValueError("unsupported CLOCKED endpoint")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("task text is required")
        total_bytes += len(text.encode("utf-8"))
        deps = task.get("depends_on", [])
        if not isinstance(deps, list) or any(d not in ids for d in deps):
            raise ValueError("dependencies must reference earlier tasks")
    if total_bytes > MAX_TASKS * MAX_TEXT_BYTES:
        raise ValueError("task plan exceeds aggregate input limit")
    limits = budget or {}
    if len(tasks) > int(limits.get("max_requests", MAX_TASKS)):
        raise ValueError("task plan exceeds request budget")
    if total_bytes > int(limits.get("max_bytes", MAX_TASKS * MAX_TEXT_BYTES)):
        raise ValueError("task plan exceeds byte budget")
    return {"task_count": len(tasks), "input_bytes": total_bytes,
            "plan_sha256": _digest(plan)}


def make_task_plan(sources: list[dict], question: str,
                   mode_plan: dict | None = None) -> dict:
    if not isinstance(sources, list) or not sources:
        raise ValueError("at least one source is required")
    mode_plan = mode_plan or propose_mode(question, sources)
    mode = mode_plan.get("mode")
    selected = mode_plan.get("selected_endpoints", [])
    if mode == "CLOCKED_PRIVACY_AUDIT":
        per_source = ["pii"]
    elif mode == "CLOCKED_INJECTION_AUDIT":
        per_source = ["injection", "script"]
    elif mode in {"CLOCKED_VERIFY", "CLOCKED_EXTRACT"}:
        per_source = ["extract", "grounding"]
    elif mode in {"CLOCKED_COMPARE", "CLOCKED_MULTI_SOURCE"}:
        per_source = ["extract", "grounding"]
    else:
        raise ValueError("unsupported autonomous CLOCKED mode")
    tasks = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not isinstance(source.get("text"), str):
            raise ValueError("each source requires text")
        source_id = str(source.get("id") or ("source-%d" % (index + 1)))
        previous = []
        for endpoint in per_source:
            task_id = endpoint + "-" + source_id
            options = {}
            if endpoint == "extract":
                options = {"fields": {
                    "claims": "Short factual claims explicitly stated in the source",
                    "dates": "Dates explicitly stated in the source",
                    "quantities": "Measurements and numeric values explicitly stated",
                    "named_entities": "Named organizations, systems and products"}}
            elif endpoint == "grounding":
                options = {"claims_from": "extract-" + source_id}
            elif endpoint == "pii":
                options = {}
            tasks.append({"id": task_id, "source_id": source_id,
                          "endpoint": endpoint, "text": source["text"],
                          "options": options, "depends_on": previous[:]})
            previous.append(task_id)
    if mode in {"CLOCKED_COMPARE", "CLOCKED_MULTI_SOURCE"} and len(sources) > 1:
        tasks.append({"id": "compare-sources", "endpoint": "compare",
                      "text": "QUESTION: " + question + "\\n\\n" +
                              "\\n\\n".join(s["text"] for s in sources),
                      "depends_on": [t["id"] for t in tasks if t["endpoint"] == "extract"]})
    tasks.append({"id": "decide", "endpoint": "decide",
                  "text": question + "\\n\\n" + "\\n\\n".join(s["text"] for s in sources),
                  "depends_on": [t["id"] for t in tasks]})
    plan = {"schema": "nanlm.clocked-task-plan/1", "version": 1,
            "plan_id": uuid.uuid4().hex, "tasks": tasks}
    validate_task_plan(plan)
    return plan

def execute_multi_source(client: ClockedGatewayClient, scratch: Scratch,
                         question: str, sources: list[dict],
                         expected_revision: str | None = None,
                         budget: dict | None = None,
                         plan: dict | None = None) -> dict:
    plan = plan or make_task_plan(sources, question)
    validation = validate_task_plan(plan, budget)
    scratch.append("hermes_plan", "hermes", {
        "mode": "CLOCKED_MULTI_SOURCE", "plan": plan, "validation": validation,
    }, expected_revision)
    scratch.append("clocked_request", "hermes", {
        "plan_sha256": validation["plan_sha256"], "task_count": validation["task_count"],
    })
    result = client.execute(plan, idempotency_key=plan["plan_id"])
    if result.get("protocol") != "nanlm.clocked-task-results":
        raise ValueError("CLOCKED gateway returned an unsupported result protocol")
    observations = result.get("results")
    if not isinstance(observations, list):
        raise ValueError("CLOCKED gateway result list is missing")
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("invalid CLOCKED observation")
        scratch.append("clocked_result", "clocked", observation)
        result_data = observation.get("result", {})
        if not isinstance(result_data, dict):
            result_data = {}
        nested = result_data.get("result")
        if isinstance(nested, dict):
            result_data = nested
        fields = result_data.get("fields", {})
        claims = fields.get("claims", []) if isinstance(fields, dict) else []
        if isinstance(claims, str):
            claims = [claims]
        if isinstance(claims, list):
            for claim in claims[:64]:
                if isinstance(claim, dict):
                    claim = claim.get("claim") or claim.get("text") or claim.get("value")
                if not isinstance(claim, str) or not claim.strip():
                    continue
                scratch.append("evidence_atoms", "hermes", {
                    "subject": str(observation.get("source_id") or "source")[:128],
                    "predicate": "states_claim",
                    "object": claim[:2048],
                    "truth_state": "UNKNOWN",
                    "resolution": 5,
                    "source_digest": observation.get("provenance", {}).get("input_sha256"),
                    "endpoint": observation.get("endpoint"),
                    "result_type": observation.get("result_type"),
                    "result_ref": _digest(observation),
                    "provenance": observation.get("provenance", {}),
                })
        scratch.append("evidence_atoms", "hermes", {
            "source_id": observation.get("source_id"),
            "endpoint": observation.get("endpoint"),
            "native_result_type": observation.get("result_type"),
            "result_ref": _digest(observation),
            "status": observation.get("status"),
            "provenance": observation.get("provenance", {}),
        })
        if observation.get("endpoint") == "compare":
            compare_data = observation.get("result", {})
            if isinstance(compare_data, dict) and isinstance(compare_data.get("result"), dict):
                compare_data = compare_data["result"]
            compare_fields = compare_data.get("fields", {}) if isinstance(compare_data, dict) else {}
            if not isinstance(compare_fields, dict):
                compare_fields = {}
            conflicts = compare_fields.get("conflicts", compare_data.get("conflicts", [])
                if isinstance(compare_data, dict) else [])
            verdict = compare_fields.get("verdict", compare_data.get("verdict")
                if isinstance(compare_data, dict) else None)
            if isinstance(conflicts, str):
                conflicts = [conflicts]
            if isinstance(conflicts, list) and conflicts or (
                    isinstance(verdict, str) and "conflict" in verdict.casefold()):
                scratch.append("conflict_map", "hermes", {
                    "kind": "clocked_cross_source_conflict",
                    "source_ids": [s.get("id") for s in sources if isinstance(s, dict)],
                    "conflicts": conflicts[:64] if isinstance(conflicts, list) else [],
                    "verdict": verdict if isinstance(verdict, str) else None,
                    "result_ref": _digest(observation),
                    "provenance": observation.get("provenance", {}),
                })
    return {"plan": plan, "validation": validation,
            "result_count": len(observations), "scratch": scratch.snapshot()}
