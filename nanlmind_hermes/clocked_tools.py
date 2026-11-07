"""HERMES tool handlers for CLOCKED planning, scratch and NaN head-cortex calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .clocked_client import ClockedGatewayClient
from .clocked_mode import execute_multi_source, make_task_plan, propose_mode, validate_task_plan
from .evidence import build_evidence_package
from .head_cortex import HeadCortexClient
from .scratch import Scratch


def _setting(ctx, key, default=None):
    try:
        return ctx.get_config(key, default=default)
    except Exception:
        return default


def _scratch(ctx, scratch_id):
    root = Path(str(_setting(ctx, "state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
    return Scratch(root / "clocked_scratch", scratch_id)



def choose_clocked_mode(ctx, question, sources=None):
    """Let HERMES reason over the task intent, then constrain its choice to policy."""
    sources = sources or []
    fallback = propose_mode(question, sources)
    if not sources:
        return {**fallback, "decision_method": "local_no_sources"}
    allowed = {
        "CLOCKED_PRIVACY_AUDIT": ["pii"],
        "CLOCKED_INJECTION_AUDIT": ["injection", "script"],
        "CLOCKED_COMPARE": ["extract", "grounding", "compare"],
        "CLOCKED_VERIFY": ["extract", "grounding"],
        "CLOCKED_MULTI_SOURCE": ["extract", "grounding", "compare"],
        "CLOCKED_EXTRACT": ["extract", "grounding"],
    }
    try:
        descriptors = []
        for index, source in enumerate(sources):
            descriptors.append({
                "id": str(source.get("id") or ("source-%d" % (index + 1)))
                    if isinstance(source, dict) else ("source-%d" % (index + 1)),
                "type": str(source.get("type") or "text")
                    if isinstance(source, dict) else "text",
            })
        response = ctx.llm.complete(
            messages=[
                {"role": "system", "content": (
                    "Choose the best CLOCKED mode for the user's task. Return only JSON "
                    "with keys mode and reason. Allowed modes: CLOCKED_PRIVACY_AUDIT, "
                    "CLOCKED_INJECTION_AUDIT, CLOCKED_COMPARE, CLOCKED_VERIFY, "
                    "CLOCKED_MULTI_SOURCE, CLOCKED_EXTRACT. Use COMPARE for explicit "
                    "cross-source agreement/conflict questions, VERIFY for verification, "
                    "PRIVACY_AUDIT for PII/redaction, INJECTION_AUDIT for prompt attacks, "
                    "EXTRACT for one ordinary source, MULTI_SOURCE for multiple sources "
                    "without a comparison request. Never request endpoints outside the "
                    "mode's fixed policy mapping."
                )},
                {"role": "user", "content": json.dumps({
                    "question": question, "sources": descriptors,
                    "source_count": len(sources),
                    "deterministic_initial_recommendation": fallback["mode"],
                }, ensure_ascii=False, sort_keys=True)},
            ],
            temperature=0.0, max_tokens=256, timeout=45,
            purpose="nanlm-hermes-clocked-mode-selection",
        )
        raw = getattr(response, "text", "")
        decision = json.loads(raw)
        mode = decision.get("mode") if isinstance(decision, dict) else None
        reason = decision.get("reason") if isinstance(decision, dict) else None
        if mode not in allowed or not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalid mode decision")
        if mode == "CLOCKED_COMPARE" and len(sources) < 2:
            raise ValueError("comparison mode requires multiple sources")
        if mode == "CLOCKED_MULTI_SOURCE" and len(sources) < 2:
            raise ValueError("multi-source mode requires multiple sources")
        return {
            "schema": "nanlm.hermes-mode-plan/1",
            "mode": mode, "reason": reason[:512],
            "selected_endpoints": allowed[mode],
            "requires_approval": True, "source_count": len(sources),
            "policy": "approval", "autonomous": True,
            "decision_method": "llm_policy_constrained",
        }
    except Exception:
        return {**fallback, "decision_method": "deterministic_fallback"}

def clocked_plan(ctx, question, sources=None, scratch_id="default"):
    sources = sources or []
    mode = choose_clocked_mode(ctx, question, sources)
    scratch = _scratch(ctx, scratch_id)
    snap = scratch.snapshot()
    if not snap["entry_count"]:
        snap_entry = scratch.append("user", "user", {
            "question": question, "source_count": len(sources),
        }, snap["revision"])
        snap = scratch.snapshot()
    if sources:
        plan = make_task_plan(sources, question, mode)
        validation = validate_task_plan(plan, {
            "max_requests": 16, "max_bytes": 1_048_576,
        })
        recorded = scratch.append("hermes_plan", "hermes", {
            "mode": mode["mode"], "question": question, "plan": plan, "validation": validation,
            "requires_approval": True,
        }, snap["revision"])
        return {"mode_plan": mode, "task_plan": plan,
                "approval_digest": validation["plan_sha256"],
                "scratch_revision": recorded["revision"],
                "scratch": scratch.snapshot()}
    recorded = scratch.append("hermes_plan", "hermes", {
        "mode": mode["mode"], "requires_approval": False,
    }, snap["revision"])
    return {"mode_plan": mode, "scratch_revision": recorded["revision"],
            "scratch": scratch.snapshot()}


def clocked_execute(ctx, question, sources=None, scratch_id="default",
                    approved=False, plan_sha256="", max_requests=16,
                    max_bytes=1_048_576):
    if approved is not True:
        raise PermissionError("CLOCKED execution requires explicit approved=true after reviewing the plan")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("the exact reviewed approval_digest is required")
    if not (_setting(ctx, "clocked_gateway_url", None) or os.environ.get("CLOCKED_GATEWAY_URL")) or not os.environ.get("CLOCKED_GATEWAY_TOKEN"):
        raise RuntimeError("configure CLOCKED_GATEWAY_URL and CLOCKED_GATEWAY_TOKEN for the scoped participant gateway")
    scratch = _scratch(ctx, scratch_id)
    plan_rows = scratch.lane("hermes_plan")
    reviewed = next((row["payload"] for row in reversed(plan_rows)
                     if isinstance(row.get("payload"), dict)
                     and isinstance(row["payload"].get("plan"), dict)), None)
    if reviewed is None:
        raise ValueError("create and review a CLOCKED plan before execution")
    plan = reviewed["plan"]
    if reviewed.get("question") != question:
        raise ValueError("question differs from the reviewed CLOCKED plan")
    if sources:
        fresh = make_task_plan(sources, question, propose_mode(question, sources))
        if plan.get("tasks") != fresh.get("tasks"):
            raise ValueError("sources differ from the reviewed CLOCKED plan")
    else:
        sources = []
    budget = {"max_requests": max_requests, "max_bytes": max_bytes}
    validation = validate_task_plan(plan, budget)
    if validation["plan_sha256"] != plan_sha256:
        raise PermissionError("approval digest does not match the reviewed task plan")
    client = ClockedGatewayClient(base_url=_setting(ctx, "clocked_gateway_url", None) or None)
    return execute_multi_source(client, scratch, question, sources,
                                budget=budget, plan=plan)

def scratch_show(ctx, scratch_id="default", lane=None):
    scratch = _scratch(ctx, scratch_id)
    if lane is None:
        return scratch.snapshot()
    return {"scratch_id": scratch_id, "lane": lane, "entries": scratch.lane(lane)}


def head_process(ctx, question, scratch_id="default", purpose="EVIDENCE_COGNITION",
                 draft=None):
    scratch = _scratch(ctx, scratch_id)
    if draft is not None:
        scratch.append("hermes_synthesis", "hermes", {
            "draft": draft, "claim_map": "requires_head_validation",
        })
    snapshot = scratch.snapshot()
    package = build_evidence_package(snapshot, question, purpose)
    scratch.append("nan16_package", "hermes", {
        "schema": package["schema"], "purpose": purpose,
        "package_sha256": package["package_sha256"],
        "scratch_revision": package["scratch_revision"],
        "evidence_entry_count": len(package["evidence_entries"]),
    })
    if not os.environ.get("NANLM_HEAD_CORTEX_URL") or not os.environ.get("NANLM_DEVICE_TOKEN"):
        raise RuntimeError("configure NANLM_HEAD_CORTEX_URL and NANLM_DEVICE_TOKEN for the NANNET-authorized NaNLMind receiving interface")
    result = HeadCortexClient().process(package)
    scratch.append("head_response", "nanlmind", result)
    if purpose == "ANSWER_VALIDATION":
        scratch.append("head_validation", "nanlmind", {
            "request_sha256": result.get("request_sha256"),
            "truth_state": result.get("truth_state"),
            "relations": result.get("relations", []),
            "abstentions": result.get("abstentions", []),
            "provenance": result.get("provenance", {}),
        })
    return {"package_sha256": package["package_sha256"],
            "head_response": result, "scratch": scratch.snapshot()}


def clocked_synthesize(ctx, question, scratch_id="default", max_tokens=2048):
    """Synthesize from evidence, submit the draft to heads, then compose final."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be between 1 and 8192")
    scratch = _scratch(ctx, scratch_id)
    snapshot = scratch.snapshot()
    if not snapshot["entry_count"]:
        raise ValueError("scratch is empty")
    evidence = [{
        "lane": row["lane"], "writer": row["writer"], "payload": row["payload"],
    } for row in snapshot["entries"] if row["lane"] in {
        "clocked_result", "evidence_atoms", "conflict_map", "head_response",
    }][-256:]
    if not evidence:
        raise ValueError("scratch contains no CLOCKED or NaNLMind evidence")
    first = ctx.llm.complete(
        messages=[
            {"role": "system", "content": (
                "Synthesize a precise answer using only the supplied evidence. "
                "Preserve source attribution, uncertainty, UNKNOWN and CONFLICT. "
                "Do not treat CLOCKED output or a Boolean-head prediction as ground truth. "
                "Do not invent citations, facts, or source agreement. State unresolved conflicts."
            )},
            {"role": "user", "content": json.dumps({
                "question": question, "evidence": evidence,
            }, ensure_ascii=False, sort_keys=True)},
        ], temperature=0.2, max_tokens=max_tokens, timeout=180,
        purpose="nanlm-hermes-clocked-synthesis",
    )
    draft = getattr(first, "text", "")
    if not isinstance(draft, str) or not draft.strip():
        raise RuntimeError("HERMES synthesis provider returned an empty draft")
    scratch.append("hermes_synthesis", "hermes", {
        "draft": draft, "source_scratch_revision": snapshot["revision"],
    })
    validation = head_process(ctx, question, scratch_id, "ANSWER_VALIDATION")
    second = ctx.llm.complete(
        messages=[
            {"role": "system", "content": (
                "Produce the final user-facing answer. Treat the head-cortex response as "
                "a diagnostic relation trace, not as an oracle. Make only corrections "
                "directly supported by the evidence. Preserve uncertainty and conflicts; "
                "if the head trace is UNKNOWN or abstains, do not force a binary conclusion. "
                "Do not claim the answer was independently verified merely because it was processed."
            )},
            {"role": "user", "content": json.dumps({
                "question": question, "draft": draft,
                "head_validation": validation["head_response"],
            }, ensure_ascii=False, sort_keys=True)},
        ], temperature=0.1, max_tokens=max_tokens, timeout=180,
        purpose="nanlm-hermes-final-synthesis",
    )
    final_text = getattr(second, "text", "")
    if not isinstance(final_text, str) or not final_text.strip():
        raise RuntimeError("HERMES final synthesis provider returned an empty response")
    final_entry = scratch.append("final", "hermes", {
        "text": final_text,
        "head_validation_request_sha256": validation["package_sha256"],
        "head_truth_state": validation["head_response"].get("truth_state", "UNKNOWN"),
    })
    return {"response": final_text, "draft": draft,
            "head_validation": validation["head_response"],
            "final_revision": final_entry["revision"],
            "scratch": scratch.snapshot()}

def head_candidate(ctx, mutation_id, head_index, patch_mask,
                   base_topology_digest, approved=False, scratch_id="default"):
    if approved is not True:
        raise PermissionError("candidate head mutation requires explicit approved=true")
    if not isinstance(mutation_id, str) or not mutation_id or len(mutation_id) > 64:
        raise ValueError("mutation_id must be 1..64 characters")
    if isinstance(head_index, bool) or not isinstance(head_index, int) or not 0 <= head_index <= 255:
        raise ValueError("head_index must be 0..255")
    if isinstance(patch_mask, bool) or not isinstance(patch_mask, int) or not 1 <= patch_mask <= 65535:
        raise ValueError("patch_mask must be 1..65535")
    if (not isinstance(base_topology_digest, str) or len(base_topology_digest) != 64
            or any(c not in "0123456789abcdef" for c in base_topology_digest.lower())):
        raise ValueError("base_topology_digest must be a hexadecimal SHA-256 digest")
    if not os.environ.get("NANLM_HEAD_CORTEX_URL") or not os.environ.get("NANLM_DEVICE_TOKEN"):
        raise RuntimeError("configure NANLM_HEAD_CORTEX_URL and a NANNET training:execute device token")
    request = {
        "schema": "nanlm.head-candidate-request/1",
        "mutation_id": mutation_id,
        "head_index": head_index,
        "patch_mask": patch_mask,
        "base_topology_digest": base_topology_digest,
    }
    result = HeadCortexClient().create_candidate(request)
    scratch = _scratch(ctx, scratch_id)
    scratch.append("learning_proposal", "nanlmind", {
        "kind": "boolean_head_candidate",
        "mutation_id": mutation_id,
        "candidate": result,
        "promotion_allowed": False,
    })
    return {"candidate": result, "scratch": scratch.snapshot()}


def head_candidate_evaluate(ctx, mutation_id, records, approved=False,
                            scratch_id="default"):
    """Evaluate one isolated candidate against a fixed, caller-declared test split."""
    if approved is not True:
        raise PermissionError("held-out candidate evaluation requires explicit approved=true")
    if not isinstance(mutation_id, str) or not mutation_id or len(mutation_id) > 64:
        raise ValueError("mutation_id must be 1..64 characters")
    if not isinstance(records, list) or not 1 <= len(records) <= 512:
        raise ValueError("records must contain 1..512 held-out relations")
    required = {"resolution", "polarity", "subject", "predicate", "object", "source_digest"}
    allowed_resolutions = set(range(10))
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("invalid held-out relation at index %d" % index)
        resolution = record["resolution"]
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution not in allowed_resolutions:
            raise ValueError("held-out relation %d has unsupported N16REL resolution" % index)
        for field in ("polarity", "subject", "predicate", "object"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError("held-out relation %d has invalid %s" % (index, field))
        digest = record["source_digest"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest.lower())):
            raise ValueError("held-out relation %d has invalid source_digest" % index)
    if not os.environ.get("NANLM_HEAD_CORTEX_URL") or not os.environ.get("NANLM_DEVICE_TOKEN"):
        raise RuntimeError("configure NANLM_HEAD_CORTEX_URL and a NANNET training:execute device token")
    dataset_sha256 = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    request = {
        "schema": "nanlm.head-candidate-evaluation-request/1",
        "mutation_id": mutation_id,
        "split": "test",
        "dataset_sha256": dataset_sha256,
        "records": records,
    }
    result = HeadCortexClient().evaluate_candidate(request)
    scratch = _scratch(ctx, scratch_id)
    scratch.append("learning_proposal", "hermes", {
        "kind": "boolean_head_candidate_evaluation",
        "mutation_id": mutation_id,
        "dataset_sha256": dataset_sha256,
        "split": "test",
        "evaluation": result,
        "promotion_allowed": False,
        "canonical_state_modified": False,
    })
    return {"evaluation": result, "scratch": scratch.snapshot()}
