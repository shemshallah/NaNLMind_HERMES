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
    if isinstance(head_index, bool) or not isinstance(head_index, int) or not 0 <= head_index <= 255:
        raise ValueError("head_index must be 0..255")
    if isinstance(patch_mask, bool) or not isinstance(patch_mask, int) or not 1 <= patch_mask <= 65535:
        raise ValueError("patch_mask must be 1..65535")
    if not isinstance(base_topology_digest, str) or len(base_topology_digest) != 64:
        raise ValueError("base_topology_digest must be a SHA-256 digest")
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
    """Evaluate one isolated candidate against a fixed caller-declared test split."""
    if approved is not True:
        raise PermissionError("held-out candidate evaluation requires explicit approved=true")
    if not isinstance(mutation_id, str) or not mutation_id or len(mutation_id) > 64:
        raise ValueError("mutation_id must be 1..64 characters")
    if not isinstance(records, list) or not 1 <= len(records) <= 512:
        raise ValueError("records must contain 1..512 held-out relations")
    required = {"resolution", "polarity", "subject", "predicate", "object", "source_digest"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("invalid held-out relation at index %d" % index)
        resolution = record["resolution"]
        if isinstance(resolution, bool) or not isinstance(resolution, int) or not 0 <= resolution <= 9:
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


def head_learn_candidate(ctx, mutation_id, base_topology_digest,
                         approved=False, scratch_id="default", epochs=1):
    if approved is not True:
        raise PermissionError("CLOCKED weak-supervision training requires explicit approved=true")
    if not isinstance(mutation_id, str) or not mutation_id or len(mutation_id) > 64:
        raise ValueError("invalid mutation_id")
    scratch = _scratch(ctx, scratch_id)
    if base_topology_digest is None:
        for row in reversed(scratch.lane("head_response")):
            payload = row.get("payload", {})
            provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
            digest = provenance.get("state_digest") if isinstance(provenance, dict) else None
            if isinstance(digest, str) and len(digest) == 64:
                base_topology_digest = digest
                break
    if not isinstance(base_topology_digest, str) or len(base_topology_digest) != 64:
        raise ValueError("base_topology_digest must be supplied or present in the latest head response")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 8:
        raise ValueError("epochs must be between 1 and 8")
    if not os.environ.get("NANLM_HEAD_CORTEX_URL") or not os.environ.get("NANLM_DEVICE_TOKEN"):
        raise RuntimeError("configure NANLM_HEAD_CORTEX_URL and a NANNET training:execute device token")
    observations = scratch.lane("clocked_result")
    if not observations:
        raise ValueError("scratch contains no CLOCKED endpoint observations")
    records = []

    def add_record(observation, entry, predicate, obj, polarity="+", resolution=5):
        provenance = observation.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
        source_digest = provenance.get("input_sha256")
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            source_digest = hashlib.sha256(json.dumps(
                observation, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode("utf-8")).hexdigest()
        observation_id = str(observation.get("task_id") or entry.get("entry_id") or "clocked")
        safe_id = "".join(ch if ch.isalnum() or ch in "_.:-" else "-" for ch in observation_id)[:120]
        if not safe_id:
            safe_id = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()[:32]
        confidence = 1
        if isinstance(observation.get("confidence"), (int, float)) and not isinstance(observation.get("confidence"), bool):
            confidence = max(1, min(65535, int(float(observation["confidence"]) * 65535)))
        text_value = str(obj).strip()[:2048]
        if not text_value:
            return
        records.append({
            "observation_id": safe_id + "-" + str(len(records)),
            "resolution": resolution, "polarity": polarity,
            "confidence": confidence,
            "subject": str(observation.get("source_id") or "cross_source")[:128],
            "predicate": predicate[:128], "object": text_value,
            "source_digest": source_digest,
        })

    for entry in observations:
        observation = entry.get("payload", {})
        if not isinstance(observation, dict):
            continue
        endpoint = str(observation.get("endpoint") or "unknown")
        result_data = observation.get("result", {})
        if not isinstance(result_data, dict):
            result_data = {}
        if isinstance(result_data.get("result"), dict):
            result_data = result_data["result"]
        fields = result_data.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}
        status = str(observation.get("status") or "unknown")
        add_record(observation, entry, "clocked_endpoint_observation",
                   endpoint + ":" + status, "+", 5)
        if endpoint == "extract":
            claims = fields.get("claims", [])
            if isinstance(claims, str):
                claims = [claims]
            if isinstance(claims, list):
                for claim in claims[:64]:
                    if isinstance(claim, dict):
                        claim = claim.get("claim") or claim.get("text") or claim.get("value")
                    if isinstance(claim, str) and claim.strip():
                        add_record(observation, entry, "states_claim", claim, "+", 5)
        elif endpoint == "grounding":
            verdict = fields.get("verdict", result_data.get("verdict"))
            if isinstance(verdict, str) and verdict.strip():
                normalized = verdict.casefold()
                polarity = "-" if normalized in {"unsupported", "not_supported", "false", "ungrounded"} else "+"
                add_record(observation, entry, "grounding_verdict", verdict, "+", 7)
                if normalized in {"supported", "grounded", "true"}:
                    add_record(observation, entry, "grounding_supports_claim", verdict, "+", 7)
                elif normalized in {"unsupported", "not_supported", "false", "ungrounded"}:
                    add_record(observation, entry, "grounding_supports_claim", verdict, "-", 7)
        elif endpoint == "compare":
            verdict = fields.get("verdict", result_data.get("verdict"))
            if isinstance(verdict, str) and verdict.strip():
                add_record(observation, entry, "source_comparison", verdict, "+", 7)
                if "conflict" in verdict.casefold():
                    add_record(observation, entry, "sources_conflict", verdict, "+", 7)
        elif endpoint == "pii":
            detected = fields.get("detected", result_data.get("detected"))
            if isinstance(detected, bool):
                add_record(observation, entry, "contains_pii", "detected" if detected else "not_detected",
                           "+" if detected else "-", 9)
        elif endpoint == "injection":
            detected = fields.get("detected", result_data.get("detected"))
            verdict = fields.get("verdict", result_data.get("verdict"))
            if isinstance(detected, bool):
                add_record(observation, entry, "contains_prompt_injection",
                           "detected" if detected else "not_detected",
                           "+" if detected else "-", 9)
            elif isinstance(verdict, str) and verdict.strip():
                add_record(observation, entry, "injection_verdict", verdict, "+", 9)
        elif endpoint == "script":
            matched = fields.get("matched", result_data.get("matched"))
            if isinstance(matched, bool):
                add_record(observation, entry, "contains_script_pattern",
                           "matched" if matched else "not_matched",
                           "+" if matched else "-", 9)
        elif endpoint == "decide":
            decision = fields.get("decision", result_data.get("decision"))
            if isinstance(decision, str) and decision.strip():
                add_record(observation, entry, "clocked_decision", decision, "+", 9)
    if not records:
        raise ValueError("CLOCKED scratch has no typed teacher signals suitable for candidate training")
    records = records[:512]
    dataset_sha256 = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    request = {
        "schema": "nanlm.head-learning-candidate-request/1",
        "mutation_id": mutation_id,
        "base_topology_digest": base_topology_digest,
        "split": "train", "teacher": "CLOCKED",
        "teacher_outputs_are_ground_truth": False,
        "dataset_sha256": dataset_sha256,
        "epochs": epochs, "records": records,
    }
    candidate = HeadCortexClient().create_learning_candidate(request)
    scratch.append("learning_proposal", "hermes", {
        "kind": "clocked_weak_supervision_head_candidate",
        "mutation_id": mutation_id,
        "training_dataset_sha256": dataset_sha256,
        "training_record_count": len(records),
        "candidate": candidate,
        "teacher_outputs_are_ground_truth": False,
        "promotion_allowed": False,
    })
    return {"candidate": candidate, "training_records": len(records),
            "training_dataset_sha256": dataset_sha256,
            "scratch": scratch.snapshot()}
