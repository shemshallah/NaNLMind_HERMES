"""HERMES tool handlers for CLOCKED planning, scratch and NaN head-cortex calls."""
from __future__ import annotations

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


def clocked_plan(ctx, question, sources=None, scratch_id="default"):
    sources = sources or []
    mode = propose_mode(question, sources)
    scratch = _scratch(ctx, scratch_id)
    snap = scratch.snapshot()
    if not snap["entry_count"]:
        snap_entry = scratch.append("user", "user", {
            "question": question, "source_count": len(sources),
        }, snap["revision"])
        snap = scratch.snapshot()
    if sources:
        plan = make_task_plan(sources, question)
        validation = validate_task_plan(plan, {
            "max_requests": 16, "max_bytes": 1_048_576,
        })
        recorded = scratch.append("hermes_plan", "hermes", {
            "mode": mode["mode"], "plan": plan, "validation": validation,
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


def clocked_execute(ctx, question, sources, scratch_id="default",
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
    fresh = make_task_plan(sources, question)
    if plan.get("tasks") != fresh.get("tasks"):
        raise ValueError("question or sources differ from the reviewed CLOCKED plan")
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
    validation = head_process(ctx, question, scratch_id, "ANSWER_VALIDATION", draft)
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
