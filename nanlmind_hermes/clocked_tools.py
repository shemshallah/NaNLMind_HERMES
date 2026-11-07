"""HERMES tool handlers for CLOCKED planning, scratch and NaN head-cortex calls."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .clocked_client import ClockedGatewayClient
from .clocked_mode import execute_multi_source, propose_mode
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
    plan = propose_mode(question, sources)
    scratch = _scratch(ctx, scratch_id)
    snap = scratch.snapshot()
    if not snap["entry_count"]:
        scratch.append("user", "user", {"question": question, "source_count": len(sources)})
    scratch.append("hermes_plan", "hermes", plan)
    return {"mode_plan": plan, "scratch": scratch.snapshot()}


def clocked_execute(ctx, question, sources, scratch_id="default",
                    approved=False, max_requests=16, max_bytes=1_048_576):
    if approved is not True:
        raise PermissionError("CLOCKED execution requires explicit approved=true after reviewing the plan")
    if os.environ.get("CLOCKED_GATEWAY_URL") is None or not os.environ.get("CLOCKED_GATEWAY_TOKEN"):
        raise RuntimeError("configure CLOCKED_GATEWAY_URL and CLOCKED_GATEWAY_TOKEN for the scoped participant gateway")
    scratch = _scratch(ctx, scratch_id)
    budget = {"max_requests": max_requests, "max_bytes": max_bytes}
    client = ClockedGatewayClient()
    return execute_multi_source(client, scratch, question, sources,
                                budget=budget)


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
    package = build_evidence_package(scratch.snapshot(), question, purpose)
    if os.environ.get("NANLM_HEAD_CORTEX_URL") is None or not os.environ.get("NANLM_DEVICE_TOKEN"):
        raise RuntimeError("configure NANLM_HEAD_CORTEX_URL and NANLM_DEVICE_TOKEN for the NaNLMind receiving interface")
    result = HeadCortexClient().process(package)
    scratch.append("head_response", "nanlmind", result)
    return {"package_sha256": package["package_sha256"],
            "head_response": result, "scratch": scratch.snapshot()}
