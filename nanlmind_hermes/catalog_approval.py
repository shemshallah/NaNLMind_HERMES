"""Hermes-native human approval gate for network and candidate mutations.

A model-supplied "approved" boolean is never sufficient authorization.
This pre_tool_call hook escalates requests to Hermes's own human approval
decision, which denies or times out unattended runs according to its policy.
It is deliberately read-only: no network request or state mutation in hook.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re

_GATED = frozenset({
    "nanlm_clocked_execute",
    "nanlm_head_candidate",
    "nanlm_head_learn_candidate",
    "nanlm_head_candidate_evaluate",
})
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _reject(reason: str) -> dict:
    return {"action": "block", "message": "NaNLMind approval preflight blocked: " + reason}


def make_approval_hook(ctx):
    """Bind the read-only CLOCKED plan inspection to this plugin context."""

    def pre_tool_call(tool_name="", args=None, **_):
        if tool_name not in _GATED:
            return None
        if type(args) is not dict:
            return _reject("object-shaped tool arguments required")
        try:
            digest = hashlib.sha256(json.dumps(
                args, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (ValueError, TypeError, OverflowError):
            return _reject("invalid tool argument encoding")

        if tool_name == "nanlm_clocked_execute":
            # The stored plan, not the model's copied digest, supplies the
            # human approval summary. No raw source text is shown.
            try:
                from .clocked_mode import validate_task_plan
                from .clocked_tools import _scratch
                scratch = _scratch(ctx, args.get("scratch_id", "default"))
                rows = scratch.lane("hermes_plan")
                reviewed = next((
                    r["payload"] for r in reversed(rows)
                    if isinstance(r.get("payload"), dict)
                    and isinstance(r["payload"].get("plan"), dict)
                ), None)
                if reviewed is None:
                    return _reject("CLOCKED plan must exist before approval")
                plan = reviewed["plan"]
                budget = {
                    "max_requests": args.get("max_requests", 16),
                    "max_bytes": args.get("max_bytes", 1048576),
                }
                actual_sha = validate_task_plan(plan, budget)["plan_sha256"]
                provided_sha = args.get("plan_sha256")
                if (type(provided_sha) is not str
                        or _SHA.fullmatch(provided_sha) is None
                        or not hmac.compare_digest(actual_sha, provided_sha)
                        or reviewed.get("question") != args.get("question")):
                    return _reject("CLOCKED plan SHA or question changed")
                tasks = plan["tasks"]
                endpoints = sorted({str(x["endpoint"]) for x in tasks})
                summary = (
                    f"Transmit approved CLOCKED sources to configured gateway. "
                    f"Tasks: {len(tasks)}. Endpoints: {', '.join(endpoints)}. "
                    f"Plan SHA-256: {actual_sha}."
                )
            except Exception:
                return _reject("cannot independently inspect stored CLOCKED plan")
        else:
            identifier = args.get("mutation_id")
            if type(identifier) is not str or _ID.fullmatch(identifier) is None:
                return _reject("invalid head candidate mutation ID")
            if tool_name == "nanlm_head_candidate":
                head = args.get("head_index")
                if type(head) is not int or not 0 <= head <= 255:
                    return _reject("invalid candidate head")
                summary = (
                    f"Create isolated head candidate {identifier!r} "
                    f"for native head {head}; no canonical promotion."
                )
            elif tool_name == "nanlm_head_learn_candidate":
                epochs = args.get("epochs", 1)
                if type(epochs) is not int or not 1 <= epochs <= 8:
                    return _reject("invalid candidate training epochs")
                summary = (
                    f"Transmit CLOCKED observations for isolated candidate "
                    f"{identifier!r}, epochs {epochs}; no canonical promotion."
                )
            else:
                records = args.get("records")
                if type(records) is not list or not 1 <= len(records) <= 512:
                    return _reject("invalid held-out candidate evaluation count")
                summary = (
                    f"Send {len(records)} held-out records to evaluate candidate "
                    f"{identifier!r}; no canonical promotion."
                )
        return {
            "action": "approve",
            "message": (summary + " This sends data or creates candidate state. "
                        f"Exact arguments SHA-256: {digest}. Human approval required."),
            "rule_key": "nanlmind:" + tool_name,
        }

    return pre_tool_call


__all__ = ("make_approval_hook",)
