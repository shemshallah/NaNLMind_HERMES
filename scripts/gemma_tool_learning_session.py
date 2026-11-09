#!/usr/bin/env python3
"""Open-ended Gemma steering session over the HERMES local tool surface.

Gemma receives a bounded tool catalog and the permanent session instructions,
then chooses tools turn by turn. The harness executes the real plugin handlers
in a temporary local state directory (no NANNET authority), records every call
and result, and writes a transcript plus a small learning report.

This tests whether the model can learn to use the tools; it is not a canonical
training run, a proof, or a NANNET/CLOCKED deployment test.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import (  # noqa: E402
    auth_tools,
    hermes_secrets,
    register,
    session_context,
)
from nanlmind_hermes.http_api import make_server as make_authority  # noqa: E402
from nanlmind_hermes.layered_learning import reset_coordinator_cache  # noqa: E402
from nanlmind_hermes.mth_leaves import reset_cache as reset_leaf_cache  # noqa: E402
from nanlmind_hermes.pairing import PairingAuthority  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_OUTPUT = ROOT / "docs" / "GEMMA_TOOL_LEARNING_SESSION.json"

SAFE_TOOLS = (
    "nanlm_status",
    "nanlm_session_context",
    "nanlm_chat",
    "nanlm_chat_history",
    "nanlm_learning_cycle",
    "nanlm_learning_status",
    "nanlm_wordnet_ingest",
    "nanlm_scratch_register",
    "nanlm_mth_catalog",
    "nanlm_mth_inspect",
    "nanlm_mth_runtime",
    "nanlm_mth_verify",
    "nanlm_mth_execute",
    "nanlm_mth_n16i_selftest",
    "nanlm_mth_dispatch",
    "nanlm_mth_hub",
    "nanlm_mth_tree",
    "nanlm_mth_leaf_create",
    "nanlm_mth_leaf_get",
    "nanlm_mth_pair",
    "nanlm_mth_leaf_verify",
)
SAFE_TOOL_SET = frozenset(SAFE_TOOLS)
DENIED_TOOLS = (
    "nanlm_learning_sync",
    "nanlm_memory_merge",
    "nanlm_head_candidate",
    "nanlm_head_learn_candidate",
    "nanlm_head_candidate_evaluate",
    "nanlm_tree_mutate",
)
# These tools are useful for observation/ingestion, but the model may never
# request their local promotion path. approved=true is stripped before dispatch.
PROMOTION_ARGUMENT_TOOLS = frozenset({
    "nanlm_learning_cycle",
    "nanlm_wordnet_ingest",
})


def _safe_arguments(name, arguments):
    """Return handler arguments with every model-supplied promotion switch removed."""
    safe = dict(arguments) if isinstance(arguments, dict) else {}
    safe.pop("approved", None)
    safe.pop("auto_sync", None)
    return safe


class Provider:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def complete(self, messages, temperature=0.7, max_tokens=2048,
                 timeout=240, purpose=""):
        from types import SimpleNamespace
        result = self.request(messages, tools=None, temperature=temperature,
                              max_tokens=max_tokens, timeout=timeout)
        message = result["choices"][0]["message"]
        return SimpleNamespace(text=message.get("content") or "",
                               provider="ollama-openai-compat",
                               model=result.get("model", self.model))

    def request(self, messages, *, tools=None, temperature=0.2,
                max_tokens=1200, timeout=300):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "max_tokens": max(1, min(int(max_tokens), 4096)),
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(524288))


class Context:
    def __init__(self, root, provider, authority_url="", issuer_key=""):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "provider",
            "chat_learning": True,
            "learning_require_approval": True,
            "learning_auto_sync": False,
            "nannet_auth_url": authority_url,
            "nannet_issuer_public_key": issuer_key,
            "wordnet_archive": os.environ.get("NANLM_WORDNET_ARCHIVE", ""),
        }
        self.llm = provider
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs


def _tool_schemas(ctx):
    schemas = []
    for name in SAFE_TOOLS:
        tool = ctx.tools.get(name)
        if not tool:
            continue
        schema = json.loads(json.dumps(tool["schema"]))
        schema["description"] = str(schema.get("description", ""))[:700]
        properties = schema.get("parameters", {}).get("properties") or {}
        if name in PROMOTION_ARGUMENT_TOOLS:
            properties.pop("approved", None)
            required = schema.get("parameters", {}).get("required")
            if isinstance(required, list) and "approved" in required:
                required.remove("approved")
        # Keep parameter descriptions short; the model needs the shape, not prose.
        for prop in properties.values():
            if isinstance(prop, dict) and isinstance(prop.get("description"), str):
                prop["description"] = prop["description"][:240]
        schemas.append({"type": "function", "function": schema})
    return schemas


def _short(value, limit=4000):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _execute(ctx, name, arguments):
    if name in DENIED_TOOLS or name not in SAFE_TOOL_SET or name not in ctx.tools:
        return {"error": "tool is not available in this session", "tool": name}
    try:
        arguments = _safe_arguments(name, arguments)
        raw = ctx.tools[name]["handler"](arguments)
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)[:1000],
                "tool": name}


def _assistant_message(message):
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}),
    }


def _summary_request(provider, transcript, calls):
    compact = []
    for item in transcript[-16:]:
        role = item.get("role")
        content = item.get("content") or ""
        compact.append({"role": role, "content": content[:1200]})
    prompt = {
        "task": ("Summarize this tool-learning session for the operator. Return "
                 "plain text with sections: What worked, What failed, Tools "
                 "learned, Free-chat learning observations, Next experiments. "
                 "Do not claim canonical commits or proofs."),
        "tool_calls": calls,
        "transcript_tail": compact,
    }
    result = provider.request(
        [{"role": "system", "content": "You are Gemma reviewing a completed "
          "open-ended HERMES tool-learning session. Be concrete and honest."},
         {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        tools=None, temperature=0.2, max_tokens=1200)
    return result["choices"][0]["message"].get("content") or ""


def _load_session_instructions():
    try:
        context = session_context.open_session(persist=False)
        return context.instructions_block()
    except Exception:
        return ""


def _run_session(ctx, provider, tools, args, *, root, nannet_info):
    system = (
        "You are Gemma steering an open-ended HERMES plugin tool-learning "
        "session. You have real tools. Use them; do not merely describe them. "
        "The NANNET pipeline is paired to a local authority and the whole "
        "pipeline is intact, but canonical merging/promotion is disabled for "
        "this session. Do not claim canonical commits, leases, proofs or a "
        "canonical merge. After each tool result, inspect it and choose the "
        "next useful test. Prefer small, bounded calls. When done, give a final "
        "summary of what you learned about using the tools and what still "
        "fails.\n\n" + _load_session_instructions()
    )
    user = (
        "Initial test goal: learn to use this plugin safely with NANNET "
        "authorization active. Cover as many of these as useful, in your own "
        "order: inspect the permanent session context; map/audit the 91-kernel "
        "MTH function tree; inspect the MTH vector catalog and runtime; create "
        "two 256-value vector data leaves (aliases gemma-v1 and gemma-v2); pair "
        "them with connect and with add; inspect a leaf and verify its NaN16 "
        "image; run two free-chat learning turns (a teaching statement and a "
        "question); inspect learning status and scratch. Do not call a merge or "
        "promotion tool. Then summarize what worked, what failed, and the next "
        "experiment. Use aliases and small arguments."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    calls = []
    transcript = list(messages)
    no_tool_turns = 0
    for turn in range(args.turns):
        response = provider.request(messages, tools=tools, max_tokens=1400)
        message = response["choices"][0]["message"]
        assistant = _assistant_message(message)
        messages.append(assistant)
        transcript.append(assistant)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            no_tool_turns += 1
            if no_tool_turns >= 2 or turn == args.turns - 1:
                break
            messages.append({"role": "user", "content":
                             "Continue with one concrete tool call, or provide your final summary now."})
            transcript.append(messages[-1])
            continue
        no_tool_turns = 0
        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                arguments = {}
            arguments = _safe_arguments(name, arguments)
            result = _execute(ctx, name, arguments)
            call_id = call.get("id") or ("call_%d_%d" % (turn, index))
            calls.append({
                "turn": turn,
                "name": name,
                "arguments": arguments,
                "result_summary": _short(result, 1200),
                "failed": bool(isinstance(result, dict) and result.get("error")),
            })
            tool_message = {"role": "tool", "tool_call_id": call_id,
                            "content": _short(result, 4000)}
            messages.append(tool_message)
            transcript.append(tool_message)
    final = _summary_request(provider, transcript, calls)

    used = sorted({call["name"] for call in calls})
    failed = [call for call in calls if call["failed"]]
    record = {
        "schema": "nanlm.gemma-tool-learning-session/1",
        "model": args.model,
        "state_dir": str(root),
        "turns_requested": args.turns,
        "tool_calls": calls,
        "tool_calls_total": len(calls),
        "tool_calls_failed": len(failed),
        "tools_used": used,
        "distinct_tools_used": len(used),
        "final_summary": final,
        "transcript": transcript,
        "nannet": nannet_info,
        "boundaries": {
            "local_unverified": False,
            "nannet_configured": True,
            "nannet_pipeline_intact": True,
            "canonical_merge_enabled": False,
            "canonical_commit": False,
            "proof_claim": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    markdown = output.with_suffix(".md")
    lines = [
        "# Gemma tool-learning session (initial test)",
        "",
        "- model: `" + args.model + "`",
        "- NANNET authority: `" + str(nannet_info.get("authority_url")) + "`",
        "- pairing mode: `" + str(nannet_info.get("authorization_mode")) + "`",
        "- canonical merge: disabled",
        "- tool calls: " + str(len(calls)),
        "- failed calls: " + str(len(failed)),
        "- distinct tools used: " + str(len(used)),
        "- state: `" + str(root) + "`",
        "",
        "## Tool calls",
        "",
    ]
    for call in calls:
        status = "FAIL" if call["failed"] else "PASS"
        lines.append("- **" + status + "** `" + call["name"] + "` turn " +
                     str(call["turn"]) + ": " + call["result_summary"][:300])
    lines.extend(["", "## Gemma's final summary", "", final, ""])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "markdown": str(markdown),
        "tool_calls": len(calls), "failed": len(failed),
        "distinct_tools": len(used), "tools": used,
        "nannet": nannet_info,
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("NANLM_REVIEW_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("NANLM_REVIEW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if not 1 <= args.turns <= 30:
        raise SystemExit("--turns must be 1..30")

    reset_coordinator_cache()
    reset_leaf_cache()
    root = Path(args.state_dir).expanduser() if args.state_dir else Path(
        os.environ.get("TMPDIR", "/tmp")) / "nanlm-gemma-tool-learning"
    root.mkdir(parents=True, exist_ok=True)
    provider = Provider(args.url, args.model)

    authority = PairingAuthority(root / "authority.sqlite", secrets.token_bytes(32))
    authority_http = make_authority(authority, "127.0.0.1", 0)
    authority_thread = threading.Thread(target=authority_http.serve_forever, daemon=True)
    authority_thread.start()
    authority_url = "http://127.0.0.1:%d" % authority_http.server_address[1]
    ctx = Context(root, provider, authority_url, authority.issuer_public_key())
    register(ctx)
    tools = _tool_schemas(ctx)
    master_b64 = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    pairing_code = authority.create_pairing_code(
        ["universe:read", "training:execute"], ttl_seconds=600)["pairing_code"]

    def secret(name):
        return master_b64 if name == "NANNET_MASTER_KEY" else ""

    try:
        with patch.object(hermes_secrets, "get_secret", side_effect=secret), \
             patch.object(auth_tools, "get_secret", side_effect=secret):
            init = json.loads(ctx.tools["nanlm_auth_init"]["handler"]({}))
            paired = json.loads(ctx.tools["nanlm_auth_pair"]["handler"](
                {"pairing_code": pairing_code}))
            nannet_info = {
                "authority_url": authority_url,
                "issuer_public_key": authority.issuer_public_key()[:16] + "...",
                "installation_id": init.get("installation_id"),
                "paired": paired.get("paired"),
                "scopes": paired.get("scopes"),
                "authorization_mode": "nannet",
                "canonical_merge_enabled": False,
            }
            return _run_session(ctx, provider, tools, args, root=root,
                                nannet_info=nannet_info)
    finally:
        authority_http.shutdown()
        authority_http.server_close()
        authority_thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
