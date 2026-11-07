#!/usr/bin/env python3
"""Run a human-readable Hermes demo: chat, native training, and CLOCKED.

This is the transcript demo, not the exhaustive tool-coverage demo. Native
chat is the default; no language-model provider is required. The real native
``nanlm-universal`` binary performs training, and the deployed NANNET / CLOCKED
/ head-cortex service supplies live integration.

    python scripts/hermes_demo.py \
      --native-binary ~/NaNLMind/build/nanlm-universal \
      --nannet-url https://nannet.apps.mistral.run \
      --pairing-code nnh_... \
      --clocked-token-file ~/.config/nanlmind/clocked-token \
      --llm-url http://127.0.0.1:11434/v1/chat/completions \
      --llm-model gemma4:31b-cloud \
      --output docs/HERMES_DEMO.md

The JSON evidence is written next to the Markdown report.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanlmind_hermes import register  # noqa: E402
from nanlmind_hermes import auth_tools, clocked_tools, hermes_secrets  # noqa: E402


class OpenAICompatProvider:
    """Minimal OpenAI-compatible chat provider, including Ollama cloud models."""

    def __init__(self, url: str, model: str = "gemma4:31b-cloud"):
        self.url = url
        self.model = model

    def complete(self, messages, temperature=0.7, max_tokens=512, timeout=180, purpose=""):
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "max_tokens": max(1, min(int(max_tokens), 4096)),
            "stream": False,
        }).encode()
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"]
        return SimpleNamespace(text=text, provider="llama.cpp", model=result.get("model", self.model))


class HybridProvider:
    """Real chat model; deterministic structured teacher for reproducible training."""

    def __init__(self, chat: OpenAICompatProvider):
        self.chat = chat

    def complete(self, messages, **kwargs):
        purpose = kwargs.get("purpose", "")
        if purpose == "nanlmind-hermes-multiresolution-reflection":
            # One explicit supervised relation so chat learning has a real,
            # bounded native update to report; the chat text itself is model-generated.
            return SimpleNamespace(
                text=json.dumps({"discoveries": [{
                    "kind": "relation", "subject": "river", "predicate": "flows",
                    "object": "The river flows.", "resolution": "lexical",
                    "polarity": "supports", "confidence": 0.9,
                    "evidence": {"demo": "chat-reflection"},
                }]}),
                provider="demo-reflection", model="deterministic/1")
        if purpose == "nanlmind-corpus-generation":
            return SimpleNamespace(
                text="\n".join(
                    f"demo corpus record {i}: the river flows through the valley and birds fly above the water"
                    for i in range(1, 65)),
                provider="demo-teacher", model="deterministic/1")
        if purpose.startswith("nanlmind-relation-teacher-"):
            return SimpleNamespace(
                text=json.dumps({"records": [
                    {"subject": f"demo:{purpose[-3:]}", "predicate": "supports",
                     "object": "The river flows.", "polarity": "supports",
                     "confidence": 0.9, "evidence": "demo teacher"},
                    {"subject": f"demo:{purpose[-3:]}", "predicate": "refutes",
                     "object": "The river is dry.", "polarity": "refutes",
                     "confidence": 0.8, "evidence": "demo teacher"},
                    {"subject": f"demo:{purpose[-3:]}", "predicate": "supports",
                     "object": "Birds fly above the water.", "polarity": "supports",
                     "confidence": 0.85, "evidence": "demo teacher"},
                    {"subject": f"demo:{purpose[-3:]}", "predicate": "refutes",
                     "object": "Boats move upstream.", "polarity": "refutes",
                     "confidence": 0.75, "evidence": "demo teacher"},
                    {"subject": f"demo:{purpose[-3:]}", "predicate": "supports",
                     "object": "The sun sets behind the hills.", "polarity": "supports",
                     "confidence": 0.8, "evidence": "demo teacher"},
                    {"subject": f"demo:{purpose[-3:]}", "predicate": "refutes",
                     "object": "The valley is empty.", "polarity": "refutes",
                     "confidence": 0.7, "evidence": "demo teacher"},
                ]}),
                provider="demo-teacher", model="deterministic/1")
        if purpose == "nanlm-hermes-clocked-mode-selection":
            user = messages[1]["content"] if len(messages) > 1 else ""
            if "PII" in user or "privacy" in user.lower():
                mode = "CLOCKED_PRIVACY_AUDIT"
            elif "compare" in user.lower():
                mode = "CLOCKED_COMPARE"
            else:
                mode = "CLOCKED_EXTRACT"
            return SimpleNamespace(text=json.dumps({"mode": mode, "reason": "demo policy"}),
                                   provider="demo-policy", model="deterministic/1")
        return self.chat.complete(messages, **kwargs)


class DemoContext:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.tools = {}
        self.command = None

    def get_config(self, name, default=None):
        return self.config.get(name, default)

    def register_tool(self, name, toolset=None, handler=None, schema=None):
        self.tools[name] = handler

    def register_command(self, name, handler, description=None):
        self.command = handler


def post_json(url, payload, token=""):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def compact(value, limit=1600):
    text = json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", default=str(Path.home() / "NaNLMind/build/nanlm-universal"))
    parser.add_argument("--nannet-url", default="https://nannet.apps.mistral.run")
    parser.add_argument("--pairing-code", required=True)
    parser.add_argument("--clocked-token-file", required=True)
    parser.add_argument("--llm-url", default="http://127.0.0.1:11434/v1/chat/completions")
    parser.add_argument("--llm-model", default="gemma4:31b-cloud")
    parser.add_argument("--chat-mode", default="native", choices=["native", "auto", "provider"])
    parser.add_argument("--phone-clocked", action="store_true",
                        help="opt in to CLOCKED during the three-way call (uses gateway credits)")
    parser.add_argument("--output", default=str(ROOT / "docs/HERMES_DEMO.md"))
    args = parser.parse_args(argv)

    native = Path(args.native_binary).expanduser()
    if not native.is_file():
        raise SystemExit("native binary not found: %s" % native)
    clocked_token = Path(args.clocked_token_file).read_text(encoding="utf-8").strip()
    root = Path(tempfile.mkdtemp(prefix="hermes-demo-"))
    master = secrets.token_bytes(32)
    master_b64 = base64.urlsafe_b64encode(master).rstrip(b"=").decode()
    base = args.nannet_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/issuer", timeout=20) as response:
        issuer_key = json.loads(response.read(32768))["issuer_public_key"]
    config = {
        "state_dir": str(root / "state"),
        "runner_command": [str(native)],
        "nannet_auth_url": base,
        "nannet_issuer_public_key": issuer_key,
        "clocked_gateway_url": base,
        "nanlm_head_cortex_url": base,
        "chat_learning": True,
        "chat_mode": args.chat_mode,
    }
    ctx = DemoContext(config)
    ctx.llm = HybridProvider(OpenAICompatProvider(args.llm_url, args.llm_model))
    register(ctx)

    # Register a scoped Hermes device for live head-cortex calls.
    device_secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    registered = post_json(base + "/v1/hermes/register", {
        "device_secret": device_secret, "consent": True,
        "terms_version": "nannet-device-registration-v1",
        "system_report": {"platform": "Hermes", "app_version": "1.3.1"},
    })
    device_token = registered["token"]
    secrets_map = {
        "NANNET_MASTER_KEY": master_b64,
        "CLOCKED_GATEWAY_TOKEN": clocked_token,
        "NANLM_DEVICE_TOKEN": device_token,
    }

    def call(name, params):
        raw = ctx.tools[name](params)
        return json.loads(raw) if isinstance(raw, str) else raw

    transcript = {"chat": {}, "training": {}, "clocked": {}}
    with patch.object(hermes_secrets, "get_secret", side_effect=lambda n: secrets_map.get(n, "")), \
         patch.object(auth_tools, "get_secret", side_effect=lambda n: secrets_map.get(n, "")), \
         patch.object(clocked_tools, "get_secret", side_effect=lambda n: secrets_map.get(n, "")):
        call("nanlm_auth_init", {})
        call("nanlm_auth_pair", {"pairing_code": args.pairing_code})
        transcript["nannet"] = call("nanlm_auth_status", {})

        # 1. Native chat: teach one relation, then ask for it back.
        teaching = call("nanlm_chat", {
            "message": "A river is water.",
            "session_id": "hermes-demo",
        })
        reply = call("nanlm_chat", {
            "message": "What is a river?",
            "session_id": "hermes-demo",
        })
        learning = teaching.get("native_learning") or {}
        transcript["chat"] = {
            "mode": args.chat_mode,
            "teach_response": teaching.get("response"),
            "ask_response": reply.get("response"),
            "provider": reply.get("provider"),
            "model": reply.get("model"),
            "transcript_count": call("nanlm_chat_history", {"session_id": "hermes-demo"}).get("count"),
            "native_synthesis": reply.get("native_synthesis"),
            "scratch": reply.get("scratch"),
            "native_learning": {
                "status": learning.get("status"),
                "boolean_foundation_status": (learning.get("boolean_foundation") or {}).get("status"),
                "changed_heads": learning.get("changed_heads"),
                "mirror_pair_updates": learning.get("mirror_pair_updates"),
                "hyperedges_added": learning.get("hyperedges_added"),
                "records": (learning.get("corpus") or {}).get("records"),
                "state_sha256_before": learning.get("native_state_sha256_before"),
                "state_sha256_after": learning.get("native_state_sha256_after"),
                "metric_scope": learning.get("metric_scope"),
            },
        }

        # 2. Native training with the real binary.
        logic = call("nanlm_logic_session", {
            "topic": "Boolean foundations", "records_per_stage": 3,
            "epochs": 1, "max_samples_per_head": 16,
        })
        training = logic.get("training", {})
        rounds = []
        for item in logic.get("rounds", []):
            metrics = item.get("resolution_metrics") or {}
            rounds.append({
                "round": item.get("round"),
                "records": item.get("records"),
                "changed_heads": item.get("changed_heads"),
                "mirror_pair_updates": item.get("mirror_pair_updates"),
                "hyperedges_added": item.get("hyperedges_added"),
                "heldout_accuracy": {
                    key: value.get("relation_polarity_accuracy")
                    for key, value in metrics.items()
                },
                "heldout_baseline": {
                    key: value.get("majority_baseline")
                    for key, value in metrics.items()
                },
            })
        transcript["training"] = {
            "mode": logic.get("mode"),
            "corpus_records": logic.get("corpus", {}).get("record_count"),
            "boolean_foundation": training.get("boolean_foundation"),
            "rounds": rounds,
            "final_state_digest": training.get("state_digest_after"),
            "note": "Round 1 made the structural updates. Later rounds reuse the same deterministic teacher data, so changed_heads=0 is expected.",
        }

        # 3. Three-way phone call: user -> Hermes -> NaNLM -> user + Hermes.
        # The call routing kernel is a raw NaN16 program (nan/phone_call.nan).
        # CLOCKED is opt-in (--phone-clocked) so a normal demo spends no
        # gateway credits; the call itself never requires CLOCKED.
        call("nanlm_phone_call", {
            "action": "start", "session_id": "phone-demo",
            "message": "A river is water.", "addressee": "hermes", "rounds": 1,
            "clocked": bool(args.phone_clocked)})
        call("nanlm_phone_call", {
            "action": "say", "session_id": "phone-demo",
            "message": "What is a river?", "addressee": "nanlm"})
        phone = call("nanlm_phone_call", {
            "action": "run", "session_id": "phone-demo", "rounds": 2})
        clocked_results = []
        lanes: dict[str, int] = {}
        if args.phone_clocked:
            call_window = call("nanlm_clocked_scratch", {
                "scratch_id": phone.get("scratch_id", "call-phone-demo")})
            for entry in call_window.get("entries", []):
                lane = entry.get("lane", "")
                lanes[lane] = lanes.get(lane, 0) + 1
            clocked_window = call("nanlm_clocked_scratch", {
                "scratch_id": phone.get("scratch_id", "call-phone-demo"),
                "lane": "clocked_result"})
            for entry in clocked_window.get("entries", [])[-3:]:
                payload = entry.get("payload", {})
                clocked_results.append({
                    "endpoint": payload.get("endpoint"),
                    "status": payload.get("status"),
                    "result_type": payload.get("result_type"),
                    "result_excerpt": json.dumps(payload.get("result"), sort_keys=True)[:280],
                })
        transcript["phone"] = {
            "session_id": phone.get("session_id"),
            "status": phone.get("status"),
            "turn": phone.get("turn"),
            "scratch_id": phone.get("scratch_id"),
            "default_path": phone.get("default_path"),
            "clocked": phone.get("clocked"),
            "kernel": phone.get("kernel"),
            "call_window_lanes": dict(sorted(lanes.items())),
            "clocked_results": clocked_results,
            "history": phone.get("history", []),
        }

        # 4. CLOCKED plan -> execute -> live head cortex -> isolated candidate.
        pii_sources = [{"id": "source-a", "text": "Contact demo@example.com for details."}]
        pii_plan = call("nanlm_clocked_plan", {
            "question": "Check PII handling in the source", "sources": pii_sources,
            "scratch_id": "hermes-live-clocked"})
        pii_run = call("nanlm_clocked_execute", {
            "question": "Check PII handling in the source", "sources": pii_sources,
            "scratch_id": "hermes-live-clocked", "approved": True,
            "plan_sha256": pii_plan["approval_digest"]})
        claim_sources = [{"id": "source-b", "text": "The river flows. Birds fly above the water."}]
        claim_plan = call("nanlm_clocked_plan", {
            "question": "Extract the supported claims", "sources": claim_sources,
            "scratch_id": "hermes-live-clocked"})
        claim_run = call("nanlm_clocked_execute", {
            "question": "Extract the supported claims", "sources": claim_sources,
            "scratch_id": "hermes-live-clocked", "approved": True,
            "plan_sha256": claim_plan["approval_digest"]})
        scratch = call("nanlm_clocked_scratch", {"scratch_id": "hermes-live-clocked"})
        head = call("nanlm_head_process", {
            "question": "Validate the extracted claims", "scratch_id": "hermes-live-clocked",
            "purpose": "ANSWER_VALIDATION", "draft": "The river flows."})
        head_response = head.get("head_response", {})
        base_digest = head_response.get("provenance", {}).get("state_digest")
        candidate_id = "demo-candidate-" + uuid.uuid4().hex[:8]
        candidate = call("nanlm_head_candidate", {
            "mutation_id": candidate_id, "head_index": 42, "patch_mask": 1,
            "base_topology_digest": base_digest, "approved": True,
            "scratch_id": "hermes-live-clocked"})
        learned = call("nanlm_head_learn_candidate", {
            "mutation_id": "demo-learn-" + uuid.uuid4().hex[:8],
            "base_topology_digest": base_digest, "approved": True,
            "scratch_id": "hermes-live-clocked", "epochs": 1})
        evaluation = call("nanlm_head_candidate_evaluate", {
            "mutation_id": candidate_id,
            "records": [{"resolution": 5, "polarity": "+", "subject": "river",
                         "predicate": "flows", "object": "The river flows.",
                         "source_digest": hashlib.sha256(b"heldout").hexdigest()}],
            "approved": True, "scratch_id": "hermes-live-clocked"})
        transcript["clocked"] = {
            "pii_plan_digest": pii_plan["approval_digest"],
            "pii_result_count": pii_run.get("result_count"),
            "claim_plan_digest": claim_plan["approval_digest"],
            "claim_result_count": claim_run.get("result_count"),
            "scratch_lanes": sorted({row["lane"] for row in scratch.get("entries", [])}),
            "head_truth_state": head_response.get("truth_state"),
            "head_state_digest": base_digest,
            "candidate": candidate.get("candidate"),
            "learned": learned.get("candidate"),
            "evaluation": evaluation.get("evaluation"),
        }
        call("nanlm_auth_revoke", {})

    report = {
        "schema": "nanlmind-hermes-demo-transcript/1",
        "version": "1.3.1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nannet_url": base,
        "native_binary": "nanlm-universal",
        "chat_mode": args.chat_mode,
        "llm_url": args.llm_url,
        "llm_model": args.llm_model,
        "transcript": transcript,
    }
    json_path = Path(args.output).with_name("hermes-demo.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Hermes demo transcript",
        "",
        "Generated by `scripts/hermes_demo.py` at **1.3.1**. NaNLM chat runs through",
        "its native parser/synthesizer (`chat_mode=%s`); no language provider is" % args.chat_mode,
        "required for that path. The three-way call uses `%s` as the Hermes" % args.llm_model,
        "bridge when reachable. Training uses the real `nanlm-universal` binary;",
        "CLOCKED and head-cortex calls use the deployed `%s` service." % base,
        "",
        "## 1. Native chat and online relation learning",
        "",
        "**User teaches:** `A river is water.`",
        "",
        "**NaNLMind replies:**",
        "",
        "```text",
        transcript["chat"].get("teach_response", ""),
        "```",
        "",
        "**User asks:** `What is a river?`",
        "",
        "**NaNLMind replies:**",
        "",
        "```text",
        transcript["chat"].get("ask_response", ""),
        "```",
        "",
        "**Reported result:**",
        "",
        "```json",
        compact({
            "provider": transcript["chat"].get("provider"),
            "model": transcript["chat"].get("model"),
            "chat_mode": transcript["chat"].get("mode"),
            "transcript_count": transcript["chat"].get("transcript_count"),
            "native_synthesis": transcript["chat"].get("native_synthesis"),
            "scratch": transcript["chat"].get("scratch"),
            "native_learning": transcript["chat"].get("native_learning"),
        }),
        "```",
        "",
        "## 2. Native training",
        "",
        "`nanlm_logic_session` ran the real Boolean foundation and relation learner.",
        "",
        "```json",
        compact(transcript["training"], 4000),
        "```",
        "",
        "## 3. Three-way phone call: User ↔ Hermes ↔ NaNLM",
        "",
        "Default path: `user -> Hermes -> NaNLM -> user + Hermes`. The call",
        "routing kernel is a raw NaN16 program (`nan/phone_call.nan`), executed",
        "on the NaNVM0 boundary; the transcript below records the kernel plan in",
        "the `phone_event` lane. CLOCKED in the call is opt-in; this run used",
        "`--phone-clocked` only if the section below reports `clocked: true`,",
        "so a default run spends no gateway credits.",
        "",
        "```json",
        compact(transcript["phone"], 6000),
        "```",
        "",
        "## 4. CLOCKED and live head cortex",
        "",
        "`nanlm_clocked_plan` → `nanlm_clocked_execute` → `nanlm_head_process` →",
        "isolated candidate creation, weak-supervision learning, and held-out",
        "evaluation.",
        "",
        "```json",
        compact(transcript["clocked"], 5000),
        "```",
        "",
        "## Reading the result",
        "",
        "- Native chat is a relation-memory synthesizer, not an LLM. Its replies",
        "  are intentionally simple until the learned relation set grows.",
        "- Boolean foundation is a hard gate: exact truth-table checks across all",
        "  256 heads, not a language-quality claim.",
        "- Relation metrics are held-out polarity accuracy on the generated",
        "  resolution split; compare them to the reported baseline.",
        "- `UNKNOWN` from the head cortex is an abstention, not `FALSE`.",
        "- Candidate metrics are review signals only. `promotion_allowed` remains",
        "  `false`; canonical state is never modified by the demo.",
        "",
    ]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": args.output, "json": str(json_path),
                      "chat_chars": len(transcript["chat"].get("ask_response", "")),
                      "training_status": transcript["training"].get("boolean_foundation", {}).get("status")
                      if isinstance(transcript["training"].get("boolean_foundation"), dict) else None,
                      "clocked_lanes": transcript["clocked"].get("scratch_lanes"),
                      "head_truth_state": transcript["clocked"].get("head_truth_state"),
                      "phone_turns": transcript["phone"].get("turn")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
