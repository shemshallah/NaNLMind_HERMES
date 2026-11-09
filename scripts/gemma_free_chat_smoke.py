#!/usr/bin/env python3
"""Live Gemma free-chat learning smoke over the plugin tool path.

This exercises the shipped permanent session context, local-unverified chat
authorization, provider answer, structured reflection, L0-L9 layered scratch
observation, and status inspection. It does not configure NANNET and therefore
cannot authorize any remote router/lease/merge action.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import register  # noqa: E402
from nanlmind_hermes.layered_learning import reset_coordinator_cache  # noqa: E402


class Provider:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def complete(self, messages, temperature=0.7, max_tokens=2048,
                 timeout=240, purpose=""):
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "max_tokens": max(1, min(int(max_tokens), 4096)),
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read(262144))
        text = result["choices"][0]["message"]["content"]
        from types import SimpleNamespace
        return SimpleNamespace(text=text, provider="ollama-openai-compat",
                               model=result.get("model", self.model))


class Context:
    def __init__(self, root, provider):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "provider",
            "chat_learning": True,
            "wordnet_archive": os.environ.get("NANLM_WORDNET_ARCHIVE", ""),
        }
        self.llm = provider
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs


def call(ctx, name, params):
    raw = ctx.tools[name]["handler"](params)
    return json.loads(raw) if isinstance(raw, str) else raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get(
        "NANLM_REVIEW_URL", "http://127.0.0.1:11434/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get(
        "NANLM_REVIEW_MODEL", "gemma4:31b-cloud"))
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--output", default=str(ROOT / "docs" / "GEMMA_FREE_CHAT_SMOKE.json"))
    args = parser.parse_args()

    reset_coordinator_cache()
    root = Path(args.state_dir).expanduser() if args.state_dir else Path(
        os.environ.get("TMPDIR", "/tmp")) / "nanlm-gemma-free-chat"
    root.mkdir(parents=True, exist_ok=True)
    ctx = Context(root, Provider(args.url, args.model))
    register(ctx)

    turns = []
    for message in (
        "A river is water.",
        "Water can flow, and rivers carry water.",
        "What is a river?",
    ):
        result = call(ctx, "nanlm_chat", {
            "message": message, "session_id": "gemma-free-chat",
        })
        layered = result.get("layered_learning", {})
        turns.append({
            "message": message,
            "response": result.get("response"),
            "provider": result.get("provider"),
            "model": result.get("model"),
            "reflection_parse_ok": (result.get("reflection") or {}).get("parse_ok"),
            "reflection_discoveries": (result.get("reflection") or {}).get(
                "proposed_discoveries"),
            "native_learning_status": (result.get("native_learning") or {}).get(
                "status"),
            "layered_learning": {
                "layers": layered.get("layers"),
                "relations": layered.get("relations"),
                "scratch_records": layered.get("scratch_records"),
                "head_balance": layered.get("head_balance"),
                "canonical_commit": layered.get("canonical_commit"),
            },
            "session_context_loaded": (result.get("session_context") or {}).get(
                "loaded"),
            "authorization": result.get("authorization"),
        })

    learning = call(ctx, "nanlm_learning_status", {})
    status = call(ctx, "nanlm_status", {})
    record = {
        "schema": "nanlm.gemma-free-chat-smoke/1",
        "model": args.model,
        "state_dir": str(root),
        "turns": turns,
        "learning_status": {
            "turns": learning.get("turns"),
            "rows": learning.get("rows"),
            "head_ensemble": learning.get("head_ensemble"),
            "scratch": learning.get("scratch"),
            "scratch_verify": learning.get("scratch_verify"),
            "concept_saturation": learning.get("concept_saturation"),
            "dictionary_hw4": learning.get("dictionary_hw4"),
        },
        "session_context": {
            "loaded": (status.get("session_context") or {}).get("loaded"),
            "scope": (status.get("session_context") or {}).get("scope"),
            "generation": (status.get("session_context") or {}).get("generation"),
            "document_ids": (status.get("session_context") or {}).get("document_ids"),
        },
        "boundaries": {
            "local_unverified": True,
            "nannet_configured": False,
            "canonical_commit": False,
            "measured_native_training": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "turns": len(turns),
        "layers_per_turn": [t["layered_learning"]["layers"] for t in turns],
        "relations_per_turn": [t["layered_learning"]["relations"] for t in turns],
        "session_context_loaded": record["session_context"]["loaded"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
