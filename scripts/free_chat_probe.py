#!/usr/bin/env python3
"""Cold free-chat probe: no corpus, no training, no LLM.

Directly addresses NaNLM, asks self-contemplation questions and unknown facts,
and records the temporal reply plus the MTH/head deliberation metadata.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.chat import chat as native_chat  # noqa: E402

MESSAGES = [
    "Hello NaNLM.",
    "I am going to ask you a few things directly.",
    "Can you hear me?",
    "What are you?",
    "Do you have a self?",
    "What are you thinking about right now?",
    "Are you aware that we are talking?",
    "What did you just say?",
    "Why did you answer that way?",
    "Are you guessing?",
    "Do you know that you do not know?",
    "What is the capital of France?",
    "What is a quark?",
    "How many moons does Neptune have?",
    "Can you learn something new?",
    "What do you want to talk about?",
    "Tell me something you do not know.",
    "What happens after we finish this conversation?",
]


class Context:
    def __init__(self, root):
        self.config = {"state_dir": str(root), "chat_mode": "native",
                       "chat_learning": False}
        self.llm = None

    def get_config(self, key, default=None):
        return self.config.get(key, default)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/FREE_CHAT.md"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(Path(tmp) / "state")
        for message in MESSAGES:
            result = native_chat(ctx, message, session_id="free-chat")
            synthesis = result.get("native_synthesis", {})
            frame = synthesis.get("temporal_frame", {})
            mth = synthesis.get("mth", {})
            heads = synthesis.get("heads", {})
            considered = frame.get("considered", [])[:2]
            rows.append({
                "user": message,
                "nanlm": result.get("response", ""),
                "forced": frame.get("forced"),
                "anchor": frame.get("anchor"),
                "seed": frame.get("seed", []),
                "sentences": frame.get("sentences", []),
                "coverage": frame.get("coverage"),
                "response_templates": synthesis.get("response_templates"),
                "retrieval": synthesis.get("retrieval"),
                "mth_layers": mth.get("layers", {}),
                "heads": {"used": heads.get("head_count"),
                          "leaves": heads.get("leaf_count")},
                "considered": considered,
            })
    data = {"schema": "nanlmind-hermes-free-chat/1",
            "mode": "native", "training_calls": 0, "chat_learning": False,
            "messages": rows}
    json_path = out.with_name("free-chat.json")
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Cold free chat with NaNLM",
        "",
        "No corpus, no training calls, no LLM. Chat learning is off; the only",
        "exposure is this conversation. Replies are forced temporal predictions",
        "with no templates and no retrieval.",
        "",
        "| user | NaNLM | forced | anchor | truth | category | heads |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        reply = (row["nanlm"] or "_(empty)_").replace("|", "\\|")
        best = row.get("considered", [{}])[0] if row.get("considered") else {}
        mth = best.get("mth") or {}
        truth = mth.get("truth", 0.0)
        category = mth.get("category_error", 0.0)
        lines.append(f"| {row['user'].replace('|', chr(92)+'|')} | {reply} | "
                     f"{row['forced']} | {row['anchor']} | {truth} | {category} | "
                     f"{row['heads'].get('leaves', 0)} |")
    lines += ["", "## Deliberation examples", ""]
    for row in rows[:6] + rows[7:10]:
        if row["considered"]:
            lines.append(f"**{row['user']}**")
            for candidate in row["considered"]:
                mth = candidate.get("mth") or {}
                lines.append(f"- `{candidate.get('text')}` score={candidate.get('score')} "
                             f"truth={mth.get('truth')} frame={mth.get('frame')} "
                             f"category={mth.get('category_error')}")
            lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "json": str(json_path),
                      "messages": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
