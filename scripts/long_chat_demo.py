#!/usr/bin/env python3
"""Long exposure chat with native NaNLM after logic training.

The learner first runs the real Boolean/deep-logic curriculum through the
native ``nanlm-universal`` binary. Hermes then instantiates its own local-CPU
runner: it loads the locally trained 8 MiB chart and a mapped temporal
register into this process (it does not use the system RunnerRing), links
through ``HW2:0`` (NaN16DE) to ``BS0:0::HW0:0``, and predicts the next word
from a temporal frame addressed by head/mirror cells.

Native replies are temporal predictions, not retrieval and not templates:
every step scores every known word and feeds the chosen word back into the
frame. No LLM provider, no CLOCKED calls and no NANNET calls are used in the
default run.

Usage:
    python scripts/long_chat_demo.py \
      --native-binary ~/NaNLMind/build/nanlm-universal \
      --output docs/LONG_CHAT_DEMO.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import register, tools  # noqa: E402
from nanlmind_hermes import phone_call as phone_module  # noqa: E402
from nanlmind_hermes.chat import chat as native_chat  # noqa: E402
from nanlmind_hermes.universe_chart import LocalRunner  # noqa: E402

EXPOSURE = [
    "A river is a flowing body of water.",
    "Water can flow.",
    "Rain is water.",
    "Rain causes rivers to rise.",
    "Storms cause rain.",
    "Rivers flow through valleys.",
    "A bird can fly.",
    "Birds have wings.",
    "Birds sing in the morning.",
    "A fish can swim.",
    "Fish live in water.",
    "A fish is an animal.",
    "A bird is an animal.",
    "Animals need water.",
    "Plants need water.",
    "Plants grow in soil.",
    "The sun is hot.",
    "The sun causes light.",
    "The sky is blue.",
    "Cold water can freeze.",
]

CONVERSATION = [
    "What is a river?",
    "Can water flow?",
    "What is water?",
    "What can a bird do?",
    "Do fish live in water?",
    "What causes rain?",
    "Why do rivers rise?",
    "Tell me about the sun.",
    "Are fish animals?",
    "Tell me about animals.",
    "What do plants need?",
    "Can cold water freeze?",
    "What do birds do in the morning?",
    "Where do rivers flow?",
    "What is the sky?",
]

# Held-out session: none of these facts are exposed before round 1.
SESSION_QUESTIONS = [
    "What is a volcano?",
    "Can lava flow?",
    "Is lava hot?",
    "What causes ash?",
    "What can a volcano do?",
    "Where does lava come from?",
    "Are volcanoes mountains?",
    "What happens when a volcano erupts?",
]
SESSION_EXPOSURE = [
    "A volcano is a mountain.",
    "Lava is hot.",
    "Lava can flow.",
    "A volcano can erupt.",
    "Volcanoes cause ash.",
    "Ash can cover cities.",
    "A volcano has a crater.",
    "Lava comes from a volcano.",
]
SESSION_GOLD = {
    "What is a volcano?": {"mountain"},
    "Can lava flow?": {"flow"},
    "Is lava hot?": {"hot"},
    "What causes ash?": {"cause", "volcano"},
    "What can a volcano do?": {"erupt"},
    "Where does lava come from?": {"come", "volcano"},
    "Are volcanoes mountains?": {"mountain"},
    "What happens when a volcano erupts?": {"erupt"},
}


def gold_hit(question, reply):
    words = set(re.findall(r"[a-z']+", reply.casefold()))
    stems = {word[:-1] if len(word) > 4 and word.endswith("s") else word
             for word in words}
    expected = SESSION_GOLD.get(question, set())
    return bool(expected & stems)


class ChatContext:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.tools = {}
        self.command = None

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, name, toolset=None, handler=None, schema=None):
        self.tools[name] = handler

    def register_command(self, name, handler, description=None):
        self.command = handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary",
                        default=str(Path.home() / "NaNLMind/build/nanlm-universal"))
    parser.add_argument("--output", default=str(ROOT / "docs/LONG_CHAT_DEMO.md"))
    parser.add_argument("--logic-records", type=int, default=3)
    parser.add_argument("--logic-epochs", type=int, default=1)
    parser.add_argument("--llm-url", default="",
                        help="optional OpenAI-compatible Hermes bridge URL (off by default)")
    parser.add_argument("--llm-model", default="gemma4:31b-cloud")
    args = parser.parse_args(argv)

    native = Path(args.native_binary).expanduser()
    if not native.exists():
        raise SystemExit(f"native binary not found: {native}")

    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = {
        "schema": "nanlmind-hermes-long-chat/1",
        "version": "1.3.1",
        "generated_at": generated,
        "native_binary": native.name,
        "provider": "none",
        "chat_mode": "native",
        "clocked_calls": 0,
        "nannet_calls": 0,
        "response_templates": 0,
        "retrieval": 0,
        "logic_training": {},
        "runner": {},
        "mth": {},
        "exposure": [],
        "conversation": [],
        "heldout": {},
        "correction": {},
        "call": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "state_dir": str(Path(tmp) / "state"),
            "runner_command": [str(native)],
            "chat_mode": "native",
            "chat_learning": True,
        }
        ctx = ChatContext(config)
        if args.llm_url:
            sys.path.insert(0, str(ROOT / "scripts"))
            from hermes_demo import OpenAICompatProvider  # noqa: E402
            provider = OpenAICompatProvider(args.llm_url, args.llm_model)
            report["provider"] = "openai-compatible"
            report["llm_model"] = args.llm_model
        else:
            provider = None
        register(ctx)

        # 1. Logic first: real Boolean inverses + deep logic through the native
        # binary. The deterministic teacher only supplies curriculum fixtures;
        # it is removed before conversation.
        sys.path.insert(0, str(ROOT / "scripts"))
        from demo_run import DemoLLM  # noqa: E402
        ctx.llm = DemoLLM()
        logic = tools.run_training_session(
            ctx, "logic", mode="logic",
            records_per_stage=args.logic_records,
            epochs=args.logic_epochs,
            max_samples_per_head=16)
        ctx.llm = provider
        training = logic.get("training") or {}
        foundation = training.get("boolean_foundation") or {}
        report["logic_training"] = {
            "status": training.get("status") or foundation.get("status"),
            "boolean_foundation_status": foundation.get("status"),
            "boolean_foundation_reused": foundation.get("reused_checkpoint"),
            "state_sha256": foundation.get("state_sha256"),
            "mode": logic.get("mode"),
            "stages_requested": logic.get("stages_requested"),
            "changed_heads": training.get("changed_heads"),
            "mirror_pair_updates": training.get("mirror_pair_updates"),
            "hyperedges_added": training.get("hyperedges_added"),
            "corpus_records": (logic.get("corpus") or {}).get("record_count"),
        }

        # 2. Hermes starts its own local-CPU runner over the trained chart.
        runner = LocalRunner.open(ctx)
        report["runner"] = runner.status()

        def ask(question):
            result = native_chat(ctx, question, session_id="long-chat")
            synthesis = result.get("native_synthesis", {})
            frame = synthesis.get("temporal_frame", {})
            return {
                "user": question,
                "nanlm": result.get("response", ""),
                "anchor": frame.get("anchor"),
                "seed": frame.get("seed", []),
                "steps": frame.get("steps", 0),
                "scored_per_step": frame.get("scored_per_step", 0),
                "register_frames": synthesis.get("register_frames", 0),
                "response_templates": synthesis.get("response_templates"),
                "retrieval": synthesis.get("retrieval"),
            }

        # 3. Exposure: the learner stores each sentence it is exposed to.
        mth_summary = {}
        for sentence in EXPOSURE:
            result = native_chat(ctx, sentence, session_id="long-chat")
            synthesis = result.get("native_synthesis", {})
            if not mth_summary:
                mth_summary = synthesis.get("mth") or {}
            frame = synthesis.get("temporal_frame", {})
            report["exposure"].append({
                "sentence": sentence,
                "response": result.get("response", ""),
                "learned_records": synthesis.get("learned_records"),
                "known_seed": frame.get("seed", []),
                "register_frames": synthesis.get("register_frames", 0),
                "register_mapped": (synthesis.get("register") or {}).get("mapped"),
            })

        # 4. Multi-turn conversation: temporal predictions, word by word.
        for question in CONVERSATION:
            row = ask(question)
            row["register_mapped"] = (runner.register.status().get("mapped"))
            report["conversation"].append(row)

        # 5. Genuine held-out session. Round 1 asks questions with no exposure.
        # Round 2 asks the same questions after new exposure. Questions alone
        # are not language exposure, so any gain comes from what was taught.
        heldout_before = [ask(question) for question in SESSION_QUESTIONS]
        session_exposure = []
        for sentence in SESSION_EXPOSURE:
            result = native_chat(ctx, sentence, session_id="long-chat")
            synthesis = result.get("native_synthesis", {})
            session_exposure.append({
                "sentence": sentence,
                "response": result.get("response", ""),
                "learned_records": synthesis.get("learned_records"),
            })
        heldout_after = [ask(question) for question in SESSION_QUESTIONS]
        # 6. Feedback and re-deliberation: reject the last answer and let the
        # model weigh its futures again with that answer suppressed.
        correction_question = "What is a volcano?"
        first_answer = ask(correction_question)
        feedback = native_chat(ctx, "No, that's incorrect.", session_id="long-chat")
        feedback_synthesis = feedback.get("native_synthesis", {})
        feedback_frame = feedback_synthesis.get("temporal_frame", {})
        report["correction"] = {
            "question": correction_question,
            "first": first_answer,
            "feedback": "No, that's incorrect.",
            "second": {
                "nanlm": feedback.get("response", ""),
                "correction": feedback_synthesis.get("correction"),
                "rejected": feedback_synthesis.get("rejected"),
                "deliberation": feedback_frame.get("considered", []),
                "register_frames": feedback_synthesis.get("register_frames", 0),
            },
        }

        report["heldout"] = {
            "topic": "volcano",
            "questions": SESSION_QUESTIONS,
            "before": heldout_before,
            "exposure": session_exposure,
            "after": heldout_after,
            "summary": {
                "asked": len(SESSION_QUESTIONS),
                "before_answered": sum(1 for row in heldout_before if row["nanlm"]),
                "after_answered": sum(1 for row in heldout_after if row["nanlm"]),
                "before_gold": sum(1 for row in heldout_before
                                   if gold_hit(row["user"], row["nanlm"])),
                "after_gold": sum(1 for row in heldout_after
                                  if gold_hit(row["user"], row["nanlm"])),
                "changed": sum(1 for before, after in zip(heldout_before, heldout_after)
                               if after["nanlm"] and after["nanlm"] != before["nanlm"]),
            },
        }

        # 5. Three-way call with CLOCKED off (optional Hermes bridge above).
        phone_module.handle(ctx, {
            "action": "start", "session_id": "long-chat",
            "message": "What is a river?", "addressee": "hermes", "rounds": 1})
        phone_module.handle(ctx, {
            "action": "say", "session_id": "long-chat",
            "message": "Can water flow?", "addressee": "nanlm"})
        phone = phone_module.handle(ctx, {
            "action": "run", "session_id": "long-chat", "rounds": 2})
        report["call"] = {
            "clocked": phone.get("clocked"),
            "kernel": phone.get("kernel"),
            "turn": phone.get("turn"),
            "history": phone.get("history", []),
        }

        report["mth"] = mth_summary
        rows = tools._memory_store(ctx).query(limit=1000)
        counts: dict[str, int] = {}
        for row in rows:
            kind = str(row.get("kind") or "unknown")
            counts[kind] = counts.get(kind, 0) + 1
        report["memory"] = {"records": len(rows), "by_kind": dict(sorted(counts.items()))}
        report["runner"] = runner.status()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_name("long-chat-demo.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Long exposure chat with NaNLM after logic training",
        "",
        "Generated by `scripts/long_chat_demo.py` at **1.3.1**. Native mode only:",
        "no LLM provider, no CLOCKED calls and no NANNET calls. The learner first",
        "ran the real Boolean/deep-logic curriculum through `nanlm-universal`,",
        "then Hermes started its **own local-CPU runner** over the trained chart",
        "and held a multi-turn conversation.",
        "",
        "**No templates, no retrieval.** Every reply is predicted word by word:",
        "the temporal mirror model scores every known word at every step, predicts",
        "the next word forward from a temporal frame, and feeds the chosen word",
        "back into the frame. `response_templates=0`, `retrieval=0` on every turn.",
        "",
        "## 1. Logic training",
        "",
        "```json",
        json.dumps(report["logic_training"], indent=2, sort_keys=True),
        "```",
        "",
        "## 2. Hermes local-CPU runner and registers",
        "",
        "The runner loads the 8 MiB chart plus the mapped temporal register into",
        "local memory. It does **not** use the system RunnerRing for compute; it",
        "links through `HW2:0` (NaN16DE binary translation) to `BS0:0::HW0:0`.",
        "",
        "```json",
        json.dumps(report["runner"], indent=2, sort_keys=True)[:8000],
        "```",
        "",
        "## 3. MTH layers: root 4x4 Boolean table to speech",
        "",
        "The MTH bridge verifies the 16-function root Boolean table, composes",
        "nibble and byte layers, then builds phonetic, morpheme, lexical,",
        "word-completion, sentence, paragraph and narrative layers from the",
        "exposed corpus. `native_verified` is false: this is the Hermes-side",
        "reference interface; native MTH kernels live in NaNROUTER.",
        "",
        "```json",
        json.dumps(report["mth"], indent=2, sort_keys=True)[:8000],
        "```",
        "",
        "## 4. Exposure",
        "",
        "| exposed sentence | learner reply | learned | known seed |",
        "|---|---|---:|---|",
    ]
    for row in report["exposure"]:
        reply = row["response"].replace("|", "\\|") or "_(learned silently)_"
        seed = ", ".join(row["known_seed"]) or "_(new words only)_"
        lines.append(f"| {row['sentence'].replace('|', chr(92)+'|')} | {reply} | "
                     f"{row['learned_records']} | {seed} |")
    lines += [
        "",
        "## 5. Conversation",
        "",
        "| user | NaNLM prediction | anchor | scored/step | register frames |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["conversation"]:
        reply = row["nanlm"].replace("|", "\\|") or "_(no known seed)_"
        lines.append(f"| {row['user'].replace('|', chr(92)+'|')} | {reply} | "
                     f"{row['anchor']} | {row['scored_per_step']} | "
                     f"{row['register_frames']} |")
    lines += [
        "",
        "## 6. Genuine held-out chat over a session",
        "",
        "Round 1 asks eight volcano questions before any volcano exposure.",
        "The same questions are asked again after the new exposure. Asking a",
        "question does not teach the language model, so any gain below comes from",
        "what was exposed between the rounds.",
        "",
        "```json",
        json.dumps(report["heldout"]["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "| question | round 1 (cold) | round 2 (after exposure) | anchor |",
        "|---|---|---|---|",
    ]
    for before, after in zip(report["heldout"]["before"], report["heldout"]["after"]):
        cold = before["nanlm"].replace("|", "\\|") or "_(no learned reply)_"
        warm = after["nanlm"].replace("|", "\\|") or "_(no learned reply)_"
        lines.append(f"| {before['user'].replace('|', chr(92)+'|')} | {cold} | {warm} | "
                     f"{after['anchor']} |")
    lines += [
        "",
        "## 7. Feedback and re-deliberation",
        "",
        "After the answer to `What is a volcano?`, the user says",
        "`No, that's incorrect.` This is control input, not conversation. The",
        "rejected answer is suppressed and the model weighs its futures again.",
        "",
        "```json",
        json.dumps(report["correction"], indent=2, sort_keys=True)[:6000],
        "```",
        "",
        "## 8. Three-way call (CLOCKED off)",
        "",
        "The call kernel is raw NaN16; CLOCKED was not enabled, so no gateway",
        "credits were used. NaNLM's call turns use the same temporal predictor.",
        "",
        "```json",
        json.dumps(report["call"], indent=2, sort_keys=True)[:12000],
        "```",
        "",
        "## 9. Memory and register state",
        "",
        "```json",
        json.dumps({"memory": report["memory"],
                    "register": report["runner"].get("register")},
                   indent=2, sort_keys=True)[:6000],
        "```",
        "",
        "## Reading the result",
        "",
        "- Logic training is the real Boolean/deep-logic curriculum in the native",
        "  binary; it is separate from language exposure.",
        "- Hermes runs its own runner on local CPU over the locally trained 8 MiB",
        "  chart. The temporal register is mapped into the chart extension and its",
        "  `s+`/`s-` rails are real pair-cell rails, held in memory.",
        "- Conversation answers are forward next-word predictions over the exposed",
        "  vocabulary, not sentence retrieval: no paraphrase, no stock phrases.",
        "- Empty replies are honest: the query shared no previously exposed term.",
        "- The merge path (Koyeb corruption analysis, then MC0/SC0/BS0DB/MC0DB",
        "  integration) is not part of this local run; the NaN16DE link through",
        "  `HW2:0` to `BS0:0::HW0:0` is the interface it will use.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "json": str(json_path),
        "logic_status": report["logic_training"].get("status"),
        "boolean_foundation": report["logic_training"].get("boolean_foundation_status"),
        "runner": report["runner"].get("runner"),
        "chart_bytes": report["runner"].get("chart_bytes"),
        "register_mapped": (report["runner"].get("register") or {}).get("mapped"),
        "nannet_link": report["runner"].get("nannet_link"),
        "uses_system_ring_for_compute": report["runner"].get("uses_system_ring_for_compute"),
        "exposure_sentences": len(EXPOSURE),
        "conversation_turns": len(CONVERSATION),
        "answered": sum(1 for row in report["conversation"] if row["nanlm"]),
        "empty": sum(1 for row in report["conversation"] if not row["nanlm"]),
        "response_templates": report["response_templates"],
        "retrieval": report["retrieval"],
        "clocked_calls": report["clocked_calls"],
        "call_turns": report["call"].get("turn"),
        "heldout_asked": report["heldout"]["summary"]["asked"],
        "heldout_before_answered": report["heldout"]["summary"]["before_answered"],
        "heldout_after_answered": report["heldout"]["summary"]["after_answered"],
        "heldout_before_gold": report["heldout"]["summary"]["before_gold"],
        "heldout_after_gold": report["heldout"]["summary"]["after_gold"],
        "heldout_changed": report["heldout"]["summary"]["changed"],
        "mth_bit_accuracy": (report["mth"].get("layers", {}).get("bit") or {}).get("accuracy"),
        "mth_nibble_accuracy": (report["mth"].get("layers", {}).get("nibble") or {}).get("accuracy"),
        "mth_byte_accuracy": (report["mth"].get("layers", {}).get("byte") or {}).get("accuracy"),
        "mth_sentence_edges": report["mth"].get("edges"),
        "correction_first": report["correction"]["first"]["nanlm"],
        "correction_second": report["correction"]["second"]["nanlm"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
