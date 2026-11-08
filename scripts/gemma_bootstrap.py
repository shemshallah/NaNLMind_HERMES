#!/usr/bin/env python3
"""Gemma-guided bootstrap conversation for NaNLM.

Gemma is given the architecture description and designs the bootstrap itself:
teacher statements (including uncertainty modelling), known probes and unknown
probes. NaNLM is exposed to the statements through native chat, then probed.
Gemma sees the results and continues the training conversation for a second
round. No "I don't know" rule is implemented; abstention is only measured.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.chat import chat as native_chat  # noqa: E402

ARCHITECTURE = """NaNLM (the Hermes plugin) is not an LLM. It is a from-scratch
word-level temporal predictor. It learns only from statements it is exposed to;
questions and requests are not learned as facts. At each step it weighs every
known word, predicts the next word forward from a head/mirror temporal frame,
finishes the sentence, and may add an explanation. MTH provides a root 4x4
Boolean table (16 functions) and ten layers L0-L9: bit, nibble, byte, phonetic,
morpheme, lexical, word_completion, sentence, paragraph, narrative. Heads map
relations through root Boolean transforms into a head scratch register. There
are no response templates and no retrieval. It has no built-in "I don't know";
abstention must develop from exposure if it develops at all."""

SYSTEM = """You are Gemma, the curriculum teacher for NaNLM.

Architecture:
""" + ARCHITECTURE + """

Design the bootstrap as a free-form training conversation made of TEACHER
STATEMENTS only (never write NaNLM's replies). Teach simple explicit facts,
definitions, capabilities and causal chains. Explicitly model uncertainty with
statements such as "I do not know the capital of France." for facts the learner
has not been taught. Then give probes: known questions with expected keywords,
and unknown questions that should expose whether abstention emerged.

Return STRICT JSON only:
{
  "reasoning": "short explanation of the bootstrap",
  "teaching": ["statement", ...],
  "uncertainty": ["statement", ...],
  "known_probes": [{"question": "...", "expected": ["keyword", ...]}, ...],
  "unknown_probes": ["question", ...]
}
Use 20-30 teaching statements, 6 uncertainty statements, 8 known probes and
8 unknown probes. Keep every statement short and explicit."""

UNCERTAINTY = ("i do not know", "i don't know", "do not know", "don't know",
               "not learned", "haven't learned", "have not learned", "unknown",
               "not sure", "no relation", "not know")


def extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("teacher did not return JSON")
    return json.loads(text[start:end + 1])


def as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def normalise_plan(plan: dict) -> dict:
    teaching = [str(item).strip() for item in as_list(plan.get("teaching"))
                if str(item).strip() and not str(item).strip().endswith("?")]
    uncertainty = [str(item).strip() for item in as_list(plan.get("uncertainty"))
                   if str(item).strip() and not str(item).strip().endswith("?")]
    known = []
    for item in as_list(plan.get("known_probes")):
        if isinstance(item, dict) and item.get("question"):
            expected = item.get("expected")
            if isinstance(expected, str):
                expected = [expected]
            known.append({"question": str(item["question"]),
                          "expected": [str(word) for word in as_list(expected)]})
        elif isinstance(item, str):
            known.append({"question": item, "expected": []})
    for item in as_list(plan.get("category_probes")):
        if isinstance(item, dict) and item.get("question"):
            expected = item.get("expected")
            if isinstance(expected, str):
                expected = [expected]
            known.append({"question": str(item["question"]),
                          "expected": [str(word) for word in as_list(expected)],
                          "member": str(item.get("member") or ""),
                          "category": True})
    unknown = [str(item).strip() for item in as_list(plan.get("unknown_probes"))
               if str(item).strip()]
    return {"reasoning": str(plan.get("reasoning") or ""),
            "teaching": teaching, "uncertainty": uncertainty,
            "known_probes": known, "unknown_probes": unknown}


def main():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from hermes_demo import OpenAICompatProvider  # noqa: E402

    url = "http://127.0.0.1:11434/v1/chat/completions"
    model = "gemma4:31b-cloud"
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/GEMMA_BOOTSTRAP.md"
    teacher = OpenAICompatProvider(url, model)

    plan = normalise_plan(extract_json(teacher.complete(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Design round 1 of the bootstrap now."}],
        temperature=0.5, max_tokens=3000, timeout=240).text))

    class Context:
        def __init__(self, root):
            self.config = {"state_dir": str(root), "chat_mode": "native",
                           "chat_learning": True}
            self.llm = None

        def get_config(self, key, default=None):
            return self.config.get(key, default)

    def feed(ctx, statements):
        rows = []
        for statement in statements:
            result = native_chat(ctx, statement, session_id="gemma-bootstrap")
            synthesis = result.get("native_synthesis", {})
            rows.append({
                "teacher": statement,
                "nanlm": result.get("response", ""),
                "learned_records": synthesis.get("learned_records"),
                "heads": (synthesis.get("heads") or {}).get("leaf_count"),
            })
        return rows

    def probe(ctx, known, unknown):
        known_rows = []
        for item in known:
            result = native_chat(ctx, item["question"], session_id="gemma-bootstrap")
            reply = result.get("response", "")
            hit = any(word.casefold() in reply.casefold()
                      for word in item["expected"]) if item["expected"] else False
            known_rows.append({**item, "nanlm": reply, "hit": hit,
                               "context_terms": result.get("native_synthesis", {})
                               .get("context_terms", [])})
        unknown_rows = []
        for question in unknown:
            result = native_chat(ctx, question, session_id="gemma-bootstrap")
            reply = result.get("response", "")
            abstained = any(phrase in reply.casefold() for phrase in UNCERTAINTY)
            unknown_rows.append({"question": question, "nanlm": reply,
                                 "abstained": abstained})
        return known_rows, unknown_rows

    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(Path(tmp) / "state")
        teaching_rows = feed(ctx, plan["teaching"] + plan["uncertainty"])
        known_1, unknown_1 = probe(ctx, plan["known_probes"], plan["unknown_probes"])

        summary_1 = {
            "known": f"{sum(1 for row in known_1 if row['hit'])}/{len(known_1)}",
            "unknown_abstained": f"{sum(1 for row in unknown_1 if row['abstained'])}/{len(unknown_1)}",
            "heads": (teaching_rows[-1]["heads"] if teaching_rows else 0),
        }
        followup = teacher.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps({
                 "round_1_plan": plan,
                 "round_1_known": known_1,
                 "round_1_unknown": unknown_1,
                 "round_1_summary": summary_1,
                 "task": ("Continue the training conversation. Add 10 teaching "
                          "statements and 4 uncertainty statements that target the "
                          "misses, plus 4 known and 4 unknown probes. Return the "
                          "same strict JSON shape.")}, indent=2)}],
            temperature=0.5, max_tokens=2500, timeout=240).text
        plan_2 = normalise_plan(extract_json(followup))
        follow_rows = feed(ctx, plan_2["teaching"] + plan_2["uncertainty"])
        known_2, unknown_2 = probe(ctx, plan_2["known_probes"], plan_2["unknown_probes"])
        summary_2 = {
            "known": f"{sum(1 for row in known_2 if row['hit'])}/{len(known_2)}",
            "unknown_abstained": f"{sum(1 for row in unknown_2 if row['abstained'])}/{len(unknown_2)}",
            "heads": (follow_rows[-1]["heads"] if follow_rows else 0),
        }
        round3_prompt = """This is round 3 of a massive from-scratch experiment, not a
normal language model. NaNLM can now anchor on new entity words supplied in the
question context, but only composes a category frame if the shared predicate was
taught from exposure. It cannot invent unseen words. Design ABSTRACT,
CATEGORY-LEVEL training:
- teach category frames with several members and shared properties. "Element"
  is polysemous: teach sense-qualified statements such as "Gold is a chemical
  element.", "A triangle is a geometric element.", "The sky is a component of
  the Earth." (an Earth component); teach animals (cat, dog, bird), celestial
  bodies (sun, moon, stars) and colours (red, blue, yellow) the same way;
- teach shared predicates so an UNTAUGHT member can be composed by analogy;
- model uncertainty when no relation is learned;
- include category_probes on members that were not directly taught, and
  wrong-sense probes (for example "Is copper a geometric element?" when copper
  was taught as a chemical element), and unknown_probes outside every taught
  category.
Return strict JSON with the same keys plus
"category_probes": [{"question", "expected", "member"}]."""
        plan_3 = normalise_plan(extract_json(teacher.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps({
                 "round_2_summary": summary_2,
                 "round_2_known": known_2,
                 "round_2_unknown": unknown_2,
                 "task": round3_prompt}, indent=2)}],
            temperature=0.55, max_tokens=3000, timeout=240).text))
        round3_rows = feed(ctx, plan_3["teaching"] + plan_3["uncertainty"])
        known_3, unknown_3 = probe(ctx, plan_3["known_probes"], plan_3["unknown_probes"])
        summary_3 = {
            "known": f"{sum(1 for row in known_3 if row['hit'])}/{len(known_3)}",
            "category_hits": f"{sum(1 for row in known_3 if row.get('category') and row['hit'])}/{sum(1 for row in known_3 if row.get('category'))}",
            "unknown_abstained": f"{sum(1 for row in unknown_3 if row['abstained'])}/{len(unknown_3)}",
            "heads": (round3_rows[-1]["heads"] if round3_rows else 0),
        }

    report = {
        "schema": "nanlmind-hermes-gemma-bootstrap/1",
        "teacher": model, "architecture": ARCHITECTURE,
        "round_1": {"plan": plan, "teaching": teaching_rows,
                    "known": known_1, "unknown": unknown_1, "summary": summary_1},
        "round_2": {"plan": plan_2, "teaching": follow_rows,
                    "known": known_2, "unknown": unknown_2, "summary": summary_2},
        "round_3": {"plan": plan_3, "teaching": round3_rows,
                    "known": known_3, "unknown": unknown_3, "summary": summary_3},
    }
    json_path = output.with_name("gemma-bootstrap.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Gemma-guided bootstrap training conversation",
        "",
        f"Teacher: `{model}`. NaNLM native mode, chat learning on, no templates.",
        "Gemma designed the curriculum from the architecture description.",
        "",
        "## Round 1 reasoning",
        "", plan["reasoning"], "",
        "### Teaching transcript",
        "", "| teacher statement | NaNLM reply | learned | heads |",
        "|---|---|---:|---:|",
    ]
    for row in teaching_rows:
        lines.append(f"| {row['teacher'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{row['learned_records']} | {row['heads']} |")
    lines += ["", "### Known probes", "",
              "| question | expected | NaNLM | hit |", "|---|---|---|---|"]
    for row in known_1:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{', '.join(row['expected'])} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{row['hit']} |")
    lines += ["", "### Unknown probes", "",
              "| question | NaNLM | abstained |", "|---|---|---|"]
    for row in unknown_1:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{row['abstained']} |")
    lines += ["", "## Round 2 (Gemma saw the misses)", "",
              plan_2["reasoning"], "",
              "| teacher statement | NaNLM reply | heads |", "|---|---|---:|"]
    for row in follow_rows:
        lines.append(f"| {row['teacher'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{row['heads']} |")
    lines += ["", "### Round 2 probes", "",
              f"- known: {summary_2['known']}",
              f"- unknown abstained: {summary_2['unknown_abstained']}",
              "",
              "| question | NaNLM | hit/abstained |", "|---|---|---|"]
    for row in known_2:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"hit={row['hit']} |")
    for row in unknown_2:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"abstained={row['abstained']} |")
    lines += ["", "## Round 3: abstract categories and untaught members", "",
              plan_3["reasoning"], "",
              f"Known: {summary_3['known']}; category transfer: {summary_3['category_hits']}; "
              f"unknown abstained: {summary_3['unknown_abstained']}", "",
              "| question | member | NaNLM | context terms | hit |", "|---|---|---|---|---|"]
    for row in known_3:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{row.get('member', '')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{', '.join(row.get('context_terms', []))} | {row['hit']} |")
    lines += ["", "| unknown question | NaNLM | abstained |", "|---|---|---|"]
    for row in unknown_3:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{row['abstained']} |")
    lines += ["", "## Summary", "",
              f"- Round 3 category transfer: {summary_3['category_hits']}",
              f"- Round 3 unknown abstained: {summary_3['unknown_abstained']}",
              f"- Round 1 known: {summary_1['known']}; unknown abstained: {summary_1['unknown_abstained']}",
              f"- Round 2 known: {summary_2['known']}; unknown abstained: {summary_2['unknown_abstained']}",
              "- Abstention was measured only; no rule or template was added.",
              "- Heads are the number of MTH head leaves after the last exposure."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "json": str(json_path),
        "round_1": summary_1, "round_2": summary_2, "round_3": summary_3,
        "teaching_statements": len(plan["teaching"]) + len(plan["uncertainty"]),
        "followup_statements": len(plan_2["teaching"]) + len(plan_2["uncertainty"]),
    }, indent=2))


if __name__ == "__main__":
    main()
