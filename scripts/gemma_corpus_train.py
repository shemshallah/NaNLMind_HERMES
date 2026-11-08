#!/usr/bin/env python3
"""Gemma-designed conceptual-superposition corpus.

Gemma designs a corpus of short statements whose concepts recur across many
contexts. Every statement contributes all concept pairs to the C(256,2) matrix,
so density grows through association, not one-word-one-cell mapping. Gemma sees
the matrix saturation and probe results and designs a second corpus.
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

from nanlmind_hermes.chat import chat as native_chat, _native_parse  # noqa: E402
from nanlmind_hermes.concept_matrix import ConceptMatrix  # noqa: E402
from nanlmind_hermes.mth_layers import relation_role  # noqa: E402

SYSTEM = """You are Gemma, designing an initial corpus for NaNLM, a massive
from-scratch experiment, not a normal language model.

Representation: 256 heads; C(256,2)=32,640 unordered pair cells, each with two
directed rails (s+ forward, s- reverse). A word is NOT mapped to a cell.
Statements bring concepts in superposition; every pair of concepts co-occurring
in a statement accumulates weight in its pair cell; typed relations add a role
and direction. A concept's leaf identity is the set of pairs it participates
in. Meaning must emerge from association and relation, not direct mapping.
Pairs are relation axes, not automatically antonyms.

Design an initial corpus that increases matrix density naturally: 30-50
recurring concepts across physical, spatial, causal, geometric, chemical and
abstract realms; varied relations (is-a, has, can, causes, part/component of,
opposite, attribute); sense-qualified statements where a word is polysemous
(e.g. chemical element, geometric element, component of the Earth). Statements
only, never questions. Return STRICT JSON:
{
  "reasoning": "...",
  "concepts": ["..."],
  "statements": ["...", ...],
  "known_probes": [{"question": "...", "expected": ["keyword"]}, ...],
  "category_probes": [{"question": "...", "expected": ["keyword"], "member": "..."}],
  "unknown_probes": ["..."]
}
40-60 statements, 8 known probes, 6 category probes, 6 unknown probes."""


def extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON from teacher")
    return json.loads(text[start:end + 1])


def normalise(plan: dict) -> dict:
    def items(key):
        value = plan.get(key)
        return value if isinstance(value, list) else []
    statements = [str(s).strip() for s in items("statements")
                  if str(s).strip() and not str(s).strip().endswith("?")]
    known = []
    for item in items("known_probes"):
        if isinstance(item, dict) and item.get("question"):
            expected = item.get("expected")
            if isinstance(expected, str):
                expected = [expected]
            known.append({"question": str(item["question"]),
                          "expected": [str(x) for x in (expected or [])]})
    category = []
    for item in items("category_probes"):
        if isinstance(item, dict) and item.get("question"):
            expected = item.get("expected")
            if isinstance(expected, str):
                expected = [expected]
            category.append({"question": str(item["question"]),
                             "expected": [str(x) for x in (expected or [])],
                             "member": str(item.get("member") or "")})
    unknown = [str(q) for q in items("unknown_probes") if str(q).strip()]
    return {"reasoning": str(plan.get("reasoning") or ""),
            "concepts": [str(c) for c in items("concepts")],
            "statements": statements, "known_probes": known,
            "category_probes": category, "unknown_probes": unknown}


def relations_for(statement: str) -> list[dict]:
    parsed = _native_parse(statement, "corpus", "0" * 64)
    out = []
    for record in parsed:
        item = record.get("item") or {}
        if item.get("kind") != "relation":
            continue
        role, _extra = relation_role(item.get("predicate", ""), item.get("object", ""))
        out.append({"subject": item.get("subject", ""),
                    "predicate": item.get("predicate", ""),
                    "object": item.get("object", ""), "role": role})
    return out


def main():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from hermes_demo import OpenAICompatProvider  # noqa: E402

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/GEMMA_CORPUS.md"
    teacher = OpenAICompatProvider("http://127.0.0.1:11434/v1/chat/completions",
                                   "gemma4:31b-cloud")

    class Context:
        def __init__(self, root):
            self.config = {"state_dir": str(root), "chat_mode": "native",
                           "chat_learning": True}
            self.llm = None

        def get_config(self, key, default=None):
            return self.config.get(key, default)

    def probe(ctx, known, category, unknown):
        known_rows = []
        for item in known + category:
            result = native_chat(ctx, item["question"], session_id="corpus")
            reply = result.get("response", "")
            hit = any(word.casefold() in reply.casefold()
                      for word in item.get("expected", []))
            known_rows.append({**item, "nanlm": reply, "hit": hit})
        unknown_rows = []
        for question in unknown:
            result = native_chat(ctx, question, session_id="corpus")
            reply = result.get("response", "")
            abstained = any(phrase in reply.casefold() for phrase in (
                "do not know", "don't know", "not learned", "haven't learned",
                "unknown", "not sure", "no relation"))
            unknown_rows.append({"question": question, "nanlm": reply,
                                 "abstained": abstained})
        return known_rows, unknown_rows

    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(Path(tmp) / "state")
        matrix = ConceptMatrix()
        plan_1 = normalise(extract_json(teacher.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Design corpus round 1 now."}],
            temperature=0.5, max_tokens=4000, timeout=300).text))
        rows_1 = []
        for statement in plan_1["statements"]:
            result = native_chat(ctx, statement, session_id="corpus")
            rels = relations_for(statement)
            for relation in rels:
                matrix.observe(statement, relation)
            if not rels:
                matrix.observe(statement)
            rows_1.append({"statement": statement, "relations": rels,
                           "nanlm": result.get("response", "")})
        known_1, unknown_1 = probe(ctx, plan_1["known_probes"],
                                   plan_1["category_probes"],
                                   plan_1["unknown_probes"])
        sat_1 = matrix.saturation()

        follow = teacher.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps({
                 "round_1": {"reasoning": plan_1["reasoning"],
                             "saturation": sat_1, "known": known_1,
                             "unknown": unknown_1},
                 "task": ("Design corpus round 2. Target underrepresented "
                          "concepts and pair regions; add sense-qualified and "
                          "cross-domain statements. 40-60 statements, 6 known, "
                          "6 category, 6 unknown probes. Same strict JSON.")},
                 indent=2)}],
            temperature=0.5, max_tokens=3000, timeout=300).text
        plan_2 = normalise(extract_json(follow))
        rows_2 = []
        for statement in plan_2["statements"]:
            result = native_chat(ctx, statement, session_id="corpus")
            rels = relations_for(statement)
            for relation in rels:
                matrix.observe(statement, relation)
            if not rels:
                matrix.observe(statement)
            rows_2.append({"statement": statement, "relations": rels,
                           "nanlm": result.get("response", "")})
        known_2, unknown_2 = probe(ctx, plan_2["known_probes"],
                                   plan_2["category_probes"],
                                   plan_2["unknown_probes"])
        sat_2 = matrix.saturation()
        matrix.save(Path(tmp) / "concept-matrix.json")
        leaves = [matrix.leaf(name) for name in list(matrix.concepts)[:24]]

        # Organization phase: chat statements use the concepts and let the
        # heads remap/reorganize; MTH transforms are recorded in heads/leaves.
        organization = []
        for statement in (
            "Copper is a chemical element.",
            "A triangle is a geometric element.",
            "The sky is a component of the earth.",
            "Hot is the opposite of cold.",
            "Gold is a chemical element.",
        ):
            result = native_chat(ctx, statement, session_id="corpus")
            rels = relations_for(statement)
            for relation in rels:
                matrix.observe(statement, relation)
            if not rels:
                matrix.observe(statement)
            synthesis = result.get("native_synthesis", {})
            organization.append({
                "statement": statement,
                "nanlm": result.get("response", ""),
                "heads": synthesis.get("heads", {}),
                "roles": [rel["role"] for rel in rels],
            })
        org_probes = []
        for question in ("Is copper a chemical element?",
                         "Is a triangle a geometric element?",
                         "Is the sky a component of the earth?",
                         "Is hot the opposite of cold?"):
            result = native_chat(ctx, question, session_id="corpus")
            org_probes.append({"question": question,
                               "nanlm": result.get("response", "")})
        sat_3 = matrix.saturation()

    report = {
        "schema": "nanlmind-hermes-gemma-corpus/1",
        "teacher": "gemma4:31b-cloud", "architecture": "concept matrix C(256,2)",
        "round_1": {"plan": plan_1, "statements": rows_1, "known": known_1,
                    "unknown": unknown_1, "saturation": sat_1},
        "round_2": {"plan": plan_2, "statements": rows_2, "known": known_2,
                    "unknown": unknown_2, "saturation": sat_2},
        "organization": {"statements": organization, "probes": org_probes,
                         "saturation": sat_3},
        "leaves": leaves,
    }
    json_path = output.with_name("gemma-corpus.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Gemma conceptual-superposition corpus",
        "",
        "Teacher: `gemma4:31b-cloud`. Statements feed NaNLM native chat and the",
        "C(256,2) concept matrix; meaning accumulates by association and typed",
        "relation, not by assigning a word to a cell.",
        "",
        "## Round 1", "", plan_1["reasoning"], "",
        f"- concepts: {sat_1['concepts']}",
        f"- occupied pairs: {sat_1['occupied_pairs']:,} / {sat_1['pair_cells']:,} "
        f"(density {sat_1['density']})",
        f"- average degree: {sat_1['average_degree']}",
        f"- known probes: {sum(1 for r in known_1 if r['hit'])}/{len(known_1)}",
        f"- unknown abstained: {sum(1 for r in unknown_1 if r['abstained'])}/{len(unknown_1)}",
        "", "| statement | NaNLM reply | relations |", "|---|---|---|",
    ]
    for row in rows_1[:80]:
        lines.append(f"| {row['statement'].replace('|', chr(92)+'|')} | "
                     f"{(row.get('nanlm') or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{', '.join(rel['role'] for rel in row['relations']) or 'co-occurrence'} |")
    lines += ["", "## Probe results", "",
              "| question | NaNLM | hit |", "|---|---|---|"]
    for row in known_1:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | {row['hit']} |")
    lines += ["", "| unknown question | NaNLM | abstained |", "|---|---|---|"]
    for row in unknown_1:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | {row['abstained']} |")
    lines += ["", "## Round 2 (targeting matrix gaps)", "", plan_2["reasoning"], "",
              f"- concepts: {sat_2['concepts']}",
              f"- occupied pairs: {sat_2['occupied_pairs']:,} / {sat_2['pair_cells']:,} "
              f"(density {sat_2['density']})",
              f"- average degree: {sat_2['average_degree']}",
              f"- known probes: {sum(1 for r in known_2 if r['hit'])}/{len(known_2)}",
              f"- unknown abstained: {sum(1 for r in unknown_2 if r['abstained'])}/{len(unknown_2)}",
              "", "## Example concept leaves", "",
              "```json", json.dumps(leaves[:12], indent=2)[:6000], "```", "",
              "## Top pairs", "", "```json",
              json.dumps(sat_2["top_pairs"], indent=2)[:4000], "```",
              "", "## Organization by chatting", "",
              "After saturation, statements combining concepts let the heads",
              "reorganize and map relations into leaves and scratchpads.", "",
              "| statement | NaNLM | roles | head leaves |", "|---|---|---|---|"]
    for row in organization:
        leaves_n = (row["heads"] or {}).get("leaf_count")
        lines.append(f"| {row['statement'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} | "
                     f"{', '.join(row['roles']) or 'co-occurrence'} | {leaves_n} |")
    lines += ["", "| organization probe | NaNLM |", "|---|---|"]
    for row in org_probes:
        lines.append(f"| {row['question'].replace('|', chr(92)+'|')} | "
                     f"{(row['nanlm'] or '_(empty)_').replace('|', chr(92)+'|')} |")
    lines += ["", "### Matrix after organization", "",
              f"- concepts: {sat_3['concepts']}",
              f"- occupied pairs: {sat_3['occupied_pairs']:,} / {sat_3['pair_cells']:,} "
              f"(density {sat_3['density']})",
              f"- average degree: {sat_3['average_degree']}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "json": str(json_path),
        "round_1": {"concepts": sat_1["concepts"], "density": sat_1["density"],
                    "known": f"{sum(1 for r in known_1 if r['hit'])}/{len(known_1)}",
                    "unknown": f"{sum(1 for r in unknown_1 if r['abstained'])}/{len(unknown_1)}"},
        "round_2": {"concepts": sat_2["concepts"], "density": sat_2["density"],
                    "known": f"{sum(1 for r in known_2 if r['hit'])}/{len(known_2)}",
                    "unknown": f"{sum(1 for r in unknown_2 if r['abstained'])}/{len(unknown_2)}"},
        "statements": len(rows_1) + len(rows_2),
        "organization_density": sat_3["density"],
    }, indent=2))


if __name__ == "__main__":
    main()
