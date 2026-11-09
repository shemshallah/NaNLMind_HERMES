#!/usr/bin/env python3
"""Longitudinal NaNLM learning experiment with Gemma as the teacher.

Gemma designs short teaching statements; NaNLM is the native student. A fixed
held-out probe set is asked before exposure and after every teaching round. The
script records probe hits, exact replies, vocabulary/head/scratch growth and
concept saturation over time.

This is a local experiment, not a canonical training run or a proof of general
language learning. Gemma does not answer as NaNLM; it only designs lessons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import register  # noqa: E402
from nanlmind_hermes.layered_learning import reset_coordinator_cache  # noqa: E402
from nanlmind_hermes.mth_leaves import reset_cache as reset_leaf_cache  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_OUTPUT = ROOT / "docs" / "NANLM_GEMMA_LEARNING_OVER_TIME.json"

TARGET_FACTS = (
    "A river is water. Water can flow. Rivers carry water.",
    "The sky is blue. Grass is green. Fire is red. Snow is white.",
    "A cat is an animal. A dog is an animal. A bird is an animal.",
    "Cats meow. Dogs bark. Birds fly.",
    "Heat causes melting. Cold causes freezing. Ice is frozen water.",
    "Red and blue make purple. Yellow and blue make green.",
    "The sun is hot. The sun causes light. The earth is a planet.",
    "A square has four sides. A circle is round.",
)

PROBES = (
    ("What is a river?", ("water",)),
    ("What color is the sky?", ("blue",)),
    ("What color is grass?", ("green",)),
    ("What do dogs do?", ("bark",)),
    ("What do cats do?", ("meow",)),
    ("What do birds do?", ("fly",)),
    ("What is a cat?", ("animal",)),
    ("What happens when water freezes?", ("ice", "frozen")),
    ("What is ice?", ("frozen", "water")),
    ("What do red and blue make?", ("purple",)),
    ("What do yellow and blue make?", ("green",)),
    ("What is the sun?", ("hot", "light", "star")),
    ("What shape has four sides?", ("square",)),
    ("What is a circle?", ("round",)),
)

TEACHER_SYSTEM = """You are Gemma, the curriculum teacher for NaNLM.
NaNLM is not an LLM. It is a from-scratch native temporal word learner that
learns only from statements it is exposed to; questions are not learned as
facts. Design short teaching statements only. Do not write NaNLM's replies and
do not ask questions. Use explicit, simple facts, varied phrasing and causal
chains. Return STRICT JSON:
{"reasoning": "short explanation", "statements": ["statement", ...]}
Use 8-12 statements. Keep every statement short and explicit."""


class Teacher:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def design(self, round_index: int, previous: list[dict]) -> dict:
        user = {
            "round": round_index,
            "target_facts": TARGET_FACTS,
            "previous_probe_results": previous[-1]["probes"] if previous else [],
            "task": ("Write the next round of teaching statements. Reinforce missed "
                     "facts with new phrasing; add short causal or category "
                     "statements. Statements only."),
        }
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": 0.35,
            "max_tokens": 1400,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.loads(response.read(262144))
        text = result["choices"][0]["message"].get("content") or ""
        return {"text": text, "model": result.get("model", self.model)}


class Context:
    def __init__(self, root, provider):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "native",
            "chat_learning": True,
            "learning_require_approval": True,
            "learning_auto_sync": False,
        }
        self.llm = provider
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


def _json_object(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _normalise_statements(plan: dict) -> list[str]:
    values = plan.get("statements")
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        text = str(value).strip()
        if text and not text.endswith("?") and len(text) <= 500:
            out.append(text)
    return out[:12]


def _call(ctx, name, params):
    raw = ctx.tools[name](params)
    return json.loads(raw) if isinstance(raw, str) else raw


def _probe(ctx, round_index: int) -> list[dict]:
    rows = []
    for question, expected in PROBES:
        result = _call(ctx, "nanlm_chat", {
            "message": question, "session_id": "nanlm-gemma-over-time",
            "probe": True,
        })
        response = str(result.get("response") or "")
        hit = any(word.casefold() in response.casefold() for word in expected)
        layered = result.get("layered_learning") or {}
        rows.append({
            "question": question,
            "expected": list(expected),
            "response": response,
            "hit": hit,
            "structured": bool((result.get("native_synthesis") or {}).get("structured")),
            "layers": layered.get("layers"),
            "relations": layered.get("relations"),
            "scratch_records": layered.get("scratch_records"),
            "heads_active": (layered.get("head_balance") or {}).get("active_heads"),
            "provider": result.get("provider"),
            "native_learning_status": (result.get("native_learning") or {}).get("status"),
        })
    return rows

def _state_fingerprint(root: Path) -> str:
    """Hash persisted learner state, excluding SQLite WAL/SHM sidecars."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(("-wal", "-shm")):
            continue
        if "conversations" in path.parts or "session" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _metrics(ctx) -> dict:
    learning = _call(ctx, "nanlm_learning_status", {})
    return {
        "turns": learning.get("turns"),
        "rows": learning.get("rows"),
        "scratch_records": (learning.get("scratch") or {}).get("records"),
        "head_balance": learning.get("head_ensemble"),
        "concept_saturation": learning.get("concept_saturation"),
        "dictionary_hw4": learning.get("dictionary_hw4"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("NANLM_REVIEW_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("NANLM_REVIEW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        raise SystemExit("--rounds must be 1..10")

    reset_coordinator_cache()
    reset_leaf_cache()
    root = Path(args.state_dir).expanduser() if args.state_dir else Path(
        os.environ.get("TMPDIR", "/tmp")) / "nanlm-gemma-learning-over-time"
    root.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.url, args.model)
    ctx = Context(root, teacher)
    register(ctx)

    record = {
        "schema": "nanlm.gemma-learning-over-time/1",
        "model": args.model,
        "state_dir": str(root),
        "rounds_requested": args.rounds,
        "target_facts": list(TARGET_FACTS),
        "probes": [{"question": q, "expected": list(e)} for q, e in PROBES],
        "baseline": None,
        "rounds": [],
        "boundaries": {
            "local_native_student": True,
            "gemma_is_teacher_only": True,
            "canonical_commit": False,
            "proof_claim": False,
            "nannet_configured": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
        os.replace(temporary, output)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    print(json.dumps({"phase": "baseline", "state_dir": str(root)}), flush=True)
    baseline_before = _state_fingerprint(root)
    baseline_probes = _probe(ctx, 0)
    baseline_after = _state_fingerprint(root)
    baseline_vector = "".join("1" if row["hit"] else "0" for row in baseline_probes)
    record["baseline"] = {
        "probes": baseline_probes,
        "hits": sum(1 for row in baseline_probes if row["hit"]),
        "total": len(baseline_probes),
        "fraction": "%d/%d" % (sum(1 for row in baseline_probes if row["hit"]), len(baseline_probes)),
        "percent": round(100.0 * sum(1 for row in baseline_probes if row["hit"]) / len(baseline_probes), 2),
        "hit_rate": sum(1 for row in baseline_probes if row["hit"]) / len(baseline_probes),
        "hit_vector": baseline_vector,
        "structured_hit_vector": "".join(
            "1" if row["hit"] and row.get("structured") else "0"
            for row in baseline_probes),
        "probe_state_digest_before": baseline_before,
        "probe_state_digest_after": baseline_after,
        "probe_state_changed": baseline_before != baseline_after,
        "metrics": _metrics(ctx),
    }
    save()

    for round_index in range(1, args.rounds + 1):
        design = teacher.design(round_index, record["rounds"])
        plan = _json_object(design["text"])
        statements = _normalise_statements(plan)
        if not statements:
            # Deterministic fallback so a teacher formatting failure does not
            # silently turn the experiment into a no-op.
            statements = list(TARGET_FACTS)[:8]
        print(json.dumps({"phase": "teach", "round": round_index,
                          "statements": len(statements)}), flush=True)
        teaching = []
        for statement_index, statement in enumerate(statements, start=1):
            print(json.dumps({"phase": "teach_turn", "round": round_index,
                              "turn": statement_index,
                              "of": len(statements)}), flush=True)
            result = _call(ctx, "nanlm_chat", {
                "message": statement, "session_id": "nanlm-gemma-over-time",
            })
            teaching.append({
                "statement": statement,
                "response": result.get("response"),
                "provider": result.get("provider"),
                "layers": (result.get("layered_learning") or {}).get("layers"),
                "relations": (result.get("layered_learning") or {}).get("relations"),
                "native_learning_status": (result.get("native_learning") or {}).get("status"),
            })
        print(json.dumps({"phase": "probe", "round": round_index}), flush=True)
        probe_before = _state_fingerprint(root)
        probes = _probe(ctx, round_index)
        probe_after = _state_fingerprint(root)
        hit_count = sum(1 for row in probes if row["hit"])
        hit_rate = hit_count / len(probes)
        structured_hit_rate = sum(
            1 for row in probes if row["hit"] and row.get("structured")) / len(probes)
        hit_vector = "".join("1" if row["hit"] else "0" for row in probes)
        previous_vector = ((record["rounds"][-1].get("hit_vector")
                            if record["rounds"] else record["baseline"]["hit_vector"])
                           or "")
        flips = []
        if len(previous_vector) == len(hit_vector):
            for index, (before, after) in enumerate(zip(previous_vector, hit_vector)):
                if before != after:
                    flips.append({
                        "index": index,
                        "question": probes[index]["question"],
                        "before": "hit" if before == "1" else "miss",
                        "after": "hit" if after == "1" else "miss",
                    })
        record["rounds"].append({
            "round": round_index,
            "teacher_reasoning": plan.get("reasoning"),
            "teacher_model": design.get("model"),
            "statements": statements,
            "teaching": teaching,
            "probes": probes,
            "hits": hit_count,
            "total": len(probes),
            "fraction": "%d/%d" % (hit_count, len(probes)),
            "percent": round(100.0 * hit_rate, 2),
            "hit_rate": hit_rate,
            "hit_vector": hit_vector,
            "structured_hit_rate": structured_hit_rate,
            "structured_hit_vector": "".join(
                "1" if row["hit"] and row.get("structured") else "0"
                for row in probes),
            "flips_from_previous": flips,
            "probe_state_digest_before": probe_before,
            "probe_state_digest_after": probe_after,
            "probe_state_changed": probe_before != probe_after,
            "delta_from_baseline": hit_rate - record["baseline"]["hit_rate"],
            "metrics": _metrics(ctx),
        })
        save()
        print(json.dumps({"phase": "round_done", "round": round_index,
                          "hit_rate": hit_rate,
                          "baseline": record["baseline"]["hit_rate"]}), flush=True)

    # Markdown report.
    lines = [
        "# NaNLM learning over time with Gemma as teacher",
        "",
        "- model/teacher: `" + args.model + "`",
        "- student: NaNLM native mode (no LLM answerer)",
        "- rounds: " + str(args.rounds),
        "- state: `" + str(root) + "`",
        "- baseline hit rate: %.3f" % record["baseline"]["hit_rate"],
        "",
        "## Probe hit rate",
        "",
        "| stage | hits | keyword % | structured hits | hit vector | flips | turns | scratch | concepts | pairs | leaves |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    def metric_row(stage, hit, structured_hit, delta, metrics, vector="", flips=0):
        hits = sum(1 for row in (record["baseline"]["probes"] if stage == "baseline" else next((r for r in record["rounds"] if "round %d" % r["round"] == stage), {}).get("probes", [])) if row.get("hit")) if stage == "baseline" else int((next((r for r in record["rounds"] if "round %d" % r["round"] == stage), {}) or {}).get("hits", 0))
        total = len(record["baseline"]["probes"]) if stage == "baseline" else int((next((r for r in record["rounds"] if "round %d" % r["round"] == stage), {}) or {}).get("total", 0))
        structured_hits = sum(1 for row in (record["baseline"]["probes"] if stage == "baseline" else next((r for r in record["rounds"] if "round %d" % r["round"] == stage), {}).get("probes", [])) if row.get("hit") and row.get("structured"))
        structured_total = total or 1
        saturation = metrics.get("concept_saturation") or {}
        head = metrics.get("head_balance") or {}
        return ("| %s | %d/%d | %.2f%% | %d/%d | `%s` | %s | %s | %s | %s | %s | %s |" % (
            stage, hits, total, 100.0 * hit, structured_hits, structured_total,
            vector, flips, metrics.get("turns"),
            metrics.get("scratch_records"),
            saturation.get("concepts"), saturation.get("occupied_pairs"),
            head.get("active_heads")))
    baseline_structured = sum(
        1 for row in record["baseline"]["probes"]
        if row["hit"] and row.get("structured")) / len(record["baseline"]["probes"])
    lines.append(metric_row("baseline", record["baseline"]["hit_rate"],
                            baseline_structured, 0.0,
                            record["baseline"]["metrics"],
                            record["baseline"].get("hit_vector", ""), 0))
    for item in record["rounds"]:
        lines.append(metric_row("round %d" % item["round"], item["hit_rate"],
                                item.get("structured_hit_rate", 0.0),
                                item["delta_from_baseline"], item["metrics"],
                                item.get("hit_vector", ""),
                                len(item.get("flips_from_previous") or [])))
    lines.extend(["", "## Probe detail (final round)", ""])
    for row in (record["rounds"][-1]["probes"] if record["rounds"] else []):
        lines.append("- %s -> `%s` | expected %s | hit=%s" % (
            row["question"], row["response"], row["expected"], row["hit"]))
    lines.extend(["", "## Boundaries", "",
                  "- Local native student; Gemma designed statements only.",
                  "- No canonical merge, NANNET route or proof claim.",
                  "- A flat or noisy curve is a real result; inspect per-probe replies.",
                  ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    save()
    print(json.dumps({
        "output": str(output), "markdown": str(output.with_suffix(".md")),
        "baseline_hit_rate": record["baseline"]["hit_rate"],
        "final_hit_rate": record["rounds"][-1]["hit_rate"] if record["rounds"] else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
