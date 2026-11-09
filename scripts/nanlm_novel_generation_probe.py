#!/usr/bin/env python3
"""Probe NaNLM's post-training output generation on novel prompts.

The student is native NaNLM in an existing state directory. Prompts are open
ended and compositional (comparison, explanation, counterfactual, story) rather
than the taught statements. Responses are measured for length, repetition,
copying and novel n-grams; an optional Gemma judge records rubric judgments.

This is not a benchmark of general language competence. It is a qualitative
post-training check of what the temporal/mirror learner actually emits.
"""
from __future__ import annotations

import argparse
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

DEFAULT_OUTPUT = ROOT / "docs" / "NANLM_NOVEL_GENERATION.json"
DEFAULT_TRAINING = ROOT / "docs" / "NANLM_GEMMA_LEARNING_OVER_TIME.json"
DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:31b-cloud"

PROMPTS = (
    "Explain why a river can carry water.",
    "Compare a cat and a bird.",
    "Why does ice melt when it gets hot?",
    "Describe what happens when snow falls on a warm day.",
    "What would happen if a bird could not fly?",
    "Tell a short story about a river, a bird, and the sun.",
    "Explain how water can become ice and then water again.",
    "Why is the sky blue but grass green?",
)

JUDGE_SYSTEM = """You are an evaluator, not the learner. Judge the NaNLM reply
against the prompt only. Return STRICT JSON with keys:
answers_prompt (0/1), coherent (0/1), novel_combination (0/1),
unsupported_claim (0/1), copied_training_sentence (0/1), notes (short string).
Do not reward fluency alone; a fluent but unrelated answer is not coherent.
Do not treat the reply as ground truth."""


class Context:
    def __init__(self, root):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "native",
            "chat_learning": False,
        }
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


class Judge:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def __call__(self, prompt: str, response: str, training: list[str]) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps({
                    "prompt": prompt,
                    "reply": response,
                    "training_statements": training[:80],
                }, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=180) as response:
            value = json.loads(response.read(262144))
        text = value["choices"][0]["message"].get("content") or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {"error": "judge returned no JSON"}
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {"error": "judge returned invalid JSON"}
        return data if isinstance(data, dict) else {"error": "judge JSON is not an object"}


def _call(ctx, name, params):
    raw = ctx.tools[name](params)
    return json.loads(raw) if isinstance(raw, str) else raw


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index:index + n])
            for index in range(max(0, len(tokens) - n + 1))}


def _training_text(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_file():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    statements = []
    for round_item in data.get("rounds", []):
        statements.extend(str(x) for x in round_item.get("statements", []))
    probes = [str(row.get("question")) for row in
              (data.get("baseline") or {}).get("probes", [])]
    return statements, probes


def _metrics(response: str, training: list[str], probes: list[str]) -> dict:
    tokens = _tokens(response)
    training_text = " ".join(training).casefold()
    probe_text = " ".join(probes).casefold()
    tri = _ngrams(tokens, 3)
    training_tri = _ngrams(_tokens(training_text), 3)
    probe_tri = _ngrams(_tokens(probe_text), 3)
    novel = tri - training_tri - probe_tri
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return {
        "characters": len(response),
        "tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "novel_3grams": len(novel),
        "novel_3gram_ratio": round(len(novel) / len(tri), 4) if tri else 0.0,
        "max_token_repetition": max(counts.values()) if counts else 0,
        "copied_training_sentence": any(
            response.strip().casefold() == statement.strip().casefold()
            for statement in training),
        "empty": not bool(response.strip()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--training-results", default=str(DEFAULT_TRAINING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--judge", action="store_true",
                        help="ask Gemma for rubric judgments (does not generate the reply)")
    parser.add_argument("--url", default=os.environ.get("NANLM_REVIEW_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("NANLM_REVIEW_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    root = Path(args.state_dir).expanduser()
    if not root.is_dir():
        raise SystemExit("state dir does not exist: " + str(root))
    training, probes = _training_text(Path(args.training_results))
    ctx = Context(root)
    register(ctx)
    judge = Judge(args.url, args.model) if args.judge else None
    rows = []
    for prompt in PROMPTS:
        result = _call(ctx, "nanlm_chat", {
            "message": prompt, "session_id": "nanlm-novel-generation",
        })
        response = str(result.get("response") or "")
        synthesis = result.get("native_synthesis") or {}
        row = {
            "prompt": prompt,
            "response": response,
            "provider": result.get("provider"),
            "model": result.get("model"),
            "surface": _metrics(response, training, probes),
            "synthesis": {
                "synthesis": synthesis.get("synthesis"),
                "response_templates": synthesis.get("response_templates"),
                "retrieval": synthesis.get("retrieval"),
                "learned_records": synthesis.get("learned_records"),
                "vocabulary": synthesis.get("vocabulary"),
                "heads": synthesis.get("heads"),
                "temporal_frame": {
                    key: (synthesis.get("temporal_frame") or {}).get(key)
                    for key in ("anchor", "steps", "score", "coverage",
                                "output_drive", "mirror_state")
                },
            },
        }
        if judge is not None:
            try:
                row["judge"] = judge(prompt, response, training)
            except Exception as exc:
                row["judge"] = {"error": type(exc).__name__ + ": " + str(exc)[:300]}
        rows.append(row)

    judged = [row.get("judge", {}) for row in rows if judge is not None]
    record = {
        "schema": "nanlm.novel-generation-probe/1",
        "state_dir": str(root),
        "training_results": str(Path(args.training_results)),
        "prompts": list(PROMPTS),
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "empty_responses": sum(1 for row in rows if row["surface"]["empty"]),
            "mean_tokens": round(sum(row["surface"]["tokens"] for row in rows) / len(rows), 2),
            "mean_novel_3gram_ratio": round(
                sum(row["surface"]["novel_3gram_ratio"] for row in rows) / len(rows), 4),
            "copied_training_sentences": sum(
                1 for row in rows if row["surface"]["copied_training_sentence"]),
            "judge": {
                "answers_prompt": sum(int(item.get("answers_prompt", 0) or 0) for item in judged),
                "coherent": sum(int(item.get("coherent", 0) or 0) for item in judged),
                "novel_combination": sum(int(item.get("novel_combination", 0) or 0) for item in judged),
                "unsupported_claim": sum(int(item.get("unsupported_claim", 0) or 0) for item in judged),
                "errors": sum(1 for item in judged if item.get("error")),
            } if judge is not None else None,
        },
        "boundaries": {
            "native_temporal_student": True,
            "llm_generated_the_reply": False,
            "judge_is_not_ground_truth": True,
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
    lines = [
        "# NaNLM novel generation probe",
        "",
        "- state: `" + str(root) + "`",
        "- student: native temporal NaNLM; no LLM wrote the replies",
        "- prompts: " + str(len(rows)),
        "- empty replies: " + str(record["summary"]["empty_responses"]),
        "- mean tokens: " + str(record["summary"]["mean_tokens"]),
        "- mean novel 3-gram ratio: " + str(record["summary"]["mean_novel_3gram_ratio"]),
        "- copied training sentences: " + str(record["summary"]["copied_training_sentences"]),
        "",
        "## Replies",
        "",
    ]
    for row in rows:
        lines.append("### " + row["prompt"])
        lines.append("")
        lines.append("> " + (row["response"] or "(empty)"))
        lines.append("")
        lines.append("- " + json.dumps(row["surface"], sort_keys=True))
        if row.get("judge"):
            lines.append("- judge: " + json.dumps(row["judge"], sort_keys=True))
        lines.append("")
    lines.extend(["## Boundaries", "",
                  "- Native temporal learner, not a general language model.",
                  "- Optional Gemma judge is a diagnostic, not ground truth.",
                  "- No canonical merge, route or proof claim.",
                  ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": record["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
