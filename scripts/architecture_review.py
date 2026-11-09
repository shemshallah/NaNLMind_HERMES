#!/usr/bin/env python3
"""Ask an attached teacher model to review the permanent session architecture.

The prompt is built from the exact NANNET/WordNet documents shipped with the
plugin, so the review is grounded in the same contract the plugin injects at
runtime. The default endpoint is the local Ollama OpenAI-compatible API and the
default model is the cloud Gemma model already configured on this host.

This is a host-side design review. It does not execute native training, route a
NANNET packet, or claim any measured result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import session_context  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_OUTPUT = ROOT / "docs" / "GEMMA_ARCHITECTURE_REVIEW.json"

SYSTEM = """You are Gemma, reviewing a from-scratch local learning architecture.
You are not the trainer and you do not own canonical state. The attached
documents are the permanent plugin-duration NANNET/WordNet/free-chat contract.
Be concrete, honest and test-oriented. Never claim a route, lease, metric or
native commit that has not actually been measured. Return STRICT JSON only."""

TASK = """Review the permanent session contract above and answer the operator's
current goal:

1. Free-chat learning: how should each user/assistant turn use the existing
   L0-L9 projections, 255-head ensemble, 1 GiB paged scratch, concept matrix,
   conservative promotion, and native relation-train path? Identify what should
   happen online versus behind explicit approval.
2. WordNet universe at HW4:0 through NANNET: how should WordNet reads be
   sequenced into free-chat learning without turning every word into a dense
   matrix or claiming prose understanding? Give a concrete flow and failure
   behavior.
3. Data leaves and hubs: how should every MTH tool use immutable,
   NaN-executable leaves with function definitions, action relations,
   identities, constraints/constants and a vector/WordNet hub? How should
   `left::right` decide connect versus mathematical pairing, and how should
   illegal pairings be denied without execution or mutation?
4. Missing architecture: list the highest-value gaps that currently prevent a
   reliable end-to-end test of free-chat learning plus HW4:0 ingestion.
5. Testability first: propose a bounded test plan with observable assertions,
   including what can run locally without NANNET and what must fail closed.
6. First experiment: one concrete, low-cost next experiment that would produce
   evidence rather than a story.

Return JSON with exactly these keys:
{
  "reasoning": "short summary",
  "architecture_assessment": ["..."],
  "free_chat_learning_plan": ["..."],
  "wordnet_hw4_plan": ["..."],
  "data_leaf_plan": ["..."],
  "missing_architecture": ["..."],
  "test_plan": ["..."],
  "first_experiment": {
     "name": "...",
     "command_or_api": "...",
     "assertions": ["..."],
     "what_it_would_prove": "...",
     "what_it_would_not_prove": "..."
  }
}
Use 4-8 concise items per list. Do not output markdown fences or extra keys."""


def build_prompt(context: session_context.SessionContext) -> str:
    return context.prompt_block() + "\n\n---\n\n" + TASK


def _extract_json(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("teacher response did not contain a JSON object")
    value = json.loads(candidate[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher response JSON is not an object")
    return value


def complete(url: str, model: str, prompt: str, *, timeout: int = 300,
             max_tokens: int = 3000) -> dict:
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.25,
        "max_tokens": max_tokens,
    }, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(262144)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("model endpoint rejected the review (HTTP %d)" % exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("model endpoint is unreachable: " + type(exc).__name__) from None
    value = json.loads(raw)
    try:
        text = value["choices"][0]["message"]["content"]
        usage = value.get("usage") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("model endpoint returned an invalid completion") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("model endpoint returned an empty completion")
    return {
        "text": text,
        "usage": usage if isinstance(usage, dict) else {},
        "model": model,
    }


def render_markdown(review: dict, *, model: str, usage: dict) -> str:
    def bullets(key: str) -> str:
        values = review.get(key)
        if not isinstance(values, list):
            return "- (not returned)\n"
        return "".join("- " + str(item) + "\n" for item in values) or "- (empty)\n"

    experiment = review.get("first_experiment")
    if not isinstance(experiment, dict):
        experiment = {}
    assertions = experiment.get("assertions")
    if not isinstance(assertions, list):
        assertions = []
    lines = [
        "# Gemma architecture review (live teacher run)",
        "",
        "Generated from the exact permanent session documents shipped with the "
        "plugin. This is a teacher review, not a measured training result.",
        "",
        "- model: `" + model + "`",
        "- prompt tokens: `" + str(usage.get("prompt_tokens", "unknown")) + "`",
        "- completion tokens: `" + str(usage.get("completion_tokens", "unknown")) + "`",
        "",
        "## Reasoning",
        "",
        str(review.get("reasoning") or "(not returned)"),
        "",
        "## Architecture assessment",
        "",
        bullets("architecture_assessment"),
        "## Free-chat learning plan",
        "",
        bullets("free_chat_learning_plan"),
        "## WordNet HW4:0 plan",
        "",
        bullets("wordnet_hw4_plan"),
        "## MTH data leaves and pairing plan",
        "",
        bullets("data_leaf_plan"),
        "## Missing architecture",
        "",
        bullets("missing_architecture"),
        "## Test plan",
        "",
        bullets("test_plan"),
        "## First experiment",
        "",
        "- name: " + str(experiment.get("name") or "(not returned)"),
        "- command/API: `" + str(experiment.get("command_or_api") or "") + "`",
        "- what it would prove: " + str(experiment.get("what_it_would_prove") or ""),
        "- what it would not prove: " + str(experiment.get("what_it_would_not_prove") or ""),
        "",
        "### Assertions",
        "",
        "".join("- " + str(item) + "\n" for item in assertions) or "- (none)\n",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("NANLM_REVIEW_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("NANLM_REVIEW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-tokens", type=int, default=3000)
    args = parser.parse_args()

    context = session_context.open_session(state_dir=None, persist=False)
    prompt = build_prompt(context)
    result = complete(args.url, args.model, prompt, max_tokens=args.max_tokens)
    review = _extract_json(result["text"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "nanlm.gemma-architecture-review/1",
        "session_context": context.manifest(),
        "prompt_sha256": session_context._sha256(prompt),
        "model": result["model"],
        "usage": result["usage"],
        "review": review,
        "raw_response": result["text"],
        "boundaries": {
            "teacher_review": True,
            "measured_training_result": False,
            "nannet_route_executed": False,
            "canonical_commit": False,
        },
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(review, model=result["model"],
                                        usage=result["usage"]), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(markdown),
                      "model": result["model"], "usage": result["usage"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
