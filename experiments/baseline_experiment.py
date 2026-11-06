#!/usr/bin/env python3
"""Model-planned logic -> chat -> words -> chat -> sentences -> chat baseline.

Runs the real HERMES plugin tools against the local NaNLMind native trainer and
records held-out metrics and conversational reflections. The model chooses each
round's topic/order from the live plan, rather than a hidden script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Model:
    def __init__(self, url: str, name: str, token: str = ""):
        self.url = url.rstrip("/")
        self.name = name
        self.token = token

    def complete(self, messages=None, **kwargs):
        body = json.dumps({
            "model": self.name,
            "messages": messages or [],
            "temperature": kwargs.get("temperature", 0.4),
            "max_tokens": kwargs.get("max_tokens", 1600),
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(
            self.url + "/chat/completions", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
        text = payload["choices"][0]["message"]["content"]
        return SimpleNamespace(text=text, provider="ollama", model=self.name)


class Context:
    def __init__(self, state_dir: str, native_command: list[str], model: Model):
        self.state_dir = state_dir
        self.native_command = native_command
        self.llm = model

    def get_config(self, key, default=None):
        if key == "state_dir":
            return self.state_dir
        if key == "runner_command":
            return self.native_command
        if key == "duration_minutes":
            return 1
        if key == "universe_layers":
            return 7
        return default


def _json_object(text: str) -> dict:
    import re
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def ask_plan(model: Model, mode: str, prior: list[dict]) -> dict:
    prompt = {
        "task": "Plan the next NaNLMind training round.",
        "mode": mode,
        "architecture": "c(256,2) unified NaNLMind; 256 heads; 32,640 pair cells; 4 NaN16 words/cell; 16 Boolean ops/head; L0-L6 learning layers; mandatory full-Boolean gate (256 heads x 4 truth rows = 1,024 exact checks plus composed gates) before language; native math-train is a bounded byte-bit task plus additive discovery memory.",
        "prior_results": prior,
        "rules": [
            "Return JSON only.",
            "Brainstorm one concrete training round for this mode before it runs.",
            "Choose a specific topic and record style, not a generic label.",
            "State a testable hypothesis and the measured signal that would support or refute it.",
            "Do not claim the round has already run or that it proves general learning.",
        ],
        "schema": {"topic": "specific topic", "reason": "one sentence",
                   "hypothesis": "testable statement", "record_style": "specific examples",
                   "success_signal": "measured comparison"},
    }
    try:
        reply = model.complete(messages=[
            {"role": "system", "content": "You are HERMES' training planner inside NaNLMind. " + __import__('nanlmind_hermes.chat', fromlist=['ARCHITECTURE_CONTEXT']).ARCHITECTURE_CONTEXT + " Return one JSON object only, with no prose around it."},
            {"role": "user", "content": json.dumps(prompt)},
        ], temperature=0.3, max_tokens=300).text
        plan = _json_object(reply)
        if not plan:
            raise ValueError("no JSON plan")
    except Exception:
        plan = {"topic": f"{mode} fundamentals", "reason": "fallback plan",
                "hypothesis": "the mode will beat its majority baseline",
                "record_style": "short varied examples", "success_signal": "held-out accuracy vs majority baseline"}
    if not plan.get("topic") and isinstance(plan.get("plan"), dict):
        nested = plan["plan"]
        if isinstance(nested, dict):
            nested.setdefault("mode", mode)
            plan = nested
    plan.setdefault("topic", f"{mode} fundamentals")
    plan["mode"] = mode
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", default=os.environ.get("NANLM_MODEL_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--model", default=os.environ.get("NANLM_MODEL", "gemma4:31b-cloud"))
    parser.add_argument("--model-token", default=os.environ.get("NANLM_MODEL_TOKEN", ""))
    parser.add_argument("--native-command", default=os.environ.get("NANLM_NATIVE_COMMAND", ""))
    parser.add_argument("--state-dir", default="experiments/baseline_state")
    parser.add_argument("--out", default="experiments/baseline_1.0.1.json")
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args(argv)
    if not args.native_command:
        parser.error("--native-command or NANLM_NATIVE_COMMAND is required")
    native = [args.native_command]
    model = Model(args.model_url, args.model, args.model_token)
    ctx = Context(args.state_dir, native, model)
    from nanlmind_hermes import tools
    from nanlmind_hermes.chat import ARCHITECTURE_CONTEXT

    record = {"schema": "nanlmind-hermes-baseline/1", "model": args.model,
              "rounds": [], "chat": []}
    prior = []
    for mode in ("logic", "words", "sentences"):
        plan = ask_plan(model, mode, prior)
        training = tools.run_training_session(
            ctx, plan["topic"], mode, args.records, args.epochs, args.samples)
        math_result = training.get("training", {})
        foundation = math_result.get("boolean_foundation") or training.get("boolean_foundation")
        result = {
            "mode": mode,
            "plan": plan,
            "corpus": training.get("corpus"),
            "boolean_foundation": {
                "status": foundation.get("status") if foundation else None,
                "state_sha256": foundation.get("state_sha256") if foundation else None,
                "stages": foundation.get("stages", []) if foundation else [],
            },
            "training": {
                "status": math_result.get("status"),
                "head_count": math_result.get("head_count"),
                "heldout_accuracy": math_result.get("metrics", {}).get("heldout_accuracy"),
                "majority_baseline_accuracy": math_result.get("metrics", {}).get("majority_baseline_accuracy"),
            },
        }
        record["rounds"].append(result)
        prior.append({"mode": mode, "plan": plan, "metrics": result["training"]})
        chat = tools.chat(
            ctx,
            f"We completed the {mode} training round using topic '{plan['topic']}'. "
            f"Before the round you planned: {json.dumps(plan, ensure_ascii=False)}. "
            f"Full-Boolean gate: {json.dumps(result['boolean_foundation'], ensure_ascii=False)}. "
            f"Measured byte held-out accuracy: {result['training'].get('heldout_accuracy')}. "
            f"Majority baseline: {result['training'].get('majority_baseline_accuracy')}. "
            "Now participate in the result: explain what the metric supports, what it does not, "
            "and propose the next concrete round. Do not claim proof.",
            session_id=f"baseline-{mode}",
            system_prompt=ARCHITECTURE_CONTEXT)
        record["chat"].append({
            "mode": mode,
            "response": chat.get("response"),
            "reflection": chat.get("reflection"),
            "memory_merge": chat.get("reflection", {}).get("memory_merge"),
        })
    record["status"] = "passed" if all(
        r["training"].get("head_count") == 256 and r["training"].get("status") == "ok"
        and r["boolean_foundation"].get("status") == "passed"
        for r in record["rounds"]) else "failed"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
