#!/usr/bin/env python3
"""Run a provider-universal LLM teacher against the HERMES layered learner.

Examples:
    # Local Gemma through Ollama, one bounded batch
    python scripts/llm_teacher_run.py --provider ollama \
        --model gemma4:31b-cloud --topic "basic metals" --count 4

    # Any OpenAI-compatible local gateway
    python scripts/llm_teacher_run.py --provider openai \
        --endpoint http://127.0.0.1:8000/v1/chat/completions \
        --model local-model --topic "physical properties" --count 6

    # Parallel loop, one batch every 60 seconds
    python scripts/llm_teacher_run.py --cycles 0 --interval 60 \
        --topic metals --topic electricity

The teacher proposes typed evidence. It never commits canonical NaNLMind state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanlmind_hermes.gemma_teacher import (  # noqa: E402
    HermesLLMTransport,
    LLMTeacher,
    OllamaTransport,
    OpenAICompatibleTransport,
)
from nanlmind_hermes.layered_learning import (  # noqa: E402
    LayeredLearningCoordinator,
    LearningPolicy,
)
from nanlmind_hermes.wordnet_store import WordNetStore  # noqa: E402


def build_teacher(args):
    if args.provider == "ollama":
        return LLMTeacher(OllamaTransport(endpoint=args.endpoint, model=args.model))
    if args.provider == "openai":
        if not args.endpoint or not args.model:
            raise SystemExit("--endpoint and --model are required for openai")
        return LLMTeacher(OpenAICompatibleTransport(
            endpoint=args.endpoint, model=args.model, api_key=args.api_key or ""))
    raise SystemExit("this CLI supports ollama or openai; use the plugin for the Hermes provider")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model", default="gemma4:31b-cloud")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=1, help="0 = run until interrupted")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--state-dir", default="~/.local/share/nanlmind-hermes")
    parser.add_argument("--wordnet", default="")
    parser.add_argument("--approve", action="store_true",
                        help="promote eligible candidates after each batch")
    args = parser.parse_args(argv)
    topics = args.topic or ["basic physical properties"]
    root = Path(args.state_dir).expanduser()
    dictionary = None
    if args.wordnet:
        try:
            dictionary = WordNetStore.discover(args.wordnet).universe
        except Exception:
            dictionary = None
    coordinator = LayeredLearningCoordinator(
        root, dictionary=dictionary,
        policy=LearningPolicy(min_support=2, min_confidence=0.5,
                              require_explicit_approval=True, auto_sync=False),
    )
    teacher = build_teacher(args)
    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            for topic in topics:
                result = teacher.teach(coordinator, topic, args.count)
                output = {
                    "schema": "nanlm.llm-teacher-run/1",
                    "cycle": cycle,
                    "topic": topic,
                    "provider": args.provider,
                    "model": args.model,
                    "records": result["records"],
                    "turn": result["observation"]["turn"],
                    "scratch_records": result["observation"]["scratch"]["records"],
                    "canonical_training_committed": False,
                }
                if args.approve:
                    output["sync"] = coordinator.synchronize(approved=True)
                print(json.dumps(output, sort_keys=True), flush=True)
            cycle += 1
            if args.cycles == 0 or cycle < args.cycles:
                time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
