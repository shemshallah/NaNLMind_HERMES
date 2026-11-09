#!/usr/bin/env python3
"""Train the native student and resident pair geometry on a local corpus.

The run is local-derived only.  It never calls NANNET, never promotes canonical
state, and never enables merge.  The temporal LM supplies the primary
next-token distribution; the resident 256-head mirror-pair cells supply a
second, independently measured associative channel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.resident.learning import ResidentLearner  # noqa: E402
from nanlmind_hermes.resident.universe import ResidentUniverse  # noqa: E402
from scripts.corpus_training import load_corpus  # noqa: E402

DEFAULT_OUTPUT = ROOT / "docs" / "RESIDENT_LEARNING.json"
DEFAULT_SIZE = 256 * 1024 * 1024


def _sample(records: list[dict], limit: int | None) -> list[dict]:
    ordered = sorted(records, key=lambda item: item.get("record_id")
                     or item.get("text", ""))
    if limit is None or limit <= 0:
        return ordered
    return ordered[:limit]


def _markdown(record: dict) -> str:
    lines = [
        "# Resident corpus learning",
        "",
        "- corpus: `" + str(record["corpus"]) + "`",
        "- resident file: `" + str(record["state_dir"]) + "/resident/universe.nlmres`",
        "- train records: " + str(record["train_records"]),
        "- held-out records: " + str(record["heldout_records"]),
        "- selected epoch: " + str(record["selected_epoch"]),
        "- canonical merge: disabled",
        "",
        "## Held-out next-token metrics",
        "",
        "| pass | temporal NLL | temporal ppl | resident NLL | resident ppl | hybrid NLL | hybrid ppl |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in record["passes"]:
        channels = item["heldout"]["channels"]
        lines.append(
            "| %s | %.4f | %s | %.4f | %s | %.4f | %s |" % (
                item["pass"],
                channels["temporal"]["mean_negative_log_likelihood"] or 0,
                channels["temporal"]["perplexity"],
                channels["resident"]["mean_negative_log_likelihood"] or 0,
                channels["resident"]["perplexity"],
                channels["hybrid"]["mean_negative_log_likelihood"] or 0,
                channels["hybrid"]["perplexity"],
            ))
    lines.extend([
        "",
        "## Resident state",
        "",
        "| item | value |",
        "|---|---:|",
        "| leaves | %d |" % record["resident"]["leaves"],
        "| relations | %d |" % record["resident"]["relations"],
        "| hyperedges | %d |" % record["resident"]["hyperedges"],
        "| features | %d |" % record["resident"]["features"],
        "| generation | %d |" % record["resident"]["generation"],
        "| digest | `%s` |" % record["resident"]["digest"],
        "",
        "## Resident association probe",
        "",
        "```json",
        json.dumps(record["probe"], indent=2, sort_keys=True),
        "```",
        "",
        "## Boundaries",
        "",
        "- Temporal and resident channels are reported separately; the stronger",
        "  temporal baseline cannot hide a flat resident channel.",
        "- The resident pair channel uses 256 hashed heads. Collisions are real and",
        "  are why it is an associative supplement, not the primary LM.",
        "- Per-source metrics are written in the JSON companion.",
        "- No canonical merge, promotion, NANNET upload, or 2 GiB overbootstrap map.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "experiments" / "corpus" / "trainable")
    parser.add_argument("--state-dir", type=Path,
                        default=Path("/data/data/com.termux/files/usr/tmp/resident-learning"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--eval-records", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--extract-relations", action="store_true",
                        help="derive typed relations from explicit definition prose")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--temporal-state", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.epochs > 20:
        parser.error("--epochs must be in 1..20")
    if args.size < 16 * 1024 * 1024:
        parser.error("--size must be at least 16 MiB")
    if args.size > 1024 * 1024 * 1024:
        parser.error("--size must not exceed 1 GiB")

    started = time.time()
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    universe_path = args.state_dir / "resident" / "universe.nlmres"
    temporal_path = args.temporal_state or (
        args.state_dir / "language" / "temporal_mirror.json")
    if args.fresh:
        for path in (universe_path, temporal_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    corpus = load_corpus(args.corpus,
                         max_records=args.max_records or None)
    train_records = list(corpus["train"])
    heldout_records = _sample(
        list(corpus["heldout"]), args.eval_records if args.eval_records > 0 else None)
    universe = ResidentUniverse.open_or_create(universe_path, size=args.size)
    learner = ResidentLearner(universe, alpha=args.alpha,
                              extract_relations=args.extract_relations)
    learner.load_temporal(temporal_path)

    before = learner.evaluate(heldout_records, limit=None)
    passes = [{"pass": "before", "heldout": before,
               "seconds": round(time.time() - started, 3)}]
    selected_epoch = 0
    selected = before
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        learned = learner.observe_batch(
            train_records, batch_size=args.batch_size)
        evaluation = learner.evaluate(heldout_records, limit=None)
        passes.append({
            "pass": "epoch-%d" % epoch,
            "heldout": evaluation,
            "learned": learned,
            "seconds": round(time.time() - epoch_started, 3),
        })
        epochs_run = epoch
        # Select the best temporal held-out NLL.  The resident channel remains
        # visible even when the temporal channel is the selected predictor.
        current = evaluation["channels"].get("temporal", {}).get(
            "mean_negative_log_likelihood")
        best = selected["channels"].get("temporal", {}).get(
            "mean_negative_log_likelihood")
        if current is not None and (best is None or current < best):
            selected_epoch = epoch
            selected = evaluation

    learner.save_temporal(temporal_path)
    universe.flush()
    checkpoint = None
    try:
        checkpoint = universe.checkpoint(args.state_dir / "checkpoints")
    except Exception as exc:  # checkpoint remains a separate gate
        checkpoint = {"error": type(exc).__name__ + ": " + str(exc)[:300]}
    status = universe.status()
    probe = learner.next_words("penguin bird", top_k=8)
    record = {
        "schema": "nanlm.resident-learning-run/1",
        "corpus": str(args.corpus),
        "state_dir": str(args.state_dir),
        "universe_path": str(universe_path),
        "temporal_state": str(temporal_path),
        "train_records": len(train_records),
        "heldout_records": len(heldout_records),
        "epochs_requested": args.epochs,
        "epochs_run": epochs_run,
        "selected_epoch": selected_epoch,
        "selected": selected,
        "passes": passes,
        "resident": {
            "generation": status["generation"],
            "digest": status["digest"],
            "leaves": status["counts"]["leaves"],
            "relations": status["counts"]["relations"],
            "hyperedges": status["counts"]["hyperedges"],
            "features": status["counts"]["features"],
            "file_bytes": status["file_bytes"],
            "committed_bytes": status["committed_bytes"],
        },
        "checkpoint": checkpoint,
        "probe": probe,
        "canonical_merge": {"enabled": False, "attempted": False},
        "seconds": round(time.time() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(args.output)
    args.output.with_suffix(".md").write_text(_markdown(record),
                                              encoding="utf-8")
    universe.close()
    print(json.dumps({
        "output": str(args.output), "markdown": str(args.output.with_suffix(".md")),
        "selected_epoch": selected_epoch,
        "train_records": len(train_records),
        "heldout_records": len(heldout_records),
        "generation": record["resident"]["generation"],
        "leaves": record["resident"]["leaves"],
        "relations": record["resident"]["relations"],
        "seconds": record["seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
