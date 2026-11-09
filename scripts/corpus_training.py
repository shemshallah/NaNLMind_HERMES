#!/usr/bin/env python3
"""Train the native NaNLM temporal student on a local corpus collection.

This is a local derived-state experiment: it does not write canonical
NaNLMind state, call NANNET/CLOCKED, or merge. Records are parsed from JSONL
`text` fields, held-out by declared split or deterministic hash, trained for
multiple passes, and evaluated with held-out next-token log-loss/perplexity
after each pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.temporal_lm import BOS, EOS, TemporalMirrorLM, tokens  # noqa: E402

DEFAULT_OUTPUT = ROOT / "docs" / "CORPUS_TRAINING.json"
MAX_TEXT_CHARS = 4000


class Context:
    def __init__(self, root):
        self.config = {"state_dir": str(root), "chat_mode": "native",
                       "chat_learning": True}
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)


def _records_from_line(line: str) -> list[dict]:
    line = line.strip()
    if not line:
        return []
    decoder = json.JSONDecoder()
    values = []
    position = 0
    while position < len(line):
        try:
            value, end = decoder.raw_decode(line, position)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            values.append(value)
        position = end
        while position < len(line) and line[position] in " \t\r\n":
            position += 1
    return values


def load_corpus(root: Path, max_records: int | None = None) -> dict:
    train, heldout = [], []
    files = sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    seen = set()
    for path in files:
        file_rows = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            for record in _records_from_line(line):
                text = record.get("text")
                if not isinstance(text, str):
                    continue
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < 8 or len(text) > MAX_TEXT_CHARS:
                    continue
                key = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                split = str(record.get("split") or "").casefold()
                item = {
                    "record_id": key,
                    "text": text,
                    "source": str(path.relative_to(root)),
                    "stage": record.get("stage") or record.get("genre") or "",
                    "split": split or "derived",
                }
                if split in {"validation", "valid", "test", "heldout"}:
                    heldout.append(item)
                elif int(key[:8], 16) % 100 < 5:
                    heldout.append(item)
                else:
                    train.append(item)
                file_rows += 1
                if max_records is not None and len(train) + len(heldout) >= max_records:
                    return {"train": train, "heldout": heldout, "files": len(files)}
    return {"train": train, "heldout": heldout, "files": len(files)}


def evaluate(model: TemporalMirrorLM, records: list[dict], limit: int) -> dict:
    # Deterministic hash order prevents the first file in directory order from
    # dominating the held-out sample. Report per-source metrics as well.
    sample = sorted(records, key=lambda item: item.get("record_id") or item.get("text", ""))[:limit]
    total_log = 0.0
    count = 0
    sentences = 0
    by_source = {}
    for record in sample:
        toks = tokens(record["text"])
        if len(toks) < 2:
            continue
        sentences += 1
        for index in range(1, len(toks)):
            if toks[index] == BOS:
                continue
            context2 = toks[index - 2] if index >= 2 else None
            context1 = toks[index - 1]
            target = toks[index]
            if not model.vocab:
                probability = 1e-6
            else:
                try:
                    probability = model._prob(context2, context1, target)
                except Exception:
                    probability = 1e-6
            nll = -math.log(max(probability, 1e-9))
            total_log += nll
            count += 1
            bucket = by_source.setdefault(record.get("source") or "unknown", [0.0, 0])
            bucket[0] += nll
            bucket[1] += 1
    mean_nll = total_log / count if count else 0.0
    source_metrics = {
        str(source): {
            "tokens": values[1],
            "mean_negative_log_likelihood": round(values[0] / values[1], 6) if values[1] else 0.0,
            "perplexity": round(math.exp(min(values[0] / values[1], 50)), 4) if values[1] else None,
        }
        for source, values in sorted(by_source.items())
    }
    return {
        "records": len(sample),
        "sentences_scored": sentences,
        "tokens_scored": count,
        "mean_negative_log_likelihood": round(mean_nll, 6),
        "perplexity": round(math.exp(min(mean_nll, 50)), 4) if count else None,
        "by_source": source_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="directory containing JSONL corpus files")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--max-eval-records", type=int, default=200)
    parser.add_argument("--patience", type=int, default=2,
                        help="stop after this many epochs without held-out NLL improvement; 0 disables")
    parser.add_argument("--min-delta", type=float, default=1e-6)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        raise SystemExit("--epochs must be 1..20")
    corpus = Path(args.corpus).expanduser()
    if not corpus.is_dir():
        raise SystemExit("corpus directory does not exist: " + str(corpus))
    root = Path(args.state_dir).expanduser() if args.state_dir else Path(
        os.environ.get("TMPDIR", "/tmp")) / "nanlm-corpus-training"
    root.mkdir(parents=True, exist_ok=True)
    ctx = Context(root)
    loaded = load_corpus(corpus, max_records=args.max_records)
    train = loaded["train"]
    heldout = loaded["heldout"]
    if not train:
        raise SystemExit("no training records found")
    if not heldout:
        heldout = train[-max(1, len(train) // 20):]
        train = train[:-len(heldout)]
    record = {
        "schema": "nanlm.corpus-training/1",
        "corpus": str(corpus),
        "state_dir": str(root),
        "files": loaded["files"],
        "train_records": len(train),
        "heldout_records": len(heldout),
        "epochs": args.epochs,
        "before": None,
        "epochs_run": [],
        "boundaries": {
            "derived_local_state": True,
            "canonical_commit": False,
            "nannet_route": False,
            "merge": False,
            "proof_claim": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
        os.replace(temporary, output)

    model = TemporalMirrorLM.load(ctx)
    record["before"] = {
        "model": model.stats(),
        "heldout": evaluate(model, heldout, args.max_eval_records),
    }
    best_nll = record["before"]["heldout"]["mean_negative_log_likelihood"]
    best_epoch = 0
    patience = 0
    record["best_epoch"] = 0
    record["best_nll"] = best_nll
    record["stopped_early"] = False
    save()
    print(json.dumps({"phase": "before", "model": record["before"]["model"],
                      "heldout": record["before"]["heldout"]}), flush=True)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        # Deliberate repeated exposure: clear the replay digest guard so later
        # passes actually train the same corpus again.
        model.seen = []
        observed = 0
        rejected = 0
        for index, item in enumerate(train, start=1):
            if model.observe(item["text"]):
                observed += 1
            else:
                rejected += 1
            if index % 5000 == 0:
                print(json.dumps({"phase": "train", "epoch": epoch,
                                  "seen": index, "observed": observed}), flush=True)
        metrics = evaluate(model, heldout, args.max_eval_records)
        improved = metrics["mean_negative_log_likelihood"] < best_nll - args.min_delta
        if improved:
            best_nll = metrics["mean_negative_log_likelihood"]
            best_epoch = epoch
            patience = 0
            model.save(ctx)
        else:
            patience += 1
        record["epochs_run"].append({
            "epoch": epoch,
            "train_records": len(train),
            "observed": observed,
            "rejected": rejected,
            "seconds": round(time.time() - started, 2),
            "model": model.stats(),
            "heldout": metrics,
            "mean_nll_delta_from_before": round(
                metrics["mean_negative_log_likelihood"]
                - record["before"]["heldout"]["mean_negative_log_likelihood"], 6),
            "best_so_far": improved,
            "saved_checkpoint": improved,
        })
        record["best_epoch"] = best_epoch
        record["best_nll"] = best_nll
        record["patience"] = patience
        save()
        print(json.dumps({"phase": "epoch_done", "epoch": epoch,
                          "heldout": metrics, "best_epoch": best_epoch,
                          "best_nll": best_nll,
                          "seconds": record["epochs_run"][-1]["seconds"]}), flush=True)
        if args.patience and patience >= args.patience:
            record["stopped_early"] = True
            print(json.dumps({"phase": "early_stop", "epoch": epoch,
                              "best_epoch": best_epoch}), flush=True)
            break

    # The best validation checkpoint is the canonical local student state;
    # later overfit epochs are not allowed to replace it.
    if best_epoch > 0:
        model = TemporalMirrorLM.load(ctx)
        record["selected"] = {"epoch": best_epoch, "heldout":
                              next(item["heldout"] for item in record["epochs_run"]
                                   if item["epoch"] == best_epoch),
                              "model": model.stats()}
    else:
        record["selected"] = {"epoch": 0, "heldout": record["before"]["heldout"],
                              "model": model.stats()}
    save()
    lines = [
        "# Corpus training",
        "",
        "- corpus: `" + str(corpus) + "`",
        "- state: `" + str(root) + "`",
        "- files: " + str(loaded["files"]),
        "- train records: " + str(len(train)),
        "- held-out records: " + str(len(heldout)),
        "- selected checkpoint: epoch " + str(record["selected"]["epoch"]),
        "- stopped early: " + str(record["stopped_early"]),
        "",
        "| epoch | observed | held-out NLL | perplexity | vocab | bigram ctx | trigram ctx | seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    before = record["before"]
    lines.append("| 0 | 0 | %.4f | %s | %s | %s | %s | 0 |" % (
        before["heldout"]["mean_negative_log_likelihood"],
        before["heldout"]["perplexity"], before["model"]["vocabulary"],
        before["model"]["bigram_contexts"], before["model"]["trigram_contexts"]))
    for item in record["epochs_run"]:
        lines.append("| %d | %d | %.4f | %s | %d | %d | %d | %s |" % (
            item["epoch"], item["observed"],
            item["heldout"]["mean_negative_log_likelihood"],
            item["heldout"]["perplexity"], item["model"]["vocabulary"],
            item["model"]["bigram_contexts"], item["model"]["trigram_contexts"],
            item["seconds"]))
    lines.extend(["", "## Selected held-out metrics by source", "",
                  "| source | tokens | NLL | perplexity |",
                  "|---|---:|---:|---:|"])
    for source, item in (record["selected"]["heldout"].get("by_source") or {}).items():
        lines.append("| `%s` | %d | %.4f | %s |" % (
            source, item["tokens"], item["mean_negative_log_likelihood"],
            item["perplexity"]))
    lines.extend(["", "## Boundaries", "",
                  "- Local derived temporal state; no canonical merge.",
                  "- Held-out NLL/perplexity is a next-token measure, not general language understanding.",
                  "- Per-source metrics prevent a hard domain from being hidden by a favorable mixture sample.",
                  "- Repeated passes clear only the replay digest guard; no corpus is uploaded.", ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    save()
    print(json.dumps({"output": str(output), "markdown": str(output.with_suffix(".md")),
                      "before_nll": before["heldout"]["mean_negative_log_likelihood"],
                      "final_nll": record["selected"]["heldout"]["mean_negative_log_likelihood"],
                      "final_perplexity": record["selected"]["heldout"]["perplexity"],
                      "selected_epoch": record["selected"]["epoch"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
