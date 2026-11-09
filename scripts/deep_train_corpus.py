#!/usr/bin/env python3
"""Deep multi-epoch training for the native NaNLM temporal student.

This experiment loads every ``*.jsonl`` record from the corpus trainable
directory (schema ``nanlm.trainable.v1``), validates and deduplicates the
records, keeps a deterministic validation split, and trains the existing
``TemporalMirrorLM`` for several deliberate passes. After every pass the local
model is saved and a fixed held-out sample is scored for next-token
log-loss/perplexity. The run is local derived state: it does not merge, publish
canonical training, or make a proof claim.

Gemma is optional and only used as a teacher/evaluator through the existing
Ollama OpenAI-compatible chat endpoint. It may propose a bounded training plan
or completion probes built from TRAIN samples. It never writes student replies,
and held-out answer text is never placed in a teacher prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.temporal_lm import (  # noqa: E402
    SCHEMA as TEMPORAL_SCHEMA,
    TemporalMirrorLM,
    tokens,
)

TRAINABLE_SCHEMA = "nanlm.trainable.v1"
REPORT_SCHEMA = "nanlm.corpus-deep-training/1"
DEFAULT_TRAINABLE_DIR = ROOT / "experiments" / "corpus" / "trainable"
RELOCATED_TRAINABLE_DIR = ROOT / "experiments" / "corpus" / "trainable"
DEFAULT_OUTPUT = ROOT / "docs" / "CORPUS_DEEP_TRAINING_RESULTS.json"
DEFAULT_STATE_DIR = Path(
    os.environ.get("NANLM_CORPUS_STATE_DIR")
    or (Path.home() / ".local" / "share" / "nanlmind-hermes"
        / "corpus-deep-training")
)
DEFAULT_GEMMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_GEMMA_MODEL = "gemma4:31b-cloud"
DEFAULT_EPOCHS = 5
MAX_EPOCHS = 20
MIN_TEXT_CHARS = 2
MAX_TEXT_CHARS = 4000
MAX_TEACHER_SAMPLE_CHARS = 400
MAX_TEACHER_SAMPLES = 24
MAX_PROBES = 32
MAX_PROBE_EXPECTED_TOKENS = 8
MAX_PROBE_PROMPT_CHARS = 1000
VALIDATION_FALLBACK_PERCENT = 5
PERPLEXITY_CAP_NLL = 50.0

TRAIN_SPLITS = frozenset({"train", "training", "fit"})
VALIDATION_SPLITS = frozenset({
    "validation", "valid", "val", "dev", "test", "heldout", "hold-out",
    "holdout",
})

PLAN_SYSTEM = (
    "You are Gemma, a bounded teacher and evaluator for the native NaNLM "
    "temporal student. You never write student replies and you never receive "
    "held-out answer text. Reply with one strict JSON object only."
)
PROBE_SYSTEM = (
    "You are Gemma, an evaluator for the native NaNLM temporal student. You "
    "create completion probes from provided TRAIN samples only. You never "
    "write student replies. Reply with one strict JSON object only."
)


class TrainingContext:
    """Minimal context compatible with ``TemporalMirrorLM.load/save``."""

    def __init__(self, state_dir):
        self.config = {
            "state_dir": str(state_dir),
            "chat_mode": "native",
            "chat_learning": True,
        }
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class OllamaOpenAIProvider:
    """Small OpenAI-compatible client for the existing Ollama endpoint."""

    def __init__(self, url=DEFAULT_GEMMA_URL, model=DEFAULT_GEMMA_MODEL,
                 timeout=120.0):
        self.url = str(url)
        self.model = str(model)
        self.timeout = float(timeout)

    def complete(self, messages, temperature=0.2, max_tokens=768,
                 timeout=None):
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "max_tokens": max(1, min(int(max_tokens), 4096)),
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
                request, timeout=self.timeout if timeout is None else timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"]
        return SimpleNamespace(text=text, model=result.get("model", self.model))


def _iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_trainable_dir():
    """Prefer the corpus path, with the relocated corpus as a fallback."""
    if DEFAULT_TRAINABLE_DIR.is_dir():
        return DEFAULT_TRAINABLE_DIR
    if RELOCATED_TRAINABLE_DIR.is_dir():
        return RELOCATED_TRAINABLE_DIR
    return DEFAULT_TRAINABLE_DIR


def _sha256_file(path):
    try:
        hasher = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    except OSError:
        return None


def _digest_ids(records):
    hasher = hashlib.sha256()
    for record_id in sorted(record["record_id"] for record in records):
        hasher.update(record_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _new_counts():
    return {
        "records": 0,
        "accepted": 0,
        "train": 0,
        "validation": 0,
        "duplicates": 0,
        "rejected": 0,
        "invalid_schema": 0,
        "invalid_record_id": 0,
        "invalid_text": 0,
        "invalid_split": 0,
        "malformed": 0,
    }


def _reject(counts, reason):
    counts[reason] += 1
    counts["rejected"] += 1


def _records_from_line(line):
    """Parse one JSONL line, allowing several concatenated JSON objects."""
    decoder = json.JSONDecoder()
    values = []
    position = 0
    length = len(line)
    while position < length:
        while position < length and line[position] in " \t\r\n":
            position += 1
        if position >= length:
            break
        try:
            value, position = decoder.raw_decode(line, position)
        except json.JSONDecodeError:
            return [], False
        if not isinstance(value, dict):
            return [], False
        values.append(value)
    return values, True


def _normalise_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def _fallback_validation(train):
    """Deterministically move the lowest-hash records into validation."""
    if len(train) < 2:
        return train, [], {
            "used": False,
            "reason": "not enough training records for a held-out split",
            "moved": 0,
        }
    count = min(len(train) - 1, max(1, len(train) * VALIDATION_FALLBACK_PERCENT // 100))
    ranked = sorted(
        train,
        key=lambda item: hashlib.sha256(
            item["record_id"].encode("utf-8")).hexdigest(),
    )
    selected_ids = {item["record_id"] for item in ranked[:count]}
    selected = [
        {**item, "split": "validation"}
        for item in train if item["record_id"] in selected_ids
    ]
    remaining = [item for item in train if item["record_id"] not in selected_ids]
    return remaining, selected, {
        "used": True,
        "method": "sha256(record_id) rank, lowest percent",
        "percent": VALIDATION_FALLBACK_PERCENT,
        "moved": len(selected),
    }


def load_trainable(root, max_text_chars=MAX_TEXT_CHARS):
    """Load, validate, deduplicate and split every trainable JSONL record."""
    root = Path(root).expanduser()
    files = sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    train = []
    validation = []
    seen_ids = set()
    totals = _new_counts()
    per_file = []
    for path in files:
        counts = _new_counts()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
            counts["malformed"] += 1
            totals["malformed"] += 1
        for line_number, line in enumerate(lines, start=1):
            values, parsed = _records_from_line(line)
            if not parsed:
                counts["malformed"] += 1
                totals["malformed"] += 1
                continue
            for record in values:
                counts["records"] += 1
                totals["records"] += 1
                if record.get("schema") != TRAINABLE_SCHEMA:
                    _reject(counts, "invalid_schema")
                    _reject(totals, "invalid_schema")
                    continue
                record_id = record.get("record_id")
                if not isinstance(record_id, str) or not record_id.strip():
                    _reject(counts, "invalid_record_id")
                    _reject(totals, "invalid_record_id")
                    continue
                record_id = record_id.strip()
                if record_id in seen_ids:
                    counts["duplicates"] += 1
                    totals["duplicates"] += 1
                    continue
                text = record.get("text")
                if not isinstance(text, str):
                    _reject(counts, "invalid_text")
                    _reject(totals, "invalid_text")
                    continue
                text = _normalise_text(text)
                if len(text) < MIN_TEXT_CHARS or len(text) > max_text_chars:
                    _reject(counts, "invalid_text")
                    _reject(totals, "invalid_text")
                    continue
                split = record.get("split")
                if not isinstance(split, str) or not split.strip():
                    _reject(counts, "invalid_split")
                    _reject(totals, "invalid_split")
                    continue
                split = split.strip().casefold()
                if split not in TRAIN_SPLITS and split not in VALIDATION_SPLITS:
                    _reject(counts, "invalid_split")
                    _reject(totals, "invalid_split")
                    continue
                seen_ids.add(record_id)
                item = {
                    "record_id": record_id,
                    "text": text,
                    "split": split,
                    "stage": str(record.get("stage") or ""),
                    "source": str(record.get("source") or ""),
                    "source_file": str(path.relative_to(root)),
                    "line": line_number,
                }
                counts["accepted"] += 1
                totals["accepted"] += 1
                if split in VALIDATION_SPLITS:
                    validation.append(item)
                    counts["validation"] += 1
                    totals["validation"] += 1
                else:
                    train.append(item)
                    counts["train"] += 1
                    totals["train"] += 1
        per_file.append({"path": str(path.relative_to(root)), **counts})

    declared_validation = len(validation)
    if not validation:
        train, validation, fallback = _fallback_validation(train)
        totals["train"] = len(train)
        totals["validation"] = len(validation)
    else:
        fallback = {
            "used": False,
            "reason": "declared validation records present",
            "moved": 0,
        }

    stages = {}
    for record in train + validation:
        stage = record["stage"] or "(none)"
        bucket = stages.setdefault(stage, {"train": 0, "validation": 0})
        bucket["validation" if record["split"] in VALIDATION_SPLITS else "train"] += 1

    return {
        "root": str(root),
        "files": [str(path.relative_to(root)) for path in files],
        "per_file": per_file,
        "totals": totals,
        "train": train,
        "validation": validation,
        "declared_validation": declared_validation,
        "validation_fallback": fallback,
        "stages": dict(sorted(stages.items())),
        "digests": {
            "corpus": _digest_ids(train + validation),
            "train": _digest_ids(train),
            "validation": _digest_ids(validation),
        },
    }


def select_eval_sample(validation, limit):
    """Return a deterministic fixed held-out sample ordered by record hash."""
    ordered = sorted(
        validation,
        key=lambda item: hashlib.sha256(
            ("eval:" + item["record_id"]).encode("utf-8")).hexdigest(),
    )
    if limit is not None and limit > 0:
        ordered = ordered[:limit]
    return ordered


def _token_probability(model, context2, context1, target):
    if not model.vocab:
        return 1e-6
    return model._prob(context2, context1, target)


def evaluate_heldout(model, records):
    """Score next-token NLL/perplexity on an already-fixed held-out sample."""
    total_log = 0.0
    count = 0
    sentences = 0
    known = 0
    for record in records:
        toks = tokens(record["text"])
        if len(toks) < 2:
            continue
        sentences += 1
        for index in range(1, len(toks)):
            context2 = toks[index - 2] if index >= 2 else None
            context1 = toks[index - 1]
            target = toks[index]
            probability = _token_probability(model, context2, context1, target)
            total_log += -math.log(max(probability, 1e-9))
            count += 1
            if target in model.vocab:
                known += 1
    mean_nll = total_log / count if count else 0.0
    return {
        "records": len(records),
        "record_ids_digest": _digest_ids(records),
        "sentences_scored": sentences,
        "tokens_scored": count,
        "target_tokens_in_vocabulary": known,
        "vocabulary_coverage": round(known / count, 6) if count else 0.0,
        "mean_negative_log_likelihood": round(mean_nll, 6),
        "perplexity": (round(math.exp(min(mean_nll, PERPLEXITY_CAP_NLL)), 4)
                       if count else None),
    }


def evaluate_probes(model, probes):
    """Score teacher-selected completion probes without generating replies."""
    rows = []
    total_log = 0.0
    count = 0
    for probe in probes:
        prompt_tokens = tokens(probe.get("prompt", ""))
        expected_tokens = tokens(probe.get("expected", ""))
        sequence = prompt_tokens + expected_tokens
        probe_log = 0.0
        probe_count = 0
        for index in range(len(prompt_tokens), len(sequence)):
            if index < 1:
                continue
            context2 = sequence[index - 2] if index >= 2 else None
            context1 = sequence[index - 1]
            target = sequence[index]
            probability = _token_probability(model, context2, context1, target)
            probe_log += -math.log(max(probability, 1e-9))
            probe_count += 1
        total_log += probe_log
        count += probe_count
        mean_nll = probe_log / probe_count if probe_count else 0.0
        rows.append({
            "source_record_id": probe.get("source_record_id", ""),
            "origin": probe.get("origin", ""),
            "prompt": str(probe.get("prompt", ""))[:160],
            "expected": str(probe.get("expected", ""))[:160],
            "tokens_scored": probe_count,
            "mean_negative_log_likelihood": round(mean_nll, 6),
            "perplexity": (round(math.exp(min(mean_nll, PERPLEXITY_CAP_NLL)), 4)
                           if probe_count else None),
        })
    mean_nll = total_log / count if count else 0.0
    return {
        "probes": len(probes),
        "tokens_scored": count,
        "mean_negative_log_likelihood": round(mean_nll, 6),
        "perplexity": (round(math.exp(min(mean_nll, PERPLEXITY_CAP_NLL)), 4)
                       if count else None),
        "rows": rows,
    }


def _metric_delta(current, baseline):
    if not baseline:
        return {"mean_nll_delta": None, "perplexity_delta": None,
                "nll_improved": None}
    delta = round(
        current["mean_negative_log_likelihood"]
        - baseline["mean_negative_log_likelihood"], 6)
    perplexity_delta = None
    if current.get("perplexity") is not None and baseline.get("perplexity") is not None:
        perplexity_delta = round(
            current["perplexity"] - baseline["perplexity"], 4)
    return {
        "mean_nll_delta": delta,
        "perplexity_delta": perplexity_delta,
        "nll_improved": delta < 0.0,
    }


def _metric_block(model, heldout, probes):
    block = {"model": model.stats(), "heldout": evaluate_heldout(model, heldout)}
    if probes is not None:
        block["probes"] = evaluate_probes(model, probes)
    return block


def train_epochs(model, train_records, heldout, epochs, save_epoch,
                 probes=None, progress=None):
    """Train repeated passes, saving and evaluating after every epoch."""
    before = _metric_block(model, heldout, probes)
    epochs_run = []
    previous_heldout = before["heldout"]
    previous_probes = before.get("probes")
    for epoch in range(1, epochs + 1):
        started = time.time()
        # Deliberate repeated exposure: clear ONLY the replay digest guard so
        # the same corpus is actually trained again. Counts and associations
        # remain intact.
        model.seen = []
        observed = 0
        rejected = 0
        for index, record in enumerate(train_records, start=1):
            if model.observe(record["text"]):
                observed += 1
            else:
                rejected += 1
            if progress is not None and index % 2000 == 0:
                progress({"phase": "train", "epoch": epoch, "seen": index,
                          "observed": observed, "rejected": rejected})
        checkpoint = save_epoch(epoch)
        if not isinstance(checkpoint, dict):
            checkpoint = {"epoch": epoch}
        heldout_metrics = evaluate_heldout(model, heldout)
        row = {
            "epoch": epoch,
            "train_records": len(train_records),
            "observed": observed,
            "rejected": rejected,
            "seconds": round(time.time() - started, 3),
            "state_dir": checkpoint.get("state_dir"),
            "checkpoint_path": checkpoint.get("model_path"),
            "checkpoint_sha256": checkpoint.get("model_sha256"),
            "checkpoint": checkpoint,
            "model": model.stats(),
            "heldout": heldout_metrics,
            "delta_from_previous": _metric_delta(heldout_metrics, previous_heldout),
            "delta_from_before": _metric_delta(heldout_metrics, before["heldout"]),
        }
        row["mean_nll_delta_from_before"] = row["delta_from_before"]["mean_nll_delta"]
        row["perplexity_delta_from_before"] = row["delta_from_before"]["perplexity_delta"]
        if probes is not None:
            probe_metrics = evaluate_probes(model, probes)
            row["probes"] = probe_metrics
            row["probes_delta_from_previous"] = _metric_delta(
                probe_metrics, previous_probes) if previous_probes else None
            row["probes_delta_from_before"] = _metric_delta(
                probe_metrics, before.get("probes"))
            previous_probes = probe_metrics
        previous_heldout = heldout_metrics
        epochs_run.append(row)
        if progress is not None:
            progress({"phase": "epoch_done", "epoch": epoch,
                      "observed": observed, "rejected": rejected,
                      "heldout": heldout_metrics,
                      "seconds": row["seconds"]})
    after = _metric_block(model, heldout, probes)
    after["delta_from_before"] = _metric_delta(
        after["heldout"], before["heldout"])
    if probes is not None:
        after["probes_delta_from_before"] = _metric_delta(
            after["probes"], before.get("probes"))
    return {"before": before, "epochs_run": epochs_run, "after": after}


def checkpoint_info(ctx, model, epoch):
    """Describe the model and optional chart checkpoint written this epoch."""
    state_dir = Path(str(ctx.get_config("state_dir"))).expanduser()
    model_path = TemporalMirrorLM.path(ctx)
    info = {
        "epoch": epoch,
        "state_dir": str(state_dir),
        "model_path": str(model_path),
        "model_present": model_path.exists(),
        "model_bytes": model_path.stat().st_size if model_path.exists() else 0,
        "model_sha256": _sha256_file(model_path),
        "temporal_schema": TEMPORAL_SCHEMA,
    }
    register_path = state_dir / "language" / "temporal_register.json"
    info["register_path"] = str(register_path)
    info["register_present"] = register_path.exists()
    info["register_sha256"] = _sha256_file(register_path)
    chart_dir = state_dir / "language" / "chart-checkpoints"
    info["chart_checkpoint_dir"] = str(chart_dir)
    runner = getattr(model, "runner", None)
    if runner is not None:
        try:
            status = runner.status()
        except Exception:
            status = {}
        generation = int(status.get("checkpoint_generation") or 0)
        info["chart_checkpoint_generation"] = generation
        info["chart_checkpoint_kind"] = status.get("checkpoint_kind")
        if generation:
            slot = (generation - 1) % 2
            chart_path = chart_dir / ("chart.%d.n16s" % slot)
            info["chart_checkpoint_path"] = str(chart_path)
            info["chart_checkpoint_present"] = chart_path.exists()
            if chart_path.exists():
                info["chart_checkpoint_bytes"] = chart_path.stat().st_size
                info["chart_checkpoint_sha256"] = _sha256_file(chart_path)
    return info


def _response_text(response):
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    raise ValueError("Gemma provider returned no text")


def parse_strict_json(text):
    """Parse exactly one JSON object, allowing one surrounding Markdown fence."""
    raw = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("teacher JSON must be an object")
    return value


def _teacher_samples(records, limit):
    ranked = sorted(
        records,
        key=lambda item: hashlib.sha256(
            item["record_id"].encode("utf-8")).hexdigest(),
    )
    return [{
        "record_id": record["record_id"],
        "stage": record["stage"],
        "source": record["source"],
        "text": record["text"][:MAX_TEACHER_SAMPLE_CHARS],
    } for record in ranked[:max(0, limit)]]


def _stage_counts(records):
    counts = {}
    for record in records:
        stage = record["stage"] or "(none)"
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _fallback_plan(default_epochs, reason):
    return {
        "rationale": "deterministic fallback:" + (" " + reason if reason else ""),
        "epochs": default_epochs,
        "max_records_per_epoch": 0,
        "focus_stages": [],
        "notes": "",
        "clamped": False,
    }


def normalise_plan(payload, default_epochs, max_epochs, train_count, stages):
    """Validate and bound a teacher plan; raise if the shape is unusable."""
    if not isinstance(payload, dict):
        raise ValueError("plan must be a JSON object")
    raw_epochs = payload.get("epochs", default_epochs)
    if isinstance(raw_epochs, bool) or not isinstance(raw_epochs, int):
        raise ValueError("plan epochs must be an integer")
    epochs = max(1, min(raw_epochs, max_epochs))
    raw_cap = payload.get("max_records_per_epoch", 0)
    if raw_cap is None:
        raw_cap = 0
    if isinstance(raw_cap, bool) or not isinstance(raw_cap, int):
        raise ValueError("plan max_records_per_epoch must be an integer")
    cap = max(0, min(raw_cap, train_count))
    raw_focus = payload.get("focus_stages", [])
    focus = []
    if isinstance(raw_focus, list):
        known = set(stages)
        for value in raw_focus:
            stage = str(value).strip()
            if stage and stage in known and stage not in focus:
                focus.append(stage)
    return {
        "rationale": str(payload.get("rationale") or "")[:2000],
        "epochs": epochs,
        "max_records_per_epoch": cap,
        "focus_stages": focus,
        "notes": str(payload.get("notes") or "")[:2000],
        "clamped": epochs != raw_epochs or cap != raw_cap,
    }


def request_plan(provider, train_records, default_epochs=DEFAULT_EPOCHS,
                 max_epochs=MAX_EPOCHS, max_samples=MAX_TEACHER_SAMPLES,
                 timeout=120.0):
    """Ask Gemma for a bounded plan using TRAIN samples only."""
    samples = _teacher_samples(train_records, max_samples)
    result = {
        "status": "fallback",
        "plan": _fallback_plan(default_epochs, "teacher unavailable"),
        "error": None,
        "samples": len(samples),
        "sample_ids_digest": _digest_ids(samples) if samples else None,
        "provider": "ollama-openai-compatible",
        "model": getattr(provider, "model", ""),
    }
    try:
        if provider is None:
            raise RuntimeError("no Gemma provider configured")
        payload = {
            "task": (
                "Design a bounded local training plan for the provided TRAIN "
                "samples. You do not write student replies. Focus stages may "
                "be ordered first; max_records_per_epoch 0 means all records."
            ),
            "constraints": {
                "epochs_min": 1,
                "epochs_max": max_epochs,
                "max_records_per_epoch_max": len(train_records),
                "max_records_per_epoch_zero_means_all": True,
            },
            "corpus": {
                "records": len(train_records),
                "stages": _stage_counts(train_records),
                "samples": samples,
            },
            "required_json": {
                "rationale": "short reason",
                "epochs": default_epochs,
                "max_records_per_epoch": 0,
                "focus_stages": [],
                "notes": "short note",
            },
        }
        response = provider.complete(
            [{"role": "system", "content": PLAN_SYSTEM},
             {"role": "user", "content": json.dumps(payload, sort_keys=True)}],
            temperature=0.2, max_tokens=700, timeout=timeout)
        plan = normalise_plan(
            parse_strict_json(_response_text(response)),
            default_epochs=default_epochs,
            max_epochs=max_epochs,
            train_count=len(train_records),
            stages=_stage_counts(train_records),
        )
        result.update({"status": "applied", "plan": plan, "error": None})
    except Exception as exc:  # Teacher failure must not stop local training.
        reason = str(exc)[:500] or exc.__class__.__name__
        result.update({
            "status": "fallback",
            "plan": _fallback_plan(default_epochs, reason),
            "error": reason,
        })
    return result


def ordered_train_records(records, plan):
    """Apply bounded focus ordering/cap without changing record contents."""
    if not plan:
        return list(records)
    focus = set(plan.get("focus_stages") or [])
    if focus:
        focused = [record for record in records if record["stage"] in focus]
        rest = [record for record in records if record["stage"] not in focus]
        records = focused + rest
    else:
        records = list(records)
    cap = int(plan.get("max_records_per_epoch") or 0)
    if cap > 0:
        records = records[:cap]
    return records


def _normalise_probes(raw_probes, by_id, limit):
    probes = []
    rejected = 0
    reasons = {}
    seen = set()

    def reject(reason):
        nonlocal rejected
        rejected += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    for item in raw_probes:
        if not isinstance(item, dict):
            reject("not_object")
            continue
        source_id = str(item.get("source_record_id") or "").strip()
        sample = by_id.get(source_id)
        prompt = item.get("prompt")
        expected = item.get("expected")
        if sample is None:
            reject("unknown_source")
            continue
        if not isinstance(prompt, str) or not isinstance(expected, str):
            reject("missing_text")
            continue
        prompt = prompt.strip()
        expected = expected.strip()
        if not prompt or not expected:
            reject("empty_text")
            continue
        source_text = sample["text"]
        if not source_text.startswith(prompt):
            reject("prompt_not_prefix")
            continue
        if (len(prompt) < len(source_text)
                and not source_text[len(prompt)].isspace()):
            reject("prompt_not_boundary")
            continue
        remainder = source_text[len(prompt):].lstrip()
        if not remainder.startswith(expected):
            reject("expected_not_continuation")
            continue
        tail = remainder[len(expected):]
        if tail and (tail[0].isalnum() or tail[0] in "'’"):
            reject("expected_not_boundary")
            continue
        if not tokens(prompt):
            reject("empty_prompt_tokens")
            continue
        expected_tokens = tokens(expected)
        if not expected_tokens:
            reject("empty_expected_tokens")
            continue
        if len(expected_tokens) > MAX_PROBE_EXPECTED_TOKENS:
            reject("expected_too_long")
            continue
        if len(prompt) > MAX_PROBE_PROMPT_CHARS:
            reject("prompt_too_long")
            continue
        key = (source_id, prompt, expected)
        if key in seen:
            reject("duplicate")
            continue
        seen.add(key)
        probes.append({
            "source_record_id": source_id,
            "prompt": prompt,
            "expected": expected,
            "origin": "gemma",
        })
        if len(probes) >= limit:
            break
    return probes, rejected, reasons


def deterministic_probes(train_records, limit=8):
    """Build completion probes directly from normalized TRAIN text."""
    probes = []
    ranked = sorted(
        train_records,
        key=lambda item: hashlib.sha256(
            item["record_id"].encode("utf-8")).hexdigest(),
    )
    for record in ranked:
        words = record["text"].split()
        if len(words) < 4:
            continue
        cut = max(1, min(len(words) - 1, int(len(words) * 2 / 3)))
        prompt = " ".join(words[:cut])
        expected = " ".join(words[cut:cut + MAX_PROBE_EXPECTED_TOKENS])
        if not tokens(prompt) or not tokens(expected):
            continue
        probes.append({
            "source_record_id": record["record_id"],
            "prompt": prompt,
            "expected": expected,
            "origin": "deterministic",
        })
        if len(probes) >= limit:
            break
    return probes


def request_probes(provider, train_records, limit=MAX_PROBES,
                   max_samples=MAX_TEACHER_SAMPLES, timeout=120.0):
    """Ask Gemma for completion probes derived only from TRAIN samples."""
    samples = _teacher_samples(train_records, max_samples)
    by_id = {record["record_id"]: record for record in train_records}
    result = {
        "status": "fallback",
        "probes": [],
        "rejected": 0,
        "rejected_reasons": {},
        "error": None,
        "samples": len(samples),
        "sample_ids_digest": _digest_ids(samples) if samples else None,
        "provider": "ollama-openai-compatible",
        "model": getattr(provider, "model", ""),
    }
    try:
        if provider is None:
            raise RuntimeError("no Gemma provider configured")
        payload = {
            "task": (
                "Create completion probes for evaluation only, derived ONLY "
                "from these TRAIN samples. Each probe must use an exact "
                "source_record_id, an exact prefix prompt ending at a word "
                "boundary, and the exact next continuation from that sample. "
                "Do not invent text, ask questions, or write student replies."
            ),
            "max_probes": limit,
            "max_expected_tokens": MAX_PROBE_EXPECTED_TOKENS,
            "samples": samples,
            "required_json": {
                "probes": [{
                    "source_record_id": "sample id",
                    "prompt": "exact prefix of sample text",
                    "expected": "exact continuation from sample text",
                }],
            },
        }
        response = provider.complete(
            [{"role": "system", "content": PROBE_SYSTEM},
             {"role": "user", "content": json.dumps(payload, sort_keys=True)}],
            temperature=0.1, max_tokens=1200, timeout=timeout)
        parsed = parse_strict_json(_response_text(response))
        raw_probes = parsed.get("probes")
        if not isinstance(raw_probes, list):
            raise ValueError("probes must be a list")
        probes, rejected, reasons = _normalise_probes(
            raw_probes, by_id, limit)
        if not probes:
            raise ValueError("no valid teacher probes")
        result.update({
            "status": "applied",
            "probes": probes,
            "rejected": rejected,
            "rejected_reasons": reasons,
            "error": None,
        })
    except Exception as exc:  # Fall back to local sample-derived probes.
        reason = str(exc)[:500] or exc.__class__.__name__
        result.update({
            "status": "fallback",
            "probes": deterministic_probes(train_records, limit=min(8, limit)),
            "error": reason,
        })
    return result


def _json_atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _num(value, digits=4):
    if value is None:
        return "-"
    return ("%." + str(digits) + "f") % value


def _markdown(record):
    corpus = record["corpus"]
    totals = corpus["totals"]
    lines = [
        "# Deep corpus training",
        "",
        "- trainable dir: `" + record["trainable_dir"] + "`",
        "- files: " + str(corpus["files"]),
        "- records parsed: " + str(totals["records"]),
        "- accepted: " + str(totals["accepted"]),
        "- train: " + str(record["split"]["train_records"]),
        "- validation: " + str(record["split"]["validation_records"]),
        "- duplicates: " + str(totals["duplicates"]),
        "- rejected: " + str(totals["rejected"]),
        "- malformed lines: " + str(totals["malformed"]),
        "- epochs: " + str(record["epochs"]["requested"])
        + " (" + record["epochs"]["source"] + ")",
        "- state dir: `" + record["state"]["state_dir"] + "`",
        "- model checkpoint: `" + record["state"]["model_path"] + "`",
        "- fixed held-out sample: " + str(record["split"]["eval_sample_records"])
        + " records, digest `" + record["split"]["eval_sample_digest"] + "`",
        "",
        "## Per-file counts",
        "",
        "| file | records | accepted | train | validation | duplicates | rejected | malformed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in corpus["per_file"]:
        lines.append("| `%s` | %d | %d | %d | %d | %d | %d | %d |" % (
            item["path"], item["records"], item["accepted"], item["train"],
            item["validation"], item["duplicates"], item["rejected"],
            item["malformed"]))
    lines.extend([
        "",
        "## Held-out next-token metrics",
        "",
        "| epoch | observed | rejected | mean NLL | perplexity | ΔNLL vs before "
        "| vocab | bigram ctx | trigram ctx | model sha256 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    before = record["before"]
    initial_digest = record["state"].get("initial_model_sha256")
    lines.append("| 0 | 0 | 0 | %s | %s | - | %d | %d | %d | `%s` |" % (
        _num(before["heldout"]["mean_negative_log_likelihood"]),
        _num(before["heldout"]["perplexity"]),
        before["model"]["vocabulary"],
        before["model"]["bigram_contexts"],
        before["model"]["trigram_contexts"],
        (initial_digest or "-")[:16]))
    for row in record["epochs_run"]:
        checkpoint = row.get("checkpoint") or {}
        digest = checkpoint.get("model_sha256") or "-"
        lines.append("| %d | %d | %d | %s | %s | %s | %d | %d | %d | `%s` |" % (
            row["epoch"], row["observed"], row["rejected"],
            _num(row["heldout"]["mean_negative_log_likelihood"]),
            _num(row["heldout"]["perplexity"]),
            _num(row["mean_nll_delta_from_before"], 6),
            row["model"]["vocabulary"],
            row["model"]["bigram_contexts"],
            row["model"]["trigram_contexts"],
            digest[:16]))
    after = record["after"]
    lines.extend([
        "",
        "- after NLL: " + _num(after["heldout"]["mean_negative_log_likelihood"]),
        "- after perplexity: " + _num(after["heldout"]["perplexity"]),
        "- NLL delta from before: "
        + _num(after["delta_from_before"]["mean_nll_delta"], 6),
        "- final model sha256: `"
        + str(record["state"].get("final_model_sha256") or "-") + "`",
    ])
    if record["split"]["validation_fallback"].get("used"):
        fallback = record["split"]["validation_fallback"]
        lines.append("- validation fallback: moved " + str(fallback["moved"])
                     + " records by " + str(fallback["method"]))
    if record.get("gemma", {}).get("enabled"):
        gemma = record["gemma"]
        lines.extend([
            "",
            "## Gemma teacher/evaluator",
            "",
            "- endpoint: `" + str(gemma.get("endpoint")) + "`",
            "- model: `" + str(gemma.get("model")) + "`",
            "- plan status: " + str(gemma["plan"].get("status")),
        ])
        plan = gemma["plan"].get("plan") or {}
        if plan:
            lines.append("- plan epochs: " + str(plan.get("epochs")))
            lines.append("- plan max records per epoch: "
                         + str(plan.get("max_records_per_epoch")))
            lines.append("- plan focus stages: "
                         + repr(plan.get("focus_stages") or []))
        if gemma["plan"].get("error"):
            lines.append("- plan fallback reason: `"
                         + str(gemma["plan"]["error"]) + "`")
        lines.append("- probes status: " + str(gemma["probes"].get("status")))
        lines.append("- probes: " + str(len(gemma["probes"].get("probes") or [])))
        if gemma["probes"].get("rejected"):
            lines.append("- probes rejected: "
                         + str(gemma["probes"]["rejected"]))
        if gemma["probes"].get("error"):
            lines.append("- probe fallback reason: `"
                         + str(gemma["probes"]["error"]) + "`")
        lines.append("- probes are evaluation-only; no student reply was "
                     "written by Gemma")
        lines.append("- held-out text sent to teacher: no")
    if before.get("probes") is not None:
        lines.extend([
            "",
            "## Probe metrics",
            "",
            "| epoch | probe NLL | perplexity | ΔNLL vs before |",
            "|---:|---:|---:|---:|",
        ])
        lines.append("| 0 | %s | %s | - |" % (
            _num(before["probes"]["mean_negative_log_likelihood"]),
            _num(before["probes"]["perplexity"])))
        for row in record["epochs_run"]:
            metrics = row.get("probes")
            if metrics is None:
                continue
            delta = (row.get("probes_delta_from_before") or {}).get(
                "mean_nll_delta")
            lines.append("| %d | %s | %s | %s |" % (
                row["epoch"], _num(metrics["mean_negative_log_likelihood"]),
                _num(metrics["perplexity"]), _num(delta, 6)))
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- Local derived temporal state; no canonical merge or commit.",
        "- No proof claim: held-out NLL/perplexity is a next-token measure, "
        "not general language understanding.",
        "- Repeated passes clear only the replay digest guard; counts and "
        "associations are preserved.",
        "- Gemma is a teacher/evaluator only; it does not write student "
        "replies and is not the NaNLM runner.",
        "- Teacher prompts contain TRAIN samples only; held-out answer text "
        "is never sent.",
        "",
    ])
    return "\n".join(lines)


def write_reports(record, output):
    output = Path(output)
    _json_atomic_write(output, record)
    markdown_path = output.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_path.with_name(markdown_path.name + ".tmp")
    temporary.write_text(_markdown(record), encoding="utf-8")
    os.replace(temporary, markdown_path)
    return {"json": str(output), "markdown": str(markdown_path)}


def run_deep_training(trainable_dir, state_dir, output, epochs=None,
                      max_eval_records=200, gemma_provider=None,
                      gemma_plan=False, gemma_probes=False,
                      gemma_url=DEFAULT_GEMMA_URL,
                      gemma_model=DEFAULT_GEMMA_MODEL,
                      gemma_max_samples=MAX_TEACHER_SAMPLES,
                      gemma_timeout=120.0, checkpoint_fn=None,
                      progress=None):
    """Run loading, optional teacher calls, deep training and reporting."""
    corpus_dir = Path(trainable_dir).expanduser()
    if not corpus_dir.is_dir():
        raise ValueError("trainable directory does not exist: " + str(corpus_dir))
    loaded = load_trainable(corpus_dir)
    if not loaded["train"]:
        raise ValueError("no training records found in " + str(corpus_dir))
    if not loaded["validation"]:
        raise ValueError("no validation records found in " + str(corpus_dir))

    eval_sample = select_eval_sample(loaded["validation"], max_eval_records)
    if not eval_sample:
        raise ValueError("held-out evaluation sample is empty")

    state_root = Path(state_dir).expanduser()
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    ctx = TrainingContext(state_root)
    model = TemporalMirrorLM.load(ctx)
    model_path = TemporalMirrorLM.path(ctx)
    initial_present = model_path.exists()
    initial_digest = _sha256_file(model_path) if initial_present else None

    if progress is not None:
        progress({"phase": "loaded", "files": len(loaded["files"]),
                  "train": len(loaded["train"]),
                  "validation": len(loaded["validation"]),
                  "eval_sample": len(eval_sample)})

    default_epochs = epochs if epochs is not None else DEFAULT_EPOCHS
    if not 1 <= int(default_epochs) <= MAX_EPOCHS:
        raise ValueError("epochs must be 1..%d" % MAX_EPOCHS)

    plan_result = {
        "status": "disabled",
        "plan": _fallback_plan(default_epochs, "Gemma plan not requested"),
        "error": None,
        "samples": 0,
        "sample_ids_digest": None,
        "provider": None,
        "model": None,
    }
    if gemma_plan:
        plan_result = request_plan(
            gemma_provider, loaded["train"], default_epochs=default_epochs,
            max_epochs=MAX_EPOCHS, max_samples=gemma_max_samples,
            timeout=gemma_timeout)
        plan_result["provider"] = "ollama-openai-compatible"
        plan_result["model"] = gemma_model
        if progress is not None:
            progress({"phase": "gemma_plan",
                      "status": plan_result["status"],
                      "epochs": plan_result["plan"].get("epochs"),
                      "error": plan_result.get("error")})

    if epochs is not None:
        effective_epochs = int(epochs)
        epochs_source = "cli"
    elif gemma_plan:
        effective_epochs = int(plan_result["plan"].get("epochs", DEFAULT_EPOCHS))
        epochs_source = ("gemma-plan" if plan_result["status"] == "applied"
                         else "gemma-plan-fallback")
    else:
        effective_epochs = DEFAULT_EPOCHS
        epochs_source = "default"
    if not 1 <= effective_epochs <= MAX_EPOCHS:
        raise ValueError("epochs must be 1..%d" % MAX_EPOCHS)

    applied_plan = (plan_result["plan"]
                    if plan_result["status"] == "applied" else None)
    train_records = ordered_train_records(loaded["train"], applied_plan)
    if not train_records:
        raise ValueError("training plan selected zero records")

    probe_result = {
        "status": "disabled",
        "probes": [],
        "rejected": 0,
        "rejected_reasons": {},
        "error": None,
        "samples": 0,
        "sample_ids_digest": None,
        "provider": None,
        "model": None,
    }
    if gemma_probes:
        probe_result = request_probes(
            gemma_provider, loaded["train"], limit=MAX_PROBES,
            max_samples=gemma_max_samples, timeout=gemma_timeout)
        probe_result["provider"] = "ollama-openai-compatible"
        probe_result["model"] = gemma_model
        if progress is not None:
            progress({"phase": "gemma_probes",
                      "status": probe_result["status"],
                      "probes": len(probe_result["probes"]),
                      "error": probe_result.get("error")})
    probes = probe_result["probes"] if gemma_probes else None

    def save_epoch(epoch):
        if checkpoint_fn is not None:
            return checkpoint_fn(epoch)
        model.save(ctx)
        return checkpoint_info(ctx, model, epoch)

    training = train_epochs(
        model, train_records, eval_sample, effective_epochs, save_epoch,
        probes=probes, progress=progress)

    observed_total = sum(row["observed"] for row in training["epochs_run"])
    rejected_total = sum(row["rejected"] for row in training["epochs_run"])
    final_checkpoint = (training["epochs_run"][-1]["checkpoint"]
                        if training["epochs_run"] else {})
    record = {
        "schema": REPORT_SCHEMA,
        "generated_at": _iso_now(),
        "trainable_dir": str(corpus_dir),
        "corpus": {
            "files": len(loaded["files"]),
            "file_list": loaded["files"],
            "per_file": loaded["per_file"],
            "totals": loaded["totals"],
            "stages": loaded["stages"],
            "digests": loaded["digests"],
        },
        "split": {
            "declared_validation": loaded["declared_validation"],
            "train_records": len(loaded["train"]),
            "validation_records": len(loaded["validation"]),
            "validation_fallback": loaded["validation_fallback"],
            "eval_sample_records": len(eval_sample),
            "eval_sample_digest": _digest_ids(eval_sample),
            "eval_sample_record_ids": [item["record_id"] for item in eval_sample],
        },
        "epochs": {
            "requested": effective_epochs,
            "source": epochs_source,
            "default": DEFAULT_EPOCHS,
            "allowed": [1, MAX_EPOCHS],
        },
        "training_set": {
            "available": len(loaded["train"]),
            "selected": len(train_records),
            "focus_stages": (applied_plan or {}).get("focus_stages", []),
            "max_records_per_epoch": (applied_plan or {}).get(
                "max_records_per_epoch", 0),
        },
        "state": {
            "state_dir": str(state_root),
            "model_path": str(model_path),
            "model_schema": TEMPORAL_SCHEMA,
            "register_path": str(state_root / "language" / "temporal_register.json"),
            "chart_checkpoint_dir": str(
                state_root / "language" / "chart-checkpoints"),
            "initial_model_present": initial_present,
            "initial_model_sha256": initial_digest,
        },
        "counts": {
            "files": len(loaded["files"]),
            "records_parsed": loaded["totals"]["records"],
            "accepted": loaded["totals"]["accepted"],
            "train": len(loaded["train"]),
            "validation": len(loaded["validation"]),
            "eval_sample": len(eval_sample),
            "duplicates": loaded["totals"]["duplicates"],
            "rejected_records": loaded["totals"]["rejected"],
            "malformed_lines": loaded["totals"]["malformed"],
            "training_observed": observed_total,
            "training_rejected": rejected_total,
            "probes": len(probes) if probes is not None else 0,
        },
        "gemma": {
            "enabled": bool(gemma_plan or gemma_probes),
            "endpoint": gemma_url if (gemma_plan or gemma_probes) else None,
            "model": gemma_model if (gemma_plan or gemma_probes) else None,
            "role": "teacher/evaluator only; never writes student replies",
            "heldout_text_sent": False,
            "plan": plan_result,
            "probes": probe_result,
        },
        "before": training["before"],
        "epochs_run": training["epochs_run"],
        "after": training["after"],
        "checkpoint_final": final_checkpoint,
        "boundaries": {
            "derived_local_state": True,
            "canonical_merge": False,
            "canonical_commit": False,
            "proof_claim": False,
            "nannet_route": False,
            "heldout_metric": "next-token NLL/perplexity only",
            "gemma_role": "teacher/evaluator only; no student replies",
            "teacher_prompts_train_only": True,
        },
    }
    record["state"].update({
        "final_model_present": final_checkpoint.get("model_present"),
        "final_model_bytes": final_checkpoint.get("model_bytes"),
        "final_model_sha256": final_checkpoint.get("model_sha256"),
        "final_register_sha256": final_checkpoint.get("register_sha256"),
        "final_chart_checkpoint_generation": final_checkpoint.get(
            "chart_checkpoint_generation"),
        "final_chart_checkpoint_path": final_checkpoint.get(
            "chart_checkpoint_path"),
        "final_chart_checkpoint_sha256": final_checkpoint.get(
            "chart_checkpoint_sha256"),
    })
    write_reports(record, output)
    if progress is not None:
        progress({"phase": "report", "json": str(output),
                  "markdown": str(Path(output).with_suffix(".md")),
                  "before_nll": training["before"]["heldout"][
                      "mean_negative_log_likelihood"],
                  "final_nll": training["after"]["heldout"][
                      "mean_negative_log_likelihood"],
                  "final_perplexity": training["after"]["heldout"]["perplexity"]})
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainable-dir", default=str(default_trainable_dir()))
    parser.add_argument("--epochs", type=int, default=None,
                        help="1..20; default 5 (or a valid Gemma plan)")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-eval-records", type=int, default=200,
                        help="fixed held-out sample size; 0 means all")
    parser.add_argument("--gemma-plan", action="store_true",
                        help="ask Gemma for a bounded training plan")
    parser.add_argument("--gemma-probes", action="store_true",
                        help="ask Gemma for evaluation probes from TRAIN samples")
    parser.add_argument("--gemma-url", default=DEFAULT_GEMMA_URL)
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--gemma-max-samples", type=int, default=MAX_TEACHER_SAMPLES)
    parser.add_argument("--gemma-timeout", type=float, default=120.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.epochs is not None and not 1 <= args.epochs <= MAX_EPOCHS:
        parser.error("--epochs must be 1..%d" % MAX_EPOCHS)
    if args.max_eval_records < 0:
        parser.error("--max-eval-records must be >= 0")
    if args.gemma_max_samples < 0:
        parser.error("--gemma-max-samples must be >= 0")

    provider = None
    if args.gemma_plan or args.gemma_probes:
        provider = OllamaOpenAIProvider(
            args.gemma_url, args.gemma_model, timeout=args.gemma_timeout)

    def progress(event):
        if not args.quiet:
            print(json.dumps(event, sort_keys=True, default=str), flush=True)

    try:
        record = run_deep_training(
            args.trainable_dir, args.state_dir, args.output,
            epochs=args.epochs,
            max_eval_records=args.max_eval_records,
            gemma_provider=provider,
            gemma_plan=args.gemma_plan,
            gemma_probes=args.gemma_probes,
            gemma_url=args.gemma_url,
            gemma_model=args.gemma_model,
            gemma_max_samples=args.gemma_max_samples,
            gemma_timeout=args.gemma_timeout,
            progress=progress,
        )
    except ValueError as exc:
        print("deep_train_corpus: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(Path(args.output).with_suffix(".md")),
        "epochs": record["epochs"]["requested"],
        "before_nll": record["before"]["heldout"]["mean_negative_log_likelihood"],
        "final_nll": record["after"]["heldout"]["mean_negative_log_likelihood"],
        "final_perplexity": record["after"]["heldout"]["perplexity"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
