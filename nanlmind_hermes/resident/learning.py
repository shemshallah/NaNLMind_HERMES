"""Corpus learning coupled to the resident universe.

The learner has two channels that observe the same sentence:

* the existing :class:`TemporalMirrorLM` word predictor, which provides the
  usable next-token distribution; and
* the resident mirror-pair cells, which receive directed head-to-head token
  counts and can answer bounded resonance/association queries without a
  separate Python graph.

This is a real online learner, not a claim that the resident geometry alone is
already a language model.  The resident pair channel is deliberately measured
separately from the temporal channel so improvements cannot hide inside a
stronger baseline.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping

from ..temporal_lm import (
    BOS,
    EOS,
    END,
    PUNCT,
    STOP,
    TemporalMirrorLM,
    fnv_head,
    mirror8,
    tokens,
)
from .cells import address_of
from .ingest import DEFAULT_FEATURE_DIM, DEFAULT_FEATURE_DTYPE, prepare_records
from .universe import ResidentUniverse

LEARNING_SCHEMA = "nanlm.resident-learning/1"
DEFAULT_ALPHA = 0.35


def _bounded(value: Any, default: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(number, maximum))


def _nll(probability: float) -> float:
    return -math.log(max(float(probability), 1e-12))


def _perplexity(log_total: float, count: int) -> float | None:
    if not count:
        return None
    return round(math.exp(min(log_total / count, 50.0)), 6)


def _tokens_with_bos(text: str) -> list[str]:
    return [BOS] + [token for token in tokens(text) if token != BOS] + [EOS]


# Deliberately small, inspectable extraction.  It only turns explicit
# definition/hypergraph wording into typed evidence; it never asks a model to
# invent a triple and it never labels an arbitrary sentence as a fact.
_RELATION_PATTERNS = (
    (re.compile(r"^(?P<s>.+?)\s+means\s+(?P<o>.+?)(?:\.|$)", re.I),
     "means", "supports"),
    (re.compile(r"^(?P<s>.+?)\s+has relation\s+(?P<p>[A-Za-z_-]+)\s+to\s+"
                r"(?P<o>.+?)(?:\.|$)", re.I), None, "supports"),
    (re.compile(r"^the intended sense of\s+(?P<s>.+?)\s+is\s+"
                r"(?P<o>.+?)(?:\.|$)", re.I), "sense", "supports"),
    (re.compile(r"^(?P<s>.+?)\s+(?:is|are)\s+(?:a|an|the)\s+"
                r"(?P<o>.+?)(?:\.|$)", re.I), "is a", "supports"),
)


def extract_relations(text: str, *, stage: str | None = None) -> list[dict[str, Any]]:
    """Derive bounded typed relations from explicit definition prose."""
    value = str(text or "").strip()
    if len(value) < 8 or value.endswith("?"):
        return []
    # Restrict the broad is-a pattern to declared definition/hypergraph stages so
    # literary prose is not silently converted into false facts.
    broad = str(stage or "").casefold() in {
        "05_word_concept", "06_hypergraph", "definition", "short_sentences",
    }
    for index, (pattern, predicate, polarity) in enumerate(_RELATION_PATTERNS):
        match = pattern.match(value[:1000])
        if not match:
            continue
        if index == 3 and not broad:
            continue
        subject = match.group("s").strip(" \t\"'")[:240]
        obj = match.group("o").strip(" \t\"'")[:512]
        if not subject or not obj:
            continue
        chosen = predicate or (match.groupdict().get("p") or "related")
        return [{
            "kind": "relation",
            "subject": subject,
            "predicate": chosen,
            "object": obj,
            "polarity": polarity,
            "confidence": 0.55,
            "source": "corpus-definition-parser",
            "provenance": {
                "source_type": "corpus_definition_parser",
                "stage": stage,
                "independently_fact_verified": False,
            },
        }]
    return []


class ResidentLearner:
    """Online corpus learner over one resident universe."""

    def __init__(self, universe: ResidentUniverse,
                 temporal: TemporalMirrorLM | None = None,
                 *, alpha: float = DEFAULT_ALPHA,
                 extract_relations: bool = False):
        self.universe = universe
        self.temporal = temporal or TemporalMirrorLM()
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.extract_relations = bool(extract_relations)
        self._outgoing = [0.0] * 256
        self._rescan_outgoing()

    # ------------------------------------------------------------- resident
    def _rescan_outgoing(self) -> None:
        self._outgoing = [0.0] * 256
        for left in range(256):
            total = 0.0
            for right in range(256):
                if left == right:
                    continue
                state = self.universe.cells.pair_state(left, right)
                weight = (state.forward.weight
                          if address_of(left, right) == state.canonical
                          else state.reverse.weight)
                if weight > 0:
                    total += weight / (1 << 16)
            self._outgoing[left] = total

    def _pair_weight(self, left: int, right: int) -> float:
        if left == right:
            return 0.0
        state = self.universe.cells.pair_state(left, right)
        value = (state.forward.weight
                 if address_of(left, right) == state.canonical
                 else state.reverse.weight)
        return max(0.0, value / (1 << 16))

    def resident_probability(self, context: str, word: str) -> float:
        """Resident bigram probability from the 256-head pair cells."""
        left, right = fnv_head(context), fnv_head(word)
        if left == right:
            return 1e-6
        weight = self._pair_weight(left, right)
        total = self._outgoing[left]
        # Add-k smoothing keeps every head reachable while the pair table is
        # sparse.  This is intentionally small; it is a fallback, not a prior
        # large enough to drown the observed counts.
        return (weight + 0.01) / (total + 0.01 * 256.0)

    def temporal_probability(self, context2: str | None,
                             context1: str, word: str) -> float:
        return self.temporal._prob(context2, context1, word)

    def hybrid_probability(self, context2: str | None,
                           context1: str, word: str) -> float:
        temporal = self.temporal_probability(context2, context1, word)
        resident = self.resident_probability(context1, word)
        return self.alpha * temporal + (1.0 - self.alpha) * resident

    # -------------------------------------------------------------- exposure
    def _learn_pairs(self, text: str) -> None:
        sequence = _tokens_with_bos(text)
        for index in range(1, len(sequence)):
            target = sequence[index]
            previous = sequence[index - 1]
            if target == BOS:
                continue
            left, right = fnv_head(previous), fnv_head(target)
            try:
                self.universe.cells.observe_pair(left, right, weight=1.0)
                self._outgoing[left] += 1.0
            except (ValueError, MemoryError):
                continue

    def observe(self, text: str, *, source: str | None = None,
                head: int | None = None,
                ingest: bool = True) -> dict[str, Any]:
        """Learn one sentence through the temporal and resident channels."""
        value = str(text or "").strip()
        if len(value) < 2:
            return {"learned": False, "reason": "short"}
        learned = self.temporal.observe(value)
        if not learned:
            return {"learned": False, "reason": "control_or_duplicate"}
        self._learn_pairs(value)
        receipt = None
        if ingest:
            prepared = prepare_records([{
                "text": value, "kind": "observation",
                "source": source, "head": head,
            }], feature_dim=DEFAULT_FEATURE_DIM,
                feature_dtype=DEFAULT_FEATURE_DTYPE)
            receipt = self.universe.ingest_records(prepared)
        return {"learned": True, "receipt": receipt}

    def observe_batch(self, records: Iterable[Mapping[str, Any]],
                      *, source: str | None = None,
                      batch_size: int = 128) -> dict[str, Any]:
        """Learn a bounded batch; one resident flush per batch."""
        size = _bounded(batch_size, 128, 1024)
        learned = skipped = 0
        rows: list[dict[str, Any]] = []
        for record in records:
            text = record.get("text")
            if not isinstance(text, str) or len(text.strip()) < 2:
                skipped += 1
                continue
            if not self.temporal.observe(text):
                skipped += 1
                continue
            self._learn_pairs(text)
            rows.append({
                "text": text,
                "kind": record.get("kind") or "observation",
                "source": record.get("source") or source,
                "stage": record.get("stage"),
                "head": record.get("head"),
            })
            learned += 1
            if len(rows) >= size:
                self._flush_rows(rows)
                rows = []
        if rows:
            self._flush_rows(rows)
        return {
            "learned": learned,
            "skipped": skipped,
            "generation": self.universe.generation,
        }

    def _flush_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        prepared_rows: list[dict[str, Any]] = list(rows)
        if self.extract_relations:
            for row in rows:
                for relation in extract_relations(
                        row.get("text", ""), stage=row.get("stage")):
                    relation["source"] = row.get("source")
                    prepared_rows.append(relation)
        prepared = prepare_records(
            prepared_rows, feature_dim=DEFAULT_FEATURE_DIM,
            feature_dtype=DEFAULT_FEATURE_DTYPE)
        self.universe.ingest_records(prepared)
        # Pair updates are direct cell writes; include the batch in the state
        # chain so a checkpoint describes the whole observation rather than
        # only its leaf records.
        self.universe._commit(
            "pairbatch",
            hashlib.sha256(json.dumps(
                [row["text"] for row in rows], sort_keys=True,
                ensure_ascii=False).encode()).hexdigest())
        self.universe.flush()

    # ------------------------------------------------------------- evaluation
    def evaluate(self, records: Iterable[Mapping[str, Any]], *,
                 limit: int | None = None) -> dict[str, Any]:
        rows = list(records)
        if limit is not None:
            rows = rows[:_bounded(limit, 512, 100000)]
        groups: dict[str, list[float]] = {
            "temporal": [0.0, 0], "resident": [0.0, 0],
            "hybrid": [0.0, 0],
        }
        by_source: dict[str, dict[str, list[float]]] = {}
        for record in rows:
            text = str(record.get("text") or "")
            if len(text) < 2:
                continue
            source = str(record.get("source") or "unknown")
            bucket = by_source.setdefault(source, {
                "temporal": [0.0, 0], "resident": [0.0, 0],
                "hybrid": [0.0, 0],
            })
            sequence = _tokens_with_bos(text)
            for index in range(1, len(sequence)):
                target = sequence[index]
                if target == BOS:
                    continue
                previous = sequence[index - 1]
                context2 = sequence[index - 2] if index >= 2 else None
                for name, probability in (
                    ("temporal", self.temporal_probability(
                        context2, previous, target)),
                    ("resident", self.resident_probability(previous, target)),
                    ("hybrid", self.hybrid_probability(
                        context2, previous, target)),
                ):
                    value = _nll(probability)
                    groups[name][0] += value
                    groups[name][1] += 1
                    bucket[name][0] += value
                    bucket[name][1] += 1
        result = {
            "records": len(rows),
            "channels": {
                name: {
                    "tokens": int(values[1]),
                    "mean_negative_log_likelihood": round(
                        values[0] / values[1], 6) if values[1] else None,
                    "perplexity": _perplexity(values[0], int(values[1])),
                }
                for name, values in groups.items()
            },
            "by_source": {
                source: {
                    channel: {
                        "tokens": int(values[1]),
                        "mean_negative_log_likelihood": round(
                            values[0] / values[1], 6) if values[1] else None,
                        "perplexity": _perplexity(values[0], int(values[1])),
                    }
                    for channel, values in channels.items()
                }
                for source, channels in sorted(by_source.items())
            },
        }
        return result

    # -------------------------------------------------------------- prediction
    def next_words(self, prompt: str, *, top_k: int = 8) -> list[dict[str, Any]]:
        """Rank vocabulary words by resident pair resonance and blend score."""
        k = _bounded(top_k, 8, 64)
        context = (tokens(prompt) or [BOS])[-1]
        scored = []
        for word in self.temporal.vocab:
            if word in {BOS, EOS} or word in PUNCT or word in END:
                continue
            resident = self.resident_probability(context, word)
            temporal = self.temporal_probability(
                None, context, word)
            scored.append((self.alpha * temporal + (1.0 - self.alpha) * resident,
                           word, resident, temporal))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{
            "word": word,
            "score": round(score, 8),
            "resident_probability": round(resident, 8),
            "temporal_probability": round(temporal, 8),
        } for score, word, resident, temporal in scored[:k]]

    # ------------------------------------------------------------- persistence
    def save_temporal(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        model = self.temporal
        payload = {
            "schema": "nanlm.temporal-mirror-lm/1",
            "vocab": model.vocab,
            "uni": model.uni,
            "bi": model.bi,
            "tri": model.tri,
            "pairs": model.pairs,
            "subject": model.subject,
            "seen": model.seen[-65536:],
            "recent": model.recent[-8:],
            "rejected": model.rejected[-32:],
            "last_query": model.last_query,
            "last_reply": model.last_reply,
            "output_drive": model.output_drive,
        }
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8")
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        temp.replace(target)
        return target

    def load_temporal(self, path: str | Path) -> bool:
        source = Path(path).expanduser()
        if not source.exists():
            return False
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict) or data.get("schema") != \
                "nanlm.temporal-mirror-lm/1":
            return False
        model = TemporalMirrorLM()
        model.vocab = {str(k): int(v) & 0xFF
                       for k, v in (data.get("vocab") or {}).items()}
        model.uni = {str(k): int(v) for k, v in (data.get("uni") or {}).items()}
        model.bi = {str(k): {str(w): int(n) for w, n in (v or {}).items()}
                    for k, v in (data.get("bi") or {}).items()}
        model.tri = {str(k): {str(w): int(n) for w, n in (v or {}).items()}
                     for k, v in (data.get("tri") or {}).items()}
        model.pairs = {str(k): {str(w): float(n) for w, n in (v or {}).items()}
                       for k, v in (data.get("pairs") or {}).items()}
        model.subject = {str(k): int(v)
                         for k, v in (data.get("subject") or {}).items()}
        model.seen = [str(v) for v in (data.get("seen") or [])]
        model.recent = [str(v) for v in (data.get("recent") or [])]
        model.rejected = [str(v) for v in (data.get("rejected") or [])]
        model.last_query = str(data.get("last_query") or "")
        model.last_reply = str(data.get("last_reply") or "")
        model.output_drive = float(data.get("output_drive", 1.0) or 1.0)
        self.temporal = model
        self._rescan_outgoing()
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "schema": LEARNING_SCHEMA,
            "alpha": self.alpha,
            "generation": self.universe.generation,
            "digest": self.universe.digest,
            "temporal": {
                "vocab": len(self.temporal.vocab),
                "unigrams": len(self.temporal.uni),
                "bigram_contexts": len(self.temporal.bi),
                "trigram_contexts": len(self.temporal.tri),
                "seen": len(self.temporal.seen),
            },
            "resident": {
                "pair_heads": sum(1 for value in self._outgoing if value > 0),
                "pairs": self.universe.cells.pair_count,
                "leaves": self.universe.leaves.count,
                "relations": len(self.universe.relations),
                "features": len(self.universe.features),
            },
        }


__all__ = ("LEARNING_SCHEMA", "DEFAULT_ALPHA", "ResidentLearner")
