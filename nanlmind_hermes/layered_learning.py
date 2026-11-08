"""Layered L0-L9 learning over the paged scratch register space.

The coordinator is the HERMES-side learning loop:

``observe -> L0-L9 projections -> 255-head ensemble -> paged scratch leaves
-> relation/concept memory -> conservative promotion into the local 8 MiB
head register``.

It is deliberately not a second native trainer and does not promote canonical
NaNLMind state.  The local `HeadScratchRegister` and the paged scratch space are
experimental/derived stores.  Canonical BS0/MC0 promotion remains behind the
existing merge gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

from .concept_matrix import ConceptMatrix, content_words
from .dictionary_universe import DictionaryUniverse
from .mth_layers import MTHHeadMapper, MTHLayerStack, ROOT_NAMES
from .scratch_register import (
    LAYER_NAMES,
    LOGICAL_BYTES,
    ScratchRegisterSpace,
    digest,
    layer_index,
)
from .universe_chart import Chart8MiB, HeadScratchRegister, LocalRunner

LEARNING_SCHEMA = "nanlm.layered-learning/1"
OBSERVATION_SCHEMA = "nanlm.layered-learning-observation/1"
POLICY_SCHEMA = "nanlm.layered-learning-policy/1"
SYNC_SCHEMA = "nanlm.layered-learning-sync/1"
HEAD_COUNT = 255
HEAD_IDS = tuple(range(1, 256))
MAX_TEXT_BYTES = 1_048_576
MAX_RELATIONS = 512
MAX_ROWS = 4096

_CAUSAL = frozenset({
    "because", "therefore", "thus", "so", "cause", "causes", "caused",
    "before", "after", "then", "while", "when", "until", "since",
})
_VOWELS = frozenset("aeiou")


class LayeredLearningError(ValueError):
    pass


def _clip_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LayeredLearningError("text must be a non-empty string")
    raw = value.encode("utf-8")
    if len(raw) > MAX_TEXT_BYTES:
        raise LayeredLearningError("text exceeds 1 MiB")
    return value


def _source(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise LayeredLearningError("source must be a bounded non-empty string")
    return value


def _tick(value: Any) -> int:
    if value is None:
        return int(time.time())
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= (1 << 63) - 1:
        raise LayeredLearningError("tick must be a non-negative integer")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _hash_int(label: str, digest_size: int = 8) -> int:
    return int.from_bytes(
        hashlib.blake2b(label.encode("utf-8"), digest_size=digest_size).digest(),
        "little",
    )


class HeadEnsemble:
    """Deterministic superposition over all 255 HW0 heads (1..255)."""

    def __init__(self):
        self.weights: dict[int, float] = {}
        self.layer_weights: dict[str, dict[int, float]] = {}
        self.observations = 0

    def distribution(self, layer: str, concepts: Iterable[str], *, top: int = 32) -> dict[int, float]:
        layer = str(layer)
        if layer not in LAYER_NAMES:
            raise LayeredLearningError("unknown L0-L9 layer")
        seeds = [str(item).casefold() for item in concepts] or [""]
        scores: dict[int, float] = {}
        for head in HEAD_IDS:
            score = 0.0
            for concept in seeds:
                raw = _hash_int(f"{layer}|{concept}|{head}") % 4093
                score += 1.0 / (1 + raw % 97)
            scores[head] = score
        total = sum(scores.values())
        if total <= 0:
            raise LayeredLearningError("empty head distribution")
        normalized = {head: value / total for head, value in scores.items()}
        ordered = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
        return dict(ordered[:max(1, min(top, HEAD_COUNT))])

    def observe(self, layer: str, concepts: Iterable[str], *, top: int = 32) -> dict[int, float]:
        distribution = self.distribution(layer, concepts, top=top)
        self.observations += 1
        bucket = self.layer_weights.setdefault(layer, {})
        for head, weight in distribution.items():
            bucket[head] = bucket.get(head, 0.0) + weight
            self.weights[head] = self.weights.get(head, 0.0) + weight
        return distribution

    def top(self, limit: int = 16) -> list[dict[str, Any]]:
        ordered = sorted(self.weights.items(), key=lambda item: (-item[1], item[0]))
        return [{"head": head, "weight": round(weight, 6)}
                for head, weight in ordered[:max(1, min(limit, HEAD_COUNT))]]

    def balance(self) -> dict[str, Any]:
        active = [value for value in self.weights.values() if value > 0]
        if not active:
            return {"active_heads": 0, "coverage": 0.0, "max_share": 0.0,
                    "min_weight": 0.0, "mean_weight": 0.0}
        total = sum(active)
        return {
            "active_heads": len(active),
            "coverage": round(len(active) / HEAD_COUNT, 6),
            "max_share": round(max(active) / total, 6),
            "min_weight": round(min(active), 6),
            "mean_weight": round(total / len(active), 6),
            "observations": self.observations,
        }

    def decorrelate(self, strength: float = 0.05) -> dict[str, Any]:
        if not 0 <= strength <= 0.5:
            raise LayeredLearningError("strength must be in 0..0.5")
        if not self.weights:
            return self.balance()
        mean = sum(self.weights.values()) / len(self.weights)
        for head in list(self.weights):
            self.weights[head] = (1 - strength) * self.weights[head] + strength * mean
        return self.balance()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "nanlm.head-ensemble/1",
            "heads": HEAD_COUNT,
            "observations": self.observations,
            "weights": {str(head): round(value, 8)
                        for head, value in sorted(self.weights.items())},
            "layer_weights": {
                layer: {str(head): round(value, 8)
                        for head, value in sorted(bucket.items())}
                for layer, bucket in sorted(self.layer_weights.items())
            },
            "balance": self.balance(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "HeadEnsemble":
        if not isinstance(value, dict):
            raise LayeredLearningError("ensemble state must be an object")
        ensemble = cls()
        ensemble.observations = int(value.get("observations", 0))
        for head, weight in (value.get("weights") or {}).items():
            index = int(head)
            if 1 <= index <= 255:
                ensemble.weights[index] = float(weight)
        for layer, bucket in (value.get("layer_weights") or {}).items():
            if layer in LAYER_NAMES and isinstance(bucket, dict):
                ensemble.layer_weights[layer] = {
                    int(head): float(weight) for head, weight in bucket.items()
                    if 1 <= int(head) <= 255
                }
        return ensemble


@dataclass(frozen=True)
class LearningPolicy:
    min_support: int = 3
    min_confidence: float = 0.6
    max_candidates: int = 64
    max_promoted_leaves: int = 100
    require_explicit_approval: bool = True
    auto_sync: bool = False
    max_scratch_bytes: int = LOGICAL_BYTES

    def __post_init__(self) -> None:
        if not 1 <= self.min_support <= 64:
            raise LayeredLearningError("min_support must be in 1..64")
        if not 0 < self.min_confidence <= 1:
            raise LayeredLearningError("min_confidence must be in (0, 1]")
        if not 1 <= self.max_candidates <= 256:
            raise LayeredLearningError("max_candidates must be in 1..256")
        if not 1 <= self.max_promoted_leaves <= 100:
            raise LayeredLearningError("max_promoted_leaves must be in 1..100")
        if self.max_scratch_bytes != LOGICAL_BYTES:
            raise LayeredLearningError("the scratch space is fixed at 1 GiB")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "max_candidates": self.max_candidates,
            "max_promoted_leaves": self.max_promoted_leaves,
            "require_explicit_approval": self.require_explicit_approval,
            "auto_sync": self.auto_sync,
            "max_scratch_bytes": self.max_scratch_bytes,
        }


def _relation(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LayeredLearningError("relation must be an object")
    subject = str(value.get("subject") or "").strip().casefold()
    predicate = str(value.get("predicate") or value.get("role") or "").strip().casefold()
    obj = str(value.get("object") or "").strip().casefold()
    if not subject or not predicate or not obj:
        return None
    if any(len(part) > 256 for part in (subject, predicate, obj)):
        raise LayeredLearningError("relation part exceeds 256 characters")
    return {"subject": subject, "predicate": predicate, "object": obj}


class LayeredLearningCoordinator:
    def __init__(
        self,
        state_dir: str | Path,
        *,
        chart: Chart8MiB | None = None,
        dictionary: DictionaryUniverse | None = None,
        policy: LearningPolicy | None = None,
        now_tick=None,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.chart = chart
        self.dictionary = dictionary
        self.policy = policy or LearningPolicy()
        self.space = ScratchRegisterSpace(self.state_dir / "scratch_register")
        self.stack = MTHLayerStack()
        self.mapper = MTHHeadMapper(self.stack)
        self.ensemble = HeadEnsemble()
        self.concepts = ConceptMatrix()
        self.head_register = HeadScratchRegister(chart, self.state_dir)
        self._now_tick = now_tick or (lambda: int(time.time()))
        self.rows: list[dict[str, Any]] = []
        self.promoted: list[dict[str, Any]] = []
        self.turns = 0
        self._load()

    # -- persistence -----------------------------------------------------
    def _state_path(self) -> Path:
        return self.state_dir / "language" / "layered_learning.json"

    def _load(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        self.turns = int(data.get("turns", 0))
        self.rows = [row for row in data.get("rows", []) if isinstance(row, dict)][-MAX_ROWS:]
        self.promoted = [row for row in data.get("promoted", []) if isinstance(row, dict)]
        self.ensemble = HeadEnsemble.from_dict(data.get("ensemble", {}))
        for row in self.rows:
            relation = _relation(row)
            if relation:
                self.concepts.observe(
                    " ".join((relation["subject"], relation["predicate"], relation["object"])),
                    relation,
                )

    def _save(self) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema": LEARNING_SCHEMA,
            "turns": self.turns,
            "rows": self.rows[-MAX_ROWS:],
            "promoted": self.promoted[-self.policy.max_promoted_leaves:],
            "ensemble": self.ensemble.to_dict(),
            "policy": self.policy.to_dict(),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, path)

    # -- layer projections ----------------------------------------------
    def _layer_features(self, layer: str, text: str, words: list[str],
                        relation: dict[str, str] | None) -> dict[str, Any]:
        if layer in ("bit", "nibble", "byte"):
            return {"signature": self.stack.signature_of(text)}
        if layer == "phonetic":
            letters = [char for char in text.casefold() if char.isalpha()]
            vowels = sum(1 for char in letters if char in _VOWELS)
            first = words[0] if words else ""
            return {"vowels": vowels, "consonants": len(letters) - vowels,
                    "onset": first[:2], "rime": first[-2:] if first else ""}
        if layer == "morpheme":
            first = words[0] if words else ""
            return {"stem": first[:max(2, len(first) - 2)] if first else "",
                    "prefix": first[:3], "suffix": first[-3:]}
        if layer == "lexical":
            return {"words": words[:64], "count": len(words)}
        if layer == "word_completion":
            return {"prefixes": sorted({word[:cut] for word in words for cut in (2, 3, 4)
                                        if len(word) >= cut})[:64]}
        if layer == "sentence":
            return {"words": words[:96], "relation": relation}
        if layer == "paragraph":
            sentences = [item.strip() for item in re.split(r"[.!?]+", text) if item.strip()]
            return {"sentences": len(sentences), "unique_words": len(set(words)),
                    "transitions": sorted({word for word in words
                                           if word in {"however", "therefore", "then",
                                                      "meanwhile", "finally", "also"}})}
        if layer == "narrative":
            events = [word for word in words if word in _CAUSAL]
            return {"causal_markers": events[:32],
                    "has_relation": relation is not None,
                    "relation": relation}
        raise LayeredLearningError("unknown layer")

    # -- observe ---------------------------------------------------------
    def observe(
        self,
        text: str,
        *,
        source: str = "chat",
        relations: Iterable[dict[str, Any]] | None = None,
        tick: int | None = None,
    ) -> dict[str, Any]:
        text = _clip_text(text)
        source = _source(source)
        tick = _tick(tick)
        words = content_words(text)
        relation_list = []
        for value in list(relations or [])[:MAX_RELATIONS]:
            relation = _relation(value)
            if relation:
                relation_list.append(relation)
        self.turns += 1
        layer_results = []
        for layer in LAYER_NAMES:
            features = self._layer_features(layer, text, words, relation_list[0] if relation_list else None)
            distribution = self.ensemble.observe(layer, words or [text[:64]])
            leaf = self.space.append(
                layer=layer,
                head=max(distribution, key=distribution.get),
                leaf_id=f"turn:{self.turns}:{layer}",
                payload={
                    "text_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "features": features,
                    "head_top": [{"head": head, "weight": round(weight, 8)}
                                 for head, weight in distribution.items()],
                    "relation": relation_list[0] if relation_list else None,
                    "source": source,
                    "nanclock_tick": tick,
                },
                kind="layer",
            )
            layer_results.append({
                "layer": layer,
                "layer_index": layer_index(layer),
                "features": features,
                "head_top": list(distribution.items())[:8],
                "scratch": {
                    "address": leaf["address"],
                    "page": leaf["page"],
                    "offset": leaf["offset"],
                    "leaf_id": leaf["leaf_id"],
                },
            })
        self.concepts.observe(text)
        row_records = []
        for relation in relation_list:
            self.concepts.observe(
                " ".join((relation["subject"], relation["predicate"], relation["object"])),
                relation,
            )
            mapped = self.mapper.map_relation({
                **relation,
                "kind": "relation",
                "provenance": {"resolution": self._resolution_for(relation)},
            })
            row = {
                **relation,
                "kind": "relation",
                "layer": mapped["layer"],
                "head": mapped["head"],
                "mirror": mapped["mirror"],
                "mask": mapped["mask"],
                "value": mapped["value"],
                "digest": mapped["digest"],
                "source": source,
                "nanclock_tick": tick,
                "support": 1,
                "confidence": 1.0,
            }
            self.rows.append(row)
            row_records.append(row)
        self.rows = self.rows[-MAX_ROWS:]
        self._save()
        return {
            "schema": OBSERVATION_SCHEMA,
            "turn": self.turns,
            "source": source,
            "nanclock_tick": tick,
            "text_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "words": words[:64],
            "layers": layer_results,
            "relations": row_records,
            "head_balance": self.ensemble.balance(),
            "concept_saturation": self.concepts.saturation(),
            "scratch": self.space.stats(),
            "promotion_requires_approval": self.policy.require_explicit_approval,
        }

    @staticmethod
    def _resolution_for(relation: dict[str, str]) -> str:
        predicate = relation["predicate"]
        if predicate in {"causes", "cause", "then", "before", "after"}:
            return "narrative"
        if predicate in {"is", "means", "defined_as", "kind_of", "type_of"}:
            return "lexical"
        return "sentence"

    # -- WordNet HW4 ingestion ------------------------------------------
    def ingest_word(self, word: str, *, depth: int = 2, source: str = "wordnet-hw4") -> dict[str, Any]:
        if self.dictionary is None:
            raise LayeredLearningError("HW4 dictionary universe is not loaded")
        if not isinstance(word, str) or not word.strip():
            raise LayeredLearningError("word is required")
        if not 1 <= depth <= 8:
            raise LayeredLearningError("depth must be in 1..8")
        query = self.dictionary.query(word)
        relations = []
        definitions = []
        for synset in query.get("synsets", []):
            offset = synset.get("offset")
            if offset is None:
                continue
            definition = self.dictionary.define(offset)
            lemma = (definition.get("lemmas") or [word])[0]
            definitions.append({
                "offset": offset,
                "address": definition["address"],
                "definition": definition.get("definition", ""),
                "pos": definition.get("pos"),
                "lemmas": definition.get("lemmas", []),
            })
            if definition.get("definition"):
                relations.append({
                    "subject": lemma, "predicate": "defined_as",
                    "object": definition["definition"][:256],
                })
            for hypernym in definition.get("hypernyms", []):
                target = hypernym.get("target_lemmas") or []
                if target:
                    relations.append({
                        "subject": lemma, "predicate": "kind_of", "object": target[0],
                    })
            for edge in definition.get("relations", []):
                target = edge.get("target_lemmas") or []
                if target:
                    relations.append({
                        "subject": lemma,
                        "predicate": edge.get("relation", "related_to"),
                        "object": target[0],
                    })
            for ancestor in definition.get("inherited", []):
                relations.append({
                    "subject": lemma, "predicate": "inherits", "object": str(ancestor),
                })
        observation = self.observe(
            " ".join([word, *[item["definition"] for item in definitions if item["definition"]]]),
            source=source,
            relations=relations,
        )
        observation["wordnet"] = {
            "word": word,
            "synsets": definitions,
            "relations": relations,
            "hw4_route": self.dictionary.interface()["route"],
            "depth": depth,
        }
        return observation

    # -- promotion -------------------------------------------------------
    def _candidate_rows(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in self.rows:
            key = (row["subject"], row["predicate"], row["object"])
            if key not in grouped:
                grouped[key] = {**row, "support": 0}
            grouped[key]["support"] += 1
        candidates = []
        for row in grouped.values():
            support = int(row["support"])
            if support < self.policy.min_support:
                continue
            confidence = min(1.0, support / (support + 1))
            if confidence < self.policy.min_confidence:
                continue
            candidates.append({**row, "support": support, "confidence": round(confidence, 6)})
        candidates.sort(key=lambda item: (-item["support"], -item["confidence"], item["subject"]))
        return candidates[:self.policy.max_candidates]

    def synchronize(self, *, approved: bool = False) -> dict[str, Any]:
        candidates = self._candidate_rows()
        if not candidates:
            return {
                "schema": SYNC_SCHEMA, "status": "no_candidates",
                "approved": approved, "candidates": 0,
                "policy": self.policy.to_dict(),
            }
        if self.policy.require_explicit_approval and not approved:
            return {
                "schema": SYNC_SCHEMA, "status": "approval_required",
                "approved": False, "candidates": len(candidates),
                "policy": self.policy.to_dict(),
            }
        seen = {(row["subject"], row["predicate"], row["object"]): row
                for row in self.promoted}
        for row in candidates:
            key = (row["subject"], row["predicate"], row["object"])
            seen[key] = row
        promoted = list(seen.values())[-self.policy.max_promoted_leaves:]
        rows = [{
            **row,
            "kind": "relation",
            "provenance": {"resolution": self._resolution_for(row)},
        } for row in promoted]
        head_status = self.head_register.organize(rows, self.mapper)
        self.promoted = promoted
        self._save()
        report = {
            "schema": SYNC_SCHEMA,
            "status": "promoted",
            "approved": approved,
            "candidates": len(candidates),
            "promoted": len(promoted),
            "head_register": head_status,
            "scratch": self.space.stats(),
            "policy": self.policy.to_dict(),
            "nanclock_tick": self._now_tick(),
        }
        history = self.state_dir / "language" / "layered_learning_sync.jsonl"
        history.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(history, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return report

    # -- chat cycle ------------------------------------------------------
    def cycle(
        self,
        text: str,
        *,
        source: str = "chat",
        relations: Iterable[dict[str, Any]] | None = None,
        tick: int | None = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        observation = self.observe(text, source=source, relations=relations, tick=tick)
        sync = None
        if self.policy.auto_sync or approved:
            sync = self.synchronize(approved=approved)
        return {"observation": observation, "sync": sync}

    def ingest_relations(self, records: Iterable[dict[str, Any]], *,
                         source: str = "chat", tick: int | None = None) -> dict[str, Any]:
        rows = []
        for record in records:
            relation = _relation(record if isinstance(record, dict) else {})
            if relation:
                rows.append(relation)
        return self.observe(
            " ".join(" ".join((row["subject"], row["predicate"], row["object"]))
                     for row in rows) or "relation",
            source=source, relations=rows, tick=tick,
        )

    # -- status ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "schema": LEARNING_SCHEMA,
            "turns": self.turns,
            "rows": len(self.rows),
            "promoted": len(self.promoted),
            "policy": self.policy.to_dict(),
            "head_ensemble": self.ensemble.balance(),
            "head_top": self.ensemble.top(16),
            "scratch": self.space.stats(),
            "scratch_verify": self.space.verify(),
            "concept_saturation": self.concepts.saturation(),
            "head_register": self.head_register.status(),
            "chart_mapped": self.chart is not None,
            "dictionary_hw4": None if self.dictionary is None else self.dictionary.interface(),
        }

    def close(self) -> None:
        self._save()
