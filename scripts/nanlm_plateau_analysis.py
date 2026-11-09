#!/usr/bin/env python3
"""Independent plateau diagnostic for the native NaNLM student.

This script is read-only with respect to the package and the training state. It
reads a ``nanlm.gemma-learning-over-time/1`` run record, inspects the
``mathematical_memory.sqlite3`` ledger in SQLite read-only mode, and explains
per-probe misses as either code-path gaps or curriculum gaps.

It deliberately does not retrain, promote relations, or modify the state dir.
The ablation plan is data only; no ablation is executed by this harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RUN_JSON = ROOT / "docs" / "NANLM_GEMMA_LEARNING_OVER_TIME.json"
DEFAULT_STATE_DIR = "/data/data/com.termux/files/usr/tmp/nanlm-train-direct"

REPORT_SCHEMA = "nanlm.plateau-diagnostic/1"
MAX_SUMMARY_LINES = 64
MAX_SUMMARY_CHARS = 12000
MAX_TABLE_ROWS = 200

CAUSES = (
    "no_structured_intent",
    "wrong_subject_interference",
    "reverse_lookup_missing",
    "causal_question_missing",
    "vocabulary/generation",
    "teacher_curriculum_missing",
    "unknown",
)

CAUSE_DEFINITIONS = {
    "no_structured_intent": (
        "The question form is not implemented by the current structured answer "
        "path, so a relation row cannot be selected even when one exists."
    ),
    "wrong_subject_interference": (
        "The reply carries an answer object from a sibling subject (for example "
        "red+blue answered with yellow+blue's object) instead of the asked subject."
    ),
    "reverse_lookup_missing": (
        "The fact exists in the forward direction (subject -> object) but the "
        "question needs the inverse direction (object/property -> subject)."
    ),
    "causal_question_missing": (
        "The question asks for a result or cause and no causal answer frame is "
        "implemented or extracted for it."
    ),
    "vocabulary/generation": (
        "A usable structured candidate exists (or the answer is otherwise in "
        "exposure), but the temporal generator did not emit it."
    ),
    "teacher_curriculum_missing": (
        "Neither the cumulative teaching statements nor the relation ledger "
        "contain the expected answer for this probe at that stage."
    ),
    "unknown": (
        "The probe is malformed or there is not enough evidence to assign a "
        "specific cause."
    ),
}

# Question words that describe the requested operation rather than the subject.
QUESTION_NOISE = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "with", "about", "as", "at", "by", "from", "it",
    "its", "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
    "do", "does", "did", "can", "could", "would", "should", "what", "how",
    "where", "when", "why", "who", "whom", "whose", "which", "color", "colour",
    "shape", "happens", "happen", "make", "makes", "made", "name", "tell",
    "kind", "type", "sort", "has", "have", "had",
})

GENERIC_OBJECT_TOKENS = frozenset({
    "ability", "able", "animal", "animals", "color", "colour", "make", "makes",
    "created", "creates", "sound", "thing", "things", "shape", "shapes", "way",
})

CODE_LEVEL_CAUSES = frozenset({
    "no_structured_intent", "wrong_subject_interference",
    "reverse_lookup_missing", "causal_question_missing",
})

SUPPORTED_STRUCTURED_INTENTS = (
    "what is/are X?",
    "what causes X?",
    "is/are X a/an/the Y?",
    "what do/does X do?",
    "what do/does X make/create/produce?",
    "what/which (shape/object/thing) has/have Y?",
    "what happens when/if X?",
)


# ---------------------------------------------------------------------------
# Text helpers (dependency-free and deterministic)
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").casefold())


def _token_variants(word: str) -> set[str]:
    """Small deterministic stem set; enough for subject/object matching."""
    word = str(word or "").casefold().strip("'")
    if not word:
        return set()
    out = {word}
    if word.endswith("'s"):
        out.add(word[:-2])
    if len(word) > 3 and word.endswith("ies"):
        out.add(word[:-3] + "y")
    if len(word) > 4 and word.endswith("es"):
        out.add(word[:-2])
        out.add(word[:-1])
    if len(word) > 3 and word.endswith("s"):
        out.add(word[:-1])
    if len(word) > 4 and word.endswith("ing"):
        stem = word[:-3]
        out.add(stem)
        out.add(stem + "e")
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        out.add(stem)
        out.add(stem + "e")
    return {item for item in out if item}


def _stems(text: str, noise: Iterable[str] = ()) -> set[str]:
    stop = set(noise)
    out: set[str] = set()
    for word in _tokens(text):
        if word in stop:
            continue
        out.update(_token_variants(word))
    return out


def _query_terms(question: str) -> set[str]:
    return _stems(question, QUESTION_NOISE)


def _expected_terms(expected: Iterable[str]) -> set[str]:
    return _stems(" ".join(str(item) for item in expected))


def _overlap(a: set[str], b: set[str]) -> bool:
    return bool(a & b)


def _norm_text(text: str) -> str:
    return " ".join(_tokens(text))


def _contains_expected(text: str, expected: Iterable[str]) -> bool:
    folded = str(text or "").casefold()
    return any(str(word).casefold() in folded for word in expected if str(word))


def _rate(hits: int, total: int) -> float:
    return (hits / total) if total else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Run record loading
# ---------------------------------------------------------------------------

def load_run(path: str | Path, retries: int = 6) -> dict[str, Any]:
    """Load a run record, tolerating a live writer replacing the JSON file."""
    target = Path(path).expanduser()
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            raw = target.read_bytes()
            record = json.loads(raw.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("run record must be a JSON object")
            record["_analysis"] = {
                "path": str(target),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "mtime_ns": target.stat().st_mtime_ns,
            }
            return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.15 * (attempt + 1))
    raise SystemExit("could not read run JSON %s: %s" % (target, last_error))


def _probe_catalog(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("probes")
    if not isinstance(raw, list) or not raw:
        raw = []
        for round_record in record.get("rounds") or []:
            for probe in round_record.get("probes") or []:
                raw.append({"question": probe.get("question"),
                            "expected": probe.get("expected")})
        if not raw:
            baseline = record.get("baseline")
            if isinstance(baseline, dict):
                raw = [
                    {"question": probe.get("question"),
                     "expected": probe.get("expected")}
                    for probe in baseline.get("probes") or []
                ]
        # Preserve first occurrence order.
        seen: set[str] = set()
        unique = []
        for probe in raw:
            key = str(probe.get("question") or "")
            if key and key not in seen:
                seen.add(key)
                unique.append(probe)
        raw = unique
    catalog = []
    for probe in raw:
        if not isinstance(probe, dict):
            continue
        question = str(probe.get("question") or "").strip()
        expected = probe.get("expected")
        if isinstance(expected, str):
            expected = [expected]
        expected = [str(item) for item in (expected or []) if str(item).strip()]
        if question:
            catalog.append({"question": question, "expected": expected})
    return catalog


def _stage_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    baseline = record.get("baseline")
    if isinstance(baseline, dict):
        stages.append({
            "stage": "baseline",
            "label": "baseline",
            "round": 0,
            "probes": baseline.get("probes") or [],
            "statements": [],
            "hit_rate": baseline.get("hit_rate"),
        })
    for position, round_record in enumerate(record.get("rounds") or [], start=1):
        if not isinstance(round_record, dict):
            continue
        number = round_record.get("round", position)
        stages.append({
            "stage": "round %s" % number,
            "label": "round %s" % number,
            "round": number,
            "probes": round_record.get("probes") or [],
            "statements": [str(item) for item in (round_record.get("statements") or [])
                           if str(item).strip()],
            "hit_rate": round_record.get("hit_rate"),
        })
    return stages


# ---------------------------------------------------------------------------
# Memory ledger inspection (strictly read-only)
# ---------------------------------------------------------------------------

def _db_uri(path: Path) -> str:
    return "file:%s?mode=ro" % quote(str(path.resolve()), safe="/")


def _sidecar_paths(path: Path) -> set[str]:
    return {
        str(sidecar)
        for suffix in ("-journal", "-wal", "-shm")
        for sidecar in (path.with_name(path.name + suffix),)
        if sidecar.exists()
    }


def _relation_rows_from_db(connection: sqlite3.Connection,
                           tables: set[str]) -> tuple[list[dict], int, int]:
    if "events" not in tables:
        return [], 0, 0
    if "projections" in tables:
        sql = ("SELECT e.event_id,e.payload,p.head,p.layer "
               "FROM events e JOIN projections p ON p.event_id=e.event_id "
               "WHERE p.active=1 ORDER BY e.event_id")
    else:
        sql = "SELECT e.event_id,e.payload,NULL AS head,NULL AS layer FROM events e ORDER BY e.event_id"
    rows = connection.execute(sql).fetchall()
    relation_rows: list[dict] = []
    observations = 0
    invalid = 0
    seen: set[str] = set()
    for row in rows:
        event_id = str(row["event_id"])
        if event_id in seen:
            continue
        seen.add(event_id)
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            invalid += 1
            continue
        if not isinstance(payload, dict):
            invalid += 1
            continue
        kind = str(payload.get("kind") or "")
        if kind == "observation":
            observations += 1
        if kind != "relation":
            continue
        subject = str(payload.get("subject") or "").strip()
        predicate = str(payload.get("predicate") or "").strip()
        obj = str(payload.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
        relation_rows.append({
            "event_id": event_id,
            "kind": "relation",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "polarity": str(payload.get("polarity") or "neutral"),
            "confidence": payload.get("confidence"),
            "head": row["head"],
            "layer": row["layer"],
            "source": provenance.get("source_type") or provenance.get("source"),
            "source_sentence": evidence.get("sentence") if isinstance(evidence.get("sentence"), str) else None,
            "evidence": evidence,
        })
    return relation_rows, observations, invalid


def inspect_memory_db(path: str | Path | None) -> dict[str, Any]:
    """Inspect the memory DB through a read-only SQLite URI.

    The function never calls ``MemoryStore`` (which creates tables and WAL
    state). It records before/after digests so callers can prove the harness
    itself did not mutate the file.
    """
    if not path:
        return {
            "status": "not_configured", "path": None, "read_only": True,
            "relation_rows": [], "relation_rows_total": 0,
            "observations_total": 0, "graph_edges": 0, "assessments": 0,
        }
    target = Path(path).expanduser()
    if not target.is_file():
        return {
            "status": "missing", "path": str(target), "read_only": True,
            "relation_rows": [], "relation_rows_total": 0,
            "observations_total": 0, "graph_edges": 0, "assessments": 0,
        }
    before_hash = _sha256_file(target)
    before_stat = target.stat()
    before_sidecars = _sidecar_paths(target)
    connection = None
    try:
        connection = sqlite3.connect(_db_uri(target), uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        relation_rows, observations, invalid = _relation_rows_from_db(connection, tables)
        graph_edges = assessments = hyperedges = 0
        if "edges" in tables:
            graph_edges = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        if "assessments" in tables:
            assessments = int(connection.execute("SELECT COUNT(*) FROM assessments").fetchone()[0])
        if "hyperedges" in tables:
            hyperedges = int(connection.execute("SELECT COUNT(*) FROM hyperedges").fetchone()[0])
        meta = {}
        if "meta" in tables:
            meta = {str(row["key"]): str(row["value"])
                    for row in connection.execute("SELECT key,value FROM meta")}
        status = "ok" if "events" in tables else "no_events_table"
        result = {
            "status": status,
            "path": str(target),
            "read_only": True,
            "sqlite_uri_mode": "ro",
            "pragma_query_only": True,
            "schema": meta.get("schema"),
            "meta": meta,
            "tables": sorted(tables),
            "relation_rows_total": len(relation_rows),
            "relation_rows": relation_rows[:MAX_TABLE_ROWS],
            "observations_total": observations,
            "invalid_event_payloads": invalid,
            "graph_edges": graph_edges,
            "hyperedges": hyperedges,
            "assessments": assessments,
        }
    except sqlite3.Error as exc:
        result = {
            "status": "sqlite_error", "path": str(target), "read_only": True,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:500]),
            "relation_rows": [], "relation_rows_total": 0,
            "observations_total": 0, "graph_edges": 0, "assessments": 0,
        }
    finally:
        if connection is not None:
            connection.close()
    after_hash = _sha256_file(target)
    after_stat = target.stat()
    result.update({
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "unchanged": before_hash == after_hash,
        "size_before": before_stat.st_size,
        "size_after": after_stat.st_size,
        "sidecars_before": sorted(before_sidecars),
        "sidecars_after": sorted(_sidecar_paths(target)),
        "sidecars_created": sorted(_sidecar_paths(target) - before_sidecars),
    })
    return result


def annotate_relation_rounds(relation_rows: list[dict],
                             stages: list[dict[str, Any]]) -> None:
    """Attach the first stage at which a relation's source sentence was taught."""
    source_rounds: dict[str, int] = {}
    for stage in stages:
        for statement in stage.get("statements") or []:
            key = _norm_text(statement)
            if key and key not in source_rounds:
                source_rounds[key] = int(stage.get("round") or 0)
    for row in relation_rows:
        sentence = row.get("source_sentence")
        key = _norm_text(sentence) if sentence else ""
        row["first_available_round"] = source_rounds.get(key, 0)
        # Rows with an explicit source sentence that was never taught in this
        # run are pre-existing ledger state. Keep them for DB reporting, but do
        # not let later/external facts leak into an earlier run stage's
        # diagnosis.
        row["traceable_to_run"] = not sentence or key in source_rounds


def _rows_available_at(relation_rows: list[dict], round_number: int) -> list[dict]:
    return [row for row in relation_rows
            if row.get("traceable_to_run", True)
            and int(row.get("first_available_round") or 0) <= round_number]


# ---------------------------------------------------------------------------
# Question intent and structured answer path
# ---------------------------------------------------------------------------

def question_kind(question: str) -> str:
    text = str(question or "").strip().casefold()
    if not text:
        return "empty"
    if (re.match(r"^what\s+happens\s+(?:when|if)\b", text)
            or re.match(r"^what\s+(?:causes?|makes?)\b", text)
            or text.startswith("why ")):
        return "causal"
    if (re.match(r"^what\s+(?:shape|thing|object|animal|color|colour|number|material)\s+(?:has|have|with)\b", text)
            or re.match(r"^what\s+[a-z ]+\s+(?:has|have)\b", text)
            or re.match(r"^what\s+has\b", text)
            or re.match(r"^which\s+.*\bhas\b", text)
            or re.match(r"^what\s+(?:is|are)\s+(?:the\s+)?(?:thing|object|shape)\b.*\bwith\b", text)):
        return "reverse"
    if re.match(r"^what\s+(?:do|does)\s+.+\s+do\?$", text):
        return "what_do"
    if re.match(r"^what\s+(?:do|does)\s+.+\s+make\?$", text):
        return "what_make"
    if re.match(r"^what\s+colou?r\b", text):
        return "what_color"
    if re.match(r"^what\s+(?:is|are)\b", text):
        return "what_is"
    if re.match(r"^(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b", text):
        return "yes_no"
    return "other"


def _fallback_structured_candidates(query: str, rows: list[dict]) -> list[dict]:
    """Small mirror of the current public intent surface, used if import fails."""
    lowered = str(query or "").strip().casefold()
    edges = []
    for row in rows:
        if row.get("kind") not in (None, "relation"):
            continue
        subject = str(row.get("subject") or "").strip()
        predicate = str(row.get("predicate") or "").strip()
        obj = str(row.get("object") or "").strip()
        if subject and predicate and obj:
            edges.append({"subject": subject, "predicate": predicate, "object": obj,
                          "confidence": row.get("confidence") or 0.55})
    candidates = []

    def add(edge):
        text = "%s %s %s" % (edge["subject"], edge["predicate"], edge["object"])
        candidates.append({"text": text[:1].upper() + text[1:] + ".",
                           "truth": float(edge["confidence"]),
                           "subject": edge["subject"], "object": edge["object"],
                           "source": "learned_relation_fallback"})

    what_is = re.match(r"^what\s+(?:is|are)\s+(.+?)\s*\?$", lowered)
    if what_is:
        subject = " ".join(word for word in _tokens(what_is.group(1))
                           if word not in {"a", "an", "the"})
        for edge in edges:
            if _norm_text(edge["subject"]) == subject:
                add(edge)
    what_causes = re.match(r"^what\s+causes?\s+(.+?)\s*\?$", lowered)
    if what_causes:
        obj = what_causes.group(1)
        for edge in edges:
            if edge["predicate"].casefold() in {"causes", "cause"} and obj in edge["object"].casefold():
                add(edge)
    return candidates


def structured_candidates_current(query: str, rows: list[dict]) -> tuple[list[dict], str]:
    """Call the package's current structured answer path without side effects."""
    try:
        from nanlmind_hermes.mth_layers import structured_candidates  # noqa: WPS433
        candidates = structured_candidates(query, rows)
        return list(candidates or []), "nanlmind_hermes.mth_layers.structured_candidates"
    except Exception as exc:  # pragma: no cover - the repo normally imports.
        candidates = _fallback_structured_candidates(query, rows)
        return candidates, "fallback (%s: %s)" % (type(exc).__name__, str(exc)[:160])


def _candidate_contains_expected(candidates: list[dict],
                                 expected: Iterable[str]) -> bool:
    return any(_contains_expected(str(candidate.get("text") or ""), expected)
               for candidate in candidates)


def current_code_snapshot() -> dict[str, Any]:
    """Record which checkout's structured_candidates() produced the evidence."""
    try:
        import nanlmind_hermes.mth_layers as module  # noqa: WPS433
        path = Path(module.__file__).expanduser()
        return {
            "module": "nanlmind_hermes.mth_layers",
            "path": str(path),
            "sha256": _sha256_file(path),
            "mtime_ns": path.stat().st_mtime_ns,
        }
    except Exception as exc:  # pragma: no cover - repo normally imports.
        return {
            "module": "nanlmind_hermes.mth_layers",
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
        }


# ---------------------------------------------------------------------------
# Teaching statement triples (diagnostic only, never written to the DB)
# ---------------------------------------------------------------------------

_TRIPLE_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<predicate>"
    r"is not|are not|is|are|was|were|has|have|had|cannot|can't|can|may|will|"
    r"causes|cause|leads to|makes|make|creates?|created|produces?|becomes?|"
    r"mix(?:es)?\s+to\s+(?:make|form)"
    r")\s+(?P<object>.+?)[.!]?$",
    re.I,
)
_TURN_RE = re.compile(
    r"^(?P<subject>.+?)\s+turns?\s+(?P<middle>.+?)\s+into\s+(?P<object>.+?)[.!]?$",
    re.I,
)


def statement_triples(statements: Iterable[dict | str]) -> list[dict]:
    triples = []
    for item in statements or []:
        if isinstance(item, dict):
            statement = str(item.get("statement") or "")
            round_number = item.get("round")
        else:
            statement = str(item or "")
            round_number = None
        sentence = statement.strip()
        if not sentence or sentence.endswith("?"):
            continue
        match = _TRIPLE_RE.match(sentence)
        if match:
            subject = match.group("subject").strip(" \t\"'")[:240]
            predicate = match.group("predicate").strip().casefold()
            obj = match.group("object").strip(" \t\"'")[:512]
        else:
            match = _TURN_RE.match(sentence)
            if not match:
                continue
            subject = match.group("subject").strip(" \t\"'")[:240]
            predicate = "turns into"
            obj = match.group("object").strip(" \t\"'")[:512]
        if not subject or not obj:
            continue
        triples.append({
            "kind": "relation",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "polarity": "supports",
            "confidence": 0.5,
            "source": "teaching_statement",
            "source_sentence": sentence,
            "round": round_number,
        })
    return triples


def _statement_family(predicate: str) -> str:
    value = str(predicate or "").casefold()
    if value in {"is", "are", "was", "were", "is not", "are not", "becomes", "become"}:
        return "is"
    if value in {"has", "have", "had"}:
        return "has"
    if value in {"can", "may", "will", "cannot", "can't"}:
        return "can"
    if value.startswith(("cause", "lead")):
        return "causes"
    if value.startswith(("make", "create", "produce", "mix", "turn")):
        return "makes"
    return value


# ---------------------------------------------------------------------------
# Miss classification
# ---------------------------------------------------------------------------

def _same_subject(subject: str, query_terms: set[str]) -> bool:
    if not query_terms:
        return False
    subject_terms = _stems(subject)
    # Require the full query subject to match; a shared color/object token is
    # exactly what makes a sibling subject a sibling.
    return query_terms.issubset(subject_terms)


def _distinctive_terms(text: str, exclude: set[str]) -> set[str]:
    terms = _stems(text, QUESTION_NOISE)
    return {
        term for term in terms - exclude
        if len(term) >= 3 and term not in GENERIC_OBJECT_TOKENS
    }


def _compatible_sibling_family(question: str) -> set[str] | None:
    kind = question_kind(question)
    if kind == "what_make":
        return {"makes", "causes"}
    if kind in {"what_is", "what_color"}:
        return {"is"}
    # "what do X do?" does not name a predicate, so has/can/is/makes are all
    # legitimate ways the sibling fact could surface.
    return None


def _sibling_evidence(question: str, response: str,
                      query_terms: set[str], expected_terms: set[str],
                      relation_rows: list[dict], statements: list[dict]) -> list[dict]:
    response_terms = _stems(response, QUESTION_NOISE)
    if not response_terms:
        return []
    excluded = query_terms | expected_terms
    family_filter = _compatible_sibling_family(question)
    evidence = []
    for triple in list(relation_rows) + statement_triples(statements):
        subject = str(triple.get("subject") or "")
        obj = str(triple.get("object") or "")
        if _same_subject(subject, query_terms):
            continue
        family = _statement_family(str(triple.get("predicate") or ""))
        if family_filter is not None and family not in family_filter:
            continue
        subject_distinct = _distinctive_terms(subject, excluded)
        object_distinct = _distinctive_terms(obj, excluded)
        if not subject_distinct or not object_distinct:
            continue
        matched = object_distinct & response_terms
        if not matched:
            continue
        evidence.append({
            "source": triple.get("source") or "relation_row",
            "subject": subject,
            "predicate": triple.get("predicate"),
            "object": obj,
            "matched_response_terms": sorted(matched),
        })
        if len(evidence) >= 5:
            break
    return evidence


def _relevant_rows(question: str, expected: Iterable[str],
                   relation_rows: list[dict], limit: int = 30) -> list[dict]:
    query_terms = _query_terms(question)
    expected_terms = _expected_terms(expected)
    matched = []
    for row in relation_rows:
        subject_terms = _stems(str(row.get("subject") or ""))
        object_terms = _stems(str(row.get("object") or ""))
        if (subject_terms & query_terms or object_terms & expected_terms
                or subject_terms & expected_terms or object_terms & query_terms):
            matched.append(row)
    matched.sort(key=lambda row: (
        -len((_stems(str(row.get("subject") or "") + " " + str(row.get("object") or "")))
             & (query_terms | expected_terms)),
        str(row.get("subject") or ""),
        str(row.get("predicate") or ""),
        str(row.get("object") or ""),
    ))
    return matched[:limit]


def _taught_statements(question: str, expected: Iterable[str],
                       statements: list[dict], limit: int = 30) -> list[dict]:
    query_terms = _query_terms(question)
    expected_terms = _expected_terms(expected)
    matched = []
    for item in statements or []:
        text = str(item.get("statement") if isinstance(item, dict) else item)
        terms = _stems(text)
        if terms & query_terms or terms & expected_terms:
            matched.append({
                "round": item.get("round") if isinstance(item, dict) else None,
                "statement": text,
            })
    return matched[:limit]


def classify_miss(probe: dict[str, Any], response: str, *,
                  statements: Iterable[dict | str] = (),
                  relation_rows: Iterable[dict] = (),
                  target_facts: Iterable[str] = ()) -> dict[str, Any]:
    """Assign one primary cause plus contributing causes and evidence.

    ``statements`` should be cumulative teaching statements for the stage.
    ``relation_rows`` should be the relation rows available at that stage.
    """
    question = str(probe.get("question") or "").strip()
    expected = [str(item) for item in (probe.get("expected") or []) if str(item)]
    response = str(response or "")
    expected_terms = _expected_terms(expected)
    query_terms = _query_terms(question)
    kind = question_kind(question)
    statements = list(statements or [])
    relation_rows = list(relation_rows or [])
    statement_rows = statement_triples(statements)
    all_triples = relation_rows + statement_rows

    if not question or not expected_terms:
        return {
            "cause": "unknown",
            "contributing_causes": [],
            "question_kind": kind or "empty",
            "evidence": {"reason": "question or expected answer is missing"},
            "structured_candidate_possible": False,
            "structured_candidates": [],
        }

    candidates, candidate_source = structured_candidates_current(question, relation_rows)
    structured_possible = _candidate_contains_expected(candidates, expected)
    direct_rows = []
    causal_direct_rows = []
    expected_rows = []
    reverse_rows = []
    for row in relation_rows:
        subject_terms = _stems(str(row.get("subject") or ""))
        object_terms = _stems(str(row.get("object") or ""))
        if subject_terms & query_terms and object_terms & expected_terms:
            direct_rows.append(row)
            if _statement_family(str(row.get("predicate") or "")) in {"causes", "makes"}:
                causal_direct_rows.append(row)
        if ((subject_terms | object_terms) & expected_terms):
            expected_rows.append(row)
        if object_terms & query_terms and subject_terms & expected_terms:
            reverse_rows.append(row)

    statement_expected = any(
        _stems(str(item.get("statement") if isinstance(item, dict) else item)) & expected_terms
        for item in statements
    )
    statement_query = any(
        _stems(str(item.get("statement") if isinstance(item, dict) else item)) & query_terms
        for item in statements
    )
    expected_evidence = statement_expected or bool(expected_rows)
    parser_gap = statement_expected and not any(
        _stems(str(row.get("subject") or "") + " " + str(row.get("object") or "")) & expected_terms
        for row in relation_rows
    )
    siblings = _sibling_evidence(question, response, query_terms, expected_terms,
                                 relation_rows, statements)
    intent_supported = kind in {
        "what_is", "causal", "yes_no", "what_do", "what_make", "reverse",
    }

    evidence = {
        "query_terms": sorted(query_terms),
        "expected_terms": sorted(expected_terms),
        "question_kind": kind,
        "structured_intent_supported": intent_supported,
        "structured_candidate_possible": structured_possible,
        "structured_candidate_source": candidate_source,
        "direct_relation_rows": len(direct_rows),
        "causal_direct_relation_rows": len(causal_direct_rows),
        "expected_relation_rows": len(expected_rows),
        "reverse_relation_rows": len(reverse_rows),
        "statement_contains_expected": statement_expected,
        "statement_contains_query_terms": statement_query,
        "taught_but_not_extracted": parser_gap,
        "sibling_evidence": siblings,
        "relation_rows_available": len(relation_rows),
    }

    cause = "unknown"
    contributing: list[str] = []
    needs_reverse = (kind == "reverse"
                     or (kind == "what_is" and not direct_rows and bool(reverse_rows)))
    if not expected_evidence:
        cause = "teacher_curriculum_missing"
    elif kind == "causal" and not causal_direct_rows:
        # A state row such as "Frozen water is ice" is not a causal answer for
        # "what happens when water freezes?". Only a causal/make row counts.
        # A current-code candidate is still reported separately below.
        cause = "causal_question_missing"
    elif needs_reverse and not direct_rows:
        cause = "reverse_lookup_missing"
    elif siblings:
        cause = "wrong_subject_interference"
    elif structured_possible:
        cause = "vocabulary/generation"
    elif expected_evidence and not structured_possible:
        cause = "no_structured_intent"
    elif expected_evidence:
        cause = "vocabulary/generation"

    if cause != "teacher_curriculum_missing" and not expected_evidence:
        contributing.append("teacher_curriculum_missing")
    if cause != "no_structured_intent" and not intent_supported:
        contributing.append("no_structured_intent")
    if cause != "wrong_subject_interference" and siblings:
        contributing.append("wrong_subject_interference")
    if cause != "vocabulary/generation" and structured_possible:
        contributing.append("vocabulary/generation")
    if cause != "reverse_lookup_missing" and reverse_rows and kind == "reverse":
        contributing.append("reverse_lookup_missing")
    if cause != "causal_question_missing" and kind == "causal":
        contributing.append("causal_question_missing")
    contributing = [item for item in dict.fromkeys(contributing) if item != cause]

    return {
        "cause": cause,
        "contributing_causes": contributing,
        "question_kind": kind,
        "evidence": evidence,
        "structured_candidate_possible": structured_possible,
        "current_code_recoverable": structured_possible,
        "structured_candidates": [
            {
                "text": candidate.get("text"),
                "truth": candidate.get("truth"),
                "subject": candidate.get("subject"),
                "object": candidate.get("object"),
                "source": candidate.get("source"),
            }
            for candidate in candidates[:5]
        ],
        "direct_rows": direct_rows[:5],
        "reverse_rows": reverse_rows[:5],
        "sibling_evidence": siblings,
        "parser_gap": parser_gap,
        "taught_statements": _taught_statements(question, expected, statements),
    }


# ---------------------------------------------------------------------------
# Coverage matrix and plateau detection
# ---------------------------------------------------------------------------

def _probe_fact_scores(probe: dict[str, Any], facts: list[str]) -> list[tuple[int, int, int]]:
    probe_terms = _stems(str(probe.get("question") or "") + " " +
                         " ".join(str(item) for item in probe.get("expected") or []))
    expected_terms = _expected_terms(probe.get("expected") or [])
    scores = []
    for index, fact in enumerate(facts):
        fact_terms = _stems(fact)
        overlap = len(probe_terms & fact_terms)
        expected_overlap = len(expected_terms & fact_terms)
        scores.append((overlap, expected_overlap, -index))
    return scores


def assign_probes_to_facts(facts: list[str],
                           probes: list[dict[str, Any]]) -> dict[int, list[str]]:
    assigned: dict[int, list[str]] = {index: [] for index in range(len(facts))}
    for probe in probes:
        if not facts:
            continue
        scores = _probe_fact_scores(probe, facts)
        best = max(scores) if scores else None
        if best is None or best[0] <= 0:
            continue
        index = -best[2]
        question = str(probe.get("question") or "")
        if question and question not in assigned[index]:
            assigned[index].append(question)
    return assigned


def build_coverage(facts: list[str], probes: list[dict[str, Any]],
                   stages: list[dict[str, Any]],
                   relation_rows: list[dict],
                   timeline: list[dict]) -> dict[str, Any]:
    assigned = assign_probes_to_facts(facts, probes)
    probe_index = {str(item.get("question")): item for item in timeline}
    rows = []
    for index, fact in enumerate(facts):
        fact_terms = _stems(fact)
        questions = assigned.get(index, [])
        probe_terms = set()
        for question in questions:
            probe = next((item for item in probes if item.get("question") == question), {})
            probe_terms |= _stems(question + " " + " ".join(probe.get("expected") or []))
        taught = []
        for stage in stages:
            for statement in stage.get("statements") or []:
                if _stems(statement) & (fact_terms | probe_terms):
                    taught.append({"round": stage.get("round"), "statement": statement})
        relevant_rows = []
        for row in relation_rows:
            row_terms = _stems(str(row.get("subject") or "") + " " + str(row.get("object") or ""))
            if row_terms & (fact_terms | probe_terms):
                relevant_rows.append({
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                    "polarity": row.get("polarity"),
                    "confidence": row.get("confidence"),
                    "first_available_round": row.get("first_available_round"),
                })
        probe_results = []
        for question in questions:
            timeline_row = probe_index.get(question)
            if timeline_row:
                probe_results.append({
                    "question": question,
                    "expected": timeline_row.get("expected"),
                    "final_status": timeline_row.get("final_status"),
                    "final_cause": timeline_row.get("final_cause"),
                    "hits": timeline_row.get("hits"),
                    "misses": timeline_row.get("misses"),
                })
        if not questions:
            status = "no_probe"
        elif not taught:
            status = "no_taught_statements"
        elif not relevant_rows:
            status = "no_relation_rows"
        elif all(item.get("final_status") == "hit" for item in probe_results):
            status = "probe_all_hit"
        elif any(item.get("final_status") == "hit" for item in probe_results):
            status = "probe_partial"
        else:
            status = "probe_all_miss"
        rows.append({
            "target_fact": fact,
            "probes": questions,
            "taught_statement_count": len(taught),
            "taught_statements": taught[:30],
            "relation_row_count": len(relevant_rows),
            "relation_rows": relevant_rows[:20],
            "probe_results": probe_results,
            "coverage_status": status,
        })
    return {
        "rows": rows,
        "unprobed_facts": [row["target_fact"] for row in rows if not row["probes"]],
        "facts_without_relation_rows": [row["target_fact"] for row in rows
                                        if not row["relation_rows"]],
        "facts_not_all_hit": [row["target_fact"] for row in rows
                              if row["probe_results"]
                              and row["coverage_status"] != "probe_all_hit"],
    }


def detect_plateaus(stages: list[dict[str, Any]],
                    hit_rates: list[float]) -> list[dict[str, Any]]:
    plateaus = []
    start = 0
    while start < len(hit_rates):
        end = start + 1
        while end < len(hit_rates) and abs(hit_rates[end] - hit_rates[start]) < 1e-9:
            end += 1
        if end - start >= 2:
            stage_slice = stages[start:end]
            misses_by_stage = []
            shared = None
            for stage in stage_slice:
                missed = {str(row.get("question"))
                          for row in stage.get("probes") or [] if not row.get("hit")}
                misses_by_stage.append({"stage": stage["stage"],
                                        "misses": sorted(missed)})
                shared = missed if shared is None else shared & missed
            probe_count = len(stage_slice[0].get("probes") or [])
            hit_count = sum(1 for row in stage_slice[0].get("probes") or [] if row.get("hit"))
            plateaus.append({
                "rate": hit_rates[start],
                "percent": round(hit_rates[start] * 100, 2),
                "hit_count": hit_count,
                "probe_count": probe_count,
                "stages": [stage["stage"] for stage in stage_slice],
                "shared_misses": sorted(shared or set()),
                "misses_by_stage": misses_by_stage,
                "length": end - start,
            })
        start = end
    return plateaus


def _fragile_hit(question: str, expected: list[str], response: str,
                 relation_rows: list[dict]) -> dict[str, Any] | None:
    candidates, source = structured_candidates_current(question, relation_rows)
    if _candidate_contains_expected(candidates, expected):
        return None
    return {
        "question": question,
        "response": response,
        "expected": expected,
        "reason": "expected token appeared in free generation, not in a structured candidate",
        "candidate_source": source,
        "candidate_count": len(candidates),
    }


# ---------------------------------------------------------------------------
# Ablation plan (never executed here)
# ---------------------------------------------------------------------------

def ablation_plan() -> list[dict[str, Any]]:
    plan = [
        {
            "id": "A1_more_repetitions",
            "status": "planned_not_run",
            "hypothesis": "Repeated, spacing-varied exposure stabilizes the temporal frame and reduces drift between sibling facts.",
            "intervention": "Expose every target fact 3x per round across rounds, with the same fact in different positions and sentence lengths. Hold probes fixed.",
            "example_statements": [
                "A river carries water.",
                "Water is what a river carries.",
                "Water can flow.",
            ],
            "does_not_leak_probe": True,
            "readout": ["hit_rate_curve", "fragile_hit_count", "wrong_subject_interference_count"],
            "pass_signal": "Fewer wrong-subject misses and fewer fragile hits; no exact expected token inserted into teaching.",
        },
        {
            "id": "A2_explicit_qa_teaching_without_exact_probe",
            "status": "planned_not_run",
            "hypothesis": "Answer-shaped exposure teaches the question frame without memorizing the held-out probe string.",
            "intervention": "Add indirect question/answer statements whose wording differs from the probe. Never include an exact probe question in teaching.",
            "example_statements": [
                "Q: Which color does the sky have? A: Blue.",
                "Q: Name the thing a river carries. A: Water.",
                "Q: Which animal says meow? A: A cat.",
            ],
            "does_not_leak_probe": True,
            "readout": ["probe_exact_question_overlap", "hit_rate_curve", "structured_candidate_rate"],
            "pass_signal": "Structured candidate rate rises while exact probe overlap stays zero.",
        },
        {
            "id": "A3_subject_balanced_color_facts",
            "status": "planned_not_run",
            "hypothesis": "Teaching sibling facts together prevents a later fact from overwriting an earlier sibling's object.",
            "intervention": "Teach sky/grass/fire/snow and red+blue/yellow+blue in the same round, balanced repetitions, with explicit subject labels.",
            "example_statements": [
                "The sky is blue.",
                "Grass is green.",
                "Fire is red.",
                "Snow is white.",
                "Red and blue make purple.",
                "Yellow and blue make green.",
            ],
            "does_not_leak_probe": True,
            "readout": ["sibling_interference_count", "per_probe_stability"],
            "pass_signal": "Red+blue and yellow+blue remain distinct across rounds; no sibling object swap.",
        },
        {
            "id": "A4_reverse_lookup",
            "status": "planned_not_run",
            "hypothesis": "Explicit inverse-direction exposure answers property-to-subject probes such as 'what shape has four sides?'.",
            "intervention": "Teach forward and reverse forms as separate statements; add a reverse lookup path over relation rows keyed by object/property.",
            "example_statements": [
                "A square has four sides.",
                "Four sides make a square.",
                "A shape with four equal sides is a square.",
                "A circle is round.",
                "The round shape is a circle.",
            ],
            "does_not_leak_probe": True,
            "readout": ["reverse_lookup_missing_count", "shape_probe_status"],
            "pass_signal": "The object/property question selects a subject from the inverse relation index.",
        },
        {
            "id": "A5_causal_paraphrase",
            "status": "planned_not_run",
            "hypothesis": "Causal paraphrases that use extractable verbs give the causal frame something to retrieve.",
            "intervention": "Teach the water-freezing causal chain with canonical causal verbs in addition to descriptive forms.",
            "example_statements": [
                "Freezing water causes ice.",
                "Cold causes water to freeze.",
                "When water freezes it becomes ice.",
                "Ice is frozen water.",
            ],
            "does_not_leak_probe": True,
            "readout": ["causal_question_missing_count", "water_freezing_probe_status"],
            "pass_signal": "The causal question returns the caused object without relying on an incidental token elsewhere in a generated sentence.",
        },
    ]
    return plan


def _plan_leaks_exact_probe(plan: list[dict[str, Any]],
                            probes: list[dict[str, Any]]) -> list[str]:
    leaks = []
    questions = [str(probe.get("question") or "").strip().casefold()
                 for probe in probes if probe.get("question")]
    for item in plan:
        for statement in item.get("example_statements") or []:
            text = str(statement).strip().casefold()
            for question in questions:
                if text == question or question in text:
                    leaks.append("%s: %s" % (item.get("id"), statement))
    return leaks


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_run(run_path: str | Path = DEFAULT_RUN_JSON,
                state_dir: str | Path | None = None,
                db_path: str | Path | None = None) -> dict[str, Any]:
    record = load_run(run_path)
    stages = _stage_rows(record)
    probes = _probe_catalog(record)
    target_facts = [str(item) for item in (record.get("target_facts") or []) if str(item).strip()]

    selected_state = state_dir or record.get("state_dir") or DEFAULT_STATE_DIR
    selected_db = db_path
    if selected_db is None and selected_state:
        selected_db = Path(str(selected_state)).expanduser() / "universe" / "mathematical_memory.sqlite3"
    db = inspect_memory_db(selected_db)
    relation_rows = db.get("relation_rows") or []
    annotate_relation_rounds(relation_rows, stages)

    cumulative_statements: list[dict] = []
    analyzed_stages = []
    hit_rates = []
    all_misses = []
    fragile_hits = []
    for stage in stages:
        cumulative_statements.extend(
            {"round": stage.get("round"), "statement": statement}
            for statement in stage.get("statements") or []
        )
        round_number = int(stage.get("round") or 0)
        available_rows = _rows_available_at(relation_rows, round_number)
        probes_by_question = {
            str(item.get("question")): item for item in stage.get("probes") or []
        }
        hits = []
        misses = []
        stage_fragile = []
        for probe in probes:
            row = probes_by_question.get(probe["question"])
            if row is None:
                continue
            response = str(row.get("response") or "")
            hit = bool(row.get("hit"))
            if hit:
                hits.append({"question": probe["question"], "response": response})
                fragile = _fragile_hit(probe["question"], probe["expected"], response,
                                       available_rows)
                if fragile:
                    fragile["stage"] = stage["stage"]
                    stage_fragile.append(fragile)
                    fragile_hits.append(fragile)
            else:
                classification = classify_miss(
                    probe, response, statements=cumulative_statements,
                    relation_rows=available_rows, target_facts=target_facts)
                miss = {
                    "question": probe["question"],
                    "expected": probe["expected"],
                    "response": response,
                    "stage": stage["stage"],
                    **classification,
                }
                misses.append(miss)
                all_misses.append(miss)
        actual_hit_rate = _rate(len(hits), len(hits) + len(misses))
        reported_rate = stage.get("hit_rate")
        if not isinstance(reported_rate, (int, float)):
            reported_rate = actual_hit_rate
        hit_rates.append(float(reported_rate))
        analyzed_stages.append({
            "stage": stage["stage"],
            "round": stage.get("round"),
            "hit_rate": float(reported_rate),
            "hit_count": len(hits),
            "miss_count": len(misses),
            "probe_count": len(hits) + len(misses),
            "hits": hits,
            "misses": misses,
            "fragile_hits": stage_fragile,
            "structured_hit_count": len(hits) - len(stage_fragile),
            "statements_taught": len(stage.get("statements") or []),
            "cumulative_statement_count": len(cumulative_statements),
            "relation_rows_available": len(available_rows),
        })

    cause_counts = {cause: 0 for cause in CAUSES}
    for miss in all_misses:
        cause_counts[miss["cause"]] = cause_counts.get(miss["cause"], 0) + 1
    contributing_counts = {cause: 0 for cause in CAUSES}
    for miss in all_misses:
        for cause in miss.get("contributing_causes") or []:
            contributing_counts[cause] = contributing_counts.get(cause, 0) + 1
    code_level_counts = {
        cause: cause_counts.get(cause, 0) + contributing_counts.get(cause, 0)
        for cause in CAUSES if cause in CODE_LEVEL_CAUSES
    }

    # Probe timeline.
    timeline = []
    for probe in probes:
        entries = []
        for stage in stages:
            row = next((item for item in stage.get("probes") or []
                        if str(item.get("question")) == probe["question"]), None)
            if row is None:
                continue
            entries.append({
                "stage": stage["stage"],
                "hit": bool(row.get("hit")),
                "response": str(row.get("response") or ""),
            })
        analyzed = next((item for item in analyzed_stages
                         if item["stage"] == entries[-1]["stage"]), None) if entries else None
        final_entry = entries[-1] if entries else None
        final_miss = None
        if analyzed and final_entry and not final_entry["hit"]:
            final_miss = next((item for item in analyzed["misses"]
                               if item["question"] == probe["question"]), None)
        timeline.append({
            "question": probe["question"],
            "expected": probe["expected"],
            "entries": entries,
            "hits": [entry["stage"] for entry in entries if entry["hit"]],
            "misses": [entry["stage"] for entry in entries if not entry["hit"]],
            "final_status": "hit" if final_entry and final_entry["hit"] else "miss",
            "final_cause": final_miss["cause"] if final_miss else None,
            "fragile_hit_stages": [
                item["stage"] for item in fragile_hits if item["question"] == probe["question"]
            ],
        })

    # Coverage matrix.
    coverage = build_coverage(target_facts, probes, stages, relation_rows, timeline)

    # DB evidence for every probe that missed at least once.
    db_findings = []
    for probe in probes:
        missed = [miss for miss in all_misses if miss["question"] == probe["question"]]
        if not missed:
            continue
        final_rows = _relevant_rows(probe["question"], probe["expected"], relation_rows)
        final_statements = _taught_statements(probe["question"], probe["expected"],
                                               cumulative_statements)
        final_candidates, candidate_source = structured_candidates_current(
            probe["question"], relation_rows)
        probe_query = _query_terms(probe["question"])
        probe_expected = _expected_terms(probe["expected"])
        traceable_rows = [row for row in relation_rows
                          if row.get("traceable_to_run", True)]
        row_expected = any(
            ((_stems(str(row.get("object") or "")) & probe_expected
              and _stems(str(row.get("subject") or "")) & probe_query)
             or (_stems(str(row.get("subject") or "")) & probe_expected
                 and _stems(str(row.get("object") or "")) & probe_query))
            for row in traceable_rows
        )
        db_findings.append({
            "question": probe["question"],
            "expected": probe["expected"],
            "question_kind": missed[-1].get("question_kind"),
            "missed_stages": [miss["stage"] for miss in missed],
            "causes": sorted({miss["cause"] for miss in missed}),
            "cause_by_stage": {miss["stage"]: miss["cause"] for miss in missed},
            "relation_rows": [
                {
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                    "polarity": row.get("polarity"),
                    "confidence": row.get("confidence"),
                    "head": row.get("head"),
                    "layer": row.get("layer"),
                    "source": row.get("source"),
                    "first_available_round": row.get("first_available_round"),
                    "traceable_to_run": row.get("traceable_to_run"),
                }
                for row in final_rows
            ],
            "relation_row_count": len(final_rows),
            "expected_in_relation_rows": row_expected,
            "taught_statements": final_statements,
            "taught_but_not_extracted": bool(final_statements) and not row_expected,
            "structured_candidate_possible": _candidate_contains_expected(
                final_candidates, probe["expected"]),
            "structured_candidates": [
                {
                    "text": candidate.get("text"),
                    "truth": candidate.get("truth"),
                    "subject": candidate.get("subject"),
                    "object": candidate.get("object"),
                    "source": candidate.get("source"),
                }
                for candidate in final_candidates[:8]
            ],
            "structured_candidate_source": candidate_source,
            "current_code_supported_intents": list(SUPPORTED_STRUCTURED_INTENTS),
        })

    plateaus = detect_plateaus(stages, hit_rates)
    final_stage = analyzed_stages[-1] if analyzed_stages else None
    plan = ablation_plan()
    leaks = _plan_leaks_exact_probe(plan, probes)
    if leaks:
        raise RuntimeError("ablation plan leaks exact probe text: %s" % "; ".join(leaks))

    rounds_present = len(record.get("rounds") or [])
    rounds_requested = record.get("rounds_requested")
    if isinstance(rounds_requested, int) and rounds_requested > 0:
        run_status = "complete" if rounds_present >= rounds_requested else "in_progress"
    else:
        run_status = "unknown"
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "path": record.get("_analysis", {}).get("path"),
            "sha256": record.get("_analysis", {}).get("sha256"),
            "bytes": record.get("_analysis", {}).get("bytes"),
            "mtime_ns": record.get("_analysis", {}).get("mtime_ns"),
            "schema": record.get("schema"),
            "model": record.get("model"),
            "state_dir": record.get("state_dir"),
            "rounds_requested": rounds_requested,
            "rounds_present": rounds_present,
            "run_status": run_status,
            "target_fact_count": len(target_facts),
            "probe_count": len(probes),
        },
        "inputs": {
            "run_json": str(Path(run_path).expanduser()),
            "state_dir": str(selected_state) if selected_state else None,
            "db_path": str(selected_db) if selected_db else None,
        },
        "db": db,
        "summary": {
            "hit_rate_curve": [
                {"stage": stage["stage"], "hit_rate": stage["hit_rate"],
                 "hit_count": stage["hit_count"], "probe_count": stage["probe_count"]}
                for stage in analyzed_stages
            ],
            "benchmark_hit_rate_curve": [
                {"stage": stage["stage"], "hit_rate": stage.get("hit_rate")}
                for stage in stages
            ],
            "plateaus": plateaus,
            "final_stage": final_stage["stage"] if final_stage else None,
            "final_hit_rate": final_stage["hit_rate"] if final_stage else None,
            "final_misses": [
                {
                    "question": miss["question"],
                    "expected": miss["expected"],
                    "response": miss["response"],
                    "cause": miss["cause"],
                    "contributing_causes": miss.get("contributing_causes") or [],
                    "structured_candidate_possible": miss.get("structured_candidate_possible"),
                }
                for miss in (final_stage or {}).get("misses", [])
            ],
            "cause_counts": cause_counts,
            "contributing_cause_counts": contributing_counts,
            "code_level_cause_counts": code_level_counts,
            "fragile_hit_count": len(fragile_hits),
            "fragile_hits": fragile_hits,
            "structured_candidate_hits": sum(
                stage["hit_count"] - len(stage["fragile_hits"])
                for stage in analyzed_stages
            ),
        },
        "rounds": analyzed_stages,
        "probe_timeline": timeline,
        "probe_db_findings": db_findings,
        "coverage": coverage,
        "ablation_plan": plan,
        "ablation_plan_leak_check": {"exact_probe_leaks": leaks, "passed": not leaks},
        "cause_definitions": CAUSE_DEFINITIONS,
        "current_code": {
            "structured_intents": list(SUPPORTED_STRUCTURED_INTENTS),
            "structured_source": "nanlmind_hermes.mth_layers.structured_candidates",
            "snapshot": current_code_snapshot(),
            "note": (
                "The harness calls the package's pure structured_candidates() "
                "function to test whether the current code can produce a "
                "candidate. It does not import or invoke the training coordinator. "
                "If the checkout changed after the run JSON was written, "
                "current_code_recoverable can be true for a probe the run missed."
            ),
        },
        "boundaries": {
            "read_only": True,
            "package_modified": False,
            "ablation_executed": False,
            "db_snapshot_is_final_state": True,
            "per_round_db_rows_filtered_by_source_teaching_round": True,
            "probe_hit_metric": "expected token appears anywhere in the reply (same as the run script)",
            "proof_claim": False,
            "canonical_commit": False,
        },
    }
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _bounded(lines: list[str], max_lines: int = MAX_SUMMARY_LINES,
             max_chars: int = MAX_SUMMARY_CHARS) -> str:
    selected = []
    used = 0
    for line in lines:
        if len(selected) >= max_lines or used + len(line) + 1 > max_chars:
            break
        selected.append(line)
        used += len(line) + 1
    if len(selected) < len(lines):
        selected.append("... summary bounded (%d lines, %d chars)" %
                        (max_lines, max_chars))
    return "\n".join(selected)


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    return "%.1f%%" % (float(value) * 100.0)


def render_text(report: dict[str, Any], max_lines: int = MAX_SUMMARY_LINES,
                max_chars: int = MAX_SUMMARY_CHARS) -> str:
    lines = ["NaNLM plateau diagnostic", ""]
    run = report.get("run") or {}
    db = report.get("db") or {}
    lines.append("run: %s" % run.get("path"))
    lines.append("sha256: %s" % run.get("sha256"))
    lines.append("rounds: %s requested / %s present | status: %s | probes: %s" %
                 (run.get("rounds_requested"), run.get("rounds_present"),
                  run.get("run_status"), run.get("probe_count")))
    if not run.get("rounds_present"):
        lines.append("no completed teaching round in this snapshot; analysis is baseline-only")
    lines.append("db: %s" % (db.get("path") or "(not configured)"))
    lines.append("db status: %s | read_only=%s | unchanged=%s | relation_rows=%s" %
                 (db.get("status"), db.get("read_only"), db.get("unchanged"),
                  db.get("relation_rows_total")))
    lines.append("")
    lines.append("hit-rate curve:")
    for item in (report.get("summary") or {}).get("hit_rate_curve") or []:
        lines.append("  %-10s %6s  (%s/%s)" % (
            item.get("stage"), _percent(item.get("hit_rate")),
            item.get("hit_count"), item.get("probe_count")))
    plateaus = (report.get("summary") or {}).get("plateaus") or []
    if plateaus:
        lines.append("")
        lines.append("plateaus:")
        for plateau in plateaus[:4]:
            lines.append("  %s across %s (%s/%s)" % (
                _percent(plateau.get("rate")), ", ".join(plateau.get("stages") or []),
                plateau.get("hit_count"), plateau.get("probe_count")))
            for miss in plateau.get("shared_misses") or []:
                lines.append("    shared miss: %s" % miss)
    final = (report.get("summary") or {}).get("final_stage")
    if final:
        lines.append("")
        lines.append("final stage %s: %s" % (
            final, _percent((report.get("summary") or {}).get("final_hit_rate"))))
        for miss in (report.get("summary") or {}).get("final_misses") or []:
            lines.append("  MISS %s -> cause=%s" % (miss.get("question"), miss.get("cause")))
    lines.append("")
    lines.append("miss causes (primary):")
    counts = (report.get("summary") or {}).get("cause_counts") or {}
    for cause in CAUSES:
        if counts.get(cause):
            lines.append("  %-30s %s" % (cause, counts[cause]))
    code_counts = (report.get("summary") or {}).get("code_level_cause_counts") or {}
    if code_counts:
        lines.append("code-level signal (primary+contributing):")
        for cause, count in sorted(code_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append("  %-30s %s" % (cause, count))
    lines.append("fragile hits (token in free generation, no structured candidate): %s" %
                 (report.get("summary") or {}).get("fragile_hit_count"))
    lines.append("")
    lines.append("probe DB evidence (missed at least once):")
    for finding in (report.get("probe_db_findings") or [])[:10]:
        lines.append("  %s" % finding.get("question"))
        lines.append("    causes=%s rows=%s taught=%s candidate_possible=%s" % (
            ",".join(finding.get("causes") or []),
            finding.get("relation_row_count"),
            len(finding.get("taught_statements") or []),
            finding.get("structured_candidate_possible")))
    lines.append("")
    lines.append("coverage gaps:")
    coverage = report.get("coverage") or {}
    lines.append("  no probe: %s" % (len(coverage.get("unprobed_facts") or [])))
    lines.append("  no relation rows: %s" % (len(coverage.get("facts_without_relation_rows") or [])))
    lines.append("  not all probes hit: %s" % (len(coverage.get("facts_not_all_hit") or [])))
    lines.append("")
    lines.append("ablation plan (planned, not run):")
    for item in report.get("ablation_plan") or []:
        lines.append("  %-46s %s" % (item.get("id"), item.get("status")))
    lines.append("")
    lines.append("Boundaries: read-only; no package changes; no ablation executed; "
                 "hit metric is expected-token substring.")
    return _bounded(lines, max_lines=max_lines, max_chars=max_chars)


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")[:300]


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(item) for item in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    run = report.get("run") or {}
    db = report.get("db") or {}
    summary = report.get("summary") or {}
    lines = [
        "# NaNLM plateau diagnostic",
        "",
        "Generated: `%s`" % report.get("generated_at"),
        "",
        "## Snapshot",
        "",
        "- run JSON: `%s`" % run.get("path"),
        "- run sha256: `%s`" % run.get("sha256"),
        "- model/teacher: `%s`" % run.get("model"),
        "- state dir: `%s`" % (report.get("inputs") or {}).get("state_dir"),
        "- memory DB: `%s`" % (db.get("path") or "(not configured)"),
        "- DB status: `%s`, read_only=`%s`, unchanged=`%s`, relation rows=`%s`" % (
            db.get("status"), db.get("read_only"), db.get("unchanged"),
            db.get("relation_rows_total")),
        "- probes: %s; target facts: %s; run status: `%s`" % (
            run.get("probe_count"), run.get("target_fact_count"),
            run.get("run_status")),
        "- current structured path: `%s` sha256=`%s`" % (
            (report.get("current_code") or {}).get("snapshot", {}).get("path"),
            (report.get("current_code") or {}).get("snapshot", {}).get("sha256")),
        "",
        "## Hit rate",
        "",
    ]
    lines.extend(_md_table(
        ["stage", "hit rate", "hits", "probes"],
        [[item.get("stage"), _percent(item.get("hit_rate")),
          item.get("hit_count"), item.get("probe_count")]
         for item in summary.get("hit_rate_curve") or []],
    ))
    lines.extend(["", "## Plateaus", ""])
    plateaus = summary.get("plateaus") or []
    if not plateaus:
        lines.append("No repeated hit-rate value was present in this snapshot.")
    for plateau in plateaus:
        lines.append("### %s across %s" % (
            _percent(plateau.get("rate")), ", ".join(plateau.get("stages") or [])))
        lines.append("")
        lines.append("Hit count: %s/%s. Shared misses: %s" % (
            plateau.get("hit_count"), plateau.get("probe_count"),
            ", ".join("`%s`" % item for item in plateau.get("shared_misses") or []) or "none"))
        lines.append("")
        by_stage = {item["stage"]: item["misses"] for item in plateau.get("misses_by_stage") or []}
        stage_names = plateau.get("stages") or []
        rows = []
        all_questions = sorted({question for misses in by_stage.values() for question in misses})
        for question in all_questions:
            row = [question]
            for stage in stage_names:
                row.append("MISS" if question in (by_stage.get(stage) or []) else "hit")
            rows.append(row)
        lines.extend(_md_table(["probe"] + stage_names, rows))
        lines.append("")
    lines.extend(["## Cause summary", ""])
    lines.extend(_md_table(
        ["cause", "primary misses", "contributing", "definition"],
        [[cause, (summary.get("cause_counts") or {}).get(cause, 0),
          (summary.get("contributing_cause_counts") or {}).get(cause, 0),
          CAUSE_DEFINITIONS.get(cause, "")]
         for cause in CAUSES],
    ))
    lines.extend([
        "",
        "Code-level signal (primary + contributing): " +
        ", ".join("%s=%s" % (key, value) for key, value in
                  sorted((summary.get("code_level_cause_counts") or {}).items(),
                         key=lambda item: (-item[1], item[0]))),
        "",
        "Fragile hits (expected token in free generation, no structured candidate): **%s**" %
        summary.get("fragile_hit_count"),
        "",
        "## Per-round misses",
        "",
    ])
    miss_rows = []
    for stage in report.get("rounds") or []:
        for miss in stage.get("misses") or []:
            miss_rows.append([
                stage.get("stage"), miss.get("question"),
                ", ".join(miss.get("expected") or []),
                miss.get("response"), miss.get("cause"),
                ", ".join(miss.get("contributing_causes") or []),
                miss.get("structured_candidate_possible"),
            ])
    lines.extend(_md_table(
        ["stage", "probe", "expected", "reply", "cause", "contributing", "candidate possible"],
        miss_rows,
    ))
    lines.extend(["", "## Probe database evidence", ""])
    for finding in report.get("probe_db_findings") or []:
        lines.append("### %s" % finding.get("question"))
        lines.append("")
        lines.append("- expected: %s" % ", ".join(finding.get("expected") or []))
        lines.append("- missed stages: %s" % ", ".join(finding.get("missed_stages") or []))
        lines.append("- causes by stage: `%s`" % json.dumps(
            finding.get("cause_by_stage") or {}, sort_keys=True))
        lines.append("- taught but not extracted: `%s`" % finding.get("taught_but_not_extracted"))
        lines.append("- structured candidate possible with current code: `%s`" %
                     finding.get("structured_candidate_possible"))
        candidates = finding.get("structured_candidates") or []
        if candidates:
            lines.append("- structured candidates: " + "; ".join(
                "`%s`" % item.get("text") for item in candidates))
        else:
            lines.append("- structured candidates: none")
        if finding.get("relation_rows"):
            lines.append("")
            lines.extend(_md_table(
                ["subject", "predicate", "object", "polarity", "confidence", "round"],
                [[row.get("subject"), row.get("predicate"), row.get("object"),
                  row.get("polarity"), row.get("confidence"),
                  row.get("first_available_round")]
                 for row in finding.get("relation_rows") or []]))
        else:
            lines.append("- relation rows: none matched")
        if finding.get("taught_statements"):
            lines.append("")
            lines.append("Taught statements:")
            for item in finding.get("taught_statements") or []:
                lines.append("- round %s: %s" % (item.get("round"), item.get("statement")))
        lines.append("")
    lines.extend(["## Coverage matrix", ""])
    coverage_rows = []
    for row in (report.get("coverage") or {}).get("rows") or []:
        result = "; ".join(
            "%s=%s%s" % (item.get("question"), item.get("final_status"),
                         " (%s)" % item.get("final_cause") if item.get("final_cause") else "")
            for item in row.get("probe_results") or []) or "no probe"
        coverage_rows.append([
            row.get("target_fact"),
            "; ".join(row.get("probes") or []) or "none",
            row.get("taught_statement_count"),
            row.get("relation_row_count"),
            result,
            row.get("coverage_status"),
        ])
    lines.extend(_md_table(
        ["target fact", "probes", "taught", "relation rows", "probe result", "status"],
        coverage_rows,
    ))
    lines.extend(["", "## Code bugs vs curriculum gaps", ""])
    code_signal = summary.get("code_level_cause_counts") or {}
    lines.append("- Structured-intent gaps: `no_structured_intent=%s`, "
                 "`causal_question_missing=%s`, `reverse_lookup_missing=%s`." % (
                     code_signal.get("no_structured_intent", 0),
                     (summary.get("cause_counts") or {}).get("causal_question_missing", 0),
                     (summary.get("cause_counts") or {}).get("reverse_lookup_missing", 0)))
    lines.append("- Subject-binding failures: `wrong_subject_interference=%s`." %
                 code_signal.get("wrong_subject_interference", 0))
    parser_gaps = [item.get("question") for item in report.get("probe_db_findings") or []
                   if item.get("taught_but_not_extracted")]
    lines.append("- Taught-but-not-extracted probes: %s." % (
        ", ".join("`%s`" % item for item in parser_gaps) or "none"))
    curriculum_gaps = [item.get("question") for item in report.get("probe_db_findings") or []
                       if "teacher_curriculum_missing" in (item.get("causes") or [])]
    lines.append("- Curriculum gaps (expected answer absent at a stage): %s." % (
        ", ".join("`%s`" % item for item in curriculum_gaps) or "none"))
    lines.append("- Fragile hits: `%s` (a substring hit from free generation is not a "
                 "structured answer)." % summary.get("fragile_hit_count"))
    lines.extend(["", "## Ablation plan", "",
                  "Status: **planned, not run**. These interventions are not executed "
                  "by this harness and no fixed result is claimed.", ""])
    lines.extend(_md_table(
        ["id", "hypothesis", "intervention", "example statements", "readout", "status"],
        [[item.get("id"), item.get("hypothesis"), item.get("intervention"),
          " / ".join(item.get("example_statements") or []),
          ", ".join(item.get("readout") or []), item.get("status")]
         for item in report.get("ablation_plan") or []],
    ))
    lines.extend([
        "",
        "## Exact fixes and tests needed",
        "",
        "1. Certify the structured intents in the current checkout. If a branch for "
        "`what do X do?`, `what color is X?`, `what happens when X?`, "
        "`what shape has ...?`, or `what do X and Y make?` is present, add a "
        "regression test that feeds a relation row and asserts the answer. The "
        "recorded run predates the current `mth_layers.py` snapshot.",
        "2. Add or certify a reverse index/query from object or property to subject. "
        "Regression tests: `A square has four sides` -> `square`; "
        "`Frozen water is ice` -> `ice` or `frozen`.",
        "3. Extend `_NATIVE_PATTERNS` for `make/makes`, `create/creates`, "
        "`produce/produces`, `become/becomes`, `turn/turns into`, `mix ... to make`, "
        "`consist/consists of`, and `possess/possesses`. Test that each taught "
        "statement produces one relation row with the expected subject/predicate/object.",
        "4. Bind generation to the asked subject. When relation rows exist for the "
        "question subject, restrict candidate objects to those rows before free "
        "generation. Regression tests alternate red+blue/purple and yellow+blue/green.",
        "5. Balance sibling facts in the same curriculum round. Add a coverage test "
        "that both siblings are taught in the plateau round and neither probe returns "
        "the sibling's object.",
        "6. Add causal paraphrases and extractable causal verbs: `Freezing water "
        "causes ice`, `Cold causes water to freeze`, `When water freezes it becomes "
        "ice`. Test causal answer selection separately from free generation.",
        "7. Add a strict answer-slot metric alongside the existing substring hit "
        "metric. The round-6 yellow+blue reply contains the expected token `green` "
        "only incidentally after the wrong object `purple`; the diagnostic should mark "
        "that as a fragile/contradictory hit, not a clean success.",
        "",
        "## Boundaries",
        "",
        "- The run JSON may be replaced by a live writer; the snapshot sha256 is recorded above.",
        "- The DB is the current final ledger, not a historical per-round backup. "
        "Per-round row availability is approximated from the source teaching statement's round.",
        "- The probe hit metric is the same expected-token substring check as the run script.",
        "- No package code, package documentation, or training state was modified.",
        "- No ablation was run and no fixed/final accuracy is claimed.",
        "",
    ])
    return "\n".join(lines)


def write_reports(report: dict[str, Any],
                  json_path: str | Path | None = None,
                  markdown_path: str | Path | None = None) -> dict[str, str]:
    written: dict[str, str] = {}
    if json_path:
        target = Path(json_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                          encoding="utf-8")
        written["json"] = str(target)
    if markdown_path:
        target = Path(markdown_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_markdown(report), encoding="utf-8")
        written["markdown"] = str(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only plateau diagnostic for the native NaNLM student.")
    parser.add_argument("--run-json", default=str(DEFAULT_RUN_JSON),
                        help="run record (default: docs/NANLM_GEMMA_LEARNING_OVER_TIME.json)")
    parser.add_argument("--state-dir", default="",
                        help="state dir containing universe/mathematical_memory.sqlite3")
    parser.add_argument("--db", default="",
                        help="explicit memory DB path (overrides --state-dir discovery)")
    parser.add_argument("--json", dest="json_path", default="",
                        help="write the full report JSON to this path")
    parser.add_argument("--markdown", dest="markdown_path", default="",
                        help="write the report Markdown to this path")
    parser.add_argument("--format", choices=("text", "json", "markdown"),
                        default="text", help="stdout format (default: text)")
    parser.add_argument("--max-summary-lines", type=int, default=MAX_SUMMARY_LINES,
                        help="bound for the text summary (default: %d)" % MAX_SUMMARY_LINES)
    args = parser.parse_args(argv)
    if args.max_summary_lines < 1:
        parser.error("--max-summary-lines must be positive")

    report = analyze_run(
        args.run_json,
        state_dir=args.state_dir or None,
        db_path=args.db or None,
    )
    written = write_reports(report, args.json_path or None, args.markdown_path or None)
    if written:
        print(json.dumps({"written": written}, sort_keys=True), file=sys.stderr)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    elif args.format == "markdown":
        print(render_markdown(report))
    else:
        print(render_text(report, max_lines=args.max_summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
