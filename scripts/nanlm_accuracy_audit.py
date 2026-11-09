#!/usr/bin/env python3
"""Independent accuracy-measurement audit for the NaNLM learning run.

This is a read-only re-score of an existing
``nanlm.gemma-learning-over-time/1`` artifact. It does not run the student,
retrain, mutate the state directory, or claim a new accuracy result.

The audit has two purposes:

1. Recompute the legacy ``hit`` vector from each stored probe and explain the
   per-probe transitions. This distinguishes a count equality from an actual
   vector equality and shows that a one-hit improvement is not necessarily a
   bit shift.
2. Re-score the stored replies with three separate measurements:

   * ``keyword_hit``  - legacy substring containment anywhere in the reply;
   * ``structured_hit`` - keyword hit *and* the recorded structured flag;
   * ``direct_exact_hit`` - expected term in a structured/direct answer span.

The last two are strictly weaker than the legacy metric. They are offline
re-scores of the stored text, not new live observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RUN_JSON = ROOT / "docs" / "NANLM_GEMMA_LEARNING_OVER_TIME.json"
REPORT_SCHEMA = "nanlm.accuracy-audit/1"
MAX_SUMMARY_LINES = 80
MAX_SUMMARY_CHARS = 16000

# A direct answer must make the expected term part of an answer slot, not an
# incidental clause. These markers end the direct answer span used offline.
DIRECT_ANSWER_CUTS = (
    " comes from ",
    " because ",
    " due to ",
    " caused by ",
    " which ",
    " while ",
    " but ",
    ";",
)

# Predicate/verb cues that can introduce the answer object. The list is
# intentionally conservative: a token merely occurring next to a random noun
# does not count as a direct answer.
DIRECT_ANSWER_PREDICATES = (
    " consists of ",
    " possesses ",
    " possess ",
    " is ",
    " are ",
    " was ",
    " were ",
    " be ",
    " been ",
    " being ",
    " has ",
    " have ",
    " had ",
    " contains ",
    " contain ",
    " includes ",
    " include ",
    " makes ",
    " make ",
    " made ",
    " creates ",
    " create ",
    " created ",
    " produces ",
    " produce ",
    " produced ",
    " causes ",
    " cause ",
    " caused ",
    " turns ",
    " turn ",
    " turned ",
    " becomes ",
    " become ",
    " means ",
    " equals ",
    " forms ",
    " form ",
    " gives ",
    " give ",
    " melts ",
    " melt ",
    " flies ",
    " fly ",
    " barks ",
    " bark ",
    " meows ",
    " meow ",
    " mixes ",
    " mix ",
    " into ",
)

METRICS = ("keyword_hit", "structured_hit", "direct_exact_hit")

DEFINITIONS = {
    "keyword_hit": (
        "Legacy reproduction: any expected term is a case-insensitive substring "
        "of the full generated reply, regardless of clause, structure, or "
        "contradiction. This is the metric stored as `hit` in the old artifact."
    ),
    "structured_hit": (
        "keyword_hit AND the probe row's recorded `structured` flag is true. "
        "Missing/null structured values are counted as not measured, not as "
        "proof that no structured answer existed."
    ),
    "direct_exact_hit": (
        "The expected term occurs in a bounded direct-answer span: the first "
        "sentence up to a non-answer connector (`comes from`, `because`, "
        "`due to`, ...), either after a predicate cue or as the answer subject. "
        "A structured candidate is treated as direct by definition. This is a "
        "deterministic heuristic over stored text, not a semantic prover."
    ),
}

DEFECTS_STATIC = (
    {
        "id": "keyword_substring_anywhere",
        "severity": "high",
        "source": "scripts/nanlm_gemma_learning_over_time.py::_probe",
        "summary": (
            "`hit = any(word.casefold() in response.casefold() for word in "
            "expected)` counts an expected token anywhere in a generated reply. "
            "Unrelated clauses, contradictions, and rambling text can all count."
        ),
    },
    {
        "id": "no_direct_or_structured_requirement",
        "severity": "high",
        "source": "scripts/nanlm_gemma_learning_over_time.py::_probe",
        "summary": (
            "The headline `hit` ignores `structured` and does not require an "
            "answer slot. The script computes a secondary structured rate, but "
            "the primary `hit_rate` remains free-generation substring matching."
        ),
    },
    {
        "id": "probe_state_mutation",
        "severity": "high",
        "source": "scripts/nanlm_gemma_learning_over_time.py (old artifact)",
        "summary": (
            "The old run did not pass `probe=True`. Each evaluation question went "
            "through the normal learning path, so a probe could teach the student "
            "the question/reply and later rounds were not held out. The current "
            "script passes `probe=True`; the old artifact remains confounded."
        ),
    },
    {
        "id": "no_per_probe_state_fingerprint_or_flip_ledger",
        "severity": "medium",
        "source": "scripts/nanlm_gemma_learning_over_time.py (old artifact)",
        "summary": (
            "The old artifact stores no per-probe state fingerprint, no before/"
            "after digest, and no flip ledger. The current script adds a "
            "stage-level digest and derived per-round flips, but still cannot "
            "attribute a state change to an individual probe."
        ),
    },
    {
        "id": "implicit_denominator_and_missing_structured_values",
        "severity": "medium",
        "source": "docs/NANLM_GEMMA_LEARNING_OVER_TIME.json",
        "summary": (
            "The old artifact stores only float hit rates, with no numerator, "
            "denominator, or vector. Its `structured` values are null throughout, "
            "so a zero structured count is missing measurement, not a measured 0%."
        ),
    },
)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def _rate(hits: int, total: int) -> float:
    return (hits / total) if total else 0.0


def _binary(value: bool) -> str:
    return "1" if value else "0"


def _vector(flags: list[bool]) -> str:
    return "".join(_binary(bool(flag)) for flag in flags)


def _integer(vector: str) -> int:
    return int(vector, 2) if vector else 0


def _xor(left: str, right: str) -> str:
    width = max(len(left), len(right))
    return "".join(
        "1" if a != b else "0"
        for a, b in zip(left.zfill(width), right.zfill(width))
    )


def _term_present(text: str, term: str) -> bool:
    """Word-boundary match used by the direct-answer metric."""
    needle = str(term or "").strip().casefold()
    if not needle:
        return False
    return re.search(
        r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])",
        str(text or "").casefold(),
    ) is not None


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _answer_span(response: str) -> str:
    """Return the first direct-answer span, stopping at non-answer clauses."""
    text = _normalise_space(response)
    if not text:
        return ""
    # First sentence only: a later unrelated sentence cannot supply the answer.
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    folded = text.casefold()
    cut_at = len(text)
    for marker in DIRECT_ANSWER_CUTS:
        position = folded.find(marker)
        if position > 0:
            cut_at = min(cut_at, position)
    return text[:cut_at].strip(" .!?;,")


def keyword_hit(response: str, expected: list[str]) -> bool:
    """Legacy scoring, reproduced exactly.

    This deliberately uses substring containment, not word boundaries: the
    purpose is to expose the old measurement, not silently replace it.
    """
    folded = str(response or "").casefold()
    return any(str(term).casefold() in folded for term in expected if str(term))


def direct_exact_hit(response: str, expected: list[str],
                     structured: bool = False) -> bool:
    """Expected term in a direct answer slot.

    ``structured=True`` is accepted as direct by definition. Otherwise the
    first answer span must contain an expected term either at the start of the
    answer (optionally after an article) or after a predicate cue. This rejects
    the canonical yellow/blue reply whose only ``green`` is inside a later
    red/blue clause introduced by ``comes from``.
    """
    if not keyword_hit(response, expected):
        return False
    if structured:
        return True
    span = _answer_span(response)
    if not span:
        return False

    folded = " " + span.casefold() + " "
    for term in expected:
        needle = str(term or "").strip().casefold()
        if not needle or not _term_present(span, needle):
            continue

        # The answer itself can begin with the expected term: "Square ...",
        # "A square ...", or simply "Water.".
        start = span.casefold().strip()
        start = re.sub(r"^(a|an|the)\s+", "", start)
        if start == needle or start.startswith(needle + " "):
            return True

        # Or the expected term appears after a predicate cue in the same span.
        for predicate in DIRECT_ANSWER_PREDICATES:
            predicate_at = folded.find(predicate)
            if predicate_at < 0:
                continue
            expected_at = folded.find(" " + needle + " ")
            if expected_at > predicate_at:
                return True
    return False


# ---------------------------------------------------------------------------
# Loading and stage extraction
# ---------------------------------------------------------------------------

def load_run(path: str | Path, retries: int = 6) -> dict[str, Any]:
    """Load a run record and attach a content identity.

    A live training writer may replace the file while it is being read; retry a
    few times and always parse one complete byte snapshot.
    """
    target = Path(path).expanduser()
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            raw = target.read_bytes()
            record = json.loads(raw.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("run record must be a JSON object")
            record["_audit_input"] = {
                "path": str(target),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "mtime_ns": target.stat().st_mtime_ns,
            }
            return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.15 * (attempt + 1))
    raise SystemExit("could not read run JSON %s: %s" % (target, last_error))


def _stage_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    baseline = record.get("baseline")
    if isinstance(baseline, dict) and isinstance(baseline.get("probes"), list):
        entries.append({
            "label": "baseline",
            "round": 0,
            "record": baseline,
            "rows": baseline["probes"],
        })
    for position, raw in enumerate(record.get("rounds") or [], start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("probes"), list):
            continue
        round_number = raw.get("round", position)
        try:
            number = int(round_number)
        except (TypeError, ValueError):
            number = position
        entries.append({
            "label": "round %d" % number,
            "round": number,
            "record": raw,
            "rows": raw["probes"],
        })
    return entries


def _catalog(record: dict[str, Any],
             entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = record.get("probes")
    if isinstance(raw, list) and raw and all(
            isinstance(row, dict) and row.get("question") is not None
            for row in raw):
        return [{
            "index": index,
            "question": str(row.get("question") or ""),
            "expected": [str(item) for item in (row.get("expected") or [])],
        } for index, row in enumerate(raw)]

    for entry in entries:
        rows = entry["rows"]
        if rows and all(isinstance(row, dict) and row.get("question") is not None
                        for row in rows):
            return [{
                "index": index,
                "question": str(row.get("question") or ""),
                "expected": [str(item) for item in (row.get("expected") or [])],
            } for index, row in enumerate(rows)]
    return []


def _align_rows(rows: list[dict[str, Any]],
                catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) == len(catalog) and all(
            str(row.get("question") or "").casefold()
            == item["question"].casefold()
            for row, item in zip(rows, catalog)):
        return list(rows)

    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_question.setdefault(
                str(row.get("question") or "").casefold(), row)
    aligned = []
    for item in catalog:
        row = by_question.get(item["question"].casefold())
        if row is None:
            row = {
                "question": item["question"],
                "expected": item["expected"],
                "response": "",
                "hit": False,
                "structured": None,
                "_missing_in_stage": True,
            }
        aligned.append(row)
    return aligned


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _row_scores(row: dict[str, Any]) -> dict[str, Any]:
    expected = [str(item) for item in (row.get("expected") or [])]
    response = str(row.get("response") or "")
    structured_value = row.get("structured")
    structured_measured = structured_value is not None
    structured_flag = bool(structured_value) if structured_measured else False
    keyword = keyword_hit(response, expected)
    direct = direct_exact_hit(response, expected, structured=structured_flag)
    return {
        "question": str(row.get("question") or ""),
        "expected": expected,
        "response": response,
        "recorded_hit": bool(row.get("hit")),
        "structured_flag": structured_value,
        "structured_measured": structured_measured,
        "keyword_hit": keyword,
        "structured_hit": bool(keyword and structured_flag),
        "direct_exact_hit": bool(direct),
    }


def _scored_stage(entry: dict[str, Any],
                  catalog: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _align_rows(entry["rows"], catalog)
    scores = [_row_scores(row) for row in rows]
    denominator = len(scores)
    recorded_flags = [score["recorded_hit"] for score in scores]
    metric_blocks = {}
    for metric in METRICS:
        flags = [bool(score[metric]) for score in scores]
        vector = _vector(flags)
        metric_blocks[metric] = {
            "vector": vector,
            "integer": _integer(vector),
            "hits": sum(flags),
            "denominator": denominator,
            "rate": _rate(sum(flags), denominator),
        }
    recorded_vector = _vector(recorded_flags)
    keyword_vector = metric_blocks["keyword_hit"]["vector"]
    structured_measured = sum(
        1 for score in scores if score["structured_measured"])
    return {
        "stage": entry["label"],
        "round": entry["round"],
        "denominator": denominator,
        "recorded": {
            "vector": recorded_vector,
            "integer": _integer(recorded_vector),
            "hits": sum(recorded_flags),
            "rate": _rate(sum(recorded_flags), denominator),
            "stored_rate": entry["record"].get("hit_rate"),
            "stored_vector": entry["record"].get("hit_vector"),
            "matches_recomputed_keyword": recorded_vector == keyword_vector,
        },
        "metrics": metric_blocks,
        "structured_measured": structured_measured,
        "structured_missing": denominator - structured_measured,
        "rows": scores,
    }


def _transition(entries: list[dict[str, Any]],
                catalog: list[dict[str, Any]],
                metric: str = "keyword_hit") -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for before_entry, after_entry in zip(entries, entries[1:]):
        before_rows = _align_rows(before_entry["rows"], catalog)
        after_rows = _align_rows(after_entry["rows"], catalog)
        before_scores = [_row_scores(row) for row in before_rows]
        after_scores = [_row_scores(row) for row in after_rows]
        width = max(len(before_scores), len(after_scores))
        before_flags = [bool(score[metric]) for score in before_scores]
        after_flags = [bool(score[metric]) for score in after_scores]
        before_vector = _vector(before_flags).ljust(width, "0")
        after_vector = _vector(after_flags).ljust(width, "0")
        xor = _xor(before_vector, after_vector)

        probes = []
        flips = []
        for index in range(width):
            before = before_flags[index] if index < len(before_flags) else False
            after = after_flags[index] if index < len(after_flags) else False
            question = (
                after_scores[index]["question"] if index < len(after_scores)
                else before_scores[index]["question"] if index < len(before_scores)
                else "")
            expected = (
                after_scores[index]["expected"] if index < len(after_scores)
                else before_scores[index]["expected"] if index < len(before_scores)
                else [])
            status = ("same" if before == after else
                      "miss->hit" if after else "hit->miss")
            item = {
                "index": index,
                "question": question,
                "expected": expected,
                "before": "hit" if before else "miss",
                "after": "hit" if after else "miss",
                "status": status,
            }
            probes.append(item)
            if status != "same":
                flips.append(item)

        miss_to_hit = sum(1 for item in flips if item["status"] == "miss->hit")
        hit_to_miss = sum(1 for item in flips if item["status"] == "hit->miss")
        net_delta = sum(after_flags) - sum(before_flags)
        transitions.append({
            "from": before_entry["label"],
            "to": after_entry["label"],
            "from_round": before_entry["round"],
            "to_round": after_entry["round"],
            "metric": metric,
            "denominator": width,
            "from_vector": before_vector,
            "to_vector": after_vector,
            "from_integer": _integer(before_vector),
            "to_integer": _integer(after_vector),
            "from_count": sum(before_flags),
            "to_count": sum(after_flags),
            "net_delta": net_delta,
            "flip_count": len(flips),
            "miss_to_hit": miss_to_hit,
            "hit_to_miss": hit_to_miss,
            "net_delta_from_flips": miss_to_hit - hit_to_miss,
            "xor": xor,
            "xor_integer": _integer(xor),
            "hamming_distance": xor.count("1"),
            "same_count": sum(before_flags) == sum(after_flags),
            "same_vector": before_vector == after_vector,
            "same_count_but_vector_changed": (
                sum(before_flags) == sum(after_flags)
                and before_vector != after_vector),
            "flipped_questions": [item["question"] for item in flips],
            "flips": flips,
            "probes": probes,
        })
    return transitions


# ---------------------------------------------------------------------------
# Observations, defects, corrected model
# ---------------------------------------------------------------------------

def _find_transition(transitions: list[dict[str, Any]],
                     from_round: int,
                     to_round: int) -> dict[str, Any] | None:
    for transition in transitions:
        if (transition["from_round"] == from_round
                and transition["to_round"] == to_round):
            return transition
    return None


def _left_shift(vector: str) -> str:
    """Drop the first bit and append zero (the usual left bit shift)."""
    return vector[1:] + "0" if vector else ""


def _observation_count_vs_vector(transition: dict[str, Any] | None) -> dict[str, Any]:
    if transition is None:
        return {"available": False}
    return {
        "available": True,
        "id": "round_3_to_4_same_count_but_vector_changed",
        "from": transition["from"],
        "to": transition["to"],
        "hit_count_before": transition["from_count"],
        "hit_count_after": transition["to_count"],
        "denominator": transition["denominator"],
        "same_hit_count": transition["same_count"],
        "same_hit_vector": transition["same_vector"],
        "vector_changed": not transition["same_vector"],
        "hamming_distance": transition["hamming_distance"],
        "xor": transition["xor"],
        "xor_integer": transition["xor_integer"],
        "integer_before": transition["from_integer"],
        "integer_after": transition["to_integer"],
        "flipped_questions": transition["flipped_questions"],
        "flips": transition["flips"],
        "conclusion": (
            "The fractions are equal but the vectors are not. The change is "
            "exactly the listed per-probe flips, not 'no change'."
        ),
    }


def _observation_no_shift(transition: dict[str, Any] | None) -> dict[str, Any]:
    if transition is None:
        return {"available": False}
    shifted = _left_shift(transition["from_vector"])
    shifted_xor = _xor(shifted, transition["to_vector"])
    return {
        "available": True,
        "id": "round_5_to_6_not_a_left_bit_shift",
        "from": transition["from"],
        "to": transition["to"],
        "hit_count_before": transition["from_count"],
        "hit_count_after": transition["to_count"],
        "denominator": transition["denominator"],
        "net_delta": transition["net_delta"],
        "flip_count": transition["flip_count"],
        "miss_to_hit": transition["miss_to_hit"],
        "hit_to_miss": transition["hit_to_miss"],
        "arithmetic": (
            "%d miss->hit - %d hit->miss = %+d"
            % (transition["miss_to_hit"], transition["hit_to_miss"],
               transition["net_delta_from_flips"])),
        "flipped_questions": transition["flipped_questions"],
        "flips": transition["flips"],
        "xor": transition["xor"],
        "xor_integer": transition["xor_integer"],
        "changed_indices": [item["index"] for item in transition["flips"]],
        "left_shift_of_from_vector": shifted,
        "left_shift_matches_to_vector": shifted == transition["to_vector"],
        "left_shift_xor": shifted_xor,
        "left_shift_hamming_distance": shifted_xor.count("1"),
        "conclusion": (
            "The later vector is not the earlier vector shifted by one bit. "
            "It contains %d individual flips, and the net %+d is their "
            "arithmetic sum." % (transition["flip_count"],
                                 transition["net_delta"])
        ),
    }


def _defect_evidence(stages: list[dict[str, Any]]) -> dict[str, Any]:
    keyword_only = []
    structured_gap = 0
    read_only_rows = 0
    total_rows = 0
    statuses: set[str] = set()
    stage_changed = []
    missing_structured = 0
    for stage in stages:
        for row, score in zip(stage.get("_raw_rows", []), stage["rows"]):
            total_rows += 1
            if score["keyword_hit"] and not score["direct_exact_hit"]:
                keyword_only.append({
                    "stage": stage["stage"],
                    "question": score["question"],
                    "expected": score["expected"],
                    "response": score["response"],
                    "keyword_hit": True,
                    "direct_exact_hit": False,
                })
            if score["keyword_hit"] and not score["structured_hit"]:
                structured_gap += 1
            if not score["structured_measured"]:
                missing_structured += 1
            status = (row.get("native_learning_status")
                      if isinstance(row, dict) else None)
            if status:
                statuses.add(str(status))
            if (isinstance(row, dict)
                    and (row.get("probe") is True
                         or row.get("state_mutated") is False
                         or status == "probe_read_only")):
                read_only_rows += 1
        recorded = stage.get("_stage_record") or {}
        if recorded.get("probe_state_changed") is True:
            stage_changed.append(stage["stage"])

    return {
        "keyword_hits_without_direct_answer": keyword_only,
        "keyword_hits_without_structured": structured_gap,
        "total_rows": total_rows,
        "probe_read_only_rows": read_only_rows,
        "native_learning_statuses": sorted(statuses),
        "stages_with_probe_state_changed": stage_changed,
        "missing_structured_values": missing_structured,
    }


def _flip_ledger_present(entries: list[dict[str, Any]]) -> bool:
    return any(isinstance(entry["record"].get("flips_from_previous"), list)
               and bool(entry["record"].get("flips_from_previous"))
               for entry in entries)


def _fingerprints_present(entries: list[dict[str, Any]]) -> bool:
    return any(
        entry["record"].get("probe_state_digest_before") is not None
        or entry["record"].get("probe_state_digest_after") is not None
        for entry in entries)


def _defects(stages: list[dict[str, Any]],
             entries: list[dict[str, Any]],
             transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = _defect_evidence(stages)
    out = []
    for definition in DEFECTS_STATIC:
        item = dict(definition)
        item["evidence"] = {}
        out.append(item)

    by_id = {item["id"]: item for item in out}
    if evidence["keyword_hits_without_direct_answer"]:
        example = evidence["keyword_hits_without_direct_answer"][0]
        # Prefer the yellow/blue example when it is present.
        for candidate in evidence["keyword_hits_without_direct_answer"]:
            if ("yellow" in candidate["question"].casefold()
                    and "blue" in candidate["question"].casefold()):
                example = candidate
                break
        by_id["keyword_substring_anywhere"]["evidence"] = {
            "stage": example["stage"],
            "question": example["question"],
            "expected": example["expected"],
            "response": example["response"],
            "keyword_hit": True,
            "direct_exact_hit": False,
            "why": (
                "The expected term is in an unrelated clause (for the yellow/"
                "blue case, `green` belongs to a red/blue clause after "
                "`comes from`), so substring containment over-credits it."
                if "yellow" in example["question"].casefold()
                else "The expected term is incidental, rambling, or outside "
                     "the direct answer span."
            ),
        }
    by_id["keyword_substring_anywhere"]["evidence"].update({
        "keyword_hits": sum(
            sum(1 for score in stage["rows"] if score["keyword_hit"])
            for stage in stages),
        "direct_exact_hits": sum(
            sum(1 for score in stage["rows"] if score["direct_exact_hit"])
            for stage in stages),
    })

    by_id["no_direct_or_structured_requirement"]["evidence"] = {
        "keyword_hits_without_structured": evidence[
            "keyword_hits_without_structured"],
        "total_rows": evidence["total_rows"],
        "structured_measured_rows": (
            evidence["total_rows"] - evidence["missing_structured_values"]),
        "structured_missing_rows": evidence["missing_structured_values"],
    }

    by_id["probe_state_mutation"]["evidence"] = {
        "probe_read_only_rows": evidence["probe_read_only_rows"],
        "total_rows": evidence["total_rows"],
        "native_learning_statuses": evidence["native_learning_statuses"],
        "fingerprints_present": _fingerprints_present(entries),
        "flip_ledger_present": _flip_ledger_present(entries),
        "stages_with_probe_state_changed": evidence[
            "stages_with_probe_state_changed"],
        "old_artifact_inference": (
            "No row carries a probe/read-only marker and no stage carries a "
            "before/after state digest, so the artifact cannot prove probes "
            "were read-only. The old script called nanlm_chat without "
            "probe=True, so later rounds may include earlier probe text as "
            "training data."
            if evidence["probe_read_only_rows"] == 0
            else "Rows carry a read-only marker; see state-change flags."
        ),
    }
    if evidence["probe_read_only_rows"] == 0:
        by_id["probe_state_mutation"]["summary"] = (
            "The direct-run artifact did not pass `probe=True`. Each evaluation "
            "question went through the normal learning path, so a probe could "
            "teach the student the question/reply and later rounds were not "
            "held out. The current script passes `probe=True`; the old artifact "
            "remains confounded."
        )
    elif evidence["stages_with_probe_state_changed"]:
        by_id["probe_state_mutation"]["summary"] = (
            "Rows are marked read-only, but the stored stage fingerprints "
            "report a state change in %s. That does not prove probe learning, "
            "but it means the read-only claim is not demonstrated by the "
            "fingerprint alone." % ", ".join(
                evidence["stages_with_probe_state_changed"])
        )
    else:
        by_id["probe_state_mutation"]["summary"] = (
            "Rows are marked read-only and no stage reports a state change. "
            "This supports the `probe=True` fix for the recorded run; the old "
            "direct-run artifact remains confounded."
        )

    by_id["no_per_probe_state_fingerprint_or_flip_ledger"]["evidence"] = {
        "fingerprints_present": _fingerprints_present(entries),
        "flip_ledger_present": _flip_ledger_present(entries),
        "note": (
            "Stage-level before/after digests identify a changed stage but not "
            "the probe that changed it. The audit recomputes flips from stored "
            "hit vectors, which is weaker than a state fingerprint per probe."
        ),
    }
    if (_fingerprints_present(entries) and _flip_ledger_present(entries)):
        by_id["no_per_probe_state_fingerprint_or_flip_ledger"]["summary"] = (
            "The artifact has a stage-level before/after digest and a derived "
            "per-round flip ledger, but still no per-probe state fingerprint. "
            "A changed stage cannot be attributed to a specific probe."
        )

    by_id["implicit_denominator_and_missing_structured_values"]["evidence"] = {
        "stored_hit_rates_only": all(
            entry["record"].get("hit_rate") is not None
            and entry["record"].get("hit_vector") is None
            for entry in entries),
        "stored_vectors_present": all(
            entry["record"].get("hit_vector") is not None for entry in entries),
        "structured_missing_rows": evidence["missing_structured_values"],
        "denominator": stages[0]["denominator"] if stages else 0,
    }
    if any(entry["record"].get("hit_vector") is not None for entry in entries):
        by_id["implicit_denominator_and_missing_structured_values"]["summary"] = (
            "The artifact stores hit vectors but no explicit denominator; the "
            "audit infers the denominator from the probe catalog. Structured "
            "values that are null are still missing measurement, not measured "
            "misses."
        )
    return out


def _corrected_model(stages: list[dict[str, Any]],
                     entries: list[dict[str, Any]],
                     catalog: list[dict[str, Any]]) -> dict[str, Any]:
    stage_rows = []
    for stage in stages:
        stage_rows.append({
            "stage": stage["stage"],
            "round": stage["round"],
            "denominator": stage["denominator"],
            "structured_measured": stage["structured_measured"],
            "structured_missing": stage["structured_missing"],
            "metrics": stage["metrics"],
        })
    metric_transitions = {}
    for metric in METRICS:
        metric_transitions[metric] = _transition(entries, catalog, metric=metric)
    return {
        "definitions": DEFINITIONS,
        "stages": stage_rows,
        "transitions": metric_transitions,
        "headline": {
            "keyword_hit_final": (
                stage_rows[-1]["metrics"]["keyword_hit"]
                if stage_rows else None),
            "structured_hit_final": (
                stage_rows[-1]["metrics"]["structured_hit"]
                if stage_rows else None),
            "direct_exact_hit_final": (
                stage_rows[-1]["metrics"]["direct_exact_hit"]
                if stage_rows else None),
            "note": (
                "All three columns are offline re-scores of the same stored "
                "replies. No live re-run or new model state is produced."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def build_report(record: dict[str, Any]) -> dict[str, Any]:
    entries = _stage_entries(record)
    catalog = _catalog(record, entries)
    stages = []
    for entry in entries:
        stage = _scored_stage(entry, catalog)
        stage["_raw_rows"] = entry["rows"]
        stage["_stage_record"] = entry["record"]
        stages.append(stage)
    transitions = _transition(entries, catalog, metric="keyword_hit")
    obs_3_4 = _observation_count_vs_vector(_find_transition(transitions, 3, 4))
    obs_5_6 = _observation_no_shift(_find_transition(transitions, 5, 6))
    corrected = _corrected_model(stages, entries, catalog)
    report = {
        "schema": REPORT_SCHEMA,
        "input": record.get("_audit_input") or {},
        "run": {
            "schema": record.get("schema"),
            "model": record.get("model"),
            "state_dir": record.get("state_dir"),
            "rounds_requested": record.get("rounds_requested"),
            "stage_count": len(stages),
            "probe_count": len(catalog),
            "state_dir_exists": Path(str(record.get("state_dir") or "")).is_dir()
            if record.get("state_dir") else False,
        },
        "stages": [
            {key: value for key, value in stage.items()
             if not key.startswith("_")}
            for stage in stages
        ],
        "transitions": transitions,
        "observations": {
            "round_3_to_4": obs_3_4,
            "round_5_to_6": obs_5_6,
        },
        "measurement_defects": _defects(stages, entries, transitions),
        "corrected_model": corrected,
        "boundaries": {
            "read_only": True,
            "new_live_run": False,
            "modifies_run_json": False,
            "modifies_state_dir": False,
            "proof_claim": False,
            "note": (
                "This audit only re-scores the stored artifact. It does not "
                "retrain, promote relations, or turn an offline re-score into "
                "a live accuracy claim."
            ),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _stage_table_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "| stage | legacy vector | int | hits | denominator | rate | stored rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        recorded = stage["recorded"]
        lines.append(
            "| %s | `%s` | %d | %d | %d | %.4f | %s |" % (
                stage["stage"], recorded["vector"], recorded["integer"],
                recorded["hits"], stage["denominator"], recorded["rate"],
                ("%.4f" % recorded["stored_rate"])
                if isinstance(recorded["stored_rate"], (int, float)) else "n/a"))
    return lines


def _transition_lines(report: dict[str, Any]) -> list[str]:
    lines = []
    for transition in report["transitions"]:
        flipped = ", ".join(
            "%s (%s->%s)" % (item["question"], item["before"], item["after"])
            for item in transition["flips"]) or "none"
        same = ", ".join(
            item["question"] for item in transition["probes"]
            if item["status"] == "same")
        lines.append(
            "- %s -> %s: %+d (%d/%d -> %d/%d); flips: %s; xor=`%s`" % (
                transition["from"], transition["to"],
                transition["net_delta"], transition["from_count"],
                transition["denominator"], transition["to_count"],
                transition["denominator"], flipped, transition["xor"]))
        if same:
            lines.append("  - same: %s" % same)
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    input_info = report["input"]
    stages = report["stages"]
    obs_3_4 = report["observations"]["round_3_to_4"]
    obs_5_6 = report["observations"]["round_5_to_6"]
    corrected = report["corrected_model"]
    lines = [
        "# NaNLM accuracy audit",
        "",
        "- input: `%s`" % input_info.get("path", "unknown"),
        "- sha256: `%s`" % input_info.get("sha256", "unknown"),
        "- schema: `%s`" % (run.get("schema") or "unknown"),
        "- state dir: `%s`" % (run.get("state_dir") or "unknown"),
        "- stages: %d, probes: %d" % (run["stage_count"], run["probe_count"]),
        "",
        "This is a read-only re-score of an existing artifact. It is not a new "
        "training run and does not claim a new live accuracy number.",
        "",
        "## Recomputed legacy hit vectors",
        "",
    ]
    lines.extend(_stage_table_lines(report))
    lines.extend([
        "",
        "## Per-stage transitions",
        "",
    ])
    lines.extend(_transition_lines(report))
    lines.extend([
        "",
        "## Observation 1: rounds 3 -> 4",
        "",
    ])
    if obs_3_4.get("available"):
        lines.extend([
            "- hit count: %d/%d -> %d/%d (same count: %s)" % (
                obs_3_4["hit_count_before"], obs_3_4["denominator"],
                obs_3_4["hit_count_after"], obs_3_4["denominator"],
                obs_3_4["same_hit_count"]),
            "- vectors: `%s` -> `%s`; same vector: %s" % (
                _find_transition(report["transitions"], 3, 4)["from_vector"],
                _find_transition(report["transitions"], 3, 4)["to_vector"],
                obs_3_4["same_hit_vector"]),
            "- xor: `%s` (integer %d), Hamming distance %d" % (
                obs_3_4["xor"], obs_3_4["xor_integer"],
                obs_3_4["hamming_distance"]),
            "- exact flips:",
        ])
        for item in obs_3_4["flips"]:
            lines.append(
                "  - `%s`: %s -> %s" % (
                    item["question"], item["before"], item["after"]))
        lines.append("- conclusion: %s" % obs_3_4["conclusion"])
    else:
        lines.append("- not available in this artifact (rounds 3 and 4 missing)")
    lines.extend([
        "",
        "## Observation 2: rounds 5 -> 6",
        "",
    ])
    if obs_5_6.get("available"):
        lines.extend([
            "- hit count: %d/%d -> %d/%d; net %+d; flips %d (%d miss->hit, "
            "%d hit->miss)" % (
                obs_5_6["hit_count_before"], obs_5_6["denominator"],
                obs_5_6["hit_count_after"], obs_5_6["denominator"],
                obs_5_6["net_delta"], obs_5_6["flip_count"],
                obs_5_6["miss_to_hit"], obs_5_6["hit_to_miss"]),
            "- arithmetic: %s" % obs_5_6["arithmetic"],
            "- xor: `%s` (integer %d); changed positions: %s" % (
                obs_5_6["xor"], obs_5_6["xor_integer"],
                obs_5_6["changed_indices"]),
            "- exact flips:",
        ])
        for item in obs_5_6["flips"]:
            lines.append("  - `%s`: %s -> %s" % (
                item["question"], item["before"], item["after"]))
        lines.extend([
            "- left shift of the earlier vector: `%s`" % (
                obs_5_6["left_shift_of_from_vector"]),
            "- shifted vector equals later vector: %s; distance to shifted "
            "vector: %d" % (
                obs_5_6["left_shift_matches_to_vector"],
                obs_5_6["left_shift_hamming_distance"]),
            "- conclusion: %s" % obs_5_6["conclusion"],
        ])
    else:
        lines.append("- not available in this artifact (rounds 5 and 6 missing)")
    lines.extend([
        "",
        "## Measurement defects",
        "",
    ])
    for defect in report["measurement_defects"]:
        lines.append("### `%s`" % defect["id"])
        lines.append("")
        lines.append(defect["summary"])
        if defect["evidence"]:
            lines.append("")
            lines.append("Evidence: `%s`" % json.dumps(
                defect["evidence"], sort_keys=True, ensure_ascii=True))
        lines.append("")
    lines.extend([
        "## Corrected measurement model",
        "",
    ])
    for metric in METRICS:
        lines.append("- `%s`: %s" % (metric, DEFINITIONS[metric]))
    lines.extend([
        "",
        "| stage | keyword vector | keyword | structured | direct exact | "
        "denominator |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for stage in corrected["stages"]:
        metrics = stage["metrics"]
        lines.append(
            "| %s | `%s` | %d | %d | %d | %d |" % (
                stage["stage"], metrics["keyword_hit"]["vector"],
                metrics["keyword_hit"]["hits"],
                metrics["structured_hit"]["hits"],
                metrics["direct_exact_hit"]["hits"],
                stage["denominator"]))
    lines.extend([
        "",
        "Structured values measured: %d/%d at the final stage; missing: %d. "
        "A missing structured flag is not a measured miss." % (
            corrected["stages"][-1]["structured_measured"]
            if corrected["stages"] else 0,
            corrected["stages"][-1]["denominator"]
            if corrected["stages"] else 0,
            corrected["stages"][-1]["structured_missing"]
            if corrected["stages"] else 0),
        "",
        "The corrected direct-answer vector is the strict readout. It is still "
        "an offline heuristic over stored text, so it does not prove that the "
        "student can answer the probes in a new run.",
        "",
        "## Boundaries",
        "",
        "- Read-only re-score; no training, state mutation, or relation promotion.",
        "- No new live result is invented; only the stored artifact is audited.",
        "- `structured` nulls are reported as missing measurement, not as 0%.",
        "- The old artifact's probe calls were not read-only; later rounds may "
        "include earlier probe text in the learned state.",
        "",
    ])
    return "\n".join(lines)


def _bounded(lines: list[str], max_lines: int = MAX_SUMMARY_LINES,
             max_chars: int = MAX_SUMMARY_CHARS) -> str:
    text = "\n".join(lines)
    clipped = "\n".join(text.splitlines()[:max_lines])
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rsplit("\n", 1)[0]
    if clipped != text:
        clipped += "\n[summary truncated; use --json/--markdown for full detail]"
    return clipped


def summary_lines(report: dict[str, Any]) -> list[str]:
    run = report["run"]
    input_info = report["input"]
    lines = [
        "NaNLM accuracy audit (read-only re-score)",
        "input: %s" % input_info.get("path", "unknown"),
        "sha256: %s" % input_info.get("sha256", "unknown"),
        "schema: %s | stages: %d | probes: %d | denominator: %s" % (
            run.get("schema"), run["stage_count"], run["probe_count"],
            report["stages"][0]["denominator"] if report["stages"] else "n/a"),
        "",
        "recomputed legacy hit vectors:",
    ]
    for stage in report["stages"]:
        recorded = stage["recorded"]
        lines.append(
            "  %-9s %s = %d/%d (%.4f, int %d)%s" % (
                stage["stage"], recorded["vector"], recorded["hits"],
                stage["denominator"], recorded["rate"], recorded["integer"],
                "" if recorded["matches_recomputed_keyword"] else " [MISMATCH]"))
    lines.append("")
    lines.append("transitions (keyword hit recomputation):")
    for transition in report["transitions"]:
        details = []
        for item in transition["flips"]:
            details.append("%s:%s->%s" % (
                item["question"], item["before"], item["after"]))
        lines.append(
            "  %-9s -> %-9s net %+d, %d flips%s" % (
                transition["from"], transition["to"],
                transition["net_delta"], transition["flip_count"],
                (": " + "; ".join(details)) if details else ""))
    lines.append("")
    obs_3_4 = report["observations"]["round_3_to_4"]
    if obs_3_4.get("available"):
        lines.append(
            "observation 3->4: same count=%s but same vector=%s; xor=%s; "
            "hamming=%d" % (
                obs_3_4["same_hit_count"], obs_3_4["same_hit_vector"],
                obs_3_4["xor"], obs_3_4["hamming_distance"]))
        lines.append("  exact flips: " + "; ".join(
            "%s %s->%s" % (item["question"], item["before"], item["after"])
            for item in obs_3_4["flips"]))
    obs_5_6 = report["observations"]["round_5_to_6"]
    if obs_5_6.get("available"):
        lines.append(
            "observation 5->6: not a left shift=%s; %d flips (%d miss->hit, "
            "%d hit->miss), net %+d; xor=%s" % (
                not obs_5_6["left_shift_matches_to_vector"],
                obs_5_6["flip_count"], obs_5_6["miss_to_hit"],
                obs_5_6["hit_to_miss"], obs_5_6["net_delta"],
                obs_5_6["xor"]))
        lines.append("  exact flips: " + "; ".join(
            "%s %s->%s" % (item["question"], item["before"], item["after"])
            for item in obs_5_6["flips"]))
    lines.append("")
    lines.append("corrected model (offline re-score):")
    for stage in report["corrected_model"]["stages"]:
        metrics = stage["metrics"]
        lines.append(
            "  %-9s keyword %d/%d %s | structured %d/%d %s | direct %d/%d %s" % (
                stage["stage"],
                metrics["keyword_hit"]["hits"], stage["denominator"],
                metrics["keyword_hit"]["vector"],
                metrics["structured_hit"]["hits"], stage["denominator"],
                metrics["structured_hit"]["vector"],
                metrics["direct_exact_hit"]["hits"], stage["denominator"],
                metrics["direct_exact_hit"]["vector"]))
    lines.append("")
    lines.append("measurement defects:")
    for defect in report["measurement_defects"]:
        lines.append("  - %s (%s): %s" % (
            defect["id"], defect["severity"], defect["summary"]))
    lines.append("")
    lines.append("boundaries: read-only; no training run; no new live result")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_markdown(report) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(DEFAULT_RUN_JSON),
        help="run JSON to audit (default: docs/NANLM_GEMMA_LEARNING_OVER_TIME.json)")
    parser.add_argument("--json", default="", metavar="PATH",
                        help="write the full audit report as JSON")
    parser.add_argument("--markdown", default="", metavar="PATH",
                        help="write a Markdown audit report")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the bounded stdout summary")
    args = parser.parse_args(argv)

    record = load_run(args.input)
    report = build_report(record)
    if args.json:
        _write_json(Path(args.json).expanduser(), report)
    if args.markdown:
        _write_markdown(Path(args.markdown).expanduser(), report)
    if not args.quiet:
        print(_bounded(summary_lines(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
