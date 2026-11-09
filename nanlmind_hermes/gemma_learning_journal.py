"""Bounded cross-session learning journal for the Gemma training series.

The journal is deliberately small and standard-library only.  It records what a
session actually did (exact tool names, call outcomes and a handful of measured
fields from tool results) so that the next session, or an after-series quiz, can
receive observed experience instead of an empty context.  It is *not* a model
weight update, a canonical memory, or a proof of learning.

Storage is one JSONL file under a persistent state directory.  Every entry is
bounded, secret-bearing keys and common token shapes are redacted, and only the
newest ``max_entries`` sessions are retained.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

JOURNAL_SCHEMA = "nanlm.gemma-learning-journal/1"
ENTRY_SCHEMA = "nanlm.gemma-learning-journal-entry/1"
DISCOVERY_SCORE_SCHEMA = "nanlm.gemma-discovery-quiz-score/1"
DEFAULT_FILENAME = "gemma_learning_journal.jsonl"

MAX_ENTRIES = 64
MAX_LINE_CHARS = 12_000
MAX_TOOL_NAMES = 32
MAX_TOOL_STATS = 24
MAX_FAILURES = 8
MAX_PROBLEMS = 4
MAX_OBSERVATIONS = 28
MAX_TEXT_CHARS = 240
MAX_BLOCK_CHARS = 4_000
DEFAULT_EXPERIENCE_SESSIONS = 3

# Only these fields are extracted automatically from tool results.  The list is
# intentionally short: enough to answer "what did the run actually observe?"
# without copying kernel catalogs, transcripts, payloads or scratch records.
OBSERVATION_KEYS = (
    "execution_mode",
    "n16i_used",
    "execution_modes",
    "leaf_execution_mode",
    "leaf_n16i_used",
    "linked_kernels",
    "expected_kernels",
    "tree_complete",
    "images_verified",
    "image_failures",
    "nan_executable",
    "marker_found",
    "kernel_id",
    "marker",
    "leaves",
    "leaf_count",
    "turns",
    "rows",
    "scratch_records",
    "scratch_pages_allocated",
    "concept_statements",
    "concept_concepts",
    "concept_occupied_pairs",
    "concept_density",
    "memory_records",
)
_OBSERVATION_KEY_SET = frozenset(OBSERVATION_KEYS)

_EXEC_TOOLS = frozenset({
    "nanlm_mth_verify",
    "nanlm_mth_execute",
    "nanlm_mth_dispatch",
    "nanlm_mth_n16i_selftest",
})
_LEAF_TOOLS = frozenset({
    "nanlm_mth_leaf_create",
    "nanlm_mth_leaf_get",
    "nanlm_mth_leaf_verify",
    "nanlm_mth_pair",
})
_LEARNING_TOOLS = frozenset({
    "nanlm_learning_status",
    "nanlm_learning_cycle",
    "nanlm_chat",
})

_SECRET_KEY_MARKERS = (
    "secret",
    "token",
    "password",
    "passwd",
    "private_key",
    "private-key",
    "master_key",
    "master-key",
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "cookie",
    "receipt_mac",
    "receipt-mac",
    "pairing_code",
    "pairing-code",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ),
)
_TRUNCATED = "...[truncated]"
_REDACTED = "[redacted]"


class JournalError(ValueError):
    """Raised for an invalid journal path, entry or bound."""


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _secret_key(name: Any) -> bool:
    lowered = str(name).strip().casefold()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def redact_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Return a bounded string with common secrets redacted.

    Large payloads are truncated rather than journaled.  Digest-shaped values
    are deliberately not redacted: fixture and image digests are evidence.
    """
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\x00", " ")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    if len(text) > limit:
        text = text[:limit] + _TRUNCATED
    return text


def sanitize_value(value: Any, *, depth: int = 0, text_limit: int = MAX_TEXT_CHARS):
    """Bound and redact an arbitrary JSON-like value for journal storage."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        return redact_text(value, text_limit)
    if depth >= 4:
        return "[depth-limit]"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        bounded = [
            sanitize_value(item, depth=depth + 1, text_limit=text_limit)
            for item in items[:12]
        ]
        if len(items) > 12:
            bounded.append("[+%d more]" % (len(items) - 12))
        return bounded
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 24:
                result["_more"] = "[+%d more]" % (len(value) - 24)
                break
            name = redact_text(key, 64)
            if _secret_key(name):
                result[name] = _REDACTED
            else:
                result[name] = sanitize_value(
                    item, depth=depth + 1, text_limit=text_limit)
        return result
    return redact_text(value, text_limit)


def _parse_json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _scalar_from_text(text: str, key: str):
    """Best-effort scalar recovery from a truncated JSON result string."""
    if not isinstance(text, str):
        return None
    pattern = re.compile(
        r'"' + re.escape(key) + r'"\s*:\s*'
        r'("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
    )
    match = pattern.search(text)
    if match is None:
        return None
    raw = match.group(1)
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    if raw.startswith('"'):
        try:
            return json.loads(raw)
        except ValueError:
            return raw[1:-1]
    number = _as_float(raw)
    if number is None:
        return None
    return int(number) if number.is_integer() else number


def _nested_mode_from_text(text: str):
    if not isinstance(text, str):
        return None
    match = re.search(
        r'"execution"\s*:\s*\{[^{}]*?"mode"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _call_result(call: dict) -> tuple[dict, str]:
    """Return a parsed result mapping and the raw text, if available."""
    result = call.get("result")
    if isinstance(result, dict):
        return result, ""
    text = call.get("result_summary")
    if isinstance(text, str):
        parsed = _parse_json_object(text)
        return parsed, text
    if isinstance(result, str):
        parsed = _parse_json_object(result)
        return parsed, result
    return {}, ""


def _nested_execution(result: dict) -> dict:
    """Find an execution description nested in leaf/function results."""
    candidates = []
    execution = result.get("execution")
    if isinstance(execution, dict):
        candidates.append(execution)
    function_definition = result.get("function_definition")
    if isinstance(function_definition, dict):
        execution = function_definition.get("execution")
        if isinstance(execution, dict):
            candidates.append(execution)
    leaf = result.get("leaf")
    if isinstance(leaf, dict):
        execution = leaf.get("execution")
        if isinstance(execution, dict):
            candidates.append(execution)
        function_definition = leaf.get("function_definition")
        if isinstance(function_definition, dict):
            execution = function_definition.get("execution")
            if isinstance(execution, dict):
                candidates.append(execution)
    return candidates[0] if candidates else {}


def _put(observations: dict, key: str, value: Any) -> None:
    if key not in _OBSERVATION_KEY_SET or value is None:
        return
    if isinstance(value, bool) or isinstance(value, (int, float)):
        observations[key] = value
    elif isinstance(value, str):
        observations[key] = redact_text(value, 120)
    elif isinstance(value, (list, tuple)):
        values = [redact_text(item, 80) for item in list(value)[:8]
                  if isinstance(item, (str, int, float, bool))]
        if values:
            observations[key] = values


def extract_observations(calls: Any) -> dict:
    """Extract a bounded set of observed fields from session tool calls.

    ``calls`` may come directly from ``gemma_training_series`` records.  Full
    ``result`` mappings are preferred; truncated ``result_summary`` strings are
    handled with a scalar fallback so a bounded journal still works on saved
    results.
    """
    observations: dict = {}
    execution_modes = set()
    for call in calls if isinstance(calls, (list, tuple)) else ():
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        result, text = _call_result(call)

        if name in _EXEC_TOOLS:
            mode = result.get("execution_mode")
            if not isinstance(mode, str):
                mode = _nested_execution(result).get("mode")
            if not isinstance(mode, str):
                mode = _scalar_from_text(text, "execution_mode")
            if isinstance(mode, str) and mode:
                execution_modes.add(mode)
                _put(observations, "execution_mode", mode)
            n16i = result.get("n16i_used")
            if not isinstance(n16i, bool):
                n16i = _nested_execution(result).get("n16i_used")
            if not isinstance(n16i, bool):
                n16i = _scalar_from_text(text, "n16i_used")
            _put(observations, "n16i_used", n16i if isinstance(n16i, bool)
                 else None)
            for key in ("marker_found", "kernel_id", "marker"):
                value = result.get(key)
                if value is None:
                    value = _scalar_from_text(text, key)
                _put(observations, key, value)

        if name in _LEAF_TOOLS:
            execution = _nested_execution(result)
            mode = execution.get("mode")
            if not isinstance(mode, str):
                mode = _scalar_from_text(text, "execution_mode")
            if not isinstance(mode, str):
                mode = _nested_mode_from_text(text)
            if isinstance(mode, str) and mode:
                execution_modes.add(mode)
                _put(observations, "leaf_execution_mode", mode)
            n16i = execution.get("n16i_used")
            if not isinstance(n16i, bool):
                n16i = _scalar_from_text(text, "n16i_used")
            _put(observations, "leaf_n16i_used",
                 n16i if isinstance(n16i, bool) else None)
            for key in ("nan_executable", "marker_found", "leaf_id"):
                value = result.get(key)
                if value is None:
                    value = _scalar_from_text(text, key)
                if key == "leaf_id":
                    continue
                _put(observations, key, value)

        if name == "nanlm_mth_tree" or (
                "linked" in result and "expected" in result
                and "images_verified" in result):
            for key in ("linked", "expected", "images_verified",
                        "complete", "image_failures"):
                value = result.get(key)
                if value is None:
                    value = _scalar_from_text(text, key)
                if key in ("linked", "expected"):
                    _put(observations, key + "_kernels", value)
                elif key == "complete":
                    _put(observations, "tree_complete", value)
                elif key == "image_failures":
                    if isinstance(value, list):
                        value = len(value)
                    _put(observations, "image_failures", value)
                else:
                    _put(observations, key, value)

        if name == "nanlm_mth_hub":
            leaves = result.get("leaves")
            if leaves is None:
                leaves = result.get("leaf_count")
            if leaves is None:
                leaves = _scalar_from_text(text, "leaves")
            _put(observations, "leaves", leaves)
            _put(observations, "leaf_count", leaves)

        if name in _LEARNING_TOOLS:
            learning = result.get("learning")
            learning = learning if isinstance(learning, dict) else result
            for key in ("turns", "rows"):
                value = learning.get(key)
                if value is None:
                    value = _scalar_from_text(text, key)
                _put(observations, key, value)
            scratch = learning.get("scratch")
            if isinstance(scratch, dict):
                _put(observations, "scratch_records", scratch.get("records"))
                _put(observations, "scratch_pages_allocated",
                     scratch.get("allocated_pages"))
            else:
                _put(observations, "scratch_records",
                     _scalar_from_text(text, "records"))
            saturation = learning.get("concept_saturation")
            if isinstance(saturation, dict):
                _put(observations, "concept_statements",
                     saturation.get("statements"))
                _put(observations, "concept_concepts",
                     saturation.get("concepts"))
                _put(observations, "concept_occupied_pairs",
                     saturation.get("occupied_pairs"))
                _put(observations, "concept_density",
                     saturation.get("density"))
            else:
                for key, stored in (
                        ("statements", "concept_statements"),
                        ("concepts", "concept_concepts"),
                        ("occupied_pairs", "concept_occupied_pairs"),
                        ("density", "concept_density")):
                    _put(observations, stored, _scalar_from_text(text, key))

    if execution_modes:
        _put(observations, "execution_modes", sorted(execution_modes))
    return observations


def _sanitize_observations(observations: Any) -> dict:
    if not isinstance(observations, dict):
        return {}
    result = {}
    for key in OBSERVATION_KEYS:
        if key in observations:
            value = observations[key]
            if value is None:
                continue
            if key in {"execution_modes", "image_failures"} and isinstance(
                    value, (list, tuple, set)):
                value = sorted({redact_text(item, 80) for item in value})[:8]
            elif isinstance(value, (dict, list, tuple, set)):
                value = sanitize_value(value, text_limit=120)
            elif isinstance(value, str):
                value = redact_text(value, 120)
            elif isinstance(value, float):
                value = _as_float(value)
                if value is None:
                    continue
            result[key] = value
    # Preserve a bounded number of caller-supplied extra fields.  Known fields
    # always come first so the small observed set remains stable.
    for key, value in observations.items():
        if key in result or key in _OBSERVATION_KEY_SET:
            continue
        if len(result) >= MAX_OBSERVATIONS:
            break
        name = redact_text(key, 48)
        if _secret_key(name):
            result[name] = _REDACTED
        else:
            result[name] = sanitize_value(value, text_limit=120)
    return result


def _tool_summary(calls: list) -> dict:
    stats: dict[str, dict] = {}
    failures = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = redact_text(call.get("name") or "unknown", 96)
        row = stats.setdefault(name, {"name": name, "calls": 0, "failures": 0})
        row["calls"] += 1
        if call.get("failed"):
            row["failures"] += 1
            if len(failures) < MAX_FAILURES:
                problems = [
                    redact_text(problem, 120)
                    for problem in list(call.get("argument_problems") or [])[
                        :MAX_PROBLEMS]
                ]
                failures.append({
                    "tool": name,
                    "turn": _as_int(call.get("turn")),
                    "problems": problems,
                })
    ordered = sorted(stats.values(),
                     key=lambda row: (-row["calls"], row["name"]))
    tools_used = sorted(stats)[:MAX_TOOL_NAMES]
    return {
        "tools_used": tools_used,
        "tool_stats": ordered[:MAX_TOOL_STATS],
        "failure_details": failures,
    }


def _fit_entry(entry: dict) -> dict:
    """Ensure a serialized entry stays inside the journal line bound."""
    def encoded(value: dict) -> int:
        return len(json.dumps(value, sort_keys=True, ensure_ascii=False,
                              default=str))

    if encoded(entry) <= MAX_LINE_CHARS:
        return entry
    observations = entry.get("observations")
    if isinstance(observations, dict):
        while observations and encoded(entry) > MAX_LINE_CHARS:
            observations.pop(next(reversed(observations)))
    if encoded(entry) > MAX_LINE_CHARS:
        details = entry.get("failure_details")
        if isinstance(details, list):
            while details and encoded(entry) > MAX_LINE_CHARS:
                details.pop()
    if encoded(entry) > MAX_LINE_CHARS:
        stats = entry.get("tool_stats")
        if isinstance(stats, list):
            while stats and encoded(entry) > MAX_LINE_CHARS:
                stats.pop()
    if encoded(entry) > MAX_LINE_CHARS:
        entry["tools_used"] = entry.get("tools_used", [])[:8]
        entry["observations"] = {}
        entry["failure_details"] = []
        entry["tool_stats"] = []
    if encoded(entry) > MAX_LINE_CHARS:
        raise JournalError("journal entry exceeds the bounded line size")
    return entry


class SessionJournal:
    """A bounded JSONL journal stored under a persistent state directory."""

    def __init__(self, state_dir: str | os.PathLike, *,
                 filename: str = DEFAULT_FILENAME,
                 max_entries: int = MAX_ENTRIES):
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise JournalError("max_entries must be an integer")
        if not 1 <= max_entries <= 4096:
            raise JournalError("max_entries must be between 1 and 4096")
        name = str(filename or "").strip()
        if not name or "/" in name or os.sep in name:
            raise JournalError("filename must be a plain file name")
        self.root = Path(state_dir).expanduser()
        self.state_dir = self.root
        self.filename = name
        self.path = self.root / name
        self.journal_path = self.path
        self.max_entries = max_entries
        self._lock = threading.RLock()

    def _read_locked(self) -> list:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        entries = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and "session_index" in value:
                entries.append(value)
        return entries

    def _write_locked(self, entries: list) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lines = [
            json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
            for entry in entries
        ]
        payload = "\n".join(lines) + ("\n" if lines else "")
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, self.path)

    def append_session(self, session_index: int, stage: int, calls: Any,
                       metrics: Any = None, observations: Any = None) -> dict:
        """Append one bounded session summary and return the stored entry."""
        index = _as_int(session_index, -1)
        stage_number = _as_int(stage, -1)
        if index < 1:
            raise JournalError("session_index must be a positive integer")
        if stage_number < 0:
            raise JournalError("stage must be a non-negative integer")
        calls = list(calls) if isinstance(calls, (list, tuple)) else []
        metrics = metrics if isinstance(metrics, dict) else {}
        if isinstance(metrics.get("metrics"), dict):
            metrics = metrics["metrics"]
        supplied = _sanitize_observations(observations)
        extracted = extract_observations(calls)
        merged = dict(extracted)
        merged.update(supplied)
        ordered = {key: merged[key] for key in OBSERVATION_KEYS
                   if key in merged}
        for key in sorted(set(merged) - set(OBSERVATION_KEYS)):
            ordered[key] = merged[key]
        merged = ordered

        summary = _tool_summary(calls)
        failed_from_calls = sum(1 for call in calls
                                if isinstance(call, dict) and call.get("failed"))
        invalid_from_calls = sum(
            1 for call in calls
            if isinstance(call, dict) and call.get("invalid_arguments"))
        total = _as_int(metrics.get("tool_calls"), len(calls))
        successful = _as_int(metrics.get("successful_calls"),
                             max(0, len(calls) - failed_from_calls))
        failed = _as_int(metrics.get("failed_calls"), failed_from_calls)
        invalid = _as_int(metrics.get("invalid_argument_calls"),
                          invalid_from_calls)
        entry = {
            "schema": ENTRY_SCHEMA,
            "session_index": index,
            "stage": stage_number,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool_calls": total,
            "successful_calls": successful,
            "successes": successful,
            "failed_calls": failed,
            "failures": failed,
            "invalid_argument_calls": invalid,
            "invalid_arguments": invalid,
            "turns_used": _as_int(metrics.get("turns_used")),
            "tools_used": summary["tools_used"],
            "tools": summary["tools_used"],
            "tool_stats": summary["tool_stats"],
            "failure_details": summary["failure_details"],
            "observations": _sanitize_observations(merged),
        }
        entry = _fit_entry(entry)
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False,
                          default=str)
        if len(line) > MAX_LINE_CHARS:
            raise JournalError("journal entry exceeds the bounded line size")

        with self._lock:
            entries = self._read_locked()
            entries.append(entry)
            if len(entries) > self.max_entries:
                entries = entries[-self.max_entries:]
                self._write_locked(entries)
            else:
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(
                    self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(descriptor, (line + "\n").encode("utf-8"))
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        return entry

    def load(self) -> list:
        """Return all retained entries in append order."""
        with self._lock:
            entries = self._read_locked()
        return entries[-self.max_entries:]

    def tail(self, limit: int = DEFAULT_EXPERIENCE_SESSIONS) -> list:
        """Return the newest ``limit`` entries, bounded by ``max_entries``."""
        count = _as_int(limit, 0)
        if count <= 0:
            return []
        count = min(count, self.max_entries)
        return self.load()[-count:]

    def build_experience_block(
            self, limit: int = DEFAULT_EXPERIENCE_SESSIONS) -> str:
        """Return a compact prompt block of exact tools and observed results."""
        entries = self.tail(limit)
        if not entries:
            return ""
        lines = [
            "Measured HERMES session journal (actual tool use, not documentation):",
        ]
        for entry in entries:
            tools = ", ".join(entry.get("tools_used") or []) or "(none)"
            observations = entry.get("observations") or {}
            observed = ", ".join(
                "%s=%s" % (key, _format_value(observations[key]))
                for key in OBSERVATION_KEYS if key in observations
            )
            line = "- session %s (stage %s): tools=%s; calls=%s ok=%s "
            line += "failed=%s invalid_args=%s"
            line = line % (
                entry.get("session_index"), entry.get("stage"), tools,
                entry.get("tool_calls"), entry.get("successful_calls"),
                entry.get("failed_calls"),
                entry.get("invalid_argument_calls"))
            if observed:
                line += "; observed: " + observed
            lines.append(line)
        lines.append(
            "Use these exact tool names and observed values when answering; "
            "they are measured results from this series, not permanent rules.")
        block = "\n".join(lines)
        if len(block) > MAX_BLOCK_CHARS:
            block = block[:MAX_BLOCK_CHARS - len(_TRUNCATED)] + _TRUNCATED
        return block


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return redact_text(value, 80)
    return redact_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str), 120)


# ---------------------------------------------------------------------------
# Discovery questions
#
# These cannot be answered from the permanent architecture documents alone:
# they ask for values observed in a specific run (execution mode, verified image
# count, leaf/scratch/concept deltas).  Expected values are supplied separately
# by ``dynamic_expected`` so the questions never leak a rubric to the model.
# ---------------------------------------------------------------------------

DISCOVERY_QUESTIONS = (
    {
        "id": "native_execution",
        "area": "observed native execution",
        "prompt": (
            "In the completed series, what exact execution_mode did the "
            "verified native MTH run report, and did that top-level result use "
            "the N16I guest interpreter? Return JSON keys: execution_mode, "
            "n16i_used."
        ),
        "keys": ("execution_mode", "n16i_used"),
    },
    {
        "id": "leaf_execution",
        "area": "observed leaf execution",
        "prompt": (
            "For the vector pair-add result leaf, which leaf-level execution "
            "mode was recorded, and did it use the N16I guest interpreter? "
            "Return JSON keys: leaf_execution_mode, leaf_n16i_used."
        ),
        "keys": ("leaf_execution_mode", "leaf_n16i_used"),
    },
    {
        "id": "function_tree",
        "area": "observed function-tree audit",
        "prompt": (
            "After the function-tree audit, how many MTH kernels were linked, "
            "how many were expected, and how many images were verified? Return "
            "JSON keys: linked_kernels, expected_kernels, images_verified."
        ),
        "keys": ("linked_kernels", "expected_kernels", "images_verified"),
    },
    {
        "id": "leaf_growth",
        "area": "observed leaf-store growth",
        "prompt": (
            "What was the leaf-store count before and after the series, and "
            "what was the leaf delta? Return JSON keys: leaves_before, "
            "leaves_after, leaf_delta."
        ),
        "keys": ("leaves_before", "leaves_after", "leaf_delta"),
    },
    {
        "id": "scratch_growth",
        "area": "observed scratch growth",
        "prompt": (
            "After the free-chat teaching turns, how many scratch records "
            "existed and what was the scratch-record delta from before the "
            "series? Return JSON keys: scratch_records, scratch_delta."
        ),
        "keys": ("scratch_records", "scratch_delta"),
    },
    {
        "id": "concept_growth",
        "area": "observed concept matrix growth",
        "prompt": (
            "After the teaching turns, how many concept statements and "
            "concepts existed, and how many concept pairs were occupied? "
            "Return JSON keys: concept_statements, concept_concepts, "
            "concept_occupied_pairs."
        ),
        "keys": ("concept_statements", "concept_concepts",
                 "concept_occupied_pairs"),
    },
    {
        "id": "learning_growth",
        "area": "observed L0-L9 learning counters",
        "prompt": (
            "What were the final L0-L9 learning turns and rows counters after "
            "the series? Return JSON keys: turns, rows."
        ),
        "keys": ("turns", "rows"),
    },
    {
        "id": "leaf_verification",
        "area": "observed leaf verification",
        "prompt": (
            "Did the verified leaf's real NaN16 image execute on the Python "
            "NaNVM0 boundary, and was the native fixture marker found? Return "
            "JSON keys: nan_executable, marker_found."
        ),
        "keys": ("nan_executable", "marker_found"),
    },
)
_DISCOVERY_BY_ID = {question["id"]: question
                    for question in DISCOVERY_QUESTIONS}


def static_discovery_questions() -> list:
    """Return the fixed discovery questions without expected values."""
    return [
        {
            "id": question["id"],
            "area": question["area"],
            "prompt": question["prompt"],
            "keys": list(question["keys"]),
        }
        for question in DISCOVERY_QUESTIONS
    ]


def _unwrap_metrics(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("metrics"), dict):
        return value["metrics"]
    return value


def _before_after(value: dict) -> tuple[dict, dict]:
    if isinstance(value.get("system_learning"), dict):
        system = value["system_learning"]
        return (_unwrap_metrics(system.get("before")),
                _unwrap_metrics(system.get("after")))
    if isinstance(value.get("after"), dict):
        return (_unwrap_metrics(value.get("before")),
                _unwrap_metrics(value.get("after")))
    return _unwrap_metrics(value.get("before")), _unwrap_metrics(value)


def _series_calls(value: dict) -> list:
    calls = []
    sessions = value.get("sessions")
    if isinstance(sessions, list):
        for session in sessions:
            if isinstance(session, dict) and isinstance(
                    session.get("tool_calls"), list):
                calls.extend(session["tool_calls"])
    return calls


def _copy_metric(expected: dict, key: str, source: dict,
                 target: str | None = None) -> None:
    if key in source and source[key] is not None:
        value = source[key]
        if isinstance(value, float) and _as_float(value) is None:
            return
        expected[target or key] = value


def dynamic_expected(after_metrics: Any) -> dict:
    """Derive expected discovery values from measured series state.

    ``after_metrics`` may be an after-metrics mapping, a system snapshot, or a
    complete ``nanlm.gemma-training-series/1`` result.  Numeric deltas are
    derived only when both before and after values are present.  Observed tool
    fields such as ``execution_mode`` are recovered from session call results
    when the full result is supplied.
    """
    if not isinstance(after_metrics, dict):
        return {}
    before, after = _before_after(after_metrics)
    expected: dict = {}
    for key in (
            "turns", "rows", "scratch_records", "scratch_pages_allocated",
            "concept_statements", "concept_concepts", "concept_occupied_pairs",
            "concept_density", "linked_kernels", "expected_kernels",
            "images_verified", "leaves", "execution_mode", "n16i_used",
            "execution_modes", "leaf_execution_mode", "leaf_n16i_used",
            "nan_executable", "marker_found", "kernel_id", "marker",
            "tree_complete", "image_failures", "leaf_count"):
        _copy_metric(expected, key, after)

    if "leaves" in after:
        expected["leaves_after"] = after["leaves"]
        expected["leaf_count"] = after["leaves"]
    if "leaves" in before:
        expected["leaves_before"] = before["leaves"]
    if "leaves" in before and "leaves" in after:
        expected["leaf_delta"] = _numeric_delta(before["leaves"],
                                                after["leaves"])
        expected["leaves_delta"] = expected["leaf_delta"]
    # Preserve explicitly recorded deltas and before values when the caller
    # supplies an already-summarized series snapshot.
    for key in (
            "leaves_before", "leaf_delta", "leaves_delta", "turns_delta",
            "rows_delta", "scratch_delta", "concept_statements_delta",
            "concept_concepts_delta", "concept_occupied_pairs_delta"):
        if key in after and key not in expected:
            expected[key] = after[key]
    for source_key, delta_key in (
            ("turns", "turns_delta"),
            ("rows", "rows_delta"),
            ("scratch_records", "scratch_delta"),
            ("concept_statements", "concept_statements_delta"),
            ("concept_concepts", "concept_concepts_delta"),
            ("concept_occupied_pairs", "concept_occupied_pairs_delta")):
        if source_key in before and source_key in after:
            delta = _numeric_delta(before[source_key], after[source_key])
            if delta is not None:
                expected[delta_key] = delta

    # A full series result also contains the session tool results.  They are
    # the only source for execution-mode and marker observations.
    observations = extract_observations(_series_calls(after_metrics))
    for key in (
            "execution_mode", "n16i_used", "execution_modes",
            "leaf_execution_mode", "leaf_n16i_used", "images_verified",
            "image_failures", "nan_executable", "marker_found", "kernel_id",
            "marker", "linked_kernels", "expected_kernels", "tree_complete",
            "leaves"):
        if key in observations and key not in expected:
            expected[key] = observations[key]
    if "leaves" in observations:
        expected.setdefault("leaves_after", observations["leaves"])
        expected.setdefault("leaf_count", observations["leaves"])
    return expected


def _numeric_delta(before: Any, after: Any):
    left = _as_float(before)
    right = _as_float(after)
    if left is None or right is None:
        return None
    delta = right - left
    return int(delta) if float(delta).is_integer() else round(delta, 6)


def _parse_answers(answers: Any) -> tuple[dict, str | None]:
    if isinstance(answers, str):
        try:
            parsed = json.loads(answers)
        except (TypeError, ValueError) as exc:
            return {}, str(exc)
        answers = parsed
    if not isinstance(answers, dict):
        return {}, "answers must be a JSON object"
    if isinstance(answers.get("answers"), dict):
        answers = answers["answers"]
    return answers, None


def _text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str)
    return text.strip().strip("`'\"").casefold()


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and "any" in expected:
        return any(_matches(actual, item) for item in expected["any"])
    if isinstance(expected, (list, tuple, set, frozenset)):
        return any(_matches(actual, item) for item in expected)
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        return _text(actual) in ({"true", "yes", "1"} if expected
                                 else {"false", "no", "0"})
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        value = _as_float(actual)
        if value is None:
            return False
        return abs(value - float(expected)) <= max(
            1e-9, abs(float(expected)) * 1e-9)
    return _text(actual) == _text(expected)


def score_discovery_answers(answers: Any, expected: Any) -> dict:
    """Score discovery answers against measured expected values.

    ``expected`` may be the flat mapping returned by ``dynamic_expected`` or a
    mapping keyed by question id.  Only questions with at least one expected
    key are scored; unavailable questions are reported but do not inflate the
    denominator.
    """
    parsed, parse_error = _parse_answers(answers)
    expected = expected if isinstance(expected, dict) else {}
    question_results = []
    total = 0.0
    maximum = 0.0
    for question in DISCOVERY_QUESTIONS:
        expected_for = {}
        nested_value = expected.get(question["id"])
        nested = nested_value if isinstance(nested_value, dict) else {}
        for key in question["keys"]:
            if key in expected:
                expected_for[key] = expected[key]
            elif key in nested:
                expected_for[key] = nested[key]
        if not expected_for and nested_value is not None:
            expected_for[question["keys"][0]] = nested_value
        if not expected_for:
            question_results.append({
                "id": question["id"],
                "area": question["area"],
                "available": False,
                "points": None,
                "expected_keys": [],
                "matched_keys": [],
                "missing_keys": [],
                "checks": [],
            })
            continue
        answer = parsed.get(question["id"])
        if not isinstance(answer, dict):
            # Accept a flat answer object for a single-question call.
            answer = {key: parsed[key] for key in question["keys"]
                      if key in parsed}
        checks = []
        matched = []
        missed = []
        for key, value in expected_for.items():
            actual = answer.get(key)
            ok = _matches(actual, value)
            checks.append({
                "key": key,
                "expected": sanitize_value(value, text_limit=120),
                "actual": sanitize_value(actual, text_limit=120),
                "matched": ok,
            })
            (matched if ok else missed).append(key)
        points = len(matched) / len(expected_for)
        total += points
        maximum += 1.0
        question_results.append({
            "id": question["id"],
            "area": question["area"],
            "available": True,
            "answered": isinstance(parsed.get(question["id"]), dict),
            "points": round(points, 4),
            "expected_keys": list(expected_for),
            "matched_keys": matched,
            "missing_keys": missed,
            "checks": checks,
        })
    percent = round(100.0 * total / maximum, 1) if maximum else 0.0
    result = {
        "schema": DISCOVERY_SCORE_SCHEMA,
        "total_points": round(total, 4),
        "max_points": maximum,
        "percent": percent,
        "available_questions": int(maximum),
        "questions": question_results,
    }
    if parse_error:
        result["parse_error"] = parse_error
    return result
