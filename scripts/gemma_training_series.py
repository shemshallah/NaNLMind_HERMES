#!/usr/bin/env python3
"""Reproducible multi-session Gemma tool-learning series over HERMES.

The script pairs the real plugin handlers to a loopback NANNET authority with
``universe:read`` and ``training:execute``, then runs several Gemma steering
sessions against the same local state directory. Before and after the series it
administers the same no-tools written quiz and snapshots the plugin's own
learning state. Canonical merge/promotion tools are excluded from the safe tool
surface and remain disabled.

This measures local, loopback behavior. It is not a canonical training run, a
proof of general learning, or a NANNET/CLOCKED deployment test.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import (  # noqa: E402
    auth_tools,
    hermes_secrets,
    register,
    session_context,
)
from nanlmind_hermes.gemma_learning_journal import SessionJournal  # noqa: E402
from nanlmind_hermes.http_api import make_server as make_authority  # noqa: E402
from nanlmind_hermes.layered_learning import reset_coordinator_cache  # noqa: E402
from nanlmind_hermes.mth_leaves import reset_cache as reset_leaf_cache  # noqa: E402
from nanlmind_hermes.pairing import PairingAuthority  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_SESSIONS = 3
DEFAULT_TURNS_PER_SESSION = 8
DEFAULT_OUTPUT = ROOT / "docs" / "GEMMA_TRAINING_SERIES_RESULTS.json"
SERIES_SCHEMA = "nanlm.gemma-training-series/1"
QUIZ_SCHEMA = "nanlm.gemma-tool-use-quiz/1"
SUMMARY_SCHEMA = "nanlm.gemma-training-series-summary/1"
MAX_TOOL_RESULT_CHARS = 4000
MAX_CALL_RESULT_CHARS = 1200
MAX_QUIZ_RESPONSE_CHARS = 8000

# The model can explore and use the real data/learning tools, but never the
# local promotion shortcut or the peer-memory merge path. Canonical merge is
# separately closed by configuration and is not exposed by any handler here.
SAFE_TOOLS = (
    "nanlm_status",
    "nanlm_session_context",
    "nanlm_chat",
    "nanlm_chat_history",
    "nanlm_learning_cycle",
    "nanlm_learning_status",
    "nanlm_wordnet_ingest",
    "nanlm_scratch_register",
    "nanlm_mth_catalog",
    "nanlm_mth_inspect",
    "nanlm_mth_runtime",
    "nanlm_mth_verify",
    "nanlm_mth_execute",
    "nanlm_mth_n16i_selftest",
    "nanlm_mth_dispatch",
    "nanlm_mth_hub",
    "nanlm_mth_tree",
    "nanlm_mth_leaf_create",
    "nanlm_mth_leaf_get",
    "nanlm_mth_pair",
    "nanlm_mth_leaf_verify",
)
SAFE_TOOL_SET = frozenset(SAFE_TOOLS)
DENIED_TOOLS = (
    "nanlm_learning_sync",
    "nanlm_memory_merge",
    "nanlm_head_candidate",
    "nanlm_head_learn_candidate",
    "nanlm_head_candidate_evaluate",
    "nanlm_tree_mutate",
)
# Observation/ingestion remains available, but no model-supplied approval may
# reach a local promotion path in this harness.
PROMOTION_ARGUMENT_TOOLS = frozenset({
    "nanlm_learning_cycle",
    "nanlm_wordnet_ingest",
})


def _safe_arguments(name, arguments):
    """Return handler arguments with model-supplied promotion switches removed."""
    safe = dict(arguments) if isinstance(arguments, dict) else {}
    safe.pop("approved", None)
    safe.pop("auto_sync", None)
    return safe

QUIZ_QUESTIONS = (
    {
        "id": "session_context",
        "area": "permanent session context",
        "prompt": (
            "You need the permanent HERMES session context and the exact shipped "
            "'permanent-instructions' document. Which tool do you call, and which "
            "argument selects that document? Return a JSON object with keys: "
            "tool, argument, document."
        ),
        "keys": ("tool", "argument", "document"),
        "keyword_groups": (
            ("nanlm_session_context",),
            ("document",),
            ("permanent-instructions",),
        ),
    },
    {
        "id": "mth_tree",
        "area": "MTH function tree",
        "prompt": (
            "You need to map and then audit all 91 bundled MTH kernels into "
            "immutable function leaves. Which tool do you call, and which two "
            "action values do you use? Return a JSON object with keys: tool, "
            "map_action, audit_action."
        ),
        "keys": ("tool", "map_action", "audit_action"),
        "keyword_groups": (
            ("nanlm_mth_tree",),
            ("map",),
            ("audit",),
        ),
    },
    {
        "id": "leaf_create_fill",
        "area": "MTH leaf create with fill",
        "prompt": (
            "Create a 256-value vector data leaf without listing 256 numbers. "
            "Which tool do you call, what kind do you create, and which payload "
            "key repeats one finite value across all 256 elements? Return a JSON "
            "object with keys: tool, kind, payload_key."
        ),
        "keys": ("tool", "kind", "payload_key"),
        "keyword_groups": (
            ("nanlm_mth_leaf_create",),
            ("vector",),
            ("fill",),
        ),
    },
    {
        "id": "pair_add",
        "area": "MTH pair add",
        "prompt": (
            "Pair two existing vector leaves with element-wise addition and create "
            "a named output leaf. Which tool and operator do you use, and which "
            "arguments name the two inputs and the output alias? Return a JSON "
            "object with keys: tool, operator, left, right, alias."
        ),
        "keys": ("tool", "operator", "left", "right", "alias"),
        "keyword_groups": (
            ("nanlm_mth_pair",),
            ("add",),
            ("left",),
            ("right",),
            ("alias",),
        ),
    },
    {
        "id": "leaf_verify",
        "area": "MTH leaf verify",
        "prompt": (
            "Verify that a data leaf's real NaN16 image executes on the Python "
            "NaNVM0 boundary. Which tool do you call, which argument identifies "
            "the leaf, and which result field reports image execution success? "
            "Return a JSON object with keys: tool, leaf_id, success_field."
        ),
        "keys": ("tool", "leaf_id", "success_field"),
        "keyword_groups": (
            ("nanlm_mth_leaf_verify",),
            ("leaf_id",),
            ("nan_executable",),
        ),
    },
    {
        "id": "chat_status",
        "area": "free chat and learning status",
        "prompt": (
            "Run two free-chat learning turns (one teaching statement and one "
            "question), then inspect L0-L9 learning status and the paged scratch "
            "register. Which three tools do you call? Return a JSON object with "
            "keys: chat_tool, status_tool, scratch_tool."
        ),
        "keys": ("chat_tool", "status_tool", "scratch_tool"),
        "keyword_groups": (
            ("nanlm_chat",),
            ("nanlm_learning_status",),
            ("nanlm_scratch_register",),
        ),
    },
)

TASK_DEFINITIONS = (
    {
        "id": "session_context", "stage": 1,
        "label": "Inspect the permanent session context",
        "tools": ("nanlm_session_context",),
    },
    {
        "id": "mth_tree", "stage": 1,
        "label": "Map or audit the 91-kernel MTH function tree",
        "tools": ("nanlm_mth_tree",),
        "argument_values": {"action": ("map", "audit")},
    },
    {
        "id": "mth_catalog", "stage": 1,
        "label": "Browse the MTH kernel catalog",
        "tools": ("nanlm_mth_catalog",),
    },
    {
        "id": "mth_runtime", "stage": 1,
        "label": "Inspect the MTH native runtime",
        "tools": ("nanlm_mth_runtime",),
    },
    {
        "id": "vector_leaf_one", "stage": 2,
        "label": "Create the first 256-value vector leaf with fill",
        "tools": ("nanlm_mth_leaf_create",), "kind": "vector",
        "payload_key": "fill", "count": 1,
    },
    {
        "id": "vector_leaf_two", "stage": 2,
        "label": "Create the second 256-value vector leaf with fill",
        "tools": ("nanlm_mth_leaf_create",), "kind": "vector",
        "payload_key": "fill", "count": 2,
    },
    {
        "id": "pair_connect", "stage": 2,
        "label": "Pair the two leaves with connect",
        "tools": ("nanlm_mth_pair",),
        "argument_values": {"operator": ("connect",)},
    },
    {
        "id": "pair_add", "stage": 2,
        "label": "Pair the two leaves with add",
        "tools": ("nanlm_mth_pair",),
        "argument_values": {"operator": ("add",)},
    },
    {
        "id": "leaf_verify", "stage": 2,
        "label": "Verify a leaf's NaN16 image",
        "tools": ("nanlm_mth_leaf_verify",),
    },
    {
        "id": "chat_teaching", "stage": 3,
        "label": "Run one free-chat teaching turn",
        "tools": ("nanlm_chat",), "count": 1,
    },
    {
        "id": "chat_question", "stage": 3,
        "label": "Run a second free-chat turn with a question",
        "tools": ("nanlm_chat",), "count": 2,
    },
    {
        "id": "learning_status", "stage": 3,
        "label": "Inspect L0-L9 learning status",
        "tools": ("nanlm_learning_status",),
    },
    {
        "id": "scratch_register", "stage": 3,
        "label": "Inspect the paged scratch register",
        "tools": ("nanlm_scratch_register",),
    },
)
TASK_BY_ID = {task["id"]: task for task in TASK_DEFINITIONS}
STAGE_TASKS = {
    1: [task["id"] for task in TASK_DEFINITIONS if task["stage"] == 1],
    2: [task["id"] for task in TASK_DEFINITIONS if task["stage"] == 2],
    3: [task["id"] for task in TASK_DEFINITIONS if task["stage"] == 3],
}
STAGE_DESCRIPTIONS = {
    1: ("Explore the permanent session context, the MTH function tree, the MTH "
        "kernel catalog, and the MTH native runtime."),
    2: ("Create two 256-value vector leaves with the fill convenience, connect "
        "them, add them, and verify a leaf's NaN16 image."),
    3: ("Run two free-chat learning turns (a teaching statement and a question), "
        "then inspect learning status and scratch. Optionally inspect existing "
        "leaves."),
}


def stage_for_session(session_index: int) -> int:
    """Cycle stages 1, 2, 3 so every session has a concrete focus."""
    return ((max(1, int(session_index)) - 1) % 3) + 1


def build_tool_use_quiz() -> dict:
    """Return the fixed, no-tools before/after tool-use quiz."""
    return {
        "schema": QUIZ_SCHEMA,
        "title": "HERMES tool-use written quiz",
        "instructions": (
            "No tools are available for this quiz. Answer from knowledge only. "
            "Return one JSON object keyed by question id; each value is an object "
            "containing the requested keys. Do not return prose or markdown."
        ),
        "questions": [
            {
                "id": question["id"],
                "area": question["area"],
                "prompt": question["prompt"],
                "keys": list(question["keys"]),
            }
            for question in QUIZ_QUESTIONS
        ],
    }


def build_quiz_messages(quiz: dict | None = None) -> list[dict]:
    quiz = quiz or build_tool_use_quiz()
    system = (
        "You are answering a written quiz about the HERMES plugin tool API for "
        "MTH session context, function-tree, data-leaf, pairing, verification, "
        "free-chat and learning-status tools. No tools are available. Do not "
        "attempt a tool call. Answer every question and return exactly one JSON "
        "object with no prose and no markdown."
    )
    user = json.dumps({
        "format": (
            "JSON object keyed by question id; each value is an object with the "
            "listed answer keys."
        ),
        "questions": quiz["questions"],
        "constraints": [
            "Use the exact tool names where asked.",
            "Keep every answer short.",
            "Return JSON only.",
        ],
    }, ensure_ascii=False, sort_keys=True)
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _json_object_from_text(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty quiz response")
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        for chunk in stripped.split("```"):
            chunk = chunk.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                candidates.append(chunk)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    last_error = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            if isinstance(value.get("answers"), dict):
                return value["answers"]
            return value
        last_error = ValueError("quiz response must be a JSON object")
    raise ValueError("cannot parse quiz JSON: " + str(last_error))


def parse_quiz_answers(text: str) -> dict:
    """Parse a quiz response, accepting a bare or fenced JSON object."""
    return _json_object_from_text(text)


def _answer_text(value) -> str:
    if isinstance(value, str):
        return value.lower()
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).lower()


def _answer_keys(value) -> set[str]:
    if not isinstance(value, dict):
        return set()
    return {str(key).lower() for key in value}


def _quiz_rubric(quiz: dict | None = None) -> dict:
    """Return the scoring rubric, merging any public quiz with the fixed one."""
    by_id = {question["id"]: question for question in QUIZ_QUESTIONS}
    if quiz is None:
        questions = [copy.deepcopy(question) for question in QUIZ_QUESTIONS]
    else:
        questions = []
        for question in quiz.get("questions", []):
            rubric = by_id.get(question.get("id"))
            questions.append(copy.deepcopy(rubric if rubric is not None
                                           else question))
    return {"questions": questions}


def score_tool_use_quiz(answers, quiz: dict | None = None) -> dict:
    """Score the fixed quiz with a deterministic keys/keywords rubric.

    Each question is worth 1 point: half for the required JSON keys and half for
    the required keyword groups. Answers may be a mapping or raw JSON text.
    """
    quiz = _quiz_rubric(quiz)
    parse_error = None
    if isinstance(answers, str):
        try:
            answers = parse_quiz_answers(answers)
        except ValueError as exc:
            parse_error = str(exc)
            answers = {}
    if not isinstance(answers, dict):
        parse_error = parse_error or "answers must be a JSON object"
        answers = {}
    if isinstance(answers.get("answers"), dict):
        answers = answers["answers"]

    question_results = []
    total_points = 0.0
    for question in quiz["questions"]:
        answer = answers.get(question["id"])
        present = _answer_keys(answer)
        required = [str(key).lower() for key in question.get("keys", [])]
        matched_keys = [key for key in required if key in present]
        missing_keys = [key for key in required if key not in present]
        key_fraction = (len(matched_keys) / len(required)) if required else 1.0

        text = _answer_text(answer) if answer is not None else ""
        groups = question.get("keyword_groups", [])
        matched_groups = []
        missing_groups = []
        for group in groups:
            alternatives = [str(item).lower() for item in group]
            if any(item in text for item in alternatives):
                matched_groups.append(list(group))
            else:
                missing_groups.append(list(group))
        keyword_fraction = (len(matched_groups) / len(groups)) if groups else 1.0

        points = round(0.5 * key_fraction + 0.5 * keyword_fraction, 4)
        total_points += points
        question_results.append({
            "id": question["id"],
            "area": question.get("area", ""),
            "answered": answer is not None,
            "keys_present": matched_keys,
            "keys_missing": missing_keys,
            "key_fraction": round(key_fraction, 4),
            "keyword_groups_matched": matched_groups,
            "keyword_groups_missing": missing_groups,
            "keyword_fraction": round(keyword_fraction, 4),
            "points": points,
        })

    maximum = float(len(question_results))
    percent = round(100.0 * total_points / maximum, 1) if maximum else 0.0
    result = {
        "schema": "nanlm.gemma-tool-use-quiz-score/1",
        "total_points": round(total_points, 4),
        "max_points": maximum,
        "percent": percent,
        "answered_questions": sum(1 for item in question_results
                                  if item["answered"]),
        "questions": question_results,
    }
    if parse_error:
        result["parse_error"] = parse_error
    return result


def validate_tool_arguments(schema: dict | None, arguments,
                            *, parse_error: str | None = None) -> list[str]:
    """Return deterministic invalid-argument problems for a tool call."""
    problems = []
    if parse_error:
        problems.append(parse_error)
    if not isinstance(arguments, dict):
        problems.append("arguments must be a JSON object")
        return problems
    if not isinstance(schema, dict):
        return problems
    parameters = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else {}
    required = parameters.get("required") or []
    for name in required:
        if name not in arguments:
            problems.append("missing required argument: " + str(name))
    properties = parameters.get("properties") or {}
    if parameters.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                problems.append("unknown argument: " + str(name))
    for name, value in arguments.items():
        specification = properties.get(name)
        if not isinstance(specification, dict):
            continue
        if "enum" in specification and value not in specification["enum"]:
            problems.append("invalid enum value for " + name)
        expected = specification.get("type")
        if expected == "string" and not isinstance(value, str):
            problems.append(name + " must be a string")
        elif expected == "integer" and (isinstance(value, bool)
                                        or not isinstance(value, int)):
            problems.append(name + " must be an integer")
        elif expected == "number" and (isinstance(value, bool)
                                       or not isinstance(value, (int, float))):
            problems.append(name + " must be a number")
        elif expected == "boolean" and not isinstance(value, bool):
            problems.append(name + " must be a boolean")
        elif expected == "array" and not isinstance(value, list):
            problems.append(name + " must be an array")
        elif expected == "object" and not isinstance(value, dict):
            problems.append(name + " must be an object")
    return problems


def _short(value, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _failed_result(result) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return True
    if result.get("ok") is False:
        return True
    if result.get("status") in {"error", "denied", "failed"}:
        return True
    if result.get("nan_executable") is False:
        return True
    return False


def _execute(ctx, name: str, arguments):
    if name in DENIED_TOOLS or name not in SAFE_TOOL_SET or name not in ctx.tools:
        return {"error": "tool is not available in this session", "tool": name}
    try:
        arguments = _safe_arguments(name, arguments)
        raw = ctx.tools[name]["handler"](arguments)
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)[:1000],
                "tool": name}


def _assistant_message(message: dict) -> dict:
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}),
    }


def call_matches_task(call: dict, task: dict) -> bool:
    """Pure matcher used by coverage metrics and tests."""
    if call.get("failed"):
        return False
    name = str(call.get("name") or "")
    if name not in task.get("tools", ()):
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    for key, allowed in (task.get("argument_values") or {}).items():
        if arguments.get(key) not in allowed:
            return False
    expected_kind = task.get("kind")
    if expected_kind is not None and arguments.get("kind") != expected_kind:
        return False
    payload_key = task.get("payload_key")
    if payload_key is not None:
        payload = arguments.get("payload")
        if not isinstance(payload, dict) or payload_key not in payload:
            return False
        value = payload.get(payload_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    return True


def task_coverage(calls: list[dict], task_ids=None) -> dict:
    """Compute required-task coverage for one session or the whole series."""
    wanted = None if task_ids is None else set(task_ids)
    results = []
    for task in TASK_DEFINITIONS:
        if wanted is not None and task["id"] not in wanted:
            continue
        matches = [index for index, call in enumerate(calls)
                   if call_matches_task(call, task)]
        required_count = int(task.get("count", 1))
        met = len(matches) >= required_count
        results.append({
            "id": task["id"],
            "stage": task["stage"],
            "label": task["label"],
            "met": met,
            "matched_calls": matches,
            "required_count": required_count,
        })
    met = [item["id"] for item in results if item["met"]]
    missing = [item["id"] for item in results if not item["met"]]
    total = len(results)
    return {
        "coverage": round(len(met) / total, 4) if total else 0.0,
        "met": met,
        "missing": missing,
        "total": total,
        "tasks": results,
    }


def recovery_metrics(calls: list[dict]) -> dict:
    """Report whether a later successful call followed each failure."""
    events = []
    for index, call in enumerate(calls):
        if not call.get("failed"):
            continue
        recovered_by = None
        recovered_index = None
        for later_index in range(index + 1, len(calls)):
            later = calls[later_index]
            if not later.get("failed"):
                recovered_by = str(later.get("name") or "")
                recovered_index = later_index
                break
        events.append({
            "call_index": index,
            "turn": call.get("turn"),
            "tool": call.get("name"),
            "recovered": recovered_index is not None,
            "recovered_by": recovered_by,
            "recovered_at_call": recovered_index,
        })
    recovered = sum(1 for event in events if event["recovered"])
    total = len(events)
    return {
        "failures": total,
        "recovered": recovered,
        "unrecovered": total - recovered,
        "rate": round(recovered / total, 4) if total else 1.0,
        "events": events,
    }


def compute_session_metrics(calls: list[dict], required_task_ids,
                            turns_used: int) -> dict:
    """Pure per-session metrics used by the live loop and tests."""
    coverage = task_coverage(calls, required_task_ids)
    failed = [call for call in calls if call.get("failed")]
    invalid = [call for call in calls if call.get("invalid_arguments")]
    used = sorted({str(call.get("name")) for call in calls
                   if call.get("name")})
    return {
        "tool_calls": len(calls),
        "successful_calls": len(calls) - len(failed),
        "failed_calls": len(failed),
        "success_rate": (round((len(calls) - len(failed)) / len(calls), 4)
                         if calls else 0.0),
        "distinct_tools": len(used),
        "tools_used": used,
        "required_tasks_met": coverage["met"],
        "required_tasks_missing": coverage["missing"],
        "required_task_coverage": coverage["coverage"],
        "required_tasks": coverage["tasks"],
        "turns_used": int(turns_used),
        "invalid_argument_calls": len(invalid),
        "invalid_arguments": [
            {"turn": call.get("turn"), "tool": call.get("name"),
             "problems": list(call.get("argument_problems") or [])}
            for call in invalid
        ],
        "recovery": recovery_metrics(calls),
    }


def system_metric_deltas(before: dict, after: dict) -> dict:
    """Numeric deltas for the fields used to describe system learning."""
    numeric_keys = (
        "turns", "rows", "promoted", "scratch_records",
        "concept_statements", "concept_concepts", "concept_occupied_pairs",
        "concept_density", "linked_kernels", "expected_kernels", "leaves",
    )
    deltas = {}
    for key in numeric_keys:
        before_value = (before or {}).get(key)
        after_value = (after or {}).get(key)
        if isinstance(before_value, bool) or isinstance(after_value, bool):
            continue
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            value = after_value - before_value
            deltas[key] = round(value, 6) if isinstance(value, float) else value
    return deltas


def _as_int(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_system_metrics(learning, status, tree, hub) -> dict:
    """Extract the bounded system-learning metrics from raw tool results."""
    learning = learning if isinstance(learning, dict) else {}
    status = status if isinstance(status, dict) else {}
    tree = tree if isinstance(tree, dict) else {}
    hub = hub if isinstance(hub, dict) else {}
    scratch = learning.get("scratch") if isinstance(learning.get("scratch"), dict) else {}
    saturation = (learning.get("concept_saturation")
                  if isinstance(learning.get("concept_saturation"), dict) else {})
    memory = (status.get("mathematical_memory")
              if isinstance(status.get("mathematical_memory"), dict) else {})
    return {
        "turns": _as_int(learning.get("turns")),
        "rows": _as_int(learning.get("rows")),
        "promoted": _as_int(learning.get("promoted")),
        "scratch_records": _as_int(scratch.get("records")),
        "scratch_pages_allocated": _as_int(scratch.get("allocated_pages")),
        "concept_statements": _as_int(saturation.get("statements")),
        "concept_concepts": _as_int(saturation.get("concepts")),
        "concept_occupied_pairs": _as_int(saturation.get("occupied_pairs")),
        "concept_density": _as_float(saturation.get("density")),
        "linked_kernels": _as_int(tree.get("linked")),
        "expected_kernels": _as_int(tree.get("expected")),
        "tree_complete": bool(tree.get("complete")),
        "leaves": _as_int(hub.get("leaves")),
        "memory_records": _as_int(memory.get("records")),
    }


def _invoke(ctx, name: str, arguments: dict | None = None):
    tool = ctx.tools.get(name)
    if not tool:
        return {"error": "tool is not registered: " + name, "tool": name}
    try:
        raw = tool["handler"](arguments or {})
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)[:1000],
                "tool": name}


def collect_system_metrics(ctx) -> dict:
    """Snapshot learning, status, MTH tree audit and leaf-store metrics."""
    learning = _invoke(ctx, "nanlm_learning_status", {})
    status = _invoke(ctx, "nanlm_status", {})
    tree = _invoke(ctx, "nanlm_mth_tree", {"action": "audit",
                                           "verify_images": False})
    hub = _invoke(ctx, "nanlm_mth_hub", {})
    errors = {
        name: value.get("error")
        for name, value in (("nanlm_learning_status", learning),
                            ("nanlm_status", status),
                            ("nanlm_mth_tree", tree),
                            ("nanlm_mth_hub", hub))
        if isinstance(value, dict) and value.get("error")
    }
    return {
        "metrics": extract_system_metrics(learning, status, tree, hub),
        "errors": errors,
    }


def _metrics_of(value) -> dict:
    if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
        return value["metrics"]
    return value if isinstance(value, dict) else {}


def _quiz_percent(value) -> float:
    if isinstance(value, dict):
        if isinstance(value.get("score"), dict):
            return _as_float(value["score"].get("percent"))
        if "percent" in value:
            return _as_float(value.get("percent"))
    return 0.0


def summarize_series(sessions: list[dict], before_system, after_system,
                     before_quiz, after_quiz) -> dict:
    """Pure aggregate metrics, coverage, quiz delta and system deltas."""
    sessions = list(sessions or [])
    all_calls = []
    session_rows = []
    for index, session in enumerate(sessions, 1):
        calls = session.get("tool_calls") or []
        all_calls.extend(calls)
        metrics = session.get("metrics") or compute_session_metrics(
            calls, [], session.get("turns_used", 0))
        recovery = metrics.get("recovery") or {}
        session_rows.append({
            "session": session.get("session_index", index),
            "stage": session.get("stage"),
            "tool_calls": _as_int(metrics.get("tool_calls")),
            "failed_calls": _as_int(metrics.get("failed_calls")),
            "distinct_tools": _as_int(metrics.get("distinct_tools")),
            "required_task_coverage": _as_float(
                metrics.get("required_task_coverage")),
            "turns_used": _as_int(metrics.get("turns_used")),
            "invalid_argument_calls": _as_int(
                metrics.get("invalid_argument_calls")),
            "failures": _as_int(recovery.get("failures")),
            "recovered_failures": _as_int(recovery.get("recovered")),
        })

    coverage = task_coverage(all_calls)
    stage_coverage = {}
    for stage in (1, 2, 3):
        stage_cov = task_coverage(all_calls, STAGE_TASKS[stage])
        stage_coverage[str(stage)] = {
            "coverage": stage_cov["coverage"],
            "met": stage_cov["met"],
            "missing": stage_cov["missing"],
            "total": stage_cov["total"],
        }

    tools_used = sorted({str(call.get("name")) for call in all_calls
                         if call.get("name")})
    total_calls = len(all_calls)
    total_failed = sum(1 for call in all_calls if call.get("failed"))
    total_invalid = sum(1 for call in all_calls if call.get("invalid_arguments"))
    failures = sum(row["failures"] for row in session_rows)
    recovered = sum(row["recovered_failures"] for row in session_rows)

    first = session_rows[0] if session_rows else {}
    last = session_rows[-1] if session_rows else {}
    delta_keys = ("tool_calls", "failed_calls", "distinct_tools",
                  "required_task_coverage", "turns_used",
                  "invalid_argument_calls")
    session_deltas = {}
    for key in delta_keys:
        first_value, last_value = first.get(key), last.get(key)
        if isinstance(first_value, (int, float)) and isinstance(last_value, (int, float)):
            session_deltas[key] = round(last_value - first_value, 6)

    before_metrics = _metrics_of(before_system)
    after_metrics = _metrics_of(after_system)
    before_percent = _quiz_percent(before_quiz)
    after_percent = _quiz_percent(after_quiz)
    return {
        "schema": SUMMARY_SCHEMA,
        "sessions": len(sessions),
        "tool_calls": total_calls,
        "successful_calls": total_calls - total_failed,
        "failed_calls": total_failed,
        "success_rate": (round((total_calls - total_failed) / total_calls, 4)
                         if total_calls else 0.0),
        "distinct_tools": len(tools_used),
        "tools_used": tools_used,
        "turns_used": sum(row["turns_used"] for row in session_rows),
        "invalid_argument_calls": total_invalid,
        "required_task_coverage": coverage["coverage"],
        "required_tasks_met": coverage["met"],
        "required_tasks_missing": coverage["missing"],
        "required_tasks_total": coverage["total"],
        "all_required_tasks_met": not coverage["missing"],
        "stage_coverage": stage_coverage,
        "recovery": {
            "failures": failures,
            "recovered": recovered,
            "unrecovered": failures - recovered,
            "rate": round(recovered / failures, 4) if failures else 1.0,
        },
        "per_session": session_rows,
        "session_deltas": session_deltas,
        "quiz": {
            "before_percent": round(before_percent, 1),
            "after_percent": round(after_percent, 1),
            "delta_percent": round(after_percent - before_percent, 1),
        },
        "system": {
            "before": before_metrics,
            "after": after_metrics,
            "deltas": system_metric_deltas(before_metrics, after_metrics),
        },
    }


def _load_session_instructions(ctx=None) -> str:
    try:
        context = session_context.open_session(ctx, persist=False)
        return context.instructions_block()
    except Exception:
        return ""


class Provider:
    """Minimal OpenAI-compatible chat-completions client used by Ollama."""

    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def complete(self, messages, temperature=0.7, max_tokens=2048,
                 timeout=240, purpose=""):
        from types import SimpleNamespace
        result = self.request(messages, tools=None, temperature=temperature,
                              max_tokens=max_tokens, timeout=timeout)
        message = result["choices"][0]["message"]
        return SimpleNamespace(text=message.get("content") or "",
                               provider="ollama-openai-compat",
                               model=result.get("model", self.model))

    def request(self, messages, *, tools=None, temperature=0.2,
                max_tokens=1200, timeout=300):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(float(temperature), 2.0)),
            "max_tokens": max(1, min(int(max_tokens), 4096)),
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(524288))


class Context:
    """Small host context matching the plugin's registration contract."""

    def __init__(self, root, provider, authority_url="", issuer_key=""):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "provider",
            "chat_learning": True,
            "learning_require_approval": True,
            "learning_auto_sync": False,
            "nannet_auth_url": authority_url,
            "nannet_issuer_public_key": issuer_key,
            "wordnet_archive": os.environ.get("NANLM_WORDNET_ARCHIVE", ""),
        }
        self.llm = provider
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs


def _tool_schemas(ctx) -> list[dict]:
    schemas = []
    for name in SAFE_TOOLS:
        tool = ctx.tools.get(name)
        if not tool:
            continue
        schema = json.loads(json.dumps(tool["schema"]))
        schema["description"] = str(schema.get("description", ""))[:700]
        properties = schema.get("parameters", {}).get("properties") or {}
        if name in PROMOTION_ARGUMENT_TOOLS:
            properties.pop("approved", None)
            required = schema.get("parameters", {}).get("required")
            if isinstance(required, list) and "approved" in required:
                required.remove("approved")
        for prop in properties.values():
            if isinstance(prop, dict) and isinstance(prop.get("description"), str):
                prop["description"] = prop["description"][:240]
        schemas.append({"type": "function", "function": schema})
    return schemas


def _stage_prompt(session_index: int, total_sessions: int, stage: int,
                  required_task_ids, prior_summary: str = "") -> str:
    tasks = [TASK_BY_ID[task_id] for task_id in required_task_ids
             if task_id in TASK_BY_ID]
    lines = [
        "This is session %d of %d; stage %d of 3." % (
            session_index, total_sessions, stage),
        STAGE_DESCRIPTIONS[stage],
        "",
        "Required-task checklist for this session (choose the order):",
    ]
    lines.extend("- " + task["label"] for task in tasks)
    if stage == 2:
        lines.extend([
            "",
            "Use the explicit fill convenience, for example payload "
            '{"fill": 2.0}, so you never list 256 numbers. Use fresh aliases '
            "for this session such as gemma-s%d-v1 and gemma-s%d-v2, then pair "
            "connect and add with distinct relation/output aliases." % (
                session_index, session_index),
        ])
    if stage == 3:
        lines.extend([
            "",
            "Use nanlm_chat for two distinct turns: first a short teaching "
            "statement, then a question. Use a session id such as "
            "gemma-series-s%d. Then inspect learning status and the scratch "
            "register." % session_index,
        ])
    lines.extend([
        "",
        "Global boundary: canonical merge and promotion are disabled. Never "
        "call nanlm_learning_sync or nanlm_memory_merge, never set approved=true "
        "on a promotion path, and do not claim a canonical commit, lease, proof "
        "or merge. The NANNET pipeline is local loopback and paired for this "
        "experiment.",
        "Prefer small bounded calls. You may issue more than one tool call in a "
        "turn. Inspect each result before choosing the next call. When the "
        "checklist is complete, give a concise final summary of what worked, "
        "what failed, and what to try next.",
    ])
    if prior_summary:
        lines.extend(["", "Previous sessions in this series:", prior_summary])
    return "\n".join(lines)


def _system_prompt(session_index: int, total_sessions: int, stage: int) -> str:
    return (
        "You are Gemma steering session %d of %d (stage %d) of a multi-session "
        "HERMES plugin tool-learning experiment. You have real tools; use them "
        "instead of merely describing them. The NANNET pipeline is intact and "
        "paired to a local loopback authority with universe:read and "
        "training:execute. Canonical merge and promotion remain disabled. Do "
        "not claim canonical commits, leases, proofs or a canonical merge. "
        "After each tool result, inspect it and choose the next useful test.\n\n"
        % (session_index, total_sessions, stage)
        + _load_session_instructions()
    )


def _summary_request(provider, transcript, calls) -> str:
    compact = []
    for item in transcript[-16:]:
        content = item.get("content") or ""
        compact.append({"role": item.get("role"),
                        "content": str(content)[:1200]})
    brief_calls = [
        {"turn": call.get("turn"), "name": call.get("name"),
         "failed": call.get("failed"),
         "arguments": _short(call.get("arguments"), 600)}
        for call in calls[-24:]
    ]
    prompt = {
        "task": ("Summarize this multi-session tool-learning session for the "
                 "operator. Return plain text with sections: What worked, What "
                 "failed, Tools learned, Learning-state observations, Next "
                 "experiment. Do not claim canonical commits or proofs."),
        "tool_calls": brief_calls,
        "transcript_tail": compact,
    }
    try:
        result = provider.request(
            [{"role": "system", "content": "You are Gemma reviewing a completed "
              "HERMES tool-learning session. Be concrete and honest."},
             {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            tools=None, temperature=0.2, max_tokens=1200)
        return result["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        return "summary request failed: " + type(exc).__name__ + ": " + str(exc)[:500]


def administer_quiz(provider, label: str, experience: str = "") -> dict:
    """Run the fixed no-tools quiz and return prompt, answer and score.

    ``experience`` is a bounded, measured session journal. The before quiz has
    no experience; the after quiz receives it so the same quiz measures
    in-context transfer from the actual session rather than isolated recall.
    """
    quiz = build_tool_use_quiz()
    messages = build_quiz_messages(quiz)
    if experience:
        messages.insert(1, {"role": "system", "content": experience})
    text = ""
    parse_error = None
    answers = {}
    try:
        result = provider.request(messages, tools=None, temperature=0.0,
                                  max_tokens=2200)
        message = (result.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content") or ""
        try:
            answers = parse_quiz_answers(text)
        except ValueError as exc:
            parse_error = str(exc)
    except Exception as exc:
        parse_error = type(exc).__name__ + ": " + str(exc)[:500]
    score = score_tool_use_quiz(answers, quiz)
    if parse_error:
        score["parse_error"] = parse_error
    return {
        "schema": "nanlm.gemma-tool-use-quiz-run/1",
        "label": label,
        "questions": quiz["questions"],
        "prompt": messages,
        "raw_response": text[:MAX_QUIZ_RESPONSE_CHARS],
        "answers": answers,
        "score": score,
    }


def run_session(ctx, provider, tools, *, session_index: int, stage: int,
                turns: int, model: str, state_dir, nannet_info: dict | None = None,
                required_task_ids=None, prior_summary: str = "") -> dict:
    """Run one Gemma steering session over the real safe tool surface."""
    required_task_ids = (required_task_ids if required_task_ids is not None
                         else STAGE_TASKS[stage])
    total_sessions = int((nannet_info or {}).get("sessions_requested") or 0) or session_index
    system = _system_prompt(session_index, total_sessions, stage)
    user = _stage_prompt(session_index, total_sessions, stage,
                         required_task_ids, prior_summary)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    calls = []
    transcript = list(messages)
    turns_used = 0
    no_tool_turns = 0
    for turn in range(turns):
        turns_used = turn + 1
        response = provider.request(messages, tools=tools, max_tokens=1400)
        message = (response.get("choices") or [{}])[0].get("message") or {}
        assistant = _assistant_message(message)
        messages.append(assistant)
        transcript.append(assistant)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            no_tool_turns += 1
            if no_tool_turns >= 2 or turn == turns - 1:
                break
            reminder = {"role": "user", "content": (
                "Continue with one concrete required tool call, or provide your "
                "final summary now.")}
            messages.append(reminder)
            transcript.append(reminder)
            continue
        no_tool_turns = 0
        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments")
            parse_error = None
            if raw_arguments is None or raw_arguments == "":
                arguments = {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    arguments = {}
                    parse_error = "invalid JSON arguments: " + str(exc)[:300]
            else:
                arguments = {}
                parse_error = "arguments must be a JSON object"

            arguments = _safe_arguments(name, arguments)
            schema = None
            if name in SAFE_TOOL_SET and name in ctx.tools:
                schema = ctx.tools[name].get("schema")
            if name in DENIED_TOOLS or name not in SAFE_TOOL_SET:
                argument_problems = []
            else:
                argument_problems = validate_tool_arguments(
                    schema, arguments, parse_error=parse_error)

            result = _execute(ctx, name, arguments)
            failed = _failed_result(result)
            call_id = call.get("id") or ("call_%d_%d" % (turn, index))
            record = {
                "turn": turn,
                "name": name,
                "arguments": arguments,
                "argument_problems": argument_problems,
                "invalid_arguments": bool(argument_problems),
                "denied_tool": name in DENIED_TOOLS,
                "failed": failed,
                "ok": not failed,
                "result_status": (result.get("status")
                                  if isinstance(result, dict) else None),
                "nan_executable": (result.get("nan_executable")
                                   if isinstance(result, dict) else None),
                "result_summary": _short(result, MAX_CALL_RESULT_CHARS),
            }
            calls.append(record)
            tool_message = {
                "role": "tool", "tool_call_id": call_id,
                "content": _short(result, MAX_TOOL_RESULT_CHARS),
            }
            messages.append(tool_message)
            transcript.append(tool_message)

    final = _summary_request(provider, transcript, calls)
    metrics = compute_session_metrics(calls, required_task_ids, turns_used)
    return {
        "schema": "nanlm.gemma-training-series-session/1",
        "session_index": session_index,
        "stage": stage,
        "model": model,
        "state_dir": str(state_dir),
        "turns_requested": turns,
        "turns_used": turns_used,
        "required_task_ids": list(required_task_ids),
        "tool_calls": calls,
        "metrics": metrics,
        "final_summary": final,
        "transcript": transcript,
    }


def _prior_summary(sessions: list[dict]) -> str:
    lines = []
    for session in sessions:
        metrics = session.get("metrics") or {}
        lines.append(
            "- session %s (stage %s): %s calls, %s failed, %.0f%% required "
            "coverage" % (
                session.get("session_index"), session.get("stage"),
                metrics.get("tool_calls", 0), metrics.get("failed_calls", 0),
                100.0 * _as_float(metrics.get("required_task_coverage"))))
    return "\n".join(lines)


def _markdown_text(value, limit: int = 500) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _render_markdown(record: dict) -> str:
    summary = record.get("summary") or {}
    quiz = record.get("quiz") or {}
    system = record.get("system_learning") or {}
    nannet = record.get("nannet") or {}
    lines = [
        "# Gemma multi-session tool-learning series",
        "",
        "Generated by `scripts/gemma_training_series.py`. This report describes "
        "one local run; it is not a proof of general learning.",
        "",
        "## Configuration",
        "",
        "- model: `%s`" % record.get("model"),
        "- endpoint: `%s`" % record.get("endpoint"),
        "- state: `%s`" % record.get("state_dir"),
        "- sessions: %s" % record.get("sessions_requested"),
        "- turns per session: %s" % record.get("turns_per_session"),
        "- NANNET authority: `%s`" % nannet.get("authority_url"),
        "- NANNET scopes: `%s`" % nannet.get("scopes"),
        "- canonical merge: disabled",
        "",
        "## Written quiz (same questions before and after)",
        "",
        "| Run | Score |",
        "|---|---:|",
        "| before | %s%% |" % ((quiz.get("before") or {}).get("score") or {}).get(
            "percent", 0.0),
        "| after | %s%% |" % ((quiz.get("after") or {}).get("score") or {}).get(
            "percent", 0.0),
        "| delta | %s points |" % (quiz.get("delta_percent", 0.0)),
        "",
        "### Quiz questions",
        "",
    ]
    before_scores = {
        item.get("id"): item.get("points", 0.0)
        for item in (((quiz.get("before") or {}).get("score") or {}).get(
            "questions") or [])
    }
    after_scores = {
        item.get("id"): item.get("points", 0.0)
        for item in (((quiz.get("after") or {}).get("score") or {}).get(
            "questions") or [])
    }
    quiz_questions = ((quiz.get("before") or {}).get("questions") or [])
    for question in quiz_questions:
        question_id = question.get("id")
        lines.append("- **%s** — before %s/1, after %s/1. %s" % (
            question_id, before_scores.get(question_id, 0.0),
            after_scores.get(question_id, 0.0),
            _markdown_text(question.get("prompt"), 400)))
    lines.extend([
        "",
        "## Sessions",
        "",
        "| Session | Stage | Calls | Failed | Distinct tools | Required coverage | "
        "Turns | Invalid args | Recovered failures |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.get("per_session", []):
        lines.append("| %s | %s | %s | %s | %s | %.0f%% | %s | %s | %s/%s |" % (
            row.get("session"), row.get("stage"), row.get("tool_calls"),
            row.get("failed_calls"), row.get("distinct_tools"),
            100.0 * _as_float(row.get("required_task_coverage")),
            row.get("turns_used"), row.get("invalid_argument_calls"),
            row.get("recovered_failures"), row.get("failures")))
    lines.extend(["", "### Per-session required-task status", ""])
    for session in record.get("sessions", []):
        metrics = session.get("metrics") or {}
        lines.append("- **Session %s, stage %s**: met %s / %s; missing %s" % (
            session.get("session_index"), session.get("stage"),
            len(metrics.get("required_tasks_met") or []),
            len(metrics.get("required_tasks") or []),
            ", ".join(metrics.get("required_tasks_missing") or []) or "none"))
        for failure in (metrics.get("recovery") or {}).get("events", []):
            lines.append("  - failure `%s` on turn %s; recovered: %s%s" % (
                failure.get("tool"), failure.get("turn"),
                failure.get("recovered"),
                (" via `%s`" % failure.get("recovered_by")
                 if failure.get("recovered_by") else "")))
    lines.extend(["", "### Bounded session transcripts", "",
                  "System prompts are omitted; tool results are truncated.", ""])
    for session in record.get("sessions", []):
        lines.append("#### Session %s (stage %s)" % (
            session.get("session_index"), session.get("stage")))
        lines.append("")
        for item in session.get("transcript", []):
            role = item.get("role")
            if role == "system":
                continue
            if role == "assistant" and item.get("tool_calls"):
                names = [
                    str((call.get("function") or {}).get("name") or "")
                    for call in item["tool_calls"]
                ]
                lines.append("- **assistant**: tool calls: `%s`" % (
                    "`, `".join(names)))
                continue
            lines.append("- **%s**: %s" % (
                role, _markdown_text(item.get("content"), 400)))
        lines.append("")
    lines.extend(["", "## System-learning metrics", "",
                  "| Metric | Before | After | Delta |",
                  "|---|---:|---:|---:|"])
    before = ((system.get("before") or {}).get("metrics") or {})
    after = ((system.get("after") or {}).get("metrics") or {})
    deltas = system.get("deltas") or {}
    for key in ("turns", "rows", "scratch_records", "concept_statements",
                "concept_concepts", "concept_occupied_pairs", "concept_density",
                "linked_kernels", "leaves"):
        lines.append("| %s | %s | %s | %s |" % (
            key, before.get(key, 0), after.get(key, 0), deltas.get(key, 0)))
    lines.extend([
        "",
        "## Summary",
        "",
        "- tool calls: %s" % summary.get("tool_calls", 0),
        "- failed calls: %s" % summary.get("failed_calls", 0),
        "- distinct tools: %s" % summary.get("distinct_tools", 0),
        "- series required-task coverage: %.0f%%" % (
            100.0 * _as_float(summary.get("required_task_coverage"))),
        "- missing required tasks: %s" % (
            ", ".join(summary.get("required_tasks_missing") or []) or "none"),
        "- quiz delta: %s points" % summary.get("quiz", {}).get(
            "delta_percent", 0.0),
        "",
        "## Boundaries",
        "",
        "- local loopback NANNET authority; no production service is contacted",
        "- canonical merge, canonical promotion, commit and proof claims are "
        "disabled",
        "- the safe surface excludes `nanlm_learning_sync` and "
        "`nanlm_memory_merge`",
        "- results apply only to this model, endpoint, state directory and run",
        "",
    ])
    return "\n".join(lines)


def write_outputs(record: dict, output_path) -> tuple[Path, Path]:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    os.replace(temporary, output)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(record), encoding="utf-8")
    return output, markdown_path


def run_series(ctx, provider, tools, *, sessions: int, turns_per_session: int,
               model: str, state_dir, nannet_info: dict, output) -> int:
    """Run the before quiz, N sessions, after quiz and system snapshots."""
    state_dir = Path(state_dir)
    before_system = collect_system_metrics(ctx)
    before_quiz = administer_quiz(provider, "before")
    journal = SessionJournal(state_dir)
    session_records = []
    for index in range(1, sessions + 1):
        stage = stage_for_session(index)
        prior = _prior_summary(session_records)
        experience = journal.build_experience_block()
        if experience:
            prior = (prior + "\n\n" + experience).strip()
        record = run_session(
            ctx, provider, tools, session_index=index, stage=stage,
            turns=turns_per_session, model=model, state_dir=state_dir,
            nannet_info={**nannet_info, "sessions_requested": sessions},
            required_task_ids=STAGE_TASKS[stage],
            prior_summary=prior,
        )
        session_records.append(record)
        journal.append_session(
            index, stage, record.get("tool_calls") or [],
            metrics=record.get("metrics") or {},
        )
    after_system = collect_system_metrics(ctx)
    after_experience = journal.build_experience_block()
    after_quiz = administer_quiz(provider, "after", experience=after_experience)
    summary = summarize_series(session_records, before_system, after_system,
                               before_quiz, after_quiz)
    record = {
        "schema": SERIES_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "endpoint": provider.url,
        "state_dir": str(state_dir),
        "sessions_requested": sessions,
        "turns_per_session": turns_per_session,
        "nannet": nannet_info,
        "safe_tools": list(SAFE_TOOLS),
        "denied_tools": list(DENIED_TOOLS),
        "quiz": {
            "before": before_quiz,
            "after": after_quiz,
            "delta_percent": summary["quiz"]["delta_percent"],
        },
        "system_learning": {
            "before": before_system,
            "after": after_system,
            "deltas": summary["system"]["deltas"],
        },
        "sessions": session_records,
        "summary": summary,
        "learning_journal": {
            "path": str(journal.path),
            "entries": len(journal.load()),
            "after_experience": after_experience,
        },
        "boundaries": {
            "local_loopback_nannet": True,
            "nannet_pipeline_intact": True,
            "nannet_scopes": ["universe:read", "training:execute"],
            "canonical_merge_enabled": False,
            "canonical_promotion_enabled": False,
            "canonical_commit": False,
            "proof_claim": False,
            "safe_tool_exclusions": ["nanlm_learning_sync",
                                     "nanlm_memory_merge"],
            "note": ("Local loopback NANNET experiment. No canonical merge, "
                     "promotion, commit or proof claim."),
        },
    }
    json_path, markdown_path = write_outputs(record, output)
    print(json.dumps({
        "output": str(json_path),
        "markdown": str(markdown_path),
        "tool_calls": summary["tool_calls"],
        "failed_calls": summary["failed_calls"],
        "distinct_tools": summary["distinct_tools"],
        "required_task_coverage": summary["required_task_coverage"],
        "quiz_delta_percent": summary["quiz"]["delta_percent"],
        "system_deltas": summary["system"]["deltas"],
        "nannet": nannet_info,
    }, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--turns-per-session", type=int,
                        default=DEFAULT_TURNS_PER_SESSION)
    parser.add_argument("--url",
                        default=os.environ.get("NANLM_REVIEW_URL", DEFAULT_URL))
    parser.add_argument("--model",
                        default=os.environ.get("NANLM_REVIEW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    if not 1 <= args.sessions <= 6:
        raise SystemExit("--sessions must be 1..6")
    if not 1 <= args.turns_per_session <= 20:
        raise SystemExit("--turns-per-session must be 1..20")

    reset_coordinator_cache()
    reset_leaf_cache()
    root = Path(args.state_dir).expanduser() if args.state_dir else Path(
        os.environ.get("TMPDIR", "/tmp")) / "nanlm-gemma-training-series"
    root.mkdir(parents=True, exist_ok=True)
    provider = Provider(args.url, args.model)

    authority = PairingAuthority(root / "authority.sqlite", secrets.token_bytes(32))
    authority_http = make_authority(authority, "127.0.0.1", 0)
    authority_thread = threading.Thread(target=authority_http.serve_forever,
                                        daemon=True)
    authority_thread.start()
    authority_url = "http://127.0.0.1:%d" % authority_http.server_address[1]
    ctx = Context(root, provider, authority_url, authority.issuer_public_key())
    register(ctx)
    tools = _tool_schemas(ctx)
    master_b64 = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    pairing_code = authority.create_pairing_code(
        ["universe:read", "training:execute"], ttl_seconds=600)["pairing_code"]

    def secret(name):
        return master_b64 if name == "NANNET_MASTER_KEY" else ""

    try:
        with patch.object(hermes_secrets, "get_secret", side_effect=secret), \
             patch.object(auth_tools, "get_secret", side_effect=secret):
            init = json.loads(ctx.tools["nanlm_auth_init"]["handler"]({}))
            paired = json.loads(ctx.tools["nanlm_auth_pair"]["handler"](
                {"pairing_code": pairing_code}))
            scopes = list(paired.get("scopes") or [])
            if not {"universe:read", "training:execute"} <= set(scopes):
                raise RuntimeError(
                    "paired installation is missing required scopes: "
                    + repr(scopes))
            nannet_info = {
                "authority_url": authority_url,
                "issuer_public_key": authority.issuer_public_key()[:16] + "...",
                "installation_id": init.get("installation_id"),
                "paired": paired.get("paired"),
                "scopes": scopes,
                "authorization_mode": "nannet",
                "canonical_merge_enabled": False,
                "canonical_promotion_enabled": False,
            }
            return run_series(
                ctx, provider, tools, sessions=args.sessions,
                turns_per_session=args.turns_per_session, model=args.model,
                state_dir=root, nannet_info=nannet_info, output=args.output)
    finally:
        authority_http.shutdown()
        authority_http.server_close()
        authority_thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
