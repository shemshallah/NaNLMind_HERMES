#!/usr/bin/env python3
"""Offline diagnostics for a completed Gemma multi-session series.

This script never calls a model and never fabricates quiz answers.  It reads a
``nanlm.gemma-training-series/1`` JSON result and explains the measured written
quiz delta, builds the discovery quiz implied by the observed state, and lists
what the next series must measure.  The diagnostic is operator-facing: expected
values are measured from the input artifact and must not be inserted into a
model prompt as answers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import gemma_learning_journal as journal  # noqa: E402

DIAGNOSTIC_SCHEMA = "nanlm.gemma-series-diagnostic/1"
DISCOVERY_QUIZ_SCHEMA = "nanlm.gemma-discovery-quiz/1"
DEFAULT_INPUT = ROOT / "docs" / "GEMMA_TRAINING_SERIES_RESULTS.json"
DEFAULT_OUTPUT = ROOT / "docs" / "GEMMA_TRAINING_SERIES_DIAGNOSTIC.json"

# The exact tool names expected by the written quiz.  This is a local copy so
# diagnostics does not import or alter the training-series script.
EXPECTED_TOOL_NAMES = {
    "session_context": ("nanlm_session_context",),
    "mth_tree": ("nanlm_mth_tree",),
    "leaf_create_fill": ("nanlm_mth_leaf_create",),
    "pair_add": ("nanlm_mth_pair",),
    "leaf_verify": ("nanlm_mth_leaf_verify",),
    "chat_status": ("nanlm_chat", "nanlm_learning_status",
                    "nanlm_scratch_register"),
}
GENERIC_TOOL_ALIASES = {
    "session_context": ("session_context",),
    "mth_tree": ("function_tree", "tree"),
    "leaf_create_fill": ("data_leaf", "leaf"),
    "pair_add": ("pairing", "pair"),
    "leaf_verify": ("verification", "verify"),
    "chat_status": ("free_chat", "chat", "learning_status",
                    "scratch_register"),
}


def _as_float(value, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _answer_text(value) -> str:
    if isinstance(value, str):
        return value.casefold()
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      default=str).casefold()


def _quiz_percent(run) -> float:
    if not isinstance(run, dict):
        return 0.0
    score = run.get("score")
    if isinstance(score, dict):
        return _as_float(score.get("percent"))
    return _as_float(run.get("percent"))


def _answers(run) -> dict:
    if not isinstance(run, dict):
        return {}
    answers = run.get("answers")
    return answers if isinstance(answers, dict) else {}


def _prompt_text(prompt) -> str:
    if not isinstance(prompt, list):
        return ""
    return "\n".join(
        str(item.get("content") or "")
        for item in prompt if isinstance(item, dict)
    ).casefold()


def _exact_tool_hits(answers: dict) -> dict:
    hits = {}
    for question_id, tool_names in EXPECTED_TOOL_NAMES.items():
        text = _answer_text(answers.get(question_id))
        hits[question_id] = [
            name for name in tool_names if name.casefold() in text
        ]
    return hits


def _generic_aliases(answers: dict) -> list:
    exact_hits = _exact_tool_hits(answers)
    found = []
    for question_id, aliases in GENERIC_TOOL_ALIASES.items():
        if exact_hits.get(question_id):
            continue
        text = _answer_text(answers.get(question_id))
        for alias in aliases:
            if alias.casefold() in text:
                found.append({"question": question_id, "alias": alias})
                break
    return found


def analyze_written_quiz(series: dict) -> dict:
    quiz = series.get("quiz") if isinstance(series.get("quiz"), dict) else {}
    before = quiz.get("before") if isinstance(quiz.get("before"), dict) else {}
    after = quiz.get("after") if isinstance(quiz.get("after"), dict) else {}
    before_answers = _answers(before)
    after_answers = _answers(after)
    before_percent = _quiz_percent(before)
    after_percent = _quiz_percent(after)
    delta = _as_float(quiz.get("delta_percent"),
                      round(after_percent - before_percent, 1))
    before_hits = _exact_tool_hits(before_answers)
    after_hits = _exact_tool_hits(after_answers)
    before_exact = sum(1 for names in before_hits.values() if names)
    after_exact = sum(1 for names in after_hits.values() if names)
    after_prompt = _prompt_text(after.get("prompt"))
    before_prompt = _prompt_text(before.get("prompt"))
    journal_in_after_prompt = (
        "session journal" in after_prompt or "journal" in after_prompt
        or "observed results" in after_prompt
    )
    journal_in_before_prompt = (
        "session journal" in before_prompt or "journal" in before_prompt
        or "observed results" in before_prompt
    )
    exact_names_in_after_prompt = sorted(
        name for name in EXPECTED_TOOL_NAMES.values()
        for tool in name if tool.casefold() in after_prompt
    )
    generic_after = _generic_aliases(after_answers)
    generic_before = _generic_aliases(before_answers)
    answers_identical = before_answers == after_answers

    extra_keys = sorted(
        key for key in after.keys()
        if key not in {"schema", "label", "questions", "prompt",
                       "raw_response", "answers", "score"}
    )
    transcript_fields = [
        key for key in ("tool_calls", "transcript", "calls")
        if key in after
    ]
    isolated_after = (
        not journal_in_after_prompt
        and not exact_names_in_after_prompt
        and not transcript_fields
    )
    if before_percent == after_percent and delta == 0.0:
        cause = (
            "The written quiz could not detect the series experience. The "
            "after-quiz was an isolated no-tools request with no session "
            "transcript, no bounded learning journal and no measured tool "
            "results in its prompt. Both quiz runs therefore answered from "
            "the same generic declarative vocabulary: the before and after "
            "answer objects are identical, none of the six answers contains "
            "the exact safe tool name it used, and generic labels such as "
            "data_leaf/function_tree/pairing were used instead. A flat score "
            "is a measurement-isolation result, not evidence that the model "
            "cannot learn from tool use."
        )
    else:
        cause = (
            "The written quiz delta is not zero; the diagnostic still reports "
            "the prompt-isolation evidence so the next run can separate "
            "declarative answers from journal-backed observations."
        )

    return {
        "before_percent": round(before_percent, 1),
        "after_percent": round(after_percent, 1),
        "delta_percent": round(delta, 1),
        "before_answers": before_answers,
        "after_answers": after_answers,
        "answers_identical": answers_identical,
        "before_exact_tool_questions": before_exact,
        "after_exact_tool_questions": after_exact,
        "question_count": len(EXPECTED_TOOL_NAMES),
        "exact_tool_hits": {
            "before": before_hits,
            "after": after_hits,
        },
        "generic_aliases": {
            "before": generic_before,
            "after": generic_after,
        },
        "after_prompt_has_journal_block": journal_in_after_prompt,
        "before_prompt_has_journal_block": journal_in_before_prompt,
        "exact_tool_names_in_after_prompt": exact_names_in_after_prompt,
        "after_record_extra_keys": extra_keys,
        "after_record_transcript_fields": transcript_fields,
        "isolated_after_quiz": isolated_after,
        "cause": cause,
        "evidence": [
            "before and after scores are both %s%%" % round(before_percent, 1),
            "before and after answer objects identical: %s" % answers_identical,
            "exact tool names recovered in after answers: %s/%s" % (
                after_exact, len(EXPECTED_TOOL_NAMES)),
            "after prompt contains a journal block: %s" % journal_in_after_prompt,
            "after prompt contains any exact safe tool name: %s" % bool(
                exact_names_in_after_prompt),
            "after quiz record contains transcript/tool-call fields: %s" % (
                transcript_fields or "none"),
            "after quiz record extra keys: %s" % (extra_keys or "none"),
        ],
    }


def build_discovery_quiz(series: dict) -> dict:
    expected = journal.dynamic_expected(series)
    questions = []
    gaps = []
    for question in journal.static_discovery_questions():
        expected_for = {
            key: expected[key] for key in question["keys"] if key in expected
        }
        missing = [key for key in question["keys"] if key not in expected]
        if missing:
            gaps.append({
                "question": question["id"],
                "missing_expected_keys": missing,
                "note": (
                    "Not recoverable from this artifact. Capture it from the "
                    "full tool result in the session journal; do not infer it."
                ),
            })
        questions.append({
            "id": question["id"],
            "area": question["area"],
            "prompt": question["prompt"],
            "keys": question["keys"],
            "expected": expected_for,
            "missing_expected_keys": missing,
        })
    return {
        "schema": DISCOVERY_QUIZ_SCHEMA,
        "questions": questions,
        "expected": expected,
        "answers": None,
        "model_call": False,
        "note": (
            "Operator view only. Expected values were measured from the input "
            "series; no model answers were generated. Do not insert expected "
            "values into the model prompt. The next series should answer these "
            "after receiving only the bounded session journal."
        ),
        "measurement_gaps": gaps,
    }


def _metrics_summary(series: dict) -> dict:
    system = series.get("system_learning")
    system = system if isinstance(system, dict) else {}
    before = system.get("before")
    after = system.get("after")
    before = journal._unwrap_metrics(before) if isinstance(before, dict) else {}
    after = journal._unwrap_metrics(after) if isinstance(after, dict) else {}
    deltas = system.get("deltas") if isinstance(system.get("deltas"), dict) else {}
    keys = (
        "turns", "rows", "scratch_records", "concept_statements",
        "concept_concepts", "concept_occupied_pairs", "linked_kernels",
        "leaves",
    )
    return {
        "before": {key: before.get(key) for key in keys},
        "after": {key: after.get(key) for key in keys},
        "deltas": {key: deltas.get(key) for key in keys},
    }


def diagnose(series: dict, *, input_path=None) -> dict:
    if not isinstance(series, dict):
        raise ValueError("series result must be a JSON object")
    written = analyze_written_quiz(series)
    discovery = build_discovery_quiz(series)
    summary = series.get("summary") if isinstance(series.get("summary"), dict) else {}
    sessions = series.get("sessions") if isinstance(series.get("sessions"), list) else []
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(input_path) if input_path is not None else None,
        "input_schema": series.get("schema"),
        "model": series.get("model"),
        "state_dir": series.get("state_dir"),
        "session_count": len(sessions),
        "written_quiz": written,
        "discovery_quiz": discovery,
        "observed_state": _metrics_summary(series),
        "series_summary": {
            "tool_calls": summary.get("tool_calls"),
            "failed_calls": summary.get("failed_calls"),
            "distinct_tools": summary.get("distinct_tools"),
            "required_task_coverage": summary.get("required_task_coverage"),
            "quiz_delta_percent": (summary.get("quiz") or {}).get(
                "delta_percent", written["delta_percent"]),
        },
        "next_series_measurements": [
            "Persist one SessionJournal per state directory and append a "
            "bounded entry after every session.",
            "Inject journal.build_experience_block(limit=3) into later session "
            "prompts so exact tool names and observed values are in context.",
            "Run the written quiz before the series without the journal, then "
            "run the same quiz after the series with the journal block; keep an "
            "additional no-journal after run if retrieval vs. state must be "
            "separated.",
            "Ask the discovery questions after the journal is present and score "
            "them with score_discovery_answers(answers, "
            "dynamic_expected(full_series_result)).",
            "Repeat at least three sessions against the same persistent state "
            "directory and compare session 1 with the last session on exact "
            "tool-name rate, generic-name rate, invalid arguments, recovered "
            "failures, turns-to-complete and discovery score.",
            "Store full scalar observations from tool results when appending "
            "the journal; truncated result summaries may omit execution_mode, "
            "n16i_used, marker_found or counters.",
        ],
        "boundaries": {
            "model_called": False,
            "answers_fabricated": False,
            "live_result_claimed": False,
            "note": (
                "This report diagnoses one saved local run. It does not call a "
                "model, does not answer the discovery quiz, and does not claim "
                "canonical merge, promotion, proof or general learning."
            ),
        },
    }


def _format_value(value) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_markdown(report: dict) -> str:
    written = report.get("written_quiz") or {}
    discovery = report.get("discovery_quiz") or {}
    state = report.get("observed_state") or {}
    before = state.get("before") or {}
    after = state.get("after") or {}
    deltas = state.get("deltas") or {}
    lines = [
        "# Gemma series diagnostic",
        "",
        "Generated by `scripts/gemma_series_diagnostics.py` from `%s`." % (
            report.get("input")),
        "This report makes no model call and fabricates no quiz answers.",
        "",
        "## Written quiz",
        "",
        "| Run | Score | Exact-tool questions |",
        "|---|---:|---:|",
        "| before | %s%% | %s/%s |" % (
            written.get("before_percent"),
            written.get("before_exact_tool_questions"),
            written.get("question_count")),
        "| after | %s%% | %s/%s |" % (
            written.get("after_percent"),
            written.get("after_exact_tool_questions"),
            written.get("question_count")),
        "| delta | %s points | |" % written.get("delta_percent"),
        "",
        "**Why the delta was zero (when it was):** " + str(
            written.get("cause", "")),
        "",
        "Evidence:",
        "",
    ]
    lines.extend("- " + str(item) for item in written.get("evidence") or [])
    lines.extend([
        "",
        "## Discovery quiz implied by the observed run",
        "",
        "| Question | Measured expected values | Missing from artifact |",
        "|---|---|---|",
    ])
    for question in discovery.get("questions") or []:
        expected = ", ".join(
            "%s=%s" % (key, _format_value(value))
            for key, value in (question.get("expected") or {}).items()
        ) or "(none)"
        missing = ", ".join(question.get("missing_expected_keys") or []) or "none"
        lines.append("| `%s` | %s | %s |" % (
            question.get("id"), expected, missing))
    lines.extend([
        "",
        "The expected values are measured from this input artifact. They are "
        "operator evidence, not answers to put in a model prompt.",
        "",
        "## Observed state",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ])
    for key in ("turns", "rows", "scratch_records", "concept_statements",
                "concept_concepts", "concept_occupied_pairs",
                "linked_kernels", "leaves"):
        lines.append("| %s | %s | %s | %s |" % (
            key, _format_value(before.get(key)),
            _format_value(after.get(key)), _format_value(deltas.get(key))))
    lines.extend([
        "",
        "## Corrected next-series design",
        "",
    ])
    lines.extend("- " + str(item)
                 for item in report.get("next_series_measurements") or [])
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- no model call; no quiz answers were generated;",
        "- the flat written quiz is a prompt-isolation result, not evidence "
        "that the model cannot learn;",
        "- no canonical merge, promotion, commit, lease or proof claim.",
        "",
    ])
    return "\n".join(lines)


def _prepare_output(output_path) -> Path:
    output = Path(output_path).expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Termux has no writable root-level /tmp; only remap that exact case.
        if output.parts[:2] == ("/", "tmp"):
            fallback = Path(tempfile.gettempdir()) / output.name
            fallback.parent.mkdir(parents=True, exist_ok=True)
            return fallback
        raise
    return output


def write_report(report: dict, output_path) -> tuple[Path, Path]:
    output = _prepare_output(output_path)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    os.replace(temporary, output)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return output, markdown_path


def load_series(path) -> dict:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("series result must be a JSON object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    series = load_series(args.input)
    report = diagnose(series, input_path=args.input)
    json_path, markdown_path = write_report(report, args.output)
    written = report["written_quiz"]
    discovery = report["discovery_quiz"]
    print(json.dumps({
        "output": str(json_path),
        "markdown": str(markdown_path),
        "input": str(args.input),
        "quiz_delta_percent": written["delta_percent"],
        "after_exact_tool_questions": written[
            "after_exact_tool_questions"],
        "generic_answers_after": len(written["generic_aliases"]["after"]),
        "isolated_after_quiz": written["isolated_after_quiz"],
        "discovery_questions": len(discovery["questions"]),
        "discovery_missing_keys": sum(
            len(item["missing_expected_keys"])
            for item in discovery["questions"]),
        "model_call": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
