#!/usr/bin/env python3
"""Verify the supplied Round 030 native-audit artifacts.

Round 030 trains a 16-class semantic classifier whose 384 effective weights are
stored as triangle circulations inside one 8,388,608-byte C++ allocation.  The
supplied ``experiments/round030`` bundle contains the measured audit JSON, the
tuned results, the training log, three snapshots and a SHA-256 manifest.

This script does **not** require WordNet, torch, numpy or scikit-learn for its
default artifact-only path.  It verifies the manifest first, parses the
checksum-verified artifacts, and checks the reported invariants:

* mean tuned accuracy 0.420833 and raw shift accuracy 0.082142;
* native gradient error 1.86e-9 and gauge-weight error 4.17e-7;
* snapshot bitwise restore, mirror antisymmetry and an 8 MiB root;
* a conventional clean linear classifier at least as accurate as the cycle
  readout on every supplied seed.

An optional local re-run compiles ``native_cycle.cpp`` and executes a temporary
copy of ``native_audit.py`` only when a C++ compiler, numpy and scikit-learn are
available.  ``native_audit.py`` imports ``run.py``, which imports scikit-learn,
although the audit itself does not use WordNet data.  If anything is missing,
the report stays in artifact-only mode with a structured skip.

Labels are explicit throughout: supplied files are a **measured artifact**;
values produced here are a **re-run**.  No language-model/LM win, semantic
superiority, canonical implementation, quantum claim, portable benchmark or
NaN-unique benefit is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

SCHEMA = "nanlmind-hermes.round030-native-audit/1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "round030"
DEFAULT_MANIFEST_NAME = "SHA256SUMS"
DEFAULT_TIMEOUT_SECONDS = 120.0
COMPILER_CANDIDATES = ("g++", "clang++", "c++")
REQUIRED_RERUN_SOURCES = ("native_cycle.cpp", "run.py", "native_audit.py")

EXPECTED_SEEDS = (17, 29, 41)
ROOT_BYTES = 8_388_608
SNAPSHOT_MAGIC = b"NANCYC30"
SNAPSHOT_BYTES = ROOT_BYTES + len(SNAPSHOT_MAGIC)
EXPECTED_PREDICTIONS = 1000

# Exact values supplied by the Round 030 artifacts, plus the rounded targets
# quoted in the round README.
MEAN_TUNED_ACCURACY_EXACT = 0.42083333333333334
MEAN_TUNED_ACCURACY = 0.420833
MEAN_RAW_SHIFT_ACCURACY_EXACT = 0.08214285714285714
MEAN_RAW_SHIFT_ACCURACY = 0.082142
GRADIENT_MAX_ABS_ERROR_EXACT = 1.862645149230957e-09
GRADIENT_MAX_ABS_ERROR = 1.86e-9
GAUGE_WEIGHT_MAX_ABS_ERROR_EXACT = 4.1653402149677277e-07
GAUGE_WEIGHT_MAX_ABS_ERROR = 4.17e-7

MEAN_ACCURACY_TOLERANCE = 1e-12
GRADIENT_MATCH_TOLERANCE = 5e-12
GAUGE_MATCH_TOLERANCE = 5e-10
GRADIENT_BOUND = 1e-6
GAUGE_BOUND = 1e-5
MAX_101_SHIFT_LOGIT_DELTA = 3e-4
LINEAR_LOGIT_BOUND = 2e-3
CYCLE_DRIFT_BOUND = 1e-5

# Only the three parsed artifacts are mandatory in the manifest.  The supplied
# bundle also covers the source, logs, README and snapshots; every listed entry
# is always verified.
REQUIRED_MANIFEST_FILES = (
    "native_audit.json",
    "tuned_results.json",
    "run.log",
)

SOURCE_ARTIFACT = "measured_artifact"
SOURCE_ARTIFACT_LABEL = "measured artifact"
SOURCE_RERUN = "re_run"
SOURCE_RERUN_LABEL = "re-run"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2
EXIT_DENIED = 3

_SUM_LINE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})[ \t]+(?P<name>\*?.+)$")


class VerificationError(RuntimeError):
    """Raised when SHA256SUMS verification must deny execution."""


class ArtifactError(RuntimeError):
    """Raised when a supplied artifact is missing, malformed or inconsistent."""


class NativeBuildError(RuntimeError):
    """Raised when the optional native compilation fails."""


class NativeRunError(RuntimeError):
    """Raised when the optional native audit cannot be executed."""


# ---------------------------------------------------------------------------
# Digest verification
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, base: Path) -> bool:
    try:
        return path.resolve().is_relative_to(base.resolve())
    except OSError:
        return False


def _manifest_name(name: str) -> str:
    """Validate one manifest path and return it as a relative POSIX string."""
    if not name or "\x00" in name:
        raise VerificationError("empty or NUL-containing path in SHA256SUMS")
    if name.startswith(("/", "\\")) or "\\" in name:
        raise VerificationError(f"path traversal rejected for {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise VerificationError(f"path traversal rejected for {name!r}")
    return pure.as_posix()


def _read_manifest(manifest: Path) -> list[dict[str, str]]:
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError(f"cannot read SHA256SUMS {manifest}: {exc}") from exc
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SUM_LINE.match(line)
        if match is None:
            raise VerificationError(
                f"invalid SHA256SUMS line {line_number}: "
                "expected '<sha256>  <path>'")
        name = match.group("name")
        if name.startswith("*"):
            name = name[1:]
        name = _manifest_name(name.strip())
        if name in seen:
            raise VerificationError(f"duplicate SHA256SUMS entry for {name!r}")
        seen.add(name)
        entries.append({"path": name, "sha256": match.group("digest").lower()})
    if not entries:
        raise VerificationError("SHA256SUMS contains no file entries")
    return entries


def verify_artifacts(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Verify every supplied digest, rejecting traversal and missing files.

    No artifact is parsed and no native code is compiled before this succeeds.
    The three parsed artifacts must be covered by the manifest.
    """
    experiment = Path(experiment_dir)
    if not experiment.is_absolute():
        experiment = (ROOT / experiment).resolve()
    else:
        experiment = experiment.resolve()
    if not experiment.is_dir():
        raise VerificationError(f"experiment directory not found: {experiment}")

    manifest_path = (
        Path(manifest) if manifest is not None
        else experiment / DEFAULT_MANIFEST_NAME
    )
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise VerificationError(f"SHA256SUMS not found: {manifest_path}")

    entries = _read_manifest(manifest_path)
    verified: list[dict[str, str]] = []
    digests: dict[str, str] = {}
    errors: list[str] = []
    for entry in entries:
        name = entry["path"]
        candidate = experiment / name
        if not _within(candidate, experiment):
            raise VerificationError(f"path traversal rejected for {name!r}")
        if not candidate.is_file():
            errors.append(f"listed file missing: {name}")
            continue
        try:
            measured = sha256_file(candidate)
        except OSError as exc:
            errors.append(f"cannot read listed file {name}: {exc}")
            continue
        if measured != entry["sha256"]:
            errors.append(
                f"SHA-256 mismatch for {name}: expected {entry['sha256']}, "
                f"measured {measured}")
            continue
        verified.append({"path": name, "sha256": measured})
        digests[name] = measured

    if errors:
        raise VerificationError("; ".join(errors))

    missing_required = [name for name in REQUIRED_MANIFEST_FILES if name not in digests]
    if missing_required:
        raise VerificationError(
            "manifest does not cover required artifact(s): "
            + ", ".join(missing_required))

    return {
        "status": "verified",
        "manifest": str(manifest_path),
        "base": str(experiment),
        "entry_count": len(entries),
        "verified_count": len(verified),
        "verified": verified,
        "digests": digests,
        "required_files": list(REQUIRED_MANIFEST_FILES),
        "required_files_covered": True,
        "source_sha256": digests.get("native_audit.py"),
    }


# ---------------------------------------------------------------------------
# Artifact parsing and validation
# ---------------------------------------------------------------------------


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactError(f"{name} must be a JSON array")
    if length is not None and len(value) != length:
        raise ArtifactError(f"{name} must contain {length} entries")
    return value


def _get(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError:
        raise ArtifactError(f"{context} is missing required key {key!r}") from None


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ArtifactError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ArtifactError(f"{name} must be <= {maximum}")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactError(f"{name} must be a boolean")
    return value


def _validate_native_audit(data: dict[str, Any]) -> None:
    _integer(_get(data, "root_bytes", "native_audit.json"), "root_bytes")
    _number(
        _get(data, "native_gradient_max_abs_error", "native_audit.json"),
        "native_gradient_max_abs_error",
        minimum=0.0,
    )
    _number(
        _get(data, "gauge_weight_max_abs_error", "native_audit.json"),
        "gauge_weight_max_abs_error",
        minimum=0.0,
    )
    _boolean(_get(data, "mirror_antisymmetry", "native_audit.json"),
             "mirror_antisymmetry")
    _boolean(_get(data, "snapshot_bitwise_restore", "native_audit.json"),
             "snapshot_bitwise_restore")
    _integer(_get(data, "predictions", "native_audit.json"), "predictions")
    _number(_get(data, "ctypes_wall_ms", "native_audit.json"),
            "ctypes_wall_ms", minimum=0.0)
    _boolean(
        _get(data, "memory_wordnet_predictor_is_native", "native_audit.json"),
        "memory_wordnet_predictor_is_native",
    )
    _boolean(_get(data, "text_encoder_is_external", "native_audit.json"),
             "text_encoder_is_external")


def _validate_accuracy_block(block: Any, name: str) -> None:
    data = _mapping(block, name)
    _number(_get(data, "accuracy", name), f"{name}.accuracy",
            minimum=0.0, maximum=1.0)
    _number(_get(data, "cross_entropy", name), f"{name}.cross_entropy",
            minimum=0.0)


def _validate_tuned_results(data: dict[str, Any]) -> None:
    results = _list(
        _get(data, "results", "tuned_results.json"),
        "tuned_results.json.results",
        length=len(EXPECTED_SEEDS),
    )
    seeds: list[int] = []
    for index, raw in enumerate(results):
        result = _mapping(raw, f"tuned_results.json.results[{index}]")
        context = f"tuned_results.json.results[{index}]"
        seeds.append(_integer(_get(result, "seed", context), f"{context}.seed"))
        for key in ("cycle_test", "shift_test", "raw_shift_test"):
            _validate_accuracy_block(_get(result, key, context), f"{context}.{key}")
        _integer(
            _get(result, "native_root_bytes", context),
            f"{context}.native_root_bytes",
        )
        _boolean(
            _get(result, "cycle_equals_same_real_linear", context),
            f"{context}.cycle_equals_same_real_linear",
        )
        _number(
            _get(result, "max_logit_delta_single_shift", context),
            f"{context}.max_logit_delta_single_shift",
            minimum=0.0,
        )
        _number(
            _get(result, "max_logit_delta_101_shifts", context),
            f"{context}.max_logit_delta_101_shifts",
            minimum=0.0,
        )
    if sorted(seeds) != sorted(EXPECTED_SEEDS) or len(set(seeds)) != len(seeds):
        raise ArtifactError(
            f"tuned_results.json seeds must be {list(EXPECTED_SEEDS)}")

    means = _mapping(
        _get(data, "mean_accuracy", "tuned_results.json"),
        "tuned_results.json.mean_accuracy",
    )
    for key in ("cycle_test", "shift_test", "raw_shift_test"):
        _number(_get(means, key, "tuned_results.json.mean_accuracy"),
                f"mean_accuracy.{key}", minimum=0.0, maximum=1.0)

    limitations = data.get("limitations", [])
    if not isinstance(limitations, list):
        raise ArtifactError("tuned_results.json.limitations must be an array")


def _validate_run_log(data: dict[str, Any]) -> None:
    runs = _list(
        _get(data, "runs", "run.log"),
        "run.log.runs",
        length=len(EXPECTED_SEEDS),
    )
    seeds: list[int] = []
    for index, raw in enumerate(runs):
        run = _mapping(raw, f"run.log.runs[{index}]")
        context = f"run.log.runs[{index}]"
        seeds.append(_integer(_get(run, "seed", context), f"{context}.seed"))
        samples = _mapping(_get(run, "samples", context), f"{context}.samples")
        for key in ("train", "val", "test"):
            _integer(_get(samples, key, f"{context}.samples"),
                     f"{context}.samples.{key}")
        for key in (
            "cycle_before",
            "cycle_after_large_gauge",
            "raw_before",
            "raw_after_large_gauge",
            "conventional_same_features",
        ):
            _validate_accuracy_block(_get(run, key, context), f"{context}.{key}")
        for key in (
            "native_training_seconds",
            "native_vs_real_linear_max_logit_error",
            "cycle_max_drift_from_gauge",
            "max_cycle_logits_change",
            "restore_max_logit_error",
        ):
            _number(_get(run, key, context), f"{context}.{key}", minimum=0.0)
        _boolean(_get(run, "mirror_audit", context), f"{context}.mirror_audit")
        _integer(_get(run, "root_bytes", context), f"{context}.root_bytes")
    if sorted(seeds) != sorted(EXPECTED_SEEDS) or len(set(seeds)) != len(seeds):
        raise ArtifactError(f"run.log seeds must be {list(EXPECTED_SEEDS)}")
    _number(_get(data, "elapsed_seconds", "run.log"),
            "run.log.elapsed_seconds", minimum=0.0)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"cannot read {name}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid JSON in {name}: {exc}") from exc
    return _mapping(data, name)


def load_artifacts(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and structurally validate the three parsed Round 030 artifacts."""
    experiment = Path(experiment_dir).resolve()
    native_path = experiment / "native_audit.json"
    tuned_path = experiment / "tuned_results.json"
    run_path = experiment / "run.log"

    native = _load_json(native_path, "native_audit.json")
    _validate_native_audit(native)
    tuned = _load_json(tuned_path, "tuned_results.json")
    _validate_tuned_results(tuned)
    run_log = _load_json(run_path, "run.log")
    _validate_run_log(run_log)

    digests = (verification or {}).get("digests", {})
    snapshots: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        path = experiment / f"root_{seed}.bin"
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            magic = handle.read(len(SNAPSHOT_MAGIC))
        snapshots.append(
            {
                "seed": seed,
                "path": str(path),
                "relative_path": path.name,
                "sha256": digests.get(path.name),
                "size": path.stat().st_size,
                "magic": magic.decode("ascii", errors="replace"),
                "magic_ok": magic == SNAPSHOT_MAGIC,
            }
        )

    def _entry(path: Path, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path),
            "relative_path": path.name,
            "sha256": digests.get(path.name),
            "data": data,
        }

    return {
        "native_audit": _entry(native_path, native),
        "tuned_results": _entry(tuned_path, tuned),
        "run_log": _entry(run_path, run_log),
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _add_check(
    details: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    *,
    source: str = SOURCE_ARTIFACT,
    tolerance: float | None = None,
    note: str | None = None,
) -> None:
    details.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "source": source,
            "source_label": (
                SOURCE_RERUN_LABEL if source == SOURCE_RERUN
                else SOURCE_ARTIFACT_LABEL
            ),
            "tolerance": tolerance,
            "note": note,
        }
    )


def _targets() -> dict[str, Any]:
    return {
        "root_bytes": ROOT_BYTES,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "predictions": EXPECTED_PREDICTIONS,
        "mean_tuned_accuracy": MEAN_TUNED_ACCURACY,
        "mean_tuned_accuracy_exact": MEAN_TUNED_ACCURACY_EXACT,
        "mean_shift_accuracy": MEAN_TUNED_ACCURACY,
        "mean_raw_shift_accuracy": MEAN_RAW_SHIFT_ACCURACY,
        "mean_raw_shift_accuracy_exact": MEAN_RAW_SHIFT_ACCURACY_EXACT,
        "native_gradient_max_abs_error": GRADIENT_MAX_ABS_ERROR,
        "native_gradient_max_abs_error_exact": GRADIENT_MAX_ABS_ERROR_EXACT,
        "gauge_weight_max_abs_error": GAUGE_WEIGHT_MAX_ABS_ERROR,
        "gauge_weight_max_abs_error_exact": GAUGE_WEIGHT_MAX_ABS_ERROR_EXACT,
        "mirror_antisymmetry": True,
        "snapshot_bitwise_restore": True,
        "conventional_clean_classifier_ge_cycle": True,
        "tolerances": {
            "mean_accuracy": MEAN_ACCURACY_TOLERANCE,
            "gradient": GRADIENT_MATCH_TOLERANCE,
            "gauge_weight": GAUGE_MATCH_TOLERANCE,
            "gradient_bound": GRADIENT_BOUND,
            "gauge_weight_bound": GAUGE_BOUND,
            "max_101_shift_logit_delta": MAX_101_SHIFT_LOGIT_DELTA,
            "linear_logit_error": LINEAR_LOGIT_BOUND,
            "cycle_drift": CYCLE_DRIFT_BOUND,
        },
    }


def evaluate_artifacts(
    artifacts: dict[str, Any],
    verification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return check details and a flattened observed-value mapping."""
    native = artifacts["native_audit"]["data"]
    tuned = artifacts["tuned_results"]["data"]
    run_log = artifacts["run_log"]["data"]
    snapshots = artifacts["snapshots"]
    details: list[dict[str, Any]] = []

    _add_check(
        details,
        "manifest.verified",
        verification.get("status") == "verified",
        verification.get("status"),
        "verified",
    )
    _add_check(
        details,
        "manifest.required_files",
        bool(verification.get("required_files_covered")),
        verification.get("required_files"),
        list(REQUIRED_MANIFEST_FILES),
    )

    # Native audit JSON.
    _add_check(
        details, "native_audit.root_bytes",
        native["root_bytes"] == ROOT_BYTES,
        native["root_bytes"], ROOT_BYTES,
    )
    gradient = float(native["native_gradient_max_abs_error"])
    _add_check(
        details, "native_audit.gradient_max_abs_error",
        _close(gradient, GRADIENT_MAX_ABS_ERROR_EXACT,
               GRADIENT_MATCH_TOLERANCE),
        gradient, GRADIENT_MAX_ABS_ERROR_EXACT,
        tolerance=GRADIENT_MATCH_TOLERANCE,
        note="rounded README target 1.86e-9",
    )
    _add_check(
        details, "native_audit.gradient_bound",
        gradient <= GRADIENT_BOUND,
        gradient, GRADIENT_BOUND,
        note="native_audit.py asserts < 1e-6",
    )
    gauge = float(native["gauge_weight_max_abs_error"])
    _add_check(
        details, "native_audit.gauge_weight_max_abs_error",
        _close(gauge, GAUGE_WEIGHT_MAX_ABS_ERROR_EXACT,
               GAUGE_MATCH_TOLERANCE),
        gauge, GAUGE_WEIGHT_MAX_ABS_ERROR_EXACT,
        tolerance=GAUGE_MATCH_TOLERANCE,
        note="rounded README target 4.17e-7",
    )
    _add_check(
        details, "native_audit.gauge_weight_bound",
        gauge <= GAUGE_BOUND,
        gauge, GAUGE_BOUND,
        note="native_audit.py asserts < 1e-5",
    )
    _add_check(
        details, "native_audit.mirror_antisymmetry",
        native["mirror_antisymmetry"] is True,
        native["mirror_antisymmetry"], True,
    )
    _add_check(
        details, "native_audit.snapshot_bitwise_restore",
        native["snapshot_bitwise_restore"] is True,
        native["snapshot_bitwise_restore"], True,
    )
    _add_check(
        details, "native_audit.predictions",
        native["predictions"] == EXPECTED_PREDICTIONS,
        native["predictions"], EXPECTED_PREDICTIONS,
    )
    wall_ms = float(native["ctypes_wall_ms"])
    _add_check(
        details, "native_audit.ctypes_wall_ms",
        math.isfinite(wall_ms) and wall_ms >= 0.0,
        wall_ms, "finite and non-negative",
        note="host-only timing; not a benchmark",
    )
    _add_check(
        details, "native_audit.memory_wordnet_predictor_is_native",
        native["memory_wordnet_predictor_is_native"] is True,
        native["memory_wordnet_predictor_is_native"], True,
        note="self-reported by the supplied audit",
    )
    _add_check(
        details, "native_audit.text_encoder_is_external",
        native["text_encoder_is_external"] is True,
        native["text_encoder_is_external"], True,
        note="the 24D SVD text encoder is outside the C++ root",
    )

    # Tuned results.
    results = tuned["results"]
    seeds = sorted(result["seed"] for result in results)
    _add_check(
        details, "tuned.seeds",
        seeds == sorted(EXPECTED_SEEDS), seeds, sorted(EXPECTED_SEEDS),
    )
    means = tuned["mean_accuracy"]
    computed = {
        key: sum(float(result[key]["accuracy"]) for result in results)
        / len(results)
        for key in ("cycle_test", "shift_test", "raw_shift_test")
    }
    mean_cycle = float(means["cycle_test"])
    mean_shift = float(means["shift_test"])
    mean_raw = float(means["raw_shift_test"])
    _add_check(
        details, "tuned.mean_cycle_accuracy",
        _close(mean_cycle, MEAN_TUNED_ACCURACY_EXACT, MEAN_ACCURACY_TOLERANCE),
        mean_cycle, MEAN_TUNED_ACCURACY_EXACT,
        tolerance=MEAN_ACCURACY_TOLERANCE,
        note="rounded README target 0.420833",
    )
    _add_check(
        details, "tuned.mean_shift_accuracy",
        _close(mean_shift, MEAN_TUNED_ACCURACY_EXACT, MEAN_ACCURACY_TOLERANCE),
        mean_shift, MEAN_TUNED_ACCURACY_EXACT,
        tolerance=MEAN_ACCURACY_TOLERANCE,
        note="identical class predictions after the synthetic gauge shift",
    )
    _add_check(
        details, "tuned.mean_raw_shift_accuracy",
        _close(mean_raw, MEAN_RAW_SHIFT_ACCURACY_EXACT,
               MEAN_ACCURACY_TOLERANCE),
        mean_raw, MEAN_RAW_SHIFT_ACCURACY_EXACT,
        tolerance=MEAN_ACCURACY_TOLERANCE,
        note="rounded README target 0.082142",
    )
    reported_match = all(
        _close(float(means[key]), computed[key], MEAN_ACCURACY_TOLERANCE)
        for key in computed
    )
    _add_check(
        details, "tuned.reported_means_match_runs",
        reported_match, computed, {k: float(means[k]) for k in computed},
        tolerance=MEAN_ACCURACY_TOLERANCE,
    )
    cycle_equal_shift = all(
        _close(
            float(result["cycle_test"]["accuracy"]),
            float(result["shift_test"]["accuracy"]),
            MEAN_ACCURACY_TOLERANCE,
        )
        for result in results
    )
    _add_check(
        details, "tuned.cycle_predictions_equal_under_shift",
        cycle_equal_shift,
        [float(result["cycle_test"]["accuracy"]) for result in results],
        [float(result["shift_test"]["accuracy"]) for result in results],
        tolerance=MEAN_ACCURACY_TOLERANCE,
    )
    _add_check(
        details, "tuned.native_root_bytes_all",
        all(result["native_root_bytes"] == ROOT_BYTES for result in results),
        [result["native_root_bytes"] for result in results],
        [ROOT_BYTES] * len(results),
    )
    _add_check(
        details, "tuned.cycle_equals_same_real_linear_all",
        all(result["cycle_equals_same_real_linear"] is True for result in results),
        [result["cycle_equals_same_real_linear"] for result in results],
        [True] * len(results),
    )
    deltas_101 = [float(result["max_logit_delta_101_shifts"])
                  for result in results]
    _add_check(
        details, "tuned.max_logit_delta_101_shifts_bounded",
        all(delta < MAX_101_SHIFT_LOGIT_DELTA for delta in deltas_101),
        deltas_101, MAX_101_SHIFT_LOGIT_DELTA,
        note="101 successive synthetic gauge shifts; float32",
    )

    # Training/evaluation log.
    runs = run_log["runs"]
    run_seeds = sorted(run["seed"] for run in runs)
    _add_check(
        details, "run_log.seeds",
        run_seeds == sorted(EXPECTED_SEEDS), run_seeds, sorted(EXPECTED_SEEDS),
    )
    sample_counts = [
        {
            "seed": run["seed"],
            "train": run["samples"]["train"],
            "val": run["samples"]["val"],
            "test": run["samples"]["test"],
        }
        for run in runs
    ]
    _add_check(
        details, "run_log.sample_counts",
        all(
            row["train"] == 2400 and row["val"] == 560 and row["test"] == 560
            for row in sample_counts
        ),
        sample_counts,
        {"train": 2400, "val": 560, "test": 560},
    )
    _add_check(
        details, "run_log.root_bytes_all",
        all(run["root_bytes"] == ROOT_BYTES for run in runs),
        [run["root_bytes"] for run in runs],
        [ROOT_BYTES] * len(runs),
    )
    restore_errors = [float(run["restore_max_logit_error"]) for run in runs]
    _add_check(
        details, "run_log.restore_max_logit_error_zero",
        all(error == 0.0 for error in restore_errors),
        restore_errors, [0.0] * len(runs),
    )
    _add_check(
        details, "run_log.mirror_audit_all",
        all(run["mirror_audit"] is True for run in runs),
        [run["mirror_audit"] for run in runs],
        [True] * len(runs),
    )
    linear_errors = [float(run["native_vs_real_linear_max_logit_error"])
                     for run in runs]
    _add_check(
        details, "run_log.native_vs_real_linear_max_logit_error_bounded",
        all(error < LINEAR_LOGIT_BOUND for error in linear_errors),
        linear_errors, LINEAR_LOGIT_BOUND,
    )
    cycle_before = [float(run["cycle_before"]["accuracy"]) for run in runs]
    cycle_after = [float(run["cycle_after_large_gauge"]["accuracy"])
                   for run in runs]
    _add_check(
        details, "run_log.cycle_gauge_invariance",
        all(
            _close(before, after, MEAN_ACCURACY_TOLERANCE)
            for before, after in zip(cycle_before, cycle_after)
        ),
        cycle_after, cycle_before,
        tolerance=MEAN_ACCURACY_TOLERANCE,
    )
    drift = [float(run["cycle_max_drift_from_gauge"]) for run in runs]
    _add_check(
        details, "run_log.cycle_weight_drift_bounded",
        all(value < CYCLE_DRIFT_BOUND for value in drift),
        drift, CYCLE_DRIFT_BOUND,
    )
    raw_before = [float(run["raw_before"]["accuracy"]) for run in runs]
    raw_after = [float(run["raw_after_large_gauge"]["accuracy"]) for run in runs]
    _add_check(
        details, "run_log.raw_readout_degrades_under_gauge",
        all(after < before for before, after in zip(raw_before, raw_after)),
        raw_after, raw_before,
    )
    conventional = [float(run["conventional_same_features"]["accuracy"])
                    for run in runs]
    mean_conventional = sum(conventional) / len(conventional)
    mean_run_cycle = sum(cycle_before) / len(cycle_before)
    per_seed_control = all(
        ordinary >= cycle for ordinary, cycle in zip(conventional, cycle_before)
    )
    _add_check(
        details, "run_log.conventional_clean_classifier_ge_cycle",
        per_seed_control,
        {"conventional": conventional, "cycle": cycle_before},
        "conventional accuracy >= cycle accuracy for every seed",
    )
    _add_check(
        details, "run_log.conventional_mean_ge_tuned_cycle_mean",
        mean_conventional >= mean_cycle,
        {
            "conventional_mean": mean_conventional,
            "run_log_cycle_mean": mean_run_cycle,
            "tuned_cycle_mean": mean_cycle,
        },
        "conventional clean classifier mean >= tuned cycle mean",
        note="16x24 no-intercept linear control on the same clean features",
    )

    # Snapshot files, when supplied.
    expected_snapshots = [f"root_{seed}.bin" for seed in EXPECTED_SEEDS]
    if snapshots:
        snapshot_ok = (
            len(snapshots) == len(EXPECTED_SEEDS)
            and all(item["size"] == SNAPSHOT_BYTES for item in snapshots)
            and all(item["magic_ok"] for item in snapshots)
        )
        _add_check(
            details, "snapshots.size_and_magic",
            snapshot_ok,
            [
                {
                    "path": item["relative_path"],
                    "size": item["size"],
                    "magic_ok": item["magic_ok"],
                }
                for item in snapshots
            ],
            {
                "files": expected_snapshots,
                "size": SNAPSHOT_BYTES,
                "magic": SNAPSHOT_MAGIC.decode("ascii"),
            },
            note="8-byte NANCYC30 marker plus the exact 8 MiB root",
        )

    observed = {
        "root_bytes": native["root_bytes"],
        "native_gradient_max_abs_error": gradient,
        "gauge_weight_max_abs_error": gauge,
        "mirror_antisymmetry": native["mirror_antisymmetry"],
        "snapshot_bitwise_restore": native["snapshot_bitwise_restore"],
        "predictions": native["predictions"],
        "ctypes_wall_ms": wall_ms,
        "memory_wordnet_predictor_is_native": native[
            "memory_wordnet_predictor_is_native"
        ],
        "text_encoder_is_external": native["text_encoder_is_external"],
        "tuned": {
            "seeds": seeds,
            "mean_accuracy": {
                "cycle_test": mean_cycle,
                "shift_test": mean_shift,
                "raw_shift_test": mean_raw,
            },
            "computed_mean_accuracy": computed,
            "cycle_accuracy": [
                float(result["cycle_test"]["accuracy"]) for result in results
            ],
            "shift_accuracy": [
                float(result["shift_test"]["accuracy"]) for result in results
            ],
            "raw_shift_accuracy": [
                float(result["raw_shift_test"]["accuracy"]) for result in results
            ],
            "max_logit_delta_101_shifts": deltas_101,
            "native_root_bytes": [
                result["native_root_bytes"] for result in results
            ],
        },
        "run_log": {
            "seeds": run_seeds,
            "samples": sample_counts,
            "cycle_before_accuracy": cycle_before,
            "cycle_after_gauge_accuracy": cycle_after,
            "raw_before_accuracy": raw_before,
            "raw_after_gauge_accuracy": raw_after,
            "conventional_accuracy": conventional,
            "mean_cycle_accuracy": mean_run_cycle,
            "mean_conventional_accuracy": mean_conventional,
            "restore_max_logit_error": restore_errors,
            "native_vs_real_linear_max_logit_error": linear_errors,
            "cycle_max_drift_from_gauge": drift,
            "mirror_audit": [run["mirror_audit"] for run in runs],
            "root_bytes": [run["root_bytes"] for run in runs],
            "elapsed_seconds": float(run_log["elapsed_seconds"]),
        },
        "snapshots": snapshots,
    }
    return details, observed


# ---------------------------------------------------------------------------
# Optional local re-run
# ---------------------------------------------------------------------------


def find_compiler(
    candidates: Sequence[str] = COMPILER_CANDIDATES,
) -> str | None:
    """Return the first available C++ compiler path, or None."""
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def missing_rerun_dependencies() -> list[str]:
    """Return optional Python modules needed by the local re-run.

    WordNet and torch are not needed.  ``native_audit.py`` imports ``run.py``,
    which imports scikit-learn at module load time even though the audit itself
    only exercises the native root.
    """
    missing: list[str] = []
    for module in ("numpy", "sklearn"):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module)
    return missing


def _trim(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def run_native_audit(
    experiment_dir: Path | str,
    *,
    compiler: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    python: str | None = None,
) -> dict[str, Any]:
    """Compile and run a temporary copy of ``native_audit.py``.

    The supplied experiment directory is never modified.  A temporary copy is
    used because ``native_audit.py`` writes ``native_audit.json`` beside itself.
    """
    experiment = Path(experiment_dir).resolve()
    python_executable = python or sys.executable
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="round030-native-audit-") as tmp:
        workspace = Path(tmp)
        for name in REQUIRED_RERUN_SOURCES:
            source = experiment / name
            if not source.is_file():
                raise NativeRunError(f"required source file missing: {source}")
            shutil.copy2(source, workspace / name)

        binary = workspace / "libnative_cycle.so"
        command = [
            compiler,
            "-std=c++20",
            "-O3",
            "-shared",
            "-fPIC",
            "native_cycle.cpp",
            "-o",
            binary.name,
        ]
        try:
            compile_proc = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(0.1, timeout),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NativeBuildError(
                f"compile timed out after {timeout:g}s") from exc
        except OSError as exc:
            raise NativeBuildError(f"cannot execute compiler {compiler}: {exc}") from exc
        if compile_proc.returncode != 0:
            raise NativeBuildError(
                "compile failed with exit "
                f"{compile_proc.returncode}: "
                f"{_trim(compile_proc.stderr or compile_proc.stdout)}")
        if not binary.is_file():
            raise NativeBuildError(
                "compiler reported success but produced no shared library")

        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "2"
        try:
            run_proc = subprocess.run(
                [python_executable, "native_audit.py"],
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(0.1, timeout),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NativeRunError(
                f"native_audit.py timed out after {timeout:g}s") from exc
        except OSError as exc:
            raise NativeRunError(
                f"cannot execute native_audit.py: {exc}") from exc

        measurement: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            candidate = json.loads(run_proc.stdout)
            measurement = _mapping(candidate, "rerun stdout")
        except (json.JSONDecodeError, ArtifactError) as exc:
            parse_error = str(exc)

        output_path = workspace / "native_audit.json"
        if measurement is None and output_path.is_file():
            try:
                measurement = _load_json(output_path, "re-run native_audit.json")
                parse_error = None
            except ArtifactError as exc:
                parse_error = str(exc)

        if run_proc.returncode != 0:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": (
                    f"native_audit.py exited {run_proc.returncode}: "
                    f"{_trim(run_proc.stderr or run_proc.stdout)}"
                ),
                "re_executed": False,
                "measurement": None,
                "compiler": compiler,
                "stdout": _trim(run_proc.stdout),
                "stderr": _trim(run_proc.stderr or ""),
                "elapsed_seconds": time.perf_counter() - start,
            }
        if measurement is None:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": f"cannot parse re-run output: {parse_error}",
                "re_executed": False,
                "measurement": None,
                "compiler": compiler,
                "stdout": _trim(run_proc.stdout),
                "stderr": _trim(run_proc.stderr or ""),
                "elapsed_seconds": time.perf_counter() - start,
            }

        _validate_native_audit(measurement)
        snapshot_removed = not (workspace / "audit_root.bin").exists()
        return {
            "label": SOURCE_RERUN_LABEL,
            "status": "ok",
            "reason": None,
            "re_executed": True,
            "measurement": measurement,
            "compiler": compiler,
            "compile": {
                "command": command,
                "returncode": compile_proc.returncode,
                "stderr": _trim(compile_proc.stderr or ""),
            },
            "stdout_sha256": hashlib.sha256(
                run_proc.stdout.encode("utf-8")).hexdigest(),
            "stderr": _trim(run_proc.stderr or ""),
            "snapshot_removed": snapshot_removed,
            "elapsed_seconds": time.perf_counter() - start,
            "workspace": "temporary workspace removed after the run",
        }


def _rerun_checks(
    rerun: dict[str, Any],
    artifact_native: dict[str, Any],
) -> list[dict[str, Any]]:
    measurement = rerun["measurement"]
    details: list[dict[str, Any]] = []
    _add_check(
        details, "rerun.root_bytes",
        measurement["root_bytes"] == ROOT_BYTES,
        measurement["root_bytes"], ROOT_BYTES,
        source=SOURCE_RERUN,
    )
    gradient = float(measurement["native_gradient_max_abs_error"])
    _add_check(
        details, "rerun.gradient_max_abs_error_bound",
        gradient <= GRADIENT_BOUND,
        gradient, GRADIENT_BOUND,
        source=SOURCE_RERUN,
    )
    gauge = float(measurement["gauge_weight_max_abs_error"])
    _add_check(
        details, "rerun.gauge_weight_max_abs_error_bound",
        gauge <= GAUGE_BOUND,
        gauge, GAUGE_BOUND,
        source=SOURCE_RERUN,
    )
    _add_check(
        details, "rerun.mirror_antisymmetry",
        measurement["mirror_antisymmetry"] is True,
        measurement["mirror_antisymmetry"], True,
        source=SOURCE_RERUN,
    )
    _add_check(
        details, "rerun.snapshot_bitwise_restore",
        measurement["snapshot_bitwise_restore"] is True,
        measurement["snapshot_bitwise_restore"], True,
        source=SOURCE_RERUN,
    )
    _add_check(
        details, "rerun.predictions",
        measurement["predictions"] == EXPECTED_PREDICTIONS,
        measurement["predictions"], EXPECTED_PREDICTIONS,
        source=SOURCE_RERUN,
    )
    _add_check(
        details, "rerun.ctypes_wall_ms",
        math.isfinite(float(measurement["ctypes_wall_ms"]))
        and float(measurement["ctypes_wall_ms"]) >= 0.0,
        float(measurement["ctypes_wall_ms"]),
        "finite and non-negative",
        source=SOURCE_RERUN,
        note="host-only timing; not a benchmark",
    )
    structural_match = (
        measurement["root_bytes"] == artifact_native["root_bytes"]
        and measurement["predictions"] == artifact_native["predictions"]
        and measurement["mirror_antisymmetry"]
        == artifact_native["mirror_antisymmetry"]
        and measurement["snapshot_bitwise_restore"]
        == artifact_native["snapshot_bitwise_restore"]
    )
    _add_check(
        details, "rerun.structural_matches_artifact",
        structural_match,
        {
            "root_bytes": measurement["root_bytes"],
            "predictions": measurement["predictions"],
            "mirror_antisymmetry": measurement["mirror_antisymmetry"],
            "snapshot_bitwise_restore": measurement[
                "snapshot_bitwise_restore"
            ],
        },
        {
            "root_bytes": artifact_native["root_bytes"],
            "predictions": artifact_native["predictions"],
            "mirror_antisymmetry": artifact_native["mirror_antisymmetry"],
            "snapshot_bitwise_restore": artifact_native[
                "snapshot_bitwise_restore"
            ],
        },
        source=SOURCE_RERUN,
    )
    rerun["comparison"] = {
        "root_bytes_match": (
            measurement["root_bytes"] == artifact_native["root_bytes"]
        ),
        "predictions_match": (
            measurement["predictions"] == artifact_native["predictions"]
        ),
        "gradient_error_delta": abs(
            gradient - float(artifact_native["native_gradient_max_abs_error"])
        ),
        "gauge_weight_error_delta": abs(
            gauge - float(artifact_native["gauge_weight_max_abs_error"])
        ),
        "structural_match": structural_match,
        "note": (
            "The artifact values remain the supplied measured artifact; "
            "these values are a separate local re-run."
        ),
    }
    return details


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _claims(native_executed: bool = False) -> dict[str, Any]:
    return {
        "canonical": False,
        "canonical_implementation": False,
        "language_model_win": False,
        "lm_win": False,
        "language_win": False,
        "nan_unique_advantage": False,
        "nan_unique_benefit": False,
        "semantic_superiority": False,
        "quantum_claim": False,
        "android_benchmark": False,
        "portable_throughput": False,
        "native_benchmark": False,
        "native_execution": bool(native_executed),
        "scope": (
            "Round 030 artifact verification only. The supplied measurements "
            "are a measured artifact; a local re-run is labeled separately. "
            "No language-model/LM win, semantic superiority, canonical "
            "implementation, quantum, benchmark or NaN-unique benefit is "
            "claimed. The conventional clean linear control has mean accuracy "
            "at least equal to the tuned cycle mean and is at least equal to "
            "the run-log cycle readout on every supplied seed."
        ),
    }


def _boundaries() -> list[str]:
    return [
        "The reported numbers are parsed from checksum-verified supplied "
        "files: a measured artifact, not a fresh training run.",
        "Values produced by the optional local execution are labeled re-run "
        "and are not the supplied artifact.",
        "This is a semantic classifier, not a generative language model. No "
        "language-model or LM win is claimed.",
        "The C++ root stores ordinary float32 values in a struct; IEEE NaN "
        "arithmetic and any NaN-unique benefit are not claimed.",
        "A conventional clean 16x24 linear classifier has a mean accuracy "
        ">= the tuned cycle mean and is >= the run-log cycle readout on every "
        "supplied seed; gauge invariance is not semantic superiority.",
        "Gauge shifts are synthetic and intentionally chosen; this is not a "
        "natural-corpus language result.",
        "The ctypes wall time is host-only and is not an Android, portable or "
        "native batched benchmark.",
        "Snapshots are experimental: not concurrency-safe, endian-neutral, "
        "transactional or crash-safe.",
        "No canonical implementation or quantum claim is made.",
    ]


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
        "host": (
            "Termux/Android"
            if os.environ.get("TERMUX_VERSION")
            or os.environ.get("ANDROID_ROOT")
            else platform.system() or "unknown"
        ),
    }


def _measurement_summary(observed: dict[str, Any] | None) -> dict[str, Any] | None:
    if observed is None:
        return None
    tuned = observed["tuned"]
    run_log = observed["run_log"]
    return {
        "root_bytes": observed["root_bytes"],
        "mean_tuned_accuracy": tuned["mean_accuracy"]["cycle_test"],
        "mean_shifted_accuracy": tuned["mean_accuracy"]["shift_test"],
        "mean_raw_shift_accuracy": tuned["mean_accuracy"]["raw_shift_test"],
        "native_gradient_max_abs_error": observed[
            "native_gradient_max_abs_error"
        ],
        "gauge_weight_max_abs_error": observed["gauge_weight_max_abs_error"],
        "mirror_antisymmetry": observed["mirror_antisymmetry"],
        "snapshot_bitwise_restore": observed["snapshot_bitwise_restore"],
        "conventional_mean_accuracy": run_log["mean_conventional_accuracy"],
        "conventional_clean_classifier_ge_cycle": (
            run_log["mean_conventional_accuracy"]
            >= tuned["mean_accuracy"]["cycle_test"]
        ),
        "source_label": SOURCE_ARTIFACT_LABEL,
    }


def _evidence_label(
    source_only: bool,
    rerun: dict[str, Any],
    status: str,
) -> str:
    if status in {"denied", "error"}:
        return "none (verification did not complete)"
    if source_only:
        return "measured artifact only (source-only)"
    if rerun.get("status") == "ok":
        return "measured artifact plus local re-run"
    if rerun.get("status") == "failed":
        return "measured artifact only (local re-run failed)"
    return "measured artifact only (no local re-run)"


def _rerun_skip(
    *,
    requested: bool,
    reason: str,
    missing: Sequence[str] | None = None,
    compiler: str | None = None,
) -> dict[str, Any]:
    return {
        "label": SOURCE_RERUN_LABEL,
        "status": "skipped" if requested else "not_requested",
        "reason": reason,
        "re_executed": False,
        "missing_dependencies": list(missing or []),
        "compiler": compiler,
        "measurement": None,
        "comparison": None,
        "dependencies": {
            "required": ["numpy", "scikit-learn", "C++ compiler"],
            "missing": list(missing or []),
            "wordnet_required": False,
            "torch_required": False,
            "note": (
                "native_audit.py itself does not read WordNet; run.py imports "
                "scikit-learn at module load time."
            ),
        },
    }


def _base_report(
    *,
    experiment: Path,
    manifest_path: Path | None,
    status: str,
    exit_code: int,
    mode: str,
    verification: dict[str, Any],
    evidence_label: str,
    rerun: dict[str, Any],
    checks: dict[str, Any],
    check_details: list[dict[str, Any]],
    targets: dict[str, Any],
    observed: dict[str, Any] | None,
    artifacts: dict[str, Any] | None,
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "exit_code": exit_code,
        "mode": mode,
        "evidence_label": evidence_label,
        "environment": _environment(),
        "experiment_dir": str(experiment),
        "evidence": {
            "artifact": {
                "label": SOURCE_ARTIFACT_LABEL,
                "re_executed": False,
                "experiment_dir": str(experiment),
                "manifest": (
                    str(manifest_path) if manifest_path is not None else None
                ),
                "note": (
                    "Supplied measurements parsed from checksum-verified "
                    "files; not a local training run."
                ),
            },
            "rerun": {
                "label": SOURCE_RERUN_LABEL,
                "status": rerun.get("status"),
                "re_executed": bool(rerun.get("re_executed")),
                "reason": rerun.get("reason"),
                "missing_dependencies": rerun.get(
                    "missing_dependencies", []),
                "compiler": rerun.get("compiler"),
                "measurement": rerun.get("measurement"),
                "comparison": rerun.get("comparison"),
                "dependencies": rerun.get("dependencies"),
                "compile": rerun.get("compile"),
                "elapsed_seconds": rerun.get("elapsed_seconds"),
                "snapshot_removed": rerun.get("snapshot_removed"),
                "note": (
                    "Values from this run are a re-run, not the supplied "
                    "measured artifact."
                ),
            },
            "label": evidence_label,
        },
        "verification": verification,
        "targets": targets,
        "measurements": _measurement_summary(observed),
        "observed": observed,
        "artifacts": (
            {
                "native_audit": {
                    "path": artifacts["native_audit"]["path"],
                    "sha256": artifacts["native_audit"]["sha256"],
                    "data": artifacts["native_audit"]["data"],
                },
                "tuned_results": {
                    "path": artifacts["tuned_results"]["path"],
                    "sha256": artifacts["tuned_results"]["sha256"],
                    "data": artifacts["tuned_results"]["data"],
                },
                "run_log": {
                    "path": artifacts["run_log"]["path"],
                    "sha256": artifacts["run_log"]["sha256"],
                    "data": artifacts["run_log"]["data"],
                },
                "snapshots": artifacts["snapshots"],
            }
            if artifacts is not None
            else None
        ),
        "checks": checks,
        "check_details": check_details,
        "claims": _claims(bool(rerun.get("status") == "ok")),
        "boundaries": _boundaries(),
        "notes": list(notes),
    }


def _denied_report(
    *,
    experiment: Path,
    manifest_path: Path | None,
    reason: str,
    source_only: bool,
) -> dict[str, Any]:
    detail = {
        "name": "manifest.verified",
        "passed": False,
        "observed": "denied",
        "expected": "verified",
        "source": SOURCE_ARTIFACT,
        "source_label": SOURCE_ARTIFACT_LABEL,
        "tolerance": None,
        "note": reason,
    }
    rerun = _rerun_skip(
        requested=False,
        reason="integrity verification failed before any re-run",
    )
    return _base_report(
        experiment=experiment,
        manifest_path=manifest_path,
        status="denied",
        exit_code=EXIT_DENIED,
        mode="source-only" if source_only else "artifact-only",
        verification={
            "status": "denied",
            "reason": reason,
            "manifest": str(manifest_path) if manifest_path else None,
            "entry_count": 0,
            "verified_count": 0,
            "verified": [],
            "digests": {},
            "required_files": list(REQUIRED_MANIFEST_FILES),
            "required_files_covered": False,
            "source_sha256": None,
        },
        evidence_label="none (verification denied)",
        rerun=rerun,
        checks={"manifest.verified": False, "all_targets_met": False},
        check_details=[detail],
        targets=_targets(),
        observed=None,
        artifacts=None,
        notes=[
            "No artifact was parsed and no native code was compiled or run "
            "because the supplied digest manifest did not verify."
        ],
    )


def _artifact_error_report(
    *,
    experiment: Path,
    manifest_path: Path | None,
    artifact_path: Path | None,
    reason: str,
    verification: dict[str, Any],
    source_only: bool,
) -> dict[str, Any]:
    detail = {
        "name": "artifact.parsed",
        "passed": False,
        "observed": str(artifact_path) if artifact_path else "invalid",
        "expected": "valid Round 030 artifact structure",
        "source": SOURCE_ARTIFACT,
        "source_label": SOURCE_ARTIFACT_LABEL,
        "tolerance": None,
        "note": reason,
    }
    rerun = _rerun_skip(
        requested=False,
        reason="artifact parsing failed before any re-run",
    )
    return _base_report(
        experiment=experiment,
        manifest_path=manifest_path,
        status="error",
        exit_code=EXIT_ERROR,
        mode="source-only" if source_only else "artifact-only",
        verification=verification,
        evidence_label="none (artifact parsing failed)",
        rerun=rerun,
        checks={"artifact.parsed": False, "all_targets_met": False},
        check_details=[detail],
        targets=_targets(),
        observed=None,
        artifacts=None,
        notes=[f"Artifact parsing failed: {reason}"],
    )


def run_audit(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    *,
    source_only: bool = False,
    manifest: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    skip_rerun: bool = False,
) -> dict[str, Any]:
    """Verify, parse and (optionally) locally re-run the Round 030 audit."""
    experiment = Path(experiment_dir)
    if not experiment.is_absolute():
        experiment = (ROOT / experiment).resolve()
    else:
        experiment = experiment.resolve()

    manifest_path = (
        Path(manifest) if manifest is not None
        else experiment / DEFAULT_MANIFEST_NAME
    )
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()

    try:
        verification = verify_artifacts(experiment, manifest_path)
    except VerificationError as exc:
        return _denied_report(
            experiment=experiment,
            manifest_path=manifest_path,
            reason=str(exc),
            source_only=source_only,
        )

    try:
        artifacts = load_artifacts(experiment, verification)
    except ArtifactError as exc:
        return _artifact_error_report(
            experiment=experiment,
            manifest_path=manifest_path,
            artifact_path=experiment,
            reason=str(exc),
            verification=verification,
            source_only=source_only,
        )

    details, observed = evaluate_artifacts(artifacts, verification)
    notes = [
        "The three parsed artifacts were covered by the verified SHA256SUMS "
        "manifest.",
        "Tuned training was not re-executed; WordNet and the 24D SVD encoder "
        "are outside this verification.",
    ]

    if source_only:
        rerun = _rerun_skip(
            requested=False,
            reason="--source-only requested; no compile or run performed",
        )
        mode = "source-only"
    elif skip_rerun:
        rerun = _rerun_skip(
            requested=False,
            reason="re-run disabled by the caller",
        )
        mode = "artifact-only"
    else:
        compiler = find_compiler()
        missing = missing_rerun_dependencies()
        missing_sources = [
            name for name in REQUIRED_RERUN_SOURCES
            if not (experiment / name).is_file()
        ]
        if missing or compiler is None or missing_sources:
            reasons = []
            if missing:
                reasons.append(
                    "missing optional Python module(s): " + ", ".join(missing))
            if compiler is None:
                reasons.append("no C++ compiler found")
            if missing_sources:
                reasons.append(
                    "missing re-run source file(s): " + ", ".join(missing_sources))
            rerun = _rerun_skip(
                requested=True,
                reason="; ".join(reasons),
                missing=missing,
                compiler=compiler,
            )
            mode = "artifact-only"
            notes.append(
                "Optional local re-run skipped; only the supplied measured "
                "artifact was verified.")
        else:
            try:
                rerun = run_native_audit(
                    experiment, compiler=compiler, timeout=timeout)
            except (NativeBuildError, NativeRunError, ArtifactError) as exc:
                rerun = {
                    "label": SOURCE_RERUN_LABEL,
                    "status": "failed",
                    "reason": str(exc),
                    "re_executed": False,
                    "missing_dependencies": [],
                    "compiler": compiler,
                    "measurement": None,
                    "comparison": None,
                }
                mode = "artifact-only"
                notes.append(
                    "Optional local re-run failed; the supplied measured "
                    "artifact checks still stand.")
            else:
                if rerun.get("status") == "ok":
                    details.extend(
                        _rerun_checks(
                            rerun,
                            artifacts["native_audit"]["data"],
                        )
                    )
                    mode = "re-run"
                    notes.append(
                        "A local re-run was executed in a temporary workspace; "
                        "its values are labeled re-run and kept separate from "
                        "the supplied measured artifact.")
                else:
                    mode = "artifact-only"
                    notes.append(
                        "Optional local re-run did not complete; only the "
                        "supplied measured artifact was verified.")

    checks = {detail["name"]: bool(detail["passed"]) for detail in details}
    all_passed = all(detail["passed"] for detail in details)
    checks["all_targets_met"] = all_passed
    status = "ok" if all_passed else "failed"
    exit_code = EXIT_OK if all_passed else EXIT_FAILED

    evidence_label = _evidence_label(source_only, rerun, status)
    return _base_report(
        experiment=experiment,
        manifest_path=manifest_path,
        status=status,
        exit_code=exit_code,
        mode=mode,
        verification=verification,
        evidence_label=evidence_label,
        rerun=rerun,
        checks=checks,
        check_details=details,
        targets=_targets(),
        observed=observed,
        artifacts=artifacts,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a deterministic Markdown report."""
    lines = [
        "# Round 030 native audit verification",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Mode:** `{report['mode']}`",
        f"- **Evidence:** {report['evidence_label']}",
        f"- **Experiment directory:** `{report['experiment_dir']}`",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        "",
        "## Digest verification",
        "",
    ]
    verification = report["verification"]
    lines += [
        f"- Manifest: `{verification.get('manifest') or 'n/a'}`",
        f"- Status: `{verification.get('status')}`",
        f"- Verified entries: {verification.get('verified_count', 0)} / "
        f"{verification.get('entry_count', 0)}",
        f"- Required artifacts covered: "
        f"{'yes' if verification.get('required_files_covered') else 'no'}",
    ]
    if verification.get("reason"):
        lines.append(f"- Reason: {verification['reason']}")
    lines.append("")

    observed = report.get("observed")
    if observed is None:
        lines += [
            "## Result",
            "",
            "No artifact measurements are available because verification or "
            "parsing did not complete.",
            "",
        ]
    else:
        tuned = observed["tuned"]
        run_log = observed["run_log"]
        lines += [
            "## Supplied measured artifact",
            "",
            "These values come from the checksum-verified supplied files. "
            "They are **not** values produced by this script.",
            "",
            "| Measurement | Observed | Target / note |",
            "| --- | ---: | --- |",
            f"| Root bytes | {_fmt(observed['root_bytes'])} | 8388608 (8 MiB) |",
            f"| Mean tuned cycle accuracy | "
            f"{_fmt(tuned['mean_accuracy']['cycle_test'])} | 0.420833 |",
            f"| Mean tuned shifted accuracy | "
            f"{_fmt(tuned['mean_accuracy']['shift_test'])} | 0.420833 |",
            f"| Mean raw shift accuracy | "
            f"{_fmt(tuned['mean_accuracy']['raw_shift_test'])} | 0.082142 |",
            f"| Native gradient max abs error | "
            f"{_fmt(observed['native_gradient_max_abs_error'], 8)} | "
            "1.86e-9 |",
            f"| Gauge weight max abs error | "
            f"{_fmt(observed['gauge_weight_max_abs_error'], 8)} | "
            "4.17e-7 |",
            f"| Mirror antisymmetry | "
            f"{_fmt(observed['mirror_antisymmetry'])} | supplied audit |",
            f"| Snapshot bitwise restore | "
            f"{_fmt(observed['snapshot_bitwise_restore'])} | supplied audit |",
            f"| Predictions | {_fmt(observed['predictions'])} | 1000 |",
            f"| ctypes wall time (host-only) | "
            f"{_fmt(observed['ctypes_wall_ms'], 5)} ms | not a benchmark |",
            f"| Conventional clean classifier mean | "
            f"{_fmt(run_log['mean_conventional_accuracy'])} | "
            f">= cycle mean {_fmt(tuned['mean_accuracy']['cycle_test'])} |",
            f"| Snapshot files | {len(observed['snapshots'])} checked | "
            "8-byte NANCYC30 + 8 MiB root |",
            "",
            "### Tuned per-seed evidence",
            "",
            "| Seed | Cycle acc. | Shifted acc. | Raw shift acc. | "
            "101-shift max logit delta |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for index, seed in enumerate(tuned["seeds"]):
            lines.append(
                f"| {seed} | {_fmt(tuned['cycle_accuracy'][index])} | "
                f"{_fmt(tuned['shift_accuracy'][index])} | "
                f"{_fmt(tuned['raw_shift_accuracy'][index])} | "
                f"{_fmt(tuned['max_logit_delta_101_shifts'][index])} |"
            )
        lines += [
            "",
            "### Training-log conventional control",
            "",
            "| Seed | Cycle before | Cycle after gauge | Conventional acc. | "
            "Restore error |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for index, seed in enumerate(run_log["seeds"]):
            lines.append(
                f"| {seed} | {_fmt(run_log['cycle_before_accuracy'][index])} | "
                f"{_fmt(run_log['cycle_after_gauge_accuracy'][index])} | "
                f"{_fmt(run_log['conventional_accuracy'][index])} | "
                f"{_fmt(run_log['restore_max_logit_error'][index])} |"
            )
        lines += [
            "",
            "The conventional control and its cycle baseline come from "
            "`run.log` using the default training schedule; the tuned "
            "accuracies above come from `tuned_results.json`. Per seed, the "
            "conventional no-intercept linear classifier is at least as "
            "accurate as the corresponding run-log cycle readout, and its mean "
            "also exceeds the tuned cycle mean. Gauge invariance is a "
            "controlled synthetic property, not a language-model result.",
            "",
        ]

        lines += [
            "## Checks",
            "",
            f"{sum(1 for item in report['check_details'] if item['passed'])}"
            f" / {len(report['check_details'])} checks passed.",
            "",
            "| Check | Source | Result | Observed | Expected |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in report["check_details"]:
            observed_text = _fmt(item["observed"])
            expected_text = _fmt(item["expected"])
            if len(observed_text) > 100:
                observed_text = observed_text[:97] + "..."
            if len(expected_text) > 100:
                expected_text = expected_text[:97] + "..."
            lines.append(
                f"| `{item['name']}` | {item['source_label']} | "
                f"{'pass' if item['passed'] else 'FAIL'} | "
                f"{observed_text} | {expected_text} |"
            )
        lines.append("")

    rerun = report["evidence"]["rerun"]
    lines += [
        "## Local re-run",
        "",
        f"- Label: **{rerun['label']}**",
        f"- Status: `{rerun['status']}`",
    ]
    if rerun.get("reason"):
        lines.append(f"- Reason: {rerun['reason']}")
    if rerun.get("measurement"):
        measurement = rerun["measurement"]
        lines += [
            f"- Root bytes: {_fmt(measurement.get('root_bytes'))}",
            f"- Gradient max abs error: "
            f"{_fmt(measurement.get('native_gradient_max_abs_error'), 8)}",
            f"- Gauge weight max abs error: "
            f"{_fmt(measurement.get('gauge_weight_max_abs_error'), 8)}",
            f"- Mirror antisymmetry: "
            f"{_fmt(measurement.get('mirror_antisymmetry'))}",
            f"- Snapshot bitwise restore: "
            f"{_fmt(measurement.get('snapshot_bitwise_restore'))}",
        ]
    else:
        lines.append(
            "- No local values were produced; the artifact-only result stands.")
    lines += [
        "",
        "## Boundaries",
        "",
    ]
    for note in report["boundaries"]:
        lines.append(f"- {note}")
    if report.get("notes"):
        lines += ["", "## Notes", ""]
        for note in report["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path | str, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path | str, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(report), encoding="utf-8")


def _receipt(
    report: dict[str, Any],
    out_json: str | None,
    out_markdown: str | None,
) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "status": report["status"],
        "mode": report["mode"],
        "evidence_label": report["evidence_label"],
        "exit_code": report["exit_code"],
        "checks_all_targets_met": report["checks"].get("all_targets_met"),
        "rerun_status": report["evidence"]["rerun"]["status"],
        "json": out_json,
        "markdown": out_markdown,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the supplied Round 030 native-audit artifacts without "
            "requiring WordNet, torch or scikit-learn; optionally compile and "
            "re-run native_audit.py when dependencies are available."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(DEFAULT_EXPERIMENT_DIR),
        help=(
            "Round 030 experiment directory "
            f"(default: {DEFAULT_EXPERIMENT_DIR})"
        ),
    )
    parser.add_argument(
        "--sha256sums",
        default=None,
        help="digest manifest (default: <experiment-dir>/SHA256SUMS)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the JSON report to this path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="write the Markdown report to this path",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "skip the optional compile/run and verify the supplied artifacts "
            "only"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "bounded compile and run timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be a positive finite number of seconds")

    report = run_audit(
        experiment_dir=args.experiment_dir,
        source_only=args.source_only,
        manifest=args.sha256sums,
        timeout=float(args.timeout),
    )

    out_json = str(args.json) if args.json is not None else None
    out_markdown = str(args.markdown) if args.markdown is not None else None
    if args.json is not None:
        _write_json(args.json, report)
    if args.markdown is not None:
        _write_markdown(args.markdown, report)

    if args.json is None and args.markdown is None:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(
            _receipt(report, out_json, out_markdown),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
