#!/usr/bin/env python3
"""Audit Round 031 and expose its 256-weight versus 32,385-cycle limitation.

Round 031 stores a byte-level language model's 16x16 cycle readout inside an
8 MiB C++ "native" root.  ``nan_get`` exposes only ``D * K = 16 * 16 = 256``
triangle fluxes.  The complete 256-head star-basis cycle space instead has
``C(255, 2) = 32,385`` independent cycles, so 32,129 cycle degrees are unused.

This script first verifies the supplied ``SHA256SUMS`` manifest and parses the
checksum-verified artifacts:

* ``results.json`` - WordNet train/validation/test runs for seeds 17 and 29;
* ``audit.json`` - native mirror/gauge/snapshot checks and 2-hop graph counts;
* ``transfer_results.json`` - short Alice transfer results;
* ``README.md``, ``native_cycle.cpp`` and ``train.py`` - stated limits and the
  actual 16x16 readout shape.

It then reports the WordNet test BpB difference between the conventional
16x16 parameter matrix and the native cycle model, the gauge invariance tests,
the memory ablation, snapshot integrity and symbolic 2-hop counts.

The supplied values are labeled **measured artifact**.  If an optional
``full_cycle_memory`` provider is available (``--full-cycle-memory``), a
deterministic **synthetic re-run** trains and audits all 32,385 star-basis
coefficients in a plain Python in-memory full-basis memory.  That synthetic
audit is not the supplied native model and does not turn the 8 MiB root into a
full-basis implementation.  Without a provider the report contains a
structured skip.

No language-model win, semantic superiority, canonical implementation,
NaN-unique advantage, quantum effect or portable/native benchmark claim is
made.  The supplied conventional 16x16 baseline has lower WordNet test BpB on
both seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import struct
import sys
import time
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

SCHEMA = "nanlmind-hermes.round031-full-cycle-audit/1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "round031" / "nan_round031"
DEFAULT_MANIFEST_NAME = "SHA256SUMS"

HEADS = 256
D = 16
K = 16
PAIRS = HEADS * (HEADS - 1) // 2                 # 32,640
EFFECTIVE_CYCLES = D * K                         # 256
FULL_BASIS_CYCLES = (HEADS - 1) * (HEADS - 2) // 2  # 32,385
UNUSED_CYCLES = FULL_BASIS_CYCLES - EFFECTIVE_CYCLES
ROOT_BYTES = 8_388_608
SNAPSHOT_MAGIC = b"NANCYC30"
SNAPSHOT_BYTES = ROOT_BYTES + len(SNAPSHOT_MAGIC)

EXPECTED_SEEDS = (17, 29)
EXPECTED_RUNS = tuple(
    f"{kind}_{seed}" for seed in EXPECTED_SEEDS for kind in ("plain", "native")
)
EXPECTED_SNAPSHOTS = tuple(
    f"{prefix}root_{seed}.bin"
    for prefix in ("", "transfer_")
    for seed in EXPECTED_SEEDS
)

SPLIT_COUNT = 1800
WORDNET_TEST_BYTES = 13_200
WORDNET_VAL_BYTES = 7_061
ABLATION_BYTES = 7_319
ALICE_VAL_BYTES = 324
ALICE_TEST_BYTES = 288

WINDOW_READ_BOUNDS = (0.005, 0.010)  # milliseconds; README says ~6-7 us.
GAUGE_WEIGHT_BOUND = 1e-5
GAUGE_BPB_BOUND = 1e-6
README_ROUNDING_TOLERANCE = 5e-6
SYNTHETIC_ERROR_BOUND = 1e-12
SYNTHETIC_GAUGE_BOUND = 1e-9

SOURCE_ARTIFACT = "measured_artifact"
SOURCE_ARTIFACT_LABEL = "measured artifact"
SOURCE_SYNTHETIC = "synthetic_rerun"
SOURCE_SYNTHETIC_LABEL = "synthetic re-run"

FULL_CYCLE_MEMORY_ENV = "NAN_ROUND031_FULL_CYCLE_MEMORY"
RESIDENT_TRAINING_MODULE = "nanlmind_hermes.resident.full_cycle_training"
RESIDENT_CYCLE_MODULE = "nanlmind_hermes.resident.full_cycle_memory"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2
EXIT_DENIED = 3

_SUM_LINE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})[ \t]+(?P<name>\*?.+)$")
_README_TABLE_ROW = re.compile(
    r"^\|\s*(17|29)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$",
    re.MULTILINE,
)


class VerificationError(RuntimeError):
    """Raised when SHA256SUMS verification must deny execution."""


class ArtifactError(RuntimeError):
    """Raised when a supplied artifact is missing, malformed or inconsistent."""


class FullCycleMemoryError(RuntimeError):
    """Raised when an optional full-cycle-memory provider cannot be loaded."""


# ---------------------------------------------------------------------------
# Cycle-count invariants
# ---------------------------------------------------------------------------


def cycle_count_invariants() -> dict[str, Any]:
    """Assert and return the effective/full cycle counts.

    The supplied model has a 16x16 readout: ``D * K = 256`` effective triangle
    weights.  The complete 256-head star basis has ``C(255, 2) = 32,385``
    independent cycles.  The difference is 32,129 unused cycle degrees.
    """
    if EFFECTIVE_CYCLES != D * K:
        raise AssertionError("effective cycle count is not D * K")
    if EFFECTIVE_CYCLES != 16 * 16:
        raise AssertionError("effective cycle count is not 16 * 16")
    if FULL_BASIS_CYCLES != math.comb(HEADS - 1, 2):
        raise AssertionError("full basis is not C(HEADS-1, 2)")
    if FULL_BASIS_CYCLES != 32_385:
        raise AssertionError("full basis is not 32,385")
    if FULL_BASIS_CYCLES != PAIRS - (HEADS - 1):
        raise AssertionError("full basis is not pairs - gauge dimensions")
    if UNUSED_CYCLES != FULL_BASIS_CYCLES - EFFECTIVE_CYCLES:
        raise AssertionError("unused cycle count is inconsistent")
    return {
        "heads": HEADS,
        "effective_dimension": D,
        "channels": K,
        "effective_cycles": EFFECTIVE_CYCLES,
        "effective_formula": "D * K = 16 * 16 = 256",
        "full_basis_cycles": FULL_BASIS_CYCLES,
        "full_basis_formula": "C(255, 2) = 255 * 254 / 2 = 32,385",
        "pair_capacity": PAIRS,
        "pair_formula": "C(256, 2) = 32,640",
        "gauge_dimension": HEADS - 1,
        "cycle_space_formula": "C(256, 2) - 255 = 32,385",
        "unused_cycles": UNUSED_CYCLES,
        "effective_fraction": EFFECTIVE_CYCLES / FULL_BASIS_CYCLES,
        "unused_fraction": UNUSED_CYCLES / FULL_BASIS_CYCLES,
        "root_bytes": ROOT_BYTES,
    }


# ---------------------------------------------------------------------------
# Digest verification
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
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
    except OSError as exc:
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
                "expected '<sha256>  <path>'"
            )
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


REQUIRED_MANIFEST_FILES = (
    "README.md",
    "native_cycle.cpp",
    "train.py",
    "audit.py",
    "transfer.py",
    "final_audit.py",
    "results.json",
    "audit.json",
    "transfer_results.json",
    "train.log",
    "transfer.log",
    "root_17.bin",
    "root_29.bin",
    "transfer_root_17.bin",
    "transfer_root_29.bin",
)


def verify_artifacts(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Verify every digest and require the Round 031 artifact set."""
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
        measured = sha256_file(candidate)
        if measured != entry["sha256"]:
            errors.append(
                f"SHA-256 mismatch for {name}: expected {entry['sha256']}, "
                f"measured {measured}"
            )
            continue
        verified.append({"path": name, "sha256": measured})
        digests[name] = measured

    if errors:
        raise VerificationError("; ".join(errors))

    missing_required = [
        name for name in REQUIRED_MANIFEST_FILES if name not in digests
    ]
    if missing_required:
        raise VerificationError(
            "manifest does not cover required artifact(s): "
            + ", ".join(missing_required)
        )

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
        "source_sha256": digests.get("native_cycle.cpp"),
    }


# ---------------------------------------------------------------------------
# JSON parsing helpers and structural validation
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


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read {name}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid JSON in {name}: {exc}") from exc
    return _mapping(data, name)


def _validate_metric(
    block: Any,
    name: str,
    *,
    n_bytes: int,
) -> dict[str, Any]:
    data = _mapping(block, name)
    if _integer(_get(data, "n_bytes", name), f"{name}.n_bytes") != n_bytes:
        raise ArtifactError(f"{name}.n_bytes must be {n_bytes}")
    _number(_get(data, "bits_per_byte", name), f"{name}.bits_per_byte",
            minimum=0.0)
    _number(_get(data, "accuracy", name), f"{name}.accuracy",
            minimum=0.0, maximum=1.0)
    return data


def _validate_results(data: dict[str, Any]) -> None:
    counts = _mapping(_get(data, "counts", "results.json"), "results.json.counts")
    for split in ("train", "val", "test"):
        if _integer(_get(counts, split, "results.json.counts"),
                    f"counts.{split}") != SPLIT_COUNT:
            raise ArtifactError(f"results.json.counts.{split} must be {SPLIT_COUNT}")

    runs = _mapping(_get(data, "runs", "results.json"), "results.json.runs")
    if set(runs) != set(EXPECTED_RUNS):
        raise ArtifactError(
            f"results.json.runs keys must be {list(EXPECTED_RUNS)}")

    for key in EXPECTED_RUNS:
        run = _mapping(runs[key], f"results.json.runs[{key}]")
        context = f"results.json.runs[{key}]"
        if _integer(_get(run, "best_step", context),
                    f"{context}.best_step") != 100:
            raise ArtifactError(f"{context}.best_step must be 100")
        _integer(_get(run, "params", context), f"{context}.params")
        _validate_metric(_get(run, "test", context), f"{context}.test",
                         n_bytes=WORDNET_TEST_BYTES)
        history = _list(_get(run, "validation", context),
                        f"{context}.validation", length=5)
        steps = []
        for index, raw in enumerate(history):
            entry = _mapping(raw, f"{context}.validation[{index}]")
            steps.append(_integer(entry.get("step"),
                                  f"{context}.validation[{index}].step"))
            _validate_metric(entry, f"{context}.validation[{index}]",
                             n_bytes=WORDNET_VAL_BYTES)
        if steps != [1, 25, 50, 75, 100]:
            raise ArtifactError(f"{context}.validation steps must be 1,25,50,75,100")

    for seed in EXPECTED_SEEDS:
        key = f"native_{seed}"
        run = _mapping(runs[key], f"results.json.runs[{key}]")
        context = f"results.json.runs[{key}]"
        gauge = _mapping(_get(run, "gauge_test", context),
                         f"{context}.gauge_test")
        _validate_metric(_get(gauge, "before", f"{context}.gauge_test"),
                         f"{context}.gauge_test.before", n_bytes=ABLATION_BYTES)
        _validate_metric(_get(gauge, "after", f"{context}.gauge_test"),
                         f"{context}.gauge_test.after", n_bytes=ABLATION_BYTES)
        _number(_get(gauge, "max_weight_diff", f"{context}.gauge_test"),
                f"{context}.gauge_test.max_weight_diff", minimum=0.0)
        _boolean(_get(gauge, "audit", f"{context}.gauge_test"),
                 f"{context}.gauge_test.audit")
        _validate_metric(_get(run, "memory_disabled", context),
                         f"{context}.memory_disabled", n_bytes=ABLATION_BYTES)


def _validate_audit(data: dict[str, Any]) -> None:
    native = _mapping(_get(data, "native", "audit.json"), "audit.json.native")
    if _integer(_get(native, "reads", "audit.json.native"),
                "audit.json.native.reads") != 10_000:
        raise ArtifactError("audit.json.native.reads must be 10000")
    _number(_get(native, "mean_read_ms", "audit.json.native"),
            "audit.json.native.mean_read_ms", minimum=0.0)
    _number(_get(native, "gauge_max_diff", "audit.json.native"),
            "audit.json.native.gauge_max_diff", minimum=0.0)
    _boolean(_get(native, "mirror_audit", "audit.json.native"),
             "audit.json.native.mirror_audit")
    _number(_get(native, "snapshot_max_diff", "audit.json.native"),
            "audit.json.native.snapshot_max_diff", minimum=0.0)

    wordnet = _mapping(_get(data, "wordnet", "audit.json"),
                       "audit.json.wordnet")
    for key in ("synsets", "two_hop_candidates", "two_hop_proofs_checked",
                "direct_edge_only_hits"):
        _integer(_get(wordnet, key, "audit.json.wordnet"),
                 f"audit.json.wordnet.{key}")
    _boolean(_get(wordnet, "all_chain_edges_verified", "audit.json.wordnet"),
             "audit.json.wordnet.all_chain_edges_verified")


def _validate_transfer(data: dict[str, Any]) -> None:
    if set(data) != set(EXPECTED_RUNS):
        raise ArtifactError(
            f"transfer_results.json keys must be {list(EXPECTED_RUNS)}")
    for key in EXPECTED_RUNS:
        run = _mapping(data[key], f"transfer_results.json[{key}]")
        context = f"transfer_results.json[{key}]"
        for phase in ("before", "after"):
            block = _mapping(_get(run, phase, context), f"{context}.{phase}")
            _validate_metric(_get(block, "wordnet", f"{context}.{phase}"),
                             f"{context}.{phase}.wordnet",
                             n_bytes=WORDNET_VAL_BYTES)
            _validate_metric(_get(block, "alice", f"{context}.{phase}"),
                             f"{context}.{phase}.alice",
                             n_bytes=ALICE_VAL_BYTES)
        _validate_metric(_get(run, "alice_test", context),
                         f"{context}.alice_test", n_bytes=ALICE_TEST_BYTES)
        if key.startswith("native_"):
            _validate_metric(_get(run, "alice_test_no_memory", context),
                             f"{context}.alice_test_no_memory",
                             n_bytes=ALICE_TEST_BYTES)


# ---------------------------------------------------------------------------
# Source and README parsing
# ---------------------------------------------------------------------------


def parse_native_source_constants(text: str) -> dict[str, Any]:
    """Parse the actual constants and formula used by ``native_cycle.cpp``."""
    compact = re.sub(r"\s+", "", text)
    dimension = re.search(
        r"staticconstexprintN=(\d+),D=(\d+),K=(\d+);", compact)
    capacity = re.search(
        r"staticconstexprsize_tPAIRS=(\d+),ROOT=(\d+);", compact)
    pair = re.search(r"structalignas\(64\)Pair\{floatf\[(\d+)\],r\[(\d+)\];\};",
                     compact)
    head = re.search(r"structalignas\(64\)Head\{unsignedcharbytes\[(\d+)\];\};",
                     compact)
    if not (dimension and capacity and pair and head):
        raise ArtifactError("cannot parse native_cycle.cpp constants")
    n, d, k = (int(value) for value in dimension.groups())
    pairs, root = (int(value) for value in capacity.groups())
    forward, reverse = (int(value) for value in pair.groups())
    head_bytes = int(head.group(1))
    pair_bytes = (forward + reverse) * 4
    parsed = {
        "heads": n,
        "dimension": d,
        "channels": k,
        "pairs": pairs,
        "root_bytes": root,
        "pair_forward_floats": forward,
        "pair_reverse_floats": reverse,
        "pair_bytes": pair_bytes,
        "head_bytes": head_bytes,
        "memory_bytes_formula": n * head_bytes + pairs * pair_bytes,
        "effective_cycles": d * k,
        "full_basis_cycles": (n - 1) * (n - 2) // 2,
        "has_size_static_assert": "static_assert(sizeof(Memory)==ROOT);" in compact,
        "has_flux_formula": (
            "returnedge(0,a,c)+edge(a,b,c)-edge(0,b,c);" in compact),
        "exports": {
            name: f"{name}(" in compact
            for name in (
                "nan_new", "nan_free", "nan_get", "nan_apply", "nan_gauge",
                "nan_predict", "nan_epoch", "nan_snapshot", "nan_restore",
                "nan_audit",
            )
        },
    }
    if n != HEADS or d != D or k != K:
        raise ArtifactError("native_cycle.cpp dimensions differ from the audit")
    if pairs != math.comb(n, 2):
        raise ArtifactError("native_cycle.cpp PAIRS is not C(256, 2)")
    if parsed["memory_bytes_formula"] != root or root != ROOT_BYTES:
        raise ArtifactError("native_cycle.cpp root size is inconsistent")
    if parsed["effective_cycles"] != EFFECTIVE_CYCLES:
        raise ArtifactError("native_cycle.cpp effective cycles are not 256")
    if parsed["full_basis_cycles"] != FULL_BASIS_CYCLES:
        raise ArtifactError("native_cycle.cpp full basis is not 32,385")
    if not parsed["has_size_static_assert"] or not parsed["has_flux_formula"]:
        raise ArtifactError("native_cycle.cpp is missing a size/readout invariant")
    return parsed


def parse_train_source_constants(text: str) -> dict[str, Any]:
    """Parse the training readout shape and optimizer settings."""
    match = re.search(r"^N\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise ArtifactError("cannot parse train.py readout width")
    n = int(match.group(1))
    compact = re.sub(r"\s+", "", text)
    return {
        "readout_width": n,
        "readout_dimension": n * n,
        "has_16x16_pair_parameter": (
            "self.pair=nn.Parameter(torch.zeros(N,N))" in compact),
        "has_nan_get": "nan_get(self.ptr" in compact,
        "has_nan_apply": "nan_apply(self.ptr" in compact,
        "neural_optimizer": (
            "AdamW" if "torch.optim.AdamW" in compact else "unknown"),
        "neural_learning_rate": (
            0.002 if "AdamW(params,lr=.002)" in compact else None),
        "native_update_rule": (
            "nan_apply SGD" if "root.apply(W.grad,.003)" in compact else "unknown"),
        "native_learning_rate": (
            0.003 if "root.apply(W.grad,.003)" in compact else None),
        "readout_matches_effective_cycles": n * n == EFFECTIVE_CYCLES,
    }


def parse_readme_table(text: str) -> dict[int, dict[str, float]]:
    rows = _README_TABLE_ROW.findall(text)
    if len(rows) != len(EXPECTED_SEEDS):
        raise ArtifactError(
            "README.md WordNet table must contain one row per seed")
    parsed: dict[int, dict[str, float]] = {}
    for seed_text, plain_text, native_text in rows:
        parsed[int(seed_text)] = {
            "plain_bits_per_byte": float(plain_text),
            "native_bits_per_byte": float(native_text),
        }
    if set(parsed) != set(EXPECTED_SEEDS):
        raise ArtifactError("README.md WordNet table seeds are incomplete")
    return parsed


def parse_readme_limits(text: str) -> dict[str, Any]:
    plain = text.replace("*", "")
    phrases = {
        "states_256_effective_weights": "256 effective triangle weights" in plain,
        "states_32129_unused_cycles": "32,129" in plain,
        "states_not_capacity_benefit": (
            "not a benefit over the capacity-matched conventional parameter "
            "matrix" in plain
        ),
        "states_not_fully_memory_native": "not a fully memory-native LM" in plain,
        "states_symbolic_two_hop_not_learned": (
            "not learned composition" in plain
        ),
        "states_no_quantum_effect": "No physical quantum effects" in plain,
        "states_alice_test_too_small": "288 bytes" in plain,
    }
    return phrases


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_artifacts(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and structurally validate the Round 031 artifacts."""
    experiment = Path(experiment_dir).resolve()
    digests = (verification or {}).get("digests", {})

    results_path = experiment / "results.json"
    audit_path = experiment / "audit.json"
    transfer_path = experiment / "transfer_results.json"
    readme_path = experiment / "README.md"
    native_path = experiment / "native_cycle.cpp"
    train_path = experiment / "train.py"

    results = _load_json(results_path, "results.json")
    _validate_results(results)
    audit = _load_json(audit_path, "audit.json")
    _validate_audit(audit)
    transfer = _load_json(transfer_path, "transfer_results.json")
    _validate_transfer(transfer)

    try:
        readme = readme_path.read_text(encoding="utf-8")
        native_source = native_path.read_text(encoding="utf-8")
        train_source = train_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read source/README artifact: {exc}") from exc

    snapshots: list[dict[str, Any]] = []
    for name in EXPECTED_SNAPSHOTS:
        path = experiment / name
        if not path.is_file():
            raise ArtifactError(f"snapshot file missing: {name}")
        with path.open("rb") as handle:
            magic = handle.read(len(SNAPSHOT_MAGIC))
        snapshots.append(
            {
                "relative_path": name,
                "path": str(path),
                "sha256": digests.get(name),
                "size": path.stat().st_size,
                "magic": magic.decode("ascii", errors="replace"),
                "magic_ok": magic == SNAPSHOT_MAGIC,
                "root_bytes": path.stat().st_size - len(SNAPSHOT_MAGIC),
            }
        )

    return {
        "results": {
            "path": str(results_path),
            "sha256": digests.get("results.json"),
            "data": results,
        },
        "audit": {
            "path": str(audit_path),
            "sha256": digests.get("audit.json"),
            "data": audit,
        },
        "transfer": {
            "path": str(transfer_path),
            "sha256": digests.get("transfer_results.json"),
            "data": transfer,
        },
        "readme": {
            "path": str(readme_path),
            "sha256": digests.get("README.md"),
            "text": readme,
            "table": parse_readme_table(readme),
            "limits": parse_readme_limits(readme),
        },
        "native_source": {
            "path": str(native_path),
            "sha256": digests.get("native_cycle.cpp"),
            "constants": parse_native_source_constants(native_source),
        },
        "train_source": {
            "path": str(train_path),
            "sha256": digests.get("train.py"),
            "constants": parse_train_source_constants(train_source),
        },
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Synthetic full-basis memory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def full_basis_pairs() -> tuple[tuple[int, int], ...]:
    """Return all 32,385 star-basis pairs ``(i, j)`` with ``1 <= i < j < 256``."""
    pairs = tuple(
        (first, second)
        for first in range(1, HEADS)
        for second in range(first + 1, HEADS)
    )
    if len(pairs) != FULL_BASIS_CYCLES:
        raise AssertionError("full-basis pair enumeration is incomplete")
    return pairs


def deterministic_full_basis_targets(seed: int = 2031) -> list[float]:
    """Deterministic exact-binary coefficients for all 32,385 cycles."""
    scale = 1024.0
    return [
        (((index * 97 + seed * 31 + 17) % 2009) - 1004) / scale
        for index in range(FULL_BASIS_CYCLES)
    ]


def _coefficient_list(value: Any, name: str) -> list[float]:
    if isinstance(value, dict):
        value = value.get("coefficients")
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise FullCycleMemoryError(f"{name} must be a sequence of coefficients")
    coefficients = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise FullCycleMemoryError(
                f"{name}[{index}] must be a finite number")
        number = float(raw)
        if not math.isfinite(number):
            raise FullCycleMemoryError(f"{name}[{index}] must be finite")
        coefficients.append(number)
    if len(coefficients) != FULL_BASIS_CYCLES:
        raise FullCycleMemoryError(
            f"{name} must contain {FULL_BASIS_CYCLES} coefficients, "
            f"found {len(coefficients)}")
    return coefficients


def _module_coefficients(module: Any, name: str) -> list[float]:
    for attribute in (
        "FULL_BASIS_COEFFICIENTS",
        "COEFFICIENTS",
        "coefficients",
    ):
        if hasattr(module, attribute):
            value = getattr(module, attribute)
            if callable(value):
                value = value()
            return _coefficient_list(value, f"{name}.{attribute}")
    for function_name in ("full_basis_coefficients", "build_coefficients"):
        if hasattr(module, function_name):
            return _coefficient_list(
                getattr(module, function_name)(), f"{name}.{function_name}()")
    if hasattr(module, "build_memory"):
        memory = module.build_memory()
        if hasattr(memory, "coefficients"):
            value = memory.coefficients
            if callable(value):
                value = value()
            return _coefficient_list(value, f"{name}.build_memory().coefficients")
    raise FullCycleMemoryError(
        f"{name} exposes no FULL_BASIS_COEFFICIENTS, coefficients() or "
        "build_memory() contract"
    )


def _load_provider(provider: str) -> list[float]:
    path = Path(provider)
    if path.is_file():
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FullCycleMemoryError(
                    f"cannot read full-cycle-memory JSON {path}: {exc}") from exc
            return _coefficient_list(value, str(path))
        spec = importlib.util.spec_from_file_location(
            "round031_full_cycle_memory_provider", path)
        if spec is None or spec.loader is None:
            raise FullCycleMemoryError(f"cannot import provider {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return _module_coefficients(module, str(path))
    try:
        module = importlib.import_module(provider)
    except Exception as exc:
        raise FullCycleMemoryError(
            f"cannot import full-cycle-memory provider {provider!r}: {exc}"
        ) from exc
    return _module_coefficients(module, provider)


def _resident_training_module() -> tuple[Any | None, str | None]:
    """Import the resident full-cycle trainer when the real module is present."""
    try:
        module = importlib.import_module(RESIDENT_TRAINING_MODULE)
    except Exception as exc:  # noqa: BLE001 - optional dependency boundary
        return None, f"cannot import {RESIDENT_TRAINING_MODULE}: {exc}"
    dimension = getattr(module, "FULL_CYCLE_COUNT", None)
    if dimension != FULL_BASIS_CYCLES:
        return None, (
            f"{RESIDENT_TRAINING_MODULE}.FULL_CYCLE_COUNT is "
            f"{dimension!r}, expected {FULL_BASIS_CYCLES}")
    for name in ("FullCycleMemory", "FullCycleTrainer"):
        if not hasattr(module, name):
            return None, f"{RESIDENT_TRAINING_MODULE} has no {name}"
    return module, None


def discover_full_cycle_memory(
    explicit: str | None = None,
    *,
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
) -> dict[str, Any]:
    """Locate an optional full-basis provider.

    The resident ``nanlmind_hermes.resident.full_cycle_training`` module is
    preferred when available.  ``explicit='builtin'`` selects the
    deterministic local synthetic target vector; a JSON/Python path or
    importable module name is also accepted.  With no explicit provider the
    environment variable, resident module, an importable ``full_cycle_memory``
    module, and a local ``full_cycle_memory.json`` are searched.  Absence is a
    structured skip, not a failure.
    """
    searched: list[str] = []
    if explicit is not None:
        if explicit == "builtin":
            return {
                "status": "available",
                "kind": "builtin",
                "source": "builtin-synthetic",
                "targets": None,
                "searched": searched,
                "reason": None,
            }
        if explicit in ("", "none", "off"):
            return {
                "status": "unavailable",
                "kind": None,
                "source": None,
                "targets": None,
                "searched": searched,
                "reason": "full-cycle-memory requested off by the caller",
            }
        if explicit == "resident":
            module, reason = _resident_training_module()
            if module is None:
                return {
                    "status": "unavailable",
                    "kind": None,
                    "source": RESIDENT_TRAINING_MODULE,
                    "targets": None,
                    "searched": [RESIDENT_TRAINING_MODULE],
                    "reason": reason,
                }
            return {
                "status": "available",
                "kind": "resident",
                "source": "resident-full-cycle-training",
                "targets": None,
                "searched": [RESIDENT_TRAINING_MODULE],
                "reason": None,
            }
        searched.append(explicit)
        try:
            targets = _load_provider(explicit)
        except FullCycleMemoryError as exc:
            return {
                "status": "unavailable",
                "kind": None,
                "source": explicit,
                "targets": None,
                "searched": searched,
                "reason": str(exc),
            }
        return {
            "status": "available",
            "kind": "coefficients",
            "source": explicit,
            "targets": targets,
            "searched": searched,
            "reason": None,
        }

    environment = os.environ.get(FULL_CYCLE_MEMORY_ENV)
    if environment:
        searched.append(f"${FULL_CYCLE_MEMORY_ENV}={environment}")
        try:
            targets = _load_provider(environment)
        except FullCycleMemoryError as exc:
            return {
                "status": "unavailable",
                "kind": None,
                "source": environment,
                "targets": None,
                "searched": searched,
                "reason": str(exc),
            }
        return {
            "status": "available",
            "kind": "coefficients",
            "source": environment,
            "targets": targets,
            "searched": searched,
            "reason": None,
        }

    resident, resident_reason = _resident_training_module()
    searched.append(RESIDENT_TRAINING_MODULE)
    if resident is not None:
        return {
            "status": "available",
            "kind": "resident",
            "source": "resident-full-cycle-training",
            "targets": None,
            "searched": searched,
            "reason": None,
        }

    spec = importlib.util.find_spec("full_cycle_memory")
    if spec is not None:
        searched.append("module:full_cycle_memory")
        try:
            targets = _load_provider("full_cycle_memory")
        except FullCycleMemoryError as exc:
            return {
                "status": "unavailable",
                "kind": None,
                "source": "module:full_cycle_memory",
                "targets": None,
                "searched": searched,
                "reason": str(exc),
            }
        return {
            "status": "available",
            "kind": "coefficients",
            "source": "module:full_cycle_memory",
            "targets": targets,
            "searched": searched,
            "reason": None,
        }

    experiment = Path(experiment_dir).resolve()
    for candidate in (
        experiment / "full_cycle_memory.json",
        experiment / "full_cycle_memory.py",
        ROOT / "full_cycle_memory.json",
        ROOT / "full_cycle_memory.py",
    ):
        searched.append(str(candidate))
        if candidate.is_file():
            try:
                targets = _load_provider(str(candidate))
            except FullCycleMemoryError as exc:
                return {
                    "status": "unavailable",
                    "kind": None,
                    "source": str(candidate),
                    "targets": None,
                    "searched": searched,
                    "reason": str(exc),
                }
            return {
                "status": "available",
                "kind": "coefficients",
                "source": str(candidate),
                "targets": targets,
                "searched": searched,
                "reason": None,
            }

    return {
        "status": "unavailable",
        "kind": None,
        "source": None,
        "targets": None,
        "searched": searched,
        "reason": (
            "full_cycle_memory is not available: no provider was given, "
            f"${FULL_CYCLE_MEMORY_ENV} is unset, "
            f"{RESIDENT_TRAINING_MODULE} is not importable, module "
            "'full_cycle_memory' is not importable, and no "
            "full_cycle_memory.json/.py file exists. "
            f"Resident import detail: {resident_reason}"
        ),
    }


class FullBasisMemory:
    """Plain-Python synthetic memory holding all 32,385 cycle coefficients."""

    def __init__(self, coefficients: Sequence[float] | None = None) -> None:
        if coefficients is None:
            self._coefficients = [0.0] * FULL_BASIS_CYCLES
        else:
            self._coefficients = _coefficient_list(
                list(coefficients), "FullBasisMemory coefficients")
        self._visits = [0] * FULL_BASIS_CYCLES

    @property
    def coefficient_count(self) -> int:
        return len(self._coefficients)

    @property
    def coefficients(self) -> list[float]:
        return list(self._coefficients)

    @property
    def visits(self) -> list[int]:
        return list(self._visits)

    def train(
        self,
        targets: Sequence[float],
        *,
        epochs: int = 1,
        learning_rate: float = 1.0,
        seed: int = 2031,
    ) -> dict[str, Any]:
        """Visit every coefficient and apply deterministic per-cycle SGD.

        The objective for coefficient ``w_i`` is ``0.5 * (w_i - t_i) ** 2``.
        With ``learning_rate=1.0`` one visit is an exact closed-form update.
        """
        target_values = _coefficient_list(list(targets), "targets")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise FullCycleMemoryError("epochs must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise FullCycleMemoryError("learning_rate must be positive and finite")
        order_hash = hashlib.sha256()
        updates = 0
        for epoch in range(epochs):
            order = list(range(FULL_BASIS_CYCLES))
            random.Random(seed + epoch).shuffle(order)
            for index in order:
                error = self._coefficients[index] - target_values[index]
                self._coefficients[index] -= learning_rate * error
                self._visits[index] += 1
                order_hash.update(struct.pack("<I", index))
                updates += 1
        return {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "updates": updates,
            "order_sha256": order_hash.hexdigest(),
            "targets_sha256": coefficient_digest(target_values),
        }

    def gauge_invariance_max_abs_error(
        self,
        gauge: Sequence[float],
        *,
        extra_gauge: Sequence[float] | None = None,
    ) -> float:
        """Build a full edge field, apply a gauge shift, and re-read all cycles.

        This uses all 32,640 pair edges and all 32,385 star-basis triangles.
        """
        if len(gauge) != HEADS:
            raise FullCycleMemoryError(f"gauge must contain {HEADS} values")
        if extra_gauge is None:
            extra_gauge = [
                ((head * 131 + 17) % 257) - 128 for head in range(HEADS)
            ]
        if len(extra_gauge) != HEADS:
            raise FullCycleMemoryError(
                f"extra_gauge must contain {HEADS} values")

        edge_values: dict[tuple[int, int], float] = {}
        coefficient_by_pair = {
            pair: self._coefficients[index]
            for index, pair in enumerate(full_basis_pairs())
        }
        for first in range(HEADS):
            for second in range(first + 1, HEADS):
                if first == 0:
                    value = float(gauge[0]) - float(gauge[second])
                else:
                    value = (
                        coefficient_by_pair[(first, second)]
                        + float(gauge[first])
                        - float(gauge[second])
                    )
                edge_values[(first, second)] = value

        # Apply the extra gauge to every edge, then read the flux of every
        # star-basis triangle. This is a synthetic in-memory audit, not the
        # supplied native root.
        for key in edge_values:
            first, second = key
            edge_values[key] += (
                float(extra_gauge[first]) - float(extra_gauge[second]))

        def edge(first: int, second: int) -> float:
            if first < second:
                return edge_values[(first, second)]
            return -edge_values[(second, first)]

        maximum = 0.0
        for index, (first, second) in enumerate(full_basis_pairs()):
            flux = edge(0, first) + edge(first, second) - edge(0, second)
            maximum = max(
                maximum, abs(flux - self._coefficients[index]))
        return maximum


def coefficient_digest(coefficients: Sequence[float]) -> str:
    """SHA-256 over little-endian float64 coefficients."""
    digest = hashlib.sha256()
    for value in coefficients:
        digest.update(struct.pack("<d", float(value)))
    return digest.hexdigest()


def run_full_basis_synthetic_audit(
    targets: Sequence[float] | None = None,
    *,
    seed: int = 2031,
    epochs: int = 1,
    source: str = "builtin-synthetic",
) -> dict[str, Any]:
    """Train and audit all 32,385 full-basis coefficients deterministically."""
    if targets is None:
        target_values = deterministic_full_basis_targets(seed)
    else:
        target_values = _coefficient_list(list(targets), "targets")

    memory = FullBasisMemory()
    training = memory.train(
        target_values, epochs=epochs, learning_rate=1.0, seed=seed)
    coefficients = memory.coefficients
    maximum_error = max(
        abs(measured - target)
        for measured, target in zip(coefficients, target_values)
    )
    gauge = [((head * 17 + 29) % 251) - 125 for head in range(HEADS)]
    gauge_error = memory.gauge_invariance_max_abs_error(gauge)
    visits = memory.visits
    used_count = sum(1 for count in visits if count > 0)
    result = {
        "label": SOURCE_SYNTHETIC_LABEL,
        "status": "ok",
        "kind": "builtin",
        "source": source,
        "seed": seed,
        "coefficient_count": len(coefficients),
        "expected_coefficient_count": FULL_BASIS_CYCLES,
        "pair_edge_count": PAIRS,
        "all_coefficients_used": used_count == FULL_BASIS_CYCLES,
        "used_count": used_count,
        "min_visit_count": min(visits),
        "max_visit_count": max(visits),
        "training": training,
        "maximum_abs_error": maximum_error,
        "gauge_invariance_max_abs_error": gauge_error,
        "coefficients_sha256": coefficient_digest(coefficients),
        "targets_sha256": coefficient_digest(target_values),
        "coefficients": coefficients,
        "audit_passed": (
            len(coefficients) == FULL_BASIS_CYCLES
            and used_count == FULL_BASIS_CYCLES
            and maximum_error <= SYNTHETIC_ERROR_BOUND
            and gauge_error <= SYNTHETIC_GAUGE_BOUND
        ),
        "note": (
            "Deterministic plain-Python synthetic full-basis re-run. It is not "
            "the supplied native root and does not change its 256-weight limit."
        ),
    }
    return result


def run_resident_full_cycle_audit(
    *,
    seed: int = 2031,
    source: str = "resident-full-cycle-training",
) -> dict[str, Any]:
    """Train and audit all 32,385 cycles through the resident implementation.

    This uses ``nanlmind_hermes.resident.full_cycle_training`` when importable.
    The synthetic sample activates every coefficient with a unit feature and a
    deterministic non-zero Q16.16 target; a unit learning rate makes one exact
    step.  The audit then checks the resident trainer's generation/loss report
    and root-star gauge invariance.  It is still a synthetic re-run, not the
    supplied Round 031 native model.
    """
    module, reason = _resident_training_module()
    if module is None:
        raise FullCycleMemoryError(reason or "resident trainer is unavailable")

    count = int(getattr(module, "FULL_CYCLE_COUNT"))
    fixed_one = int(getattr(module, "FIXED_ONE", 1 << 16))
    memory_class = getattr(module, "FullCycleMemory")
    trainer_class = getattr(module, "FullCycleTrainer")

    # Every coefficient gets a non-zero target; Q16.16 units keep the
    # arithmetic exact and JSON-safe after dividing by FIXED_ONE.
    target_units = [
        ((((index * 97 + seed * 31 + 17) % 2009) - 1004) or 1) * 64
        for index in range(count)
    ]
    features = [fixed_one] * count
    memory = memory_class()
    trainer = trainer_class(memory)
    receipt = trainer.train_batch([(features, target_units)], lr=fixed_one)

    units = [int(value) for value in trainer.coefficients]
    if len(units) != FULL_BASIS_CYCLES:
        raise FullCycleMemoryError(
            f"resident trainer exposed {len(units)} coefficients, expected "
            f"{FULL_BASIS_CYCLES}")
    coefficients = [value / fixed_one for value in units]
    target_values = [value / fixed_one for value in target_units]
    maximum_error = max(
        abs(measured - target)
        for measured, target in zip(coefficients, target_values)
    )

    gauge_units = [((head * 17 + 29) % 251) - 125 for head in range(HEADS)]
    resident_gauge = trainer.apply_root_star_gauge(gauge_units)
    generic_memory = FullBasisMemory(coefficients)
    gauge_error = generic_memory.gauge_invariance_max_abs_error(
        [value / fixed_one for value in gauge_units])

    used_count = int(receipt.get("weights_updated", 0))
    loss_before_units = int(receipt.get("loss_units_before", 0))
    loss_after_units = int(receipt.get("loss_units_after", 0))
    resident_gauge_ok = (
        resident_gauge.get("cycle_readout_unchanged") is True
        and resident_gauge.get("weights_unchanged") is True
    )
    training_loss_ok = loss_after_units <= loss_before_units
    audit_passed = (
        len(coefficients) == FULL_BASIS_CYCLES
        and used_count == FULL_BASIS_CYCLES
        and maximum_error <= SYNTHETIC_ERROR_BOUND
        and gauge_error <= SYNTHETIC_GAUGE_BOUND
        and resident_gauge_ok
        and training_loss_ok
    )
    return {
        "label": SOURCE_SYNTHETIC_LABEL,
        "status": "ok",
        "kind": "resident",
        "source": source,
        "framework": f"{RESIDENT_TRAINING_MODULE}.FullCycleTrainer",
        "memory_class": type(memory).__name__,
        "trainer_class": type(trainer).__name__,
        "seed": seed,
        "coefficient_count": len(coefficients),
        "expected_coefficient_count": FULL_BASIS_CYCLES,
        "pair_edge_count": PAIRS,
        "all_coefficients_used": used_count == FULL_BASIS_CYCLES,
        "used_count": used_count,
        "min_visit_count": 1 if used_count == FULL_BASIS_CYCLES else 0,
        "max_visit_count": 1 if used_count == FULL_BASIS_CYCLES else 0,
        "training": {
            "framework": f"{RESIDENT_TRAINING_MODULE}.FullCycleTrainer",
            "schema": receipt.get("schema"),
            "epochs": 1,
            "batches": 1,
            "learning_rate": fixed_one / (1 << 16),
            "learning_rate_units": fixed_one,
            "updates": used_count,
            "loss_units_before": loss_before_units,
            "loss_units_after": loss_after_units,
            "loss_before": receipt.get("loss_before"),
            "loss_after": receipt.get("loss_after"),
            "generation_before": receipt.get("generation_before"),
            "generation_after": receipt.get("generation_after"),
            "weights_digest": receipt.get("weights_digest"),
            "samples": receipt.get("samples"),
            "active_features": receipt.get("active_features"),
        },
        "maximum_abs_error": maximum_error,
        "gauge_invariance_max_abs_error": gauge_error,
        "resident_gauge": {
            "schema": resident_gauge.get("schema"),
            "applied": resident_gauge.get("applied"),
            "cycle_readout_unchanged": resident_gauge.get(
                "cycle_readout_unchanged"),
            "weights_unchanged": resident_gauge.get("weights_unchanged"),
            "generation_before": resident_gauge.get("generation_before"),
            "generation_after": resident_gauge.get("generation_after"),
            "weights_digest": resident_gauge.get("weights_digest"),
        },
        "coefficients_sha256": trainer.weight_digest(),
        "coefficients_float_sha256": coefficient_digest(coefficients),
        "targets_sha256": coefficient_digest(target_values),
        "coefficients": coefficients,
        "audit_passed": audit_passed,
        "note": (
            "Deterministic synthetic re-run using the resident full-cycle "
            "trainer. The supplied Round 031 native root still exposes only "
            "256 effective weights; this re-run is not that model."
        ),
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
    skipped: bool = False,
) -> None:
    details.append(
        {
            "name": name,
            "passed": bool(passed),
            "skipped": bool(skipped),
            "status": "skipped" if skipped else ("pass" if passed else "fail"),
            "observed": observed,
            "expected": expected,
            "source": source,
            "source_label": (
                SOURCE_SYNTHETIC_LABEL if source == SOURCE_SYNTHETIC
                else SOURCE_ARTIFACT_LABEL
            ),
            "tolerance": tolerance,
            "note": note,
        }
    )


def targets() -> dict[str, Any]:
    invariants = cycle_count_invariants()
    return {
        **invariants,
        "wordnet_test_bytes": WORDNET_TEST_BYTES,
        "wordnet_validation_bytes": WORDNET_VAL_BYTES,
        "ablation_bytes": ABLATION_BYTES,
        "alice_validation_bytes": ALICE_VAL_BYTES,
        "alice_test_bytes": ALICE_TEST_BYTES,
        "split_count": SPLIT_COUNT,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "snapshot_magic": SNAPSHOT_MAGIC.decode("ascii"),
        "full_basis_coefficient_count": FULL_BASIS_CYCLES,
        "tolerances": {
            "gauge_weight": GAUGE_WEIGHT_BOUND,
            "gauge_bits_per_byte": GAUGE_BPB_BOUND,
            "readme_rounding": README_ROUNDING_TOLERANCE,
            "synthetic_error": SYNTHETIC_ERROR_BOUND,
            "synthetic_gauge": SYNTHETIC_GAUGE_BOUND,
        },
    }


def evaluate_artifacts(
    artifacts: dict[str, Any],
    verification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return invariant check details and observed measured values."""
    results = artifacts["results"]["data"]
    audit = artifacts["audit"]["data"]
    transfer = artifacts["transfer"]["data"]
    source = artifacts["native_source"]["constants"]
    train_source = artifacts["train_source"]["constants"]
    readme = artifacts["readme"]
    runs = results["runs"]
    details: list[dict[str, Any]] = []
    invariants = cycle_count_invariants()

    _add_check(
        details, "manifest.verified",
        verification.get("status") == "verified",
        verification.get("status"), "verified",
    )
    _add_check(
        details, "manifest.required_files",
        bool(verification.get("required_files_covered")),
        verification.get("verified_count"),
        len(verification.get("required_files", [])),
    )
    _add_check(
        details, "counts.split_sizes",
        results["counts"] == {"train": SPLIT_COUNT, "val": SPLIT_COUNT,
                              "test": SPLIT_COUNT},
        results["counts"], {"train": SPLIT_COUNT, "val": SPLIT_COUNT,
                            "test": SPLIT_COUNT},
    )

    # The 256 / 32,385 limitation.
    _add_check(
        details, "cycle.effective_equals_16x16",
        EFFECTIVE_CYCLES == 16 * 16 == 256,
        EFFECTIVE_CYCLES, 256,
        note="D * K = 16 * 16",
    )
    _add_check(
        details, "cycle.full_equals_c_255_2",
        FULL_BASIS_CYCLES == math.comb(HEADS - 1, 2) == 32_385,
        FULL_BASIS_CYCLES, 32_385,
        note="C(255, 2)",
    )
    _add_check(
        details, "cycle.full_equals_pairs_minus_gauge",
        FULL_BASIS_CYCLES == PAIRS - (HEADS - 1),
        FULL_BASIS_CYCLES, PAIRS - (HEADS - 1),
        note="C(256, 2) - 255",
    )
    _add_check(
        details, "cycle.unused_equals_32129",
        UNUSED_CYCLES == 32_129,
        UNUSED_CYCLES, 32_129,
        note="32,385 - 256",
    )
    _add_check(
        details, "source.native_constants",
        (
            source["heads"] == HEADS
            and source["dimension"] == D
            and source["channels"] == K
            and source["pairs"] == PAIRS
            and source["root_bytes"] == ROOT_BYTES
            and source["has_size_static_assert"]
            and source["has_flux_formula"]
            and all(source["exports"].values())
        ),
        {
            "heads": source["heads"],
            "dimension": source["dimension"],
            "channels": source["channels"],
            "pairs": source["pairs"],
            "root_bytes": source["root_bytes"],
        },
        {
            "heads": HEADS,
            "dimension": D,
            "channels": K,
            "pairs": PAIRS,
            "root_bytes": ROOT_BYTES,
        },
    )
    _add_check(
        details, "source.train_readout",
        (
            train_source["readout_dimension"] == EFFECTIVE_CYCLES
            and train_source["has_16x16_pair_parameter"]
            and train_source["has_nan_get"]
            and train_source["has_nan_apply"]
            and train_source["readout_matches_effective_cycles"]
        ),
        {
            "readout_dimension": train_source["readout_dimension"],
            "has_16x16_pair_parameter": train_source[
                "has_16x16_pair_parameter"],
            "has_nan_get": train_source["has_nan_get"],
            "has_nan_apply": train_source["has_nan_apply"],
        },
        {
            "readout_dimension": EFFECTIVE_CYCLES,
            "has_16x16_pair_parameter": True,
            "has_nan_get": True,
            "has_nan_apply": True,
        },
    )

    params = {}
    for seed in EXPECTED_SEEDS:
        plain = int(runs[f"plain_{seed}"]["params"])
        native = int(runs[f"native_{seed}"]["params"])
        params[seed] = {
            "plain_parameters": plain,
            "native_neural_parameters": native,
            "native_minus_plain": plain - native,
            "expected_pair_parameters": EFFECTIVE_CYCLES,
        }
    params_ok = all(
        row["native_minus_plain"] == EFFECTIVE_CYCLES for row in params.values()
    )
    _add_check(
        details, "params.native_excludes_256_pair_weights",
        params_ok, params,
        {"native_minus_plain": EFFECTIVE_CYCLES},
        note="the plain 16x16 pair tensor has 256 parameters; native excludes it",
    )

    # WordNet held-out test BpB.
    wordnet: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        plain = runs[f"plain_{seed}"]["test"]
        native = runs[f"native_{seed}"]["test"]
        wordnet[seed] = {
            "plain": plain,
            "native": native,
            "native_minus_plain_bits_per_byte": (
                float(native["bits_per_byte"]) - float(plain["bits_per_byte"])
            ),
            "native_minus_plain_accuracy": (
                float(native["accuracy"]) - float(plain["accuracy"])
            ),
        }
    mean_plain = sum(row["plain"]["bits_per_byte"]
                     for row in wordnet.values()) / len(wordnet)
    mean_native = sum(row["native"]["bits_per_byte"]
                      for row in wordnet.values()) / len(wordnet)
    mean_delta = mean_native - mean_plain
    native_worse = all(
        row["native_minus_plain_bits_per_byte"] > 0 for row in wordnet.values()
    )
    conventional_better = all(
        row["plain"]["bits_per_byte"] < row["native"]["bits_per_byte"]
        for row in wordnet.values()
    )
    _add_check(
        details, "wordnet_test.native_bpb_worse_all_seeds",
        native_worse,
        {seed: row["native_minus_plain_bits_per_byte"]
         for seed, row in wordnet.items()},
        "native BpB > plain BpB for both seeds",
        tolerance=0.0,
    )
    _add_check(
        details, "wordnet_test.conventional_16x16_better_all_seeds",
        conventional_better,
        {seed: {
            "plain": row["plain"]["bits_per_byte"],
            "native": row["native"]["bits_per_byte"],
        } for seed, row in wordnet.items()},
        "plain 16x16 BpB < native BpB for both seeds",
    )
    _add_check(
        details, "wordnet_test.mean_delta_positive",
        mean_delta > 0,
        mean_delta, "> 0",
        note="mean(native - plain) across seeds 17 and 29",
    )

    # Gauge invariance.
    gauge: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        block = runs[f"native_{seed}"]["gauge_test"]
        before = block["before"]
        after = block["after"]
        gauge[seed] = {
            "before": before,
            "after": after,
            "max_weight_diff": float(block["max_weight_diff"]),
            "audit": bool(block["audit"]),
            "bits_per_byte_abs_delta": abs(
                float(after["bits_per_byte"]) - float(before["bits_per_byte"])),
            "accuracy_abs_delta": abs(
                float(after["accuracy"]) - float(before["accuracy"])),
        }
    _add_check(
        details, "gauge.weight_diff_bounded",
        all(row["max_weight_diff"] <= GAUGE_WEIGHT_BOUND
            for row in gauge.values()),
        {seed: row["max_weight_diff"] for seed, row in gauge.items()},
        f"<= {GAUGE_WEIGHT_BOUND:g}",
    )
    _add_check(
        details, "gauge.predictions_unchanged",
        all(row["accuracy_abs_delta"] == 0.0 for row in gauge.values()),
        {seed: row["accuracy_abs_delta"] for seed, row in gauge.items()},
        0.0,
    )
    _add_check(
        details, "gauge.audit_true",
        all(row["audit"] is True for row in gauge.values()),
        {seed: row["audit"] for seed, row in gauge.items()}, True,
    )
    native_audit = audit["native"]
    _add_check(
        details, "gauge.api_mirror_and_snapshot_audit",
        (
            native_audit["mirror_audit"] is True
            and float(native_audit["gauge_max_diff"]) <= GAUGE_WEIGHT_BOUND
            and float(native_audit["snapshot_max_diff"]) == 0.0
        ),
        {
            "mirror_audit": native_audit["mirror_audit"],
            "gauge_max_diff": native_audit["gauge_max_diff"],
            "snapshot_max_diff": native_audit["snapshot_max_diff"],
        },
        {
            "mirror_audit": True,
            "gauge_max_diff": f"<= {GAUGE_WEIGHT_BOUND:g}",
            "snapshot_max_diff": 0.0,
        },
    )
    read_ms = float(native_audit["mean_read_ms"])
    _add_check(
        details, "native_audit.read_timing_host_only",
        WINDOW_READ_BOUNDS[0] <= read_ms <= WINDOW_READ_BOUNDS[1],
        read_ms, f"{WINDOW_READ_BOUNDS[0]:g}..{WINDOW_READ_BOUNDS[1]:g} ms",
        note="host-side ctypes read; not an Android or portable benchmark",
    )

    # Matched 7,319-byte memory ablation.
    ablation: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        enabled = runs[f"native_{seed}"]["gauge_test"]["before"]
        disabled = runs[f"native_{seed}"]["memory_disabled"]
        ablation[seed] = {
            "enabled": enabled,
            "disabled": disabled,
            "bits_per_byte_increase_without_memory": (
                float(disabled["bits_per_byte"])
                - float(enabled["bits_per_byte"])
            ),
            "accuracy_change_without_memory": (
                float(disabled["accuracy"]) - float(enabled["accuracy"])
            ),
        }
    loss_worse = all(
        row["bits_per_byte_increase_without_memory"] > 0
        for row in ablation.values()
    )
    accuracy_benefit = all(
        row["accuracy_change_without_memory"] < 0 for row in ablation.values()
    )
    _add_check(
        details, "memory_ablation.loss_worse_without_memory",
        loss_worse,
        {seed: row["bits_per_byte_increase_without_memory"]
         for seed, row in ablation.items()},
        "> 0 for both seeds",
    )
    _add_check(
        details, "memory_ablation.no_consistent_accuracy_benefit",
        not accuracy_benefit,
        {
            "accuracy_change_without_memory": {
                seed: row["accuracy_change_without_memory"]
                for seed, row in ablation.items()
            },
            "memory_improves_accuracy_all_seeds": accuracy_benefit,
        },
        "disabled-memory accuracy does not fall for both seeds",
    )

    # Snapshots and symbolic graph counts.
    snapshots = artifacts["snapshots"]
    snapshot_ok = (
        len(snapshots) == len(EXPECTED_SNAPSHOTS)
        and all(item["size"] == SNAPSHOT_BYTES for item in snapshots)
        and all(item["magic_ok"] for item in snapshots)
        and all(item["root_bytes"] == ROOT_BYTES for item in snapshots)
        and all(item["sha256"] for item in snapshots)
    )
    _add_check(
        details, "snapshots.exact_8mib_plus_magic",
        snapshot_ok,
        [
            {
                "path": item["relative_path"],
                "size": item["size"],
                "magic_ok": item["magic_ok"],
                "sha256": item["sha256"],
            }
            for item in snapshots
        ],
        {
            "files": list(EXPECTED_SNAPSHOTS),
            "size": SNAPSHOT_BYTES,
            "magic": SNAPSHOT_MAGIC.decode("ascii"),
        },
    )
    wordnet_graph = audit["wordnet"]
    graph_counts_ok = (
        wordnet_graph["synsets"] == 117_570
        and wordnet_graph["two_hop_candidates"] == 87_655
        and wordnet_graph["two_hop_proofs_checked"] == 1_500
    )
    _add_check(
        details, "two_hop.counts",
        graph_counts_ok,
        {
            "synsets": wordnet_graph["synsets"],
            "two_hop_candidates": wordnet_graph["two_hop_candidates"],
            "two_hop_proofs_checked": wordnet_graph[
                "two_hop_proofs_checked"],
        },
        {
            "synsets": 117_570,
            "two_hop_candidates": 87_655,
            "two_hop_proofs_checked": 1_500,
        },
    )
    _add_check(
        details, "two_hop.all_edges_verified",
        wordnet_graph["all_chain_edges_verified"] is True,
        wordnet_graph["all_chain_edges_verified"], True,
        note="symbolic path lookup; not learned composition",
    )
    _add_check(
        details, "two_hop.no_direct_edge_only_hits",
        wordnet_graph["direct_edge_only_hits"] == 0,
        wordnet_graph["direct_edge_only_hits"], 0,
    )

    # Transfer results.
    transfer_summary: dict[str, dict[str, Any]] = {}
    for key in EXPECTED_RUNS:
        run = transfer[key]
        before = run["before"]
        after = run["after"]
        transfer_summary[key] = {
            "wordnet_bits_per_byte_improvement": (
                float(before["wordnet"]["bits_per_byte"])
                - float(after["wordnet"]["bits_per_byte"])
            ),
            "alice_validation_bits_per_byte_improvement": (
                float(before["alice"]["bits_per_byte"])
                - float(after["alice"]["bits_per_byte"])
            ),
            "alice_test": run["alice_test"],
        }
        if key.startswith("native_"):
            transfer_summary[key]["alice_test_no_memory"] = run[
                "alice_test_no_memory"]
    validation_improves = all(
        row["wordnet_bits_per_byte_improvement"] > 0
        and row["alice_validation_bits_per_byte_improvement"] > 0
        for row in transfer_summary.values()
    )
    _add_check(
        details, "transfer.validation_improves_all_models",
        validation_improves,
        {
            key: {
                "wordnet": row["wordnet_bits_per_byte_improvement"],
                "alice": row[
                    "alice_validation_bits_per_byte_improvement"],
            }
            for key, row in transfer_summary.items()
        },
        "WordNet and Alice validation BpB improve after 30 updates",
    )
    _add_check(
        details, "transfer.alice_test_is_tiny",
        all(
            row["alice_test"]["n_bytes"] == ALICE_TEST_BYTES
            for row in transfer_summary.values()
        ),
        {key: row["alice_test"]["n_bytes"]
         for key, row in transfer_summary.items()},
        ALICE_TEST_BYTES,
        note="288 bytes cannot support a strong generalization claim",
    )
    native_memory_better = all(
        float(transfer_summary[f"native_{seed}"]["alice_test_no_memory"][
            "bits_per_byte"])
        > float(transfer_summary[f"native_{seed}"]["alice_test"][
            "bits_per_byte"])
        for seed in EXPECTED_SEEDS
    )
    _add_check(
        details, "transfer.native_memory_effect_inconsistent",
        not native_memory_better,
        {
            seed: {
                "memory": transfer_summary[f"native_{seed}"]["alice_test"][
                    "bits_per_byte"],
                "no_memory": transfer_summary[f"native_{seed}"][
                    "alice_test_no_memory"]["bits_per_byte"],
            }
            for seed in EXPECTED_SEEDS
        },
        "memory is not better on both seeds",
        note="288-byte Alice test only; no language-win claim",
    )
    alice_plain_better = all(
        float(transfer_summary[f"plain_{seed}"]["alice_test"][
            "bits_per_byte"])
        < float(transfer_summary[f"native_{seed}"]["alice_test"][
            "bits_per_byte"])
        for seed in EXPECTED_SEEDS
    )
    _add_check(
        details, "transfer.conventional_16x16_better_alice_test",
        alice_plain_better,
        {
            seed: {
                "plain": transfer_summary[f"plain_{seed}"]["alice_test"][
                    "bits_per_byte"],
                "native": transfer_summary[f"native_{seed}"]["alice_test"][
                    "bits_per_byte"],
            }
            for seed in EXPECTED_SEEDS
        },
        "plain BpB < native BpB for both seeds",
    )

    # README consistency.
    readme_table = readme["table"]
    table_ok = all(
        _close(
            readme_table[seed]["plain_bits_per_byte"],
            round(float(wordnet[seed]["plain"]["bits_per_byte"]), 5),
            README_ROUNDING_TOLERANCE,
        )
        and _close(
            readme_table[seed]["native_bits_per_byte"],
            round(float(wordnet[seed]["native"]["bits_per_byte"]), 5),
            README_ROUNDING_TOLERANCE,
        )
        for seed in EXPECTED_SEEDS
    )
    _add_check(
        details, "readme.wordnet_table_matches_results",
        table_ok,
        readme_table,
        {seed: {
            "plain_bits_per_byte": round(
                float(wordnet[seed]["plain"]["bits_per_byte"]), 5),
            "native_bits_per_byte": round(
                float(wordnet[seed]["native"]["bits_per_byte"]), 5),
        } for seed in EXPECTED_SEEDS},
        tolerance=README_ROUNDING_TOLERANCE,
    )
    limits = readme["limits"]
    limits_ok = all(limits.values())
    _add_check(
        details, "readme.limitations_present",
        limits_ok, limits, "all documented limitation phrases present",
    )

    observed = {
        "source_label": SOURCE_ARTIFACT_LABEL,
        "effective_cycles": EFFECTIVE_CYCLES,
        "full_basis_cycles": FULL_BASIS_CYCLES,
        "unused_cycles": UNUSED_CYCLES,
        "cycle_fractions": {
            "effective": EFFECTIVE_CYCLES / FULL_BASIS_CYCLES,
            "unused": UNUSED_CYCLES / FULL_BASIS_CYCLES,
        },
        "params": params,
        "wordnet_test": {
            "per_seed": wordnet,
            "mean_plain_bits_per_byte": mean_plain,
            "mean_native_bits_per_byte": mean_native,
            "mean_native_minus_plain_bits_per_byte": mean_delta,
            "native_worse_all_seeds": native_worse,
            "conventional_16x16_better_all_seeds": conventional_better,
        },
        "gauge": gauge,
        "memory_ablation": {
            "n_bytes": ABLATION_BYTES,
            "per_seed": ablation,
            "loss_worse_without_memory": loss_worse,
            "accuracy_benefit_all_seeds": accuracy_benefit,
            "note": (
                "Matched 100-document ablation subset; the full 13,200-byte "
                "test BpB is not directly comparable to this 7,319-byte subset."
            ),
        },
        "snapshot": {
            "audit_snapshot_max_diff": native_audit["snapshot_max_diff"],
            "files": snapshots,
        },
        "two_hop": {
            **wordnet_graph,
            "note": "symbolic graph lookup, not learned composition",
        },
        "transfer": transfer_summary,
        "readme": {
            "table": readme_table,
            "limits": limits,
        },
        "native_audit": {
            "reads": native_audit["reads"],
            "mean_read_ms": read_ms,
            "gauge_max_diff": native_audit["gauge_max_diff"],
            "mirror_audit": native_audit["mirror_audit"],
            "snapshot_max_diff": native_audit["snapshot_max_diff"],
        },
        "native_source": source,
        "train_source": train_source,
    }
    return details, observed


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _full_basis_skip(
    *,
    reason: str,
    requested: bool,
    searched: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "label": SOURCE_SYNTHETIC_LABEL,
        "status": "skipped",
        "kind": None,
        "reason": reason,
        "requested": bool(requested),
        "source": None,
        "searched": list(searched or []),
        "expected_coefficient_count": FULL_BASIS_CYCLES,
        "coefficient_count": None,
        "pair_edge_count": PAIRS,
        "all_coefficients_used": False,
        "used_count": 0,
        "coefficients": None,
        "coefficients_sha256": None,
        "maximum_abs_error": None,
        "gauge_invariance_max_abs_error": None,
        "resident_gauge": None,
        "audit_passed": None,
        "note": (
            "Structured skip: no full_cycle_memory provider was available. "
            "The supplied native model still exposes only 256 effective "
            "weights and is not a full 32,385-coefficient implementation."
        ),
    }


def _full_basis_checks(
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    _add_check(
        details, "full_basis.coefficient_count",
        audit["coefficient_count"] == FULL_BASIS_CYCLES,
        audit["coefficient_count"], FULL_BASIS_CYCLES,
        source=SOURCE_SYNTHETIC,
    )
    _add_check(
        details, "full_basis.all_coefficients_used",
        audit["all_coefficients_used"] is True
        and audit["used_count"] == FULL_BASIS_CYCLES,
        {
            "used_count": audit["used_count"],
            "min_visit_count": audit["min_visit_count"],
            "max_visit_count": audit["max_visit_count"],
        },
        {"used_count": FULL_BASIS_CYCLES},
        source=SOURCE_SYNTHETIC,
    )
    _add_check(
        details, "full_basis.training_error",
        audit["maximum_abs_error"] <= SYNTHETIC_ERROR_BOUND,
        audit["maximum_abs_error"], f"<= {SYNTHETIC_ERROR_BOUND:g}",
        source=SOURCE_SYNTHETIC,
        tolerance=SYNTHETIC_ERROR_BOUND,
    )
    _add_check(
        details, "full_basis.gauge_invariance",
        audit["gauge_invariance_max_abs_error"] <= SYNTHETIC_GAUGE_BOUND,
        audit["gauge_invariance_max_abs_error"],
        f"<= {SYNTHETIC_GAUGE_BOUND:g}",
        source=SOURCE_SYNTHETIC,
        tolerance=SYNTHETIC_GAUGE_BOUND,
    )
    _add_check(
        details, "full_basis.reports_all_coefficients",
        isinstance(audit["coefficients"], list)
        and len(audit["coefficients"]) == FULL_BASIS_CYCLES,
        len(audit["coefficients"]) if isinstance(audit["coefficients"], list)
        else None,
        FULL_BASIS_CYCLES,
        source=SOURCE_SYNTHETIC,
    )
    if audit.get("kind") == "resident":
        training = audit.get("training", {})
        _add_check(
            details, "full_basis.resident_training_loss",
            int(training.get("loss_units_after", 1))
            <= int(training.get("loss_units_before", 0))
            and int(training.get("updates", 0)) == FULL_BASIS_CYCLES,
            {
                "loss_units_before": training.get("loss_units_before"),
                "loss_units_after": training.get("loss_units_after"),
                "updates": training.get("updates"),
            },
            {
                "loss_units_after": "<= loss_units_before",
                "updates": FULL_BASIS_CYCLES,
            },
            source=SOURCE_SYNTHETIC,
        )
        resident_gauge = audit.get("resident_gauge") or {}
        _add_check(
            details, "full_basis.resident_gauge",
            resident_gauge.get("cycle_readout_unchanged") is True
            and resident_gauge.get("weights_unchanged") is True,
            {
                "cycle_readout_unchanged": resident_gauge.get(
                    "cycle_readout_unchanged"),
                "weights_unchanged": resident_gauge.get("weights_unchanged"),
            },
            {
                "cycle_readout_unchanged": True,
                "weights_unchanged": True,
            },
            source=SOURCE_SYNTHETIC,
        )
    _add_check(
        details, "full_basis.audit_passed",
        audit["audit_passed"] is True,
        audit["audit_passed"], True,
        source=SOURCE_SYNTHETIC,
    )
    return details


def _claims(full_cycle_memory_executed: bool) -> dict[str, Any]:
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
        "native_execution": False,
        "full_cycle_memory_available": bool(full_cycle_memory_executed),
        "full_cycle_memory_used_in_supplied_model": False,
        "scope": (
            "Round 031 artifact audit. The supplied 16x16 native model exposes "
            "256 effective triangle weights, not the 32,385-dimensional star "
            "basis. No language-model win, semantic superiority, canonical "
            "implementation, quantum effect, NaN-unique benefit or portable "
            "benchmark is claimed. A conventional 16x16 parameter matrix has "
            "lower WordNet test BpB on both supplied seeds."
        ),
    }


def _boundaries() -> list[str]:
    return [
        "The supplied native model has 256 effective triangle weights "
        "(16x16) versus C(255,2) = 32,385 full star-basis cycles; 32,129 "
        "cycle degrees are unused.",
        "The root is RAM allocated but not mlocked in this run; it is not a "
        "full-basis cycle memory.",
        "The byte GRU encoder/decoder remains external PyTorch; this is not a "
        "fully memory-native language model.",
        "The native update uses SGD at 0.003 while neural parameters use "
        "AdamW at 0.002; matched nominal counts do not imply optimizer-"
        "equivalent training.",
        "All supplied numbers are a measured artifact parsed from checksum-"
        "verified files. A synthetic full-basis audit, if run, is a separate "
        "plain-Python re-run and not the supplied native model.",
        "Gauge shifts are synthetic and intentionally chosen; gauge invariance "
        "is not semantic superiority.",
        "The conventional 16x16 parameter matrix has lower WordNet test BpB "
        "and higher accuracy on both supplied seeds; no language win is "
        "claimed.",
        "WordNet glosses are selected in corpus file order, not a randomized "
        "representative sample.",
        "The 2-hop graph count is symbolic path lookup, not learned "
        "composition, and is separate from the trained language model.",
        "The untouched Alice test has only 288 bytes and cannot support a "
        "strong generalization claim.",
        "The ~6.9 microsecond ctypes read is a host-only measurement, not an "
        "Android, portable-throughput or native batched benchmark.",
        "The root stores IEEE float values, not payload-preserving NaN "
        "instructions; no quantum or NaN-unique effect is claimed.",
        "No canonical implementation is claimed.",
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


def _base_report(
    *,
    experiment: Path,
    manifest_path: Path | None,
    status: str,
    exit_code: int,
    mode: str,
    evidence_label: str,
    verification: dict[str, Any],
    details: list[dict[str, Any]],
    observed: dict[str, Any] | None,
    artifacts: dict[str, Any] | None,
    full_basis_audit: dict[str, Any],
    notes: Sequence[str],
) -> dict[str, Any]:
    checks = {
        item["name"]: (None if item["skipped"] else bool(item["passed"]))
        for item in details
    }
    all_passed = all(
        item["passed"] or item["skipped"] for item in details
    )
    checks["all_targets_met"] = all_passed
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
            "synthetic_full_basis": {
                "label": SOURCE_SYNTHETIC_LABEL,
                "status": full_basis_audit["status"],
                "re_executed": full_basis_audit["status"] == "ok",
                "source": full_basis_audit.get("source"),
                "reason": full_basis_audit.get("reason"),
                "note": (
                    "A synthetic full-basis audit is a separate deterministic "
                    "re-run; it is not the supplied native model."
                ),
            },
            "label": evidence_label,
        },
        "verification": verification,
        "targets": targets(),
        "observed": observed,
        "full_basis_synthetic_audit": full_basis_audit,
        "artifacts": (
            {
                "results": {
                    "path": artifacts["results"]["path"],
                    "sha256": artifacts["results"]["sha256"],
                    "data": artifacts["results"]["data"],
                },
                "audit": {
                    "path": artifacts["audit"]["path"],
                    "sha256": artifacts["audit"]["sha256"],
                    "data": artifacts["audit"]["data"],
                },
                "transfer": {
                    "path": artifacts["transfer"]["path"],
                    "sha256": artifacts["transfer"]["sha256"],
                    "data": artifacts["transfer"]["data"],
                },
                "readme": {
                    "path": artifacts["readme"]["path"],
                    "sha256": artifacts["readme"]["sha256"],
                    "table": artifacts["readme"]["table"],
                    "limits": artifacts["readme"]["limits"],
                },
                "native_source": {
                    "path": artifacts["native_source"]["path"],
                    "sha256": artifacts["native_source"]["sha256"],
                    "constants": artifacts["native_source"]["constants"],
                },
                "train_source": {
                    "path": artifacts["train_source"]["path"],
                    "sha256": artifacts["train_source"]["sha256"],
                    "constants": artifacts["train_source"]["constants"],
                },
                "snapshots": artifacts["snapshots"],
            }
            if artifacts is not None
            else None
        ),
        "checks": checks,
        "check_details": details,
        "claims": _claims(full_basis_audit["status"] == "ok"),
        "boundaries": _boundaries(),
        "notes": list(notes),
    }


def _denied_report(
    *,
    experiment: Path,
    manifest_path: Path | None,
    reason: str,
) -> dict[str, Any]:
    detail = {
        "name": "manifest.verified",
        "passed": False,
        "skipped": False,
        "status": "fail",
        "observed": "denied",
        "expected": "verified",
        "source": SOURCE_ARTIFACT,
        "source_label": SOURCE_ARTIFACT_LABEL,
        "tolerance": None,
        "note": reason,
    }
    full_basis = _full_basis_skip(
        reason="integrity verification failed before any full-basis re-run",
        requested=False,
    )
    return _base_report(
        experiment=experiment,
        manifest_path=manifest_path,
        status="denied",
        exit_code=EXIT_DENIED,
        mode="artifact-only",
        evidence_label="none (verification denied)",
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
        details=[detail],
        observed=None,
        artifacts=None,
        full_basis_audit=full_basis,
        notes=[
            "No artifact was parsed and no synthetic full-basis audit was run "
            "because the supplied digest manifest did not verify."
        ],
    )


def _artifact_error_report(
    *,
    experiment: Path,
    manifest_path: Path | None,
    reason: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    detail = {
        "name": "artifact.parsed",
        "passed": False,
        "skipped": False,
        "status": "fail",
        "observed": "invalid",
        "expected": "valid Round 031 artifact structure",
        "source": SOURCE_ARTIFACT,
        "source_label": SOURCE_ARTIFACT_LABEL,
        "tolerance": None,
        "note": reason,
    }
    full_basis = _full_basis_skip(
        reason="artifact parsing failed before any full-basis re-run",
        requested=False,
    )
    return _base_report(
        experiment=experiment,
        manifest_path=manifest_path,
        status="error",
        exit_code=EXIT_ERROR,
        mode="artifact-only",
        evidence_label="none (artifact parsing failed)",
        verification=verification,
        details=[detail],
        observed=None,
        artifacts=None,
        full_basis_audit=full_basis,
        notes=[f"Artifact parsing failed: {reason}"],
    )


def run_audit(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    *,
    manifest: Path | str | None = None,
    full_cycle_memory: str | None = "auto",
) -> dict[str, Any]:
    """Verify, parse and audit Round 031.

    ``full_cycle_memory='auto'`` searches for an optional provider.  ``None``
    disables the synthetic full-basis re-run and yields a structured skip.
    ``'builtin'`` uses the deterministic built-in synthetic target vector.
    """
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
        )

    try:
        artifacts = load_artifacts(experiment, verification)
    except ArtifactError as exc:
        return _artifact_error_report(
            experiment=experiment,
            manifest_path=manifest_path,
            reason=str(exc),
            verification=verification,
        )

    details, observed = evaluate_artifacts(artifacts, verification)
    notes = [
        "All parsed artifacts were covered by the verified SHA256SUMS manifest.",
        "The supplied 16x16 native model exposes 256 effective triangle "
        "weights; the full star basis has 32,385 cycles and 32,129 are unused.",
    ]

    if full_cycle_memory is None:
        full_basis = _full_basis_skip(
            reason=(
                "full-basis synthetic audit disabled by the caller "
                "(full_cycle_memory=None)"
            ),
            requested=False,
        )
        mode = "artifact-only"
        evidence_label = "measured artifact only (full-basis audit skipped)"
        notes.append(
            "No synthetic full-basis audit was run; the 256-weight limitation "
            "is reported from the supplied artifact and source.")
    else:
        provider = discover_full_cycle_memory(
            None if full_cycle_memory == "auto" else full_cycle_memory,
            experiment_dir=experiment,
        )
        if provider["status"] != "available":
            full_basis = _full_basis_skip(
                reason=provider["reason"],
                requested=full_cycle_memory != "auto",
                searched=provider.get("searched", []),
            )
            mode = "artifact-only"
            evidence_label = (
                "measured artifact only (full-basis provider unavailable; "
                "structured skip)"
            )
            notes.append(
                "The optional full_cycle_memory provider was unavailable; the "
                "report keeps a structured skip and does not claim a full-basis "
                "measurement.")
        else:
            kind = provider.get("kind")
            try:
                if kind == "resident":
                    full_basis = run_resident_full_cycle_audit(
                        source=provider["source"])
                else:
                    full_basis = run_full_basis_synthetic_audit(
                        provider.get("targets"),
                        source=provider["source"],
                    )
            except Exception as exc:  # noqa: BLE001 - optional audit boundary
                full_basis = {
                    "label": SOURCE_SYNTHETIC_LABEL,
                    "status": "failed",
                    "kind": kind,
                    "reason": str(exc),
                    "requested": True,
                    "source": provider.get("source"),
                    "searched": provider.get("searched", []),
                    "expected_coefficient_count": FULL_BASIS_CYCLES,
                    "coefficient_count": None,
                    "pair_edge_count": PAIRS,
                    "all_coefficients_used": False,
                    "used_count": 0,
                    "coefficients": None,
                    "maximum_abs_error": None,
                    "gauge_invariance_max_abs_error": None,
                    "resident_gauge": None,
                    "audit_passed": False,
                }
                _add_check(
                    details, "full_basis.provider",
                    False,
                    str(exc),
                    "a valid 32,385-coefficient provider",
                    source=SOURCE_SYNTHETIC,
                )
                mode = "artifact-only"
                evidence_label = (
                    "measured artifact only (full-basis provider failed)")
                notes.append(
                    "The optional full_cycle_memory provider was found but "
                    "failed validation; no full-basis claim is made.")
            else:
                details.extend(_full_basis_checks(full_basis))
                mode = "artifact+synthetic-full-basis"
                evidence_label = (
                    "measured artifact plus synthetic full-basis re-run"
                )
                notes.append(
                    "The synthetic full-basis audit used all 32,385 "
                    "coefficients through the "
                    f"{full_basis.get('kind', 'builtin')} provider; it is "
                    "separate from the supplied native root and is not a "
                    "native measurement.")

    all_passed = all(
        item["passed"] or item["skipped"] for item in details
    ) and full_basis.get("status") != "failed"
    status = "ok" if all_passed else "failed"
    exit_code = EXIT_OK if all_passed else EXIT_FAILED
    return _base_report(
        experiment=experiment,
        manifest_path=manifest_path,
        status=status,
        exit_code=exit_code,
        mode=mode,
        evidence_label=evidence_label,
        verification=verification,
        details=details,
        observed=observed,
        artifacts=artifacts,
        full_basis_audit=full_basis,
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
    """Render the Round 031 audit as deterministic Markdown."""
    targets_block = report["targets"]
    lines = [
        "# Round 031 full-cycle audit",
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
        lines += [
            "## Effective weights versus the full basis",
            "",
            "| Quantity | Value | Formula / note |",
            "| --- | ---: | --- |",
            f"| Effective cycles in the supplied model | "
            f"{targets_block['effective_cycles']} | "
            f"{targets_block['effective_formula']} |",
            f"| Full star-basis cycles | "
            f"{targets_block['full_basis_cycles']} | "
            f"{targets_block['full_basis_formula']} |",
            f"| Pair capacity | {targets_block['pair_capacity']} | "
            f"{targets_block['pair_formula']} |",
            f"| Gauge dimensions | {targets_block['gauge_dimension']} | "
            "256 - 1 |",
            f"| Unused cycle degrees | {targets_block['unused_cycles']} | "
            "32,385 - 256 |",
            f"| Effective fraction | "
            f"{_fmt(observed['cycle_fractions']['effective'], 6)} | "
            f"256 / 32,385 |",
            "",
            f"The supplied model uses **256** effective weights and leaves "
            f"**{targets_block['unused_cycles']:,}** of **"
            f"{targets_block['full_basis_cycles']:,}** cycle degrees unused.",
            "",
            "## Supplied measured artifact",
            "",
            "These values come from the checksum-verified supplied files. "
            "They are **not** values produced by this script.",
            "",
            "### WordNet held-out test bits per byte (lower is better)",
            "",
            "| Seed | Conventional 16x16 | Native cycle | Native - plain |",
            "| ---: | ---: | ---: | ---: |",
        ]
        wordnet = observed["wordnet_test"]["per_seed"]
        for seed in EXPECTED_SEEDS:
            row = wordnet[seed]
            lines.append(
                f"| {seed} | {_fmt(row['plain']['bits_per_byte'], 8)} | "
                f"{_fmt(row['native']['bits_per_byte'], 8)} | "
                f"{_fmt(row['native_minus_plain_bits_per_byte'], 8)} |"
            )
        lines += [
            f"| Mean | "
            f"{_fmt(observed['wordnet_test']['mean_plain_bits_per_byte'], 8)} | "
            f"{_fmt(observed['wordnet_test']['mean_native_bits_per_byte'], 8)} | "
            f"{_fmt(observed['wordnet_test']['mean_native_minus_plain_bits_per_byte'], 8)} |",
            "",
            "The conventional 16x16 matrix has lower BpB on both seeds; this is "
            "the capacity-matched baseline comparison, not a language-model "
            "win.",
            "",
            "### Gauge invariance",
            "",
            "| Seed | Max weight diff | BpB abs. delta | Accuracy delta | Audit |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
        for seed in EXPECTED_SEEDS:
            row = observed["gauge"][seed]
            lines.append(
                f"| {seed} | {_fmt(row['max_weight_diff'], 8)} | "
                f"{_fmt(row['bits_per_byte_abs_delta'], 8)} | "
                f"{_fmt(row['accuracy_abs_delta'], 8)} | "
                f"{_fmt(row['audit'])} |"
            )
        lines += [
            "",
            "### Memory ablation",
            "",
            f"Matched {observed['memory_ablation']['n_bytes']}-byte subset. "
            "The full 13,200-byte test BpB is not directly comparable to this "
            "subset.",
            "",
            "| Seed | With memory BpB | Without memory BpB | BpB increase | "
            "Accuracy change |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for seed in EXPECTED_SEEDS:
            row = observed["memory_ablation"]["per_seed"][seed]
            lines.append(
                f"| {seed} | {_fmt(row['enabled']['bits_per_byte'], 8)} | "
                f"{_fmt(row['disabled']['bits_per_byte'], 8)} | "
                f"{_fmt(row['bits_per_byte_increase_without_memory'], 8)} | "
                f"{_fmt(row['accuracy_change_without_memory'], 8)} |"
            )
        lines += [
            "",
            "Disabling memory worsens loss on both seeds but improves accuracy "
            "on both seeds; there is no consistent accuracy benefit from cycle "
            "memory in this ablation.",
            "",
            "### Snapshots and symbolic 2-hop graph",
            "",
            f"- Snapshot files checked: {len(observed['snapshot']['files'])} "
            f"(8-byte `{SNAPSHOT_MAGIC.decode('ascii')}` magic + "
            f"{ROOT_BYTES:,} bytes each).",
            f"- Snapshot restore max diff: "
            f"{_fmt(observed['snapshot']['audit_snapshot_max_diff'], 8)}.",
            f"- WordNet synsets: {observed['two_hop']['synsets']:,}; "
            f"two-hop candidates: "
            f"{observed['two_hop']['two_hop_candidates']:,}; proofs checked: "
            f"{observed['two_hop']['two_hop_proofs_checked']:,}.",
            f"- All chain edges verified: "
            f"{_fmt(observed['two_hop']['all_chain_edges_verified'])}; "
            f"direct-edge-only hits: "
            f"{observed['two_hop']['direct_edge_only_hits']}.",
            "- This graph audit is symbolic path lookup, not learned "
            "composition.",
            "",
            "### Alice transfer",
            "",
            "| Run | WordNet BpB improvement | Alice val BpB improvement | "
            "Alice test BpB |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key in EXPECTED_RUNS:
            row = observed["transfer"][key]
            lines.append(
                f"| {key} | "
                f"{_fmt(row['wordnet_bits_per_byte_improvement'], 8)} | "
                f"{_fmt(row['alice_validation_bits_per_byte_improvement'], 8)} | "
                f"{_fmt(row['alice_test']['bits_per_byte'], 8)} |"
            )
        lines += [
            "",
            "The untouched Alice test has only 288 bytes, so no strong "
            "generalization claim is made. Native memory is not better on both "
            "Alice-test seeds.",
            "",
            "## Full-basis synthetic audit",
            "",
        ]
        full_basis = report["full_basis_synthetic_audit"]
        lines.append(f"- Label: **{full_basis['label']}**")
        lines.append(f"- Status: `{full_basis['status']}`")
        if full_basis.get("reason"):
            lines.append(f"- Reason: {full_basis['reason']}")
        if full_basis.get("source"):
            lines.append(f"- Provider: `{full_basis['source']}`")
        if full_basis["status"] == "ok":
            lines += [
                f"- Coefficients used: {full_basis['used_count']} / "
                f"{full_basis['expected_coefficient_count']}",
                f"- All coefficients used: "
                f"{_fmt(full_basis['all_coefficients_used'])}",
                f"- Training updates: {full_basis['training']['updates']} "
                f"({full_basis['training']['epochs']} epoch(s))",
                f"- Maximum training error: "
                f"{_fmt(full_basis['maximum_abs_error'], 8)}",
                f"- Gauge invariance max error: "
                f"{_fmt(full_basis['gauge_invariance_max_abs_error'], 8)}",
                f"- Coefficients SHA-256: "
                f"`{full_basis['coefficients_sha256']}`",
                "- The full coefficient vector (32,385 values) is in the JSON "
                "report.",
            ]
        else:
            lines.append(
                "- Structured skip: the supplied native model is unchanged and "
                "still exposes only 256 effective weights.")
        lines.append("")

        lines += [
            "## Checks",
            "",
            f"{sum(1 for item in report['check_details'] if item['passed'] and not item['skipped'])}"
            f" passed, "
            f"{sum(1 for item in report['check_details'] if item['skipped'])}"
            f" skipped, "
            f"{sum(1 for item in report['check_details'] if not item['passed'] and not item['skipped'])}"
            f" failed.",
            "",
            "| Check | Source | Result | Observed | Expected |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in report["check_details"]:
            observed_text = _fmt(item["observed"])
            expected_text = _fmt(item["expected"])
            if len(observed_text) > 120:
                observed_text = observed_text[:117] + "..."
            if len(expected_text) > 120:
                expected_text = expected_text[:117] + "..."
            lines.append(
                f"| `{item['name']}` | {item['source_label']} | "
                f"{item['status']} | {observed_text} | {expected_text} |"
            )
        lines.append("")

    lines += [
        "## Claims",
        "",
    ]
    claims = report["claims"]
    for key in (
        "canonical",
        "canonical_implementation",
        "language_model_win",
        "lm_win",
        "language_win",
        "nan_unique_advantage",
        "nan_unique_benefit",
        "semantic_superiority",
        "quantum_claim",
        "android_benchmark",
        "portable_throughput",
        "native_benchmark",
        "native_execution",
        "full_cycle_memory_used_in_supplied_model",
    ):
        lines.append(f"- `{key}`: {_fmt(claims[key])}")
    lines += [
        "",
        "No language-model win, NaN-unique advantage, semantic superiority, "
        "canonical implementation or quantum claim is made.",
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
        "effective_cycles": report["targets"]["effective_cycles"],
        "full_basis_cycles": report["targets"]["full_basis_cycles"],
        "unused_cycles": report["targets"]["unused_cycles"],
        "full_basis_synthetic_audit": report[
            "full_basis_synthetic_audit"]["status"],
        "json": out_json,
        "markdown": out_markdown,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the supplied Round 031 artifacts and expose the 256-weight "
            "readout versus the 32,385-dimensional full cycle basis; optionally "
            "run a deterministic synthetic full-basis audit."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(DEFAULT_EXPERIMENT_DIR),
        help=f"Round 031 experiment directory (default: {DEFAULT_EXPERIMENT_DIR})",
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
        "--full-cycle-memory",
        nargs="?",
        const="builtin",
        default="auto",
        metavar="PROVIDER",
        help=(
            "full-basis provider: 'builtin' for the deterministic synthetic "
            "target vector, or a JSON/Python path or importable module; "
            "default 'auto' searches for one"
        ),
    )
    parser.add_argument(
        "--no-full-cycle-memory",
        action="store_true",
        help="disable the optional synthetic full-basis audit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider: str | None
    if args.no_full_cycle_memory:
        provider = None
    else:
        provider = args.full_cycle_memory
    report = run_audit(
        experiment_dir=args.experiment_dir,
        manifest=args.sha256sums,
        full_cycle_memory=provider,
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
