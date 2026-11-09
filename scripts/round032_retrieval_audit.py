#!/usr/bin/env python3
"""Verify and replay the Round 032 sparse mirror-pair retrieval experiment.

Round 032 stores WordNet hypernym facts in a 256-head / 32,640-pair mirror
layout and benchmarks exact and two-hop retrieval against a conventional hash
index.  The supplied ``experiments/round032`` bundle contains the facts, the
sampled two-hop queries, two measured JSON results and a SHA-256 manifest.

This script verifies every digest before parsing anything, reconstructs both
retrieval indexes over the same facts, and reports the artifact values next to
an explicitly labelled local Python replay.  It does **not** claim that the
pair/posting index is faster than the hash comparator: in the supplied
artifact the hash comparator is faster, and this audit only checks retrieval
counts and structural invariants.

An optional native path compiles ``native_recursive.cpp`` in a temporary
directory when a C++ compiler and the source are present.  Otherwise the
report stays in artifact-only mode with a structured skip.

Labels are explicit throughout: supplied files are a **measured artifact**;
values produced here are a **re-run**.  No canonical implementation, neural
decoder, language-model win, learned-generalization or mirror-speed advantage
is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

SCHEMA = "nanlmind-hermes.round032-retrieval-audit/1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "round032"
DEFAULT_MANIFEST_NAME = "SHA256SUMS"
DEFAULT_TIMEOUT_SECONDS = 180.0

HEAD_CELLS = 256
PAIR_CELLS = 32_640
HEAD_BYTES = 128
PAIR_BYTES = 256
ROOT_BYTES = 8_388_608

EXPECTED_FACTS = 50_268
EXPECTED_LOOKUPS = 250_000
EXPECTED_TWO_HOP = 2_000
EXPECTED_LONG_CONTEXT_QUERIES = 1_000
EXPECTED_DISTRACTORS = 512
LONG_CONTEXT_SEED = 2026
LONG_CONTEXT_WINDOW = 32

DEFAULT_QUERY_TRIALS = EXPECTED_LOOKUPS
MAX_QUERY_TRIALS = 5_000_000

REQUIRED_MANIFEST_FILES = (
    "results.json",
    "long_context_results.json",
    "data/facts.tsv",
    "data/queries.tsv",
)
REQUIRED_RERUN_SOURCES = ("native_recursive.cpp",)
COMPILER_CANDIDATES = ("g++", "clang++", "c++")

SOURCE_ARTIFACT = "measured_artifact"
SOURCE_ARTIFACT_LABEL = "measured artifact"
SOURCE_RERUN = "re_run"
SOURCE_RERUN_LABEL = "re-run"
SOURCE_DERIVED = "derived_layout"
SOURCE_DERIVED_LABEL = "derived layout"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2
EXIT_DENIED = 3

_SUM_LINE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})[ \t]+(?P<name>\*?.+)$")


class VerificationError(RuntimeError):
    """Raised when SHA256SUMS verification must deny execution."""


class ArtifactError(RuntimeError):
    """Raised when a supplied artifact is missing, malformed or inconsistent."""


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
        raise VerificationError(
            f"cannot read SHA256SUMS {manifest}: {exc}") from exc
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

    missing_required = [
        name for name in REQUIRED_MANIFEST_FILES if name not in digests
    ]
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
    }


# ---------------------------------------------------------------------------
# Artifact parsing
# ---------------------------------------------------------------------------


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be a JSON object")
    return value


def _get(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError:
        raise ArtifactError(
            f"{context} is missing required key {key!r}") from None


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


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ArtifactError(f"{name} must be >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactError(f"{name} must be a boolean")
    return value


def _validate_results(data: dict[str, Any]) -> None:
    context = "results.json"
    for key in (
        "facts",
        "root_bytes",
        "membership_count",
        "queries",
        "posting_found",
        "hash_found",
        "twohop_queries",
        "twohop_found",
        "direct_found",
    ):
        _integer(_get(data, key, context), f"{context}.{key}", minimum=0)
    _integer(_get(data, "root_locked", context), f"{context}.root_locked",
             minimum=0)
    _boolean(_get(data, "pair_map_pass", context), f"{context}.pair_map_pass")
    for key in (
        "ingest_ms",
        "posting_ms",
        "hash_ms",
        "cycle_before",
        "cycle_after_update",
        "cycle_after_gauge",
    ):
        _number(_get(data, key, context), f"{context}.{key}", minimum=0.0)


def _validate_long_context(data: dict[str, Any]) -> None:
    context = "long_context_results.json"
    for key in (
        "queries",
        "distractor_facts",
        "short_context_correct",
        "pair_index_conceptual_correct",
        "hash_index_correct",
    ):
        _integer(_get(data, key, context), f"{context}.{key}", minimum=0)
    _number(
        _get(data, "evaluation_ms", context),
        f"{context}.evaluation_ms",
        minimum=0.0,
    )
    if not isinstance(data.get("note"), str):
        raise ArtifactError(f"{context}.note must be a string")


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


def _parse_three_ints(path: Path, name: str) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read {name}: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ArtifactError(
                    f"{name} line {line_number}: expected 3 integers")
            try:
                values = tuple(int(part) for part in parts)
            except ValueError:
                raise ArtifactError(
                    f"{name} line {line_number}: expected 3 integers") from None
            if any(value <= 0 for value in values):
                raise ArtifactError(
                    f"{name} line {line_number}: integers must be positive")
            rows.append(values)  # type: ignore[arg-type]
    if not rows:
        raise ArtifactError(f"{name} contains no rows")
    return rows


def parse_facts_tsv(path: Path | str) -> list[tuple[int, int, int]]:
    """Parse ``source relation destination`` rows as positive integers."""
    return _parse_three_ints(Path(path), "facts.tsv")


def parse_queries_tsv(path: Path | str) -> list[tuple[int, int, int]]:
    """Parse ``source relation destination`` two-hop query rows."""
    return _parse_three_ints(Path(path), "queries.tsv")


def _summarize_facts(
    facts: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    relations = Counter(relation for _, relation, _ in facts)
    keys = {(source, relation) for source, relation, _ in facts}
    return {
        "rows": len(facts),
        "sources": len({source for source, _, _ in facts}),
        "destinations": len({destination for _, _, destination in facts}),
        "unique_source_relation_keys": len(keys),
        "duplicate_source_relation_rows": len(facts) - len(keys),
        "relations": {str(key): value for key, value in sorted(relations.items())},
    }


def load_artifacts(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and structurally validate the four parsed Round 032 artifacts."""
    experiment = Path(experiment_dir).resolve()
    results_path = experiment / "results.json"
    long_context_path = experiment / "long_context_results.json"
    facts_path = experiment / "data" / "facts.tsv"
    queries_path = experiment / "data" / "queries.tsv"

    results = _load_json(results_path, "results.json")
    _validate_results(results)
    long_context = _load_json(long_context_path, "long_context_results.json")
    _validate_long_context(long_context)
    facts = parse_facts_tsv(facts_path)
    queries = parse_queries_tsv(queries_path)

    digests = (verification or {}).get("digests", {})

    def _entry(path: Path, relative: str, data: Any) -> dict[str, Any]:
        return {
            "path": str(path),
            "relative_path": relative,
            "sha256": digests.get(relative),
            "data": data,
        }

    return {
        "results": _entry(results_path, "results.json", results),
        "long_context": _entry(
            long_context_path, "long_context_results.json", long_context),
        "facts": {
            "path": str(facts_path),
            "relative_path": "data/facts.tsv",
            "sha256": digests.get("data/facts.tsv"),
            "summary": _summarize_facts(facts),
            "rows": facts,
        },
        "queries": {
            "path": str(queries_path),
            "relative_path": "data/queries.tsv",
            "sha256": digests.get("data/queries.tsv"),
            "summary": {
                "rows": len(queries),
                "unique_rows": len(set(queries)),
            },
            "rows": queries,
        },
    }


# ---------------------------------------------------------------------------
# Mirror layout and retrieval indexes
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_ANCHOR_SALT = 0x9E3779B97F4A7C15


def hash64(value: int) -> int:
    """The 64-bit splitmix-style hash used by ``native_recursive.cpp``."""
    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def anchor_pair(concept_id: int) -> tuple[int, int]:
    """Map a concept id to the canonical pair of head cells (1..255)."""
    key = hash64(concept_id + _ANCHOR_SALT)
    first = 1 + key % 255
    second = 1 + ((key >> 15) % 255)
    if first == second:
        second = second % 255 + 1
    if first > second:
        first, second = second, first
    return first, second


def pair_index(first: int, second: int) -> int:
    """The C++ triangular pair index for ``0 <= first < second < 256``."""
    if first > second:
        first, second = second, first
    return first * (511 - first) // 2 + second - first - 1


def root_layout() -> dict[str, Any]:
    """Describe the fixed 8 MiB root layout without allocating it in Python."""
    head_total = HEAD_CELLS * HEAD_BYTES
    pair_total = PAIR_CELLS * PAIR_BYTES
    computed_pairs = HEAD_CELLS * (HEAD_CELLS - 1) // 2
    return {
        "head_cells": HEAD_CELLS,
        "head_bytes": HEAD_BYTES,
        "head_total_bytes": head_total,
        "pair_cells": PAIR_CELLS,
        "pair_bytes": PAIR_BYTES,
        "pair_total_bytes": pair_total,
        "computed_pair_cells": computed_pairs,
        "pair_count_matches": computed_pairs == PAIR_CELLS,
        "root_bytes": head_total + pair_total,
        "expected_root_bytes": ROOT_BYTES,
        "matches_expected": head_total + pair_total == ROOT_BYTES,
    }


class HashIndex:
    """Conventional exact-key hash map plus an adjacency map for two hops.

    The exact map keeps the first fact for a duplicated ``(source, relation)``
    key, matching the C++ comparator's insertion rule.  Lookup queries used for
    identity checks are deduplicated, so this choice never changes the answer.
    """

    def __init__(self) -> None:
        self.exact: dict[tuple[int, int], int] = {}
        self.edges: dict[int, list[tuple[int, int]]] = {}
        self.ingest_ms = 0.0

    def add(self, source: int, relation: int, destination: int) -> None:
        self.exact.setdefault((source, relation), destination)
        self.edges.setdefault(source, []).append((relation, destination))

    def get(self, source: int, relation: int) -> int:
        return self.exact.get((source, relation), 0)

    def two_hop(self, source: int, relation: int, goal: int) -> bool:
        for first_relation, middle in self.edges.get(source, ()):
            if first_relation != relation:
                continue
            for second_relation, destination in self.edges.get(middle, ()):
                if second_relation == relation and destination == goal:
                    return True
        return False


class MirrorPostingIndex:
    """The C++ mirror-pair posting index reconstructed in Python.

    Facts are stored in a 1-based arena with a null sentinel.  Each fact is
    chained into the posting head selected by the source concept's anchored
    pair.  Exact source/relation comparison resolves pair collisions exactly as
    the native implementation does.  Two-hop traversal uses typed adjacency
    lists outside the posting chains, again mirroring the native code.
    """

    def __init__(self) -> None:
        self.facts: list[tuple[int, int, int] | None] = [None]
        self.next_ids: list[int] = [0]
        self.first: list[int] = [0] * PAIR_CELLS
        self.edges: dict[int, list[tuple[int, int]]] = {}
        self.ingest_ms = 0.0

    def add(self, source: int, relation: int, destination: int) -> int:
        fact_id = len(self.facts)
        first_cell, second_cell = anchor_pair(source)
        cell = pair_index(first_cell, second_cell)
        self.facts.append((source, relation, destination))
        self.next_ids.append(self.first[cell])
        self.first[cell] = fact_id
        self.edges.setdefault(source, []).append((relation, destination))
        return fact_id

    def get(self, source: int, relation: int) -> int:
        first_cell, second_cell = anchor_pair(source)
        fact_id = self.first[pair_index(first_cell, second_cell)]
        while fact_id:
            fact = self.facts[fact_id]
            assert fact is not None
            if fact[0] == source and fact[1] == relation:
                return fact_id
            fact_id = self.next_ids[fact_id]
        return 0

    def destination(self, fact_id: int) -> int:
        if not fact_id:
            return 0
        fact = self.facts[fact_id]
        assert fact is not None
        return fact[2]

    def two_hop(self, source: int, relation: int, goal: int) -> bool:
        for first_relation, middle in self.edges.get(source, ()):
            if first_relation != relation:
                continue
            for second_relation, destination in self.edges.get(middle, ()):
                if second_relation == relation and destination == goal:
                    return True
        return False


def build_indexes(
    facts: Sequence[tuple[int, int, int]],
) -> tuple[MirrorPostingIndex, HashIndex]:
    """Build both indexes over exactly the same fact rows."""
    started = time.perf_counter_ns()
    posting = MirrorPostingIndex()
    for source, relation, destination in facts:
        posting.add(source, relation, destination)
    posting.ingest_ms = (time.perf_counter_ns() - started) / 1e6

    started = time.perf_counter_ns()
    hashed = HashIndex()
    for source, relation, destination in facts:
        hashed.add(source, relation, destination)
    hashed.ingest_ms = (time.perf_counter_ns() - started) / 1e6
    return posting, hashed


def build_lookup_pool(
    facts: Sequence[tuple[int, int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Sample stored ``(source, relation)`` keys as the native code does.

    Duplicate keys are removed so the exact-identity comparison against the
    first-insertion hash map is unambiguous.  Facts are sampled at the same
    stride used by the native binary before shuffling with seed 2026.
    """
    stride = max(1, len(facts) // 5000)
    pool: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index in range(0, len(facts), stride):
        source, relation, _ = facts[index]
        key = (source, relation)
        if key in seen:
            continue
        seen.add(key)
        pool.append(key)
    random.Random(2026).shuffle(pool)
    return pool, stride


def benchmark_exact(
    index: MirrorPostingIndex | HashIndex,
    pool: Sequence[tuple[int, int]],
    trials: int,
) -> tuple[int, float]:
    """Run exactly ``trials`` exact lookups, cycling through ``pool``."""
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if trials and not pool:
        raise ValueError("cannot benchmark a non-empty trial count without keys")
    found = 0
    size = len(pool)
    get = index.get
    started = time.perf_counter_ns()
    for index_number in range(trials):
        source, relation = pool[index_number % size]
        if get(source, relation):
            found += 1
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return found, elapsed_ms


def benchmark_two_hop(
    index: MirrorPostingIndex | HashIndex,
    queries: Sequence[tuple[int, int, int]],
) -> tuple[int, float]:
    """Run two-hop lookups against one index."""
    found = 0
    started = time.perf_counter_ns()
    for source, relation, goal in queries:
        if index.two_hop(source, relation, goal):
            found += 1
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return found, elapsed_ms


def replay_long_context(
    facts: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    """Deterministically replay ``long_context.py`` over the supplied facts."""
    parents: dict[int, set[int]] = {}
    for source, _, destination in facts:
        parents.setdefault(source, set()).add(destination)
    valid = [
        (source, next(iter(destinations)))
        for source, destinations in parents.items()
        if len(destinations) == 1
    ]
    rng = random.Random(LONG_CONTEXT_SEED)
    rng.shuffle(valid)
    valid = valid[:EXPECTED_LONG_CONTEXT_QUERIES]
    answerbank = dict(valid)
    distractors = EXPECTED_DISTRACTORS
    hits_short = 0
    hits_index = 0
    hits_dict = 0
    started = time.perf_counter()
    for source, correct in valid:
        sampled = rng.sample(valid, min(distractors, len(valid)))
        recent = [(other, answer) for other, answer in sampled if other != source]
        short_recent = dict(recent[-LONG_CONTEXT_WINDOW:])
        hits_short += int(short_recent.get(source) == correct)
        hits_index += int(answerbank.get(source) == correct)
        hits_dict += int(answerbank.get(source) == correct)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "label": SOURCE_RERUN_LABEL,
        "seed": LONG_CONTEXT_SEED,
        "distractor_facts": distractors,
        "short_window": LONG_CONTEXT_WINDOW,
        "queries": len(valid),
        "short_context_correct": hits_short,
        "pair_index_conceptual_correct": hits_index,
        "hash_index_correct": hits_dict,
        "evaluation_ms": elapsed_ms,
        "note": (
            "Python replay of long_context.py. This is exact indexed retrieval, "
            "not neural language generation or decoder evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Optional native re-run
# ---------------------------------------------------------------------------


def find_compiler() -> str | None:
    for candidate in COMPILER_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _trim(value: str, limit: int = 4000) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "...[trimmed]"


def run_native(
    experiment_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Compile and run the supplied C++ binary in a temporary directory."""
    source = experiment_dir / "native_recursive.cpp"
    facts = experiment_dir / "data" / "facts.tsv"
    queries = experiment_dir / "data" / "queries.tsv"
    missing = [
        path.name
        for path in (source, facts, queries)
        if not path.is_file()
    ]
    if missing:
        return {
            "label": SOURCE_RERUN_LABEL,
            "status": "skipped",
            "reason": "missing source/data file(s): " + ", ".join(missing),
            "compiler": None,
            "compile_command": None,
            "returncode": None,
            "measurement": None,
            "comparison": None,
        }

    compiler = find_compiler()
    if compiler is None:
        return {
            "label": SOURCE_RERUN_LABEL,
            "status": "skipped",
            "reason": "no C++ compiler found",
            "compiler": None,
            "compile_command": None,
            "returncode": None,
            "measurement": None,
            "comparison": None,
        }

    compile_command = [
        compiler,
        "-O3",
        "-std=c++20",
        "-Wall",
        "-Wextra",
        str(source),
        "-o",
        "native_recursive",
    ]
    display_command = [
        Path(compiler).name,
        "-O3",
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "native_recursive.cpp",
        "-o",
        "native_recursive",
    ]
    with tempfile.TemporaryDirectory(prefix="round032_native_") as directory:
        temporary = Path(directory)
        try:
            compiled = subprocess.run(
                compile_command,
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": f"compile failed: {exc}",
                "compiler": compiler,
                "compile_command": display_command,
                "returncode": None,
                "measurement": None,
                "comparison": None,
            }
        if compiled.returncode != 0:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": "compile failed",
                "compiler": compiler,
                "compile_command": display_command,
                "returncode": compiled.returncode,
                "measurement": None,
                "comparison": None,
                "stderr": _trim(compiled.stderr),
            }

        executable = temporary / "native_recursive"
        try:
            executed = subprocess.run(
                [str(executable), str(facts), str(queries)],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": f"native run failed: {exc}",
                "compiler": compiler,
                "compile_command": display_command,
                "returncode": None,
                "measurement": None,
                "comparison": None,
            }
        if executed.returncode != 0:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": "native run returned a non-zero exit code",
                "compiler": compiler,
                "compile_command": display_command,
                "returncode": executed.returncode,
                "measurement": None,
                "comparison": None,
                "stderr": _trim(executed.stderr),
            }
        lines = [line for line in executed.stdout.splitlines() if line.strip()]
        try:
            measurement = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": f"cannot parse native JSON output: {exc}",
                "compiler": compiler,
                "compile_command": display_command,
                "returncode": executed.returncode,
                "measurement": None,
                "comparison": None,
                "stderr": _trim(executed.stderr),
            }
        if not isinstance(measurement, dict):
            return {
                "label": SOURCE_RERUN_LABEL,
                "status": "failed",
                "reason": "native output is not a JSON object",
                "compiler": compiler,
                "compile_command": display_command,
                "returncode": executed.returncode,
                "measurement": None,
                "comparison": None,
            }
        return {
            "label": SOURCE_RERUN_LABEL,
            "status": "ok",
            "reason": None,
            "compiler": compiler,
            "compile_command": display_command,
            "returncode": executed.returncode,
            "measurement": measurement,
            "comparison": None,
            "stderr": _trim(compiled.stderr) or None,
        }


def compare_native(
    measurement: dict[str, Any] | None,
    results: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare native re-run counts with the supplied measured artifact."""
    if measurement is None:
        return None
    keys = (
        "facts",
        "root_bytes",
        "queries",
        "posting_found",
        "hash_found",
        "twohop_queries",
        "twohop_found",
        "direct_found",
    )
    comparison: dict[str, Any] = {}
    for key in keys:
        comparison[f"{key}_match"] = measurement.get(key) == results.get(key)
    comparison["all_match"] = all(
        comparison[f"{key}_match"] for key in keys)
    return comparison


# ---------------------------------------------------------------------------
# Checks and report assembly
# ---------------------------------------------------------------------------


def _add_check(
    details: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    *,
    source: str,
    source_label: str,
    note: str = "",
    tolerance: float | None = None,
) -> None:
    details.append({
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "source": source,
        "source_label": source_label,
        "note": note,
        "tolerance": tolerance,
    })


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
        "host": (
            "Termux/Android"
            if "TERMUX_VERSION" in __import__("os").environ
            or "ANDROID_ROOT" in __import__("os").environ
            else platform.system() or "unknown"
        ),
    }


def _claims(native_executed: bool = False) -> dict[str, Any]:
    return {
        "canonical": False,
        "canonical_implementation": False,
        "neural_decoder": False,
        "language_model": False,
        "language_model_win": False,
        "lm_win": False,
        "learned_generalization": False,
        "pair_index_faster": False,
        "mirror_performance_advantage": False,
        "quantum_claim": False,
        "nan_unique_benefit": False,
        "native_execution": bool(native_executed),
        "scope": (
            "Round 032 retrieval audit only. Supplied values are a measured "
            "artifact; local values are a re-run. The corpus contains known "
            "WordNet links, so lookup accuracy is not learned generalization. "
            "No canonical implementation, neural decoder, language-model win, "
            "pair-index speed advantage or quantum claim is made."
        ),
    }


def _boundaries() -> list[str]:
    return [
        "The reported artifact numbers are parsed from checksum-verified "
        "supplied files: a measured artifact, not a fresh Round 032 run.",
        "The Python index replay uses the same facts and query files, but its "
        "timings are host-local re-run values and are not comparable across "
        "machines.",
        "The supplied hash comparator is faster than the posting index in this "
        "microbenchmark. No pair-index or mirror-geometry performance "
        "advantage is claimed.",
        "Hypernym chains come from already stored WordNet links. Two-hop "
        "accuracy is not unseen-word inference or learned generalization.",
        "The long-context replay retrieves existing facts after synthetic "
        "distractors. It does not test a neural decoder or language model.",
        "The 8 MiB root is verified from the fixed layout arithmetic and the "
        "artifact field; the Python replay does not allocate the native root.",
        "The optional native path only re-executes the supplied C++ source "
        "when a compiler is available. A skip is not a failure of the artifact.",
        "No canonical implementation, neural decoder, language-model win or "
        "quantum claim is made.",
    ]


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


def _base_report(
    *,
    experiment: Path,
    verification: dict[str, Any],
    status: str,
    exit_code: int,
    mode: str,
    evidence_label: str,
    artifacts: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
    native: dict[str, Any] | None,
    checks: dict[str, Any],
    check_details: list[dict[str, Any]],
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "exit_code": exit_code,
        "mode": mode,
        "evidence_label": evidence_label,
        "environment": _environment(),
        "experiment_dir": str(experiment),
        "verification": verification,
        "artifacts": (
            None
            if artifacts is None
            else {
                "results": {
                    "path": artifacts["results"]["path"],
                    "relative_path": artifacts["results"]["relative_path"],
                    "sha256": artifacts["results"]["sha256"],
                    "data": artifacts["results"]["data"],
                },
                "long_context_results": {
                    "path": artifacts["long_context"]["path"],
                    "relative_path": artifacts["long_context"][
                        "relative_path"],
                    "sha256": artifacts["long_context"]["sha256"],
                    "data": artifacts["long_context"]["data"],
                },
                "facts": {
                    "path": artifacts["facts"]["path"],
                    "relative_path": artifacts["facts"]["relative_path"],
                    "sha256": artifacts["facts"]["sha256"],
                    "summary": artifacts["facts"]["summary"],
                },
                "queries": {
                    "path": artifacts["queries"]["path"],
                    "relative_path": artifacts["queries"]["relative_path"],
                    "sha256": artifacts["queries"]["sha256"],
                    "summary": artifacts["queries"]["summary"],
                },
            }
        ),
        "benchmark": benchmark,
        "native": native,
        "checks": checks,
        "check_details": check_details,
        "claims": _claims(
            bool(native and native.get("status") == "ok")),
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
        "observed": "denied",
        "expected": "verified",
        "source": SOURCE_ARTIFACT,
        "source_label": SOURCE_ARTIFACT_LABEL,
        "note": reason,
        "tolerance": None,
    }
    return _base_report(
        experiment=experiment,
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
        },
        status="denied",
        exit_code=EXIT_DENIED,
        mode="artifact-only",
        evidence_label="none (verification denied)",
        artifacts=None,
        benchmark=None,
        native=None,
        checks={"manifest.verified": False, "all_targets_met": False},
        check_details=[detail],
        notes=[
            "No artifact was parsed and no native code was compiled or run "
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
        "observed": "invalid",
        "expected": "valid Round 032 artifact structure",
        "source": SOURCE_ARTIFACT,
        "source_label": SOURCE_ARTIFACT_LABEL,
        "note": reason,
        "tolerance": None,
    }
    return _base_report(
        experiment=experiment,
        verification=verification,
        status="error",
        exit_code=EXIT_ERROR,
        mode="artifact-only",
        evidence_label="none (artifact parsing failed)",
        artifacts=None,
        benchmark=None,
        native=None,
        checks={"artifact.parsed": False, "all_targets_met": False},
        check_details=[detail],
        notes=[f"Artifact parsing failed: {reason}"],
    )


def _evaluate(
    artifacts: dict[str, Any],
    verification: dict[str, Any],
    benchmark: dict[str, Any],
    native: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = artifacts["results"]["data"]
    long_context = artifacts["long_context"]["data"]
    facts_summary = artifacts["facts"]["summary"]
    queries_summary = artifacts["queries"]["summary"]
    layout = benchmark["root_layout"]
    exact = benchmark["exact"]
    two_hop = benchmark["two_hop"]
    replay = benchmark["long_context"]

    details: list[dict[str, Any]] = []
    _add_check(
        details, "manifest.verified", verification.get("status") == "verified",
        verification.get("status"), "verified",
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "manifest.required_files",
        bool(verification.get("required_files_covered")),
        verification.get("required_files_covered"), True,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)

    _add_check(
        details, "results.facts", results["facts"] == EXPECTED_FACTS,
        results["facts"], EXPECTED_FACTS,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.root_bytes", results["root_bytes"] == ROOT_BYTES,
        results["root_bytes"], ROOT_BYTES,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.pair_map_pass", results["pair_map_pass"] is True,
        results["pair_map_pass"], True,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.membership_count",
        results["membership_count"] == results["facts"],
        results["membership_count"], results["facts"],
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL,
        note="Every stored fact must be reachable from a posting head.")
    _add_check(
        details, "results.lookup_count",
        results["queries"] == EXPECTED_LOOKUPS,
        results["queries"], EXPECTED_LOOKUPS,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.posting_found",
        results["posting_found"] == results["queries"],
        results["posting_found"], results["queries"],
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.hash_found",
        results["hash_found"] == results["queries"],
        results["hash_found"], results["queries"],
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.hash_comparator_faster",
        results["hash_ms"] < results["posting_ms"],
        {"hash_ms": results["hash_ms"], "posting_ms": results["posting_ms"]},
        "hash_ms < posting_ms",
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL,
        note=(
            "Measured artifact only. This is the opposite of a pair-index "
            "speed claim; no mirror advantage is asserted."))
    _add_check(
        details, "results.twohop_queries",
        results["twohop_queries"] == EXPECTED_TWO_HOP,
        results["twohop_queries"], EXPECTED_TWO_HOP,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.twohop_found",
        results["twohop_found"] == results["twohop_queries"],
        results["twohop_found"], results["twohop_queries"],
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "results.direct_found",
        results["direct_found"] == 0,
        results["direct_found"], 0,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL,
        note="The supplied two-hop queries are not direct edges.")
    _add_check(
        details, "results.cycle_update_applied",
        math.isclose(
            results["cycle_after_update"] - results["cycle_before"],
            1.0, rel_tol=0.0, abs_tol=1e-3),
        results["cycle_after_update"] - results["cycle_before"], 1.0,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL,
        tolerance=1e-3)
    _add_check(
        details, "results.gauge_invariant",
        math.isclose(
            results["cycle_after_gauge"],
            results["cycle_after_update"],
            rel_tol=0.0,
            abs_tol=1e-4,
        ),
        results["cycle_after_gauge"], results["cycle_after_update"],
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL,
        tolerance=1e-4)

    _add_check(
        details, "facts.count", facts_summary["rows"] == EXPECTED_FACTS,
        facts_summary["rows"], EXPECTED_FACTS,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)
    _add_check(
        details, "queries.count",
        queries_summary["rows"] == EXPECTED_TWO_HOP,
        queries_summary["rows"], EXPECTED_TWO_HOP,
        source=SOURCE_ARTIFACT, source_label=SOURCE_ARTIFACT_LABEL)

    _add_check(
        details, "root.layout_root_bytes", layout["matches_expected"],
        layout["root_bytes"], ROOT_BYTES,
        source=SOURCE_DERIVED, source_label=SOURCE_DERIVED_LABEL)
    _add_check(
        details, "root.pair_count", layout["pair_count_matches"],
        layout["computed_pair_cells"], PAIR_CELLS,
        source=SOURCE_DERIVED, source_label=SOURCE_DERIVED_LABEL)

    _add_check(
        details, "re_run.exact.query_count",
        exact["queries"] == benchmark["query_trials"],
        exact["queries"], benchmark["query_trials"],
        source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL,
        note="The requested lookup count is a hard bound.")
    _add_check(
        details, "re_run.exact.posting_found",
        exact["posting_found"] == exact["queries"],
        exact["posting_found"], exact["queries"],
        source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
    _add_check(
        details, "re_run.exact.hash_found",
        exact["hash_found"] == exact["queries"],
        exact["hash_found"], exact["queries"],
        source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
    _add_check(
        details, "re_run.exact.found_counts_equal",
        exact["posting_found"] == exact["hash_found"],
        {"posting_found": exact["posting_found"],
         "hash_found": exact["hash_found"]},
        "equal found counts",
        source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
    _add_check(
        details, "re_run.exact.identity_mismatches",
        exact["identity_mismatches"] == 0,
        exact["identity_mismatches"], 0,
        source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL,
        note=(
            "Destination identity on lookup keys with a single source/relation "
            "row. Ambiguous duplicate keys are excluded and reported "
            "separately because the native posting head is LIFO while the "
            "comparator map keeps the first row."))

    if two_hop is not None:
        _add_check(
            details, "re_run.two_hop.query_count",
            two_hop["queries"] == benchmark["two_hop_requested"],
            two_hop["queries"], benchmark["two_hop_requested"],
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
        _add_check(
            details, "re_run.two_hop.posting_found",
            two_hop["posting_found"] == two_hop["queries"],
            two_hop["posting_found"], two_hop["queries"],
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
        _add_check(
            details, "re_run.two_hop.hash_found",
            two_hop["hash_found"] == two_hop["queries"],
            two_hop["hash_found"], two_hop["queries"],
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
        _add_check(
            details, "re_run.two_hop.found_counts_equal",
            two_hop["posting_found"] == two_hop["hash_found"],
            {"posting_found": two_hop["posting_found"],
             "hash_found": two_hop["hash_found"]},
            "equal found counts",
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
        if two_hop["queries"] == EXPECTED_TWO_HOP:
            _add_check(
                details, "re_run.two_hop.matches_artifact",
                two_hop["posting_found"] == results["twohop_found"],
                two_hop["posting_found"], results["twohop_found"],
                source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)

    if replay is not None:
        _add_check(
            details, "re_run.long_context.queries",
            replay["queries"] == EXPECTED_LONG_CONTEXT_QUERIES,
            replay["queries"], EXPECTED_LONG_CONTEXT_QUERIES,
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
        _add_check(
            details, "re_run.long_context.matches_artifact",
            {
                "queries": replay["queries"],
                "distractor_facts": replay["distractor_facts"],
                "short_context_correct": replay["short_context_correct"],
                "pair_index_conceptual_correct": replay[
                    "pair_index_conceptual_correct"],
                "hash_index_correct": replay["hash_index_correct"],
            }
            == {
                "queries": long_context["queries"],
                "distractor_facts": long_context["distractor_facts"],
                "short_context_correct": long_context[
                    "short_context_correct"],
                "pair_index_conceptual_correct": long_context[
                    "pair_index_conceptual_correct"],
                "hash_index_correct": long_context["hash_index_correct"],
            },
            {
                "queries": replay["queries"],
                "short_context_correct": replay["short_context_correct"],
                "pair_index_conceptual_correct": replay[
                    "pair_index_conceptual_correct"],
                "hash_index_correct": replay["hash_index_correct"],
            },
            {
                "queries": long_context["queries"],
                "short_context_correct": long_context[
                    "short_context_correct"],
                "pair_index_conceptual_correct": long_context[
                    "pair_index_conceptual_correct"],
                "hash_index_correct": long_context["hash_index_correct"],
            },
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)
        _add_check(
            details, "re_run.long_context.short_context_zero",
            replay["short_context_correct"] == 0,
            replay["short_context_correct"], 0,
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL)

    if native is not None and native.get("status") == "ok":
        comparison = native.get("comparison") or {}
        _add_check(
            details, "native.re_run.matches_artifact",
            comparison.get("all_match") is True,
            comparison, "all compared counts match",
            source=SOURCE_RERUN, source_label=SOURCE_RERUN_LABEL,
            note="Optional local C++ re-execution of the supplied source.")

    checks = {detail["name"]: bool(detail["passed"]) for detail in details}
    all_passed = all(detail["passed"] for detail in details)
    checks["all_targets_met"] = all_passed
    return details, checks


def run_audit(
    experiment_dir: Path | str = DEFAULT_EXPERIMENT_DIR,
    *,
    manifest: Path | str | None = None,
    queries: int = DEFAULT_QUERY_TRIALS,
    two_hop: bool = True,
    two_hop_limit: int | None = None,
    long_context: bool = True,
    native: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Verify, parse and replay the Round 032 retrieval experiment."""
    if isinstance(queries, bool) or not isinstance(queries, int):
        raise ValueError("queries must be an integer")
    if not 1 <= queries <= MAX_QUERY_TRIALS:
        raise ValueError(
            f"queries must be between 1 and {MAX_QUERY_TRIALS}")
    if two_hop_limit is not None:
        if isinstance(two_hop_limit, bool) or not isinstance(two_hop_limit, int):
            raise ValueError("two_hop_limit must be an integer or None")
        if two_hop_limit < 0:
            raise ValueError("two_hop_limit must be non-negative")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")

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

    facts = artifacts["facts"]["rows"]
    query_rows = artifacts["queries"]["rows"]

    posting, hashed = build_indexes(facts)
    pool, stride = build_lookup_pool(facts)
    posting_found, posting_ms = benchmark_exact(posting, pool, queries)
    hash_found, hash_ms = benchmark_exact(hashed, pool, queries)
    key_counts = Counter((source, relation) for source, relation, _ in facts)
    ambiguous_keys = {key for key, count in key_counts.items() if count > 1}
    duplicate_key_pool = sum(1 for key in pool if key in ambiguous_keys)
    identity_mismatches = sum(
        1
        for source, relation in pool
        if (source, relation) not in ambiguous_keys
        and hashed.get(source, relation)
        != posting.destination(posting.get(source, relation))
    )

    two_hop_requested: int | None = None
    two_hop_block: dict[str, Any] | None = None
    if two_hop:
        selected = (
            query_rows if two_hop_limit is None
            else query_rows[:two_hop_limit]
        )
        two_hop_requested = len(selected)
        if selected:
            mirror_found, mirror_ms = benchmark_two_hop(posting, selected)
            hash_two_found, hash_two_ms = benchmark_two_hop(hashed, selected)
        else:
            mirror_found = hash_two_found = 0
            mirror_ms = hash_two_ms = 0.0
        two_hop_block = {
            "label": SOURCE_RERUN_LABEL,
            "available": len(query_rows),
            "queries": len(selected),
            "posting_found": mirror_found,
            "hash_found": hash_two_found,
            "posting_ms": mirror_ms,
            "hash_ms": hash_two_ms,
            "all_queries_found": mirror_found == len(selected),
            "note": (
                "Both indexes use the same typed adjacency rows; this replay "
                "does not establish a mirror speed advantage."
            ),
        }

    replay = replay_long_context(facts) if long_context else None

    layout = root_layout()
    benchmark = {
        "label": SOURCE_RERUN_LABEL,
        "facts": len(facts),
        "root_bytes": ROOT_BYTES,
        "root_layout": layout,
        "source_relation_stride": stride,
        "query_pool": len(pool),
        "query_trials": queries,
        "exact": {
            "label": SOURCE_RERUN_LABEL,
            "queries": queries,
            "pool": len(pool),
            "posting_found": posting_found,
            "hash_found": hash_found,
            "posting_ms": posting_ms,
            "hash_ms": hash_ms,
            "identity_mismatches": identity_mismatches,
            "duplicate_key_pool": duplicate_key_pool,
            "duplicate_key_rows": len(facts) - len(key_counts),
            "timing_order": "posting index then conventional hash index",
        },
        "two_hop": two_hop_block,
        "two_hop_requested": two_hop_requested,
        "long_context": replay,
        "ingest_ms": {
            "posting": posting.ingest_ms,
            "hash": hashed.ingest_ms,
        },
        "note": (
            "Host-local Python replay. The conventional hash comparator is "
            "not claimed to be slower in general; the supplied artifact reports "
            "it as faster, and no pair-index speed advantage is asserted."
        ),
    }

    native_block: dict[str, Any] | None = None
    if native:
        native_block = run_native(experiment, timeout=timeout)
        if native_block.get("status") == "ok":
            native_block["comparison"] = compare_native(
                native_block.get("measurement"),
                artifacts["results"]["data"],
            )
            if not native_block["comparison"]["all_match"]:
                native_block["status"] = "failed"
                native_block["reason"] = (
                    "native re-run counts differ from the measured artifact")
    else:
        native_block = {
            "label": SOURCE_RERUN_LABEL,
            "status": "skipped",
            "reason": "native re-run disabled by the caller",
            "compiler": None,
            "compile_command": None,
            "returncode": None,
            "measurement": None,
            "comparison": None,
        }

    details, checks = _evaluate(
        artifacts, verification, benchmark, native_block)
    status = "ok" if checks["all_targets_met"] else "failed"
    exit_code = EXIT_OK if status == "ok" else EXIT_FAILED
    native_ok = bool(native_block and native_block.get("status") == "ok")
    mode = "native-re-run" if native_ok else "artifact-only"
    evidence_label = (
        "measured artifact plus Python replay and native re-run"
        if native_ok
        else "measured artifact plus Python replay"
    )

    notes = [
        "All four parsed artifacts were covered by the verified SHA256SUMS "
        "manifest.",
        "The Python replay rebuilds both indexes from the same facts.tsv rows; "
        "it does not reuse the native root allocation or its timings.",
    ]
    if native_block is not None:
        if native_block.get("status") == "ok":
            notes.append(
                "The optional native source was compiled and re-executed in a "
                "temporary directory; its values are labelled re-run.")
        elif native_block.get("status") == "skipped":
            notes.append(
                "Optional native re-run skipped: "
                + str(native_block.get("reason")))
        else:
            notes.append(
                "Optional native re-run failed without invalidating the "
                "measured artifact: "
                + str(native_block.get("reason")))
    if two_hop_limit is not None and two_hop:
        notes.append(
            f"Two-hop evaluation was bounded to {two_hop_requested} query row(s) "
            f"of {len(query_rows)} available.")
    if not long_context:
        notes.append("Long-context replay was disabled by the caller.")

    report = _base_report(
        experiment=experiment,
        verification=verification,
        status=status,
        exit_code=exit_code,
        mode=mode,
        evidence_label=evidence_label,
        artifacts=artifacts,
        benchmark=benchmark,
        native=native_block,
        checks=checks,
        check_details=details,
        notes=notes,
    )
    report["assertions"] = {
        "facts": EXPECTED_FACTS,
        "root_bytes": ROOT_BYTES,
        "lookups": EXPECTED_LOOKUPS,
        "two_hop": [EXPECTED_TWO_HOP, EXPECTED_TWO_HOP],
        "long_context": [
            EXPECTED_LONG_CONTEXT_QUERIES,
            EXPECTED_LONG_CONTEXT_QUERIES,
        ],
        "short_context": 0,
        "pair_index_faster_claimed": False,
    }
    return report


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    """Render a deterministic Markdown report from a report mapping."""
    lines = [
        "# Round 032 retrieval audit",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Mode:** `{report['mode']}`",
        f"- **Evidence:** {report['evidence_label']}",
        f"- **Experiment directory:** `{report['experiment_dir']}`",
        f"- **Schema:** `{report['schema']}`",
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

    benchmark = report.get("benchmark")
    if benchmark is None:
        lines += [
            "## Result",
            "",
            "No retrieval values are available because verification or parsing "
            "did not complete.",
            "",
        ]
    else:
        exact = benchmark["exact"]
        lines += [
            "## Supplied measured artifact",
            "",
            "These values come from the checksum-verified supplied files. They "
            "are **not** values produced by this script.",
            "",
            "| Measurement | Observed | Expected / note |",
            "| --- | ---: | --- |",
            f"| Facts | {_fmt(benchmark['facts'])} | {EXPECTED_FACTS} |",
            f"| Root bytes | {_fmt(benchmark['root_bytes'])} | 8388608 (8 MiB) |",
            f"| Artifact lookups | "
            f"{_fmt(report['artifacts']['results']['data']['queries'])} | "
            f"{EXPECTED_LOOKUPS} |",
            f"| Artifact posting found | "
            f"{_fmt(report['artifacts']['results']['data']['posting_found'])} | "
            f"{EXPECTED_LOOKUPS} |",
            f"| Artifact hash found | "
            f"{_fmt(report['artifacts']['results']['data']['hash_found'])} | "
            f"{EXPECTED_LOOKUPS} |",
            f"| Artifact two-hop found | "
            f"{_fmt(report['artifacts']['results']['data']['twohop_found'])} | "
            f"{EXPECTED_TWO_HOP} / {EXPECTED_TWO_HOP} |",
            f"| Artifact direct found | "
            f"{_fmt(report['artifacts']['results']['data']['direct_found'])} | "
            "0 |",
            f"| Artifact hash comparator faster | "
            f"{_fmt(report['artifacts']['results']['data']['hash_ms'] < report['artifacts']['results']['data']['posting_ms'])} | "
            "measured artifact only; no mirror advantage claimed |",
            "",
            "### Long-context artifact",
            "",
            "| Measurement | Observed | Expected / note |",
            "| --- | ---: | --- |",
            f"| Queries | "
            f"{_fmt(report['artifacts']['long_context_results']['data']['queries'])} | "
            f"{EXPECTED_LONG_CONTEXT_QUERIES} |",
            f"| Distractor facts | "
            f"{_fmt(report['artifacts']['long_context_results']['data']['distractor_facts'])} | "
            f"{EXPECTED_DISTRACTORS} |",
            f"| Short-context correct | "
            f"{_fmt(report['artifacts']['long_context_results']['data']['short_context_correct'])} | "
            "0 |",
            f"| Pair-index conceptual correct | "
            f"{_fmt(report['artifacts']['long_context_results']['data']['pair_index_conceptual_correct'])} | "
            f"{EXPECTED_LONG_CONTEXT_QUERIES} / "
            f"{EXPECTED_LONG_CONTEXT_QUERIES} |",
            f"| Hash index correct | "
            f"{_fmt(report['artifacts']['long_context_results']['data']['hash_index_correct'])} | "
            f"{EXPECTED_LONG_CONTEXT_QUERIES} / "
            f"{EXPECTED_LONG_CONTEXT_QUERIES} |",
            "",
            "## Python replay",
            "",
            "| Measurement | Re-run | Bound / note |",
            "| --- | ---: | --- |",
            f"| Exact lookups | {_fmt(exact['queries'])} | "
            f"requested {_fmt(benchmark['query_trials'])} |",
            f"| Posting found | {_fmt(exact['posting_found'])} | "
            f"{_fmt(exact['queries'])} |",
            f"| Hash found | {_fmt(exact['hash_found'])} | "
            f"{_fmt(exact['queries'])} |",
            f"| Posting time | {_fmt(exact['posting_ms'], 6)} ms | "
            "host-local, not a portable benchmark |",
            f"| Hash time | {_fmt(exact['hash_ms'], 6)} ms | "
            "host-local, not a portable benchmark |",
            f"| Identity mismatches | {_fmt(exact['identity_mismatches'])} | 0 |",
            f"| Duplicate-key lookup keys | "
            f"{_fmt(exact['duplicate_key_pool'])} | excluded from identity "
            "check; reported separately |",
            f"| Query pool | {_fmt(exact['pool'])} | "
            "deduplicated stored keys |",
        ]
        two_hop = benchmark["two_hop"]
        if two_hop is None:
            lines.append("| Two-hop replay | skipped | disabled by caller |")
        else:
            lines += [
                f"| Two-hop queries | {_fmt(two_hop['queries'])} | "
                f"{_fmt(benchmark['two_hop_requested'])} requested |",
                f"| Two-hop posting found | "
                f"{_fmt(two_hop['posting_found'])} | "
                f"{_fmt(two_hop['queries'])} |",
                f"| Two-hop hash found | {_fmt(two_hop['hash_found'])} | "
                f"{_fmt(two_hop['queries'])} |",
                f"| Two-hop posting time | "
                f"{_fmt(two_hop['posting_ms'], 6)} ms | host-local |",
                f"| Two-hop hash time | "
                f"{_fmt(two_hop['hash_ms'], 6)} ms | host-local |",
            ]
        replay = benchmark.get("long_context")
        if replay is not None:
            lines += [
                f"| Long-context queries | {_fmt(replay['queries'])} | "
                f"{EXPECTED_LONG_CONTEXT_QUERIES} |",
                f"| Long-context short correct | "
                f"{_fmt(replay['short_context_correct'])} | 0 |",
                f"| Long-context pair correct | "
                f"{_fmt(replay['pair_index_conceptual_correct'])} | "
                f"{EXPECTED_LONG_CONTEXT_QUERIES} |",
                f"| Long-context hash correct | "
                f"{_fmt(replay['hash_index_correct'])} | "
                f"{EXPECTED_LONG_CONTEXT_QUERIES} |",
            ]
        lines += [
            "",
            "The conventional hash comparator is **not** claimed to be slower "
            "than the pair-posting index. The supplied artifact reports the "
            "hash comparator as faster in this microbenchmark.",
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

    native = report.get("native")
    lines += ["## Native re-run", ""]
    if native is None:
        lines.append("- Not attempted; verification or parsing did not complete.")
    else:
        lines += [
            f"- Label: **{native.get('label')}**",
            f"- Status: `{native.get('status')}`",
        ]
        if native.get("reason"):
            lines.append(f"- Reason: {native.get('reason')}")
        if native.get("compiler"):
            lines.append(f"- Compiler: `{native.get('compiler')}`")
        measurement = native.get("measurement")
        if measurement:
            lines += [
                f"- Facts: {_fmt(measurement.get('facts'))}",
                f"- Root bytes: {_fmt(measurement.get('root_bytes'))}",
                f"- Lookups found: "
                f"{_fmt(measurement.get('posting_found'))} / "
                f"{_fmt(measurement.get('hash_found'))}",
                f"- Two-hop found: {_fmt(measurement.get('twohop_found'))}",
            ]
    lines += ["", "## Claims and boundaries", ""]
    for key, value in sorted(report["claims"].items()):
        if key == "scope":
            continue
        lines.append(f"- Claim `{key}`: **{_fmt(value)}**")
    lines.append(f"- Scope: {report['claims']['scope']}")
    lines += [""]
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
        "native_status": (
            report["native"].get("status")
            if report.get("native") is not None
            else None
        ),
        "json": out_json,
        "markdown": out_markdown,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and replay the Round 032 mirror-pair retrieval artifacts; "
            "optionally compile and re-run native_recursive.cpp."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(DEFAULT_EXPERIMENT_DIR),
        help=f"Round 032 experiment directory (default: {DEFAULT_EXPERIMENT_DIR})",
    )
    parser.add_argument(
        "--sha256sums",
        default=None,
        help="digest manifest (default: <experiment-dir>/SHA256SUMS)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_QUERY_TRIALS,
        help=(
            "number of exact lookups to run, bounded to 1.."
            f"{MAX_QUERY_TRIALS} (default: {DEFAULT_QUERY_TRIALS})"
        ),
    )
    parser.add_argument(
        "--two-hop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the supplied two-hop queries (default: enabled)",
    )
    parser.add_argument(
        "--two-hop-limit",
        type=int,
        default=None,
        help="bound two-hop evaluation to this many query rows",
    )
    parser.add_argument(
        "--long-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replay long_context.py (default: enabled)",
    )
    parser.add_argument(
        "--native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile and re-run native_recursive.cpp when possible",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "bounded compile/run timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.queries <= MAX_QUERY_TRIALS:
        parser.error(f"--queries must be between 1 and {MAX_QUERY_TRIALS}")
    if args.two_hop_limit is not None and args.two_hop_limit < 0:
        parser.error("--two-hop-limit must be non-negative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number of seconds")

    report = run_audit(
        experiment_dir=args.experiment_dir,
        manifest=args.sha256sums,
        queries=int(args.queries),
        two_hop=bool(args.two_hop),
        two_hop_limit=args.two_hop_limit,
        long_context=bool(args.long_context),
        native=bool(args.native),
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
