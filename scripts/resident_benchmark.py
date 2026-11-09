#!/usr/bin/env python3
"""Memory and latency harness for a resident-universe-style store.

This is deliberately self-contained: it does not import the in-progress
``nanlmind_hermes.resident`` package, run a model, train, or perform any
canonical merge. It builds a small append-only, memory-mapped synthetic store
on a temp file and measures the properties the resident universe needs to
respect on a phone:

* configured budget, mapped bytes and file size;
* resident-page estimate and optional process RSS delta;
* ingest throughput;
* query latency p50/p95;
* exact top-k agreement against an independent brute-force scan;
* checkpoint/fsync time and reopen time.

The default budget is 256 MiB and the hard ceiling is 1 GiB. A request above
the ceiling is rejected before a file is created.

Examples::

    # CI-sized run, explicit JSON and Markdown reports
    python3 scripts/resident_benchmark.py --profile low --quick \\
        --json /tmp/resident-low.json --markdown /tmp/resident-low.md

    # Opt-in larger phone-profile runs
    python3 scripts/resident_benchmark.py --profile medium --full \\
        --json /tmp/resident-medium.json --markdown /tmp/resident-medium.md
    python3 scripts/resident_benchmark.py --profile high --full \\
        --budget-mb 256 --json /tmp/resident-high.json \\
        --markdown /tmp/resident-high.md

Without ``--json``/``--markdown`` the JSON report is printed to stdout.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import math
import mmap
import os
import random
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIB = 1024 * 1024
DEFAULT_BUDGET_BYTES = 256 * MIB
HARD_BUDGET_BYTES = 1024 * MIB
# Small budgets are useful for CI. The real resident implementation has a
# larger minimum; this harness only needs enough room for a tiny synthetic run.
MIN_BUDGET_BYTES = 256 * 1024
DEFAULT_PROFILE = "high"
DEFAULT_SEED = 20261108
REPORT_SCHEMA = "nanlmind-hermes-resident-benchmark/1"

MAGIC = b"NRESB001"
FORMAT_VERSION = 1
DATA_OFFSET = 64
SUPERBLOCK = struct.Struct("<8sIIQQQ16s")
RECORD_HEADER = struct.Struct("<QII")
DIGEST_SIZE = 16

try:
    PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, ValueError):
    PAGE_SIZE = 4096

# Workload sizes are intentionally modest. ``--full`` is an opt-in larger run,
# not a stress test that pretends a synthetic index is a production model.
PHONE_PROFILES: dict[str, dict[str, Any]] = {
    "low": {
        "label": "low",
        "description": "entry-level phone / tight memory",
        "budget_bytes": 64 * MIB,
        "quick": {"records": 96, "queries": 24, "dimensions": 32, "top_k": 5},
        "full": {"records": 1024, "queries": 120, "dimensions": 32, "top_k": 10},
    },
    "medium": {
        "label": "medium",
        "description": "mainstream phone / balanced profile",
        "budget_bytes": 128 * MIB,
        "quick": {"records": 160, "queries": 36, "dimensions": 48, "top_k": 5},
        "full": {"records": 1536, "queries": 160, "dimensions": 48, "top_k": 10},
    },
    "high": {
        "label": "high",
        "description": "flagship phone / default 256 MiB budget",
        "budget_bytes": DEFAULT_BUDGET_BYTES,
        "quick": {"records": 256, "queries": 48, "dimensions": 64, "top_k": 5},
        "full": {"records": 2048, "queries": 200, "dimensions": 64, "top_k": 10},
    },
}

LIMITATIONS = [
    "The store is a synthetic append-only mmap fixture, not the unfinished resident package.",
    "Queries are exact squared-L2 scans, not the eventual resonance/superposition index.",
    "Top-k correctness is a sanity check against brute force, not a semantic retrieval claim.",
    "Resident pages are estimated from touched mapped bytes; RSS is reported "
    "only when /proc is readable.",
    "Timings are host-specific and are not a guarantee for every Android device.",
    "No live model, training, canonical merge, promotion, or durable user state is involved.",
]


class BudgetError(ValueError):
    """Raised when a requested resident budget is invalid or too large."""


@dataclass(frozen=True)
class WorkloadPlan:
    """Resolved workload and memory plan for one benchmark run."""

    profile: str
    mode: str
    records: int
    requested_records: int
    queries: int
    dimensions: int
    top_k: int
    record_stride: int
    dataset_bytes: int
    budget_bytes: int
    budget_clamped: bool

    @property
    def mapped_bytes(self) -> int:
        return self.budget_bytes


def _profile_config(profile: str) -> dict[str, Any]:
    try:
        return PHONE_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(PHONE_PROFILES))
        raise BudgetError("unknown phone profile %r (choose %s)" % (profile, choices)) from exc


def resolve_budget(budget_bytes: int | None = None, *,
                   profile: str = DEFAULT_PROFILE) -> int:
    """Return an accepted budget in bytes or raise :class:`BudgetError`.

    ``None`` selects the phone profile's recommended budget. The default
    profile is ``high``, so the default resolved budget is 256 MiB. Any
    explicit request above 1 GiB is rejected rather than silently clamped.
    """
    config = _profile_config(profile)
    if budget_bytes is None:
        return int(config["budget_bytes"])
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise BudgetError("budget must be an integer number of bytes")
    if budget_bytes < MIN_BUDGET_BYTES:
        raise BudgetError(
            "budget %d bytes is below the synthetic harness minimum of %d bytes"
            % (budget_bytes, MIN_BUDGET_BYTES))
    if budget_bytes > HARD_BUDGET_BYTES:
        raise BudgetError(
            "requested budget %d bytes exceeds the hard 1 GiB ceiling (%d bytes)"
            % (budget_bytes, HARD_BUDGET_BYTES))
    return budget_bytes


def budget_from_mb(megabytes: int | float | None) -> int | None:
    """Convert a CLI megabyte value to bytes without applying the ceiling."""
    if megabytes is None:
        return None
    if isinstance(megabytes, bool) or not isinstance(megabytes, (int, float)):
        raise BudgetError("budget in MB must be a number")
    value = float(megabytes)
    if not math.isfinite(value) or value <= 0:
        raise BudgetError("budget in MB must be a positive finite number")
    return int(round(value * MIB))


def build_workload(profile: str = DEFAULT_PROFILE, mode: str = "quick",
                   budget_bytes: int | None = None) -> WorkloadPlan:
    """Build a workload sized for ``profile`` and clamped to the budget.

    The record count is reduced when the requested profile workload would not
    fit inside the configured budget. The mapped file itself is always exactly
    the resolved budget size.
    """
    config = _profile_config(profile)
    if mode not in ("quick", "full"):
        raise ValueError("mode must be 'quick' or 'full'")
    budget = resolve_budget(budget_bytes, profile=profile)
    spec = config[mode]
    dimensions = int(spec["dimensions"])
    stride = RECORD_HEADER.size + dimensions * 4
    capacity = max(0, (budget - DATA_OFFSET) // stride)
    if capacity < 1:
        raise BudgetError(
            "budget %d bytes cannot hold one %d-byte synthetic record"
            % (budget, stride))
    requested_records = int(spec["records"])
    records = min(requested_records, capacity)
    top_k = min(int(spec["top_k"]), records)
    dataset_bytes = DATA_OFFSET + records * stride
    return WorkloadPlan(
        profile=profile,
        mode=mode,
        records=records,
        requested_records=requested_records,
        queries=int(spec["queries"]),
        dimensions=dimensions,
        top_k=top_k,
        record_stride=stride,
        dataset_bytes=dataset_bytes,
        budget_bytes=budget,
        budget_clamped=records < requested_records,
    )


def _read_rss_bytes() -> int | None:
    """Read current RSS from /proc when available; return None otherwise."""
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(fields) < 2:
            return None
        return int(fields[1]) * PAGE_SIZE
    except (OSError, UnicodeError, ValueError):
        return None


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _squared_l2(left: Sequence[float], right: Sequence[float]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        delta = a - b
        total += delta * delta
    return total


def _brute_force_topk(vectors: Sequence[Sequence[float]],
                      record_ids: Sequence[int],
                      query: Sequence[float], k: int) -> list[tuple[float, int]]:
    """Independent reference top-k used only for the correctness sanity check."""
    scored = [(_squared_l2(query, vector), record_ids[index])
              for index, vector in enumerate(vectors)]
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[:k]


class SyntheticResidentStore:
    """Small append-only mmap store with an in-memory exact vector index.

    The fixture deliberately mirrors the resident universe's file-level shape:
    one sparse file fixed to the budget, a superblock, append-only records, a
    generation counter and a content digest. The query index is a Python list;
    it is not the unfinished production resonance index.
    """

    def __init__(self, path: str | Path, budget_bytes: int, dimensions: int,
                 record_stride: int, *, create: bool) -> None:
        self.path = Path(path)
        self.budget_bytes = int(budget_bytes)
        self.dimensions = int(dimensions)
        self.record_stride = int(record_stride)
        expected_stride = RECORD_HEADER.size + self.dimensions * 4
        if self.record_stride != expected_stride:
            raise ValueError(
                "record stride %d does not match dimensions %d (expected %d)"
                % (self.record_stride, self.dimensions, expected_stride))
        if self.budget_bytes < DATA_OFFSET + self.record_stride:
            raise BudgetError("budget is too small for the store superblock and one record")
        if self.budget_bytes > HARD_BUDGET_BYTES:
            raise BudgetError("resident store budget exceeds the hard 1 GiB ceiling")

        flags = os.O_RDWR | (os.O_CREAT | os.O_TRUNC if create else 0)
        self._fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(self._fd, self.budget_bytes)
                self._map = mmap.mmap(self._fd, self.budget_bytes,
                                      access=mmap.ACCESS_WRITE)
                self.count = 0
                self.generation = 0
                self._ids: list[int] = []
                self._vectors: list[tuple[float, ...]] = []
                self._hasher = hashlib.blake2b(digest_size=DIGEST_SIZE)
                self._digest = self._hasher.digest()
                self._write_superblock()
            else:
                file_size = os.fstat(self._fd).st_size
                if file_size < DATA_OFFSET:
                    raise ValueError("resident store file is shorter than its superblock")
                self._map = mmap.mmap(self._fd, file_size,
                                      access=mmap.ACCESS_WRITE)
                self._read_superblock()
                self._load_records()
        except Exception:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1
            raise

    # -- persistence -----------------------------------------------------
    def _write_superblock(self) -> None:
        SUPERBLOCK.pack_into(
            self._map, 0, MAGIC, FORMAT_VERSION, 0, self.count,
            self.generation, self.record_stride, self._digest)

    def _read_superblock(self) -> None:
        (magic, version, _flags, count, generation, record_stride,
         digest) = SUPERBLOCK.unpack_from(self._map, 0)
        if magic != MAGIC:
            raise ValueError("resident store magic mismatch")
        if version != FORMAT_VERSION:
            raise ValueError("unsupported resident store version %d" % version)
        if record_stride != self.record_stride:
            raise ValueError("resident store record stride mismatch")
        if DATA_OFFSET + count * record_stride > len(self._map):
            raise ValueError("resident store record count exceeds file size")
        self.count = int(count)
        self.generation = int(generation)
        self._stored_digest = bytes(digest)

    def _load_records(self) -> None:
        self._ids = []
        self._vectors = []
        hasher = hashlib.blake2b(digest_size=DIGEST_SIZE)
        for index in range(self.count):
            offset = DATA_OFFSET + index * self.record_stride
            record_id, _generation, dimensions = RECORD_HEADER.unpack_from(
                self._map, offset)
            if dimensions != self.dimensions:
                raise ValueError("resident store record dimension mismatch")
            values = struct.unpack_from(
                "<%df" % self.dimensions, self._map,
                offset + RECORD_HEADER.size)
            self._ids.append(int(record_id))
            self._vectors.append(tuple(float(value) for value in values))
            hasher.update(self._map[offset:offset + self.record_stride])
        self._hasher = hasher
        self._digest = hasher.digest()
        if self._digest != self._stored_digest:
            raise ValueError("resident store digest mismatch (torn snapshot)")

    def append(self, vector: Sequence[float], record_id: int | None = None) -> int:
        if len(vector) != self.dimensions:
            raise ValueError("vector dimension mismatch")
        if DATA_OFFSET + (self.count + 1) * self.record_stride > len(self._map):
            raise BudgetError("resident store budget is full")
        value = int(self.count if record_id is None else record_id)
        self.generation += 1
        offset = DATA_OFFSET + self.count * self.record_stride
        header = RECORD_HEADER.pack(value, self.generation, self.dimensions)
        payload = struct.pack("<%df" % self.dimensions, *vector)
        record = header + payload
        self._map[offset:offset + len(record)] = record
        self._hasher.update(record)
        self._digest = self._hasher.digest()
        self._ids.append(value)
        self._vectors.append(tuple(float(item) for item in vector))
        self.count += 1
        self._write_superblock()
        return value

    def checkpoint(self) -> dict[str, Any]:
        """Flush the map and fsync the backing file, with component timings."""
        total_start = time.perf_counter()
        digest_start = time.perf_counter()
        expected = self._hasher.digest()
        if expected != self._digest:
            raise ValueError("resident store digest drifted before checkpoint")
        digest_ms = (time.perf_counter() - digest_start) * 1000.0

        header_start = time.perf_counter()
        self._write_superblock()
        header_ms = (time.perf_counter() - header_start) * 1000.0

        flush_start = time.perf_counter()
        self._map.flush()
        flush_ms = (time.perf_counter() - flush_start) * 1000.0

        fsync_start = time.perf_counter()
        os.fsync(self._fd)
        fsync_ms = (time.perf_counter() - fsync_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "total_ms": total_ms,
            "digest_ms": digest_ms,
            "header_ms": header_ms,
            "flush_ms": flush_ms,
            "fsync_ms": fsync_ms,
            "bytes_flushed": DATA_OFFSET + self.count * self.record_stride,
        }

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "SyntheticResidentStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # -- query -----------------------------------------------------------
    def query(self, vector: Sequence[float], k: int) -> list[tuple[float, int]]:
        if len(vector) != self.dimensions:
            raise ValueError("query dimension mismatch")
        if k <= 0:
            return []
        heap: list[tuple[float, int]] = []
        for index, candidate in enumerate(self._vectors):
            score = _squared_l2(vector, candidate)
            item = (-score, self._ids[index])
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        return [(-negative_score, record_id)
                for negative_score, record_id in sorted(
                    heap, key=lambda item: (-item[0], item[1]))]

    @property
    def digest_hex(self) -> str:
        return self._digest.hex()

    @property
    def mapped_bytes(self) -> int:
        return len(self._map)


def _make_vectors(count: int, dimensions: int, seed: int) -> list[tuple[float, ...]]:
    rng = random.Random(seed)
    return [tuple(rng.uniform(-1.0, 1.0) for _ in range(dimensions))
            for _ in range(count)]


def _make_queries(vectors: Sequence[Sequence[float]], count: int,
                  dimensions: int, seed: int
                  ) -> tuple[list[tuple[float, ...]], list[int | None]]:
    rng = random.Random(seed ^ 0x5EED5EED)
    queries: list[tuple[float, ...]] = []
    planted: list[int | None] = []
    for index in range(count):
        # Two thirds are near a stored vector, which gives a known planted
        # top-1. The rest are random queries checked against brute force.
        if vectors and index % 3 != 2:
            source = (index * 17 + 3) % len(vectors)
            base = vectors[source]
            queries.append(tuple(value + rng.gauss(0.0, 0.01)
                                 for value in base))
            planted.append(source)
        else:
            queries.append(tuple(rng.uniform(-1.0, 1.0)
                                 for _ in range(dimensions)))
            planted.append(None)
    return queries, planted


def _memory_snapshot(store: SyntheticResidentStore,
                     rss_before: int | None,
                     page_size: int) -> dict[str, Any]:
    stat = store.path.stat()
    file_size = int(stat.st_size)
    allocated = int(getattr(stat, "st_blocks", 0)) * 512
    mapped = int(store.mapped_bytes)
    touched = DATA_OFFSET + store.count * store.record_stride
    resident_pages = math.ceil(touched / page_size) if touched else 0
    rss_after = _read_rss_bytes()
    rss_delta = (rss_after - rss_before
                 if rss_after is not None and rss_before is not None else None)
    return {
        "file_size_bytes": file_size,
        "allocated_bytes": allocated,
        "mapped_bytes": mapped,
        "mapped_pages": math.ceil(mapped / page_size) if mapped else 0,
        "touched_bytes": touched,
        "resident_pages_estimate": resident_pages,
        "resident_bytes_estimate": resident_pages * page_size,
        "page_size_bytes": page_size,
        "budget_utilization": (touched / store.budget_bytes
                               if store.budget_bytes else 0.0),
        "budget_headroom_bytes": max(0, store.budget_bytes - touched),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_delta,
    }


def _run_store(plan: WorkloadPlan, store_path: Path, seed: int) -> dict[str, Any]:
    vectors = _make_vectors(plan.records, plan.dimensions, seed)
    queries, planted = _make_queries(vectors, plan.queries, plan.dimensions, seed)
    rss_before = _read_rss_bytes()

    store = SyntheticResidentStore(
        store_path, plan.budget_bytes, plan.dimensions, plan.record_stride,
        create=True)
    try:
        ingest_start = time.perf_counter()
        for index, vector in enumerate(vectors):
            store.append(vector, record_id=index)
        ingest_seconds = time.perf_counter() - ingest_start
        records = store.count
        ops_per_sec = records / ingest_seconds if ingest_seconds > 0 else 0.0
        record_bytes = records * plan.record_stride
        mib_per_sec = ((record_bytes / MIB) / ingest_seconds
                       if ingest_seconds > 0 else 0.0)

        warmup = min(3, len(queries))
        for query in queries[:warmup]:
            store.query(query, plan.top_k)

        latencies: list[float] = []
        top1_hits = 0
        planted_hits = 0
        planted_queries = 0
        recall_numerator = 0
        exact_queries = 0
        for query, planted_index in zip(queries, planted):
            start_ns = time.perf_counter_ns()
            actual = store.query(query, plan.top_k)
            latencies.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
            expected = _brute_force_topk(
                vectors, list(range(plan.records)), query, plan.top_k)
            actual_ids = [record_id for _score, record_id in actual]
            expected_ids = [record_id for _score, record_id in expected]
            if actual_ids and expected_ids and actual_ids[0] == expected_ids[0]:
                top1_hits += 1
            overlap = len(set(actual_ids) & set(expected_ids))
            recall_numerator += overlap
            if actual_ids == expected_ids:
                exact_queries += 1
            if planted_index is not None:
                planted_queries += 1
                if actual_ids and actual_ids[0] == planted_index:
                    planted_hits += 1

        memory = _memory_snapshot(store, rss_before, PAGE_SIZE)
        checkpoint = store.checkpoint()
        digest_before_reopen = store.digest_hex
        count_before_reopen = store.count
        store_file_size = store.path.stat().st_size
    finally:
        store.close()

    reopen_start = time.perf_counter()
    reopened = SyntheticResidentStore(
        store_path, plan.budget_bytes, plan.dimensions, plan.record_stride,
        create=False)
    reopen_ms = (time.perf_counter() - reopen_start) * 1000.0
    try:
        reopen_digest_matches = reopened.digest_hex == digest_before_reopen
        reopen_count_matches = reopened.count == count_before_reopen
        reopen_query_ok = True
        if queries and reopened.count:
            sample = _brute_force_topk(
                vectors, list(range(plan.records)), queries[0], plan.top_k)
            reopened_ids = [record_id for _score, record_id
                            in reopened.query(queries[0], plan.top_k)]
            reopen_query_ok = reopened_ids == [record_id for _score, record_id
                                               in sample]
        reopened_records = reopened.count
    finally:
        reopened.close()

    query_count = len(latencies)
    top1_accuracy = top1_hits / query_count if query_count else 0.0
    planted_accuracy = planted_hits / planted_queries if planted_queries else 1.0
    recall_at_k = (recall_numerator / (query_count * plan.top_k)
                   if query_count and plan.top_k else 0.0)
    top_k_passed = (query_count > 0 and recall_at_k >= 0.999
                    and top1_accuracy >= 0.999 and planted_accuracy >= 0.999)

    return {
        "memory": memory,
        "ingest": {
            "records": records,
            "seconds": ingest_seconds,
            "ops_per_sec": ops_per_sec,
            "mib_per_sec": mib_per_sec,
        },
        "query_latency": {
            "samples": query_count,
            "warmup_queries": warmup,
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
            "mean_ms": (sum(latencies) / query_count if query_count else 0.0),
            "min_ms": min(latencies) if latencies else 0.0,
            "max_ms": max(latencies) if latencies else 0.0,
        },
        "top_k": {
            "metric": "squared_l2",
            "k": plan.top_k,
            "queries": query_count,
            "top1_accuracy": top1_accuracy,
            "planted_top1_queries": planted_queries,
            "planted_top1_accuracy": planted_accuracy,
            "recall_at_k": recall_at_k,
            "exact_queries": exact_queries,
            "against_bruteforce": True,
            "passed": top_k_passed,
        },
        "checkpoint": checkpoint,
        "reopen": {
            "reopen_ms": reopen_ms,
            "records": reopened_records,
            "count_verified": reopen_count_matches,
            "digest_verified": reopen_digest_matches,
            "query_ok": reopen_query_ok,
            "file_size_bytes": store_file_size,
        },
    }


def run_benchmark(profile: str = DEFAULT_PROFILE, *, quick: bool = False,
                  full: bool = False, budget_bytes: int | None = None,
                  store_dir: str | Path | None = None,
                  seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Run one benchmark and return its JSON-serialisable report.

    With neither ``quick`` nor ``full`` supplied, the CI-sized quick mode is
    used. ``--full`` is always an explicit opt-in.
    """
    if quick and full:
        raise ValueError("quick and full modes are mutually exclusive")
    mode = "full" if full else "quick"
    config = _profile_config(profile)
    resolved_budget = resolve_budget(budget_bytes, profile=profile)
    plan = build_workload(profile, mode, resolved_budget)
    if plan.mapped_bytes > HARD_BUDGET_BYTES:
        raise BudgetError("resolved mapped bytes exceed the hard 1 GiB ceiling")

    with contextlib.ExitStack() as stack:
        if store_dir is None:
            temporary = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="resident-benchmark-"))
            base = Path(temporary)
        else:
            base = Path(store_dir)
            base.mkdir(parents=True, exist_ok=True)
        store_path = base / "resident_universe.bin"
        measurements = _run_store(plan, store_path, seed)

    budget_source = "profile" if budget_bytes is None else "explicit"
    memory = measurements["memory"]
    budget_ok = (plan.budget_bytes <= HARD_BUDGET_BYTES
                 and memory["mapped_bytes"] <= plan.budget_bytes
                 and memory["file_size_bytes"] <= HARD_BUDGET_BYTES)
    reopen = measurements["reopen"]
    reopen_ok = (reopen["count_verified"] and reopen["digest_verified"]
                 and reopen["query_ok"])
    passed = bool(budget_ok and measurements["top_k"]["passed"] and reopen_ok)

    return {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "profile": profile,
        "profile_description": config["description"],
        "seed": seed,
        "scope": {
            "synthetic_store": True,
            "resident_module_dependency": False,
            "live_model": False,
            "training": False,
            "canonical_merge": False,
            "persistent_user_state": False,
        },
        "budget": {
            "default_bytes": DEFAULT_BUDGET_BYTES,
            "hard_bytes": HARD_BUDGET_BYTES,
            "profile_recommended_bytes": int(config["budget_bytes"]),
            "requested_bytes": plan.budget_bytes,
            "effective_bytes": plan.budget_bytes,
            "source": budget_source,
            "within_hard_ceiling": plan.budget_bytes <= HARD_BUDGET_BYTES,
        },
        "workload": {
            "requested_records": plan.requested_records,
            "records": plan.records,
            "queries": plan.queries,
            "dimensions": plan.dimensions,
            "top_k": plan.top_k,
            "record_stride_bytes": plan.record_stride,
            "dataset_bytes": plan.dataset_bytes,
            "mapped_bytes": plan.mapped_bytes,
            "budget_clamped": plan.budget_clamped,
        },
        "store": {
            "kind": "synthetic-append-only-mmap",
            "file_name": "resident_universe.bin",
            "format_version": FORMAT_VERSION,
            "superblock_bytes": SUPERBLOCK.size,
            "data_offset_bytes": DATA_OFFSET,
            "page_size_bytes": PAGE_SIZE,
        },
        "memory": memory,
        "ingest": measurements["ingest"],
        "query_latency": measurements["query_latency"],
        "top_k": measurements["top_k"],
        "checkpoint": measurements["checkpoint"],
        "reopen": reopen,
        "verdict": {
            "budget_within_ceiling": bool(budget_ok),
            "top_k_sanity": bool(measurements["top_k"]["passed"]),
            "reopen_verified": bool(reopen_ok),
            "passed": passed,
        },
        "limitations": list(LIMITATIONS),
    }


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024.0 or unit == "GiB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f GiB" % size


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else "%.3f ms" % value


def render_markdown(report: dict[str, Any]) -> str:
    """Render a stable Markdown view of a benchmark report."""
    budget = report["budget"]
    workload = report["workload"]
    memory = report["memory"]
    ingest = report["ingest"]
    latency = report["query_latency"]
    top_k = report["top_k"]
    checkpoint = report["checkpoint"]
    reopen = report["reopen"]
    verdict = report["verdict"]

    lines = [
        "# Resident universe benchmark",
        "",
        "Synthetic append-only mmap fixture. No live model, no training, and no "
        "canonical merge.",
        "",
        "## Run",
        "",
        "| field | value |",
        "|---|---|",
        "| profile | `%s` (%s) |" % (report["profile"],
                                      report["profile_description"]),
        "| mode | `%s` |" % report["mode"],
        "| budget source | `%s` |" % budget["source"],
        "| configured budget | %s |" % _format_bytes(budget["effective_bytes"]),
        "| hard ceiling | %s |" % _format_bytes(budget["hard_bytes"]),
        "| records | %d / %d requested |" % (
            workload["records"], workload["requested_records"]),
        "| budget clamped | %s |" % ("yes" if workload["budget_clamped"] else "no"),
        "| dimensions | %d |" % workload["dimensions"],
        "| top-k | %d |" % top_k["k"],
        "| queries | %d |" % latency["samples"],
        "",
        "## Memory",
        "",
        "| metric | value |",
        "|---|---:|",
        "| mapped bytes | %s |" % _format_bytes(memory["mapped_bytes"]),
        "| file size | %s |" % _format_bytes(memory["file_size_bytes"]),
        "| allocated blocks | %s |" % _format_bytes(memory["allocated_bytes"]),
        "| touched bytes | %s |" % _format_bytes(memory["touched_bytes"]),
        "| resident pages estimate | %d (%s) |" % (
            memory["resident_pages_estimate"],
            _format_bytes(memory["resident_bytes_estimate"])),
        "| budget utilisation | %.2f%% |" % (memory["budget_utilization"] * 100.0),
        "| budget headroom | %s |" % _format_bytes(memory["budget_headroom_bytes"]),
        "| RSS delta | %s |" % _format_bytes(memory["rss_delta_bytes"]),
        "",
        "## Latency and throughput",
        "",
        "| metric | value |",
        "|---|---:|",
        "| ingest | %.1f records/s |" % ingest["ops_per_sec"],
        "| ingest bandwidth | %.2f MiB/s |" % ingest["mib_per_sec"],
        "| query p50 | %s |" % _format_ms(latency["p50_ms"]),
        "| query p95 | %s |" % _format_ms(latency["p95_ms"]),
        "| query mean | %s |" % _format_ms(latency["mean_ms"]),
        "| checkpoint total | %s |" % _format_ms(checkpoint["total_ms"]),
        "| checkpoint fsync | %s |" % _format_ms(checkpoint["fsync_ms"]),
        "| reopen | %s |" % _format_ms(reopen["reopen_ms"]),
        "",
        "## Correctness",
        "",
        "| metric | value |",
        "|---|---:|",
        "| top-1 accuracy vs brute force | %.4f |" % top_k["top1_accuracy"],
        "| planted top-1 accuracy | %.4f |" % top_k["planted_top1_accuracy"],
        "| recall@k | %.4f |" % top_k["recall_at_k"],
        "| exact top-k queries | %d / %d |" % (
            top_k["exact_queries"], top_k["queries"]),
        "| reopen count verified | %s |" % reopen["count_verified"],
        "| reopen digest verified | %s |" % reopen["digest_verified"],
        "| overall | %s |" % ("PASS" if verdict["passed"] else "FAIL"),
        "",
        "## Boundaries",
        "",
    ]
    lines.extend("- %s" % item for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_reports(report: dict[str, Any], json_path: str | Path | None = None,
                  markdown_path: str | Path | None = None) -> None:
    """Write JSON and/or Markdown reports. Parent directories are created."""
    if json_path is not None:
        _write_text(Path(json_path),
                    json.dumps(report, indent=2, sort_keys=True) + "\n")
    if markdown_path is not None:
        _write_text(Path(markdown_path), render_markdown(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Benchmark a synthetic resident-universe mmap store under "
                     "a phone memory budget. Default profile: high (256 MiB)."),
        epilog=("Quick CI example: python3 scripts/resident_benchmark.py "
                "--profile low --quick --json /tmp/resident-low.json "
                "--markdown /tmp/resident-low.md"),
    )
    parser.add_argument("--profile", choices=sorted(PHONE_PROFILES),
                        default=DEFAULT_PROFILE,
                        help="phone profile (default: high)")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help=("resident budget in MiB; default is the profile "
                              "recommendation (high = 256 MiB); hard ceiling "
                              "is 1024 MiB"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="small CI-sized run (default)")
    mode.add_argument("--full", action="store_true",
                      help="larger opt-in run; no live model is used")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="deterministic synthetic data seed")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the JSON report to this path")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="write the Markdown report to this path")
    parser.add_argument("--store-dir", type=Path, default=None,
                        help="keep the synthetic store in this directory")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        budget_bytes = budget_from_mb(args.budget_mb)
        report = run_benchmark(
            profile=args.profile,
            quick=not args.full,
            full=args.full,
            budget_bytes=budget_bytes,
            store_dir=args.store_dir,
            seed=args.seed,
        )
    except BudgetError as exc:
        print("resident_benchmark: error: %s" % exc, file=sys.stderr)
        return 2

    write_reports(report, args.json, args.markdown)
    if args.json is None and args.markdown is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = {
            "schema": report["schema"],
            "profile": report["profile"],
            "mode": report["mode"],
            "budget_bytes": report["budget"]["effective_bytes"],
            "records": report["workload"]["records"],
            "query_p50_ms": report["query_latency"]["p50_ms"],
            "query_p95_ms": report["query_latency"]["p95_ms"],
            "top_k_passed": report["top_k"]["passed"],
            "passed": report["verdict"]["passed"],
            "json": str(args.json) if args.json else None,
            "markdown": str(args.markdown) if args.markdown else None,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
