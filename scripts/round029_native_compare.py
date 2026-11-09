#!/usr/bin/env python3
"""Compare the Round 029 C++ root with a Python raw-payload implementation.

The supplied C++ program (``experiments/round029/nan_round029/native_mirror.cpp``)
builds one 8,388,608-byte root of 256 heads and 32,640 mirror pairs, then checks
skew-pair storage, gauge cancellation, finite-difference gradients and a local
loss step.  This script verifies the supplied SHA256SUMS manifest first, then
compiles the program in a temporary directory with a bounded timeout, parses
its JSON stdout, and compares the measured invariants with a Python
bytearray-backed implementation of the same raw geometry.

The C++ read timing is a host measurement only.  It is not an Android
benchmark and does not support a portable-throughput claim.  With
``--python-only``, or when no compiler is available, only the Python raw
payload invariants are computed; native execution is explicitly not claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

SCHEMA = "nanlm.round029-native-compare/1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REL = Path("experiments/round029/nan_round029/native_mirror.cpp")
DEFAULT_SOURCE = ROOT / DEFAULT_SOURCE_REL
DEFAULT_TIMEOUT_SECONDS = 120.0
COMPILER_CANDIDATES = ("g++", "clang++", "c++")

# The supplied Round 029 native test targets an Android/aarch64 device.  The
# report always states whether the local measurement actually matched it.
DEFAULT_TARGET_PLATFORM = "android"
DEFAULT_TARGET_ARCHITECTURE = "aarch64"

# Mirrors the compile-time constants and struct layout in native_mirror.cpp.
N = 256
HEAD_BYTES = 128
PAIR_BYTES = 256
PAIRS = N * (N - 1) // 2
ROOT_BYTES = N * HEAD_BYTES + PAIRS * PAIR_BYTES
READS = 500_000
GAUGE_BOUND = 3e-6
FINITE_DIFFERENCE_BOUND = 2e-3
_LITTLE_FLOAT = struct.Struct("<f")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2
EXIT_DENIED = 3

_INT_FIELDS = ("root_bytes", "mirror_pairs", "reads", "mlock_status")
_FLOAT_FIELDS = (
    "max_gauge_error",
    "max_reverse_error",
    "finite_difference_error",
    "loss_before_step",
    "loss_after_step",
    "read_milliseconds",
)
_REQUIRED_NATIVE_FIELDS = _INT_FIELDS + _FLOAT_FIELDS + ("passed",)
_SUM_LINE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})[ \t]+(?P<name>\*?.+)$")
_ARCH_ALIASES = {
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "armv7l": "armv7",
    "armv7a": "armv7",
    "armv8l": "aarch64",
    "amd64": "x86_64",
    "x86-64": "x86_64",
    "x86_64": "x86_64",
    "i386": "x86",
    "i486": "x86",
    "i586": "x86",
    "i686": "x86",
}


class VerificationError(RuntimeError):
    """Raised when SHA256SUMS verification must deny execution."""


class NativeBuildError(RuntimeError):
    """Raised when native compilation fails."""


class NativeRunError(RuntimeError):
    """Raised when the native executable cannot be run."""


class NativeOutputError(RuntimeError):
    """Raised when native stdout is missing or malformed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
                f"invalid SHA256SUMS line {line_number}: expected '<sha256>  <path>'")
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


def verify_supplied_files(
    source: Path | str,
    manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Verify every supplied digest, rejecting traversal and source mismatch.

    ``manifest`` defaults to ``SHA256SUMS`` next to the source.  The source must
    be covered by the manifest, live under the manifest directory, and match its
    recorded SHA-256 digest.  No native code is run before this succeeds.
    """
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (ROOT / source_path).resolve()
    else:
        source_path = source_path.resolve()
    manifest_path = (
        Path(manifest) if manifest is not None else source_path.parent / "SHA256SUMS"
    )
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise VerificationError(f"SHA256SUMS not found: {manifest_path}")
    base = manifest_path.parent.resolve()
    if not source_path.is_file():
        raise VerificationError(f"source file not found: {source_path}")
    if not _within(source_path, base):
        raise VerificationError(
            f"source escapes manifest directory: {source_path} not under {base}")

    entries = _read_manifest(manifest_path)
    verified: list[dict[str, str]] = []
    errors: list[str] = []
    source_sha256: str | None = None
    source_listed = False
    for entry in entries:
        name = entry["path"]
        candidate = base / name
        if not _within(candidate, base):
            raise VerificationError(f"path traversal rejected for {name!r}")
        if not candidate.is_file():
            errors.append(f"listed file missing: {name}")
            continue
        measured = _sha256_file(candidate)
        if measured != entry["sha256"]:
            errors.append(
                f"SHA-256 mismatch for {name}: expected {entry['sha256']}, "
                f"measured {measured}")
            continue
        verified.append({"path": name, "sha256": measured})
        try:
            is_source = candidate.resolve() == source_path
        except OSError:
            is_source = False
        if is_source:
            source_listed = True
            source_sha256 = measured

    if errors:
        raise VerificationError("; ".join(errors))
    if not source_listed or source_sha256 is None:
        raise VerificationError(
            f"source is not covered by SHA256SUMS: {source_path.name}")
    return {
        "status": "verified",
        "manifest": str(manifest_path),
        "base": str(base),
        "entry_count": len(entries),
        "verified": verified,
        "source_listed": True,
        "source_sha256": source_sha256,
    }


def _normalise_arch(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "unknown"
    return _ARCH_ALIASES.get(raw, raw)


def _host_platform() -> str:
    if hasattr(sys, "getandroidapilevel"):
        return "android"
    if os.environ.get("ANDROID_ROOT") or os.environ.get("TERMUX_VERSION"):
        return "android"
    raw = (platform.system() or "").strip().lower()
    if "android" in raw:
        return "android"
    if raw.startswith("linux"):
        return "linux"
    if raw.startswith("darwin"):
        return "darwin"
    if raw.startswith("windows"):
        return "windows"
    return raw or "unknown"


def _capture(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=max(0.1, timeout),
        check=False,
    )


def _trim(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def find_compiler(candidates: Sequence[str] = COMPILER_CANDIDATES) -> str | None:
    """Return the first available C++ compiler path, or None."""
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def inspect_compiler(
    compiler: str,
    *,
    timeout: float,
    target_platform: str,
    target_architecture: str,
) -> dict[str, Any]:
    """Report the compiler version and its own target triple."""
    version: str | None = None
    triple: str | None = None
    version_error: str | None = None
    try:
        proc = _capture([compiler, "--version"], min(timeout, 10.0))
        first = (proc.stdout or proc.stderr or "").strip().splitlines()
        version = first[0] if first else None
        if proc.returncode != 0:
            version_error = f"compiler --version exited {proc.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        version_error = f"compiler --version failed: {exc}"
    try:
        proc = _capture([compiler, "-dumpmachine"], min(timeout, 10.0))
        text = (proc.stdout or "").strip().splitlines()
        triple = text[0].strip() if text else None
    except (OSError, subprocess.TimeoutExpired):
        triple = None

    compiler_arch = _normalise_arch(triple.split("-", 1)[0]) if triple else "unknown"
    triple_lower = (triple or "").lower()
    target_platform_norm = target_platform.strip().lower()
    target_arch_norm = _normalise_arch(target_architecture)
    platform_matches = (
        target_platform_norm != "android" or "android" in triple_lower
    )
    return {
        "name": Path(compiler).name,
        "path": compiler,
        "version": version,
        "version_error": version_error,
        "target_triple": triple,
        "target_architecture": compiler_arch,
        "target_matches": compiler_arch == target_arch_norm and platform_matches,
        "target_platform": target_platform_norm,
        "target_architecture_expected": target_arch_norm,
    }


def build_environment(
    *,
    compiler: dict[str, Any] | None,
    timeout: float,
    target_platform: str,
    target_architecture: str,
) -> dict[str, Any]:
    host_platform = _host_platform()
    host_arch = _normalise_arch(platform.machine())
    target_platform_norm = target_platform.strip().lower() or "unknown"
    target_arch_norm = _normalise_arch(target_architecture)
    return {
        "python": platform.python_version(),
        "host": {
            "platform": host_platform,
            "architecture": host_arch,
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
        },
        "target": {
            "platform": target_platform_norm,
            "architecture": target_arch_norm,
        },
        "matches_target": (
            host_platform == target_platform_norm and host_arch == target_arch_norm
        ),
        "compiler": compiler,
        "timeout_seconds": float(timeout),
    }


def compile_native(
    source: Path | str,
    compiler: str,
    *,
    timeout: float,
    workdir: Path | str,
) -> dict[str, Any]:
    """Compile into ``workdir`` with a bounded timeout."""
    binary = Path(workdir) / "native_mirror"
    command = [
        compiler,
        "-std=c++20",
        "-O3",
        "-Wall",
        "-Wextra",
        str(source),
        "-o",
        str(binary),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeBuildError(
            f"compile timed out after {timeout:g}s: {' '.join(command)}") from exc
    except OSError as exc:
        raise NativeBuildError(f"cannot execute compiler {compiler}: {exc}") from exc
    if proc.returncode != 0:
        raise NativeBuildError(
            f"compile failed with exit {proc.returncode}: "
            f"{_trim(proc.stderr or proc.stdout)}")
    if not binary.is_file():
        raise NativeBuildError("compiler reported success but produced no binary")
    return {
        "command": command,
        "returncode": proc.returncode,
        "stderr": _trim(proc.stderr or ""),
        "binary": str(binary),
    }


def run_native(
    binary: Path | str,
    *,
    timeout: float,
    workdir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the compiled root with a bounded timeout."""
    try:
        proc = subprocess.run(
            [str(binary)],
            cwd=str(workdir) if workdir is not None else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeRunError(
            f"native run timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise NativeRunError(f"cannot run native binary: {exc}") from exc
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": _trim(proc.stderr or ""),
        "timed_out": False,
    }


def _normalise_measurement(obj: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise NativeOutputError(f"{label} is not a JSON object")
    missing = [field for field in _REQUIRED_NATIVE_FIELDS if field not in obj]
    if missing:
        raise NativeOutputError(f"{label} missing fields: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for field in _INT_FIELDS:
        value = obj[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NativeOutputError(f"{label} field {field} is not numeric")
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise NativeOutputError(f"{label} field {field} is not an integer")
        result[field] = int(number)
    for field in _FLOAT_FIELDS:
        value = obj[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NativeOutputError(f"{label} field {field} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise NativeOutputError(f"{label} field {field} is not finite")
        result[field] = number
    if not isinstance(obj["passed"], bool):
        raise NativeOutputError(f"{label} field passed is not boolean")
    result["passed"] = obj["passed"]
    return result


def parse_native_stdout(stdout: str) -> dict[str, Any]:
    """Parse the JSON object emitted by native_mirror.cpp."""
    candidates: list[str] = []
    stripped = (stdout or "").strip()
    if stripped:
        candidates.append(stripped)
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line and line not in candidates:
            candidates.append(line)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(
            field in obj for field in _REQUIRED_NATIVE_FIELDS
        ):
            return _normalise_measurement(obj, label="native output")
    raise NativeOutputError(
        "native stdout did not contain a complete JSON measurement object")


def _pair_id(i: int, j: int) -> int:
    if i == j or i < 0 or j < 0 or i >= N or j >= N:
        raise ValueError(f"invalid mirror-pair indices ({i}, {j})")
    if i > j:
        i, j = j, i
    return i * (2 * N - i - 1) // 2 + (j - i - 1)


def _pair_base(i: int, j: int) -> int:
    return N * HEAD_BYTES + _pair_id(i, j) * PAIR_BYTES


def _lane_offset(i: int, j: int) -> int:
    # forward[0] at +0, reverse[0] at +HEAD_BYTES; the pair id is canonical.
    return _pair_base(i, j) + (0 if i < j else HEAD_BYTES)


def _read_direction(payload: bytearray, i: int, j: int) -> float:
    return _LITTLE_FLOAT.unpack_from(payload, _lane_offset(i, j))[0]


def _write_direction(payload: bytearray, i: int, j: int, value: float) -> None:
    _LITTLE_FLOAT.pack_into(payload, _lane_offset(i, j), value)


def _store_skew(payload: bytearray, i: int, j: int, value: float) -> None:
    _write_direction(payload, i, j, value)
    _write_direction(payload, j, i, -value)


def _circulation(payload: bytearray, i: int, j: int) -> float:
    return (
        _read_direction(payload, 0, i)
        + _read_direction(payload, i, j)
        + _read_direction(payload, j, 0)
    )


def _gradient_step(
    payload: bytearray,
    i: int,
    j: int,
    gradient: float,
    learning_rate: float,
) -> None:
    delta = -learning_rate * gradient
    _store_skew(payload, 0, i, _read_direction(payload, 0, i) + delta)
    _store_skew(payload, i, j, _read_direction(payload, i, j) + delta)
    _store_skew(payload, j, 0, _read_direction(payload, j, 0) + delta)


def python_raw_measurement(reads: int = READS) -> dict[str, Any]:
    """Compute the Round 029 raw payload invariants in pure Python.

    This is a bytearray-backed mirror of the C++ layout, not native execution.
    """
    if reads < 1:
        raise ValueError("reads must be positive")
    payload = bytearray(ROOT_BYTES)
    if len(payload) != ROOT_BYTES:
        raise AssertionError("payload allocation mismatch")

    checked = 0
    alignment_ok = True
    for i in range(N):
        for j in range(i + 1, N):
            forward = _pair_base(i, j)
            reverse = _pair_base(j, i)
            if forward != reverse:
                raise AssertionError("mirror pairs do not share storage")
            if _lane_offset(i, j) != forward or _lane_offset(j, i) != reverse + HEAD_BYTES:
                raise AssertionError("forward/reverse lanes are not addressable")
            if forward % 64 or reverse % 64:
                alignment_ok = False
            checked += 1
    if checked != PAIRS:
        raise AssertionError(f"expected {PAIRS} mirror pairs, checked {checked}")

    import random

    rng = random.Random(2029)
    max_gauge = 0.0
    max_reverse = 0.0
    for _ in range(100):
        gauge = [rng.gauss(0.0, 1.0) * 3.0 for _ in range(N)]
        for k in range(48):
            a = 1 + 2 * k
            b = 2 + 2 * k
            semantic = rng.gauss(0.0, 1.0)
            _store_skew(payload, 0, a, gauge[0] - gauge[a])
            _store_skew(payload, 0, b, gauge[0] - gauge[b])
            _store_skew(payload, a, b, semantic + gauge[a] - gauge[b])
            max_gauge = max(
                max_gauge,
                abs(_circulation(payload, a, b) - semantic),
            )
            max_reverse = max(
                max_reverse,
                abs(_read_direction(payload, b, a) + _read_direction(payload, a, b)),
            )

    a, b = 3, 4
    target = 0.42
    before = _read_direction(payload, a, b)

    def loss() -> float:
        error = _circulation(payload, a, b) - target
        return 0.5 * error * error

    delta = 1e-3
    _store_skew(payload, a, b, before + delta)
    high = loss()
    _store_skew(payload, a, b, before - delta)
    low = loss()
    _store_skew(payload, a, b, before)
    finite_difference = (high - low) / (2.0 * delta)
    analytic = _circulation(payload, a, b) - target
    gradient_error = abs(finite_difference - analytic)
    loss_before = loss()
    _gradient_step(payload, a, b, analytic, 0.1)
    loss_after = loss()

    triangle_offsets = [
        (
            _lane_offset(0, 1 + 2 * k),
            _lane_offset(1 + 2 * k, 2 + 2 * k),
            _lane_offset(2 + 2 * k, 0),
        )
        for k in range(48)
    ]
    unpack = _LITTLE_FLOAT.unpack_from
    sink = 0.0
    started = time.perf_counter()
    for k in range(reads):
        offset_a, offset_b, offset_c = triangle_offsets[k % 48]
        sink += (
            unpack(payload, offset_a)[0]
            + unpack(payload, offset_b)[0]
            + unpack(payload, offset_c)[0]
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    passed = (
        max_gauge < GAUGE_BOUND
        and max_reverse == 0.0
        and gradient_error < FINITE_DIFFERENCE_BOUND
        and loss_after < loss_before
    )
    return {
        "implementation": "python-bytearray-float32",
        "root_bytes": len(payload),
        "mirror_pairs": checked,
        "max_gauge_error": max_gauge,
        "max_reverse_error": max_reverse,
        "finite_difference_error": gradient_error,
        "loss_before_step": loss_before,
        "loss_after_step": loss_after,
        "reads": reads,
        "read_milliseconds": elapsed_ms,
        "mlock_status": None,
        "passed": passed,
        "layout": {
            "heads": N,
            "head_bytes": HEAD_BYTES,
            "pair_bytes": PAIR_BYTES,
            "pairs": PAIRS,
            "root_bytes": ROOT_BYTES,
            "alignment_ok": alignment_ok,
        },
        "read_sink": sink,
    }


def compare_measurements(
    native: dict[str, Any],
    python: dict[str, Any],
) -> dict[str, Any]:
    """Compare native and Python raw-geometry invariants against fixed bounds."""
    fields: dict[str, dict[str, Any]] = {}

    native_root = int(native["root_bytes"])
    python_root = int(python["root_bytes"])
    fields["root_bytes"] = {
        "native": native_root,
        "python": python_root,
        "expected": ROOT_BYTES,
        "match": native_root == python_root == ROOT_BYTES,
    }

    native_pairs = int(native["mirror_pairs"])
    python_pairs = int(python["mirror_pairs"])
    fields["mirror_pairs"] = {
        "native": native_pairs,
        "python": python_pairs,
        "expected": PAIRS,
        "match": native_pairs == python_pairs == PAIRS,
    }

    native_gauge = float(native["max_gauge_error"])
    python_gauge = float(python["max_gauge_error"])
    fields["max_gauge_error"] = {
        "native": native_gauge,
        "python": python_gauge,
        "bound": GAUGE_BOUND,
        "match": (
            0.0 <= native_gauge < GAUGE_BOUND
            and 0.0 <= python_gauge < GAUGE_BOUND
            and abs(native_gauge - python_gauge) < GAUGE_BOUND
        ),
    }

    native_reverse = float(native["max_reverse_error"])
    python_reverse = float(python["max_reverse_error"])
    fields["max_reverse_error"] = {
        "native": native_reverse,
        "python": python_reverse,
        "expected": 0.0,
        "match": native_reverse == 0.0 and python_reverse == 0.0,
    }

    native_fd = float(native["finite_difference_error"])
    python_fd = float(python["finite_difference_error"])
    fields["finite_difference_error"] = {
        "native": native_fd,
        "python": python_fd,
        "bound": FINITE_DIFFERENCE_BOUND,
        "match": (
            0.0 <= native_fd < FINITE_DIFFERENCE_BOUND
            and 0.0 <= python_fd < FINITE_DIFFERENCE_BOUND
            and abs(native_fd - python_fd) < FINITE_DIFFERENCE_BOUND
        ),
    }

    native_before = float(native["loss_before_step"])
    native_after = float(native["loss_after_step"])
    python_before = float(python["loss_before_step"])
    python_after = float(python["loss_after_step"])
    fields["loss_decrease"] = {
        "native_before": native_before,
        "native_after": native_after,
        "python_before": python_before,
        "python_after": python_after,
        "native_decreased": native_after < native_before,
        "python_decreased": python_after < python_before,
        "match": native_after < native_before and python_after < python_before,
    }

    native_reads = int(native["reads"])
    python_reads = int(python["reads"])
    fields["reads"] = {
        "native": native_reads,
        "python": python_reads,
        "expected": READS,
        "match": native_reads == python_reads == READS,
    }

    native_time = float(native["read_milliseconds"])
    python_time = float(python["read_milliseconds"])
    fields["read_milliseconds"] = {
        "native": native_time,
        "python": python_time,
        "comparable": False,
        "native_label": "C++ host time, not Android",
        "python_label": "Python host time, not native",
        "note": (
            "Timing is host-specific and is never compared for equality; only "
            "the 500,000-read count is compared."
        ),
    }

    matched = all(
        bool(value["match"]) for value in fields.values() if "match" in value
    ) and bool(native.get("passed")) and bool(python.get("passed"))
    return {
        "native_available": True,
        "matched": matched,
        "fields": fields,
        "bounds": {
            "root_bytes": ROOT_BYTES,
            "mirror_pairs": PAIRS,
            "max_gauge_error": GAUGE_BOUND,
            "max_reverse_error": 0.0,
            "finite_difference_error": FINITE_DIFFERENCE_BOUND,
            "loss_decrease": True,
            "reads": READS,
        },
    }


def _python_only_comparison() -> dict[str, Any]:
    return {
        "native_available": False,
        "matched": None,
        "fields": {},
        "bounds": {
            "root_bytes": ROOT_BYTES,
            "mirror_pairs": PAIRS,
            "max_gauge_error": GAUGE_BOUND,
            "max_reverse_error": 0.0,
            "finite_difference_error": FINITE_DIFFERENCE_BOUND,
            "loss_decrease": True,
            "reads": READS,
        },
        "note": (
            "No native measurement was made; Python raw-payload invariants are "
            "reported without claiming native execution."
        ),
    }


def load_reference(path: Path | str) -> dict[str, Any]:
    """Load the supplied prior native_results.json as a reference artifact."""
    reference_path = Path(path)
    result: dict[str, Any] = {
        "path": str(reference_path),
        "status": "missing",
        "measurement": None,
        "note": "supplied artifact; it is not a measurement from this run",
    }
    if not reference_path.is_file():
        return result
    try:
        obj = json.loads(reference_path.read_text(encoding="utf-8"))
        result["measurement"] = _normalise_measurement(obj, label="reference")
        result["status"] = "loaded"
    except (OSError, json.JSONDecodeError, NativeOutputError) as exc:
        result["status"] = "invalid"
        result["reason"] = str(exc)
    return result


def _claims(native_executed: bool) -> dict[str, Any]:
    return {
        "canonical": False,
        "canonical_implementation": False,
        "native_execution": bool(native_executed),
        "native_benchmark": False,
        "android_benchmark": False,
        "portable_throughput": False,
        "semantic_superiority": False,
        "quantum_claim": False,
        "scope": (
            "local host-side raw-geometry invariant comparison only; no "
            "canonical, Android, portable-throughput, semantic or quantum claim"
        ),
    }


def build_report(
    *,
    status: str,
    mode: str,
    source: Path,
    verification: dict[str, Any],
    environment: dict[str, Any],
    native: dict[str, Any],
    python: dict[str, Any] | None,
    comparison: dict[str, Any],
    reference: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    try:
        display_path = source.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        display_path = str(source)
    return {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "mode": mode,
        "source": {
            "path": str(source),
            "display_path": display_path,
            "sha256": verification.get("source_sha256"),
        },
        "verification": verification,
        "environment": environment,
        "native": native,
        "python": python,
        "reference": reference,
        "comparison": comparison,
        "claims": _claims(native.get("measurement") is not None),
        "notes": notes,
    }


def _fmt_number(value: Any, digits: int = 6) -> str:
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
    """Render the comparison report as Markdown."""
    environment = report["environment"]
    host = environment["host"]
    target = environment["target"]
    verification = report["verification"]
    native = report["native"]
    python = report.get("python")
    comparison = report["comparison"]
    reference = report["reference"]
    fields = comparison.get("fields", {})

    native_measurement = native.get("measurement")
    python_measurement = python.get("measurement") if python else None

    def field(name: str, key: str = "native") -> Any:
        entry = fields.get(name, {})
        if key in entry:
            return entry[key]
        if key in ("native", "native_before", "native_after") and native_measurement:
            measurement_key = {
                "native": name,
                "native_before": "loss_before_step",
                "native_after": "loss_after_step",
            }[key]
            return native_measurement.get(measurement_key)
        if key in ("python", "python_before", "python_after") and python_measurement:
            measurement_key = {
                "python": name,
                "python_before": "loss_before_step",
                "python_after": "loss_after_step",
            }[key]
            return python_measurement.get(measurement_key)
        return None

    lines = [
        "# Round 029 native / Python raw geometry comparison",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Mode:** `{report['mode']}`",
        f"- **Source:** `{report['source']['display_path']}`",
        f"- **Source SHA-256:** `{report['source'].get('sha256') or 'n/a'}`",
        "",
        "## Environment",
        "",
        "| Item | Measured | Target |",
        "| --- | --- | --- |",
        f"| Platform | {host['platform']} | {target['platform']} |",
        f"| Architecture | {host['architecture']} | {target['architecture']} |",
        f"| Target match | {'yes' if environment['matches_target'] else 'no'} | - |",
    ]
    compiler = environment.get("compiler")
    if compiler:
        lines.append(
            f"| Compiler | {compiler.get('name')}: "
            f"{compiler.get('version') or 'version unknown'} | "
            f"{compiler.get('target_triple') or 'triple unknown'} |")
        lines.append(
            f"| Compiler target match | "
            f"{'yes' if compiler.get('target_matches') else 'no'} | - |")
    else:
        lines.append("| Compiler | not found / not used | - |")
    lines += [
        f"| Timeout | {_fmt_number(environment.get('timeout_seconds'), 4)} s | "
        "bounded compile and run |",
        "",
        "## Digest verification",
        "",
        f"- Manifest: `{verification.get('manifest', 'n/a')}`",
        f"- Status: `{verification.get('status', 'n/a')}`",
        f"- Verified entries: {verification.get('entry_count', 'n/a')}",
        f"- Source covered: {'yes' if verification.get('source_listed') else 'no'}",
    ]
    if verification.get("reason"):
        lines.append(f"- Reason: {verification['reason']}")
    lines += [
        "",
        "## Measurements",
        "",
        "The root is 8,388,608 bytes: 256 × 128-byte heads plus 32,640 × "
        "256-byte mirror-pair records.",
        "",
        "| Invariant | C++ native | Python raw | Comparison |",
        "| --- | ---: | ---: | --- |",
        f"| Root bytes | {_fmt_number(field('root_bytes'))} | "
        f"{_fmt_number(field('root_bytes', 'python'))} | "
        f"{_fmt_number(field('root_bytes', 'match'))} |",
        f"| Mirror pairs | {_fmt_number(field('mirror_pairs'))} | "
        f"{_fmt_number(field('mirror_pairs', 'python'))} | "
        f"{_fmt_number(field('mirror_pairs', 'match'))} |",
        f"| Max gauge error | {_fmt_number(field('max_gauge_error'))} | "
        f"{_fmt_number(field('max_gauge_error', 'python'))} | "
        f"{_fmt_number(field('max_gauge_error', 'match'))} |",
        f"| Max reverse error | {_fmt_number(field('max_reverse_error'))} | "
        f"{_fmt_number(field('max_reverse_error', 'python'))} | "
        f"{_fmt_number(field('max_reverse_error', 'match'))} |",
        f"| Finite-difference error | "
        f"{_fmt_number(field('finite_difference_error'))} | "
        f"{_fmt_number(field('finite_difference_error', 'python'))} | "
        f"{_fmt_number(field('finite_difference_error', 'match'))} |",
        f"| Loss before step | "
        f"{_fmt_number(field('loss_decrease', 'native_before'))} | "
        f"{_fmt_number(field('loss_decrease', 'python_before'))} | observed |",
        f"| Loss after step | "
        f"{_fmt_number(field('loss_decrease', 'native_after'))} | "
        f"{_fmt_number(field('loss_decrease', 'python_after'))} | "
        f"{_fmt_number(field('loss_decrease', 'match'))} |",
        f"| Reads | {_fmt_number(field('reads'))} | "
        f"{_fmt_number(field('reads', 'python'))} | "
        f"{_fmt_number(field('reads', 'match'))} |",
    ]
    read_time = fields.get("read_milliseconds")
    if read_time is None:
        read_time = {
            "native": (
                native_measurement.get("read_milliseconds")
                if native_measurement else None
            ),
            "python": (
                python_measurement.get("read_milliseconds")
                if python_measurement else None
            ),
        }
    lines.append(
        f"| Read time (host-only) | "
        f"{_fmt_number(read_time.get('native'), 5)} ms | "
        f"{_fmt_number(read_time.get('python'), 5)} ms | "
        "not comparable |")
    lines += [""]

    if comparison.get("native_available"):
        lines += [
            f"Overall invariant match: **{'yes' if comparison.get('matched') else 'no'}**.",
            "",
        ]
    else:
        lines += [
            "Native comparison unavailable: "
            f"{native.get('reason') or comparison.get('note') or 'not run'}.",
            "",
        ]

    reference_measurement = reference.get("measurement")
    lines += [
        "## Supplied reference",
        "",
        f"- Status: `{reference.get('status', 'n/a')}`",
        f"- Path: `{reference.get('path', 'n/a')}`",
    ]
    if reference_measurement:
        lines.append(
            "- Reference root bytes / pairs: "
            f"{reference_measurement['root_bytes']} / "
            f"{reference_measurement['mirror_pairs']} (prior artifact, not "
            "re-executed)")
    lines += [
        "",
        "## Boundaries",
        "",
        "- The C++ read time is host-only. It is **not** an Android benchmark "
        "and must not be used as a portable-throughput claim.",
        "- Python-only mode executes a bytearray model of the raw geometry; it "
        "does **not** execute native code or claim a native measurement.",
        "- No canonical implementation, semantic-superiority, quantum, or "
        "device-performance claim is made.",
        "- SHA-256 digests are checked before any compilation or execution.",
    ]
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path | str, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path | str, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(report), encoding="utf-8")


def _print_receipt(report: dict[str, Any], out_json: str | None, out_md: str | None) -> None:
    receipt = {
        "schema": report["schema"],
        "status": report["status"],
        "mode": report["mode"],
        "native_status": report["native"].get("status"),
        "comparison_matched": report["comparison"].get("matched"),
        "out_json": out_json,
        "out_md": out_md,
    }
    print(json.dumps(receipt, indent=2, allow_nan=False))


def _base_source_path(value: str | None) -> Path:
    if value is None:
        return DEFAULT_SOURCE
    source = Path(value)
    if not source.is_absolute():
        source = ROOT / source
    return source.resolve()


def _denied_report(
    *,
    source: Path,
    reason: str,
    timeout: float,
    target_platform: str,
    target_architecture: str,
    manifest: Path | None = None,
) -> dict[str, Any]:
    verification = {
        "status": "denied",
        "reason": reason,
        "manifest": str(manifest) if manifest is not None else None,
        "entry_count": 0,
        "verified": [],
        "source_listed": False,
        "source_sha256": None,
    }
    environment = build_environment(
        compiler=None,
        timeout=timeout,
        target_platform=target_platform,
        target_architecture=target_architecture,
    )
    native = {
        "status": "not_run",
        "reason": "integrity verification failed before native execution",
        "measurement": None,
    }
    return build_report(
        status="denied",
        mode="python-only",
        source=source,
        verification=verification,
        environment=environment,
        native=native,
        python=None,
        comparison=_python_only_comparison(),
        reference={"path": str(source.parent / "native_results.json"),
                   "status": "not_loaded", "measurement": None},
        notes=[
            "Execution was denied because the supplied digest manifest did not "
            "verify.",
        ],
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify, compile and run the Round 029 C++ root, then compare its "
            "raw geometry with a Python raw-payload implementation."
        )
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_REL),
        help=f"C++ source (default: {DEFAULT_SOURCE_REL})",
    )
    parser.add_argument(
        "--sha256sums",
        default=None,
        help="digest manifest (default: SHA256SUMS next to --source)",
    )
    parser.add_argument("--out-json", default=None, help="write the JSON report here")
    parser.add_argument("--out-md", default=None, help="write the Markdown report here")
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="skip compiler detection/execution and report Python raw invariants",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"compile/run timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--target-platform",
        default=DEFAULT_TARGET_PLATFORM,
        help=f"expected target platform (default: {DEFAULT_TARGET_PLATFORM})",
    )
    parser.add_argument(
        "--target-architecture",
        default=DEFAULT_TARGET_ARCHITECTURE,
        help=f"expected target architecture (default: {DEFAULT_TARGET_ARCHITECTURE})",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number of seconds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source = _base_source_path(args.source)
    out_json = args.out_json
    out_md = args.out_md
    timeout = float(args.timeout)
    target_platform = args.target_platform
    target_architecture = args.target_architecture

    if args.sha256sums:
        manifest_path = Path(args.sha256sums)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        manifest_path = manifest_path.resolve()
    else:
        manifest_path = source.parent / "SHA256SUMS"
    try:
        verification = verify_supplied_files(source, manifest_path)
    except VerificationError as exc:
        report = _denied_report(
            source=source,
            reason=str(exc),
            timeout=timeout,
            target_platform=target_platform,
            target_architecture=target_architecture,
            manifest=manifest_path,
        )
        if out_json:
            _write_json(out_json, report)
        if out_md:
            _write_markdown(out_md, report)
        _print_receipt(report, out_json, out_md)
        return EXIT_DENIED

    python_measurement = python_raw_measurement()
    python_block = {
        "status": "ok" if python_measurement["passed"] else "failed",
        "measurement": python_measurement,
        "implementation": "python-bytearray-float32",
    }

    compiler_info: dict[str, Any] | None = None
    compile_info: dict[str, Any] | None = None
    native_measurement: dict[str, Any] | None = None
    native_status = "not_requested"
    native_reason: str | None = None
    mode = "python-only"
    notes: list[str] = []

    if args.python_only:
        native = {
            "status": "not_requested",
            "reason": "--python-only selected",
            "measurement": None,
        }
    else:
        compiler = find_compiler()
        if compiler is None:
            native_status = "skipped"
            native_reason = (
                "no C++ compiler found (tried g++, clang++, c++); Python raw "
                "fallback only"
            )
            native = {
                "status": native_status,
                "reason": native_reason,
                "measurement": None,
            }
            notes.append(
                "Native execution was skipped because no C++ compiler was "
                "available; no native measurement is claimed.")
        else:
            compiler_info = inspect_compiler(
                compiler,
                timeout=timeout,
                target_platform=target_platform,
                target_architecture=target_architecture,
            )
            try:
                with tempfile.TemporaryDirectory(prefix="round029-native-") as tmp:
                    workdir = Path(tmp)
                    compile_info = compile_native(
                        source, compiler, timeout=timeout, workdir=workdir)
                    run = run_native(
                        compile_info["binary"], timeout=timeout, workdir=workdir)
                    native_measurement = parse_native_stdout(run["stdout"])
                    native_status = "ok"
                    mode = "native-compare"
                    native = {
                        "status": native_status,
                        "reason": None,
                        "compiler": compiler,
                        "returncode": run["returncode"],
                        "command": compile_info["command"],
                        "stderr": run["stderr"],
                        "measurement": native_measurement,
                    }
                    if run["returncode"] not in (0, 5):
                        native_status = "failed"
                        native["status"] = native_status
                        native["reason"] = (
                            f"native binary exited {run['returncode']} with a "
                            "parseable measurement")
            except NativeBuildError as exc:
                native_status = (
                    "timeout" if "timed out" in str(exc) else "compile_failed")
                native = {
                    "status": native_status,
                    "reason": str(exc),
                    "compiler": compiler,
                    "measurement": None,
                }
                notes.append("Native comparison was not available because "
                             "compilation failed or exceeded its timeout.")
            except NativeRunError as exc:
                native_status = "timeout" if "timed out" in str(exc) else "run_failed"
                native = {
                    "status": native_status,
                    "reason": str(exc),
                    "compiler": compiler,
                    "measurement": None,
                }
                notes.append("Native comparison was not available because the "
                             "native run failed or exceeded its timeout.")
            except NativeOutputError as exc:
                native_status = "invalid_output"
                native = {
                    "status": native_status,
                    "reason": str(exc),
                    "compiler": compiler,
                    "measurement": None,
                }
                notes.append("Native comparison was not available because stdout "
                             "did not contain the expected JSON measurement.")

    comparison = (
        compare_measurements(native_measurement, python_measurement)
        if native_measurement is not None
        else _python_only_comparison()
    )
    environment = build_environment(
        compiler=compiler_info,
        timeout=timeout,
        target_platform=target_platform,
        target_architecture=target_architecture,
    )
    reference = load_reference(source.parent / "native_results.json")
    if reference["status"] == "loaded":
        notes.append(
            "The supplied native_results.json is reported as a prior reference "
            "artifact and was not re-executed.")
    if not environment["matches_target"]:
        notes.append(
            "Measured host platform/architecture differs from the stated target; "
            "timing is therefore local-host only.")

    if native_measurement is not None:
        status = (
            "ok"
            if (
                native_status == "ok"
                and comparison["matched"]
                and python_measurement["passed"]
            )
            else "failed"
        )
    elif native_status in {"compile_failed", "run_failed", "timeout", "invalid_output"}:
        status = "error"
    else:
        # --python-only and missing-compiler fallback are deliberate, reported
        # modes; they are not native measurements.
        status = "python-only"

    report = build_report(
        status=status,
        mode=mode,
        source=source,
        verification=verification,
        environment=environment,
        native=native,
        python=python_block,
        comparison=comparison,
        reference=reference,
        notes=notes,
    )
    if out_json:
        _write_json(out_json, report)
    if out_md:
        _write_markdown(out_md, report)
    if not out_json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _print_receipt(report, out_json, out_md)
    if status == "ok" or status == "python-only":
        return EXIT_OK
    if status == "failed":
        return EXIT_FAILED
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
