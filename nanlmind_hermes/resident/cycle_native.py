"""Optional, dependency-free ctypes bridge to the Round 030 native root.

The Round 030 experiment stores its classifier in one C++ allocation of
``8_388_608`` bytes (256 heads plus 32,640 mirror pairs).  This module exposes
that native root to Python without importing NumPy or PyTorch:

* the supplied ``native_cycle.cpp`` is verified against its ``SHA256SUMS``
  entry before anything is compiled;
* only the verified bytes are compiled, in a private temporary cache directory,
  with ``c++``/``g++``/``clang++`` and a bounded timeout;
* the resulting shared library is cached next to a digest manifest and that
  digest is checked again on every load;
* all Python values crossing the boundary are quantised to IEEE-754 binary32
  and widened back to Python ``float`` on return.  Records contain ordinary
  float32 values; this module makes **no** raw-NaN payload or NaN-arithmetic
  claim.
* no raw native pointer is exposed and no canonical model is merged into the
  resident universe.  Snapshots are the experiment's private ``NANCYC30``
  format: an 8-byte marker plus exactly 8 MiB of native root bytes.

If no compiler or library is available, :func:`probe_native_cycle` reports a
plain ``unavailable`` receipt and :class:`NativeCycleRoot` raises
:class:`NativeCycleUnavailable`.  Native measurements are never simulated.

This is an experimental, same-architecture, non-transactional bridge.  It is
not concurrency-safe across processes, endian-neutral, crash-safe or a
replacement for the resident universe format.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import numbers
import operator
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SCHEMA = "nanlm.hermes-native-cycle/1"
BUILD_SCHEMA = "nanlm.hermes-native-cycle-build/1"
SNAPSHOT_SCHEMA = "nanlm.hermes-native-cycle-snapshot/1"

# Layout constants copied from experiments/round030/native_cycle.cpp.  They are
# asserted against the compiled library through its behaviour and root size.
N = 256
D = 24
K = 16
PAIRS = N * (N - 1) // 2  # 32,640
ROOT_BYTES = N * 128 + PAIRS * 256  # 8,388,608
SNAPSHOT_MAGIC = b"NANCYC30"
SNAPSHOT_MARKER_BYTES = 8
SNAPSHOT_BYTES = ROOT_BYTES + SNAPSHOT_MARKER_BYTES

DEFAULT_SOURCE_REL = Path("experiments/round030/native_cycle.cpp")
COMPILER_CANDIDATES = ("c++", "g++", "clang++")
DEFAULT_COMPILE_TIMEOUT = 120.0
MAX_COMPILE_TIMEOUT = 600.0
MAX_SAMPLES = 8192
MAX_EPOCHS = 64
MAX_LR = 1.0
MAX_GAUGE_SIGMA = 32.0
MAX_PATH_BYTES = 4096
MAX_LIBRARY_BYTES = 64 * 1024 * 1024

_F32 = struct.Struct("<f")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SUM_LINE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})[ \t]+(?P<name>\*?.+)$")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SOURCE = _REPO_ROOT / DEFAULT_SOURCE_REL
_BUILD_LOCK = threading.RLock()

CLAIMS = {
    "native_execution": True,
    "canonical": False,
    "canonical_merge": False,
    "raw_nan_payload_arithmetic": False,
    "raw_nan_acceleration_claim": False,
    "numpy_required": False,
    "torch_required": False,
    "quantum_claim": False,
    "float32_boundary": True,
    "snapshot_format": "NANCYC30/raw-native-root",
    "scope": (
        "optional ctypes access to the experimental Round 030 native root; "
        "ordinary float32 records, no raw-NaN claim, no canonical merge"
    ),
}


class NativeCycleError(RuntimeError):
    """Base class for native-cycle bridge failures."""


class NativeCycleUnavailable(NativeCycleError):
    """The bridge cannot be made available (missing compiler/source/library)."""

    def __init__(self, reason: str, *, receipt: dict[str, Any] | None = None):
        super().__init__("unavailable: " + str(reason))
        self.reason = str(reason)
        self.receipt = dict(receipt or {})


class NativeCycleVerificationError(NativeCycleError):
    """A source/library digest, manifest or path failed verification."""

    def __init__(self, reason: str, *, receipt: dict[str, Any] | None = None):
        super().__init__("verification denied: " + str(reason))
        self.reason = str(reason)
        self.receipt = dict(receipt or {})


class NativeCycleBuildError(NativeCycleError):
    """A verified source could not be compiled within the bounded timeout."""

    def __init__(self, reason: str, *, receipt: dict[str, Any] | None = None):
        super().__init__("native build failed: " + str(reason))
        self.reason = str(reason)
        self.receipt = dict(receipt or {})


class NativeCycleLibraryError(NativeCycleError):
    """The shared library loaded but is missing the expected ABI."""


class NativeCycleClosedError(NativeCycleError):
    """The native root has already been freed."""


class NativeCycleSnapshotError(NativeCycleError):
    """A snapshot failed its exact marker/size check or native I/O."""


class NativeCycleNumericalError(NativeCycleError):
    """A native result was non-finite and is not reported as a valid result."""


class NativeCycleBoundsError(ValueError):
    """A Python argument exceeds a documented dimension/range/path bound."""


@dataclass(frozen=True)
class NativeCycleBuild:
    """Receipt for a verified compile (or verified cache hit)."""

    source: Path
    source_sha256: str
    library: Path
    library_sha256: str
    compiler: str
    command: tuple[str, ...]
    cache_dir: Path
    cached: bool
    verification: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BUILD_SCHEMA,
            "source": str(self.source),
            "source_sha256": self.source_sha256,
            "library": str(self.library),
            "library_sha256": self.library_sha256,
            "compiler": self.compiler,
            "command": list(self.command),
            "cache_dir": str(self.cache_dir),
            "cached": bool(self.cached),
            "verification": dict(self.verification),
        }


# ---------------------------------------------------------------------------
# Bounded Python values and paths
# ---------------------------------------------------------------------------


def _float32(value: Any, field: str) -> float:
    """Quantise one Python number to finite binary32 at the native boundary."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise NativeCycleBoundsError(f"{field} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NativeCycleBoundsError(f"{field} is not a finite number") from exc
    if not math.isfinite(number):
        raise NativeCycleBoundsError(f"{field} must be finite")
    try:
        quantised = _F32.unpack(_F32.pack(number))[0]
    except (OverflowError, struct.error) as exc:
        raise NativeCycleBoundsError(
            f"{field} is outside the finite float32 range") from exc
    if not math.isfinite(quantised):
        raise NativeCycleBoundsError(f"{field} is outside the finite float32 range")
    return quantised


def _uint32(value: Any, field: str, *, maximum: int = 0xFFFFFFFF) -> int:
    if isinstance(value, bool):
        raise NativeCycleBoundsError(f"{field} must be an integer")
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise NativeCycleBoundsError(f"{field} must be an integer") from exc
    if number < 0 or number > maximum:
        raise NativeCycleBoundsError(
            f"{field} must be between 0 and {maximum}")
    return int(number)


def _bounded_timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise NativeCycleBoundsError("timeout must be a number of seconds")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise NativeCycleBoundsError("timeout must be positive and finite")
    if timeout > MAX_COMPILE_TIMEOUT:
        raise NativeCycleBoundsError(
            f"timeout must not exceed {MAX_COMPILE_TIMEOUT:g} seconds")
    return timeout


def _coerce_vector(values: Iterable[Any], field: str) -> list[float]:
    if isinstance(values, (str, bytes, bytearray)):
        raise NativeCycleBoundsError(f"{field} must be a sequence of numbers")
    try:
        items = list(values)
    except TypeError as exc:
        raise NativeCycleBoundsError(
            f"{field} must be an iterable of {D} numbers") from exc
    if len(items) != D:
        raise NativeCycleBoundsError(
            f"{field} must contain exactly {D} values, got {len(items)}")
    return [_float32(value, f"{field}[{index}]")
            for index, value in enumerate(items)]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _coerce_input(sample: Any) -> tuple[list[list[float]], bool]:
    """Return ``(rows, was_batch)`` for one sample or a bounded batch."""
    if isinstance(sample, (str, bytes, bytearray)):
        raise NativeCycleBoundsError("sample must be a sequence of numbers")
    try:
        items = list(sample)
    except TypeError as exc:
        raise NativeCycleBoundsError(
            "sample must be an iterable of numbers") from exc
    if not items:
        raise NativeCycleBoundsError("sample must not be empty")
    if _is_scalar(items[0]):
        return [_coerce_vector(items, "sample")], False
    if len(items) > MAX_SAMPLES:
        raise NativeCycleBoundsError(
            f"at most {MAX_SAMPLES} samples may cross one boundary call")
    rows = [_coerce_vector(row, f"samples[{index}]")
            for index, row in enumerate(items)]
    return rows, True


def _coerce_batch(samples: Any) -> list[list[float]]:
    """Return validated rows for an explicit batch call."""
    if isinstance(samples, (str, bytes, bytearray)):
        raise NativeCycleBoundsError("samples must be an iterable of samples")
    try:
        items = list(samples)
    except TypeError as exc:
        raise NativeCycleBoundsError("samples must be an iterable") from exc
    if not items:
        raise NativeCycleBoundsError("samples must not be empty")
    if len(items) > MAX_SAMPLES:
        raise NativeCycleBoundsError(
            f"at most {MAX_SAMPLES} samples may cross one boundary call")
    return [_coerce_vector(row, f"samples[{index}]")
            for index, row in enumerate(items)]


def _coerce_labels(labels: Any, count: int) -> list[int]:
    if isinstance(labels, (str, bytes, bytearray)):
        raise NativeCycleBoundsError("labels must be an iterable of integers")
    try:
        items = list(labels)
    except TypeError as exc:
        raise NativeCycleBoundsError("labels must be an iterable") from exc
    if len(items) != count:
        raise NativeCycleBoundsError(
            f"labels must contain {count} values, got {len(items)}")
    checked: list[int] = []
    for index, value in enumerate(items):
        label = _uint32(value, f"labels[{index}]", maximum=K - 1)
        checked.append(label)
    return checked


def _bounded_path(path: Any, field: str) -> Path:
    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)):
        raise NativeCycleBoundsError(f"{field} must be a filesystem path")
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise NativeCycleBoundsError(f"{field} must be a filesystem path") from exc
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not raw or "\x00" in raw:
        raise NativeCycleBoundsError(f"{field} must be a non-empty path without NUL")
    try:
        encoded = os.fsencode(raw)
    except (OSError, UnicodeError) as exc:
        raise NativeCycleBoundsError(f"{field} cannot be encoded") from exc
    if len(encoded) > MAX_PATH_BYTES:
        raise NativeCycleBoundsError(
            f"{field} must not exceed {MAX_PATH_BYTES} bytes")
    candidate = Path(raw).expanduser()
    if not candidate.name or candidate.name in (".", ".."):
        raise NativeCycleBoundsError(f"{field} must name a file")
    return candidate


def _snapshot_write_path(path: Any) -> Path:
    candidate = _bounded_path(path, "snapshot path")
    parent = candidate.parent if str(candidate.parent) else Path(".")
    try:
        parent = parent.resolve(strict=True)
    except OSError as exc:
        raise NativeCycleBoundsError(
            f"snapshot directory does not exist: {parent}") from exc
    if not parent.is_dir():
        raise NativeCycleBoundsError(f"snapshot parent is not a directory: {parent}")
    target = parent / candidate.name
    if target.exists() and target.is_dir():
        raise NativeCycleBoundsError(f"snapshot path is a directory: {target}")
    return target


def _snapshot_read_path(path: Any) -> Path:
    candidate = _bounded_path(path, "restore path")
    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise NativeCycleSnapshotError(f"snapshot not found: {candidate}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise NativeCycleSnapshotError(f"snapshot is not a regular file: {resolved}")
    if info.st_size != SNAPSHOT_BYTES:
        raise NativeCycleSnapshotError(
            f"snapshot must be exactly {SNAPSHOT_BYTES} bytes "
            f"(8 marker + {ROOT_BYTES} root), got {info.st_size}")
    try:
        with resolved.open("rb") as handle:
            marker = handle.read(SNAPSHOT_MARKER_BYTES)
    except OSError as exc:
        raise NativeCycleSnapshotError(f"cannot read snapshot: {exc}") from exc
    if marker != SNAPSHOT_MAGIC:
        raise NativeCycleSnapshotError("snapshot marker is not NANCYC30")
    return resolved


# ---------------------------------------------------------------------------
# Source verification and bounded compilation
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_name(name: str) -> str:
    if not name or "\x00" in name:
        raise NativeCycleVerificationError("empty or NUL-containing SHA256SUMS path")
    if name.startswith(("/", "\\")) or "\\" in name:
        raise NativeCycleVerificationError(
            f"path traversal rejected in SHA256SUMS: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise NativeCycleVerificationError(
            f"path traversal rejected in SHA256SUMS: {name!r}")
    return pure.as_posix()


def _read_manifest(manifest: Path) -> list[tuple[str, str]]:
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        raise NativeCycleVerificationError(
            f"cannot read SHA256SUMS {manifest}: {exc}") from exc
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SUM_LINE.match(line)
        if match is None:
            raise NativeCycleVerificationError(
                f"invalid SHA256SUMS line {line_number}: expected '<sha256>  <path>'")
        name = match.group("name")
        if name.startswith("*"):
            name = name[1:]
        name = _manifest_name(name.strip())
        if name in seen:
            raise NativeCycleVerificationError(
                f"duplicate SHA256SUMS entry for {name!r}")
        seen.add(name)
        entries.append((name, match.group("digest").lower()))
    if not entries:
        raise NativeCycleVerificationError("SHA256SUMS contains no file entries")
    return entries


def default_source_path() -> Path:
    """Return the Round 030 source path (environment override honoured)."""
    override = os.environ.get("NANLMIND_HERMES_CYCLE_SOURCE")
    if override:
        return _bounded_path(override, "NANLMIND_HERMES_CYCLE_SOURCE").resolve()
    return _DEFAULT_SOURCE


def default_cache_dir() -> Path:
    """Return the private compile cache directory (temporary by default)."""
    override = os.environ.get("NANLMIND_HERMES_NATIVE_CACHE")
    if override:
        return _bounded_path(override, "NANLMIND_HERMES_NATIVE_CACHE")
    return Path(tempfile.gettempdir()) / "nanlmind-hermes-native-cycle"


def verify_native_cycle_source(
    source: Path | str | None = None,
    manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Verify the supplied source is listed in SHA256SUMS and matches it.

    Every manifest path is parsed for traversal/duplicate safety, but only the
    source digest is executed.  No compiler is run before this succeeds.
    """
    source_path = default_source_path() if source is None else _bounded_path(
        source, "source").resolve()
    if not source_path.is_file():
        raise NativeCycleUnavailable(f"native_cycle.cpp source not found: {source_path}")
    manifest_path = (
        _bounded_path(manifest, "manifest")
        if manifest is not None
        else source_path.parent / "SHA256SUMS"
    )
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise NativeCycleVerificationError(
            f"SHA256SUMS not found: {manifest_path}") from exc
    if not manifest_path.is_file():
        raise NativeCycleVerificationError(
            f"SHA256SUMS is not a regular file: {manifest_path}")
    base = manifest_path.parent.resolve()
    try:
        source_path.relative_to(base)
    except ValueError as exc:
        raise NativeCycleVerificationError(
            f"source escapes manifest directory: {source_path} not under {base}") from exc

    entries = _read_manifest(manifest_path)
    listed_digest: str | None = None
    for name, digest in entries:
        candidate = base / name
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise NativeCycleVerificationError(
                f"cannot resolve manifest entry {name!r}: {exc}") from exc
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise NativeCycleVerificationError(
                f"manifest entry escapes its directory: {name!r}") from exc
        if resolved == source_path:
            listed_digest = digest
    if listed_digest is None:
        raise NativeCycleVerificationError(
            f"source is not covered by SHA256SUMS: {source_path.name}")
    measured = _sha256_file(source_path)
    if measured != listed_digest:
        raise NativeCycleVerificationError(
            f"SHA-256 mismatch for {source_path.name}: expected {listed_digest}, "
            f"measured {measured}")
    return {
        "schema": SCHEMA,
        "status": "verified",
        "source": str(source_path),
        "source_sha256": measured,
        "manifest": str(manifest_path),
        "entry_count": len(entries),
    }


def find_compiler(candidates: Sequence[str] = COMPILER_CANDIDATES) -> str | None:
    """Return the first executable C++ compiler from *candidates*, or ``None``."""
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        if os.sep in text or (os.altsep and os.altsep in text):
            path = Path(text).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                # Preserve the invoked name (e.g. the ``c++`` symlink): the
                # Clang driver chooses its C++ runtime from argv[0].
                return os.path.abspath(os.fspath(path))
            continue
        found = shutil.which(text)
        if found and os.path.isfile(found) and os.access(found, os.X_OK):
            return found
    return None


def _compiler_signature(compiler: str, timeout: float) -> str:
    identity = {
        "path": os.path.realpath(compiler),
        "invoked": Path(compiler).name,
        "platform": sys.platform,
        "machine": os.uname().machine if hasattr(os, "uname") else "unknown",
    }
    try:
        info = os.stat(compiler)
        identity["size"] = info.st_size
        identity["mtime_ns"] = info.st_mtime_ns
    except OSError:
        pass
    try:
        proc = subprocess.run(
            [compiler, "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=min(max(0.1, timeout), 5.0),
            check=False,
        )
        first = (proc.stdout or proc.stderr or "").strip().splitlines()
        identity["version"] = first[0] if first else ""
    except (OSError, subprocess.SubprocessError):
        identity["version"] = ""
    return _sha256_bytes(
        json.dumps(identity, sort_keys=True).encode("utf-8"))


def _library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def _compile_command(compiler: str, source: Path, output: Path) -> list[str]:
    common = [compiler, "-std=c++20", "-O3", "-fPIC"]
    if sys.platform == "darwin":
        return common + ["-dynamiclib", str(source), "-o", str(output)]
    return common + ["-shared", str(source), "-o", str(output)]


def _cache_paths(
    cache_dir: Path,
    source_sha256: str,
    compiler: str,
    timeout: float,
) -> tuple[Path, Path]:
    key = _sha256_bytes(
        json.dumps(
            {
                "schema": BUILD_SCHEMA,
                "source": source_sha256,
                "compiler": _compiler_signature(compiler, timeout),
                "platform": sys.platform,
                "suffix": _library_suffix(),
                "flags": ["-std=c++20", "-O3", "-fPIC"],
            },
            sort_keys=True,
        ).encode("utf-8")
    )[:24]
    name = "libnative_cycle-" + key + _library_suffix()
    return cache_dir / name, cache_dir / (name + ".sha256")


def _write_manifest_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(
        prefix=".cycle-manifest-", dir=str(path.parent))
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _cache_lookup(
    library: Path,
    manifest: Path,
    source_sha256: str,
) -> dict[str, Any] | None:
    if not library.is_file() or not manifest.is_file():
        return None
    try:
        info = library.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            return None
        if info.st_size > MAX_LIBRARY_BYTES:
            return None
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema") != BUILD_SCHEMA:
            return None
        if payload.get("library") != library.name:
            return None
        if payload.get("source_sha256") != source_sha256:
            return None
        expected = payload.get("library_sha256")
        if not isinstance(expected, str) or not _SHA256_RE.match(expected):
            return None
        measured = _sha256_file(library)
        if measured.lower() != expected.lower():
            return None
        return {
            "schema": BUILD_SCHEMA,
            "library": str(library),
            "library_sha256": measured,
            "source_sha256": source_sha256,
            "cached": True,
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def compile_native_cycle(
    source: Path | str | None = None,
    *,
    manifest: Path | str | None = None,
    compiler: str | None = None,
    compiler_candidates: Sequence[str] = COMPILER_CANDIDATES,
    cache_dir: Path | str | None = None,
    timeout: float = DEFAULT_COMPILE_TIMEOUT,
    force: bool = False,
) -> NativeCycleBuild:
    """Verify, compile and cache the Round 030 shared library.

    The source digest is checked before compiler detection.  The verified bytes
    are copied into the cache and compiled there, so the compiler never reads a
    file that changed after verification.  Compilation has a bounded timeout.
    """
    timeout = _bounded_timeout(timeout)
    verification = verify_native_cycle_source(source, manifest)
    source_path = Path(verification["source"])
    source_sha256 = str(verification["source_sha256"])

    # Re-read once and compile exactly these bytes (verification -> build copy).
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise NativeCycleVerificationError(
            f"cannot read verified source: {exc}") from exc
    if _sha256_bytes(payload) != source_sha256:
        raise NativeCycleVerificationError(
            "source changed between verification and compilation")

    chosen = compiler if compiler is not None else find_compiler(compiler_candidates)
    if not chosen:
        tried = ", ".join(str(item) for item in compiler_candidates) or "no candidates"
        reason = f"no C++ compiler found (tried {tried})"
        receipt = {
            "schema": SCHEMA,
            "status": "unavailable",
            "available": False,
            "reason": reason,
            "source": str(source_path),
            "source_sha256": source_sha256,
        }
        raise NativeCycleUnavailable(reason, receipt=receipt)
    chosen_path = find_compiler([chosen])
    if chosen_path is None:
        reason = f"C++ compiler is not executable: {chosen}"
        raise NativeCycleUnavailable(
            reason,
            receipt={
                "schema": SCHEMA,
                "status": "unavailable",
                "available": False,
                "reason": reason,
                "source": str(source_path),
                "source_sha256": source_sha256,
            },
        )

    cache = (
        _bounded_path(cache_dir, "cache_dir")
        if cache_dir is not None
        else default_cache_dir()
    )
    try:
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise NativeCycleUnavailable(
            f"cannot create native cache directory {cache}: {exc}") from exc
    if not cache.is_dir():
        raise NativeCycleUnavailable(f"native cache is not a directory: {cache}")

    library, cache_manifest = _cache_paths(
        cache, source_sha256, chosen_path, timeout)
    with _BUILD_LOCK:
        if not force:
            hit = _cache_lookup(library, cache_manifest, source_sha256)
            if hit is not None:
                return NativeCycleBuild(
                    source=source_path,
                    source_sha256=source_sha256,
                    library=library,
                    library_sha256=str(hit["library_sha256"]),
                    compiler=chosen_path,
                    command=(),
                    cache_dir=cache,
                    cached=True,
                    verification=verification,
                )

        staging_fd, staging_name = tempfile.mkstemp(
            prefix=".native-cycle-source-", suffix=".cpp", dir=str(cache))
        os.close(staging_fd)
        staging_source = Path(staging_name)
        output_fd, output_name = tempfile.mkstemp(
            prefix=".native-cycle-build-", suffix=_library_suffix(), dir=str(cache))
        os.close(output_fd)
        try:
            os.unlink(output_name)
        except OSError:
            pass
        command = _compile_command(chosen_path, staging_source, Path(output_name))
        try:
            try:
                staging_source.write_bytes(payload)
                os.chmod(staging_source, 0o600)
                proc = subprocess.run(
                    command,
                    cwd=str(cache),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise NativeCycleBuildError(
                    f"compile timed out after {timeout:g}s") from exc
            except OSError as exc:
                raise NativeCycleBuildError(
                    f"cannot execute compiler {chosen_path}: {exc}") from exc
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                if len(detail) > 4000:
                    detail = detail[:4000] + "\n...[truncated]"
                raise NativeCycleBuildError(
                    f"compiler exited {proc.returncode}: {detail}")
            output = Path(output_name)
            if not output.is_file():
                raise NativeCycleBuildError(
                    "compiler reported success but produced no library")
            info = output.stat()
            if info.st_size <= 0 or info.st_size > MAX_LIBRARY_BYTES:
                raise NativeCycleBuildError(
                    f"compiled library has an invalid size: {info.st_size}")
            library_sha256 = _sha256_file(output)
            try:
                os.chmod(output, 0o700)
                os.replace(output, library)
                _write_manifest_atomic(
                    cache_manifest,
                    {
                        "schema": BUILD_SCHEMA,
                        "library": library.name,
                        "library_sha256": library_sha256,
                        "source_sha256": source_sha256,
                        "source": str(source_path),
                        "compiler": chosen_path,
                        "command": command,
                        "created_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            except OSError as exc:
                raise NativeCycleBuildError(
                    f"cannot publish compiled library in {cache}: {exc}") from exc
        finally:
            for temporary in (Path(output_name), staging_source):
                try:
                    temporary.unlink()
                except OSError:
                    pass

    return NativeCycleBuild(
        source=source_path,
        source_sha256=source_sha256,
        library=library,
        library_sha256=library_sha256,
        compiler=chosen_path,
        command=tuple(command),
        cache_dir=cache,
        cached=False,
        verification=verification,
    )


# ---------------------------------------------------------------------------
# Library verification and availability probe
# ---------------------------------------------------------------------------


def _is_library_name(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".so")
        or ".so." in name
        or name.endswith(".dylib")
        or name.endswith(".dll")
    )


def _classify_target(target: Path | str | None) -> tuple[Path, Path | None]:
    if target is None:
        return default_source_path(), None
    candidate = _bounded_path(target, "lib_path_or_source")
    if _is_library_name(candidate):
        return candidate, candidate
    return candidate, None


def _library_manifest_candidates(
    library: Path,
    manifest: Path | str | None,
) -> list[Path]:
    if manifest is not None:
        candidate = _bounded_path(manifest, "manifest")
        return [candidate]
    candidates = [
        library.with_name(library.name + ".sha256"),
        library.parent / "SHA256SUMS",
    ]
    return [path for path in candidates if path.is_file()]


def _read_any_manifest(manifest: Path) -> list[tuple[str, str]]:
    """Read either a SHA256SUMS text file or a cache JSON manifest."""
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        raise NativeCycleVerificationError(
            f"cannot read library manifest {manifest}: {exc}") from exc
    try:
        payload = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and "library" in payload:
        name = payload.get("library")
        digest = payload.get("library_sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise NativeCycleVerificationError(
                f"invalid cache manifest fields in {manifest}")
        if not _SHA256_RE.match(digest):
            raise NativeCycleVerificationError(
                f"invalid cache manifest digest in {manifest}")
        return [(_manifest_name(name), digest.lower())]
    return _read_manifest(manifest)


def _verify_library_digest(
    library: Path,
    manifest: Path | str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not library.is_file():
        raise NativeCycleUnavailable(f"native library not found: {library}")
    info = library.stat()
    if not stat.S_ISREG(info.st_mode):
        raise NativeCycleVerificationError(
            f"native library is not a regular file: {library}")
    if info.st_size <= 0 or info.st_size > MAX_LIBRARY_BYTES:
        raise NativeCycleVerificationError(
            f"native library has an invalid size: {info.st_size}")
    measured = _sha256_file(library)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not _SHA256_RE.match(
                expected_sha256):
            raise NativeCycleVerificationError(
                "expected_sha256 must be 64 hexadecimal characters")
        if measured.lower() != expected_sha256.lower():
            raise NativeCycleVerificationError(
                f"SHA-256 mismatch for {library.name}: expected "
                f"{expected_sha256.lower()}, measured {measured}")

    manifests = _library_manifest_candidates(library, manifest)
    if manifests:
        try:
            manifest_path = manifests[0].resolve(strict=True)
        except OSError as exc:
            raise NativeCycleVerificationError(
                f"library manifest not found: {manifests[0]}") from exc
        base = manifest_path.parent
        try:
            library.resolve().relative_to(base)
        except ValueError as exc:
            raise NativeCycleVerificationError(
                f"library escapes its manifest directory: {library}") from exc
        listed: str | None = None
        for name, digest in _read_any_manifest(manifest_path):
            candidate = (base / name).resolve(strict=False)
            if candidate == library.resolve():
                listed = digest
        if listed is None:
            raise NativeCycleVerificationError(
                f"library is not covered by manifest: {library.name}")
        if measured != listed:
            raise NativeCycleVerificationError(
                f"SHA-256 mismatch for {library.name}: expected {listed}, "
                f"measured {measured}")
        status = "verified"
        manifest_text = str(manifest_path)
    else:
        status = "explicit-library"
        manifest_text = None
    return {
        "schema": SCHEMA,
        "status": status,
        "library": str(library.resolve()),
        "library_sha256": measured,
        "manifest": manifest_text,
    }


def probe_native_cycle(
    lib_path_or_source: Path | str | None = None,
    *,
    manifest: Path | str | None = None,
    compiler: str | None = None,
    compiler_candidates: Sequence[str] = COMPILER_CANDIDATES,
    cache_dir: Path | str | None = None,
    timeout: float = DEFAULT_COMPILE_TIMEOUT,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Report whether the bridge can be opened, without compiling anything.

    The receipt always has ``status`` in ``available``/``unavailable``/
    ``denied`` and an ``available`` boolean.  An absent compiler or library is
    reported, never replaced by a fabricated native result.
    """
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "unavailable",
        "available": False,
        "reason": None,
        "root_bytes": ROOT_BYTES,
        "sample_dimensions": D,
        "class_count": K,
        "snapshot_bytes": SNAPSHOT_BYTES,
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "claims": dict(CLAIMS),
    }
    try:
        timeout = _bounded_timeout(timeout)
        source, library = _classify_target(lib_path_or_source)
        if library is not None:
            verification = _verify_library_digest(
                library, manifest, expected_sha256)
            receipt.update(
                status="available",
                available=True,
                library=verification["library"],
                library_sha256=verification["library_sha256"],
                verification=verification,
            )
            return receipt
        verification = verify_native_cycle_source(source, manifest)
        chosen = compiler if compiler is not None else find_compiler(
            compiler_candidates)
        chosen = find_compiler([chosen]) if chosen else None
        if chosen is None:
            tried = ", ".join(str(item) for item in compiler_candidates) or (
                "no candidates")
            receipt.update(
                status="unavailable",
                reason=f"no C++ compiler found (tried {tried})",
                source=verification["source"],
                source_sha256=verification["source_sha256"],
                verification=verification,
            )
            return receipt
        cache = (
            _bounded_path(cache_dir, "cache_dir")
            if cache_dir is not None
            else default_cache_dir()
        )
        receipt.update(
            status="available",
            available=True,
            source=verification["source"],
            source_sha256=verification["source_sha256"],
            compiler=chosen,
            cache_dir=str(cache),
            verification=verification,
        )
        return receipt
    except NativeCycleUnavailable as exc:
        receipt.update(status="unavailable", available=False, reason=exc.reason)
        return receipt
    except NativeCycleVerificationError as exc:
        receipt.update(status="denied", available=False, reason=exc.reason)
        return receipt
    except NativeCycleBoundsError as exc:
        receipt.update(status="denied", available=False, reason=str(exc))
        return receipt
    except NativeCycleError as exc:
        receipt.update(status="unavailable", available=False, reason=str(exc))
        return receipt


# ---------------------------------------------------------------------------
# Root handle
# ---------------------------------------------------------------------------


class NativeCycleRoot:
    """Safe handle around one 8 MiB native Round 030 root.

    Parameters
    ----------
    lib_path_or_source:
        ``None`` for ``experiments/round030/native_cycle.cpp``, a ``.cpp``
        source to verify/compile, or an existing ``.so``/``.dylib``/``.dll``.
        A library is checked against its sibling ``<name>.sha256`` manifest,
        an explicit manifest, or an explicit ``expected_sha256``.  An explicit
        library without any manifest is loaded as caller-trusted native code;
        source builds are always digest-verified.
    """

    _FUNCTION_SPECS = {
        "nan_new": ([ctypes.c_int], ctypes.c_void_p),
        "nan_free": ([ctypes.c_void_p], None),
        "nan_get": (
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)], None),
        "nan_predict": (
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
             ctypes.POINTER(ctypes.c_float)], None),
        "nan_raw_predict": (
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
             ctypes.POINTER(ctypes.c_float)], None),
        "nan_epoch": (
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
             ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_float,
             ctypes.c_uint], ctypes.c_double),
        "nan_gauge": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_float], None),
        "nan_snapshot": (
            [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
        "nan_restore": (
            [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
        "nan_audit": ([ctypes.c_void_p], ctypes.c_int),
    }

    def __init__(
        self,
        lib_path_or_source: Path | str | None = None,
        *,
        manifest: Path | str | None = None,
        cache_dir: Path | str | None = None,
        compiler: str | None = None,
        compiler_candidates: Sequence[str] = COMPILER_CANDIDATES,
        timeout: float = DEFAULT_COMPILE_TIMEOUT,
        expected_sha256: str | None = None,
    ):
        self._lock = threading.RLock()
        self._library = None
        self._handle: int | None = None
        self._source_path: Path | None = None
        self._library_path: Path | None = None
        self._build_receipt: dict[str, Any] | None = None
        self._verification: dict[str, Any] | None = None

        timeout = _bounded_timeout(timeout)
        source, library = _classify_target(lib_path_or_source)
        try:
            if library is not None:
                verification = _verify_library_digest(
                    library, manifest, expected_sha256)
                self._library_path = Path(verification["library"])
                self._verification = verification
            else:
                build = compile_native_cycle(
                    source,
                    manifest=manifest,
                    compiler=compiler,
                    compiler_candidates=compiler_candidates,
                    cache_dir=cache_dir,
                    timeout=timeout,
                )
                self._source_path = build.source
                self._library_path = build.library
                self._build_receipt = build.as_dict()
                self._verification = dict(build.verification)
            self._load_library(self._library_path)
        except BaseException:
            self._free_handle()
            raise

    # -- lifecycle ---------------------------------------------------------

    def _load_library(self, path: Path) -> None:
        try:
            library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise NativeCycleLibraryError(
                f"cannot load native library {path}: {exc}") from exc
        try:
            for name, (argtypes, restype) in self._FUNCTION_SPECS.items():
                function = getattr(library, name)
                function.argtypes = argtypes
                function.restype = restype
        except AttributeError as exc:
            raise NativeCycleLibraryError(
                f"native library is missing required symbol: {exc}") from exc
        self._library = library
        handle = library.nan_new(0)
        if not handle:
            self._library = None
            raise NativeCycleLibraryError(
                "native nan_new(0) returned a null root")
        self._handle = int(handle)
        if not bool(library.nan_audit(ctypes.c_void_p(self._handle))):
            self._free_handle()
            raise NativeCycleLibraryError(
                "new native root failed the mirror-antisymmetry audit")

    def _free_handle(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None and self._library is not None:
            try:
                self._library.nan_free(ctypes.c_void_p(handle))
            except Exception:
                pass

    def close(self) -> None:
        """Free the native root.  Idempotent; no pointer is returned."""
        with self._lock:
            self._free_handle()
            self._library = None

    @property
    def closed(self) -> bool:
        return self._handle is None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def __enter__(self) -> "NativeCycleRoot":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"<NativeCycleRoot {state} root_bytes={ROOT_BYTES}>"

    # -- receipts ----------------------------------------------------------

    def claims(self) -> dict[str, Any]:
        """Return the explicit scope/claim boundary for this bridge."""
        return dict(CLAIMS)

    def receipt(self) -> dict[str, Any]:
        """Return a JSON-friendly status receipt (never a raw pointer)."""
        with self._lock:
            return {
                "schema": SCHEMA,
                "status": "available" if self.available else "closed",
                "available": self.available,
                "root_bytes": ROOT_BYTES,
                "sample_dimensions": D,
                "class_count": K,
                "pair_count": PAIRS,
                "snapshot_bytes": SNAPSHOT_BYTES,
                "snapshot_magic": SNAPSHOT_MAGIC.decode("ascii"),
                "snapshot_schema": SNAPSHOT_SCHEMA,
                "float32_boundary": True,
                "source": str(self._source_path) if self._source_path else None,
                "library": str(self._library_path) if self._library_path else None,
                "verification": dict(self._verification or {}),
                "build": dict(self._build_receipt or {}),
                "claims": dict(CLAIMS),
            }

    def _require_open(self) -> int:
        handle = self._handle
        library = self._library
        if handle is None or library is None:
            raise NativeCycleClosedError("native root has been freed")
        return handle

    # -- reads -------------------------------------------------------------

    def _predict_rows(self, rows: list[list[float]], raw: bool) -> list[list[float]]:
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            count = len(rows)
            flat = (ctypes.c_float * (count * D))(
                *(value for row in rows for value in row))
            output = (ctypes.c_float * (count * K))()
            method = library.nan_raw_predict if raw else library.nan_predict
            method(ctypes.c_void_p(handle), flat, output)
            results: list[list[float]] = []
            for index in range(count):
                vector = [
                    float(output[index * K + class_index])
                    for class_index in range(K)
                ]
                if any(not math.isfinite(value) for value in vector):
                    raise NativeCycleNumericalError(
                        "native prediction produced a non-finite logit")
                results.append(vector)
            return results

    def predict(self, sample: Any) -> list[float] | list[list[float]]:
        """Predict cycle-gauge logits for one sample (or an auto batch).

        A single 24-value sample returns 16 Python floats.  A nested sequence
        of up to :data:`MAX_SAMPLES` samples returns one 16-float row each.
        Every input value is quantised to binary32 and validated finite.
        """
        rows, was_batch = _coerce_input(sample)
        results = self._predict_rows(rows, raw=False)
        return results if was_batch else results[0]

    def predict_batch(self, samples: Any) -> list[list[float]]:
        """Predict an explicit batch of 24-value samples."""
        return self._predict_rows(_coerce_batch(samples), raw=False)

    def raw_predict(self, sample: Any) -> list[float] | list[list[float]]:
        """Return the raw directed-pair readout (no cycle cancellation)."""
        rows, was_batch = _coerce_input(sample)
        results = self._predict_rows(rows, raw=True)
        return results if was_batch else results[0]

    def weights(self) -> tuple[tuple[float, ...], ...]:
        """Return the effective 16 x 24 gauge-invariant weights.

        The returned tuple is a fresh Python copy of the native fluxes.  Values
        are exact binary32 values widened to Python ``float``; no raw NaN
        payload interpretation is made.
        """
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            output = (ctypes.c_float * (K * D))()
            library.nan_get(ctypes.c_void_p(handle), output)
            rows: list[tuple[float, ...]] = []
            for class_index in range(K):
                row = tuple(
                    float(output[class_index * D + dimension])
                    for dimension in range(D)
                )
                if any(not math.isfinite(value) for value in row):
                    raise NativeCycleNumericalError(
                        "native weights contain a non-finite value")
                rows.append(row)
            return tuple(rows)

    def audit(self) -> bool:
        """Return the native mirror/antisymmetry audit result."""
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            return bool(library.nan_audit(ctypes.c_void_p(handle)))

    # -- training ----------------------------------------------------------

    def train_epoch(
        self,
        samples: Any,
        labels: Any,
        lr: float = 0.02,
        seed: int = 0,
    ) -> float:
        """Run exactly one native softmax-SGD epoch and return mean loss.

        ``samples`` is 1..:data:`MAX_SAMPLES` rows of 24 finite values;
        ``labels`` must contain one class index in 0..15 per row.  ``lr`` is
        finite, positive and at most :data:`MAX_LR`; ``seed`` is uint32.
        """
        rows = _coerce_batch(samples)
        checked_labels = _coerce_labels(labels, len(rows))
        learning_rate = _float32(lr, "lr")
        if learning_rate <= 0 or learning_rate > MAX_LR:
            raise NativeCycleBoundsError(
                f"lr must be > 0 and <= {MAX_LR:g}")
        checked_seed = _uint32(seed, "seed")
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            flat = (ctypes.c_float * (len(rows) * D))(
                *(value for row in rows for value in row))
            targets = (ctypes.c_int * len(rows))(*checked_labels)
            loss = float(library.nan_epoch(
                ctypes.c_void_p(handle),
                flat,
                targets,
                len(rows),
                learning_rate,
                checked_seed,
            ))
            if not math.isfinite(loss):
                raise NativeCycleNumericalError(
                    "native training epoch produced a non-finite loss")
            return loss

    def train(
        self,
        samples: Any,
        labels: Any,
        *,
        epochs: int = 1,
        lr: float = 0.02,
        seed: int = 0,
    ) -> list[float]:
        """Run 1..:data:`MAX_EPOCHS` epochs, returning one loss per epoch."""
        if isinstance(epochs, bool):
            raise NativeCycleBoundsError("epochs must be an integer")
        try:
            count = operator.index(epochs)
        except TypeError as exc:
            raise NativeCycleBoundsError("epochs must be an integer") from exc
        if count < 1 or count > MAX_EPOCHS:
            raise NativeCycleBoundsError(
                f"epochs must be between 1 and {MAX_EPOCHS}")
        checked_seed = _uint32(seed, "seed")
        losses: list[float] = []
        for offset in range(count):
            losses.append(
                self.train_epoch(samples, labels, lr=lr,
                                 seed=(checked_seed + offset) & 0xFFFFFFFF))
        return losses

    def gauge(self, seed: int, sigma: float) -> None:
        """Apply one synthetic additive gauge shift (vertex-gradient form)."""
        checked_seed = _uint32(seed, "seed", maximum=0x7FFFFFFF)
        checked_sigma = _float32(sigma, "sigma")
        if checked_sigma < 0 or checked_sigma > MAX_GAUGE_SIGMA:
            raise NativeCycleBoundsError(
                f"sigma must be between 0 and {MAX_GAUGE_SIGMA:g}")
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            library.nan_gauge(
                ctypes.c_void_p(handle), checked_seed, checked_sigma)

    # -- snapshots ---------------------------------------------------------

    def snapshot(self, path: Path | str) -> bool:
        """Write an exact ``NANCYC30`` + 8 MiB snapshot atomically.

        The final path is only replaced after the marker and exact byte count
        have been validated.  This is not a canonical/transactional format.
        """
        target = _snapshot_write_path(path)
        for protected in (self._source_path, self._library_path):
            if protected is not None:
                try:
                    if target.resolve(strict=False) == protected.resolve(
                            strict=False):
                        raise NativeCycleBoundsError(
                            "snapshot path must not overwrite source/library")
                except OSError:
                    pass
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            handle_fd, temp_name = tempfile.mkstemp(
                prefix=".native-cycle-snapshot-", suffix=".bin",
                dir=str(target.parent))
            os.close(handle_fd)
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            temp_path = Path(temp_name)
            try:
                if not bool(library.nan_snapshot(
                        ctypes.c_void_p(handle), os.fsencode(temp_name))):
                    raise NativeCycleSnapshotError(
                        "native nan_snapshot returned failure")
                info = temp_path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise NativeCycleSnapshotError(
                        "native snapshot is not a regular file")
                if info.st_size != SNAPSHOT_BYTES:
                    raise NativeCycleSnapshotError(
                        f"native snapshot has {info.st_size} bytes; expected "
                        f"{SNAPSHOT_BYTES} (8 marker + {ROOT_BYTES} root)")
                with temp_path.open("rb") as handle_in:
                    marker = handle_in.read(SNAPSHOT_MARKER_BYTES)
                if marker != SNAPSHOT_MAGIC:
                    raise NativeCycleSnapshotError(
                        "native snapshot marker is not NANCYC30")
                try:
                    os.chmod(temp_path, 0o600)
                    os.replace(temp_path, target)
                except OSError as exc:
                    raise NativeCycleSnapshotError(
                        f"cannot publish snapshot {target}: {exc}") from exc
                return True
            finally:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def restore(self, path: Path | str) -> bool:
        """Restore a validated exact snapshot into this root."""
        source = _snapshot_read_path(path)
        with self._lock:
            handle = self._require_open()
            library = self._library
            assert library is not None
            if not bool(library.nan_restore(
                    ctypes.c_void_p(handle), os.fsencode(str(source)))):
                raise NativeCycleSnapshotError(
                    "native nan_restore rejected the snapshot")
            if not bool(library.nan_audit(ctypes.c_void_p(handle))):
                raise NativeCycleSnapshotError(
                    "restored root failed the mirror-antisymmetry audit")
            return True


__all__ = (
    "SCHEMA",
    "BUILD_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "N",
    "D",
    "K",
    "PAIRS",
    "ROOT_BYTES",
    "SNAPSHOT_MAGIC",
    "SNAPSHOT_MARKER_BYTES",
    "SNAPSHOT_BYTES",
    "DEFAULT_SOURCE_REL",
    "COMPILER_CANDIDATES",
    "DEFAULT_COMPILE_TIMEOUT",
    "MAX_COMPILE_TIMEOUT",
    "MAX_SAMPLES",
    "MAX_EPOCHS",
    "MAX_LR",
    "MAX_GAUGE_SIGMA",
    "MAX_PATH_BYTES",
    "CLAIMS",
    "NativeCycleError",
    "NativeCycleUnavailable",
    "NativeCycleVerificationError",
    "NativeCycleBuildError",
    "NativeCycleLibraryError",
    "NativeCycleClosedError",
    "NativeCycleSnapshotError",
    "NativeCycleNumericalError",
    "NativeCycleBoundsError",
    "NativeCycleBuild",
    "NativeCycleRoot",
    "default_source_path",
    "default_cache_dir",
    "verify_native_cycle_source",
    "find_compiler",
    "compile_native_cycle",
    "probe_native_cycle",
)
