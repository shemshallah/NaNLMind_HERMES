"""Crash-recoverable checkpoint persistence for the resident universe.

The resident universe is a bounded sparse memory-mapped file (at most 1 GiB).
This module does not own or reinterpret that file: it publishes and verifies
small, checksummed *manifests* that describe which bytes are authoritative.
Payload files stay where the caller created them, so a sparse mmap is never
copied, materialised or expanded here.

Layout
------

``checkpoint(path, generation, manifest)`` treats ``path`` as a checkpoint
directory and maintains exactly two alternating slots::

    <path>/
        manifest.0.json     # slot 0, checksummed manifest record
        manifest.1.json     # slot 1, checksummed manifest record
        .write.lock         # writer serialisation, mode 0600
        .manifest.<uuid>.tmp # transient; removed after commit or failure

The next generation replaces the slot that is *not* the newest valid one, so a
torn, truncated or digest-mismatched newest slot can always fall back to the
previous generation.  Writing is ``write temp -> fsync -> os.replace ->
fsync(dir)`` with mode 0600.  A damaged pair of slots fails closed: recovery
raises instead of silently returning genesis, and ``checkpoint`` refuses to
overwrite an all-invalid slot pair with a fresh low generation.

Manifest schema
---------------

A manifest is a JSON-serialisable mapping.  The optional ``files`` key (alias:
``entries``) lists payload descriptors::

    {
        "note": "free-form metadata is preserved",
        "files": [
            {
                "path": "universe.nlm",       # relative to <path>, or absolute
                "size": 268435456,            # optional; verified/filled
                "digest": "<64 hex chars>",   # optional; verified/filled
                "regions": [                  # optional sparse used regions
                    {"offset": 0, "length": 4096, "digest": "<optional>"},
                    {"offset": 1048576, "length": 8192}
                ]
            }
        ]
    }

Digest rules, matching :func:`manifest_digest` and :func:`verify_file`:

* no ``regions``: the digest is SHA-256 over the whole file;
* with ``regions``: the top-level digest is SHA-256 over the concatenation of
  the declared regions in ascending offset order, and each region may carry
  its own digest.  Regions must not overlap.

This lets a caller hash only the used regions of a mostly-sparse file while
still detecting truncation (the declared ``size`` is checked) and corruption.

Only regular, non-symlink files no larger than :data:`MAX_FILE_BYTES` (1 GiB)
are accepted.  ``checkpoint`` refuses a manifest whose declared digest or size
does not match the file on disk, and a generation that is not strictly greater
than the newest valid generation.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import mmap as _mmap
import os
from pathlib import Path
import stat
import time
from typing import Any, Iterator
import uuid

from . import MAX_BYTES

try:  # pragma: no cover - POSIX-only fallback for exotic hosts
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None  # type: ignore[assignment]


SCHEMA = "nanlm.resident-checkpoint/1"
MANIFEST_SCHEMA = "nanlm.resident-checkpoint-manifest/1"

SLOT_NAMES = ("manifest.0.json", "manifest.1.json")
TEMP_PREFIX = ".manifest."
TEMP_SUFFIX = ".tmp"
LOCK_NAME = ".write.lock"

MAX_FILE_BYTES = MAX_BYTES  # hard resident ceiling: 1 GiB
MAX_MANIFEST_BYTES = 1 * 1024 * 1024
HASH_CHUNK = 1 * 1024 * 1024
_HEX_DIGEST_CHARS = 64


class PersistenceError(ValueError):
    """Raised when a checkpoint cannot be written or safely recovered."""


# Convenience alias for callers that prefer an explicit checkpoint name.
CheckpointError = PersistenceError


@dataclass(frozen=True)
class Checkpoint:
    """A committed (or recovered) checkpoint generation."""

    generation: int
    manifest: dict[str, Any]
    slot: int
    digest: str
    path: Path
    created_ns: int = 0
    rejected: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "manifest": self.manifest,
            "slot": self.slot,
            "digest": self.digest,
            "path": str(self.path),
            "created_ns": self.created_ns,
            "rejected": list(self.rejected),
        }


# ---------------------------------------------------------------------------
# Small validation and hashing primitives
# ---------------------------------------------------------------------------


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PersistenceError("%s must be an integer >= %d" % (name, minimum))
    return value


def _require_digest(value: Any, name: str = "digest") -> str:
    if not isinstance(value, str) or len(value) != _HEX_DIGEST_CHARS:
        raise PersistenceError("%s must be a 64-character hex SHA-256" % name)
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise PersistenceError("%s must be lowercase hex SHA-256" % name) from exc
    if raw.hex() != value:
        raise PersistenceError("%s must be lowercase hex SHA-256" % name)
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError("checkpoint manifest is not JSON-serialisable") from exc
    return text.encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PersistenceError("duplicate JSON key in checkpoint: %s" % key)
        result[key] = value
    return result


def _decode_manifest(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PersistenceError("non-finite JSON constant: %s" % token)),
        )
    except PersistenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PersistenceError("invalid checkpoint JSON: %s" % exc) from exc
    if not isinstance(value, dict):
        raise PersistenceError("checkpoint record must be a JSON object")
    return value


def _pread(fd: int, length: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(fd, length, offset)
    os.lseek(fd, offset, os.SEEK_SET)  # pragma: no cover - non-POSIX
    return os.read(fd, length)


def _hash_range(fd: int, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    position = offset
    remaining = length
    while remaining > 0:
        chunk = _pread(fd, min(HASH_CHUNK, remaining), position)
        if not chunk:
            raise PersistenceError("file truncated while hashing")
        digest.update(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _hash_regions(fd: int, regions: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Hash declared regions once, returning aggregate and per-region digests."""
    aggregate = hashlib.sha256()
    per_region: list[str] = []
    for region in regions:
        digest = hashlib.sha256()
        position = int(region["offset"])
        remaining = int(region["length"])
        while remaining > 0:
            chunk = _pread(fd, min(HASH_CHUNK, remaining), position)
            if not chunk:
                raise PersistenceError("file truncated while hashing regions")
            digest.update(chunk)
            aggregate.update(chunk)
            position += len(chunk)
            remaining -= len(chunk)
        per_region.append(digest.hexdigest())
    return aggregate.hexdigest(), per_region


def _hash_mmap_range(source: _mmap.mmap, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    position = offset
    remaining = length
    while remaining > 0:
        chunk = source[position:position + min(HASH_CHUNK, remaining)]
        if not chunk:
            raise PersistenceError("mmap shorter than declared size")
        digest.update(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _hash_mmap_regions(source: _mmap.mmap,
                       regions: list[dict[str, Any]]) -> tuple[str, list[str]]:
    aggregate = hashlib.sha256()
    per_region: list[str] = []
    for region in regions:
        digest = hashlib.sha256()
        position = int(region["offset"])
        remaining = int(region["length"])
        while remaining > 0:
            chunk = source[position:position + min(HASH_CHUNK, remaining)]
            if not chunk:
                raise PersistenceError("mmap shorter than declared region")
            digest.update(chunk)
            aggregate.update(chunk)
            position += len(chunk)
            remaining -= len(chunk)
        per_region.append(digest.hexdigest())
    return aggregate.hexdigest(), per_region


def _open_regular(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(os.fspath(path), flags)
    except FileNotFoundError as exc:
        raise PersistenceError("checkpoint payload is missing: %s" % path) from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise PersistenceError(
                "checkpoint payload must not be a symlink: %s" % path) from exc
        raise PersistenceError(
            "cannot open checkpoint payload %s: %s" % (path, exc)) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PersistenceError(
                "checkpoint payload must be a regular file: %s" % path)
        if info.st_size > MAX_FILE_BYTES:
            raise PersistenceError(
                "checkpoint payload exceeds the 1 GiB resident bound: %s" % path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_all(fd: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise PersistenceError("checkpoint manifest truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _resolve(root: Path, raw: Any) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise PersistenceError("checkpoint file path must be a string")
    text = os.fspath(raw)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise PersistenceError("checkpoint file path is invalid")
    candidate = Path(text).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _normalise_regions(raw: Any, size: int,
                       *, name: str = "regions") -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise PersistenceError("%s must be a non-empty list" % name)
    prepared: list[tuple[int, dict[str, Any]]] = []
    spans: list[tuple[int, int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise PersistenceError("%s[%d] must be an object" % (name, index))
        offset = _require_int(item.get("offset"), "%s[%d].offset" % (name, index))
        length = _require_int(
            item.get("length"), "%s[%d].length" % (name, index), minimum=1)
        end = offset + length
        if end > size:
            raise PersistenceError(
                "%s[%d] lies outside the %d-byte file" % (name, index, size))
        for other_start, other_end in spans:
            if offset < other_end and other_start < end:
                raise PersistenceError("%s regions must not overlap" % name)
        spans.append((offset, end))
        region = dict(item)
        region["offset"] = offset
        region["length"] = length
        if "digest" in region:
            region["digest"] = _require_digest(
                region["digest"], "%s[%d].digest" % (name, index))
        prepared.append((offset, region))
    prepared.sort(key=lambda pair: pair[0])
    return [region for _offset, region in prepared]


# ---------------------------------------------------------------------------
# Public digest and verification helpers
# ---------------------------------------------------------------------------


def manifest_digest(path: "str | os.PathLike[str] | _mmap.mmap | Mapping[str, Any]",
                    manifest: Mapping[str, Any] | None = None) -> str:
    """Return the SHA-256 digest a manifest would declare for ``path``.

    ``path`` may be a filesystem path or an open :class:`mmap.mmap` (including
    a mapping of a sparse file).  Without a manifest, or when the manifest
    declares no ``regions``, the whole file is hashed.  With ``regions``, only
    the declared byte ranges are hashed, in ascending offset order, exactly as
    :func:`checkpoint` and :func:`verify_file` do.

    For convenience, a mapping may be passed instead of a path (with no second
    argument); its canonical JSON SHA-256 is returned.
    """
    if isinstance(path, Mapping) and manifest is None:
        return hashlib.sha256(_canonical_json(path)).hexdigest()
    entry = _entry_for_source(manifest, path)
    if isinstance(path, _mmap.mmap):
        size = len(path)
        if size > MAX_FILE_BYTES:
            raise PersistenceError("mmap exceeds the 1 GiB resident bound")
        _check_declared_size(entry, size)
        regions = _regions_from_entry(entry, size)
        if regions is not None:
            digest, _per_region = _hash_mmap_regions(path, regions)
            return digest
        return _hash_mmap_range(path, 0, size)

    source = Path(path)
    fd = _open_regular(source)
    try:
        size = os.fstat(fd).st_size
        _check_declared_size(entry, size)
        regions = _regions_from_entry(entry, size)
        if regions is not None:
            digest, _per_region = _hash_regions(fd, regions)
            return digest
        return _hash_range(fd, 0, size)
    finally:
        os.close(fd)


def verify_file(path: "str | os.PathLike[str] | _mmap.mmap",
                manifest: Mapping[str, Any] | None) -> bool:
    """Return ``True`` only if ``path`` satisfies every declared checksum.

    The manifest may be a single file descriptor (as stored in
    ``manifest["files"][i]``) or a full checkpoint manifest containing one or
    more descriptors.  Missing files, symlinks, size mismatch, region digest
    mismatch and full/aggregate digest mismatch all return ``False``; this
    function never mutates the filesystem.
    """
    try:
        return _verify_source(path, manifest, require_digest=False)
    except (PersistenceError, OSError, ValueError, TypeError):
        return False


def _check_declared_size(entry: Mapping[str, Any] | None, size: int) -> None:
    if entry is not None and "size" in entry:
        declared = _require_int(entry.get("size"), "size")
        if declared != size:
            raise PersistenceError(
                "checkpoint file size mismatch: declared %d, actual %d"
                % (declared, size))


def _regions_from_entry(entry: Mapping[str, Any] | None,
                        size: int) -> list[dict[str, Any]] | None:
    if entry is None or entry.get("regions") is None:
        return None
    return _normalise_regions(entry.get("regions"), size)


def _entry_for_source(manifest: Mapping[str, Any] | None,
                      source: Any) -> Mapping[str, Any] | None:
    if manifest is None:
        return None
    if not isinstance(manifest, Mapping):
        raise PersistenceError("manifest must be a mapping")
    files = manifest.get("files")
    if files is None and "entries" in manifest:
        files = manifest.get("entries")
    if files is None:
        return manifest
    if not isinstance(files, (list, tuple)):
        raise PersistenceError("manifest files must be a list")
    if len(files) == 1 and isinstance(files[0], Mapping):
        return files[0]
    wanted = os.fspath(source) if isinstance(source, (str, os.PathLike)) else None
    for item in files:
        if isinstance(item, Mapping):
            declared = item.get("path", item.get("file"))
            if declared is not None and wanted is not None and os.fspath(declared) == wanted:
                return item
    raise PersistenceError("manifest does not describe this file")


def _verify_declared_hashes(entry: Mapping[str, Any], size: int,
                            full_digest, region_digests, *,
                            require_digest: bool) -> bool:
    """Apply the manifest digest contract to a path or mmap source."""
    regions = _regions_from_entry(entry, size)
    if regions is not None:
        aggregate, per_region = region_digests(regions)
        for index, (region, actual) in enumerate(zip(regions, per_region)):
            declared = region.get("digest")
            if declared is None:
                if require_digest:
                    raise PersistenceError("region %d has no digest" % index)
                continue
            if _require_digest(declared) != actual:
                raise PersistenceError("region %d digest mismatch" % index)
        declared = entry.get("digest")
        if declared is None:
            if require_digest:
                raise PersistenceError("manifest has no aggregate digest")
            return True
        if _require_digest(declared) != aggregate:
            raise PersistenceError("aggregate digest mismatch")
        return True

    declared = entry.get("digest")
    if declared is None:
        if require_digest:
            raise PersistenceError("manifest has no digest")
        return True
    if _require_digest(declared) != full_digest():
        raise PersistenceError("digest mismatch")
    return True


def _verify_source(source: Any, manifest: Mapping[str, Any] | None,
                   *, require_digest: bool) -> bool:
    entry = _entry_for_source(manifest, source)
    if entry is None:
        return False
    return _verify_entry_source(source, entry, require_digest=require_digest)


def _verify_entry_source(source: Any, entry: Mapping[str, Any],
                         *, require_digest: bool) -> bool:
    """Verify one already-selected file descriptor against its source."""
    if isinstance(source, _mmap.mmap):
        size = len(source)
        if size > MAX_FILE_BYTES:
            raise PersistenceError("mmap exceeds the 1 GiB resident bound")
        _check_declared_size(entry, size)
        return _verify_declared_hashes(
            entry, size,
            lambda: _hash_mmap_range(source, 0, size),
            lambda regions: _hash_mmap_regions(source, regions),
            require_digest=require_digest)

    path = Path(source)
    fd = _open_regular(path)
    try:
        size = os.fstat(fd).st_size
        _check_declared_size(entry, size)
        result = _verify_declared_hashes(
            entry, size,
            lambda: _hash_range(fd, 0, size),
            lambda regions: _hash_regions(fd, regions),
            require_digest=require_digest)
        if os.fstat(fd).st_size != size:
            raise PersistenceError(
                "checkpoint payload changed size during verification")
        return result
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Manifest normalisation and verification
# ---------------------------------------------------------------------------


def _normalise_file_entry(root: Path, raw: Any) -> dict[str, Any]:
    if isinstance(raw, (str, os.PathLike)):
        raw = {"path": os.fspath(raw)}
    if not isinstance(raw, Mapping):
        raise PersistenceError("manifest file entry must be an object")
    record = dict(raw)
    declared_path = record.get("path", record.get("file"))
    if declared_path is None:
        raise PersistenceError("manifest file entry has no path")
    if not isinstance(declared_path, (str, os.PathLike)):
        raise PersistenceError("manifest file path must be a string")
    record["path"] = os.fspath(declared_path)
    record.pop("file", None)

    resolved = _resolve(root, record["path"])
    fd = _open_regular(resolved)
    try:
        size = os.fstat(fd).st_size
        if "size" in record:
            declared_size = _require_int(record.get("size"), "size")
            if declared_size != size:
                raise PersistenceError(
                    "checkpoint file size mismatch for %s: declared %d, actual %d"
                    % (record["path"], declared_size, size))
        record["size"] = size

        regions = _regions_from_entry(record, size)
        if regions is not None:
            aggregate, per_region = _hash_regions(fd, regions)
            for index, (region, actual) in enumerate(zip(regions, per_region)):
                declared = region.get("digest")
                if declared is not None and declared != actual:
                    raise PersistenceError(
                        "region %d digest mismatch for %s"
                        % (index, record["path"]))
                region["digest"] = actual
            declared = record.get("digest")
            if declared is not None and _require_digest(declared) != aggregate:
                raise PersistenceError(
                    "aggregate digest mismatch for %s" % record["path"])
            record["digest"] = aggregate
            record["regions"] = regions
            if os.fstat(fd).st_size != size:
                raise PersistenceError(
                    "checkpoint payload changed size while hashing: %s"
                    % record["path"])
            return record

        actual = _hash_range(fd, 0, size)
        declared = record.get("digest")
        if declared is not None and _require_digest(declared) != actual:
            raise PersistenceError("digest mismatch for %s" % record["path"])
        record["digest"] = actual
        if os.fstat(fd).st_size != size:
            raise PersistenceError(
                "checkpoint payload changed size while hashing: %s"
                % record["path"])
        return record
    finally:
        os.close(fd)


def _normalise_manifest(root: Path, generation: int,
                        manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise PersistenceError("checkpoint manifest must be a mapping")
    payload = dict(manifest)

    raw_files: Any
    if "files" in payload:
        raw_files = payload.get("files")
    elif "entries" in payload:
        raw_files = payload.get("entries")
    elif "path" in payload or "file" in payload:
        raw_files = [{key: payload[key] for key in
                      ("path", "file", "size", "digest", "regions")
                      if key in payload}]
    else:
        raw_files = []
    if not isinstance(raw_files, (list, tuple)):
        raise PersistenceError("checkpoint manifest files must be a list")

    files = [_normalise_file_entry(root, item) for item in raw_files]
    payload.pop("entries", None)
    if "files" not in payload and raw_files and "path" not in payload:
        payload.pop("file", None)
    payload["schema"] = MANIFEST_SCHEMA
    payload["generation"] = generation
    payload["files"] = files
    return payload


def _verify_manifest(root: Path, manifest: Any, *, generation: int | None = None) -> None:
    if not isinstance(manifest, Mapping):
        raise PersistenceError("checkpoint manifest must be an object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PersistenceError("unknown checkpoint manifest schema")
    declared_generation = _require_int(
        manifest.get("generation"), "manifest generation", minimum=1)
    if generation is not None and declared_generation != generation:
        raise PersistenceError("manifest generation does not match its record")
    files = manifest.get("files", [])
    if not isinstance(files, (list, tuple)):
        raise PersistenceError("checkpoint manifest files must be a list")
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise PersistenceError("manifest file entry %d is not an object" % index)
        path = _resolve(root, entry.get("path"))
        try:
            _verify_entry_source(path, entry, require_digest=True)
        except PersistenceError as exc:
            raise PersistenceError(
                "payload verification failed for %s: %s" % (entry.get("path"), exc)
            ) from exc


# ---------------------------------------------------------------------------
# Slot records: reads, writes, atomic replacement
# ---------------------------------------------------------------------------


def _ensure_root(root: Path) -> None:
    if root.is_symlink():
        raise PersistenceError("checkpoint directory must not be a symlink")
    if root.exists():
        if not root.is_dir():
            raise PersistenceError("checkpoint path is not a directory")
    else:
        try:
            root.mkdir(parents=True, mode=0o700)
        except OSError as exc:
            raise PersistenceError(
                "cannot create checkpoint directory %s: %s" % (root, exc)) from exc
    try:
        os.chmod(root, 0o700)
    except OSError as exc:
        raise PersistenceError(
            "cannot secure checkpoint directory %s: %s" % (root, exc)) from exc


@contextmanager
def _write_lock(root: Path) -> Iterator[None]:
    lock_path = root / LOCK_NAME
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise PersistenceError("cannot open checkpoint write lock: %s" % exc) from exc
    try:
        os.fchmod(fd, 0o600)
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
        yield
    finally:
        if _fcntl is not None:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - unlock best effort
                pass
        os.close(fd)


def _cleanup_temps(root: Path) -> None:
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(TEMP_PREFIX) and entry.name.endswith(TEMP_SUFFIX):
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # A stale temp never participates in recovery; if it cannot be
                # removed it must not turn a healthy checkpoint into a failure.
                pass


def _fsync_dir(root: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(root, flags)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:  # pragma: no cover - depends on filesystem
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
                raise
    finally:
        os.close(fd)


def _atomic_write(root: Path, target: Path, data: bytes) -> None:
    temporary = root / (TEMP_PREFIX + uuid.uuid4().hex + TEMP_SUFFIX)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = -1  # fdopen owns and closes the descriptor now
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_dir(root)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover - preserve the original failure
            pass
        raise


def _slot_record(generation: int, manifest: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "generation": generation,
        "created_ns": time.time_ns(),
        "manifest": manifest,
    }
    checksum = hashlib.sha256(_canonical_json(body)).hexdigest()
    record = dict(body)
    record["checksum"] = checksum
    return record


def _read_slot(root: Path, path: Path, slot: int) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise PersistenceError("%s must not be a symlink" % path.name) from exc
        raise PersistenceError("cannot open %s: %s" % (path.name, exc)) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PersistenceError("%s is not a regular file" % path.name)
        if info.st_size > MAX_MANIFEST_BYTES:
            raise PersistenceError("%s exceeds the manifest size bound" % path.name)
        raw = _read_all(fd, info.st_size)
    finally:
        os.close(fd)

    record = _decode_manifest(raw)
    if record.get("schema") != SCHEMA:
        raise PersistenceError("%s has an unknown checkpoint schema" % path.name)
    generation = _require_int(
        record.get("generation"), "checkpoint generation", minimum=1)
    checksum = _require_digest(record.get("checksum"), "checkpoint checksum")
    body = {key: value for key, value in record.items() if key != "checksum"}
    actual = hashlib.sha256(_canonical_json(body)).hexdigest()
    if actual != checksum:
        raise PersistenceError("%s checksum mismatch" % path.name)
    created_ns = _require_int(record.get("created_ns", 0), "created_ns")
    manifest = record.get("manifest")
    _verify_manifest(root, manifest, generation=generation)
    return {
        "generation": generation,
        "manifest": dict(manifest),
        "slot": slot,
        "digest": checksum,
        "path": path,
        "created_ns": created_ns,
    }


def _scan_slots(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    rejected: list[str] = []
    for slot, name in enumerate(SLOT_NAMES):
        path = root / name
        try:
            record = _read_slot(root, path, slot)
        except (PersistenceError, OSError, ValueError) as exc:
            if path.exists() or path.is_symlink():
                rejected.append("%s: %s" % (name, exc))
            continue
        if record is not None:
            valid.append(record)
    return valid, rejected


def _select_newest(valid: list[dict[str, Any]],
                   rejected: list[str]) -> dict[str, Any]:
    if not valid:
        detail = "; ".join(rejected) if rejected else "no checkpoint slots found"
        raise PersistenceError(
            "no valid resident checkpoint; refusing to recover (%s)" % detail)
    newest = max(valid, key=lambda item: (item["generation"], item["slot"]))
    same_generation = [item for item in valid
                       if item["generation"] == newest["generation"]]
    if len(same_generation) > 1:
        checksums = {item["digest"] for item in same_generation}
        if len(checksums) != 1:
            raise PersistenceError(
                "divergent same-generation resident checkpoints; refusing to recover")
    return newest


# ---------------------------------------------------------------------------
# Public checkpoint / recovery API
# ---------------------------------------------------------------------------


def checkpoint(path: "str | os.PathLike[str]", generation: int,
               manifest: Mapping[str, Any]) -> Checkpoint:
    """Atomically publish ``manifest`` as the checkpoint for ``generation``.

    Payload files declared by the manifest are hashed and size-checked before
    anything is published.  Missing digests are filled in the returned/stored
    manifest.  The new generation must be strictly greater than the newest
    valid generation on disk; stale writers are refused.  The older of the two
    slots survives, so the immediately previous valid generation remains
    recoverable after a torn newest write.
    """
    root = Path(path).expanduser()
    generation = _require_int(generation, "generation", minimum=1)
    _ensure_root(root)
    with _write_lock(root):
        _cleanup_temps(root)
        valid, rejected = _scan_slots(root)
        if not valid and rejected:
            raise PersistenceError(
                "no valid resident checkpoint to continue from; refusing to "
                "reinitialize (%s)" % "; ".join(rejected))
        newest = _select_newest(valid, rejected) if valid else None
        if newest is not None and generation <= newest["generation"]:
            raise PersistenceError(
                "stale checkpoint generation %d; newest valid is %d"
                % (generation, newest["generation"]))
        normalised = _normalise_manifest(root, generation, manifest)
        record = _slot_record(generation, normalised)
        data = _canonical_json(record) + b"\n"
        if len(data) > MAX_MANIFEST_BYTES:
            raise PersistenceError(
                "checkpoint manifest exceeds the %d-byte slot bound"
                % MAX_MANIFEST_BYTES)
        target_slot = 0 if newest is None else (int(newest["slot"]) ^ 1)
        target = root / SLOT_NAMES[target_slot]
        _atomic_write(root, target, data)
        return Checkpoint(
            generation=generation,
            manifest=normalised,
            slot=target_slot,
            digest=record["checksum"],
            path=target,
            created_ns=record["created_ns"],
        )


def recover(path: "str | os.PathLike[str]") -> Checkpoint:
    """Return the newest fully verified checkpoint, or fail closed.

    Both slots are independently parsed, checksum-verified and matched against
    their declared payload files.  A torn, truncated, checksum-mismatched or
    payload-corrupt newest slot is ignored in favour of the other valid slot.
    If neither slot is valid, :class:`PersistenceError` is raised; recovery
    never falls back to an empty/genesis state.
    """
    root = Path(path).expanduser()
    if root.is_symlink():
        raise PersistenceError("checkpoint directory must not be a symlink")
    if not root.exists():
        raise PersistenceError(
            "no valid resident checkpoint; refusing to recover "
            "(checkpoint directory is missing)")
    if not root.is_dir():
        raise PersistenceError("checkpoint path is not a directory")
    valid, rejected = _scan_slots(root)
    newest = _select_newest(valid, rejected)
    return Checkpoint(
        generation=newest["generation"],
        manifest=newest["manifest"],
        slot=newest["slot"],
        digest=newest["digest"],
        path=newest["path"],
        created_ns=newest["created_ns"],
        rejected=tuple(rejected),
    )


__all__ = (
    "SCHEMA",
    "MANIFEST_SCHEMA",
    "SLOT_NAMES",
    "TEMP_PREFIX",
    "TEMP_SUFFIX",
    "LOCK_NAME",
    "MAX_FILE_BYTES",
    "MAX_MANIFEST_BYTES",
    "PersistenceError",
    "CheckpointError",
    "Checkpoint",
    "checkpoint",
    "recover",
    "manifest_digest",
    "verify_file",
)
