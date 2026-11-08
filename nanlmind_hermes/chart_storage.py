"""Crash-recoverable byte transport for a NaN16 universe chart.

This is ONLY a host-side persistence adapter: no mathematical op, native head
mutation, admission, or NaN16 instruction is interpreted here. The original
NLMGLOB3 canonical checkpoint remains read-only and authoritative. Two
generation-stamped slots hold a local *unmerged* 8 MiB chart overlay. An
SHA-256 digest and canonical parent root bind every slot to its source.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import struct
import uuid
from pathlib import Path

MAGIC = b"N16SNAP1"
CHART_MAGIC = b"NLMCHRT1"
PAGE_BYTES = 8 * 1024 * 1024
HEADER = struct.Struct("<8sQ32s32s")
SLOT_NAMES = ("chart.0.n16s", "chart.1.n16s")


class ChartStorageError(ValueError):
    pass


def _root(state_dir: str | Path) -> Path:
    return Path(state_dir).expanduser() / "language" / "chart-checkpoints"


def _parent(root_hash: str) -> bytes:
    if type(root_hash) is not str or len(root_hash) != 64:
        raise ChartStorageError("invalid canonical parent root")
    try:
        raw = bytes.fromhex(root_hash)
    except ValueError as exc:
        raise ChartStorageError("invalid canonical parent root") from exc
    if raw.hex() != root_hash:
        raise ChartStorageError("canonical parent must be lowercase SHA-256")
    return raw


def _validate_chart(page: bytes) -> None:
    if len(page) != PAGE_BYTES or page[:8] != CHART_MAGIC:
        raise ChartStorageError("invalid NaN16 chart page")
    if struct.unpack_from("<I", page, 36)[0] != PAGE_BYTES:
        raise ChartStorageError("invalid NaN16 chart size")
    if struct.unpack_from("<I", page, 32)[0] != 256:
        raise ChartStorageError("invalid NaN16 chart head count")


def _read(path: Path):
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ChartStorageError("snapshot slot is not a regular file")
    if path.stat().st_size != HEADER.size + PAGE_BYTES:
        raise ChartStorageError("snapshot slot length mismatch")
    with path.open("rb") as stream:
        head = stream.read(HEADER.size)
        body = stream.read(PAGE_BYTES)
        if stream.read(1):
            raise ChartStorageError("snapshot has trailing bytes")
    magic, generation, root, digest = HEADER.unpack(head)
    if magic != MAGIC or generation == 0:
        raise ChartStorageError("invalid snapshot header")
    if hashlib.sha256(body).digest() != digest:
        raise ChartStorageError("snapshot digest mismatch")
    _validate_chart(body)
    return (generation, root, body)


def _candidates(folder: Path, parent: bytes):
    valid = []
    corrupt = []
    for slot, name in enumerate(SLOT_NAMES):
        path = folder / name
        try:
            record = _read(path)
            if record is not None and record[1] == parent:
                valid.append((record[0], slot, record[2]))
        except (OSError, ValueError, struct.error) as exc:
            corrupt.append(str(exc))
    if valid:
        valid.sort(key=lambda item: (item[0], item[1]))
        # Two individually SHA-valid slots at the same generation are not
        # necessarily the same chart. Choosing a slot by filename would
        # silently fork the one-universe history and authorize arbitrary
        # native training against one of two equally current pages.
        # Exact duplicate pages are harmless (e.g. restored copies), but
        # divergent equal-generation pages must fail closed for readers AND
        # the under-lock writer/CAS path that calls this helper.
        if (len(valid) == 2 and valid[0][0] == valid[1][0]
                and valid[0][2] != valid[1][2]):
            raise ChartStorageError(
                "ambiguous same-generation chart snapshots; manual recovery required"
            )
        return valid[-1]
    if corrupt:
        raise ChartStorageError("no valid snapshot: " + "; ".join(corrupt))
    return None


def load_chart_snapshot(state_dir: str | Path, root_hash: str):
    """Return (generation, eight-megabyte-page), or (0, None).

    A damaged newest slot is ignored when the other valid slot can be used.
    A damaged pair never silently resets the chart to genesis.
    A checkpoint for another canonical root is never applied.
    """
    folder = _root(state_dir)
    entry = _candidates(folder, _parent(root_hash))
    return (entry[0], entry[2]) if entry else (0, None)


def load_chart_snapshot_pair(state_dir: str | Path, root_hash: str,
                             prior_generation: int) -> tuple[bytes, bytes]:
    """Read an exact adjacent, SHA-valid N16SNAP1 transition.

    An unacknowledged native head commit can only be reconciled if BOTH
    pre-commit and post-commit snapshots survive with consecutive generation
    numbers under the same immutable parent. No torn-slot fallback here:
    this operation needs evidence of the precise byte transition, rather
    than just the newest recoverable chart.
    """
    if type(prior_generation) is not int or prior_generation < 1:
        raise ChartStorageError("prior chart generation must be positive")
    parent = _parent(root_hash)
    folder = _root(state_dir)
    current = _candidates(folder, parent)
    if current is None or current[0] != prior_generation + 1:
        raise ChartStorageError("exact next-generation snapshot unavailable")
    prior_path = folder / SLOT_NAMES[current[1] ^ 1]
    try:
        prior = _read(prior_path)
    except (ValueError, OSError, struct.error) as exc:
        raise ChartStorageError("adjacent prior snapshot is corrupt") from exc
    if (prior is None or prior[0] != prior_generation
            or prior[1] != parent):
        raise ChartStorageError("adjacent parent-bound prior snapshot unavailable")
    return prior[2], current[2]


def save_chart_snapshot(state_dir: str | Path, root_hash: str,
                        page: bytes | bytearray, expected_generation: int,
                        *, expected_page_sha256: str | None = None):
    """Atomically publish the alternate slot, using a generation lease.

    Refuses stale writers instead of silently overwriting another runner's
    work. The previous valid slot survives a partial/torn newest write.
    """
    if type(expected_generation) is not int or expected_generation < 0:
        raise ChartStorageError("invalid expected generation")
    if expected_page_sha256 is not None:
        if (type(expected_page_sha256) is not str
                or len(expected_page_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_page_sha256)):
            raise ChartStorageError("invalid expected chart page digest")
        if expected_generation < 1:
            raise ChartStorageError("chart page digest requires an existing generation")
    data = bytes(page)
    _validate_chart(data)
    root = _parent(root_hash)
    folder = _root(state_dir)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if folder.is_symlink():
        raise ChartStorageError("checkpoint directory must not be a symlink")
    os.chmod(folder, 0o700)
    lockpath = folder / ".write.lock"
    if lockpath.is_symlink():
        raise ChartStorageError("snapshot lock must not be a symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lockpath, flags, 0o600)
    try:
        with os.fdopen(fd, "rb") as lock:
            os.chmod(lockpath, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = _candidates(folder, root)
            generation = current[0] if current else 0
            if generation != expected_generation:
                raise ChartStorageError("stale snapshot generation; reload before writing")
            # The digest fence must be checked UNDER the same exclusive file
            # lock as the generation check. A valid same-generation but
            # different chart page must not be overwritten by a native bank
            # handoff that was prepared against unrelated chart bytes.
            if expected_page_sha256 is not None:
                if current is None or (
                    hashlib.sha256(current[2]).hexdigest() != expected_page_sha256
                ):
                    raise ChartStorageError("stale chart page digest; reload before writing")
            if generation == 0xFFFFFFFFFFFFFFFF:
                raise ChartStorageError("snapshot generation exhausted")
            target_slot = (current[1] ^ 1) if current else 0
            target = folder / SLOT_NAMES[target_slot]
            if target.is_symlink():
                raise ChartStorageError("target slot must not be a symlink")
            new_generation = generation + 1
            header = HEADER.pack(MAGIC, new_generation, root,
                                 hashlib.sha256(data).digest())
            temporary = folder / (".chart." + uuid.uuid4().hex + ".tmp")
            temp_fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(temp_fd, "wb") as stream:
                    stream.write(header)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                dir_fd = os.open(folder, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return new_generation
    except BaseException:
        # fdopen owns the descriptor once entered; avoid closing twice.
        raise
