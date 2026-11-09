"""Crash-recoverable byte transport for a NaN16 universe chart.

This is ONLY a host-side persistence adapter: no mathematical op, native head
mutation, admission, or NaN16 instruction is interpreted here. The original
NLMGLOB3 canonical checkpoint remains read-only and authoritative. Two
generation-stamped slots hold a local *unmerged* 8 MiB chart overlay. An
SHA-256 digest and canonical parent root bind every slot to its source.
"""
from __future__ import annotations

from contextlib import contextmanager
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
    """Resolve the fixed checkpoint location without accepting child aliases.

    The caller (LocalRunner / SnapshotHeadBankFence) pins its trusted physical
    state home, but a *nested* language/ or chart-checkpoints/ symlink could
    still redirect a later read/write to another, individually SHA-valid
    N16SNAP1 chart. Reject pre-existing aliases or non-directory nodes
    before reading, creating or replacing a slot. This is not a substitute
    for exclusive host filesystem ownership or a cross-process lease.
    """
    base = Path(state_dir).expanduser()
    language = base / "language"
    folder = language / "chart-checkpoints"
    for component in (language, folder):
        if component.is_symlink():
            raise ChartStorageError("checkpoint directory components cannot be symlinks")
        if component.exists() and not component.is_dir():
            raise ChartStorageError("checkpoint directory component is not a directory")
    return folder


@contextmanager
def _opened_folder(state_dir: str | Path, *, create: bool):
    """Open the fixed checkpoint chain with O_NOFOLLOW at *each* level.

    All subsequent slot, lock, rename and fsync operations use the opened
    chart-checkpoints directory fd, never a re-resolved path. This prevents
    replacing language/ or chart-checkpoints/ with aliases between a path
    validation and a native checkpoint operation.
    """
    base = Path(state_dir).expanduser()
    # Preserve the existing precise rejection of pre-existing aliases while
    # descriptor-relative O_NOFOLLOW closes the post-check redirect window.
    _root(base)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    opened = []
    try:
        if create:
            base.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            parent_fd = os.open(base, flags)
        except FileNotFoundError:
            if not create:
                yield None
                return
            raise
        opened.append(parent_fd)
        for name in ("language", "chart-checkpoints"):
            try:
                child_fd = os.open(name, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                if not create:
                    yield None
                    return
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(name, flags, dir_fd=parent_fd)
            opened.append(child_fd)
            parent_fd = child_fd
        if create:
            os.fchmod(parent_fd, 0o700)
        yield parent_fd
    except OSError as exc:
        raise ChartStorageError("checkpoint directory descriptor could not be opened") from exc
    finally:
        for fd in reversed(opened):
            os.close(fd)


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


def _read_at(folder_fd: int, name: str):
    """Open a snapshot exactly once, refusing symlinks and nonregular nodes.

    fstat/read share one descriptor: swapping the filename after validation
    cannot cause the reader to verify a different object's bytes.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, flags, dir_fd=folder_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ChartStorageError("snapshot slot cannot be opened without following links") from exc
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ChartStorageError("snapshot slot is not a regular file")
        if info.st_size != HEADER.size + PAGE_BYTES:
            raise ChartStorageError("snapshot slot length mismatch")
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



def _candidates(folder_fd: int, parent: bytes):
    valid = []
    corrupt = []
    for slot, name in enumerate(SLOT_NAMES):
        try:
            record = _read_at(folder_fd, name)
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
    parent = _parent(root_hash)
    with _opened_folder(state_dir, create=False) as folder_fd:
        if folder_fd is None:
            return (0, None)
        entry = _candidates(folder_fd, parent)
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
    with _opened_folder(state_dir, create=False) as folder_fd:
        if folder_fd is None:
            raise ChartStorageError("exact next-generation snapshot unavailable")
        current = _candidates(folder_fd, parent)
        if current is None or current[0] != prior_generation + 1:
            raise ChartStorageError("exact next-generation snapshot unavailable")
        try:
            prior = _read_at(folder_fd, SLOT_NAMES[current[1] ^ 1])
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
    with _opened_folder(state_dir, create=True) as folder_fd:
        if folder_fd is None:
            raise ChartStorageError("checkpoint directory unavailable")
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            lock_fd = os.open(".write.lock", flags, 0o600, dir_fd=folder_fd)
        except OSError as exc:
            raise ChartStorageError("snapshot lock cannot be followed or opened") from exc
        with os.fdopen(lock_fd, "rb") as lock:
            lock_stat = os.fstat(lock.fileno())
            if not stat.S_ISREG(lock_stat.st_mode):
                raise ChartStorageError("snapshot lock is not a regular file")
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            # Check lock ownership after admission. This closes a common
            # pre-flock replacement case, but hostile concurrent unlink of
            # the lock by the same privileged filesystem owner still needs
            # an OS-level exclusive directory owner/lease.
            current_lock = os.stat(
                ".write.lock", dir_fd=folder_fd, follow_symlinks=False
            )
            if (current_lock.st_dev != lock_stat.st_dev or
                    current_lock.st_ino != lock_stat.st_ino):
                raise ChartStorageError("snapshot lock identity changed")
            current = _candidates(folder_fd, root)
            generation = current[0] if current else 0
            if generation != expected_generation:
                raise ChartStorageError("stale snapshot generation; reload before writing")
            if expected_page_sha256 is not None:
                if current is None or (
                    hashlib.sha256(current[2]).hexdigest() != expected_page_sha256
                ):
                    raise ChartStorageError("stale chart page digest; reload before writing")
            if generation == 0xFFFFFFFFFFFFFFFF:
                raise ChartStorageError("snapshot generation exhausted")
            target_slot = (current[1] ^ 1) if current else 0
            target = SLOT_NAMES[target_slot]
            try:
                target_stat = os.stat(target, dir_fd=folder_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
                    raise ChartStorageError("target slot must be a regular file, not a symlink")
            new_generation = generation + 1
            header = HEADER.pack(
                MAGIC, new_generation, root, hashlib.sha256(data).digest()
            )
            temporary = ".chart." + uuid.uuid4().hex + ".tmp"
            temp_fd = os.open(
                temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600, dir_fd=folder_fd,
            )
            try:
                with os.fdopen(temp_fd, "wb") as stream:
                    stream.write(header)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Both endpoints refer to the same pinned physical folder.
                # A retargeted path alias cannot change the CAS destination.
                os.replace(
                    temporary, target, src_dir_fd=folder_fd,
                    dst_dir_fd=folder_fd,
                )
                os.fsync(folder_fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=folder_fd)
                except FileNotFoundError:
                    pass
            return new_generation
