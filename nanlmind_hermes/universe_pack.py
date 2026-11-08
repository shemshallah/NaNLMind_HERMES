"""Portable N16PACK1: one self-contained, verified *offline* universe image.

A live SQLite WAL is NEVER embedded into writable NaN16 guest memory.
Instead, this one-file archive carries a verified N16ROOT1/8MiB chart
snapshot plus online-backup images of the existing BS0DB and MC0DB.
All entries have bounded lengths and SHA-256 digests; no path names are
deserialized from untrusted archives, and unpack-to-live is forbidden.

Crucial limitation: separate online SQLite backups and the chart snapshot
cannot prove a distributed cross-store atomic instant. A caller must arrange
a quiesced external lease to make a logically consistent capture. The file
is research/transport evidence, NOT a canonical NANNET merge/restore.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from types import SimpleNamespace
from urllib.parse import quote

from .chart_storage import load_chart_snapshot, PAGE_BYTES
from .universe_root import read_universe_root

MAGIC = b"N16PACK1"
SCHEMA = "nanlm.universe-offline-pack/1"
VERSION = 1
HEADER = struct.Struct("<8sII32s")
SECTIONS = ("chart", "mc0db", "bs0db")
CHUNK = 1024 * 1024
MAX_MANIFEST = 64 * 1024
MAX_DB_BYTES = 4 * 1024 * 1024 * 1024


class UniversePackError(ValueError):
    pass


def _digest(source):
    sha = hashlib.sha256()
    with open(source, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def _safe_file(path, label):
    file = Path(path).expanduser()
    if file.is_symlink() or not file.is_file():
        raise UniversePackError(f"{label} must be an existing regular file")
    return file


def _backup(source: Path, destination: Path, identity: str) -> None:
    """SQLite online backup, never a byte copy of an active WAL database."""
    # SQLite URI mode=ro does not mutate source or its WAL.
    if source.stat().st_size > MAX_DB_BYTES:
        raise UniversePackError(f"{identity} source exceeds maximum")
    uri = "file:" + quote(source.resolve().as_posix(), safe="/") + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=15)) as original:
            with closing(sqlite3.connect(str(destination), timeout=15)) as backup:
                original.backup(backup)
                backup.commit()
                result = backup.execute("PRAGMA quick_check").fetchone()
                if result is None or result[0] != "ok":
                    raise UniversePackError(f"{identity} backup failed integrity check")
                # Do not mistake an arbitrary SQLite DB for the typed mapping store.
                row = backup.execute(
                    "SELECT version FROM math_schema WHERE store=?", (identity,)
                ).fetchone()
                if row is None or type(row[0]) is not int or row[0] < 1:
                    raise UniversePackError(f"{identity} math schema missing")
    except sqlite3.DatabaseError as exc:
        raise UniversePackError(f"{identity} is not a valid math store") from exc
    if destination.stat().st_size > MAX_DB_BYTES:
        raise UniversePackError(f"{identity} backup exceeds maximum")


def _validate_chart(page: bytes, parent_root: str) -> dict:
    if type(parent_root) is not str or len(parent_root) != 64:
        raise UniversePackError("expected canonical parent root digest")
    try:
        if bytes.fromhex(parent_root).hex() != parent_root:
            raise ValueError("mixed-case digest")
    except ValueError as exc:
        raise UniversePackError("expected canonical parent root digest") from exc
    if len(page) != PAGE_BYTES:
        raise UniversePackError("chart is not exactly 8 MiB")
    try:
        root = read_universe_root(SimpleNamespace(bytes=page, root_hash=parent_root))
    except ValueError as exc:
        raise UniversePackError("N16ROOT1 chart does not match canonical parent") from exc
    if not root["mounted"]:
        raise UniversePackError("N16ROOT1 was not mounted before checkpoint")
    return root


def _write_stream(out, source):
    if isinstance(source, bytes):
        out.write(source)
        return
    with open(source, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            out.write(chunk)


def export_n16pack(*, state_dir: str | Path, parent_root_sha256: str,
                   expected_chart_generation: int, mc0db: str | Path,
                   bs0db: str | Path, destination: str | Path) -> dict:
    """Export snapshot and existing stores as a verified one-file archive.

    No implicit checkpoint, no DB schema initialization, no NANNET publication
    and no overwrite of an existing destination. A caller must quiesce all
    writers externally for a genuinely cross-store-consistent capture.
    """
    if type(expected_chart_generation) is not int or expected_chart_generation < 1:
        raise UniversePackError("a positive checkpoint generation is required")
    try:
        generation, page = load_chart_snapshot(state_dir, parent_root_sha256)
    except ValueError as exc:
        raise UniversePackError("valid chart snapshot unavailable") from exc
    if generation != expected_chart_generation or page is None:
        raise UniversePackError("stale or unavailable chart checkpoint")
    root = _validate_chart(page, parent_root_sha256)
    mcfile = _safe_file(mc0db, "MC0DB")
    bsfile = _safe_file(bs0db, "BS0DB")
    outpath = Path(destination).expanduser()
    if outpath.is_symlink() or outpath.exists():
        raise UniversePackError("refusing to overwrite an existing N16PACK")
    if not outpath.parent.is_dir():
        raise UniversePackError("N16PACK destination parent does not exist")

    with tempfile.TemporaryDirectory(prefix=".n16pack-", dir=outpath.parent) as tmp:
        temp = Path(tmp)
        mc_backup = temp / "mc0.sqlite"
        bs_backup = temp / "bs0.sqlite"
        _backup(mcfile, mc_backup, "MC0DB")
        _backup(bsfile, bs_backup, "BS0DB")
        blobs = {"chart": page, "mc0db": mc_backup, "bs0db": bs_backup}
        specs = []
        for name in SECTIONS:
            obj = blobs[name]
            length = len(obj) if isinstance(obj, bytes) else obj.stat().st_size
            digest = hashlib.sha256(obj).hexdigest() if isinstance(obj, bytes) else _digest(obj)
            specs.append({"name": name, "length": length, "sha256": digest})
        manifest = {
            "schema": SCHEMA,
            "version": VERSION,
            "parent_root_sha256": parent_root_sha256,
            "chart_generation": generation,
            "root_block_sha256": root["mount_block_sha256"],
            "chart_bytes": PAGE_BYTES,
            "head_fixed_points": 256,
            "mirror_points": 32640,
            "conceptual_layers": 10,
            "sections": specs,
            "native_execution_verified": False,
            "canonical_training_committed": False,
            "cross_store_atomicity_attested": False,
            "import_to_live_allowed": False,
        }
        encoded = _json(manifest)
        if len(encoded) > MAX_MANIFEST:
            raise UniversePackError("manifest too large")
        archive = temp / "archive.tmp"
        with open(archive, "xb") as stream:
            os.chmod(archive, 0o600)
            stream.write(HEADER.pack(
                MAGIC, VERSION, len(encoded), hashlib.sha256(encoded).digest()
            ))
            stream.write(encoded)
            for name in SECTIONS:
                _write_stream(stream, blobs[name])
            stream.flush()
            os.fsync(stream.fileno())
        # Install by hard-linking in the same filesystem: fails closed if the
        # destination appeared during export, rather than overwriting it.
        try:
            os.link(archive, outpath, follow_symlinks=False)
        except FileExistsError as exc:
            raise UniversePackError("destination already exists") from exc
    return inspect_n16pack(outpath)


def inspect_n16pack(path: str | Path) -> dict:
    """Verify every byte of an untrusted N16PACK; never execute or restore."""
    file = _safe_file(path, "N16PACK")
    with file.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise UniversePackError("truncated N16PACK header")
        magic, version, manifest_size, manifest_digest = HEADER.unpack(header)
        if magic != MAGIC or version != VERSION or not 0 < manifest_size <= MAX_MANIFEST:
            raise UniversePackError("invalid N16PACK version, magic or length")
        encoded = stream.read(manifest_size)
        if len(encoded) != manifest_size or hashlib.sha256(encoded).digest() != manifest_digest:
            raise UniversePackError("N16PACK manifest digest mismatch")
        try:
            manifest = json.loads(encoded.decode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise UniversePackError("invalid N16PACK JSON manifest") from exc
        if _json(manifest) != encoded or manifest.get("schema") != SCHEMA:
            raise UniversePackError("noncanonical N16PACK manifest")
        if manifest.get("version") != VERSION:
            raise UniversePackError("wrong N16PACK schema version")
        if (manifest.get("chart_bytes") != PAGE_BYTES
                or type(manifest.get("chart_generation")) is not int
                or manifest["chart_generation"] < 1
                or manifest.get("head_fixed_points") != 256
                or manifest.get("mirror_points") != 32640
                or manifest.get("conceptual_layers") != 10
                or manifest.get("native_execution_verified") is not False
                or manifest.get("canonical_training_committed") is not False
                or manifest.get("import_to_live_allowed") is not False
                or manifest.get("cross_store_atomicity_attested") is not False):
            raise UniversePackError("N16PACK invariant or non-attestation field mismatch")
        specs = manifest.get("sections")
        if (type(specs) is not list or len(specs) != 3
                or [s.get("name") for s in specs if isinstance(s, dict)] != list(SECTIONS)):
            raise UniversePackError("missing or reordered N16PACK sections")
        for i, spec in enumerate(specs):
            name = spec["name"]
            limit = PAGE_BYTES if name == "chart" else MAX_DB_BYTES
            length = spec.get("length")
            digest = spec.get("sha256")
            if (type(length) is not int or length < 0 or length > limit
                    or name == "chart" and length != PAGE_BYTES
                    or type(digest) is not str or len(digest) != 64):
                raise UniversePackError("invalid section length/digest")
            try:
                if bytes.fromhex(digest).hex() != digest:
                    raise ValueError("mixed case")
            except ValueError as exc:
                raise UniversePackError("invalid section SHA-256") from exc
            sha = hashlib.sha256()
            remaining = length
            chart = bytearray() if name == "chart" else None
            prefix = b""
            while remaining:
                data = stream.read(min(remaining, CHUNK))
                if not data:
                    raise UniversePackError("truncated N16PACK payload")
                if len(prefix) < 16:
                    prefix += data[:16-len(prefix)]
                sha.update(data)
                if chart is not None:
                    chart.extend(data)
                remaining -= len(data)
            if sha.hexdigest() != digest:
                raise UniversePackError("section checksum mismatch")
            if name == "chart":
                root = _validate_chart(bytes(chart), manifest["parent_root_sha256"])
                if root["mount_block_sha256"] != manifest.get("root_block_sha256"):
                    raise UniversePackError("N16ROOT1 manifest digest mismatch")
            elif prefix != b"SQLite format 3\0":
                raise UniversePackError("not an SQLite database snapshot")
        if stream.read(1):
            raise UniversePackError("N16PACK contains unlisted trailing payload")
    return {
        "schema": SCHEMA,
        "verified": True,
        "path": str(file),
        "file_bytes": file.stat().st_size,
        "parent_root_sha256": manifest["parent_root_sha256"],
        "chart_generation": manifest["chart_generation"],
        "head_addresses": 256,
        "mirror_points": 32640,
        "layers": 10,
        "sections": [dict(s) for s in specs],
        "canonical_training_committed": False,
        "native_execution_verified": False,
        "import_to_live_allowed": False,
        "cross_store_atomicity_attested": False,
    }
