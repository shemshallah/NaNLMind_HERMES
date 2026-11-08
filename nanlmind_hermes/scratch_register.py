"""One-gigabyte logical scratch register space for layered learning.

The space is 128 pages of 8 MiB.  Pages 0..119 are ten L0-L9 layer blocks of
twelve pages each; pages 120..127 are reserved for index/manifest data.  Pages
are allocated sparsely on first write, so the logical capacity is 1 GiB but the
physical files grow only as leaves are learned.

A page is an append-only record log:
``magic | version | layer | page | used | generation | count | sha256 | records``.
Every record is ``u32 length | canonical JSON``.  The digest covers the used
region and is verified before a page is read.  The module stores NaN-facing
learning leaves; it performs no arithmetic on the NaN word itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterable

SPACE_SCHEMA = "nanlm.scratch-register-space/1"
PAGE_SCHEMA = "nanlm.scratch-register-page/1"
RECORD_SCHEMA = "nanlm.scratch-register-record/1"

PAGE_BYTES = 8 * 1024 * 1024
HEADER_BYTES = 256
MAGIC = b"NLMSPACE"
VERSION = 1
LAYERS = 10
LAYER_NAMES = (
    "bit", "nibble", "byte", "phonetic", "morpheme",
    "lexical", "word_completion", "sentence", "paragraph", "narrative",
)
PAGES_PER_LAYER = 12
DATA_PAGES = LAYERS * PAGES_PER_LAYER
TOTAL_PAGES = 128
LOGICAL_BYTES = TOTAL_PAGES * PAGE_BYTES
MAX_RECORD_BYTES = 65_536
MAX_RECORDS_PER_PAGE = 65_535
ZERO = "0" * 64
_ADDRESS = re.compile(r"^nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW(\d{1,3}):register#(\d{1,3}):subrelation\((\d+)\)$")


class ScratchRegisterError(ValueError):
    pass


class ScratchRegisterCapacityError(ScratchRegisterError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def layer_index(layer: int | str) -> int:
    if isinstance(layer, str):
        try:
            return LAYER_NAMES.index(layer.casefold())
        except ValueError as exc:
            raise ScratchRegisterError("unknown L0-L9 layer name") from exc
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < LAYERS:
        raise ScratchRegisterError("layer must be an integer 0..9")
    return layer


def register_address(head: int, page: int, offset: int) -> str:
    if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head <= 255:
        raise ScratchRegisterError("head must be 0..255")
    if not 0 <= page < DATA_PAGES:
        raise ScratchRegisterError("page is not a data page")
    if not HEADER_BYTES <= offset < PAGE_BYTES:
        raise ScratchRegisterError("offset is outside the page record area")
    return (
        "nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0"
        f"::HW{head}:register#{page}:subrelation({offset})"
    )


def parse_register_address(value: str) -> tuple[int, int, int]:
    match = _ADDRESS.fullmatch(str(value))
    if match is None:
        raise ScratchRegisterError("invalid scratch register address")
    head, page, offset = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if page >= DATA_PAGES or not HEADER_BYTES <= offset < PAGE_BYTES:
        raise ScratchRegisterError("scratch address is outside the data pages")
    return head, page, offset


@dataclass
class _PageHeader:
    layer: int
    page: int
    used: int
    generation: int
    count: int
    digest: str


class ScratchPage:
    def __init__(self, root: str | Path, page: int, layer: int):
        if not 0 <= page < DATA_PAGES:
            raise ScratchRegisterError("page must be a data page")
        self.layer = layer_index(layer)
        self.page = page
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / f"page-{page:03d}.n16"
        self.lock_path = self.root / f"page-{page:03d}.lock"
        self._lock = threading.RLock()

    def _open(self, create: bool) -> int:
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        return os.open(self.path, flags, 0o600)

    def _header(self, fd: int) -> _PageHeader:
        raw = os.pread(fd, HEADER_BYTES, 0)
        if not raw:
            return _PageHeader(self.layer, self.page, HEADER_BYTES, 0, 0, ZERO)
        if len(raw) != HEADER_BYTES or raw[:8] != MAGIC:
            raise ScratchRegisterError(f"page {self.page} header is invalid")
        version = int.from_bytes(raw[8:10], "big")
        if version != VERSION:
            raise ScratchRegisterError(f"page {self.page} version is unsupported")
        layer = raw[10]
        page = int.from_bytes(raw[11:13], "big")
        used = int.from_bytes(raw[16:20], "big")
        generation = int.from_bytes(raw[20:24], "big")
        count = int.from_bytes(raw[24:28], "big")
        expected = raw[28:60].hex()
        if layer != self.layer or page != self.page:
            raise ScratchRegisterError(f"page {self.page} identity mismatch")
        if not HEADER_BYTES <= used <= PAGE_BYTES:
            raise ScratchRegisterError(f"page {self.page} used bound is invalid")
        if count > MAX_RECORDS_PER_PAGE:
            raise ScratchRegisterError(f"page {self.page} record count is invalid")
        return _PageHeader(layer, page, used, generation, count, expected)

    def _write_header(self, fd: int, used: int, generation: int, count: int, digest_hex: str) -> None:
        header = bytearray(HEADER_BYTES)
        header[0:8] = MAGIC
        header[8:10] = VERSION.to_bytes(2, "big")
        header[10] = self.layer
        header[11:13] = self.page.to_bytes(2, "big")
        header[16:20] = used.to_bytes(4, "big")
        header[20:24] = generation.to_bytes(4, "big")
        header[24:28] = count.to_bytes(4, "big")
        header[28:60] = bytes.fromhex(digest_hex)
        os.pwrite(fd, bytes(header), 0)
        os.fsync(fd)

    def _verify_region(self, fd: int, header: _PageHeader) -> None:
        used = header.used - HEADER_BYTES
        data = os.pread(fd, used, HEADER_BYTES)
        if len(data) != used:
            raise ScratchRegisterError(f"page {self.page} is truncated")
        actual = hashlib.sha256(data).hexdigest()
        if header.count == 0 and header.used == HEADER_BYTES:
            return
        if actual != header.digest:
            raise ScratchRegisterError(f"page {self.page} digest mismatch")
        position = 0
        count = 0
        while position < len(data):
            if position + 4 > len(data):
                raise ScratchRegisterError(f"page {self.page} record length is truncated")
            size = int.from_bytes(data[position:position + 4], "big")
            position += 4
            if size < 2 or size > MAX_RECORD_BYTES or position + size > len(data):
                raise ScratchRegisterError(f"page {self.page} record is invalid")
            position += size
            count += 1
        if count != header.count or position != len(data):
            raise ScratchRegisterError(f"page {self.page} record count mismatch")

    def header(self) -> _PageHeader:
        with self._lock:
            fd = self._open(create=False)
            try:
                return self._header(fd)
            finally:
                os.close(fd)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        raw = canonical(record)
        if not 2 <= len(raw) <= MAX_RECORD_BYTES:
            raise ScratchRegisterError("record payload is outside the page bound")
        with self._lock:
            lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                fd = self._open(create=True)
                try:
                    header = self._header(fd)
                    if header.count == 0 and header.used == HEADER_BYTES and os.fstat(fd).st_size == 0:
                        self._write_header(fd, HEADER_BYTES, 1, 0, ZERO)
                        header = self._header(fd)
                    self._verify_region(fd, header)
                    if header.count >= MAX_RECORDS_PER_PAGE:
                        raise ScratchRegisterCapacityError(f"page {self.page} record count is full")
                    if header.used + 4 + len(raw) > PAGE_BYTES:
                        raise ScratchRegisterCapacityError(f"page {self.page} is full")
                    offset = header.used
                    os.pwrite(fd, len(raw).to_bytes(4, "big") + raw, offset)
                    used = offset + 4 + len(raw)
                    generation = header.generation + 1
                    count = header.count + 1
                    data = os.pread(fd, used - HEADER_BYTES, HEADER_BYTES)
                    new_digest = hashlib.sha256(data).hexdigest()
                    self._write_header(fd, used, generation, count, new_digest)
                finally:
                    os.close(fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
        return {
            "page": self.page, "layer": self.layer, "offset": offset,
            "length": len(raw), "generation": generation,
            "record_index": count - 1,
            "address": register_address(int(record.get("head", 0)), self.page, offset),
        }

    def read(self, offset: int, length: int) -> bytes:
        if not HEADER_BYTES <= offset < PAGE_BYTES or length < 1 or offset + length > PAGE_BYTES:
            raise ScratchRegisterError("read is outside the page bounds")
        with self._lock:
            fd = self._open(create=False)
            try:
                header = self._header(fd)
                self._verify_region(fd, header)
                if offset + length > header.used:
                    raise ScratchRegisterError("read exceeds the used page region")
                data = os.pread(fd, length, offset)
            finally:
                os.close(fd)
        if len(data) != length:
            raise ScratchRegisterError("short page read")
        return data

    def records(self) -> Iterable[dict[str, Any]]:
        with self._lock:
            fd = self._open(create=False)
            try:
                header = self._header(fd)
                self._verify_region(fd, header)
                data = os.pread(fd, header.used - HEADER_BYTES, HEADER_BYTES)
            finally:
                os.close(fd)
        position = 0
        while position < len(data):
            size = int.from_bytes(data[position:position + 4], "big")
            position += 4
            payload = data[position:position + size]
            position += size
            yield json.loads(payload)

    def stats(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"page": self.page, "layer": self.layer, "allocated": False,
                    "used_bytes": 0, "records": 0, "generation": 0}
        header = self.header()
        return {
            "page": self.page, "layer": self.layer, "allocated": True,
            "used_bytes": header.used - HEADER_BYTES,
            "records": header.count, "generation": header.generation,
            "page_bytes": PAGE_BYTES,
            "free_bytes": PAGE_BYTES - header.used,
        }


class ScratchRegisterSpace:
    """Sparse 1 GiB logical L0-L9 scratch space."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.pages = {
            page: ScratchPage(self.root / "pages", page, page // PAGES_PER_LAYER)
            for page in range(DATA_PAGES)
        }
        self._lock = threading.RLock()

    def _candidate_pages(self, layer: int, head: int) -> list[int]:
        base = layer * PAGES_PER_LAYER
        start = head % PAGES_PER_LAYER
        return [base + ((start + offset) % PAGES_PER_LAYER)
                for offset in range(PAGES_PER_LAYER)]

    def append(self, *, layer: int | str, head: int, leaf_id: str,
               payload: Any, kind: str = "leaf") -> dict[str, Any]:
        layer_number = layer_index(layer)
        if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head <= 255:
            raise ScratchRegisterError("head must be 0..255")
        if not isinstance(leaf_id, str) or not leaf_id or len(leaf_id) > 128:
            raise ScratchRegisterError("leaf_id must be a bounded non-empty string")
        if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,31}", kind):
            raise ScratchRegisterError("kind has an unsupported value")
        record = {
            "schema": RECORD_SCHEMA,
            "layer": layer_number,
            "layer_name": LAYER_NAMES[layer_number],
            "head": head,
            "leaf_id": leaf_id,
            "kind": kind,
            "payload": payload,
            "leaf_digest": digest(payload),
        }
        size = len(canonical(record)) + 4
        for page_number in self._candidate_pages(layer_number, head):
            page = self.pages[page_number]
            try:
                if page.path.exists():
                    stats = page.stats()
                    if stats["free_bytes"] < size:
                        continue
                result = page.append(record)
            except ScratchRegisterCapacityError:
                continue
            result["layer"] = layer_number
            result["layer_name"] = LAYER_NAMES[layer_number]
            result["head"] = head
            result["leaf_id"] = leaf_id
            return result
        raise ScratchRegisterCapacityError(
            f"L{layer_number} scratch layer has no page with {size} free bytes"
        )

    def append_address(self, address: str, payload: Any, *, kind: str = "leaf") -> dict[str, Any]:
        head, page, _offset = parse_register_address(address)
        record = {
            "schema": RECORD_SCHEMA,
            "layer": self.pages[page].layer,
            "layer_name": LAYER_NAMES[self.pages[page].layer],
            "head": head,
            "leaf_id": address,
            "kind": kind,
            "payload": payload,
            "leaf_digest": digest(payload),
        }
        result = self.pages[page].append(record)
        result["layer"] = self.pages[page].layer
        result["layer_name"] = LAYER_NAMES[self.pages[page].layer]
        result["head"] = head
        result["leaf_id"] = address
        return result

    def records(self, *, layer: int | str | None = None,
                head: int | None = None) -> Iterable[dict[str, Any]]:
        layer_number = None if layer is None else layer_index(layer)
        if head is not None and (isinstance(head, bool) or not isinstance(head, int)
                                 or not 0 <= head <= 255):
            raise ScratchRegisterError("head must be 0..255")
        pages = range(DATA_PAGES)
        if layer_number is not None:
            pages = range(layer_number * PAGES_PER_LAYER,
                          (layer_number + 1) * PAGES_PER_LAYER)
        for page_number in pages:
            page = self.pages[page_number]
            if not page.path.exists():
                continue
            for record in page.records():
                if head is None or record.get("head") == head:
                    yield record

    def stats(self) -> dict[str, Any]:
        layers = {
            LAYER_NAMES[index]: {
                "pages": PAGES_PER_LAYER,
                "allocated_pages": sum(
                    1 for page in range(index * PAGES_PER_LAYER,
                                        (index + 1) * PAGES_PER_LAYER)
                    if self.pages[page].path.exists()
                ),
                "used_bytes": sum(
                    self.pages[page].stats().get("used_bytes", 0)
                    for page in range(index * PAGES_PER_LAYER,
                                      (index + 1) * PAGES_PER_LAYER)
                ),
                "records": sum(
                    self.pages[page].stats().get("records", 0)
                    for page in range(index * PAGES_PER_LAYER,
                                      (index + 1) * PAGES_PER_LAYER)
                ),
            }
            for index in range(LAYERS)
        }
        allocated = [page for page in self.pages.values() if page.path.exists()]
        used = sum(page.stats()["used_bytes"] for page in allocated)
        records = sum(page.stats()["records"] for page in allocated)
        return {
            "schema": SPACE_SCHEMA,
            "logical_bytes": LOGICAL_BYTES,
            "page_bytes": PAGE_BYTES,
            "total_pages": TOTAL_PAGES,
            "data_pages": DATA_PAGES,
            "index_pages": TOTAL_PAGES - DATA_PAGES,
            "allocated_pages": len(allocated),
            "used_bytes": used,
            "records": records,
            "layers": layers,
            "root": str(self.root),
        }

    def verify(self) -> dict[str, Any]:
        errors = []
        digests = {}
        for page in self.pages.values():
            if not page.path.exists():
                continue
            try:
                header = page.header()
                list(page.records())
                digests[str(page.page)] = header.digest
            except (ScratchRegisterError, OSError) as exc:
                errors.append({"page": page.page, "error": str(exc)[:200]})
        return {
            "schema": "nanlm.scratch-register-verify/1",
            "ok": not errors,
            "errors": errors,
            "page_digests": digests,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "nanlm.scratch-register-manifest/1",
            "stats": self.stats(),
            "verify": self.verify(),
        }
