"""Duration and N16TXT policy shared by plugin tools and tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_DURATION_MINUTES = 360
MIN_DURATION_MINUTES = 1
DEFAULT_DURATION_MINUTES = 30
MAX_RECORD_BYTES = 4096
MAX_CORPUS_BYTES = 1_048_576
N16TXT_HEADER = b"N16TXT/1"


@dataclass(frozen=True)
class CorpusInfo:
    records: tuple[str, ...]
    canonical_bytes: bytes
    unique_records: int


def validate_duration(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("duration_minutes must be an integer")
    if not MIN_DURATION_MINUTES <= value <= MAX_DURATION_MINUTES:
        raise ValueError("duration_minutes must be between 1 and 360")
    return value


def parse_n16txt(raw: bytes) -> CorpusInfo:
    if len(raw) > MAX_CORPUS_BYTES:
        raise ValueError("corpus exceeds the 1 MiB N16TXT v1 limit")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is not permitted")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8: {exc}") from exc
    # CRLF is accepted; bare CR is not part of the canonical grammar.
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        raise ValueError("bare CR line endings are not permitted")
    lines = text.split("\n")
    if not lines or lines[0] != "N16TXT/1":
        raise ValueError("missing exact N16TXT/1 header")
    records: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        encoded = line.encode("utf-8")
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError(f"record on line {line_no} exceeds 4096 UTF-8 bytes")
        for ch in line:
            cp = ord(ch)
            if cp == 0 or (cp < 0x20 and ch != "\t") or 0x7f <= cp <= 0x9f:
                raise ValueError(f"forbidden control character on line {line_no}")
            if 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFF) in (0xFFFE, 0xFFFF):
                raise ValueError(f"Unicode noncharacter on line {line_no}")
        if line not in seen:
            seen.add(line)
            records.append(line)
    canonical = N16TXT_HEADER + b"\n" + b"".join(r.encode("utf-8") + b"\n" for r in records)
    if len(canonical) > MAX_CORPUS_BYTES:
        raise ValueError("canonical corpus exceeds the 1 MiB N16TXT v1 limit")
    return CorpusInfo(tuple(records), canonical, len(records))


def load_n16txt(path: str | Path) -> CorpusInfo:
    return parse_n16txt(Path(path).expanduser().read_bytes())
