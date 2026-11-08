"""Local universe chart mapping with real temporal rails.

Hermes trains its own local universe (``global_universe.nmg``) before any merge
to Koyeb. This module holds that universe in memory as the documented 8 MiB
chart page and binds a temporal register to a pair cell's two real rails:

    s+  forward rail   past -> future
    s-  reverse rail   future -> past

The chart is assembled from the serialized ``NLMU0001`` page exactly as the
native ``Universe`` layout defines it (cells, address registers, descriptors,
shared registers, heads, scratch, canon/pair tables). It is held in a
module-level cache so the 8 MiB page stays in memory across chat turns.

The mapped register lives in the chart's free extension area
(``0x63BA10``+): magic, frame, forward/back rail words and a 256-cell
temporal state. The pair cell's ``s+``/``s-`` words in the shared-register
region are the real rails; register commits advance a logical frame and copy
forward into back. The on-disk ``.nmg`` is opened read-only and never mutated:
writes live in the in-memory chart plus a small ``temporal_register.json``
sidecar, which is the local state a later merge can inspect.

If no trained chart exists yet, the register is still usable in memory
(``mapped=False``) so chat is never blocked; metadata always says which mode
was used.
"""
from __future__ import annotations

import struct
import threading
from pathlib import Path

from .chart_storage import load_chart_snapshot, save_chart_snapshot, ChartStorageError

MAGIC_GLOBAL = b"NLMGLOB3"
MAGIC_UNIVERSE = b"NLMU0001"
MAGIC_CHART = b"NLMCHRT1"
MAGIC_REGISTER = b"NLMREG01"

PAGE_BYTES = 8 * 1024 * 1024
ADDRESS_COUNT = 65_536
FIXED_POINT_COUNT = 256
MIRROR_PAIR_COUNT = 32_640
CANONICAL_COUNT = 32_896
HEAD_COUNT = 256
CELL_BYTES = 64
ADDRESS_REG_BYTES = 32
DESCRIPTOR_BYTES = 48
HEAD_BYTES = 32
HEAD_SCRATCH_BYTES = 1_024
CHART_SCRATCH_BYTES = 65_536
ADDRESS_DIRTY_BYTES = 8_192
ADJACENCY_BYTES = 8_192

OFF_INDEX = 0x001000
OFF_CANON = 0x021000
OFF_PAIR_TABLE = 0x031100
OFF_ADDRESS_REGS = 0x041000
OFF_DESCRIPTORS = 0x241000
OFF_SHARED_REGS = 0x3C2800
OFF_HEAD_TABLE = 0x5C4800
OFF_ADJACENCY = 0x5C6800
OFF_DIRTY_CELLS = 0x5C8800
OFF_DIRTY_ADDRESSES = 0x5C9810
OFF_CHECKSUMS = 0x5CB810
OFF_HEAD_SCRATCH = 0x5EBA10
OFF_CHART_SCRATCH = 0x62BA10
OFF_EXTENSION = 0x63BA10
REGISTER_MAGIC = b"NLMREG01"
REGISTER_BYTES = 4_096
HEAD_REGISTER_MAGIC = b"NLMHEAD1"
HEAD_REGISTER_OFFSET = OFF_EXTENSION + REGISTER_BYTES
HEAD_REGISTER_BYTES = 4_096
LEAF_MAGIC = b"NLMLEAF1"
RUNNER_ADDRESS = "nan://HW2:BS0:MC0:SC0:GM0"
NANNET_LINK = "HW2:0"
RING_ENDPOINT = "BS0:0::HW0:0"
BINARY_TRANSLATION = "NaN16DE"

# s+ at cell+0, s- at cell+8; NaN16 word payload is bits 16..47.
WORD_S_PLUS = 0
WORD_S_MINUS = 8
RAIL_NAMES = {0: "s+", 1: "s-"}

_CACHE: dict[tuple[str, int, int], "Chart8MiB"] = {}


def reverse8(value: int) -> int:
    return int(format(value & 0xFF, "08b")[::-1], 2)


def reverse16(value: int) -> int:
    return int(format(value & 0xFFFF, "016b")[::-1], 2)


def address_of(a: int, b: int) -> int:
    """Native pair address: addr(a,b) = (a << 8) | reverse8(b)."""
    return ((a & 0xFF) << 8) | reverse8(b)


def canonical_address(a: int, b: int) -> int:
    raw = address_of(a, b)
    return min(raw, reverse16(raw))


def make_word(payload: int, near: int = 0x7FF8, far: int = 0x7FF8) -> int:
    stored_far = (far >> 8) | ((far & 0xFF) << 8)
    return ((near & 0xFFFF) << 48) | ((payload & 0xFFFFFFFF) << 16) | (stored_far & 0xFFFF)


def word_payload(word: int) -> int:
    return (word >> 16) & 0xFFFFFFFF


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


class ChartError(ValueError):
    pass


class Chart8MiB:
    """In-memory 8 MiB chart assembled from a serialized local universe."""

    @classmethod
    def genesis(cls) -> "Chart8MiB":
        """Build an empty local chart before any training has run.

        Heads can map into the 8 MiB register file and the temporal register
        from plugin boot; a later trained chart replaces it in the same runner
        while preserving the register frame/state.
        """
        import hashlib

        genome = 0x4E414E4C4D494E44  # "NANLMIND"
        chart = cls.__new__(cls)
        chart.source = None
        chart.root_hash = hashlib.sha256(
            b"genesis:" + struct.pack("<Q", genome)).hexdigest()
        chart.bytes = bytearray(PAGE_BYTES)
        chart._dirty = True
        chart.snapshot_generation = 0
        chart._checkpoint_page_sha256 = None
        chart._parents = {}
        header = chart.bytes
        header[0:8] = MAGIC_CHART
        struct.pack_into("<I", header, 8, 1)
        struct.pack_into("<Q", header, 12, genome)
        struct.pack_into("<I", header, 20, CANONICAL_COUNT)
        struct.pack_into("<I", header, 24, ADDRESS_COUNT)
        struct.pack_into("<I", header, 28, MIRROR_PAIR_COUNT)
        struct.pack_into("<I", header, 32, HEAD_COUNT)
        struct.pack_into("<I", header, 36, PAGE_BYTES)
        fixed = [(a << 8) | reverse8(a) for a in range(256)]
        pairs = [canonical_address(a, b)
                 for a in range(256) for b in range(a + 1, 256)]
        chart.canon = fixed + pairs
        if len(chart.canon) != CANONICAL_COUNT:
            raise ChartError("genesis canonical count")
        index = [0xFFFF] * ADDRESS_COUNT
        for position, address in enumerate(chart.canon):
            struct.pack_into("<H", header, OFF_CANON + 2 * position, address)
            index[address] = position
        for address in range(ADDRESS_COUNT):
            if index[address] == 0xFFFF:
                index[address] = index[reverse16(address)]
            struct.pack_into("<H", header, OFF_INDEX + 2 * address, index[address])
        chart.address_to_index = index
        for pair_index, address in enumerate(pairs):
            canonical_index = 256 + pair_index
            struct.pack_into("<H", header, OFF_PAIR_TABLE + 2 * pair_index,
                             canonical_index)
        empty_word = make_word(0)
        for position, address in enumerate(chart.canon):
            cell = OFF_SHARED_REGS + position * CELL_BYTES
            for word in range(4):
                struct.pack_into("<Q", header, cell + 8 * word, empty_word)
            head = address >> 8
            mirrored = address & 0xFF
            header[cell + 45] = 1 if head == mirrored else 2  # head / pair
            header[cell + 40:cell + 42] = struct.pack("<H", head)
        for head in range(HEAD_COUNT):
            record = OFF_HEAD_TABLE + head * HEAD_BYTES
            struct.pack_into("<H", header, record,
                             (head << 8) | reverse8(head))
            header[record + 20] = 255
            struct.pack_into("<I", header, record + 24, genome & 0xFFFFFFFF)
        return chart

    def __init__(self, page: bytes, source: Path, root_hash: bytes):
        if len(page) != 6_264_872:
            raise ChartError("unexpected serialized universe size %d" % len(page))
        if page[:8] != MAGIC_UNIVERSE:
            raise ChartError("unexpected universe magic %r" % page[:8])
        self.source = source
        self.root_hash = root_hash.hex()
        self.bytes = bytearray(PAGE_BYTES)
        self._write_header(page)
        self._copy(page)
        self._build_index()
        self._parents: dict[int, int] = {}
        self._dirty = False
        self.snapshot_generation = 0
        self._checkpoint_page_sha256 = None

    # -- assembly -------------------------------------------------------
    def _write_header(self, page: bytes) -> None:
        genome = struct.unpack_from("<Q", page, 8)[0]
        counts = struct.unpack_from("<6I", page, 16)
        header = self.bytes
        header[0:8] = MAGIC_CHART
        struct.pack_into("<I", header, 8, 1)
        struct.pack_into("<Q", header, 12, genome)
        struct.pack_into("<I", header, 20, counts[0])
        struct.pack_into("<I", header, 24, counts[1])
        struct.pack_into("<I", header, 28, CANONICAL_COUNT - FIXED_POINT_COUNT)
        struct.pack_into("<I", header, 32, HEAD_COUNT)
        struct.pack_into("<I", header, 36, PAGE_BYTES)
        struct.pack_into("<I", header, 109, 0)
        struct.pack_into("<Q", header, 113, 0)

    def _copy(self, page: bytes) -> None:
        offset = 40
        sizes = (
            (OFF_SHARED_REGS, CANONICAL_COUNT * CELL_BYTES),
            (OFF_ADDRESS_REGS, ADDRESS_COUNT * ADDRESS_REG_BYTES),
            (OFF_DIRTY_ADDRESSES, ADDRESS_DIRTY_BYTES),
            (OFF_HEAD_SCRATCH, HEAD_COUNT * HEAD_SCRATCH_BYTES),
            (OFF_CHART_SCRATCH, CHART_SCRATCH_BYTES),
            (OFF_DESCRIPTORS, CANONICAL_COUNT * DESCRIPTOR_BYTES),
            (OFF_HEAD_TABLE, HEAD_COUNT * HEAD_BYTES),
            (OFF_ADJACENCY, ADJACENCY_BYTES),
            (OFF_CANON, CANONICAL_COUNT * 2),
            (OFF_PAIR_TABLE, MIRROR_PAIR_COUNT * 2),
        )
        for destination, size in sizes:
            chunk = page[offset:offset + size]
            if len(chunk) != size:
                raise ChartError("truncated universe page at offset %d" % offset)
            self.bytes[destination:destination + size] = chunk
            offset += size
        if offset != len(page):
            raise ChartError("serialized universe length mismatch")
        self.bytes[OFF_DIRTY_CELLS:OFF_DIRTY_CELLS + 4_112] = b"\x00" * 4_112

    def _build_index(self) -> None:
        self.canon: list[int] = [
            _u16(self.bytes, OFF_CANON + 2 * index) for index in range(CANONICAL_COUNT)
        ]
        self.address_to_index = [0xFFFF] * ADDRESS_COUNT
        for index, address in enumerate(self.canon):
            self.address_to_index[address] = index
        for address in range(ADDRESS_COUNT):
            if self.address_to_index[address] == 0xFFFF:
                self.address_to_index[address] = self.address_to_index[reverse16(address)]

    # -- cells ----------------------------------------------------------
    def cell_index(self, a: int, b: int) -> int:
        address = canonical_address(a, b)
        index = self.address_to_index[address]
        if index == 0xFFFF:
            raise ChartError("address %04x is not mapped" % address)
        return index

    def cell_offset(self, a: int, b: int) -> int:
        return OFF_SHARED_REGS + self.cell_index(a, b) * CELL_BYTES

    def read_rail(self, a: int, b: int, rail: int) -> int:
        offset = self.cell_offset(a, b) + (WORD_S_PLUS if rail == 0 else WORD_S_MINUS)
        raw = self.bytes[offset:offset + 8]
        return struct.unpack("<Q", raw)[0]

    def write_rail(self, a: int, b: int, rail: int, word: int) -> None:
        offset = self.cell_offset(a, b) + (WORD_S_PLUS if rail == 0 else WORD_S_MINUS)
        self.bytes[offset:offset + 8] = struct.pack("<Q", word & 0xFFFFFFFFFFFFFFFF)
        self._dirty = True

    def head_offset(self, head: int) -> int:
        return OFF_HEAD_TABLE + (head & 0xFF) * HEAD_BYTES

    def head_edge_count(self, head: int) -> int:
        return self.bytes[self.head_offset(head) + 20]

    def active_heads(self, limit: int = 8) -> list[int]:
        ranked = sorted(
            range(HEAD_COUNT),
            key=lambda head: (-self.head_edge_count(head), head),
        )
        return ranked[:limit]

    def verify(self) -> list[str]:
        problems = []
        if self.bytes[:8] != MAGIC_CHART:
            problems.append("chart magic")
        if struct.unpack_from("<I", self.bytes, 36)[0] != PAGE_BYTES:
            problems.append("page size")
        if struct.unpack_from("<I", self.bytes, 32)[0] != HEAD_COUNT:
            problems.append("head count")
        for address in range(ADDRESS_COUNT):
            if self.address_to_index[address] == 0xFFFF:
                problems.append("unmapped address %04x" % address)
                break
        return problems

    def stats(self) -> dict:
        return {
            "source": str(self.source) if self.source else None,
            "mode": "trained" if self.source is not None else "genesis",
            "trained": self.source is not None,
            "root_hash": self.root_hash,
            "page_bytes": PAGE_BYTES,
            "heads": HEAD_COUNT,
            "addresses": ADDRESS_COUNT,
            "cells": CANONICAL_COUNT,
            "dirty": self._dirty,
        }


def find_universe(state_dir: str | Path) -> Path | None:
    root = Path(state_dir).expanduser()
    candidates = sorted(
        root.glob("universe/native/global_universe.nmg"),
        key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(
            root.glob("universe/**/*.nmg"),
            key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_global(path: Path) -> tuple[bytes, bytes]:
    data = path.read_bytes()
    if data[:8] != MAGIC_GLOBAL:
        raise ChartError("not an NLMGLOB3 container: %s" % path)
    count = _u32(data, 8)
    offset = 12
    page = None
    for _ in range(count):
        offset += 5  # depth + 4 path bytes
        size = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        chunk = data[offset:offset + size]
        offset += size
        if page is None and chunk[:8] == MAGIC_UNIVERSE:
            page = chunk
    if page is None:
        raise ChartError("no universe page inside %s" % path)
    root_hash = data[offset:offset + 32] if len(data) - offset >= 32 else b"\x00" * 32
    return page, root_hash


def load_chart(state_dir: str | Path) -> Chart8MiB | None:
    path = find_universe(state_dir)
    if path is None:
        return None
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    page, root_hash = parse_global(path)
    chart = Chart8MiB(page, path, root_hash)
    _CACHE[key] = chart
    return chart


def restore_local_chart(chart: Chart8MiB, state_dir: str | Path) -> int:
    """Restore an unmerged local byte overlay only on its exact canonical root.

    Immutable address/canon/pair mapping must be identical to a freshly
    instantiated native layout. Never promote an overlay into the .nmg.
    """
    generation, payload = load_chart_snapshot(state_dir, chart.root_hash)
    if payload is not None:
        for start, size in ((0, 40),
                            (OFF_INDEX, ADDRESS_COUNT * 2),
                            (OFF_CANON, CANONICAL_COUNT * 2),
                            (OFF_PAIR_TABLE, MIRROR_PAIR_COUNT * 2)):
            if payload[start:start + size] != chart.bytes[start:start + size]:
                raise ChartError("snapshot changed immutable chart addressing")
        chart.bytes[:] = payload
        chart._dirty = False
    # Record the exact SHA-verified durable page loaded from storage, not a
    # later dirty in-memory page. Subsequent local checkpoints CAS against it.
    import hashlib
    chart._checkpoint_page_sha256 = (
        hashlib.sha256(payload).hexdigest() if payload is not None else None
    )
    chart.snapshot_generation = generation
    return generation


class TemporalRegister:
    """Temporal register bound to one pair cell's s+/s- rails."""

    def __init__(self, chart: Chart8MiB | None, state_dir: Path, pair: tuple[int, int],
                 reason: str = ""):
        self.chart = chart
        self.state_dir = state_dir
        self.pair = pair
        self.reason = reason
        self.mapped = chart is not None
        self.offset = OFF_EXTENSION
        self.frame = 0
        self.forward_payload = 0
        self.back_payload = 0
        self.state = [0] * 256
        self._load_sidecar()
        if self.mapped and self.chart.snapshot_generation:
            self._restore_extension()
        self._sync_extension()

    @classmethod
    def open(cls, ctx, chart: "Chart8MiB | None" = None) -> "TemporalRegister":
        state_dir = Path(str(ctx.get_config("state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
        if chart is None:
            try:
                chart = load_chart(state_dir)
            except Exception:
                chart = None
        reason = ("trained chart" if chart is not None
                  else "genesis chart (no trained universe yet)")
        if chart is None:
            chart = Chart8MiB.genesis()
        if chart.source is None:
            pair = (17, 136)
        else:
            active = chart.active_heads(8)
            pair = (active[0], active[1]) if len(active) >= 2 and active[0] != active[1] else (17, 136)
        register = cls(chart, state_dir, pair)
        register.reason = reason
        register._refresh_from_rail()
        return register

    # -- sidecar --------------------------------------------------------
    def _sidecar(self) -> Path:
        return self.state_dir / "language" / "temporal_register.json"

    def _load_sidecar(self) -> None:
        try:
            import json
            data = json.loads(self._sidecar().read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict) or data.get("pair") != list(self.pair):
            return
        if self.mapped and data.get("chart_root") not in {None, self.chart.root_hash}:
            return
        self.frame = int(data.get("frame", 0))
        self.forward_payload = int(data.get("forward_payload", 0))
        self.back_payload = int(data.get("back_payload", 0))
        state = data.get("state")
        if isinstance(state, list) and len(state) == 256:
            self.state = [int(value) & 0xFFFF for value in state]

    def _restore_extension(self) -> None:
        """Recovery uses the SHA-verified chart, not a potentially stale JSON."""
        data = self.chart.bytes
        region = self.offset
        if data[region:region + 8] != REGISTER_MAGIC:
            raise ChartError("committed snapshot lacks native temporal register")
        self.frame = struct.unpack_from("<Q", data, region + 8)[0]
        self.forward_payload = struct.unpack_from("<Q", data, region + 16)[0]
        self.back_payload = struct.unpack_from("<Q", data, region + 24)[0]
        self.state = list(struct.unpack_from("<256H", data, region + 32))

    def save(self) -> None:
        import json
        path = self._sidecar()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema": "nanlm.temporal-register/1",
            "chart_root": self.chart.root_hash if self.chart else None,
            "chart_source": str(self.chart.source) if self.chart else None,
            "pair": list(self.pair),
            "mapped": self.mapped,
            "frame": self.frame,
            "forward_payload": self.forward_payload,
            "back_payload": self.back_payload,
            "state": self.state,
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8")
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        temp.replace(path)

    # -- rail IO --------------------------------------------------------
    def rail_addresses(self) -> dict:
        a, b = self.pair
        canonical = canonical_address(a, b)
        mirror = reverse16(canonical)
        base = RUNNER_ADDRESS + "::pair/%d/%d" % self.pair
        return {
            "canonical": canonical,
            "canonical_hex": "%04X" % canonical,
            "mirror": mirror,
            "mirror_hex": "%04X" % mirror,
            "s_plus": base + "/s+",
            "s_minus": base + "/s-",
        }

    def _sync_extension(self) -> None:
        """Mirror the temporal register into the chart's mapped extension."""
        if not self.mapped:
            return
        region = self.offset
        self.chart.bytes[region:region + 8] = REGISTER_MAGIC
        struct.pack_into("<Q", self.chart.bytes, region + 8, self.frame & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<Q", self.chart.bytes, region + 16, self.forward_payload & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<Q", self.chart.bytes, region + 24, self.back_payload & 0xFFFFFFFFFFFFFFFF)
        state = b"".join(struct.pack("<H", int(value) & 0xFFFF) for value in self.state)
        self.chart.bytes[region + 32:region + 32 + 512] = state
        self.chart._dirty = True

    def _refresh_from_rail(self) -> None:
        if not self.mapped:
            return
        forward = self.chart.read_rail(*self.pair, 0)
        back = self.chart.read_rail(*self.pair, 1)
        if word_payload(forward):
            self.forward_payload = word_payload(forward)
        if word_payload(back):
            self.back_payload = word_payload(back)

    def _payload_for(self, head: int) -> int:
        return ((head & 0xFF) << 24) | (self.frame & 0xFFFFFF)

    def project(self, head: int) -> int:
        """Write the predicted next head on the forward rail (s+)."""
        self.forward_payload = self._payload_for(head)
        if self.mapped:
            self.chart.write_rail(*self.pair, 0, make_word(self.forward_payload))
            read_back = word_payload(self.chart.read_rail(*self.pair, 0))
            if read_back != self.forward_payload:
                raise ChartError("forward rail read-back mismatch")
        self._sync_extension()
        return self.forward_payload

    def commit(self, head: int, update_state: bool = True) -> int:
        """Commit one predicted word: s+ -> s-, advance the logical frame."""
        payload = self.project(head)
        self.back_payload = payload
        if self.mapped:
            self.chart.write_rail(*self.pair, 1, make_word(payload))
            read_back = word_payload(self.chart.read_rail(*self.pair, 1))
            if read_back != payload:
                raise ChartError("back rail read-back mismatch")
        self.frame += 1
        if update_state:
            self.state = [value * 0.97 for value in self.state]
            self.state[head & 0xFF] = min(0xFFFF, self.state[head & 0xFF] + 1)
            mirror = reverse8(head & 0xFF)
            self.state[mirror] = min(0xFFFF, self.state[mirror] + 1)
        self._sync_extension()
        return self.frame

    def rollback(self) -> None:
        """Restore the forward rail from the back rail without losing the past."""
        self.forward_payload = self.back_payload
        if self.mapped:
            self.chart.write_rail(*self.pair, 0, make_word(self.back_payload))
        self._sync_extension()

    def frame_meet(self, forward_start: int, forward_delta: int,
                   back_start: int, back_delta: int) -> dict:
        """Native FrameSync rule: opposite rails may meet at one target."""
        if forward_delta == 0 or back_delta == 0:
            return {"ok": False, "reason": "zero-length frame request"}
        if (forward_delta > 0) == (back_delta > 0):
            return {"ok": False, "reason": "both requests use the same rail"}
        if forward_start + forward_delta != back_start + back_delta:
            return {"ok": False, "reason": "no common frame"}
        return {"ok": True, "frame": forward_start + forward_delta,
                "reason": "forward and rollback meet"}

    def status(self) -> dict:
        rails = self.rail_addresses()
        forward = word_payload(self.chart.read_rail(*self.pair, 0)) if self.mapped else self.forward_payload
        back = word_payload(self.chart.read_rail(*self.pair, 1)) if self.mapped else self.back_payload
        return {
            "mapped": self.mapped,
            "reason": self.reason,
            "chart": self.chart.stats() if self.chart else None,
            "pair": list(self.pair),
            "rails": rails,
            "register_offset": self.offset,
            "frame": self.frame,
            "forward_payload": forward,
            "back_payload": back,
            "state_cells": 256,
            "state_nonzero": sum(1 for value in self.state if value),
        }


class HeadScratchRegister:
    """Special scratch register organizing relations by head and MTH leaf.

    The aggregate register lives in the chart extension next to the temporal
    register; each used head also receives a 1 KiB leaf record in its native
    head scratchpad. Both are held in the in-memory 8 MiB chart.
    """

    MAX_LEAVES = 100

    def __init__(self, chart: Chart8MiB | None, state_dir: Path):
        self.chart = chart
        self.state_dir = state_dir
        self.offset = HEAD_REGISTER_OFFSET
        self.leaves: list[dict] = []
        self.heads: dict[int, list[dict]] = {}
        self._load_sidecar()

    def _sidecar(self) -> Path:
        return self.state_dir / "language" / "mth_heads.json"

    def _load_sidecar(self) -> None:
        try:
            import json
            data = json.loads(self._sidecar().read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        leaves = data.get("leaves")
        if isinstance(leaves, list):
            self.leaves = [leaf for leaf in leaves if isinstance(leaf, dict)][-self.MAX_LEAVES:]
            self._rebuild_heads()

    def _rebuild_heads(self) -> None:
        self.heads = {}
        for leaf in self.leaves:
            self.heads.setdefault(int(leaf.get("head", 0)) & 0xFF, []).append(leaf)

    def organize(self, rows: list[dict], mapper=None) -> dict:
        if mapper is None:
            from .mth_layers import MTHHeadMapper, MTHLayerStack
            mapper = MTHHeadMapper(MTHLayerStack())
        leaves = []
        for row in rows:
            if row.get("kind") != "relation":
                continue
            leaves.append(mapper.map_relation(row))
        self.leaves = leaves[-self.MAX_LEAVES:]
        self._rebuild_heads()
        if self.chart is not None:
            self._write_chart()
        self._save()
        return self.status()

    def _write_chart(self) -> None:
        data = self.chart.bytes
        region = self.offset
        data[region:region + 8] = HEAD_REGISTER_MAGIC
        struct.pack_into("<I", data, region + 8, len(self.leaves))
        struct.pack_into("<I", data, region + 12, len(self.heads))
        table = region + 16
        start = 0
        for head in range(256):
            bucket = self.heads.get(head, [])
            mask = int(bucket[0]["mask"]) & 0x0F if bucket else 0
            struct.pack_into("<HHBB", data, table + head * 8, start,
                             len(bucket), mask, 0)
            start += len(bucket)
        leaf_base = table + 256 * 8
        for index, leaf in enumerate(self.leaves):
            struct.pack_into("<BBBBII", data, leaf_base + index * 16,
                             int(leaf.get("head", 0)) & 0xFF,
                             int(leaf.get("mirror", 0)) & 0xFF,
                             int(leaf.get("mask", 0)) & 0x0F,
                             0,
                             int(leaf.get("value", 0)) & 0xFFFFFFFF,
                             int(str(leaf.get("digest", "0")), 16) & 0xFFFFFFFF)
        for head, bucket in self.heads.items():
            scratch = OFF_HEAD_SCRATCH + (head & 0xFF) * HEAD_SCRATCH_BYTES
            leaf = bucket[0]
            data[scratch:scratch + 8] = LEAF_MAGIC
            struct.pack_into("<HBBII", data, scratch + 8, len(bucket),
                             int(leaf.get("mask", 0)) & 0x0F, 0,
                             int(leaf.get("value", 0)) & 0xFFFFFFFF,
                             int(str(leaf.get("digest", "0")), 16) & 0xFFFFFFFF)
        self.chart._dirty = True

    def _save(self) -> None:
        import json
        path = self._sidecar()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema": "nanlm.mth-heads/1",
            "chart_root": self.chart.root_hash if self.chart else None,
            "offset": self.offset,
            "leaves": self.leaves[-self.MAX_LEAVES:],
            "heads": {str(head): len(bucket) for head, bucket in self.heads.items()},
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8")
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        temp.replace(path)

    def status(self) -> dict:
        return {
            "mapped": self.chart is not None,
            "register_offset": self.offset,
            "magic": HEAD_REGISTER_MAGIC.decode("ascii"),
            "leaf_count": len(self.leaves),
            "head_count": len(self.heads),
            "head_scratchpads": len(self.heads),
            "masks": sorted({int(leaf.get("mask", 0)) for leaf in self.leaves}),
            "layers": sorted({str(leaf.get("layer", "")) for leaf in self.leaves}),
            "leaves": self.leaves[-8:],
        }


_RUNNERS: dict[str, "LocalRunner"] = {}


class LocalRunner:
    """Hermes' own local-CPU runner over the local universe registers.

    Loads the trained 8 MiB chart and the mapped temporal register into this
    process memory. It does not use the system RunnerRing for compute; it
    links through ``HW2:0`` (the NaN16DE binary translation interface) to the
    ``BS0:0::HW0:0`` runner-ring NaNNet interface, and shares the chart,
    frame-sync and dual-rail code/format with the native stack.

    A genesis chart is instantiated at plugin boot so heads can map into
    registers before any Boolean/local training. When a trained chart appears,
    the cached runner rebinds to it and keeps the register frame/state.
    """

    ADDRESS = RUNNER_ADDRESS
    LINK = NANNET_LINK
    RING = RING_ENDPOINT
    BINARY_TRANSLATION = BINARY_TRANSLATION

    def __init__(self, chart: Chart8MiB | None, register: TemporalRegister):
        self.chart = chart
        self.register = register
        # Serialize the live chart's native bank rebase and ordinary writers.
        # Durable N16SNAP1 CAS separately fences other processes.
        self._write_lock = threading.RLock()
        self.heads = HeadScratchRegister(chart, register.state_dir)
        self.local_memory_bytes = PAGE_BYTES + (2 * REGISTER_BYTES if chart else 0)

    @classmethod
    def open(cls, ctx) -> "LocalRunner":
        state_dir = Path(str(ctx.get_config("state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
        key = str(state_dir)
        runner = _RUNNERS.get(key)
        latest = None
        try:
            latest = find_universe(state_dir)
        except Exception:
            latest = None
        if runner is None:
            chart = load_chart(state_dir) or Chart8MiB.genesis()
            restore_local_chart(chart, state_dir)
            from .universe_root import attach_universe_root
            attach_universe_root(chart)
            register = TemporalRegister.open(ctx, chart)
            runner = cls(chart, register)
            _RUNNERS[key] = runner
        elif latest is not None:
            chart = load_chart(state_dir)
            if chart is not None and runner.chart is not chart \
                    and (runner.chart is None or runner.chart.root_hash != chart.root_hash):
                restore_local_chart(chart, state_dir)
                from .universe_root import attach_universe_root
                attach_universe_root(chart)
                runner.chart = chart
                runner.register.chart = chart
                runner.register.reason = "trained chart"
                runner.register._refresh_from_rail()
                runner.register._sync_extension()
                runner.heads.chart = chart
                runner.heads._write_chart()
                runner.local_memory_bytes = PAGE_BYTES + 2 * REGISTER_BYTES
        return runner

    def commit_native_head_observation(self, *, coordinator, request,
                                       lease_receipt, network_head: int) -> dict:
        """Serialize authenticated native rebase against local chart writers."""
        with self._write_lock:
            return self._commit_native_head_observation_locked(
                coordinator=coordinator, request=request,
                lease_receipt=lease_receipt, network_head=network_head,
            )

    def _commit_native_head_observation_locked(self, *, coordinator, request,
                                               lease_receipt, network_head: int) -> dict:
        """Rebase one verified native NaN16 bank into THIS existing chart.

        The router journal and HERMES N16SNAP1 writer must already be
        deployment-owned, never assembled from an untrusted plugin request.
        Preserve any pending temporal/MP/MTH edits in memory while replacing
        ONLY the native bank's reserved byte region with its durable output.
        This method does not allocate another chart or train a second model.
        The runner serializes its ordinary in-process writers during handoff.
        """
        import hashlib
        from .head_bank_transport import BANK_OFFSET, BANK_END, BANK_BYTES
        from .native_head_checkpoint import (
            NativeHeadCheckpointCoordinator,
        )
        if type(coordinator) is not NativeHeadCheckpointCoordinator:
            raise ChartError("trusted native head checkpoint coordinator required")
        if self.chart is None:
            raise ChartError("local runner has no binary universe")
        if (self.register.chart is not self.chart
                or self.heads.chart is not self.chart):
            raise ChartError("local runner heads and temporal rails are not bound")
        if self.register.state_dir.resolve() != coordinator.fence._state_dir.resolve():
            raise ChartError("native checkpoint belongs to another state directory")
        if (type(request.chart_generation) is not int
                or self.chart.root_hash != request.parent_root_sha256
                or self.chart.snapshot_generation not in (
                    request.chart_generation, request.chart_generation + 1
                )):
            raise ChartError("local runner checkpoint parent/generation mismatch")
        current = bytes(self.chart.bytes[BANK_OFFSET:BANK_END])
        if self.chart.snapshot_generation == request.chart_generation:
            if hashlib.sha256(current).hexdigest() != request.bank_before_sha256:
                raise ChartError("native head bank changed in cached local chart")
        before_dirty = bool(self.chart._dirty)
        # Only the durable router journal supplies the native result. The
        # commit writes its own disk snapshot before we touch cached bytes.
        committed = coordinator.commit(
            request=request, lease_receipt=lease_receipt,
            network_head=network_head,
        )
        next_generation = request.chart_generation + 1
        if (committed.get("local_checkpoint_verified") is not True
                or committed.get("chart_generation") != next_generation
                or committed.get("canonical_training_committed") is not False):
            raise ChartError("native checkpoint evidence is not locally verified")
        generation, page = load_chart_snapshot(
            self.register.state_dir, self.chart.root_hash,
        )
        if generation != next_generation or page is None:
            raise ChartError("native checkpoint was superseded during local rebase")
        new_bank = bytes(page[BANK_OFFSET:BANK_END])
        if (len(new_bank) != BANK_BYTES
                or hashlib.sha256(new_bank).hexdigest() != committed.get(
                    "bank_after_sha256"
                )):
            raise ChartError("native checkpoint and retrieved bank disagree")
        if self.chart.snapshot_generation == next_generation:
            # Repeated dispatch: don't mutate the live object again.
            if hashlib.sha256(current).hexdigest() != committed["bank_after_sha256"]:
                raise ChartError("replay would replace a different cached native bank")
            rebased = False
        else:
            # Preserve live edits to EVERY OTHER address, mirror pair, and
            # temporal register. Do not replace the whole cached chart
            # with the older disk page or lose uncheckpointed local work.
            if hashlib.sha256(
                self.chart.bytes[BANK_OFFSET:BANK_END]
            ).hexdigest() != request.bank_before_sha256:
                raise ChartError("native bank changed while checkpoint was committed")
            self.chart.bytes[BANK_OFFSET:BANK_END] = new_bank
            self.chart.snapshot_generation = generation
            rebased = True
        # If temporal/semantic changes were already dirty, they remain so;
        # the next ordinary checkpoint must commit them using the NEW
        # expected generation. Do not falsely mark everything clean.
        # Native CAS has already published the exact verified page. Preserve
        # its full-page digest as the baseline for a later ordinary HERMES
        # checkpoint that will merge unrelated pending temporal/MP edits.
        self.chart._checkpoint_page_sha256 = hashlib.sha256(page).hexdigest()
        self.chart._dirty = before_dirty
        return {
            "schema": "nanlm.local-runner-native-head-rebase/1",
            "head_bank_rebased": rebased,
            "single_chart_object_preserved": True,
            "chart_generation": self.chart.snapshot_generation,
            "native_bank_sha256": committed["bank_after_sha256"],
            "pending_nonbank_edits_preserved": before_dirty,
            "local_checkpoint_verified": True,
            "canonical_training_committed": False,
            "nannet_merge_issued": False,
        }

    def checkpoint(self) -> dict:
        """Serialize ordinary full-page CAS against native in-process rebase."""
        with self._write_lock:
            return self._checkpoint_locked()

    def _checkpoint_locked(self) -> dict:
        """Durably store local chart bytes with full-page source CAS."""
        import hashlib
        if self.chart is None:
            raise ChartError("no chart to checkpoint")
        generation_before = self.chart.snapshot_generation
        source_sha = getattr(self.chart, "_checkpoint_page_sha256", None)
        if generation_before and source_sha is None:
            raise ChartError("verified durable chart source digest missing")
        # The cached chart can contain pending temporal/MP edits. Its bytes
        # are NOT the expected old page; compare the last verified durable
        # chart digest under the N16SNAP1 exclusive writer lock.
        data = bytes(self.chart.bytes)
        generation = save_chart_snapshot(
            self.register.state_dir, self.chart.root_hash, data,
            generation_before, expected_page_sha256=source_sha,
        )
        self.chart.snapshot_generation = generation
        self.chart._checkpoint_page_sha256 = hashlib.sha256(data).hexdigest()
        self.chart._dirty = False
        return {"kind": "unmerged-local-chart", "generation": generation,
                "parent_root": self.chart.root_hash, "bytes": PAGE_BYTES,
                "canonical_training_committed": False}

    def commit(self, head: int) -> int:
        with self._write_lock:
            return self.register.commit(head)

    def plan(self, heads: list[int]) -> list[int]:
        with self._write_lock:
            return [self.register.commit(head) for head in heads]

    def organize(self, rows: list[dict], mapper=None) -> dict:
        with self._write_lock:
            return self.heads.organize(rows, mapper)

    def status(self) -> dict:
        return {
            "runner": "hermes-local-cpu",
            "address": self.ADDRESS,
            "nannet_link": self.LINK,
            "ring_endpoint": self.RING,
            "binary_translation": self.BINARY_TRANSLATION,
            "uses_system_ring_for_compute": False,
            "shares_code_with": [
                "nanlm.chart-layout/1",
                "nanlm.frame-sync/1",
                "nanlm.dual-rails/1",
                "nan16de",
            ],
            "local_memory_bytes": self.local_memory_bytes,
            "chart_bytes": PAGE_BYTES if self.chart else 0,
            "register_bytes": REGISTER_BYTES if self.chart else 0,
            "chart": self.chart.stats() if self.chart else None,
            "checkpoint_generation": self.chart.snapshot_generation if self.chart else 0,
            "checkpoint_kind": "unmerged-local-chart",
            "universe_root": __import__(
                "nanlmind_hermes.universe_root", fromlist=["read_universe_root"]
            ).read_universe_root(self.chart),
            "register": self.register.status(),
            "head_register": self.heads.status(),
        }
