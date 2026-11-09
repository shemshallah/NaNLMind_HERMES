"""Unified memory-mapped resident universe primitives.

This package is a bounded, phone-friendly unified store. The implementation is
split into small modules so the superblock, cells, leaves, relations, resonance
query, ingestion, export/intake, persistence and CLI can be developed and
tested independently.

Hard constraints:
- one sparse memory-mapped file per universe;
- default budget 256 MiB, hard ceiling 1 GiB;
- append-only writes with a monotonic generation;
- no 2 GiB overbootstrap copy on Android;
- queries may use resonance/superposition, not only key lookup;
- no canonical merge is performed by this package.
"""
from __future__ import annotations

MAGIC = b"NLMRES01"
RESIDENT_VERSION = 1
DEFAULT_BYTES = 256 * 1024 * 1024
MAX_BYTES = 1024 * 1024 * 1024
MIN_BYTES = 16 * 1024 * 1024
ALIGN = 64
HEAD_COUNT = 256
PAIR_COUNT = 256 * 255 // 2  # C(256,2) = 32,640
RAILS = 2

SECTION_HEADER = 1
SECTION_CHART = 2
SECTION_HEADS = 3
SECTION_PAIRS = 4
SECTION_LEAVES = 5
SECTION_RELATIONS = 6
SECTION_HYPEREDGES = 7
SECTION_FEATURES = 8
SECTION_TOKENS = 9
SECTION_INDEX = 10
SECTION_ARENA = 11
SECTION_JOURNAL = 12


def align(value: int, alignment: int = ALIGN) -> int:
    """Round ``value`` up to the requested alignment."""
    if not isinstance(value, int) or value < 0:
        raise ValueError("value must be a non-negative integer")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def clamp_size(size: int) -> int:
    """Validate and clamp a requested resident file size to the hard ceiling."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("size must be an integer")
    if size < MIN_BYTES:
        raise ValueError("resident universe must be at least %d bytes" % MIN_BYTES)
    return min(size, MAX_BYTES)


__all__ = (
    "MAGIC", "RESIDENT_VERSION", "DEFAULT_BYTES", "MAX_BYTES", "MIN_BYTES",
    "ALIGN", "HEAD_COUNT", "PAIR_COUNT", "RAILS", "SECTION_HEADER",
    "SECTION_CHART", "SECTION_HEADS", "SECTION_PAIRS", "SECTION_LEAVES",
    "SECTION_RELATIONS", "SECTION_HYPEREDGES", "SECTION_FEATURES",
    "SECTION_TOKENS", "SECTION_INDEX", "SECTION_ARENA", "SECTION_JOURNAL",
    "align", "clamp_size", "ResidentUniverse", "ResidentUniverseError",
    "ResidentUniverseFull", "open_universe", "ResidentHeadField",
    "ScratchLayers", "SelfModificationGate", "FieldLearner",
    "FieldPersistence", "MTHHeadBridge", "CycleField", "NativeCycleRoot",
    "Round030Snapshot", "FullCycleMemory", "FullCycleTrainer",
    "EpisodicGate", "CompositionMemory",
)


def __getattr__(name):
    """Lazy-load the integration layer without making primitives circular."""
    if name == "ResidentUniverse":
        from .universe import ResidentUniverse
        return ResidentUniverse
    if name == "ResidentUniverseError":
        from .universe import ResidentUniverseError
        return ResidentUniverseError
    if name == "ResidentUniverseFull":
        from .universe import ResidentUniverseFull
        return ResidentUniverseFull
    if name == "open_universe":
        from .universe import open_universe
        return open_universe
    if name == "ResidentHeadField":
        from .head_field import ResidentHeadField
        return ResidentHeadField
    if name == "ScratchLayers":
        from .scratch_layers import ScratchLayers
        return ScratchLayers
    if name == "SelfModificationGate":
        from .selfmod_gate import SelfModificationGate
        return SelfModificationGate
    if name == "FieldLearner":
        from .field_learning import FieldLearner
        return FieldLearner
    if name == "FieldPersistence":
        from .field_persistence import FieldPersistence
        return FieldPersistence
    if name == "MTHHeadBridge":
        from .mth_head_bridge import MTHHeadBridge
        return MTHHeadBridge
    if name == "CycleField":
        from .cycle_observable import CycleField
        return CycleField
    if name == "NativeCycleRoot":
        from .cycle_native import NativeCycleRoot
        return NativeCycleRoot
    if name == "Round030Snapshot":
        from .cycle_snapshot import Snapshot as Round030Snapshot
        return Round030Snapshot
    if name == "FullCycleMemory":
        from .full_cycle_memory import FullCycleMemory
        return FullCycleMemory
    if name == "FullCycleTrainer":
        from .full_cycle_training import FullCycleTrainer
        return FullCycleTrainer
    if name == "EpisodicGate":
        from .episodic_gate import LogisticGate as EpisodicGate
        return EpisodicGate
    if name == "CompositionMemory":
        from .composition_gate import CompositionMemory
        return CompositionMemory
    if name == "EvidenceLanes":
        from .evidence_lanes import EvidenceLanes
        return EvidenceLanes
    if name == "ConventionalEvidence":
        from .evidence_lanes import ConventionalEvidence
        return ConventionalEvidence
    if name == "EvidenceSession":
        from .evidence_session import EvidenceSession
        return EvidenceSession
    raise AttributeError(name)
