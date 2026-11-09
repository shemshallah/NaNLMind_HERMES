"""Durable student state versus process-lifetime runner/runtime state.

The tests drive the real native chat/learning path in an isolated ``state_dir``:
they create a valid local native universe, teach one fact, record the durable
files and their digests, drop every process cache, create a new context, and
reload.  No Gemma/Ollama provider is configured.  Native mode must answer from
the reloaded local student state alone.

Run with::

    python3 -m unittest tests.test_student_state_persistence -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanlmind_hermes.chat import chat, chat_history
from nanlmind_hermes.layered_learning import (
    coordinator_for_context,
    reset_coordinator_cache,
)
from nanlmind_hermes.memory import MemoryStore
from nanlmind_hermes.temporal_lm import TemporalMirrorLM
from nanlmind_hermes.universe_chart import (
    ADDRESS_COUNT,
    CANONICAL_COUNT,
    HEAD_COUNT,
    MIRROR_PAIR_COUNT,
    _CACHE,
    _RUNNERS,
    canonical_address,
    make_word,
    reverse8,
)

SESSION_ID = "student-state"
TEACHING_FACT = "Dogs bark."
QUESTION = "What do dogs do?"
STRUCTURED_ANSWER = "Dogs can bark."


class StudentContext:
    """Minimal plugin context for the native chat/learning path."""

    def __init__(self, root: Path, llm=None):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "native",
            "chat_learning": True,
            "merge_enabled": False,
            "learning_require_approval": True,
            "learning_auto_sync": False,
        }
        if llm is not None:
            self.llm = llm

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class PoisonLLM:
    """Fail if native reload accidentally reaches a model provider."""

    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError(
            "native student reload attempted provider/Gemma/Ollama access")


def build_native_universe(pair=(17, 136), s_plus=11, s_minus=22) -> bytes:
    """Build a minimal valid NLMGLOB3/NLMU0001 native universe.

    The layout mirrors the real serialized native universe: 32,896 canonical
    cells, 65,536 address registers, 256 head records and one bound pair cell
    with real forward/back rail words.
    """
    fixed = [((a << 8) | reverse8(a)) for a in range(256)]
    seen = set(fixed)
    pairs = []
    for a in range(256):
        for b in range(a + 1, 256):
            canonical = canonical_address(a, b)
            if canonical in seen:
                continue
            seen.add(canonical)
            pairs.append(canonical)
    canon = fixed + pairs
    assert len(canon) == CANONICAL_COUNT
    index = {address: position for position, address in enumerate(canon)}

    cells = bytearray(CANONICAL_COUNT * 64)
    cell = index[canonical_address(*pair)] * 64
    struct.pack_into("<Q", cells, cell, make_word(s_plus))
    struct.pack_into("<Q", cells, cell + 8, make_word(s_minus))

    heads = bytearray(HEAD_COUNT * 32)
    for head in pair:
        heads[head * 32 + 20] = 255

    page = b"NLMU0001"
    page += struct.pack("<Q", 42)
    page += struct.pack("<6I", CANONICAL_COUNT, ADDRESS_COUNT,
                        CANONICAL_COUNT, HEAD_COUNT, CANONICAL_COUNT,
                        MIRROR_PAIR_COUNT)
    page += bytes(cells)
    page += b"\x00" * (ADDRESS_COUNT * 32)
    page += b"\x00" * 8_192
    page += b"\x00" * (HEAD_COUNT * 1_024)
    page += b"\x00" * 65_536
    page += b"\x00" * (CANONICAL_COUNT * 48)
    page += bytes(heads)
    page += b"\x00" * 8_192
    page += b"".join(struct.pack("<H", address) for address in canon)
    page += b"\x00" * (MIRROR_PAIR_COUNT * 2)
    assert len(page) == 6_264_872

    container = b"NLMGLOB3" + struct.pack("<I", 1)
    container += bytes([0, 0, 0, 0, 0])
    container += struct.pack("<Q", len(page)) + page
    container += b"R" * 32
    return container


def write_native_universe(root: Path) -> Path:
    path = root / "universe" / "native" / "global_universe.nmg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_native_universe())
    return path


def reset_process_state() -> None:
    """Drop every process-lifetime cache, as a process restart would."""
    from nanlmind_hermes import session_context

    reset_coordinator_cache()
    session_context.reset_cache()
    _RUNNERS.clear()
    _CACHE.clear()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _database_logical_digest(root: Path) -> str | None:
    database = root / "universe" / "mathematical_memory.sqlite3"
    if not database.is_file():
        return None
    connection = sqlite3.connect(str(database), timeout=10)
    try:
        dump = "\n".join(connection.iterdump())
    finally:
        connection.close()
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def learner_state_fingerprint(root: Path) -> tuple[str, dict]:
    """Fingerprint every durable learner file, excluding transient sidecars.

    Excluded: lock files, SQLite WAL/SHM journals and temporary atomic-write
    files.  Those are coordination/journal artifacts, not learner evidence.
    """
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if path.name.endswith(".lock") or relative.endswith(("-wal", "-shm")):
            continue
        if relative == "universe/mathematical_memory.sqlite3":
            entries[relative] = {
                "logical_dump_sha256": _database_logical_digest(root),
            }
        else:
            entries[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), entries


def relation_rows(store: MemoryStore) -> list[dict]:
    return [row for row in store.query(limit=1000)
            if row.get("kind") == "relation"]


class StudentStatePersistenceTests(unittest.TestCase):
    def setUp(self):
        reset_process_state()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(reset_process_state)
        self.root = Path(self.temporary.name) / "state"

    def _teach_native_student(self):
        self.root.mkdir(parents=True, exist_ok=True)
        native_path = write_native_universe(self.root)
        ctx = StudentContext(self.root)
        result = chat(ctx, TEACHING_FACT, session_id=SESSION_ID, probe=False)
        self.assertIs(result.get("probe"), False)
        self.assertIs(result.get("state_mutated"), True)
        self.assertTrue(result.get("response"))
        self.assertTrue(native_path.is_file())
        return ctx, result, native_path

    def test_restart_reloads_vocabulary_relations_layers_and_chart(self):
        ctx, _, native_path = self._teach_native_student()
        native_digest = sha256_file(native_path)

        # A native training turn writes the temporal model, chart checkpoint,
        # register sidecar, head-register sidecar, layered state, memory ledger,
        # scratch pages and transcript.
        for relative in (
            "language/temporal_mirror.json",
            "language/temporal_register.json",
            "language/mth_heads.json",
            "language/layered_learning.json",
            "universe/mathematical_memory.sqlite3",
            "conversations/%s.jsonl" % SESSION_ID,
            "clocked_scratch/chat-%s.jsonl" % SESSION_ID,
        ):
            with self.subTest(path=relative):
                self.assertTrue((self.root / relative).is_file(), relative)
        self.assertTrue(list((self.root / "language" / "chart-checkpoints")
                             .glob("chart.*.n16s")))
        self.assertTrue(list((self.root / "scratch_register" / "pages")
                             .glob("page-*.n16")))

        model = TemporalMirrorLM.load(ctx)
        self.assertIn("dogs", model.vocab)
        self.assertIn("bark", model.vocab)
        self.assertGreater(model.uni.get("dogs", 0), 0)
        self.assertTrue(model.seen)
        self.assertTrue(model.runner.chart.stats()["trained"])
        self.assertTrue(model.register.mapped)
        frame = model.register.frame
        state_16bit = [int(value) & 0xFFFF for value in model.register.state]
        generation = model.runner.chart.snapshot_generation
        self.assertGreater(generation, 0)
        self.assertTrue(any(state_16bit))
        runner = model.runner
        self.assertGreaterEqual(runner.heads.status()["leaf_count"], 1)

        store = MemoryStore(
            self.root / "universe" / "mathematical_memory.sqlite3")
        learned = relation_rows(store)
        self.assertTrue(any(
            str(row.get("subject", "")).casefold() == "dogs"
            and str(row.get("predicate", "")).casefold() == "can"
            and str(row.get("object", "")).casefold() == "bark"
            for row in learned), learned)

        coordinator = coordinator_for_context(ctx)
        self.assertEqual(coordinator.turns, 1)
        self.assertTrue(any(
            row.get("subject") == "dogs"
            and row.get("predicate") == "can"
            and row.get("object") == "bark"
            for row in coordinator.rows), coordinator.rows)
        self.assertGreaterEqual(coordinator.space.stats()["records"], 10)
        self.assertTrue(coordinator.space.verify()["ok"])

        transcript = chat_history(ctx, SESSION_ID, limit=10)
        self.assertEqual([row["role"] for row in transcript["messages"]],
                         ["user", "assistant"])

        before_restart, before_entries = learner_state_fingerprint(self.root)
        reset_process_state()

        # A brand-new context sees only the durable files and the native .nmg.
        fresh_ctx = StudentContext(self.root)
        reloaded = TemporalMirrorLM.load(fresh_ctx)
        self.assertIn("dogs", reloaded.vocab)
        self.assertIn("bark", reloaded.vocab)
        self.assertGreater(reloaded.uni.get("bark", 0), 0)
        self.assertTrue(reloaded.seen)
        self.assertTrue(reloaded.runner.chart.stats()["trained"])
        self.assertEqual(reloaded.runner.chart.snapshot_generation, generation)
        self.assertEqual(reloaded.register.frame, frame)
        self.assertEqual(
            reloaded.register.state,
            [int(value) & 0xFFFF for value in state_16bit])
        self.assertGreaterEqual(
            reloaded.runner.heads.status()["leaf_count"], 1)

        reloaded_store = MemoryStore(
            self.root / "universe" / "mathematical_memory.sqlite3")
        self.assertTrue(any(
            str(row.get("subject", "")).casefold() == "dogs"
            and str(row.get("predicate", "")).casefold() == "can"
            and str(row.get("object", "")).casefold() == "bark"
            for row in relation_rows(reloaded_store)))
        self.assertTrue(reloaded_store.status()["append_only"])
        self.assertFalse(reloaded_store.status()["weights_checkpoint"])

        reloaded_coordinator = coordinator_for_context(fresh_ctx)
        self.assertEqual(reloaded_coordinator.turns, 1)
        self.assertTrue(any(
            row.get("subject") == "dogs"
            and row.get("predicate") == "can"
            and row.get("object") == "bark"
            for row in reloaded_coordinator.rows))
        self.assertGreaterEqual(
            reloaded_coordinator.space.stats()["records"], 10)
        self.assertTrue(reloaded_coordinator.space.verify()["ok"])

        # Reopening is read-only: no durable file changed, and the canonical
        # native .nmg was never rewritten by the local chart checkpoint.
        after_restart, after_entries = learner_state_fingerprint(self.root)
        changed = sorted(
            key for key in set(before_entries) | set(after_entries)
            if before_entries.get(key) != after_entries.get(key))
        self.assertEqual(after_restart, before_restart,
                         "reload changed durable files: %r" % (changed,))
        self.assertEqual(sha256_file(native_path), native_digest)

    def test_probe_only_turns_do_not_persist_or_advance_register(self):
        ctx, _, _ = self._teach_native_student()
        model = TemporalMirrorLM.load(ctx)
        before_digest, before_entries = learner_state_fingerprint(self.root)
        frame = model.register.frame
        state_before = [int(value) & 0xFFFF for value in model.register.state]

        responses = []
        for _ in range(3):
            result = chat(ctx, QUESTION, session_id=SESSION_ID, probe=True)
            self.assertIs(result.get("probe"), True)
            self.assertIs(result.get("state_mutated"), False)
            self.assertEqual(result.get("layered_learning"),
                             {"status": "probe_read_only"})
            synthesis = result.get("native_synthesis") or {}
            self.assertIs(synthesis.get("probe"), True)
            self.assertIs(synthesis.get("state_mutated"), False)
            self.assertEqual(synthesis.get("register_frames"), 0)
            self.assertTrue(result.get("response"))
            responses.append(result["response"])

        after_digest, after_entries = learner_state_fingerprint(self.root)
        self.assertEqual(after_digest, before_digest,
                         "a read-only probe changed durable learner state: "
                         "before=%r after=%r" % (before_entries, after_entries))
        self.assertEqual(sorted(after_entries), sorted(before_entries))
        self.assertEqual(len(set(responses)), 1,
                         "repeated probes were not deterministic: %r" % responses)
        self.assertEqual(model.register.frame, frame,
                         "a read-only probe advanced the in-process register")
        self.assertEqual(
            [int(value) & 0xFFFF for value in model.register.state],
            state_before,
            "a read-only probe changed the in-process temporal state")

    def test_reopened_state_answers_without_gemma_or_ollama(self):
        ctx, _, native_path = self._teach_native_student()
        first = chat(ctx, QUESTION, session_id=SESSION_ID, probe=True)
        self.assertIs(first.get("probe"), True)
        self.assertIs(first.get("state_mutated"), False)
        self.assertTrue(first["native_synthesis"]["structured"])
        self.assertEqual(first["response"], STRUCTURED_ANSWER)
        self.assertEqual(first.get("provider"), "nanlm-native")

        reset_process_state()
        poison = PoisonLLM()
        fresh_ctx = StudentContext(self.root, llm=poison)
        clean_environment = {
            key: value for key, value in os.environ.items()
            if "OLLAMA" not in key.upper() and "GEMMA" not in key.upper()
        }
        with mock.patch.dict(os.environ, clean_environment, clear=True):
            second = chat(fresh_ctx, QUESTION, session_id=SESSION_ID, probe=True)

        self.assertEqual(poison.calls, [],
                         "native student reload called the provider")
        self.assertEqual(second.get("provider"), "nanlm-native")
        self.assertIs(second.get("probe"), True)
        self.assertIs(second.get("state_mutated"), False)
        self.assertTrue(second["native_synthesis"]["structured"])
        self.assertEqual(second["response"], first["response"])
        self.assertEqual(second["response"], STRUCTURED_ANSWER)

        names = [path.name.casefold() for path in self.root.rglob("*")
                 if path.is_file()]
        self.assertFalse(
            [name for name in names
             if "gemma" in name or "ollama" in name],
            "student state unexpectedly contains teacher-provider files")
        self.assertTrue(native_path.is_file())


if __name__ == "__main__":
    unittest.main()
