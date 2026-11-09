"""Read-only evaluation probes must not mutate persisted learner state.

The tests drive the real plugin registration and the ``nanlm_chat`` handler in
native mode over an isolated state directory.  They need no NANNET authority,
language provider or WordNet archive.

The strict fingerprint deliberately covers only durable learner-facing files:
the mathematical-memory main database, the two language state files, the
paged scratch-register pages and conversation transcripts.  SQLite ``-wal``
and ``-shm`` sidecars are reported separately but excluded from the strict
fingerprint because they are transient journal/index files, not learner
evidence.  See ``docs/PROBE_IMMUTABILITY.md`` for the exact scope and the
remaining in-process register caveat.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import register
from nanlmind_hermes.layered_learning import reset_coordinator_cache


SESSION_ID = "probe-immutability"
PROBE_REPEATS = 3

TEACHING_FACTS = (
    "Dogs bark.",
    "Cats meow.",
    "Red and blue make purple.",
    "Yellow and blue make green.",
    "Water flows.",
)


class PluginContext:
    """Minimal Hermes/OpenCode-style context used by the plugin register hook."""

    def __init__(self, root: Path):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "native",
            "chat_learning": True,
            "merge_enabled": False,
            "learning_require_approval": True,
            "learning_auto_sync": False,
        }
        self.tools: dict[str, object] = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


def _call(ctx: PluginContext, name: str, params: dict) -> dict:
    raw = ctx.tools[name](params)
    return json.loads(raw) if isinstance(raw, str) else raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conversation_files(root: Path) -> list[Path]:
    directory = root / "conversations"
    return sorted(directory.glob("*.jsonl")) if directory.is_dir() else []


def _scratch_page_files(root: Path) -> list[Path]:
    directory = root / "scratch_register"
    return sorted(directory.rglob("page-*.n16")) if directory.is_dir() else []


def learner_state_fingerprint(root: Path) -> tuple[str, dict]:
    """Return a deterministic SHA-256 over the durable learner state.

    Fixed files are included even when they are absent (``None``) so that a
    probe which creates one of them cannot hide behind an empty directory.
    """
    entries: dict[str, object] = {}
    for relative in (
        "universe/mathematical_memory.sqlite3",
        "language/layered_learning.json",
        "language/temporal_mirror.json",
    ):
        path = root / relative
        entries[relative] = None if not path.is_file() else {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    entries["scratch_register/pages"] = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _scratch_page_files(root)
    ]
    entries["conversations"] = [
        {
            "path": str(path.relative_to(root)),
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _conversation_files(root)
    ]

    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), entries


def learner_counts(root: Path) -> dict[str, int]:
    """Count semantic learner items independently of the byte fingerprint."""
    layered_path = root / "language" / "layered_learning.json"
    layered: dict = {}
    if layered_path.is_file():
        try:
            value = json.loads(layered_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                layered = value
        except (OSError, ValueError):
            layered = {}

    database = root / "universe" / "mathematical_memory.sqlite3"
    memory_events = 0
    if database.is_file():
        connection = sqlite3.connect(
            "file:" + str(database) + "?mode=ro", uri=True, timeout=10)
        try:
            row = connection.execute("SELECT COUNT(*) FROM events").fetchone()
            memory_events = int(row[0]) if row is not None else 0
        finally:
            connection.close()

    pages = _scratch_page_files(root)
    return {
        "conversation_lines": sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in _conversation_files(root)),
        "layered_turns": int(layered.get("turns", 0) or 0),
        "memory_events": memory_events,
        "scratch_pages": len(pages),
        "scratch_bytes": sum(path.stat().st_size for path in pages),
    }


def _sidecar_report(root: Path) -> dict[str, dict | None]:
    """Report WAL/SHM presence without adding them to the strict fingerprint."""
    database = root / "universe" / "mathematical_memory.sqlite3"
    report = {}
    for suffix in ("-wal", "-shm"):
        path = Path(str(database) + suffix)
        report[suffix] = None if not path.is_file() else {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return report


def _snapshot(root: Path) -> dict:
    fingerprint, entries = learner_state_fingerprint(root)
    return {
        "fingerprint": fingerprint,
        "entries": entries,
        "counts": learner_counts(root),
        "sidecars": _sidecar_report(root),
    }


class ProbeImmutabilityTests(unittest.TestCase):
    def setUp(self):
        reset_coordinator_cache()
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name) / "state"
        self.ctx = PluginContext(self.root)
        register(self.ctx)
        if "nanlm_chat" not in self.ctx.tools:
            self.skipTest("nanlm_chat is not registered by this build")

    def tearDown(self):
        reset_coordinator_cache()

    def _teach(self) -> None:
        for fact in TEACHING_FACTS:
            result = _call(self.ctx, "nanlm_chat", {
                "message": fact,
                "session_id": SESSION_ID,
                "probe": False,
            })
            self.assertIs(result.get("probe"), False, fact)
            self.assertIs(result.get("state_mutated"), True, fact)

    def _probe(self, question: str) -> dict:
        result = _call(self.ctx, "nanlm_chat", {
            "message": question,
            "session_id": SESSION_ID,
            "probe": True,
        })
        self.assertIs(result.get("probe"), True, question)
        self.assertIs(result.get("state_mutated"), False, question)
        synthesis = result.get("native_synthesis") or {}
        self.assertIs(synthesis.get("probe"), True, question)
        self.assertIs(synthesis.get("state_mutated"), False, question)
        scratch = result.get("scratch") or {}
        if isinstance(scratch, dict) and "state_mutated" in scratch:
            self.assertIs(scratch["state_mutated"], False, question)
        self.assertTrue(result.get("response"), question)
        return result

    def _assert_probe_counts_unchanged(self, before: dict, after: dict,
                                       question: str) -> None:
        for key in ("conversation_lines", "layered_turns", "memory_events",
                    "scratch_pages"):
            with self.subTest(question=question, counter=key):
                self.assertEqual(
                    after["counts"][key], before["counts"][key],
                    "%s changed across read-only probes: %r -> %r"
                    % (key, before["counts"][key], after["counts"][key]))
        with self.subTest(question=question, counter="scratch_bytes"):
            self.assertEqual(
                after["counts"]["scratch_bytes"],
                before["counts"]["scratch_bytes"],
                "scratch page bytes changed across read-only probes")
        with self.subTest(question=question, counter="learner_state_fingerprint"):
            self.assertEqual(
                after["fingerprint"], before["fingerprint"],
                "strict learner-state fingerprint changed across read-only "
                "probes; sidecars before=%r after=%r"
                % (before["sidecars"], after["sidecars"]))

    def _run_repeated_probes(self, question: str, *,
                             require_zero_register_frames: bool = False
                             ) -> list[dict]:
        before = _snapshot(self.root)
        results = [self._probe(question) for _ in range(PROBE_REPEATS)]
        after = _snapshot(self.root)
        self._assert_probe_counts_unchanged(before, after, question)

        responses = [str(result.get("response") or "") for result in results]
        frames = [
            (result.get("native_synthesis") or {}).get("register_frames")
            for result in results
        ]
        if require_zero_register_frames:
            self.assertEqual(
                frames, [0] * len(frames),
                "read-only probes advanced the shared in-process temporal "
                "register: register_frames=%r; this is not persisted by the "
                "probe receipt but it is observable learner state" % (frames,))
        self.assertEqual(
            len(set(responses)), 1,
            "read-only probe responses were not deterministic across %d runs: "
            "%r; temporal register frames=%r"
            % (PROBE_REPEATS, responses, frames))
        return results

    def test_structured_action_probe_is_read_only_and_deterministic(self):
        self._teach()
        results = self._run_repeated_probes("What do dogs do?")
        synthesis = results[0].get("native_synthesis") or {}
        if not synthesis.get("structured"):
            self.skipTest(
                "structured action intent is unavailable in this build")
        self.assertIn("bark", results[0]["response"].casefold())
        self.assertNotIn("meow", results[0]["response"].casefold())

    def test_structured_colour_probe_is_read_only_and_deterministic(self):
        self._teach()
        results = self._run_repeated_probes("What do red and blue make?")
        synthesis = results[0].get("native_synthesis") or {}
        if not synthesis.get("structured"):
            self.skipTest(
                "structured colour intent is unavailable in this build")
        self.assertIn("purple", results[0]["response"].casefold())
        self.assertNotIn("green", results[0]["response"].casefold())

    def test_unstructured_open_probe_is_read_only_and_deterministic(self):
        self._teach()
        results = self._run_repeated_probes("Why do people learn things?")
        self.assertFalse((results[0].get("native_synthesis") or {}).get(
            "structured"), "the open question unexpectedly used a structured "
            "answer")

    def test_probe_does_not_advance_in_process_temporal_register(self):
        self._teach()
        self._run_repeated_probes(
            "What do dogs do?", require_zero_register_frames=True)

    def test_non_probe_chat_control_changes_learner_state(self):
        self._teach()
        before = _snapshot(self.root)
        result = _call(self.ctx, "nanlm_chat", {
            "message": "Birds fly.",
            "session_id": SESSION_ID,
            "probe": False,
        })
        self.assertIs(result.get("probe"), False)
        self.assertIs(result.get("state_mutated"), True)
        after = _snapshot(self.root)

        self.assertNotEqual(
            after["fingerprint"], before["fingerprint"],
            "a non-probe teaching turn must change the learner-state "
            "fingerprint; counts before=%r after=%r"
            % (before["counts"], after["counts"]))
        self.assertGreater(
            after["counts"]["conversation_lines"],
            before["counts"]["conversation_lines"],
            "non-probe chat must append transcript lines")
        self.assertGreater(
            after["counts"]["layered_turns"],
            before["counts"]["layered_turns"],
            "non-probe chat must advance layered learning")
        self.assertGreater(
            after["counts"]["memory_events"],
            before["counts"]["memory_events"],
            "non-probe chat must add mathematical-memory events")
        self.assertGreaterEqual(
            after["counts"]["scratch_pages"],
            before["counts"]["scratch_pages"],
            "non-probe chat must not remove scratch pages")
        self.assertGreater(
            after["counts"]["scratch_bytes"],
            before["counts"]["scratch_bytes"],
            "non-probe chat must append scratch-register bytes")


if __name__ == "__main__":
    unittest.main()
