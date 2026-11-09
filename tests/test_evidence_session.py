"""Bounded add/ask session over the resident support/refute lanes."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.evidence_lanes import EvidenceLanes
from nanlmind_hermes.resident.evidence_session import (
    SCHEMA,
    EvidenceSession,
    SessionError,
)
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory
from nanlmind_hermes.resident.universe import ResidentUniverse


def _lanes():
    cells = ResidentCells.format_region(bytearray(REGION_BYTES))
    return EvidenceLanes(FullCycleMemory(cells))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.lanes = _lanes()
        self.session = EvidenceSession(self.lanes)

    def test_tell_support_and_ask(self):
        result = self.session.tell("quokka is a kind of animal")
        self.assertEqual(result.polarity, "supports")
        self.assertEqual(result.subject, "quokka")
        self.assertEqual(result.relation, "is a kind of")
        self.assertEqual(result.destination, "animal")
        ask = self.session.ask("is quokka a kind of animal?")
        self.assertEqual(ask.status, "supports")
        self.assertIn("Yes", ask.answer)

    def test_refute_does_not_cancel_support(self):
        self.session.tell("quokka is a kind of animal")
        self.session.tell("quokka is not a kind of animal")
        ask = self.session.ask("is quokka a kind of animal?")
        self.assertEqual(ask.status, "both")
        self.assertEqual(len(ask.state.supports), 1)
        self.assertEqual(len(ask.state.refutes), 1)
        self.assertIn("both supports and refutes", ask.answer)

    def test_unknown_and_article_grammar(self):
        ask = self.session.ask("is quokka a plant?")
        self.assertEqual(ask.status, "unknown")
        self.assertIn("not have enough", ask.answer)
        result = self.session.tell("quokka is an animal")
        self.assertEqual(result.relation, "is a")
        self.assertEqual(self.session.ask("is quokka an animal?").status,
                         "supports")
        self.assertIn("an animal", self.session.ask(
            "is quokka an animal?").answer)

    def test_explicit_polarity_and_rejection(self):
        result = self.session.tell("quokka is a kind of animal",
                                   polarity="refutes")
        self.assertEqual(result.polarity, "refutes")
        with self.assertRaises(SessionError):
            self.session.tell("the weather is pleasant")
        with self.assertRaises(SessionError):
            self.session.ask("why is quokka an animal?")

    def test_save_load_and_status(self):
        self.session.tell("quokka is a kind of animal")
        self.session.tell("quokka is not a kind of plant")
        snapshot = self.session.save()
        status = self.session.status()
        self.assertEqual(status["schema"], SCHEMA)
        self.assertEqual(status["records"], 2)
        self.assertEqual(status["supports"], 1)
        self.assertEqual(status["refutes"], 1)
        fresh = EvidenceSession(_lanes())
        fresh.load(snapshot)
        self.assertEqual(fresh.ask("is quokka a kind of animal?").status,
                         "supports")
        self.assertEqual(fresh.ask("is quokka a kind of plant?").status,
                         "refutes")

    def test_universe_session_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.nlm"
            with ResidentUniverse.open(path, 32 * 1024 * 1024) as universe:
                session = universe.evidence_session()
                self.assertIs(session, universe.evidence_session())
                session.tell("quokka is a kind of animal")
                self.assertEqual(
                    session.ask("is quokka a kind of animal?").status,
                    "supports")
            with ResidentUniverse.open(path, 32 * 1024 * 1024) as reopened:
                self.assertEqual(
                    reopened.evidence_session().ask(
                        "is quokka a kind of animal?").status,
                    "supports")


if __name__ == "__main__":
    unittest.main()
