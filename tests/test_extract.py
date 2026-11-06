import hashlib
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.extract import extract_discoveries
from nanlmind_hermes.memory import MemoryStore


class CorpusExtractionTests(unittest.TestCase):
    def test_explicit_relations_equations_and_implications_keep_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text(
                "penguin cannot fly. penguin can swim. robin is a bird. x = y. if rain then wet\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
            records = extract_discoveries(corpus, digest)
            self.assertEqual(len(records), 5)
            by_subject = {record["item"]["subject"]: record for record in records}
            self.assertEqual(by_subject["ability:penguin:fly"]["item"]["polarity"], "refutes")
            self.assertEqual(by_subject["ability:penguin:swim"]["item"]["polarity"], "supports")
            self.assertEqual(by_subject["entity:robin:is_a:bird"]["item"]["object"], "bird")
            self.assertEqual(by_subject["equation:x=y"]["item"]["expression"]["op"], "equals")
            self.assertEqual(by_subject["implication:rain:wet"]["item"]["conditions"], ["rain"])
            self.assertEqual(by_subject["ability:penguin:fly"]["item"]["evidence"]["source_line"], 1)

    def test_conflicting_surface_claims_become_reviewable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text("penguin can fly. penguin cannot fly.\n", encoding="utf-8")
            digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
            records = extract_discoveries(corpus, digest)
            store = MemoryStore(Path(tmp) / "memory.sqlite3")
            store.merge(records)
            result = store.reorganize()
            self.assertEqual(result["contested_subjects"], 1)
            assessment = store.query_assessments("ability:penguin:fly")[0]
            self.assertEqual(assessment["status"], "contested")
            self.assertEqual(assessment["evidence_count"], 2)


if __name__ == "__main__":
    unittest.main()
