import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanlmind_hermes.resident import ingest


SOURCE = hashlib.sha256(b"resident-ingest-source").hexdigest()
OTHER_SOURCE = hashlib.sha256(b"resident-ingest-other").hexdigest()


def relation_row(**overrides):
    row = {
        "kind": "relation",
        "subject": "river",
        "predicate": "flows_to",
        "object": "sea",
        "expression": {"op": "relation", "args": ["river", "flows_to", "sea"]},
        "provenance": {"source": "unit-test", "native_head": 17},
        "polarity": "supports",
        "confidence": 0.8,
        "head": 17,
        "layer": 2,
    }
    row.update(overrides)
    return row


def observation_row(**overrides):
    row = {
        "kind": "observation",
        "subject": "river",
        "predicate": "observed",
        "object": "water",
        "expression": "The river carries water.",
        "provenance": {"source": "unit-test"},
        "polarity": "neutral",
        "confidence": 0.6,
        "head": 21,
        "layer": 0,
    }
    row.update(overrides)
    return row


def trainable_row(**overrides):
    row = {
        "schema": ingest.TRAINABLE_SCHEMA,
        "record_id": "record-1",
        "text": "A deterministic trainable sentence.",
        "stage": "short_sentences",
        "source": "authored",
        "split": "train",
        "provenance": {"source_file": "unit.jsonl"},
    }
    row.update(overrides)
    return row


class PrepareRecordTests(unittest.TestCase):
    def test_relation_row_becomes_three_leaves_and_one_relation(self):
        records = ingest.prepare_records([relation_row()], source_digest=SOURCE)
        self.assertEqual([record["type"] for record in records],
                         ["leaf", "leaf", "leaf", "relation"])
        self.assertEqual([record["role"] for record in records[:3]],
                         ["subject", "predicate", "object"])
        relation = records[-1]
        self.assertEqual(relation["kind"], "relation")
        self.assertEqual(relation["subject"], "river")
        self.assertEqual(relation["predicate"], "flows_to")
        self.assertEqual(relation["object"], "sea")
        self.assertEqual(relation["polarity"], "supports")
        self.assertEqual(relation["confidence"], 0.8)
        self.assertEqual(relation["head"], 17)
        self.assertEqual(relation["layer"], 2)
        self.assertEqual(relation["source_digest"], SOURCE)
        self.assertEqual(relation["provenance"], relation_row()["provenance"])
        self.assertEqual(relation["expression"], relation_row()["expression"])
        self.assertEqual(relation["subject_leaf_id"], records[0]["leaf_id"])
        self.assertEqual(relation["predicate_leaf_id"], records[1]["leaf_id"])
        self.assertEqual(relation["object_leaf_id"], records[2]["leaf_id"])
        self.assertFalse(relation["observation"])
        for record in records:
            self.assertEqual(record["source_digest"], SOURCE)
            self.assertEqual(record["feature_dim"], 256)
            self.assertEqual(record["feature_dtype"], "int16")
            self.assertEqual(len(record["features"]), 256)
            self.assertTrue(all(-32767 <= value <= 32767
                                for value in record["features"]))
            self.assertEqual(len(record["feature_digest"]), 64)

    def test_observation_row_becomes_leaf_not_relation(self):
        records = ingest.prepare_records([observation_row()],
                                         source_digest=SOURCE)
        self.assertEqual(len(records), 1)
        leaf = records[0]
        self.assertEqual(leaf["type"], "leaf")
        self.assertEqual(leaf["kind"], "observation")
        self.assertTrue(leaf["observation"])
        self.assertEqual(leaf["text"], "The river carries water.")
        self.assertEqual(leaf["subject"], "river")
        self.assertEqual(leaf["predicate"], "observed")
        self.assertEqual(leaf["object"], "water")
        self.assertEqual(leaf["source_digest"], SOURCE)
        self.assertNotIn("relation", {record["type"] for record in records})

    def test_observation_with_complete_triple_never_creates_relation(self):
        records = ingest.prepare_records(
            [{"kind": "observation", "subject": "a", "predicate": "b",
              "object": "c"}],
            source_digest=SOURCE,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "leaf")
        self.assertEqual(records[0]["text"], "c")

    def test_trainable_jsonl_record_becomes_observation_leaf(self):
        records = ingest.prepare_records([trainable_row()])
        self.assertEqual(len(records), 1)
        leaf = records[0]
        self.assertEqual(leaf["type"], "leaf")
        self.assertEqual(leaf["kind"], "observation")
        self.assertTrue(leaf["observation"])
        self.assertEqual(leaf["record_id"], "record-1")
        self.assertEqual(leaf["origin"], ingest.TRAINABLE_SCHEMA)
        self.assertEqual(leaf["stage"], "short_sentences")
        self.assertEqual(leaf["source"], "authored")
        self.assertEqual(leaf["split"], "train")
        self.assertEqual(leaf["text"], "A deterministic trainable sentence.")
        self.assertEqual(len(leaf["source_digest"]), 64)
        self.assertNotIn(leaf["source_digest"], {None, ""})

    def test_wrapper_row_outer_source_and_route_are_preserved(self):
        records = ingest.prepare_records([{
            "item": relation_row(head=None, layer=None),
            "source_digest": OTHER_SOURCE,
            "head": 42,
            "layer": 4,
        }])
        relation = [record for record in records
                    if record["type"] == "relation"][0]
        self.assertEqual(relation["source_digest"], OTHER_SOURCE)
        self.assertEqual(relation["head"], 42)
        self.assertEqual(relation["layer"], 4)

    def test_role_alias_and_polarity_confidence_mapping(self):
        records = ingest.prepare_records([{
            "kind": "hypothesis", "subject": "A", "role": "implies",
            "object": "B", "polarity": "positive", "confidence": 3,
        }], source_digest=SOURCE)
        relation = records[-1]
        self.assertEqual(relation["predicate"], "implies")
        self.assertEqual(relation["polarity"], "supports")
        self.assertEqual(relation["confidence"], 1.0)
        records = ingest.prepare_records([{
            "kind": "relation", "subject": "A", "predicate": "p",
            "object": "B", "polarity": "contradicts", "confidence": -1,
        }], source_digest=SOURCE)
        relation = records[-1]
        self.assertEqual(relation["polarity"], "refutes")
        self.assertEqual(relation["confidence"], 0.0)

    def test_head_and_layer_are_bounded(self):
        records = ingest.prepare_records([relation_row(head=999, layer=99)],
                                         source_digest=SOURCE)
        relation = records[-1]
        self.assertEqual(relation["head"], ingest.MAX_HEAD)
        self.assertEqual(relation["layer"], ingest.MAX_LAYER)
        records = ingest.prepare_records([relation_row(head=-4, layer=-2)],
                                         source_digest=SOURCE)
        relation = records[-1]
        self.assertEqual(relation["head"], 0)
        self.assertEqual(relation["layer"], 0)


class IdentityAndFeatureTests(unittest.TestCase):
    def test_ids_are_deterministic_and_case_insensitive(self):
        first = ingest.prepare_records([relation_row()], source_digest=SOURCE)
        second = ingest.prepare_records([relation_row()], source_digest=SOURCE)
        self.assertEqual([record["leaf_id"] for record in first[:3]],
                         [record["leaf_id"] for record in second[:3]])
        self.assertEqual(first[-1]["relation_id"],
                         second[-1]["relation_id"])
        self.assertEqual(first[-1]["relation_hash"],
                         second[-1]["relation_hash"])
        self.assertEqual(len(first[-1]["relation_hash"]), 64)
        changed = ingest.prepare_records(
            [relation_row(subject="  RIVER  ", predicate="Flows_To",
                          object="SEA")],
            source_digest=SOURCE,
        )
        self.assertEqual(first[-1]["relation_id"],
                         changed[-1]["relation_id"])
        self.assertEqual(first[-1]["subject_leaf_id"],
                         changed[-1]["subject_leaf_id"])
        self.assertEqual(ingest.leaf_id_for_text("Sea"),
                         ingest.leaf_id_for_text("  sea  "))

    def test_feature_projection_is_deterministic_and_text_sensitive(self):
        first = ingest.encode_features("the river flows to the sea")
        second = ingest.encode_features("the river flows to the sea")
        different = ingest.encode_features("an unrelated sentence")
        self.assertEqual(first, second)
        self.assertNotEqual(first["digest"], different["digest"])
        self.assertNotEqual(first["values"], different["values"])
        self.assertEqual(first["dim"], 256)
        self.assertEqual(first["dtype"], "int16")
        self.assertTrue(any(value != 0 for value in first["values"]))

    def test_feature_dimensions_and_quantization_bounds(self):
        int8 = ingest.encode_features("bounded features", dim=128, dtype="int8")
        self.assertEqual(int8["dim"], 128)
        self.assertEqual(len(int8["values"]), 128)
        self.assertTrue(all(-127 <= value <= 127 for value in int8["values"]))
        int16 = ingest.encode_features("bounded features", dim=256,
                                       dtype="int16")
        self.assertEqual(int16["dim"], 256)
        self.assertEqual(len(int16["values"]), 256)
        self.assertTrue(all(-32767 <= value <= 32767
                            for value in int16["values"]))
        with self.assertRaises(ValueError):
            ingest.encode_features("x", dim=64)
        with self.assertRaises(ValueError):
            ingest.encode_features("x", dim=256, dtype="float32")
        self.assertEqual(ingest.feature_vector("x", dim=128, dtype="int8"),
                         ingest.project_features("x", dim=128, dtype="int8"))

    def test_expression_participates_in_relation_features(self):
        left = ingest.prepare_records(
            [relation_row(expression={"op": "left"})],
            source_digest=SOURCE,
        )[-1]
        right = ingest.prepare_records(
            [relation_row(expression={"op": "right"})],
            source_digest=SOURCE,
        )[-1]
        self.assertEqual(left["relation_id"], right["relation_id"])
        self.assertNotEqual(left["feature_digest"], right["feature_digest"])
        self.assertNotEqual(left["record_id"], right["record_id"])

    def test_source_digest_validation_and_derivation(self):
        with self.assertRaises(ValueError):
            ingest.prepare_records([relation_row()], source_digest="not-a-digest")
        with self.assertRaises(ValueError):
            ingest.prepare_records([relation_row(source_digest="bad")])
        derived = ingest.prepare_records([relation_row()])[-1]
        self.assertRegex(derived["source_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(derived["source_digest"],
                         ingest.derive_source_digest({
                             "kind": "relation",
                             "relation": {
                                 "subject": "river",
                                 "predicate": "flows_to",
                                 "object": "sea",
                             },
                             "expression": relation_row()["expression"],
                         }))


class DuplicateAndBatchTests(unittest.TestCase):
    def test_duplicate_rows_are_idempotent(self):
        rows = [relation_row(), relation_row()]
        records = ingest.prepare_records(rows, source_digest=SOURCE)
        self.assertEqual(len(records), 4)
        self.assertEqual([record["type"] for record in records],
                         ["leaf", "leaf", "leaf", "relation"])
        once = ingest.prepare_records([relation_row()], source_digest=SOURCE)
        twice = ingest.prepare_records([relation_row()], source_digest=OTHER_SOURCE)
        self.assertEqual(once[-1]["relation_id"], twice[-1]["relation_id"])
        self.assertNotEqual(once[-1]["record_id"], twice[-1]["record_id"])

    def test_conflicting_polarities_keep_distinct_relations_and_one_triple(self):
        records = ingest.prepare_records([
            relation_row(polarity="supports"),
            relation_row(polarity="refutes"),
        ], source_digest=SOURCE)
        relations = [record for record in records
                     if record["type"] == "relation"]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0]["triple_hash"],
                         relations[1]["triple_hash"])
        self.assertNotEqual(relations[0]["relation_id"],
                            relations[1]["relation_id"])
        self.assertNotEqual(relations[0]["record_id"],
                            relations[1]["record_id"])
        self.assertEqual({record["polarity"] for record in relations},
                         {"supports", "refutes"})

    def test_prepare_enforces_max_batch(self):
        rows = [observation_row(expression="obs-%d" % index)
                for index in range(ingest.MAX_BATCH + 1)]
        with self.assertRaises(ValueError):
            ingest.prepare_records(rows, source_digest=SOURCE)
        with self.assertRaises(ValueError):
            ingest.prepare_records(rows[:3], source_digest=SOURCE,
                                   max_batch=2)
        with self.assertRaises(ValueError):
            ingest.iter_batches(rows, max_batch=0)
        with self.assertRaises(ValueError):
            ingest.iter_batches(rows, max_batch=ingest.MAX_BATCH + 1)

    def test_iter_batches_is_bounded(self):
        rows = [observation_row(expression="obs-%d" % index)
                for index in range(5)]
        batches = list(ingest.iter_batches(rows, max_batch=2))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertTrue(all(len(batch) <= 2 for batch in batches))

    def test_prepare_batch_reports_counts_and_sources(self):
        batch = ingest.prepare_batch(
            [relation_row(), observation_row(), trainable_row()],
            source_digest=SOURCE,
        )
        self.assertEqual(batch["counts"], {
            "records": 6, "leaves": 5, "relations": 1, "observations": 2,
        })
        self.assertEqual(batch["source_digests"], [SOURCE])


class JsonlTests(unittest.TestCase):
    def test_parse_jsonl_accepts_multiple_objects_per_line(self):
        line = json.dumps(trainable_row()) + " " + json.dumps(relation_row())
        rows = ingest.parse_jsonl(line + "\n\n" + json.dumps(observation_row()))
        self.assertEqual(len(rows), 3)
        records = ingest.prepare_records(rows, source_digest=SOURCE)
        self.assertEqual(sum(record["type"] == "relation" for record in records), 1)
        self.assertEqual(sum(bool(record.get("observation")) for record in records), 2)

    def test_load_local_jsonl_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            path.write_text(
                json.dumps(trainable_row()) + "\n" +
                json.dumps(observation_row()) + "\n",
                encoding="utf-8",
            )
            rows = ingest.load_jsonl(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(ingest.read_jsonl(path), rows)
        records = ingest.prepare_records(rows, source_digest=SOURCE)
        self.assertTrue(all(record["type"] == "leaf" for record in records))

    def test_invalid_jsonl_is_rejected(self):
        with self.assertRaises(ValueError):
            ingest.parse_jsonl("{not-json}\n")
        with self.assertRaises(ValueError):
            ingest.parse_jsonl("[1, 2, 3]\n")

    def test_ingest_jsonl_uses_bounded_limit_and_universe(self):
        universe = StubUniverse()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            path.write_text(
                "\n".join(json.dumps(observation_row(
                    expression="observation %d" % index))
                    for index in range(3)) + "\n",
                encoding="utf-8",
            )
            summary = ingest.ingest_jsonl(
                path, limit=2, universe=universe, source_digest=SOURCE)
        self.assertEqual(summary["limit"], 2)
        self.assertEqual(summary["requested_limit"], 2)
        self.assertFalse(summary["limit_clamped"])
        self.assertEqual(summary["ingested"], 2)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(len(universe.leaves), 2)
        self.assertEqual(summary["source"], str(path))


class StubUniverse:
    """Small stand-in for the resident append API."""

    def __init__(self):
        self.leaves = {}
        self.relations = {}
        self.leaf_calls = []
        self.relation_calls = []
        self.batch_sizes = []

    def append_leaf(self, record, source_digest=None):
        self.leaf_calls.append((record["leaf_id"], source_digest))
        self.batch_sizes.append(1)
        if record["leaf_id"] in self.leaves:
            return {"leaf_id": record["leaf_id"], "duplicate": True}
        self.leaves[record["leaf_id"]] = dict(record)
        return {"leaf_id": record["leaf_id"], "duplicate": False}

    def append_relation(self, record, source_digest=None):
        self.relation_calls.append((record["record_id"], source_digest))
        self.batch_sizes.append(1)
        if record["record_id"] in self.relations:
            return {"record_id": record["record_id"], "duplicate": True}
        self.relations[record["record_id"]] = dict(record)
        return {"record_id": record["record_id"], "duplicate": False}


class ResidentIntegrationTests(unittest.TestCase):
    def test_stub_universe_receives_leaves_then_relations(self):
        universe = StubUniverse()
        summary = ingest.ingest_rows(
            [relation_row(), observation_row(), trainable_row()],
            universe=universe,
            source_digest=SOURCE,
        )
        self.assertTrue(summary["applied"])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["batches"], 1)
        self.assertEqual(summary["records"], 6)
        self.assertEqual(summary["relations"], 1)
        self.assertEqual(summary["observations"], 2)
        self.assertEqual(summary["accepted"], 6)
        self.assertEqual(summary["duplicates"], 0)
        self.assertEqual(len(universe.relations), 1)
        self.assertEqual(len(universe.leaves), 5)
        relation = next(iter(universe.relations.values()))
        self.assertEqual(relation["source_digest"], SOURCE)
        self.assertTrue(all(source == SOURCE
                            for _, source in universe.leaf_calls))
        self.assertTrue(all(source == SOURCE
                            for _, source in universe.relation_calls))

    def test_stub_reingest_is_idempotent(self):
        universe = StubUniverse()
        rows = [relation_row(), observation_row()]
        first = ingest.ingest_rows(rows, universe=universe,
                                   source_digest=SOURCE)
        second = ingest.ingest_rows(rows, universe=universe,
                                    source_digest=SOURCE)
        self.assertEqual(first["accepted"], 5)
        self.assertEqual(second["accepted"], 0)
        self.assertEqual(second["duplicates"], 5)
        self.assertEqual(len(universe.leaves), 4)
        self.assertEqual(len(universe.relations), 1)

    def test_ingest_splits_large_input_into_bounded_batches(self):
        universe = StubUniverse()
        rows = [observation_row(expression="observation %d" % index)
                for index in range(5)]
        summary = ingest.ingest_rows(rows, universe=universe,
                                     source_digest=SOURCE, max_batch=2)
        self.assertEqual(summary["batches"], 3)
        self.assertEqual(len(universe.leaves), 5)
        self.assertEqual(summary["accepted"], 5)

    def test_lazy_loader_falls_back_to_prepared_records(self):
        with mock.patch.object(ingest, "load_resident_universe",
                               return_value=None):
            summary = ingest.ingest_rows([relation_row()],
                                         source_digest=SOURCE)
        self.assertFalse(summary["applied"])
        self.assertFalse(summary["available"])
        self.assertEqual(summary["records"], 4)
        self.assertEqual(len(summary["prepared_records"]), 4)
        self.assertEqual(summary["source_digests"], [SOURCE])

    def test_lazy_loader_is_used_when_available(self):
        universe = StubUniverse()
        loader = mock.Mock(return_value=universe)
        summary = ingest.ingest_rows([observation_row()],
                                     source_digest=SOURCE, loader=loader)
        loader.assert_called_once_with(context=None, state_dir=None)
        self.assertTrue(summary["applied"])
        self.assertEqual(len(universe.leaves), 1)

    def test_apply_records_requires_an_append_api(self):
        with self.assertRaises(TypeError):
            ingest.apply_records(object(), ingest.prepare_records(
                [observation_row()], source_digest=SOURCE))

    def test_apply_records_enforces_batch_bound(self):
        with self.assertRaises(ValueError):
            ingest.apply_records(StubUniverse(),
                                 [{"type": "leaf"}] * (ingest.MAX_BATCH + 1))

    def test_many_relations_split_into_bounded_record_batches(self):
        universe = StubUniverse()
        rows = [relation_row(subject="entity-%d" % index,
                             predicate="links-%d" % index,
                             object="target-%d" % index)
                for index in range(1025)]
        summary = ingest.ingest_rows(rows, universe=universe,
                                     source_digest=SOURCE)
        self.assertEqual(summary["records"], 4100)
        self.assertEqual(summary["record_batches"], 2)
        self.assertEqual(summary["accepted"], 4100)
        self.assertEqual(len(universe.relations), 1025)
        self.assertEqual(len(universe.leaves), 3075)


if __name__ == "__main__":
    unittest.main()
