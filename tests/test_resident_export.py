"""Tests for bounded resident candidate export, validation and intake.

These tests use a stub universe and stay entirely local. They exercise the
bundle format, digest checks, generation compare-and-swap, duplicate intake,
bundle bounds, explicit approval and the disabled canonical merge gate.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.export import (
    SCHEMA,
    SECTIONS,
    MergeGate,
    canonical_json,
    compute_manifest_digest,
    export_candidate,
    integrate_local_corpus,
    intake_candidate,
    validate_candidate,
)

SOURCE_DIGEST = hashlib.sha256(b"source-a").hexdigest()


class StubUniverse:
    """Minimal generation-bound universe implementing the export/intake protocol."""

    def __init__(self, universe_id: str = "stub-universe"):
        self.universe_id = universe_id
        self.generation = 0
        self._digests = {
            0: hashlib.sha256(("base:" + universe_id).encode()).hexdigest(),
        }
        self.records: dict[str, list[dict]] = {section: [] for section in SECTIONS}
        self.source_digests = {"corpus:test": SOURCE_DIGEST}
        self.external_references = {"leaves": [], "relations": [], "records": []}
        self.applied_candidate_digests: set[str] = set()
        self.local_corpus: list[dict] = []
        self.snapshot = b"stub resident snapshot"

    @property
    def digest(self) -> str:
        return self._digests[self.generation]

    def digest_at(self, generation: int) -> str:
        return self._digests[generation]

    def set_generation(self, generation: int) -> None:
        self.generation = generation
        self._digests[generation] = hashlib.sha256(
            ("%s:%d" % (self.universe_id, generation)).encode()).hexdigest()

    def delta_records(self, since_generation=None):
        if since_generation is not None and since_generation >= self.generation:
            return {section: [] for section in SECTIONS}
        return {section: list(self.records[section]) for section in SECTIONS}

    def snapshot_bytes(self) -> bytes:
        return self.snapshot

    def has_candidate(self, candidate_digest: str) -> bool:
        return candidate_digest in self.applied_candidate_digests

    def apply_candidate(self, records, manifest, *, approved=False,
                        expected_generation=None, target_generation=None,
                        candidate_digest=None):
        if approved is not True:
            return {"accepted": False, "reasons": ["approval required"]}
        if expected_generation != self.generation:
            return {"accepted": False, "status": "stale_generation",
                    "reasons": ["generation compare-and-swap failed"]}
        if candidate_digest in self.applied_candidate_digests:
            return {"accepted": False, "status": "duplicate"}
        appended = 0
        for section in SECTIONS:
            values = records.get(section, ())
            self.records[section].extend(values)
            appended += len(values)
        self.generation = target_generation
        self._digests[self.generation] = hashlib.sha256(
            (self._digests[expected_generation] + candidate_digest).encode()
        ).hexdigest()
        self.applied_candidate_digests.add(candidate_digest)
        return {"accepted": True, "applied": appended,
                "generation": self.generation}

    def integrate_local_corpus(self, records, manifest, *, approved=False,
                               purpose=None, candidate_digest=None):
        if approved is not True:
            return {"accepted": False, "reasons": ["approval required"]}
        count = sum(len(records.get(section, ())) for section in SECTIONS)
        self.local_corpus.append({
            "purpose": purpose,
            "candidate_digest": candidate_digest,
            "records": count,
        })
        return {"accepted": True, "records": count}


class MockExternalGate:
    """Stand-in for a future external canonical merge gate."""

    def __init__(self, authorized=True):
        self.authorized = authorized

    def authorize_canonical(self, candidate_digest, *, approved=False):
        return {
            "authorized": bool(
                self.authorized
                and approved is True
                and candidate_digest == "a" * 64
            )
        }


class ResidentCandidateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _source(self, universe_id="source-universe", generation=1):
        universe = StubUniverse(universe_id)
        universe.records["leaves"] = [
            {"id": "leaf:river", "text": "river"},
            {"id": "leaf:water", "text": "water"},
        ]
        universe.records["relations"] = [{
            "id": "rel:river-water",
            "subject": "river",
            "predicate": "is_a",
            "object": "water-body",
            "leaf_id": "leaf:river",
            "source_digest": SOURCE_DIGEST,
        }]
        universe.records["pairs"] = [{
            "id": "pair:river-water",
            "left": "leaf:river",
            "right": "leaf:water",
            "operator": "resonance",
        }]
        universe.records["hyperedges"] = [{
            "id": "edge:river",
            "members": ["leaf:river", "rel:river-water"],
            "arity": 2,
        }]
        universe.set_generation(generation)
        return universe

    def _export(self, name="bundle", **kwargs):
        bundle = self.root / name
        result = export_candidate(self._source(), bundle, **kwargs)
        return bundle, result

    def _copy(self, bundle, name):
        destination = self.root / name
        shutil.copytree(bundle, destination)
        return destination

    def _read_records(self, bundle):
        records = {}
        for section in SECTIONS:
            path = Path(bundle) / "delta" / ("%s.jsonl" % section)
            lines = path.read_text(encoding="utf-8").splitlines()
            records[section] = [json.loads(line) for line in lines if line]
        return records

    def _rewrite(self, bundle, records=None, mutate_manifest=None):
        """Rewrite a bundle with recomputed section and manifest digests."""
        bundle = Path(bundle)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = self._read_records(bundle) if records is None else records
        for section in SECTIONS:
            entries = sorted(current[section], key=lambda item: item["id"])
            payload = b"".join(
                canonical_json(record) + b"\n" for record in entries)
            path = bundle / "delta" / ("%s.jsonl" % section)
            path.write_bytes(payload)
            manifest["sections"][section] = {
                "file": "delta/%s.jsonl" % section,
                "count": len(entries),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            manifest["section_digests"][section] = hashlib.sha256(
                payload).hexdigest()
            manifest["counts"][section] = len(entries)
        total = sum(len(current[section]) for section in SECTIONS)
        manifest["counts"]["total"] = total
        manifest["delta"]["record_count"] = total
        if mutate_manifest is not None:
            mutate_manifest(manifest)
        manifest["manifest_digest"] = compute_manifest_digest(manifest)
        manifest_path.write_bytes(canonical_json(manifest) + b"\n")
        (bundle / "manifest.sha256").write_text(
            manifest["manifest_digest"] + "\n", encoding="utf-8")
        return manifest

    def test_export_writes_bounded_manifest_and_delta_sections(self):
        bundle, result = self._export()
        manifest = json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], SCHEMA)
        self.assertEqual(manifest["universe_id"], "source-universe")
        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(manifest["parent_generation"], 0)
        self.assertEqual(manifest["parent_digest"],
                         self._source().digest_at(0))
        self.assertEqual(manifest["counts"], {
            "relations": 1, "leaves": 2, "pairs": 1, "hyperedges": 1,
            "total": 5,
        })
        self.assertEqual(manifest["source_digests"],
                         {"corpus:test": SOURCE_DIGEST})
        self.assertFalse(manifest["include_full"])
        self.assertIsNone(manifest["full_snapshot"])
        self.assertTrue(manifest["no_canonical_claim"])
        self.assertFalse(manifest["merge_enabled"])
        self.assertFalse(manifest["canonical_merge_enabled"])
        self.assertFalse((bundle / "full" / "snapshot.bin").exists())
        self.assertFalse(result["canonical"])
        self.assertEqual(manifest["manifest_digest"],
                         compute_manifest_digest(manifest))
        self.assertEqual(
            (bundle / "manifest.sha256").read_text(encoding="utf-8").strip(),
            manifest["manifest_digest"])

        records = self._read_records(bundle)
        self.assertEqual([record["id"] for record in records["leaves"]],
                         ["leaf:river", "leaf:water"])
        self.assertEqual(records["relations"][0]["id"], "rel:river-water")
        self.assertEqual(records["pairs"][0]["id"], "pair:river-water")
        self.assertEqual(records["hyperedges"][0]["id"], "edge:river")
        self.assertEqual(records["relations"][0]["source_digest"],
                         SOURCE_DIGEST)

    def test_export_validate_intake_round_trip_and_duplicate_replay(self):
        bundle, _ = self._export()
        validation = validate_candidate(bundle)
        self.assertTrue(validation.accepted, validation.reasons)
        self.assertEqual(validation.reasons, ())
        self.assertEqual(validation.counts["total"], 5)
        self.assertFalse(validation.include_full)

        target = StubUniverse("target-universe")
        unapproved = intake_candidate(target, bundle)
        self.assertEqual(unapproved["status"], "approval_required")
        self.assertFalse(unapproved["accepted"])
        self.assertFalse(unapproved["applied"])
        self.assertEqual(target.generation, 0)
        self.assertTrue(all(not target.records[section] for section in SECTIONS))

        receipt = intake_candidate(target, bundle, approved=True)
        self.assertEqual(receipt["status"], "accepted")
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["applied"])
        self.assertFalse(receipt["duplicate"])
        self.assertEqual(receipt["target_generation_before"], 0)
        self.assertEqual(receipt["target_generation_after"], 1)
        self.assertTrue(receipt["generation_compare_and_swap"]["ok"])
        self.assertFalse(receipt["canonical"])
        self.assertFalse(receipt["canonical_merge"])
        self.assertFalse(receipt["canonical_authority"])
        self.assertFalse(receipt["canonical_merge_enabled"])
        self.assertFalse(receipt["canonical_commit"])
        self.assertFalse(receipt["canonical_state_modified"])
        self.assertFalse(receipt["promotion_allowed"])
        self.assertEqual(receipt["six_runner_barrier"], "absent")
        self.assertEqual(target.generation, 1)
        for section in SECTIONS:
            self.assertEqual(target.records[section],
                             self._source().records[section])
        self.assertIn(receipt["manifest_digest"],
                      target.applied_candidate_digests)

        replay = intake_candidate(target, bundle, approved=True)
        self.assertEqual(replay["status"], "duplicate")
        self.assertFalse(replay["accepted"])
        self.assertFalse(replay["applied"])
        self.assertTrue(replay["duplicate"])
        self.assertTrue(replay["already_applied"])
        self.assertEqual(replay["target_generation_after"], 1)
        self.assertEqual(target.generation, 1)
        for section in SECTIONS:
            self.assertEqual(len(target.records[section]),
                             len(self._source().records[section]))

    def test_tampered_records_and_wrong_manifest_digest_are_rejected(self):
        bundle, _ = self._export()
        self.assertTrue(validate_candidate(bundle).accepted)

        tampered = self._copy(bundle, "tampered-records")
        relations = tampered / "delta" / "relations.jsonl"
        raw = relations.read_bytes()
        updated = raw.replace(b'"predicate":"is_a"', b'"predicate":"was_a"')
        self.assertNotEqual(raw, updated)
        relations.write_bytes(updated)
        result = validate_candidate(tampered)
        self.assertFalse(result.accepted)
        self.assertTrue(any("section digest mismatch" in reason
                            for reason in result.reasons), result.reasons)

        wrong_manifest = self._copy(bundle, "tampered-manifest")
        manifest = json.loads(
            (wrong_manifest / "manifest.json").read_text(encoding="utf-8"))
        manifest["generation"] = 2
        (wrong_manifest / "manifest.json").write_bytes(
            canonical_json(manifest) + b"\n")
        result = validate_candidate(wrong_manifest)
        self.assertFalse(result.accepted)
        self.assertTrue(any("manifest digest mismatch" in reason
                            for reason in result.reasons), result.reasons)

        wrong_sidecar = self._copy(bundle, "tampered-sidecar")
        (wrong_sidecar / "manifest.sha256").write_text(
            "0" * 64 + "\n", encoding="utf-8")
        result = validate_candidate(wrong_sidecar)
        self.assertFalse(result.accepted)
        self.assertTrue(any("manifest.sha256" in reason
                            for reason in result.reasons), result.reasons)

        wrong_schema = self._copy(bundle, "tampered-schema")
        self._rewrite(
            wrong_schema,
            mutate_manifest=lambda manifest: manifest.__setitem__(
                "schema", "nanlmind.resident.candidate/999"))
        result = validate_candidate(wrong_schema)
        self.assertFalse(result.accepted)
        self.assertTrue(any("unsupported candidate schema" in reason
                            for reason in result.reasons), result.reasons)

        undeclared_source = self._copy(bundle, "tampered-source")
        records = self._read_records(undeclared_source)
        records["relations"][0]["source_digest"] = "f" * 64
        self._rewrite(undeclared_source, records=records)
        result = validate_candidate(undeclared_source)
        self.assertFalse(result.accepted)
        self.assertTrue(any("undeclared source digest" in reason
                            for reason in result.reasons), result.reasons)

    def test_full_snapshot_is_opt_in_and_verified(self):
        default_bundle, _ = self._export("default-bundle")
        self.assertFalse((default_bundle / "full" / "snapshot.bin").exists())

        full_bundle, result = self._export(
            "full-bundle", include_full=True, max_bytes=1024 * 1024)
        manifest = json.loads(
            (full_bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["include_full"])
        self.assertEqual(manifest["full_snapshot"]["bytes"],
                         len(b"stub resident snapshot"))
        self.assertEqual((full_bundle / "full" / "snapshot.bin").read_bytes(),
                         b"stub resident snapshot")
        validation = validate_candidate(full_bundle, max_bytes=1024 * 1024)
        self.assertTrue(validation.accepted, validation.reasons)
        self.assertTrue(validation.include_full)

        tampered = self._copy(full_bundle, "tampered-full")
        with (tampered / "full" / "snapshot.bin").open("ab") as handle:
            handle.write(b"!")
        validation = validate_candidate(tampered, max_bytes=1024 * 1024)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("full snapshot" in reason
                            for reason in validation.reasons),
                        validation.reasons)

    def test_stale_generation_is_rejected_without_mutation(self):
        bundle, _ = self._export()
        target = StubUniverse("stale-target")
        target.set_generation(2)
        receipt = intake_candidate(target, bundle, approved=True)
        self.assertEqual(receipt["status"], "stale_generation")
        self.assertFalse(receipt["accepted"])
        self.assertFalse(receipt["applied"])
        self.assertEqual(receipt["target_generation_before"], 2)
        self.assertEqual(receipt["target_generation_after"], 2)
        self.assertFalse(receipt["generation_compare_and_swap"]["ok"])
        self.assertTrue(any("compare-and-swap" in reason
                            for reason in receipt["reasons"]),
                        receipt["reasons"])
        self.assertEqual(target.generation, 2)
        self.assertTrue(all(not target.records[section] for section in SECTIONS))

    def test_duplicate_record_ids_are_rejected(self):
        bundle, _ = self._export()
        duplicated = self._copy(bundle, "duplicate-ids")
        records = self._read_records(duplicated)
        records["relations"].append(dict(records["relations"][0]))
        self._rewrite(duplicated, records=records)

        validation = validate_candidate(duplicated)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("duplicate id" in reason
                            for reason in validation.reasons),
                        validation.reasons)

        target = StubUniverse("duplicate-target")
        receipt = intake_candidate(target, duplicated, approved=True)
        self.assertEqual(receipt["status"], "rejected")
        self.assertFalse(receipt["applied"])
        self.assertEqual(target.generation, 0)
        self.assertTrue(all(not target.records[section] for section in SECTIONS))

    def test_dangling_relation_and_leaf_references_are_rejected(self):
        bundle, _ = self._export()

        dangling_relation = self._copy(bundle, "dangling-relation")
        records = self._read_records(dangling_relation)
        records["relations"][0]["leaf_id"] = "leaf:missing"
        self._rewrite(dangling_relation, records=records)
        validation = validate_candidate(dangling_relation)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("references unknown leaf" in reason
                            for reason in validation.reasons),
                        validation.reasons)

        dangling_leaf = self._copy(bundle, "dangling-leaf")
        records = self._read_records(dangling_leaf)
        records["leaves"][0]["relation_id"] = "rel:missing"
        self._rewrite(dangling_leaf, records=records)
        validation = validate_candidate(dangling_leaf)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("references unknown relation" in reason
                            for reason in validation.reasons),
                        validation.reasons)

    def test_paths_and_secrets_are_rejected(self):
        bundle, _ = self._export()

        path_case = self._copy(bundle, "path-case")
        records = self._read_records(path_case)
        records["leaves"].append({"id": "leaf:path", "path": "/etc/passwd"})
        self._rewrite(path_case, records=records)
        validation = validate_candidate(path_case)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("path-like" in reason
                            for reason in validation.reasons),
                        validation.reasons)

        secret_case = self._copy(bundle, "secret-case")
        records = self._read_records(secret_case)
        records["leaves"].append({
            "id": "leaf:secret", "password": "hunter2",
        })
        self._rewrite(secret_case, records=records)
        validation = validate_candidate(secret_case)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("secret-like" in reason
                            for reason in validation.reasons),
                        validation.reasons)

    def test_oversized_bundle_is_rejected_at_validation_and_export(self):
        bundle, _ = self._export()
        validation = validate_candidate(bundle, max_bytes=32)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("max_bytes" in reason
                            for reason in validation.reasons),
                        validation.reasons)

        destination = self.root / "too-large"
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            export_candidate(self._source(), destination, max_bytes=32)
        self.assertFalse(destination.exists())

    def test_explicit_literal_approval_is_required(self):
        bundle, _ = self._export()
        for approved in (False, None, "yes", 1):
            with self.subTest(approved=approved):
                target = StubUniverse("approval-target")
                receipt = intake_candidate(target, bundle, approved=approved)
                self.assertEqual(receipt["status"], "approval_required")
                self.assertFalse(receipt["accepted"])
                self.assertFalse(receipt["applied"])
                self.assertFalse(receipt["approved"])
                self.assertEqual(target.generation, 0)
                self.assertTrue(all(
                    not target.records[section] for section in SECTIONS))

    def test_canonical_merge_is_disabled_by_default(self):
        gate = MergeGate()
        self.assertFalse(gate.canonical_merge_enabled)
        self.assertFalse(gate.canonical_promotion_allowed)
        status = gate.status()
        self.assertFalse(status["canonical_merge_enabled"])
        self.assertFalse(status["canonical_promotion_allowed"])
        self.assertTrue(status["external_gate_required"])
        self.assertFalse(status["local_integration_is_canonical"])
        with self.assertRaisesRegex(PermissionError, "disabled"):
            gate.require_canonical("a" * 64, approved=True)

        enabled_but_no_external_gate = MergeGate(canonical_merge_enabled=True)
        self.assertFalse(enabled_but_no_external_gate.canonical_promotion_allowed)
        with self.assertRaisesRegex(PermissionError, "external"):
            enabled_but_no_external_gate.require_canonical(
                "a" * 64, approved=True)

        allowing_gate = MergeGate(
            canonical_merge_enabled=True,
            external_gate=MockExternalGate(authorized=True),
        )
        self.assertTrue(allowing_gate.canonical_promotion_allowed)
        decision = allowing_gate.require_canonical("a" * 64, approved=True)
        self.assertTrue(decision["authorized"])

        denying_gate = MergeGate(
            canonical_merge_enabled=True,
            external_gate=MockExternalGate(authorized=False),
        )
        with self.assertRaisesRegex(PermissionError, "did not authorize"):
            denying_gate.require_canonical("a" * 64, approved=True)

        bundle, _ = self._export()
        canonical_claim = self._copy(bundle, "canonical-claim")
        self._rewrite(
            canonical_claim,
            mutate_manifest=lambda manifest: manifest.__setitem__(
                "canonical_merge_enabled", True))
        validation = validate_candidate(canonical_claim)
        self.assertFalse(validation.accepted)
        self.assertTrue(any("canonical_merge_enabled must be false" in reason
                            for reason in validation.reasons),
                        validation.reasons)

        target = StubUniverse("gate-target")
        receipt = intake_candidate(target, bundle, approved=True)
        self.assertEqual(receipt["status"], "accepted")
        self.assertFalse(receipt["canonical"])
        self.assertFalse(receipt["canonical_authority"])
        self.assertFalse(receipt["merge_gate"]["canonical_merge_enabled"])
        self.assertFalse(receipt["merge_gate"]["canonical_promotion_allowed"])

    def test_local_corpus_integration_is_separate_and_explicit(self):
        bundle, _ = self._export()
        target = StubUniverse("local-target")

        unapproved = integrate_local_corpus(
            target, bundle, purpose="training_corpus")
        self.assertEqual(unapproved["status"], "approval_required")
        self.assertFalse(unapproved["accepted"])
        self.assertEqual(target.local_corpus, [])
        self.assertEqual(target.generation, 0)

        receipt = integrate_local_corpus(
            target, bundle, purpose="training_corpus", approved=True)
        self.assertEqual(receipt["status"], "accepted")
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["local_only"])
        self.assertFalse(receipt["canonical"])
        self.assertFalse(receipt["canonical_authority"])
        self.assertEqual(receipt["purpose"], "training_corpus")
        self.assertEqual(target.generation, 0)
        self.assertEqual(len(target.local_corpus), 1)
        self.assertEqual(target.local_corpus[0]["purpose"], "training_corpus")
        self.assertEqual(target.local_corpus[0]["records"], 5)

        with self.assertRaisesRegex(ValueError, "purpose"):
            integrate_local_corpus(
                target, bundle, purpose="canonical_merge", approved=True)


if __name__ == "__main__":
    unittest.main()
