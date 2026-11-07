import unittest

from nanlmind_hermes.evidence import build_evidence_package
from nanlmind_hermes.transform_chain import preflight, postflight


def package(entries=None):
    snapshot = {
        "scratch_id": "test",
        "revision": 4,
        "entries": entries or [
            {"entry_id": "u1", "lane": "user", "writer": "user",
             "sequence": 1, "revision": "d" * 64, "payload": {"question": "What happened?"}},
            {"entry_id": "c1", "lane": "clocked_result", "writer": "clocked",
             "sequence": 2, "revision": "e" * 64, "payload": {
                 "endpoint": "extract", "status": "ok",
                 "provenance": {"input_sha256": "a" * 64},
                 "result": {"fields": {"claims": ["A claim"]}},
             }},
            {"entry_id": "e1", "lane": "evidence_atoms", "writer": "hermes",
             "sequence": 3, "revision": "f" * 64, "payload": {
                 "subject": "source", "predicate": "supports",
                 "object": "claim", "source_digest": "b" * 64,
                 "resolution": 5,
             }},
        ],
    }
    return build_evidence_package(snapshot, "What happened?")


class TransformChainTests(unittest.TestCase):
    def test_preflight_runs_m0_through_m5_in_order(self):
        value = package()
        result = preflight(value)
        self.assertEqual([x["pass"] for x in result["passes"]],
                         ["M0", "M1", "M2", "M3", "M4", "M5"])
        self.assertEqual(result["relation_counts"]["evidence_atoms"], 1)
        self.assertEqual(result["relation_counts"]["clocked_observations"], 1)
        self.assertEqual(result["relation_counts"]["typed_relations"], 2)
        self.assertTrue(all(len(x["input_sha256"]) == 64 for x in result["passes"]))

    def test_postflight_runs_m6_through_m9_and_never_promotes(self):
        value = package()
        prepared = preflight(value)
        response = {
            "schema": "nanlm.head-cortex-response",
            "request_sha256": value["package_sha256"],
            "generation": 7,
            "truth_state": "CONFLICT",
            "relations": [
                {"truth_state": "TRUE", "head_id": 5, "premises": ["e1"]},
                {"truth_state": "FALSE", "head_id": 9, "premises": ["e1"]},
            ],
            "abstentions": [],
            "provenance": {
                "read_only": True, "state_digest": "c" * 64,
                "source_entry_ids": ["e1"],
            },
        }
        result = postflight(value, response, prepared)
        self.assertEqual([x["pass"] for x in result["passes"]],
                         ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"])
        self.assertEqual(result["truth_state"], "CONFLICT")
        self.assertEqual(result["decision"], "abstain")
        self.assertFalse(result["canonical_mutation_allowed"])
        self.assertFalse(result["promotion_allowed"])

    def test_unknown_is_not_false_and_forces_abstention(self):
        value = package()
        prepared = preflight(value)
        response = {
            "request_sha256": value["package_sha256"],
            "truth_state": "UNKNOWN",
            "relations": [{"truth_state": "UNKNOWN", "head_id": 0}],
            "abstentions": [],
            "provenance": {"read_only": True, "source_entry_ids": []},
        }
        result = postflight(value, response, prepared)
        self.assertEqual(result["truth_state"], "UNKNOWN")
        self.assertEqual(result["decision"], "abstain")

    def test_true_result_remains_review_only_despite_inference_disclaimer(self):
        value = package()
        prepared = preflight(value)
        response = {
            "request_sha256": value["package_sha256"],
            "truth_state": "TRUE",
            "relations": [{"truth_state": "TRUE", "head_id": 5}],
            "abstentions": [{"reason": "model_output_is_inference_not_ground_truth"}],
            "provenance": {"read_only": True, "source_entry_ids": []},
        }
        result = postflight(value, response, prepared)
        self.assertEqual(result["decision"], "review_only")
        self.assertEqual(result["hard_abstention_count"], 0)
        self.assertFalse(result["canonical_mutation_allowed"])

    def test_m0_rejects_truth_policy_that_collapses_unknown_or_conflict(self):
        value = package()
        value["policy"]["unknown_is_not_false"] = False
        unsigned = dict(value)
        unsigned.pop("package_sha256")
        import hashlib, json
        value["package_sha256"] = hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "M0"):
            preflight(value)

    def test_m4_rejects_non_monotonic_scratch_revisions(self):
        value = package([
            {"entry_id": "a", "lane": "user", "sequence": 2, "revision": "a" * 64, "payload": {}},
            {"entry_id": "b", "lane": "clocked_result", "sequence": 1, "revision": "b" * 64, "payload": {}},
        ])
        with self.assertRaisesRegex(ValueError, "M4"):
            preflight(value)

    def test_m7_rejects_response_from_another_package(self):
        value = package()
        prepared = preflight(value)
        response = {
            "request_sha256": "d" * 64, "relations": [],
            "abstentions": [], "provenance": {"read_only": True},
        }
        with self.assertRaisesRegex(ValueError, "M7"):
            postflight(value, response, prepared)


if __name__ == "__main__":
    unittest.main()
