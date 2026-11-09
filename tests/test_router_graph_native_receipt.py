"""A signed native-result statement is not a learning authorization."""
import copy
import unittest

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import (
    _DOMAIN as SENDER_DOMAIN, verify_graph_sender_attestation,
)
from nanlmind_hermes.router_graph_evidence import audit_graph_revision_evidence
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_native_input import pack_native_graph_claim_input
from nanlmind_hermes.router_graph_native_receipt import (
    _RECEIPT_DOMAIN, SCHEMA, NativeGraphReceiptError, _canonical,
    verify_native_graph_claim_receipt,
)
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from test_router_graph_history import edge, fixture
from test_router_graph_walk import state_page


class Leaf:
    leaf_id = "nml:evidence:1"
    version = 1

    def digest(self):
        return "f" * 64


class NativeGraphReceiptTests(unittest.TestCase):
    def setUp(self):
        self.now = 1750000000
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        self.revisions = [
            dict(edge("a"), relation="supports"),
            dict(edge("b"), relation="refutes"),
        ]
        self.pages = [
            state_page(self.chart, self.revisions, i) for i in range(2)
        ]
        self.history = [
            fixture(self.chart, self.revisions, history_offset=i)
            for i in range(2)
        ]
        self.verifier = RouterGraphHistoryVerifier(self.chart)
        for page in self.pages:
            self.verifier.add_state_page(page)
        for page in self.history:
            self.verifier.add_history_page(page)
        self.assertTrue(self.verifier.status()["history_verified"])
        self.sender_key = Ed25519PrivateKey.generate()
        self.worker_key = Ed25519PrivateKey.generate()
        self.rogue_key = Ed25519PrivateKey.generate()
        self.nonce = "a" * 32
        self.guest_hash = "b" * 64
        self.lease_hash = "c" * 64
        self.sender_attestation = self.attestation()

    @staticmethod
    def public(key):
        return key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw).hex()

    def attestation(self):
        header = self.verifier.state._header
        body = {
            "schema": "nanrouter.graph-sender-attestation/1",
            "router_id": "router-one", "key_id": "router-key",
            "graph_commitment_sha256": self.verifier.state._commitment,
            "topology_sha256": header["topology_sha256"],
            "parent_root_sha256": header["parent_root_sha256"],
            "chart_generation": header["chart_generation"],
            "history_sha256": header["history_sha256"],
            "nonce": "d" * 32, "issued_at": self.now,
        }
        return {
            **body,
            "signature_hex": self.sender_key.sign(
                SENDER_DOMAIN + _canonical(body)).hex(),
            "signed_by_router": True,
            "native_mth_receipt": False,
            "mathematical_truth_promoted": False,
        }

    def receipt(self, signer=None, **changes):
        staged = pack_native_graph_claim_input(
            self.verifier, self.pages, self.history)
        sender = verify_graph_sender_attestation(
            self.verifier, self.sender_attestation,
            expected_router_id="router-one", expected_key_id="router-key",
            public_key_hex=self.public(self.sender_key), now=self.now)
        evidence = audit_graph_revision_evidence(
            self.verifier, self.pages, self.history,
            resolve_trusted_leaf=lambda _id, _version: Leaf())
        body = {
            "schema": SCHEMA,
            "worker_id": "native-worker-one",
            "key_id": "worker-key",
            "graph_envelope_sha256": staged["envelope_sha256"],
            "word_digest_sha256": staged["word_digest_sha256"],
            "provenance_sha256": staged["provenance_sha256"],
            "graph_commitment_sha256": staged["graph_commitment_sha256"],
            "history_sha256": staged["history_sha256"],
            "chart_generation": staged["chart_generation"],
            "router_attestation_sha256": sender["attestation_sha256"],
            "evidence_reference_digest_sha256":
                evidence["evidence_reference_digest_sha256"],
            "guest_image_sha256": self.guest_hash,
            "rule_epoch": 17,
            "lease_sha256": self.lease_hash,
            "nonce": self.nonce,
            "query_src": 7, "query_dst": 265, "query_layer": 4,
            "guest_status": 1, "result_mask": 3,
            "issued_at": self.now,
        }
        body.update(changes)
        signer = signer or self.worker_key
        return {
            **body,
            "signature_hex": signer.sign(
                _RECEIPT_DOMAIN + _canonical(body)).hex(),
        }

    def check(self, receipt=None, **kwargs):
        args = dict(
            expected_router_id="router-one",
            expected_router_key_id="router-key",
            pinned_router_public_key_hex=self.public(self.sender_key),
            resolve_trusted_leaf=lambda _id, _version: Leaf(),
            expected_worker_id="native-worker-one",
            expected_worker_key_id="worker-key",
            pinned_worker_public_key_hex=self.public(self.worker_key),
            expected_guest_image_sha256=self.guest_hash,
            expected_rule_epoch=17, expected_lease_sha256=self.lease_hash,
            expected_nonce=self.nonce, expected_query=(7, 265, 4),
            now=self.now,
        )
        args.update(kwargs)
        return verify_native_graph_claim_receipt(
            self.verifier, self.pages, self.history,
            self.sender_attestation, self.receipt() if receipt is None else receipt,
            **args)

    def test_all_independent_gates_are_read_only_and_not_learning(self):
        before = bytes(self.chart.bytes)
        verdict = self.check()
        self.assertEqual(verdict["candidate_mask"], 3)
        for key in ("graph_history_verified", "sender_authenticated",
                    "evidence_digest_integrity_verified",
                    "native_result_worker_attested"):
            self.assertTrue(verdict[key])
        for key in ("nonce_replay_protected", "lease_dispatch_authorized",
                    "native_execution_independently_verified",
                    "native_admission_verified", "learning_admitted",
                    "mathematical_truth_promoted"):
            self.assertFalse(verdict[key])
        self.assertEqual(bytes(self.chart.bytes), before)

    def test_cryptographic_pins_and_forged_worker_result_reject(self):
        for receipt, settings in (
            (self.receipt(signer=self.rogue_key), {}),
            (self.receipt(), {"pinned_worker_public_key_hex":
                              self.public(self.rogue_key)}),
            (self.receipt(), {"pinned_router_public_key_hex":
                              self.public(self.rogue_key)}),
            (self.receipt(), {"expected_worker_id": "foreign-worker"}),
        ):
            with self.subTest(settings=settings), self.assertRaises(Exception):
                self.check(receipt, **settings)
        tampered = self.receipt()
        tampered["result_mask"] = 0
        with self.assertRaises(NativeGraphReceiptError):
            self.check(tampered)
        tampered = self.receipt()
        tampered["learning_admitted"] = True
        with self.assertRaises(NativeGraphReceiptError):
            self.check(tampered)

    def test_signed_but_unbound_receipts_reject_even_with_valid_signature(self):
        variants = {
            "graph_envelope_sha256": "0" * 64,
            "word_digest_sha256": "0" * 64,
            "provenance_sha256": "0" * 64,
            "guest_image_sha256": "0" * 64,
            "lease_sha256": "0" * 64,
            "graph_commitment_sha256": "0" * 64,
            "router_attestation_sha256": "0" * 64,
            "evidence_reference_digest_sha256": "0" * 64,
            "rule_epoch": 16,
            "nonce": "0" * 32,
            "query_dst": 266,
            "guest_status": 0,
            "result_mask": 4,
            "issued_at": self.now - 301,
            "chart_generation": self.chart.snapshot_generation + 1,
        }
        for field, wrong in variants.items():
            with self.subTest(field=field), self.assertRaises(NativeGraphReceiptError):
                self.check(self.receipt(**{field: wrong}))

    def test_history_evidence_and_chart_changes_reject_after_signing(self):
        receipt = self.receipt()
        pages = copy.deepcopy(self.pages)
        pages[0]["entries"][0]["relation"] = "refutes"
        with self.assertRaises(Exception):
            verify_native_graph_claim_receipt(
                self.verifier, pages, self.history,
                self.sender_attestation, receipt,
                expected_router_id="router-one",
                expected_router_key_id="router-key",
                pinned_router_public_key_hex=self.public(self.sender_key),
                resolve_trusted_leaf=lambda _id, _version: Leaf(),
                expected_worker_id="native-worker-one",
                expected_worker_key_id="worker-key",
                pinned_worker_public_key_hex=self.public(self.worker_key),
                expected_guest_image_sha256=self.guest_hash,
                expected_rule_epoch=17, expected_lease_sha256=self.lease_hash,
                expected_nonce=self.nonce, expected_query=(7, 265, 4),
                now=self.now)
        with self.assertRaises(Exception):
            self.check(receipt, resolve_trusted_leaf=lambda _id, _v: None)
        self.chart.snapshot_generation += 1
        with self.assertRaises(Exception):
            self.check(receipt)


if __name__ == "__main__":
    unittest.main()
