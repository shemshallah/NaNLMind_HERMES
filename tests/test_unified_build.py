"""Contract tests for the read-only NaNLMind unified-build adapter."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanlmind_hermes import unification


def contract() -> dict:
    return {
        "schema": "nanlm.unified-build/1",
        "canonical_repository": "NaNLMind",
        "canonical_binary": "build/nanlm-universal",
        "canonical_state_owner": "NaNLMind",
        "transport_plane": "NaNROUTER",
        "adapter_plane": "NaNLMind_HERMES",
        "topology": {
            "addresses": 65536,
            "heads": 256,
            "triad_member_heads": 255,
            "field_head": 255,
            "triads": 85,
            "triad_size": 3,
            "mirror_pairs": 32640,
            "canonical_cells": 32896,
            "chart_bytes": 8388608,
            "hypermatrix_rows": 32640,
            "hypermatrix_row_bits": 256,
            "resolution_layers": 10,
            "transform_passes": 10,
            "runner_slots": 6,
            "active_runner_slots": 1,
        },
        "role_views": {
            "physical_heads": 256,
            "mp_writer_eligible_heads": 255,
            "boolean_curriculum_heads": 250,
            "nannet_root_control_head": 0,
            "bsr_control_head_indices": [0, 4, 5, 6, 7, 8],
            "triad_member_heads": 255,
            "triad_field_head": 255,
            "nannet_port_is_native_head": False,
            "role_overlay_may_reassign_native_heads": False,
        },
        "authority": {
            "topology": "NaNLMind",
            "native_build": "NaNLMind",
            "canonical_state": "NaNLMind",
            "transport_and_control_plane": "NaNROUTER",
            "hermes_role": "read-only adapter and candidate producer",
            "hermes_may_commit_canonical_state": False,
            "nanrouter_may_copy_or_fork_native_universe": False,
        },
    }


NATIVE_LINE = (
    "unified_build schema=nanlm.unified-build/1 canonical_repository=NaNLMind "
    "canonical_binary=build/nanlm-universal heads=256 triad_member_heads=255 "
    "field_head=255 triads=85 triad_size=3 mp_writer_eligible_heads=255 "
    "boolean_curriculum_heads=250 nannet_root_control_head=0 bsr_control_heads=6 "
    "mirror_pairs=32640 canonical_cells=32896 chart_bytes=8388608 "
    "hypermatrix_rows=32640 hypermatrix_row_bits=256 resolution_layers=10 "
    "transform_passes=10 runner_slots=6 active_runner_slots=1"
)


class ContractCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / unification.CONTRACT_RELATIVE
        self.path.parent.mkdir(parents=True)
        self.write(contract())

    def write(self, document: dict) -> None:
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def fake_native(self, line: str = NATIVE_LINE) -> Path:
        binary = self.root / unification.BINARY_RELATIVE
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text(
            "#!/bin/sh\nprintf '%s\\n' " + json.dumps(line) + "\n",
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | 0o111)
        return binary


class LoadTests(ContractCase):
    def test_load_returns_canonical_digest_and_root(self):
        loaded = unification.load(self.path)
        self.assertEqual(loaded["root"], self.root)
        self.assertEqual(loaded["document"]["canonical_repository"], "NaNLMind")
        self.assertEqual(len(loaded["sha256"]), 64)

    def test_missing_contract_is_unavailable_not_silently_defaulted(self):
        result = unification.status(self.root / "missing.json")
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["hermes_is_canonical"])

    def test_wrong_topology_is_rejected(self):
        document = contract()
        document["topology"]["heads"] = 255
        self.write(document)
        with self.assertRaisesRegex(unification.UnificationError, "heads"):
            unification.load(self.path)

    def test_authority_cannot_grant_hermes_canonical_write(self):
        document = contract()
        document["authority"]["hermes_may_commit_canonical_state"] = True
        self.write(document)
        with self.assertRaisesRegex(unification.UnificationError, "HERMES"):
            unification.load(self.path)

    def test_router_role_overlay_cannot_reassign_native_heads(self):
        document = contract()
        document["role_views"]["role_overlay_may_reassign_native_heads"] = True
        self.write(document)
        with self.assertRaisesRegex(unification.UnificationError, "role overlay"):
            unification.load(self.path)

    def test_role_views_are_required(self):
        document = contract()
        del document["role_views"]
        self.write(document)
        with self.assertRaisesRegex(unification.UnificationError, "role_views"):
            unification.load(self.path)

    def test_environment_root_finds_the_contract(self):
        with mock.patch.dict(os.environ, {"NANLMIND_ROOT": str(self.root)}):
            self.assertEqual(unification.find_contract(), self.path)


@unittest.skipIf(os.name == "nt", "shell fixture is POSIX-only")
class NativeTests(ContractCase):
    def test_green_native_command_is_reported(self):
        self.fake_native()
        result = unification.status(self.path)
        self.assertEqual(result["status"], "green")
        self.assertEqual(result["native"]["status"], "green")
        self.assertEqual(result["native"]["triad_member_heads"], "255")
        self.assertFalse(result["hermes_is_canonical"])

    def test_native_mismatch_is_rejected(self):
        self.fake_native(NATIVE_LINE.replace("heads=256", "heads=255"))
        result = unification.status(self.path)
        self.assertEqual(result["status"], "green")
        self.assertEqual(result["native"]["status"], "rejected")
        self.assertIn("heads", result["native"]["mismatches"])

    def test_missing_native_binary_is_not_a_contract_failure(self):
        result = unification.status(self.path)
        self.assertEqual(result["status"], "green")
        self.assertEqual(result["native"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
