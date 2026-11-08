import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.scratch_register import (
    DATA_PAGES,
    LAYER_NAMES,
    LOGICAL_BYTES,
    PAGE_BYTES,
    ScratchRegisterError,
    ScratchRegisterSpace,
    parse_register_address,
    register_address,
)


class ScratchRegisterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.space = ScratchRegisterSpace(Path(self.tmp.name) / "scratch")

    def test_one_gib_logical_space_and_sparse_allocation(self):
        stats = self.space.stats()
        self.assertEqual(stats["logical_bytes"], 1024 ** 3)
        self.assertEqual(stats["logical_bytes"], LOGICAL_BYTES)
        self.assertEqual(stats["page_bytes"], PAGE_BYTES)
        self.assertEqual(stats["data_pages"], DATA_PAGES)
        self.assertEqual(stats["allocated_pages"], 0)
        self.assertEqual(len(stats["layers"]), 10)

    def test_append_leaf_record_and_read_back(self):
        result = self.space.append(
            layer=5, head=17, leaf_id="leaf-1",
            payload={"subject": "copper", "predicate": "is", "object": "metal"},
        )
        self.assertEqual(result["layer"], 5)
        self.assertEqual(result["layer_name"], LAYER_NAMES[5])
        self.assertEqual(result["head"], 17)
        address = result["address"]
        head, page, offset = parse_register_address(address)
        self.assertEqual(head, 17)
        self.assertEqual(page, result["page"])
        self.assertEqual(offset, result["offset"])
        self.assertEqual(
            register_address(head, page, offset), address
        )
        records = list(self.space.records(layer="lexical", head=17))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["payload"]["subject"], "copper")
        self.assertEqual(records[0]["leaf_id"], "leaf-1")

    def test_records_are_durable_across_instances(self):
        self.space.append(layer=9, head=255, leaf_id="narrative-1",
                          payload={"event": "rain", "then": "wet"})
        reopened = ScratchRegisterSpace(Path(self.tmp.name) / "scratch")
        records = list(reopened.records(layer=9))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["layer_name"], "narrative")
        self.assertEqual(records[0]["head"], 255)

    def test_address_append_and_verify(self):
        first = self.space.append(layer=0, head=1, leaf_id="bit-1",
                                  payload={"truth": "and"})
        second = self.space.append_address(first["address"], {"truth": "or"})
        self.assertEqual(second["page"], first["page"])
        self.assertGreater(second["offset"], first["offset"])
        verification = self.space.verify()
        self.assertTrue(verification["ok"])
        self.assertEqual(verification["errors"], [])

    def test_tampered_page_fails_closed(self):
        result = self.space.append(layer=3, head=9, leaf_id="phon-1",
                                   payload={"onset": "k", "rime": "at"})
        page = self.space.pages[result["page"]]
        raw = bytearray(page.path.read_bytes())
        raw[-1] ^= 0xFF
        page.path.write_bytes(bytes(raw))
        verification = self.space.verify()
        self.assertFalse(verification["ok"])
        self.assertTrue(verification["errors"])
        with self.assertRaises(ScratchRegisterError):
            list(page.records())

    def test_bounds_and_address_validation(self):
        with self.assertRaises(ScratchRegisterError):
            self.space.append(layer=10, head=1, leaf_id="x", payload={})
        with self.assertRaises(ScratchRegisterError):
            self.space.append(layer=0, head=256, leaf_id="x", payload={})
        with self.assertRaises(ScratchRegisterError):
            parse_register_address("nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0")
        with self.assertRaises(ScratchRegisterError):
            register_address(1, DATA_PAGES, 256)

    def test_manifest_reports_layers_and_digests(self):
        self.space.append(layer=7, head=3, leaf_id="sentence-1",
                          payload={"text": "the sky is blue"})
        manifest = self.space.manifest()
        self.assertEqual(manifest["schema"], "nanlm.scratch-register-manifest/1")
        self.assertTrue(manifest["verify"]["ok"])
        self.assertEqual(manifest["stats"]["allocated_pages"], 1)
        self.assertIn("sentence", manifest["stats"]["layers"])


if __name__ == "__main__":
    unittest.main()
