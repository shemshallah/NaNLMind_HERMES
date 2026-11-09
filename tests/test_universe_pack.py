"""N16PACK1: secure single-file offline chart + typed SQLite backups."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.chart_storage import save_chart_snapshot
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import (
    ROOT_OFFSET, attach_universe_root,
)
from nanlmind_hermes.universe_pack import (
    MAGIC, HEADER, UniversePackError, export_n16pack, inspect_n16pack,
)


class PackTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.state = self.base / "state"
        self.state.mkdir()
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        self.parent = self.chart.root_hash
        self.generation = save_chart_snapshot(
            self.state, self.parent, self.chart.bytes, 0,
        )
        self.mc = self.base / "mc0.db"
        self.bs = self.base / "bs0.db"
        for db, name in ((self.mc, "MC0DB"), (self.bs, "BS0DB")):
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "CREATE TABLE math_schema("
                    "store TEXT PRIMARY KEY,version INTEGER NOT NULL)"
                )
                conn.execute("INSERT INTO math_schema VALUES (?,?)", (name,1))
                conn.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY, data TEXT)")
                conn.execute("INSERT INTO sample(data) VALUES (?)", (name,))
        self.dest = self.base / "single_universe.n16pack"

    def export(self, **changes):
        params = dict(state_dir=self.state, parent_root_sha256=self.parent,
                      expected_chart_generation=self.generation,
                      mc0db=self.mc, bs0db=self.bs, destination=self.dest)
        params.update(changes)
        return export_n16pack(**params)

    def _forge_sections(self, replacements, filename):
        """Rewrite manifest and hashes, modelling an adversarial pack author."""
        archive = self.dest.read_bytes()
        magic, version, manifest_size, _ = HEADER.unpack(archive[:HEADER.size])
        manifest = json.loads(archive[HEADER.size:HEADER.size + manifest_size])
        offset = HEADER.size + manifest_size
        contents = {}
        for spec in manifest["sections"]:
            size = spec["length"]
            contents[spec["name"]] = archive[offset:offset + size]
            offset += size
        for name, payload in replacements.items():
            contents[name] = bytes(payload)
        for spec in manifest["sections"]:
            payload = contents[spec["name"]]
            spec["length"] = len(payload)
            spec["sha256"] = hashlib.sha256(payload).hexdigest()
        encoded = json.dumps(
            manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        forged = (HEADER.pack(magic, version, len(encoded),
                              hashlib.sha256(encoded).digest())
                  + encoded + b"".join(
                      contents[spec["name"]] for spec in manifest["sections"]
                  ))
        result = self.base / filename
        result.write_bytes(forged)
        return result, contents

    def test_reject_digest_consistent_swapped_typed_store(self):
        self.export()
        _, payloads = self._forge_sections({}, "copy.n16pack")
        # Valid, integrity-checked SQLite bytes with the wrong store ID.
        swapped, _ = self._forge_sections(
            {"mc0db": payloads["bs0db"], "bs0db": payloads["mc0db"]},
            "wrong-store.n16pack",
        )
        with self.assertRaisesRegex(UniversePackError, "MC0DB.*schema"):
            inspect_n16pack(swapped)
        self.assertTrue(inspect_n16pack(self.dest)["verified"])

    def test_reject_digest_consistent_corrupt_sqlite_header(self):
        self.export()
        _, payloads = self._forge_sections({}, "copy.n16pack")
        corrupted = bytearray(payloads["mc0db"])
        self.assertTrue(corrupted.startswith(b"SQLite format 3\0"))
        # SQLite page-size field is at offsets 16..17: 3 is invalid.
        corrupted[16:18] = bytes((0, 3))
        forged, _ = self._forge_sections(
            {"mc0db": corrupted}, "invalid-page-size.n16pack",
        )
        with self.assertRaises(UniversePackError):
            inspect_n16pack(forged)

    def test_reject_hash_consistent_chart_header_geometry_tamper(self):
        self.export()
        _, payloads = self._forge_sections({}, "copy.n16pack")
        chart = bytearray(payloads["chart"])
        chart[0] ^= 0x01  # Mounted N16ROOT1 remains intact.
        forged, _ = self._forge_sections(
            {"chart": chart}, "corrupt-chart-header.n16pack",
        )
        with self.assertRaisesRegex(UniversePackError, "chart header"):
            inspect_n16pack(forged)

    def test_roundtrip_single_file_chart_and_databases(self):
        result = self.export()
        self.assertTrue(result["verified"])
        self.assertFalse(result["import_to_live_allowed"])
        self.assertFalse(result["cross_store_atomicity_attested"])
        self.assertFalse(result["native_execution_verified"])
        self.assertEqual(result["head_addresses"], 256)
        self.assertEqual(result["mirror_points"], 32640)
        self.assertEqual(result["layers"], 10)
        self.assertEqual(result["chart_generation"], 1)
        self.assertEqual([s["name"] for s in result["sections"]],
                         ["chart","mc0db","bs0db"])
        self.assertEqual(result["sections"][0]["length"], 8*1024*1024)
        self.assertEqual(self.dest.read_bytes()[:8], MAGIC)
        self.assertEqual(self.dest.stat().st_mode & 0o777, 0o600)
        self.assertEqual(result, inspect_n16pack(self.dest))

    def test_corruption_manifest_payload_truncation_and_trailing_denied(self):
        self.export()
        archive = self.dest.read_bytes()
        for name, corrupted in (
            ("manifest", archive[:HEADER.size] +
             bytes([archive[HEADER.size] ^ 1]) + archive[HEADER.size+1:]),
            ("chart", archive[:8000] + bytes([archive[8000] ^ 1]) + archive[8001:]),
            ("truncated", archive[:-10]),
            ("appended", archive + b"unlisted"),
        ):
            target = self.base / (name + ".n16pack")
            target.write_bytes(corrupted)
            with self.subTest(name=name), self.assertRaises(UniversePackError):
                inspect_n16pack(target)

    def test_stale_checkpoint_wrong_root_and_destination_reuse(self):
        with self.assertRaises(UniversePackError):
            self.export(expected_chart_generation=2)
        with self.assertRaises(UniversePackError):
            self.export(parent_root_sha256="f"*64)
        self.export()
        old = hashlib.sha256(self.dest.read_bytes()).hexdigest()
        with self.assertRaises(UniversePackError):
            self.export()
        self.assertEqual(hashlib.sha256(self.dest.read_bytes()).hexdigest(), old)

    def test_reject_uninitialized_root_corrupt_root_swapped_db_and_bad_schema(self):
        chart = Chart8MiB.genesis()
        fresh_state = self.base / "unmounted"
        save_chart_snapshot(fresh_state, chart.root_hash, chart.bytes, 0)
        with self.assertRaises(UniversePackError):
            self.export(state_dir=fresh_state, parent_root_sha256=chart.root_hash)
        corrupted = Chart8MiB.genesis()
        attach_universe_root(corrupted)
        corrupted.bytes[ROOT_OFFSET + 8] ^= 1
        damaged_state = self.base / "damaged"
        save_chart_snapshot(damaged_state, corrupted.root_hash, corrupted.bytes, 0)
        with self.assertRaises(UniversePackError):
            self.export(state_dir=damaged_state,
                        parent_root_sha256=corrupted.root_hash)
        with self.assertRaises(UniversePackError):
            self.export(mc0db=self.bs,bs0db=self.mc)
        invalid = self.base / "foreign.db"
        with sqlite3.connect(invalid) as db:
            db.execute("CREATE TABLE unrelated(a TEXT)")
        with self.assertRaises(UniversePackError):
            self.export(mc0db=invalid)
        self.assertFalse(self.dest.exists())

    def test_wal_backups_are_readable_and_dont_mutate_sources(self):
        for p in (self.mc,self.bs):
            with sqlite3.connect(p) as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("INSERT INTO sample(data) VALUES ('new')")
        digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (self.mc,self.bs)}
        self.export()
        self.assertTrue(inspect_n16pack(self.dest)["verified"])
        for p in (self.mc,self.bs):
            with sqlite3.connect(p) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM sample").fetchone()[0], 2)
                self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")


if __name__ == "__main__":
    unittest.main()
