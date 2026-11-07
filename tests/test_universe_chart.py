"""Universe chart mapping and real temporal rail tests."""
import struct
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.temporal_lm import TemporalMirrorLM
from nanlmind_hermes.universe_chart import (
    ADDRESS_COUNT,
    CANONICAL_COUNT,
    HEAD_COUNT,
    MIRROR_PAIR_COUNT,
    NANNET_LINK,
    OFF_EXTENSION,
    PAGE_BYTES,
    REGISTER_MAGIC,
    RING_ENDPOINT,
    RUNNER_ADDRESS,
    Chart8MiB,
    LocalRunner,
    TemporalRegister,
    address_of,
    canonical_address,
    load_chart,
    make_word,
    reverse16,
    reverse8,
    word_payload,
)


class Context:
    def __init__(self, root):
        self.config = {"state_dir": str(root), "chat_mode": "native", "chat_learning": False}

    def get_config(self, key, default=None):
        return self.config.get(key, default)


def build_universe(pair=(17, 136), s_plus=11, s_minus=22) -> bytes:
    fixed = [((a << 8) | reverse8(a)) for a in range(256)]
    seen = set(fixed)
    pairs = []
    for a in range(256):
        for b in range(a + 1, 256):
            canonical = canonical_address(a, b)
            if canonical in seen:
                continue
            seen.add(canonical)
            pairs.append(canonical)
    canon = fixed + pairs
    assert len(canon) == CANONICAL_COUNT
    index = {address: position for position, address in enumerate(canon)}
    cells = bytearray(CANONICAL_COUNT * 64)
    cell = index[canonical_address(*pair)] * 64
    struct.pack_into("<Q", cells, cell, make_word(s_plus))
    struct.pack_into("<Q", cells, cell + 8, make_word(s_minus))
    heads = bytearray(HEAD_COUNT * 32)
    for head in pair:
        heads[head * 32 + 20] = 255
    page = b"NLMU0001"
    page += struct.pack("<Q", 42)
    page += struct.pack("<6I", CANONICAL_COUNT, ADDRESS_COUNT, CANONICAL_COUNT,
                        HEAD_COUNT, CANONICAL_COUNT, MIRROR_PAIR_COUNT)
    page += bytes(cells)
    page += b"\x00" * (ADDRESS_COUNT * 32)
    page += b"\x00" * 8_192
    page += b"\x00" * (HEAD_COUNT * 1_024)
    page += b"\x00" * 65_536
    page += b"\x00" * (CANONICAL_COUNT * 48)
    page += bytes(heads)
    page += b"\x00" * 8_192
    page += b"".join(struct.pack("<H", address) for address in canon)
    page += b"\x00" * (MIRROR_PAIR_COUNT * 2)
    assert len(page) == 6_264_872
    container = b"NLMGLOB3" + struct.pack("<I", 1)
    container += bytes([0, 0, 0, 0, 0])
    container += struct.pack("<Q", len(page)) + page
    container += b"R" * 32
    return container


def write_universe(root: Path, **kwargs) -> Path:
    path = root / "universe" / "native" / "global_universe.nmg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_universe(**kwargs))
    return path


class ChartTests(unittest.TestCase):
    def test_chart_assembles_eight_mib_and_reads_real_rails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_universe(root)
            chart = load_chart(root)
            self.assertIsInstance(chart, Chart8MiB)
            self.assertEqual(chart.stats()["page_bytes"], PAGE_BYTES)
            self.assertEqual(chart.verify(), [])
            self.assertEqual(word_payload(chart.read_rail(17, 136, 0)), 11)
            self.assertEqual(word_payload(chart.read_rail(17, 136, 1)), 22)
            canonical = canonical_address(17, 136)
            self.assertEqual(canonical, min(address_of(17, 136), address_of(136, 17)))
            self.assertEqual(reverse16(canonical), address_of(136, 17))
            self.assertIs(load_chart(root), chart)  # held in the in-memory cache

    def test_register_commits_on_rails_without_mutating_the_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_universe(root)
            before = path.read_bytes()
            ctx = Context(root)
            register = TemporalRegister.open(ctx)
            self.assertTrue(register.mapped)
            self.assertEqual(register.pair, (17, 136))
            status = register.status()
            self.assertEqual(status["rails"]["canonical_hex"],
                             "%04X" % canonical_address(17, 136))
            self.assertEqual(status["rails"]["mirror_hex"],
                             "%04X" % reverse16(canonical_address(17, 136)))
            self.assertTrue(status["rails"]["s_plus"].startswith(
                RUNNER_ADDRESS + "::pair/17/136/s+"))
            self.assertTrue(status["rails"]["s_minus"].startswith(
                RUNNER_ADDRESS + "::pair/17/136/s-"))
            self.assertEqual(register.commit(5), 1)
            self.assertEqual(register.frame, 1)
            self.assertEqual(word_payload(register.chart.read_rail(17, 136, 0)),
                             register.forward_payload)
            self.assertEqual(word_payload(register.chart.read_rail(17, 136, 1)),
                             register.back_payload)
            self.assertEqual(register.state[5], 1)
            self.assertEqual(register.state[reverse8(5)], 1)
            self.assertEqual(register.chart.bytes[OFF_EXTENSION:OFF_EXTENSION + 8],
                             REGISTER_MAGIC)
            register.rollback()
            self.assertEqual(register.forward_payload, register.back_payload)
            self.assertTrue(register.frame_meet(0, 3, 6, -3)["ok"])
            self.assertFalse(register.frame_meet(0, 3, 7, -3)["ok"])
            self.assertEqual(path.read_bytes(), before)  # native file never mutated

    def test_local_runner_loads_registers_and_links_through_hw2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_universe(root)
            runner = LocalRunner.open(Context(root))
            status = runner.status()
            self.assertEqual(status["runner"], "hermes-local-cpu")
            self.assertEqual(status["address"], "nan://HW2:BS0:MC0:SC0:GM0")
            self.assertEqual(status["nannet_link"], "HW2:0")
            self.assertEqual(status["ring_endpoint"], "BS0:0::HW0:0")
            self.assertEqual(status["binary_translation"], "NaN16DE")
            self.assertFalse(status["uses_system_ring_for_compute"])
            self.assertEqual(status["chart_bytes"], PAGE_BYTES)
            self.assertGreater(status["local_memory_bytes"], PAGE_BYTES)
            self.assertTrue(status["register"]["mapped"])
            self.assertEqual(runner.commit(9), 1)
            self.assertEqual(runner.register.state[9], 1)
            frame = int.from_bytes(
                runner.chart.bytes[OFF_EXTENSION + 8:OFF_EXTENSION + 16], "little")
            self.assertEqual(frame, runner.register.frame)

    def test_register_falls_back_without_a_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            register = TemporalRegister.open(Context(Path(tmp)))
            self.assertTrue(register.mapped)
            self.assertEqual(register.status()["chart"]["mode"], "genesis")
            self.assertEqual(register.status()["chart"]["page_bytes"], PAGE_BYTES)
            self.assertEqual(register.commit(3), 1)
            self.assertEqual(register.state[3], 1)

    def test_plugin_boot_instantiates_local_runner_before_training(self):
        from nanlmind_hermes import register as register_plugin

        class BootContext:
            def __init__(self, root):
                self.config = {"state_dir": str(root), "chat_mode": "native"}
                self.tools = {}
                self.command = None

            def get_config(self, key, default=None):
                return self.config.get(key, default)

            def register_tool(self, name, toolset=None, handler=None, schema=None):
                self.tools[name] = handler

            def register_command(self, name, handler, description=None):
                self.command = handler

        with tempfile.TemporaryDirectory() as tmp:
            ctx = BootContext(Path(tmp))
            register_plugin(ctx)
            runner = LocalRunner.open(ctx)
            status = runner.status()
            self.assertEqual(status["runner"], "hermes-local-cpu")
            self.assertEqual(status["chart"]["mode"], "genesis")
            self.assertEqual(status["chart_bytes"], PAGE_BYTES)
            self.assertFalse(status["uses_system_ring_for_compute"])
            self.assertIs(LocalRunner.open(ctx), runner)  # held in memory
            self.assertEqual(runner.commit(4), 1)
            self.assertGreaterEqual(runner.register.frame, 1)

    def test_runner_rebinds_to_trained_chart_and_keeps_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = Context(root)
            runner = LocalRunner.open(ctx)
            self.assertEqual(runner.chart.stats()["mode"], "genesis")
            runner.commit(11)
            frame = runner.register.frame
            write_universe(root)
            refreshed = LocalRunner.open(ctx)
            self.assertIs(refreshed, runner)
            self.assertEqual(refreshed.chart.stats()["mode"], "trained")
            self.assertEqual(refreshed.register.frame, frame)

    def test_temporal_model_uses_mapped_register_and_keeps_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_universe(root)
            ctx = Context(root)
            model = TemporalMirrorLM.load(ctx)
            self.assertIsNotNone(model.register)
            self.assertTrue(model.register.mapped)
            model.observe("A river is water.")
            model.observe("Water can flow.")
            prediction = model.generate("What is a river?",
                                        allowed_seed={"river", "water", "flow"})
            self.assertTrue(prediction["text"])
            self.assertTrue(prediction["register"]["mapped"])
            self.assertGreaterEqual(prediction["register_frames"], 1)
            self.assertGreaterEqual(prediction["register"]["frame"], 1)
            self.assertEqual(prediction["runner"]["nannet_link"], "HW2:0")
            self.assertEqual(prediction["runner"]["ring_endpoint"], "BS0:0::HW0:0")
            self.assertFalse(prediction["runner"]["uses_system_ring_for_compute"])
            model.save(ctx)
            self.assertTrue((root / "language" / "temporal_register.json").exists())
            reloaded = TemporalMirrorLM.load(ctx)
            self.assertGreaterEqual(reloaded.register.frame, 1)


if __name__ == "__main__":
    unittest.main()
