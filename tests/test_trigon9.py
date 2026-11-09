"""TRIGON-9 reference-model tests (optional PyTorch path)."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "experiments" / "round042"
for path in (ROUND, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import torch
except ImportError:  # pragma: no cover - pure path must remain importable
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TrigonReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from trigon import TrigonLM
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.model = TrigonLM(vocab=259, d_model=32, n_heads=2, n_layers=10)

    def test_basis_constants(self) -> None:
        from trigon import CYCLE_COUNT, HEAD_COUNT, PAIR_COUNT
        self.assertEqual(HEAD_COUNT, 256)
        self.assertEqual(PAIR_COUNT, 32640)
        self.assertEqual(CYCLE_COUNT, 32385)

    def test_gauge_is_exactly_invariant(self) -> None:
        indices = torch.arange(0, 2000)
        signs = torch.ones(2000)
        report = self.model.cycle.audit_gauge(indices, signs, scale=3.0)
        self.assertTrue(report["exact"])
        self.assertLess(report["max_cycle_phase_error"], 1e-4)

    def test_genome_rejects_unbounded_values(self) -> None:
        from trigon import Genome
        with self.assertRaises(ValueError):
            Genome(head_k=2)
        with self.assertRaises(ValueError):
            Genome(order=(0, 0, 2, 3, 4, 5, 6, 7, 8, 9))
        genome = Genome().mutated(scale=0.1)
        self.assertEqual(sorted(genome.order), list(range(10)))
        self.assertGreaterEqual(genome.head_k, 3)
        self.assertLessEqual(genome.head_k, 12)

    def test_forward_backward_and_memory_ablation(self) -> None:
        ids = torch.randint(0, 256, (2, 6))
        labels = torch.randint(0, 256, (2, 6))
        out = self.model(ids, labels=labels)
        self.assertEqual(out["logits"].shape, (2, 6, 259))
        self.assertTrue(torch.isfinite(out["logits"]).all())
        out["loss"].backward()
        self.assertIsNotNone(self.model.cycle.edge_phase.grad)
        self.model.ablate_memory = True
        ablated = self.model(ids)
        self.model.ablate_memory = False
        self.assertEqual(ablated["logits"].shape, (2, 6, 259))

    def test_relation_task_write_read_shape(self) -> None:
        from trigon import RelationTask, head_for_name
        task = RelationTask(d_model=32)
        left = torch.randint(0, 256, (3, 12))
        right = torch.randint(0, 256, (3, 12))
        phase, encoded = task(left, right, self.model.cycle)
        heads_l = torch.tensor([head_for_name("dog")] * 3)
        heads_r = torch.tensor([head_for_name("animal")] * 3)
        logits = task.classify(
            left, right, phase, self.model.cycle, heads_l, heads_r, encoded
        )
        self.assertEqual(logits.shape, (3, 4))

    def test_nannet_p_never_enables_merge(self) -> None:
        from nannet_p import NannetP
        from train import make_batch
        import random
        nannet = NannetP(self.model, seed=5)
        ids, labels = make_batch(
            [b"a small test definition of a thing"] * 2, random.Random(5), 2, 6
        )
        state = nannet.state(self.model, 4.0, 0.0, 0)
        record = nannet.step(self.model, state, [(ids, labels)], count=2,
                             margin=1e-6)
        self.assertFalse(record["canonical_merge"])
        self.assertEqual(record["mode"], "local-unverified")
        self.assertFalse(record["live_nannet_route"])
        self.assertIn("accepted", record)


if __name__ == "__main__":
    unittest.main()
