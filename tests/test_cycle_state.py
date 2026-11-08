"""Persistent gauge-preserving cycle-state tests."""
import unittest

from nanlmind_hermes.cycle_state import CycleState
from nanlmind_hermes.gauge_memory import project_cycle


class CycleStateTests(unittest.TestCase):
    def setUp(self):
        self.nodes = 8
        self.raw = {(first, second): complex(first + 1, second - 2)
                    for first in range(self.nodes)
                    for second in range(first + 1, self.nodes)}
        self.cycle = project_cycle(self.nodes, self.raw)

    def test_observation_projects_out_vertex_gradient(self):
        state = CycleState(self.nodes)
        state.observe(self.raw, rate=1.0)
        expected = project_cycle(self.nodes, self.raw)
        for edge, value in expected.items():
            self.assertAlmostEqual(abs(state.values[edge] - value), 0.0, places=10)

    def test_vertex_gauge_does_not_change_cycle_content(self):
        state = CycleState(self.nodes)
        state.observe(self.cycle, rate=1.0)
        before = dict(state.values)
        potential = [complex(index, -index / 3) for index in range(self.nodes)]
        shifted = state.with_gauge(potential)
        shifted.project()
        for edge, value in before.items():
            self.assertAlmostEqual(abs(shifted.values[edge] - value), 0.0, places=10)

    def test_snapshot_restore_preserves_state(self):
        state = CycleState(self.nodes, decay=0.9)
        state.observe(self.cycle, rate=0.7)
        snapshot = state.snapshot()
        restored = CycleState(self.nodes)
        restored.restore(snapshot)
        self.assertEqual(restored.values.keys(), state.values.keys())
        self.assertAlmostEqual(restored.energy(), state.energy(), places=10)

    def test_edge_order_is_normalised(self):
        state = CycleState(self.nodes)
        state.observe({(3, 1): 2 + 1j}, rate=1.0)
        self.assertEqual(set(state.values), {(1, 3)})


if __name__ == "__main__":
    unittest.main()
