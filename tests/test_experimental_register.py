import hashlib
import tempfile
import unittest
from pathlib import Path
from nanlmind_hermes import experimental_register as tree


class Context:
    def __init__(self, root):
        self.root = root

    def get_config(self, key, default=None):
        return str(self.root) if key == "state_dir" else default


class ExperimentalRegisterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = Context(Path(self.tmp.name))

    def test_layered_tree_and_digest_logged_optimistic_mutation(self):
        initial = tree.inspect(self.ctx)
        self.assertEqual(initial["heads"], 255)
        self.assertFalse(initial["populated"])
        a = tree.mutate(self.ctx, 1, "vector123", "coordinate", "010203")
        self.assertEqual(a["revision"], 1)
        self.assertEqual(a["sha256"], hashlib.sha256(b"\x01\x02\x03").hexdigest())
        self.assertFalse(a["canonical_merged"])
        self.assertFalse(a["native_nan16_execution"])
        self.assertEqual(tree.inspect(self.ctx, 1)["branches"][0]["branch"], "vector123")
        self.assertEqual(tree.inspect(self.ctx, 1, "vector123")["leaves"][0]["leaf"], "coordinate")
        self.assertEqual(tree.inspect(self.ctx, 1, "vector123", "coordinate")["data_hex"], "010203")
        with self.assertRaises(tree.ScratchError):
            tree.mutate(self.ctx, 1, "vector123", "coordinate", "00", 0)
        b = tree.mutate(self.ctx, 1, "vector123", "coordinate", "fefe", 1)
        self.assertEqual(b["revision"], 2)
        self.assertEqual(len(tree.audit(self.ctx)["operations"]), 2)
        tree.mutate(self.ctx, 255, "hyperedge1", "relations", "ff")
        self.assertEqual(len(tree.inspect(self.ctx)["populated"]), 2)

    def test_reject_invalid_head_branch_leaf_and_oversized_payload(self):
        for head in (0, 256, -1, True):
            with self.assertRaises(tree.ScratchError):
                tree.mutate(self.ctx, head, "vector123", "coordinate", "00")
        for branch in ("../vector1", "vector", "script1", "vector1/../../x"):
            with self.assertRaises(tree.ScratchError):
                tree.mutate(self.ctx, 1, branch, "coordinate", "00")
        for leaf in ("../foo", "leaf/2", "UPPER"):
            with self.assertRaises(tree.ScratchError):
                tree.mutate(self.ctx, 1, "vector123", leaf, "00")
        with self.assertRaises(tree.ScratchError):
            tree.mutate(self.ctx, 1, "vector123", "coordinate", "ab" * 65537)
        with self.assertRaises(tree.ScratchError):
            tree.mutate(self.ctx, 1, "vector123", "coordinate", "xz")
        with self.assertRaises(tree.ScratchError):
            tree.inspect(self.ctx, branch="vector123")

    def test_persistent_restart_without_implicit_training(self):
        tree.mutate(self.ctx, 2, "math42", "scalar", "102030")
        other = Context(Path(self.tmp.name))
        result = tree.inspect(other, 2, "math42", "scalar")
        self.assertTrue(result["found"])
        self.assertEqual(result["revision"], 1)
        self.assertEqual(result["data_hex"], "102030")


if __name__ == "__main__":
    unittest.main()
