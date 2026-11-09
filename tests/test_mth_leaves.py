import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import mth_catalog, mth_leaves, nan0s_runtime, register


class Context:
    def __init__(self, root):
        self.config = {
            "state_dir": str(root),
            "nan0s_prefer_bundled": True,
            "chat_learning": False,
        }
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]


class LeafStoreTests(unittest.TestCase):
    def setUp(self):
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = Context(Path(self._tmp.name) / "state")
        self.store = mth_leaves.store_for_context(self.ctx)

    def tearDown(self):
        mth_leaves.reset_cache()
        self._tmp.cleanup()

    def test_seed_hubs_define_vector_wordnet_and_constraints(self):
        status = self.store.status()
        self.assertEqual(status["leaves"], 9)
        self.assertEqual(status["kinds"], {"hub": 9})
        vector = self.store.get(mth_leaves.VECTOR_HUB)
        self.assertEqual(vector["payload"]["defines"], "vector")
        self.assertIn("vector.dimension == 256", vector["constraints"])
        self.assertIn("dot(x,y) == dot(y,x)", vector["identities"])
        self.assertEqual(vector["wordnet"]["route"],
                         "nan://GM0:SC0:MC0:BS0:0:HW4:0")
        self.assertIn("vector::vector", status["pair_rules"])
        self.assertEqual(status["wordnet"]["hub"], "hub:mth.wordnet")
        root = self.store.get(mth_leaves.ROOT_HUB)
        self.assertEqual(root["kind"], "hub")

    def test_leaf_image_is_real_nan_executable_and_immutable(self):
        leaf = self.store.create(
            kind="vector", payload={"values": [2.0] * 256}, alias="v1")
        self.assertTrue(leaf["leaf_id"].startswith("leaf:"))
        self.assertEqual(leaf["kind"], "vector")
        self.assertEqual(leaf["hub"], mth_leaves.VECTOR_HUB)
        self.assertTrue(leaf["immutable"])
        self.assertTrue(leaf["nan_image"]["path"].endswith(".nan"))
        self.assertTrue(Path(leaf["nan_image"]["path"]).is_file())
        verified = mth_leaves.verify_leaf(leaf)
        self.assertTrue(verified["nan_executable"])
        self.assertEqual(verified["output"], "L")
        self.assertEqual(verified["boundary"], "python-nanvm0")
        text = Path(leaf["nan_image"]["path"]).read_text(encoding="utf-8")
        self.assertIn("4C454146", text)  # LEAF magic payload
        # Re-creating identical content is idempotent, not a mutation.
        again = self.store.create(
            kind="vector", payload={"values": [2.0] * 256})
        self.assertEqual(again["leaf_id"], leaf["leaf_id"])
        self.assertEqual(again["content_sha256"], leaf["content_sha256"])
        filled = self.store.create(
            kind="vector", payload={"fill": 4.0}, alias="filled")
        self.assertEqual(len(filled["payload"]["values"]), 256)
        self.assertEqual(filled["payload"]["values"][0], 4.0)
        self.assertEqual(filled["payload"]["values"][-1], 4.0)
        normalized = self.store.create(
            kind="vector", payload={"fill": 5.0},
            parent=mth_leaves.FUNCTION_HUB, alias="normalized-parent")
        self.assertEqual(normalized["parent"], mth_leaves.VECTOR_HUB)
        with self.assertRaises(mth_leaves.MthLeafError) as raised:
            self.store.create(kind="vector", payload={"fill": 6.0},
                              parent=mth_leaves.ROOT_HUB)
        self.assertEqual(raised.exception.code, "invalid_parent_hub")

    def test_connect_creates_relation_without_native_execution(self):
        left = self.store.create(kind="vector", payload={"values": [1.0] * 256},
                                 alias="v1")
        right = self.store.create(kind="vector", payload={"values": [2.0] * 256},
                                  alias="v2")
        before = self.store.status()["leaves"]
        result = self.store.pair("v1", "v2", operator="connect", alias="link")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.leaf["kind"], "relation")
        self.assertEqual(result.leaf["function_definition"]["operator"], "connect")
        self.assertEqual(result.leaf["payload"]["left"], left["leaf_id"])
        self.assertEqual(result.leaf["payload"]["right"], right["leaf_id"])
        self.assertEqual(self.store.status()["leaves"], before + 1)
        self.assertEqual(self.store.get("v1")["content_sha256"],
                         left["content_sha256"])

    def test_illegal_and_unknown_pairings_deny_without_new_leaf(self):
        vector = self.store.create(kind="vector", payload={"values": [1.0] * 256},
                                   alias="v1")
        scalar = self.store.create(kind="scalar", payload={"value": 2.0},
                                   alias="s1")
        before = self.store.status()["leaves"]
        illegal = self.store.pair("v1", "s1", operator="add")
        self.assertEqual(illegal.status, "denied")
        self.assertEqual(illegal.error["code"], "illegal_pairing")
        unknown = self.store.pair("v1", "v1", operator="illegalpairing")
        self.assertEqual(unknown.status, "denied")
        self.assertEqual(unknown.error["code"], "unknown_operator")
        hub = self.store.pair("v1", mth_leaves.VECTOR_HUB)
        self.assertEqual(hub.status, "denied")
        self.assertEqual(hub.error["code"], "hub_pairing_denied")
        self.assertEqual(self.store.status()["leaves"], before)
        self.assertEqual(vector["leaf_id"], self.store.get("v1")["leaf_id"])
        self.assertEqual(scalar["leaf_id"], self.store.get("s1")["leaf_id"])

    def test_function_tree_links_every_mth_kernel_exactly_once(self):
        audit = self.store.ensure_function_tree()
        self.assertEqual(audit["expected"], 91)
        self.assertEqual(audit["linked"], 91)
        self.assertEqual(audit["missing"], [])
        self.assertEqual(audit["extra"], [])
        self.assertEqual(audit["duplicate_aliases"], [])
        self.assertEqual(audit["digest_mismatches"], [])
        self.assertEqual(audit["orphans"], [])
        self.assertEqual(audit["ancestry_failures"], [])
        self.assertEqual(audit["image_failures"], [])
        self.assertEqual(audit["images_verified"], 91)
        self.assertTrue(audit["complete"])
        self.assertEqual(sum(audit["family_counts"].values()), 91)
        self.assertGreater(len(audit["family_counts"]), 30)
        vector_kernel = self.store.get("kernel:mth.vector.dot256.f32")
        self.assertEqual(vector_kernel["kind"], "kernel")
        self.assertEqual(vector_kernel["parent"],
                         mth_leaves.FAMILY_HUB_PREFIX + "vector")
        definition = vector_kernel["function_definition"]
        self.assertEqual(definition["kernel_id"], "mth.vector.dot256.f32")
        self.assertEqual(definition["source_digest"],
                         mth_catalog.require(
                             "mth.vector.dot256.f32").source_digest)
        actions = {item["action"] for item in vector_kernel["action_relations"]}
        self.assertIn("member_of_family", actions)
        self.assertIn("member_of_function_root", actions)
        self.assertIn("operates_on", actions)
        self.assertIn("wordnet_link", actions)
        family = self.store.get(mth_leaves.FAMILY_HUB_PREFIX + "vector")
        self.assertEqual(family["parent"], mth_leaves.FUNCTION_HUB)
        root = self.store.get(mth_leaves.FUNCTION_HUB)
        self.assertEqual(root["parent"], mth_leaves.ROOT_HUB)
        # Idempotent mapping must not duplicate or rewrite kernel leaves.
        before = self.store.status()["leaves"]
        again = self.store.ensure_function_tree()
        self.assertTrue(again["complete"])
        self.assertEqual(self.store.status()["leaves"], before)

    def test_audit_log_records_denials_without_payloads(self):
        self.store.create(kind="vector", payload={"values": [1.0] * 256}, alias="v1")
        self.store.pair("v1", "v1", operator="not-an-operator")
        lines = Path(self.store.audit_path).read_text(encoding="utf-8").splitlines()
        denied = [json.loads(line) for line in lines if '"denied"' in line]
        self.assertTrue(denied)
        self.assertEqual(denied[-1]["code"], "unknown_operator")
        self.assertNotIn("values", denied[-1])


@unittest.skipUnless(
    nan0s_runtime.compiler_available(),
    "C compiler required for native vector pairing")
class LeafMathTests(unittest.TestCase):
    def setUp(self):
        mth_leaves.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = Context(Path(self._tmp.name) / "state")
        self.store = mth_leaves.store_for_context(self.ctx)

    def tearDown(self):
        mth_leaves.reset_cache()
        self._tmp.cleanup()

    def test_add_and_dot_create_new_hash_leaves_and_preserve_inputs(self):
        left = self.store.create(kind="vector", payload={"values": [2.0] * 256},
                                 alias="v1")
        right = self.store.create(kind="vector", payload={"values": [3.0] * 256},
                                  alias="v2")
        added = self.store.pair("v1", "v2", operator="add", alias="sum")
        self.assertEqual(added.status, "ok")
        self.assertTrue(added.hashleaf.startswith("leaf:"))
        self.assertEqual(added.leaf["kind"], "vector")
        self.assertEqual(added.leaf["payload"]["values"][0], 5.0)
        self.assertEqual(added.leaf["payload"]["values"][-1], 5.0)
        self.assertEqual(added.leaf["function_definition"]["kernel_id"],
                         "mth.vector.add256.f32")
        self.assertIn("add(x,y) == add(y,x)", added.leaf["identities"])
        add_execution = added.leaf["function_definition"]["execution"]
        self.assertEqual(add_execution["mode"], "n16i_guest_interpreter")
        self.assertIs(add_execution["n16i_used"], True)
        self.assertEqual(add_execution["status_register"], 1)
        self.assertEqual(add_execution["trap"], "")
        self.assertGreater(add_execution["steps"], 0)
        self.assertEqual(self.store.get("v1")["content_sha256"],
                         left["content_sha256"])
        self.assertEqual(self.store.get("v2")["content_sha256"],
                         right["content_sha256"])
        dot = self.store.pair("v1", "v2", operator="dot", alias="dot")
        self.assertEqual(dot.status, "ok")
        self.assertEqual(dot.leaf["kind"], "scalar")
        self.assertEqual(dot.leaf["payload"]["value"], 1536.0)
        self.assertEqual(dot.leaf["function_definition"]["program_id"], 900)
        self.assertEqual(dot.leaf["payload"]["precision"], "float32")
        dot_execution = dot.leaf["function_definition"]["execution"]
        self.assertEqual(dot_execution["mode"], "n16i_guest_interpreter")
        self.assertIs(dot_execution["n16i_used"], True)
        self.assertEqual(dot_execution["status_register"], 1)
        self.assertEqual(dot_execution["trap"], "")
        self.assertGreater(dot_execution["steps"], 0)

    def test_tool_handlers_create_pair_get_and_verify_leaves(self):
        register(self.ctx)
        vector = json.loads(self.ctx.tools["nanlm_mth_leaf_create"]({
            "kind": "vector", "payload": {"values": [2.0] * 256},
            "alias": "tool-v1"}))
        self.assertEqual(vector["status"], "ok")
        self.assertEqual(vector["authorization"]["mode"], "local-unverified")
        second = json.loads(self.ctx.tools["nanlm_mth_leaf_create"]({
            "kind": "vector", "payload": {"values": [3.0] * 256},
            "alias": "tool-v2"}))
        self.assertEqual(second["status"], "ok")
        paired = json.loads(self.ctx.tools["nanlm_mth_pair"]({
            "left": "tool-v1", "right": "tool-v2", "operator": "add"}))
        self.assertEqual(paired["status"], "ok")
        self.assertEqual(paired["leaf"]["payload"]["values"][0], 5.0)
        self.assertEqual(
            paired["leaf"]["function_definition"]["execution"]["mode"],
            "n16i_guest_interpreter")
        self.assertIs(
            paired["leaf"]["function_definition"]["execution"]["n16i_used"], True)
        fetched = json.loads(self.ctx.tools["nanlm_mth_leaf_get"]({
            "leaf_id": "tool-v1", "include_image": True}))
        self.assertIn("4C454146", fetched["nan_image"]["text"])
        verified = json.loads(self.ctx.tools["nanlm_mth_leaf_verify"]({
            "leaf_id": "tool-v1"}))
        self.assertTrue(verified["nan_executable"])
        hub = json.loads(self.ctx.tools["nanlm_mth_hub"]({}))
        self.assertEqual(hub["root_hub"], "hub:mth.root")
        self.assertEqual(hub["wordnet"]["route"],
                         "nan://GM0:SC0:MC0:BS0:0:HW4:0")
        tree = json.loads(self.ctx.tools["nanlm_mth_tree"]({"action": "audit"}))
        self.assertTrue(tree["complete"])
        self.assertEqual(tree["linked"], 91)
        self.assertEqual(tree["authorization"]["mode"], "local-unverified")


if __name__ == "__main__":
    unittest.main()
