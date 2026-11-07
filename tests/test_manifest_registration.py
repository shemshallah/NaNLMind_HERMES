"""Keep the catalog manifest synchronized with the runtime registration surface."""
import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def registered_tool_names():
    source = (ROOT / "nanlmind_hermes" / "__init__.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    register = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "register"
    )
    specs = next(
        node.value for node in register.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "specs"
                for target in node.targets)
    )
    return {
        item.elts[0].value
        for item in specs.elts
        if isinstance(item, ast.Tuple)
        and item.elts
        and isinstance(item.elts[0], ast.Constant)
        and isinstance(item.elts[0].value, str)
    }


def manifest_tool_names():
    text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^provides_tools:\s*\n((?:[ \t]+- [^\n]+\n)+)", text)
    if not match:
        raise AssertionError("plugin.yaml must declare provides_tools")
    return {
        line.split("- ", 1)[1].strip()
        for line in match.group(1).splitlines()
    }


class ManifestRegistrationTests(unittest.TestCase):
    def test_manifest_tool_list_matches_runtime_registration_exactly(self):
        declared = manifest_tool_names()
        registered = registered_tool_names()
        self.assertEqual(declared, registered,
                         "plugin.yaml provides_tools must exactly match register(ctx)")

    def test_package_and_manifest_versions_match(self):
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest_version = re.search(
            r"(?m)^version:\s*['\"]?([0-9]+\.[0-9]+\.[0-9]+)", manifest
        )
        project_version = re.search(
            r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', project
        )
        self.assertIsNotNone(manifest_version, "plugin.yaml version is missing")
        self.assertIsNotNone(project_version, "pyproject.toml version is missing")
        self.assertEqual(manifest_version.group(1), project_version.group(1))

    def test_license_and_build_metadata_match_the_release_files(self):
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertRegex(manifest, r"(?m)^license:\s*Apache-2\.0\s*$")
        self.assertRegex(project, r'(?m)^license\s*=\s*"Apache-2\.0"\s*$')
        self.assertIn("Apache License", license_text)
        self.assertIn("[build-system]", project)
        self.assertIn('build-backend = "setuptools.build_meta"', project)

    def test_phone_call_schema_and_nan16_kernel_ship(self):
        source = (ROOT / "nanlmind_hermes" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('"clocked": {"type": "boolean"', source,
                      "nanlm_phone_call schema must expose clocked")
        kernel = ROOT / "nanlmind_hermes" / "nan" / "phone_call.nan"
        self.assertTrue(kernel.exists(), "raw NaN16 kernel is missing")
        from nanlmind_hermes.nanvm import parse_nan_image, word_payload
        words = parse_nan_image(kernel.read_text(encoding="utf-8"))
        self.assertEqual(len(words), 99)
        first = word_payload(words[0])
        self.assertEqual(first >> 24, 0x01)          # MOVI ...
        self.assertEqual((first >> 20) & 15, 13)     # ... r13,1 executes first
        self.assertIn("*.nan", (ROOT / "MANIFEST.in").read_text(encoding="utf-8"))
        self.assertIn("nan/*.nan",
                      (ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_crypto_backend_is_optional_and_has_one_authoritative_declaration(self):
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("python_dependencies:", manifest)
        self.assertRegex(project, r'(?m)^dependencies\s*=\s*\[\]\s*$')
        self.assertIn('auth = ["cryptography>=42,<47"]', project)

    def test_full_feature_demo_covers_every_registered_tool(self):
        demo = (ROOT / "scripts" / "demo_run.py").read_text(encoding="utf-8")
        missing = sorted(name for name in registered_tool_names()
                         if '"%s"' % name not in demo)
        self.assertFalse(missing, "demo_run.py does not call: " + ", ".join(missing))

    def test_secret_environment_names_are_unique_and_masked(self):
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        expected = {"CLOCKED_GATEWAY_TOKEN", "NANLM_DEVICE_TOKEN", "NANNET_MASTER_KEY"}
        declared = set(re.findall(r"(?m)^\s+env:\s*([A-Z][A-Z0-9_]*)\s*$", manifest))
        self.assertEqual(declared, expected)
        self.assertEqual(
            len(re.findall(r"(?m)^\s+env:\s*[A-Z][A-Z0-9_]*\s*$", manifest)),
            len(expected),
        )


if __name__ == "__main__":
    unittest.main()
