"""Catalog credential policy regression tests."""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nanlmind_hermes"


class CredentialPolicyTests(unittest.TestCase):
    def test_plugin_source_never_reads_process_environment_or_credential_files(self):
        forbidden = re.compile(r"os\.environ|NANLM_AUTH_TEST_BYPASS|HW_HERMES_API")
        for path in PACKAGE.glob("*.py"):
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIsNone(forbidden.search(source), path.name)
                self.assertNotIn('".env"', source, path.name)
                self.assertNotIn('"auth.json"', source, path.name)

    def test_manifest_declares_all_external_credentials_as_masked_secrets(self):
        manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        expected = {
            "clocked_gateway_token": "CLOCKED_GATEWAY_TOKEN",
            "nanlm_device_token": "NANLM_DEVICE_TOKEN",
            "nannet_master_key": "NANNET_MASTER_KEY",
        }
        for key, env_name in expected.items():
            with self.subTest(key=key):
                lines = manifest.splitlines()
                index = lines.index("  " + key + ":")
                block = "\n".join(lines[index + 1:index + 4])
                self.assertIn("type: secret", block)
                self.assertIn("env: " + env_name, block)

    def test_production_authorization_bypass_is_absent(self):
        source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("NANLM_AUTH_TEST_BYPASS", source)
        self.assertIn("_auth_tools().require_scope(ctx, scope)", source)


if __name__ == "__main__":
    unittest.main()
