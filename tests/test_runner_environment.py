"""Verify native subprocesses never inherit Hermes provider credentials."""
import os
import unittest
from unittest.mock import Mock, patch

from nanlmind_hermes.runner import _run, _native_process_environment


class NativeSubprocessSecretIsolationTests(unittest.TestCase):
    def test_minimal_whitelist_drops_provider_tokens_and_profile_variables(self):
        poisoned = {
            "PATH": "/usr/bin:/bin", "HOME": "/test/homedir", "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8", "TMPDIR": "/test/tmp",
            "OPENAI_API_KEY": "NEVER_LEAK_THIS_SECRET",
            "ANTHROPIC_API_KEY": "NEVER_LEAK_THIS_SECRET",
            "GITHUB_TOKEN": "NEVER_LEAK_THIS_SECRET",
            "NANLM_AUTH_TEST_BYPASS": "true",
            "NANLM_UNREVIEWED_SECRET": "NEVER_LEAK_THIS_SECRET",
            "HERMES_YOLO_MODE": "1", "PYTHONPATH": "do-not-pass",
            "CLOCKED_GATEWAY_TOKEN": "NEVER_LEAK_THIS_SECRET",
            "NANLM_DEVICE_TOKEN": "NEVER_LEAK_THIS_SECRET",
            "NANNET_MASTER_KEY": "NEVER_LEAK_THIS_SECRET",
        }
        with patch.dict(os.environ, poisoned, clear=True):
            clean = _native_process_environment()
            self.assertEqual(set(clean), {
                "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"})
            self.assertEqual(clean["PATH"], "/usr/bin:/bin")
            self.assertEqual(clean["HOME"], "/test/homedir")
            self.assertNotIn("NEVER_LEAK_THIS_SECRET", repr(clean))
            proc = Mock()
            proc.communicate.return_value = ("native output", None)
            proc.returncode = 0
            with patch("nanlmind_hermes.runner.subprocess.Popen",
                       return_value=proc) as spawn:
                result = _run(["/opt/bin/nanlm-universal", "status"], 2)
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["output"], "native output")
            kwargs = spawn.call_args.kwargs
            self.assertEqual(kwargs["env"], clean)
            self.assertFalse(kwargs.get("shell", False))

    def test_locale_default_and_future_nanlm_wildcards_not_forwarded(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "SECRET",
            "NANLM_ABC": "UNREVIEWED",
        }, clear=True):
            env = _native_process_environment()
            self.assertEqual(env["LANG"], "C.UTF-8")
            self.assertEqual(set(env), {"HOME", "PATH", "LANG"})
            self.assertNotIn("NANLM_ABC", env)


if __name__ == "__main__":
    unittest.main()
