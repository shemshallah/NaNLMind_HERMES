"""Dependency-injected tests for ``scripts/resident_universe_cli.py``.

The tests never import the real ``nanlmind_hermes.resident`` implementation.
A temporary fake backend/universe is passed through ``main(backend=...)`` so
the CLI contract can be exercised while the resident package is still being
built in parallel.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resident_universe_cli.py"

_spec = importlib.util.spec_from_file_location(
    "resident_universe_cli_under_test", SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scripts/resident_universe_cli.py")
cli = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cli
_spec.loader.exec_module(cli)

_DEFAULT = object()


class FakeUniverse:
    """A minimal universe that records every operation the CLI performs."""

    def __init__(self, backend, path, size):
        self.backend = backend
        self.path = Path(path)
        self.size = size
        self.calls = []
        self.closed = False

    def status(self):
        self.calls.append(("status",))
        self.backend.maybe_raise(self.backend.status_error)
        value = self.backend.status_value
        return value() if callable(value) else value

    def ingest_jsonl(self, source, limit):
        source = Path(source)
        self.calls.append(("ingest_jsonl", source, limit))
        self.backend.maybe_raise(self.backend.ingest_error)
        records = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= limit:
                    break
                if line.strip():
                    records.append(json.loads(line))
        return {
            "ingested": len(records),
            "limit": limit,
            "source": str(source),
        }

    def query(self, text, top_k):
        self.calls.append(("query", text, top_k))
        self.backend.maybe_raise(self.backend.query_error)
        if self.backend.query_value is not None:
            return self.backend.query_value
        return list(self.backend.query_items)

    def link(self, parent, child):
        self.calls.append(("link", parent, child))
        self.backend.maybe_raise(self.backend.link_error)
        if self.backend.link_value is not None:
            return self.backend.link_value
        return {"linked": True, "parent": parent, "child": child}

    def export(self, destination, full):
        destination = Path(destination)
        self.calls.append(("export", destination, full))
        self.backend.maybe_raise(self.backend.export_error)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").write_text(
            json.dumps({"full": full}), encoding="utf-8")
        return {"exported": True, "full": full, "entries": 1}

    def checkpoint(self):
        self.calls.append(("checkpoint",))
        self.backend.maybe_raise(self.backend.checkpoint_error)
        return {"checkpointed": True, "generation": 7}

    def close(self):
        self.closed = True
        self.calls.append(("close",))


class FakeBackend:
    """Backend seam shared by all tests."""

    def __init__(self, root):
        self.root = Path(root)
        self.open_calls = []
        self.universes = []
        self.validate_calls = []
        self.status_value = lambda: {
            "path": str(self.root / "universe.nan"),
            "generation": 7,
            "size_bytes": 16 * 1024 * 1024,
            "leaves": 4,
        }
        self.query_items = [
            {"leaf": "leaf:%03d" % index, "score": 1.0 / (index + 1)}
            for index in range(200)
        ]
        self.query_value = None
        self.link_value = None
        self.validation = {"valid": True, "entries": 3}
        self.open_error = None
        self.status_error = None
        self.ingest_error = None
        self.query_error = None
        self.link_error = None
        self.export_error = None
        self.checkpoint_error = None
        self.validate_error = None

    @staticmethod
    def maybe_raise(error):
        if error is not None:
            raise error

    def open_universe(self, path, create=False, size=None):
        self.open_calls.append((Path(path), create, size))
        self.maybe_raise(self.open_error)
        universe = FakeUniverse(self, path, size)
        self.universes.append(universe)
        return universe

    def validate_bundle(self, bundle):
        self.validate_calls.append(Path(bundle))
        self.maybe_raise(self.validate_error)
        return dict(self.validation)


class ResidentUniverseCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.universe_path = self.root / "universe.nan"
        self.source = self.root / "records.jsonl"
        self.source.write_text(
            "\n".join(json.dumps({"id": index, "text": "record %d" % index})
                      for index in range(3)) + "\n",
            encoding="utf-8",
        )
        self.backend = FakeBackend(self.root)

    def run_cli(self, *arguments, backend=_DEFAULT):
        chosen = self.backend if backend is _DEFAULT else backend
        stdout = io.StringIO()
        stderr = io.StringIO()
        code = cli.main(list(arguments), backend=chosen,
                        stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def payload(text):
        return json.loads(text)

    # -- init -----------------------------------------------------------

    def test_init_reports_bounded_size_and_closes_universe(self):
        size = 16 * 1024 * 1024
        code, out, err = self.run_cli(
            "init", "--path", str(self.universe_path), "--size", str(size))

        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        data = self.payload(out)
        self.assertEqual(data["schema"], cli.SCHEMA)
        self.assertTrue(data["ok"])
        self.assertEqual(data["command"], "init")
        self.assertEqual(data["bytes"], size)
        self.assertEqual(data["requested_bytes"], size)
        self.assertFalse(data["size_clamped"])
        self.assertTrue(data["result"]["created"])
        self.assertEqual(data["canonical_merge"],
                         {"enabled": False, "attempted": False})
        self.assertEqual(self.backend.open_calls,
                         [(self.universe_path, True, size)])
        self.assertTrue(self.backend.universes[0].closed)

    def test_init_clamps_oversized_request_and_accepts_binary_suffix(self):
        requested = 2 * cli.MAX_BYTES
        code, out, err = self.run_cli(
            "init", "--path", str(self.universe_path), "--size", "2GiB")

        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertEqual(data["requested_bytes"], requested)
        self.assertEqual(data["bytes"], cli.MAX_BYTES)
        self.assertTrue(data["size_clamped"])
        self.assertEqual(self.backend.open_calls[0][2], cli.MAX_BYTES)

    def test_init_refuses_to_overwrite_a_non_empty_path(self):
        self.universe_path.write_text("existing", encoding="utf-8")
        code, out, err = self.run_cli(
            "init", "--path", str(self.universe_path), "--size", "16777216")

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "already_exists")
        self.assertEqual(self.backend.open_calls, [])

    # -- status ---------------------------------------------------------

    def test_status_bounds_and_sanitizes_values(self):
        self.backend.status_value = lambda: {
            "path": str(self.root / "universe.nan"),
            "size_bytes": 10 ** 30,
            "negative_bytes": -7,
            "big_list": list(range(200)),
            "huge_text": "z" * 4000,
            "payload": b"\x00" * 32,
            "nested": {"deep": {"deeper": {"deepest": [1, 2, 3]}}},
        }

        with mock.patch.object(cli, "_home_text", return_value=str(self.root)):
            code, out, err = self.run_cli(
                "status", "--path", str(self.universe_path))

        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertLessEqual(len(out.strip()), cli.MAX_JSON_CHARS)
        data = self.payload(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["command"], "status")
        self.assertEqual(data["path"], "~/universe.nan")
        result = data["result"]
        self.assertEqual(result["path"], "~/universe.nan")
        self.assertEqual(result["size_bytes"], cli.MAX_REPORTED_INT)
        self.assertEqual(result["negative_bytes"], 0)
        self.assertLessEqual(len(result["big_list"]), cli.MAX_ITEMS + 1)
        self.assertIn("truncated", result["big_list"][-1])
        self.assertLessEqual(len(result["huge_text"]), cli.MAX_STRING)
        self.assertEqual(result["payload"]["type"], "bytes")
        self.assertEqual(result["payload"]["bytes"], 32)
        self.assertNotIn("\\u0000", out)
        self.assertTrue(data["truncated"])
        self.assertTrue(self.backend.universes[0].closed)

    # -- ingest ---------------------------------------------------------

    def test_ingest_forwards_limit_source_and_closes(self):
        code, out, err = self.run_cli(
            "ingest", "--path", str(self.universe_path),
            "--jsonl", str(self.source), "--limit", "2")

        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        data = self.payload(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["command"], "ingest")
        self.assertEqual(data["path"], cli._display_path(self.universe_path))
        self.assertEqual(data["source"], cli._display_path(self.source))
        self.assertEqual(data["limit"], 2)
        self.assertFalse(data["limit_clamped"])
        self.assertEqual(data["result"]["ingested"], 2)
        universe = self.backend.universes[0]
        self.assertIn(("ingest_jsonl", self.source, 2), universe.calls)
        self.assertTrue(universe.closed)

    def test_ingest_clamps_limit_and_rejects_missing_source(self):
        code, out, err = self.run_cli(
            "ingest", "--path", str(self.universe_path),
            "--jsonl", str(self.source), "--limit", "1000000000")
        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertEqual(data["limit"], cli.MAX_INGEST_LIMIT)
        self.assertTrue(data["limit_clamped"])

        missing = self.root / "absent.jsonl"
        code, out, err = self.run_cli(
            "ingest", "--path", str(self.universe_path),
            "--jsonl", str(missing))
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertEqual(data["error"]["code"], "invalid_input")
        self.assertEqual(len(self.backend.open_calls), 1)

    # -- query ----------------------------------------------------------

    def test_query_trims_results_and_clamps_top_k(self):
        code, out, err = self.run_cli(
            "query", "--path", str(self.universe_path),
            "--text", "alpha", "--top-k", "999")

        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        data = self.payload(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["command"], "query")
        self.assertEqual(data["top_k"], cli.MAX_TOP_K)
        self.assertTrue(data["top_k_clamped"])
        self.assertEqual(data["text_chars"], len("alpha"))
        self.assertEqual(data["returned"], cli.MAX_TOP_K)
        self.assertEqual(len(data["result"]), cli.MAX_TOP_K)
        self.assertIn(("query", "alpha", cli.MAX_TOP_K),
                      self.backend.universes[0].calls)
        self.assertTrue(self.backend.universes[0].closed)

    def test_query_trims_mapping_results(self):
        self.backend.query_value = {
            "results": list(self.backend.query_items),
            "total": len(self.backend.query_items),
        }
        code, out, err = self.run_cli(
            "query", "--path", str(self.universe_path),
            "--text", "alpha", "--top-k", "3")
        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertEqual(len(data["result"]["results"]), 3)
        self.assertEqual(data["returned"], 3)

    def test_query_rejects_empty_and_oversized_text(self):
        code, out, err = self.run_cli(
            "query", "--path", str(self.universe_path), "--text", "   ")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(self.payload(err)["error"]["code"], "invalid_text")

        code, out, err = self.run_cli(
            "query", "--path", str(self.universe_path),
            "--text", "x" * (cli.MAX_TEXT_CHARS + 1))
        self.assertEqual(code, 2)
        self.assertEqual(self.payload(err)["error"]["code"], "invalid_text")
        self.assertEqual(self.backend.open_calls, [])

    # -- link -----------------------------------------------------------

    def test_link_reports_and_forwards_leaf_names(self):
        code, out, err = self.run_cli(
            "link", "--path", str(self.universe_path),
            "--parent", "leaf:parent", "--child", "leaf:child")

        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["parent"], "leaf:parent")
        self.assertEqual(data["child"], "leaf:child")
        self.assertTrue(data["result"]["linked"])
        self.assertIn(("link", "leaf:parent", "leaf:child"),
                      self.backend.universes[0].calls)

    def test_link_reported_failure_exits_nonzero_with_json(self):
        self.backend.link_value = {
            "linked": False,
            "message": "missing parent leaf",
        }
        code, out, err = self.run_cli(
            "link", "--path", str(self.universe_path),
            "--parent", "leaf:parent", "--child", "leaf:child")

        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        data = self.payload(out)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "operation_failed")
        self.assertIn("missing parent leaf", data["error"]["message"])

    # -- export ---------------------------------------------------------

    def test_export_forwards_full_flag_and_destination(self):
        destination = self.root / "bundle"
        code, out, err = self.run_cli(
            "export", "--path", str(self.universe_path),
            "--destination", str(destination))
        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertFalse(data["full"])
        self.assertEqual(data["destination"], cli._display_path(destination))
        self.assertTrue((destination / "manifest.json").is_file())
        self.assertIn(("export", destination, False),
                      self.backend.universes[0].calls)

        full_destination = self.root / "bundle-full"
        code, out, err = self.run_cli(
            "export", "--path", str(self.universe_path),
            "--destination", str(full_destination), "--full")
        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertTrue(data["full"])
        self.assertIn(("export", full_destination, True),
                      self.backend.universes[1].calls)

    def test_export_rejects_destination_equal_to_universe_path(self):
        code, out, err = self.run_cli(
            "export", "--path", str(self.universe_path),
            "--destination", str(self.universe_path))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(self.payload(err)["error"]["code"],
                         "invalid_destination")
        self.assertEqual(self.backend.open_calls, [])

    # -- validate -------------------------------------------------------

    def test_validate_success_and_invalid_bundle(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        code, out, err = self.run_cli("validate", "--bundle", str(bundle))
        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["command"], "validate")
        self.assertEqual(data["bundle"], cli._display_path(bundle))
        self.assertEqual(data["result"]["entries"], 3)
        self.assertEqual(self.backend.validate_calls, [bundle])

        self.backend.validation = {
            "valid": False,
            "errors": ["manifest.json is missing"],
        }
        code, out, err = self.run_cli("validate", "--bundle", str(bundle))
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        data = self.payload(out)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "invalid_bundle")
        self.assertIn("manifest.json is missing", data["error"]["message"])

    def test_validate_missing_directory_and_no_result_fail_closed(self):
        missing = self.root / "missing-bundle"
        code, out, err = self.run_cli("validate", "--bundle", str(missing))
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertEqual(data["error"]["code"], "invalid_bundle")
        self.assertEqual(self.backend.validate_calls, [])

        bundle = self.root / "empty-bundle"
        bundle.mkdir()
        self.backend.validation = None
        self.backend.validate_bundle = lambda value: None
        code, out, err = self.run_cli("validate", "--bundle", str(bundle))
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(self.payload(err)["error"]["code"],
                         "backend_contract_error")

    # -- checkpoint -----------------------------------------------------

    def test_checkpoint_reports_generation_and_closes(self):
        code, out, err = self.run_cli(
            "checkpoint", "--path", str(self.universe_path))
        self.assertEqual(code, 0, err)
        data = self.payload(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["command"], "checkpoint")
        self.assertTrue(data["result"]["checkpointed"])
        self.assertEqual(data["result"]["generation"], 7)
        self.assertIn(("checkpoint",), self.backend.universes[0].calls)
        self.assertTrue(self.backend.universes[0].closed)

    # -- failure and usage handling ------------------------------------

    def test_backend_failure_is_sanitized_json_without_traceback(self):
        self.backend.open_error = RuntimeError(
            "cannot open " + str(self.root / "private" / "universe.nan"))
        with mock.patch.object(cli, "_home_text", return_value=str(self.root)):
            code, out, err = self.run_cli(
                "status", "--path", str(self.universe_path))

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "open_failed")
        self.assertIn("~/private/universe.nan", data["error"]["message"])
        self.assertNotIn(str(self.root), err)
        self.assertNotIn("Traceback", err)

    def test_operation_exception_closes_universe_and_emits_json(self):
        self.backend.query_error = RuntimeError("query backend exploded")
        code, out, err = self.run_cli(
            "query", "--path", str(self.universe_path), "--text", "alpha")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertEqual(data["error"]["code"], "internal_error")
        self.assertIn("query backend exploded", data["error"]["message"])
        self.assertTrue(self.backend.universes[0].closed)

    def test_usage_errors_are_json_and_exit_two(self):
        code, out, err = self.run_cli("status")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertEqual(data["error"]["code"], "usage")

        code, out, err = self.run_cli("not-a-command")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(self.payload(err)["error"]["code"], "usage")

        code, out, err = self.run_cli(
            "init", "--path", str(self.universe_path), "--size", "potato")
        self.assertEqual(code, 2)
        self.assertEqual(self.payload(err)["error"]["code"], "usage")

        code, out, err = self.run_cli(
            "init", "--path", str(self.universe_path), "--size", "9" * 400)
        self.assertEqual(code, 2)
        self.assertEqual(self.payload(err)["error"]["code"], "usage")

        code, out, err = self.run_cli(
            "query", "--path", str(self.universe_path),
            "--text", "alpha", "--top-k", "0")
        self.assertEqual(code, 2)
        self.assertEqual(self.payload(err)["error"]["code"], "usage")

    def test_unavailable_default_backend_is_json_not_traceback(self):
        with mock.patch.object(
                cli._ResidentBackend, "_load_modules",
                side_effect=cli.CliError(
                    "resident universe backend is unavailable",
                    "backend_unavailable")):
            code, out, err = self.run_cli(
                "status", "--path", str(self.universe_path), backend=None)

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        data = self.payload(err)
        self.assertEqual(data["error"]["code"], "backend_unavailable")
        self.assertNotIn("Traceback", err)

    # -- static boundaries ---------------------------------------------

    def test_module_imports_resident_lazily_and_never_imports_network(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_level_modules = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_modules.append(node.module)

        self.assertFalse(any(
            module.startswith("nanlmind_hermes.resident")
            for module in top_level_modules), top_level_modules)
        forbidden = {
            "socket", "ssl", "urllib", "http", "requests", "httpx",
            "aiohttp", "ftplib", "smtplib", "telnetlib",
        }
        imported_roots = {module.split(".", 1)[0] for module in top_level_modules}
        self.assertFalse(imported_roots & forbidden)
        self.assertFalse(hasattr(cli, "resident"))

    def test_command_surface_has_no_network_or_merge_command(self):
        parser = cli.build_parser()
        subparsers = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        self.assertEqual(len(subparsers), 1)
        commands = set(subparsers[0].choices)
        self.assertEqual(commands, {
            "init", "status", "ingest", "query", "link", "export",
            "validate", "checkpoint",
        })
        self.assertNotIn("merge", commands)
        self.assertNotIn("serve", commands)
        self.assertNotIn("fetch", commands)


if __name__ == "__main__":
    unittest.main()
