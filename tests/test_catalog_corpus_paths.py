"""Prevent model-selected corpus paths from reading arbitrary host files."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.tools import resolve_tool_corpus_path


class Context:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)

    def get_config(self, name, default=None):
        return str(self.state_dir) if name == "state_dir" else default


class CorpusPathPolicyTests(unittest.TestCase):
    def test_only_canonical_files_below_state_corpora_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            corpora = state / "corpora"
            corpora.mkdir(parents=True)
            valid = corpora / "safe.n16txt"
            valid.write_text("N16TXT/1\nhello world\n", encoding="utf-8")
            outside = Path(tmp) / "PRIVATE_TOKEN_FILE"
            outside.write_text("do-not-train-this", encoding="utf-8")
            ctx = Context(state)
            self.assertEqual(resolve_tool_corpus_path(ctx, str(valid)), str(valid))
            cases = [
                outside, corpora / ".." / ".." / outside.name,
                state / "not-corpus.txt",
            ]
            for path in cases:
                with self.subTest(path=path), self.assertRaises(
                        (PermissionError, ValueError)):
                    resolve_tool_corpus_path(ctx, str(path))
            alias = corpora / "outbound.n16txt"
            alias.symlink_to(outside)
            with self.assertRaises(PermissionError):
                resolve_tool_corpus_path(ctx, str(alias))
            with self.assertRaises(ValueError):
                resolve_tool_corpus_path(ctx, "")
            with self.assertRaises(ValueError):
                resolve_tool_corpus_path(ctx, None)
            with self.assertRaises(ValueError):
                resolve_tool_corpus_path(ctx, str(corpora / "missing.n16txt"))

    def test_nested_owned_file_and_internal_symlink_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpora = Path(tmp) / "corpora"
            child = corpora / "nested"
            child.mkdir(parents=True)
            file = child / "owned.n16txt"
            file.write_text("N16TXT/1\ntext\n", encoding="utf-8")
            alias = corpora / "alias.n16txt"
            alias.symlink_to(file)
            ctx = Context(tmp)
            self.assertEqual(resolve_tool_corpus_path(ctx, str(alias)), str(file))
            self.assertEqual(resolve_tool_corpus_path(ctx, str(file)), str(file))


if __name__ == "__main__":
    unittest.main()
