#!/usr/bin/env python3
"""Build and inspect NaNLMind HERMES release archives.

The verifier builds a wheel (and an sdist when the host tooling supports it) in
a temporary workspace, inspects the archive member lists, checks the bundled
MTH/NaN0S/session payload, rejects secret/state/model/oversized artifacts and
prints a single JSON receipt.  It exits non-zero when any check fails.

Typical use::

    python3 scripts/verify_package.py

Set ``NANLM_PACKAGING_TIMEOUT`` to bound each build phase (seconds).  Build
artifacts are kept in the reported ``work_dir`` so the receipt paths can be
inspected after the process exits.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from typing import Any, Iterable

SCHEMA = "nanlm.package-verify/1"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "nanlmind_hermes"
EXPECTED_SESSION_DOCS = 7
EXPECTED_MTH_KERNELS = 91
EXPECTED_NAN0S_CORE = 9

MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
MAX_TEXT_SCAN_BYTES = 1024 * 1024
MAX_TOTAL_TEXT_SCAN_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT = 240.0

# High-confidence filename markers.  ``hermes_secrets.py`` is legitimate code,
# so filename checks target state/credential/artifact extensions, not words.
_SECRET_NAME_RE = re.compile(
    r"(?ix)(?:^|/)(?:"
    r"\.env(?:$|\.[^/]*)|"
    r"[^/]*\.(?:pem|key|pat|token|sqlite|sqlite3|sqlite-wal|sqlite-shm|"
    r"safetensors|gguf|onnx|pt|pth|ckpt|h5|hdf5|keras|tflite|npy|npz|"
    r"pkl|pickle|joblib|model|weights|bin|parquet|arrow|feather|mat|bag|"
    r"jsonl|db)|"
    r"[^/]*(?:token|secret)[^/]*\.json|"
    r"\.netrc|\.pypirc|id_rsa|id_ed25519|credentials\.json|secrets?\.json|"
    r"nannet_token\.json|installation\.json"
    r")$"
)

_SECRET_CONTENT_PATTERNS = (
    ("private-key", re.compile(
        rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("slack-token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai-key", re.compile(rb"\bsk-[A-Za-z0-9]{32,}\b")),
    ("google-api-key", re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("pypi-token", re.compile(rb"\bpypi-[A-Za-z0-9_-]{20,}\b")),
)

_TEXT_SUFFIXES = frozenset({
    ".awk", ".bat", ".c", ".cfg", ".conf", ".csh", ".css", ".h", ".html",
    ".inc", ".ini", ".in", ".js", ".json", ".ksh", ".md", ".nan", ".nas",
    ".ps1", ".py", ".rst", ".service", ".sh", ".toml", ".ts", ".txt",
    ".xml", ".yaml", ".yml", ".zsh",
})


class PackagingVerificationError(RuntimeError):
    """Raised when an archive cannot be inspected safely."""


def _rel_files(root: Path, pattern: str) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    )


def _safe_mth_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise PackagingVerificationError(f"invalid MTH {field}: {value!r}")
    pure = PurePosixPath(value)
    if ".." in pure.parts:
        raise PackagingVerificationError(f"MTH {field} escapes bundle: {value!r}")
    return f"{PACKAGE}/mth/{value}"


def build_requirements(root: Path | str = ROOT) -> dict[str, Any]:
    """Return the exact file set and counts a release archive must contain."""
    root = Path(root).resolve()
    required: set[str] = {
        "plugin.yaml",
        f"{PACKAGE}/__init__.py",
        f"{PACKAGE}/mth/spec/wave-capabilities.json",
        f"{PACKAGE}/nan0s/tools/nan_cli.c",
        f"{PACKAGE}/nan0s/tools/nmk.awk",
        f"{PACKAGE}/nan0s/src/tools/n16i.nas",
        f"{PACKAGE}/nan0s/src/tests/n16i_test.nas",
    }
    globs = (
        f"{PACKAGE}/nan/*.nan",
        f"{PACKAGE}/nan/*.nas",
        f"{PACKAGE}/session_docs/*.md",
        f"{PACKAGE}/mth/nan16/*.nas",
        f"{PACKAGE}/mth/tests/*.nas",
        f"{PACKAGE}/nan0s/core/*.c",
        f"{PACKAGE}/nan0s/core/*.h",
        f"{PACKAGE}/nan0s/core/*.inc",
    )
    for pattern in globs:
        required.update(_rel_files(root, pattern))

    source_errors: list[str] = []
    kernels: list[dict[str, Any]] = []
    pairs: list[dict[str, str]] = []
    manifest_path = root / PACKAGE / "mth" / "spec" / "wave-capabilities.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("kernels")
        if not isinstance(entries, list):
            raise PackagingVerificationError("manifest has no kernels list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise PackagingVerificationError("manifest kernel entry is not an object")
            source = _safe_mth_path(entry.get("source"), "source")
            fixture = _safe_mth_path(entry.get("fixture"), "fixture")
            required.add(source)
            required.add(fixture)
            rows = [(source, fixture)]
            if any(item.get("source") == source and item.get("fixture") == fixture
                   for item in pairs):
                continue
            pairs.extend({"source": src, "fixture": fixture} for src, fixture in rows)
            kernels.append(entry)
    except (OSError, ValueError, PackagingVerificationError) as exc:
        source_errors.append(f"cannot load MTH manifest: {exc}")

    sdist_required = sorted(required)
    wheel_required = sorted(path for path in required if path.startswith(PACKAGE + "/"))
    source_missing = sorted(path for path in required if not (root / path).is_file())

    counts = {
        "required_files": len(required),
        "package_files": len(wheel_required),
        "nan_kernels": len(_rel_files(root, f"{PACKAGE}/nan/*.nan")),
        "nan_sources": len(_rel_files(root, f"{PACKAGE}/nan/*.nas")),
        "session_docs": len(_rel_files(root, f"{PACKAGE}/session_docs/*.md")),
        "mth_nan16": len(_rel_files(root, f"{PACKAGE}/mth/nan16/*.nas")),
        "mth_fixtures": len(_rel_files(root, f"{PACKAGE}/mth/tests/*.nas")),
        "mth_manifest_kernels": len(kernels),
        "mth_pairs": len(pairs),
        "nan0s_core": len([
            path for path in _rel_files(root, f"{PACKAGE}/nan0s/core/*")
            if PurePosixPath(path).suffix in {".c", ".h", ".inc"}
        ]),
    }
    if source_missing:
        source_errors.append("missing source files: " + ", ".join(source_missing))
    if counts["session_docs"] != EXPECTED_SESSION_DOCS:
        source_errors.append(
            f"expected {EXPECTED_SESSION_DOCS} session docs, found {counts['session_docs']}")
    if counts["mth_manifest_kernels"] != EXPECTED_MTH_KERNELS:
        source_errors.append(
            f"expected {EXPECTED_MTH_KERNELS} MTH kernels, found "
            f"{counts['mth_manifest_kernels']}")
    if counts["mth_fixtures"] != EXPECTED_MTH_KERNELS:
        source_errors.append(
            f"expected {EXPECTED_MTH_KERNELS} MTH fixtures, found "
            f"{counts['mth_fixtures']}")
    if counts["mth_nan16"] < EXPECTED_MTH_KERNELS:
        source_errors.append(
            f"expected at least {EXPECTED_MTH_KERNELS} MTH sources, found "
            f"{counts['mth_nan16']}")
    if counts["nan0s_core"] < EXPECTED_NAN0S_CORE:
        source_errors.append(
            f"expected at least {EXPECTED_NAN0S_CORE} NaN0S core files, found "
            f"{counts['nan0s_core']}")

    return {
        "root": str(root),
        "sdist_required": sdist_required,
        "wheel_required": wheel_required,
        "pairs": pairs,
        "counts": counts,
        "source_errors": source_errors,
    }


def _manifest_entries(text: str) -> Iterable[tuple[str, list[str]]]:
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        directive = parts[0].lower()
        if directive in {"include", "exclude", "global-include", "global-exclude"}:
            yield directive, parts[1:]
        elif directive in {"recursive-include", "recursive-exclude"}:
            yield directive, parts[1:]
        elif directive in {"graft", "prune"}:
            yield directive, parts[1:]


def manifest_covers(manifest_text: str, relative_path: str) -> bool:
    """Return True when MANIFEST.in includes *relative_path* (no excludes)."""
    rel = relative_path.replace("\\", "/").lstrip("./")
    base = rel.rsplit("/", 1)[-1]
    for directive, args in _manifest_entries(manifest_text):
        if directive == "include":
            if any(fnmatch.fnmatchcase(rel, pattern)
                   or fnmatch.fnmatchcase(base, pattern) for pattern in args):
                return True
        elif directive == "recursive-include" and args:
            directory = args[0].rstrip("/")
            if rel.startswith(directory + "/") and any(
                    fnmatch.fnmatchcase(base, pattern) for pattern in args[1:]):
                return True
        elif directive == "graft" and args:
            directory = args[0].rstrip("/")
            if rel.startswith(directory + "/"):
                return True
        elif directive == "global-include":
            if any(fnmatch.fnmatchcase(base, pattern) for pattern in args):
                return True
    return False


def _regex_package_data(text: str) -> list[str]:
    section = re.search(
        r"(?ms)^[ \t]*\[tool\.setuptools\.package-data\][ \t]*\n(.*?)(?=^[ \t]*\[|\Z)",
        text)
    if section is None:
        return []
    match = re.search(
        r"(?ms)^[ \t]*[\"']?nanlmind_hermes[\"']?[ \t]*=[ \t]*\[(.*?)\]",
        section.group(1))
    if match is None:
        return []
    return [
        first or second
        for first, second in re.findall(r'"([^"]+)"|\'([^\']+)\'', match.group(1))
    ]


def read_package_data(root: Path | str = ROOT) -> list[str]:
    """Read ``[tool.setuptools.package-data].nanlmind_hermes`` patterns."""
    text = (Path(root) / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        return _regex_package_data(text)
    try:
        data = tomllib.loads(text)
        patterns = data["tool"]["setuptools"]["package-data"][PACKAGE]
    except (KeyError, TypeError, ValueError):
        return []
    return [str(pattern) for pattern in patterns]


def package_data_missing(requirements: dict[str, Any],
                         patterns: Iterable[str]) -> list[str]:
    patterns = list(patterns)
    missing = []
    for path in requirements["wheel_required"]:
        relative = path[len(PACKAGE) + 1:]
        # Python modules ship through packages.find, not package-data.
        if relative.endswith(".py"):
            continue
        if not any(PurePosixPath(relative).match(pattern) for pattern in patterns):
            missing.append(relative)
    return sorted(set(missing))


def forbidden_member_reason(name: str) -> str | None:
    """Return a reason when *name* is a forbidden release artifact."""
    normalized = name.replace("\\", "/").lstrip("./")
    parts = [part for part in normalized.split("/") if part]
    if any(part == "__pycache__" for part in parts):
        return "python-bytecode-cache"
    if normalized.endswith((".pyc", ".pyo")):
        return "python-bytecode"
    if any(part == ".git" for part in parts):
        return "git-metadata"
    if _SECRET_NAME_RE.search(normalized):
        return "secret-state-or-model-artifact"
    return None


def _is_text_candidate(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    suffix = PurePosixPath(name).suffix.lower()
    return suffix in _TEXT_SUFFIXES or "." not in base


def _read_zip(source: Any) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    scanned = 0
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise PackagingVerificationError(
                f"archive has {len(infos)} members (limit {MAX_ARCHIVE_MEMBERS})")
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            size = info.file_size
            total += size
            data = None
            if (size <= MAX_TEXT_SCAN_BYTES
                    and scanned + size <= MAX_TOTAL_TEXT_SCAN_BYTES
                    and _is_text_candidate(name)):
                with archive.open(info) as handle:
                    data = handle.read(MAX_TEXT_SCAN_BYTES + 1)
                scanned += len(data)
            records.append({"name": name, "size": size, "data": data})
    return records, total


def _read_tar(source: Any) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    scanned = 0
    if isinstance(source, (str, Path)):
        context = tarfile.open(source, mode="r:*")
    else:
        context = tarfile.open(fileobj=source, mode="r:*")
    with context as archive:
        for member in archive:
            if not member.isfile():
                continue
            if len(records) >= MAX_ARCHIVE_MEMBERS:
                raise PackagingVerificationError(
                    f"archive has more than {MAX_ARCHIVE_MEMBERS} files")
            name = member.name
            size = member.size
            total += size
            data = None
            if (size <= MAX_TEXT_SCAN_BYTES
                    and scanned + size <= MAX_TOTAL_TEXT_SCAN_BYTES
                    and _is_text_candidate(name)):
                handle = archive.extractfile(member)
                if handle is not None:
                    data = handle.read(MAX_TEXT_SCAN_BYTES + 1)
                    scanned += len(data)
            records.append({"name": name, "size": size, "data": data})
    return records, total


def _normalize_names(kind: str, records: list[dict[str, Any]]) -> list[str]:
    names = [
        record["name"][2:] if record["name"].startswith("./") else record["name"]
        for record in records
    ]
    if kind == "sdist":
        candidates = {
            name.split("/", 1)[0] for name in names if "/" in name
        }
        prefix = ""
        for candidate in candidates:
            if f"{candidate}/{PACKAGE}/__init__.py" in names:
                prefix = candidate + "/"
                break
        if prefix:
            names = [name[len(prefix):] if name.startswith(prefix) else name
                     for name in names]
    return names


def _count(names: Iterable[str], pattern: str) -> int:
    return sum(1 for name in names if PurePosixPath(name).match(pattern))


def _evaluate_archive(kind: str, records: list[dict[str, Any]], total: int,
                      requirements: dict[str, Any], path: str,
                      archive_bytes: int) -> dict[str, Any]:
    names = _normalize_names(kind, records)
    name_set = set(names)
    required = (requirements["sdist_required"] if kind == "sdist"
                else requirements["wheel_required"])
    missing = sorted(set(required) - name_set)

    forbidden: list[dict[str, str]] = []
    oversized: list[dict[str, Any]] = []
    for record, name in zip(records, names):
        reason = forbidden_member_reason(name)
        if reason is not None:
            forbidden.append({"name": name, "reason": reason})
        size = int(record["size"])
        if size > MAX_MEMBER_BYTES:
            oversized.append({"name": name, "bytes": size})
        data = record.get("data")
        if data:
            for label, pattern in _SECRET_CONTENT_PATTERNS:
                if pattern.search(data):
                    forbidden.append({"name": name, "reason": "secret-content:" + label})

    counts = {
        "files": len(records),
        "uncompressed_bytes": total,
        "session_docs": _count(names, f"{PACKAGE}/session_docs/*.md"),
        "nan_kernels": _count(names, f"{PACKAGE}/nan/*.nan"),
        "nan_sources": _count(names, f"{PACKAGE}/nan/*.nas"),
        "mth_nan16": _count(names, f"{PACKAGE}/mth/nan16/*.nas"),
        "mth_fixtures": _count(names, f"{PACKAGE}/mth/tests/*.nas"),
        "nan0s_core": len({
            name for name in names
            if any(PurePosixPath(name).match(pattern) for pattern in (
                f"{PACKAGE}/nan0s/core/*.c",
                f"{PACKAGE}/nan0s/core/*.h",
                f"{PACKAGE}/nan0s/core/*.inc",
            ))
        }),
    }
    expected = requirements["counts"]
    pair_missing = [
        pair for pair in requirements["pairs"]
        if pair["source"] not in name_set or pair["fixture"] not in name_set
    ]
    checks = {
        "plugin_manifest": {
            "pass": kind == "wheel" or "plugin.yaml" in name_set,
            "applicable": kind == "sdist",
        },
        "package_init": {
            "pass": f"{PACKAGE}/__init__.py" in name_set,
        },
        "nan_kernels": {
            "pass": counts["nan_kernels"] >= expected["nan_kernels"],
            "expected": expected["nan_kernels"],
            "found": counts["nan_kernels"],
        },
        "session_docs": {
            "pass": counts["session_docs"] >= EXPECTED_SESSION_DOCS,
            "expected": EXPECTED_SESSION_DOCS,
            "found": counts["session_docs"],
        },
        "mth_manifest": {
            "pass": f"{PACKAGE}/mth/spec/wave-capabilities.json" in name_set,
            "expected_kernels": expected["mth_manifest_kernels"],
        },
        "mth_nan16_sources": {
            "pass": counts["mth_nan16"] >= expected["mth_nan16"],
            "expected": expected["mth_nan16"],
            "found": counts["mth_nan16"],
        },
        "mth_test_fixtures": {
            "pass": counts["mth_fixtures"] >= EXPECTED_MTH_KERNELS,
            "expected": EXPECTED_MTH_KERNELS,
            "found": counts["mth_fixtures"],
        },
        "mth_source_fixture_pairs": {
            "pass": not pair_missing,
            "expected": len(requirements["pairs"]),
            "missing": pair_missing,
        },
        "nan0s_core": {
            "pass": counts["nan0s_core"] >= EXPECTED_NAN0S_CORE,
            "expected": EXPECTED_NAN0S_CORE,
            "found": counts["nan0s_core"],
        },
        "nan0s_tools": {
            "pass": all(name in name_set for name in (
                f"{PACKAGE}/nan0s/tools/nan_cli.c",
                f"{PACKAGE}/nan0s/tools/nmk.awk",
            )),
        },
        "n16i_payload": {
            "pass": all(name in name_set for name in (
                f"{PACKAGE}/nan0s/src/tools/n16i.nas",
                f"{PACKAGE}/nan0s/src/tests/n16i_test.nas",
            )),
        },
        "no_forbidden_files": {
            "pass": not forbidden,
            "found": len(forbidden),
        },
        "size_bounds": {
            "pass": not oversized and total <= MAX_ARCHIVE_UNCOMPRESSED,
            "member_limit": MAX_MEMBER_BYTES,
            "archive_limit": MAX_ARCHIVE_UNCOMPRESSED,
        },
    }
    passed = (not missing and not forbidden and not oversized
              and total <= MAX_ARCHIVE_UNCOMPRESSED
              and all(check["pass"] for check in checks.values()))
    return {
        "kind": kind,
        "path": path,
        "bytes": archive_bytes,
        "pass": passed,
        "counts": counts,
        "missing": missing,
        "forbidden": forbidden,
        "oversized": oversized,
        "checks": checks,
    }


def inspect_archive(path: Path | str,
                    requirements: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    kind = "wheel" if path.suffix == ".whl" else "sdist"
    try:
        records, total = (_read_zip(path) if kind == "wheel"
                          else _read_tar(path))
        return _evaluate_archive(kind, records, total, requirements,
                                 str(path), path.stat().st_size)
    except (OSError, tarfile.TarError, zipfile.BadZipFile,
            PackagingVerificationError) as exc:
        return {
            "kind": kind,
            "path": str(path),
            "bytes": path.stat().st_size if path.exists() else 0,
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def inspect_archive_bytes(data: bytes, kind: str,
                          requirements: dict[str, Any],
                          label: str = "<memory>") -> dict[str, Any]:
    """Inspect in-memory wheel/sdist bytes; used by the static test suite."""
    source = io.BytesIO(data)
    try:
        records, total = (_read_zip(source) if kind == "wheel"
                          else _read_tar(source))
        return _evaluate_archive(kind, records, total, requirements,
                                 label, len(data))
    except (OSError, tarfile.TarError, zipfile.BadZipFile,
            PackagingVerificationError) as exc:
        return {
            "kind": kind,
            "path": label,
            "bytes": len(data),
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _module_available(module: str, timeout: float = 20.0) -> bool:
    with tempfile.TemporaryDirectory(prefix="nanlm-import-") as temporary:
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                cwd=temporary, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return False
    return result.returncode == 0


def real_build_available() -> bool:
    """True when ``python -m build`` or ``pip wheel`` can be attempted."""
    return _module_available("build") or _module_available("pip")


def _copy_source(root: Path, work: Path) -> Path:
    destination = work / "source"
    ignore = shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "*.pyo", "build", "dist",
        "*.egg-info", ".venv", ".mypy_cache", ".pytest_cache")
    shutil.copytree(root, destination, ignore=ignore, symlinks=True)
    return destination


def _trim(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def _run_command(label: str, command: list[str], cwd: Path,
                 timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"
    try:
        completed = subprocess.run(
            command, cwd=str(cwd), env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=max(1.0, timeout), check=False)
        return {
            "label": label,
            "command": command,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
            "output_tail": _trim(completed.stdout),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return {
            "label": label,
            "command": command,
            "returncode": None,
            "seconds": round(time.monotonic() - started, 3),
            "timed_out": True,
            "output_tail": _trim(str(output)),
        }
    except OSError as exc:
        return {
            "label": label,
            "command": command,
            "returncode": None,
            "seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
            "output_tail": f"{type(exc).__name__}: {exc}",
        }


def _has_artifact(output: Path, suffix: str) -> bool:
    return any(output.glob(f"*{suffix}"))


def _build_artifacts(source: Path, work: Path,
                     timeout: float) -> dict[str, Any]:
    output = work / "dist"
    output.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(1.0, timeout)

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    def attempt(label: str, command: list[str], cwd: Path) -> bool:
        if remaining() <= 0:
            attempts.append({
                "label": label, "command": command, "returncode": None,
                "seconds": 0.0, "timed_out": True,
                "output_tail": "skipped: overall build timeout reached",
            })
            return False
        result = _run_command(label, command, cwd, remaining())
        attempts.append(result)
        return result["returncode"] == 0

    python = sys.executable
    if _module_available("build"):
        if not attempt(
                "python -m build",
                [python, "-m", "build", "--outdir", str(output), str(source)],
                work):
            attempt(
                "python -m build --no-isolation",
                [python, "-m", "build", "--no-isolation", "--outdir",
                 str(output), str(source)],
                work)
    if not _has_artifact(output, ".whl"):
        attempt(
            "pip wheel",
            [python, "-m", "pip", "wheel", "--no-deps", "--wheel-dir",
             str(output), str(source)],
            work)
    if not _has_artifact(output, ".whl"):
        attempt(
            "pip wheel --no-build-isolation",
            [python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--wheel-dir", str(output), str(source)],
            work)
    if not _has_artifact(output, ".whl") and (source / "setup.py").is_file():
        attempt(
            "setup.py bdist_wheel",
            [python, "setup.py", "bdist_wheel", "--dist-dir", str(output)],
            source)
    if not _has_artifact(output, ".tar.gz"):
        if (source / "setup.py").is_file():
            attempt(
                "setup.py sdist",
                [python, "setup.py", "sdist", "--dist-dir", str(output)],
                source)
    if not _has_artifact(output, ".tar.gz"):
        sdist_script = (
            "import sys\n"
            "from setuptools import build_meta\n"
            "print(build_meta.build_sdist(sys.argv[1]))\n")
        attempt(
            "setuptools build_sdist",
            [python, "-c", sdist_script, str(output)],
            source)

    artifacts = sorted(
        [path for path in output.glob("*.whl")]
        + [path for path in output.glob("*.tar.gz")],
        key=lambda path: path.name)
    return {
        "attempts": attempts,
        "artifacts": [str(path) for path in artifacts],
        "wheel": next((str(path) for path in artifacts
                       if path.suffix == ".whl"), None),
        "sdist": next((str(path) for path in artifacts
                       if path.name.endswith(".tar.gz")), None),
        "work_dir": str(work),
    }


def verify_package(root: Path | str = ROOT,
                   timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Build and inspect release archives; return a JSON-serialisable receipt."""
    root = Path(root).resolve()
    requirements = build_requirements(root)
    errors = list(requirements["source_errors"])
    work = Path(tempfile.mkdtemp(prefix="nanlm-package-verify-"))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "root": str(root),
        "pass": False,
        "work_dir": str(work),
        "requirements": {
            "counts": requirements["counts"],
            "sdist_files": len(requirements["sdist_required"]),
            "wheel_files": len(requirements["wheel_required"]),
            "source_errors": requirements["source_errors"],
        },
        "build": {},
        "archives": [],
        "errors": errors,
        "warnings": [],
    }
    try:
        source = _copy_source(root, work)
        build = _build_artifacts(source, work, timeout)
        receipt["build"] = build
        if not build["artifacts"]:
            errors.append("no wheel or sdist was produced")
        if build["wheel"] is None:
            receipt["warnings"].append("wheel was not produced")
        if build["sdist"] is None:
            receipt["warnings"].append(
                "sdist was not produced; plugin.yaml could not be checked")
        for archive in build["artifacts"]:
            receipt["archives"].append(inspect_archive(archive, requirements))
    except Exception as exc:  # keep the receipt machine-readable on failure
        errors.append(f"{type(exc).__name__}: {exc}")

    archives = receipt["archives"]
    failed = [archive for archive in archives if not archive.get("pass")]
    if failed:
        errors.append(f"{len(failed)} archive(s) failed inspection")
    receipt["pass"] = bool(archives) and not failed and not errors
    receipt["errors"] = errors
    receipt["summary"] = {
        "archives": len(archives),
        "wheel": sum(1 for archive in archives if archive["kind"] == "wheel"),
        "sdist": sum(1 for archive in archives if archive["kind"] == "sdist"),
        "archives_passed": len(archives) - len(failed),
        "archives_failed": len(failed),
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify the NaNLMind HERMES wheel/sdist payload")
    parser.add_argument("--root", default=str(ROOT),
                        help="repository root (default: %(default)s)")
    parser.add_argument(
        "--timeout", type=float,
        default=float(os.environ.get("NANLM_PACKAGING_TIMEOUT", DEFAULT_TIMEOUT)),
        help="total build timeout in seconds (default: %(default)s)")
    args = parser.parse_args(argv)
    receipt = verify_package(args.root, timeout=args.timeout)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
