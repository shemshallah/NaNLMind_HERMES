#!/usr/bin/env python3
"""Bounded operator CLI for the unified resident universe.

The CLI is deliberately small and offline:

* every command prints one bounded JSON document;
* paths are normalised and home-directory prefixes are redacted;
* integer sizes are clamped and raw byte payloads are summarised by length;
* failures return a non-zero exit code (``2`` for usage, ``1`` otherwise);
* no network client is imported and no canonical merge is attempted.

The real backend is imported lazily from :mod:`nanlmind_hermes.resident` only
when a command runs.  That keeps ``--help`` and this module importable while
the resident package is still under construction.  Tests and embedders can
pass a stub backend to :func:`main` instead of touching the real package.

Backend protocol
----------------
A backend is either a callable or an object exposing::

    open_universe(path, create=False, size=None) -> universe
    validate_bundle(bundle) -> dict              # optional

A universe exposes::

    status() -> dict
    ingest_jsonl(source, limit=...) -> dict
    query(text, top_k=...) -> dict | list
    link(parent, child) -> dict
    export(destination, full=...) -> dict
    checkpoint() -> dict
    close() -> None                              # optional

Aliases used by the resident package are accepted as well, so a backend may
name the ingest source ``path`` or the export target ``output``.  A backend
that does not provide an operation fails closed with a JSON error instead of
being guessed at.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "resident-universe-cli/1"

# Keep these local so importing this script never imports the resident package.
MAX_BYTES = 1024 * 1024 * 1024
DEFAULT_INGEST_LIMIT = 1000
MAX_INGEST_LIMIT = 100_000
DEFAULT_TOP_K = 8
MAX_TOP_K = 64
MAX_TEXT_CHARS = 8192
MAX_PATH_CHARS = 4096
MAX_LEAF_CHARS = 256

# Output bounds.  These apply to every successful and failed JSON document.
MAX_ITEMS = 64
MAX_STRING = 512
MAX_DEPTH = 6
NODE_BUDGET = 2048
MAX_JSON_CHARS = 65_536
MAX_REPORTED_INT = 1 << 60

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SIZE_KEY_RE = re.compile(r"(?:^|_)(?:bytes?|sizes?|capacity|budget)(?:$|_)")
_SIZE_RE = re.compile(
    r"^(?P<number>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[kmgt]?i?b?|bytes?)?$",
    re.IGNORECASE,
)
_SIZE_FACTORS = {
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "k": 1024,
    "kb": 1024,
    "ki": 1024,
    "kib": 1024,
    "m": 1024 ** 2,
    "mb": 1024 ** 2,
    "mi": 1024 ** 2,
    "mib": 1024 ** 2,
    "g": 1024 ** 3,
    "gb": 1024 ** 3,
    "gi": 1024 ** 3,
    "gib": 1024 ** 3,
    "t": 1024 ** 4,
    "tb": 1024 ** 4,
    "ti": 1024 ** 4,
    "tib": 1024 ** 4,
}

OPEN_ALIASES = {
    "path": ("universe_path", "file", "filename", "source"),
    "create": ("create_if_missing", "new"),
    "size": ("bytes", "capacity", "budget"),
}
INGEST_ALIASES = {
    "source": ("path", "jsonl", "jsonl_path", "file", "filename"),
    "limit": ("max_records", "max_items", "count"),
}
QUERY_ALIASES = {
    "text": ("query_text", "query", "term", "q"),
    "top_k": ("k", "limit", "max_results", "count"),
}
LINK_ALIASES = {
    "parent": ("left", "source", "parent_leaf"),
    "child": ("right", "target", "child_leaf"),
}
EXPORT_ALIASES = {
    "destination": ("dest", "output", "directory", "dir", "bundle"),
    "full": ("complete", "include_payloads"),
}
VALIDATE_ALIASES = {
    "bundle": ("path", "directory", "dir", "bundle_path"),
}

_OPEN_FACTORIES_CREATE = (
    "create_universe",
    "create_resident_universe",
    "create_resident",
    "open_universe",
    "open_resident_universe",
    "open_store",
    "new_universe",
)
_OPEN_FACTORIES_OPEN = (
    "open_universe",
    "open_resident_universe",
    "open_store",
    "load_universe",
)
_UNIVERSE_CLASSES = (
    "ResidentUniverse",
    "ResidentUniverseStore",
    "UnifiedUniverse",
    "UniverseStore",
    "Universe",
    "ResidentStore",
    "Store",
)
_VALIDATOR_NAMES = (
    "validate_bundle",
    "validate_resident_bundle",
    "validate_export",
    "verify_bundle",
)
_BACKEND_SUBMODULES = (
    "universe", "store", "core", "persistence", "export", "intake",
    "runtime", "api",
)
_INGEST_METHODS = (
    "ingest_jsonl", "ingest_file", "ingest_jsonl_file", "ingest")
_QUERY_METHODS = (
    "query", "resonance_query", "resonance_search", "resonance", "search")
_LINK_METHODS = ("link", "link_leaves", "append_link", "add_link")
_EXPORT_METHODS = ("export", "export_bundle", "export_to", "write_bundle")
_STATUS_METHODS = ("status", "stats", "describe")
_CHECKPOINT_METHODS = ("checkpoint", "save_checkpoint", "flush")


class CliError(RuntimeError):
    """A fail-closed CLI error with a stable machine-readable code."""

    def __init__(self, message: str, code: str = "cli_error",
                 exit_code: int = 1):
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


class _Parser(argparse.ArgumentParser):
    """Argument parser that emits a JSON error instead of raw usage text."""

    def error(self, message):  # noqa: D401 - argparse override
        raise CliError(
            "invalid command line: " + _clean_text(message, 300),
            "usage",
            exit_code=2,
        )


class _Budget:
    """A shared node budget for one bounded JSON document."""

    __slots__ = ("remaining", "truncated")

    def __init__(self) -> None:
        self.remaining = NODE_BUDGET
        self.truncated = False

    def take(self) -> bool:
        if self.remaining <= 0:
            self.truncated = True
            return False
        self.remaining -= 1
        return True


def _home_text() -> str:
    try:
        home = os.path.abspath(os.path.expanduser("~"))
    except (OSError, RuntimeError):
        return ""
    return home if home and home != os.sep else ""


def _strip_home(text: str) -> str:
    home = _home_text()
    if not home:
        return text
    if text == home:
        return "~"
    marker = home + os.sep
    if text.startswith(marker):
        return "~" + text[len(home):]
    return text.replace(marker, "~" + os.sep)


def _clean_text(value: Any, limit: int = MAX_STRING) -> str:
    text = str(value)
    text = _strip_home(text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _CONTROL_RE.sub("", text)
    return text[:limit]


def _clean_path(value: Any) -> str:
    text = _strip_home(str(value))
    text = text.replace(os.sep, "/")
    text = _CONTROL_RE.sub("", text)
    return text[:MAX_PATH_CHARS]


def _display_path(value: Any) -> str:
    try:
        text = os.path.normpath(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError):
        text = str(value)
    return _clean_path(text)


def _safe_exception(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    text = _clean_text(text, 300)
    return text or type(exc).__name__


def _parse_size(value: str) -> int:
    text = str(value).strip().lower().replace("_", "")
    match = _SIZE_RE.fullmatch(text)
    if not match:
        raise ValueError(
            "size must be a positive byte count such as 16777216 or 16MiB")
    unit = (match.group("unit") or "").lower()
    factor = _SIZE_FACTORS.get(unit)
    if factor is None:
        raise ValueError("unsupported size unit; use bytes, KiB, MiB or GiB")
    try:
        number = float(match.group("number"))
    except ValueError:
        raise ValueError("size must be a number") from None
    if not math.isfinite(number):
        raise ValueError("size is too large")
    try:
        size = int(number * factor)
    except (OverflowError, ValueError):
        raise ValueError("size is too large") from None
    if size <= 0:
        raise ValueError("size must be greater than zero")
    return size


def _size_arg(value: str) -> int:
    try:
        return _parse_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _positive_int_arg(value: str) -> int:
    try:
        number = int(str(value), 10)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _checked_path(value: str, label: str = "path") -> Path:
    text = str(value)
    if not text or "\x00" in text:
        raise CliError(label + " must be a non-empty path", "invalid_path",
                       exit_code=2)
    if len(text) > MAX_PATH_CHARS:
        raise CliError(label + " is longer than %d characters" % MAX_PATH_CHARS,
                       "invalid_path", exit_code=2)
    try:
        return Path(text).expanduser()
    except (OSError, RuntimeError):
        raise CliError(label + " could not be expanded", "invalid_path",
                       exit_code=2) from None


def _checked_leaf(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise CliError(label + " must not be empty", "invalid_leaf",
                       exit_code=2)
    if len(text) > MAX_LEAF_CHARS:
        raise CliError(label + " is longer than %d characters" % MAX_LEAF_CHARS,
                       "invalid_leaf", exit_code=2)
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise CliError(label + " must not contain control characters",
                       "invalid_leaf", exit_code=2)
    return text


def _clean_key(value: Any) -> str:
    text = _clean_text(value, 128)
    return text or "_"


def _bounded(value: Any, budget: _Budget, depth: int = 0,
             key: str | None = None) -> Any:
    """Return a JSON-safe, depth/item/string bounded copy of ``value``."""
    if depth > MAX_DEPTH:
        budget.truncated = True
        return "[truncated]"
    if not budget.take():
        return "[truncated]"

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if key and _SIZE_KEY_RE.search(key.lower()) and value < 0:
            return 0
        return max(-MAX_REPORTED_INT, min(value, MAX_REPORTED_INT))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value) > MAX_REPORTED_INT:
            return math.copysign(MAX_REPORTED_INT, value)
        return value
    if isinstance(value, str):
        raw = str(value)
        text = _clean_text(raw, MAX_STRING + 1)
        if len(raw) > MAX_STRING or len(text) > MAX_STRING:
            budget.truncated = True
            text = text[:MAX_STRING]
        return text
    if isinstance(value, (bytes, bytearray, memoryview)):
        budget.truncated = True
        return {
            "type": "bytes",
            "bytes": min(len(value), MAX_REPORTED_INT),
            "note": "payload omitted",
        }
    if isinstance(value, Mapping):
        try:
            items = sorted(value.items(), key=lambda item: str(item[0]))
        except (TypeError, ValueError):
            items = list(value.items())
        out: dict[str, Any] = {}
        for index, (nested_key, nested_value) in enumerate(items):
            if index >= MAX_ITEMS:
                budget.truncated = True
                out["_truncated_items"] = len(items) - MAX_ITEMS
                break
            cleaned_key = _clean_key(nested_key)
            out[cleaned_key] = _bounded(
                nested_value, budget, depth + 1, cleaned_key)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        if isinstance(value, (set, frozenset)):
            sequence.sort(key=lambda item: repr(item))
        out_list: list[Any] = []
        for index, nested_value in enumerate(sequence):
            if index >= MAX_ITEMS:
                budget.truncated = True
                out_list.append(
                    "[truncated %d items]" % (len(sequence) - MAX_ITEMS))
                break
            out_list.append(_bounded(nested_value, budget, depth + 1, key))
        return out_list
    if isinstance(value, range):
        return _bounded(list(value), budget, depth + 1, key)

    for attribute in ("to_dict", "as_dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return _bounded(method(), budget, depth + 1, key)
            except Exception:
                break
    if hasattr(value, "__dict__"):
        return _bounded(vars(value), budget, depth + 1, key)
    return _clean_text(repr(value), MAX_STRING)


def _fallback_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": bool(payload.get("ok")),
        "command": payload.get("command") or "unknown",
        "canonical_merge": {"enabled": False, "attempted": False},
        "truncated": True,
    }
    if payload.get("error") is not None:
        document["error"] = payload["error"]
    else:
        document["result"] = {"note": "bounded output was truncated"}
    return document


def _encode(payload: Mapping[str, Any]) -> str:
    try:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        text = json.dumps(_fallback_document(payload), sort_keys=True,
                          ensure_ascii=True, separators=(",", ":"))
    if len(text) <= MAX_JSON_CHARS:
        return text
    text = json.dumps(_fallback_document(payload), sort_keys=True,
                      ensure_ascii=True, separators=(",", ":"))
    if len(text) <= MAX_JSON_CHARS:
        return text
    return json.dumps({
        "schema": SCHEMA,
        "ok": bool(payload.get("ok")),
        "command": "unknown",
        "canonical_merge": {"enabled": False, "attempted": False},
        "truncated": True,
    }, sort_keys=True, separators=(",", ":"))


def _write(payload: Mapping[str, Any], stream: Any) -> str:
    text = _encode(payload)
    try:
        print(text, file=stream)
    except BrokenPipeError:
        pass
    return text


def _adapted_kwargs(function: Any, kwargs: Mapping[str, Any],
                    aliases: Mapping[str, Sequence[str]],
                    label: str) -> tuple[list[Any], dict[str, Any]]:
    """Map logical keyword arguments onto a backend callable's signature."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return [], dict(kwargs)

    parameters = signature.parameters
    has_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values())
    adapted: dict[str, Any] = {}
    positional: list[Any] = []
    consumed: set[str] = set()

    for parameter in parameters.values():
        if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            continue
        for logical, value in kwargs.items():
            if logical in consumed:
                continue
            candidates = (logical,) + tuple(aliases.get(logical, ()))
            if parameter.name in candidates:
                positional.append(value)
                consumed.add(logical)
                break

    for logical, value in kwargs.items():
        if logical in consumed:
            continue
        candidates = (logical,) + tuple(aliases.get(logical, ()))
        for candidate in candidates:
            if candidate in parameters:
                adapted[candidate] = value
                consumed.add(logical)
                break
        else:
            if has_var_keyword:
                adapted[logical] = value

    missing = [
        name for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
        and parameter.default is inspect.Parameter.empty
        and name not in adapted
    ]
    if missing:
        raise CliError(
            "%s does not accept required argument(s): %s"
            % (label, ", ".join(missing)),
            "backend_contract_error",
        )
    return positional, adapted


def _call_adapted(function: Any, kwargs: Mapping[str, Any],
                  aliases: Mapping[str, Sequence[str]], label: str) -> Any:
    positional, adapted = _adapted_kwargs(function, kwargs, aliases, label)
    return function(*positional, **adapted)


def _invoke(universe: Any, method_names: Sequence[str],
            kwargs: Mapping[str, Any], aliases: Mapping[str, Sequence[str]],
            label: str) -> Any:
    for name in method_names:
        method = getattr(universe, name, None)
        if callable(method):
            return _call_adapted(method, kwargs, aliases, label)
    raise CliError(
        "resident universe backend does not implement " + label,
        "unsupported_operation",
    )


def _status(universe: Any) -> Any:
    for name in _STATUS_METHODS:
        value = getattr(universe, name, None)
        if callable(value):
            return _call_adapted(value, {}, {}, "status")
        if value is not None:
            return value
    raise CliError("resident universe backend does not implement status",
                   "unsupported_operation")


def _optional_status(universe: Any) -> Any:
    try:
        return _status(universe)
    except Exception:
        # Status is informational during init; a status failure must not turn
        # a successful create into a failed command.
        return None


def _close_universe(universe: Any) -> None:
    closer = getattr(universe, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            # A close/release failure must not replace the bounded report.
            pass


class _ResidentBackend:
    """Lazily discover the evolving ``nanlmind_hermes.resident`` backend."""

    def __init__(self) -> None:
        self._modules: list[Any] | None = None

    def _load_modules(self) -> list[Any]:
        if self._modules is not None:
            return self._modules
        try:
            resident = importlib.import_module("nanlmind_hermes.resident")
        except Exception as exc:
            raise CliError(
                "resident universe backend is unavailable: " + _safe_exception(exc),
                "backend_unavailable",
            ) from None
        modules = [resident]
        for suffix in _BACKEND_SUBMODULES:
            try:
                modules.append(importlib.import_module(
                    "nanlmind_hermes.resident." + suffix))
            except ImportError:
                continue
            except Exception:
                # A partly available submodule must not take down the package
                # when the aggregate module still exposes the public API.
                continue
        self._modules = modules
        return modules

    @staticmethod
    def _find(modules: Sequence[Any], names: Sequence[str],
              predicate: Any = None) -> Any:
        for module in modules:
            for name in names:
                value = getattr(module, name, None)
                if value is None:
                    continue
                if predicate is not None and not predicate(value):
                    continue
                return value
        return None

    def open_universe(self, path: Path, create: bool = False,
                      size: int | None = None) -> Any:
        modules = self._load_modules()
        names = _OPEN_FACTORIES_CREATE if create else _OPEN_FACTORIES_OPEN
        factory = self._find(modules, names, callable)
        kwargs: dict[str, Any] = {"path": path, "create": create}
        if size is not None:
            kwargs["size"] = size
        if factory is not None:
            try:
                return _call_adapted(factory, kwargs, OPEN_ALIASES,
                                     "open universe")
            except CliError:
                raise
            except Exception as exc:
                raise CliError(
                    "could not open resident universe: " + _safe_exception(exc),
                    "open_failed",
                ) from None

        universe_class = self._find(
            modules, _UNIVERSE_CLASSES, lambda value: isinstance(value, type))
        if universe_class is None:
            raise CliError(
                "resident universe backend is unavailable: no open factory "
                "or universe class found",
                "backend_unavailable",
            )
        return self._open_class(universe_class, path, create, size)

    def _open_class(self, universe_class: type, path: Path, create: bool,
                    size: int | None) -> Any:
        kwargs: dict[str, Any] = {"path": path, "create": create}
        if size is not None:
            kwargs["size"] = size
        names = ("create", "initialize", "new", "open", "load") if create \
            else ("open", "load")
        for name in names:
            try:
                raw = inspect.getattr_static(universe_class, name)
            except AttributeError:
                continue
            if not isinstance(raw, (classmethod, staticmethod)):
                continue
            method = getattr(universe_class, name)
            try:
                return _call_adapted(method, kwargs, OPEN_ALIASES,
                                     "open universe")
            except CliError:
                raise
            except Exception as exc:
                raise CliError(
                    "could not open resident universe: " + _safe_exception(exc),
                    "open_failed",
                ) from None
        try:
            return _call_adapted(universe_class, kwargs, OPEN_ALIASES,
                                 "open universe")
        except CliError:
            raise
        except Exception as exc:
            raise CliError(
                "could not open resident universe: " + _safe_exception(exc),
                "open_failed",
            ) from None

    def validate_bundle(self, bundle: Path) -> Any:
        modules = self._load_modules()
        for module in modules:
            for name in _VALIDATOR_NAMES:
                validator = getattr(module, name, None)
                if callable(validator):
                    try:
                        return _call_adapted(
                            validator, {"bundle": bundle}, VALIDATE_ALIASES,
                            "validate bundle")
                    except CliError:
                        raise
                    except Exception as exc:
                        raise CliError(
                            "bundle validation failed: " + _safe_exception(exc),
                            "invalid_bundle",
                        ) from None
        universe_class = self._find(
            modules, _UNIVERSE_CLASSES, lambda value: isinstance(value, type))
        if universe_class is not None:
            for name in _VALIDATOR_NAMES + ("validate",):
                try:
                    raw = inspect.getattr_static(universe_class, name)
                except AttributeError:
                    continue
                if not isinstance(raw, (classmethod, staticmethod)):
                    continue
                validator = getattr(universe_class, name)
                try:
                    return _call_adapted(
                        validator, {"bundle": bundle}, VALIDATE_ALIASES,
                        "validate bundle")
                except CliError:
                    raise
                except Exception as exc:
                    raise CliError(
                        "bundle validation failed: " + _safe_exception(exc),
                        "invalid_bundle",
                    ) from None
        raise CliError(
            "resident universe validator is unavailable; the backend does "
            "not expose validate_bundle",
            "backend_unavailable",
        )


def _open_universe(backend: Any, path: Path, create: bool,
                   size: int | None = None) -> Any:
    if backend is None:
        backend = _ResidentBackend()
    opener = getattr(backend, "open_universe", None)
    if opener is None and callable(backend):
        opener = backend
    if opener is None:
        opener = getattr(backend, "open", None)
    if not callable(opener):
        raise CliError(
            "backend does not expose open_universe", "backend_unavailable")
    kwargs: dict[str, Any] = {"path": path, "create": create}
    if size is not None:
        kwargs["size"] = size
    try:
        return _call_adapted(opener, kwargs, OPEN_ALIASES, "open universe")
    except CliError:
        raise
    except Exception as exc:
        raise CliError(
            "could not open resident universe: " + _safe_exception(exc),
            "open_failed",
        ) from None


def _validate_bundle(backend: Any, bundle: Path) -> Any:
    if backend is None:
        backend = _ResidentBackend()
    validator = getattr(backend, "validate_bundle", None)
    if not callable(validator):
        raise CliError(
            "backend does not expose validate_bundle", "backend_unavailable")
    try:
        return _call_adapted(
            validator, {"bundle": bundle}, VALIDATE_ALIASES,
            "validate bundle")
    except CliError:
        raise
    except Exception as exc:
        raise CliError(
            "bundle validation failed: " + _safe_exception(exc),
            "invalid_bundle",
        ) from None


def _operation_error(result: Any, false_keys: Sequence[str]) -> str | None:
    if isinstance(result, bool):
        return None if result else "operation reported failure"
    if isinstance(result, Mapping):
        if result.get("ok") is False:
            error = result.get("error")
            if isinstance(error, Mapping):
                message = error.get("message") or error.get("code")
                if message:
                    return _clean_text(message, 300)
            if isinstance(error, str) and error:
                return _clean_text(error, 300)
            return _clean_text(
                result.get("message") or result.get("reason")
                or "operation reported failure", 300)
        for key in false_keys:
            if result.get(key) is False:
                message = result.get("message") or result.get("reason")
                return _clean_text(message or (key + " reported false"), 300)
    return None


def _validation_error(result: Any) -> str | None:
    if isinstance(result, bool):
        return None if result else "bundle is invalid"
    if isinstance(result, int) and not isinstance(result, bool):
        return None if result else "bundle is invalid"
    if isinstance(result, Mapping):
        if (result.get("ok") is False or result.get("valid") is False
                or result.get("complete") is False):
            message = result.get("message") or result.get("reason")
            if message:
                return _clean_text(message, 300)
            errors = result.get("errors")
            if isinstance(errors, (list, tuple)) and errors:
                first = errors[0]
                if isinstance(first, str) and first:
                    return _clean_text(first, 300)
                return "bundle validation reported errors"
            return "bundle is invalid"
    return None


def _count_results(result: Any) -> int | None:
    if isinstance(result, (list, tuple)):
        return len(result)
    if isinstance(result, Mapping):
        for key in ("results", "matches", "hits", "leaves", "items", "rows"):
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                return len(value)
        for key in ("count", "returned", "total"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _trim_query_result(result: Any, top_k: int) -> Any:
    if isinstance(result, (list, tuple)):
        return list(result)[:top_k]
    if isinstance(result, Mapping):
        copy = dict(result)
        for key in ("results", "matches", "hits", "leaves", "items", "rows"):
            value = copy.get(key)
            if isinstance(value, (list, tuple)):
                copy[key] = list(value)[:top_k]
                return copy
    return result


def _success(command: str, result: Any,
             subject: Mapping[str, Any]) -> dict[str, Any]:
    budget = _Budget()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": True,
        "command": command,
        "canonical_merge": {"enabled": False, "attempted": False},
    }
    for key, value in subject.items():
        payload[key] = _bounded(value, budget)
    payload["result"] = _bounded({} if result is None else result, budget)
    payload["truncated"] = budget.truncated
    return payload


def _report(command: str, result: Any, subject: Mapping[str, Any],
            false_keys: Sequence[str] = (),
            failure_code: str = "operation_failed") -> tuple[dict[str, Any], int]:
    payload = _success(command, result, subject)
    message = _operation_error(result, false_keys)
    if message is None:
        return payload, 0
    payload["ok"] = False
    payload["error"] = {"code": failure_code, "message": message}
    return payload, 1


def _error_payload(command: str | None, error: CliError) -> dict[str, Any]:
    message = _clean_text(str(error), 300) or error.code
    return {
        "schema": SCHEMA,
        "ok": False,
        "command": command or "unknown",
        "canonical_merge": {"enabled": False, "attempted": False},
        "truncated": False,
        "error": {"code": error.code, "message": message},
    }


def _cmd_init(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    requested = int(args.size)
    size = min(requested, MAX_BYTES)
    if path.exists():
        try:
            if path.is_dir() or path.stat().st_size > 0:
                raise CliError(
                    "universe path already exists and is not empty",
                    "already_exists")
        except OSError:
            raise CliError("cannot inspect universe path", "path_unreadable") \
                from None
    universe = _open_universe(backend, path, create=True, size=size)
    try:
        result: dict[str, Any] = {"created": True}
        status = _optional_status(universe)
        if status is not None:
            result["status"] = status
    finally:
        _close_universe(universe)
    subject = {
        "path": _display_path(path),
        "bytes": size,
        "requested_bytes": requested,
        "size_clamped": size != requested,
    }
    return _report("init", result, subject)


def _cmd_status(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    universe = _open_universe(backend, path, create=False)
    try:
        result = _status(universe)
    finally:
        _close_universe(universe)
    return _report("status", result, {"path": _display_path(path)})


def _cmd_ingest(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    source = _checked_path(args.jsonl, "--jsonl")
    if not source.is_file():
        raise CliError(
            "JSONL source must be an existing regular file", "invalid_input")
    requested_limit = int(args.limit)
    limit = min(max(requested_limit, 1), MAX_INGEST_LIMIT)
    universe = _open_universe(backend, path, create=False)
    try:
        result = _invoke(
            universe, _INGEST_METHODS,
            {"source": source, "limit": limit},
            INGEST_ALIASES, "ingest JSONL")
    finally:
        _close_universe(universe)
    if result is None:
        result = {"ingested": 0, "limit": limit}
    elif isinstance(result, int) and not isinstance(result, bool):
        result = {"ingested": result, "limit": limit}
    subject = {
        "path": _display_path(path),
        "source": _display_path(source),
        "limit": limit,
        "limit_clamped": limit != requested_limit,
    }
    return _report("ingest", result, subject, ("ok", "success"))


def _cmd_query(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    text = str(args.text)
    if not text.strip():
        raise CliError("--text must not be empty", "invalid_text", exit_code=2)
    if len(text) > MAX_TEXT_CHARS:
        raise CliError(
            "--text is longer than %d characters" % MAX_TEXT_CHARS,
            "invalid_text", exit_code=2)
    requested_top_k = int(args.top_k)
    top_k = min(max(requested_top_k, 1), MAX_TOP_K)
    universe = _open_universe(backend, path, create=False)
    try:
        result = _invoke(
            universe, _QUERY_METHODS,
            {"text": text, "top_k": top_k},
            QUERY_ALIASES, "query")
    finally:
        _close_universe(universe)
    result = _trim_query_result(result, top_k)
    subject: dict[str, Any] = {
        "path": _display_path(path),
        "top_k": top_k,
        "top_k_clamped": top_k != requested_top_k,
        "text_chars": len(text),
    }
    returned = _count_results(result)
    if returned is not None:
        subject["returned"] = returned
    return _report("query", result, subject, ("ok", "success"))


def _cmd_link(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    parent = _checked_leaf(args.parent, "--parent")
    child = _checked_leaf(args.child, "--child")
    universe = _open_universe(backend, path, create=False)
    try:
        result = _invoke(
            universe, _LINK_METHODS,
            {"parent": parent, "child": child},
            LINK_ALIASES, "link")
    finally:
        _close_universe(universe)
    if result is None:
        result = {"linked": True}
    elif isinstance(result, bool):
        result = {"linked": result}
    subject = {
        "path": _display_path(path),
        "parent": _clean_text(parent, MAX_LEAF_CHARS),
        "child": _clean_text(child, MAX_LEAF_CHARS),
    }
    return _report("link", result, subject, ("linked", "ok", "success"))


def _cmd_export(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    destination = _checked_path(args.destination, "--destination")
    full = bool(args.full)
    try:
        same = os.path.abspath(os.fspath(path)) == os.path.abspath(
            os.fspath(destination))
    except (OSError, ValueError):
        same = False
    if same:
        raise CliError(
            "export destination must differ from the universe path",
            "invalid_destination", exit_code=2)
    universe = _open_universe(backend, path, create=False)
    try:
        result = _invoke(
            universe, _EXPORT_METHODS,
            {"destination": destination, "full": full},
            EXPORT_ALIASES, "export")
    finally:
        _close_universe(universe)
    if result is None:
        result = {"exported": True, "full": full}
    subject = {
        "path": _display_path(path),
        "destination": _display_path(destination),
        "full": full,
    }
    return _report("export", result, subject, ("exported", "ok", "complete"))


def _cmd_validate(args: argparse.Namespace, backend: Any):
    bundle = _checked_path(args.bundle, "--bundle")
    if not bundle.is_dir():
        raise CliError(
            "bundle directory does not exist or is not a directory",
            "invalid_bundle")
    result = _validate_bundle(backend, bundle)
    if result is None:
        raise CliError(
            "validator returned no result", "backend_contract_error")
    subject = {"bundle": _display_path(bundle)}
    payload = _success("validate", result, subject)
    message = _validation_error(result)
    if message is None:
        return payload, 0
    payload["ok"] = False
    payload["error"] = {"code": "invalid_bundle", "message": message}
    return payload, 1


def _cmd_checkpoint(args: argparse.Namespace, backend: Any):
    path = _checked_path(args.path, "--path")
    universe = _open_universe(backend, path, create=False)
    try:
        result = _invoke(
            universe, _CHECKPOINT_METHODS, {}, {}, "checkpoint")
    finally:
        _close_universe(universe)
    if result is None:
        result = {"checkpointed": True}
    return _report(
        "checkpoint", result, {"path": _display_path(path)},
        ("checkpointed", "ok", "success"))


_HANDLERS = {
    "init": _cmd_init,
    "status": _cmd_status,
    "ingest": _cmd_ingest,
    "query": _cmd_query,
    "link": _cmd_link,
    "export": _cmd_export,
    "validate": _cmd_validate,
    "checkpoint": _cmd_checkpoint,
}


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="resident-universe-cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND")

    init = subparsers.add_parser(
        "init", help="create a bounded resident universe file",
        allow_abbrev=False)
    init.add_argument("--path", required=True, metavar="PATH",
                      help="resident universe file to create")
    init.add_argument("--size", required=True, type=_size_arg, metavar="BYTES",
                      help="requested capacity; values above 1 GiB are clamped")

    status = subparsers.add_parser(
        "status", help="print bounded universe status",
        allow_abbrev=False)
    status.add_argument("--path", required=True, metavar="PATH")

    ingest = subparsers.add_parser(
        "ingest", help="ingest bounded JSONL records",
        allow_abbrev=False)
    ingest.add_argument("--path", required=True, metavar="PATH")
    ingest.add_argument("--jsonl", required=True, metavar="FILE")
    ingest.add_argument("--limit", type=_positive_int_arg,
                        default=DEFAULT_INGEST_LIMIT, metavar="N",
                        help="maximum records; values above %d are clamped"
                             % MAX_INGEST_LIMIT)

    query = subparsers.add_parser(
        "query", help="run a bounded resonance query",
        allow_abbrev=False)
    query.add_argument("--path", required=True, metavar="PATH")
    query.add_argument("--text", required=True, metavar="TEXT")
    query.add_argument("--top-k", type=_positive_int_arg, default=DEFAULT_TOP_K,
                       metavar="K",
                       help="maximum results; values above %d are clamped"
                            % MAX_TOP_K)

    link = subparsers.add_parser(
        "link", help="link a parent leaf to a child leaf",
        allow_abbrev=False)
    link.add_argument("--path", required=True, metavar="PATH")
    link.add_argument("--parent", required=True, metavar="LEAF")
    link.add_argument("--child", required=True, metavar="LEAF")

    export = subparsers.add_parser(
        "export", help="export a bounded bundle",
        allow_abbrev=False)
    export.add_argument("--path", required=True, metavar="PATH")
    export.add_argument("--destination", required=True, metavar="DIR")
    export.add_argument("--full", action="store_true",
                        help="include full payloads instead of the index view")

    validate = subparsers.add_parser(
        "validate", help="validate an exported bundle directory",
        allow_abbrev=False)
    validate.add_argument("--bundle", required=True, metavar="DIR")

    checkpoint = subparsers.add_parser(
        "checkpoint", help="checkpoint a resident universe",
        allow_abbrev=False)
    checkpoint.add_argument("--path", required=True, metavar="PATH")

    return parser


def _peek_command(argv: Sequence[str] | None) -> str | None:
    values = list(sys.argv[1:] if argv is None else argv)
    for value in values[:16]:
        if value in _HANDLERS:
            return value
    return None


def main(argv: Sequence[str] | None = None, *,
         backend: Any = None,
         stdout: Any = None,
         stderr: Any = None) -> int:
    """Run one command and return its process exit code.

    ``backend`` is the dependency-injection seam for tests.  When it is
    omitted the resident package is imported lazily by ``_ResidentBackend``.
    ``stdout``/``stderr`` are injectable streams and default to the process
    streams.
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    command = _peek_command(argv)
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        command = args.command
        handler = _HANDLERS.get(command)
        if handler is None:
            raise CliError("unknown command", "usage", exit_code=2)
        payload, exit_code = handler(args, backend)
    except CliError as exc:
        _write(_error_payload(command, exc), err)
        return exc.exit_code
    except SystemExit:
        raise
    except Exception as exc:
        error = CliError(
            "internal error: " + _safe_exception(exc), "internal_error")
        _write(_error_payload(command, error), err)
        return 1
    _write(payload, out)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
