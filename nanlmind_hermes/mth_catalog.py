"""Bundled NaN16 MTH kernel catalog.

The manifest and every kernel/fixture source ship inside the plugin package
under ``nanlmind_hermes/mth``. This module only validates and indexes that
manifest; it does not execute a kernel. Native execution is provided by
``nan0s_runtime`` using a host-built or bundled portable NaN0S core.

This is the same 91-kernel contract as NaNROUTER ``spec/wave-capabilities.json``
with source paths rewritten for the plugin bundle. A catalog entry is not a
mathematical proof or a claim that every kernel has a general operand ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

SCHEMA = "nanlm.mth-catalog/1"
MANIFEST_SCHEMA = "nanrouter.wave-capabilities.v1"
PRECISIONS = {"float32", "u16->i32", "u32", "q16.16", "q32.32",
              "bool", "i32", "u32->u64"}
STATUSES = {"declared", "source_written", "native_verified", "degraded", "revoked"}
MAX_CATALOG_PAGE = 100

# The only kernels with an admitted dynamic program/operator dispatcher in
# mth/nan16/mth_execute.nas. Every other entry is native-fixture verifiable.
NATIVE_DISPATCH = frozenset({
    "mth.vector.dot256.f32",
    "mth.vector.cosine256.f32",
    "mth.vector.add256.f32",
    "mth.vector.sub256.f32",
})

_MTH_ROOT = Path(__file__).resolve().with_name("mth")
_MANIFEST = _MTH_ROOT / "spec" / "wave-capabilities.json"
_CACHE: tuple["MthKernel", ...] | None = None


class MthCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class MthKernel:
    kernel_id: str
    source: str
    fixture: str
    marker: str
    steps: int
    precision: str
    workspace_words: int
    shape: tuple[int, ...]
    status: str
    ci_verified: bool
    version: int
    tags: tuple[str, ...]
    source_digest: str
    fixture_digest: str
    source_bytes: int
    fixture_bytes: int

    @property
    def family(self) -> str:
        return self.kernel_id.split(".")[1]

    @property
    def execution_mode(self) -> str:
        return ("native_dispatch_abi" if self.kernel_id in NATIVE_DISPATCH
                else "native_fixture_only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "version": self.version,
            "family": self.family,
            "source": self.source,
            "fixture": self.fixture,
            "marker": self.marker,
            "steps": self.steps,
            "precision": self.precision,
            "workspace_words": self.workspace_words,
            "shape": list(self.shape),
            "status": self.status,
            "ci_verified": self.ci_verified,
            "tags": list(self.tags),
            "source_digest": self.source_digest,
            "fixture_digest": self.fixture_digest,
            "source_bytes": self.source_bytes,
            "fixture_bytes": self.fixture_bytes,
            "execution_mode": self.execution_mode,
            "dynamic_execution_available": self.kernel_id in NATIVE_DISPATCH,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise MthCatalogError("invalid " + field)
    path = Path(value)
    if ".." in path.parts:
        raise MthCatalogError(field + " escapes the MTH bundle")
    return value


def _int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MthCatalogError("invalid " + field)
    return value


def _derive_tags(entry: dict[str, Any], family: str) -> tuple[str, ...]:
    declared = entry.get("tags")
    if isinstance(declared, list):
        values = {str(item).lower() for item in declared if isinstance(item, str)}
    else:
        values = set()
    shape = entry["shape"]
    values.update({
        "family:" + family,
        "precision:" + str(entry["precision"]).lower().replace("->", "-"),
        "status:" + str(entry["status"]),
        "ci:" + ("verified" if entry["ci_verified"] else "unverified"),
        "workspace:" + ("zero" if entry["workspace_words"] == 0 else "bounded"),
    })
    if shape:
        values.add("shape:" + "x".join(str(item) for item in shape))
    return tuple(sorted(values))


def _load_unchecked() -> tuple[MthKernel, ...]:
    if not _MANIFEST.is_file():
        raise MthCatalogError("bundled MTH manifest is missing")
    try:
        data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MthCatalogError("cannot read bundled MTH manifest") from exc
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise MthCatalogError("unexpected MTH manifest schema")
    entries = data.get("kernels")
    if not isinstance(entries, list) or not entries:
        raise MthCatalogError("MTH manifest has no kernels")
    kernels = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise MthCatalogError("MTH kernel entry must be an object")
        kernel_id = entry.get("kernel_id")
        if (not isinstance(kernel_id, str)
                or re.fullmatch(r"mth\.[a-z0-9_.-]+", kernel_id) is None
                or kernel_id in seen):
            raise MthCatalogError("invalid or duplicate MTH kernel_id")
        seen.add(kernel_id)
        source = _safe_relative(entry.get("source"), "source")
        fixture = _safe_relative(entry.get("fixture"), "fixture")
        source_path = _MTH_ROOT / source
        fixture_path = _MTH_ROOT / fixture
        if not source_path.is_file() or not fixture_path.is_file():
            raise MthCatalogError("bundled source/fixture is missing: " + kernel_id)
        marker = entry.get("marker")
        if not isinstance(marker, str) or not marker.strip():
            raise MthCatalogError("invalid marker: " + kernel_id)
        precision = entry.get("precision")
        if precision not in PRECISIONS:
            raise MthCatalogError("invalid precision: " + kernel_id)
        status = entry.get("status")
        if status not in STATUSES:
            raise MthCatalogError("invalid status: " + kernel_id)
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape:
            raise MthCatalogError("invalid shape: " + kernel_id)
        shape_tuple = tuple(_int(item, "shape dimension", 1) for item in shape)
        family = kernel_id.split(".")[1]
        kernels.append(MthKernel(
            kernel_id=kernel_id, source=source, fixture=fixture, marker=marker,
            steps=_int(entry.get("steps"), "steps", 1),
            precision=precision,
            workspace_words=_int(entry.get("workspace_words"), "workspace_words"),
            shape=shape_tuple, status=status,
            ci_verified=entry.get("ci_verified") is True,
            version=_int(entry.get("version", 1), "version", 1),
            tags=_derive_tags(entry, family),
            source_digest=_sha256(source_path),
            fixture_digest=_sha256(fixture_path),
            source_bytes=source_path.stat().st_size,
            fixture_bytes=fixture_path.stat().st_size,
        ))
    return tuple(kernels)


def kernels() -> tuple[MthKernel, ...]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_unchecked()
    return _CACHE


def reset_cache() -> None:
    global _CACHE
    _CACHE = None


def lookup(kernel_id: str) -> MthKernel | None:
    for kernel in kernels():
        if kernel.kernel_id == kernel_id:
            return kernel
    return None


def require(kernel_id: str) -> MthKernel:
    kernel = lookup(kernel_id)
    if kernel is None:
        raise MthCatalogError("unknown MTH kernel: " + str(kernel_id))
    return kernel


def source_path(kernel_id: str) -> Path:
    return _MTH_ROOT / require(kernel_id).source


def fixture_path(kernel_id: str) -> Path:
    return _MTH_ROOT / require(kernel_id).fixture


def bundle_root() -> Path:
    return _MTH_ROOT


def health() -> dict[str, Any]:
    entries = kernels()
    families: dict[str, int] = {}
    precisions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    verified = 0
    for kernel in entries:
        families[kernel.family] = families.get(kernel.family, 0) + 1
        precisions[kernel.precision] = precisions.get(kernel.precision, 0) + 1
        statuses[kernel.status] = statuses.get(kernel.status, 0) + 1
        verified += 1 if kernel.ci_verified else 0
    return {
        "schema": SCHEMA,
        "manifest": str(_MANIFEST),
        "total": len(entries),
        "ci_verified": verified,
        "families": dict(sorted(families.items())),
        "precisions": dict(sorted(precisions.items())),
        "statuses": dict(sorted(statuses.items())),
        "native_dispatch": sorted(NATIVE_DISPATCH),
        "native_dispatch_count": len(NATIVE_DISPATCH),
    }


def list_kernels(*, family: str | None = None,
                 tags: Iterable[str] | None = None,
                 offset: int = 0, limit: int = 32,
                 require_ci: bool = False) -> dict[str, Any]:
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1_000_000:
        raise MthCatalogError("offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CATALOG_PAGE:
        raise MthCatalogError("limit must be between 1 and " + str(MAX_CATALOG_PAGE))
    wanted = tuple(str(item) for item in (tags or ()))
    if family is not None and (not isinstance(family, str) or not family.isalnum()):
        raise MthCatalogError("invalid family")
    matches = []
    for kernel in kernels():
        if family is not None and kernel.family != family:
            continue
        if require_ci and not (kernel.ci_verified and kernel.status == "native_verified"):
            continue
        if wanted and not all(tag in kernel.tags for tag in wanted):
            continue
        matches.append(kernel)
    selected = matches[offset:offset + limit]
    return {
        "schema": SCHEMA,
        "total": len(matches),
        "offset": offset,
        "next_offset": (offset + len(selected)
                        if offset + len(selected) < len(matches) else None),
        "kernels": [kernel.to_dict() for kernel in selected],
    }


def inspect(kernel_id: str) -> dict[str, Any]:
    kernel = require(kernel_id)
    return {
        "schema": SCHEMA,
        "kernel": {
            **kernel.to_dict(),
            "source_path": str(source_path(kernel_id)),
            "fixture_path": str(fixture_path(kernel_id)),
            "bundled": True,
        },
    }


def catalog_document() -> dict[str, Any]:
    return {
        **health(),
        "bundle_root": str(_MTH_ROOT),
        "kernels": [kernel.to_dict() for kernel in kernels()],
    }
