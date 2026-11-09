"""Read-only adapter for the canonical NaNLMind unified-build contract.

NaNLMind owns the topology and the native build.  HERMES never embeds a second
copy of the constants as authority: it reads ``spec/unified-build.v1.json`` from
a configured NaNLMind checkout (sibling, current directory, or home fallback)
and, when available, checks the native ``unification`` command.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "nanlm.unified-build/1"
CONTRACT_RELATIVE = Path("spec") / "unified-build.v1.json"
BINARY_RELATIVE = Path("build") / "nanlm-universal"
EXPECTED_TOPOLOGY = {
    "heads": 256,
    "triad_member_heads": 255,
    "field_head": 255,
    "triads": 85,
    "mirror_pairs": 32640,
    "canonical_cells": 32896,
    "chart_bytes": 8388608,
    "hypermatrix_rows": 32640,
    "hypermatrix_row_bits": 256,
    "resolution_layers": 10,
    "transform_passes": 10,
    "runner_slots": 6,
    "active_runner_slots": 1,
}
EXPECTED_ROLE_VIEWS = {
    "physical_heads": 256,
    "mp_writer_eligible_heads": 255,
    "boolean_curriculum_heads": 250,
    "nannet_root_control_head": 0,
    "triad_member_heads": 255,
    "triad_field_head": 255,
}
EXPECTED_NATIVE = {
    "canonical_repository": "NaNLMind",
    "heads": "256",
    "triad_member_heads": "255",
    "field_head": "255",
    "triads": "85",
    "mp_writer_eligible_heads": "255",
    "boolean_curriculum_heads": "250",
    "nannet_root_control_head": "0",
    "bsr_control_heads": "6",
    "mirror_pairs": "32640",
    "chart_bytes": "8388608",
}


class UnificationError(ValueError):
    """The canonical contract is missing, malformed, or contradictory."""


def _candidate_roots() -> Iterable[Path]:
    here = Path(__file__).resolve()
    # repository root is .../NaNLMind_HERMES
    yield here.parents[1].parent / "NaNLMind"
    yield Path.cwd() / "NaNLMind"
    yield Path.home() / "NaNLMind"


def find_contract() -> Path:
    for root in _candidate_roots():
        path = root / CONTRACT_RELATIVE
        if path.is_file():
            return path
    raise UnificationError(
        "canonical NaNLMind contract not found; configure nanlmind_root or install a sibling NaNLMind checkout"
    )


def load(path: Path | None = None) -> dict[str, Any]:
    contract_path = Path(path) if path else find_contract()
    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnificationError(f"cannot read unified-build contract: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise UnificationError("unexpected unified-build schema")
    if document.get("canonical_repository") != "NaNLMind":
        raise UnificationError("NaNLMind must be the canonical repository")
    topology = document.get("topology")
    if not isinstance(topology, dict):
        raise UnificationError("contract has no topology")
    for key, expected in EXPECTED_TOPOLOGY.items():
        if topology.get(key) != expected:
            raise UnificationError(
                f"contract topology {key}={topology.get(key)!r}, expected {expected}"
            )
    roles = document.get("role_views")
    if not isinstance(roles, dict):
        raise UnificationError("contract has no role_views block")
    for key, expected in EXPECTED_ROLE_VIEWS.items():
        if roles.get(key) != expected:
            raise UnificationError(
                f"role view {key}={roles.get(key)!r}, expected {expected}"
            )
    if roles.get("nannet_port_is_native_head"):
        raise UnificationError("the NANNET port must not be a native head")
    if roles.get("role_overlay_may_reassign_native_heads"):
        raise UnificationError("a role overlay must not reassign native heads")
    authority = document.get("authority")
    if not isinstance(authority, dict):
        raise UnificationError("contract has no authority block")
    if authority.get("hermes_may_commit_canonical_state"):
        raise UnificationError("HERMES must not commit canonical state")
    if authority.get("nanrouter_may_copy_or_fork_native_universe"):
        raise UnificationError("NaNROUTER must not fork the native universe")
    return {
        "document": document,
        "path": contract_path,
        "root": contract_path.parents[1],
        "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }


def _native_binary(root: Path) -> Path | None:
    binary = root / BINARY_RELATIVE
    return binary if binary.is_file() else None


def native_status(root: Path, timeout: float = 3.0) -> dict[str, Any]:
    binary = _native_binary(root)
    if binary is None:
        return {"available": False, "binary": None, "status": "missing"}
    try:
        proc = subprocess.run(
            [str(binary), "unification"],
            text=True,
            capture_output=True,
            timeout=timeout,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(Path.home()),
                "LANG": "C.UTF-8",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "binary": str(binary),
            "status": "error",
            "error": str(exc),
        }
    line = next(
        (item for item in proc.stdout.splitlines() if item.startswith("unified_build ")),
        None,
    )
    if proc.returncode != 0 or line is None:
        return {
            "available": True,
            "binary": str(binary),
            "status": "rejected",
            "exit_code": proc.returncode,
            "stderr": proc.stderr[-2000:],
        }
    values = dict(
        part.split("=", 1) for part in line.split()[1:] if "=" in part
    )
    mismatches = {
        key: values.get(key)
        for key, expected in EXPECTED_NATIVE.items()
        if values.get(key) != expected
    }
    if mismatches:
        return {
            "available": True,
            "binary": str(binary),
            "status": "rejected",
            "mismatches": mismatches,
        }
    return {
        "available": True,
        "binary": str(binary),
        "status": "green",
        "canonical_repository": values.get("canonical_repository"),
        "heads": values.get("heads"),
        "triad_member_heads": values.get("triad_member_heads"),
        "field_head": values.get("field_head"),
        "triads": values.get("triads"),
        "mp_writer_eligible_heads": values.get("mp_writer_eligible_heads"),
        "boolean_curriculum_heads": values.get("boolean_curriculum_heads"),
        "mirror_pairs": values.get("mirror_pairs"),
        "chart_bytes": values.get("chart_bytes"),
    }


def status(path: Path | None = None) -> dict[str, Any]:
    """Return adapter status without making HERMES a second authority."""
    try:
        loaded = load(path)
    except UnificationError as exc:
        return {
            "schema": SCHEMA,
            "status": "unavailable",
            "canonical_repository": "NaNLMind",
            "error": str(exc),
            "hermes_is_canonical": False,
        }
    document = loaded["document"]
    return {
        "schema": SCHEMA,
        "status": "green",
        "canonical_repository": document["canonical_repository"],
        "canonical_state_owner": document.get("canonical_state_owner"),
        "transport_plane": document.get("transport_plane"),
        "adapter_plane": document.get("adapter_plane"),
        "contract_path": str(loaded["path"]),
        "contract_sha256": loaded["sha256"],
        "topology": document["topology"],
        "role_views": document["role_views"],
        "authority": document["authority"],
        "native": native_status(loaded["root"]),
        "hermes_is_canonical": False,
    }


__all__ = [
    "CONTRACT_RELATIVE",
    "EXPECTED_ROLE_VIEWS",
    "EXPECTED_TOPOLOGY",
    "SCHEMA",
    "UnificationError",
    "find_contract",
    "load",
    "native_status",
    "status",
]
