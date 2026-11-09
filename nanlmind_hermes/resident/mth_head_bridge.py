"""Bounded per-head bridge to the MTH/N16I Boolean execution path.

The resident universe has exactly 256 permanent head identities.  This module
lets one head ask the bundled MTH Boolean kernel for one two-input truth-table
decision, and lets a caller aggregate the independent decisions of all 256
heads into a field report.

The bridge deliberately has a very small input surface: a head index, two
Boolean operands and one truth-table operation.  It accepts no native pointer,
path, shell command or arbitrary kernel id.  When a real MTH/N16I runtime is
available the decision is executed by that runtime; when it is not available
the call returns a structured ``execution_mode="unavailable"`` denial and no
Boolean output.  A Python truth-table value is used only to *check* the real
output, never to replace it.

Receipts are local experimental receipts.  They never claim a canonical
commit, NANNET route, lease, proof or merge (``canonical_merge`` is always
``False``).

Runtime protocol
----------------
An injected runtime may expose one of these callables::

    dispatch_boolean(mask, a, b, *, head=..., steps=..., timeout=...)
    boolean_op(head, a, b, mask)
    execute_boolean(mask, a, b, ...)

It may also expose ``available()`` returning either a bool or a mapping with
an ``available`` key.  The default backend uses the bundled
``nanlmind_hermes.nan0s_runtime`` and the ``mth.bool.all`` catalog kernel
(``nanlmind_hermes/mth/nan16/mth_bool.nas``).  If a future
``nanlmind_hermes.mth_runtime`` module exposes a Boolean dispatch callable it
is preferred automatically.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

SCHEMA = "nanlm.mth-head-bridge/1"
SELECTION_SCHEMA = "nanlm.mth-head-selection/1"
RECEIPT_SCHEMA = "nanlm.mth-head-receipt/1"
FIELD_SCHEMA = "nanlm.mth-head-field/1"

HEAD_COUNT = 256
BOOL_MASK_COUNT = 16

# Fixed structural bounds.  Constructor arguments may lower these values but
# can never raise them.
MAX_STEPS = 20_000_000
DEFAULT_STEPS = 5_000_000
MAX_TIMEOUT = 60
DEFAULT_TIMEOUT = 30
MAX_PLAN_BYTES = 64 * 1024
MAX_AGGREGATE_BYTES = 1024 * 1024
MAX_FIELD_SECONDS = 120
MAX_IMAGE_BYTES = 1024 * 1024
MAX_GUEST_WORDS = 8192

OP_NAMES: tuple[str, ...] = (
    "ZERO", "NOR", "NOT_A_AND_B", "NOT_A", "A_AND_NOT_B", "NOT_B",
    "XOR", "NAND", "AND", "XNOR", "B", "A_OR_NOT_B", "A", "NOT_A_OR_B",
    "OR", "ONE",
)

_NAME_TO_MASK = {name: mask for mask, name in enumerate(OP_NAMES)}
_OP_ALIASES = {
    "FALSE": 0,
    "TRUE": 15,
    "NOT_A_AND_B": 2,
    "A_AND_NOT_B": 4,
    "A_OR_NOT_B": 11,
    "NOT_A_OR_B": 13,
    "IMPLIES": 13,
    "A_IMPLIES_B": 13,
    "B_IMPLIES_A": 11,
    "NOT_AND": 7,
    "NOT_OR": 1,
    "NOT_XOR": 9,
    "EQUIV": 9,
    "EQV": 9,
    "XOR": 6,
    "AND": 8,
    "OR": 14,
}

_MISSING = object()

try:  # The resident head identity is the source of truth when available.
    from .cells import HEAD_COUNT as _CELL_HEAD_COUNT
    from .cells import identity_address as _cell_identity_address
    from .cells import reverse8 as _reverse8
except Exception:  # pragma: no cover - only reached in an incomplete install
    _CELL_HEAD_COUNT = HEAD_COUNT

    def _reverse8(value: int) -> int:
        return int(format(value & 0xFF, "08b")[::-1], 2)

    def _cell_identity_address(head: int) -> int:
        return (head << 8) | _reverse8(head)


class MTHHeadBridgeError(ValueError):
    """Raised for invalid constructor bounds or an invalid bridge mode."""


def _bounded_int(value: Any, name: str, maximum: int,
                 minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MTHHeadBridgeError(name + " must be an integer")
    if not minimum <= value <= maximum:
        raise MTHHeadBridgeError(
            "%s must be between %d and %d" % (name, minimum, maximum))
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)[:64]


def _truth(mask: int, a: int, b: int) -> int:
    """The root 4x4 truth table indexed by ``2*a + b``."""
    return (mask >> ((a << 1) | b)) & 1


def _head_identity(head: int) -> int | None:
    """Return the permanent ``(head, head)`` identity, or ``None``.

    The expected identity is computed structurally and compared with the
    resident cell helper.  A mismatch is a closed failure, not a fallback to a
    fabricated address.
    """
    if isinstance(head, bool) or not isinstance(head, int):
        return None
    if not 0 <= head < HEAD_COUNT:
        return None
    expected = (head << 8) | _reverse8(head)
    if _CELL_HEAD_COUNT != HEAD_COUNT:
        return None
    try:
        actual = int(_cell_identity_address(head))
    except Exception:
        return None
    if actual != expected:
        return None
    return actual


def _normalise_bit(value: Any, name: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return int(value), None
    if isinstance(value, int) and value in (0, 1):
        return value, None
    return None, name + " must be Boolean (0/1 or bool)"


def _normalise_op(value: Any) -> tuple[int | None, str | None, str | None]:
    """Return ``(mask, canonical_name, error)`` for a supported operation."""
    if isinstance(value, bool):
        return None, None, "op must be a name or an integer mask in [0, 16)"
    if isinstance(value, int):
        if 0 <= value < BOOL_MASK_COUNT:
            return value, OP_NAMES[value], None
        return None, None, "op mask must be in [0, 16)"
    if isinstance(value, str):
        token = value.strip().upper().replace("-", "_").replace(" ", "_")
        if token in _NAME_TO_MASK:
            mask = _NAME_TO_MASK[token]
            return mask, OP_NAMES[mask], None
        if token in _OP_ALIASES:
            mask = _OP_ALIASES[token]
            return mask, OP_NAMES[mask], None
        return None, None, "unknown Boolean op: " + value[:64]
    return None, None, "op must be a name or an integer mask in [0, 16)"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_optional_mth_runtime() -> Any | None:
    """Load a future ``nanlmind_hermes.mth_runtime`` Boolean dispatcher.

    Only a module that actually exposes a callable Boolean entry point is
    accepted.  Import failures and unrelated modules are ignored so the
    bundled NaN0S backend can be used instead.
    """
    try:
        package = __package__.rsplit(".", 1)[0]
        module = importlib.import_module(package + ".mth_runtime")
    except Exception:
        return None
    for name in ("dispatch_boolean", "boolean_op", "execute_boolean",
                 "run_boolean", "evaluate_boolean"):
        if callable(getattr(module, name, None)):
            return module
    return None


def _runtime_name(runtime: Any) -> str:
    name = getattr(runtime, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()[:96]
    module = getattr(runtime, "__name__", None)
    if isinstance(module, str) and module:
        return module.rsplit(".", 1)[-1][:96]
    return type(runtime).__name__[:96]


class _Nan0sBooleanBackend:
    """Real bundled MTH Boolean kernel driven through NaN0S/N16I.

    The backend synthesises one fixed-shape guest driver that calls the
    bundled ``mth_bool2`` routine with a truth-table mask and two Boolean
    operands.  The driver is the only native artifact the bridge can create;
    no caller-controlled path, pointer or command reaches the runtime.
    """

    name = "nan0s_runtime"

    _DRIVER = """.org 0
JMP @head_bool_main
.include "mth_bool.nas"

head_bool_main:
  MOVI r13,1
  MOVI r1,{mask}
  MOVI r5,{a}
  MOVI r6,{b}
  CALL @mth_bool2
  MOVI r2,@head_bool_out
  ST r4,r2,0
  MOVI r7,1
  MOVI r2,@head_bool_status
  ST r7,r2,0
  HALT
.org 0x100
head_bool_out:
.word 0
head_bool_status:
.word 0
"""

    def __init__(self, ctx: Any = None, *, mode: str = "auto"):
        if mode not in ("auto", "native", "interpreted"):
            raise MTHHeadBridgeError("mode must be auto, native or interpreted")
        self.ctx = ctx
        self.mode = mode
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.RLock()

    def _modules(self):
        from .. import mth_catalog, nan0s_runtime
        return mth_catalog, nan0s_runtime

    def _resolve_mode(self, runtime, core: Mapping[str, Any]) -> str:
        if self.mode in ("native", "interpreted"):
            return self.mode
        root = Path(str(core.get("root") or ""))
        if (root / "src" / "tools" / "n16i.nas").is_file():
            return "interpreted"
        return "native"

    def available(self) -> dict[str, Any]:
        try:
            catalog, runtime = self._modules()
        except Exception as exc:
            return {"available": False, "name": self.name,
                    "reason": "runtime_import_failed",
                    "detail": type(exc).__name__ + ": " + str(exc)[:300]}
        kernel = catalog.lookup("mth.bool.all")
        if kernel is None:
            return {"available": False, "name": self.name,
                    "reason": "catalog_kernel_missing",
                    "detail": "mth.bool.all is absent from the catalog"}
        try:
            core = runtime.discover_core(self.ctx)
        except Exception as exc:
            return {"available": False, "name": self.name,
                    "reason": "core_unavailable",
                    "detail": type(exc).__name__ + ": " + str(exc)[:300]}
        mode = self._resolve_mode(runtime, core)
        root = Path(str(core.get("root") or ""))
        if mode == "interpreted" and not (root / "src" / "tools" / "n16i.nas").is_file():
            return {"available": False, "name": self.name,
                    "reason": "n16i_interpreter_missing",
                    "detail": "N16I interpreter source is not present"}
        try:
            source = catalog.source_path("mth.bool.all")
        except Exception as exc:
            return {"available": False, "name": self.name,
                    "reason": "catalog_source_missing",
                    "detail": type(exc).__name__ + ": " + str(exc)[:300]}
        if not source.is_file():
            return {"available": False, "name": self.name,
                    "reason": "catalog_source_missing",
                    "detail": "mth_bool.nas is not present"}
        compiler = False
        try:
            compiler = bool(runtime.compiler_available(self.ctx))
        except Exception:
            compiler = False
        cli = None
        try:
            cli = runtime.ensure_native_cli(self.ctx, build=False).get("cli")
        except Exception:
            cli = None
        if not compiler and not cli and not core.get("external_cli"):
            return {"available": False, "name": self.name,
                    "reason": "native_cli_unavailable",
                    "detail": ("no C compiler, cached CLI or prebuilt CLI for "
                               "the bundled NaN0S core")}
        return {
            "available": True,
            "name": self.name,
            "execution_mode": mode,
            "mode": mode,
            "kernel_id": "mth.bool.all",
            "catalog_status": kernel.status,
            "source_digest": kernel.source_digest,
            "core_layout": core.get("layout"),
            "compiler": compiler,
            "cli_cached": bool(cli),
        }

    def _ensure_workdir(self) -> Path:
        with self._lock:
            if self._workdir is None:
                self._workdir = tempfile.TemporaryDirectory(
                    prefix="nanlm-mth-head-bridge-")
            return Path(self._workdir.name)

    def _driver_source(self, mask: int, a: int, b: int) -> str:
        return self._DRIVER.format(mask=mask, a=a, b=b)

    @staticmethod
    def _peek_words(stdout: str) -> list[int]:
        return [int(line, 16) for line in (stdout or "").splitlines()
                if len(line.strip()) == 8
                and all(ch in "0123456789abcdefABCDEF" for ch in line.strip())]

    def dispatch_boolean(self, mask: int, a: int, b: int, *,
                         head: int, steps: int, timeout: int) -> dict[str, Any]:
        catalog, runtime = self._modules()
        core = runtime.discover_core(self.ctx)
        mode = self._resolve_mode(runtime, core)
        with self._lock:
            workdir = self._ensure_workdir()
            source = catalog.source_path("mth.bool.all")
            local_source = workdir / "mth_bool.nas"
            if not local_source.is_file():
                local_source.write_bytes(source.read_bytes())
            driver = workdir / ("driver_%d_%d_%d.nas" % (mask, a, b))
            driver.write_text(self._driver_source(mask, a, b), encoding="utf-8")
            assembled = runtime.assemble(driver, ctx=self.ctx)
            image = Path(assembled["image"])
            image_bytes = image.stat().st_size
            if image_bytes > MAX_IMAGE_BYTES:
                raise RuntimeError("assembled Boolean image exceeds the byte bound")
            guest_words = len(runtime._nan_word_lines(image))
            if guest_words > MAX_GUEST_WORDS:
                raise RuntimeError("assembled Boolean image exceeds the guest-word bound")
            if mode == "interpreted":
                try:
                    dispatched = runtime.dispatch_image_n16i(
                        self.ctx, image, max_steps=steps, timeout=timeout,
                        peek_guest="0x100:1,0x101:1")
                except Exception:
                    if self.mode != "auto":
                        raise
                    mode = "native"
                    run = runtime.run_image(
                        self.ctx, image, steps=steps, timeout=timeout,
                        peek="0x100:1,0x101:1")
                    stdout = run.get("stdout") or ""
                    ok = bool(run.get("ok"))
                else:
                    run = dispatched["run"]
                    stdout = dispatched.get("stdout") or ""
                    ok = bool(dispatched.get("ok"))
            else:
                mode = "native"
                run = runtime.run_image(
                    self.ctx, image, steps=steps, timeout=timeout,
                    peek="0x100:1,0x101:1")
                stdout = run.get("stdout") or ""
                ok = bool(run.get("ok"))
        words = self._peek_words(stdout)
        if not ok or len(words) < 2:
            raise RuntimeError("Boolean guest execution did not return a readback")
        output, status = words[-2], words[-1]
        if status != 1:
            raise RuntimeError("Boolean guest driver did not signal status 1")
        if output not in (0, 1):
            raise RuntimeError("Boolean guest output is not a Boolean value")
        return {
            "available": True,
            "output": output,
            "execution_mode": mode,
            "mode": mode,
            "steps": run.get("steps"),
            "kernel_id": "mth.bool.all",
            "source_digest": catalog.require("mth.bool.all").source_digest,
            "image_bytes": image_bytes,
            "guest_words": guest_words,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }

    def close(self) -> None:
        with self._lock:
            workdir, self._workdir = self._workdir, None
        if workdir is not None:
            workdir.cleanup()


class MTHHeadBridge:
    """Expose the MTH/N16I Boolean path to independent per-head decisions."""

    def __init__(self, runtime: Any = None, *, ctx: Any = None,
                 mode: str = "auto",
                 max_steps: int = DEFAULT_STEPS,
                 max_timeout: int = DEFAULT_TIMEOUT,
                 max_plan_bytes: int = MAX_PLAN_BYTES,
                 max_aggregate_bytes: int = MAX_AGGREGATE_BYTES,
                 max_field_seconds: int = MAX_FIELD_SECONDS):
        if mode not in ("auto", "native", "interpreted"):
            raise MTHHeadBridgeError("mode must be auto, native or interpreted")
        self.max_steps = _bounded_int(max_steps, "max_steps", MAX_STEPS)
        self.max_timeout = _bounded_int(max_timeout, "max_timeout", MAX_TIMEOUT)
        self.max_plan_bytes = _bounded_int(
            max_plan_bytes, "max_plan_bytes", MAX_PLAN_BYTES)
        self.max_aggregate_bytes = _bounded_int(
            max_aggregate_bytes, "max_aggregate_bytes", MAX_AGGREGATE_BYTES)
        self.max_field_seconds = _bounded_int(
            max_field_seconds, "max_field_seconds", MAX_FIELD_SECONDS)
        self._ctx = ctx
        self._mode = mode
        if runtime is None:
            optional = _load_optional_mth_runtime()
            if optional is not None:
                self._runtime = optional
                self._runtime_origin = "mth_runtime"
            else:
                self._runtime = _Nan0sBooleanBackend(ctx=ctx, mode=mode)
                self._runtime_origin = "nan0s"
        else:
            self._runtime = runtime
            self._runtime_origin = "injected"
        self._runtime_label = _runtime_name(self._runtime)
        self._generations = [0] * HEAD_COUNT
        self._field_generation = 0
        self._lock = threading.RLock()
        self._closed = False

    # ------------------------------------------------------------- properties
    @property
    def runtime_name(self) -> str:
        return self._runtime_label

    @property
    def runtime_origin(self) -> str:
        return self._runtime_origin

    @property
    def bounds(self) -> dict[str, int]:
        return {
            "heads": HEAD_COUNT,
            "operations": BOOL_MASK_COUNT,
            "max_steps": self.max_steps,
            "max_timeout": self.max_timeout,
            "max_plan_bytes": self.max_plan_bytes,
            "max_aggregate_bytes": self.max_aggregate_bytes,
            "max_field_seconds": self.max_field_seconds,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_guest_words": MAX_GUEST_WORDS,
        }

    @staticmethod
    def supported_ops() -> list[dict[str, Any]]:
        return [{"op": name, "mask": mask, "truth_table": [
            _truth(mask, a, b) for a in (0, 1) for b in (0, 1)
        ]} for mask, name in enumerate(OP_NAMES)]

    def close(self) -> None:
        self._closed = True
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "MTHHeadBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------------------------------------------------------------- selection
    def select_head(self, head: int) -> dict[str, Any]:
        """Select one permanent head identity within the fixed 0..255 bounds."""
        if self._closed:
            return self._selection_denial(head, "bridge_closed")
        if isinstance(head, bool) or not isinstance(head, int) \
                or not 0 <= head < HEAD_COUNT:
            return self._selection_denial(head, "invalid_head")
        identity = _head_identity(head)
        if identity is None:
            return self._selection_denial(head, "identity_invalid")
        with self._lock:
            generation = self._generations[head]
        return {
            "schema": SELECTION_SCHEMA,
            "status": "ok",
            "ok": True,
            "denied": False,
            "selected": True,
            "head": head,
            "head_id": head,
            "identity": identity,
            "identity_hex": "0x%04x" % identity,
            "identity_valid": True,
            "generation": generation,
            "runtime": self._probe_runtime(),
            "canonical_merge": False,
            "canonical_commit": False,
            "bounds": self.bounds,
        }

    def _selection_denial(self, head: Any, reason: str) -> dict[str, Any]:
        return {
            "schema": SELECTION_SCHEMA,
            "status": "denied",
            "ok": False,
            "denied": True,
            "selected": False,
            "head": _json_safe(head),
            "head_id": None,
            "identity": None,
            "identity_hex": None,
            "identity_valid": False,
            "generation": 0,
            "reason": reason,
            "runtime": {"available": False, "reason": reason,
                        "name": self._runtime_label},
            "canonical_merge": False,
            "canonical_commit": False,
            "bounds": self.bounds,
        }

    # ---------------------------------------------------------- single-head ops
    def boolean_op(self, head: int, a: Any, b: Any, op: Any, *,
                   steps: int | None = None,
                   timeout: int | None = None) -> dict[str, Any]:
        """Evaluate one Boolean operation for one head."""
        return self._single(head, a, b, op, steps=steps, timeout=timeout)

    def evaluate(self, head: int, a: Any, b: Any, op: Any, *,
                 steps: int | None = None,
                 timeout: int | None = None) -> dict[str, Any]:
        """Alias for :meth:`boolean_op`; one head, one field decision."""
        return self._single(head, a, b, op, steps=steps, timeout=timeout)

    def dispatch_boolean(self, head: int, a: Any, b: Any, op: Any, *,
                         steps: int | None = None,
                         timeout: int | None = None) -> dict[str, Any]:
        """Dispatch one bounded Boolean operation to the configured runtime."""
        return self._single(head, a, b, op, steps=steps, timeout=timeout)

    def _single(self, head: int, a: Any, b: Any, op: Any, *,
                steps: int | None, timeout: int | None) -> dict[str, Any]:
        if self._closed:
            return self._receipt(head, a, b, op, None, reason="bridge_closed")
        receipt = self._receipt(head, a, b, op, None)
        if receipt["status"] == "denied":
            return receipt
        head = receipt["head"]
        mask = receipt["op_mask"]
        a_value = receipt["inputs"]["a"]
        b_value = receipt["inputs"]["b"]
        step_budget = self._resolve_steps(steps)
        if step_budget is None:
            receipt = self._receipt(head, a_value, b_value, op, None,
                                    reason="steps_out_of_range")
            receipt["outputs"] = []
            receipt["requested_bounds"] = {"steps": _json_safe(steps)}
            return receipt
        timeout_budget = self._resolve_timeout(timeout)
        if timeout_budget is None:
            receipt = self._receipt(head, a_value, b_value, op, None,
                                    reason="timeout_out_of_range")
            receipt["requested_bounds"] = {"timeout": _json_safe(timeout)}
            return receipt
        probe = self._probe_runtime()
        with self._lock:
            generation = self._generations[head]
            receipt = self._receipt(head, a_value, b_value, op, mask,
                                    generation=generation, steps=step_budget,
                                    timeout=timeout_budget, probe=probe)
            if not probe["available"]:
                receipt.update({
                    "status": "denied", "ok": False, "denied": True,
                    "reason": "runtime_unavailable",
                    "execution_mode": "unavailable",
                    "available": False,
                    "runtime_unavailable_reason": probe.get("reason"),
                    "runtime_detail": probe.get("detail"),
                })
                receipt["outputs"] = []
                return receipt
            result, error = self._invoke_runtime(
                mask, a_value, b_value, head=head,
                steps=step_budget, timeout=timeout_budget)
            self._apply_result(receipt, result, error, mask, a_value, b_value)
            if receipt["status"] == "ok":
                self._generations[head] = generation + 1
                receipt["generation"] = generation + 1
            return receipt

    # --------------------------------------------------------- field aggregate
    def field_aggregate(self, plan: Any, *, steps: int | None = None,
                        timeout: int | None = None) -> dict[str, Any]:
        """Evaluate a complete 256-head plan and aggregate the decisions.

        ``plan`` may be a mapping ``{head: (op, a, b)}`` or a sequence of 256
        entries.  Each sequence entry is either ``(op, a, b)`` (head inferred
        from position) or ``(head, op, a, b)`` or a mapping with ``head``,
        ``op``, ``a`` and ``b`` keys.  Every head 0..255 must appear exactly
        once; a missing, duplicate or out-of-range head fails closed before
        any runtime call.
        """
        if self._closed:
            return self._field_denial("bridge_closed")
        step_budget = self._resolve_steps(steps)
        if step_budget is None:
            return self._field_denial("steps_out_of_range")
        timeout_budget = self._resolve_timeout(timeout)
        if timeout_budget is None:
            return self._field_denial("timeout_out_of_range")
        entries, plan_bytes, plan_digest, error = self._normalise_plan(plan)
        if error is not None:
            return self._field_denial(
                "plan_invalid", detail=error,
                plan_bytes=plan_bytes, plan_digest=plan_digest)
        probe = self._probe_runtime()
        with self._lock:
            receipts: list[dict[str, Any]] = []
            deadline = time.monotonic() + self.max_field_seconds
            for entry in entries:
                head = entry["head"]
                generation = self._generations[head]
                if not probe["available"]:
                    receipt = self._receipt(
                        head, entry["a"], entry["b"], entry["op"], entry["op_mask"],
                        generation=generation, steps=step_budget,
                        timeout=timeout_budget, probe=probe)
                    receipt.update({
                        "status": "denied", "ok": False, "denied": True,
                        "reason": "runtime_unavailable",
                        "execution_mode": "unavailable",
                        "available": False,
                        "runtime_unavailable_reason": probe.get("reason"),
                        "runtime_detail": probe.get("detail"),
                    })
                    receipt["outputs"] = []
                    receipts.append(receipt)
                    continue
                if time.monotonic() > deadline:
                    receipt = self._receipt(
                        head, entry["a"], entry["b"], entry["op"], entry["op_mask"],
                        generation=generation, steps=step_budget,
                        timeout=timeout_budget, probe=probe)
                    receipt.update({
                        "status": "denied", "ok": False, "denied": True,
                        "reason": "field_deadline_exceeded",
                        "execution_mode": "unavailable",
                        "available": False,
                    })
                    receipt["outputs"] = []
                    receipts.append(receipt)
                    continue
                receipt = self._receipt(
                    head, entry["a"], entry["b"], entry["op"], entry["op_mask"],
                    generation=generation, steps=step_budget,
                    timeout=timeout_budget, probe=probe)
                result, error = self._invoke_runtime(
                    entry["op_mask"], entry["a"], entry["b"], head=head,
                    steps=step_budget, timeout=timeout_budget)
                self._apply_result(receipt, result, error, entry["op_mask"],
                                   entry["a"], entry["b"])
                if receipt["status"] == "ok":
                    receipt["generation"] = generation + 1
                receipts.append(receipt)
            all_ok = all(receipt["status"] == "ok" for receipt in receipts)
        aggregate = self._aggregate(receipts, entries, plan_bytes, plan_digest,
                                    probe)
        aggregate_bytes = len(_canonical_json(aggregate))
        if aggregate_bytes > self.max_aggregate_bytes:
            return self._field_denial(
                "aggregate_bytes_exceeded",
                detail="aggregate exceeded the configured byte bound",
                plan_bytes=plan_bytes, plan_digest=plan_digest,
                extra={"aggregate_bytes": aggregate_bytes,
                       "head_count": len(receipts)})
        if all_ok:
            with self._lock:
                for receipt in receipts:
                    self._generations[receipt["head"]] = receipt["generation"]
                self._field_generation += 1
        aggregate["aggregate_bytes"] = aggregate_bytes
        return aggregate

    def evaluate_field(self, plan: Any, *, steps: int | None = None,
                       timeout: int | None = None) -> dict[str, Any]:
        """Alias for :meth:`field_aggregate`."""
        return self.field_aggregate(plan, steps=steps, timeout=timeout)

    def field(self, plan: Any, *, steps: int | None = None,
              timeout: int | None = None) -> dict[str, Any]:
        """Alias for :meth:`field_aggregate`."""
        return self.field_aggregate(plan, steps=steps, timeout=timeout)

    def uniform_field(self, op: Any, a: Any, b: Any, *,
                      steps: int | None = None,
                      timeout: int | None = None) -> dict[str, Any]:
        """Evaluate the same operation for all 256 heads, keeping them independent."""
        return self.field_aggregate(
            {head: (op, a, b) for head in range(HEAD_COUNT)},
            steps=steps, timeout=timeout)

    # --------------------------------------------------------------- receipts
    def _receipt(self, head: Any, a: Any, b: Any, op: Any,
                 mask: int | None, *,
                 generation: int = 0,
                 steps: int | None = None,
                 timeout: int | None = None,
                 probe: Mapping[str, Any] | None = None,
                 reason: str | None = None) -> dict[str, Any]:
        valid_head = isinstance(head, int) and not isinstance(head, bool) \
            and 0 <= head < HEAD_COUNT
        identity = _head_identity(head) if valid_head else None
        op_mask, op_name, op_error = _normalise_op(op)
        a_value, a_error = _normalise_bit(a, "a")
        b_value, b_error = _normalise_bit(b, "b")
        if op_error is not None:
            reason = reason or "invalid_op"
        elif a_error is not None or b_error is not None:
            reason = reason or "invalid_operand"
        elif not valid_head:
            reason = reason or "invalid_head"
        elif identity is None:
            reason = reason or "identity_invalid"
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "denied" if reason else "ok",
            "ok": False if reason else True,
            "denied": bool(reason),
            "reason": reason,
            "head": head if valid_head else _json_safe(head),
            "head_id": head if valid_head else None,
            "identity": identity,
            "identity_hex": ("0x%04x" % identity) if identity is not None else None,
            "op": op_name if op_name is not None else _json_safe(op),
            "op_mask": op_mask if op_mask is not None else mask,
            "inputs": {
                "a": a_value if a_value is not None else _json_safe(a),
                "b": b_value if b_value is not None else _json_safe(b),
            },
            "outputs": [],
            "output": None,
            "execution_mode": "unavailable",
            "available": False,
            "generation": generation,
            "runtime": dict(probe or self._probe_runtime()),
            "runtime_origin": self._runtime_origin,
            "canonical_merge": False,
            "canonical_commit": False,
            "native_pointer": False,
            "command_execution": False,
            "bounds": {
                **self.bounds,
                "steps": steps if steps is not None else self.max_steps,
                "timeout": timeout if timeout is not None else self.max_timeout,
            },
            "claim": "local per-head MTH Boolean decision",
            "does_not_prove": [
                "mathematical correctness beyond the selected truth table",
                "a canonical training, NANNET route, lease or merge commit",
                "a native pointer ABI or command execution interface",
            ],
        }
        if reason:
            receipt["unavailable_reason"] = reason
        return receipt

    def _apply_result(self, receipt: dict[str, Any], result: Mapping[str, Any] | None,
                      error: Mapping[str, Any] | None,
                      mask: int, a: int, b: int) -> None:
        if error is not None:
            receipt.update({
                "status": "denied",
                "ok": False,
                "denied": True,
                "reason": "runtime_unavailable",
                "execution_mode": "unavailable",
                "available": False,
                "runtime_error_type": error.get("type"),
                "runtime_detail": error.get("detail"),
            })
            receipt["outputs"] = []
            return
        output = int(result["output"])
        mode = str(result["execution_mode"])
        expected = _truth(mask, a, b)
        receipt.update({
            "execution_mode": mode,
            "available": True,
            "runtime_generation": result.get("generation"),
            "runtime_steps": result.get("steps"),
            "runtime_detail": result.get("detail"),
            "expected_outputs": [expected],
        })
        if output != expected:
            receipt.update({
                "status": "denied",
                "ok": False,
                "denied": True,
                "reason": "runtime_output_mismatch",
                "output": output,
                "outputs": [output],
                "matches_expected": False,
            })
            return
        receipt.update({
            "status": "ok",
            "ok": True,
            "denied": False,
            "reason": None,
            "output": output,
            "outputs": [output],
            "matches_expected": True,
        })

    def _aggregate(self, receipts: list[dict[str, Any]],
                   entries: list[dict[str, Any]],
                   plan_bytes: int, plan_digest: str,
                   probe: Mapping[str, Any]) -> dict[str, Any]:
        ok_receipts = [receipt for receipt in receipts
                       if receipt["status"] == "ok"]
        modes = {receipt["execution_mode"] for receipt in ok_receipts}
        if not ok_receipts:
            mode = "unavailable"
        elif len(modes) == 1:
            mode = next(iter(modes))
        else:
            mode = "mixed"
        outputs = [receipt["output"] if receipt["status"] == "ok" else None
                   for receipt in receipts]
        output_digest = _digest([
            {"head": receipt["head"],
             "op_mask": receipt["op_mask"],
             "a": receipt["inputs"]["a"],
             "b": receipt["inputs"]["b"],
             "output": receipt["output"]}
            for receipt in receipts if receipt["status"] == "ok"
        ])
        identities = [
            {"head": entry["head"], "identity": _head_identity(entry["head"])}
            for entry in entries
        ]
        failed = [receipt["head"] for receipt in receipts
                  if receipt["status"] != "ok"]
        all_ok = not failed
        with self._lock:
            generation_before = self._field_generation
        field_generation = generation_before + (1 if all_ok else 0)
        if all_ok:
            failure_reason = None
        elif all(receipt["execution_mode"] == "unavailable"
                 for receipt in receipts):
            failure_reason = "runtime_unavailable"
        elif any(receipt.get("reason") == "field_deadline_exceeded"
                 for receipt in receipts):
            failure_reason = "field_deadline_exceeded"
        else:
            failure_reason = "head_execution_failed"
        return {
            "schema": FIELD_SCHEMA,
            "status": "ok" if all_ok else "denied",
            "ok": all_ok,
            "denied": not all_ok,
            "reason": failure_reason,
            "head_count": HEAD_COUNT,
            "selected_heads": len(receipts),
            "identities_valid": (
                len(identities) == HEAD_COUNT
                and all(item["identity"] is not None for item in identities)
                and len({item["identity"] for item in identities}) == HEAD_COUNT
            ),
            "identity_count": len(identities),
            "identity_digest": _digest(identities),
            "independent": True,
            "execution_mode": mode,
            "generation": field_generation,
            "generation_before": generation_before,
            "outputs": outputs,
            "output_digest": output_digest,
            "true_heads": sum(1 for value in outputs if value == 1),
            "false_heads": sum(1 for value in outputs if value == 0),
            "unavailable_heads": sum(
                1 for receipt in receipts
                if receipt["execution_mode"] == "unavailable"),
            "failed_heads": failed,
            "plan": {
                "bytes": plan_bytes,
                "digest": plan_digest,
                "entries": len(entries),
            },
            "runtime": dict(probe),
            "runtime_origin": self._runtime_origin,
            "canonical_merge": False,
            "canonical_commit": False,
            "native_pointer": False,
            "command_execution": False,
            "receipts": receipts,
            "bounds": self.bounds,
            "claim": "local independent per-head MTH Boolean field aggregate",
            "does_not_prove": [
                "a canonical field decision or canonical head state",
                "a NANNET route, lease, proof or merge commit",
                "a native pointer ABI or command execution interface",
            ],
        }

    def _field_denial(self, reason: str, *, detail: Any = None,
                      plan_bytes: int = 0, plan_digest: str | None = None,
                      extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": FIELD_SCHEMA,
            "status": "denied",
            "ok": False,
            "denied": True,
            "reason": reason,
            "detail": detail,
            "head_count": 0,
            "selected_heads": 0,
            "identities_valid": False,
            "identity_count": 0,
            "identity_digest": None,
            "independent": True,
            "execution_mode": "unavailable",
            "generation": self._field_generation,
            "outputs": [],
            "output_digest": None,
            "true_heads": 0,
            "false_heads": 0,
            "unavailable_heads": 0,
            "failed_heads": [],
            "plan": {"bytes": plan_bytes, "digest": plan_digest, "entries": 0},
            "runtime": {"available": False, "name": self._runtime_label,
                        "reason": reason},
            "runtime_origin": self._runtime_origin,
            "canonical_merge": False,
            "canonical_commit": False,
            "native_pointer": False,
            "command_execution": False,
            "receipts": [],
            "bounds": self.bounds,
            "claim": "local per-head MTH Boolean field aggregate",
            "does_not_prove": [
                "a canonical field decision or canonical head state",
                "a NANNET route, lease, proof or merge commit",
            ],
        }
        if extra:
            result.update(dict(extra))
        return result

    # ---------------------------------------------------------- plan handling
    def _normalise_plan(self, plan: Any):
        entries: list[tuple[Any, Any, Any, Any]] = []
        if isinstance(plan, Mapping):
            for head, value in plan.items():
                parsed, error = self._plan_value(value, head)
                if error is not None:
                    return None, 0, None, error
                entries.append((head, parsed[0], parsed[1], parsed[2]))
        elif isinstance(plan, Sequence) and not isinstance(plan, (str, bytes, bytearray)):
            positional = len(plan) == HEAD_COUNT
            for index, value in enumerate(plan):
                head = index if positional else None
                if isinstance(value, Mapping) and "head" in value:
                    head = value.get("head")
                parsed, error = self._plan_value(value, head, index=index)
                if error is not None:
                    return None, 0, None, error
                if head is None:
                    return None, 0, None, (
                        "entry %d must name a head or be part of a 256-entry "
                        "positional plan" % index)
                entries.append((head, parsed[0], parsed[1], parsed[2]))
        else:
            return None, 0, None, "plan must be a mapping or a sequence"

        heads: list[int] = []
        for head, _op, _a, _b in entries:
            if isinstance(head, bool) or not isinstance(head, int) \
                    or not 0 <= head < HEAD_COUNT:
                return None, 0, None, "plan head must be an integer in [0, 256)"
            heads.append(head)
        if len(entries) != HEAD_COUNT:
            return None, 0, None, (
                "plan must contain exactly %d heads, got %d"
                % (HEAD_COUNT, len(entries)))
        if len(set(heads)) != HEAD_COUNT or set(heads) != set(range(HEAD_COUNT)):
            return None, 0, None, "plan must contain every head 0..255 exactly once"

        payload = []
        seen_identities = set()
        for head, op, a, b in sorted(entries, key=lambda item: item[0]):
            mask, name, op_error = _normalise_op(op)
            if op_error is not None:
                return None, 0, None, "head %d: %s" % (head, op_error)
            a_value, a_error = _normalise_bit(a, "a")
            if a_error is not None:
                return None, 0, None, "head %d: %s" % (head, a_error)
            b_value, b_error = _normalise_bit(b, "b")
            if b_error is not None:
                return None, 0, None, "head %d: %s" % (head, b_error)
            identity = _head_identity(head)
            if identity is None:
                return None, 0, None, "head %d has no valid permanent identity" % head
            if identity in seen_identities:
                return None, 0, None, "duplicate permanent head identity"
            seen_identities.add(identity)
            payload.append({
                "head": head,
                "op": name,
                "op_mask": mask,
                "a": a_value,
                "b": b_value,
            })
        raw = _canonical_json(payload)
        if len(raw) > self.max_plan_bytes:
            return None, len(raw), None, (
                "plan is %d bytes; the bound is %d bytes"
                % (len(raw), self.max_plan_bytes))
        return payload, len(raw), hashlib.sha256(raw).hexdigest(), None

    @staticmethod
    def _plan_value(value: Any, head: Any, *, index: int | None = None):
        if isinstance(value, Mapping):
            allowed = {"head", "op", "a", "b", "op_mask"}
            extras = set(value) - allowed
            if extras:
                return None, ("entry contains unsupported keys: "
                              + ", ".join(sorted(str(item) for item in extras)))
            op = value.get("op", value.get("op_mask"))
            if op is None:
                return None, "entry is missing op"
            if "a" not in value or "b" not in value:
                return None, "entry is missing a or b"
            return (op, value.get("a"), value.get("b")), None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) == 3:
                return (value[0], value[1], value[2]), None
            if len(value) == 4:
                if head is not None and value[0] != head:
                    return None, "entry head does not match its position or key"
                return (value[1], value[2], value[3]), None
        return None, "entry must be (op, a, b), (head, op, a, b) or a mapping"

    # ------------------------------------------------------------ runtime glue
    def _resolve_steps(self, steps: int | None) -> int | None:
        if steps is None:
            return self.max_steps
        try:
            return _bounded_int(steps, "steps", self.max_steps)
        except MTHHeadBridgeError:
            return None

    def _resolve_timeout(self, timeout: int | None) -> int | None:
        if timeout is None:
            return self.max_timeout
        try:
            return _bounded_int(timeout, "timeout", self.max_timeout)
        except MTHHeadBridgeError:
            return None

    def _probe_runtime(self) -> dict[str, Any]:
        probe = getattr(self._runtime, "available", None)
        if not callable(probe):
            return {"available": True, "name": self._runtime_label,
                    "reason": None, "detail": None}
        try:
            value = probe()
        except Exception as exc:
            return {"available": False, "name": self._runtime_label,
                    "reason": "runtime_probe_failed",
                    "detail": type(exc).__name__ + ": " + str(exc)[:300]}
        if isinstance(value, bool):
            return {"available": value, "name": self._runtime_label,
                    "reason": None if value else "runtime_unavailable",
                    "detail": None}
        if isinstance(value, Mapping):
            result = dict(value)
            result.setdefault("name", self._runtime_label)
            result["available"] = bool(result.get("available"))
            if not result["available"]:
                result.setdefault("reason", "runtime_unavailable")
                result.setdefault("detail", None)
            return result
        return {"available": False, "name": self._runtime_label,
                "reason": "runtime_probe_invalid",
                "detail": "available() did not return a bool or mapping"}

    def _runtime_method(self):
        for name in ("dispatch_boolean", "boolean_op", "execute_boolean",
                     "run_boolean", "evaluate_boolean"):
            method = getattr(self._runtime, name, None)
            if callable(method):
                return name, method
        if callable(self._runtime):
            return "call", self._runtime
        return None, None

    def _invoke_runtime(self, mask: int, a: int, b: int, *,
                        head: int, steps: int, timeout: int):
        name, method = self._runtime_method()
        if method is None:
            return None, {
                "type": "MissingRuntimeInterface",
                "detail": ("runtime has no dispatch_boolean/boolean_op/"
                           "execute_boolean callable"),
            }
        try:
            result = self._call_method(name, method, mask, a, b,
                                       head=head, steps=steps, timeout=timeout)
        except Exception as exc:
            return None, {"type": type(exc).__name__,
                          "detail": str(exc)[:500]}
        normalised, error = self._normalise_runtime_result(result)
        return normalised, error

    def _call_method(self, name: str, method: Any, mask: int, a: int, b: int,
                     *, head: int, steps: int, timeout: int) -> Any:
        parameters: Mapping[str, inspect.Parameter] | None
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = None
        if name == "boolean_op" or (
                parameters is not None
                and next(iter(parameters), None) in ("head", "head_id")):
            return method(head, a, b, mask)
        if parameters is None:
            return method(mask, a, b)
        kwargs: dict[str, Any] = {}
        names = set(parameters)
        accepts_any = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                          for parameter in parameters.values())
        for candidate, value in (("head", head), ("steps", steps),
                                 ("timeout", timeout), ("op_mask", mask),
                                 ("ctx", self._ctx)):
            if accepts_any or candidate in names:
                kwargs[candidate] = value
        return method(mask, a, b, **kwargs)

    @staticmethod
    def _normalise_runtime_result(result: Any):
        if not isinstance(result, Mapping):
            return None, {"type": "RuntimeProtocolError",
                          "detail": "runtime result is not a mapping"}
        if result.get("available") is False or result.get("ok") is False:
            return None, {
                "type": "RuntimeUnavailable",
                "detail": str(result.get("detail")
                               or result.get("reason")
                               or result.get("error")
                               or "runtime reported unavailable")[:500],
            }
        mode = result.get("execution_mode")
        if mode is None:
            mode = result.get("mode")
        if mode == "unavailable":
            return None, {"type": "RuntimeUnavailable",
                          "detail": "runtime reported execution_mode=unavailable"}
        if result.get("trap"):
            return None, {"type": "RuntimeTrap",
                          "detail": str(result.get("trap"))[:500]}
        if mode not in ("native", "interpreted"):
            if result.get("n16i_used") is True:
                mode = "interpreted"
            elif result.get("interpreted") is True:
                mode = "interpreted"
            elif result.get("native") is True or result.get("n16i_used") is False:
                mode = "native"
            else:
                return None, {"type": "RuntimeProtocolError",
                              "detail": "runtime did not identify native or "
                                        "interpreted execution"}
        output = result.get("output", _MISSING)
        if output is _MISSING:
            output = result.get("result", _MISSING)
        if output is _MISSING:
            output = result.get("value", _MISSING)
        if output is _MISSING:
            outputs = result.get("outputs")
            if isinstance(outputs, Sequence) and not isinstance(
                    outputs, (str, bytes, bytearray)) and len(outputs) == 1:
                output = outputs[0]
        if output is _MISSING:
            return None, {"type": "RuntimeProtocolError",
                          "detail": "runtime result has no output value"}
        if isinstance(output, bool):
            output = int(output)
        elif isinstance(output, float) and output.is_integer():
            output = int(output)
        if not isinstance(output, int) or output not in (0, 1):
            return None, {"type": "RuntimeProtocolError",
                          "detail": "runtime output is not Boolean"}
        return {
            "output": output,
            "execution_mode": mode,
            "generation": result.get("generation"),
            "steps": result.get("steps"),
            "detail": result.get("detail"),
        }, None


__all__ = (
    "SCHEMA", "SELECTION_SCHEMA", "RECEIPT_SCHEMA", "FIELD_SCHEMA",
    "HEAD_COUNT", "BOOL_MASK_COUNT", "OP_NAMES",
    "MAX_STEPS", "DEFAULT_STEPS", "MAX_TIMEOUT", "DEFAULT_TIMEOUT",
    "MAX_PLAN_BYTES", "MAX_AGGREGATE_BYTES", "MAX_FIELD_SECONDS",
    "MAX_IMAGE_BYTES", "MAX_GUEST_WORDS", "MTHHeadBridgeError",
    "MTHHeadBridge",
)
