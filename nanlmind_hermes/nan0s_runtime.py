"""Universal host-side NaN0S runtime for the bundled MTH kernels.

The plugin ships architecture-neutral NaN16 kernels plus a small source
snapshot of the portable NaN0S C core. At first native MTH use this module:

1. detects the host architecture and C compiler;
2. uses an explicitly configured NaN0S checkout/prebuilt CLI when available;
3. otherwise compiles the bundled portable core into the plugin state cache;
4. assembles a kernel/fixture from NaN16 source with the bundled POSIX-awk
   emitter;
5. runs it with a bounded step budget and returns serial output, trap and
   guest-word readback.

The portable C core is the universal substrate. ``n16i`` remains the guest
NaN16 interpreter-in-NaN16 fallback; this module reports its availability and
never pretends it executed a kernel unless a real driver ran.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

from . import mth_catalog

SCHEMA = "nanlm.nan0s-runtime/1"
BUILD_SCHEMA = "nanlm.nan0s-build/1"
HALT_RC = 3
DEFAULT_MAX_STEPS = 5_000_000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 900
MAX_STEPS_CAP = 500_000_000

_ARCH = {
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "i386": "x86",
    "i486": "x86",
    "i586": "x86",
    "i686": "x86",
    "riscv64": "riscv64",
    "ppc64le": "ppc64le",
    "s390x": "s390x",
}
_BUNDLED = Path(__file__).resolve().with_name("nan0s")
_BUILD_LOCK = threading.RLock()
_AWK_CACHE: dict[str, str | None] = {}


class NativeRuntimeError(RuntimeError):
    pass


def _setting(ctx, key: str, default: Any) -> Any:
    try:
        return ctx.get_config(key, default=default)
    except Exception:
        return default


def host_architecture() -> dict[str, Any]:
    machine = (platform.machine() or "").lower()
    return {
        "machine": machine,
        "target": _ARCH.get(machine, "portable-c"),
        "system": platform.system(),
        "release": platform.release(),
        "python_bits": 64 if sys.maxsize > 2 ** 32 else 32,
        "byteorder": sys.byteorder,
        "portable_c": True,
    }


def _layout(root: Path) -> dict[str, Any] | None:
    if not root.is_dir():
        return None
    bundled_core = root / "core" / "nan16.c"
    external_core = root / "aarch64" / "android" / "core" / "nan16.c"
    if bundled_core.is_file() and (root / "tools" / "nan_cli.c").is_file():
        return {
            "root": str(root), "layout": "bundled",
            "core_dir": str(root / "core"),
            "cli_source": str(root / "tools" / "nan_cli.c"),
            "nmk": str(root / "tools" / "nmk.awk"),
            "external_cli": None,
        }
    if external_core.is_file():
        base = root / "aarch64" / "android"
        return {
            "root": str(root), "layout": "external",
            "core_dir": str(base / "core"),
            "cli_source": str(base / "tools" / "nan_cli.c"),
            "nmk": str(root / "tools" / "nmk.awk"),
            "external_cli": str(base / "nan_cli") if (base / "nan_cli").is_file() else None,
        }
    return None


def discover_core(ctx=None) -> dict[str, Any]:
    """Find a configured, sibling or bundled NaN0S source layout."""
    # NaN0S is the canonical dependency. The bundled snapshot is only a
    # last-resort wheel fallback, never preferred over a real checkout.
    configured = str(_setting(ctx, "nan0s_root", "") or "").strip()
    prefer_bundled = bool(_setting(ctx, "nan0s_prefer_bundled", False))
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    if not (configured or prefer_bundled):
        candidates.extend([
            Path(__file__).resolve().parents[1].parent / "nan0s",
            Path.home() / "nan0s",
            Path.home() / "NaN0S",
            Path("/data/data/com.termux/files/home/nan0s"),
        ])
    for root in candidates:
        found = _layout(root)
        if found is not None:
            found["discovered_from"] = "config"
            return found
    builtin = _layout(_BUNDLED)
    if builtin is not None:
        builtin["discovered_from"] = "bundled"
        return builtin
    raise NativeRuntimeError(
        "no NaN0S core found; set nan0s_root to a NaN0S checkout or install a C compiler")


def _cache_root(ctx) -> Path:
    state = Path(str(_setting(ctx, "state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
    return state / "nan0s"


def _compiler(ctx) -> str | None:
    configured = str(_setting(ctx, "nan0s_cc", "") or "").strip()
    if configured:
        return configured
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    return None


def compiler_available(ctx=None) -> bool:
    """Whether a host C compiler can build the bundled portable core."""
    return _compiler(ctx) is not None


def _awk(nmk: str | Path | None = None) -> str | None:
    """Return a NaN16-emitter-compatible awk.

    BusyBox awk mis-emits at least the HALT word (0x7FFFFFFF instead of
    0xFF000000), so plain ``awk`` is accepted only after a real assembly probe.
    gawk and mawk are preferred. Never silently produce a wrong guest image.
    """
    candidates = []
    for name in ("gawk", "mawk", "awk"):
        found = shutil.which(name)
        if found and found not in candidates:
            candidates.append(found)
    if nmk is None:
        return candidates[0] if candidates else None
    cache_key = str(nmk)
    if cache_key in _AWK_CACHE:
        return _AWK_CACHE[cache_key]
    for candidate in candidates:
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".nas", delete=False) as handle:
                handle.write("HALT\n")
                source = handle.name
            completed = subprocess.run(
                [candidate, "-f", str(nmk), "-v", "CMD=emit", source],
                capture_output=True, text=True, timeout=20)
            os.unlink(source)
            text = (completed.stdout or "").upper()
            if (completed.returncode == 0 and "FF000000" in text
                    and "7FFFFFFF" not in text):
                _AWK_CACHE[cache_key] = candidate
                return candidate
        except (OSError, subprocess.SubprocessError):
            try:
                os.unlink(source)
            except OSError:
                pass
    _AWK_CACHE[cache_key] = None
    return None


def _source_files(core: dict[str, Any]) -> list[Path]:
    core_dir = Path(core["core_dir"])
    files = [core_dir / "nan16.c", core_dir / "nan_dev.c", core_dir / "nan_load.c",
             core_dir / "nan_render.c", core_dir / "nan_state.c",
             Path(core["cli_source"])]
    return [path for path in files if path.is_file()]


def _digest_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def ensure_native_cli(ctx=None, *, build: bool = True) -> dict[str, Any]:
    """Return a usable native ``nan_cli``, compiling the bundled core if needed."""
    core = discover_core(ctx)
    arch = host_architecture()
    cache = _cache_root(ctx) / arch["target"]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "nan_cli"
    sources = _source_files(core)
    digest = _digest_files(sources)

    external = core.get("external_cli")
    if external and os.access(str(external), os.X_OK):
        return {"schema": BUILD_SCHEMA, "cli": str(external), "arch": arch,
                "core": core, "built": False, "source": "prebuilt",
                "source_digest": digest, "compiler": None}

    receipt_path = cache / "build.json"
    if target.is_file() and os.access(str(target), os.X_OK):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = {}
        if receipt.get("source_digest") == digest:
            return {"schema": BUILD_SCHEMA, "cli": str(target), "arch": arch,
                    "core": core, "built": False, "source": "cache",
                    "source_digest": digest,
                    "compiler": receipt.get("compiler")}
    if not build:
        return {"schema": BUILD_SCHEMA, "cli": None, "arch": arch,
                "core": core, "built": False, "source": "not_built",
                "source_digest": digest, "compiler": None}
    compiler = _compiler(ctx)
    if not compiler:
        raise NativeRuntimeError(
            "no C compiler found; install cc/gcc/clang or set nan0s_root to a prebuilt NaN0S CLI")
    command = [
        compiler, "-std=gnu99", "-O2", "-ffp-contract=off", "-fno-fast-math",
        "core/nan16.c", "core/nan_dev.c", "core/nan_load.c",
        "core/nan_render.c", "core/nan_state.c", "tools/nan_cli.c",
        "-o", str(target), "-lm",
    ]
    with _BUILD_LOCK:
        completed = subprocess.run(
            command, cwd=core["root"], capture_output=True, text=True,
            timeout=300)
        if completed.returncode != 0 or not target.is_file():
            tail = (completed.stderr or completed.stdout or "")[-2000:]
            raise NativeRuntimeError("NaN0S CLI build failed: " + tail)
    receipt = {
        "schema": BUILD_SCHEMA,
        "arch": arch,
        "compiler": compiler,
        "command": command,
        "source_digest": digest,
        "sources": [str(path) for path in sources],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, receipt_path)
    return {"schema": BUILD_SCHEMA, "cli": str(target), "arch": arch,
            "core": core, "built": True, "source": "compiled",
            "source_digest": digest, "compiler": compiler}


def assemble(source: str | Path, output: str | Path | None = None,
             ctx=None, labels: str | Path | None = None) -> dict[str, Any]:
    """Assemble one NaN16 ``.nas`` source with the bundled/external nmk.awk."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise NativeRuntimeError("NaN16 source does not exist")
    core = discover_core(ctx)
    nmk = Path(core["nmk"])
    if not nmk.is_file():
        raise NativeRuntimeError("nmk.awk is missing from the NaN0S core")
    awk = _awk(nmk)
    if not awk:
        raise NativeRuntimeError(
            "no compatible POSIX awk found for NaN16 assembly; install gawk or "
            "mawk (BusyBox awk mis-emits HALT words)")
    if output is None:
        cache = _cache_root(ctx) / "assembled"
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:32]
        output_path = cache / (digest + ".nan")
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [awk, "-f", str(nmk), "-v", "CMD=emit"]
    if labels is not None:
        labels_path = Path(labels)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["-v", "LABELS=" + str(labels_path)])
    command.append(str(source_path))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise NativeRuntimeError("NaN16 assembly failed: "
                                 + (completed.stderr or completed.stdout)[-2000:])
    image = completed.stdout
    if not image.strip():
        raise NativeRuntimeError("NaN16 assembler produced an empty image")
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(image, encoding="utf-8")
    os.replace(temporary, output_path)
    return {
        "source": str(source_path), "image": str(output_path),
        "assembler": awk, "nmk": str(nmk),
        "labels": str(labels) if labels is not None else None,
        "bytes": output_path.stat().st_size,
    }


def _parse_run(stderr: str) -> dict[str, Any]:
    match = re.search(
        r":\s*(\d+)\s+steps,\s*rc=(\d+),\s*pc=(\d+),\s*trap='([^']*)'",
        stderr or "")
    if match is None:
        return {"steps": None, "rc": None, "pc": None, "trap": None}
    return {"steps": int(match.group(1)), "rc": int(match.group(2)),
            "pc": int(match.group(3)), "trap": match.group(4)}


def run_image(ctx, image: str | Path, *, steps: int = DEFAULT_MAX_STEPS,
              timeout: int = DEFAULT_TIMEOUT, peek: str | None = None,
              build: bool = True) -> dict[str, Any]:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS_CAP:
        raise NativeRuntimeError("steps must be a positive bounded integer")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT:
        raise NativeRuntimeError("timeout must be between 1 and " + str(MAX_TIMEOUT))
    runtime = ensure_native_cli(ctx, build=build)
    cli = runtime.get("cli")
    if not cli:
        raise NativeRuntimeError("native NaN0S CLI is not built")
    command = [str(cli), "--img", str(image), "--steps", str(steps), "--serial"]
    if peek:
        command.extend(["--peek", str(peek)])
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=timeout)
    parsed = _parse_run(completed.stderr)
    return {
        "schema": SCHEMA,
        "cli": str(cli),
        "architecture": runtime["arch"],
        "core": runtime["core"],
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "steps": parsed["steps"],
        "rc": parsed["rc"],
        "pc": parsed["pc"],
        "trap": parsed["trap"],
        "halted": parsed["rc"] == HALT_RC and not parsed["trap"],
        "ok": completed.returncode == 0 and parsed["rc"] == HALT_RC
              and not parsed["trap"],
        "execution_mode": "native_nan0s",
        "n16i_used": False,
    }


def verify_kernel(ctx, kernel_id: str, *, max_steps: int | None = None,
                  timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Assemble and run one bundled kernel fixture natively."""
    kernel = mth_catalog.require(kernel_id)
    fixture = mth_catalog.fixture_path(kernel_id)
    image = assemble(fixture, ctx=ctx)
    budget = max_steps or max(kernel.steps * 4, 1_000_000)
    budget = min(max(int(budget), 1), MAX_STEPS_CAP)
    run = run_image(ctx, image["image"], steps=budget, timeout=timeout)
    marker_found = kernel.marker in (run.get("stdout") or "")
    verified = bool(run["ok"] and marker_found)
    return {
        "schema": SCHEMA,
        "kernel_id": kernel_id,
        "verified": verified,
        "marker": kernel.marker,
        "marker_found": marker_found,
        "budget_steps": budget,
        "catalog_status": kernel.status,
        "ci_verified": kernel.ci_verified,
        "source_digest": kernel.source_digest,
        "fixture_digest": kernel.fixture_digest,
        "image": image,
        "run": run,
        "execution_mode": "native_nan0s",
        "claim": ("local native fixture execution" if verified
                  else "fixture did not produce its declared marker"),
        "does_not_prove": [
            "mathematical correctness beyond this fixture",
            "a general operand ABI for kernels outside the four dispatchers",
            "a canonical training, NANNET route, lease or merge commit",
        ],
    }


_DISPATCH_FILES = {
    "mth.vector.add256.f32": ("mth_vector_add256_f32.nas",),
    "mth.vector.sub256.f32": ("mth_vector_sub256_f32.nas",),
    "mth.vector.dot256.f32": ("mth_dot256_f32.nas",),
    "mth.vector.cosine256.f32": ("mth_cosine256_f32.nas", "mth_dot256_f32.nas",
                                 "mth_norm256_f32.nas"),
}


def _word_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _bits_word(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _driver_source(kernel_id: str, x: list[float], y: list[float]) -> str:
    files = _DISPATCH_FILES[kernel_id]
    symbol = {
        "mth.vector.add256.f32": "mth_vector_add256_f32",
        "mth.vector.sub256.f32": "mth_vector_sub256_f32",
        "mth.vector.dot256.f32": "mth_dot256_f32",
        "mth.vector.cosine256.f32": "mth_cosine256_f32",
    }[kernel_id]
    lines = [".org 0", "JMP @main"]
    lines.extend('.include "' + name + '"' for name in files)
    lines.extend(["main:", "MOVI r13,1", "MOVI r1,@buf_x"])
    for index, value in enumerate(x):
        bits = _word_bits(value)
        lines.append("MOVI r7,%d" % (bits & 0xFFFF))
        lines.append("MOVHI r7,%d" % ((bits >> 16) & 0xFFFF))
        lines.append("ST r7,r1,%d" % index)
    lines.extend(["MOVI r1,@buf_y"])
    for index, value in enumerate(y):
        bits = _word_bits(value)
        lines.append("MOVI r7,%d" % (bits & 0xFFFF))
        lines.append("MOVHI r7,%d" % ((bits >> 16) & 0xFFFF))
        lines.append("ST r7,r1,%d" % index)
    lines.extend([
        "MOVI r1,@buf_x",
        "MOVI r2,@buf_y",
        "MOVI r3,@buf_out",
    ])
    if kernel_id == "mth.vector.cosine256.f32":
        lines.append("MOVI r4,@buf_scratch")
    lines.extend([
        "CALL @" + symbol,
        "MOVI r2,@buf_status",
        "ST r1,r2,0",
        "HALT",
        ".org 0x4000",
        "buf_x:",
    ])
    lines.extend([".word 0"] * 256)
    lines.append("buf_y:")
    lines.extend([".word 0"] * 256)
    lines.append("buf_out:")
    lines.extend([".word 0"] * 256)
    lines.append("buf_scratch:")
    lines.extend([".word 0"] * 4)
    lines.append("buf_status:")
    lines.append(".word 0")
    return "\n".join(lines) + "\n"


def _prepare_vector_driver(ctx, kernel_id: str, x: list[float],
                           y: list[float]) -> dict[str, Any]:
    """Validate operands and assemble the shared 256-element driver image."""
    if kernel_id not in mth_catalog.NATIVE_DISPATCH:
        raise NativeRuntimeError(
            "kernel has no admitted dynamic operand ABI; use verify_kernel for its native fixture")
    if (not isinstance(x, list) or not isinstance(y, list)
            or len(x) != 256 or len(y) != 256):
        raise NativeRuntimeError("x and y must each contain exactly 256 float32 values")
    values = []
    for item in list(x) + list(y):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise NativeRuntimeError("x and y must contain numeric values")
        value = float(item)
        if not (-3.4028235e38 <= value <= 3.4028235e38):
            raise NativeRuntimeError("value is outside the float32 finite range")
        values.append(value)
    x = values[:256]
    y = values[256:]
    symbols = _DISPATCH_FILES[kernel_id]
    driver_dir = _cache_root(ctx) / "drivers" / hashlib.sha256(
        (kernel_id + repr(x) + repr(y)).encode("utf-8")).hexdigest()[:24]
    driver_dir.mkdir(parents=True, exist_ok=True)
    source_dir = mth_catalog.bundle_root() / "nan16"
    for name in symbols:
        source = source_dir / name
        if not source.is_file():
            raise NativeRuntimeError("vector kernel dependency is missing: " + name)
        shutil.copyfile(source, driver_dir / name)
    driver = driver_dir / "driver.nas"
    driver.write_text(_driver_source(kernel_id, x, y), encoding="utf-8")
    image = assemble(driver, ctx=ctx)
    # .org 0x4000: x=0x4000, y=0x4100, out=0x4200, scratch=0x4300..0x4303,
    # status=0x4304.
    peek_guest = ("0x4200:1,0x4304:1" if kernel_id in {
        "mth.vector.dot256.f32", "mth.vector.cosine256.f32"}
        else "0x4200:256,0x4304:1")
    expected = 2 if kernel_id in {"mth.vector.dot256.f32",
                                  "mth.vector.cosine256.f32"} else 257
    return {
        "image": image,
        "driver": str(driver),
        "peek_guest": peek_guest,
        "expected_words": expected,
    }


def _decode_vector_readback(stdout: str, expected: int) -> tuple[int, Any]:
    """Decode peek words into (status, scalar-or-vector result)."""
    words = [int(line, 16) for line in (stdout or "").splitlines()
             if re.fullmatch(r"[0-9a-fA-F]{8}", line.strip())]
    if len(words) < expected:
        raise NativeRuntimeError("vector result readback was incomplete")
    output_words = words[-expected:-1]
    status = words[-1]
    result = (_bits_word(output_words[0]) if expected == 2
              else [_bits_word(word) for word in output_words])
    return status, result


def execute_vector_kernel_native(ctx, kernel_id: str, x: list[float],
                                 y: list[float], *,
                                 timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Diagnostic helper: run an admitted vector kernel directly on native NaN0S.

    The unified default path is :func:`execute_vector_kernel`, which runs the
    same driver image through N16I. This helper keeps direct native execution
    available for differential checks and fallback diagnosis only.
    """
    prepared = _prepare_vector_driver(ctx, kernel_id, x, y)
    run = run_image(ctx, prepared["image"]["image"], steps=DEFAULT_MAX_STEPS,
                    timeout=timeout, peek=prepared["peek_guest"])
    status, result = _decode_vector_readback(
        run.get("stdout") or "", prepared["expected_words"])
    return {
        "schema": SCHEMA,
        "kernel_id": kernel_id,
        "status": status,
        "ok": bool(run["ok"] and status == 1),
        "result": result,
        "result_precision": "float32",
        "input_elements": 256,
        "execution_mode": "native_nan0s",
        "n16i_used": False,
        "architecture": run["architecture"],
        "cli": run["cli"],
        "steps": run["steps"],
        "trap": run["trap"],
        "image": prepared["image"],
        "boundary": {
            "canonical_commit": False,
            "nannet_route_executed": False,
            "n16i_used": False,
            "host_python_math": False,
        },
    }


def execute_vector_kernel(ctx, kernel_id: str, x: list[float], y: list[float],
                          *, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Run an admitted vector kernel through the N16I guest interpreter.

    The same 256-element driver image used by direct native execution is
    assembled, then dispatched through ``dispatch_image_n16i`` so the guest
    program is interpreted by the bundled N16I implementation running on the
    native core. Direct native execution remains available as
    :func:`execute_vector_kernel_native` for diagnostics.
    """
    prepared = _prepare_vector_driver(ctx, kernel_id, x, y)
    dispatched = dispatch_image_n16i(
        ctx, prepared["image"]["image"],
        peek_guest=prepared["peek_guest"], timeout=timeout)
    run = dispatched["run"]
    status, result = _decode_vector_readback(
        dispatched.get("stdout") or "", prepared["expected_words"])
    return {
        "schema": SCHEMA,
        "kernel_id": kernel_id,
        "status": status,
        "ok": bool(dispatched["ok"] and status == 1),
        "result": result,
        "result_precision": "float32",
        "input_elements": 256,
        "execution_mode": "n16i_guest_interpreter",
        "n16i_used": True,
        "native": False,
        "guest_base": dispatched["guest_base"],
        "architecture": run["architecture"],
        "cli": run["cli"],
        "steps": run["steps"],
        "trap": run["trap"],
        "image": prepared["image"],
        "dispatch": dispatched,
        "boundary": {
            "canonical_commit": False,
            "nannet_route_executed": False,
            "n16i_used": True,
            "host_python_math": False,
        },
    }


def _nan_word_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split(";", 1)[0].strip()
        if code:
            lines.append(code)
    return lines


def _emit_load32(register: int, value: int) -> list[str]:
    value &= 0xFFFFFFFF
    return ["MOVI r%d,%d" % (register, value & 0xFFFF),
            "MOVHI r%d,%d" % (register, (value >> 16) & 0xFFFF)]


def _translate_peek_guest(spec: str, guest_base: int, guest_mask: int) -> str:
    """Translate ``0x4200:256`` guest peeks into host ``--peek`` addresses."""
    if not isinstance(spec, str) or not spec.strip():
        raise NativeRuntimeError("peek_guest must be a non-empty guest-address spec")
    translated = []
    for raw in spec.split(","):
        token = raw.strip()
        match = re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)(?::(\d+))?", token)
        if match is None:
            raise NativeRuntimeError("invalid peek_guest token: " + token)
        address = (int(match.group(1), 16) if match.group(1).lower().startswith("0x")
                   else int(match.group(1), 10))
        length = int(match.group(2), 10) if match.group(2) else 1
        if address > 0xFFFFFFFF:
            raise NativeRuntimeError("peek_guest address is outside the guest word space")
        if not 1 <= length <= 65536:
            raise NativeRuntimeError("peek_guest length is outside the bounded window")
        translated.append("0x%X:%d" % (guest_base + (address & guest_mask), length))
    return ",".join(translated)


def dispatch_image_n16i(ctx, image: str | Path, *, entry: int = 0,
                        guest_base: int | None = None,
                        max_steps: int = 100_000_000,
                        timeout: int = DEFAULT_TIMEOUT,
                        peek_guest: str | None = None) -> dict[str, Any]:
    """Run an assembled NaN16 image under the bundled N16I interpreter.

    This is the generic N16 dispatch path: the guest words are placed at a
    host base, N16I is entered with guest pc/GB/mask, and its serial output is
    captured by the native core. It works for arbitrary guest images that fit
    the host memory, not only the four MTH vector dispatchers.

    ``peek_guest`` accepts guest-address readback specs such as
    ``0x4200:256,0x4304:1``. Each address is translated to the host window with
    ``guest_base + (address & guest_mask)`` before being passed to
    ``run_image(--peek ...)``.
    """
    if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
        raise NativeRuntimeError("entry must be a non-negative guest word address")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS_CAP:
        raise NativeRuntimeError("max_steps must be a positive bounded integer")
    if peek_guest is not None and not isinstance(peek_guest, str):
        raise NativeRuntimeError("peek_guest must be a guest-address spec string")
    image_path = Path(image).resolve()
    guest = _nan_word_lines(image_path)
    if not guest:
        raise NativeRuntimeError("guest image has no NaN16 words")
    if len(guest) > 3_500_000:
        raise NativeRuntimeError("guest image exceeds the bounded N16I dispatch window")
    if guest_base is None:
        guest_base = 0x10000
    if isinstance(guest_base, bool) or not isinstance(guest_base, int) or guest_base < 0x1000:
        raise NativeRuntimeError("guest_base must be at least 0x1000")
    core = discover_core(ctx)
    n16i = Path(core["root"]) / "src" / "tools" / "n16i.nas"
    if not n16i.is_file():
        raise NativeRuntimeError("N16I interpreter source is missing")
    size = 1
    while size < len(guest):
        size <<= 1
    mask = size - 1
    cache = _cache_root(ctx) / "n16i-dispatch" / hashlib.sha256(
        (str(image_path) + ":" + str(entry) + ":" + str(guest_base)).encode("utf-8")
    ).hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(n16i, cache / "n16i.nas")
    driver = cache / "dispatch.nas"
    lines = [
        ".org 0",
        "JMP @drv",
        '.include "n16i.nas"',
        "drv:",
        "MOVI r13,1",
    ]
    lines.extend(_emit_load32(1, entry))
    lines.extend(_emit_load32(2, guest_base))
    lines.extend(_emit_load32(3, mask))
    lines.extend(["CALL @n16i_run", "HALT", ".org %d" % guest_base])
    lines.extend([".word 0"] * len(guest))
    driver.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assembled = assemble(driver, output=cache / "dispatch.nan", ctx=ctx)
    driver_lines = Path(assembled["image"]).read_text(encoding="utf-8").splitlines()
    word_indexes = [index for index, line in enumerate(driver_lines)
                    if line.split(";", 1)[0].strip()]
    if len(word_indexes) < guest_base + len(guest):
        raise NativeRuntimeError("N16I driver image is shorter than the guest window")
    for offset, word in enumerate(guest):
        line_index = word_indexes[guest_base + offset]
        driver_lines[line_index] = word
    patched = cache / "dispatch.patched.nan"
    patched.write_text("\n".join(driver_lines) + "\n", encoding="utf-8")
    host_peek = (None if peek_guest is None
                 else _translate_peek_guest(peek_guest, guest_base, mask))
    run = run_image(ctx, patched, steps=max_steps, timeout=timeout,
                    peek=host_peek)
    return {
        "schema": SCHEMA,
        "execution_mode": "n16i_guest_interpreter",
        "n16i_used": True,
        "guest_words": len(guest),
        "guest_base": guest_base,
        "guest_mask": mask,
        "entry": entry,
        "peek_guest": peek_guest,
        "peek": host_peek,
        "image": assembled,
        "run": run,
        "ok": bool(run["ok"]),
        "trap": run["trap"],
        "stdout": run["stdout"],
        "boundary": {
            "canonical_commit": False,
            "nannet_route_executed": False,
        },
    }


def verify_kernel_n16i(ctx, kernel_id: str, *, use_fixture: bool = True,
                       entry: int = 0, max_steps: int = 100_000_000,
                       timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Assemble a bundled kernel/fixture and dispatch it under N16I."""
    kernel = mth_catalog.require(kernel_id)
    source = mth_catalog.fixture_path(kernel_id) if use_fixture else mth_catalog.source_path(kernel_id)
    image = assemble(source, ctx=ctx)
    dispatched = dispatch_image_n16i(
        ctx, image["image"], entry=entry, max_steps=max_steps, timeout=timeout)
    marker_found = kernel.marker in (dispatched.get("stdout") or "")
    return {
        "schema": SCHEMA,
        "kernel_id": kernel_id,
        "use_fixture": use_fixture,
        "verified": bool(dispatched["ok"] and (marker_found or not use_fixture)),
        "marker": kernel.marker,
        "marker_found": marker_found,
        "execution_mode": "n16i_guest_interpreter",
        "n16i_used": True,
        "source_digest": kernel.source_digest,
        "fixture_digest": kernel.fixture_digest,
        "dispatch": dispatched,
        "claim": ("arbitrary guest image dispatched through N16I"
                  if dispatched["ok"] else "N16I dispatch failed"),
        "does_not_prove": [
            "a canonical training, NANNET route, lease or merge commit",
            "a general typed operand ABI for every MTH kernel",
        ],
    }


def _label(labels_path: Path, name: str) -> int:
    try:
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == name:
                return int(fields[1], 0)
    except (OSError, ValueError) as exc:
        raise NativeRuntimeError("cannot read N16I label table") from exc
    raise NativeRuntimeError("N16I label is missing: " + name)


def _patch_word(image: Path, word_index: int, payload: int) -> Path:
    lines = image.read_text(encoding="utf-8").splitlines()
    count = 0
    for index, line in enumerate(lines):
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        if count == word_index:
            lines[index] = "7FF8_%08X_F87F   ; patched word %d" % (
                payload & 0xFFFFFFFF, word_index)
            patched = image.with_name(image.stem + ".patched.nan")
            patched.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return patched
        count += 1
    raise NativeRuntimeError("N16I patch address is outside the image")


def n16i_self_test(ctx=None, *, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Run the nan0s differential gate: same image native and under N16I.

    The bundled sources are the real ``src/tests/n16i_test.nas`` driver, the
    NaN16 ``n16i.nas`` interpreter and its disassembler. The test image is
    assembled once, then its ``mode`` word is patched from 0 (native) to 1
    (run the guest under N16I). Serial bytes must be identical.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT:
        raise NativeRuntimeError("timeout must be between 1 and " + str(MAX_TIMEOUT))
    core = discover_core(ctx)
    root = Path(core["root"])
    test = root / "src" / "tests" / "n16i_test.nas"
    for required in (test, root / "src" / "tools" / "n16i.nas",
                     root / "src" / "tools" / "disasm.nas"):
        if not required.is_file():
            raise NativeRuntimeError("N16I self-test source is missing: " + str(required))
    cache = _cache_root(ctx) / "n16i"
    cache.mkdir(parents=True, exist_ok=True)
    labels = cache / "n16i.labels"
    image = assemble(test, output=cache / "n16i_test.nan", ctx=ctx,
                     labels=labels)
    mode = _label(labels, "mode")
    patched = _patch_word(Path(image["image"]), mode, 1)
    native = run_image(ctx, image["image"], steps=20_000_000, timeout=timeout)
    interpreted = run_image(ctx, patched, steps=20_000_000, timeout=timeout)
    native_bytes = (native.get("stdout") or "").encode("utf-8")
    interpreted_bytes = (interpreted.get("stdout") or "").encode("utf-8")
    identical = (native_bytes == interpreted_bytes
                 and native["halted"] and interpreted["halted"])
    return {
        "schema": SCHEMA,
        "test": "nan0s.n16i.differential",
        "identical": identical,
        "native_steps": native.get("steps"),
        "interpreted_steps": interpreted.get("steps"),
        "output_bytes": len(native_bytes),
        "output_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "mode_word": mode,
        "native": native,
        "interpreted": interpreted,
        "execution_mode": "n16i_guest_interpreter",
        "n16i_used": True,
        "claim": ("native and N16I interpreted serial output are identical"
                  if identical else "native and N16I output differ"),
        "does_not_prove": [
            "that every MTH kernel has an N16I driver",
            "a NANNET route, lease, merge or canonical commit",
        ],
    }


def runtime_status(ctx=None, *, build: bool = False) -> dict[str, Any]:
    arch = host_architecture()
    try:
        core = discover_core(ctx)
    except Exception as exc:
        return {"schema": SCHEMA, "architecture": arch, "native_available": False,
                "error": type(exc).__name__ + ": " + str(exc)[:500],
                "catalog": mth_catalog.health()}
    try:
        cli = ensure_native_cli(ctx, build=build)
        error = None
    except Exception as exc:
        cli = {"cli": None, "source": "unavailable", "compiler": _compiler(ctx),
               "source_digest": None}
        error = type(exc).__name__ + ": " + str(exc)[:500]
    n16i = Path(core["root"]) / "src" / "tools" / "n16i.nas"
    self_test_sources = (Path(core["root"]) / "src" / "tests" / "n16i_test.nas",
                         n16i,
                         Path(core["root"]) / "src" / "tools" / "disasm.nas")
    nmk_path = Path(core["nmk"])
    compatible_awk = _awk(nmk_path)
    return {
        "schema": SCHEMA,
        "architecture": arch,
        "core": core,
        "cli": cli,
        "compiler": _compiler(ctx),
        "awk": compatible_awk,
        "awk_compatible": compatible_awk is not None,
        "awk_candidates": [shutil.which(name) for name in ("gawk", "mawk", "awk")
                           if shutil.which(name)],
        "native_available": bool(cli.get("cli")),
        "native_build_required": bool(not cli.get("cli")),
        "error": error,
        "n16i": {
            "source_present": n16i.is_file(),
            "path": str(n16i) if n16i.is_file() else None,
            "self_test_available": all(path.is_file() for path in self_test_sources),
            "self_test_tool": "nanlm_mth_n16i_selftest",
            "guest_fallback_wired": all(path.is_file() for path in self_test_sources),
            "arbitrary_dispatch_tool": "nanlm_mth_dispatch",
            "note": ("n16i is the NaN16-in-NaN16 interpreter. The bundled "
                     "portable core is preferred for MTH kernels; N16I has a "
                     "differential self-test and a generic guest-image "
                     "dispatch path. Typed operand ABIs for every kernel are "
                     "still separate work."),
        },
        "execution_modes": {
            "native_nan0s": bool(cli.get("cli")),
            "portable_c_source": True,
            "n16i_self_test": all(path.is_file() for path in self_test_sources),
            "n16i_arbitrary_kernel_fallback": all(path.is_file() for path in self_test_sources),
            "python_nanvm0": "limited to NaNVM0 phone-call kernel",
        },
        "catalog": mth_catalog.health(),
    }


def reset_build_cache(ctx=None) -> None:
    """Testing/setup helper: remove the compiled CLI cache only."""
    cache = _cache_root(ctx)
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
