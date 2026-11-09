"""Python binding to the one canonical NaN0S VM.

There is no second opcode implementation here.  Raw 0xNaN images are executed
by the portable C core from the pinned ``nanlmind_hermes/nan0s`` snapshot
(itself a byte-identical mirror of canonical ``shemshallah/nan0s``).  This
module only parses images, builds/loads the native shared object, and exposes
memory/register access for tests and the resident kernels.

Compiling the binding requires a C compiler; the canonical raw-NaN path is
native.  A missing compiler is reported, never replaced by a Python VM.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import tempfile
import threading

MASK32 = 0xFFFFFFFF
MEM_WORDS = 8192
MEM_MASK = MEM_WORDS - 1
NAN_HEADS = (0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8)


class NanVMError(ValueError):
    pass


def head_valid(head: int) -> bool:
    return (head & 0x7FF0) == 0x7FF0 and (head & 0x0F) in (1, 8)


def head_class(head: int) -> int:
    return ((head >> 2) & 2) | (head >> 15)


def far_head(word: int) -> int:
    raw = word & 0xFFFF
    return ((raw & 0xFF) << 8) | (raw >> 8)


def near_head(word: int) -> int:
    return (word >> 48) & 0xFFFF


def word_payload(word: int) -> int:
    return (word >> 16) & MASK32


def make_word(payload: int, n: int) -> int:
    far = NAN_HEADS[n & 3]
    near = NAN_HEADS[(n >> 2) & 3]
    return ((near << 48) | ((payload & MASK32) << 16)
            | ((far >> 8) | ((far & 0xFF) << 8)))


def parse_nan_image(text: str) -> list[int]:
    words: list[int] = []
    for number, line in enumerate(text.splitlines(), 1):
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        parts = code.split("_")
        if len(parts) != 3:
            raise NanVMError(f"bad .nan line {number}: {line.strip()!r}")
        try:
            near = int(parts[0], 16)
            payload = int(parts[1], 16)
            far_le = int(parts[2], 16)
        except ValueError as exc:
            raise NanVMError(f"bad .nan line {number}: {line.strip()!r}") from exc
        far = ((far_le & 0xFF) << 8) | (far_le >> 8)
        if not head_valid(near) or not head_valid(far):
            raise NanVMError(f"illegal head on .nan line {number}")
        words.append((near << 48) | ((payload & MASK32) << 16)
                     | (far_le & 0xFFFF))
    return words


_LIB_LOCK = threading.Lock()


def _source_root() -> Path:
    return Path(__file__).resolve().with_name("nan0s")


@lru_cache(maxsize=1)
def _build_library() -> Path:
    root = _source_root()
    core = root / "core"
    sources = [core / name for name in (
        "nan16.c", "nan_dev.c", "nan_load.c", "nan_render.c", "nan_state.c")]
    if any(not path.is_file() for path in sources):
        raise NanVMError("pinned NaN0S core sources are missing")
    from shutil import which
    compiler = None
    for name in ("cc", "clang", "gcc"):
        compiler = which(name)
        if compiler:
            break
    if compiler is None:
        raise NanVMError(
            "the one native NaN0S VM needs a C compiler; install cc")
    target_dir = Path(tempfile.gettempdir()) / "nanlmind-nan0s"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "libnan16_native.so"
    if target.is_file():
        return target
    with _LIB_LOCK:
        if target.is_file():
            return target
        tmp = target_dir / ("libnan16.%d.tmp" % os.getpid())
        command = [
            compiler, "-std=gnu99", "-O2", "-fPIC", "-shared",
            "-ffp-contract=off", "-fno-fast-math",
            "-DNAN_WORDS=8192",
            *map(str, sources), "-lm", "-o", str(tmp),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise NanVMError("cannot build native NaN0S VM: " +
                             (result.stderr or result.stdout).strip())
        tmp.replace(target)
        return target


class _NativeMemory:
    __slots__ = ("_vm",)

    def __init__(self, vm: "NanVM"):
        self._vm = vm

    def __len__(self) -> int:
        return MEM_WORDS

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(MEM_WORDS))]
        return int(self._vm._api.nan_word(self._vm._handle, int(index) & MEM_MASK))

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            raise NanVMError("native memory slice assignment is unsupported")
        self._vm._api.nan_set_word(
            self._vm._handle, int(index) & MEM_MASK, int(value) & 0xFFFFFFFFFFFFFFFF)


class _NativeRegisters:
    __slots__ = ("_vm",)

    def __init__(self, vm: "NanVM"):
        self._vm = vm

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index):
        return int(self._vm._api.nan_reg(self._vm._handle, int(index) & 15))

    def __setitem__(self, index, value) -> None:
        self._vm._api.nan_set_reg(
            self._vm._handle, int(index) & 15, int(value) & MASK32)


class NanVM:
    """One native NaN0S machine: r0..r15, Z/L flags, 8192 wrapping words."""

    def __init__(self, image: list[int]):
        if len(image) > MEM_WORDS:
            raise NanVMError("image exceeds 8192 words")
        api = _api()
        self._api = api
        self._handle = api.nan_create()
        if not self._handle:
            raise NanVMError("cannot allocate native NaNVM")
        self.mem = _NativeMemory(self)
        self.reg = _NativeRegisters(self)
        self._output = ""
        for index, word in enumerate(image):
            self.mem[index] = word

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                self._api.nan_destroy(self._handle)
                self._handle = None
        except Exception:
            pass

    @property
    def output(self) -> str:
        # nan_serial_read drains the native ring.  Keep the collected serial
        # text so repeated reads (nan_executable + output) are stable.
        return self._output

    @property
    def steps(self) -> int:
        return int(self._api.nan_steps(self._handle))

    @property
    def pc(self) -> int:
        return int(self._api.nan_pc(self._handle))

    @property
    def z(self) -> int:
        return 1 if (int(self._api.nan_flags(self._handle)) & 2) else 0

    @property
    def l(self) -> int:
        return 1 if (int(self._api.nan_flags(self._handle)) & 1) else 0

    def run(self, max_steps: int = 1_000_000) -> None:
        rc = int(self._api.nan_run(self._handle, int(max_steps)))
        trap = self._api.nan_trap_text(self._handle)
        trap = trap.decode("ascii", "replace") if trap else ""
        buf = ctypes.create_string_buffer(max(1, 1 << 16))
        count = self._api.nan_serial_read(self._handle, buf, len(buf))
        self._output += bytes(buf[:count]).decode("utf-8", "replace")
        if trap:
            raise NanVMError(trap)
        # NAN_BUDGET=0, WAIT_KEY=1, VSYNC=2, HALTED=3.  Only a real halt is
        # success here; a host wait/budget means the guest did not finish.
        if rc != 3:
            raise NanVMError("native NaNVM did not halt (rc=%d)" % rc)


@lru_cache(maxsize=1)
def _api():
    path = _build_library()
    lib = ctypes.CDLL(str(path))
    lib.nan_create.restype = ctypes.c_void_p
    lib.nan_destroy.argtypes = [ctypes.c_void_p]
    lib.nan_run.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.nan_run.restype = ctypes.c_int
    lib.nan_steps.argtypes = [ctypes.c_void_p]
    lib.nan_steps.restype = ctypes.c_uint64
    lib.nan_trap_text.argtypes = [ctypes.c_void_p]
    lib.nan_trap_text.restype = ctypes.c_char_p
    lib.nan_reg.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.nan_reg.restype = ctypes.c_uint32
    lib.nan_pc.argtypes = [ctypes.c_void_p]
    lib.nan_pc.restype = ctypes.c_uint32
    lib.nan_flags.argtypes = [ctypes.c_void_p]
    lib.nan_flags.restype = ctypes.c_int
    lib.nan_word.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.nan_word.restype = ctypes.c_uint64
    lib.nan_set_word.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                 ctypes.c_uint64]
    lib.nan_set_reg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    lib.nan_serial_read.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_size_t]
    lib.nan_serial_read.restype = ctypes.c_size_t
    return lib
