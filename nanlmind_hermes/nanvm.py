"""NaNVM0: a thin NaN16 v0.1 machine boundary.

This module is deliberately small. It is the I/O shim that loads raw 0xNaN
word images (the ``NNNN_PPPPPPPP_FFFF`` format frozen in the NaN16 v0.1
spec) into 8192-word memory and executes them. All call logic lives in the
raw NaN16 program under ``nanlmind_hermes/nan/``; nothing here knows about
phone calls. The op subset mirrors the native nan0s core (``nan16.c``)
exactly: same word layout, same head-class predication, same flag rules,
same wrapping memory. Only the NaNVM0 op range (0x00-0x1B) is implemented;
device ops are headless stubs (no screen, no keyboard, serial captured).
"""
from __future__ import annotations

MASK32 = 0xFFFFFFFF
MEM_WORDS = 8192
MEM_MASK = MEM_WORDS - 1

# Head classes: c = 2*Q + S for the four legal heads (+s, -s, +q, -q).
NAN_HEADS = (0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8)

# NaNVM0 opcodes implemented here (NAN16_SPEC v0.1, 0x00-0x1B).
_IMPLEMENTED = frozenset(
    [*range(0x00, 0x0E), 0x0F,
     0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
     0x1A, 0x1B]
)
HALT = 0xFF


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
    """Build a machine word exactly like ``nan_mkword`` in nan16.c."""
    far = NAN_HEADS[n & 3]
    near = NAN_HEADS[(n >> 2) & 3]
    return ((near << 48) | ((payload & MASK32) << 16)
            | ((far >> 8) | ((far & 0xFF) << 8)))


def parse_nan_image(text: str) -> list[int]:
    """Parse a raw .nan image into 64-bit machine words.

    Source line: ``NNNN_PPPPPPPP_FFFF`` where NNNN is the near head,
    PPPPPPPP the 32-bit payload and FFFF the far head as it sits in the
    stored low word (byte-swapped; e.g. far head 7FF8 is written F87F).
    """
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
        words.append((near << 48) | ((payload & MASK32) << 16) | (far_le & 0xFFFF))
    return words


class NanVM:
    """One NaNVM0 machine: r0..r15, Z/L flags, 8192 wrapping words."""

    def __init__(self, image: list[int]):
        if len(image) > MEM_WORDS:
            raise NanVMError("image exceeds 8192 words")
        self.mem = [0] * MEM_WORDS
        for index, word in enumerate(image):
            self.mem[index] = word & 0xFFFFFFFFFFFFFFFF
        self.reg = [0] * 16
        self.z = 0
        self.l = 0
        self.pc = 0
        self.steps = 0
        self.putc: list[str] = []

    @property
    def output(self) -> str:
        return "".join(self.putc)

    def _set_flags(self, z: int, l: int) -> None:
        self.z = 1 if z else 0
        self.l = 1 if l else 0

    def run(self, max_steps: int = 1_000_000) -> None:
        while self.steps < max_steps:
            word = self.mem[self.pc & MEM_MASK]
            near, far = near_head(word), far_head(word)
            if not head_valid(near) or not head_valid(far):
                raise NanVMError(f"!G gate trap at {self.pc & MEM_MASK}")
            payload = word_payload(word)
            op = payload >> 24
            if op == HALT:
                return
            if op not in _IMPLEMENTED:
                raise NanVMError(f"!O bad op 0x{op:02X} at {self.pc & MEM_MASK}")
            n = (head_class(near) << 2) | head_class(far)
            at = self.pc & MEM_MASK
            self.pc = (at + 1) & MEM_MASK
            self.steps += 1
            if op != 0x00 and not ((n >> (2 * self.z + self.l)) & 1):
                continue
            self._execute(op, (payload >> 20) & 15, (payload >> 16) & 15,
                          payload & 0xFFFF)
        raise NanVMError("step limit exceeded")

    def _execute(self, op: int, d: int, a: int, imm: int) -> None:
        reg = self.reg
        b = imm & 15
        av, bv = reg[a], reg[b]
        if op == 0x00:  # BOOL: truth table is this word's class nibble
            word = self.mem[(self.pc - 1) & MEM_MASK]
            n = (head_class(near_head(word)) << 2) | head_class(far_head(word))
            res = 0
            if n & 1:
                res |= ~(av | bv)
            if n & 2:
                res |= ~av & bv
            if n & 4:
                res |= av & ~bv
            if n & 8:
                res |= av & bv
            reg[d] = res & MASK32
            self._set_flags(reg[d] == 0, 0)
        elif op == 0x01:  # MOVI
            reg[d] = imm
        elif op == 0x02:  # MOVHI: keep low half, set high half
            reg[d] = (reg[d] & 0xFFFF) | (imm << 16)
        elif op == 0x03:  # ADD
            total = av + bv
            reg[d] = total & MASK32
            self._set_flags(reg[d] == 0, total >> 32)
        elif op == 0x04:  # SUB
            res = (av - bv) & MASK32
            reg[d] = res
            self._set_flags(res == 0, av < bv)
        elif op == 0x05:  # SHL
            res = (av << (imm & 31)) & MASK32
            reg[d] = res
            self._set_flags(res == 0, 0)
        elif op == 0x06:  # SHR
            res = av >> (imm & 31)
            reg[d] = res
            self._set_flags(res == 0, 0)
        elif op == 0x07:  # CMP a,b
            self._set_flags(av == bv, av < bv)
        elif op == 0x08:  # LD d,a,imm
            reg[d] = word_payload(self.mem[(av + imm) & MEM_MASK])
        elif op == 0x09:  # ST d,a,imm: +q/+q word, payload r[d]
            self.mem[(av + imm) & MEM_MASK] = make_word(reg[d], 10)
        elif op == 0x0A:  # JMP imm
            self.pc = imm
        elif op == 0x0B:  # CALL imm
            reg[15] = self.pc
            self.pc = imm
        elif op == 0x0C:  # RET
            self.pc = reg[15]
        elif op == 0x0D:  # JMPR a
            self.pc = av
        elif op == 0x0F:  # NOP
            pass
        elif op == 0x10:  # VPOKE: headless, discarded
            pass
        elif op == 0x11:  # VPEEK: headless
            reg[d] = 0
        elif op == 0x12:  # KEY: headless boundary sees end of input
            reg[d] = 0
        elif op == 0x13:  # TICK
            reg[d] = self.steps & MASK32
        elif op == 0x14:  # BSWAPM a,imm
            index = (av + imm) & MEM_MASK
            self.mem[index] = int.from_bytes(
                self.mem[index].to_bytes(8, "little")[::-1], "little")
        elif op == 0x15:  # PUTC
            self.putc.append(chr(reg[d] & 0xFF))
        elif op == 0x16:  # STW d,a,b: word class r[b]&15, payload r[d]
            self.mem[av & MEM_MASK] = make_word(reg[d], reg[b] & 15)
        elif op == 0x17:  # LDN d,a,imm
            q = self.mem[(av + imm) & MEM_MASK]
            nh, fh = near_head(q), far_head(q)
            reg[d] = ((head_class(nh) << 2) | head_class(fh)
                      if head_valid(nh) and head_valid(fh) else MASK32)
        elif op == 0x1A:  # KEYQ: headless
            reg[d] = 0
        elif op == 0x1B:  # NOP (reserved)
            pass
