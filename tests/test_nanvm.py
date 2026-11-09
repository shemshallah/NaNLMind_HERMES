"""NaNVM0 boundary tests: the raw NaN16 machine behind the phone-call kernel."""
import math
import struct
import unittest

from nanlmind_hermes.nanvm import (MEM_WORDS, NanVM, NanVMError, make_word,
                                   parse_nan_image, word_payload)

PROGRAM = """
FFF8_01D00001_F8FF   ; 000 MOVI r13,1        (program sets the r13=1 convention)
FFF8_0A000005_F8FF   ; 001 JMP 5
FFF8_09150000_F8FF   ; 002 ST r1,r5,0        emit: mem[r5]=r1, r5+=1
FFF8_0355000D_F8FF   ; 003 ADD r5,r5,r13
FFF8_0C000000_F8FF   ; 004 RET
FFF8_01500800_F8FF   ; 005 MOVI r5,2048      main
FFF8_01100007_F8FF   ; 006 MOVI r1,7
FFF8_0B000002_F8FF   ; 007 CALL 2
FFF8_01100009_F8FF   ; 008 MOVI r1,9
FFF8_0B000002_F8FF   ; 009 CALL 2
FFF8_01100005_F8FF   ; 010 MOVI r1,5
FFF8_01200005_F8FF   ; 011 MOVI r2,5
FFF8_07010002_F8FF   ; 012 CMP r1,r2
7FF1_0A000021_F8FF   ; 013 JMP[NE] 33        equal: must NOT jump
FFF8_01300001_F8FF   ; 014 MOVI r3,1
FFF8_01100003_F8FF   ; 015 MOVI r1,3
FFF8_01200009_F8FF   ; 016 MOVI r2,9
FFF8_07010002_F8FF   ; 017 CMP r1,r2
7FF1_0A000021_F1FF   ; 018 JMP[GT] 33        3<9: must NOT jump
FFF8_01400001_F8FF   ; 019 MOVI r4,1
FFF8_01500800_F8FF   ; 020 MOVI r5,2048
FFF8_08650000_F8FF   ; 021 LD r6,r5,0        r6=7
FFF8_08750001_F8FF   ; 022 LD r7,r5,1        r7=9
FFF8_03860007_F8FF   ; 023 ADD r8,r6,r7      r8=16
FFF8_09850002_F8FF   ; 024 ST r8,r5,2
FFF8_00980000_F8FF   ; 025 BOOL.12 r9,r8,r0  copy r8
FFF8_0499000D_F8FF   ; 026 SUB r9,r9,r13     15
FFF8_07090000_F8FF   ; 027 CMP r9,r0
FFF1_0A000021_F17F   ; 028 JMP[EQ] 33        nonzero: must NOT jump
FFF8_01A00063_F8FF   ; 029 MOVI r10,99
FFF8_09A50003_F8FF   ; 030 ST r10,r5,3
FFF8_0A000023_F8FF   ; 031 JMP 35
FFF8_0F000000_F8FF   ; 032 NOP
FFF8_01A00042_F8FF   ; 033 MOVI r10,66       error marker
FFF8_09A50003_F8FF   ; 034 ST r10,r5,3
FFF8_FF000000_F8FF   ; 035 HALT
"""


class ParseTests(unittest.TestCase):
    def test_far_head_is_byteswapped_from_source(self):
        words = parse_nan_image("7FF8_00000001_F87F\n")
        self.assertEqual(words[0],
                         (0x7FF8 << 48) | (1 << 16) | 0xF87F)

    def test_make_word_roundtrip(self):
        word = make_word(0xDEADBEEF, 3)
        self.assertEqual(word_payload(word), 0xDEADBEEF)
        self.assertEqual(word >> 48, 0x7FF1)        # near class 0 (+s)
        self.assertEqual(word & 0xFFFF, 0xF8FF)     # far class 3 (-q) stored

    def test_illegal_head_rejected(self):
        with self.assertRaises(NanVMError):
            parse_nan_image("0000_00000000_0000\n")

    def test_comments_and_blank_lines_skipped(self):
        words = parse_nan_image("; only a comment\n\n7FF8_00000000_F87F\n")
        self.assertEqual(len(words), 1)


class MachineTests(unittest.TestCase):
    def test_program_runs_and_computes(self):
        vm = NanVM(parse_nan_image(PROGRAM))
        vm.run()
        self.assertEqual(word_payload(vm.mem[2048]), 7)
        self.assertEqual(word_payload(vm.mem[2049]), 9)
        self.assertEqual(word_payload(vm.mem[2050]), 16)
        self.assertEqual(word_payload(vm.mem[2051]), 99)
        self.assertEqual(vm.reg[3], 1)   # EQ check passed
        self.assertEqual(vm.reg[4], 1)   # GT check passed

    def test_movi_does_not_touch_flags(self):
        vm = NanVM(parse_nan_image(
            "FFF8_01100001_F8FF\n"   # 0 MOVI r1,1
            "FFF8_01200001_F8FF\n"   # 1 MOVI r2,1
            "FFF8_07010002_F8FF\n"   # 2 CMP r1,r2 -> Z=1
            "FFF8_01300005_F8FF\n"   # 3 MOVI r3,5 (must not clear Z)
            "FFF1_0A000007_F17F\n"   # 4 JMP[EQ] 7 (must jump)
            "FFF8_01200063_F8FF\n"   # 5 MOVI r2,99 (error path)
            "FFF8_0A000009_F8FF\n"   # 6 JMP 9 (error path)
            "FFF8_01400001_F8FF\n"   # 7 MOVI r4,1 (EQ path)
            "FFF8_FF000000_F8FF\n"))  # 8 HALT
        vm.run()
        self.assertEqual(vm.reg[3], 5)
        self.assertEqual(vm.reg[4], 1)
        self.assertEqual(vm.reg[2], 1)

    def test_memory_wraps_within_8192_words(self):
        vm = NanVM(parse_nan_image(
            "FFF8_01501FFF_F8FF\n"   # MOVI r5,8191
            "FFF8_0110002A_F8FF\n"   # MOVI r1,42
            "FFF8_09150001_F8FF\n"   # ST r1,r5,1 -> wraps to word 0
            "FFF8_FF000000_F8FF\n"))
        vm.run()
        self.assertEqual(word_payload(vm.mem[0]), 42)

    def test_gate_trap_on_illegal_head(self):
        vm = NanVM([0])
        with self.assertRaises(NanVMError) as caught:
            vm.run()
        self.assertIn("!G", str(caught.exception))

    def test_bad_op_traps(self):
        vm = NanVM(parse_nan_image(
            "FFF8_1F000000_F8FF\n"   # op 0x1F is not in NaNVM0
            "FFF8_FF000000_F8FF\n"))
        with self.assertRaises(NanVMError) as caught:
            vm.run()
        self.assertIn("!O", str(caught.exception))

    def test_step_limit(self):
        vm = NanVM(parse_nan_image("FFF8_0A000000_F8FF\n"))  # JMP 0
        with self.assertRaises(NanVMError):
            vm.run(max_steps=1000)

    def test_fexp_ftanh_flog_primitives(self):
        def run(op):
            vm = NanVM(parse_nan_image(
                "FFF8_%08X_F8FF\n" % ((op << 24) | (1 << 20)) +
                "FFF8_FF000000_F8FF\n"))
            vm.reg[0] = 0x3F800000  # 1.0f
            vm.run()
            return struct.unpack("<f", struct.pack("<I", vm.reg[1]))[0]
        self.assertAlmostEqual(run(0x46), math.e, places=5)
        self.assertAlmostEqual(run(0x47), math.tanh(1.0), places=5)
        self.assertAlmostEqual(run(0x48), 0.0, places=5)

    def test_memory_size(self):
        vm = NanVM([])
        self.assertEqual(len(vm.mem), MEM_WORDS)


if __name__ == "__main__":
    unittest.main()
