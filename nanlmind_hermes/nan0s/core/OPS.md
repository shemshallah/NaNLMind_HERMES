# NaN16/NaNVM1 native-core semantics (from src/native/kan16.xa, the x86 gold core) - REVISED r3 (re-verified line-by-line against r28d kan16.xa)
Authority: kan16.xa (AT&T order: `MOV src,dst`). Every line below was re-read against the full 1156-line source
this round. Items marked [FIX] correct the first draft of this file.

## Image / memory
- build/nan0s.img = 1,474,560 B. Boot sector 0-511, kernel 512-8703, guest words from byte 0x2200 (8704), 65,536 words
  (524,288 B). The kernel copies them to guest word 0 after zero-filling 4M words (MASK 0x3FFFFF, 32 MB).
- Word = 8 raw bytes: b0-1 FAR head (big-endian u16), b2-5 payload (LE u32), b6-7 NEAR head (LE u16).
  Heads: 0x7FF1,0xFFF1,0x7FF8,0xFFF8 = class 0..3. Valid iff (h&0x7FF0)==0x7FF0 and (h&15) is 1 or 8.
  class(h) = ((h>>2)&2) | (h>>15).  nibble n = class(NEAR)<<2 | class(FAR).   (nib: near first, then far)
- putw(index,payload,n): FAR = heads[n&3], NEAR = heads[n>>2]. ST/STB/BFILL use n=10 (both 0x7FF8). STW uses r[b]&15.
- [FIX] `.nan` text line `AAAA_PPPPPPPP_BBBB`: field1 = NEAR head VALUE (e.g. FFF1 -> b6=F1,b7=FF), field2 = payload as the
  big-endian hex of the LE u32 (0C000000 -> b2..b5 = 00 00 00 0C), field3 = FAR head written byte-swapped
  (F17F -> head 0x7FF1 -> b0=7F,b1=F1). Verified against image words 0,35,37,40 of museum.nan.
  e.g. `FFF1_0C000000_F17F` (RET[EQ]) = bytes `7FF1 0000000C F1FF`; `FFF8_0A000200_F8FF` (JMP 512) = `FFF8 0002000A F8FF`.
  (mkimg.awk's names NEAR/FAR for the text fields are the same convention: n = idx(field1)*4 + idx(field3).)

## Fetch / decode
loop: steps++ (kernel stops at MAXS=0x7FFFFFFF silently; the C core uses a per-call budget instead).
gate: BOTH heads valid else trap "!G" . pc++ . op==0xFF HALT (silent, unconditional, before predication).
op>0x48 -> trap "!O" (also unconditional). Then: op!=0 requires (n>>k)&1 else the op is skipped silently - this applies
to ALL ops incl. the bad ones in the table (0x0E, 0x1C-0x1F), so a predicated-off bad op does not trap. BOOL (0x00) ignores n for
the predicate (n is its truth table). Payload p: op=p>>24 d=(p>>20)&15 a=(p>>16)&15 imm=p&0xFFFF A=r[a] B=r[imm&15].
pc is a full 32-bit counter (JMPF/CALLF 24-bit); the word fetched is (pc & MASK). JMP/CALL load imm (16-bit).
Trap text: "!" + 'G'|'O' + uppercase hex of (pc after the fetch increment), min 4 digits, to COM1, then the CPU stops
(CLI;HLT). Gate: shown value = pc+1 (the kernel INCs before printing). Flags: k=2Z+L, FLAGM=1<<k, initial k=0.

## Op table
00 BOOL(n = truth table: bit0 ~(A|B), bit1 ~A&B, bit2 A&~B, bit3 A&B; result Z=res==0, L=0)
01 MOVI r[d]=imm  02 MOVHI r[d]=r[d]&0xFFFF | imm<<16
03 ADD A+B (Z,L=carry)  04 SUB A-B (Z,L=borrow)  05 SHL 06 SHR 27 SAR: count = imm&31 (payload low 5 bits, NOT B), value A, Z only (L=0)
07 CMP (Z=eq, L=A<B unsigned)  28 CMPS (Z=eq, L=A<B signed)
08 LD r[d]=payload[(A+imm)&M] (no gate check)  09 ST payload[(A+imm)&M]=r[d], n=10
0A JMP imm  0B CALL(r15=pc,pc=imm)  0C RET(pc=r15)  0D JMPR pc=A  2C CALLR(r15=pc,pc=A)  0E bad  0F NOP
10 VPOKE(idx=A,val=r[d])  11 VPEEK  12 KEY(blocking)  13 TICK r[d]=ms  14 BSWAPM word[(A+imm)&M] = bswap64(word)
15 PUTC serial<-r[d]&255  16 STW word r[A]&M payload=r[d], nibble r[b]&15 (b=imm&15)  17 LDN r[d]=n of word (A+imm)&M, or 0xFFFFFFFF if a head is bad
18 DISPQ  19 DISPSET(r[d])  1A KEYQ(non-blocking, 0 if empty)  1B NOP  1C-1F bad
20 MUL low32 (Z)  21 MULH signed hi (Z)  22 MULHU unsigned hi (Z)
23 DIV 24 MOD (signed) 25 DIVU 26 MODU: B==0 -> DIV/DIVU=0, MOD/MODU=A, and k=3, result still written; DIV by -1 = -A; MOD by -1 = 0 (Z from result)
29 ADDI A+sext(imm) (Z, L=carry of the 32-bit add)
2A LDB byte addr=A+imm: word=(addr>>2)&M, byte=(addr&3), payload LE  2B STB (read-modify-write payload, n=10)
2D BCOPY dst r[d]&M, src A&M, count B (full 32-bit): if ((dst-src)&M) < count copy BACKWARD from the last word, else forward; raw 8-byte words; wraps with &M
2E BFILL dst r[d], payload A, count B: n=10 words, address &M each step  2F NOP [FIX r3: table entry 0x2F is h_nop - it is NOT a bad op]
30 FADD 31 FSUB 32 FMUL 33 FDIV 34 FSQRT 35 FCMP 36 CVTIF 37 CVTFI (32-bit IEEE float in r[]). All FP ops set k=0 except FCMP/CVTFI (below).
38 JMPF pc=payload&0xFFFFFF  39 CALLF (r15=pc first)
3A FBINFO r[d]=FBW<<16|FBH  3B VSYNC  3C FBCOPY(count r[d], dst A, src B)  3D FBFILL(count r[d], start A, colour B&255)
3E OUTB(port A&0xFFFF, val r[d])  3F INB  40 INW  41 OUTW  42 OUTSW(port A, src word index B, count r[d]; low16 of each payload, B not masked in x86)  43 OUTL (no INL exists)
44 CPUID(leaf A, subleaf 0; r[d] = reg B&3 of EAX,EBX,ECX,EDX)  45 RDTSC (r[d] = A==0 ? low32 : high32)
46 FEXP d,a = expf(A)  47 FTANH d,a = tanhf(A)  48 FLOG d,a = logf(A)
Plain ops leave flags alone: MOVI MOVHI LD ST LDN STW KEY KEYQ TICK VPEEK VPOKE BSWAPM PUTC DISPQ FBINFO INB... BOOL/ADD/SUB/shifts/CMP*/MUL*/DIV*/ADDI set flags.
[FIX] VPEEK with an out-of-range index leaves r[d] UNCHANGED (kernel jumps to loop without storing). VPOKE out of range is ignored.

## Floating point (x86 SSE single, MXCSR default: no FTZ/DAZ, round-to-nearest-even)
FADD/FSUB/FMUL: if A is NaN -> A|0x00400000; else if B is NaN -> B|0x00400000; else IEEE op; if the result is NaN (inf-inf, 0*inf) -> 0xFFC00000.
FDIV: if B is +-0: A is +-0 or NaN -> 0x7FC00000; otherwise +-inf (sign xor) as IEEE. Else the same NaN rules as FADD (inf/inf -> 0xFFC00000).
FSQRT: A NaN or A<0 (strict; -0 -> -0) -> 0x7FC00000.
FEXP/FTANH/FLOG: A NaN -> A|0x00400000; result NaN -> 0x7FC00000; FLOG(+-0) -> -inf, FLOG(negative) -> qNaN; all set k=0. FCMP: unordered -> k=3, else Z=eq, L=A<B. CVTIF = (float)(int32) RNE.
CVTFI: NaN -> 0; trunc toward zero; >=2^31 -> 0x7FFFFFFF; <=-2^31 -> 0x80000000; flags: Z from result, L=0.
ARM note: AArch64/ARMv7 produce 0x7FC00000 for invalid ops and propagate NaNs differently, so the C core implements the rules in software
(bit tests) and never trusts the host FPU for NaN results. Build with -ffp-contract=off -fno-fast-math.

## Display
Modes: text 80x25 (u16 cells char|attr<<8), mode 13h 320x200x8, VBE LFB WxH x8 (pitch = W), 16 MB VRAM.
DISPSET v: v>=0xFFFF0000 -> RAWKB=v&255, clear ring (display unchanged); v>=0x10000 -> VBE (w=v>>16,h=v&0xFFFF; FBTOT=w*h; clear FBTOT bytes; VMODE=1);
else (v&255)==0x13 -> mode 13h (FBW 320 FBH 200 FBTOT 64000, clear); else TEXT: VMODE=0, FBW=80, FBH=25, all 2000 cells set to 0x0000 (not 0x0720!).
FBTOT is NOT reset by text mode. The DAC is NOT touched by any mode switch (the guest loads/keeps its own palette).
Initial (BIOS) state: text mode, cells 0x0720 (assumed; confirm against qemu); FBW/FBH are 0 until the first DISPSET; FBTOT starts at 64000 (set in _start), FBBASE 0xA0000. [FIX r3]
VBE mode: FBBASE = BAR0&~0xF of the first PCI dev (bus0, dev 0..31) whose class>>16 == 0x0300 (else 0xE0000000); the C core uses a flat 16 MB VRAM offset instead.
DISPQ: VMODE==1 -> 0x1300C800 else 0x00031950. VPOKE/VPEEK: text idx<2000 (u16 cell); gfx idx<FBTOT (byte at FB+idx).
FBFILL/FBCOPY are unchecked in x86 (offscreen rows used by the GUI): count is a full 32-bit; copy = forward dword loop then trailing bytes (overlap order matters).
The C core clamps to a 16 MB VRAM instead of crashing.
VGA register writes (0x3C0-0x3D5, fb.nas mode13/mode3 tables) are ignored; fb.nas always follows them with DISPSET.
DAC: 0x3C8 write index, 0x3C9 3x 6-bit RGB (auto-increment), 0x3C7 read index + 0x3C9 reads. Text colour c (0..15) -> DAC index
c<8 ? (c==6 ? 0x14 : c) : 0x38+(c-8); the BIOS text palette = 64-entry EGA table (index bits r g b R G B: 0x14=AA5500, 0x38..3F bright), so a palette
left by a graphics exhibit shows up in text mode exactly as on hardware.
[FIX r4, probed on qemu/SeaBIOS with a boot sector that reads 0x3C9] There is NO default VGA palette: after BIOS mode 3 the DAC is entries 0..63 = EGA table (above) and 64..255 = black. An exhibit that does not load a palette sees exactly that in mode 13h/VBE.
DAC 6->8 bit (qemu c6_to_8, used by screendump and the C core): v8 = (v6<<2) | ((v6&1)<<1) | (v6&1), e.g. 42->168, 21->87, 63->255.
Initial text cells 0x0720 CONFIRMED (menu screen byte-identical to qemu). Text attr: fg = attr&15, bg = (attr>>4)&7 (blink on, BIOS default; guests so far only use attrs 0,8,11,14,15,31).
FP both-operands-NaN: the core returns A (real SSE: first source). qemu 8.2 TCG returns the operand with the larger significand (x87 rule), so fuzz seeds with two NaN inputs can differ from qemu without being a core bug (seed 10 of the first batch).

## Keyboard
Ring of 16 entries (cap 15; full = (tail+1)&15 == head; a push into a full ring is dropped). KEY: pops, or blocks (HLT) until an IRQ1 key arrives;
KEYQ: pops or 0. The C core returns NAN_WAIT_KEY and REWINDS pc to the KEY op so it re-executes when a key arrives (host can sleep).
Cooked IRQ1 path (RAWKB==0): 0xE0 sets E0 flag (no push). Break (bit7): ignored if E0, else 0x2A/0x36 clear SHIFT. Make: if E0 -> ktab[256+sc];
0x2A/0x36 set SHIFT; 0x3A toggles CAPS; else idx = (SHIFT ^ (CAPS if plain ktab[sc] is a..z))*128 + sc. ktab value 0 -> nothing pushed. E0 cleared after every code except the E0 prefix itself.
Specials: up128 down129 left130 right131 home132 end133 pgup134 pgdn135 ins136 del137. Raw (DISPSET 0xFFFF00xx, xx!=0): every scancode byte (incl. E0 and breaks) is pushed as-is.

## Ports used by the guest (everything else: INB reads 0xFF, INW 0xFFFF, writes ignored)
0x3C8/0x3C9/0x3C7 DAC; 0x3DA retrace status (toggle bit3); 0x42/0x43/0x61 PIT ch2 + speaker (audio.nas: 0xB6 to 0x43, divisor lo/hi to 0x42,
bits0-1 of 0x61 = gate+enable; hz = 1193182/divisor); 0x3F8 COM1 data -> serial, 0x3FD reads 0x60;
PCI mech #1: OUTL 0xCF8 then INW 0xCFC/0xCFE (gui_cart pci_scan: bus 0, dev 0..31): dev0 8086:1237 class 060000, dev1 8086:7000 class 060100,
dev2 1234:1111 class 030000 BAR0 0xFD000008; others 0xFFFFFFFF. ATA primary master 0x1F0-0x1F7/0x3F6 (ata.nas, gui ut_atas IDENTIFY x4 slots):
PIO LBA28, commands 0xEC IDENTIFY, 0x20 READ, 0x30 WRITE, 0xE7 FLUSH; no drive attached => status reads 0xFF (floating bus). Secondary/slaves absent.
CPUID leaf0: vendor in EBX,EDX,ECX; leaf 1 family/model/stepping in EAX; 0x80000000 max ext leaf; 0x80000001 EDX bit29 long mode; 0x80000002-4 brand string.
RDTSC: any monotonic counter (the C core uses the executed-instruction count; TICK is host milliseconds, so the GUI's "MHz" is real emulation speed).
TICK: x86 programs the PIT to ~1000 Hz and counts ms from boot; the C core returns the host-supplied ms (nan_set_time_ms).

## Verified in r3 against kan16.xa (no change needed)
Op table 00-45 order, gate/trap text (min 4 hex digits, no newline, CLI;HLT), predication TEST EDI,[FLAGM], flag packing k=2Z+L, DIV/MOD zero & -1 paths (k=3 on B==0),
FP via SSE (software rules in OPS match: FDIV zero path, FSQRT strict <0, CVTFI saturation, FCMP unordered k=3), BCOPY/BFILL, STB/LDB shift, FBCOPY (dst=A src=B count=r[d]), FBFILL,
DISPSET forms, keyboard IRQ1 state machine (raw path skips E0/shift logic), KEY/KEYQ, TICK=[MS] ~1000 Hz, CPUID = REAL host cpuid (the C core returns a fixed table; cannot match bit-for-bit), RDTSC = real TSC.
