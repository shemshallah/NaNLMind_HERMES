/* nan16.c - host conformance harness for the NaN16/NaNVM1 core (S6).
   NOT the VM: spec/NANVM_SPEC.md is normative; no semantics land here first.
   Semantics ref: OPS.md (verified against kan16.xa r28d). */
#include "nan16_int.h"
#include <math.h>

#define M NAN_MASK

static inline int is_nan32(uint32_t x) { return (x & 0x7F800000u) == 0x7F800000u && (x & 0x007FFFFFu); }
static inline float f32(uint32_t x) { float f; memcpy(&f, &x, 4); return f; }
static inline uint32_t u32(float f) { uint32_t x; memcpy(&x, &f, 4); return x; }

static uint32_t fp_bin(int op, uint32_t A, uint32_t B) {
  if (is_nan32(A)) return A | 0x00400000u;
  if (is_nan32(B)) return B | 0x00400000u;
  volatile float a = f32(A), b = f32(B), r;
  switch (op) { case 0x30: r = a + b; break; case 0x31: r = a - b; break;
                case 0x32: r = a * b; break; default: r = a / b; break; }
  uint32_t x = u32(r);
  return is_nan32(x) ? 0xFFC00000u : x;
}
static uint32_t fp_div(uint32_t A, uint32_t B) {
  if ((B & 0x7FFFFFFFu) == 0) {
    if ((A & 0x7FFFFFFFu) == 0 || is_nan32(A)) return 0x7FC00000u;
    return ((A ^ B) & 0x80000000u) | 0x7F800000u;
  }
  return fp_bin(0x33, A, B);
}
static uint32_t fp_sqrt(uint32_t A) {
  if (is_nan32(A) || ((A & 0x80000000u) && (A & 0x7FFFFFFFu))) return 0x7FC00000u;
  volatile float a = f32(A), r = sqrtf(a);
  return u32(r);
}
static uint32_t fp_cvtfi(uint32_t A) {
  if (is_nan32(A)) return 0;
  float f = f32(A);
  if (f >= 2147483648.0f) return 0x7FFFFFFFu;
  if (f <= -2147483648.0f) return 0x80000000u;
  return (uint32_t)(int32_t)f;
}

nan_vm *nan_create(void) {
  nan_vm *v = (nan_vm *)calloc(1, sizeof *v);
  if (!v) return NULL;
  v->mem = (uint64_t *)calloc(NAN_WORDS, 8);
  v->vram = (uint8_t *)calloc(NAN_VRAM, 1);
  v->ser = (char *)calloc(NAN_SER_CAP, 1);
  if (!v->mem || !v->vram || !v->ser) { nan_destroy(v); return NULL; }
  nan_reset(v);
  return v;
}
void nan_destroy(nan_vm *v) {
  if (!v) return;
  free(v->mem); free(v->vram); free(v->ser); free(v);
}
void nan_reset(nan_vm *v) {
  memset(v->mem, 0, (size_t)NAN_WORDS * 8);
  memset(v->vram, 0, NAN_VRAM);
  memset(v->r, 0, sizeof v->r);
  v->pc = 0; v->k = 0; v->halted = 0; v->steps = 0; v->ms = 0;
  for (int i = 0; i < 2000; i++) v->cells[i] = 0x0720;
  v->vmode = 0; v->fbw = 0; v->fbh = 0; v->fbtot = 64000;
  v->trap[0] = 0; v->ser_h = v->ser_t = 0;
  v->tick_last = 0xFFFFFFFFu; v->tick_same = 0;
  v->dirty++; v->frame++;
  nan_dev_reset(v);
}

/* ---- accessors ---- */
uint64_t nan_steps(const nan_vm *v) { return v->steps; }
const char *nan_trap_text(const nan_vm *v) { return v->trap; }
void nan_set_time_ms(nan_vm *v, uint32_t ms) { v->ms = ms; }
uint32_t nan_reg(const nan_vm *v, int i) { return v->r[i & 15]; }
uint32_t nan_pc(const nan_vm *v) { return v->pc; }
uint64_t nan_word(const nan_vm *v, uint32_t i) { return v->mem[i & M]; }
void nan_set_word(nan_vm *v, uint32_t i, uint64_t raw) { v->mem[i & M] = raw; }
void nan_set_reg(nan_vm *v, int i, uint32_t x) { v->r[i & 15] = x; }
void nan_set_pc(nan_vm *v, uint32_t pc) { v->pc = pc; v->halted = 0; }
int nan_flags(const nan_vm *v) { return v->k; }
void nan_set_trace(nan_vm *v, nan_trace_fn cb, void *ctx) { v->trace = cb; v->trace_ctx = ctx; }
uint32_t nan_dirty(const nan_vm *v) { return v->dirty; }
uint32_t nan_frame_counter(const nan_vm *v) { return v->frame; }
int nan_display_mode(const nan_vm *v) { return v->vmode; }
void nan_display_size(const nan_vm *v, int *w, int *h) {
  if (v->vmode) { *w = (int)v->fbw; *h = (int)v->fbh; } else { *w = 80; *h = 25; }
}
uint16_t nan_text_cell(const nan_vm *v, int i) { return (i >= 0 && i < 2000) ? v->cells[i] : 0; }

static void trap(nan_vm *v, char kind, uint32_t shown) {
  char b[24]; snprintf(b, sizeof b, "!%c%04X", kind, (unsigned)shown);
  for (const char *p = b; *p; p++) nan_ser_putc(v, (uint8_t)*p);
  snprintf(v->trap, sizeof v->trap, "%s", b);
  v->pc = shown; v->halted = 1;
}

/* ---- VRAM helpers (clamped to 16 MB; x86 is unchecked) ---- */
static void fb_copy(nan_vm *v, uint32_t cnt, uint32_t dst, uint32_t src) {
  uint64_t cap = NAN_VRAM, n4 = cnt >> 2; uint8_t *f = v->vram;
  uint64_t lo = dst < src ? dst : src;
  if (lo >= cap) return;
  if ((uint64_t)dst + cnt <= cap && (uint64_t)src + cnt <= cap &&
      ((uint64_t)dst + cnt <= src || (uint64_t)src + cnt <= dst)) { memcpy(f + dst, f + src, cnt); v->dirty++; return; }
  for (uint64_t i = 0; i < n4; i++) {           /* forward dword loop (rep movsd order) */
    uint64_t d = (uint64_t)dst + i * 4, s = (uint64_t)src + i * 4;
    if (d >= cap && s >= cap) return;
    uint8_t t[4];
    for (int j = 0; j < 4; j++) t[j] = (s + j < cap) ? f[s + j] : 0;
    for (int j = 0; j < 4; j++) if (d + j < cap) f[d + j] = t[j];
  }
  for (uint32_t j = 0; j < (cnt & 3); j++) {       /* trailing bytes */
    uint64_t d = (uint64_t)dst + n4 * 4 + j, s = (uint64_t)src + n4 * 4 + j;
    if (d < cap) f[d] = (s < cap) ? f[s] : 0;
  }
  v->dirty++;
}
static void fb_fill(nan_vm *v, uint32_t cnt, uint32_t start, uint8_t c) {
  uint64_t cap = NAN_VRAM;
  if (start >= cap) return;
  uint64_t n = cnt; if (start + n > cap) n = cap - start;
  memset(v->vram + start, c, (size_t)n); v->dirty++;
}

static void dispset(nan_vm *v, uint32_t x) {
  if (x >= 0xFFFF0000u) { v->rawkb = x & 255; v->rhead = v->rtail = 0; return; }
  if (x >= 0x10000u) {
    v->fbw = x >> 16; v->fbh = x & 0xFFFF; uint64_t t = (uint64_t)v->fbw * v->fbh;
    v->fbtot = (uint32_t)t; v->vmode = 1;
    memset(v->vram, 0, t > NAN_VRAM ? NAN_VRAM : (size_t)(t & ~3ull));
  } else if ((x & 255) == 0x13) {
    v->vmode = 1; v->fbtot = 64000; v->fbw = 320; v->fbh = 200; memset(v->vram, 0, 64000);
  } else {
    v->vmode = 0; v->fbw = 80; v->fbh = 25;
    for (int i = 0; i < 2000; i++) v->cells[i] = 0;
  }
  v->dirty++; v->frame++;
}

int nan_run(nan_vm *v, uint32_t max_steps) {
  uint32_t *r = v->r; uint64_t *mem = v->mem;
  uint32_t pc = v->pc; uint32_t done = 0;
  int rc = NAN_BUDGET;
#define SETK(z, l) (v->k = 2 * (int)(z) + (int)(l))
#define LEAVE(code) do { rc = (code); goto out; } while (0)
  if (v->halted) return NAN_HALTED;
  while (done < max_steps) {
    done++; v->steps++;
    uint64_t w = mem[pc & M];
    uint32_t nh = nan_near(w), fh = nan_far(w);
    if (!nan_hvalid(nh) || !nan_hvalid(fh)) { trap(v, 'G', pc + 1); pc = v->pc; LEAVE(NAN_HALTED); }
    uint32_t at = pc; pc++;
    uint32_t p = (uint32_t)(w >> 16), op = p >> 24;
    int n = (int)((nan_hcls(nh) << 2) | nan_hcls(fh));
    if (op == 0xFF) { v->halted = 1; LEAVE(NAN_HALTED); }
    if (op > 0x45) { trap(v, 'O', pc); pc = v->pc; LEAVE(NAN_HALTED); }
    if (op != 0 && !((n >> v->k) & 1)) continue;
    if (v->trace) v->trace(v->trace_ctx, at, p, n, v->k);
    uint32_t d = (p >> 20) & 15, a = (p >> 16) & 15, imm = p & 0xFFFF;
    uint32_t A = r[a], B = r[imm & 15];
    switch (op) {
    case 0x00: { uint32_t res = 0;
      if (n & 1) res |= ~(A | B); if (n & 2) res |= ~A & B; if (n & 4) res |= A & ~B; if (n & 8) res |= A & B;
      r[d] = res; SETK(res == 0, 0); break; }
    case 0x01: r[d] = imm; break;
    case 0x02: r[d] = (r[d] & 0xFFFF) | (imm << 16); break;
    case 0x03: { uint64_t s = (uint64_t)A + B; r[d] = (uint32_t)s; SETK((uint32_t)s == 0, s >> 32); break; }
    case 0x04: { uint32_t res = A - B; r[d] = res; SETK(res == 0, A < B); break; }
    case 0x05: { uint32_t res = A << (p & 31); r[d] = res; SETK(res == 0, 0); break; }
    case 0x06: { uint32_t res = A >> (p & 31); r[d] = res; SETK(res == 0, 0); break; }
    case 0x27: { uint32_t res = (uint32_t)((int32_t)A >> (p & 31)); r[d] = res; SETK(res == 0, 0); break; }
    case 0x07: SETK(A == B, A < B); break;
    case 0x28: SETK(A == B, (int32_t)A < (int32_t)B); break;
    case 0x08: r[d] = (uint32_t)(mem[(A + imm) & M] >> 16); break;
    case 0x09: mem[(A + imm) & M] = nan_mkword(r[d], 10); break;
    case 0x0A: pc = imm; break;
    case 0x0B: r[15] = pc; pc = imm; break;
    case 0x0C: pc = r[15]; break;
    case 0x0D: pc = A; break;
    case 0x2C: r[15] = pc; pc = A; break;
    case 0x0E: case 0x1C: case 0x1D: case 0x1E: case 0x1F:
      trap(v, 'O', pc); pc = v->pc; LEAVE(NAN_HALTED);
    case 0x0F: case 0x1B: case 0x2F: break;
    case 0x10: { uint32_t i = A;
      if (v->vmode == 1) { if (i < v->fbtot && i < NAN_VRAM) { v->vram[i] = (uint8_t)r[d]; v->dirty++; } }
      else if (i < 2000) { v->cells[i] = (uint16_t)r[d]; v->dirty++; } break; }
    case 0x11: { uint32_t i = A;
      if (v->vmode == 1) { if (i < v->fbtot && i < NAN_VRAM) r[d] = v->vram[i]; }
      else if (i < 2000) r[d] = v->cells[i]; break; }
    case 0x12: { int c = nan_ring_pop(v);
      if (c < 0) { pc = at; v->steps--; LEAVE(NAN_WAIT_KEY); }
      r[d] = (uint32_t)c; break; }
    case 0x1A: { int c = nan_ring_pop(v); r[d] = c < 0 ? 0 : (uint32_t)c; break; }
    case 0x13: r[d] = v->ms;
      if (v->ms == v->tick_last) { if (++v->tick_same >= 256) { v->tick_same = 0; LEAVE(NAN_TIME_WAIT); } }
      else { v->tick_last = v->ms; v->tick_same = 0; }
      break;
    case 0x14: { uint64_t *q = &mem[(A + imm) & M]; *q = __builtin_bswap64(*q); break; }
    case 0x15: nan_ser_putc(v, (uint8_t)r[d]); break;
    case 0x16: mem[A & M] = nan_mkword(r[d], (int)(B & 15)); break;
    case 0x17: { uint64_t q = mem[(A + imm) & M];
      r[d] = (nan_hvalid(nan_near(q)) && nan_hvalid(nan_far(q))) ? ((nan_hcls(nan_near(q)) << 2) | nan_hcls(nan_far(q))) : 0xFFFFFFFFu; break; }
    case 0x18: r[d] = v->vmode == 1 ? 0x1300C800u : 0x00031950u; break;
    case 0x19: dispset(v, r[d]); break;
    case 0x20: { uint32_t res = A * B; r[d] = res; SETK(res == 0, 0); break; }
    case 0x21: { uint32_t res = (uint32_t)(((int64_t)(int32_t)A * (int32_t)B) >> 32); r[d] = res; SETK(res == 0, 0); break; }
    case 0x22: { uint32_t res = (uint32_t)(((uint64_t)A * B) >> 32); r[d] = res; SETK(res == 0, 0); break; }
    case 0x23: case 0x24: case 0x25: case 0x26: {
      uint32_t res;
      if (B == 0) { res = (op == 0x23 || op == 0x25) ? 0 : A; r[d] = res; v->k = 3; break; }
      if (op == 0x23) res = (B == 0xFFFFFFFFu) ? 0u - A : (uint32_t)((int32_t)A / (int32_t)B);
      else if (op == 0x24) res = (B == 0xFFFFFFFFu) ? 0 : (uint32_t)((int32_t)A % (int32_t)B);
      else if (op == 0x25) res = A / B; else res = A % B;
      r[d] = res; SETK(res == 0, 0); break; }
    case 0x29: { uint32_t s2 = (uint32_t)(int32_t)(int16_t)imm; uint64_t s = (uint64_t)A + s2;
      r[d] = (uint32_t)s; SETK((uint32_t)s == 0, s >> 32); break; }
    case 0x2A: { uint32_t ad = A + imm; r[d] = ((uint32_t)(mem[(ad >> 2) & M] >> 16) >> ((ad & 3) * 8)) & 0xFF; break; }
    case 0x2B: { uint32_t ad = A + imm, sh = (ad & 3) * 8; uint64_t *q = &mem[(ad >> 2) & M];
      uint32_t pl = (uint32_t)(*q >> 16); pl = (pl & ~(0xFFu << sh)) | ((r[d] & 0xFF) << sh); *q = nan_mkword(pl, 10); break; }
    case 0x2D: { uint32_t src = A & M, dst = r[d] & M, cnt = B;
      if (cnt) {
        if (((dst - src) & M) < cnt) { uint32_t s = (src + cnt - 1) & M, t = (dst + cnt - 1) & M;
          for (uint32_t i = 0; i < cnt; i++) { mem[t] = mem[s]; s = (s - 1) & M; t = (t - 1) & M; } }
        else for (uint32_t i = 0; i < cnt; i++) { mem[dst] = mem[src]; src = (src + 1) & M; dst = (dst + 1) & M; } }
      break; }
    case 0x2E: { uint32_t dst = r[d], cnt = B; uint64_t wd = nan_mkword(A, 10);
      for (uint32_t i = 0; i < cnt; i++) mem[(dst + i) & M] = wd; break; }
    case 0x30: case 0x31: case 0x32: r[d] = fp_bin((int)op, A, B); v->k = 0; break;
    case 0x33: r[d] = fp_div(A, B); v->k = 0; break;
    case 0x34: r[d] = fp_sqrt(A); v->k = 0; break;
    case 0x35: if (is_nan32(A) || is_nan32(B)) v->k = 3; else SETK(f32(A) == f32(B), f32(A) < f32(B)); break;
    case 0x36: { volatile float f = (float)(int32_t)A; r[d] = u32(f); v->k = 0; break; }
    case 0x37: { uint32_t res = fp_cvtfi(A); r[d] = res; SETK(res == 0, 0); break; }
    case 0x38: pc = p & 0xFFFFFF; break;
    case 0x39: r[15] = pc; pc = p & 0xFFFFFF; break;
    case 0x3A: r[d] = (v->fbw << 16) | v->fbh; break;
    case 0x3B: if (v->vmode == 1) { v->frame++; LEAVE(NAN_VSYNC); } break;
    case 0x3C: fb_copy(v, r[d], A, B); break;
    case 0x3D: fb_fill(v, r[d], A, (uint8_t)B); break;
    case 0x3E: nan_port_out(v, A & 0xFFFF, r[d], 1); break;
    case 0x3F: r[d] = nan_port_in(v, A & 0xFFFF, 1) & 0xFF; break;
    case 0x40: r[d] = nan_port_in(v, A & 0xFFFF, 2) & 0xFFFF; break;
    case 0x41: nan_port_out(v, A & 0xFFFF, r[d] & 0xFFFF, 2); break;
    case 0x42: { uint32_t cnt = r[d], idx = B;
      for (uint32_t i = 0; i < cnt; i++, idx++) nan_port_out(v, A & 0xFFFF, (uint32_t)(mem[idx & M] >> 16) & 0xFFFF, 2); break; }
    case 0x43: nan_port_out(v, A & 0xFFFF, r[d], 4); break;
    case 0x44: { uint32_t reg[4] = {0, 0, 0, 0};
      switch (A) {
      case 0: reg[0] = 0x16; reg[1] = 0x314E614Eu /* "NaN1" */; reg[3] = 0x726F4336u /* "6Cor" */; reg[2] = 0x4D565F65u /* "e_VM" */; break;
      case 1: reg[0] = 0x00000F10; reg[3] = 0x00000001; break;
      case 0x80000000u: reg[0] = 0x80000004u; break;
      case 0x80000001u: reg[3] = 1u << 29; break;
      case 0x80000002u: memcpy(reg, "NaN16 portable C", 16); break;
      case 0x80000003u: memcpy(reg, " core (not x86) ", 16); break;
      case 0x80000004u: break; }
      r[d] = reg[B & 3]; break; }
    case 0x45: r[d] = A == 0 ? (uint32_t)v->steps : (uint32_t)(v->steps >> 32); break;
    default: trap(v, 'O', pc); pc = v->pc; LEAVE(NAN_HALTED);
    }
  }
out:
  v->pc = pc;
  return rc;
}
