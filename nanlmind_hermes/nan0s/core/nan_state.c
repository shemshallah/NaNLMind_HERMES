/* nan_state.c - sparse snapshot of a nan_vm ("NAN1" format, little-endian, version 1).
 *
 *   magic "NAN1" | u32 version | u32 scalar_len | scalar block | u32 n_mem_pages | u32 n_vram_pages | pages...
 *   a page = u32 page_index + 4096 bytes (mem: 512 words of 8 bytes, vram: 4096 bytes); only non-zero pages are stored.
 *
 * The disk backend, the serial FIFO and the trace/port hooks are host-owned and are NOT part of a snapshot.
 * nan_restore_state() validates every field and every page index BEFORE touching the VM.
 */
#include "nan16_int.h"

#define PAGE 4096u
#define MEM_PAGES ((uint32_t)(((uint64_t)NAN_WORDS * 8) / PAGE))
#define VRAM_PAGES (NAN_VRAM / PAGE)
#define STATE_VERSION 1u
#define SCALAR_LEN 6144u   /* fixed size of the scalar block (about 5.5 KB used, rest reserved), see put_scalars() */

typedef struct { uint8_t *p; size_t n, cap; int ok; } wr;
static void w8(wr *w, uint32_t x)  { if (w->p) { if (w->n + 1 > w->cap) w->ok = 0; else w->p[w->n] = (uint8_t)x; } w->n += 1; }
static void w16(wr *w, uint32_t x) { w8(w, x); w8(w, x >> 8); }
static void w32(wr *w, uint32_t x) { w16(w, x); w16(w, x >> 16); }
static void w64(wr *w, uint64_t x) { w32(w, (uint32_t)x); w32(w, (uint32_t)(x >> 32)); }
static void wb(wr *w, const void *s, size_t len) {
  if (w->p) { if (w->n + len > w->cap) w->ok = 0; else memcpy(w->p + w->n, s, len); }
  w->n += len;
}

typedef struct { const uint8_t *p; size_t n, len; int ok; } rd;
static uint32_t r8(rd *r)  { if (r->n + 1 > r->len) { r->ok = 0; return 0; } return r->p[r->n++]; }
static uint32_t r16(rd *r) { uint32_t a = r8(r); return a | (r8(r) << 8); }
static uint32_t r32(rd *r) { uint32_t a = r16(r); return a | (r16(r) << 16); }
static uint64_t r64(rd *r) { uint64_t a = r32(r); return a | ((uint64_t)r32(r) << 32); }
static void rb(rd *r, void *d, size_t len) { if (r->n + len > r->len) { r->ok = 0; memset(d, 0, len); return; } memcpy(d, r->p + r->n, len); r->n += len; }

static int page_zero(const uint8_t *p) {
  for (uint32_t i = 0; i < PAGE; i += 8) { uint64_t x; memcpy(&x, p + i, 8); if (x) return 0; }
  return 1;
}

static void put_scalars(wr *w, const nan_vm *v) {
  size_t start = w->n;
  for (int i = 0; i < 16; i++) w32(w, v->r[i]);
  w32(w, v->pc); w32(w, (uint32_t)v->k); w32(w, (uint32_t)v->halted); w64(w, v->steps); w32(w, v->ms);
  w32(w, (uint32_t)v->vmode); w32(w, v->fbw); w32(w, v->fbh); w32(w, v->fbtot);
  for (int i = 0; i < 2000; i++) w16(w, v->cells[i]);
  wb(w, v->dac, 768);
  w32(w, (uint32_t)v->dac_loaded); w32(w, (uint32_t)v->dac_wi); w32(w, (uint32_t)v->dac_wp); w32(w, (uint32_t)v->dac_ri); w32(w, (uint32_t)v->dac_rp);
  wb(w, v->ring, 16); w32(w, (uint32_t)v->rhead); w32(w, (uint32_t)v->rtail); w32(w, (uint32_t)v->rawkb);
  w32(w, (uint32_t)v->shift); w32(w, (uint32_t)v->caps); w32(w, (uint32_t)v->e0);
  { char t[24]; memset(t, 0, sizeof t); memcpy(t, v->trap, sizeof t - 1); wb(w, t, 24); }
  w32(w, v->pit_div); w32(w, (uint32_t)v->pit_phase); w8(w, v->p61); w8(w, v->pit_ctl); w32(w, v->pci_addr); w8(w, v->retrace);
  { const nan_ata *a = &v->ata;
    w8(w, a->err); w8(w, a->count); wb(w, a->lba, 3); w8(w, a->drive); w8(w, a->status); w8(w, a->cmd);
    for (int i = 0; i < 256; i++) w16(w, a->buf[i]);
    w32(w, (uint32_t)a->pos); w32(w, (uint32_t)a->mode); w32(w, a->lba_cur); w32(w, (uint32_t)a->remain); }
  w32(w, v->tick_last); w32(w, v->tick_same); w32(w, v->frame); w32(w, v->dirty);
  if (w->n - start > SCALAR_LEN) w->ok = 0;            /* layout outgrew the block: bump STATE_VERSION and SCALAR_LEN */
  while (w->n - start < SCALAR_LEN) w8(w, 0);          /* fixed-size block (room for later additions) */
}

static size_t build(const nan_vm *v, uint8_t *buf, size_t cap) {
  wr w = { buf, 0, cap, 1 };
  wb(&w, "NAN1", 4); w32(&w, STATE_VERSION); w32(&w, SCALAR_LEN);
  put_scalars(&w, v);
  uint32_t nm = 0, nv = 0;
  const uint8_t *m = (const uint8_t *)v->mem;
  for (uint32_t i = 0; i < MEM_PAGES; i++) if (!page_zero(m + (size_t)i * PAGE)) nm++;
  for (uint32_t i = 0; i < VRAM_PAGES; i++) if (!page_zero(v->vram + (size_t)i * PAGE)) nv++;
  w32(&w, nm); w32(&w, nv);
  for (uint32_t i = 0; i < MEM_PAGES; i++) if (!page_zero(m + (size_t)i * PAGE)) { w32(&w, i); wb(&w, m + (size_t)i * PAGE, PAGE); }
  for (uint32_t i = 0; i < VRAM_PAGES; i++) if (!page_zero(v->vram + (size_t)i * PAGE)) { w32(&w, i); wb(&w, v->vram + (size_t)i * PAGE, PAGE); }
  return w.ok ? w.n : 0;
}

size_t nan_state_size(const nan_vm *v) { return build(v, NULL, 0); }
size_t nan_save_state(const nan_vm *v, uint8_t *buf, size_t cap) { return buf ? build(v, buf, cap) : 0; }

/* little-endian words in the file; the in-memory words are the LE u64 of the raw bytes, so a memcpy of the page is exact only on LE hosts */
static inline uint64_t le64(const uint8_t *p) { uint64_t x = 0; for (int i = 7; i >= 0; i--) x = (x << 8) | p[i]; return x; }

int nan_restore_state(nan_vm *v, const uint8_t *buf, size_t len) {
  if (!buf || len < 16 || memcmp(buf, "NAN1", 4)) return -1;
  rd h = { buf, 4, len, 1 };
  if (r32(&h) != STATE_VERSION || r32(&h) != SCALAR_LEN) return -1;
  size_t scalars_at = h.n, pages_at = scalars_at + SCALAR_LEN;
  if (pages_at + 8 > len) return -1;
  rd pr = { buf, pages_at, len, 1 };
  uint32_t nm = r32(&pr), nv = r32(&pr);
  if (nm > MEM_PAGES || nv > VRAM_PAGES) return -1;
  size_t need = pr.n + ((size_t)nm + nv) * (4 + (size_t)PAGE);
  if (need != len) return -1;
  /* validate page indices (strictly increasing, in range) before changing anything */
  { rd q = pr; int64_t last = -1;
    for (uint32_t i = 0; i < nm; i++) { uint32_t ix = r32(&q); if (ix >= MEM_PAGES || (int64_t)ix <= last) return -1; last = ix; q.n += PAGE; }
    last = -1;
    for (uint32_t i = 0; i < nv; i++) { uint32_t ix = r32(&q); if (ix >= VRAM_PAGES || (int64_t)ix <= last) return -1; last = ix; q.n += PAGE; } }

  /* parse the scalar block into locals, sanitising everything the interpreter indexes with */
  rd r = { buf, scalars_at, scalars_at + SCALAR_LEN, 1 };
  uint32_t reg[16]; for (int i = 0; i < 16; i++) reg[i] = r32(&r);
  uint32_t pc = r32(&r), k = r32(&r), halted = r32(&r); uint64_t steps = r64(&r); uint32_t ms = r32(&r);
  uint32_t vmode = r32(&r), fbw = r32(&r), fbh = r32(&r), fbtot = r32(&r);
  uint16_t cells[2000]; for (int i = 0; i < 2000; i++) cells[i] = (uint16_t)r16(&r);
  uint8_t dac[256][3]; rb(&r, dac, 768);
  uint32_t dloaded = r32(&r), dwi = r32(&r), dwp = r32(&r), dri = r32(&r), drp = r32(&r);
  uint8_t ring[16]; rb(&r, ring, 16); uint32_t rhead = r32(&r), rtail = r32(&r), rawkb = r32(&r), shift = r32(&r), caps = r32(&r), e0 = r32(&r);
  char trap[24]; rb(&r, trap, 24); trap[23] = 0;
  uint32_t pit_div = r32(&r), pit_phase = r32(&r); uint8_t p61 = (uint8_t)r8(&r), pit_ctl = (uint8_t)r8(&r); uint32_t pci = r32(&r); uint8_t retrace = (uint8_t)r8(&r);
  nan_ata a; memset(&a, 0, sizeof a);
  a.err = (uint8_t)r8(&r); a.count = (uint8_t)r8(&r); rb(&r, a.lba, 3); a.drive = (uint8_t)r8(&r); a.status = (uint8_t)r8(&r); a.cmd = (uint8_t)r8(&r);
  for (int i = 0; i < 256; i++) a.buf[i] = (uint16_t)r16(&r);
  uint32_t apos = r32(&r), amode = r32(&r), alba = r32(&r), arem = r32(&r);
  uint32_t tick_last = r32(&r), tick_same = r32(&r), frame = r32(&r), dirty = r32(&r);
  if (!r.ok) return -1;
  if (k > 3 || vmode > 1 || fbtot > NAN_VRAM || (vmode && (uint64_t)fbw * fbh > NAN_VRAM)) return -1;
  if (dwi > 255 || dwp > 2 || dri > 255 || drp > 2 || rhead > 15 || rtail > 15 || rawkb > 255 || apos > 255 || amode > 2 || arem > 256) return -1;

  /* apply */
  memset(v->mem, 0, (size_t)NAN_WORDS * 8); memset(v->vram, 0, NAN_VRAM);
  { rd q = pr;
    for (uint32_t i = 0; i < nm; i++) { uint32_t ix = r32(&q); uint64_t *dst = v->mem + (size_t)ix * (PAGE / 8);
      for (uint32_t j = 0; j < PAGE / 8; j++) dst[j] = le64(q.p + q.n + (size_t)j * 8);
      q.n += PAGE; }
    for (uint32_t i = 0; i < nv; i++) { uint32_t ix = r32(&q); memcpy(v->vram + (size_t)ix * PAGE, q.p + q.n, PAGE); q.n += PAGE; } }
  memcpy(v->r, reg, sizeof reg); v->pc = pc; v->k = (int)k; v->halted = halted ? 1 : 0; v->steps = steps; v->ms = ms;
  v->vmode = (int)vmode; v->fbw = fbw; v->fbh = fbh; v->fbtot = fbtot; memcpy(v->cells, cells, sizeof cells);
  memcpy(v->dac, dac, sizeof dac); v->dac_loaded = dloaded ? 1 : 0; v->dac_wi = (int)dwi; v->dac_wp = (int)dwp; v->dac_ri = (int)dri; v->dac_rp = (int)drp;
  memcpy(v->ring, ring, 16); v->rhead = (int)rhead; v->rtail = (int)rtail; v->rawkb = (int)rawkb; v->shift = shift ? 1 : 0; v->caps = caps ? 1 : 0; v->e0 = e0 ? 1 : 0;
  memcpy(v->trap, trap, sizeof v->trap);
  v->pit_div = pit_div & 0xFFFF; v->pit_phase = pit_phase ? 1 : 0; v->p61 = p61; v->pit_ctl = pit_ctl; v->pci_addr = pci; v->retrace = retrace;
  { nan_ata *d = &v->ata;        /* keep the host-owned backend (mem/fd/size), replace the register file */
    d->err = a.err; d->count = a.count; memcpy(d->lba, a.lba, 3); d->drive = a.drive; d->status = a.status; d->cmd = a.cmd;
    memcpy(d->buf, a.buf, sizeof a.buf); d->pos = (int)apos; d->mode = (int)amode; d->lba_cur = alba; d->remain = (int)arem; }
  v->tick_last = tick_last; v->tick_same = tick_same; v->frame = frame; v->dirty = dirty + 1;
  v->ser_h = v->ser_t = 0;
  return 0;
}
