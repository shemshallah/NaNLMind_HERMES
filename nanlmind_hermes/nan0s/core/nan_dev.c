/* nan_dev.c - ports, DAC, PIT/speaker, PCI mech#1, ATA PIO, COM1, keyboard (IRQ1 path). */
#include "nan16_int.h"
#ifndef _WIN32
#include <unistd.h>
#endif

static const uint8_t ktab[384] = {
#include "ktab.inc"
};

/* ---------- serial ---------- */
void nan_ser_putc(nan_vm *v, uint8_t c) {
  size_t nt = (v->ser_t + 1) % NAN_SER_CAP;
  if (nt == v->ser_h) return;
  v->ser[v->ser_t] = (char)c; v->ser_t = nt;
}
size_t nan_serial_read(nan_vm *v, char *buf, size_t max) {
  size_t n = 0;
  while (n < max && v->ser_h != v->ser_t) { buf[n++] = v->ser[v->ser_h]; v->ser_h = (v->ser_h + 1) % NAN_SER_CAP; }
  return n;
}

/* ---------- palettes ---------- */
void nan_vga_default(uint8_t t[256][3]) {
  static const uint8_t cga[16][3] = {{0,0,0},{0,0,42},{0,42,0},{0,42,42},{42,0,0},{42,0,42},{42,21,0},{42,42,42},
    {21,21,21},{21,21,63},{21,63,21},{21,63,63},{63,21,21},{63,21,63},{63,63,21},{63,63,63}};
  static const uint8_t grey[16] = {0x00,0x05,0x08,0x0B,0x0E,0x11,0x14,0x18,0x1C,0x20,0x24,0x28,0x2D,0x32,0x38,0x3F};
  static const uint8_t sets[9][2] = {{0x3F,0},{0x3F,0x1F},{0x3F,0x2D},{0x1C,0},{0x1C,0x0E},{0x1C,0x14},{0x10,0},{0x10,0x08},{0x10,0x0B}};
  static const uint8_t walk[24][3] = {{0,0,4},{1,0,4},{2,0,4},{3,0,4},{4,0,4},{4,0,3},{4,0,2},{4,0,1},{4,0,0},{4,1,0},{4,2,0},{4,3,0},
    {4,4,0},{3,4,0},{2,4,0},{1,4,0},{0,4,0},{0,4,1},{0,4,2},{0,4,3},{0,4,4},{0,3,4},{0,2,4},{0,1,4}};
  memset(t, 0, 256 * 3);
  for (int i = 0; i < 16; i++) { memcpy(t[i], cga[i], 3); t[16 + i][0] = t[16 + i][1] = t[16 + i][2] = grey[i]; }
  for (int s = 0; s < 9; s++) {
    int hi = sets[s][0], lo = sets[s][1], dd = hi - lo;
    int lv[5] = {lo, lo + (2 * dd + 3) / 8, lo + dd / 2, lo + (6 * dd + 3) / 8, hi};
    for (int i = 0; i < 24; i++) for (int c = 0; c < 3; c++) t[32 + s * 24 + i][c] = (uint8_t)lv[walk[i][c]];
  }
}
static void ega_table(uint8_t t[64][3]) { /* index bits r g b R G B (msb..lsb) */
  for (int i = 0; i < 64; i++) {
    t[i][0] = (uint8_t)(((i >> 2) & 1 ? 0x2A : 0) | ((i >> 5) & 1 ? 0x15 : 0));
    t[i][1] = (uint8_t)(((i >> 1) & 1 ? 0x2A : 0) | ((i >> 4) & 1 ? 0x15 : 0));
    t[i][2] = (uint8_t)((i & 1 ? 0x2A : 0) | ((i >> 3) & 1 ? 0x15 : 0));
  }
}
void nan_dev_reset(nan_vm *v) {
  memset(v->dac, 0, sizeof v->dac); ega_table(v->dac); /* qemu/SeaBIOS mode-3 state: 0..63 EGA table, 64..255 black (probed with a boot sector) */
  v->dac_loaded = 0; v->dac_wi = v->dac_wp = v->dac_ri = v->dac_rp = 0;
  v->rhead = v->rtail = 0; v->rawkb = 0; v->shift = v->caps = v->e0 = 0;
  v->pit_div = 0; v->pit_phase = 0; v->p61 = 0; v->pit_ctl = 0; v->pci_addr = 0; v->retrace = 0;
  nan_ata *a = &v->ata;  /* keep the attached backend across reset */
  a->err = a->count = a->drive = a->cmd = 0; memset(a->lba, 0, 3); a->status = 0x50; a->pos = 0; a->mode = 0; a->remain = 0;
}
/* qemu c6_to_8: low bit replicated into the two low bits (byte-exact with the qemu oracle). */
static uint32_t to8(uint8_t x6) { uint32_t v = x6 & 63u, b = v & 1u; return (v << 2) | (b << 1) | b; }
void nan_dac_rgb(const nan_vm *v, int idx, uint8_t rgb[3]) {
  const uint8_t (*t)[3] = v->dac;
  for (int c = 0; c < 3; c++) rgb[c] = (uint8_t)to8(t[idx & 255][c]);
}
static const uint8_t *pal_for_gfx(const nan_vm *v, uint8_t def[256][3]) {
  (void)def; return &v->dac[0][0];
}
int nan_render_gfx(const nan_vm *v, uint32_t *argb, size_t cap) {
  if (v->vmode != 1) return -1;
  size_t n = (size_t)v->fbw * v->fbh;
  if (n > cap || n > NAN_VRAM) return -1;
  uint8_t def[256][3]; const uint8_t *p = pal_for_gfx(v, def); uint32_t lut[256];
  for (int i = 0; i < 256; i++) lut[i] = 0xFF000000u | (to8(p[i*3]) << 16) | (to8(p[i*3+1]) << 8) | to8(p[i*3+2]);
  for (size_t i = 0; i < n; i++) argb[i] = lut[v->vram[i]];
  return 0;
}
uint32_t nan_text_color_vm(const nan_vm *v, int c) {
  c &= 15; int idx = c < 8 ? (c == 6 ? 0x14 : c) : 0x38 + (c - 8);
  return 0xFF000000u | (to8(v->dac[idx][0]) << 16) | (to8(v->dac[idx][1]) << 8) | to8(v->dac[idx][2]);
}
uint32_t nan_text_color(int c) {
  uint8_t t[64][3]; ega_table(t); c &= 15; int idx = c < 8 ? (c == 6 ? 0x14 : c) : 0x38 + (c - 8);
  return 0xFF000000u | (to8(t[idx][0]) << 16) | (to8(t[idx][1]) << 8) | to8(t[idx][2]);
}

/* ---------- keyboard ring ---------- */
int nan_ring_pop(nan_vm *v) {
  if (v->rhead == v->rtail) return -1;
  int c = v->ring[v->rhead]; v->rhead = (v->rhead + 1) & 15; return c;
}
void nan_ring_push(nan_vm *v, uint8_t c) {
  int nt = (v->rtail + 1) & 15;
  if (nt == v->rhead) return;
  v->ring[v->rtail] = c; v->rtail = nt;
}
void nan_scancode(nan_vm *v, uint8_t sc) {            /* IRQ1 handler, kan16.xa irq1 */
  if (v->rawkb) { nan_ring_push(v, sc); return; }
  if (sc == 0xE0) { v->e0 = 1; return; }
  int e0 = v->e0; v->e0 = 0; int push = 0;
  if (sc & 0x80) {
    if (!e0 && ((sc & 0x7F) == 0x2A || (sc & 0x7F) == 0x36)) v->shift = 0;
    return;
  }
  if (e0) push = ktab[256 + sc];
  else if (sc == 0x2A || sc == 0x36) v->shift = 1;
  else if (sc == 0x3A) v->caps ^= 1;
  else { int s = v->shift; if ((unsigned)(ktab[sc] - 97) < 26) s ^= v->caps; push = ktab[(s << 7) + sc]; }
  if (push) nan_ring_push(v, (uint8_t)push);
}
void nan_key_event(nan_vm *v, int sc, int down) {
  if (sc >= 0x100) nan_scancode(v, 0xE0);
  nan_scancode(v, (uint8_t)((sc & 0x7F) | (down ? 0 : 0x80)));
}
void nan_key_char(nan_vm *v, int c) {
  if (!v->rawkb) { nan_ring_push(v, (uint8_t)c); return; }
  if (c == 10) c = 13;
  for (int sc = 1; sc < 128; sc++) if (ktab[sc] == c) { nan_scancode(v, (uint8_t)sc); nan_scancode(v, (uint8_t)(sc | 0x80)); return; }
  for (int sc = 1; sc < 128; sc++) if (ktab[128 + sc] == c) {
    nan_scancode(v, 0x2A); nan_scancode(v, (uint8_t)sc); nan_scancode(v, (uint8_t)(sc | 0x80)); nan_scancode(v, 0xAA); return; }
  for (int sc = 1; sc < 128; sc++) if (ktab[256 + sc] == c) {
    nan_scancode(v, 0xE0); nan_scancode(v, (uint8_t)sc); nan_scancode(v, 0xE0); nan_scancode(v, (uint8_t)(sc | 0x80)); return; }
}
int nan_raw_keyboard(const nan_vm *v) { return v->rawkb; }
uint32_t nan_speaker_hz(const nan_vm *v) { return ((v->p61 & 3) == 3 && v->pit_div) ? 1193182u / v->pit_div : 0; }

/* ---------- ATA PIO primary master ---------- */
void nan_attach_disk_mem(nan_vm *v, uint8_t *d, size_t len) { v->ata.mem = d; v->ata.size = len; v->ata.fd = -1; v->ata.status = 0x50; }
void nan_attach_disk_fd(nan_vm *v, int fd, size_t len) { v->ata.mem = NULL; v->ata.size = len; v->ata.fd = fd; v->ata.status = 0x50; }
void nan_detach_disk(nan_vm *v) { v->ata.mem = NULL; v->ata.size = 0; v->ata.fd = -1; }
static int ata_present(const nan_vm *v) { return v->ata.mem != NULL || (v->ata.fd >= 0 && v->ata.size); }
static void ata_rd(nan_vm *v, uint32_t lba) {
  nan_ata *a = &v->ata; uint8_t tmp[512]; memset(tmp, 0, 512); size_t off = (size_t)lba * 512;
  if (off + 512 <= a->size) {
    if (a->mem) memcpy(tmp, a->mem + off, 512);
#ifndef _WIN32
    else if (pread(a->fd, tmp, 512, (off_t)off) < 0) memset(tmp, 0, 512);
#endif
  }
  for (int i = 0; i < 256; i++) a->buf[i] = (uint16_t)(tmp[2 * i] | (tmp[2 * i + 1] << 8));
  a->pos = 0;
}
static void ata_wr(nan_vm *v, uint32_t lba) {
  nan_ata *a = &v->ata; uint8_t tmp[512]; size_t off = (size_t)lba * 512;
  for (int i = 0; i < 256; i++) { tmp[2 * i] = (uint8_t)a->buf[i]; tmp[2 * i + 1] = (uint8_t)(a->buf[i] >> 8); }
  if (off + 512 > a->size) return;
  if (a->mem) memcpy(a->mem + off, tmp, 512);
#ifndef _WIN32
  else if (pwrite(a->fd, tmp, 512, (off_t)off) < 0) { /* ignore */ }
#endif
}
static uint32_t ata_lba(const nan_ata *a) { return ((uint32_t)(a->drive & 15) << 24) | ((uint32_t)a->lba[2] << 16) | ((uint32_t)a->lba[1] << 8) | a->lba[0]; }
static void ata_cmd(nan_vm *v, uint8_t c) {
  nan_ata *a = &v->ata; a->cmd = c;
  if (!ata_present(v)) return;
  a->err = 0;
  if (c == 0xEC) {
    memset(a->buf, 0, sizeof a->buf); uint32_t sec = (uint32_t)(a->size / 512);
    a->buf[0] = 0x0040; a->buf[49] = 0x0200; a->buf[60] = (uint16_t)sec; a->buf[61] = (uint16_t)(sec >> 16); a->buf[83] = 0;
    const char *m = "NaN16 ATA DISK                          ";
    for (int i = 0; i < 20; i++) a->buf[27 + i] = (uint16_t)((m[2 * i] << 8) | m[2 * i + 1]);
    a->pos = 0; a->mode = 1; a->remain = 1; a->status = 0x58;
  } else if (c == 0x20) {
    a->remain = a->count ? a->count : 256; a->lba_cur = ata_lba(a); ata_rd(v, a->lba_cur); a->mode = 1; a->status = 0x58;
  } else if (c == 0x30) {
    a->remain = a->count ? a->count : 256; a->lba_cur = ata_lba(a); a->pos = 0; a->mode = 2; a->status = 0x58;
  } else if (c == 0xE7) a->status = 0x50;
  else { a->err = 4; a->status = 0x51; }
}

/* ---------- PCI mech #1 ---------- */
static uint32_t pci_cfg(uint32_t addr) {
  if (!(addr & 0x80000000u) || ((addr >> 16) & 0xFF) || ((addr >> 8) & 7)) return 0xFFFFFFFFu;
  uint32_t dev = (addr >> 11) & 31, reg = addr & 0xFC;
  static const uint32_t id[3] = {0x12378086u, 0x70008086u, 0x11111234u}, cl[3] = {0x06000000u, 0x06010000u, 0x03000000u};
  if (dev > 2) return 0xFFFFFFFFu;
  if (reg == 0) return id[dev]; if (reg == 8) return cl[dev]; if (reg == 0x10 && dev == 2) return 0xFD000008u;
  return 0;
}

static uint32_t port_in(nan_vm *v, uint32_t port, int size) {
  nan_ata *a = &v->ata;
  switch (port) {
  case 0x3C9: { uint32_t x = v->dac[v->dac_ri][v->dac_rp]; if (++v->dac_rp == 3) { v->dac_rp = 0; v->dac_ri = (v->dac_ri + 1) & 255; } return x; }
  case 0x3DA: v->retrace ^= 8; return v->retrace;
  case 0x3FD: return 0x60;
  case 0x61: return v->p61;
  case 0xCFC: case 0xCFD: case 0xCFE: case 0xCFF: {
    uint32_t c = pci_cfg(v->pci_addr); if (c == 0xFFFFFFFFu) return size == 2 ? 0xFFFF : 0xFF;
    return (c >> ((port & 3) * 8)) & (size == 2 ? 0xFFFFu : 0xFFu); }
  case 0x1F0: if (size == 2 && ata_present(v) && a->mode == 1) {
      uint32_t x = a->buf[a->pos++];
      if (a->pos >= 256) { a->pos = 0;
        if (--a->remain > 0 && a->cmd == 0x20) { a->lba_cur++; ata_rd(v, a->lba_cur); } else { a->mode = 0; a->status = 0x50; } }
      return x; }
    return size == 2 ? 0xFFFF : 0xFF;
  case 0x1F1: return ata_present(v) ? a->err : 0xFF;
  case 0x1F2: return ata_present(v) ? a->count : 0xFF;
  case 0x1F3: case 0x1F4: case 0x1F5: return ata_present(v) ? a->lba[port - 0x1F3] : 0xFF;
  case 0x1F6: return ata_present(v) ? a->drive : 0xFF;
  case 0x1F7: case 0x3F6: return ata_present(v) ? a->status : 0xFF;
  default: return size == 2 ? 0xFFFF : size == 4 ? 0xFFFFFFFFu : 0xFF;
  }
}
static void port_out(nan_vm *v, uint32_t port, uint32_t val, int size) {
  nan_ata *a = &v->ata;
  switch (port) {
  case 0x3C8: v->dac_wi = val & 255; v->dac_wp = 0; break;
  case 0x3C7: v->dac_ri = val & 255; v->dac_rp = 0; break;
  case 0x3C9: v->dac[v->dac_wi][v->dac_wp] = val & 63; v->dac_loaded = 1; v->dirty++;
    if (++v->dac_wp == 3) { v->dac_wp = 0; v->dac_wi = (v->dac_wi + 1) & 255; } break;
  case 0x43: v->pit_ctl = (uint8_t)val; v->pit_phase = 0; break;
  case 0x42: if (v->pit_phase == 0) v->pit_div = (v->pit_div & 0xFF00) | (val & 255); else v->pit_div = (v->pit_div & 0xFF) | ((val & 255) << 8);
    v->pit_phase ^= 1; break;
  case 0x61: v->p61 = (uint8_t)val; break;
  case 0x3F8: nan_ser_putc(v, (uint8_t)val); break;
  case 0xCF8: if (size == 4) v->pci_addr = val; break;
  case 0x1F0: if (size == 2 && ata_present(v) && a->mode == 2) {
      a->buf[a->pos++] = (uint16_t)val;
      if (a->pos >= 256) { ata_wr(v, a->lba_cur); a->pos = 0; a->lba_cur++;
        if (--a->remain <= 0) { a->mode = 0; a->status = 0x50; } } }
    break;
  case 0x1F1: a->err = (uint8_t)val; break;
  case 0x1F2: a->count = (uint8_t)val; break;
  case 0x1F3: case 0x1F4: case 0x1F5: a->lba[port - 0x1F3] = (uint8_t)val; break;
  case 0x1F6: a->drive = (uint8_t)val; break;
  case 0x1F7: ata_cmd(v, (uint8_t)val); break;
  default: break;
  }
}

uint32_t nan_port_in(nan_vm *v, uint32_t port, int size) {
  uint32_t x = port_in(v, port, size);
  if (v->portlog) v->portlog(v->portlog_ctx, 0, port, x, size);
  return x;
}
void nan_port_out(nan_vm *v, uint32_t port, uint32_t val, int size) {
  if (v->portlog) v->portlog(v->portlog_ctx, 1, port, val, size);
  port_out(v, port, val, size);
}
void nan_set_port_log(nan_vm *v, nan_port_fn cb, void *ctx) { v->portlog = cb; v->portlog_ctx = ctx; }
const char *nan_strerror(int rc) {
  switch (rc) { case NAN_BUDGET: return "budget"; case NAN_WAIT_KEY: return "wait-key"; case NAN_VSYNC: return "vsync";
                case NAN_HALTED: return "halted"; case NAN_TIME_WAIT: return "time-wait"; default: return "?"; }
}
