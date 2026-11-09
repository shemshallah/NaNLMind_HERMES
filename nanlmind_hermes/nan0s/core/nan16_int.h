/* nan16_int.h - private state of the NaN16 core (not part of the public API). */
#ifndef NAN16_INT_H
#define NAN16_INT_H
#include "nan16.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define NAN_MASK  (NAN_WORDS - 1u)
/* NAN_WORDS must be a power of two. Default keeps the current machine; the
   platform revision raises it toward 2^31 words (16 GB) and then widens the
   index to 64 bits (spec/NANVM_SPEC.md §4.1, §8.1). */
#ifndef NAN_WORDS
#define NAN_WORDS 0x400000u
#endif
#define NAN_VRAM  (16u << 20)
#define NAN_SER_CAP 262144u

static const uint16_t NAN_HEADS[4] = {0x7FF1, 0xFFF1, 0x7FF8, 0xFFF8};
static inline int nan_hvalid(uint32_t h) { return (h & 0x7FF0) == 0x7FF0 && ((h & 15) == 1 || (h & 15) == 8); }
static inline uint32_t nan_hcls(uint32_t h) { return ((h >> 2) & 2) | (h >> 15); }
/* raw word: b0-1 FAR head (BE), b2-5 payload (LE), b6-7 NEAR head (LE); stored as the LE u64 of those bytes */
static inline uint64_t nan_mkword(uint32_t payload, int n) {
  uint32_t far = NAN_HEADS[n & 3], near = NAN_HEADS[(n >> 2) & 3];
  return (uint64_t)(far >> 8) | ((uint64_t)(far & 0xFF) << 8) | ((uint64_t)payload << 16) | ((uint64_t)near << 48);
}
static inline uint32_t nan_far(uint64_t w) { return (uint32_t)(((w & 0xFF) << 8) | ((w >> 8) & 0xFF)); }
static inline uint32_t nan_near(uint64_t w) { return (uint32_t)(w >> 48); }

typedef struct {
  uint8_t *mem; size_t size; int fd;      /* one of mem / fd */
  uint8_t err, count, lba[3], drive, status, cmd;
  uint16_t buf[256]; int pos, mode;       /* mode 0 idle, 1 reading, 2 writing */
  uint32_t lba_cur; int remain;
} nan_ata;

struct nan_vm {
  uint32_t r[16], pc; int k, halted; uint64_t steps; uint32_t ms;
  uint64_t *mem; uint8_t *vram;
  uint16_t cells[2000];
  int vmode; uint32_t fbw, fbh, fbtot;
  uint8_t dac[256][3]; int dac_loaded, dac_wi, dac_wp, dac_ri, dac_rp;
  uint8_t ring[16]; int rhead, rtail, rawkb, shift, caps, e0;
  char trap[24];
  char *ser; size_t ser_h, ser_t;
  uint32_t pit_div; int pit_phase; uint8_t p61, pit_ctl; uint32_t pci_addr; uint8_t retrace;
  nan_ata ata;
  uint32_t dirty, frame, tick_last, tick_same;
  nan_trace_fn trace; void *trace_ctx;
  nan_port_fn portlog; void *portlog_ctx;
  int load_err;
};

void nan_dev_reset(nan_vm *v);
void nan_ser_putc(nan_vm *v, uint8_t c);
uint32_t nan_port_in(nan_vm *v, uint32_t port, int size);
void nan_port_out(nan_vm *v, uint32_t port, uint32_t val, int size);
int  nan_ring_pop(nan_vm *v);               /* -1 when empty */
void nan_ring_push(nan_vm *v, uint8_t c);
void nan_vga_default(uint8_t t[256][3]);
#endif
