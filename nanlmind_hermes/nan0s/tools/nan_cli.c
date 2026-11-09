/* nan_cli - run a NaN16 image headless. See docs/DESIGN.md. */
#include "../core/nan16.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static uint8_t *slurp(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); if (!f) return NULL;
  fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc((size_t)l + 1); if (!b || fread(b, 1, (size_t)l, f) != (size_t)l) { fclose(f); return NULL; }
  fclose(f); b[l] = 0; *n = (size_t)l; return b;
}
static int keycode(const char *t) {
  static const struct { const char *n; int c; } k[] = {{"ESC",27},{"ENTER",13},{"TAB",9},{"BS",8},{"SPACE",32},{"COMMA",','},
    {"UP",NANK_UP},{"DOWN",NANK_DOWN},{"LEFT",NANK_LEFT},{"RIGHT",NANK_RIGHT},{"HOME",NANK_HOME},{"END",NANK_END},
    {"PGUP",NANK_PGUP},{"PGDN",NANK_PGDN},{"INS",NANK_INS},{"DEL",NANK_DEL}};
  for (size_t i = 0; i < sizeof k / sizeof k[0]; i++) if (!strcmp(t, k[i].n)) return k[i].c;
  return t[1] ? -1 : (unsigned char)t[0];
}
static void drain(nan_vm *v, int echo) {
  char b[512]; size_t n; while ((n = nan_serial_read(v, b, sizeof b))) if (echo) fwrite(b, 1, n, stdout);
}
static uint8_t *build_training_guest(size_t *len) {
  const char *cmd = access("tools/nmk.awk", R_OK) == 0
      ? "awk -f tools/nmk.awk -v CMD=emit src/training/n16_train.nas"
      : "awk -f ../../tools/nmk.awk -v CMD=emit ../../src/training/n16_train.nas";
  FILE *p = popen(cmd, "r");
  if (!p) return NULL;
  size_t cap = 65536, n = 0;
  uint8_t *b = (uint8_t *)malloc(cap);
  if (!b) { pclose(p); return NULL; }
  for (;;) {
    if (n == cap) {
      if (cap >= 4u * 1024u * 1024u) { free(b); pclose(p); return NULL; }
      cap *= 2;
      uint8_t *next = (uint8_t *)realloc(b, cap);
      if (!next) { free(b); pclose(p); return NULL; }
      b = next;
    }
    size_t got = fread(b + n, 1, cap - n, p);
    n += got;
    if (got == 0) break;
  }
  int status = pclose(p);
  if (status != 0 || n == 0) { free(b); return NULL; }
  *len = n;
  return b;
}
int main(int argc, char **argv) {
  const char *vga = NULL, *img = NULL, *keys = NULL, *ppm = NULL, *disk = NULL, *peek = NULL, *corpus = NULL, *candidate_out = NULL; uint64_t steps = 50000000ull, per_ms = 1000;
  int train_mode = 0;
  int dump_text = 0, serial = 0, trim = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--img") && i + 1 < argc) img = argv[++i];
    else if (!strcmp(argv[i], "--nannet-train")) { train_mode = 1; if (i + 1 < argc && argv[i + 1][0] != '-') corpus = argv[++i]; }
    else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) { train_mode = 1; corpus = argv[++i]; }
    else if (!strcmp(argv[i], "--candidate-out") && i + 1 < argc) candidate_out = argv[++i];
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = strtoull(argv[++i], 0, 0);
    else if (!strcmp(argv[i], "--keys") && i + 1 < argc) keys = argv[++i];
    else if (!strcmp(argv[i], "--dump-ppm") && i + 1 < argc) ppm = argv[++i];
    else if (!strcmp(argv[i], "--dump-vga") && i + 1 < argc) vga = argv[++i];
    else if (!strcmp(argv[i], "--disk") && i + 1 < argc) disk = argv[++i];
    else if (!strcmp(argv[i], "--peek") && i + 1 < argc) peek = argv[++i];
    else if (!strcmp(argv[i], "--clock") && i + 1 < argc) { const char *c = argv[++i]; if (!strncmp(c, "steps:", 6)) per_ms = strtoull(c + 6, 0, 0); }
    else if (!strcmp(argv[i], "--dump-text")) dump_text = 1;
    else if (!strcmp(argv[i], "--no-trim")) trim = 0;
    else if (!strcmp(argv[i], "--serial")) serial = 1;
    else { fprintf(stderr, "usage: nan_cli [--img f] [--nannet-train [corpus.txt]] [--candidate-out state.nan1] [--steps N] [--keys \"1,ESC,~,@5000000\"] [--serial] [--dump-text] [--dump-ppm f] [--peek ADDR[:LEN]] [--disk f] [--clock steps:N]\n"); return 2; }
  }
  if (!img && !train_mode) { fprintf(stderr, "need --img or --nannet-train\n"); return 2; }
  size_t len = 0;
  uint8_t *data = (!img && train_mode) ? build_training_guest(&len) : slurp(img, &len);
  if (!data) { perror(img ? img : "NaN16 training guest"); return 1; }
  nan_vm *v = nan_create(); if (!v) { fprintf(stderr, "out of memory\n"); return 1; }
  const char *kind; if (nan_load_auto(v, data, len, &kind)) { fprintf(stderr, "cannot load %s (text error line %d)\n", img ? img : "<built trainer>", nan_load_error(v)); return 1; }

  if (train_mode && corpus) {
    size_t corpus_len = 0;
    uint8_t *raw = slurp(corpus, &corpus_len);
    if (!raw) { perror(corpus); return 1; }
    if (corpus_len > 1048576u || memchr(raw, 0, corpus_len)) {
      fprintf(stderr, "corpus must be at most 1 MiB and contain no NUL bytes\n");
      free(raw); return 2;
    }
    static const uint8_t header_lf[] = "N16TXT/1\n";
    static const uint8_t header_crlf[] = "N16TXT/1\r\n";
    int has_header_lf = corpus_len >= sizeof(header_lf) - 1 &&
                        memcmp(raw, header_lf, sizeof(header_lf) - 1) == 0;
    int has_header_crlf = corpus_len >= sizeof(header_crlf) - 1 &&
                          memcmp(raw, header_crlf, sizeof(header_crlf) - 1) == 0;
    size_t header_len = sizeof(header_lf) - 1;
    size_t crlf_len = sizeof(header_crlf) - 1;
    size_t body_offset = has_header_crlf ? crlf_len : (has_header_lf ? header_len : 0);
    size_t body_len = corpus_len - body_offset;
    size_t framed_len = header_len + body_len;
    uint8_t *framed = (uint8_t *)malloc(framed_len + 1);
    if (!framed) { free(raw); return 1; }
    memcpy(framed, header_lf, header_len);
    memcpy(framed + header_len, raw + body_offset, body_len);
    framed[framed_len] = 0;
    int wrc = nan_write_bytes(v, 327680u, framed, framed_len + 1);
    free(framed); free(raw);
    if (wrc) { fprintf(stderr, "corpus does not fit the NaN16 guest buffer\n"); return 1; }
  } else if (train_mode) {
    static const uint8_t sample[] =
      "N16TXT/1\n"
      "NaN16 owns the corpus codec.\n"
      "Runner and Android share one trainer.\n"
      "The intake stages updates before merge.\n";
    if (nan_write_bytes(v, 327680u, sample, sizeof(sample))) {
      fprintf(stderr, "built-in sample does not fit the NaN16 guest buffer\n");
      return 1;
    }
  }
  if (train_mode) serial = 1;
  if (disk) { int fd = open(disk, O_RDWR); struct stat st; if (fd < 0 || fstat(fd, &st)) { perror(disk); return 1; } nan_attach_disk_fd(v, fd, (size_t)st.st_size); }
  char *kcopy = keys ? strdup(keys) : NULL; char *save = NULL; char *tok = kcopy ? strtok_r(kcopy, ",", &save) : NULL;
  uint64_t pause_until = 0; int rc = NAN_BUDGET;
  while (nan_steps(v) < steps) {
    nan_set_time_ms(v, (uint32_t)(nan_steps(v) / per_ms));
    uint64_t left = steps - nan_steps(v); uint32_t slice = left > 100000 ? 100000 : (uint32_t)left;
    rc = nan_run(v, slice); drain(v, serial);
    if (rc == NAN_HALTED) break;
    int can_key = nan_steps(v) >= pause_until || rc == NAN_WAIT_KEY;
    if (can_key && tok) {
      if (tok[0] == '@') { pause_until = nan_steps(v) + strtoull(tok + 1, 0, 0); tok = strtok_r(NULL, ",", &save); continue; }
      int c = keycode(tok); if (c < 0) { fprintf(stderr, "bad key '%s'\n", tok); return 2; }
      nan_key_char(v, c); tok = strtok_r(NULL, ",", &save); continue;
    }
    if (rc == NAN_WAIT_KEY && !tok) break;
  }
  drain(v, serial);
  if (serial && isatty(1)) fputc('\n', stdout);
  if (dump_text) { static char b[16384]; nan_text_utf8(v, b, sizeof b, trim); fputs(b, stdout); }
  if (peek) { /* --peek ADDR[:LEN][,ADDR[:LEN]...]: hex-dump guest words (gate/debug) */
    char *spec = strdup(peek), *sv = NULL;
    for (char *tok = strtok_r(spec, ",", &sv); tok; tok = strtok_r(NULL, ",", &sv)) {
      unsigned long a0 = strtoul(tok, NULL, 0), nw = 1; const char *c = strchr(tok, ':');
      if (c) nw = strtoul(c + 1, NULL, 0);
      for (unsigned long i = 0; i < nw; i++) printf("%08X\n", (uint32_t)(nan_word(v, (uint32_t)(a0 + i)) >> 16));
    }
    free(spec);
  }
  if (vga) { FILE *f = fopen(vga, "wb"); if (f) { for (int i = 0; i < 2000; i++) { uint16_t c = nan_text_cell(v, i); fputc(c & 255, f); fputc(c >> 8, f); } fclose(f); } }
  if (ppm) {
    int w, h; nan_display_size(v, &w, &h);
    if (nan_display_mode(v) == 1) {
      uint32_t *px = (uint32_t *)malloc((size_t)w * h * 4); FILE *f = fopen(ppm, "wb");
      if (px && f && nan_render_gfx(v, px, (size_t)w * h) == 0) {
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (int i = 0; i < w * h; i++) { uint8_t c[3] = {(uint8_t)(px[i] >> 16), (uint8_t)(px[i] >> 8), (uint8_t)px[i]}; fwrite(c, 1, 3, f); }
      }
      if (f) fclose(f); free(px);
    } else fprintf(stderr, "(text mode: no PPM written)\n");
  }
  fprintf(stderr, "[nan_cli] %s: %llu steps, rc=%d, pc=%u, trap='%s'\n", kind, (unsigned long long)nan_steps(v), rc, nan_pc(v), nan_trap_text(v));
  int train_ok = !train_mode || (rc == NAN_HALTED && (uint32_t)(nan_word(v, 8449u) >> 16) != 0);
  if (candidate_out) {
    if (!train_mode || !train_ok) {
      fprintf(stderr, "candidate export requires a successful training run\n");
      train_ok = 0;
    } else {
      size_t state_len = nan_state_size(v);
      uint8_t *state = (uint8_t *)malloc(state_len ? state_len : 1);
      if (!state || !state_len || nan_save_state(v, state, state_len) != state_len) {
        fprintf(stderr, "could not serialize candidate VM state\n");
        train_ok = 0;
      } else {
        FILE *out = fopen(candidate_out, "wb");
        int write_ok = 0;
        if (out) {
          write_ok = fwrite(state, 1, state_len, out) == state_len;
          if (fclose(out) != 0) write_ok = 0;
        }
        if (!write_ok) {
          fprintf(stderr, "could not write candidate checkpoint\n");
          train_ok = 0;
        }
      }
      free(state);
    }
  }
  nan_destroy(v); return train_ok ? 0 : 1;
}
