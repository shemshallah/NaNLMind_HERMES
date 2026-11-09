/* nan_load.c - floppy image / raw words / .nan text loaders. */
#include "nan16_int.h"
#include <ctype.h>

static uint64_t rd64le(const uint8_t *p) {
  uint64_t x = 0; for (int i = 7; i >= 0; i--) x = (x << 8) | p[i]; return x;
}
int nan_load_words(nan_vm *v, const uint8_t *w, size_t len) {
  size_t n = len / 8; if (!n || n > NAN_WORDS) return -1;
  nan_reset(v);
  for (size_t i = 0; i < n; i++) v->mem[i] = rd64le(w + i * 8);
  return 0;
}
int nan_load_image(nan_vm *v, const uint8_t *img, size_t len) {
  if (len < 0x2200 + 8) return -1;
  size_t n = (len - 0x2200) / 8; if (n > 65536) n = 65536;
  return nan_load_words(v, img + 0x2200, n * 8);
}
static int hexn(const char *s, int n, uint32_t *out) {
  uint32_t x = 0;
  for (int i = 0; i < n; i++) { if (!isxdigit((unsigned char)s[i])) return 0; x = x * 16 + (uint32_t)(isdigit((unsigned char)s[i]) ? s[i] - '0' : (tolower((unsigned char)s[i]) - 'a' + 10)); }
  *out = x; return 1;
}
int nan_load_error(const nan_vm *v) { return v->load_err; }
int nan_load_text(nan_vm *v, const char *t, size_t len) {
  nan_reset(v); v->load_err = 0; size_t i = 0, idx = 0; int line = 0;
  while (i < len) {
    line++; size_t e = i; while (e < len && t[e] != '\n') e++;
    char b[256]; size_t m = e - i; if (m >= sizeof b) m = sizeof b - 1; memcpy(b, t + i, m); b[m] = 0; i = e + 1;
    char *c = strchr(b, ';'); if (c) *c = 0;
    char *s = b; while (isspace((unsigned char)*s)) s++;
    if (!*s) continue;
    if (!strncmp(s, ".org", 4)) { idx = (size_t)strtoul(s + 4, NULL, 0); continue; }
    uint32_t f1, f2, f3;
    if (strlen(s) < 18 || !hexn(s, 4, &f1) || s[4] != '_' || !hexn(s + 5, 8, &f2) || s[13] != '_' || !hexn(s + 14, 4, &f3) || idx >= NAN_WORDS) {
      v->load_err = line; return -1; }
    v->mem[idx++] = ((uint64_t)f1 << 48) | ((uint64_t)f2 << 16) | f3;   /* NEAR | payload | FAR byte-swapped */
  }
  return 0;
}
int nan_load_auto(nan_vm *v, const uint8_t *d, size_t len, const char **kind) {
  const char *k = "";
  int rc = -1;
  if (len >= 16 && !memcmp(d, "NAN1", 4)) { k = "state"; rc = nan_restore_state(v, d, len); }
  else if (len == 1474560) { k = "floppy"; rc = nan_load_image(v, d, len); }
  else {
    int texty = len > 0; for (size_t i = 0; i < len && i < 4096; i++) if (d[i] >= 0x80 || (d[i] < 32 && !isspace(d[i]))) { texty = 0; break; }
    if (texty && (rc = nan_load_text(v, (const char *)d, len)) == 0) k = "nan-text";
    else if (len % 8 == 0 && len >= 8) {
      uint64_t w0 = rd64le(d);
      if (nan_hvalid(nan_near(w0)) && nan_hvalid(nan_far(w0))) { k = "words"; rc = nan_load_words(v, d, len); } }
  }
  if (kind) *kind = k;
  return rc;
}
