/* nan_render.c - text screen -> UTF-8 (CP437). */
#include "nan16_int.h"
static const char *const cp437[] = {
#include "cp437_utf8.inc"
};
typedef char cp437_has_256[(sizeof cp437 / sizeof cp437[0]) == 256 ? 1 : -1];
size_t nan_text_utf8(const nan_vm *v, char *buf, size_t cap, int trim) {
  size_t o = 0;
  for (int y = 0; y < 25; y++) {
    int last = 79;
    if (trim) while (last >= 0 && (v->cells[y * 80 + last] & 0xFF) <= 0x20) last--;
    for (int x = 0; x <= last; x++) {
      const char *s = cp437[v->cells[y * 80 + x] & 0xFF]; size_t l = strlen(s);
      if (o + l + 2 > cap) return o;
      memcpy(buf + o, s, l); o += l;
    }
    if (o + 2 > cap) return o;
    buf[o++] = '\n';
  }
  buf[o] = 0; return o;
}
