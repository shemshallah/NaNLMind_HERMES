#include "nan_net_driver.h"
#include <errno.h>
#include <sys/socket.h>

static int read_exact(int fd, uint8_t *p, size_t n) {
    while (n) {
        ssize_t got = recv(fd, p, n, 0);
        if (got == 0) return -1;
        if (got < 0) { if (errno == EINTR) continue; return -1; }
        p += (size_t)got; n -= (size_t)got;
    }
    return 0;
}

int nan_net_send_frame_fd(int fd, const uint8_t *frame, size_t length) {
    if (!frame || length < NAN_NET_HEADER_BYTES + NAN_NET_TRAILER_BYTES ||
        length > NAN_NET_MAX_FRAME) return -1;
    size_t sent = 0;
    while (sent < length) {
        ssize_t n = send(fd, frame + sent, length - sent,
#ifdef MSG_NOSIGNAL
             MSG_NOSIGNAL
#else
             0
#endif
        );
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

int nan_net_recv_frame_fd(int fd, uint8_t *frame, size_t capacity, size_t *length) {
    if (!frame || !length || capacity < NAN_NET_HEADER_BYTES + NAN_NET_TRAILER_BYTES) return -1;
    if (read_exact(fd, frame, NAN_NET_HEADER_BYTES)) return -1;
    /* Length is read only to delimit the stream and bound allocation. NaN16
       independently validates the frame and all protocol semantics. */
    uint32_t payload = (uint32_t)frame[28] |
        ((uint32_t)frame[29] << 8) | ((uint32_t)frame[30] << 16) |
        ((uint32_t)frame[31] << 24);
    if (payload > NAN_NET_MAX_PAYLOAD) return -1;
    size_t total = NAN_NET_HEADER_BYTES + (size_t)payload + NAN_NET_TRAILER_BYTES;
    if (total > capacity) return -1;
    if (read_exact(fd, frame + NAN_NET_HEADER_BYTES,
                   (size_t)payload + NAN_NET_TRAILER_BYTES)) return -1;
    *length = total;
    return 0;
}

int nan_net_stage_frame(nan_vm *vm, uint32_t byte_address,
                        const uint8_t *frame, size_t length) {
    if (!vm || !frame || length < NAN_NET_HEADER_BYTES + NAN_NET_TRAILER_BYTES ||
        length > NAN_NET_MAX_FRAME) return -1;
    return nan_write_bytes(vm, byte_address, frame, length);
}

int nan_net_extract_frame(const nan_vm *vm, uint32_t byte_address,
                          uint8_t *frame, size_t capacity, size_t length) {
    if (!vm || !frame || length < NAN_NET_HEADER_BYTES + NAN_NET_TRAILER_BYTES ||
        length > NAN_NET_MAX_FRAME || length > capacity) return -1;
    return nan_read_bytes(vm, byte_address, frame, length);
}
