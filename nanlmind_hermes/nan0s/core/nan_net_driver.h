#ifndef NAN_NET_DRIVER_H
#define NAN_NET_DRIVER_H
#include "nan16.h"
#include <stddef.h>
#include <stdint.h>

/* Transport-only boundary. The NaN16 guest validates magic, version, state,
 * sequence, CRC and route policy. This C driver only moves bounded bytes. */
#define NAN_NET_HEADER_BYTES 32u
#define NAN_NET_TRAILER_BYTES 4u
#define NAN_NET_MAX_PAYLOAD 1048576u
#define NAN_NET_MAX_FRAME (NAN_NET_HEADER_BYTES + NAN_NET_MAX_PAYLOAD + NAN_NET_TRAILER_BYTES)

int nan_net_send_frame_fd(int fd, const uint8_t *frame, size_t length);
int nan_net_recv_frame_fd(int fd, uint8_t *frame, size_t capacity, size_t *length);
int nan_net_stage_frame(nan_vm *vm, uint32_t byte_address, const uint8_t *frame, size_t length);
int nan_net_extract_frame(const nan_vm *vm, uint32_t byte_address, uint8_t *frame, size_t capacity, size_t length);
#endif
