# Opaque 255-head NaN16 bank transport in the 8 MiB HERMES chart

The native head bank from NaN0S #19 has 8 header words plus 255 x 266 words (two 133-word generations per head), exactly **67,838 NaN16 words / 542,704 bytes**. Each generation contains 128 class-bearing NaN16 words. This is **not** the entire 8 MiB universe and not the existing 256 KiB HERMES `OFF_HEAD_SCRATCH` leaf projection.

## Frozen chart placement

| Region | Byte range | Ownership |
| --- | --- | --- |
| Temporal register | `OFF_EXTENSION` onward, 4096 bytes | HERMES |
| Aggregate MTH head register | `HEAD_REGISTER_OFFSET` onward, 4096 bytes | HERMES |
| Native head bank window | `0x640000..0x6C47F0` (exclusive) | native NaN16 guest bytes |
| Remaining extension | untouched | reserved |

The native bank's chart-relative guest word address is `0xC8000` (=819,200). That is a **mapping contract**, not proof the VM has mounted this region. Real runtime attachment requires the VM's guest memory to map this chart slice, then call `nhr_attach` against the native `nh_open` ABI. Never interpret HERMES's `NLMLEAF1` summaries as equivalent to a native `NH_SLOT_MAGIC` bank.

`nanlmind_hermes.head_bank_transport` exports exact raw bytes, stages a fully sized incoming raw bank behind four byte/root/generation leases, and exposes an inspection digest and layout. It does not parse the payload, select a generation, increment a counter or mark `native_guest_attached=true`. The native guest must check its own header, NaN class, checksum and slot commit marker. A SHA digest by itself does not authenticate native execution. The HERMES chart's full-page dual-slot SHA-256 checkpoint now also preserves these opaque bytes.

**Still required before a live teaching claim:** a VM attachment/mount, an authenticated native execution receipt tied to the HERMES chart root and bank generation, an atomic update/restore handoff with NANNET lease/preflight, and a held-out before/after language-learning test. The host is only the byte transport; no implicit math authorization, native execution or global merge is conferred.
