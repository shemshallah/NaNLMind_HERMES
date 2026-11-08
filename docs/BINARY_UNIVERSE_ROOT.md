# HERMES binary universe root (boot-time mount descriptor)

The physical **8 MiB NaN16 chart remains one shared universe**. During
`LocalRunner.open()`, HERMES either restores the SHA-verified previous
`N16SNAP1` overlay or instantiates `Chart8MiB.genesis()`, and then attaches
the deterministic `N16ROOT1` binary root in a **4 KiB unallocated chart window**.
No second model, chart, or generic SQLite database is created by this step.

| Chart region | Offset | Size | Owner |
|---|---|---|---|
| Temporal L0-L9/rails | `0x63BA10` | 4 KiB | HERMES |
| Aggregate head scratch | `0x63CA10` | 4 KiB | HERMES |
| New binary root header | `0x63DA10` | 4 KiB | HERMES mount |
| Native 255-head scratch bank | `0x640000` | 542,704 B | NaN0S native guest |

The root includes magic/version, canonical chart parent digest, fixed-point
count (256), MP count (32,640), canonical cell count (32,896), 10 layer
coordinates (L0–L9), chart and native bank geometry and a SHA-256
integrity field. It names `nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0`.
It is **not a replacement** for native `.nmg` or NaNROUTER's
versioned MC0DB / immutable BS0DB leaf store.

### One logical addressing tree, distinct physical authorities

```text
N16ROOT1 (inside the single 8 MiB chart)
  -> HW0:0..255  (256 fixed points, HW0:0 reserved NANNET0)
  -> HW0:MP0..32639 (each MP -> two directional 16-bit addresses)
    -> layer L0..L9 and typed immutable leaf versions (MC0DB/BS0DB)
    -> per-head scratch/candidate projections
  -> native NaN16 chart + dual rails + class-bearing head bank
```

The topology is generated in NaNROUTER and **only becomes attached after a
separate authenticated router handshake**. In this PR the binary descriptor
truthfully returns `router_topology_attached=false`. This prevents the
falsely reassuring condition where a format declaration is treated as an
authenticated live mapping. HERMES status and chart inspection now expose
root geometry but not raw native words.

Corrupt, partially initialized, or wrong-parent root blocks fail closed.
The header is initialized only when its entire 4 KiB reserved region is
zero, and all bytes persist via the existing dual-slot SHA checkpoint.

### Why not put the SQLite database inside the chart?

SQLite WAL, concurrent reads and page recovery require its own transactions.
Wrapping a live SQLite database as writable bytes inside a NaN16 chart makes
the two independent commit protocols unsafe. The **one-universe property**
is a logical mount identity, not one filesystem inode. An offline portable
single-file `.n16pack` export can later contain a consistent chart snapshot,
a SQLite online-backup image, content-addressed leaf blocks and a manifest;
it must be assembled from quiesced, verified generations and never edited in
place as a running SQLite file.

L0–L9 is an index of conceptual granularity, not ten copies of the same data.
Iterative mathematical experiments should create versioned typed leaf
relations with source evidence, negative controls and measured utility before
independent math promotion. No arbitrary HERMES rewrite can grant V3 proof.
