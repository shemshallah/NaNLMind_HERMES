# Unified resident universe: one memory-mapped file per local universe

Status: **partly implemented**. The compact local-derived implementation is
`nanlmind_hermes/resident/universe.py` together with `cells.py`, `leaves.py`,
`relations.py`, `resonance.py`, `export.py`, `persistence.py`, `ingest.py`,
`learning.py`, the CLI and benchmark. The full 12-section chart/hot-window
profile below remains a design target; the current file uses combined physical
regions for heads+pairs and relations+hyperedges and exposes a compatibility
section view. See [RESIDENT_IMPLEMENTATION.md](RESIDENT_IMPLEMENTATION.md) for
the actual file map, APIs, tests and measured learning loop. The larger
memory-window, 12-slot and six-runner canonical promotion design in this
document is **not** claimed as implemented.

This contract is deliberately **local-derived**, not canonical. The canonical
NaNLMind universe remains `HW0::BS0`; a HERMES resident file is a bounded local
materialization that can produce candidate evidence. Only the existing BS0
generation barrier may promote canonical state, and that barrier is absent in
this repository (see [RESIDENT_INTAKE.md](RESIDENT_INTAKE.md)).

## 1. Scope, identity and file location

- **One live file per local universe.** A local universe is identified by
  `(installation_id, universe_id, state_dir)`. It has exactly one live
  `universe.nlmres` file and one writer lock. It is not a segment of a global
  file, and two universe IDs must never share a file.
- **Default path:** `<state_dir>/resident/universe.nlmres`.
- **Lock path:** `<state_dir>/resident/universe.nlmres.lock`.
- **Rollback image:** `<state_dir>/resident/universe.prev.nlmres`. It is a
  replaced, non-live image used for rollback; it is not mapped as a second
  universe.
- **Permissions:** directory `0700`, file and lock `0600`; reject symlinks and
  non-regular files (`O_NOFOLLOW` where available). Do not place the file on
  shared external storage.
- **One writer:** an exclusive non-blocking `flock` on the lock file. A second
  writer either opens a read-only snapshot or fails with `writer_busy`; it never
  interleaves writes. Readers map read-only and pin a generation.
- **Not a canonical path:** `state_dir/universe/` remains the native canonical
  checkpoint location. The resident file must not be advertised as canonical,
  and no API returns `canonical_commit: true`.

The file is sparse-capable. Creation uses `ftruncate` to the selected profile
size; it does not preallocate or populate every page.

## 2. Section map

The section IDs match `nanlmind_hermes/resident/__init__.py`. The default
256 MiB (`phone-256`) profile is laid out as follows. All offsets and lengths
are MiB unless stated otherwise.

```text
0                                                              256 MiB
+---------+---------+-------+-------+---------+---------+--------+----------+
| header  | chart   | heads | pairs | leaves  | relations| hyper  | features |
| 1 MiB   | 8 MiB   | 1 MiB | 4 MiB | 56 MiB  | 40 MiB   | 16 MiB | 16 MiB   |
+---------+---------+-------+-------+---------+---------+--------+----------+
| tokens  | index   | arena         | journal  |
| 16 MiB  | 32 MiB  | 50 MiB        | 16 MiB   |
+---------+---------+---------------+----------+
```

| ID | section | offset (MiB) | length (MiB) | record size | capacity |
|---:|---|---:|---:|---:|---:|
| 1 | `header` | 0 | 1 | — | 2 superblocks + 2 checkpoints |
| 2 | `chart` | 1 | 8 | 8 MiB page | 1 |
| 3 | `heads` | 9 | 1 | 4,096 B | 256 |
| 4 | `pairs` | 10 | 4 | 128 B | 32,640 |
| 5 | `leaves` | 14 | 56 | 256 B | 224,000 |
| 6 | `relations` | 70 | 40 | 512 B | 80,000 |
| 7 | `hyperedges` | 110 | 16 | 512 B | 32,768 |
| 8 | `features` | 126 | 16 | 1,024 B | 16,384 |
| 9 | `tokens` | 142 | 16 | 256 B | 65,536 |
| 10 | `index` | 158 | 32 | 64 KiB page | 512 pages |
| 11 | `arena` | 190 | 50 | byte | 52,428,800 B |
| 12 | `journal` | 240 | 16 | 4 KiB slot | 16,777,216 B / 4,096 slots |

The `chart` section is the exact 8 MiB chart page described by
`nanlmind_hermes/universe_chart.py` (8,388,608 bytes; `NLMCHRT1` header;
65,536 address registers; 32,896 canonical cells). It is kept as one page so
existing chart code can continue to operate on it. The other sections are new
resident records, not a reinterpretation of the chart page.

`phone-128` is the low-memory profile:

| section | offset | length |
|---|---:|---:|
| header | 0 | 1 MiB |
| chart | 1 | 8 MiB |
| heads | 9 | 1 MiB |
| pairs | 10 | 4 MiB |
| leaves | 14 | 28 MiB |
| relations | 42 | 20 MiB |
| hyperedges | 62 | 8 MiB |
| features | 70 | 8 MiB |
| tokens | 78 | 8 MiB |
| index | 86 | 16 MiB |
| arena | 102 | 18 MiB |
| journal | 120 | 8 MiB |

`desktop-1g` is the only profile allowed to reach the hard 1 GiB ceiling:

| section | offset | length |
|---|---:|---:|
| header | 0 | 1 MiB |
| chart | 1 | 8 MiB |
| heads | 9 | 1 MiB |
| pairs | 10 | 4 MiB |
| leaves | 14 | 256 MiB |
| relations | 270 | 192 MiB |
| hyperedges | 462 | 64 MiB |
| features | 526 | 16 MiB |
| tokens | 542 | 16 MiB |
| index | 558 | 32 MiB |
| arena | 590 | 370 MiB |
| journal | 960 | 64 MiB |

The 16 MiB value in `MIN_BYTES` is a fixture floor, not a production profile.
A fixture profile must be generated by the same layout validator, may shrink
paged sections, and must still contain a valid header and journal.

Section validation is fail-closed: IDs 1–12 exactly once; offsets aligned to 64
bytes (the shipped profiles use MiB alignment); no section overlaps another;
`offset + length <= file_size`; `record_size` divides fixed sections;
`used <= capacity`; `used` never exceeds the physical section. A file that
fails validation is not repaired implicitly.

## 3. Memory budgets and resident/paged split

Budgets are **file-backed RSS targets** for the resident layer, excluding
Python runtime and caller allocations. They are design targets, not measured
results.

| profile | file size | resident target | resident hard cap | process target | process hard cap |
|---|---:|---:|---:|---:|---:|
| `phone-256` | 256 MiB | 48 MiB | 80 MiB | 128 MiB | 192 MiB |
| `phone-128` | 128 MiB | 32 MiB | 56 MiB | 96 MiB | 144 MiB |
| `desktop-1g` | 1 GiB | 128 MiB | 256 MiB | 384 MiB | 768 MiB |

Resident by contract:

- header and both checkpoint slots (1 MiB section);
- chart (8 MiB);
- heads (1 MiB);
- pairs directory (4 MiB);
- the active index root and a bounded hot page set;
- the current journal head page and checkpoint descriptor.

Paged by contract:

- leaf records and leaf payloads in the arena;
- relation records outside the hot relation window;
- hyperedge records;
- feature vectors outside the hot feature window;
- token records outside the hot token window;
- cold index pages and journal history.

Window policy:

- Map the file with `mmap`; do **not** use `MAP_POPULATE` on Android.
- Use bounded windows of 1 MiB; default maximum 32 windows; release windows
  after a query or after the configured idle interval.
- LRU page cache: 24 MiB for `phone-256`, 16 MiB for `phone-128`, 96 MiB for
  `desktop-1g`.
- `madvise(MADV_RANDOM)` for sparse access; `MADV_WILLNEED` only for a bounded
  range, never the whole file.
- `mlock` at most 4 MiB, only for header/journal pages. Do not `mlock` the
  chart or the whole file.
- If `mmap` is unavailable or refused on a platform, use bounded
  `pread`/`pwrite` over the same offsets and the same digest/journal protocol;
  never fall back to reading the whole file.
- A `MemoryError` or budget refusal closes windows and returns
  `memory_budget_exceeded`; it never silently disables durability.

The 256 MiB default is a **file** budget, not a promise that 256 MiB is
resident. The phone profile is expected to run far below the file size.

## 4. Superblock, section table and checkpoint slots

The header section contains:

```text
0x0000  superblock copy A (512 B, own 4 KiB page)
0x1000  superblock copy B (512 B, own 4 KiB page)
0x2000  section table, 12 entries x 64 B = 768 B
0x3000  checkpoint slot A (4 KiB)
0x4000  checkpoint slot B (4 KiB)
0x5000  reserved for writer epoch / diagnostics
```

Superblock fields:

| field | type | meaning |
|---|---|---|
| `magic` | 8 B | `NLMRES01` |
| `version` | u32 | `RESIDENT_VERSION` (1) |
| `header_bytes` | u32 | 1,048,576 |
| `generation` | u64 | monotonic committed generation; 0 means uninitialized |
| `parent_generation` | u64 | generation this state descends from |
| `state_digest` | 32 B | digest defined in §6 |
| `parent_digest` | 32 B | digest of the parent generation |
| `active_checkpoint` | u32 | 0 or 1; selector, not authority by itself |
| `profile_id` | u16 | `phone-256`, `phone-128`, `desktop-1g`, fixture |
| `section_count` | u16 | 12 |
| `flags` | u32 | read-only, recovery, compaction; canonical flag must stay 0 |
| `journal_head_seq` | u64 | next journal sequence to write |
| `journal_checkpoint_seq` | u64 | journal sequence covered by active checkpoint |
| `created_ns` | u64 | creation time |
| `committed_ns` | u64 | last commit time |
| `writer_epoch` | 16 B | random process epoch, diagnostic only |
| `superblock_crc32` | u32 | CRC over the preceding bytes |
| `reserved` | — | zeroed |

The two superblock copies live in separate 4 KiB pages. A torn A copy falls
back to B, and vice versa; both invalid is `bad_superblock`. The superblock is
never the sole authority for a committed transaction: the journal commit record
and checkpoint digest must agree with it.

Section table entry (64 bytes):

| field | type | meaning |
|---|---|---|
| `id` | u16 | 1–12 |
| `flags` | u16 | fixed/paged/append-only |
| `offset` | u64 | absolute file offset |
| `length` | u64 | section length |
| `record_size` | u32 | 0 for byte arena |
| `capacity` | u32 | records or bytes |
| `used` | u32 | committed records/bytes at the checkpoint |
| `name` | 12 B | NUL-padded ASCII |
| `root_prefix` | 16 B | first 16 bytes of the data-section root; zero for header |
| `reserved` | 4 B | zero |

Checkpoint slots are independent 4 KiB records:

```text
magic "NLMCHK01", version, slot, generation, parent_generation,
journal_seq, journal_checkpoint_seq, created_ns, profile, section_count,
state_digest, section_roots[12] (32 B each; indices 1 and 12 are zero/reserved),
mirror_residual (f64), used_bytes (u64), flags, crc32
```

A checkpoint is valid only when its CRC, state digest and the 10 state-section
roots (IDs 2..11) verify. The header and journal sections are not part of the
state digest, so a checkpoint or commit record cannot recursively change the
digest it stores; the header is covered by the superblock CRC and the journal
by its record chain. The newest valid checkpoint wins; a newer torn slot is
ignored and the older slot is used. If neither slot is valid, recovery may use
the journal only
when a complete commit chain exists. Otherwise the file is opened read-only with
`no_valid_checkpoint`; it is never silently reinitialized.

## 5. Record layouts

All record digests are SHA-256 over the canonical record bytes with the digest
field zeroed. All integers are little-endian. Local handles are `u32` and are
file-local; portable bundles use 32-byte content digests, never handles.

### 5.1 Pair record (128 B)

| offset | field | notes |
|---:|---|---|
| 0 | `rank` u32 | `rank_of_pair(a,b) = a*(511-a)/2 + (b-a-1)` |
| 4 | `a` u16, `b` u16 | `0 <= a < b <= 255` |
| 8 | `canonical_addr` u32 | `min(addr(a,b), reverse16(addr(a,b)))` |
| 12 | `mirror_addr` u32 | `reverse16(canonical_addr)` |
| 16 | `mirror_rank` u32 | `32639 - rank` |
| 20 | `cell_index` u32 | index into the chart's canonical cells (0..32,895) |
| 24 | `subject_head` u16, `predicate_head` u16, `object_head` u16 | role hints |
| 30 | `flags` u8, `mirror_flags` u8 | occupied/valid/mirror-valid |
| 32 | `s_plus` u64 | forward temporal rail, past → future |
| 40 | `s_minus` u64 | reverse temporal rail, future → past |
| 48 | `amplitude` f32 | pair weight |
| 52 | `phase_rad` f32 | phase channel |
| 56 | `gauge_residual` f32 | last gauge fit residual |
| 60 | `cycle_re` f32, `cycle_im` f32 | complex cycle flux |
| 68 | `relation_count` u32 | number of relations citing this cell |
| 72 | `record_digest` 32 B | canonical record digest |
| 104 | `reserved` 24 B | zero |

`rank` and `mirror_rank` are the **true rank mirror**. `canonical_addr` and
`mirror_addr` are the **address mirror**. They are different involutions and
must not be conflated. `s+`/`s-` are temporal rails, not the mirror pair.

The pair cell, the chart cell words and the head record are **fixed mutable
cells**, unlike append-only leaves/relations/hyperedges. A transaction that
changes one writes the before and after images to the journal before touching
the cell; an uncommitted transaction is restored from the before image during
recovery.
Snapshot isolation for scalar pair/head values is therefore “latest committed”;
a query that needs a frozen scalar copies it into its generation snapshot. The
append-only records remain generation-pinned.

### 5.2 Other records (summary)

- **Head (4,096 B):** magic, head `u16`, mirror `reverse8(head)` `u16`,
  generation, 16 Boolean score words, 256-bit active mask, 64 phase channels,
  last leaf/relation/hyperedge handles, local scratch. Existing head scratchpad
  semantics from `universe_chart.py` and `mth_layers.py` are preserved.
- **Leaf (256 B):** content SHA-256, schema/kind, generation, parent handle,
  head/mirror, resolution L0–L9, arena offset/length, payload digest, feature
  and token handles, relation/hyperedge counts, source digest, record digest.
- **Relation (512 B):** relation ID, schema, resolution, polarity, generation,
  subject/predicate/object leaf handles, three head handles, three pair ranks,
  three mirror ranks, hyperedge handle, confidence/support/refute, evidence
  handle, source digest, record digest. See
  [RESIDENT_TREE_AND_SUPERPOSITION.md](RESIDENT_TREE_AND_SUPERPOSITION.md) for
  the mandatory link closure.
- **Hyperedge (512 B):** hyperedge ID, schema, arity, kind, generation,
  relation handle, 8 primary member handles, 8 member kinds, secondary mirror
  members, evidence handle, confidence, source digest, record digest. An
  `N16REL/1` relation creates a 6-member primary hyperedge: subject, predicate
  and object heads plus the three pairwise cells.
- **Feature (1,024 B):** exactly 256 `float32` values. No short-array
  broadcast; non-finite values are rejected.
- **Token (256 B):** token SHA-256, UTF-8 bytes bounded to 128 B, kind,
  resolution hint, counts, first/last leaf handles, generation.
- **Arena:** append-only, 64-byte-aligned payload records with a 32-byte header
  (length, digest, generation, flags). Default maximum single payload 64 KiB;
  16 KiB in `phone-128`. Deduplicate by digest; no in-place overwrite.
- **Index:** 64 KiB pages containing exact token/leaf maps, relation triple
  maps, pair adjacency, hyperedge membership and feature buckets. Index entries
  carry a generation and are rebuildable from records.

## 6. Generation, journal, checkpoint and recovery

### 6.1 Digest

```text
record_digest = SHA256(canonical_record_bytes_with_digest_zeroed)

# fixed-record sections:
section_root  = SHA256("NLMRES-SEC1" || u16(id) || u32(record_size)
                       || u32(used) || SHA256(record_digest[0..used]))

# byte sections (arena) and journal slots:
section_root  = SHA256("NLMRES-SEC1" || u16(id) || u32(0)
                       || u32(used) || SHA256(bytes[0..used]))

state_digest  = SHA256("NLMRES-STATE1" || u32(version) || u16(profile)
                       || u64(generation) || u64(parent_generation)
                       || parent_digest || section_root[2..11])
```

`used` and the roots are checkpoint values. Sections 1 (header) and 12
(journal) are excluded from `state_digest`; the header is validated by the
superblock CRC, and the journal by its record chain/CRC. This avoids a
checkpoint or commit record changing the digest it is trying to store. For the
journal ring, `used` is the committed byte span (slots × 4,096), not a logical
record count. A digest computed over a different `used` set is a different
state, even if the bytes happen to coincide. Logical recovery uses sequence
numbers and commit records, not a digest of the ring image.

### 6.2 Transaction protocol

1. Acquire the writer lock and complete recovery.
2. Require `expected_generation` (and optionally `expected_digest`) and compare
   it with the current committed generation. A missing CAS on a mutation is
   `cas_required`, not an implicit overwrite.
3. Append a `BEGIN` journal record with a random 16-byte `txn_id`.
4. Append new leaf/relation/hyperedge/token records and append
   `APPEND`/`INDEX` journal records. For fixed pair/head cells and chart cell
   words, append before/after images (`CELL` records; a 4 KiB head may span
   multiple journal records) before updating the cell. Append-only records are
   never modified in place.
5. Compute the new state-section roots (2..11) and state digest. Append a
   `COMMIT` record containing the new generation, roots and digest.
6. `msync`/`flush` touched data windows and `fsync` the file. Then update both
   superblock copies with the new generation/digest and `fsync` the header
   pages. A crash before the superblock update leaves the old generation
   visible; recovery adopts the committed journal record.
7. Return a receipt containing `generation`, `state_digest`, `journal_seq`,
   counts, `durable: true|false`, and `canonical_commit: false`.

`fsync` order is data before journal before superblock. `durable: true` is
returned only after all three complete. A group-commit mode may acknowledge
before the final `fsync`, but it must report `durable: false` and must not be
used by intake export.

### 6.3 Journal

The journal is a 4,096-slot ring of 4 KiB records. A slot has a 192-byte
header (magic, version, seq, txn ID, op, flags, section, offset, length,
payload CRC/digest, previous-record digest, state digest) and at most 3,904
bytes of payload. `op` is `BEGIN`, `APPEND`, `INDEX`, `CELL`, `COMMIT`,
`ABORT`, `CHECKPOINT` or `ROLLBACK`. Sequence numbers are monotonic. A slot may be
overwritten only after its sequence is covered by a valid checkpoint; if the
writer needs a slot that is not covered, it forces a checkpoint or fails
`journal_full`.

Recovery:

- choose the newest valid checkpoint;
- scan forward from its `journal_checkpoint_seq + 1` to the first invalid,
  torn or non-contiguous record;
- find the last `COMMIT`, recompute state-section roots (2..11), verify the
  journal chain and compare the digest;
- adopt a committed generation newer than the superblock;
- on a torn tail, logically truncate at the last valid record, restore any
  uncommitted fixed cell from its before image, and report `journal_torn_tail`;
  never replay a `BEGIN` without `COMMIT`;
- on digest divergence, fail closed (`state_digest_mismatch`), open read-only
  and require operator action.

### 6.4 Checkpoints

Checkpoints are written to the inactive slot at most every 256 commits or 30
seconds (configurable, disabled during a transaction). A checkpoint records the
section `used` counters, roots, state digest, journal sequence and mirror
residual. After the inactive slot is fsynced, the superblock selector is
advanced and fsynced. The previous slot remains the fallback. Checkpointing does
not advance the generation; it only makes an already committed generation
faster to recover.

### 6.5 Compaction and rollback

Compaction is a committed maintenance transaction. It writes a new file
containing only live records, fsyncs it, and atomically renames it over the live
path. It consumes the next generation, `parent_generation = old generation`, and
sets a `compaction` flag in the journal commit; the old file becomes `.prev`
until the next successful checkpoint. Readers already mapped to the old inode
finish their pinned snapshot; new readers open the new path.

Rollback is a **new generation**, never a reused number. A verified checkpoint
image (or `.prev` image) is copied to a temporary file, validated, and atomically
renamed over the live path. The new generation is `max(current, checkpoint)+1`;
its state digest is recomputed for that generation, not copied from the target
checkpoint. The old file is retained as `.prev`. The rollback receipt names the
target checkpoint, old generation, new generation, old/new digests and the
reason. In-place rewind of the live file is forbidden.

## 7. Mirror rails

The chart already defines a pair address and mirror:

```text
addr(a,b)          = (a << 8) | reverse8(b)
canonical_addr     = min(addr(a,b), reverse16(addr(a,b)))
mirror_addr        = reverse16(canonical_addr)
rank(a,b)          = a*(511-a)/2 + (b-a-1)          # 0..32639
mirror_rank        = 32639 - rank
head_mirror        = reverse8(head)
```

The resident universe stores **both** mirrors:

- **Address mirror:** `mirror_addr = reverse16(canonical_addr)`, matching the
  native chart.
- **Rank mirror:** `mirror_rank = 32639 - rank`, matching
  `nanlmind_hermes/temporal_spine.py`. This is not `255 - i` complement
  pairing and has no fixed point because 32,639 is odd.
- **Head mirror:** `reverse8(head)` for the 256 heads.

Each pair cell also has two temporal rails:

```text
s+ : forward rail, past -> future
s- : reverse rail, future -> past
```

A temporal commit advances the frame and copies the committed forward word into
the reverse rail, preserving the complementary pair invariant. For complex
couplings, the mirror cell satisfies `A_mirror = conj(A)`; for integer rails,
the mirror record carries the corresponding copied word. `mirror_residual` is
the maximum violation of that relation.

The rank-local update is the existing `temporal_spine.py` form, with `B = s-`
past, `F = s+` future and `A_r` the coupling at rank `r`:

```text
epsilon   = F - A_r * B
A_r       = A_r - eta * epsilon * conj(B) / (|B|^2 + eps)
A_mirror  = conj(A_r)
```

This is a classical local update; the resident layer may wrap it but must not
reimplement a different formula under the same name. No convergence or
accuracy result is claimed here.

The chart cell remains the native serialization. The pair record's cached rails
must match the chart cell on read; a mismatch is `rail_divergence`. The pair is
not served from the cache until it is rebuilt from the chart and the repair is
journaled. Divergence is corruption, not a tie-break.

## 8. The 2 GiB overbootstrap map is never copied

The 2 GiB overbootstrap map is an external bootstrap artifact. It is not a
section, not a payload and not an input to this file.

- The resident API accepts only a `bootstrap_ref` (opaque source ID, size,
  SHA-256) and never opens, reads, mmaps or copies the referenced map.
- `open` rejects any file or request whose mapped length exceeds
  `MAX_BYTES`; `import`/`export` reject any section or bundle that would carry
  the map or a payload larger than its section.
- The error is `overbootstrap_copy_denied`. There is no fallback that copies
  “just the hot part” of the 2 GiB map.
- The phone profile must work with no 2 GiB file present at all.
- A test may create a fake 2 GiB path with a canary; opening the resident
  universe must not open it, read it or increase file-backed RSS by more than
  the allowed window budget.

## 9. Failure modes

| failure | detection | effect | response / recovery |
|---|---|---|---|
| bad magic/version/profile | superblock validation | cannot open | `bad_magic`/`bad_version`/`bad_profile`; no write |
| section overlap/out of bounds | section table validation | cannot open | `bad_section_table`; operator action |
| torn superblock copy | CRC on A/B | fallback to other copy | open; report warning |
| both superblocks invalid | CRC | no authority | `bad_superblock`; read-only or refuse |
| torn newest checkpoint | CRC/digest/roots | use older slot | `checkpoint_fallback`; write after recovery |
| both checkpoints invalid | validation | no checkpoint | journal-only recovery if complete; else `no_valid_checkpoint` |
| torn journal tail | magic/CRC/sequence | ignore tail | truncate logically; restore uncommitted cell before images; `journal_torn_tail` |
| torn fixed-cell write | cell digest vs journal | cell not committed | restore before image; `journal_torn_tail` |
| journal full | ring check | cannot append | force checkpoint; else `journal_full` |
| stale writer | CAS mismatch | no write | `stale_generation`/`stale_digest`; reload |
| second writer | `flock` | no write | `writer_busy`; open read-only |
| section full | `used + n > capacity` | no partial commit | `section_capacity_exceeded` |
| index corruption | page/record digest | query cannot trust route | rebuild from records; else read-only |
| rail divergence | pair record vs chart | pair not served | rebuild from chart, journal repair |
| mirror violation | rank/address/coupling check | relation quarantined | `mirror_violation`; no silent fix |
| state digest mismatch | recompute roots | corruption/attack | `state_digest_mismatch`; fail closed |
| power loss before superblock | journal commit newer | old generation visible | replay committed journal |
| power loss during compaction | temp/rename | old or new file | keep `.prev`; `compaction_failed` if both invalid |
| read-only filesystem | open/mmap/write | no mutation | `read_only`; reads still work |
| memory pressure | window/cache budget | no silent eviction of dirty data | `memory_budget_exceeded`; close windows |
| 2 GiB copy request | size/path/canary check | denied | `overbootstrap_copy_denied` |

## 10. Invariants

1. Exactly one live `.nlmres` file and one writer per local universe.
2. `generation` is monotonic and increases by exactly one per committed
   transaction, including compaction and rollback transactions; it is never
   reused after rollback.
3. A mutation without `expected_generation` is rejected; stale writers never
   overwrite newer state.
4. No committed leaf/relation/hyperedge/token record is modified in place.
   Pair/head cells and chart cell words are mutable only through a journaled
   before/after image and are restored from the before image if the transaction
   does not commit. Append-only records are the source of truth; indexes are
   rebuildable.
5. A reader sees a generation-bound snapshot for append-only records.
   Pair/head/chart scalar cells are latest-committed unless copied into that
   snapshot; records newer than the pinned generation are invisible. A fully
   frozen scalar view requires a compacted checkpoint image, not an in-place
   read of a live cell.
6. A checkpoint is valid only if CRC, state digest and the state-section roots
   (2..11) verify; the journal is validated separately by its record chain. The
   newest valid checkpoint wins.
7. A committed transaction is recoverable from the journal even if the
   superblock or checkpoint lags.
8. `s+`/`s-` are temporal rails, not the mirror link; both mirrors (address and
   rank) are stored and validated.
9. `mirror_rank(rank) = 32639 - rank` and
   `mirror_addr(canonical) = reverse16(canonical)` are never conflated.
10. Every pair record has exactly one mirror link (possibly self for a
    palindromic address); involution and coupling residual are checked.
11. `chart` remains the native 8 MiB serialization; pair rails that disagree
    with it are a fail-closed corruption.
12. The 2 GiB overbootstrap map is never opened, mapped, read or copied.
13. `canonical_commit`, `six_runner_barrier` and `merge_enabled` claims are
    explicit in receipts; resident writes never set them true.
14. Local handles never cross a file or bundle boundary; portable references
    are 32-byte content digests.
15. `phone-256` is the default; `desktop-1g` requires explicit opt-in; 1 GiB is
    a hard ceiling, not a target.

## 11. Acceptance tests (proposed, not executed)

These tests do not exist yet. They are the required gates for an implementation.
All assertions are against the proposed contract; no test result is implied.

| ID | test | assertion |
|---|---|---|
| U-01 | `test_resident_layout.py::test_default_table` | 12 sections, exact offsets/lengths, no overlap, total 256 MiB, all aligned |
| U-02 | `test_resident_layout.py::test_capacities` | pair capacity 32,640; head 256; relation/leaf/hyperedge capacities match §2 |
| U-03 | `test_resident_open.py::test_genesis_and_reopen` | create, commit one generation, close, reopen; generation/digest match |
| U-04 | `test_resident_cas.py::test_stale_writer` | two writers over one file; exactly one commit; other `stale_generation` |
| U-05 | `test_resident_recovery.py::test_torn_checkpoint` | corrupt newest slot; older slot is used; open succeeds |
| U-06 | `test_resident_recovery.py::test_both_checkpoints_bad` | fail closed; no silent reset |
| U-07 | `test_resident_recovery.py::test_torn_journal_tail` | partial tail ignored; last commit survives |
| U-08 | `test_resident_recovery.py::test_commit_after_checkpoint` | journal commit newer than superblock is adopted |
| U-09 | `test_resident_generation.py::test_reader_pin` | old reader sees old snapshot; new reader sees new generation |
| U-10 | `test_resident_mirror.py::test_involutions` | rank mirror `32639-r`; address mirror `reverse16`; head `reverse8`; no conflation |
| U-11 | `test_resident_mirror.py::test_rail_divergence` | chart/pair mismatch returns `rail_divergence`, no silent repair on read |
| U-12 | `test_resident_budget.py::test_section_full` | filling a section returns `section_capacity_exceeded`; no partial generation |
| U-13 | `test_resident_budget.py::test_no_overbootstrap` | fake 2 GiB path is never opened; no copy; `overbootstrap_copy_denied` on request |
| U-14 | `test_resident_compaction.py::test_atomic_compaction_and_rollback` | `.prev` retained; rollback creates a new generation with target digest |
| U-15 | `test_resident_corruption.py::test_digest_mismatch` | mismatch fails closed; read-only mode; no rewrite |
| U-16 | `test_resident_phone_profile.py::test_rss_target` | optional environment-gated RSS check; skip with explicit reason if unavailable |

A test that only checks constants is not sufficient. U-04, U-05, U-07, U-08,
U-10, U-13 and U-15 are mandatory before any intake integration.

## 12. Exact file/module ownership

| path | status | owner / responsibility |
|---|---|---|
| `nanlmind_hermes/resident/__init__.py` | **exists** | constants only: magic, version, section IDs, sizes, `align`, `clamp_size` |
| `nanlmind_hermes/resident/layout.py` | proposed | profile tables, offsets, section-entry encoding, validation |
| `nanlmind_hermes/resident/superblock.py` | proposed | superblock A/B, CRC, checkpoint slot encode/decode |
| `nanlmind_hermes/resident/journal.py` | proposed | ring records, sequence, `BEGIN`/`COMMIT`/`ROLLBACK`, recovery scan |
| `nanlmind_hermes/resident/store.py` | proposed | mmap windows, `flock`, open/close, CAS, msync/fsync, compaction |
| `nanlmind_hermes/resident/records.py` | proposed | pair/head/leaf/relation/hyperedge/feature/token codecs |
| `nanlmind_hermes/resident/index.py` | proposed | exact/triple/pair/hyperedge/feature indexes; rebuild |
| `nanlmind_hermes/resident/universe.py` | proposed | `ResidentUniverse.open/append/lookup/snapshot/checkpoint/rollback` |
| `nanlmind_hermes/resident/resonance.py` | proposed | dot/cosine/phase/cycle/gauge scoring wrappers |
| `nanlmind_hermes/resident/tree.py` | proposed | hub/leaf/relation/mirror link validation and insertion |
| `nanlmind_hermes/resident/query.py` | proposed | candidate union, fusion, abstention, generation visibility |
| `nanlmind_hermes/resident/bundle.py` | proposed | portable candidate bundle container/manifest |
| `nanlmind_hermes/resident/intake.py` | proposed | intake validation, plan, apply, receipts, rollback |
| `nanlmind_hermes/resident/receipts.py` | proposed | local receipt journal and rollback tokens |
| `nanlmind_hermes/resident/tools.py` | proposed | plugin handlers; requires registration changes outside this doc |
| `nanlmind_hermes/universe_chart.py` | **exists** | 8 MiB chart layout and native cell semantics |
| `nanlmind_hermes/chart_storage.py` | **exists** | two-slot snapshot generation lease; precedent, not the resident file |
| `nanlmind_hermes/temporal_spine.py` | **exists** | rank mirror `32639-r` and local coupling update |
| `nanlmind_hermes/gauge_memory.py` | **exists** | gauge residual/retrieval math |
| `nanlmind_hermes/phase_memory.py` | **exists** | phase/gauge scoring |
| `nanlmind_hermes/cycle_space.py` | **exists** | cycle/gradient decomposition |
| `nanlmind_hermes/mth_leaves.py` | **exists** | immutable leaf/hub semantics and N16I execution receipts |
| `nanlmind_hermes/memory.py` | **exists** | append-only evidence event ledger; not the resident file |
| `tests/test_resident_layout.py` | proposed | U-01/U-02 |
| `tests/test_resident_open.py` | proposed | U-03 |
| `tests/test_resident_cas.py` | proposed | U-04 |
| `tests/test_resident_recovery.py` | proposed | U-05..U-09 |
| `tests/test_resident_mirror.py` | proposed | U-10/U-11 |
| `tests/test_resident_budget.py` | proposed | U-12/U-13 |
| `tests/test_resident_compaction.py` | proposed | U-14 |
| `tests/test_resident_corruption.py` | proposed | U-15 |
| `tests/test_resident_phone_profile.py` | proposed | U-16, environment-gated |

Existing modules keep their ownership. The resident layer must call
`universe_chart`, `temporal_spine`, `gauge_memory`, `phase_memory`,
`cycle_space` and `mth_leaves` rather than duplicating their math or leaf
formats. No changes to those modules are part of this docs-only change.

## 13. Integration status and non-claims

- **Implemented today:** `resident/__init__.py` constants; the existing 8 MiB
  chart, two-slot chart snapshot CAS, evidence ledger and math modules named
  above. None of them is a unified resident mmap.
- **Proposed:** the file format, profiles, records, journal, recovery,
  mirror rails, quotas, APIs and tests in this document.
- **Not claimed:** test results, throughput, latency, RSS measurements, intake
  deployment, canonical merge, six-runner barrier, production durability or
  Android certification.
- **Hard boundary:** the resident file is local-derived. It cannot promote
  canonical state and cannot substitute for the BS0 generation barrier.
