# TOTAL integration plan: one bounded resident universe, continuous corpus + chat learning

Status: **implementation plan with current-state audit**, 2026-11-08. Repository
`/data/data/com.termux/files/home/NaNLMind_HERMES` at dirty `a14e135`. Values are
**measured** (observed here), **artifact** (recorded result), or **target** (not
run). This is not a release note or proof.
## 1. Executive truth
**One line:** the repository has a real bounded one-file resident store, a real
count-based temporal student, and corpus/evaluation scripts, but not the loop that
feeds corpus and free chat through concept leaves, typed relations, pair state and
the student into one checkpointed universe. The Round 23/24 C++ objects are
reference prototypes, not the Python implementation.

| Area | Exists now | Measured / artifact | Design-only or missing |
|---|---|---|---|
| Resident core | `resident/` layout, cells, leaves, relations, resonance, ingest, export, persistence, CLI, plugin tools | **measured:** 191 resident tests OK in 46.5 s; 256 MiB create/ingest/query/reopen on Termux/aarch64; `nanlm_resident_status/ingest/query` registered | writer lock, six-runner CAS barrier, LRU window manager |
| Layout | v1: 64 KiB header, chart, payload, 2 MiB cells, leaves, relations+hyperedges, features, journal | **measured:** `phone-256` commits 123,470,080 B; `desktop-1g` 445,448,448 B | `UNIFIED_RESIDENT_UNIVERSE.md` still describes a different 12-section table |
| Heads/pairs | 256 x 32 B heads + 32,640 x 64 B cells (two 32 B rails) | **measured:** tests | rank mirror, pair handle, cycle cache, Hodge maintenance |
| Leaves/tree | 112 B records, 128-bit content ids, parent/child links | **measured:** tests | concept registry and leaf-ref resolver partial |
| Relations/hyperedges | 304 B relations, 96 B hyperedges, leaf/relation/pair/mirror members | **measured:** 3 relations + 3 hyperedges from 792 rows; reopen preserved | parent edge, concept member, incidence index, traversal API |
| Temporal LM | interpolated uni/bi/tri counts, mirror-8 frame, beam generation | **artifact:** held-out NLL 2.513986 / ppl 12.3541 | no optimizer; resident hybrid scoring only in `learning.py`, not main generation; pair/cycle not in `_prob` |
| Corpus training | `corpus_training.py`, `deep_train_corpus.py`, `resident_learning.py` | **artifact:** 14,039 train / 1,449 held-out; 1,000 scored / 28,521 tokens; resident two-channel loop 13,655 learned | repeated raw-count epochs add no information; full resident run is one pass |
| Free chat | deterministic parser + L0-L9 learner | **artifact:** 15/15 calls; quiz 76.1% -> 100.0% in-context | resident pair exposure exists; typed-relation/hyperedge feedback loop partial |
| Gemma | teacher/evaluator scripts over Ollama | **artifact:** 12/12 calls; 14/14 probe curve; novel ratio 0.1825 | not the student; no resident integration |
| MTH/N16I | 91 kernels, 44 families, 91/91 linked, four typed vector kernels | **artifact:** 73,982 native vs 3,343,783 interpreted steps; 2,640 identical bytes | 87 kernels have no typed ABI; resident does not call MTH |
| NANNET HW2 | pairing, scopes, router client, probe | **artifact:** 11/11 live-smoke checks; loopback 8/8 | fixed HW2/node 3 production binding open |
| Tools/CLI | 79 manifest tools; resident CLI 8 commands | **measured:** CLI init/status/ingest/query/export/validate/checkpoint run today; resident tools registered | CLI ingest caps at 256 rows |
| Canonical merge | default false; no merge tool | **artifact:** all receipts false | six-runner barrier and generation-checked validator absent |
| Quantum math | gauge/cycle/phase/spectral/robust recovery + read-only bridge | **artifact:** 22 targeted tests passed | mostly documented-not-wired; no physical claim |

| Artifact | Value | Class |
|---|---|---:|
| Resident targeted tests | 191 OK, 46.5 s | **measured** |
| `phone-256` file / committed | 268,435,456 / 123,470,080 B | **measured** |
| `desktop-1g` file / committed | 1,073,741,824 / 445,448,448 B | **measured** |
| 792-row ingest | 2.235 s; 1,046 leaves; 799 features; gen 792 | **measured** |
| Resident query, 799 features | 58.9 ms top-3 cosine | **measured**, host-specific |
| Deep corpus epoch 1 | NLL 2.513986, ppl 12.3541, vocab 17,962, bigram 17,963, trigram 60,299 | **artifact** |
| Gemma probes | 0/14 -> 14/14, structured 11/14; 48 turns, 480 scratch, 60 concepts, 134 pairs, 255 leaves | **artifact**, confounded |
| Re-score of another direct run | keyword 12/14 (85.7%), direct-exact 10/14 (71.4%), baseline 1/14 | **artifact**, different lineage |
| Novel generation | 8 prompts, 0 empty, mean 8.0 tokens, novel-3-gram 0.1825, 2 copied, judge answers 0/8 | **artifact** |
| Round 23 C++ | 8,388,608 B; 65,280 mappings; Hodge 1.032 ms; max divergence 8.84e-7 | **prototype**, container |
Do not merge the 14/14 curve and the 12/14 read-only re-score: different runs,
different baselines. The 14/14 table is not a clean held-out learning proof. C++
times are not Android/Python times; the synthetic benchmark is not resident RSS
evidence.

**Non-claims:** no canonical merge/commit or six-runner barrier; no physical
quantum behavior/speedup/proof; no native performance for Python mirrors; no
replacement of `universe/native/global_universe.nmg`; Gemma is not the student and
never receives held-out text; the 2 GiB map is never copied.
## 2. One-file memory architecture
### 2.1 Decision: the active v1 layout is the target

The integration target is the layout the code opens today, not the 12-section
table in `docs/UNIFIED_RESIDENT_UNIVERSE.md` section 2. That table has different
section IDs, 1 MiB headers, 4 MiB pair records, token/index sections and a 50 MiB
arena; it has not been exercised. Using both creates two formats by documentation.
The first implementation task is to replace that table with the measured v1 table.

Header: two independent 4 KiB JSON slots at offsets 0 and 4096, each with
`NLMUHDR1`, version, body length and SHA-256; the complete section table is in
each body. The newer valid slot wins, the other is fallback.
### 2.2 Exact sections: `phone-256` (measured 2026-11-08)
| Section | Start byte | Length | Capacity / record |
|---|---:|---:|---|
| header | 0 | 65,536 | 2 x 4 KiB slots |
| chart | 65,536 | 8,388,608 | 8 MiB `NLMCHRT1` placeholder |
| payload arena | 8,454,144 | 33,554,464 | 33,554,432 B payload |
| heads | 42,008,640 | 8,192 | 256 x 32 B |
| pairs | 42,016,832 | 2,088,960 | 32,640 x 64 B (2 rails) |
| leaves | 44,105,792 | 20,971,568 | 131,072 leaves |
| relations+hyperedges | 65,077,376 | 41,615,464 | 32,768 each |
| hyperedge-index | 106,692,864 | 65,536 | compatibility placeholder |
| features | 106,758,400 | 16,777,216 | 32,768 x 512 B (256 int16) |
| tokens | 123,535,616 | 65,536 | compatibility placeholder |
| index | 123,601,152 | 65,536 | compatibility placeholder |
| journal | 123,666,688 | 8,388,640 | 131,072 x 64 B records |
`status()` reports `committed_bytes: 132,055,296`; the last section ends at
132,055,360 (alignment bookkeeping). The file is sparse: the unused ~124 MiB is
not capacity and must not be called resident memory.
### 2.3 Exact sections: `desktop-1g` (measured 2026-11-08)
| Section | Start byte | Length | Capacity / record |
|---|---:|---:|---|
| header | 0 | 65,536 | 2 x 4 KiB slots |
| chart | 65,536 | 8,388,608 | 8 MiB chart page |
| payload arena | 8,454,144 | 134,217,760 | 134,217,728 B |
| heads | 142,671,936 | 8,192 | 256 x 32 B |
| pairs | 142,680,128 | 2,088,960 | 32,640 x 64 B |
| leaves | 144,769,088 | 83,886,128 | 524,288 leaves |
| relations+hyperedges | 228,655,232 | 166,461,544 | 131,072 each |
| hyperedge-index | 395,116,800 | 65,536 | compatibility placeholder |
| features | 395,182,336 | 67,108,864 | 131,072 x 512 B |
| tokens | 462,291,200 | 65,536 | compatibility placeholder |
| index | 462,356,736 | 65,536 | compatibility placeholder |
| journal | 462,422,272 | 16,777,248 | 262,144 x 64 B |
`committed_bytes` is 479,199,488. `_plan_for_size()` thresholds: <=48 MiB
fixture, <=160 MiB `phone-128`, <=768 MiB `phone-256`, else `desktop-1g`. A
768 MiB request still selects `phone-256`; use an explicit 1 GiB.

Hard ceiling **1,073,741,824 B**. API requests above it return `budget_exceeded`;
a CLI may clamp only with `size_clamped: true`. A 2 GiB file is never opened,
mapped, read, copied or used as bootstrap. `ResidentUniverse.create()` rejects
requests above the ceiling before file creation; the denial is covered by
`tests.test_resident_universe`.
### 2.4 Resident vs paged
| Region | Current | Required |
|---|---|---|
| header | mmap, 2 slots | resident; `mlock` at most 4 MiB incl. journal head |
| cells 2 MiB | mmap, demand-paged | resident/touched; no `MAP_POPULATE` |
| chart 8 MiB | mmap placeholder | mmap but do not mlock whole page |
| leaves | mmap per lookup | paged, bounded hot set |
| relations/hyperedges | mmap append-only | paged outside hot window |
| features | mmap, linear scan | paged; bounded 24 MiB query cache |
| payload | mmap append-only | paged; no whole-arena copy |
| journal history | mmap append-only | head resident; history paged |
The whole-file mmap lets the OS page; `resident: true` in status means "mapped",
not physically resident. No LRU, `madvise`, `mlock`, eviction or RSS meter exists
yet; those are targets.
### 2.5 Growth, writer and recovery rules

1. Capacities are fixed at creation; no in-place expansion. Duplicate ids are
   idempotent and consume no capacity. At 70% of leaves/relations/features or 75%
   of journal, warn and checkpoint; at 90%, refuse new unique records with
   `capacity_exceeded` (duplicates still work).
2. Grow by checkpoint -> export candidate -> create next profile -> approved
   import -> verify counts/digest -> fsync -> atomic rename; retain old as `.prev`.
   New generation is `old + 1`, not a continuation inode. If `desktop-1g` fills,
   stop; cold evidence goes to an export bundle or SQLite, never a 2 GiB map.
3. Compaction is new file + atomic rename. Rollback copies a verified checkpoint
   or `.prev` to a new generation; never rewinds numbers.
4. **Current writer:** one mmap, alternating 4 KiB JSON headers, 64 B state journal
   chaining generation/digest, two-slot checkpoint manifest. No `flock`, CAS,
   per-record commit, `.prev` or torn-record truncation; a crash can leave records
   newer than the header.
5. **Required writer:** exclusive `flock`; second writer `writer_busy` or read-only;
   every mutation carries `expected_generation` (missing `cas_required`, mismatch
   `stale_generation`); write records, append `COMMIT`, fsync data, fsync journal,
   write alternate header, fsync; readers serve only `generation <= committed`.
6. **Required recovery:** newest valid header; scan to first torn record; adopt a
   complete commit newer than header; truncate torn tail (`journal_torn_tail`);
   invalid headers or digest mismatch -> read-only (`bad_header`,
   `state_digest_mismatch`); never silently reinitialize. Checkpoint every 256
   commits or 30 s. No local handle crosses the file; exports use 32-byte digests.

## 3. Round 23 integration: 256-head / 32,640-pair mirrors

`cells.py` already has one canonical cell for `(a,b)` and `(b,a)` with two
independently writable 32-byte rails in a 64-byte record. `pair_index` is
0..32,639; `pair_state` returns canonical/mirror addresses, `s+`, `s-`, phase,
amplitude, weight, relation chain and generation.

| Round 23 element | Current | Action |
|---|---|---|
| 256 heads | 256 x 32 B | Keep; do not expand to 128 B |
| 32,640 pairs | 32,640 x 64 B | Keep; do not expand to 256 B |
| two 128 B lanes | two 32 B rails | Keep two-lane semantics; extra bytes only as versioned extensions |
| handle 1..32640, 0 void | dense index 0..32639 | add `pair_handle = index + 1` |
| addr/canonical/mirror | `address_of`, `canonical_address`, `reverse16` | keep; test against rank mirror |
| rank / mirror rank | absent | add `rank=a*(511-a)/2+(b-a-1)`, `mirror_rank=32639-rank` |
| cycle flux | absent | use rail reserved u32 at +28 as Q16.16 |
Exact mapping: `addr(a,b)=(a<<8)|reverse8(b)`;
`canonical=min(addr, reverse16(addr))`; `mirror=reverse16(canonical)`. Add rank
but do not replace address mirror: `mirror_rank(rank)=32639-rank`;
`mirror_addr=reverse16(canonical_addr)`. They are different involutions; assert
they are never conflated. Pair cell = `heads_offset + head_region + index*64`;
`s+` at +0, `s-` at +32; rail reserved +28 holds Q16.16 flux; head reserved +28
stays zero unless a versioned extension index is added. Add `pair_handle`,
`pair_from_handle`, `rank_of_pair`, `mirror_rank`, `head_mirror`, `mirror_report`.

Add `fundamental_flux(root=0)` for 32,385 triangles:
`anti(i,j)=0.5*(w(i,j)-w(j,i))`;
`F_ij=anti(0,i)+anti(i,j)-anti(0,j)`; store `F` forward and `-F` reverse. This is
derived observable, not new information. Add `hodge_project()`: divergence, remove
vertex-gradient from the antisymmetric part, preserve symmetric part; O(256^2),
O(256) workspace, checkpoint or explicit maintenance only. Round 23's 1.032 ms
Hodge pass is one C++ container run at 8,388,608 B, not Android/Python. Python will
be slower; keep geometry, do not inherit timing. Native output must report
`execution_mode` and `native: true`; Python must not present its result as native.
A v2 128 B head / 256 B pair region needs a new format version and migration.

## 4. Round 24 integration: recursive typed hyperedges

`relations.py` has 304 B relations (32-byte content ids, three role leaves,
pair/mirror ids, source, confidence, generation, polarity, role chains) and 96 B
hyperedges (id, members offset, arity, common kind, flags). Default relation
members: subject, predicate, object, pair, mirror, relation. Missing: concept
kind, nested-edge kind, parent edge, reverse incidence, bounded traversal.

| Round 24 idea | Resident mapping | Change |
|---|---|---|
| stable concept id | `leaves.py` kind `concept`, 128-bit id | add `concept:<32hex>` alias and resolver |
| concept payload | leaf + payload arena | no duplicate token table |
| typed hyperedge | 96 B record + member arena | add kinds `concept=5`, `hyperedge=6`, `parent=7` |
| parent edge | parent member position 0 | no record growth; not semantic arity |
| incidence | absent | add 80 B incidence record in hyperedge arena |
| pair membership | member kind `pair` | use pair handle/digest, never pointer |
| concept parent | `leaves.parent`/child links | acyclic, depth <=32 |
| recursive traversal | absent | iterative `neighbors` with budgets |
| growable arena | fixed capacity | append-only; grow by profile migration |
Incidence record: append-only 80 B, 8-byte aligned. `member_id` 32 B at 0;
`hyperedge_id` 32 B at 32; `next_member` u32 at 64; `next_incidence` u32 at 68;
`kind` u8 at 72; `flags` u8 at 73; zero 6 B at 74. The hyperedge section gets
`member_id -> first_incidence` and `hyperedge_id -> first_member` hash tables;
first incidence is a u32 record index plus one, zero empty. No raw pointer persists.

A concept is a leaf with kind `concept`; its parent is a leaf slot. Unknown parent
is rejected; a parent cycle is `parent_cycle`; depth >32 is
`parent_depth_exceeded`. A parent hyperedge is a member kind `parent`, not semantic
arity, not returned unless `include_parents=True`. A hyperedge may contain
concepts, pairs, heads, other hyperedges and one parent; self/indirect parent cycles
are rejected at append. Deletion is a tombstone; records never move.

`neighbors(ref, max_hops=2, max_edges=64, max_members=512)` is iterative and returns
`{start, reached, edges_visited, members_visited, truncated, reason}`. Hard limits:
hops <=4, edges <=256, members <=2048; over-limit is `traversal_budget` with partial
evidence. Visited sets are content ids, not Python object identities. Round 24's
200,002-edge / 33.20 ms append and 54.96 ms incidence are C++ single-process
results; they justify shape, not Python/phone performance. The Python gate is
5,000 hyperedges and 5,000 incidence walks on the target phone, plus reopen and
parent-cycle rejection.

## 5. The learning loop
```text
corpus row -> validate/normalize/dedupe/split -> concept/observation leaf
-> 256-head sparse activation (head + mirror) -> typed relation or observation
-> hyperedge with pair/mirror/relation members -> pair rails + cycle flux
-> temporal next-token counts -> free-chat feedback (non-probe) -> checkpoint
```
1. **Corpus intake.** `nanlm.trainable.v1`; normalize; 2..4000 chars; dedupe by
   id and SHA-256; declared train/validation wins, else lowest 5% of
   `sha256(record_id)` moves to validation (at least one, never empty train).
2. **Stable leaves.** Concept id = first 16 bytes of
   `SHA-256("concept\\0" + casefold(term))`; observations keep text in the payload
   arena; relations create three role leaves plus evidence. All ids content-addressed.
3. **Sparse activation.** `head=fnv_head(term)`, `mirror=reverse8(head)`; add 1.0
   to head, 0.5 to mirror; decay 0.97 per token; max 32 heads per step; never scan
   or update all 32,640 pairs.
4. **Typed relations.** Explicit `X is Y`, `X has Y`, `X can Y`, `X causes Y` and
   bounded intransitive forms; unmatched text is a neutral observation, never an
   invented label. Store polarity, confidence and source digest.
5. **Pairs/cycles.** Co-occurring heads within 4 tokens update the canonical rail by
   `confidence * 0.5^distance`; at checkpoint recompute 32,385 fundamental fluxes;
   Hodge only on demand.
6. **Next token.** `P=0.12*P1+0.33*P2+0.55*P3`; objective is maximum-likelihood
   count estimation; reported loss is mean NLL. No gradient descent or optimizer.
7. **Free chat.** Non-probe parses, merges supported/refuted relations, observes in
   the layered learner and saves. Probe writes nothing (no merge/observe/scratch/
   transcript/checkpoint). Correction lowers `output_drive` by 0.85 and records the
   rejected reply; no gradient retrain.
8. **Checkpoint.** Only committed generation visible; receipt has generation,
   digest, counts, durable flag and `canonical_commit: false`; reopen matches.

| Objective | Current implementation | Status |
|---|---|---|
| Next-token NLL | `mean NLL=-(1/N)sum log(max(P,1e-9))`; `ppl=exp(mean NLL)`, display cap `exp(50)` | implemented, measured |
| Pair association | additive Hebbian `w_ab += eta*a_i*a_j`; rails accumulate signed weight/amplitude | cells implemented, not wired |
| Relation strength | evidence confidence; polarity separate; duplicate id idempotent | partial |
| Cycle consistency | `L_cycle=mean(F_ij^2)` over cached fundamental fluxes; diagnostic | target |
| Free-chat correction | reject last reply, `output_drive *= 0.85`, record rejected | implemented, heuristic |
| Typed operator (future) | cross-entropy over predicate logits with L2 on held-out split | design-only |
This is count/association learning; lower NLL is next-token fit, not understanding.
The 14/14 curve is not proof: a read-only audit of a different direct run found
keyword 12/14 but direct-exact 10/14, and old probes mutated state. Future runs use
`probe=True`, fixed held-out prompts and per-stage state fingerprints.

Splits: use declared splits; otherwise the deterministic 5% fallback is allowed and
logged. Fix one sample by `sha256("eval:"+record_id)`, take first
`--max-eval-records`, write ids/count/digest; the measured 1,000-record sample
scored 28,521 tokens. Report per-source NLL/perplexity. Held-out text never goes to
a teacher; teacher prompts use bounded train samples and report
`heldout_text_sent: false`. Novel generation uses `probe=True`; judge labels are
diagnostic, copy detection uses train-only text.

Escape: model output never chooses a path, module, shell, `eval`/`exec` string or raw
pointer; resident paths derive from `state_dir`; unknown schemas denied. A model may
propose a relation; the parser and authorization policy decide, and a model cannot
approve its own promotion. `merge_enabled` defaults false; promotion needs an
operator, `approved=true`, matching `expected_generation`, policy version and audit;
canonical merge additionally needs an absent external gate. Resident writes are
local-derived/append-only: no in-place rewrite, code mutation, self-hosted install or
network side effect. Bounded batch/payload/traversal/top-k; cycles, unknown kinds,
missing leaves and digest mismatches fail closed. Chat text never becomes a system
instruction, tool authorization or scope bypass.
## 6. Point-by-point TODO

`done` = code and local test exist; `partial` = some code/test but acceptance not
met; `open` = no usable implementation.
| ID | Work item | Files | Depends | Acceptance test | Failure/rollback | Status |
|---|---|---|---|---|---|---|
| T01 | Freeze v1 format; replace stale 12-section doc table with measured 7-section table | `docs/UNIFIED_RESIDENT_UNIVERSE.md` | none | doc offsets/capacities equal `status()` on 256 MiB and 1 GiB; no 4 MiB pair claim | revert doc; no data change | partial |
| T02 | Add one-writer `flock`, expected-generation CAS, read-only snapshot | `resident/universe.py`, `persistence.py` | T01 | two writers: one commit, other `writer_busy`/`stale_generation` | release lock; old header valid | open |
| T03 | Add generation-filtered commit journal and torn-tail recovery | `universe.py`, `persistence.py` | T02 | kill after data/journal/header; reopen old or new complete; `journal_torn_tail` | keep previous header/`.prev`; no reset | open |
| T04 | Add `.prev`, checkpoint fallback and rollback as new generation | `persistence.py`, `universe.py` | T03 | corrupt newest -> older slot; rollback digest matches; old retained | restore `.prev`; refuse if both invalid | partial |
| T05 | Reject >1 GiB; CLI reports clamp; 2 GiB canary denial test | `resident/__init__.py`, `universe.py`, `export.py` | T01 | 2 GiB open denied; canary never opened; RSS unchanged | close windows; budget error | partial |
| T06 | Add `madvise`, bounded 4 MiB `mlock`, 24 MiB feature cache, RSS telemetry | `universe.py`, `resonance.py` | T02 | 10k run RSS delta <=64 MiB; no `MAP_POPULATE` | disable mlock; bounded pread/pwrite | open |
| T07 | Add pair handle, rank mirror, head mirror report, reserved-word flux | `cells.py` | T01 | 65,280 views; rank involution; mirrors not conflated | keep reserved zero; no format change | open |
| T08 | Add fundamental-cycle and Hodge maintenance in Python | `cells.py`, `resident_benchmark.py` | T07 | gauge-shift flux delta <=1e-3 Q16.16; divergence reduced; no per-token call | disable maintenance; old rails remain | open |
| T09 | Decide optional v2 8 MiB cells only by measured need | `layout.py`, `universe.py` | T07,T08 | v1->v2 export/import preserves ids/digests; no in-place reuse | keep v1; abandon v2 | open |
| T10 | Add member kinds concept/hyperedge/parent; keep 6-member default | `relations.py`, `ingest.py` | T01 | concept resolves to leaf; nested edge accepted; parent not arity | reject unknown kind; no record growth | open |
| T11 | Add 80 B incidence records and member-id index | `relations.py`, `universe.py` | T10 | pair/head/concept incidence complete; reopen preserved; clean refusal | keep edge list; rebuild incidence | open |
| T12 | Add bounded iterative traversal API | `relations.py`, `universe.py`, CLI | T11 | 2-hop penguin->bird->animal; limits enforced; cycle rejected | partial + `traversal_budget`; no hang | open |
| T13 | Add concept registry and leaf-ref resolver | `leaves.py`, `universe.py` | T10 | unknown leaf denied; parent/child correct; duplicate alias retained/refused | keep old leaves; no implicit creation | partial |
| T14 | Fix per-role head derivation; relation leaves must not share one head | `ingest.py`, `universe.py`, tests | none | relation creates >=2 active heads; pair weight nonzero after ingest | revert head field; no pair claim | open |
| T15 | Add paged corpus intake with progress, 256-row compatibility, dedupe | `ingest.py`, `corpus_training.py` | T14 | 10,000 rows accepted; duplicates <5%; generation <= records | resume last committed batch | partial |
| T16 | Add 10k resident ingest gate and capacity telemetry | `resident_universe_cli.py`, tests | T15 | counts match receipts; no capacity exceeded | checkpoint/stop at 90% | open |
| T17 | Wire `TemporalMirrorLM.observe`/`observe_memory_rows` to resident ingest | `universe.py`, `temporal_lm.py`, `chat.py`, `resident/learning.py` | T15 | corpus turn updates model + resident; probe does not | disable resident write; keep model checkpoint | partial |
| T18 | Implement sparse activation and top-k pair updates | `universe.py`, `cells.py`, `resident/learning.py` | T14,T17 | <=32 heads and <=128 pair updates per sentence; no full scan | skip pair update; count remains | partial |
| T19 | Route parser output to resident relations/hyperedges | `chat.py`, `layered_learning.py`, `ingest.py` | T10,T17 | `Dogs bark.` creates supported relation; question none; probe no write | fall back to memory ledger | partial |
| T20 | Feed bounded pair/cycle summary into next-token rescoring | `temporal_lm.py`, `resonance.py`, `resident/learning.py` | T08,T18 | NLL not worse than 2.60; disabled path equals baseline | disable flag; restore baseline | partial |
| T21 | Make free-chat feedback/correction explicit; never train on probe | `chat.py`, `layered_learning.py`, `temporal_lm.py` | T17 | probe fingerprint unchanged; non-probe changes; correction recorded | probe mode only; no write | partial |
| T22 | Checkpoint resident + temporal + layered as one receipt (not atomic) | `universe.py`, `temporal_lm.py`, `chat.py` | T03 | reopen restores generation/digest and model counts; names files | keep last valid; report `partial_commit` | partial |
| T23 | Standardize split, fixed eval ids, per-source report | `deep_train_corpus.py`, `corpus_training.py` | T15 | 1,000-record digest stable; no validation in train; per-source table | rerun from checkpoint | partial |
| T24 | Keep teacher to bounded train samples; never send held-out | `deep_train_corpus.py`, teacher scripts | T23 | `heldout_text_sent:false`; max 24 samples; fallback plan | deterministic fallback | partial |
| T25 | Keep judge diagnostic; record errors; no ground truth | `corpus_generation_eval.py` | T23 | judge errors counted; no LLM-written reply | omit judge; deterministic metrics | partial |
| T26 | Add MTH/N16I 256-wide scoring path with explicit receipt | `resonance.py`, `nan0s_runtime.py`, `mth_leaves.py` | T07 | dot/cosine match native within 1e-5; mode `n16i_guest_interpreter`, `native:false` | disable MTH; labelled Python only | partial |
| T27 | Publish per-kernel ABI; 4 typed, 87 fixture-only; refuse others | `mth_catalog.py`, docs | T26 | `abi` field; refusal code; no Python fallback | keep fixture-only status | partial |
| T28 | Bind permanent HW2/node 3/BS0:0; pin router receipts | `nannet_hw2_connect.py`, `router_universe_client.py` | auth | fixed reservation; reject point 4/256, heads 256, foreign, stale | release dynamic lease; no relabel | partial |
| T29 | Require scopes for resident tools; local-unverified without authority | `__init__.py`, `auth_tools.py`, `plugin.yaml` | T30 | scope table; unpaired write denied; read scope enforced | disable tool; stable code | done |
| T30 | Register resident open/status/ingest/query/link/hyperedge/checkpoint tools | `plugin.yaml`, `__init__.py`, `TOOLS_REFERENCE.md` | T02 | status/ingest/query registered; remaining four need manifest/register/demo equality | unregister extras; keep CLI | partial |
| T31 | Extend CLI hyperedge/traversal/learning status; paginate beyond 256 | `resident_universe_cli.py` | T12,T15 | 10k via bounded pages; bounded JSON; no path traversal | keep 256 mode; explicit error | partial |
| T32 | Add real resident backend option to benchmark; keep synthetic labelled | `resident_benchmark.py` | T02,T06 | create/ingest/query/reopen/RSS on real package | keep synthetic; no resident claim | partial |
| T33 | Synchronize tool/doc counts and claims | `README.md`, `TOOLS_REFERENCE.md`, `CONNECTION_AUDIT.md` | T30 | manifest count = documented count; no stale 33/34/59/72 | freeze docs at measured count | partial |
| T34 | Crash/fault injection at header, journal, record, checkpoint | `tests/test_resident_recovery.py`, persistence | T03 | each kill yields old/new complete; no half record served | restore `.prev`; fail closed | open |
| T35 | Full-suite and packaging gate with reproducible counts | CI, `MANIFEST.in`, tests | all | manifest: Python, test, tool, kernel counts | publish only tested archive | partial |
| T36 | Test merge-disabled matrix for all promotion paths | `layered_learning.py`, `__init__.py`, `plugin.yaml` | T29 | `approved=true` returns `merge_disabled`, promoted 0, no file change | keep false; audit event | partial |
| T37 | Keep quantum bridge/matrix read-only diagnostics; no learner dependency | `quantum_bridge.py`, docs | none | all claims false; no runtime caller outside scripts/tests | disable bridge; no state change | partial |
| T38 | Ship session docs/package data for resident format | `session_docs/`, `MANIFEST.in`, `pyproject.toml` | T30 | clean wheel/sdist contain docs/tests; import works | exclude; document | partial |
| T39 | Enforce concept taxonomy acyclicity, depth <=32, tombstones | `leaves.py`, `relations.py` | T13 | cycle/depth denied; tombstone skipped; reopen stable | reject append; old tree remains | open |
| T40 | Record NANNET production evidence separately; never call loopback production | `NANNET_HW2_PRODUCTION.md`, `LIVE_SMOKE.md` | T28 | deployment ID, issuer fingerprint, command digest, date; missing -> open | mark unverified; no production claim | open |
## 7. Staged milestones M0..M8
| Milestone | Scope | Measurable gate |
|---|---|---|
| **M0 Truth freeze** | Read repo; classify measured/artifact/target; touch no other file | 191 resident/field tests; this plan only new file; full-suite log archived |
| **M1 Create/reopen** | One 256 MiB file, section table, close, reopen | `phone-256`, 268,435,456 B, 123,470,080 committed, same gen/digest, 7 sections, no 2 GiB |
| **M2 Corpus ingest** | 10,000 trainable rows in bounded batches | accepted >=9,500; duplicates <=5%; leaves/features >=9,500; gen <= records; no capacity error |
| **M3 Query** | Text/vector resonance; top-k vs brute force; read-only | 100 queries recall@10=1.0 vs scan; p95 <=250 ms at 16,384 features; probe digest unchanged |
| **M4 Traversal** | Typed relation + 6-member hyperedge + concept chain + parent + incidence | 2-hop `penguin`->`animal`; incidence counts match; limits enforced; parent cycle rejected; reopen preserved |
| **M5 Learning** | Deep corpus fixed split/sample | NLL <=2.60, ppl <=13.5 (measured 2.513986/12.3541); per-source; no selected epoch >baseline +1% |
| **M6 Generation** | Native `probe=True`, 8 prompts, optional diagnostic judge | 0 empty; novel-3-gram >=0.18 (measured 0.1825); copied <=2; NLL reported; no LLM reply |
| **M7 Budget/RSS** | 256 MiB, 10k rows, 1 GiB ceiling, 2 GiB canary | file <=256 MiB; RSS delta <=64 MiB; mlock <=4 MiB; 2 GiB denied; canary untouched |
| **M8 Tests/merge-off** | Full suite, tools, CLI, docs, packaging, merge | all tests pass; `canonical_merge_enabled=false`; every promotion `merge_disabled`; counts equal; wheel/sdist import |
M0-M2 are partly measured. M3-M8 are targets until backing TODO rows close. Do not
pass a milestone on one happy-path command.
## 8. Risks and denial rules

1. **Canonical merge stays off.** No default `canonical_commit:true` or
   `merge_enabled:true`; no six-runner claim. Resident state is local-derived.
2. **No raw pointers.** Never persist Python object ids, mmap addresses, `ctypes`
   pointers, vector addresses or handles crossing a file boundary; use 128-bit
   leaf ids and 32-byte content digests.
3. **No uncontrolled self-modification.** No model-authored path, module, shell,
   `eval`/`exec`, plugin install, network call or approval. Promotion needs an
   operator, generation CAS and audit record.
4. **No 2 GiB overbootstrap.** 1 GiB hard ceiling; never copy a "hot part", open a
   canary, or count sparse file size as resident memory.
5. **No false performance/quantum/MTH/NANNET claim.** Python mirrors are not the
   C++ prototype; gauge/cycle/phase/spectral is classical math and bridge flags
   stay false; only four MTH kernels have typed operands (87 fixture-only); loopback
   pairing is not production, and HW2/node 3/issuer/MAC evidence remains open.
6. **No judge-as-truth.** Gemma labels are diagnostic; held-out text is not sent;
   the 14/14 curve is confounded; report keyword and direct-exact separately.
7. **Test these risks:** concurrent writer/partial record; header/journal digest
   mismatch; pair-head collapse (relation leaves currently share one head); O(N)
   feature scan at 16k; capacity exhaustion; chat contamination; weak novelty;
   doc/code count drift; sparse-file page faults; Android `mlock` refusal; missing
   compiler/awk.
8. **Denial codes:** `merge_disabled`, `canonical_merge_disabled`,
   `overbootstrap_copy_denied`, `writer_busy`, `stale_generation`, `cas_required`,
   `state_digest_mismatch`, `no_valid_checkpoint`, `journal_torn_tail`,
   `capacity_exceeded`, `traversal_budget`, `unknown_schema`, `raw_pointer_denied`,
   `self_modification_denied`.

## 9. Exact commands to reproduce each gate
```sh
export REPO=/data/data/com.termux/files/home/NaNLMind_HERMES
export GATE=/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate
rm -rf "$GATE" && mkdir -p "$GATE" && cd "$REPO"
```
**M0 tests**
```sh
python3 -m unittest discover -s tests -p 'test_resident*.py' -v
python3 -m unittest tests.test_n16i_unification -v
python3 scripts/resident_universe_cli.py --help
```
**M1 create/reopen**
```sh
python3 scripts/resident_universe_cli.py init --path "$GATE/universe.nlmres" --size 256MiB
python3 scripts/resident_universe_cli.py status --path "$GATE/universe.nlmres"
python3 -c 'from nanlmind_hermes.resident.universe import open_universe as o; p="/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/universe.nlmres"; u=o(p); a=(u.generation,u.digest,u.status()["profile"],u.status()["committed_bytes"]); u.close(); u=o(p); b=(u.generation,u.digest,u.status()["profile"],u.status()["committed_bytes"]); u.close(); assert a==b and a[2:]==("phone-256",123470080), (a,b); print(a)'
```
**M2 10k bounded ingest**
```sh
python3 - <<'PY'
import json; from pathlib import Path; from nanlmind_hermes.resident.universe import open_universe
rows=[]
for p in sorted(Path("experiments/corpus/trainable").glob("*.jsonl")):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
        if len(rows)>=10000: break
    if len(rows)>=10000: break
u=open_universe("/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/universe.nlmres"); acc=0
for i in range(0,len(rows),256): acc+=u.ingest_rows(rows[i:i+256])["accepted"]
s=u.status(); u.close(); print({"accepted":acc,"counts":s["counts"],"generation":s["generation"]}); assert acc>=9500 and s["counts"]["leaves"]>=9500 and s["counts"]["features"]>=9500
PY
```
**M3 query/digest**
```sh
python3 scripts/resident_universe_cli.py query --path "$GATE/universe.nlmres" --text "water flows in a river" --top-k 10
python3 -c 'import time; from nanlmind_hermes.resident.universe import open_universe as o; u=o("/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/universe.nlmres"); d=u.digest; ts=[]
for _ in range(100): t=time.perf_counter(); q=u.query("water river stone",top_k=10); ts.append((time.perf_counter()-t)*1000); assert len(q["hits"])<=10
assert u.digest==d; ts.sort(); print({"p50_ms":round(ts[50],3),"p95_ms":round(ts[95],3)}); u.close()'
```
**M4 relations/hyperedges**
```sh
python3 - <<'PY'
import json; from nanlmind_hermes.resident.universe import open_universe
u=open_universe("/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/universe.nlmres")
for s,p,o in (("penguin","is_a","bird"),("bird","is_a","animal")): u.append_relation({"subject":s,"predicate":p,"object":o,"confidence":0.8,"polarity":"supports"})
r=u.relations.relations(limit=2); assert len(r)==2 and all(x["hyperedge_id"] for x in r)
for x in r: assert len(u.relations.get_hyperedge(x["hyperedge_id"])["members"])>=6
s=u.status(); print(json.dumps({"relations":s["counts"]["relations"],"hyperedges":s["counts"]["hyperedges"]})); u.close()
PY
```
**M5 learning curve**
```sh
python3 scripts/deep_train_corpus.py --trainable-dir experiments/corpus/trainable --state-dir "$GATE/corpus-state" --epochs 3 --max-eval-records 1000 --output "$GATE/deep-training.json"
python3 -c 'import json; s=json.load(open("/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/deep-training.json"))["selected"]; print(s["epoch"],s["heldout"]["mean_negative_log_likelihood"],s["heldout"]["perplexity"]); assert s["heldout"]["mean_negative_log_likelihood"]<=2.60 and s["heldout"]["perplexity"]<=13.5'
```
**M6 native generation**
```sh
python3 scripts/corpus_generation_eval.py --state-dir "$GATE/corpus-state" --corpus experiments/corpus/trainable --nll --output "$GATE/generation.json"
python3 -c 'import json; s=json.load(open("/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/generation.json"))["summary"]; print(s); assert s["empty_responses"]==0 and s["mean_novel_3gram_ratio"]>=0.18 and s["copied_training_sentences"]<=2'
```
**M7 budget/RSS/2 GiB denial**
```sh
python3 -c 'import os,resource; from nanlmind_hermes.resident.universe import open_universe as o; p="/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/universe.nlmres"; b=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; u=o(p); q=u.query("water river",top_k=10); a=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; print({"file_bytes":os.path.getsize(p),"rss_delta_kb":max(0,a-b),"hits":len(q["hits"])}); u.close(); assert os.path.getsize(p)<=268435456'
python3 -c 'from nanlmind_hermes.resident.universe import open_universe as o
try: o("/data/data/com.termux/files/usr/tmp/opencode/hermes-total-plan/gate/too-big.nlmres",create=True,size=2*1024*1024*1024)
except Exception as e: print(type(e).__name__,e)
else: raise AssertionError("2 GiB request was not denied")'
```
**M8 full suite/merge-off/package**
```sh
python3 -m unittest discover -s tests
python3 scripts/resident_universe_cli.py status --path "$GATE/universe.nlmres" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["canonical_merge"]["enabled"] is False; print("merge-off ok")'
python3 -m build && python3 -m twine check --strict dist/*
```
**Reference-only C++ gates (outside repo; not Android/Python)**
```sh
cd /data/data/com.termux/files/usr/tmp/opencode/round2324
g++ -std=c++20 -O3 -Wall -Wextra mirror_object.cpp -o mirror_test && ./mirror_test
g++ -std=c++20 -O3 -Wall -Wextra nan_mirror_round024/recursive_memory.cpp -o recursive_memory && ./recursive_memory 50000
```
Done means M0-M8 pass on one named commit/device; the full suite and
package build are reproducible; resident tools have scopes and receipts; learning
writes only local-derived state with `canonical_commit: false`; and a reader can
reproduce the corpus NLL, query latency, traversal, generation metrics and memory
ceiling from these commands. A green probe curve alone is not completion.

## 10. Head-field addendum (Round 26)

The next integration layer makes all 256 heads independently active and then
couples them as a field. It is deliberately **scratch-first**:

1. Preserve the mandatory 16-function Boolean foundation; no new field path may
   replace, weaken or bypass `ensure_boolean_foundation`.
2. Give each head an independent Boolean rule/score/activation record. A head
   may diverge from the field; the field is a weighted aggregate, not a
   replacement for per-head state.
3. Give every head its own L0-L9 scratch layers. A write to any layer is
   scratch-only until the gate approves a main commit.
4. Stage every self-modification as a proposal with a token, target head,
   layer, operation, value, reason and expected generation.
5. Evaluate the proposal against Boolean invariants and denial rules before any
   main write. Contradiction, unknown head/layer, oversized value, duplicate or
   stale token, and missing approval must deny without mutation.
6. Commit only with literal `approved=True` and an exact generation CAS. The
   commit writes the main head register once and advances the generation.
7. Keep field coupling additive through pair cells; never rewrite permanent
   head identity or `head_id`, and never alias two heads into one state.
8. Feed corpus and free-chat activation through many heads: hash/sparse
   activations per token/relation, per-head losses, field entropy, active-head
   count and collision rate. A single global head activation is a failure.
9. Persist the field snapshot and per-head scratch payloads through the
   resident leaf/relation API; reload must compare generation and digest.
10. Expose bounded status/activate/propose/commit/query tool handlers; all
    canonical merge/promotion flags remain false.
11. Benchmark independent-head mode against one-head and uniform-field
    controls; report active-head distribution, entropy, collision rate, time,
    ops/sec and memory. Never claim native/phone performance for Python.
12. Preserve rollback: denied or failed proposals leave main state byte-for-byte
    unchanged; accepted commits have a before/after receipt.

### Current measured resident learning result

- Corpus: 14,039 train records, 500 held-out records, 13,655 learned exposures
  after control/duplicate filtering.
- Typed relation extraction: **2,522 relations / 2,522 hyperedges** from explicit
  definition prose; no relation is treated as independently fact-verified.
- Held-out before -> epoch 1: temporal NLL **6.7254 -> 2.7400** (ppl
  **833.3 -> 15.49**), resident pair NLL **5.5953 -> 3.4160** (ppl
  **269.1 -> 30.45**), hybrid NLL **5.8347 -> 2.5995** (ppl
  **341.96 -> 13.46**).
- Resident state: 19,858 leaves, 2,522 relations, 2,522 hyperedges, 19,602
  features, generation 19,709, local digest
  `82786994c686ae18318cfde6decbb493dccf9816b456b7643825ec524e7e32a4`.
- A repeated identical epoch adds no new exposure because the temporal replay
  guard has already seen it; the selected checkpoint is epoch 1.
- This is measured on the Termux/aarch64 host, not an Android benchmark claim.

### Round-26 implementation status (2026-11-08)

Implemented and tested as separate, non-overlapping modules:

| layer | file | measured |
|---|---|---|
| Boolean head field | `nanlmind_hermes/head_boolean_field.py` | all 16 functions, exact XOR/AND/OR base |
| resident head adapter | `resident/head_field.py` | independent 256-head scores, field entropy/coupling |
| L0-L9 scratch | `resident/scratch_layers.py` | 256 heads x 10 layers, scratch-only writes |
| approval/CAS gate | `resident/selfmod_gate.py` | approval, Boolean rules, stale-generation denial |
| field learning | `resident/field_learning.py` | 16-64 active heads per sentence; proposal acceptance measured |
| field persistence | `resident/field_persistence.py` | approved main snapshot + versioned per-head scratch reload |
| field tools | `resident/field_tools.py` | status/activate/propose/commit/query, 79th plugin tool |
| MTH bridge | `resident/mth_head_bridge.py` | all 256 heads, all 16 ops, real N16I preference |
| benchmark | `scripts/head_field_benchmark.py` | independent/coupled/single/uniform controls |
| integration tests | `tests/test_resident_field_integration.py` | 14/14 pass |

Plugin behavior now: accepted free-chat exposure computes a bounded activation
map over many heads, stages it through the field learner's scratch adapter,
evaluates bounded invariants, and only then applies accepted activation/coupling
to the resident cells. `nanlm_head_field` stages explicit proposals in the
per-head `head % 10` scratch layer and refuses a main commit without
`approved=true`, the Boolean base, and exact generation CAS. Single-head
activation remains available but is not the default corpus/chat behavior.
