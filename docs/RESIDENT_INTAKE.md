# Resident intake: local training to bounded candidate bundles

Status: **design/proposed contract**. There is no intake implementation in this
repository. `nanlmind_hermes/resident/__init__.py` contains constants only.
This document does not report a live upload, a test run, an accepted bundle or
a canonical merge. Existing `merge_enabled` behavior is documented as the
current closed gate; it is not evidence that intake exists.

The intake layer is an **optional** staging boundary. HERMES trains a local
universe first; a user or operator may then choose to export a bounded candidate
bundle. Nothing is uploaded automatically. Acceptance means “structurally valid
candidate evidence integrated into staging”, not “true”, “trained” or
“canonical”.

## 1. Trust path

```text
HERMES local process
  resident universe (universe.nlmres, local-derived)
        |
        |  train locally: relation-train / layered / MTH / chat evidence
        v
  local generation G, checkpoint, local receipt
        |
        |  explicit export (opt-in, bounded, no 2 GiB map)
        v
  candidate bundle (manifest + typed records + optional corpus/evidence)
        |
        |  HTTPS upload, scoped token, idempotency key
        v
  intake layer (candidate staging, separate owner)
        |
        |  validate manifest/generation/digests/references
        |  dry-run plan -> all-or-nothing apply
        v
  accepted candidate projection + training corpus/evidence
        |
        |  canonical merge: OFF
        |  merge_enabled=false
        |  six_runner_barrier=absent
        v
  no canonical BS0 write
```

The intake layer may be a NaNLMind/NaNROUTER service or a local test service.
Its exact deployment is outside this repository. The contract is the same:
acceptance cannot promote canonical state.

## 2. What HERMES trains and what may be exported

Local training may produce:

- typed relations (subject, predicate, object, polarity, confidence,
  resolution L0–L9, source digest);
- immutable leaves and hubs (lexical, evidence, function/data leaves);
- pair cells, mirror links and hyperedges;
- evidence rows with provenance and contradiction retention;
- optional corpus chunks (N16TXT/1 text or N16REL/1 rows);
- local generation, checkpoint digest and state digest.

Export is **not** a copy of the resident mmap file. The bundle contains only
bounded logical records. It never contains:

- the 8 MiB chart page as a raw dump (the intake layer validates logical
  records; it does not need the native page);
- the 2 GiB overbootstrap map;
- raw native pointers, open file descriptors, shell commands or process
  memory;
- credentials, private keys, bearer tokens or absolute local paths;
- data from another universe or installation.

Default export mode is `relations_and_evidence`: typed relations, leaves,
hyperedges and evidence. Raw corpus text is included only with
`include_corpus=true` and explicit user approval. Corpus export requires a
license/consent declaration.

## 3. Bundle container

A bundle is one deterministic container file, media type
`application/vnd.nanlm.resident-bundle.v1`, optional HTTP `Content-Encoding:
gzip`. The container is uncompressed for digest purposes; gzip is transport
only.

```text
NLMCSB01 header (4 KiB)
  magic "NLMCSB01", version u32, header_bytes u32, flags u16,
  section_count u16, container_digest 32, manifest_digest 32,
  created_ns u64, reserved
section table (16 entries x 80 B)
section payloads, 64-byte aligned
```

Section entry:

| field | type | meaning |
|---|---|---|
| `id` | u16 | 1–16 |
| `flags` | u16 | required/compressed/reserved |
| `offset` | u64 | absolute offset in the container |
| `length` | u64 | payload length |
| `record_size` | u32 | fixed record size or 0 for JSONL |
| `count` | u32 | records/lines |
| `digest` | 32 B | SHA-256 of the section payload |
| `name` | 16 B | e.g. `manifest`, `leaves` |
| `reserved` | 4 B | zero |
| `reserved` | 4 B | zero |

Bundle sections:

| ID | section | encoding |
|---:|---|---|
| 1 | `manifest` | canonical UTF-8 JSON, ≤1 MiB |
| 2 | `leaves` | canonical JSONL, one leaf per line |
| 3 | `relations` | canonical JSONL, one relation per line |
| 4 | `hyperedges` | canonical JSONL, one hyperedge per line |
| 5 | `tokens` | canonical JSONL |
| 6 | `features` | canonical JSONL: 256 finite float32 values |
| 7 | `corpus` | optional N16TXT/1 or N16REL/1 chunks |
| 8 | `evidence` | canonical JSONL: source-bound evidence rows |
| 9 | `receipts` | optional prior local receipts, canonical JSONL |
| 10–16 | reserved | must be empty |

`container_digest` is computed over the header with the `container_digest`
field zeroed, the complete section table, and every section payload.
`manifest_digest` is the SHA-256 of the canonical manifest section bytes.
Neither is computed over a field that contains itself.

Canonical JSON means sorted keys, no insignificant whitespace, UTF-8, no NaN or
Infinity, and integers/digests as strings where the schema requires it. JSONL
lines are not reordered; `record_digest` is computed over the canonical line
without the digest field.

Portable references use `leaf:<sha256>`, `rel:<sha256>`, `he:<sha256>` or a
32-byte digest. Local `u32` handles from the resident file are forbidden in a
bundle. This is what makes a bundle portable across files and generations.

## 4. Manifest

The manifest is the authority for bounds, generation and policy. Required
fields:

```json
{
  "schema": "hermes.resident.candidate/1",
  "bundle_id": "<32 hex>",
  "created_utc": "2026-11-08T00:00:00Z",
  "installation_id": "<opaque installation id>",
  "universe_id": "<opaque local universe id>",
  "profile": "phone-256",
  "local_generation": 42,
  "base_generation": 41,
  "parent_digest": "<64 hex>",
  "state_digest": "<64 hex>",
  "checkpoint_digest": "<64 hex>",
  "section_digests": {"leaves": "...", "relations": "...", "hyperedges": "..."},
  "counts": {
    "leaves": 30, "relations": 12, "hyperedges": 12,
    "tokens": 22, "features": 22, "evidence": 12, "corpus_chunks": 0
  },
  "resolutions": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0,
                   "5": 12, "6": 0, "7": 0, "8": 0, "9": 0},
  "splits": {"train": 10, "validation": 1, "test": 1},
  "source_digests": ["<64 hex>"],
  "license": "operator-declared identifier",
  "consent": "operator-declared consent record",
  "include_corpus": false,
  "operations": ["append_leaf", "append_relation", "append_hyperedge"],
  "no_canonical_claim": true,
  "merge_enabled": false,
  "canonical_merge_enabled": false,
  "six_runner_barrier": "absent"
}
```

`section_digests` covers every present non-manifest section. The manifest's own
digest is the header's `manifest_digest`, never a field inside the manifest
(that would be circular).

Additional required fields for corpus export: `corpus_manifest_digest`,
`corpus_license`, `corpus_consent`, `corpus_split_policy`. The manifest must
not contain secrets or absolute paths.

`bundle_id` is an opaque 128-bit identifier generated by the producer.
`manifest_digest` binds the manifest bytes; `container_digest` binds the whole
container. Reusing a `bundle_id` with different bytes is a conflict, not an
update.

The manifest's `state_digest` and `checkpoint_digest` are **claims about the
local state**. Intake can verify that they are well-formed and bound by the
manifest digest, but without the resident file it cannot recompute the local
mmap state. If an installation attestation is configured, the manifest may
carry an Ed25519 signature over `manifest_digest`; otherwise the bundle is
marked `unattested`. An unattested bundle is accepted only by an explicitly
configured local/test policy. It is never described as cryptographically
proven.

## 5. Bounds

Default bounds (hard limits in parentheses):

| item | default | hard |
|---|---:|---:|
| container bytes | 32 MiB | 128 MiB |
| decompressed bytes | 128 MiB | 512 MiB |
| compression expansion ratio | 8:1 | 8:1 |
| manifest | 1 MiB | 1 MiB |
| leaves | 32,768 | 65,536 |
| relations | 65,536 | 131,072 |
| hyperedges | 16,384 | 32,768 |
| tokens | 16,384 | 32,768 |
| features | 16,384 | 32,768 |
| evidence rows | 8,192 | 16,384 |
| corpus chunks | 4,096 | 8,192 |
| single corpus chunk | 64 KiB | 256 KiB |
| single JSONL line | 16 KiB | 64 KiB |
| total payload bytes | 32 MiB | 128 MiB |
| any one section | 64 MiB | 256 MiB |

`phone-128` export uses half of the default limits. The intake layer applies
the **smaller** of the manifest profile limit and the server limit. A
decompressor must enforce the limit while streaming; it must not allocate the
declared uncompressed size first.

### 5.1 Working-set budget

These are design targets for validation/apply, not measured results:

| scope | target | hard cap |
|---|---:|---:|
| one streamed chunk | 1 MiB | 4 MiB |
| manifest in memory | 1 MiB | 1 MiB |
| one JSONL line | 16 KiB | 64 KiB |
| validation working set | 32 MiB | 64 MiB |
| per-bundle temporary spool | 64 MiB | 128 MiB |
| concurrent applies per universe | 1 | 1 |

The validator streams sections to a temporary spool, computes digests while
reading, and never holds the whole decompressed bundle plus a second copy. A
bundle that would exceed the working-set budget is rejected with
`bundle_too_large`; the server does not rely on the garbage collector to recover
from an oversized allocation.

## 6. Validation order

Validation is deterministic and stops at the first fatal error. Every error is
a stable code; diagnostics never include secret material or raw corpus text.

1. **Transport:** HTTPS, content type, content length, authenticated scoped
   token, no redirect to a non-HTTPS origin, no query-string secrets.
2. **Container:** magic, version, header size, section count, offsets,
   alignment, no overlap, `offset + length <= total`, all required sections
   present exactly once.
3. **Decompression:** streaming bound and expansion ratio.
4. **Manifest:** schema, required fields, canonical JSON, count/type bounds,
   at least one substantive record (an empty bundle is `empty_bundle`),
   resolution 0–9, splits, license/consent, `no_canonical_claim=true`,
   `merge_enabled=false`, `canonical_merge_enabled=false`,
   `six_runner_barrier="absent"` unless a separately configured future gate is
   explicitly enabled.
5. **Digests:** `container_digest`, `manifest_digest`, each `section_digest`,
   each `record_digest`; reject any mismatch before applying.
6. **Generation:** `local_generation >= base_generation`;
   `base_generation` equals the intake's accepted generation or a valid
   contiguous rebase proof exists; `parent_digest` matches
   `accepted_local_digest`; no future generation.
7. **References:** every relation resolves its subject/predicate/object leaf;
   every leaf resolves its payload/hub; every pair rank/address and mirror
   exists in the declared range; every hyperedge member resolves; no
   cross-universe or cross-installation reference.
8. **Typed semantics:** polarity is supports/refutes/neutral; confidence is
   finite in [0,1]; relation resolution is L0–L9; no non-finite vector;
   feature length exactly 256; no duplicate content ID with different bytes;
   per-section `counts` match the parsed records; the resolution histogram sums
   to the relation count; split counts sum to the relation/corpus rows.
9. **Policy:** license/consent present where required; test-split rows are not
   relabeled as train; no `2 GiB` object; no canonical/promotion request; no
   forbidden pointer/shell field.
10. **Idempotency:** same `bundle_id` + same digests returns the prior receipt;
    same `bundle_id` + different bytes returns `bundle_id_conflict`.
11. **Dry-run plan:** counts, duplicates, conflicts, projected new digest,
    projected corpus bytes. No write in dry-run.

`state_digest`/`checkpoint_digest` are validated for shape and binding, not
recomputed from the local resident file. This limitation is explicit in the
receipt (`local_state_recomputed: false`).

## 7. Mathematical and corpus integration

Acceptance has two effects, both in the candidate staging scope:

### 7.1 Mathematical integration

- Append immutable leaf, relation, hyperedge and pair/mirror links using the
  closure contract in
  [RESIDENT_TREE_AND_SUPERPOSITION.md](RESIDENT_TREE_AND_SUPERPOSITION.md).
- Resolve and validate every reference.
- Deduplicate by content-addressed relation/event ID; a replay adds zero
  records.
- Retain contradictory claims. Do not average away support and refute into a
  single truth value.
- Build deterministic candidate projections:
  - relation triple index `(subject, predicate, object)`;
  - pair cells and mirrors;
  - hyperedge membership;
  - evidence assessment per subject.
- Evidence aggregation, if performed, is a derived projection with an explicit
  formula. For a pair with support weight `S` and refute weight `R`:

  ```text
  support = Σ confidence over supports
  refute  = Σ confidence over refutes
  weight  = (support - refute) / (support + refute)   if support+refute > 0
  weight  = 0                                        otherwise
  ```

  The counts and contributing event IDs remain stored; the weight is not a
  probability and does not overwrite either side.

### 7.2 Training corpus/evidence integration

- Derive `N16REL/1` rows from accepted typed relations, preserving split,
  resolution, polarity, confidence, subject/predicate/object digests and source
  digest.
- Optional corpus chunks are stored with their own source digest, license and
  split. They are evidence/training candidates, not verified facts.
- Acceptance does **not** train a model. A separate trainer must request the
  corpus, cite the `bundle_id` and `intake_generation`, and produce its own
  training receipt. Training success/failure is separate evidence.
- Test rows remain test rows. A bundle that relabels test as train is rejected
  with `split_leakage`.

## 8. Generation, CAS and rebase

The intake layer maintains, per `universe_id`:

```text
accepted_generation          u64   # last applied local generation
accepted_local_digest        hex   # last accepted local state/checkpoint digest
accepted_projection_digest   hex   # digest of the accepted candidate projection
intake_generation            u64   # intake's own monotonic generation
receipt_head                 hex   # last receipt ID
```

Apply requires generation CAS:

- `expected_intake_generation` must equal the current intake generation;
- `expected_accepted_generation` must equal the current accepted generation;
- `expected_accepted_digest` must match `accepted_projection_digest` if
  supplied;
- `parent_digest` must match `accepted_local_digest` when the base is current;
- a stale value returns `stale_intake_generation` or `stale_base_generation`
  **before any write**;
- exactly one concurrent apply may win.

If `base_generation < accepted_generation`, the bundle is a delayed delta.
Default policy rejects it (`stale_base_generation`). A deployment may enable
`allow_rebase=true` only with a contiguous, digest-linked chain of prior
bundles or a signed rebase proof; an unproven rebase is quarantined. A future
base (`base_generation > accepted_generation`) is always rejected.

`local_generation` is the producer's generation at export. It is not trusted to
be newer than the intake's state; it is used for ordering and diagnostics. The
intake's `intake_generation` advances by exactly one per accepted apply.

## 9. Canonical merge is explicitly OFF

Current gate:

```text
merge_enabled              = false   (HERMES local config default)
canonical_merge_enabled    = false
six_runner_barrier         = absent
promotion_allowed          = false
canonical_commit           = false
canonical_state_modified   = false
```

The existing `_merge_enabled()` gate in `nanlmind_hermes/__init__.py` may allow
some **local** promotion when explicitly configured; it does not authorize a
canonical merge. The intake layer must not infer canonical authority from it.
Every receipt states the flags separately. In the current gate an intake
manifest must still declare `merge_enabled=false`; a local process that enables
its own local gate may not export a bundle that claims canonical authority.
Setting a local `merge_enabled=true` with `canonical_merge_enabled=false` never
yields `canonical_commit:true`, and the intake layer rejects such a claim.

Canonical promotion remains blocked until all of the following exist and pass
their own contract tests:

1. **Generation CAS in canonical BS0:** every promotion carries
   `expected_generation` and `expected_root`; a stale writer fails before any
   canonical write; exactly one concurrent promotion wins.
2. **Six-runner barrier:** six distinct runner signatures over the same
   base/candidate generation, all 256 heads accounted for, no duplicate runner,
   no missing or extra head, and a barrier receipt.
3. **Rollback pointer and audit:** the previous canonical generation is
   recoverable, and every attempt (accepted, refused, dry-run, rollback) is an
   append-only receipt.
4. **Independent gate test:** a contract test proves that a valid bundle cannot
   promote without the barrier, and that a barrier with five runners or a stale
   generation is refused.

Until then, a bundle may be accepted, rejected or quarantined, but it cannot
change canonical state. The phrase “integrate mathematically” in this document
always means candidate staging, never canonical promotion.

## 10. Receipts

An accepted bundle returns a receipt. A rejected bundle returns a rejection
record with the same identity fields and no rollback token.

Receipt schema: `hermes.resident.intake_receipt/1`.

| field | meaning |
|---|---|
| `receipt_id` | SHA-256 of the canonical receipt body |
| `status` | `accepted`, `rejected`, `quarantined`, `duplicate`, `rolled_back` |
| `bundle_id` | producer bundle ID |
| `manifest_digest`, `container_digest` | verified digests |
| `installation_id`, `universe_id` | scope |
| `local_generation`, `base_generation` | producer claims |
| `accepted_generation`, `intake_generation` | intake state after apply |
| `accepted_local_digest`, `accepted_projection_digest` | post-apply CAS anchors |
| `counts.received/accepted/duplicates/rejected` | record accounting |
| `digests.before`, `digests.after` | candidate projection digests |
| `validation.errors` | stable codes and bounded diagnostics |
| `local_state_recomputed` | false when the resident file was not uploaded |
| `attestation` | `verified`, `unattested` or `not_configured` |
| `merge_enabled` | false in the current gate |
| `canonical_merge_enabled` | false |
| `six_runner_barrier` | `absent` |
| `promotion_allowed` | false |
| `canonical_commit` | false |
| `canonical_state_modified` | false |
| `rollback_token` | only on accepted; capability, never logged |
| `rollback_expires_ns` | expiry |
| `policy_version`, `created_ns` | audit metadata |

`receipt_id` is content-addressed over the canonical JSON body excluding the
receipt ID itself. The local receipt is stored at
`<state_dir>/resident/receipts/<receipt_id>.json` with mode `0600`; the
append-only index is `<state_dir>/resident/receipts.jsonl`. The service stores
the receipt and a hash of the rollback token, not the token itself.

HTTP status mapping:

- `200` accepted or idempotent replay;
- `202` quarantined pending review;
- `400` malformed container/manifest;
- `401/403` authentication, scope, license/consent or canonical gate;
- `409` generation/CAS/idempotency conflict;
- `413` size/decompression/overbootstrap denial;
- `422` schema, digest, reference or typed-semantics failure;
- `429` rate limit;
- `500` apply failure after a pre-apply checkpoint;
- `507` intake capacity exhausted.

## 11. Rollback

There are two independent rollbacks:

1. **Local resident rollback** in [UNIFIED_RESIDENT_UNIVERSE.md](UNIFIED_RESIDENT_UNIVERSE.md):
   creates a new local generation from a verified checkpoint and does not touch
   intake or canonical state.
2. **Intake rollback:** withdraws the bundle's contribution from candidate
   staging and creates a new intake generation.

Intake rollback rules:

- request: `receipt_id`, `rollback_token`, `expected_intake_generation`,
  `expected_accepted_digest`; CAS is mandatory;
- the token is random 32 bytes, stored hashed by the service;
- rollback creates a new intake generation whose logical content is the
  pre-apply checkpoint; it never reuses a generation number;
- records are soft-withdrawn with `withdrawn_by=<receipt_id>`; history remains
  queryable;
- corpus rows derived from the bundle are marked withdrawn; a training run that
  already consumed them is not rewritten, but its manifest must be flagged
  `withdrawn_source=true` when queried;
- rollback default TTL 24 hours, maximum 7 days; after expiry the receipt
  remains but `rollback_expired` is returned;
- replay of the same rollback request returns the same receipt; a different
  body with the same token is `rollback_conflict`;
- rollback never changes canonical state and never claims to reverse a
  canonical training run.

## 12. Failure modes

| failure | stable code | status | effect |
|---|---|---:|---|
| bad container magic/version | `bad_container` | 400 | no read/apply |
| section overlap/out of bounds | `bad_section_table` | 400 | no apply |
| oversize container/decompressed | `bundle_too_large` | 413 | rejected before allocation |
| compression bomb | `decompression_ratio_exceeded` | 413 | rejected while streaming |
| bad manifest/schema | `bad_manifest` | 422 | rejected |
| empty bundle | `empty_bundle` | 422 | rejected |
| manifest digest mismatch | `manifest_digest_mismatch` | 422 | rejected |
| container digest mismatch | `container_digest_mismatch` | 422 | rejected |
| section digest mismatch | `section_digest_mismatch` | 422 | rejected |
| record digest mismatch | `record_digest_mismatch` | 422 | rejected |
| stale base | `stale_base_generation` | 409 | rejected; no write |
| future base | `future_base_generation` | 409 | rejected |
| stale intake generation | `stale_intake_generation` | 409 | rejected; reload |
| bundle ID reuse with different bytes | `bundle_id_conflict` | 409 | rejected |
| dangling leaf/pair/hyperedge ref | `dangling_reference` | 422 | rejected |
| cross-universe ref | `cross_universe_reference` | 422 | rejected |
| pair rank/address/mirror mismatch | `pair_mirror_mismatch` | 422 | rejected |
| resolution outside L0–L9 | `resolution_out_of_range` | 422 | rejected |
| test row relabeled train | `split_leakage` | 422 | rejected |
| missing license/consent | `license_missing`/`consent_missing` | 403 | rejected |
| unsigned bundle under strict policy | `unsigned_bundle` | 403 | quarantined/rejected |
| canonical/promotion claim | `canonical_gate_violation` | 403 | rejected; gate stays off |
| 2 GiB map in bundle | `overbootstrap_copy_denied` | 413 | rejected |
| duplicate replay | `duplicate` | 200 | prior receipt |
| rollback after expiry | `rollback_expired` | 409 | no change |
| rollback CAS conflict | `rollback_conflict` | 409 | no change |
| intake capacity | `intake_capacity_exceeded` | 507 | no apply |
| apply crash | `apply_failed` | 500 | restore pre-apply checkpoint; quarantine |
| receipt write failure | `receipt_write_failed` | 500 | no success claim; retry idempotently |

## 13. Invariants

1. Upload is explicit and opt-in; no background exfiltration.
2. A bundle contains logical records, not the resident mmap, native pointers,
   process memory or the 2 GiB map.
3. Every accepted bundle has verified container, manifest, section and record
   digests.
4. Every relation resolves subject, predicate and object leaves, three pairwise
   distinct-head cells, their mirrors and a hyperedge. No dangling references.
5. References never cross `installation_id` or `universe_id`.
6. Generation CAS is mandatory; stale or future bases are rejected before
   mutation.
7. Apply is all-or-nothing and idempotent; a replay does not double-count.
8. Contradictions are retained; acceptance is not a truth verdict.
9. Acceptance is not training success. Corpus consumption is a separate,
   receipted action.
10. `merge_enabled=false`, `canonical_merge_enabled=false`,
    `promotion_allowed=false` and `canonical_commit=false` in the current gate.
11. No accepted bundle can become canonical without the generation CAS and
    six-runner barrier, and even then promotion is a separate request.
12. Every accepted bundle returns a receipt and a rollback capability.
13. Rollback creates a new generation; it never deletes history or rewinds a
    generation number.
14. Local handles are not portable; wire references are content digests.
15. Test/validation splits and source/provenance/license are preserved.
16. The 2 GiB overbootstrap map is never copied, read or mapped.

## 14. Acceptance tests (proposed, not executed)

These tests do not exist yet. They are required before any intake endpoint is
enabled. No result is implied.

| ID | test | assertion |
|---|---|---|
| I-01 | `test_resident_intake.py::test_happy_path` | fixture bundle accepted; counts/digests match; candidate relations/hyperedges; all canonical flags false |
| I-02 | `test_resident_intake.py::test_stale_base` | `base_generation` behind accepted state -> 409, no write |
| I-03 | `test_resident_intake.py::test_future_base` | future base -> 409, no write |
| I-04 | `test_resident_intake.py::test_manifest_digest` | one-byte manifest change -> 422 |
| I-05 | `test_resident_intake.py::test_container_digest` | one-byte payload change -> 422 |
| I-06 | `test_resident_intake.py::test_dangling_leaf` | relation missing subject leaf -> 422 |
| I-07 | `test_resident_intake.py::test_pair_and_mirror` | wrong rank/address/mirror -> 422 |
| I-08 | `test_resident_intake.py::test_cross_universe` | foreign universe/installation ref -> 422 |
| I-09 | `test_resident_intake.py::test_bundle_id_conflict` | same ID, different bytes -> 409 |
| I-10 | `test_resident_intake.py::test_idempotent_replay` | same bundle -> identical receipt, no double count |
| I-11 | `test_resident_intake.py::test_rollback` | pre/post digests match; soft withdrawal; new intake generation |
| I-12 | `test_resident_intake.py::test_bounds_and_bomb` | oversize/ratio -> 413 before allocation |
| I-13 | `test_resident_intake.py::test_canonical_gate` | `merge_enabled=true` still cannot canonical commit; receipt false |
| I-14 | `test_resident_intake.py::test_corpus_derivation` | N16REL rows preserve split/resolution/source; test remains test |
| I-15 | `test_resident_intake.py::test_concurrent_apply` | exactly one apply wins; other 409 |
| I-16 | `test_resident_intake.py::test_no_overbootstrap` | 2 GiB object denied and never read |
| I-17 | `test_resident_intake.py::test_receipt_persistence` | receipt survives restart; replay returns it |
| I-18 | `test_resident_intake.py::test_attestation_policy` | strict policy rejects unattested; local policy marks `unattested` |

The canonical-gate test is mandatory. A happy-path intake test alone is not
sufficient because it could accidentally enable promotion.

## 15. Exact file/module ownership

HERMES local side:

| path | status | responsibility |
|---|---|---|
| `nanlmind_hermes/resident/bundle.py` | proposed | container/manifest encode/decode, bounds, canonical JSONL |
| `nanlmind_hermes/resident/intake.py` | proposed | validate/plan/apply client-side orchestration |
| `nanlmind_hermes/resident/receipts.py` | proposed | local receipt journal, token storage, replay |
| `nanlmind_hermes/resident/tools.py` | proposed | export/upload/status/rollback handlers |
| `nanlmind_hermes/resident/__init__.py` | **exists** | constants only; no intake code |
| `nanlmind_hermes/__init__.py` | **exists** | `_merge_enabled` gate and tool registration; no canonical promotion |
| `plugin.yaml` | **exists** | config/tool manifest; adding intake tools would require a separate change |
| `tests/test_resident_intake.py` | proposed | I-01..I-18 |

Intake service side (external to this repository; proposed ownership, not
implemented here):

| path | owner | responsibility |
|---|---|---|
| `NaNLMind/runner/resident_intake.py` | NaNLMind intake service | validate, plan, apply, receipts, rollback, staging store |
| `NaNLMind/deploy/resident_intake/` | NaNLMind deployment | HTTPS service, storage, auth scopes |
| BS0 six-runner barrier | NaNLMind canonical trainer | only canonical promotion path |
| `runner/training_merge.py` | NaNLMind/bootstrap merge | existing delta merge precedent; not the resident intake |
| HERMES `chat.py`, `memory.py`, `mth_leaves.py` | HERMES | producers of evidence/leaves; never canonical owners |

New auth scopes must be registered before deployment, for example
`resident:intake:write`, `resident:intake:read` and
`resident:intake:rollback`. These names are proposed; the existing NANNET
authority must approve them. Until then, no intake endpoint should be enabled.

## 16. Non-claims and current status

- **Not implemented:** bundle container, manifest validator, intake endpoint,
  rollback, receipts, auth scopes.
- **Not claimed:** any upload, acceptance, training result, accuracy, coverage,
  canonical merge, six-runner consensus or production durability.
- **Current gate:** `merge_enabled` defaults false; `canonical_merge_enabled`
  is false; the six-runner barrier is absent. Existing code reports this, but
  this document did not run the status command.
- **Hard boundary:** intake acceptance is candidate evidence. It is not truth,
  not a model update, not a canonical commit, and not a license to train.
