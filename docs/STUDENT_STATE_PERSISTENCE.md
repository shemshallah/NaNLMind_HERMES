# What persists for the NaNLM student (and what does not)

This document answers one question precisely: **when the local runner/process
stops, what is left of the NaNLM student, and what has to be re-created?**

The short answer is:

> **No, the student is not only in memory until the runner shuts off.** The live
> 8 MiB chart, temporal-register state, loaded model objects and process caches
> are process-lifetime working copies. Their writable state is checkpointed to
> durable files on every non-probe native turn. A normal shutdown throws away the
> working copies, not the learning. On restart the runner rebuilds the chart,
> restores the latest valid local chart snapshot, reloads the language model,
> layered state, memory ledger, scratch pages and transcripts from
> `state_dir`.

There is one honest loss window: changes made **after** the last successful save,
or a crash during a multi-file save. Each individual file is written atomically,
but the three-file temporal-model save is not one cross-file transaction.
Previously committed chart generations remain readable.

## Roles: Gemma/Ollama is the teacher, not the student

| Component | What it is | Where its state lives |
| --- | --- | --- |
| Gemma via Ollama (`gemma4:31b-cloud` in the recorded runs) | Teacher/model provider. It writes lessons, statements, quiz answers and tool-call text. | Provider weights, context/KV cache and service state live outside the HERMES `state_dir` (typically in the Ollama installation). |
| NaNLM native student | `TemporalMirrorLM`, the local 8 MiB chart, `TemporalRegister`, `HeadScratchRegister`, `LayeredLearningCoordinator`, `MemoryStore` and the paged scratch register. | Durable files under `state_dir` (see the table below). |

The NaNLM universe is **not** stored in Gemma. Gemma never receives the chart
bytes, register state, SQLite ledger or scratch pages as its own model state. It
may see bounded text prompts and tool results; those are inputs, not the
student. Native chat (`chat_mode=native`) does not call a provider at all.
Stopping Ollama, deleting a Gemma model, or changing an Ollama endpoint does not
remove the student. Native reload needs no Gemma weights, no Ollama service and
no teacher transcript.

The scripts that use Gemma (`scripts/gemma_training_series.py`,
`scripts/nanlm_gemma_learning_over_time.py`, etc.) are teacher harnesses. The
learning they cause is written through the same native student paths described
here.

## Durable versus in-process-only

### Durable (survives process exit)

- Temporal language-model counts and vocabulary:
  `language/temporal_mirror.json`.
- The local 8 MiB chart overlay:
  `language/chart-checkpoints/chart.{0,1}.n16s`.
- Register frame, forward/back payloads and 256-cell state:
  `language/temporal_register.json` (auxiliary) and the chart extension inside
  the checkpoint (authoritative after a snapshot restore).
- Head-register relation leaves: `language/mth_heads.json`.
- Layered L0-L9 state: `language/layered_learning.json`; promotion history:
  `language/layered_learning_sync.jsonl`.
- Paged scratch records: `scratch_register/pages/page-*.n16`.
- Additive mathematical memory: `universe/mathematical_memory.sqlite3`.
- Conversation text: `conversations/<session_id>.jsonl`.
- Native chat scratch journal: `clocked_scratch/chat-<session_id>.jsonl`.
- The canonical native universe, when training has produced one:
  `universe/native/global_universe.nmg` and the associated training receipts,
  corpora, traces and challenge files.
- Permanent session manifest: `session/plugin_session_context.json`.

### In-process-only (rebuilt or discarded on restart)

- `universe_chart._CACHE`: `Chart8MiB` objects keyed by path, mtime and size.
- `universe_chart._RUNNERS`: one `LocalRunner` per state directory.
- The resident 8 MiB `bytearray` chart page and its extension words.
- `TemporalRegister.state` as a live Python list (including its higher-precision
  floating-point decay values before the next checkpoint).
- `LayeredLearningCoordinator` objects in `layered_learning._COORDINATORS`.
- Loaded `TemporalMirrorLM` instances, `MTHLayerStack`, `MTHScorer`,
  `MTHHeadMapper` and `ConceptMatrix` objects.
- Session documents cached by `session_context._CACHE`.
- SQLite connections plus transient `-wal`/`-shm` and lock files.

The chart is held in a process-local Python bytearray. It is not an OS shared
memory mapping. The repository calls it the "mapped" chart because it is laid
out like the native 8 MiB chart page and registers are mapped into its
extension area.

## File-by-file persistence map

All paths are relative to the configured `state_dir` (default
`~/.local/share/nanlmind-hermes`).

| Path | Written by | What it contains | Read on reload | Role |
| --- | --- | --- | --- | --- |
| `language/temporal_mirror.json` | `TemporalMirrorLM.save` | Schema, vocabulary/head addresses, unigram/bigram/trigram counts, head/mirror pair counts, subject weights, seen-sentence SHA-256 list, recent/rejected replies, last query/reply, `output_drive` | `TemporalMirrorLM.load` | Durable language counts. Missing/corrupt file falls back to a fresh model, not to Gemma. |
| `language/temporal_register.json` | `TemporalRegister.save` (called by `TemporalMirrorLM.save`) | `chart_root`, `chart_source`, pair, mapped flag, frame, forward/back payload, 256-cell state | `TemporalRegister._load_sidecar` | Auxiliary register snapshot; overridden by the SHA-verified chart extension after a checkpoint restore. |
| `language/chart-checkpoints/chart.0.n16s`, `chart.1.n16s` | `LocalRunner.checkpoint` via `save_chart_snapshot` | `N16SNAP1` header (generation, canonical parent root, SHA-256) plus the full 8,388,608-byte `NLMCHRT1` page | `restore_local_chart` / `load_chart_snapshot` | Durable local, unmerged chart overlay. Two-generation rolling slots; newest valid slot wins; corrupt newest falls back to the older valid slot. |
| `language/chart-checkpoints/.write.lock` | `save_chart_snapshot` | Empty advisory lock | Writers | Transient coordination, excluded from learner-state fingerprints. |
| `language/mth_heads.json` | `HeadScratchRegister._save` (`organize`) | Chart root, extension offset, up to 100 mapped relation leaves, per-head counts | `HeadScratchRegister._load_sidecar` | Durable head/mirror/MTH relation mapping. The chart also gets a byte copy in its extension/head scratchpads at checkpoint time. |
| `language/layered_learning.json` | `LayeredLearningCoordinator._save` (`observe`, `synchronize`, `close`) | Turn count, up to 4,096 relation rows, promoted rows, 255-head ensemble weights, policy | `LayeredLearningCoordinator._load` | Durable layered learning. `ConceptMatrix` is rebuilt from the saved relation rows; it is not a separate file. |
| `language/layered_learning_sync.jsonl` | `LayeredLearningCoordinator.synchronize` | Append-only promotion reports, fsynced | Inspection/audit | Promotion history; not required to answer a probe. |
| `scratch_register/pages/page-NNN.n16` | `ScratchPage.append` via `ScratchRegisterSpace.append` | 256-byte page header (magic, version, layer, page, used, generation, count, SHA-256) plus length-prefixed canonical-JSON records | `ScratchPage.records`, `stats`, `verify` | Durable, sparse 1 GiB logical L0-L9 scratch space. One record per layer per observed text (10 per layered turn). |
| `scratch_register/pages/page-NNN.lock` | Page append | Empty advisory lock | Writers | Transient coordination, not learner evidence. |
| `universe/mathematical_memory.sqlite3` | `MemoryStore` (`ingest`, `merge`, `reorganize`) | Append-only `events`; rebuildable `projections`, `edges`, `hyperedges`, `assessments`; `meta` | `MemoryStore.query`, `query_assessments`, `status` | Durable additive mathematical/relation memory. It is a ledger, not neural weights (`weights_checkpoint=false`). |
| `universe/mathematical_memory.sqlite3-wal`, `-shm` | SQLite | Write-ahead journal and shared-memory index | SQLite | Transient journal/index files; excluded from the strict byte fingerprint. Committed events remain in the main database. |
| `conversations/<session_id>.jsonl` | Non-probe `chat` / `_native_chat` | Canonical JSON lines with role, content, timestamp, source/provider metadata | `_read_history` (last 24 for context), `chat_history` | Durable transcript, including full user/assistant text. Probe turns append nothing. |
| `clocked_scratch/chat-<session_id>.jsonl` | `_append_native_scratch` on non-probe native chat | Five hash-chained entries: user, plan, learning proposal, synthesis, final | `Scratch` readers / audit | Durable native-chat journal. Separate from the paged scratch register. |
| `universe/native/global_universe.nmg` | Native training (`math-train` / relation training / foundation seed) | `NLMGLOB3` container with the embedded `NLMU0001` universe page and canonical root | `find_universe`, `parse_global`, `load_chart` | Canonical serialized native base. Chat/runner reads it; local chart checkpoints never rewrite it. |
| `universe/boolean_foundation.nmg`, `.json`, `foundation_seed.json` | Explicit native Boolean foundation | Native Boolean checkpoint, pass receipt, seed receipt | Training gates | Durable training prerequisites; not needed for a native chat reload. |
| `corpora/*.n16rel`, `corpora/*.n16txt`, `traces/*.jsonl`, `universe/challenges/*`, `universe/last_*`, `universe/relation_metric_history.json` | Explicit training/curriculum tools | Generated corpora, discovery traces, frozen challenge sets, run manifests and metric history | Training tools / audit | Durable training artifacts. A failed chat-learning attempt can still leave a generated relation corpus. |
| `universe/identity.json` | `_universe_id` | Stable local universe UUID and creation time | Status/tools | Durable installation identity, not model state. |
| `session/plugin_session_context.json` | `session_context.open_session` | Packaged document IDs, digests and generation/start time | `session_context` | Durable permanent plugin-session manifest; not student learning. |

Not every file is required for every answer. A native probe uses
`universe/native/global_universe.nmg` (or the genesis fallback), the temporal
model JSON, the chart checkpoint, the register sidecar and the memory database
when present; each has an explicit empty/genesis fallback. Layered state and
scratch pages are read for learning status and promotion; transcripts are
conversation history.

## Trace: `TemporalMirrorLM.load` and `save`

`nanlmind_hermes/temporal_lm.py`:

1. `path(ctx)` resolves
   `state_dir/language/temporal_mirror.json`.
2. `load(ctx)` creates a fresh `TemporalMirrorLM`, then:
   - if the file is absent, unreadable or has the wrong schema, it leaves the
     model empty and binds a `LocalRunner`;
   - otherwise it restores vocabulary (masked to 8-bit heads), unigram/bigram/
     trigram counts, pair counts, subject weights, seen digests (capped at
     65,536), recent replies (8), rejected replies (32), last query/reply and
     `output_drive`;
   - it always calls `LocalRunner.open(ctx)` and assigns
     `model.register = runner.register`;
   - it does **not** restore `mth`; the MTH scorer is rebuilt from memory rows
     on each chat turn.
3. `save(ctx)`:
   - creates `language/` with mode 0700;
   - writes a compact, sorted JSON payload through a `.tmp` file, chmods it to
     0600 and atomically replaces the target;
   - then calls `runner.checkpoint()`, which publishes the full 8 MiB chart
     snapshot;
   - then calls `register.save()`, which writes the JSON register sidecar.
4. Save order is model JSON, chart checkpoint, register sidecar. Each step is
   atomic; the sequence is not a single transaction. A crash between steps can
   leave the JSON newer than the chart, but the previous chart generation is
   still valid and the next save repairs the register copy.

Training turns and non-probe question turns both call `save`. A statement adds
counts, pair weights and relation exposure; a question updates
`recent`/`last_query`/`last_reply` and may add question terms to the vocabulary
without teaching a fact. A probe calls neither `observe` nor `save`.

## Trace: `LocalRunner.open` and chart checkpoints

`nanlmind_hermes/universe_chart.py`:

1. `LocalRunner.open(ctx)` keys a runner by `state_dir`.
2. On first open in a process:
   - `find_universe(state_dir)` prefers
     `universe/native/global_universe.nmg`, then the newest `universe/**/*.nmg`;
   - `load_chart` parses `NLMGLOB3`, extracts the embedded `NLMU0001` page and
     builds the 8 MiB `NLMCHRT1` in-memory chart (or builds a genesis chart);
   - `restore_local_chart` loads the newest valid checkpoint for that exact
     canonical root and copies the local overlay into memory after verifying
     that the immutable address/canonical/pair tables were not changed;
   - `TemporalRegister.open` binds the register to the chart;
   - the runner is stored in `_RUNNERS`.
3. If a runner is already cached and a different canonical `.nmg` root appears,
   the cached runner rebinds to the new chart, restores the matching overlay,
   refreshes the rails, rewrites the in-memory head-register extension from
   `mth_heads.json`, and keeps the register frame/state.
4. `checkpoint()` calls `save_chart_snapshot` with the current snapshot
   generation. The receipt is explicitly
   `kind=unmerged-local-chart`, `canonical_training_committed=false`.

`nanlmind_hermes/chart_storage.py` owns the durable slots:

- two files, `chart.0.n16s` and `chart.1.n16s`;
- 80-byte header: `N16SNAP1`, uint64 generation, 32-byte canonical parent root,
  32-byte SHA-256 of the 8 MiB body;
- named advisory lock plus generation comparison, so a stale process cannot
  overwrite a newer generation;
- temporary 0600 file, `fsync`, atomic rename, directory `fsync`;
- corrupt/torn newest slot falls back to the older valid slot; a fully corrupt
  store fails closed instead of silently resetting to genesis;
- a snapshot is applied only against the exact `.nmg` root (or the genesis
  root). If the canonical `.nmg` root changes, the old overlay is ineligible.

This is host byte storage and crash recovery, not native training, not a
canonical merge and not a rewrite of `global_universe.nmg`.

## Trace: `TemporalRegister`

- The register lives at `OFF_EXTENSION` (`0x63BA10`) in the chart extension:
  magic `NLMREG01`, logical frame, forward payload, back payload and 256
  uint16 state cells.
- It is bound to one canonical pair cell and its real `s+`/`s-` rails.
- `project(head)` writes the predicted head to `s+`, reads it back and syncs
  the extension. `commit(head)` projects, copies `s+` to `s-`, advances the
  frame, decays the 256-cell state by 0.97, increments the emitted head and its
  8-bit mirror, and syncs the extension. `rollback` restores `s+` from `s-`.
- `save()` writes `language/temporal_register.json`.
- On restart, `_load_sidecar` applies the sidecar only when the pair and chart
  root match. If the chart has a restored snapshot generation,
  `_restore_extension` overrides it from the SHA-verified chart bytes and
  requires the `NLMREG01` magic. Finally `_refresh_from_rail` copies nonzero
  forward/back payloads from the real chart rails.
- Therefore the chart checkpoint's 16-bit state is the durable recovery form;
  the JSON sidecar is the auxiliary/fallback form and may carry float decay
  values from the live process.

## Trace: layered learning, head register and MTH

- `LayeredLearningCoordinator.observe` increments `turns`, projects the text
  through all ten L0-L9 layers, updates the 255-head `HeadEnsemble`, appends one
  scratch record per layer, updates `ConceptMatrix`, appends typed relation rows,
  then saves `language/layered_learning.json`.
- `synchronize(approved=True)` groups relation rows by full subject/predicate/
  object/polarity/confidence, applies support and confidence gates, promotes up
  to 100 leaves into `HeadScratchRegister`, saves `mth_heads.json`, and appends a
  report to `language/layered_learning_sync.jsonl`.
- `HeadScratchRegister.organize` writes only relation rows. It maps each row to
  a head, mirror, leaf mask/value/digest, writes the aggregate register and
  per-head scratchpads into the in-memory chart, and saves
  `language/mth_heads.json`. `LocalRunner.checkpoint` later persists the chart
  byte copy.
- On reload, `mth_heads.json` is the source used to reconstruct the head
  register object. If it is missing, the chart bytes still contain the last
  checkpointed extension copy, but the current loader does not rebuild leaves
  from that copy; keep both files together.
- `ConceptMatrix` and the MTH layer stack/scorer are derived in memory.
  `layered_learning.json` does not contain the full raw text of every observed
  turn. Scratch records store a text digest and features; the transcript and
  SQLite rows store the conversation/relation evidence separately.

This whole path is explicitly derived/experimental. `canonical_commit` remains
false; local promotion is not a BS0/MC0 merge.

## Trace: mathematical memory (SQLite)

- Path: `universe/mathematical_memory.sqlite3`.
- `MemoryStore.ingest` validates each discovery, derives a content-addressed
  event ID, and inserts idempotently into `events`; it also writes projection,
  edge and hyperedge rows.
- `merge` is an idempotent union, not last-writer-wins.
- `reorganize` rebuilds active projections and evidence assessments from the
  immutable event log.
- `query`/`query_assessments` read active projections.
- Non-probe native chat merges surface measurements and parsed relations, then
  runs cross-resolution feedback; probe chat uses
  `sqlite3.connect("file:...?mode=ro", uri=True)` and does not merge or
  reorganize.
- This ledger persists discoveries, not native Context2Model weights. Its status
  reports `append_only=true` and `weights_checkpoint=false`.

## Trace: paged scratch register

- Path: `scratch_register/`; pages are created sparsely on first write.
- 128 logical pages x 8 MiB = 1 GiB logical space. Pages 0..119 are ten layer
  blocks of twelve pages; pages 120..127 are reserved for index/manifest data.
- Each page is an append-only log with a 256-byte header and SHA-256 over the
  used region. Records are `u32 length | canonical JSON`. A digest mismatch
  fails closed on read.
- `LayeredLearningCoordinator.observe` writes ten leaves per turn (one per
  layer). `verify()` audits every allocated page.
- This is durable but derived. It is not the chart's 64 KiB chart scratch area
  and it is not the separate `clocked_scratch/` chat journal.

## Trace: transcripts

- Path: `conversations/<session_id>.jsonl`; session IDs are restricted to
  1..80 letters, digits, underscores or hyphens.
- Non-probe chat appends one user line and one assistant line. Provider chat
  records provider/model metadata; native chat records
  `provider=nanlm-native` and `model=temporal-mirror/1`.
- `_read_history` reads only the last 24 valid user/assistant lines for prompt
  context; `chat_history` returns the requested tail for inspection.
- Probe turns write no transcript lines. Deleting a transcript removes
  conversational history but not the learned model counts, memory ledger,
  layered state or scratch records.

## Probes versus training turns

`nanlm_chat(..., probe=True)` is an evaluation read:

| Effect | Non-probe native turn | `probe=True` |
| --- | --- | --- |
| Merge surface measurements/relations into memory | yes | no |
| Observe/update layered L0-L9 state | yes | no |
| Append scratch-register pages | yes | no |
| Append transcript/clocked-scratch entries | yes | no |
| `runner.organize` / `mth_heads.json` | yes | no |
| `TemporalMirrorLM.observe` / `observe_memory_rows` | yes | no |
| `TemporalMirrorLM.save` / chart checkpoint / register sidecar | yes | no |
| `TemporalRegister.commit` and `register_frames` | yes | no (`mutate_register=False`, frames = 0) |
| Response source | learned local state | same local state, read-only |

The probe still builds in-memory `MTHLayerStack`/`MTHScorer` views from memory
rows and still reads the chart/register. Those are read-only derivations, not
durable mutations. One non-learning exception: the first-ever chat/registration
can create `session/plugin_session_context.json` because the permanent packaged
session context is loaded at entry. That manifest is not student learning and is
stable after creation.

## What happens on process shutdown and restart

1. The process exits. `_CACHE`, `_RUNNERS`, `_COORDINATORS` and all loaded
   objects disappear. The live chart bytearray and register state list are gone.
   SQLite closes and may checkpoint its WAL.
2. A new process starts. Plugin boot calls `LocalRunner.open`:
   - loads `universe/native/global_universe.nmg` (or genesis),
   - restores the newest valid chart checkpoint for that exact root,
   - opens the temporal register from the chart extension/sidecar.
3. `TemporalMirrorLM.load` reloads language counts and vocabulary from
   `language/temporal_mirror.json` and rebinds to the runner.
4. `coordinator_for_context` reloads `language/layered_learning.json`; scratch
   pages are read on demand and verified by digest.
5. `MemoryStore` opens the SQLite ledger and replays active projections.
6. `chat` reads the transcript tail for conversational context.

The reopened native student can answer the same structured question without
Gemma, Ollama, the network or the original process. The test suite verifies
exactly this.

## Verification

`tests/test_student_state_persistence.py` runs the real native path in a
temporary state directory:

1. Builds a valid local `NLMGLOB3`/`NLMU0001` universe, so the chart is a
   trained local chart rather than only genesis.
2. Teaches `Dogs bark.` with `probe=False`.
3. Asserts the temporal model, chart checkpoint, register sidecar, head-register
   sidecar, layered JSON, memory SQLite, scratch pages and transcript exist.
4. Records the register frame/state, chart generation, file fingerprint and the
   original `.nmg` digest.
5. Clears every process cache and creates a new context.
6. Reloads and asserts the vocabulary (`dogs`, `bark`), relation
   (`dogs can bark`), layered turn/row, head-register leaves, register
   frame/state, chart generation, memory row and scratch records are present.
7. Asserts reopening did not change any durable file and did not rewrite the
   canonical `.nmg`.
8. Runs three `probe=True` questions and asserts the complete durable file
   fingerprint is unchanged, the in-process register frame/state is unchanged,
   `register_frames=0`, and repeated answers are identical.
9. Reopens the trained state with a provider object that fails if called and an
   environment with `OLLAMA*`/`GEMMA*` variables removed. The same structured
   answer (`Dogs can bark.`) is produced with no provider call and no
   Gemma/Ollama file in the state directory.

Run it with:

```sh
python3 -m unittest tests.test_student_state_persistence -v
```

Observed in this workspace (elapsed time varies by host):

```text
Ran 3 tests

OK
```

The run may print pre-existing SQLite `ResourceWarning: unclosed database`
messages from short-lived `MemoryStore` connections in the package. They do not
change any assertion or the durable-state fingerprint.

## Boundaries

- A chart checkpoint is a local, unmerged overlay. It is not canonical training,
  not a BS0/MC0 merge and not a NANNET commit.
- `global_universe.nmg` is read-only to chat and to this persistence layer.
- The checkpoint is a full 8 MiB page write per save, not incremental.
- There is no cross-file transaction across `temporal_mirror.json`, the chart
  slot and `temporal_register.json`; a hard kill can leave them at different
  generations. The dual-slot chart ensures an older valid chart survives.
- Layered learning, scratch records, MTH head mappings and memory rows are
  derived evidence. They do not claim general language competence or replace
  native head training.
- Gemma/Ollama is a teacher/provider input. It is not a storage location for the
  student universe.
