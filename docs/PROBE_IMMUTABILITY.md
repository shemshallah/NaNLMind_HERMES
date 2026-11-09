# Read-only probe immutability

`tests/test_probe_immutability.py` verifies that `nanlm_chat` with
`probe=True` does not mutate persisted learner state. It uses the real plugin
`register(ctx)` hook, the `nanlm_chat` handler, native chat mode and an
isolated temporary `state_dir`. No NANNET authority, cloud provider, language
model or WordNet archive is required.

Run it with:

```sh
python3 -m unittest tests.test_probe_immutability -v
```

## What the tests do

1. Register the plugin into a temporary state directory with `chat_mode=native`
   and `chat_learning=True`.
2. Teach several facts through `nanlm_chat` with `probe=False`.
3. Capture the strict learner-state fingerprint and semantic counters.
4. Ask the same structured action question, structured red/blue question and
   unstructured open question three times each with `probe=True`.
5. Assert, for every probe result:
   - top-level `probe=True` and `state_mutated=False`;
   - `native_synthesis.probe=True` and
     `native_synthesis.state_mutated=False`;
   - a non-empty response.
6. Assert that the repeated responses are identical, the strict fingerprint is
   unchanged, and the transcript-line, layered-turn, memory-event and
   scratch-page counters are unchanged.
7. Run a control turn with `probe=False` and assert that it does change the
   fingerprint and increases the transcript, layered-learning, memory-event
   and scratch-byte counters.

A structured question is skipped only when the current build does not provide
the corresponding question intent.

## Exact strict fingerprint scope

The fingerprint is a SHA-256 over canonical JSON. Each entry contains the
relative path plus the file size and SHA-256 of its bytes.

Included:

| Path | Included data |
| --- | --- |
| `universe/mathematical_memory.sqlite3` | Main database file bytes only |
| `language/layered_learning.json` | Full file bytes |
| `language/temporal_mirror.json` | Full file bytes |
| `scratch_register/pages/page-*.n16` | Every page, sorted by relative path; path, size and SHA-256 |
| `conversations/*.jsonl` | Every transcript, sorted by relative path; path, line count, size and SHA-256 |

A fixed file that is absent is recorded as `null`, so a probe that creates one
of those files cannot hide behind an empty directory.

Deliberately excluded from the strict fingerprint:

- SQLite `mathematical_memory.sqlite3-wal` and `-shm` sidecars.
- Scratch page `.lock` files; these are coordination artifacts, not learner
  pages.
- `language/temporal_register.json`, `language/mth_heads.json` and
  `language/chart-checkpoints/*`.
- `clocked_scratch/*`, `corpora/*` and `session/*`.
- Temporary files, directory mtimes and file mtimes.

`learner_counts()` separately reads the logical state: transcript lines,
`layered_learning.json.turns`, `SELECT COUNT(*) FROM events` against the main
database in read-only mode, and the number/size of scratch pages. The event
count reads through the WAL, so a probe that wrote only an uncheckpointed WAL
record would still be detected even though the WAL is outside the byte
fingerprint.

## Why WAL/SHM sidecars are excluded

`-wal` is SQLite's write-ahead journal and `-shm` is its shared-memory index.
Their contents depend on connection lifetime, read marks and checkpoint
timing; they can be created, reused or touched by read-only opens. They are
not durable learner evidence and including them would make the fingerprint
sensitive to SQLite housekeeping rather than to learning.

The main database file plus the read-only event count covers durable memory
semantics. The tests also record WAL/SHM size and SHA-256 separately and print
them in fingerprint-failure messages, but do not assert on them.

### Read-only SQLite caveat

The probe path in `nanlmind_hermes/chat.py` uses
`sqlite3.connect("file:...?mode=ro", uri=True)`. Before that fix, a probe could
open the writable `MemoryStore`, which initializes the schema and enables WAL,
so merely probing could create or advance sidecars. With the current read-only
path, a SQLite build may still touch sidecars depending on version, platform,
WAL state and whether `-shm` already exists.

Observed on this Termux/Android host with the current code:

- after five teaching turns the sidecars already existed:
  `-wal` 49,472 bytes and `-shm` 32,768 bytes;
- after nine read-only probes both sidecars were byte-for-byte identical
  (same size and SHA-256) and no sidecar was created or removed;
- the main database file was also byte-for-byte identical.

If a future environment reports sidecar-only traffic, treat it as transient
SQLite bookkeeping, not a durable learner mutation. For strict sidecar byte
stability, use an external copy or an immutable snapshot; do not use
`immutable=1` on a live WAL database unless ignoring uncheckpointed WAL records
is explicitly acceptable.

## Observed result

With the bundled five-fact curriculum, the strict fingerprint was identical
before and after all structured and unstructured probes. The observed counters
were unchanged:

| Counter | After teaching | After probes |
| --- | ---: | ---: |
| Conversation lines | 10 | 10 |
| Layered-learning turns | 5 | 5 |
| Mathematical-memory events | 35 | 35 |
| Scratch pages | 40 | 40 |
| Scratch page bytes | 96,640 | 96,640 |

The non-probe control (`Birds fly.`) changed the fingerprint and moved the same
counters to 12 transcript lines, 6 layered turns, 42 memory events and 47
scratch pages (115,578 page bytes).

## In-process temporal register (fixed)

Persisted learner files are unchanged, and the probe no longer advances the
shared **in-process** temporal register. `_native_synthesize(..., probe=True)`
skips memory/bootstrap mutation and `TemporalMirrorLM.save(ctx)`, and
`TemporalMirrorLM.generate(..., mutate_register=False)` suppresses
`TemporalRegister.commit()` for evaluation probes. The probe receipt reports
`state_mutated=False` and `register_frames=0`.

The register test now passes. Previously, probes advanced the register by
5–15 frames per call without saving; a later non-probe chat could checkpoint
that probe-induced state. That path is closed.

## Test status in this workspace

`python3 -m unittest tests.test_probe_immutability -v` runs five tests and all
pass:

- structured action probe: pass;
- structured red/blue probe: pass;
- unstructured open probe (persisted state and text determinism): pass;
- non-probe control changes state: pass;
- probe does not advance the in-process temporal register: pass.
