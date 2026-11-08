# N16PACK1: a single-file binary-root universe with indexed leaves

The live architecture is intentionally **hybrid**:
`N16ROOT1` is mounted at HERMES start inside one 8 MiB chart;
NaNROUTER deterministically maps all 65,536 raw addresses into 256 fixed
heads and 32,640 MPs; MC0DB stores sparse leaf pointers, relation revisions
and L0–L9 candidate edges; BS0DB stores immutable content and provenance.

The new `nanlmind_hermes.universe_pack` module can export a **single
self-verifying `.n16pack` file** from an already durable chart checkpoint,
a valid MC0DB and a valid BS0DB. This is an offline bundle, not a second
online universe or an alternate host VM.

## Binary format

```text
magic N16PACK1 [8]
format_version [u32 LE]
manifest_length [u32 LE]
sha256(canonical JSON manifest) [32]
canonical JSON manifest [bounded to 64 KiB]
8 MiB exact chart snapshot
MC0DB SQLite *online-backup* bytes
BS0DB SQLite *online-backup* bytes
EOF (no implicit extensions or trailing sections)
```

The manifest binds the parent chart root, checkpoint generation, chart root
block SHA-256, physical geometry, exact ordered section lengths and section
SHA-256 digests. The loader reads only fixed known sections and rejects
wrong magic, noncanonical JSON, unexpected names, overlong lengths,
truncated content, corrupt hashes, malformed root, wrong parent, bad
SQLite headers and trailing bytes. It **never extracts or executes** input
contents. No file paths from the manifest are trusted. Pack files are
written with mode 0600 and installed without replacing an existing output.
The chart is sourced from the existing dual-slot SHA-verified `N16SNAP1`
checkpoint, **not** a concurrently mutating Python chart object.

Database bytes are captured using SQLite's `backup()` API so WAL mode is
handled correctly. The captured stores must contain their expected
`math_schema` row and pass `PRAGMA quick_check`; no new live SQLite files
are initialized by the archive logic. The archive may contain enormous
knowledge leaves because no JSON materialization of every cell occurs.

## Critical consistency boundary

A chart checkpoint, an MC0DB online backup and a BS0DB online backup **are
not jointly atomic**. A caller must externally quiesce the authenticated
NaNROUTER/HERMES writers, lock the relevant epoch, capture the chart/root
and both stores, then verify the epoch did not change. This export does
*not* implement that distributed barrier yet, and its metadata expressly
returns `cross_store_atomicity_attested=false`.
It cannot be automatically imported into live state, merged to NANNET0,
trusted as native head training, or used to promote a mathematical rule.
The production restore/commit path still needs a real native VM lease,
full MTH/NMP preflight and journal-verified receipt.

## Operational usage

From a host with already checkpointed HERMES state *and* co-located
authorized router databases:

```python
from nanlmind_hermes.universe_pack import export_n16pack, inspect_n16pack
receipt = export_n16pack(
    state_dir="/private/hermes",
    parent_root_sha256="<64-hex-parent>",
    expected_chart_generation=1,
    mc0db="/private/router/MC0DB.sqlite3",
    bs0db="/private/router/BS0DB.sqlite3",
    destination="/private/snapshots/one-universe.n16pack",
)
assert inspect_n16pack("/private/snapshots/one-universe.n16pack")["verified"]
```

These paths are illustrative, not claims of a currently configured
deployment. The export API is not exposed as a general plugin call and
never reads or transmits private SQLite bytes unless explicitly invoked
by an authorized local caller.
