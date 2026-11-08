# NaNVerse layered learning, 1 GiB scratch and HW5 runner ring

This is the HERMES-side learning loop over the one HW0::BS0 universe. It is
derived/experimental state until a separate canonical merge gate accepts it.

## Data flow

```text
free-form chat / HW4 WordNet reads
        |
        v
L0 bit -> L1 nibble -> L2 byte -> L3 phonetic -> L4 morpheme
-> L5 lexical -> L6 word_completion -> L7 sentence
-> L8 paragraph -> L9 narrative
        |
        v
255-head ensemble (HW0:1..255), deterministic superposition
        |
        v
1 GiB paged scratch register space (128 x 8 MiB, sparse)
        |
        v
ConceptMatrix (32,640 unordered relation cells)
        |
        v (conservative, explicit approval)
local 8 MiB HeadScratchRegister + temporal register sidecar
        |
        v
bounded handoff checkpoint through the active HW5:0 runner-ring slot
```

The canonical native NaNLMind topology and the Koyeb merge gates are not
bypassed by this path.

## 1 GiB scratch register

`nanlmind_hermes/scratch_register.py` implements:

- 128 pages x 8 MiB = 1 GiB logical capacity;
- pages 0..119 are ten L0-L9 layer blocks of twelve pages each;
- pages 120..127 are reserved for future index/manifest data;
- sparse allocation: only written pages exist on disk;
- append-only records with a SHA-256 page digest and per-record length;
- addresses of the form
  `nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW17:register#5:subrelation(4096)`;
- `verify()`, `manifest()`, `stats()` and bounded `records()` readback.

The page digest is checked before a record is returned. A tampered page fails
closed.

## L0-L9 and 255 heads

`nanlmind_hermes/layered_learning.py` projects every observed text into all ten
canonical resolutions. It distributes each layer across all 255 HW0 heads with
a deterministic superposition and records:

- layer features (Boolean signature, phonetic onset/rime, morpheme parts,
  lexical words, completion prefixes, sentence relations, paragraph
  transitions, narrative causal markers);
- top head weights;
- the exact paged-scratch address of the layer leaf;
- relation rows for typed subject/predicate/object evidence.

`HeadEnsemble.balance()` reports active-head coverage and concentration.
`decorrelate()` is available for experiments but is not run automatically.

## WordNet HW4

`nanlmind_hermes/wordnet_store.py` streams the WordNet 3.1 archive into a sparse
synset dictionary and lemma index. It does not materialise a dense per-word
matrix. `LayeredLearningCoordinator.ingest_word()` reads definitions, hypernyms
and typed relations through `DictionaryUniverse`, emits `defined_as`,
`kind_of` and inherited-category rows, and writes them into the L4/L5 scratch
pages. The HW4 interface remains:

```text
nan://HW4:BS0:MC0:SC0:GM0   (legacy-compatible HW4 route)
HW4:0                        (link)
HW2:0                        (HERMES upstream)
```

## Promotion policy

The default policy is deliberately conservative:

| Rule | Default |
|---|---|
| minimum repeated support | 3 |
| minimum confidence | 0.6 |
| maximum candidates per pass | 64 |
| maximum local promoted leaves | 100 |
| explicit approval required | true |
| automatic synchronization | false |

`nanlm_learning_cycle` observes and writes scratch. `nanlm_learning_sync` with
`approved=true` is the only path that promotes candidates into the local
`HeadScratchRegister`. A promotion report is appended to
`state_dir/language/layered_learning_sync.jsonl`.

These rules are a starting point, not a claim that the truth-maintenance
problem is solved. Contradiction, source trust, temporal supersession and
cross-head consensus remain explicit future work.

## Runner-ring handoff

The six `nanrouter-runner-0..5` repositories are identical hot standbys. They
are the internal BSR0..BSR5 members of MC0:0 and do **not** receive individual
HW addresses. The ring as a whole is **HW5:0**.

The Koyeb portal provides:

- an atomic, strictly monotone `nanclock_tick`;
- one active lease with epoch and expiry;
- a bounded checkpoint containing the exact HW5:0 fabric inbox.

On handoff, the next slot restores the checkpoint and continues. See
`NaNLMind/runner/runner_ring_authority.py` and
`NaNROUTER/docs/POINT_FABRIC.md`.

## Hermes tools

| Tool | Scope | Purpose |
|---|---|---|
| `nanlm_learning_cycle` | `training:execute` | one observe/L0-L9/scratch cycle |
| `nanlm_learning_status` | `universe:read` | scratch, ensemble, concepts, heads |
| `nanlm_learning_sync` | `training:execute` | explicit promotion into local head register |
| `nanlm_wordnet_ingest` | `training:execute` | HW4 definition/hypernym ingestion |
| `nanlm_scratch_register` | `universe:read` | bounded paged-scratch inspection |

## Boundaries

- No host numerical kernel replaces the native MTH execution path.
- No canonical NaNLMind topology commit is made by the plugin.
- No secret, token or private key is written to scratch or logs.
- The WordNet graph is sparse; no dense 256x256 per-word matrix is created.
- The 1 GiB space is logical and sparse, not a 1 GiB RAM reservation.
