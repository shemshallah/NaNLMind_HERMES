# NaNLMind Hermes

Hermes plugin for a local NaNLMind 256-head universe, with NANNET installation identity, authority-side scope checks, native mathematical training, and additive discovery memory.

## Native mathematical trainer

Build the core repository and configure Hermes `runner_command` as an argv array containing the absolute path to `build/nanlm-universal`:

    ["/absolute/path/to/NaNLMind/build/nanlm-universal"]

Use `nanlm_math_train` for the new path. It invokes the native `math-train` command, trains the 256 palindromic heads in the shared global topology, evaluates a held-out corpus split, writes a resumable serialized global topology, adds one learned hyperedge per head to the native global hypergraph (with a ledger commit), validates the 256-record JSONL discovery trace against those hyperedge IDs, and merges the discoveries into the additive memory ledger.

This first native trainer is deliberately scoped: it learns per-head Boolean truth-table rules for byte-bit transition tasks. It is a real persistent native training path, but it is not yet a general symbolic mathematics solver, a language model, or the complete multi-layer self-reorganizing trainer. Its held-out accuracy is an evaluation metric, not proof of broad mathematical reasoning.

The state file includes the global topology and serialized universe/head scratchpads. A later run loads that exact state before training. Hermes keeps a separate append-only event ledger so extracted claims and contradictory evidence survive even when native state evolves.

## Other tools

- `nanlm_status` — local universe, latest native math run, checkpoint presence, and memory status.
- `nanlm_generate_corpus` — produce a content-addressed N16TXT/1 corpus.
- `nanlm_generate_curriculum` — ask the active Hermes model for a balanced 12-stage corpus, from Boolean inverses and deep logic through morphology, word choice, sentences, paragraphs and stories.
- `nanlm_curriculum_experiment` — generate that corpus, run the native 256-head trainer, merge runner discoveries into the same additive memory as chat, and save a reproducible experiment manifest.
- `nanlm_train` — run the legacy Context2 bootstrap trainer (its parameters remain in-memory).
- `nanlm_math_train` — run native 256-head training, resume the global topology, validate discoveries, merge, and reorganize.
- `nanlm_evaluate` — compare topology-byte, Kneser-Ney order-2 and Context2 controls.
- `nanlm_memory_ingest`, `nanlm_memory_query`, `nanlm_memory_assessments`, `nanlm_memory_reorganize`, `nanlm_memory_merge` — manage durable additive mathematical memory and evidence review.
- `nanlm_memory_cross_resolution` — compute the 10×10 language-resolution interaction matrix, compare aligned claims, preserve contradictions, and merge derived mathematical hyperedges.
- `nanlm_auth_init`, `nanlm_auth_pair`, `nanlm_auth_status`, `nanlm_auth_revoke` — manage the local NANNET installation identity.

## What persists

The native global topology is serialized to `universe/native/global_universe.nmg` and resumed on subsequent math-training runs. Structured discoveries are stored separately in a SQLite WAL event ledger with content-addressed IDs, typed graph edges and n-ary hyperedges. Chat reflection and curriculum generation read a bounded snapshot of that same universe before proposing new evidence; native runner discoveries append to it and retain their native head/layer routes. Reorganization rebuilds deterministic projections without deleting evidence. Conflicting hypotheses remain independently auditable.

NANNET is the authorization and delta-coordination boundary. Training and corpus processing run locally; Koyeb is not the native trainer or corpus store.

See `docs/HERMES_UNIVERSE_MEMORY.md` for the memory model and current limitations.


## Open-ended Hermes conversation

The conversational path is the primary language-model interface; it is separate
from the legacy N16TXT corpus trainer. Expose `nanlmind_hermes.tools.chat` as
`nanlm_chat` in the Hermes plugin tool registry and
`nanlmind_hermes.tools.chat_history` as `nanlm_chat_history`.

`nanlm_chat` accepts ordinary UTF-8 text up to 1 MiB, keeps a durable JSONL
transcript per session, calls the active Hermes language provider for an
open-ended answer, then records deterministic UTF-8 bit, nibble and byte histograms plus
orthographic token, sentence and paragraph measurements in ten independent
resolution channels. A structured reflection pass supplies phonetic, morpheme,
word-completion and narrative evidence when supported by the exchange.
It then makes a separate best-effort structured reflection call. The reflection
worker proposes typed discoveries tagged with linguistic resolution (including
phonetic, morpheme, lexical, word completion and narrative hypotheses). Valid
proposals are merged into the additive memory ledger with a source digest and
provenance. Stable concept alignment IDs let the cross-resolution engine compare
the same concept across different linguistic resolutions. Reflection failure
never blocks the conversational answer.

This is an interaction and evidence-learning loop, not online gradient training:
the provider response is generated by the configured model, while reflected
discoveries update Hermes' durable symbolic memory. They do not silently alter
provider weights or claim that the native 256-head topology has learned language.
Large-text conversation does not require N16TXT framing. The active provider
selection is whatever the host Hermes runtime configures; multi-provider
experimentation requires configuring/selecting additional providers at the host
integration layer.

## Android / Termux installation

The base plugin has no third-party Python runtime dependencies. On Termux, install
the editable package without the optional NANNET cryptography stack:

    python -m pip install -e .

This enables plugin registration, `nanlm_chat`, `nanlm_chat_history`, SQLite
mathematical memory, corpus handling and local runner integration. The NANNET
identity/pairing/signature/encryption tools remain registered but require the
optional `auth` extra on a platform with a supported cryptography wheel:

    python -m pip install -e ".[auth]"

Do not attempt to substitute pure-Python homemade cryptography. In standalone
local mode the runner does not require NANNET cryptography; configure the
authority URL and pinned issuer key only when using paired authority mode.


## Cross-resolution discovery and controls

The ten resolutions are not isolated bins. When evidence carries an explicit stable
`alignment_id` or `concept_id`, the engine can compare numeric observations across
resolutions. It currently discovers exact equality, ratios, and differences only
when both observations declare matching non-empty units. Numeric expressions are
treated as inert data: only a literal numeric `value` or numeric `object` is read;
arbitrary expression trees are never executed. Each accepted relation becomes an
auditable equation/hyperedge that points back to its source event IDs and is routed
to the narrative/review channel.

Controls and baselines:

- **Known-answer positive control:** one aligned concept represented at all ten
  resolutions produces 45 pairwise level comparisons and a consistency score of 1.
- **Contradiction control:** a deliberately refuting level remains visible as
  contradiction evidence; it is not averaged away.
- **Unrelated-concept negative control:** ten unique alignment IDs produce zero
  cross-resolution links.
- **Numeric unit control:** matching values and units can produce an equality;
  a known 12-to-3 ratio produces 4. Incompatible or missing units are excluded.
- **Permutation negative control:** a seeded label permutation reports observed
  versus shuffled comparison counts and consistency magnitudes. It is a diagnostic,
  not a statistical p-value.

Run the deterministic synthetic suite with:

    python -m benchmarks.cross_resolution_baseline

Run the complete unit suite with:

    python -m unittest discover -s tests -v

These synthetic numbers validate implementation behavior, not real-world language
understanding. Real corpus baselines still need held-out corpora, repeated seeds,
per-resolution coverage, relation precision/recall against adjudicated labels,
and comparison against shuffled-label and single-resolution controls. No learned
neural weights are updated by these symbolic relation discoveries.

## Ten-resolution mathematical feedback

The language substrate has ten distinct resolution coordinates, not seven coarse buckets:

| Index | Resolution | Mathematical role |
|---:|---|---|
| 0 | Bit | Boolean features and binary evidence |
| 1 | Nibble | Four-bit symbol partitions |
| 2 | Byte | UTF-8 code-unit and byte distributions |
| 3 | Phonetic | Consonant/vowel and sound-pattern hypotheses |
| 4 | Morpheme | Roots, stems, prefixes and suffixes |
| 5 | Lexical | Word identity and semantic relations |
| 6 | Word completion | Candidate next-token alternatives |
| 7 | Sentence | Syntax, agreement and local sequence constraints |
| 8 | Paragraph | Discourse continuity and reference |
| 9 | Narrative | Temporal, causal and story-level structure |

`nanlm_memory_cross_resolution` computes a symmetric 10×10 co-occurrence matrix plus separate fine-to-coarse and coarse-to-fine feedback matrices. It compares claims sharing a subject or explicit `alignment_id` / `concept_id`, measures polarity alignment, contradiction and content divergence, computes a bounded consistency score and polarity entropy, and emits deterministic n-ary equation/hyperedge records linking the original event IDs. Derived records are routed to resolution channel 9 for review; original evidence is never overwritten.

The score is descriptive evidence, not a truth oracle: `(aligned - contradicted) / (aligned + contradicted)`, or zero when no support/refutation comparison exists. Neutral evidence is retained but does not inflate the score. The same operation is invoked by chat and native mathematical training, and is exposed directly as a Hermes tool for runners.
