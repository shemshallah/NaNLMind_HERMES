# WordNet universe at HW4:0 (permanent session contract)

Source contract: NaNROUTER `spec/point-map.toml`,
`docs/BS0_HARDWARE_SLOT_CONTRACT.md`, and the plugin's
`dictionary_universe.py` / `wordnet_store.py`.

## Identity and route

- WordNet is the permanently reserved `HW4` universe inside Bootstrap BS0.
- Canonical route: `nan://GM0:SC0:MC0:BS0:0:HW4:0`.
- Compatibility route: `nan://HW4:BS0:MC0:SC0:GM0`.
- HERMES reads WordNet through NANNET. HERMES upstream link is `HW2:0`.
- Binary translation is NaN16DE. The local link name is `HW4:0`.
- `HW4` is never allocated to a public portal user and is not repurposed as a
  live lease after migration.

## Data shape

WordNet is a sparse synset/lemma/typed-edge graph. Counts are archive-dependent;
the checked-in 3.1 extraction reported 82,192 synsets, 117,953 lemmas, 84,505
hypernym edges, 378,203 typed edges and 26 relation types. The local archive used
for the live smoke reported 117,458 synsets, 147,982 lemmas and 377,031 pointers.
Report the actual `WordNetStore.stats()` for the archive in use; never present
one extraction's counts as universal.

A word leaf points to its synset; a synset participates in the concept graph.
Polysemy is a superposition of synset leaves. There is no dense per-word
256x256 matrix and no one-word-one-head allocation.

## Read path

```text
word
  -> lemma index
  -> synset offsets
  -> definitions, hypernyms, inherited categories, typed relations
  -> L4 morpheme / L5 lexical scratch leaves
  -> relation rows: defined_as, kind_of, inherits, typed edge name
  -> optional conservative promotion
```

`nanlm_wordnet_ingest` reads a word, projects definitions and typed hypernym
relations into L0-L9 scratch, and can optionally promote under the normal
policy. It does not mutate canonical NaNLMind state.

## Boundaries

- A definition is a graph edge/gloss, not parsed prose understanding.
- Inheritance is hypernym-closure lookup, not a learned semantic proof.
- The dictionary universe is read-only evidence. It does not authorize a route,
  lease, merge or canonical head update.
- Unknown words return no fabricated synset. Report absence honestly.
- The WordNet archive is configured with `wordnet_archive`; if it is absent the
  HW4 dictionary is unavailable, not silently replaced by an LLM's memory.
