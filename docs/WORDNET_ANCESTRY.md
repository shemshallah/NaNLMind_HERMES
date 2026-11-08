# WordNet exact ancestry traversal

The source synset is supplied. Positive targets are 2–4-hop ancestors;
negative targets share the target's lexicographer class but are not
ancestors. This is symbolic graph composition, not language learning.

- noun synsets: **82,192**
- positives: **1,000**
- negatives: **1,000**
- direct-edge-only balanced accuracy: **0.5000**
- multi-hop balanced accuracy: **1.0000**
- hop distribution: `{2: 303, 4: 330, 3: 367}`

## Examples

- bone of contention → message (2 hops; noun.communication → noun.communication)
- organic light-emitting diode → conductor (4 hops; noun.artifact → noun.artifact)
- clothes tree → instrumentality (4 hops; noun.artifact → noun.artifact)
- part-timer → organism (3 hops; noun.person → noun.Tops)
- flintlock → firearm (2 hops; noun.artifact → noun.artifact)
- gaggle → biological group (3 hops; noun.group → noun.group)
- muralist → person (4 hops; noun.person → noun.Tops)
- romance → indo-european (3 hops; noun.communication → noun.communication)
- el cid → imaginary being (2 hops; noun.person → noun.person)
- luxembourg → country (2 hops; noun.location → noun.location)
- adam's peak → point (4 hops; noun.object → noun.location)
- retardation → cognition (3 hops; noun.cognition → noun.Tops)

The 50% direct baseline is expected: positives are deliberately not
direct edges, while negatives are rejected by direct lookup. The
multi-hop result only verifies traversal over the real graph.
