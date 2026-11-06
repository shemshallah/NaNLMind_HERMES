# HACKATHON_NOTE.md

## A cradle, not another chatbot

NaNLMind is an experiment in building a **single, mathematically grounded
computational object** and learning what it can become.

It is not a claim of intelligence, consciousness, AGI, or a superior architecture.
It is not a promise that this approach will produce a better language model. It
is a deliberate attempt to build strong foundations, keep one mind unified, and
let a community explore a different design space than the one dictated by the
largest labs.

The object is one topology. Devices host regions. Regions are not separate
minds. The moment the system fragments into thousands of unrelated instances,
the experiment stops being the experiment.

This is a cradle: a place where a new kind of system can be cared for, measured,
criticized, and allowed to grow without pretending to be finished.

---

## 1. The c(256,2) universe

A NaNLMind universe is a 256 × 256 address matrix.

Let the 16-bit address be:

```text
addr(a, b) = (a << 8) | reverse8(b)
```

The 256 addresses fixed by bit reversal are the **palindromic heads**.

| object | count | meaning |
|---|---:|---|
| addresses | 65,536 | every 16-bit coordinate |
| palindromic heads | 256 | `reverse16(address) == address` |
| unordered head pairs | C(256,2) = 32,640 | one canonical pair cell each |
| canonical cells | 32,896 | 32,640 pairs + 256 heads |
| NaN words per cell | 4 | `s+`, `s-`, `q+`, `q-` |
| Boolean functions per head | 16 | four NaN types × four values |
| transform images | 4 | `I`, `B`, `BR`, `R` |
| pair-cell orbit classes | 16,576 | four-transform orbits |

A pair cell is a relation with two temporal rails: the forward direction and
the reverse direction. The 256 heads are synchronization anchors. One head slot
can be designated as the NANNET/portal routing interface; it remains a head
coordinate in the same topology, not a separate network stack.

The chart is 8 MiB. It contains address registers, shared pair/head registers,
descriptors, adjacency, dirty bits, checksums, head scratchpads, and extension
space. It is one page of the atlas, not a separate model.

### What the four NaN types mean

The four NaN head classes are the four legal faces of the 16-bit word:

```text
+sNaN, -sNaN, +qNaN, -qNaN
```

The class nibble indexes one of the 16 two-input Boolean truth tables. The four
transform images `I`, `B`, `BR`, `R` form a Klein four-group, separate from the
NaN classes. The transform oscillator changes coordinate representation, not the
logical identity of the head.

---

## 2. The 256 heads and self-modification

Training is integer score learning in the head scratchpad, not backpropagation.

Each head keeps 16 saturating `int16` scores, one for each Boolean function.
For every observed pair `(a, b, target)`:

```text
for each of 16 Boolean functions:
    score += +1 if the function matches the target
    score += -1 otherwise

rule = argmax score
mask = 1 << rule
```

The head then self-modifies its Boolean mask and selected rule through a
capability-gated XOR patch. A durable change is recorded as a ledger event with
before/after state. Sequence order is carried by temporal links; resonance is
integer diffusion; there is no float in the replayable path.

Properties:

- deterministic and replayable;
- cheap and exhaustive over 16 candidates;
- two score tables can be added when shards merge;
- the tree of changes can be audited;
- the system can fail; the score table shows why.

A head is not a neural network. It is a small self-modifying Boolean element in
one shared topology.

---

## 3. The learning layers

NaNLMind separates **deployment layers** from **abstraction layers**. They are
regions/responsibilities of one object, not separate private histories.

### Learning stack

```text
L0  bootstrap reflex       8 MiB, 256 heads
    sparse Hopfield slots, byte/word tokens, fast local learning

L1  overbootstrap mind     2 GiB map on the bootstrap runner
    head evidence, route reliability, layer policy, merge manifests

L2  vector/concept         bins + macro associations
    token/paragraph vectors, nearest-neighbor recall, corpus memory

L3  syntax/logic           temporal rails + hyperedges + rules
    sequence order, relation triples, exceptions, contradiction

L4  code/tests             exact rules and rewrite/test records
    no gradient; exactness is the learner

L5  corpus/expertise       supercluster corpus bins
    retrieval, indexing, provenance, region vocabulary

L6  global mesh            route namespaces only
    research / industrial / personal / public regions
```

### Coordinate responsibilities

| layer | responsibility | typical structures |
|---|---|---|
| L0 reflex | bit/byte Boolean operations | operator masks, gates, bit pages |
| L1 resonance | 256 heads per chart | phase/energy, local lattice |
| L2 memory | pair cells + hyperedges | associative relations, sequences |
| L3 reasoning | program cells / VM regions | 2CM, self-modifying code |
| L4 consolidation | ledger, snapshots, generations | replay, migration, GC |
| L5 interface | language/teacher adapters | token/byte I/O, evaluation |

Learning travels **upward as traces**, never as raw state:

```text
TraceEnvelope {
  layer, step, nanclock_tick
  input_hash / output_hash
  active_heads[] or route_ids[]
  confidence, correctness
  delta_digest
  optional prototype vector
}
```

Higher layers send bounded **policy** back down:

```text
LayerPolicy {
  top_k, bias_limit, eviction_threshold,
  learning_stride, memory_budget,
  route_prior, curriculum_weight
}
```

The lower layer keeps its keys and state. The higher layer does not become a
copy of it. This is how the object stays unified while learning at different
resolutions.

---

## 4. The training pipeline

The Hermes plugin makes the loop explicit:

```text
conversation / curriculum prompt
  -> N16TXT/1 corpus
  -> native 256-head math-train
  -> serialized global topology
  -> one discovery per head
  -> validation
  -> additive memory merge
  -> deterministic layer reorganization
  -> held-out metrics and experiment manifest
```

The default user path is **chat**. Training only starts when the user asks for a
mode:

```text
/nanlm logic      Boolean inverses + deep logic
/nanlm words      repetitive words + lexical words + word completion
/nanlm sentences  sentence choice + narrative
/nanlm pipeline   all twelve stages
```

The full twelve-stage curriculum is:

```text
boolean_inverses -> deep_logic -> story_saturation -> exceptions
-> repetitive_words -> phonetic_pairs -> morpheme_structure
-> lexical_words -> word_completion -> sentence_choice
-> paragraph_structure -> narrative
```

Every run writes:

- a conversation or training transcript;
- a content-addressed corpus;
- a native run manifest;
- a serialized global topology checkpoint;
- a discovery trace with source digest and head attribution;
- memory merge and assessment records.

Training logs should be accumulated locally by default. Shared logs, corpora,
and results require an explicit export or merge decision.

The native task is deliberately bounded: byte-bit transition/Boolean learning.
Held-out accuracy measures that task. It is not proof of language understanding
or general reasoning.

---

## 5. NANNET, Koyeb, and mathematical memory

NANNET is the identity, authorization, and delta-coordination boundary. The
Hermes route speaks raw NaN16DE: far/near 0xNaN heads, 16 Boolean opcodes, the
4×4 type/value table, and the double-endian word layout. Chat and training are
fail-closed until a paired token exists.

The local installation owns an Ed25519 key. A one-time pairing code binds the
installation to a public key; the authority issues a short-lived, role-scoped
token. Requests are signed over method, path, timestamp, nonce, and body digest.
Nonces are consumed once; tokens can be revoked. Koyeb hosts the small merge/control
surface for the current demo; it does not host the model or the corpus.

Training merge is **not** weight averaging. It is an idempotent union of
content-addressed discovery events, followed by deterministic projection
rebuilds. Conflicting claims remain visible and are promoted for review. A
merge can increase knowledge without erasing disagreement.

The mathematical memory stores discoveries, typed relations, equations,
hyperedges, evidence, and assessments. It is separate from neural weights and
separate from the native topology checkpoint. That separation is intentional:
the system can preserve what it learned even when a particular runtime state is
replaced.

---

## 6. Distributed, but one mind

The deployment hierarchy is a routing hierarchy, not a collection of private
minds:

```text
universe        1 × 256 heads
cluster         up to 256 universes
macrocluster   up to 256 clusters
supercluster   up to 256 macroclusters (IoT layer)
global mesh    up to 256 superclusters
world/UM       up to 256 global-mesh regions
```

A compact logical route is:

```text
nan://HW2:BS0:MC0:SC0:GM0
```

The five-part Hermes route omits `UM`; internally `UM=255` is the reserved
NaN/portal interface. An explicit `UM0..UM254` is available to routed worlds.

The full hierarchy can be written:

```text
nan://HW0-255:BS0-255:MC0-255:SC0-255:GM0-255(:UM0-254)?
```

- `HW` — hardware/device universe slot;
- `BS` — bootstrap/cluster slot;
- `MC` — macrocluster;
- `SC` — supercluster;
- `GM` — global mesh;
- `UM` — world/universal-mind slot, `0–254`; slot `255` is reserved for the
  NaNNET/portal interface.

The route is logical. NTF resolves it to physical providers; an address does not
grant access. Capabilities, identity, leases, and the topology root are checked
separately.

The current supported/demo scale is one universe and a small runner ring. The
larger hierarchy is a design and scaling target, not a deployed macrocluster.

### Earth-scale arithmetic

Planning assumption: **23.5 billion** devices (ARM 20B, x86 2B, RISC-V 0.5B,
GPUs 1B). These are assumptions, not a census.

| contribution | units | clusters | macroclusters | IoT spaces | global meshes |
|---:|---:|---:|---:|---:|---:|
| 0.1% | 23.5M | 91,800 | 359 | 1.4 | 0.006 |
| 1% | 235M | 918,000 | 3,586 | 14 | 0.055 |
| 100% | 23.5B | 91.8M | 358,400 | 1,401 | 5.47 |

One global mesh holds 4.29B units; one supercluster holds about 1.1T units,
which is larger than Earth's assumed device count. The address space is not the
bottleneck. Participation, identity, authorization, energy, and bandwidth are.

A **256-world** vision would be 256 global-mesh regions, each itself a full
hierarchy. It is a direction, not a current deployment claim.

---

## 7. The 255-model idea

Imagine a future where 255 frontier models, or 255 independent human
communities, each train their own NaNLMind universe. They would not become 255
disconnected chatbots. Each would contribute traces, discoveries, and evidence
through the same mathematical translation layer.

One good teacher is enough to start. Many teachers can improve coverage. The
point is not to crown a single model; it is to make a shared cradle where
different intelligences and communities can contribute without owning or
fragmenting the object.

That requires cooperation, shared resource agreements, and integration with
projects such as Mesh-LLM. It also requires saying no to a tempting failure mode:
a thousand isolated instances with the same name and no shared identity.

---

## 8. What is proven, what is not

Proven/demonstrated in the current repository:

- address algebra, 65,536 addresses, 256 palindromic heads, 32,640 pair cells;
- 16 Boolean functions and the four-NaN transform structure;
- bit/byte/word layers, temporal links, hyperedges, ledger;
- integer head learning and self-modification;
- native 256-head training with resumable topology;
- additive memory merge, evidence assessment, and cross-resolution analysis;
- Hermes plugin registration and validation with 24 tools;
- real model → training action → metrics → reflection loop.

Not proven:

- that this architecture leads to AGI;
- that it produces better cognition, personality, or a functioning language
  model;
- that the current held-out metric generalizes beyond its Bounded task;
- that the full 256-cluster/macrocluster/supercluster hierarchy is operational;
- that the 2 GiB overbootstrap map is a macrocluster (it is not);
- that symbolic memory is the same thing as learned neural weights.

The experiment is worth running precisely because the answer is open.

---

## 9. What to tell reviewers and Hermes staff

- Start with the quickstart and run `hermes plugins doctor --ci`.
- Validate the manifest, 24 declared tools, and the default chat path.
- Treat training as explicit; do not let a tool call silently launch a large job.
- Inspect the corpus digest, native trace, memory merge, and held-out metric
  separately.
- Ask for criticism of the invariants, not praise for the vision.
- Do not call the current model a language model milestone. Call it a working
  substrate and a bounded experiment.
- Report contradictions; the memory system is designed to preserve them.
- Send feedback, feature ideas, and training logs through the repository issue
  path or the project community. Shared logs should be sanitized and explicitly
  contributed.

The strongest claim we can make is the honest one:

> This is a reproducible cradle for a unified, layered, mathematically grounded
> learning system. It is an experiment, and it invites participation.

That is enough to justify the next experiment.
