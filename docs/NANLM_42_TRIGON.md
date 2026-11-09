# TRIGON-9 — harmonious synthesis of Rounds 030–041

Status: **experimental reference architecture implemented and trained; native
trigonometric kernel implemented in raw NaN16; open research gates remain.**
Last updated 2026-11-09.

This document is the Round 042 synthesis requested after Rounds 030–041. It
takes the strongest measured contribution of each round and couples them into
one deliberately experimental generative language model:

> **TRIGON-9** — a Trigonometric Recursive Invariant Generative Operator
> Network, with nine semantic L0–L9 stages, 256 sparse heads, the full
> 32,385-cycle mirror memory, ordered SO(3) transport, a byte-level
> autoregressive decoder, and a bounded architecture-evolution gate.

The native low-level contract is raw 0xNaN16:
`nanlmind_hermes/nan/trig_cycle.nan` (513 pure `NNNN_PPPPPPPP_FFFF` words).
The trainable reference model is host PyTorch under
`experiments/round042/`. **The PyTorch model is not the native runtime.**
No live NANNET lease, canonical merge, BS0/MC0 promotion, or production claim
is made.

---

## 1. What each round actually contributes

The synthesis does not average the rounds. It assigns each one a role where it
was strongest.

| Round | Strongest measured contribution | Role in TRIGON-9 |
|---|---|---|
| 030 | Native-root-trained classifier as gauge-invariant triangle circulations; exact mirror antisymmetry | **Memory coordinate system.** Every relation/token write is a three-edge circulation; reads are anchored triangles. |
| 031 | Full 32,385-coefficient basis; native gradient updates; 256-effective-weight limit made explicit | **Capacity rule.** The model exposes the full basis; no silent 256-only fallback. Effective adapter size is reported separately. |
| 032 | Typed hyperedges, stable IDs, pair postings, two-hop traversal, long-context retrieval | **Evidence graph.** Relation provenance and entity keys live beside the cycle field; traversal is bounded. |
| 033 | Episodic copy from the actual 8 MiB root; exact byte completion; gauge robustness | **Copy channel.** A pointer/copy gate can emit stored spans rather than hallucinate them. |
| 034 | Learned copy-or-abstain router; calibrated false-answer rate | **Abstention discipline.** The model has an explicit unsupported/unknown path; no forced answer. |
| 035 | Two-/three-source composition; answer assembled after reading; broken-link abstention | **Composer.** Multi-source paths are validated before rendering. |
| 036 | Neural encoder–attention–decoder; adversarial `opposes`; four-valued evidence | **Language decoder and contradiction rule.** `opposes` is not traversed as positive ancestry. |
| 037 | Prose-to-relation writer; native storage; retention after new writes; generation from native evidence | **Dynamic writer.** Content-dependent writes change the resident relation state. |
| 038 | SO(2) vs SO(3) order sensitivity; conjugation-invariant holonomy trace; token-conditioned angles | **Trigonometric transport layer.** L7–L9 use ordered Givens products and trace readouts. |
| 039 | True unrestricted subword sampling; native gradient updates; honest negative result for a static adapter | **Decoder lesson.** The model must sample its own tokens, and memory must be dynamic, not a fixed learned matrix. |
| 040 | Dynamic episode writes, bounded typed BFS, add/ask/save CLI, snapshot pair, retention | **Mutable memory contract.** Writes are transactional at the generation level; old and new queries are evaluated. |
| 041 | Differentiable propagator; deep 4-/5-hop chains; causal edge deletion/restore; copy decoder | **Reasoning/readout bridge.** Path evidence feeds token generation; deletion and restore are explicit interventions. |

The synthesis also keeps the two controls that every round forced:

1. **A conventional memory/index can reproduce gauge robustness.** The
   geometric gain is invariance, not exclusive storage superiority.
2. **A static cycle adapter does not beat a matched dense adapter.** R039 is
   the reason TRIGON-9 writes content-dependent state instead of enlarging a
   fixed 16×16 matrix.

---

## 2. The architecture

```
                          HW4 WordNet matrix
                    nan://GM0:SC0:MC0:BS0:0:HW0:0::HW4:0
                                  │
                                  ▼
   bytes ──► L0 bit ──► L1 nibble ──► L2 byte ──► L3 phonetic
                │            │             │            │
                └────────────┴──────┬──────┴────────────┘
                                    ▼
                    L4 morpheme ──► L5 lexical ──► L6 completion
                                    │
                                    ▼
              ┌──────────── 256 sparse heads (HeadRouter) ────────────┐
              │  content-dependent top-k selection, phase/amplitude     │
              └───────────────────────┬───────────────────────────────┘
                                      │  write pairs / read pairs
                                      ▼
       ┌──────────── full 32,385-cycle mirror memory ─────────────┐
       │  φ(i,j) = e(0,i) + e(i,j) - e(0,j)   (gauge invariant)   │
       │  dynamic sparse writes + decay + typed provenance         │
       └───────────────────────┬───────────────────────────────┘
                               │  complex read + triangle holonomy
                               ▼
        L7 sentence ──► L8 paragraph ──► L9 narrative
          SO(3) ordered transport, trace/holonomy, four-valued evidence
                               │
                               ▼
        token decoder  +  pointer-copy  +  copy/abstain/compose gate
                               │
                               ▼
                     sampled bytes / abstention / path answer
```

### 2.1 Four planes, one state

TRIGON-9 keeps four planes explicitly separate so none can masquerade as the
others:

1. **Geometric memory plane** — 256 heads, 32,640 mirror pairs, 32,385 cycle
   slots. Scalar/phase state is stored in anchored triangle coordinates.
2. **Trigonometric computation plane** — phase/amplitude rotors in every layer,
   cosine phase-alignment attention, cumulative complex state, SO(3) ordered
   transport and closed-loop holonomy.
3. **Neural language plane** — byte embeddings, L0–L6 fixed features, L5
   learned lexical embeddings, L7–L9 relation/paragraph/narrative transport,
   autoregressive decoder, pointer copy.
4. **Architecture plane** — bounded genome, NANNET-P mutation/validation gate,
   generation-CAS acceptance.

The language plane never reads a separate Python mirror of the memory. It reads
the same `CycleField` that the relation probe writes.

### 2.2 L0–L9 with trigonometry as a first-class component

The initial L0–L9 stack was a semantic hierarchy. In TRIGON-9 each layer also
has a trigonometric role; this is not a positional add-on.

| Layer | Semantic role | Trigonometric role in TRIGON-9 |
|---|---|---|
| L0 | bit | 16 root Boolean functions from the MTH root table map to 16 phase offsets; `cos/sin` of the root signature |
| L1 | nibble | two 4-bit rotors; phase of the low/high nibble |
| L2 | byte | multi-frequency byte phase (`1×, 2×, 4×, 8×` turns) |
| L3 | phonetic | vowel/consonant/space/digit oscillator flags |
| L4 | morpheme | rolling 3-byte hash phases for stem/affix-like units |
| L5 | lexical | learned token embedding plus a learned complex rotor state |
| L6 | completion | causal prefix features; pointer-copy and continuation gate |
| L7 | sentence | typed relation transport; SO(3) Givens rotations; four-valued evidence |
| L8 | paragraph | phase coherence across sentence states; ordered transport continues |
| L9 | narrative | long-horizon ordered product and trace/holonomy readout |

Every layer `l` contains:

- a **complex rotor state**:
  `θ_t = Σ_{s≤t}(ω_l + W_θ h_s)`,
  `z_t = A_t e^{iθ_t}`, with `A_t = softplus(W_A h_t)`;
- a **trigonometric attention bias**:
  `score(q,k) = q·k/√d + γ_l cos(θ_q - θ_k)`;
- a **layer gate** `γ_l` from the genome; setting it to zero removes the layer;
- at L7–L9, an **SO(3) transport cell**:
  `R_t = R_{t-1} G(axis_t, angle_t)`, read through `tr(R_t)` and matrix entries.

The `TrigState` and `TrigAttention` are large components of the parameter and
compute budget, not a small auxiliary head.

### 2.3 The 256 heads

The heads are a sparse interface between neural state and geometric memory.
For each token, `HeadRouter` computes 256 scores and selects `k` heads
(`k` is a genome gene, default 6, accepted search moved it to 4). Head 0 is the
permanent root/hub and is excluded from pair selection; heads 1–255 form the
C(255,2)=32,385 cycle basis.

The router returns:

- sparse activation `a_t` over 256 heads;
- selected head indices `H_t = {h_1…h_k}`;
- a head context `[a·K, a·cos φ, a·sin φ, a·amp]` fed back into the decoder.

The decoder also has a direct **head readout** `Linear(256 → vocab)`, trained
with an auxiliary next-token loss. This prevents the heads from remaining a
decorative routing label: they must carry next-token information.

### 2.4 Full-cycle mirror memory

For heads `i,j`, edge coordinates `e(i,j)` are stored in a 256×256 field. The
cycle value is

```
φ(i,j) = e(0,i) + e(i,j) - e(0,j)
```

and the additive vertex gauge is

```
e(i,j) → e(i,j) + g_i - g_j
φ(i,j) → φ(i,j) + (g_0 - g_i) + (g_i - g_j) - (g_0 - g_j) = φ(i,j)
```

The cancellation is exact algebra, limited only by floating-point roundoff in
the host reference and by fixed-point wrap in the native kernel.

A write of `δ` to one cycle uses the minimum-norm three-edge distribution:

```
e(0,i) += δ/3,  e(i,j) += δ/3,  e(0,j) -= δ/3
```

The relation probe writes `δ` from a learned content phase and reads it back;
the language model writes a sparse, content-dependent `δ_t` at every token,
decays the fast state, and reads it at the next token. The full 32,385 basis is
present; the number of *active* pairs per token is sparse and reported, not
silently truncated.

### 2.5 Ordered transport and holonomy

Commutative 2D phases cannot distinguish path order. TRIGON-9 therefore keeps
both:

- **SO(2) rotor bank** for cheap phase/amplitude state and frequency analysis;
- **SO(3) Givens transport** at L7–L9 for order-sensitive composition.

For a path of edges with rotations `R_1…R_n`:

```
R_path = R_1 R_2 … R_n
```

and for a closed loop `A→B→C→A` the trace `tr(R_loop)` is invariant under a
consistent change of local frame. The model exposes `tr`, matrix entries, and
the cycle holonomy

```
θ_loop = φ(a,b) + φ(b,c) + φ(c,a)
```

as features. Contradictory evidence is kept four-valued:

```
unknown, supports, refutes, both
```

so `opposes` is not cancelled by an opposite phase.

### 2.6 Language head, copy and abstention

The decoder produces a byte distribution from the final hidden state. A
pointer-copy distribution over the current context is added in log space:

```
p(token) ∝ exp(logits_lm + g_copy · log p_copy)
```

The copy gate is learned and genome-bounded. The relation/composition probe
provides the four-valued evidence path; a supported path can be rendered, an
unsupported or contradictory path can abstain. The current reference model
trains the byte decoder and relation probe jointly; the full copy/abstain
renderer from R034/R041 is an integration gate, not a claim of this run.

---

## 3. NANNET-P: self-modifying architecture under a gate

The user asked for a highly self-modifying, experimental system that learns its
architecture. TRIGON-9 implements a bounded architecture genome and a local
evolution protocol, `NANNET-P` (Neural Architecture Nanite Evolution Protocol).

The canonical route string used for the WordNet evidence plane is

```
nan://GM0:SC0:MC0:BS0:0:HW0:0::HW4:0
```

The `HW0:0::HW4:0` linkage is a read-only evidence link in this experiment.
NANNET-P is **not** the NANNET identity/authorization service. The local run is
recorded as `mode: "local-unverified"` and `live_nannet_route: false`.

### 3.1 Genome

| Gene | Range | Effect |
|---|---|---|
| `layer_gates[10]` | 0.15–1.5 | residual scale per L0–L9 stage |
| `order[10]` | permutation | layer execution order |
| `head_k` | 3–12 | sparse heads selected per token |
| `write_rate` | 0.01–0.5 | dynamic cycle write strength |
| `read_temp` | 0.5–2.0 | cycle read temperature |
| `holonomy_gain` | 0.0–1.0 | SO(3) transport residual |
| `copy_gain` | 0.0–0.5 | pointer-copy mixture |
| `phase_gain[10]` | 0.02–1.0 | trig-attention phase bias |
| `memory_decay` | 0.90–0.999 | fast-state retention |

### 3.2 Protocol

At each generation:

1. a diagonal-Gaussian policy observes the current genes and validation state;
2. it samples four candidate genomes;
3. each candidate is evaluated on six held-out validation batches;
4. the candidate is accepted only if loss improves by at least `5e-4` **and**
   its parent digest still matches the current genome (a generation-CAS check);
5. an accepted genome increments the generation and is recorded; rejected
   candidates roll back;
6. the policy is updated by a policy-gradient step using the measured
   validation improvement.

No candidate can enable canonical merge, exceed the gene bounds, or change the
1 GiB ceiling.

### 3.3 Measured evolution in the reference run

Six generations, 1.08M-parameter model, seed 17. One candidate was accepted:

| Gen | Baseline CE | Candidate CE | Accepted | Gene effect |
|---:|---:|---:|---|---|
| 0 | 2.51495 | 2.51539 | no | — |
| 1 | 2.51495 | 2.51446 | no | — |
| 2 | 2.51495 | **2.51041** | **yes** | `head_k` 6 → 4 |
| 3 | 2.51041 | 2.51409 | no | — |
| 4 | 2.51041 | 2.51079 | no | — |
| 5 | 2.51041 | 2.51095 | no | — |

This is a small but real architecture change selected by held-out loss and
protected by a generation digest. It is not evidence of a large search
advantage; six generations are only a pilot.

---

## 4. HW4 WordNet matrix and training data

The compact mirror is prepared from WordNet 3.1 `data.noun`:

| Quantity | Value |
|---|---:|
| Parsed noun synsets | 82,192 |
| LM records | 25,000 train / 1,500 val / 1,500 test |
| Typed relation examples | 2,800 train / 2,285 val / 2,292 test |
| Relation labels | none, hypernym, hyponym, opposes |
| Split rule | SHA-256 of synset ID, deterministic |
| Route | `nan://GM0:SC0:MC0:BS0:0:HW0:0::HW4:0` |
| Live NANNET route | **false — local mirror** |

The WordNet matrix is used twice:

1. to initialize the base cycle field with typed relation phases (2,090 train
   relations written in the reference run);
2. to train the relation probe that writes a content phase into a cycle and
   reads it back through the same field the language model uses.

This is a controlled graph/definition benchmark. It is not an open-domain
relation extractor and not a proof that the model understands WordNet prose.

---

## 5. Raw NaN16 trigonometric kernel

`nanlmind_hermes/nan/trig_cycle.nan` is **513 pure 0xNaN words**. Every
non-empty line is:

```
NNNN_PPPPPPPP_FFFF
```

There are no comments, Python, C, or assembly lines in the committed image.

### 5.1 Contract

All angles are Q16.16 **turns** (1.0 = one full turn).

| Address | Input | Meaning |
|---|---|---|
| `0x400` | `mode` | 0 = triangle, 1 = holonomy loop |
| `0x401` | `e0a` | edge `e(0,a)` |
| `0x402` | `eab` | edge `e(a,b)` |
| `0x403` | `e0b` | edge `e(0,b)` |
| `0x404` | `ebc` | edge `e(b,c)` (mode 1) |
| `0x405` | `e0c` | edge `e(0,c)` (mode 1) |
| `0x406` | `eca` | edge `e(c,a)` (mode 1) |

| Address | Output | Meaning |
|---|---|---|
| `0x420` | `cycle` | invariant triangle/loop phase |
| `0x421` | `cos` | Q15 cosine of the cycle |
| `0x422` | `sin` | Q15 sine of the cycle |
| `0x423` | `status` | 1 = success |

The kernel reduces the phase to the low 16 fractional bits, selects a quadrant,
and looks up a 257-entry Q15 quarter-wave table at `0x100`. It uses only
NaNVM0 ops (`MOVI`, `ADD`, `SUB`, `SHL`, `SHR`, `CMP`, `LD`, `ST`, `JMP`,
`HALT`).

### 5.2 Verified algebra

`tests/test_trig_cycle_nan.py` executes the raw image through the included
NaNVM0 boundary and verifies:

- the image is pure raw words;
- triangle `e(0,a)+e(a,b)-e(0,b)` readout;
- additive vertex gauge cancellation;
- a closed loop `eab+ebc+eca` readout;
- gauge invariance of the loop;
- Q15 `cos/sin` close to the mathematical reference.

The kernel is a native algebra/trig primitive. It is **not** the full trained
language model executing in NaN16. Running the 1.08M-parameter model natively
would require an N16I tensor/operator ABI that does not exist yet.

---

## 6. Measured reference results

Run: `experiments/round042/results/trigon_results_17.json`
(120 LM steps + 60 relation steps + 6 architecture generations, seed 17,
d_model 80, 10 layers, 4 attention heads, byte vocabulary 259).

### 6.1 Model and language modelling

| Quantity | Value |
|---|---:|
| Total parameters | 1,084,110 |
| Layer parameters | 815,605 |
| Cycle field parameters | 131,072 |
| Router parameters | 11,296 |
| Byte vocabulary | 259 |
| Train / val / test records | 24,989 / 1,499 / 1,499 |
| Best validation bpb | 3.6889 (seed 17) / 3.6599 (seed 29) |
| Held-out test bpb | **3.6009** (17) / **3.5858** (29) |
| Held-out test perplexity | **12.134** (17) / **12.007** (29) |
| Held-out next-byte top-1 | **26.97%** (17) / **27.68%** (29) |
| Evaluation tokens | 4,483 per seed |

Seed 43 was interrupted during final evaluation after its LM training had
completed; its best validation bpb was 3.9029 and its resumed test bpb was
3.8336. It is recorded but not used in the headline. The relation-task
checkpoint for seed 43 was not retained, so its relation score is not reported.

This is a real byte-level autoregressive language model: it samples its own
next byte, not a stored definition. It is **not fluent**; see §8.

### 6.1b Matched byte Transformer control

Same corpus, split, token budget, context (48), batch (8), optimizer family
(AdamW) and 120 steps. The control is a standard causal byte Transformer with
learned positions, 10 layers, 4 heads, d_model 96, no cycle field, no
trigonometric rotor bank, no 256-head interface and no NANNET-P.

| Model | Seed | Parameters | Wall time | Test bpb | Test ppl | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| TRIGON-9 | 17 | 1,084,110 | 651 s | **3.6009** | **12.134** | **26.97%** |
| TRIGON-9 | 29 | 1,084,110 | 582 s | **3.5858** | **12.007** | **27.68%** |
| byte Transformer | 17 | 1,217,731 | 55 s | 3.7377 | 13.340 | 24.05% |
| byte Transformer | 29 | 1,217,731 | 53 s | 3.7329 | 13.296 | 23.67% |
| byte Transformer | 43 | 1,217,731 | 56 s | 3.7234 | 13.209 | 24.65% |

TRIGON-9 is better on this internal split with fewer parameters, but it uses
about 12× the wall time because the sparse cycle field and trigonometric
blocks are not optimized on CPU. This is a data/parameter-comparable control,
**not a compute-matched control**. A compute-matched baseline may close the
gap; that is an open gate (T-09).

### 6.2 Typed relation probe

| Metric | Seed 17 | Seed 29 |
|---|---:|---:|
| Test correct | 1,041 / 2,292 | 1,018 / 2,292 |
| Accuracy | **45.42%** | **44.42%** |
| Chance (4 balanced classes) | 25% | 25% |
| Test split | source synsets held out by hash | same |

The probe writes a content phase into the shared cycle field and reads it back.
It is a controlled relation benchmark, not open-domain extraction. Hypernym vs
hyponym confusion is the dominant error; `opposes` is the strongest class.

### 6.3 Gauge invariance

| Metric | Before gauge | After gauge |
|---|---:|---:|
| Test bpb | 3.596665 | 3.596665 |
| Test perplexity | 12.097733 | 12.097734 |
| Top-1 | 27.144% | 27.144% |
| Argmax agreement on fixed batch | — | **100%** |
| Max cycle phase error | — | **1.19e-6** |

The gauge is applied to the physical edge coordinates, not to the cycle
readout. The readout is invariant by construction. A conventional memory is not
exposed to this coordinate change; the control for a *unique* geometric
advantage remains open.

### 6.4 Component ablations

All ablations use the same trained weights and the same 2,472-token evaluation
slice; only the named component is disabled.

| Variant | bpb | Δ vs full | Interpretation |
|---|---:|---:|---|
| full | **3.5967** | — | reference |
| no dynamic write (`write_rate=0`) | 3.5977 | +0.0010 | dynamic writes have a small positive effect |
| no memory read | 3.6025 | +0.0058 | memory channel carries measurable next-token information |
| no trig rotor | 3.5902 | **−0.0065** | trig path is slightly negative on byte LM at this budget |
| no SO(3) holonomy | 3.5868 | **−0.0099** | holonomy is slightly negative on byte LM at this budget |
| no head readout | 3.5983 | +0.0016 | heads carry a small amount of next-token information |
| no copy | 3.5978 | +0.0011 | copy channel has a small positive effect |

Seed 29 keeps the memory advantage (no memory 3.6185 vs full 3.5903) and the
trig ablation goes the other way (no trig 3.5951), while holonomy again helps
the ablation slightly (no holonomy 3.5676). The component effects are small and
seed-dependent; do not turn any one of them into a headline.

**Honest reading:** the memory/head/copy channels are measurably used but small.
The trig/holonomy paths are not yet earning their compute on the byte-language
objective. They are retained because the order probe below shows the algebra is
necessary for order-sensitive composition, and because the user explicitly
required the trigonometric layer to remain a large component. This is a
negative result for the current LM objective, not a hidden success.

### 6.5 Order probe: where the trigonometric layer wins

`experiments/round042/order_probe.py` uses the same `rotation_holonomy`
algebra. It presents the same three relation symbols in different orders and
balances the three labels.

| Seed | Full ordered SO(3) features | Phase-only control |
|---|---:|---:|
| 17 | **100%** | 33.33% (chance) |
| 29 | **100%** | 33.33% (chance) |
| 43 | **100%** | 33.33% (chance) |

The phase-only control is permutation-invariant; it cannot solve the probe.
The ordered product can. This is a controlled synthetic probe, not open-domain
reasoning, and a conventional sequence model could also solve it. It establishes
that the trig/holonomy layer is doing real order-sensitive work, even though
the byte LM does not yet reward it.

### 6.6 Free samples (honest)

Prompt → sampled continuation (temperature 0.82, top-k 40, 40 bytes):

```
a person who        → a person whocoor use or; orrorildibs wirthes s thlru
a place where       → a place wheres helAbiteang of of fovil ched she te pe
an instrument used for → an instrument used forineaoeth fa; tomaluilalansisin the oregr
the act of          → the act of gre heat of cers shing thexa therlly on
a small animal      → a small animal: dit ise ind of alm-s of a ig theinh th
```

These are genuine free samples, not templates, but they are **not fluent**.
The model is undertrained for open-ended text and the vocabulary is raw bytes.
R039's fluency problem is not solved by this run.

---

## 7. What is genuinely new in this synthesis

1. **Trigonometry is structural, not positional.** Every L0–L9 layer has a
   complex rotor state and a phase-alignment attention bias; L7–L9 add ordered
   SO(3) transport.
2. **The same cycle field is trained by language and typed relations.** The
   relation probe writes and reads the same 32,385-cycle field that conditions
   the decoder.
3. **Dynamic content-dependent writes replace a static adapter.** The model
   writes sparse cycle updates at every token; the R039 fixed 16×16 adapter
   lesson is addressed structurally.
4. **Heads have a direct token-loss path.** A 256-way head readout is trained
   with an auxiliary next-token loss, so heads are not decorative.
5. **Architecture evolution is gated.** NANNET-P changes `head_k`/gates/order
   under a held-out-loss and generation-digest gate; one change was accepted in
   seed 17 and two in seed 29.
6. **The native contract is raw NaN16.** The triangle/holonomy/sin-cos kernel
   is pure 0xNaN words and executes on the included NaNVM0 boundary.

---

## 8. Honest boundaries and open gates

- **Not a fluent LM.** Samples are malformed; 120 LM steps and byte-level
  vocabulary are insufficient for open-domain fluency.
- **Not a compute-matched win.** The byte Transformer control is
  data/parameter-comparable and has more parameters, but it is about 12× faster
  on CPU. The TRIGON-9 advantage is not yet separated from extra compute.
- **Head usage collapsed in the seed-17 checkpoint.** With the NANNET-P
  accepted `head_k=4`, a 16-document probe selected only **4 unique heads out
  of 255**. The router's hard top-k selection is rich-get-richer. The head
  readout still carries next-token information, but this is not "all 256 heads
  at full capability". A load-balancing/diversity pressure is a required next
  step (T-19); it was not present in the measured run.
- **Not open-domain relation extraction.** The relation probe uses WordNet
  lemma pairs and a controlled graph; it does not parse arbitrary prose.
- **Not a live NANNET route.** `mode: local-unverified`; the HW4 address is a
  local read-only mirror. No canonical merge, promotion, or lease.
- **Not the whole model in NaN16.** The native kernel is the trig/cycle
  primitive. The 1.08M-parameter model runs in host PyTorch.
- **Holonomy is currently neutral/negative on byte LM.** It is retained for
  order-sensitive composition and must be evaluated on a task that rewards
  order, not hidden behind the LM number.
- **Two complete LM seeds, one interrupted.** Seeds 17 and 29 completed; seed
  43 was interrupted during final evaluation. The order probe has three seeds.
- **Snapshot/transaction semantics** from R040 are not yet ported to the
  reference cycle field. The model is experimental, not crash-safe.
- **No Android ARM energy/RSS claim.** The reference run is host Termux.

### Required next gates

| Gate | Test |
|---|---|
| G-01 | Matched conventional byte LM (same parameters/compute/data) |
| G-02 | Three-seed TRIGON-9 LM replication |
| G-03 | Order-sensitive composition task inside the LM, not only the probe |
| G-04 | Retention after additional writes and after continued gradient training |
| G-05 | Copy/abstain/compose end-to-end with rendered answers |
| G-06 | Native N16I operator ABI for the trained model’s hot path |
| G-07 | Android ARM RSS/cache/energy measurement |
| G-08 | Gauge audit on the resident 8 MiB root, not only the reference field |
| G-09 | `opposes` and four-valued evidence in generated text |
| G-10 | Reproducible snapshot pair with generation digest |

---

## 9. Reproduce

```bash
# 1. Prepare the compact HW4 mirror (requires WordNet 3.1 data.noun)
python3 experiments/round042/prepare_data.py \
  --wordnet /path/to/wn3.1.dict/dict/data.noun

# 2. Execute the raw NaN16 trigonometric kernel tests
python3 -m unittest tests.test_trig_cycle_nan -v

# 3. Train the reference model (host PyTorch; ~8-12 min on Termux)
python3 experiments/round042/train.py \
  --steps 120 --relation-steps 60 --arch-generations 6 \
  --d-model 80 --batch 8 --context 48 --seed 17 \
  --out experiments/round042/results

# 4. Order-sensitive trigonometric probe (3 seeds)
python3 experiments/round042/order_probe.py

# 5. Matched byte Transformer control (same data/token budget)
python3 experiments/round042/baseline.py --steps 120 --d-model 96 --seed 17   --out experiments/round042/results
```

Artifacts:

| Path | Contents |
|---|---|
| `experiments/round042/trigon.py` | reference model, cycle field, genome |
| `experiments/round042/wordnet_hw4.py` | HW4 mirror, relations, prior writer |
| `experiments/round042/nannet_p.py` | architecture policy/gate |
| `experiments/round042/train.py` | training/eval/ablation/sampling |
| `experiments/round042/order_probe.py` | SO(3) vs phase control |
| `experiments/round042/baseline.py` | matched byte Transformer control |
| `experiments/round042/results/trigon_results_17.json` | measured run |
| `experiments/round042/results/samples_17.json` | free samples |
| `experiments/round042/results/order_probe.json` | three-seed order probe |
| `experiments/round042/results/baseline_transformer.json` | matched control |
| `nanlmind_hermes/nan/trig_cycle.nan` | 513-word raw NaN16 kernel |
| `nanlmind_hermes/resident/trig_cycle.py` | bounded NaNVM0 bridge for the raw kernel |
| `tests/test_trig_cycle_nan.py` | native kernel tests |
| `tests/test_resident_trig_cycle.py` | resident bridge contract tests |

---

## 10. The one-sentence claim

TRIGON-9 is a working experimental synthesis in which a byte-level generative
language model, a full 32,385-cycle gauge-invariant mirror memory, 256 sparse
heads, L0–L9 trigonometric rotor layers, ordered SO(3) holonomy, a typed HW4
WordNet probe, and a generation-gated architecture evolver are trained
together; it reaches 3.60 held-out bits/byte and 26.97% next-byte accuracy on
its internal split, beating a larger byte Transformer control (3.74 bpb,
24.05%) at ~12× the wall time, preserves predictions exactly under a vertex
gauge, and still produces malformed free text. The trigonometric layer is proven
order-sensitive by the phase-only control, but it does not yet improve the byte
LM; that is the next falsifiable problem, not a claim to hide.
