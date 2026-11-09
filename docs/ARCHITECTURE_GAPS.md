# Architecture gaps after the Alpine + Gemma test round

Date: 2026-11-08. Baseline at the start of this round: **312/312 tests passed in
Termux and in Alpine/proot**; the current suite includes the new session,
free-chat, WordNet, MTH, N16I and pipeline tests. The live Gemma cloud endpoint
was reachable and used for architecture review, free-chat smoke and tool-use
runs.

## Fixed in this round

| gap | change |
|---|---|
| No shipped NANNET/WordNet session contract | seven permanent documents ship under `nanlmind_hermes/session_docs/`, load at `register()`, persist a session manifest, and are injected into chat/reflection/curriculum/CLOCKED/synthesis provider calls |
| Free chat did not feed the L0-L9/255-head/scratch learner | `tools.chat` now performs one layered observation per turn (best effort) and returns a `layered_learning` receipt |
| Local demo claimed no NANNET needed but plugin tools required a paired token | local-only mode is explicit `local-unverified` when no authority is configured; a configured authority still fails closed without pairing |
| WordNet route used the legacy HW-first form as authority | canonical `nan://GM0:SC0:MC0:BS0:0:HW4:0` is used; legacy `nan://HW4:BS0:MC0:SC0:GM0` is compatibility only |
| No universal native MTH surface | 91 kernels/fixtures + portable NaN0S core ship with the plugin; architecture/compiler autodetect, native fixture verification, four real vector dispatchers, N16I differential self-test and arbitrary guest-image N16 dispatch |
| No generic N16 dispatch | `nanlm_mth_dispatch` relocates an assembled guest image and runs it under N16I with bounded steps and exact receipt |
| No constrained MTH data model | immutable NaN-executable leaves, vector/scalar/matrix/relation hubs, WordNet HW4 vector link, identities/constraints and `left::right` pairing now exist |
| Ambiguous/unsafe pairing | hub/kind/operator rules, dimension/finiteness checks, symmetry/antisymmetry identity checks and deny-by-default unknown pairings |
| MTH functions not linked into the tree | `nanlm_mth_tree` maps all 91 kernels to 44 family hubs with digests/ancestry/image audit (`91/91` complete locally) |
| No reproducible Gemma architecture review | `scripts/architecture_review.py` reviews the exact session documents and writes `docs/GEMMA_ARCHITECTURE_REVIEW.{json,md}` |

## Open gaps

1. **Typed operand ABI, not dispatch.** Generic N16 dispatch is now wired:
   `nanlm_mth_dispatch` relocates an assembled guest image and runs it under
   `n16i.nas`. The remaining gap is a typed input/output contract for each of
   the 87 fixture-only kernels; dispatch alone does not define their operands
   or output layout.
2. **General operand ABI for 87 kernels.** Only dot/cosine/add/sub have the
   admitted `mth_execute.nas` dispatcher. The other 87 are native-fixture
   verifiable, not callable with arbitrary typed inputs. Each needs an input/
   output ABI, workspace layout, bounded fuel and differential tests.
3. **WordNet object grammar.** `::synset/<offset>`, `::word/<lemma>/<offset>`
   and `::relation/<type>/<a>/<b>` are plugin-local typed references. The
   NaNROUTER v1 grammar only defines `vector|hyperedge|rule|route|bin|temporal`;
   a v2 registration or an explicit router association is required before a
   live canonical resolution may claim them.
4. **Promotion and canonical merge gate.** The local promotion policy is now
   specified exactly (support, confidence, caps, explicit approval), but the
   generation-checked BS0/MC0 merge validator and six-runner promotion barrier
   remain separate work.
5. **Native relation training in the smoke.** Free-chat `native_learning`
   reports `native_update_failed` when no `runner_command` is configured. The
   layered scratch path is real; the native topology update is not fabricated.
6. **Live NANNET lease/route.** The plugin has local device/lease clients and
   the router has native lease code, but the production Koyeb cutover and
   single-issuer migration are not performed in this round.
7. **CLOCKED/head-cortex live path.** Not exercised in this round; credentials
   and deployment remain separate.
8. **Cross-architecture evidence.** The portable C core compiles on the aarch64
   test host. x86_64/RISC-V builds and fixture runs need their own CI evidence
   before claiming universal target coverage.
9. **Gemma free-chat quality.** Gemma's responses are verbose and often report
   hypotheses as measured-sounding prose. The deterministic parser now supplies
   missing explicit `X is Y` / `X can Y` relations to native/layered learning,
   but response style and calibration remain prompt/evaluation work.

## Initial Gemma tool-learning test

`scripts/gemma_tool_learning_session.py` runs an open-ended model-steered tool
session against a **real loopback NANNET authority**. The installation is
paired with `universe:read` and `training:execute`; the whole pipeline is intact
for scoped tool use, while canonical merge/promotion is disabled. The session
uses the real plugin handlers, not mocks.

Aggregated evidence: `docs/GEMMA_INITIAL_TRAINING_REPORT.md`.
Result (`docs/GEMMA_TOOL_LEARNING_SESSION.md`):

- **12/12 tool calls passed**, **9 distinct tools used**.
- Covered: session context, 91-kernel function-tree audit, vector catalog,
  runtime status, two vector leaf creations, `connect` and `add` pairing,
  NaN16 leaf verification, two free-chat turns and learning-status inspection.
- **0 critical failures**. The first attempt exposed a real usability gap:
  Gemma supplied short vectors to `nanlm_mth_leaf_create`. The schema and
  payload now support an explicit `{"fill": <finite number>}` form that expands
  across exactly 256 elements; no silent short-array broadcast was added.
- Canonical merge was not called and remains disabled. NANNET is not blocked;
  it is the paired authority path. Local-unverified mode remains only an
  offline fallback when no authority is configured.

Journaled 3-session series (`docs/GEMMA_TRAINING_SERIES_RESULTS.md`):

- 15/15 tool calls, 0 failures, 12 distinct tools, 100% required-task coverage;
- before/after quiz **76.1% → 100.0% (+23.9)** after carrying the measured
  session journal into later sessions and the after-quiz;
- the earlier non-journaled run was flat (76.1% → 76.1%); its after-quiz was
  isolated from the session, which was an experiment bug, not a learning result;
- system state grew: turns 0→2, scratch 0→20, concepts 0→30, occupied pairs
  0→339, leaves 144→148; kernel tree stayed 91/91;
- this is in-context transfer from measured experience, not a model-weight
  update or a general-learning proof.

## Gemma's review

`docs/GEMMA_ARCHITECTURE_REVIEW.md` records the latest live
7,373-prompt-token / 1,011-completion-token review. Its highest-value asks were:

- a concrete promotion trigger (now specified in the session contract);
- a defined signed micro-routing packet for HW4 (router-owned; still open);
- merge-gate validation logic (still open);
- L0-L9 scratch-to-head mapping (implemented in the ensemble/register, needs a
  clearer public status view);
- formal supported/refuted triple syntax (now specified).

It also proposed the HW4-to-scratch projection test; `tests/test_wordnet_tool.py`
now exercises exactly that path with a mock archive and canonical HW4:0 route.
Its data-leaf review explicitly asks for content-addressed leaves, NaN16
metadata/payload images, no in-place mutation, hub ancestry checks,
symmetry/antisymmetry identity checks, denial without execution, and a new
immutable hash leaf on success. Those are now covered by `mth_leaves.py` and
`tests/test_mth_leaves.py`; all 91 kernels are linked into the function tree by
`nanlm_mth_tree`.
