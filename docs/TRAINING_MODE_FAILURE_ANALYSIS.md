# HERMES training-mode failure analysis and implementation contract

This document records verified defects in the current integration and the acceptance criteria for the repair.

## Confirmed causes of flat metrics

1. `nanlm_training_pipeline` and the `logic`, `words`, `sentences`, and `pipeline` presets all call `run_curriculum_experiment`, which always calls `math_train`. The native objective is the same byte-bit transition task for every mode.
2. The native Boolean foundation is rerun as a gate. Its trainer resets each head, assigns deterministic target truth functions, trains on all four rows, and evaluates those same rows. The resulting 100% is a calibration result, not a cross-run learning metric.
3. The native word-after-logic stage constructs a fresh `WordMemory` from a fixed small corpus and ten fixed queries. It does not update persistent head/mirror-pair state.
4. Normal `nanlm_chat` appends exact surface statistics and provider-proposed reflection records to the Hermes SQLite event store. It does not currently call the native runner to mutate the 256-head/mirror-pair universe. The current cross-resolution hyperedges are deterministic derived memory events, not native hypergraph mutations.
5. The byte-bit task's single held-out accuracy is currently the only reported native language-training metric. The curriculum's semantic stage names do not make it a lexical or sentence evaluator.

## Required user-facing separation

### Structured mode
Teacher-generated, schema-validated examples; deterministic oracle where possible; independent train/validation/test; native update; per-task metrics and error report. Boolean, relation/logic, phonetic, morphemic, lexical, word-completion, sentence, paragraph and narrative modes must not all route to one byte objective.

### Free-form chat mode
Ordinary dialogue remains the input. The assistant answers normally, then extracts candidate observations and relations from both sides of the exchange. Validated discoveries are submitted to the native shared-universe ingest path. Each update must affect assigned heads and mirror-pair cells and add typed hyperedges with source, evidence, polarity, uncertainty and epoch. Contradictions remain first-class. Report extraction separately from native mutation and from measured learning.

## Acceptance report

Every run reports all ten resolution coordinates independently: baseline, post-training score, test count, coverage, error examples, source/test digests, and before/after native state digests. Unsupported probes are explicitly marked unsupported. Also report changed heads, changed mirror pairs, inserted hyperedges, duplicate events, rejected events, and whether the update survived restart.

A training call cannot return `status=ok` solely because the process exited 0 or a JSON trace exists. It must verify a state mutation (or explicitly report `no_op`) and run the relevant held-out probe.

See the core repository's `docs/82_TRAINING_MODES_AND_CONVERSATIONAL_LEARNING.md` for the native contract.
