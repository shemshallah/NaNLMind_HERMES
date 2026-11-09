# Claim and theorem ledger — Round 038

| Claim | Status | Evidence and limits |
|---|---|---|
| Exact 8 MiB native pair root and preserved legacy ABI | **Verified** | C++ static_assert and `test_trig_geometry.py`, legacy `test_native.py` |
| Additive vertex gauge cancels around anchored triangles (float32 tolerance) | **Verified (numerical)** | `geometry_checks.json`, `round038_seed29_results.json`; not bitwise invariant under finite precision |
| Snapshot can restore native memory exactly | **Verified** | Both C++ native tests |
| SO(2) rotations commute; generic SO(3) rotations do not | **Verified** | Independent matrix identity/stress test |
| Trace of closed transport holonomy is invariant under consistent local-frame changes | **Verified (math/numerics)** | `test_trig_geometry.py`; does not imply whole learned network is equivariant under arbitrary frame gauges |
| Angle and token-conditioned phase receive gradient from next-token cross-entropy | **Verified** | Nonzero gradient norms in `geometry_checks.json` |
| Contradictions survive rather than phase-cancel | **Verified in four-state task** | Both state produced from independent support/refute truth channels, not general real-world inconsistency resolution |
| End-to-end native evidence -> differentiable reasoning -> autoregressive restricted generation | **Verified in controlled episodes** | `generated_answers_038.json`, trained checkpoint; writer frozen and intermediate entities supplied |
| Full hybrid test accuracy across 3 seeds = 100% | **Verified for the specified two synthetic/controlled tasks** | Per-seed reports, no open-domain extrapolation |
| Trigonometry inherently outperforms all conventional models | **NOT ESTABLISHED** | DeepSets can also achieve 100% |
| Native root learns parametric rotations by itself | **NOT ESTABLISHED** | Rotation matrices stored in PyTorch checkpoint, not in native root |
| General natural-language understanding or free text generation | **NOT ESTABLISHED** | Restricted writer and output token vocabulary |
| Genuine mathematical/geometric advantage at equal compute and parameters in natural prose | **OPEN** | Needs matched naturally occurring prose, nontrivial extraction, retention and reasoning |
