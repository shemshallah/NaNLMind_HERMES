# Round 038 research log

- Source: recovered original `nan_recursive_round037.zip` from user Library; archive includes unchanged writer/checkpoints/native C++/sequence generator.
- Dependency: WordNet dictionary separate ZIP SHA-256 `1820414c1056652b56717df5794609ef0c91aae033d3fc54ddc9846249294285`.
- Implemented independent gauge-invariant numerical feature lanes (`native_trig.cpp`), SO(2) and SO(3) operators, closure-trace features, autoregressive token-conditioned offsets and learned candidate attention in `round038.py`.
- Added a strong DeepSets control, a conventional GRU, learned/frozen phase and rotation ablations, holonomy-only and full models.
- Trained principal WordNet seed 29 with 420 updates/variant, auxiliary seeds 17/43 with 420 updates for 4 targeted variants.
- Trained principal order seed 41 with 500 updates/variant, auxiliary seeds 17/43 with 500 updates for 4 targeted variants.
- Independent math/native audit: `geometry_checks.json`.
- Generated answers replay: 180/180 correct restricted-state labels from `trig_wordnet_full_29.pt`.
- Original R037 native checks rerun: 600 cells, max error ~1.81e-5, exact restore.
- Original R037 neural generator rerun: 90/90 semantically correct; 90/90 unchanged after gauge.
- Findings: Full hybrid consistently reaches 100% across six task-seed instances. Commutative phase encoding cannot distinguish permutations of the same multiset; trained SO(3) and DeepSets can. More robust comparisons required.
- Non-results: no demonstration of emergent free-form prose understanding, native gradient updates or unique representation/storage advantage.
