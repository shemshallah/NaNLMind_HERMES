# NaN Recursive Generation — Round 034

## Scope
This is a **reproducible open-set generation/retrieval milestone**, NOT a fluent language model, NOT a physical quantum effect, and NOT a demonstration of NaN-specific superiority. It extends the actual Round 033 WordNet key encoder and native 8 MiB cycle-copy memory with a learned **copy-or-abstain router** and byte-level **LM/copy mixture**.

## Core architecture
* A held-out WordNet lemma name selects one of 12 separately provided episodic records. Each record contains a 24-byte gloss prefix.
* A previously trained key encoder maps noisy names and candidate names to 48D vectors. A new learned router consumes cosine similarity, top-two score margin, nearest edit distance, distance to the top selected candidate, and edit-distance margin. Router selects copy or abstention.
* The copy path uses the actual native 8 MiB `C(256,2)` object. For slot t, its payload is `e(a,b)`; the physical memory computes the invariant `e(0,a) + e(a,b) - e(0,b)`. Random vertex-gauge shifts are applied before decoding.
* A 256-byte autoregressive GRU checkpoint provides a next-byte distribution. In an accepted episode the hybrid mixture is `0.05*p_LM + 0.95*p_copy`; otherwise `p_LM`. This is a tested mixture inference rule, **not jointly optimized token-level gradients**.
* A conventional indexed memory with the same retrieved record and copying rule is an exact-matched control and TIES the native cycle method.

## Data and provenance
`wordnet_episodic_pool.json` contains 3,500 previously prepared lemma/gloss-prefix examples per split (train, val, test). These were derived from the user's WordNet 3.1 archive via Round 033's split logic; it is a compact task-specific subset, not a replacement for WordNet. The task intentionally **gives the 24-byte definition to the episodic memory**. Test lemma identities are absent from the key encoder's training identities, but their definitions are available at query time as candidate records. We measure choosing and copying correctly, not deriving unseen definitions from scratch.

## Actual measured results
Three router random seeds; 600 test queries per seed (50% present and 50% absent):
* Learned router: 99.0% correct accept/abstain, 0.444% false-answer rate (over all queries), 48.944% correct copy across all queries.
* Validation-tuned edit-distance threshold: 98.167% accept/abstain; 0.444% false-answer rate.
* Validation-tuned cosine threshold: 92.0% accept/abstain, 3.222% false-answer rate.
* Under 6-character corruption (untrained shift): mean 61.89% accept/abstain, with 23.78% acceptance of present records. This is a real failure mode.
* Native audit on 60 episodes per seed: 89 of 89 accepted present-record queries returned EXACT 24-byte glosses through anchored triangle memory after gauge shifts; raw single-edge method succeeded on 0/89. Conventional indexed memory also achieved 89/89.
* Teacher-forced held-out next-byte loss on 90 present-record episodes per seed: standard LM 3.61–3.66 bits/byte; retrieved LM/copy mixture 0.18–0.45 bits/byte. Conventional indexed copy achieves **exactly the same** result. This large gain is expected when the correct answer is already stored and copyable; it is not evidence of language synthesis or a cycle-specific quality gain.

## Build and commands
```
python -m pip install numpy torch
c++ -std=c++20 -O3 -fPIC -shared native_episodic.cpp -o libepisodic.so
python demo.py
python demo.py --absent
python route_train.py
python stress.py
python hybrid_eval.py
```
The included `.so` is compiled for this Linux environment; rebuild on other systems. The project also uses existing `plain_17.pt` and `episodic_{17,29,41}.pt` checkpoints from Round 033, copied into this package. No external corpus download is required for the scripts' compact tests.

## Important limitations and next engineering direction
1. Train copy/generate gate jointly with next-token loss, including variable corruption and absent-memory negative examples.
2. Replace flat prefix byte storage by persistent typed concept/hyperedge records with evidence provenance and update-on-train behavior.
3. Test on entire held-out books and unseen questions requiring synthesis and composition, not exact copies of available gloss bytes.
4. Compare with conventional indexed-memory and gauge-invariant graph memory matched in data, parameters, and runtime; do not attribute advantages from copying to NaN encodings.
5. Benchmark resident C++ root and hot-path indices on the target Android ARM device; Linux `mlock` is not a phone guarantee.
