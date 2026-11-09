# NaN Recursive Generation — Round 035

## What changed
This round moves beyond single-memory definition copying. A source-disjoint, two-relation WordNet benchmark trains a 13-feature decision gate to determine whether two stored typed relations compose into a valid ancestry chain. A per-byte confidence mixture assigns probability to the newly **assembled** target answer. The final text is *not* stored in either source record. This is a small pointer/template generator; it is **not a fluent open-ended neural language model**, and its grammar is fixed.

A second 3-hop experiment reads all source strings from the **actual 8 MiB native C++ mirror memory**, after a large vertex-gauge change. It composes facts into an answer or abstains when the third source does not connect. The 3-hop algorithm is explicitly symbolic and not a trained neural 3-hop generalizer.

## Native geometry
The C++ module uses one exact 8,388,608-byte object: 256 × 128-byte head records plus 32,640 × 256-byte pair records. For each mirror pair, two directed 128-byte lanes share one physical cell; the source bytes in the experiment use a float32 slot in each independent pair cell. The native reading operation computes `e(0,a)+e(a,b)-e(0,b)`, cancelling arbitrary additive vertex gauges. The pair address mapping remains that of Round 034. Source records are inserted as independent byte strings, then the gauge is shifted. Generation assembles the final sentence *after* reading both or all three source records.

## Input and evaluation
WordNet 3.1 noun data from the user's `wn3.1.dict.zip` are parsed directly. Positives are real child → parent → grandparent WordNet chains; negatives replace a source with an unrelated edge. The two-hop train/validation/test sets are grouped by child synset and split 70/15/15. **The model has easy explicit equality indicators (whether the intermediate names match); a perfect score is not evidence of autonomous natural-language understanding.** The exact 3-hop test uses 60 real chains and 60 intentionally broken chains.

## Reproduction
```bash
python -m pip install numpy torch
python experiment.py /path/to/wn3.1.dict.zip
python recursive_test.py /path/to/wn3.1.dict.zip
```
A C++20 compiler is needed for the native `.so`; the Python scripts build it from included `native_episodic.cpp` if missing. All random seeds are fixed. Result snapshots are `results.json` and `recursive_results.json`, plus three trained router checkpoints.

## Controls and interpretation
- Conventional exact-index symbolic composition has identical reasoning accuracy on the two-hop task.
- Missing the second source makes half the balanced two-hop questions unanswerable.
- Native cycle readings preserve both stored sources after large gauge shifts, whereas raw single directed-edge readings do not.
- A two-source gate trained using explicit string-link features does not imply that the neural model has learned to parse arbitrary prose or perform free-form composition.
- The 3-hop test is a separate symbolic recursive composition path, not a generalization claim for the two-hop trained gate.
- Ordinary indexed storage can match the composition result without the 8 MiB pair field; the geometric benefit is **robustness to additive gauge drift**, not superior language modeling under matched conditions.

## Where to go next
Train a real pointer-generator on variable-length answers and multiple relation types, include WordNet gloss-derived uncertain links, and evaluate **held-out relation extraction** (from text), **answer generation** (from source graph), and long-context **perplexity** as separate benchmarks. Give both geometric memory and conventional memory the same evidence, compute, and parameters. Use a typed provenance and contradiction-aware store rather than accepting arbitrary corpus text as truth.
