# NaN Recursive Learning — Round 029: falsifiable geometric advantage

## Finding and scope

A **geometric inductive-bias advantage** has been demonstrated on a **controlled, distribution-shifted WordNet 3.1 gloss-classification task**. The exact oriented triangle observable `x(0,a) + x(a,b) - x(0,b)` cancels arbitrary vertex-gradient gauge shifts. A matched nonlinear classifier on these observables generalizes to previously unseen gauge shifts far better than reading individual mirror-pair lanes or assembling inconsistent triangles.

**This is NOT evidence that NaN encodings are intrinsically superior to classical graph representations.** A conventional graph implementing the same gauge-invariant observable matches the NaN readout exactly. This is NOT a general language-model win and NOT a quantum effect. It is an experimentally tested example of a geometrically constrained feature map providing an appropriate invariance and greatly improving out-of-distribution inference.

## What was tested

- Source: the extracted **WordNet 3.1** `corpus.jsonl` from Round 025. The user-supplied dictionary is the upstream source. Requires `WORDNET` corpus in `/mnt/data/nan_round025/corpus.jsonl` (edit `SRC` to use another path).
- WordNet task: 16 well-supported noun lexicographer categories, 480 synsets each (7,680 total), stratified 70/15/15 **synset-disjoint** train/validation/test. The TF-IDF vectorizer and 48-dimensional SVD are fit **only on training glosses**. No WordNet relation label or test gloss is used during representation fitting.
- The 48 semantic coordinates are embedded in 48 disjoint fundamental triangles of the complete 256-head graph. The sampled oriented edges per triangle are `(0,a)`, `(a,b)`, and `(0,b)` for `a=1+2k,b=a+1`.
- Write oriented edge field: `x_0a=g0-ga`, `x_ab=z_k+ga-gb`, `x_0b=g0-gb`, so triangle flux `F_k=x_0a+x_ab-x_0b=z_k` for every gauge `g`.
- Training/validation gauge sigma: 0.02; test-time unseen vertex gauge sigma: 3.0. This deliberately measures **robustness to a specified gauge distribution shift** rather than general prose comprehension. The 48-dimensional classifiers use exactly 4,176 parameters and matched optimization/hyperparameters. All models use validation loss for checkpoint selection. Each test set has 1,152 held-out synsets; three seeds.
- Readouts: (1) correct fundamental triangles; (2) raw `(a,b)` pair; (3) incorrectly linked triangles (shifted spokes); (4) a conventional implementation of exactly the correct triangle formula; (5) raw clean-signal control. Additional retrained model sees all 144 oriented edges and strong gauge noise during *training* (5,168 parameters versus 4,176 for cycle). Its purpose is to test whether learned nuisance cancellation catches up with geometric cancellation given these data/training budgets.

## Results (mean of seeds 17, 29, 41)

| Readout | Held-out accuracy | CE (nats) |
| --- | ---: | ---: |
| True cycle triangle, distribution-shift gauge | 52.691% | 1.63859 |
| Raw directed pair, distribution-shift gauge | 10.966% | 11.67543 |
| Wrong triangle indexing, distribution-shift gauge | 10.706% | 11.62996 |
| Conventional correct triangle observable | 52.691% | 1.63859 |
| Raw pair with zero gauge train and test | 52.691% | 1.63859 |
| Learned all-three-edge control, gauge-augmented training/test | 26.823% | 2.65367 |
| Correct triangle, gauge-augmented training/test | 52.691% | 1.63859 |

The two conditions with identical correct invariant are intentionally tied. Accuracy varies by seed (roughly 51.0–53.6% for cycles). Baselines are not an exhaustive search of large transformer, GNN or gauge-equivariant architectures. In particular, a fixed correct triangle readout is a strong conventional baseline that **ties**. The input construction is *designed to obey* gauge geometry, and the success is specific to that setting. The experiment does not prove advantages on ordinary unstructured Gutenberg language data.

## Mathematical proof

Let `B` be the oriented vertex-edge incidence matrix, `z` the semantic cycle field, and `g` a vertex gauge. Observed edge field is `x=z+B^T g`. For every cycle vector `c` obeying `Bc=0`:

`c^T x = c^T z + (Bc)^T g = c^T z`.

Thus **cycle measurements exactly eliminate any additive vertex gauge**; no model needs to learn this invariance from samples. The full complete-graph cycle space has dimension 32,385. The experiment uses a 48-dimensional subspace, rather than all cycle dimensions, to make a controlled held-out learning test feasible.

## Native 8 MiB test

`native_mirror.cpp` constructs one contiguous 8,388,608-byte root:

- 256 head records × 128 bytes
- 32,640 unique pair records × 256 bytes
- Each pair has independently addressable 128-byte forward and reverse lanes.
- All 32,640 forward/reverse pair mapping identities pass.
- In-place skew-pair writes, triangle reads, and explicit local gradient descent update work.
- 100 random batches of 48 triangles have max float32 gauge error 1.86e-6 and exact reverse antisymmetry.
- Finite-difference gradient error approximately 6e-5; local step reduces synthetic loss 0.0658 to 0.0322.
- 500,000 triangle reads took 1.41 ms in one Linux benchmark, which **must not** be interpreted as portable/Android throughput.
- Linux `mlock` succeeded during this run. Residency lock depends on host privileges and memory limits; 8 MiB is not CPU register-resident.

`native_mirror.cpp` is the direct-memory proof-of-implementation; `experiment.py` is the batched, reproducible semantic-learning experiment of the *same exact edge mathematics*. The neural experiment does **not** currently train the C++ root during supervised optimization; that integration from Round 026 is a subsequent engineering target.

## Reproduction

```
python experiment.py --seeds 17 29 41
python strong_control.py
g++ -std=c++20 -O3 -Wall -Wextra native_mirror.cpp -o native_mirror
./native_mirror
```

Requires Python 3, NumPy, PyTorch, scikit-learn, C++20. Adjust `SRC` in `experiment.py` to your extracted WordNet corpus path.

## Follow-on falsification

1. Use the same geometric triangle front-end on a **real, naturally gauge-shifted** task, not only generated shifts.
2. Train a 256-head/32,385-cycle sparse operator in resident C++ memory end-to-end; compare to a feature-identical strong conventional graph model.
3. Use full book-disjoint language modeling, multi-hop WordNet queries, longer training and multiple seeds; report perplexity, exact inference, throughput and memory costs.
4. Include learned gauge-equivariant architectures and group-augmentation baselines, not just raw edge baselines.
5. Measure effective use of mirror pairing versus separate directed storage on actual Android hardware with cache/memory profiling.
