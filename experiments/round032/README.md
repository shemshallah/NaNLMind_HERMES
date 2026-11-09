# Round 032 — Sparse Mirror-Pair / Typed Hyperedge Retrieval

This is the next executable step after Round 031, but **not yet a fully trained retrieval-augmented LM**. It implements native content addressing in the same 8 MiB mirror-root layout, stores actual WordNet hypernym facts as typed hyperedges with stable IDs and root-pair membership postings, and benchmarks exact and two-hop lookup against a conventional hash index.

## Reproduce

```
python prepare.py /path/to/round025/corpus.jsonl --out data --limit 50000
c++ -O3 -std=c++20 -Wall -Wextra native_recursive.cpp -o native_recursive
./native_recursive data/facts.tsv data/queries.tsv > results.json
python long_context.py
```

All inputs are derived from the previous Round 025 WordNet corpus and are included as a small numeric TSV in this archive. The unmodified user WordNet dictionary is not duplicated. Test platform: local Linux container. `mlock` succeeded here; not guaranteed on Android.

## What is actually computed

- Root: exactly 8,388,608 bytes, 256 head cells at 128 bytes, and 32,640 pair cells at 256 bytes. Two 128-byte directional lanes share each pair cell.
- Hashes of concept IDs select an anchored pair. The pair owns an inverted hyperedge posting list. Collisions are resolved with explicit source/relation comparison. All growing hyperedges live outside the immutable-location root.
- The triangle `x[0,a]+x[a,b]-x[0,b]` is read from root memory. A three-edge learning update changes that value; applying a random gauge change leaves it intact to float32 precision.
- Hypernym chains are traversed as directed typed edges. 2,000 queries are sampled from *already stored links* and are not unseen-word inference.
- The hash-map comparator indexes exactly the same facts, and is faster in this microbenchmark. No unique performance advantage established for mirror geometry.
- The delay/long-context test retrieves existing facts after 512 synthetic distractor facts. It does not test any neural decoder. A short-window-only control cannot recover facts absent from its window; both indexed memory implementations can.

## Known limitations

The current pair posting lists are headed in a separate fixed index array, not embedded directly into the root record. There is no concurrent writer, no transactional WAL, no persistent hyperedge snapshot in this round, and no GPU/quantum operation. The root's float lanes receive a demonstrative triangle update but are **not trained from next-token loss in this implementation**. Round 031 contains the direct native next-token training experiment; integrating its decoder with these postings is the remaining end-to-end task.

The corpus includes known graph links, so lookup accuracy should not be interpreted as learned generalization. The dictionary has multiple hypernyms per synset, which is why the long-context test restricts to single-parent sources.
