# Head-population diagnostics: routing, void head, specialization

The head Gram rank is the key measure: rank 1 means every head carries
the same distribution, so the spectral operator has only one mode.

| active | context | balance | router | accuracy | top-3 | rank | effective rank | mean cosine |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 8 | 2 | False | hash | 0.1935 | 0.4335 | 27 | 4.7746 | 0.068631 |
| 8 | 3 | False | hash | 0.1598 | 0.3635 | 27 | 2.4877 | 0.068641 |
| 8 | 3 | True | hash | 0.1328 | 0.3353 | 27 | 1.8469 | 0.063906 |
| 8 | 4 | True | hash | 0.1322 | 0.3399 | 27 | 1.7897 | 0.06391 |
| 16 | 3 | True | hash | 0.1302 | 0.308 | 27 | 1.4316 | 0.063926 |
| 8 | 3 | True | hash | 0.1392 | 0.3353 | 27 | 2.0158 | 0.063793 |
| 8 | 2 | False | reward | 0.1412 | 0.3632 | 27 | 11.6794 | 0.070584 |
| 255 | 2 | False | hash | 0.1162 | 0.2949 | 1 | 1.0 | 0.063944 |

## Reading

- `active=255` trains every head on every context, so all heads converge
  to the global letter distribution (rank 1). That was the bottleneck.
- Sparse routing keeps all 256 heads available over the corpus while only
  a subset fires per context, which is what creates specialization.
- Head 0 should be the void/prior; training it or letting it dominate
  the collapse destroys the distinction between void and content.
- Higher rank/effective rank and lower mean cosine are the targets.
- Longer contexts and load-balanced routing are expected to raise rank and
  accuracy; decorrelation removes the shared global component.

## What to do next

1. Keep all 256 heads, but choose active heads sparsely per context.
2. Freeze head 0 as the void/prior; do not train it on the corpus.
3. Add load-balancing and a decorrelation term to the head updates.
4. Extend context length to 3-4 letters for more distinct routes.
5. Rebuild the spectral A from a diversified population and re-measure
   the low-mode/cut leakage.
