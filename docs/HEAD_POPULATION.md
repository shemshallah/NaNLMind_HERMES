# Head-population diagnostics: routing, void head, specialization

The head Gram rank is the key measure: rank 1 means every head carries
the same distribution, so the spectral operator has only one mode.

| active | head0 prior | train head0 | accuracy | top-3 | rank | effective rank | mean cosine |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 0.0 | False | 0.1987 | 0.4303 | 27 | 4.9311 | 0.068551 |
| 16 | 1.0 | True | 0.1415 | 0.3841 | 27 | 2.6388 | 0.067329 |
| 16 | 0.0 | False | 0.1778 | 0.4137 | 27 | 2.6444 | 0.067344 |
| 32 | 0.0 | False | 0.1397 | 0.3748 | 27 | 1.9058 | 0.066307 |
| 64 | 0.0 | False | 0.1302 | 0.3495 | 27 | 1.4876 | 0.06508 |
| 255 | 0.0 | False | 0.1162 | 0.2949 | 1 | 1.0 | 0.063944 |

## Reading

- `active=255` trains every head on every context, so all heads converge
  to the global letter distribution (rank 1). That was the bottleneck.
- Sparse routing keeps all 256 heads available over the corpus while only
  a subset fires per context, which is what creates specialization.
- Head 0 should be the void/prior; training it or letting it dominate
  the collapse destroys the distinction between void and content.
- Higher rank/effective rank and lower mean cosine are the targets.

## What to do next

1. Keep all 256 heads, but choose active heads sparsely per context.
2. Freeze head 0 as the void/prior; do not train it on the corpus.
3. Add load-balancing and a decorrelation term to the head updates.
4. Extend context length to 3-4 letters for more distinct routes.
5. Rebuild the spectral A from a diversified population and re-measure
   the low-mode/cut leakage.
