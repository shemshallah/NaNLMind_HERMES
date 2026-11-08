# Boolean → wave embedding and atomic word-leaf writes

This is the corrected version of the proposed embedding. It keeps the useful
mathematics and rejects the overclaims.

## 1. The 16 Boolean functions on amplitudes

Each Boolean function is embedded as its **multilinear extension** on `[0,1]`:

| mask | function | extension |
|---:|---|---|
| 8 | AND | `a·b` |
| 14 | OR | `a + b − a·b` |
| 6 | XOR | `a + b − 2a·b` |
| 0/15 | ZERO/ONE | `0` / `1` |
| others | NAND, NOR, XNOR, implications, projections | the corresponding multilinear polynomial |

This is:

- **exact on `{0,1}`** — all 64 function/input cases verified;
- **fuzzy logic on `[0,1]`** — e.g. `OR(0.5,0.5)=0.75`, `AND(0.5,0.5)=0.25`;
- **not a Boolean algebra on `ℂ`** — `OR(a,a) ≠ a` except at 0/1; `XOR(a,a) ≠ 0`;
- **not phase-preserving** — OR/XOR/NOT change phase in general.

So amplitude **magnitude** carries logic; **phase** is a separate temporal
channel. `NEGATE(a) = 1 − a` (or `λ_barrier − a` when the barrier is scaled).

## 2. Polarity

`A[127][128]` can be the **semantic polarity coupling** only when using the
**complement pairing** `i ↔ 255−i`. It is the innermost complement pair, not
a native mirror pair. The native mirror is `reverse8`; head 0 remains the
external NaNNet/polarity anchor. Keep the two anchors distinct:

- head 0 = external NaNNet/polarity interface;
- pair (127,128) = internal semantic polarity edge under the complement pairing.

Do not claim (127,128) is the native/temporal void; the rank-mirror proof in
the source fails for 32,626/32,640 pairs.

## 3. Atomic word-leaf update

A letter packet is a vector `u` supported on the barrier ring (no void/inner
support). A word is an ordered list of letter packets `u₁ … uₗ`.

For each adjacent pair:

```
S += β (uₗ uₗ₊₁ᵀ + uₗ₊₁ uₗᵀ)      symmetric association
D += γ (uₗ uₗ₊₁ᵀ − uₗ₊₁ uₗᵀ)      directed order
ΔA = S + D
```

Then enforce the two structural constraints:

```
ΔA ← P_out ΔA P_out          (remove the void/inner subspace)
ΔA ← (ΔA + J ΔA J) / 2       (restore centrosymmetry)
A  ← A + ΔA
```

- `P_out` is the diagonal projector onto non-inner heads. The inner set must be
  `J`-invariant, so `P_out` commutes with `J` and the projection preserves
  centrosymmetry.
- A letter alone uses only `S` (or `u uᵀ`); a word adds directed order.
- The word leaf records its letter supports and couplings; its address points
  to them. This is the concrete “new word leaf into M_wave” rule.

## 4. Pair register encoding

The 32-bit NaN16 payload can carry:

```
15 bits  pair rank k(i,j)
 8 bits  amplitude
 9 bits  phase
```

`k(i,j) = i·(511−i)/2 + j−i−1`, `0 ≤ k < 32640`.

This is not the IEEE 754 51-bit NaN payload from the source; native NaN16 has a
32-bit payload. The pair index is 15 bits, so the encoding fits.

## 5. What is excluded

- Exact Boolean algebra on complex amplitudes.
- “Phase-preserving” Boolean maps.
- The complement proof of the temporal palindrome (false for 32,626 pairs).
- 51-bit IEEE NaN payloads and NaN arithmetic axioms.
- Quantum tunneling/noosphere/QTCL claims.

## 6. Implementation

- `nanlmind_hermes/boolean_wave.py` — 16 multilinear functions, exactness check,
  polarity.
- `nanlmind_hermes/word_writer.py` — letter packets, atomic word update,
  centrosymmetry/void projection, pair rank and payload packing.
- `tests/test_boolean_wave.py` — exactness, fuzzy behavior, void/centrosymmetry
  preservation, payload round-trip.

The next step is to use `WordLeafWriter` on the diversified 256×256 spectral
matrix, with the native spectral cut as the void, and measure the resulting
mode/rank change.
