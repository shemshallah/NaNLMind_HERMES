# Geometric theorem and limits — Round 042

For a graph on 256 labelled vertices, let each undirected physical pair carry two antisymmetric oriented scalar fields `E^(k)` (`k=0` support, `k=1` refutation). Define anchored triangles relative to vertex 0 by `F^(k)_{ab}=E^(k)_{0a}+E^(k)_{ab}-E^(k)_{0b}` for `1<=a<b<=255`.

**Theorem (gauge invariance).** For any vertex potentials `g_i`, the transformation `E^(k)_{ij} <- E^(k)_{ij} + g_i-g_j` preserves every `F^(k)_{ab}` exactly in real arithmetic.

Proof: the added terms sum to `(g_0-g_a)+(g_a-g_b)-(g_0-g_b)=0`. This holds independently for each lane; it does not assume that either field is flat.

**Dimension statement.** There are `256*255/2=32,640` unoriented edges. The image of the potential-gradient operator on a connected 256-vertex graph has dimension 255 (constant potentials are in its kernel). The orthogonal complement, and equivalently the independent cycle coordinates, has dimension `32,640-255=32,385`. A complete anchored triangle basis can be selected using all pairs `1<=a<b<=255`, of which there are `255*254/2=32,385`.

**Useful but limited consequence.** A value stored as triangle circulation is unaffected by potentially large gauge drift, while a raw individual edge value is not. The same invariant may be realized using conventional floating-point graph arrays; this is a benefit of the *chosen geometric representation*, not a proof that NaN encodings provide computational advantage or that physical quantum superposition occurs.

**Training interface.** The C++ setter adjusts a physical pair so its anchored circulation equals the requested coefficient. The Python autograd custom function exposes a surrogate identity Jacobian from requested coefficient to retrieved circulation, which is exact in real arithmetic for this setter and approximate to float32 precision in native execution. This permits token-loss gradients to reach the neural writer that chooses the coefficients. It does not mean the physical floats implement an autonomous gradient optimizer.

**Four-valued evidence.** Two independent cycle lanes allow `support ∈ {0,1}` and `refute ∈ {0,1}`. The four combinations correspond to neither, support, refute, and both. No rule of explosion follows from `both`, because the system does not derive arbitrary additional claims from it. The trained speaker uses these native evidence signals to generate bounded grammatical responses. This structure is related to familiar four-valued/paraconsistent reasoning and is not asserted as a novel mathematical theorem.
