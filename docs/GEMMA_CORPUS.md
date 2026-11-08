# Gemma conceptual-superposition corpus

Teacher: `gemma4:31b-cloud`. Statements feed NaNLM native chat and the
C(256,2) concept matrix; meaning accumulates by association and typed
relation, not by assigning a word to a cell.

## Round 1

To maximize matrix density in a 32,640-cell space, I have selected 42 concepts that overlap across domains. By using polysemous terms like 'element' and 'field', the model is forced to distinguish meaning via the surrounding pair-cluster (superposition). The relations are designed to create a dense web of associations: physical properties link to chemical elements, which link to geometric structures, which link to abstract logic. This ensures that no concept is an island, forcing the identity of a concept to be defined by its relational coordinates rather than a static index.

- concepts: 74
- occupied pairs: 217 / 32,640 (density 0.006648)
- average degree: 4.76
- known probes: 1/14
- unknown abstained: 0/6

| statement | NaNLM reply | relations |
|---|---|---|
| An atom is a member of a chemical element. | Atom is a member of a chemical element. | member_of:chemical |
| A chemical element is a component of a molecule. | Chemical element is a component of a molecule. | component_of |
| A molecule is formed by a chemical bond. | Molecule is formed by a chemical bond. | is |
| An electron is a part of an atom. | Atom is a part of an atom. | is |
| A proton is a part of a nucleus. | Part of a molecule is formed by a chemical bond. | is |
| A nucleus is a part of an atom. | Nucleus is a part of an atom. | is |
| Heat is a form of energy. | Heat is a form of energy. | is |
| Pressure is a result of force applied to a plane. | Pressure is a result of force applied to a plane. | is |
| A point is a geometric element. | Element is a part of an atom. | member_of:geometric |
| A line is a geometric element. | Geometric element is a result of force applied to a plane. | member_of:geometric |
| A plane is a geometric element. | Plane is a geometric element. | member_of:geometric |
| A volume is a property of a sphere. | Volume is a part of an atom. | is |
| A sphere is a three-dimensional volume. | Sphere is a three dimensional volume. | is |
| An angle is a relation between two lines. | Angle is a part of an atom is a relation between two lines. | is |
| A dimension is an attribute of space. | Dimension is an attribute of space. | is |
| Symmetry is an attribute of a sphere. | Sphere is a part of an attribute of space. | is |
| A vector is a representation of force. | Force applied to a plane. | is |
| A field is a region of space. | Space a part of an atom is a geometric element. | is |
| A field is a set of values. | Field is a part of an atom. | is |
| Gravity is a force. | Force applied to a plane is a geometric element. | is |
| Gravity is a result of mass. | Gravity is a result of mass. | is |
| Mass is an attribute of an atom. | Mass is an attribute of an atom. | is |
| Acceleration is a change in velocity. | Change in velocity. | is |
| Velocity is a change in position over time. | Velocity is a change in position over time. | is |
| Time is a dimension of space. | Dimension is an attribute of space. | is |
| Entropy is a measure of disorder in a state. | Measure of disorder in a state. | is |
| Causality is the relation between a cause and an effect. | Relation between a cause and an effect. | is |
| A cause produces an effect. | Cause and an effect. | co-occurrence |
| Logic is a set of relations. | Set of a molecule is formed by a chemical element. | is |
| A member is a part of a set. | Member is a part of an atom. | is |
| A subset is a part of a set. | Part of an atom is a set of relations. | is |
| An attribute is a property of an identity. | Attribute of an atom is a geometric element. | is |
| An opposite is a relation of contrast. | Relation between a cause produces an effect. | is |
| A state is a condition of a system. | State is a part of an atom. | is |
| Change is a transition between states. | Change in position over time. | is |
| Energy causes a change in state. | Energy causes a change in state. | causes |
| Heat causes a change in volume. | Heat causes a change in volume. | causes |
| Pressure causes a change in state. | Pressure causes a change in state. | causes |
| An electron is an opposite of a proton in charge. | Electron is a part of an atom. | opposition |
| A point has no dimension. | Point has no dimension. | has |
| A line is a one-dimensional volume. | Line is a change in volume. | is |
| A plane is a two-dimensional volume. | Two dimensional volume. | is |
| Force causes acceleration. | Force causes a change in state. | causes |

## Probe results

| question | NaNLM | hit |
|---|---|---|
| What is a part of an atom? | Part of an atom is a change in position over time. | False |
| What is a geometric element? | Geometric element is a part of an atom. | False |
| What causes acceleration? | Causes a change in position over time. | False |
| What is a form of energy? | Energy causes a change in position over time. | False |
| What is a result of mass? | Mass is an attribute of space. | False |
| What is a dimension of space? | Dimension is an attribute of space. | False |
| What is a relation between a cause and an effect? | Relation between a cause and an effect. | False |
| What is a property of a sphere? | Sphere is a part of an atom. | False |
| Which elements are geometric? | Geometric element is a change in state. | False |
| Which elements are chemical? | Chemical element is a part of an atom. | True |
| Which concepts relate to change? | Change in state is a part of an atom. | False |
| Which concepts relate to sets? | Concepts is a part of an atom is a change in state. | False |
| Which concepts are forces? | Concepts is a part of an atom is a change in volume. | False |
| Which concepts are attributes? | Attributes a part of an atom is a change in state. | False |

| unknown question | NaNLM | abstained |
|---|---|---|
| Does a molecule have a dimension? | Molecule is formed by a chemical bond. | False |
| Is entropy a geometric element? | Entropy is a geometric element. | False |
| Can a vector be a member of a set? | Vector is a part of an atom. | False |
| Does a proton cause gravity? | Gravity is a part of an atom. | False |
| Is a line an opposite of a sphere? | Line is a part of an atom. | False |
| Does logic have a volume? | Logic is a part of an atom. | False |

## Round 2 (targeting matrix gaps)

Round 1 showed a collapse where 'part of an atom' and 'change in state' became attractor basins, causing the model to hallucinate these patterns across unrelated concepts (e.g., 'Sphere is a part of an atom'). To counteract this, Round 2 introduces 'structural anchors' and 'disjoint domain markers'. I am introducing concepts that force the model to differentiate between 'element' as a chemical entity, 'element' as a geometric primitive, and 'element' as a set member. By creating tight, domain-specific clusters (e.g., the 'Thermodynamics' cluster and the 'Topology' cluster) that only share a few high-level bridge concepts (like 'energy' or 'state'), the matrix density will increase without collapsing into a single global attractor. I am specifically targeting the 'opposite' and 'attribute' relations which were under-represented.

- concepts: 138
- occupied pairs: 543 / 32,640 (density 0.016636)
- average degree: 6.7
- known probes: 1/12
- unknown abstained: 0/6

## Example concept leaves

```json
[
  {
    "concept": "atom",
    "count": 6,
    "degree": 14,
    "links": [
      "atom\u0000attribute",
      "atom\u0000bonding",
      "atom\u0000capacity",
      "atom\u0000chemical",
      "atom\u0000component",
      "atom\u0000determine",
      "atom\u0000electron",
      "atom\u0000electrons",
      "atom\u0000element",
      "atom\u0000mass",
      "atom\u0000member",
      "atom\u0000nucleus",
      "atom\u0000part",
      "atom\u0000valence"
    ],
    "leaf_id": "298a1e7e1f064cd1fc823fa9288b9f3c"
  },
  {
    "concept": "member",
    "count": 4,
    "degree": 7,
    "links": [
      "atom\u0000member",
      "chemical\u0000member",
      "collection\u0000member",
      "element\u0000member",
      "larger\u0000member",
      "member\u0000part",
      "member\u0000set"
    ],
    "leaf_id": "6362c5bd15fffd3d7ceea63808e3072e"
  },
  {
    "concept": "chemical",
    "count": 8,
    "degree": 20,
    "links": [
      "atom\u0000chemical",
      "atoms\u0000chemical",
      "between\u0000chemical",
      "bond\u0000chemical",
      "chemical\u0000component",
      "chemical\u0000consists",
      "chemical\u0000covalent",
      "chemical\u0000defined",
      "chemical\u0000element",
      "chemical\u0000formed",
      "chemical\u0000ionic",
      "chemical\u0000isotopes",
      "chemical\u0000member",
      "chemical\u0000molecule",
      "chemical\u0000more",
      "chemical\u0000nucleus",
      "chemical\u0000number",
      "chemical\u0000one",
      "chemical\u0000protons",
      "chemical\u0000substance"
    ],
    "leaf_id": "84680e0c5b3215d8463ad6bddd20a98d"
  },
  {
    "concept": "element",
    "count": 15,
    "degree": 36,
    "links": [
      "atom\u0000element",
      "chemical\u0000element",
      "collection\u0000element",
      "component\u0000element",
      "consists\u0000element",
      "constant\u0000element",
      "cube\u0000element",
      "curvature\u0000element",
      "defined\u0000element",
      "dimension\u0000element",
      "direction\u0000element",
      "element\u0000faces",
      "element\u0000geometric",
      "element\u0000has",
      "element\u0000isotopes",
      "element\u0000larger",
      "element\u0000line",
      "element\u0000logic",
      "element\u0000magnitude",
      "element\u0000member",
      "element\u0000molecule",
      "element\u0000more",
      "element\u0000no",
      "element\u0000nucleus",
      "element\u0000number",
      "element\u0000one",
      "element\u0000plane",
      "element\u0000point",
      "element\u0000proposition",
      "element\u0000protons",
      "element\u0000set",
      "element\u0000six",
      "element\u0000sphere",
      "element\u0000square",
      "element\u0000such",
      "element\u0000vector"
    ],
    "leaf_id": "27f45c2e28d5cccbc3fe01f40fa1b9ac"
  },
  {
    "concept": "component",
    "count": 6,
    "degree": 10,
    "links": [
      "atom\u0000component",
      "chemical\u0000component",
      "component\u0000cube",
      "component\u0000electron",
      "component\u0000element",
      "component\u0000line",
      "component\u0000molecule",
      "component\u0000plane",
      "component\u0000point",
      "component\u0000substance"
    ],
    "leaf_id": "96177e62c46737531a97cad762bbbbce"
  },
  {
    "concept": "molecule",
    "count": 4,
    "degree": 8,
    "links": [
      "atoms\u0000molecule",
      "between\u0000molecule",
      "bond\u0000molecule",
      "chemical\u0000molecule",
      "component\u0000molecule",
      "element\u0000molecule",
      "formed\u0000molecule",
      "molecule\u0000substance"
    ],
    "leaf_id": "85b2603c4711fabd9b658d1dbf2af633"
  },
  {
    "concept": "formed",
    "count": 2,
    "degree": 5,
    "links": [
      "atoms\u0000formed",
      "between\u0000formed",
      "bond\u0000formed",
      "chemical\u0000formed",
      "formed\u0000molecule"
    ],
    "leaf_id": "d91ef08263c4708c18eb55db37b8bf3b"
  },
  {
    "concept": "bond",
    "count": 3,
    "degree": 7,
    "links": [
      "atoms\u0000bond",
      "between\u0000bond",
      "bond\u0000chemical",
      "bond\u0000covalent",
      "bond\u0000formed",
      "bond\u0000ionic",
      "bond\u0000molecule"
    ],
    "leaf_id": "0ca3cc195fb9b048ca1a28adc859e4d5"
  },
  {
    "concept": "electron",
    "count": 4,
    "degree": 6,
    "links": [
      "atom\u0000electron",
      "charge\u0000electron",
      "component\u0000electron",
      "electron\u0000opposite",
      "electron\u0000part",
      "electron\u0000proton"
    ],
    "leaf_id": "ab461f3edfcdcdc528e2a1d987fbaa51"
  },
  {
    "concept": "part",
    "count": 6,
    "degree": 7,
    "links": [
      "atom\u0000part",
      "electron\u0000part",
      "member\u0000part",
      "nucleus\u0000part",
      "part\u0000proton",
      "part\u0000set",
      "part\u0000subset"
    ],
    "leaf_id": "5465d1d15d00bffc1083d87a994d2780"
  },
  {
    "concept": "proton",
    "count": 3,
    "degree": 5,
    "links": [
      "charge\u0000proton",
      "electron\u0000proton",
      "nucleus\u0000proton",
      "opposite\u0000proton",
      "part\u0000proton"
    ],
    "leaf_id": "bcae61689a2a40fc896db9a6a3d416ee"
  },
  {
    "concept": "nucleus",
    "count": 5,
    "degree": 12,
    "links": [
      "atom\u0000nucleus",
      "attribute\u0000nucleus",
      "chemical\u0000nucleus",
      "contains\u0000nucleus",
      "defined\u0000nucleus",
      "element\u0000nucleus",
      "mass\u0000nucleus",
      "neutrons\u0000nucleus",
      "nucleus\u0000number",
      "nucleus\u0000part",
      "nucleus\u0000proton",
      "nucleus\u0000protons"
    ],
    "leaf_id": "1e7c4554ad7a9b9d93bd9ef3fb0744c5"
  }
]
```

## Top pairs

```json
[
  {
    "pair": [
      "element",
      "geometric"
    ],
    "weight": 4.0,
    "roles": {}
  },
  {
    "pair": [
      "causes",
      "change"
    ],
    "weight": 2.33,
    "roles": {}
  },
  {
    "pair": [
      "chemical",
      "element"
    ],
    "weight": 2.0,
    "roles": {}
  },
  {
    "pair": [
      "dimensional",
      "volume"
    ],
    "weight": 1.75,
    "roles": {}
  },
  {
    "pair": [
      "bond",
      "chemical"
    ],
    "weight": 1.5,
    "roles": {}
  },
  {
    "pair": [
      "attribute",
      "mass"
    ],
    "weight": 1.5,
    "roles": {}
  },
  {
    "pair": [
      "part",
      "set"
    ],
    "weight": 1.5,
    "roles": {}
  },
  {
    "pair": [
      "element",
      "member"
    ],
    "weight": 1.33,
    "roles": {}
  },
  {
    "pair": [
      "dimension",
      "space"
    ],
    "weight": 1.33,
    "roles": {}
  },
  {
    "pair": [
      "change",
      "velocity"
    ],
    "weight": 1.33,
    "roles": {}
  },
  {
    "pair": [
      "chemical",
      "molecule"
    ],
    "weight": 1.25,
    "roles": {}
  },
  {
    "pair": [
      "component",
      "molecule"
    ],
    "weight": 1.0,
    "roles": {}
  }
]
```

## Organization by chatting

After saturation, statements combining concepts let the heads
reorganize and map relations into leaves and scratchpads.

| statement | NaNLM | roles | head leaves |
|---|---|---|---|
| Copper is a chemical element. | Chemical element consists of one or more isotopes. | member_of:chemical | 83 |
| A triangle is a geometric element. | Geometric element such as a point has no dimension. | member_of:geometric | 84 |
| The sky is a component of the earth. | Component of a chemical bond can be ionic or covalent. | component_of | 85 |
| Hot is the opposite of cold. | Opposite charge of an atom is a geometric element with six square faces. | opposition | 86 |
| Gold is a chemical element. | Chemical bond can be ionic or more isotopes. | member_of:chemical | 87 |

| organization probe | NaNLM |
|---|---|
| Is copper a chemical element? | Chemical bond can be ionic or covalent. |
| Is a triangle a geometric element? | Triangle is a geometric element with six square faces. |
| Is the sky a component of the earth? | Sky is a geometric element with six square faces. |
| Is hot the opposite of cold? | Hot is the opposite of cold. |

### Matrix after organization

- concepts: 145
- occupied pairs: 560 / 32,640 (density 0.017157)
- average degree: 6.54
