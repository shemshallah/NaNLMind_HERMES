# WordNet 3.1 concept/synset graph

WordNet is extracted as synset nodes, lemma edges and hypernym edges.
A word is not assigned to a head: a word leaf points to its synset;
the synset participates in the concept graph.

- synsets: **82,192**
- lemmas: **117,953**
- hypernym edges: **84,505**
- definition-inheritance probes: **32**
- Canonical HW4 route: `nan://GM0:SC0:MC0:BS0:0:HW4:0` via `HW4:0`
  (upstream `HW2:0`, NANNET port `BS0:0`, NaN16DE).
- Legacy compatibility envelope: `nan://HW4:BS0:MC0:SC0:GM0`. It must be
  explicitly translated and is not a second authority. Historical snapshots in
  this document may still show the legacy form.

## Hypernym paths (sample)

### animal
- 15568: animal, animate being, beast, brute — a living organism characterized by voluntary movement
- 4475: organism, being — a living thing that has (or can develop) the ability to act or function independently
- 4258: living thing, animate thing — a living (or once living) entity
- 3553: whole, unit — an assemblage of parts that is regarded as a single entity; "how big is that part compared to the whole?"; "the team is 
- 2684: object, physical object — a tangible and visible entity; an entity that can cast a shadow; "it was full of rackets, balls and other objects"
- 1930: physical entity — an entity that has physical existence
- 1740: entity — that which is perceived or known or inferred to have its own distinct existence (living or nonliving)
### bird
- 1505702: bird — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modified as wings
- 1474323: vertebrate, craniate — animals having a bony or cartilaginous skeleton with a segmented spinal column and a large brain enclosed in a skull or 
- 1468898: chordate — any animal of the phylum Chordata having a notochord or spinal column
- 15568: animal, animate being, beast, brute — a living organism characterized by voluntary movement
- 4475: organism, being — a living thing that has (or can develop) the ability to act or function independently
- 4258: living thing, animate thing — a living (or once living) entity
- 3553: whole, unit — an assemblage of parts that is regarded as a single entity; "how big is that part compared to the whole?"; "the team is 
- 2684: object, physical object — a tangible and visible entity; an entity that can cast a shadow; "it was full of rackets, balls and other objects"
- 1930: physical entity — an entity that has physical existence
### penguin
- 2058457: penguin — short-legged flightless birds of cold southern especially Antarctic regions having webbed feet and wings modified as fli
- 2058312: sphenisciform seabird — flightless cold-water seabirds: penguins
- 2024449: seabird, sea bird, seafowl — a bird that frequents coastal waters and the open ocean: gulls; pelicans; gannets; cormorants; albatrosses; petrels; etc
- 1847558: aquatic bird — wading and swimming and diving birds of either fresh or salt water
- 1505702: bird — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modified as wings
- 1474323: vertebrate, craniate — animals having a bony or cartilaginous skeleton with a segmented spinal column and a large brain enclosed in a skull or 
- 1468898: chordate — any animal of the phylum Chordata having a notochord or spinal column
- 15568: animal, animate being, beast, brute — a living organism characterized by voluntary movement
- 4475: organism, being — a living thing that has (or can develop) the ability to act or function independently
### vehicle
- 4531608: vehicle — a conveyance that transports people or objects
- 3105141: conveyance, transport — something that serves as a means of transportation
- 3580409: instrumentality, instrumentation — an artifact (or system of artifacts) that is instrumental in accomplishing some end
- 22119: artifact, artefact — a man-made object taken as a whole
- 3553: whole, unit — an assemblage of parts that is regarded as a single entity; "how big is that part compared to the whole?"; "the team is 
- 2684: object, physical object — a tangible and visible entity; an entity that can cast a shadow; "it was full of rackets, balls and other objects"
- 1930: physical entity — an entity that has physical existence
- 1740: entity — that which is perceived or known or inferred to have its own distinct existence (living or nonliving)
### car
- 2961779: car, auto, automobile, machine — a motor vehicle with four wheels; usually propelled by an internal combustion engine; "he needs a car to get to work"
- 3796768: motor vehicle, automotive vehicle — a self-propelled wheeled vehicle that does not run on rails
- 4177098: self-propelled vehicle — a wheeled vehicle that carries in itself a means of propulsion
- 4583497: wheeled vehicle — a vehicle that moves on wheels and usually has a container for transporting things or people; "the oldest known wheeled 
- 4531608: vehicle — a conveyance that transports people or objects
- 3105141: conveyance, transport — something that serves as a means of transportation
- 3580409: instrumentality, instrumentation — an artifact (or system of artifacts) that is instrumental in accomplishing some end
- 22119: artifact, artefact — a man-made object taken as a whole
- 3553: whole, unit — an assemblage of parts that is regarded as a single entity; "how big is that part compared to the whole?"; "the team is 
### entity
- 1740: entity — that which is perceived or known or inferred to have its own distinct existence (living or nonliving)

## Definition inheritance

A synthetic property is attached to a high-level synset; a word defined
only by a synset link inherits it through the hypernym closure.

```json
[
  {
    "word": "4wd",
    "definition_synset": 3394607,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/03394607",
    "inherited": [
      "artifact"
    ]
  },
  {
    "word": "a. testudineus",
    "definition_synset": 2559004,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02559004",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "aardvark",
    "definition_synset": 2085443,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02085443",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "aardwolf",
    "definition_synset": 2120828,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02120828",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "abalone",
    "definition_synset": 1945510,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/01945510",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "abandoned ship",
    "definition_synset": 2669131,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02669131",
    "inherited": [
      "artifact"
    ]
  },
  {
    "word": "abm",
    "definition_synset": 2719537,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02719537",
    "inherited": [
      "artifact"
    ]
  },
  {
    "word": "abortus",
    "definition_synset": 1462944,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/01462944",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "abrocome",
    "definition_synset": 2371051,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02371051",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "abyssinian",
    "definition_synset": 2126965,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02126965",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "abyssinian cat",
    "definition_synset": 2126965,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/02126965",
    "inherited": [
      "living"
    ]
  },
  {
    "word": "acanthisitta chloris",
    "definition_synset": 1591637,
    "address": "nan://HW4:BS0:MC0:SC0:GM0::synset/01591637",
    "inherited": [
      "living",
      "winged"
    ]
  }
]
```

## Hosting and encoding

- Do not allocate one 256×256 matrix per word. Use sparse synset/lemma
  addresses and a dedicated HW4 universe/chart hierarchy behind NaNROUTER.
- A word leaf is a hard-mapped letter hyperedge plus its synset membership;
  polysemy is a superposition of synset leaves.
- Hypernym/meronym/similar edges are typed; order-sensitive relations use
  non-Abelian operators, not scalar phases.
- Cycle fields and gauge-aware retrieval enforce consistency; definitions
  are graph edges, not yet parsed prose.

## Honest boundary

This is a real dictionary graph and compositional inheritance test, not a
trained language model. It supplies the lexical/concept substrate the
cycle memory and head population need.
