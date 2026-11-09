# Typed WordNet 3.1 graph

Every WordNet pointer is retained as a typed directed edge. The graph is
addressed sparsely at HW4; it is not a dense matrix per word.

- synsets: **117,791**
- lemma occurrences: **207,272**
- typed pointer edges: **378,203**
- hypernym/instance edges: **97,761**
- gloss characters: **8,873,693**
- canonical route: `nan://GM0:SC0:MC0:BS0:0:HW4:0` via `HW4:0`
  (upstream `HW2:0`, NANNET port `BS0:0`, NaN16DE).
- legacy compatibility envelope: `nan://HW4:BS0:MC0:SC0:GM0` (translation
  required; not a second authority). Historical JSON snapshots below may show
  the legacy form.

## Positions

| POS | synsets | lemma occurrences | typed pointers | gloss chars |
|---|---:|---:|---:|---:|
| n | 82,192 | 146,547 | 269,711 | 6,188,706 |
| v | 13,789 | 25,061 | 54,963 | 963,906 |
| a | 7,468 | 9,688 | 30,169 | 595,275 |
| s | 10,717 | 20,384 | 19,277 | 845,930 |
| r | 3,625 | 5,592 | 4,083 | 279,876 |

## Relation inventory

| relation | edges |
|---|---:|
| hypernym | 89,172 |
| hyponym | 89,172 |
| derivationally_related_form | 74,680 |
| similar_to | 21,434 |
| member_holonym | 12,288 |
| member_meronym | 12,288 |
| part_holonym | 9,111 |
| part_meronym | 9,111 |
| instance_hypernym | 8,589 |
| instance_hyponym | 8,589 |
| pertainym | 8,074 |
| antonym | 7,983 |
| domain_topic | 6,690 |
| member_of_domain_topic | 6,690 |
| also_see | 3,276 |
| verb_group | 1,744 |
| domain_region | 1,499 |
| member_of_domain_region | 1,499 |
| domain_usage | 1,370 |
| member_of_domain_usage | 1,370 |
| attribute | 1,278 |
| substance_holonym | 797 |
| substance_meronym | 797 |
| entailment | 408 |
| cause | 221 |
| participle_of | 73 |

## Sample concept records

### entity

- synset: `n:00001740`
- lexname: `noun.Tops`
- definition: that which is perceived or known or inferred to have its own distinct existence (living or nonliving)
- examples: []
- hypernym chain:
  - `n:00001740` entity — that which is perceived or known or inferred to have its own distinct existence (living or nonliving)
- typed relations: `[{"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1930}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2137}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 4431553}]`

### animal

- synset: `n:00015568`
- lexname: `noun.Tops`
- definition: a living organism characterized by voluntary movement
- examples: []
- hypernym chain:
  - `n:00015568` animal, animate being, beast, brute — a living organism characterized by voluntary movement
  - `n:00004475` organism, being — a living thing that has (or can develop) the ability to act or function independently
  - `n:00004258` living thing, animate thing — a living (or once living) entity
  - `n:00003553` whole, unit — an assemblage of parts that is regarded as a single entity
  - `n:00002684` object, physical object — a tangible and visible entity; an entity that can cast a shadow
  - `n:00001930` physical entity — an entity that has physical existence
  - `n:00001740` entity — that which is perceived or known or inferred to have its own distinct existence (living or nonliving)
- typed relations: `[{"relation": "hypernym", "symbol": "@", "target_pos": "n", "target_offset": 4475}, {"relation": "member_meronym", "symbol": "#m", "target_pos": "n", "target_offset": 1315584}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "v", "target_offset": 1620211}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "a", "target_offset": 1266555}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "a", "target_offset": 1266555}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "v", "target_offset": 1684651}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "v", "target_offset": 1684651}, {"relation": "member_of_domain_topic", "symbol": "-c", "target_pos": "a", "target_offset": 58232}, {"relation": "member_of_domain_topic", "symbol": "-c", "target_pos": "a", "target_offset": 148320}, {"relation": "member_of_domain_topic", "symbol": "-c", "target_pos": "a", "target_offset": 161553}, {"relation": "member_of_domain_topic", "symbol": "-c", "target_pos": "a", "target_offset": 216053}, {"relation": "member_of_domain_topic", "symbol": "-c", "target_pos": "a", "target_offset": 314598}]`

### bird

- synset: `n:01505702`
- lexname: `noun.animal`
- definition: warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modified as wings
- examples: []
- hypernym chain:
  - `n:01505702` bird — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modified as wings
  - `n:01474323` vertebrate, craniate — animals having a bony or cartilaginous skeleton with a segmented spinal column and a large brain enclosed in a skull or cranium
  - `n:01468898` chordate — any animal of the phylum Chordata having a notochord or spinal column
  - `n:00015568` animal, animate being, beast, brute — a living organism characterized by voluntary movement
  - `n:00004475` organism, being — a living thing that has (or can develop) the ability to act or function independently
  - `n:00004258` living thing, animate thing — a living (or once living) entity
  - `n:00003553` whole, unit — an assemblage of parts that is regarded as a single entity
  - `n:00002684` object, physical object — a tangible and visible entity; an entity that can cast a shadow
  - `n:00001930` physical entity — an entity that has physical existence
- typed relations: `[{"relation": "hypernym", "symbol": "@", "target_pos": "n", "target_offset": 1474323}, {"relation": "member_meronym", "symbol": "#m", "target_pos": "n", "target_offset": 1504903}, {"relation": "member_meronym", "symbol": "#m", "target_pos": "n", "target_offset": 8007398}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "v", "target_offset": 1142291}, {"relation": "member_of_domain_topic", "symbol": "-c", "target_pos": "a", "target_offset": 2705521}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1506617}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1517309}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1517500}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1517567}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1517719}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1517944}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 1518224}]`

### penguin

- synset: `n:02058457`
- lexname: `noun.animal`
- definition: short-legged flightless birds of cold southern especially Antarctic regions having webbed feet and wings modified as flippers
- examples: []
- hypernym chain:
  - `n:02058457` penguin — short-legged flightless birds of cold southern especially Antarctic regions having webbed feet and wings modified as flippers
  - `n:02058312` sphenisciform seabird — flightless cold-water seabirds: penguins
  - `n:02024449` seabird, sea bird, seafowl — a bird that frequents coastal waters and the open ocean: gulls; pelicans; gannets; cormorants; albatrosses; petrels; etc.
  - `n:01847558` aquatic bird — wading and swimming and diving birds of either fresh or salt water
  - `n:01505702` bird — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modified as wings
  - `n:01474323` vertebrate, craniate — animals having a bony or cartilaginous skeleton with a segmented spinal column and a large brain enclosed in a skull or cranium
  - `n:01468898` chordate — any animal of the phylum Chordata having a notochord or spinal column
  - `n:00015568` animal, animate being, beast, brute — a living organism characterized by voluntary movement
  - `n:00004475` organism, being — a living thing that has (or can develop) the ability to act or function independently
- typed relations: `[{"relation": "hypernym", "symbol": "@", "target_pos": "n", "target_offset": 2058312}, {"relation": "member_meronym", "symbol": "#m", "target_pos": "n", "target_offset": 2058085}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2058882}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2059224}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2059382}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2059689}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2059984}]`

### vehicle

- synset: `n:04531608`
- lexname: `noun.artifact`
- definition: a conveyance that transports people or objects
- examples: []
- hypernym chain:
  - `n:04531608` vehicle — a conveyance that transports people or objects
  - `n:03105141` conveyance, transport — something that serves as a means of transportation
  - `n:03580409` instrumentality, instrumentation — an artifact (or system of artifacts) that is instrumental in accomplishing some end
  - `n:00022119` artifact, artefact — a man-made object taken as a whole
  - `n:00003553` whole, unit — an assemblage of parts that is regarded as a single entity
  - `n:00002684` object, physical object — a tangible and visible entity; an entity that can cast a shadow
  - `n:00001930` physical entity — an entity that has physical existence
  - `n:00001740` entity — that which is perceived or known or inferred to have its own distinct existence (living or nonliving)
- typed relations: `[{"relation": "hypernym", "symbol": "@", "target_pos": "n", "target_offset": 3105141}, {"relation": "derivationally_related_form", "symbol": "+", "target_pos": "a", "target_offset": 3139995}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 2922348}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 3130521}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 3769811}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 4106523}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 4235484}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 4242397}, {"relation": "part_holonym", "symbol": "%p", "target_pos": "n", "target_offset": 4288513}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 4317295}, {"relation": "hyponym", "symbol": "~", "target_pos": "n", "target_offset": 4583497}]`

### car

- synset: `n:02937835`
- lexname: `noun.artifact`
- definition: a conveyance for passengers or freight on a cable railway
- examples: ["they took a cable car to the top of the mountain"]
- hypernym chain:
  - `n:02937835` cable car, car — a conveyance for passengers or freight on a cable railway
  - `n:03083745` compartment — a partitioned section, chamber, or separate room within a larger enclosed area
  - `n:04112987` room — an area within a building enclosed by walls and floor and ceiling
  - `n:02738693` area — a part of a structure having some specific characteristic or function
  - `n:04348764` structure, construction — a thing constructed; a complex entity constructed of many parts
  - `n:00022119` artifact, artefact — a man-made object taken as a whole
  - `n:00003553` whole, unit — an assemblage of parts that is regarded as a single entity
  - `n:00002684` object, physical object — a tangible and visible entity; an entity that can cast a shadow
  - `n:00001930` physical entity — an entity that has physical existence
- typed relations: `[{"relation": "hypernym", "symbol": "@", "target_pos": "n", "target_offset": 3083745}, {"relation": "part_meronym", "symbol": "#p", "target_pos": "n", "target_offset": 2938025}]`

## Boundaries

- This is a typed lexical graph, not a learned language model.
- Directed graph edges are not a quantum resource; gauge loops are classical
  matrix algebra.
- The next experiment trains operators on these typed edges and enforces
  composition only where the relation semantics justify it.
