# New word from a definition link

This is the controlled graph step, not a learned prose parser: the new
word is attached to an existing synset, then queried through typed
hypernym pointers.

- new word: **sparklebeak**
- word address: `nan://HW4:BS0:MC0:SC0:GM0::word/sparklebeak/02058457`
- linked synset: `nan://HW4:BS0:MC0:SC0:GM0::synset/02058457`
- source lemmas: penguin
- source definition: short-legged flightless birds of cold southern especially Antarctic regions having webbed feet and wings modified as flippers
- inherited properties: **living, winged**

## Hypernym chain

- `nan://HW4:BS0:MC0:SC0:GM0::synset/02058312` sphenisciform seabird — flightless cold-water seabirds: penguins
- `nan://HW4:BS0:MC0:SC0:GM0::synset/02024449` seabird, sea bird, seafowl — a bird that frequents coastal waters and the open ocean: gulls; pelicans; gannets; cormorants; albatrosses; petrels; etc.
- `nan://HW4:BS0:MC0:SC0:GM0::synset/01847558` aquatic bird — wading and swimming and diving birds of either fresh or salt water
- `nan://HW4:BS0:MC0:SC0:GM0::synset/01505702` bird — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modified as wings
- `nan://HW4:BS0:MC0:SC0:GM0::synset/01474323` vertebrate, craniate — animals having a bony or cartilaginous skeleton with a segmented spinal column and a large brain enclosed in a skull or cranium

## Boundaries

- The new word does not get a dense per-word matrix; it is a sparse leaf.
- Linking a word to a synset is not parsing arbitrary natural-language
  definitions. A prose parser and a learned definition decoder remain open.
- Inheritance is graph traversal; it does not establish language understanding.
