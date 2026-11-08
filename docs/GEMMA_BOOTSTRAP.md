# Gemma-guided bootstrap training conversation

Teacher: `gemma4:31b-cloud`. NaNLM native mode, chat learning on, no templates.
Gemma designed the curriculum from the architecture description.

## Round 1 reasoning

Round 1 establishes basic identity, the fundamental mechanism of temporal prediction, and simple causal associations between colors and objects to test the L6 (lexical) and L7 (word_completion) layers.

### Teaching transcript

| teacher statement | NaNLM reply | learned | heads |
|---|---|---:|---:|
| NaNLM is a word-level temporal predictor. | Nanlm is a word level temporal predictor. | 1 | 1 |
| NaNLM learns from statements. | Nanlm learns from statements. | 0 | 1 |
| A word is a unit of meaning. | Word level temporal predictor. | 1 | 2 |
| A sentence is a sequence of words. | Sentence is a sequence of words. | 1 | 3 |
| Temporal prediction is guessing the next word. | Temporal prediction is guessing the next word. | 1 | 4 |
| The sky is blue. | Sky is blue. | 1 | 5 |
| Grass is green. | Grass is green. | 1 | 6 |
| Fire is red. | Fire is red. | 1 | 7 |
| Snow is white. | Snow is white. | 1 | 8 |
| The sun is yellow. | Sun is yellow. | 1 | 9 |
| Water is clear. | Water is clear. | 1 | 10 |
| A cat is an animal. | Cat is an animal. | 1 | 11 |
| A dog is an animal. | Animal a word level temporal predictor. | 1 | 12 |
| A bird is an animal. | Animal is a word level temporal predictor. | 1 | 13 |
| Cats meow. | Cats meow is an animal. | 0 | 13 |
| Dogs bark. | Dogs bark is an animal. | 0 | 13 |
| Birds fly. | Birds fly is an animal. | 0 | 13 |
| Heat causes melting. | Heat causes melting. | 1 | 14 |
| Cold causes freezing. | Causes freezing is a word level temporal predictor. | 1 | 15 |
| Ice is frozen water. | Water is clear. | 1 | 16 |
| Steam is boiled water. | Water is a sequence of words. | 1 | 17 |
| A circle is round. | Circle is round. | 1 | 18 |
| A square has four sides. | Square has four sides. | 1 | 19 |
| Red and blue make purple. | Red and blue make purple. | 0 | 19 |
| Yellow and blue make green. | Yellow and blue make green. | 0 | 19 |
| Red and yellow make orange. | Red and yellow make orange. | 0 | 19 |
| The earth is a planet. | Earth is a planet. | 1 | 20 |
| The moon orbits the earth. | Earth is a sequence of words. | 0 | 20 |
| Stars are in space. | Stars are in space. | 1 | 21 |
| Light travels fast. | Light travels fast. | 0 | 21 |
| I do not know the capital of France. | Not know the capital of france. | 0 | 21 |
| I do not know the name of the president. | Not know the capital of words. | 0 | 21 |
| I do not know how many stars are in the galaxy. | Not know how many stars are in space. | 1 | 22 |
| I do not know the taste of salt. | Not know the taste of salt. | 0 | 22 |
| I do not know the date of tomorrow. | Not know the date of tomorrow. | 0 | 22 |
| I do not know who built the pyramids. | Not know who built the pyramids. | 0 | 22 |

### Known probes

| question | expected | NaNLM | hit |
|---|---|---|---|
| What is NaNLM? | word-level, temporal, predictor | Nanlm learns from statements. | False |
| What color is the sky? | blue | Sky is blue. | True |
| What color is grass? | green | Grass is green. | True |
| What do dogs do? | bark | Dogs bark not know the date of tomorrow. | True |
| What happens when water freezes? | ice | Water is clear. | False |
| What is a cat? | animal | Cat is an animal. | True |
| What do red and blue make? | purple | Red and blue make green. | False |
| What orbits the earth? | moon | Orbits the earth. | False |

### Unknown probes

| question | NaNLM | abstained |
|---|---|---|
| What is the capital of Japan? | Capital of france. | False |
| Who wrote Hamlet? | I do not know the date of tomorrow. | True |
| What is the speed of sound? | I do not know the capital of france. | True |
| How many legs does a spider have? | Many stars are in space. | False |
| What is the chemical symbol for gold? | I do not know who built the pyramids. | True |
| Where is the Eiffel Tower located? | Red and yellow make orange. | False |
| Who discovered gravity? | I do not know how many stars are in space. | True |
| What is the largest ocean? | I do not know the. | True |

## Round 2 (Gemma saw the misses)

Round 2 focuses on reinforcing failed causal links from Round 1 (freezing, color mixing, orbital mechanics) and refining the 'I do not know' abstention pattern to reduce hallucinations of unrelated facts.

| teacher statement | NaNLM reply | heads |
|---|---|---:|
| Water freezes into ice. | Water freezes into ice. | 22 |
| Freezing water creates ice. | Freezing water creates ice. | 22 |
| Red and blue make purple. | Red and blue make purple. | 22 |
| Purple is made from red and blue. | Red and blue make green. | 23 |
| The moon orbits the earth. | Moon orbits the earth. | 23 |
| The earth is orbited by the moon. | Earth is orbited by the moon. | 24 |
| NaNLM predicts the next word. | Nanlm predicts the next word. | 24 |
| Prediction uses temporal frames. | Prediction uses temporal frames. | 24 |
| Ice is cold. | Ice is frozen water. | 25 |
| Purple is a color. | Purple is made from statements. | 26 |
| I do not know the capital of Japan. | Not know the capital of japan. | 26 |
| I do not know the location of the Eiffel Tower. | Not know who built the pyramids. | 26 |
| I do not know the number of spider legs. | Not know the capital of france. | 26 |
| I do not know the chemical symbol for gold. | Not know the chemical symbol for gold. | 26 |

### Round 2 probes

- known: 1/4
- unknown abstained: 3/4

| question | NaNLM | hit/abstained |
|---|---|---|
| What happens when water freezes? | Water freezes into ice. | hit=True |
| What do red and blue make? | Red and blue make green. | hit=False |
| What orbits the earth? | Orbits the earth. | hit=False |
| What does NaNLM do? | Nanlm learns from statements. | hit=False |
| Who is the king of Spain? | I do not know who built the pyramids. | abstained=True |
| What is the square root of 144? | Square has four sides. | abstained=False |
| Where is the Great Wall located? | I do not know the capital of france. | abstained=True |
| What is the boiling point of silver? | I do not know the chemical symbol for gold. | abstained=True |

## Summary

- Round 1 known: 4/8; unknown abstained: 5/8
- Round 2 known: 1/4; unknown abstained: 3/4
- Abstention was measured only; no rule or template was added.
- Heads are the number of MTH head leaves after the last exposure.
