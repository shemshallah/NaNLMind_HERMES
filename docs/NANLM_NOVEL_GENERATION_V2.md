# NaNLM novel generation probe

- state: `/data/data/com.termux/files/usr/tmp/nanlm-train-v2`
- student: native temporal NaNLM; no LLM wrote the replies
- prompts: 8
- empty replies: 0
- mean tokens: 9.75
- mean novel 3-gram ratio: 0.247
- copied training sentences: 1

## Replies

### Explain why a river can carry water.

> River consists of water. Carry water has the ability to flow.

- {"characters": 61, "copied_training_sentence": false, "empty": false, "max_token_repetition": 2, "novel_3gram_ratio": 0.3333, "novel_3grams": 3, "tokens": 11, "unique_tokens": 10}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is incoherent and fails to explain 'why'. It contains a copied sentence and a grammatically broken phrase.", "novel_combination": 0, "unsupported_claim": 0}

### Compare a cat and a bird.

> Cat and a bird is an animal.

- {"characters": 28, "copied_training_sentence": false, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.4, "novel_3grams": 2, "tokens": 7, "unique_tokens": 7}
- judge: {"answers_prompt": 0, "coherent": 1, "copied_training_sentence": 0, "notes": "The reply identifies a commonality but does not 'compare' (analyze similarities and differences) as requested. It is a simplistic statement rather than a comparison.", "novel_combination": 0, "unsupported_claim": 0}

### Why does ice melt when it gets hot?

> Ice is water that is frozen water has the ability to fly. Melt the sun causes water to freeze. Gets the sun causes water to freeze.

- {"characters": 131, "copied_training_sentence": false, "empty": false, "max_token_repetition": 4, "novel_3gram_ratio": 0.4, "novel_3grams": 8, "tokens": 26, "unique_tokens": 15}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is incoherent, claims water can fly, and incorrectly states the sun causes water to freeze.", "novel_combination": 0, "unsupported_claim": 1}

### Describe what happens when snow falls on a warm day.

> Snow is white.

- {"characters": 14, "copied_training_sentence": true, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.0, "novel_3grams": 0, "tokens": 3, "unique_tokens": 3}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is a factually true statement from the training set, but it does not answer the prompt regarding what happens when snow falls on a warm day.", "novel_combination": 0, "unsupported_claim": 0}

### What would happen if a bird could not fly?

> Bird is an animal.

- {"characters": 18, "copied_training_sentence": false, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.0, "novel_3grams": 0, "tokens": 4, "unique_tokens": 4}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is a factual statement from the training data but does not answer the hypothetical question asked in the prompt.", "novel_combination": 0, "unsupported_claim": 0}

### Tell a short story about a river, a bird, and the sun.

> River consists of water has the ability to flow.

- {"characters": 48, "copied_training_sentence": false, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.1429, "novel_3grams": 1, "tokens": 9, "unique_tokens": 9}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is not a story and ignores two of the three required elements (bird, sun). It is a concatenation of training statements.", "novel_combination": 0, "unsupported_claim": 0}

### Explain how water can become ice and then water again.

> Water has the ability to flow. Ice is water that is frozen.

- {"characters": 59, "copied_training_sentence": false, "empty": false, "max_token_repetition": 2, "novel_3gram_ratio": 0.2, "novel_3grams": 2, "tokens": 12, "unique_tokens": 10}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply fails to explain the process of water becoming ice and then water again. It provides two unrelated statements, both of which are copied directly from the training set.", "novel_combination": 0, "unsupported_claim": 0}

### Why is the sky blue but grass green?

> Grass is green. Sky is blue.

- {"characters": 28, "copied_training_sentence": false, "empty": false, "max_token_repetition": 2, "novel_3gram_ratio": 0.5, "novel_3grams": 2, "tokens": 6, "unique_tokens": 5}
- judge: {"answers_prompt": 0, "coherent": 1, "copied_training_sentence": 1, "notes": "The reply states the facts but fails to answer the 'Why' part of the prompt.", "novel_combination": 0, "unsupported_claim": 0}

## Boundaries

- Native temporal learner, not a general language model.
- Optional Gemma judge is a diagnostic, not ground truth.
- No canonical merge, route or proof claim.
