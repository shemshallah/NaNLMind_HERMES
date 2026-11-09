# NaNLM novel generation probe

- state: `/data/data/com.termux/files/usr/tmp/nanlm-train-canonical`
- student: native temporal NaNLM; no LLM wrote the replies
- prompts: 8
- empty replies: 0
- mean tokens: 8.0
- mean novel 3-gram ratio: 0.1825
- copied training sentences: 2

## Replies

### Explain why a river can carry water.

> River consists of water. Carry flowing water has the ability to flow.

- {"characters": 69, "copied_training_sentence": false, "empty": false, "max_token_repetition": 2, "novel_3gram_ratio": 0.3, "novel_3grams": 3, "tokens": 12, "unique_tokens": 11}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 0, "notes": "The reply is grammatically incoherent and fails to explain 'why' a river can carry water; it merely repeats facts from the training set in a fragmented way.", "novel_combination": 0, "unsupported_claim": 0}

### Compare a cat and a bird.

> Cat and a bird is an animal.

- {"characters": 28, "copied_training_sentence": false, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.4, "novel_3grams": 2, "tokens": 7, "unique_tokens": 7}
- judge: {"answers_prompt": 0, "coherent": 1, "copied_training_sentence": 0, "notes": "The reply identifies a commonality but fails to 'compare' the two animals as requested; it provides a simple classification instead of a comparison.", "novel_combination": 0, "unsupported_claim": 0}

### Why does ice melt when it gets hot?

> Ice is water that froze. Gets the sun is a hot star. Melt the sun is a hot star.

- {"characters": 80, "copied_training_sentence": false, "empty": false, "max_token_repetition": 3, "novel_3gram_ratio": 0.3846, "novel_3grams": 5, "tokens": 19, "unique_tokens": 12}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is a collection of fragmented phrases and repeated training sentences. It fails to explain why ice melts.", "novel_combination": 0, "unsupported_claim": 0}

### Describe what happens when snow falls on a warm day.

> Snow is white.

- {"characters": 14, "copied_training_sentence": true, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.0, "novel_3grams": 0, "tokens": 3, "unique_tokens": 3}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is a fact from the training set but does not answer the prompt's question about what happens when snow falls on a warm day.", "novel_combination": 0, "unsupported_claim": 0}

### What would happen if a bird could not fly?

> Bird is an animal.

- {"characters": 18, "copied_training_sentence": false, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.0, "novel_3grams": 0, "tokens": 4, "unique_tokens": 4}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply is a factual statement from the training data but does not answer the hypothetical question asked in the prompt.", "novel_combination": 0, "unsupported_claim": 0}

### Tell a short story about a river, a bird, and the sun.

> Dogs, and birds are animals.

- {"characters": 28, "copied_training_sentence": false, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.0, "novel_3grams": 0, "tokens": 5, "unique_tokens": 5}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 1, "notes": "The reply fails to tell a story and instead provides a factual statement copied from the training data.", "novel_combination": 0, "unsupported_claim": 0}

### Explain how water can become ice and then water again.

> Ice is frozen water.

- {"characters": 20, "copied_training_sentence": true, "empty": false, "max_token_repetition": 1, "novel_3gram_ratio": 0.0, "novel_3grams": 0, "tokens": 4, "unique_tokens": 4}
- judge: {"answers_prompt": 0, "coherent": 1, "copied_training_sentence": 1, "notes": "The reply is a true statement but fails to explain the process of water becoming ice and then water again as requested.", "novel_combination": 0, "unsupported_claim": 0}

### Why is the sky blue but grass green?

> Sky is blue mix to make green. Grass is green.

- {"characters": 46, "copied_training_sentence": false, "empty": false, "max_token_repetition": 2, "novel_3gram_ratio": 0.375, "novel_3grams": 3, "tokens": 10, "unique_tokens": 8}
- judge: {"answers_prompt": 0, "coherent": 0, "copied_training_sentence": 0, "notes": "The reply is incoherent and makes an unsupported claim that the sky is blue mix to make green.", "novel_combination": 0, "unsupported_claim": 1}

## Boundaries

- Native temporal learner, not a general language model.
- Optional Gemma judge is a diagnostic, not ground truth.
- No canonical merge, route or proof claim.
