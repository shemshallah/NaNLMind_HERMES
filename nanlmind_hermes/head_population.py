"""Head-population next-letter completion.

Heads do not own letters. Each head holds a distribution over the whole
alphabet. A context activates a population of heads; their distributions are
collapsed into the next-letter distribution, with per-head votes retained as
diagnostics. Head 0 is the NaNNet/polarity prior and always participates.
"""
from __future__ import annotations

from collections import Counter

ALPHABET = "abcdefghijklmnopqrstuvwxyz$"  # $ = end of word


def fnv(text: str, salt: int = 0) -> int:
    value = 2166136261 ^ (salt * 0x9E3779B1 & 0xFFFFFFFF)
    for char in text.encode("utf-8"):
        value ^= char
        value = (value * 16777619) & 0xFFFFFFFF
    return value


class HeadPopulation:
    def __init__(self, heads: int = 256, active: int = 16,
                 prior_weight: float = 1.0):
        self.head_count = heads
        self.active_count = active
        self.prior_weight = prior_weight
        self.heads: list[Counter] = [Counter() for _ in range(heads)]
        self.prior = Counter()
        self.observations = 0

    def _active(self, context: str) -> list[int]:
        # Head 0 is the shared NaNNet/polarity prior; 1..255 are population heads.
        if self.active_count >= self.head_count - 1:
            return list(range(self.head_count))
        result = {0}
        for salt in range(self.active_count):
            result.add(fnv(context, salt) % (self.head_count - 1) + 1)
        return sorted(result)

    def observe(self, word: str, weight: float = 1.0) -> None:
        word = str(word).strip().lower()
        if not word or not word.isalpha():
            return
        padded = "^" + word + "$"
        for index in range(2, len(padded)):
            context = padded[index - 2:index]
            target = padded[index]
            if target not in ALPHABET:
                continue
            self.observations += 1
            self.prior[target] += weight
            for head in self._active(context):
                self.heads[head][target] += weight

    def observe_partial(self, prefix: str, next_letter: str,
                        weight: float = 1.0) -> None:
        """Train explicitly on a partial word: prefix -> next letter."""
        prefix = str(prefix).strip().lower()
        next_letter = str(next_letter).strip().lower()
        if not prefix or len(next_letter) != 1 or next_letter not in ALPHABET:
            return
        padded = "^" + prefix
        context = padded[-2:]
        self.observations += 1
        self.prior[next_letter] += weight
        for head in self._active(context):
            self.heads[head][next_letter] += weight

    def distribution(self, context: str) -> dict[str, float]:
        scores: Counter = Counter()
        for head in self._active(context):
            if head == 0:
                for letter, count in self.heads[head].items():
                    scores[letter] += count * self.prior_weight
            else:
                scores.update(self.heads[head])
        total = sum(scores.values())
        if not total:
            scores.update(self.prior)
            total = sum(scores.values())
        return {letter: round(count / total, 6) if total else 0.0
                for letter, count in scores.most_common()}

    def head_votes(self, context: str) -> dict[str, int]:
        votes: Counter = Counter()
        for head in self._active(context):
            if head == 0 and self.prior_weight <= 0.0:
                continue
            counts = self.heads[head]
            if not counts:
                continue
            votes[max(counts.items(), key=lambda item: (item[1], -ord(item[0])))[0]] += 1
        return dict(votes.most_common())

    def predict(self, context: str) -> str:
        distribution = self.distribution(context)
        if not distribution:
            return "e"
        return next(iter(distribution))

    def evaluate(self, words: list[str]) -> dict:
        correct = total = 0
        top3 = 0
        for word in words:
            padded = "^" + word + "$"
            for index in range(2, len(padded)):
                context = padded[index - 2:index]
                target = padded[index]
                if target not in ALPHABET:
                    continue
                distribution = list(self.distribution(context))
                total += 1
                correct += bool(distribution) and distribution[0] == target
                top3 += target in distribution[:3]
        return {"accuracy": round(correct / total, 4) if total else 0.0,
                "top3": round(top3 / total, 4) if total else 0.0,
                "letters": total}

    def superposition(self) -> list[dict]:
        """The dictionary as it is actually stored: head letter distributions."""
        result = []
        for index, head in enumerate(self.heads):
            total = sum(head.values())
            if not total:
                continue
            result.append({
                "head": index,
                "letters": {letter: round(count / total, 4)
                            for letter, count in head.most_common()[:8]},
                "mass": round(total, 3),
            })
        return result

    def status(self) -> dict:
        nonzero = sum(1 for head in self.heads if head)
        return {
            "heads": self.head_count,
            "active_per_context": self.active_count + 1,
            "heads_with_distribution": nonzero,
            "head_0": "NaNNet + polarity prior",
            "observations": self.observations,
            "letters_per_head": len(ALPHABET),
            "letter_to_head_mapping": False,
            "prior_weight": self.prior_weight,
            "dictionary_in_superposition": True,
            "stored_words": 0,
            "superposition_heads": nonzero,
        }


class BigramBaseline:
    def __init__(self):
        self.counts: dict[str, Counter] = {}

    def train(self, words: list[str]) -> None:
        for word in words:
            padded = "^" + word + "$"
            for index in range(1, len(padded)):
                context = padded[index - 1]
                target = padded[index]
                if target not in ALPHABET:
                    continue
                self.counts.setdefault(context, Counter())[target] += 1

    def predict(self, context: str) -> str:
        counts = self.counts.get(context[-1])
        if not counts:
            return "e"
        return max(counts.items(), key=lambda item: (item[1], -ord(item[0])))[0]

    def evaluate(self, words: list[str]) -> dict:
        correct = total = 0
        for word in words:
            padded = "^" + word + "$"
            for index in range(2, len(padded)):
                context = padded[index - 2:index]
                target = padded[index]
                if target not in ALPHABET:
                    continue
                total += 1
                correct += self.predict(context) == target
        return {"accuracy": round(correct / total, 4) if total else 0.0,
                "letters": total}
