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
    def __init__(self, heads: int = 256, active: int = 8,
                 prior_weight: float = 0.0, train_head0: bool = False,
                 context_length: int = 2, balance: bool = False):
        self.head_count = heads
        self.active_count = active
        self.prior_weight = prior_weight
        self.train_head0 = train_head0
        self.context_length = max(1, int(context_length))
        self.balance = balance
        self.usage = [0] * heads
        self.heads: list[Counter] = [Counter() for _ in range(heads)]
        self.prior = Counter()
        self.observations = 0

    def _active(self, context: str) -> list[int]:
        # Head 0 is the shared NaNNet/polarity prior; 1..255 are population heads.
        if self.active_count >= self.head_count - 1:
            return list(range(self.head_count))
        candidates = []
        seen = set()
        for salt in range(self.active_count * 5):
            head = fnv(context, salt) % (self.head_count - 1) + 1
            if head not in seen:
                seen.add(head)
                candidates.append(head)
        if self.balance:
            candidates.sort(key=lambda head: (self.usage[head], head))
        selected = candidates[:self.active_count]
        return sorted([0] + selected)

    def observe(self, word: str, weight: float = 1.0) -> None:
        word = str(word).strip().lower()
        if not word or not word.isalpha():
            return
        width = self.context_length
        padded = "^" * (width - 1) + word + "$"
        for index in range(width, len(padded)):
            context = padded[index - width:index]
            target = padded[index]
            if target not in ALPHABET:
                continue
            self.observations += 1
            self.prior[target] += weight
            for head in self._active(context):
                if head == 0 and not self.train_head0:
                    continue
                self.heads[head][target] += weight
                self.usage[head] += 1

    def observe_partial(self, prefix: str, next_letter: str,
                        weight: float = 1.0) -> None:
        """Train explicitly on a partial word: prefix -> next letter."""
        prefix = str(prefix).strip().lower()
        next_letter = str(next_letter).strip().lower()
        if not prefix or len(next_letter) != 1 or next_letter not in ALPHABET:
            return
        width = self.context_length
        padded = "^" * (width - 1) + prefix
        context = padded[-width:]
        self.observations += 1
        self.prior[next_letter] += weight
        for head in self._active(context):
            if head == 0 and not self.train_head0:
                continue
            self.heads[head][next_letter] += weight
            self.usage[head] += 1

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
        width = self.context_length
        for word in words:
            padded = "^" * (width - 1) + word + "$"
            for index in range(width, len(padded)):
                context = padded[index - width:index]
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

    def decorrelate(self, strength: float = 0.1) -> None:
        """Remove the shared global component from every head distribution."""
        letters = "abcdefghijklmnopqrstuvwxyz$"
        totals = {letter: 0.0 for letter in letters}
        for head in self.heads:
            for letter in letters:
                totals[letter] += head.get(letter, 0)
        count = max(1, len(self.heads))
        mean = {letter: totals[letter] / count for letter in letters}
        for head in self.heads:
            for letter in letters:
                head[letter] = max(0.0, head.get(letter, 0) - strength * mean[letter])

    def vectors(self, letters: str = "abcdefghijklmnopqrstuvwxyz$") -> list[list[float]]:
        vectors = []
        for head in self.heads:
            total = sum(head.values())
            if total:
                vectors.append([head.get(letter, 0) / total for letter in letters])
            else:
                vectors.append([0.0] * len(letters))
        return vectors

    def diversity(self) -> dict:
        vectors = [vector for vector in self.vectors() if any(vector)]
        if not vectors:
            return {"heads": 0, "rank": 0, "effective_rank": 0.0,
                    "mean_cosine": 0.0, "mean_entropy": 0.0}
        import math
        gram = [[sum(a * b for a, b in zip(left, right))
                 for right in vectors] for left in vectors]
        n = len(gram)
        # Jacobi eigenvalues of the small Gram matrix (diagnostic only).
        values = [gram[i][i] for i in range(n)]
        # Use power iteration for the top values and trace for effective rank.
        trace = sum(values) or 1.0
        squared = sum(value * value for value in values)
        effective = (trace * trace / squared) if squared else 0.0
        cosines = []
        for i in range(n):
            for j in range(i + 1, n):
                cosines.append(sum(a * b for a, b in zip(vectors[i], vectors[j])))
        entropies = []
        for vector in vectors:
            entropy = -sum(value * math.log(value) for value in vector if value > 0)
            entropies.append(entropy)
        return {
            "heads": n,
            "trace": round(trace, 4),
            "effective_rank": round(effective, 4),
            "mean_cosine": round(sum(cosines) / len(cosines), 6) if cosines else 0.0,
            "mean_entropy": round(sum(entropies) / len(entropies), 6),
        }

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
    def __init__(self, context_length: int = 2):
        self.counts: dict[str, Counter] = {}
        self.context_length = context_length

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
        width = self.context_length
        for word in words:
            padded = "^" * (width - 1) + word + "$"
            for index in range(width, len(padded)):
                context = padded[index - width:index]
                target = padded[index]
                if target not in ALPHABET:
                    continue
                total += 1
                correct += self.predict(context) == target
        return {"accuracy": round(correct / total, 4) if total else 0.0,
                "letters": total}
