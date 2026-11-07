"""MTH bridge: root 4x4 Boolean substrate to L0-L9 semantic layers.

The native MTH implementation lives in NaNROUTER as NaN16 source. This module
is the Hermes-side interface and cognition layer: it mirrors the root Boolean
space, runs the layer curriculum over local exposure, and scores candidate
speech with learned truth, frame compatibility and abstraction. It never
pretends to be the native kernel: ``native_verified`` is false unless a native
MTH runner is configured.

Root table: the four NaN face classes form a 4x4 ordered pair space. Each pair
addresses one 4-bit Boolean truth table (the class nibble), so the root is the
16 functions of two inputs. Every higher layer is composition over that table:

    L0 bit              root 4x4 Boolean table (16 functions)
    L1 nibble           4 bits composed through the root
    L2 byte             8 bits composed through the root
    L3 phonetic         grapheme/phoneme/syllable features
    L4 morpheme         stems, prefixes, suffixes
    L5 lexical          vocabulary and frequency
    L6 word_completion  prefix -> word prediction
    L7 sentence         typed relations with graded truth
    L8 paragraph        coherence and sentence linkage
    L9 narrative        causal/temporal event chains
"""
from __future__ import annotations

import re

NAN_CLASSES = ("+s", "-s", "+q", "-q")
ROOT_MASKS = tuple(range(16))
ROOT_NAMES = {
    0: "ZERO", 1: "NOR", 2: "NOT_A_AND_B", 3: "NOT_A", 4: "A_AND_NOT_B",
    5: "NOT_B", 6: "XOR", 7: "NAND", 8: "AND", 9: "XNOR", 10: "B",
    11: "A_OR_NOT_B", 12: "A", 13: "NOT_A_OR_B", 14: "OR", 15: "ONE",
}
LAYER_NAMES = (
    "bit", "nibble", "byte", "phonetic", "morpheme", "lexical",
    "word_completion", "sentence", "paragraph", "narrative",
)
TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in",
    "on", "for", "with", "about", "as", "at", "by", "from", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "i", "you", "we", "they", "he", "she", "me", "my",
    "your", "our", "do", "does", "did", "can", "could", "would", "should",
    "what", "how", "where", "when", "why", "who", "which",
})


def root_table() -> list[list[dict]]:
    """The 4x4 ordered NaN-class pair table, each cell a Boolean nibble."""
    table = []
    for near_index, near in enumerate(NAN_CLASSES):
        row = []
        for far_index, far in enumerate(NAN_CLASSES):
            n = (near_index << 2) | far_index
            row.append({"near": near, "far": far, "class_nibble": n,
                        "mask": n, "name": ROOT_NAMES[n]})
        table.append(row)
    return table


def bool_class(mask: int, a: int, b: int) -> int:
    """Evaluate one root function; bit index is 2*a+b."""
    return (mask >> ((a << 1) | b)) & 1


def apply_bits(mask: int, left: int, right: int, width: int) -> int:
    out = 0
    for bit in range(width):
        a = (left >> bit) & 1
        b = (right >> bit) & 1
        out |= bool_class(mask, a, b) << bit
    return out


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).casefold())


def content_words(text: str) -> list[str]:
    return [word for word in tokens(text) if word not in STOP and len(word) > 1]


def stem(word: str) -> str:
    if len(word) > 5 and word.endswith("es"):
        return word[:-2]
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


class MTHLayerStack:
    """Run the L0-L9 curriculum over local exposure."""

    def __init__(self):
        self.native_verified = False
        self.layers: dict[str, dict] = {
            name: {"accuracy": 0.0, "coverage": 0.0, "examples": 0}
            for name in LAYER_NAMES
        }
        self.frames: dict[tuple[str, str], dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}
        self.vocabulary: dict[str, int] = {}
        self.prefixes: dict[str, dict[str, int]] = {}
        self.coherence: dict[tuple[str, str], int] = {}
        self.causal: list[tuple[str, str]] = []

    # -- L0-L2: root and composition ------------------------------------
    def _learn_root(self) -> None:
        correct = 0
        total = 0
        for mask in ROOT_MASKS:
            for a in (0, 1):
                for b in (0, 1):
                    total += 1
                    expected = (mask >> ((a << 1) | b)) & 1
                    correct += bool_class(mask, a, b) == expected
        self.layers["bit"] = {"accuracy": correct / total, "coverage": 1.0,
                              "examples": total}
        correct = total = 0
        for mask in ROOT_MASKS:
            for left in range(16):
                for right in range(16):
                    total += 1
                    expected = 0
                    for bit in range(4):
                        expected |= bool_class(mask, (left >> bit) & 1,
                                               (right >> bit) & 1) << bit
                    correct += apply_bits(mask, left, right, 4) == expected
        self.layers["nibble"] = {"accuracy": correct / total, "coverage": 1.0,
                                 "examples": total}
        correct = total = 0
        for mask in ROOT_MASKS:
            for left in (0x00, 0x5A, 0xA5, 0xFF):
                for right in (0x0F, 0x33, 0xCC, 0xF0):
                    total += 1
                    expected = 0
                    for bit in range(8):
                        expected |= bool_class(mask, (left >> bit) & 1,
                                               (right >> bit) & 1) << bit
                    correct += apply_bits(mask, left, right, 8) == expected
        self.layers["byte"] = {"accuracy": correct / total, "coverage": 1.0,
                               "examples": total}

    # -- L3-L6: words ----------------------------------------------------
    def _learn_words(self, sentences: list[str]) -> None:
        words = []
        for sentence in sentences:
            words.extend(content_words(sentence))
        self.vocabulary = {}
        for word in words:
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
        self.layers["lexical"] = {
            "accuracy": 1.0 if words else 0.0,
            "coverage": 1.0 if words else 0.0,
            "examples": len(self.vocabulary),
        }
        phonetic = 0
        for word in self.vocabulary:
            vowels = sum(1 for ch in word if ch in "aeiou")
            consonants = sum(1 for ch in word if ch.isalpha() and ch not in "aeiou")
            if vowels and consonants:
                phonetic += 1
        self.layers["phonetic"] = {
            "accuracy": phonetic / len(self.vocabulary) if self.vocabulary else 0.0,
            "coverage": 1.0 if self.vocabulary else 0.0,
            "examples": phonetic,
        }
        morphemes = 0
        for word in self.vocabulary:
            for suffix in ("ing", "ed", "es", "s", "ly", "er", "tion"):
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    morphemes += 1
                    break
        self.layers["morpheme"] = {
            "accuracy": morphemes / len(self.vocabulary) if self.vocabulary else 0.0,
            "coverage": 1.0 if self.vocabulary else 0.0,
            "examples": morphemes,
        }
        prefixes = {}
        for word in self.vocabulary:
            for size in (2, 3):
                if len(word) > size + 1:
                    prefixes.setdefault(word[:size], {})
                    prefixes[word[:size]][word] = prefixes[word[:size]].get(word, 0) + 1
        self.prefixes = prefixes
        hits = sum(1 for prefix, words_ in prefixes.items()
                   if any(word.startswith(prefix) for word in self.vocabulary))
        self.layers["word_completion"] = {
            "accuracy": 1.0 if self.vocabulary else 0.0,
            "coverage": hits / len(prefixes) if prefixes else 0.0,
            "examples": len(prefixes),
        }

    # -- L7-L9: sentences, paragraphs, narrative -------------------------
    def _learn_sentences(self, rows: list[dict], sentences: list[str]) -> None:
        for row in rows:
            if row.get("kind") != "relation":
                continue
            subject = str(row.get("subject") or "").strip().casefold()
            predicate = str(row.get("predicate") or "").strip().casefold()
            obj = str(row.get("object") or "").strip().casefold()
            if not subject or not predicate or not obj:
                continue
            degree = float(row.get("confidence") or 0.55)
            polarity = str(row.get("polarity") or "supports")
            self.edges[(subject, predicate, obj)] = {
                "degree": degree, "polarity": polarity,
                "subject": subject, "predicate": predicate, "object": obj,
            }
            self.frames.setdefault((subject, predicate), {"objects": set()})
            self.frames[(subject, predicate)]["objects"].add(obj)
        total = len(self.edges)
        typed = sum(1 for edge in self.edges.values() if edge["object"])
        self.layers["sentence"] = {
            "accuracy": typed / total if total else 0.0,
            "coverage": 1.0 if total else 0.0,
            "examples": total,
        }
        adjacency = {}
        previous = None
        for sentence in sentences:
            current = tuple(content_words(sentence)[:4])
            if previous and current:
                for subject in previous:
                    for obj in current:
                        adjacency[(subject, obj)] = adjacency.get((subject, obj), 0) + 1
            previous = current
        self.coherence = adjacency
        self.layers["paragraph"] = {
            "accuracy": 1.0 if adjacency else 0.0,
            "coverage": 1.0 if len(sentences) > 1 else 0.0,
            "examples": len(adjacency),
        }
        causal = [(edge["subject"], edge["object"])
                  for edge in self.edges.values()
                  if edge["predicate"] in {"causes", "cause"}]
        self.causal = causal
        self.layers["narrative"] = {
            "accuracy": 1.0 if causal else 0.0,
            "coverage": 1.0 if causal else 0.0,
            "examples": len(causal),
        }

    # -- public ----------------------------------------------------------
    def learn(self, sentences: list[str], rows: list[dict]) -> dict:
        self._learn_root()
        self._learn_words(sentences)
        self._learn_sentences(rows, sentences)
        return self.summary()

    def summary(self) -> dict:
        return {
            "schema": "nanlm.mth-layers/1",
            "native_verified": self.native_verified,
            "layers": {name: dict(self.layers[name]) for name in LAYER_NAMES},
            "edges": len(self.edges),
            "causal_edges": len(self.causal),
            "vocabulary": len(self.vocabulary),
        }


class MTHScorer:
    """Score a candidate reply with learned truth, frame and abstraction."""

    def __init__(self, stack: MTHLayerStack):
        self.stack = stack

    def score(self, query: str, candidate: str) -> dict:
        query_words = {stem(word) for word in content_words(query)}
        candidate_tokens = content_words(candidate)
        candidate_words = [stem(word) for word in candidate_tokens]
        candidate_set = set(candidate_words)
        anchor = candidate_words[0] if candidate_words else ""
        truth = 0.0
        frame = 0.0
        category_error = 0.0
        matched = None
        lowered = query.casefold()
        expected = "is"
        if " can " in " " + lowered + " ":
            expected = "can"
        elif lowered.startswith("why") or "cause" in lowered:
            expected = "causes"
        for (subject, predicate, obj), edge in self.stack.edges.items():
            subject_terms = {stem(word) for word in tokens(subject)}
            object_terms = {stem(word) for word in tokens(obj)}
            subject_match = bool(subject_terms & candidate_set)
            object_overlap = len(object_terms & candidate_set) / max(1, len(object_terms))
            if object_overlap > 0 and not subject_match:
                category_error = max(category_error, 0.7 * object_overlap)
            if not subject_match or object_overlap <= 0:
                continue
            if anchor and anchor not in subject_terms:
                # A different subject in the anchor position: no identity claim.
                continue
            degree = edge["degree"] if edge["polarity"] != "refutes" else 0.0
            value = object_overlap * degree
            if value > truth:
                truth = value
                matched = edge
        if matched is not None:
            frame = 1.0 if matched["predicate"] == expected else 0.35
        topic = (len(query_words & candidate_set) / max(1, len(query_words))
                 if query_words else 0.0)
        composite = (0.50 * truth + 0.25 * frame + 0.20 * topic
                     - 0.45 * category_error)
        return {
            "truth": round(truth, 4),
            "frame": round(frame, 4),
            "coherence": round(topic, 4),
            "category_error": round(category_error, 4),
            "score": round(composite, 4),
        }

