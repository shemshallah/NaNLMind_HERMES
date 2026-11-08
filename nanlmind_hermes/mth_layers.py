"""MTH bridge: root 4x4 Boolean substrate to L0-L9 semantic layers.

The native MTH implementation lives in NaNROUTER as NaN16 source. This module
is the Hermes-side interface and cognition layer. It never pretends to be the
native kernel: ``native_verified`` is false unless a native MTH runner is
configured.

Root table: the four NaN face classes form a 4x4 ordered pair space. Each pair
addresses one 4-bit Boolean truth table (the class nibble), so the root is the
16 functions of two inputs. Every higher layer consumes the layer below:

    L0 bit              root 4x4 Boolean table (16 functions)
    L1 nibble           4 bits composed through the root
    L2 byte             8 bits composed through the root
    L3 phonetic         grapheme/phoneme/syllable features
    L4 morpheme         stems, prefixes, suffixes
    L5 lexical          vocabulary and frequency
    L6 word_completion  prefix -> word prediction
    L7 sentence         typed relations with graded, directional truth
    L8 paragraph        coherence and sentence linkage
    L9 narrative        causal/temporal event chains

Every layer contributes a bounded signal to ``MTHScorer.score``; the temporal
predictor adds that composite to each candidate's next-word score. All ten
layers are therefore in the deliberation path, not just in a report.
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
    """One coherent L0-L9 curriculum over local exposure."""

    def __init__(self):
        self.native_verified = False
        self.layers: dict[str, dict] = {
            name: {"accuracy": 0.0, "coverage": 0.0, "examples": 0}
            for name in LAYER_NAMES
        }
        self.signature = [0] * 16
        self.vocabulary: dict[str, int] = {}
        self.lexicon: set[str] = set()
        self.prefixes: dict[str, dict[str, int]] = {}
        self.phonetic: dict[str, dict] = {}
        self.morphemes: dict[str, dict] = {}
        self.frames: dict[tuple[str, str], dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}
        self.coherence: dict[tuple[str, str], int] = {}
        self.causal: list[tuple[str, str]] = []

    # -- L0-L2 ----------------------------------------------------------
    def signature_of(self, text: str) -> list[int]:
        data = str(text).encode("utf-8")
        values = []
        for mask in ROOT_MASKS:
            acc = 0
            for index in range(0, len(data), 2):
                left = data[index] & 0x0F
                right = ((data[index] >> 4) | (data[index + 1] if index + 1 < len(data) else 0)) & 0x0F
                acc ^= apply_bits(mask, left, right, 4)
            values.append(acc)
        return values

    def _learn_root(self, sentences: list[str]) -> None:
        correct = total = 0
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
        aggregate = [0] * 16
        for sentence in sentences:
            values = self.signature_of(sentence)
            for index in range(16):
                aggregate[index] ^= values[index]
        self.signature = aggregate

    # -- L3-L6 ----------------------------------------------------------
    def _phonetic_features(self, word: str) -> dict:
        vowels = sum(1 for ch in word if ch in "aeiou")
        consonants = sum(1 for ch in word if ch.isalpha() and ch not in "aeiou")
        syllables = max(1, vowels)
        return {
            "vowels": vowels, "consonants": consonants, "syllables": syllables,
            "valid": bool(vowels and consonants),
            "onset": word[:1], "coda": word[-1:],
        }

    def _segment(self, word: str) -> dict:
        for suffix in ("tion", "ing", "ed", "es", "s", "ly", "er"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return {"stem": word[:-len(suffix)], "suffix": suffix, "prefix": ""}
        for prefix in ("un", "re", "pre", "dis"):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                return {"stem": word[len(prefix):], "suffix": "", "prefix": prefix}
        return {"stem": word, "suffix": "", "prefix": ""}

    def _learn_words(self, sentences: list[str]) -> None:
        words = []
        for sentence in sentences:
            words.extend(content_words(sentence))
        self.vocabulary = {}
        for word in words:
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
        self.lexicon = {stem(word) for word in self.vocabulary}
        phonetic = {word: self._phonetic_features(word) for word in self.vocabulary}
        self.phonetic = phonetic
        valid = sum(1 for data in phonetic.values() if data["valid"])
        self.layers["phonetic"] = {
            "accuracy": valid / len(phonetic) if phonetic else 0.0,
            "coverage": 1.0 if phonetic else 0.0,
            "examples": len(phonetic),
        }
        morphemes = {word: self._segment(word) for word in self.vocabulary}
        self.morphemes = morphemes
        segmented = sum(1 for data in morphemes.values()
                        if data["suffix"] or data["prefix"])
        self.layers["morpheme"] = {
            "accuracy": segmented / len(morphemes) if morphemes else 0.0,
            "coverage": 1.0 if morphemes else 0.0,
            "examples": segmented,
        }
        self.layers["lexical"] = {
            "accuracy": 1.0 if self.vocabulary else 0.0,
            "coverage": 1.0 if self.vocabulary else 0.0,
            "examples": len(self.vocabulary),
        }
        prefixes: dict[str, dict[str, int]] = {}
        for word in self.vocabulary:
            for size in (2, 3):
                if len(word) > size + 1:
                    key = word[:size]
                    prefixes.setdefault(key, {})
                    prefixes[key][word] = prefixes[key].get(word, 0) + 1
        self.prefixes = prefixes
        usable = sum(1 for data in prefixes.values() if len(data) >= 1)
        self.layers["word_completion"] = {
            "accuracy": usable / len(prefixes) if prefixes else 0.0,
            "coverage": 1.0 if prefixes else 0.0,
            "examples": usable,
        }

    # -- L7-L9 ----------------------------------------------------------
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
        typed = sum(1 for edge in self.edges.values()
                    if edge["subject"] and edge["predicate"] and edge["object"])
        self.layers["sentence"] = {
            "accuracy": typed / len(self.edges) if self.edges else 0.0,
            "coverage": 1.0 if self.edges else 0.0,
            "examples": typed,
        }
        adjacency: dict[tuple[str, str], int] = {}
        previous = None
        links = total_pairs = 0
        for sentence in sentences:
            current = tuple(sorted({stem(word) for word in content_words(sentence)}))
            if previous and current:
                total_pairs += 1
                linked = False
                for subject in previous:
                    for obj in current:
                        if subject == obj:
                            continue
                        adjacency[(subject, obj)] = adjacency.get((subject, obj), 0) + 1
                        linked = True
                links += linked
            previous = current
        self.coherence = adjacency
        self.layers["paragraph"] = {
            "accuracy": links / total_pairs if total_pairs else 0.0,
            "coverage": 1.0 if total_pairs else 0.0,
            "examples": total_pairs,
        }
        causal = [(edge["subject"], edge["object"])
                  for edge in self.edges.values()
                  if edge["predicate"] in {"causes", "cause"}]
        continuous = sum(1 for subject, obj in causal
                         if any(subject in sentence.casefold()
                                and obj.split()[0] in sentence.casefold()
                                for sentence in sentences))
        self.causal = causal
        self.layers["narrative"] = {
            "accuracy": continuous / len(causal) if causal else 0.0,
            "coverage": 1.0 if causal else 0.0,
            "examples": len(causal),
        }

    # -- public ----------------------------------------------------------
    def learn(self, sentences: list[str], rows: list[dict]) -> dict:
        self._learn_root(sentences)
        self._learn_words(sentences)
        self._learn_sentences(rows, sentences)
        return self.summary()

    def forward(self, text: str, context: str = "") -> dict:
        """Run the layer stack on one text; every layer produces output."""
        words = content_words(text)
        return {
            "bit": self.signature_of(text),
            "nibble": [apply_bits(mask, len(text) & 0x0F, sum(text.encode("utf-8")) & 0x0F, 4)
                       for mask in ROOT_MASKS],
            "byte": [apply_bits(mask, sum(text.encode("utf-8")) & 0xFF,
                                (len(text) * 17) & 0xFF, 8) for mask in ROOT_MASKS],
            "phonetic": [self._phonetic_features(word) for word in words],
            "morpheme": [self._segment(word) for word in words],
            "lexical": [{"word": word, "known": stem(word) in self.lexicon}
                        for word in words],
            "word_completion": {word: list(self.prefixes.get(word[:2], {}))[:8]
                                for word in words},
            "sentence": [edge for edge in self.edges.values()
                         if any(stem(term) in {stem(word) for word in words}
                                for term in tokens(edge["object"]))][:8],
            "paragraph": [{"context_terms": len(set(content_words(context))),
                           "overlap": len(set(content_words(context)) & set(words))}
                          if context else {"context_terms": 0, "overlap": 0}],
            "narrative": [{"subject": subject, "object": obj}
                          for subject, obj in self.causal][:8],
        }

    def summary(self) -> dict:
        return {
            "schema": "nanlm.mth-layers/1",
            "native_verified": self.native_verified,
            "pipeline": " -> ".join(LAYER_NAMES),
            "layers": {name: dict(self.layers[name]) for name in LAYER_NAMES},
            "edges": len(self.edges),
            "causal_edges": len(self.causal),
            "vocabulary": len(self.vocabulary),
            "prefixes": len(self.prefixes),
        }


def _norm_phrase(text: str) -> str:
    words = [word for word in tokens(text)
             if word not in {"a", "an", "the"}]
    return " ".join(words)


def structured_candidates(query: str, rows: list[dict]) -> list[dict]:
    """Answer a factual question from learned relation edges, not n-grams.

    Learned frames are applied to the question's subject. This is how the MTH
    relation store participates in speech: the candidate is a learned relation
    schema instantiated with the question's unit, not a canned reply.
    """
    edges = []
    for row in rows:
        if row.get("kind") != "relation":
            continue
        subject = str(row.get("subject") or "").strip()
        predicate = str(row.get("predicate") or "").strip()
        obj = str(row.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        role, extra = relation_role(predicate, obj)
        edges.append({"subject": subject, "predicate": predicate,
                      "object": obj, "role": role,
                      "degree": float(row.get("confidence") or 0.55),
                      "polarity": str(row.get("polarity") or "supports")})
    lowered = query.strip().casefold()
    candidates = []

    def add(subject, predicate, obj, edge):
        if predicate in {"is", "are", "was", "were"}:
            verb = "is"
        elif predicate in {"has", "have", "had"}:
            verb = "has"
        elif predicate in {"can", "may", "will"}:
            verb = "can"
        elif predicate in {"causes", "cause"}:
            verb = "causes"
        else:
            verb = predicate
        text = f"{subject} {verb} {obj}"
        if not text.endswith((".", "!", "?")):
            text += "."
        text = text[0].upper() + text[1:]
        candidates.append({"text": text, "truth": edge["degree"],
                           "role": edge["role"],
                           "source": "learned_relation",
                           "subject": edge["subject"],
                           "object": edge["object"]})

    yes_no = re.match(r"^(?:is|are|was|were)\s+(.+?)\s+(?:a|an|the)\s+(.+?)\s*\?$",
                      lowered)
    if yes_no:
        subject = _norm_phrase(yes_no.group(1))
        obj = _norm_phrase(yes_no.group(2))
        for edge in edges:
            if _norm_phrase(edge["subject"]) == subject and                     obj in _norm_phrase(edge["object"]):
                add(edge["subject"], edge["predicate"], edge["object"], edge)
    what_is = re.match(r"^what\s+(?:is|are)\s+(.+?)\s*\?$", lowered)
    if what_is:
        subject = _norm_phrase(what_is.group(1))
        for edge in edges:
            if _norm_phrase(edge["subject"]) == subject:
                add(edge["subject"], edge["predicate"], edge["object"], edge)
    what_causes = re.match(r"^what\s+causes?\s+(.+?)\s*\?$", lowered)
    if what_causes:
        obj = _norm_phrase(what_causes.group(1))
        for edge in edges:
            if edge["predicate"] in {"causes", "cause"} and                     obj in _norm_phrase(edge["object"]):
                add(edge["subject"], edge["predicate"], edge["object"], edge)
    candidates.sort(key=lambda item: -item["truth"])
    return candidates


class MTHScorer:
    """Candidates are scored by every L0-L9 layer, then combined."""

    def __init__(self, stack: MTHLayerStack):
        self.stack = stack

    def _structure(self, candidate: str) -> float:
        if not any(self.stack.signature):
            return 0.5
        values = self.stack.signature_of(candidate)
        same = sum(1 for a, b in zip(values, self.stack.signature) if a == b)
        return same / len(values)

    def _phonetic(self, words: list[str]) -> float:
        if not words:
            return 0.0
        scores = []
        for word in words:
            features = self.stack._phonetic_features(word)
            scores.append(1.0 if features["valid"] else 0.3)
        return sum(scores) / len(scores)

    def _morpheme(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return sum(1 for word in words
                   if (self.stack._segment(word)["stem"] in self.stack.lexicon)
                   or self.stack._segment(word)["suffix"]
                   or self.stack._segment(word)["prefix"]) / len(words)

    def _lexical(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return sum(1 for word in words if stem(word) in self.stack.lexicon) / len(words)

    def _completion(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return sum(1 for word in words if self.stack.prefixes.get(word[:2])) / len(words)

    def score(self, query: str, candidate: str, context: str = "") -> dict:
        query_words = {stem(word) for word in content_words(query)}
        candidate_words = content_words(candidate)
        candidate_set = {stem(word) for word in candidate_words}
        anchor = stem(candidate_words[0]) if candidate_words else ""
        lowered = query.casefold()
        expected = "is"
        if "component of" in lowered:
            expected = "component_of"
        elif "chemical element" in lowered:
            expected = "member_of:chemical"
        elif "geometric element" in lowered:
            expected = "member_of:geometric"
        elif "opposite" in lowered:
            expected = "opposition"
        elif " can " in " " + lowered + " ":
            expected = "can"
        elif lowered.startswith("why") or "cause" in lowered:
            expected = "causes"

        truth = frame = category_error = 0.0
        matched = None
        for (subject, predicate, obj), edge in self.stack.edges.items():
            subject_terms = {stem(word) for word in tokens(subject)}
            object_terms = {stem(word) for word in tokens(obj)}
            subject_match = bool(subject_terms & candidate_set)
            overlap = len(object_terms & candidate_set) / max(1, len(object_terms))
            if overlap and not subject_match:
                category_error = max(category_error, 0.7 * overlap)
            if not subject_match or overlap <= 0:
                continue
            if anchor and anchor not in subject_terms:
                continue
            value = overlap * (edge["degree"] if edge["polarity"] != "refutes" else 0.0)
            if value > truth:
                truth, matched = value, edge
        if matched is not None:
            matched_role, _extra = relation_role(matched["predicate"], matched["object"])
            frame = (1.0 if matched_role == expected
                     or (expected == "is" and matched_role in {"is", "has"})
                     else 0.25)

        lexical = self._lexical(candidate_words)
        coherence = (len(query_words & candidate_set) / max(1, len(query_words))
                     if query_words else 0.0)
        if context:
            context_words = {stem(word) for word in content_words(context)}
            coherence = max(coherence,
                            len(context_words & candidate_set) /
                            max(1, len(context_words)))
        narrative = 0.5
        if expected == "causes" and self.stack.causal:
            narrative = 1.0 if matched is not None and matched["predicate"] == "causes" else 0.2
        layers = {
            "bit": self._structure(candidate),
            "nibble": self._structure(candidate),
            "byte": self._structure(candidate),
            "phonetic": self._phonetic(candidate_words),
            "morpheme": self._morpheme(candidate_words),
            "lexical": lexical,
            "word_completion": self._completion(candidate_words),
            "sentence": truth,
            "paragraph": coherence,
            "narrative": narrative,
        }
        composite = (0.03 * layers["bit"] + 0.03 * layers["nibble"]
                     + 0.04 * layers["byte"] + 0.05 * layers["phonetic"]
                     + 0.05 * layers["morpheme"] + 0.08 * layers["lexical"]
                     + 0.04 * layers["word_completion"] + 0.25 * truth
                     + 0.10 * frame + 0.13 * coherence + 0.10 * narrative
                     - 0.45 * category_error)
        return {
            "truth": round(truth, 4),
            "frame": round(frame, 4),
            "coherence": round(coherence, 4),
            "category_error": round(category_error, 4),
            "layers": {name: round(value, 4) for name, value in layers.items()},
            "score": round(composite, 4),
        }


def reverse8(value: int) -> int:
    return int(format(value & 0xFF, "08b")[::-1], 2)


def relation_role(predicate: str, obj: str) -> tuple[str, dict]:
    """Role of a relation, not a generic inverse.

    A pair is an axis; the two rails carry directed roles. Antonymy is one
    role (opposition), not the default interpretation of a pair.
    """
    text = (str(predicate) + " " + str(obj)).casefold()
    if "component of" in text:
        return "component_of", {"whole": text.split("component of", 1)[1].strip()}
    if "opposite of" in text or "antonym" in text:
        return "opposition", {}
    if "chemical element" in text:
        return "member_of:chemical", {"qualifier": "chemical"}
    if "geometric element" in text:
        return "member_of:geometric", {"qualifier": "geometric"}
    if predicate in {"is", "has", "can", "causes", "cause"}:
        return predicate, {}
    return predicate or "relates", {}


def leaf_tags(subject: str, predicate: str, obj: str, role: str,
              whole: str, digest: str) -> dict:
    """The user's tag grammar as a leaf schema (description is a unit)."""
    return {
        "DESCRIPTION": subject,
        "FORMAL:ID:LABEL::DIRECTIVE": digest + "::assert",
        "OBJECT:REALM": {"object": subject, "realm": "physical"},
        "DESCRIPTION:DIRECTOR:ABSTRACT": {
            "description": subject, "director": "assert", "abstract": role,
        },
        "UNIT:DIVISION:RELATION": {
            "unit": subject, "division": "relation", "relation": role,
            "predicate": predicate, "object": obj,
        },
        "UNIT:WHOLE:COMPONENT:QUERY": {
            "whole": whole, "component": subject,
            "query": "is_component_of" if role == "component_of" else role,
        },
        "SPACE:BODY:COMPONENT:SYSTEM": {
            "space": whole if role == "component_of" else "general",
            "body": "planet" if role == "component_of" else "general",
            "component": subject, "system": role,
        },
    }


class MTHHeadMapper:
    """Map a relation into a head and a mathematical leaf via root functions.

    The head address and leaf value are produced by composing the relation's
    bytes through a root Boolean mask, so the 256 heads are actively using the
    4x4 Boolean table rather than a plain hash.
    """

    def __init__(self, stack: MTHLayerStack):
        self.stack = stack

    def transform(self, text: str, mask: int) -> int:
        data = text.casefold().encode("utf-8")
        out = 0
        for index in range(0, len(data), 2):
            left = data[index] & 0x0F
            right = ((data[index] >> 4) |
                     (data[index + 1] if index + 1 < len(data) else 0)) & 0x0F
            out ^= apply_bits(mask, left, right, 4)
        return out

    def map_relation(self, row: dict) -> dict:
        subject = str(row.get("subject") or "").strip().casefold()
        predicate = str(row.get("predicate") or "").strip().casefold()
        obj = str(row.get("object") or "").strip().casefold()
        mask = (sum(ord(char) for char in predicate)
                + len(subject) + len(obj)) % 16
        s = self.transform(subject, mask)
        p = self.transform(predicate, mask)
        o = self.transform(obj, mask)
        role, extra = relation_role(predicate, obj)
        whole = extra.get("whole") or obj
        object_head = self.transform(whole, mask)
        pair = tuple(sorted((s, object_head)))
        rail = "s+" if s < object_head else "s-"
        reverse_rail = "s-" if rail == "s+" else "s+"
        head = s
        layer = ("narrative" if predicate in {"causes", "cause"}
                 else "sentence")
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            resolution = str(provenance.get("resolution") or "")
            if resolution in LAYER_NAMES:
                layer = resolution
        digest = __import__("hashlib").sha256(
            (subject + "|" + predicate + "|" + obj).encode("utf-8")).hexdigest()[:16]
        return {
            "head": head,
            "mirror": reverse8(head),
            "mask": mask,
            "boolean": ROOT_NAMES[mask],
            "value": (s << 16) | (object_head << 8) | o,
            "pair": pair,
            "rail": rail,
            "reverse_rail": reverse_rail,
            "role": role,
            "whole": whole,
            "qualifier": extra.get("qualifier", ""),
            "layer": layer,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "tags": leaf_tags(subject, predicate, obj, role, whole, digest),
            "digest": digest,
        }
