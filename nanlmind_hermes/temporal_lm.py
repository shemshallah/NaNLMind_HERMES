"""Temporal mirror language model: predict every next word from exposure.

This is a small, dependency-free word-level predictor trained online from the
sentences the learner is exposed to. It does not retrieve or mirror stored
sentences: generation always scores every word in the vocabulary and predicts
the next word forward from a temporal frame, then feeds the chosen word back
into the frame (temporal feedback loop).

Addressing follows the native universe convention: every word owns an 8-bit
head address and its mirror is the bit-reversal of that address
(``mirror8(h) = reverse8(h)``, the 8-bit analogue of the native reverse16
mirror link). Mirror cells participate in two ways:

- the temporal frame is a 256-cell activation vector updated after every
  emitted word (``state[h] += 1`` and ``state[mirror(h)] += 0.5``), decayed
  each step; and
- candidate scoring adds learned mirror-address transition mass
  ``P(mirror(next) | mirror(context))``.

Candidate scoring interpolates trigram, bigram and unigram counts learned from
exposure, adds the temporal/mirror terms, and applies repetition penalties.
Beam search keeps several forward hypotheses alive, then chooses the best
completed continuation for the query anchor. There are no response templates:
the output is only vocabulary the learner has been exposed to.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .universe_chart import LocalRunner, TemporalRegister

SCHEMA = "nanlm.temporal-mirror-lm/1"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.!?,;:]")
BOS = "<s>"
EOS = "</s>"
END = frozenset({".", "!", "?"})
PUNCT = frozenset({".", "!", "?", ",", ";", ":"})
_REQUEST_STARTS = frozenset({
    "tell", "show", "list", "explain", "describe", "summarize", "name",
})

STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "with", "about", "as", "at", "by", "from", "it",
    "its", "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
    "do", "does", "did", "can", "could", "would", "should", "what", "how",
    "where", "when", "why", "who", "whom", "whose", "which", "tell", "show",
    "list", "explain", "describe", "summarize", "name",
})

MAX_VOCAB = 65_536
MAX_SEEN = 65_536
NEGATIVE_STARTS = (
    "no", "nope", "wrong", "incorrect", "not correct", "not right",
    "that's wrong", "that is wrong", "that's incorrect", "that is incorrect",
    "try again", "that is not right", "that's not right",
)


def is_correction(text: str) -> bool:
    """Control input, not conversation: the user rejects the last answer."""
    value = str(text).strip().casefold()
    return any(value == word or value.startswith(word + ",") or
               value.startswith(word + ".") or value.startswith(word + " ")
               for word in NEGATIVE_STARTS)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def stem(token: str) -> str:
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    return token[:-1] if len(token) > 4 and token.endswith("s") else token


def mirror8(head: int) -> int:
    """Native mirror link: reverse the 8-bit head address."""
    return int(format(head & 0xFF, "08b")[::-1], 2)


def fnv_head(word: str) -> int:
    h = 2166136261
    for ch in word.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % 256


def _detokenize(words: list[str]) -> str:
    cleaned = []
    for token in words:
        if (cleaned and token not in PUNCT
                and cleaned[-1].casefold() == token.casefold()):
            continue
        if token in PUNCT and not cleaned:
            continue
        cleaned.append(token)
    text = ""
    for token in cleaned:
        if token in PUNCT:
            text = text.rstrip() + token
        else:
            text += ("" if not text else " ") + token
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


class TemporalMirrorLM:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.uni: dict[str, int] = {}
        self.bi: dict[str, dict[str, int]] = {}
        self.tri: dict[str, dict[str, int]] = {}
        self.pairs: dict[str, dict[str, float]] = {}
        self.subject: dict[str, int] = {}
        self.seen: list[str] = []
        self.recent: list[str] = []
        self.rejected: list[str] = []
        self.last_query: str = ""
        self.last_reply: str = ""
        self.output_drive: float = 1.0
        self.register: TemporalRegister | None = None
        self.runner: LocalRunner | None = None
        self.mth = None

    # -- persistence ----------------------------------------------------
    @staticmethod
    def path(ctx) -> Path:
        root = Path(str(ctx.get_config("state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
        return root / "language" / "temporal_mirror.json"

    @classmethod
    def load(cls, ctx) -> "TemporalMirrorLM":
        model = cls()
        path = cls.path(ctx)
        if not path.exists():
            model.runner = LocalRunner.open(ctx)
            model.register = model.runner.register
            return model
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            model.runner = LocalRunner.open(ctx)
            model.register = model.runner.register
            return model
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            model.runner = LocalRunner.open(ctx)
            model.register = model.runner.register
            return model
        model.vocab = {str(k): int(v) & 0xFF for k, v in (data.get("vocab") or {}).items()}
        model.uni = {str(k): int(v) for k, v in (data.get("uni") or {}).items()}
        model.bi = {str(k): {str(w): int(n) for w, n in (v or {}).items()}
                    for k, v in (data.get("bi") or {}).items()}
        model.tri = {str(k): {str(w): int(n) for w, n in (v or {}).items()}
                     for k, v in (data.get("tri") or {}).items()}
        model.pairs = {str(k): {str(w): float(n) for w, n in (v or {}).items()}
                       for k, v in (data.get("pairs") or {}).items()}
        model.subject = {str(k): int(v) for k, v in (data.get("subject") or {}).items()}
        model.seen = [str(s) for s in (data.get("seen") or [])][-MAX_SEEN:]
        model.recent = [str(s) for s in (data.get("recent") or [])][-8:]
        model.rejected = [str(s) for s in (data.get("rejected") or [])][-32:]
        model.last_query = str(data.get("last_query") or "")
        model.last_reply = str(data.get("last_reply") or "")
        model.output_drive = float(data.get("output_drive", 1.0) or 1.0)
        model.runner = LocalRunner.open(ctx)
        model.register = model.runner.register
        return model

    def save(self, ctx) -> None:
        path = self.path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema": SCHEMA,
            "vocab": self.vocab,
            "uni": self.uni,
            "bi": self.bi,
            "tri": self.tri,
            "pairs": self.pairs,
            "subject": self.subject,
            "seen": self.seen[-MAX_SEEN:],
            "recent": self.recent[-8:],
            "rejected": self.rejected[-32:],
            "last_query": self.last_query,
            "last_reply": self.last_reply,
            "output_drive": self.output_drive,
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8")
        try:
            temp.chmod(0o600)
        except OSError:
            pass
        temp.replace(path)
        # Publish the verified local 8 MiB byte snapshot before optional JSON
        # mirrors. This is NOT a canonical NaNLMind training commit.
        if self.runner is not None:
            self.runner.checkpoint()
        if self.register is not None:
            self.register.save()

    # -- exposure -------------------------------------------------------
    def observe(self, sentence: str) -> bool:
        """Learn from one exposed sentence. Returns False if already seen."""
        sentence = str(sentence).strip()
        if len(sentence) < 2:
            return False
        if sentence.rstrip().endswith("?"):
            return False  # questions are prompts, not language exposure
        first_word = re.match(r"[A-Za-z']+", sentence)
        if first_word and first_word.group(0).casefold() in _REQUEST_STARTS:
            return False  # requests are not facts
        digest = hashlib.sha256(sentence.casefold().encode("utf-8")).hexdigest()
        if digest in self.seen:
            return False
        self.seen.append(digest)
        toks = [BOS] + tokens(sentence) + [EOS]
        for token in toks:
            if token not in (BOS, EOS):
                if token not in self.vocab and len(self.vocab) >= MAX_VOCAB:
                    continue
                self.vocab.setdefault(token, fnv_head(token))
            self.uni[token] = self.uni.get(token, 0) + 1
        for index in range(1, len(toks)):
            target = toks[index]
            self.bi.setdefault(toks[index - 1], {}).setdefault(target, 0)
            self.bi[toks[index - 1]][target] += 1
            if index >= 2:
                key = toks[index - 2] + "\x00" + toks[index - 1]
                self.tri.setdefault(key, {}).setdefault(target, 0)
                self.tri[key][target] += 1
        raw = tokens(sentence)
        for position, word in enumerate(raw[:3]):
            if word not in STOP and word not in END:
                self.subject[word] = self.subject.get(word, 0) + (2 if position == 0 else 1)
                break
        lowered = sentence.casefold()
        if any(phrase in lowered for phrase in
               ("do not know", "don't know", "not know", "unknown",
                "not learned", "haven't learned")):
            self.output_drive = max(0.05, self.output_drive * 0.985)
        else:
            self.output_drive = min(1.0, self.output_drive + 0.002)
        for index in range(1, len(toks)):
            if toks[index] in (BOS, EOS):
                continue
            for back in range(index - 1, max(-1, index - 4), -1):
                if toks[back] == BOS:
                    continue
                weight = 0.5 ** (index - 1 - back)
                self._pair(toks[back], toks[index], weight)
        return True

    def question_terms(self, text: str) -> list[str]:
        return [token for token in tokens(text)
                if token not in STOP and token not in END and token != BOS]

    def add_context_terms(self, text: str) -> list[str]:
        """Add question entities to the anchor vocabulary without teaching them."""
        added = []
        for token in self.question_terms(text):
            if token not in self.vocab:
                self.vocab[token] = fnv_head(token)
                added.append(token)
        return added

    def observe_memory_rows(self, rows: list[dict], exclude: set[str] | None = None) -> int:
        learned = 0
        for row in rows:
            sentence = None
            if row.get("kind") in {"relation", "equation", "constraint", "operator"}:
                evidence = row.get("evidence")
                if isinstance(evidence, dict):
                    sentence = evidence.get("sentence")
            elif row.get("kind") == "observation":
                sentence = row.get("object")
            if not isinstance(sentence, str):
                continue
            if exclude and sentence.strip().casefold() in exclude:
                continue
            if self.observe(sentence):
                learned += 1
        return learned

    def _pair(self, first: str, second: str, weight: float) -> None:
        head_a, head_b = fnv_head(first), fnv_head(second)
        self.pairs.setdefault(str(head_a), {}).setdefault(str(head_b), 0.0)
        self.pairs[str(head_a)][str(head_b)] += weight
        mirror_key = "m" + str(mirror8(head_a))
        mirror_target = str(mirror8(head_b))
        self.pairs.setdefault(mirror_key, {}).setdefault(mirror_target, 0.0)
        self.pairs[mirror_key][mirror_target] += weight

    # -- prediction -----------------------------------------------------
    def _prob(self, context2: str | None, context1: str, word: str) -> float:
        total_uni = sum(self.uni.values()) or 1
        p1 = (self.uni.get(word, 0) + 0.01) / (total_uni + 0.01 * len(self.vocab))
        bigram = self.bi.get(context1)
        p2 = bigram.get(word, 0) / sum(bigram.values()) if bigram else 0.0
        trigram = self.tri.get(context2 + "\x00" + context1) if context2 else None
        p3 = trigram.get(word, 0) / sum(trigram.values()) if trigram else 0.0
        return 0.12 * p1 + 0.33 * p2 + 0.55 * p3

    def _seed_state(self, seed: list[str], decay: float = 0.75) -> list[float]:
        if self.register is not None and any(self.register.state):
            total = sum(self.register.state) or 1.0
            state = [0.25 * value / total for value in self.register.state]
            forward = self.register.forward_payload
            if forward:
                head = (forward >> 24) & 0xFF
                state[head] += 0.02
                state[mirror8(head)] += 0.01
        else:
            state = [0.0] * 256
        for index, word in enumerate(seed):
            weight = decay ** (len(seed) - 1 - index)
            head = self.vocab.get(word, fnv_head(word))
            state[head] += weight
            state[mirror8(head)] += 0.5 * weight
        total = sum(state) or 1.0
        return [value / total for value in state]

    def _advance(self, state: list[float], word: str, decay: float = 0.7) -> list[float]:
        state = [value * decay for value in state]
        head = self.vocab.get(word, fnv_head(word))
        state[head] += 1.0
        state[mirror8(head)] += 0.5
        total = sum(state) or 1.0
        return [value / total for value in state]

    def _beam(self, anchor: str, state: list[float], beam: int = 4,
              max_tokens: int = 14, mirror_gain: float = 0.18,
              pair_gain: float = 0.15, repetition: float = 1.4) -> list[dict]:
        """Weight every vocabulary word at every step and keep k futures."""
        candidates = {"version": 0}
        active = [{"logp": 0.0, "words": [anchor], "counts": {}, "state": state}]
        completed: list[dict] = []
        vocabulary = [w for w in self.vocab if w != BOS]
        for _ in range(max_tokens):
            next_active = []
            for hypothesis in active:
                words = hypothesis["words"]
                context1 = words[-1]
                context2 = words[-2] if len(words) >= 2 else None
                for word in vocabulary:
                    probability = self._prob(context2, context1, word)
                    if probability <= 0:
                        continue
                    score = math.log(probability + 1e-9)
                    head = self.vocab[word]
                    if word not in (EOS,):
                        score += mirror_gain * hypothesis["state"][head]
                    if context1 != BOS and word != EOS:
                        mirrored = self.pairs.get("m" + str(mirror8(fnv_head(context1)))) or {}
                        mirrored_total = sum(mirrored.values()) or 1.0
                        score += pair_gain * mirrored.get(str(mirror8(head)), 0.0) / mirrored_total
                    score -= repetition * hypothesis["counts"].get(word, 0)
                    if word in words[-2:]:
                        score -= 0.7
                    if word == BOS:
                        continue
                    if word == EOS or (word in END and len(words) - 1 >= 2):
                        completed.append({"logp": hypothesis["logp"] + score,
                                          "words": list(words),
                                          "counts": dict(hypothesis["counts"]),
                                          "ended": True})
                        continue
                    if word in END and len(words) - 1 < 2:
                        continue
                    counts = dict(hypothesis["counts"])
                    counts[word] = counts.get(word, 0) + 1
                    next_active.append({
                        "logp": hypothesis["logp"] + score,
                        "words": words + [word],
                        "counts": counts,
                        "state": self._advance(hypothesis["state"], word),
                    })
            if not next_active:
                break
            next_active.sort(key=lambda h: -h["logp"])
            active = next_active[:beam]
        for hypothesis in active:
            completed.append({"logp": hypothesis["logp"], "words": hypothesis["words"],
                              "counts": hypothesis["counts"], "ended": False})
        return completed

    def _select_sentence(self, anchors, state, seed_stems, beam, max_tokens,
                         query: str = "", context: str = ""):
        max_subject = max(self.subject.values() or [1])
        considered = []
        best = None
        for anchor_index, anchor in enumerate(anchors):
            for candidate in self._beam(anchor, state, beam=beam, max_tokens=max_tokens):
                words = candidate["words"]
                if len(words) < 2:
                    continue
                covered = {stem(word) for word in words} & seed_stems
                length = max(1, len(words) - 1)
                score = candidate["logp"] / length
                if seed_stems:
                    score += 0.6 * (len(covered) / len(seed_stems))
                score += 0.6 * (self.subject.get(anchor, 0) / max_subject)
                score += 0.12 * (1.0 - anchor_index / max(1, len(anchors)))
                score += 0.6 if candidate["ended"] else -0.6
                score -= 0.05 * max(0, len(words) - 14)
                if len(words) - 1 > 12:
                    score -= 0.3
                text = _detokenize([w for w in words if w not in (BOS, EOS)])
                if self.mth is not None:
                    mth_score = self.mth.score(query, text, context=context)
                    score += 0.35 * mth_score["score"]
                else:
                    mth_score = None
                if text in self.rejected:
                    score -= 6.0
                if text in self.recent:
                    score -= 2.5
                score += 0.12 * ((fnv_head(query.casefold() + "|" + text) % 97) / 97.0)
                considered.append({
                    "text": text, "score": round(score, 4),
                    "anchor": anchor, "ended": candidate["ended"],
                    "mth": mth_score})
                if best is None or score > best["score"]:
                    best = {"score": score, "anchor": anchor, "words": words,
                            "ended": candidate["ended"]}
        considered.sort(key=lambda row: -row["score"])
        return best, considered[:8]

    def _complete_sentence(self, words, state, max_extra: int = 12):
        out = list(words)
        context2 = out[-2] if len(out) >= 2 else None
        context1 = out[-1]
        current = list(state)
        counts = {}
        for word in out:
            if word not in (BOS, EOS):
                counts[word] = counts.get(word, 0) + 1
        while len(out) - len(words) < max_extra:
            best = None
            for word in self.vocab:
                if word == BOS:
                    continue
                probability = self._prob(context2, context1, word)
                if probability <= 0:
                    continue
                score = math.log(probability + 1e-9)
                head = self.vocab[word]
                if word != EOS:
                    score += 0.18 * current[head]
                if context1 != BOS and word != EOS:
                    mirrored = self.pairs.get("m" + str(mirror8(fnv_head(context1)))) or {}
                    total = sum(mirrored.values()) or 1.0
                    score += 0.15 * mirrored.get(str(mirror8(head)), 0.0) / total
                score -= 1.4 * counts.get(word, 0)
                if word in out[-2:]:
                    score -= 0.7
                if best is None or score > best[0]:
                    best = (score, word)
            if best is None:
                break
            word = best[1]
            if word in (BOS, EOS) or word in END:
                break
            out.append(word)
            counts[word] = counts.get(word, 0) + 1
            current = self._advance(current, word)
            context2, context1 = context1, word
        return out

    def generate(self, query: str, beam: int = 4, max_tokens: int = 16,
                 temperature: float = 0.0,
                 allowed_seed: set[str] | None = None,
                 mutate_register: bool = True) -> dict:
        raw_seed = [token for token in tokens(query)
                    if token not in STOP and token not in END and token != BOS]
        seed = [token for token in raw_seed
                if token in self.vocab
                and (allowed_seed is None or token in allowed_seed)]
        started_from_bos = False
        if not seed:
            seed = [token for token in raw_seed if token in self.vocab]
        if not seed:
            if self.output_drive < 0.35:
                return {"text": "", "seed": [], "anchor": None, "words": [],
                        "started_from_bos": False, "output_drive": round(self.output_drive, 4),
                        "sentences": [], "explanation_rounds": 0, "coverage": 0.0,
                        "steps": 0, "scored_per_step": len(self.vocab),
                        "mirror_state": True, "vocabulary": len(self.vocab),
                        "register_frames": 0,
                        "register": self.register.status() if self.register else None,
                        "runner": self.runner.status() if self.runner else None}
            seed = [BOS]
            started_from_bos = True
        seed_stems = {stem(token) for token in seed if token != BOS}
        state = self._seed_state([token for token in seed if token != BOS])
        sentences = []
        used_words = []
        rounds = 0
        pending = list(seed)
        coverage = 0.0
        anchor_used = None
        weighed = []
        for round_index in range(4):
            budget = max_tokens if round_index == 0 else 8
            best, considered = self._select_sentence(
                pending, state, seed_stems, beam, budget, query=query,
                context=sentences[-1] if sentences else "")
            if best is None:
                break
            words = best["words"]
            if not best["ended"]:
                words = self._complete_sentence(words, state)
            spoken = [word for word in words if word not in (BOS, EOS)]
            text = _detokenize(spoken)
            if not text or text in sentences:
                break
            if anchor_used is None:
                anchor_used = best["anchor"]
            weighed.extend(considered)
            sentences.append(text)
            used_words.extend(spoken)
            for word in spoken:
                state = self._advance(state, word)
            used_stems = {stem(word) for word in used_words}
            coverage = (len(used_stems & seed_stems) / len(seed_stems)
                        if seed_stems else 1.0)
            open_ended = query.strip().casefold().startswith(
                ("why", "how", "explain", "what happens"))
            uncovered = [token for token in seed
                         if token != BOS and stem(token) not in used_stems]
            if round_index >= 2 or not open_ended:
                break
            if coverage >= 1.0:
                break
            if uncovered:
                pending = uncovered[:2]
            elif open_ended:
                pending = [BOS]
            else:
                break
            rounds += 1

        if not sentences and self.vocab:
            best, considered = self._select_sentence(
                [BOS], state, set(), beam, 8, query=query,
                context=sentences[-1] if sentences else "")
            if best is not None:
                words = best["words"]
                if not best["ended"]:
                    words = self._complete_sentence(words, state)
                spoken = [word for word in words if word not in (BOS, EOS)]
                text = _detokenize(spoken)
                if text:
                    sentences.append(text)
                    used_words.extend(spoken)
                    anchor_used = BOS

        text = " ".join(sentences)
        for sentence in sentences:
            self.recent.append(sentence)
        self.recent = self.recent[-8:]
        self.last_query = str(query)
        self.last_reply = text
        frames = 0
        if self.register is not None and mutate_register:
            for word in used_words:
                if word in PUNCT:
                    continue
                try:
                    self.register.commit(
                        fnv_head(word),
                        update_state=word not in STOP and word not in PUNCT)
                    frames += 1
                except Exception:
                    break
        return {
            "text": text,
            "seed": seed,
            "anchor": anchor_used,
            "words": used_words,
            "started_from_bos": started_from_bos,
            "output_drive": round(self.output_drive, 4),
            "sentences": sentences,
            "explanation_rounds": rounds,
            "coverage": round(coverage, 4),
            "considered": weighed[:12],
            "steps": len(used_words),
            "scored_per_step": len(self.vocab),
            "mirror_state": True,
            "mirror_addresses": sorted({self.vocab.get(word, 0) for word in used_words})[:32],
            "score": round(best["score"], 4) if sentences else None,
            "ended": all(sentence.endswith((".", "!", "?")) for sentence in sentences),
            "vocabulary": len(self.vocab),
            "register_frames": frames,
            "register": self.register.status() if self.register else None,
            "runner": self.runner.status() if self.runner else None,
        }

    def correct(self) -> dict:
        """Reject the last reply and force a different future for the question."""
        if not self.last_query or not self.last_reply:
            return {}
        self.rejected.append(self.last_reply)
        self.rejected = self.rejected[-32:]
        self.output_drive = max(0.1, self.output_drive * 0.85)
        prediction = self.generate(self.last_query, allowed_seed=None)
        prediction["correction"] = True
        prediction["rejected"] = self.last_reply
        return prediction

    def stats(self) -> dict:
        return {
            "schema": SCHEMA,
            "vocabulary": len(self.vocab),
            "exposed_sentences": len(self.seen),
            "bigram_contexts": len(self.bi),
            "trigram_contexts": len(self.tri),
            "mirror_address_cells": len(self.pairs),
        }
