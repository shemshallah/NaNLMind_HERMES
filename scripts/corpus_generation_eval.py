#!/usr/bin/env python3
"""Evaluate a deeply trained native NaNLM student on held-out generation.

The student is a real ``TemporalMirrorLM``/memory/layered state directory.  Its
replies are produced by the native ``nanlm_chat`` tool with ``probe=True`` (or
by any injected read-only student callable in tests); no LLM writes the student
reply.  The optional Gemma judge only labels an already-written reply against a
small rubric and is never treated as ground truth.

The deterministic metrics describe the emitted text: emptiness, length,
vocabulary diversity, repetition, exact sentence copying, novel 3-grams,
keyword presence and (optionally) next-token NLL/perplexity.  They are surface
measurements, not a proof of language understanding.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Sequence
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import register  # noqa: E402
from nanlmind_hermes.temporal_lm import (  # noqa: E402
    BOS,
    TemporalMirrorLM,
    tokens as lm_tokens,
)

SCHEMA = "nanlm.corpus-generation-eval/1"
DEFAULT_OUTPUT = ROOT / "docs" / "CORPUS_GENERATION_EVAL.json"
DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_SESSION_ID = "corpus-generation-eval"
MIN_COPIED_SENTENCE_TOKENS = 3
JUDGE_REFERENCE_LIMIT = 40
MAX_JUDGE_RAW = 2000
JUDGE_BINARY_KEYS = (
    "answers_prompt",
    "coherent",
    "novel_combination",
    "unsupported_claim",
)
JUDGE_REQUIRED_KEYS = JUDGE_BINARY_KEYS + ("notes",)

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

# Built-in probes cover the task families requested for post-training review.
# Each alternative inside a tuple is a match for the same coverage group.
BUILTIN_PROMPTS = (
    {
        "id": "explain-river",
        "category": "explanation",
        "prompt": "Explain why a river can carry water.",
        "keyword_groups": (
            ("river",),
            ("water",),
            ("carry", "carries", "carried", "flow", "flows", "move", "moves"),
        ),
    },
    {
        "id": "explain-water-cycle",
        "category": "explanation",
        "prompt": "Explain how water can become ice and then water again.",
        "keyword_groups": (
            ("water",),
            ("ice", "frozen"),
            ("freeze", "freezes", "frozen", "melt", "melts", "melting"),
        ),
    },
    {
        "id": "compare-cat-bird",
        "category": "comparison",
        "prompt": "Compare a cat and a bird.",
        "keyword_groups": (
            ("cat", "cats"),
            ("bird", "birds"),
            ("fly", "flies", "fur", "feather", "feathers", "animal",
             "animals", "pet", "pets"),
        ),
    },
    {
        "id": "causal-ice-melt",
        "category": "causal",
        "prompt": "Why does ice melt when it gets hot?",
        "keyword_groups": (
            ("ice",),
            ("melt", "melts", "melting", "melted"),
            ("hot", "heat", "heated", "warm", "temperature"),
        ),
    },
    {
        "id": "counterfactual-bird",
        "category": "counterfactual",
        "prompt": "What would happen if a bird could not fly?",
        "keyword_groups": (
            ("bird", "birds"),
            ("fly", "flies", "flying"),
            ("could not", "cannot", "can't", "unable", "ground", "walk",
             "walks"),
        ),
    },
    {
        "id": "story-river-bird-sun",
        "category": "story",
        "prompt": "Tell a short story about a river, a bird, and the sun.",
        "keyword_groups": (
            ("river",),
            ("bird",),
            ("sun", "sunlight", "sunny"),
        ),
    },
    {
        "id": "define-river",
        "category": "definition",
        "prompt": "What is a river?",
        "keyword_groups": (
            ("river",),
            ("water",),
            ("flow", "flows", "flowing", "body", "stream", "carry"),
        ),
    },
    {
        "id": "define-square",
        "category": "definition",
        "prompt": "What is a square?",
        "keyword_groups": (
            ("square",),
            ("four", "4"),
            ("side", "sides", "shape"),
        ),
    },
    {
        "id": "arithmetic-two-plus-three",
        "category": "arithmetic",
        "prompt": "What is two plus three?",
        "keyword_groups": (
            ("five", "5"),
        ),
    },
    {
        "id": "arithmetic-apples",
        "category": "arithmetic",
        "prompt": ("If there are four apples and two are eaten, "
                   "how many remain?"),
        "keyword_groups": (
            ("two", "2"),
            ("apple", "apples"),
        ),
    },
)

JUDGE_SYSTEM = """You are an evaluator, not the learner. Judge only the supplied
prompt and the already-written NaNLM reply. Return exactly one JSON object with
no markdown and no extra keys:
{"answers_prompt": 0 or 1, "coherent": 0 or 1,
 "novel_combination": 0 or 1, "unsupported_claim": 0 or 1,
 "notes": "short diagnostic string"}
Do not write or replace the reply. Do not reward fluency alone: a fluent but
unrelated answer is not coherent. A claim is unsupported when the reply asserts
something the supplied reference excerpts do not support; do not treat the judge
as ground truth."""

BOUNDARY_NOTES = (
    "Replies are emitted by the native NaNLM student path. The optional Gemma "
    "judge never writes a student reply.",
    "Deterministic metrics are surface measurements over the emitted text, not "
    "a semantic prover and not an accuracy benchmark.",
    "Copied-training detection matches exact corpus sentences of at least "
    f"{MIN_COPIED_SENTENCE_TOKENS} word tokens; paraphrase and shorter phrases "
    "are not detected.",
    "Novel 3-grams are relative only to the supplied corpus and prompt. A novel "
    "n-gram is not necessarily true, useful, or grammatical.",
    "Keyword coverage checks whether an alternative is present. It does not "
    "check correctness, negation, entailment, or whether the term is used in "
    "the right clause.",
    "NLL/perplexity is the student's own next-token fit for the reply given the "
    "prompt. Low NLL can be memorized text; high NLL can be novel but valid.",
    "The optional judge is an LLM diagnostic. It can be wrong, biased, or "
    "fooled by fluent text and is not ground truth.",
    "Held-out status is caller-declared. With --corpus the script reports exact "
    "prompt/corpus matches only; it cannot prove that no paraphrase or other "
    "training path exposed the probe.",
    "No training, canonical commit, route, merge, or proof claim is made.",
)


class EvalContext:
    """Minimal plugin context for tool registration and state access."""

    def __init__(self, root: Path):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "native",
            "chat_learning": False,
            "merge_enabled": False,
            "learning_require_approval": True,
            "learning_auto_sync": False,
        }
        self.tools: dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def register_tool(self, **kwargs: Any) -> None:
        self.tools[kwargs["name"]] = kwargs["handler"]


class CorpusReference:
    """Precomputed corpus view for copied-sentence and novelty metrics."""

    def __init__(self, texts: Iterable[str]):
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in texts:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        self.texts = tuple(cleaned)
        self.normalized_texts = frozenset(text.casefold() for text in self.texts)

        trigrams: set[tuple[str, ...]] = set()
        by_prefix: dict[tuple[str, ...], list[tuple[tuple[str, ...], str]]] = {}
        for text in self.texts:
            tokens = word_tokens(text)
            trigrams.update(_ngrams(tokens, 3))
            if len(tokens) >= MIN_COPIED_SENTENCE_TOKENS:
                prefix = tuple(tokens[:MIN_COPIED_SENTENCE_TOKENS])
                by_prefix.setdefault(prefix, []).append((tuple(tokens), text))
        self.trigrams = frozenset(trigrams)
        self._by_prefix = by_prefix

    def contains_prompt(self, prompt: str) -> bool:
        return _normalise_space(prompt).casefold() in self.normalized_texts

    def copied_sentences(self, response: str, limit: int = 5) -> list[str]:
        """Return corpus sentences copied verbatim inside *response*.

        Matching is word-token based, so punctuation and whitespace variants do
        not hide a copy. Only sentences with at least
        ``MIN_COPIED_SENTENCE_TOKENS`` tokens are considered.
        """
        tokens = word_tokens(response)
        found: list[str] = []
        seen: set[str] = set()
        for index in range(max(0, len(tokens) - MIN_COPIED_SENTENCE_TOKENS + 1)):
            prefix = tuple(tokens[index:index + MIN_COPIED_SENTENCE_TOKENS])
            for sentence_tokens, sentence in self._by_prefix.get(prefix, ()):
                end = index + len(sentence_tokens)
                if end > len(tokens):
                    continue
                if tuple(tokens[index:end]) != sentence_tokens:
                    continue
                if sentence not in seen:
                    seen.add(sentence)
                    found.append(sentence)
                    if len(found) >= limit:
                        return found
        return found


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_tokens(text: str) -> list[str]:
    """Case-folded word/number tokens used by the deterministic metrics."""
    return [match.group(0).casefold()
            for match in WORD_RE.finditer(str(text or ""))]


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index:index + n])
            for index in range(max(0, len(tokens) - n + 1))]


def _term_present(text: str, term: str) -> bool:
    needle = _normalise_space(term).casefold()
    if not needle:
        return False
    pattern = (r"(?<![^\W_])" + re.escape(needle) + r"(?![^\W_])")
    return re.search(pattern, str(text or "").casefold(), flags=re.UNICODE) is not None


def normalise_keyword_groups(value: Any) -> list[list[str]]:
    """Normalise ``keywords``/``keyword_groups``/``expected`` input.

    A plain list of strings becomes one group per string.  A list of lists (or
    tuples) keeps each inner sequence as alternatives for one group.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("keyword groups must be a list or tuple")
    groups: list[list[str]] = []
    for item in value:
        if isinstance(item, str):
            alternatives = [item]
        elif isinstance(item, (list, tuple)):
            alternatives = [str(entry) for entry in item]
        else:
            raise ValueError("each keyword group must be a string or list")
        cleaned = [text.strip() for text in alternatives if text.strip()]
        if cleaned:
            groups.append(cleaned)
    return groups


def _keyword_metric(response: str, groups: Sequence[Sequence[str]]) -> dict:
    matched: list[list[str]] = []
    missing: list[list[str]] = []
    for group in groups:
        if any(_term_present(response, term) for term in group):
            matched.append(list(group))
        else:
            missing.append(list(group))
    total = len(groups)
    return {
        "keyword_coverage": round(len(matched) / total, 4) if total else None,
        "keywords_matched": len(matched),
        "keywords_total": total,
        "missing_keywords": missing,
    }


def _prompt_public(prompt: dict) -> dict:
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "keyword_groups": [list(group) for group in prompt["keyword_groups"]],
    }


def _prompt_from_value(value: Any, index: int) -> dict:
    if isinstance(value, str):
        prompt_text = value.strip()
        identifier = "prompt-%03d" % index
        category = "custom"
        groups: list[list[str]] = []
    elif isinstance(value, dict):
        prompt_text = str(
            value.get("prompt") or value.get("question") or
            value.get("message") or "").strip()
        identifier = str(value.get("id") or "prompt-%03d" % index).strip()
        category = str(value.get("category") or value.get("kind") or
                       "custom").strip() or "custom"
        if "keyword_groups" in value:
            groups = normalise_keyword_groups(value.get("keyword_groups"))
        elif "keywords" in value:
            groups = normalise_keyword_groups(value.get("keywords"))
        elif "expected" in value:
            groups = normalise_keyword_groups(value.get("expected"))
        else:
            groups = []
    else:
        raise ValueError("prompt entry must be a string or JSON object")
    if not prompt_text:
        raise ValueError("prompt entry has no prompt/question/message text")
    if not identifier:
        identifier = "prompt-%03d" % index
    return {
        "id": identifier,
        "category": category,
        "prompt": prompt_text,
        "keyword_groups": groups,
    }


def _prompt_meta(prompts: Sequence[dict], source: str) -> dict:
    payload = json.dumps([_prompt_public(prompt) for prompt in prompts],
                         sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return {
        "source": source,
        "count": len(prompts),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _json_values_from_line(line: str) -> list[Any]:
    """Decode one or more concatenated JSON values from a JSONL line."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    position = 0
    while position < len(line):
        while position < len(line) and line[position] in " \t\r\n":
            position += 1
        if position >= len(line):
            break
        value, end = decoder.raw_decode(line, position)
        values.append(value)
        position = end
    return values


def load_prompts(path: str | Path | None = None) -> tuple[list[dict], dict]:
    """Load built-in prompts or a JSONL/JSON probe file.

    JSONL entries may be plain strings or objects with ``prompt`` (or
    ``question``/``message``), optional ``id``/``category`` and optional
    ``keywords``, ``keyword_groups`` or ``expected``.
    """
    if path is None or str(path).strip() == "":
        prompts = [_prompt_from_value(dict(item), index)
                   for index, item in enumerate(BUILTIN_PROMPTS, start=1)]
        return prompts, _prompt_meta(prompts, "builtin")

    target = Path(str(path)).expanduser()
    if not target.is_file():
        raise ValueError("prompt file does not exist: " + str(target))
    suffix = target.suffix.lower()
    values: list[Any] = []
    if suffix == ".jsonl":
        for line_number, line in enumerate(
                target.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.extend(_json_values_from_line(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: invalid JSON: %s" %
                                 (target, line_number, exc)) from exc
    elif suffix == ".json":
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("%s: invalid JSON: %s" % (target, exc)) from exc
        if isinstance(loaded, dict):
            for key in ("prompts", "probes", "rows", "items"):
                if isinstance(loaded.get(key), list):
                    loaded = loaded[key]
                    break
        if not isinstance(loaded, list):
            raise ValueError("prompt JSON must be a list or contain a prompt list")
        values = loaded
    else:
        raise ValueError("prompt file must end in .jsonl or .json")

    if not values:
        raise ValueError("prompt file contains no prompts: " + str(target))
    prompts = []
    used_ids: set[str] = set()
    for index, value in enumerate(values, start=1):
        try:
            prompt = _prompt_from_value(value, index)
        except ValueError as exc:
            raise ValueError("%s: prompt %d: %s" %
                             (target, index, exc)) from exc
        if prompt["id"] in used_ids:
            raise ValueError("%s: duplicate prompt id %r" %
                             (target, prompt["id"]))
        used_ids.add(prompt["id"])
        prompts.append(prompt)
    return prompts, _prompt_meta(prompts, str(target))


def _collect_corpus_texts(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            output.append(text)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_corpus_texts(item, output)
        return
    if not isinstance(value, dict):
        return
    for key in ("text", "sentence", "statement", "content"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            output.append(text.strip())
            return
    for key in ("sentences", "statements", "texts", "records", "rows",
                "train", "data", "rounds", "baseline"):
        if key in value:
            _collect_corpus_texts(value[key], output)


def _corpus_file_texts(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    texts: list[str] = []
    if suffix == ".jsonl":
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1):
            if not line.strip():
                continue
            try:
                values = _json_values_from_line(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: invalid JSON: %s" %
                                 (path, line_number, exc)) from exc
            for value in values:
                _collect_corpus_texts(value, texts)
    elif suffix == ".json":
        try:
            value = json.loads(
                path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ValueError("%s: invalid JSON: %s" % (path, exc)) from exc
        _collect_corpus_texts(value, texts)
    else:
        for line in path.read_text(
                encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if text:
                texts.append(text)
    return texts


def load_corpus(path: str | Path | None = None) -> tuple[CorpusReference | None, dict]:
    """Load a text/JSONL/JSON corpus used for copy and novelty references."""
    if path is None or str(path).strip() == "":
        return None, {"provided": False}
    target = Path(str(path)).expanduser()
    if target.is_dir():
        files = sorted(
            item for item in target.rglob("*")
            if item.is_file() and item.suffix.lower() in
            {".txt", ".jsonl", ".json", ".md"})
    elif target.is_file():
        files = [target]
    else:
        raise ValueError("corpus path does not exist: " + str(target))
    if not files:
        raise ValueError("corpus contains no .txt/.jsonl/.json/.md files: "
                         + str(target))

    collected: list[str] = []
    for item in files:
        collected.extend(_corpus_file_texts(item))
    reference = CorpusReference(collected)
    if not reference.texts:
        raise ValueError("corpus contains no text: " + str(target))
    texts = list(reference.texts)
    token_count = sum(len(word_tokens(text)) for text in texts)
    payload = "\n".join(text.casefold() for text in texts)
    meta = {
        "provided": True,
        "path": str(target),
        "files": len(files),
        "sentences": len(texts),
        "tokens": token_count,
        "characters": sum(len(text) for text in texts),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "min_copied_sentence_tokens": MIN_COPIED_SENTENCE_TOKENS,
    }
    return reference, meta


def score_next_token_nll(model: Any, prompt: str, response: str) -> dict | None:
    """Score response tokens with the native model's own next-token predictor.

    The prompt is used as read-only context.  This function only calls
    ``_prob``; it never calls ``observe`` or ``save``.
    """
    if model is None:
        return None
    response_tokens = lm_tokens(response)
    if not response_tokens:
        return {
            "tokens_scored": 0,
            "negative_log_likelihood": 0.0,
            "mean_negative_log_likelihood": None,
            "perplexity": None,
        }
    prompt_tokens = lm_tokens(prompt)
    sequence = [BOS] + prompt_tokens + response_tokens
    start = 1 + len(prompt_tokens)
    total = 0.0
    count = 0
    for index in range(start, len(sequence)):
        context1 = sequence[index - 1]
        context2 = sequence[index - 2] if index >= 2 else None
        target = sequence[index]
        try:
            probability = float(model._prob(context2, context1, target))
        except Exception:
            probability = 0.0
        total += -math.log(max(probability, 1e-9))
        count += 1
    mean_nll = total / count if count else None
    return {
        "tokens_scored": count,
        "negative_log_likelihood": round(total, 6),
        "mean_negative_log_likelihood": (
            round(mean_nll, 6) if mean_nll is not None else None),
        "perplexity": (round(math.exp(min(mean_nll, 50)), 4)
                       if mean_nll is not None else None),
    }


def measure_response(response: str, prompt: str,
                     keyword_groups: Sequence[Sequence[str]] | None = None,
                     corpus: CorpusReference | None = None,
                     nll_model: Any = None) -> dict:
    """Return all deterministic per-reply surface metrics."""
    text = str(response or "")
    tokens = word_tokens(text)
    counts = Counter(tokens)
    response_trigrams = set(_ngrams(tokens, 3))
    prompt_trigrams = set(_ngrams(word_tokens(prompt), 3))
    reference_trigrams = set(prompt_trigrams)
    if corpus is not None:
        reference_trigrams |= set(corpus.trigrams)
    novel = response_trigrams - reference_trigrams
    trigram_counts = Counter(_ngrams(tokens, 3))
    keyword = _keyword_metric(
        text, normalise_keyword_groups(keyword_groups or []))

    if corpus is None:
        copied: bool | None = None
        copied_sentences: list[str] | None = None
        novel_reference = "prompt_only"
        prompt_in_corpus: bool | None = None
    else:
        copied_sentences = corpus.copied_sentences(text)
        copied = bool(copied_sentences)
        novel_reference = "corpus+prompt"
        prompt_in_corpus = corpus.contains_prompt(prompt)

    unique = len(set(tokens))
    return {
        "characters": len(text),
        "tokens": len(tokens),
        "unique_tokens": unique,
        "unique_token_ratio": (round(unique / len(tokens), 4)
                               if tokens else 0.0),
        "repeated_token_ratio": (round(1.0 - unique / len(tokens), 4)
                                 if tokens else 0.0),
        "max_token_repetition": max(counts.values()) if counts else 0,
        "response_3grams": len(response_trigrams),
        "max_trigram_repetition": (max(trigram_counts.values())
                                   if trigram_counts else 0),
        "novel_3grams": len(novel),
        "novel_3gram_ratio": (round(len(novel) / len(response_trigrams), 4)
                              if response_trigrams else 0.0),
        "novel_reference": novel_reference,
        "copied_training_sentence": copied,
        "copied_training_sentences": copied_sentences,
        "empty": not bool(text.strip()),
        "prompt_in_corpus": prompt_in_corpus,
        "next_token_nll": score_next_token_nll(nll_model, prompt, text),
        **keyword,
    }


def _judge_error(message: str, raw: Any = None) -> dict:
    result = {"error": message}
    if raw is not None:
        text = str(raw)
        result["raw"] = text[:MAX_JUDGE_RAW]
    return result


def normalise_judge_object(value: Any) -> dict:
    """Validate a judge rubric object and coerce binary fields to bool.

    Invalid or partial rubric objects become ``{"error": ...}`` so a broken
    judge never silently becomes an aggregate score.
    """
    if not isinstance(value, dict):
        return _judge_error("judge result is not a JSON object")
    missing = [key for key in JUDGE_REQUIRED_KEYS if key not in value]
    if missing:
        return _judge_error("judge rubric missing keys: " + ", ".join(missing),
                            value)
    result: dict[str, Any] = {}
    for key in JUDGE_BINARY_KEYS:
        raw = value[key]
        if isinstance(raw, bool):
            result[key] = raw
        elif isinstance(raw, (int, float)) and raw in (0, 1):
            result[key] = bool(raw)
        elif isinstance(raw, str) and raw.strip().casefold() in {
                "0", "1", "true", "false", "yes", "no"}:
            result[key] = raw.strip().casefold() in {"1", "true", "yes"}
        else:
            return _judge_error(
                "judge key %s is not binary: %r" % (key, raw), value)
    notes = value.get("notes")
    result["notes"] = "" if notes is None else str(notes)[:2000]
    return result


def parse_judge_content(content: str) -> dict:
    """Parse a strict JSON rubric from a judge message, including fenced JSON."""
    text = str(content or "").strip()
    if not text:
        return _judge_error("judge returned no content")
    value: Any = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return _judge_error("judge returned no JSON", text)
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            return _judge_error("judge returned invalid JSON: %s" % exc, text)
    return normalise_judge_object(value)


class OllamaJudge:
    """Optional Gemma judge over an Ollama OpenAI-compatible endpoint."""

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 timeout: float = 180.0):
        self.url = url
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str, response: str,
                 references: Iterable[str] | None = None) -> dict:
        user: dict[str, Any] = {"prompt": prompt, "reply": response}
        excerpts = [str(item) for item in (references or [])
                    if str(item).strip()][:JUDGE_REFERENCE_LIMIT]
        if excerpts:
            user["reference_excerpts"] = excerpts
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(
                    user, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            value = json.loads(reply.read(262144))
        try:
            content = value["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            return _judge_error("judge response has no message content: %s"
                                % exc, value)
        return parse_judge_content(content)


def _decode_tool_result(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class NativeChatStudent:
    """Call the registered ``nanlm_chat`` tool with ``probe=True``."""

    interface = "nanlm_chat"

    def __init__(self, state_dir: str | Path,
                 session_id: str = DEFAULT_SESSION_ID):
        self.state_dir = Path(state_dir).expanduser()
        self.session_id = session_id
        self.ctx = EvalContext(self.state_dir)
        register(self.ctx)
        handler = self.ctx.tools.get("nanlm_chat")
        if handler is None:
            raise RuntimeError("nanlm_chat was not registered by this build")
        self._handler = handler

    def reply(self, prompt: str) -> dict:
        result = _decode_tool_result(self._handler({
            "message": prompt,
            "session_id": self.session_id,
            "probe": True,
        }))
        if not isinstance(result, dict):
            raise TypeError("nanlm_chat returned a non-object result")
        return result

    __call__ = reply


def load_nll_model(state_dir: str | Path) -> tuple[Any | None, str | None]:
    """Load ``TemporalMirrorLM`` for scoring without calling ``save``."""
    root = Path(state_dir).expanduser()
    ctx = EvalContext(root)
    path = TemporalMirrorLM.path(ctx)
    if not path.is_file():
        return None, "temporal mirror state not found: " + str(path)
    try:
        return TemporalMirrorLM.load(ctx), None
    except Exception as exc:
        return None, type(exc).__name__ + ": " + str(exc)[:500]


def _student_meta(result: Any, error: str | None) -> dict:
    data = result if isinstance(result, dict) else {}
    synthesis = data.get("native_synthesis")
    synthesis = synthesis if isinstance(synthesis, dict) else {}
    frame = synthesis.get("temporal_frame")
    frame = frame if isinstance(frame, dict) else {}
    meta = {
        "interface": data.get("interface"),
        "provider": data.get("provider"),
        "model": data.get("model"),
        "probe": data.get("probe"),
        "state_mutated": data.get("state_mutated"),
        "synthesis": synthesis.get("synthesis"),
        "response_templates": synthesis.get("response_templates"),
        "retrieval": synthesis.get("retrieval"),
        "structured": synthesis.get("structured"),
        "temporal_frame": {
            key: frame.get(key)
            for key in ("anchor", "steps", "score", "coverage",
                        "output_drive", "mirror_state")
        } if frame else None,
        "error": error,
    }
    return meta


def _reply_from_student(student: Any, prompt: str) -> Any:
    if hasattr(student, "reply"):
        return student.reply(prompt)
    if callable(student):
        return student(prompt)
    raise TypeError("student must be callable or expose reply(prompt)")


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _category_summary(rows: Sequence[dict], category: str,
                      judge_enabled: bool) -> dict:
    subset = [row for row in rows if row["category"] == category]
    metrics = [row["metrics"] for row in subset]
    judged = [row["judge"] for row in subset
              if isinstance(row.get("judge"), dict)
              and "error" not in row["judge"]]
    result = {
        "prompts": len(subset),
        "empty_responses": sum(1 for item in metrics if item["empty"]),
        "empty_rate": _rate(
            sum(1 for item in metrics if item["empty"]), len(metrics)),
        "mean_tokens": _mean(item["tokens"] for item in metrics) or 0.0,
        "mean_unique_token_ratio": (
            _mean(item["unique_token_ratio"] for item in metrics) or 0.0),
        "mean_novel_3gram_ratio": (
            _mean(item["novel_3gram_ratio"] for item in metrics) or 0.0),
        "mean_keyword_coverage": _mean(
            item["keyword_coverage"] for item in metrics
            if item["keywords_total"]) if any(
                item["keywords_total"] for item in metrics) else None,
    }
    if judge_enabled:
        result["judge"] = {
            "judged": len(judged),
            "errors": sum(1 for row in subset
                          if isinstance(row.get("judge"), dict)
                          and "error" in row["judge"]),
            "answers_prompt": sum(
                1 for item in judged if item.get("answers_prompt")),
            "coherent": sum(1 for item in judged if item.get("coherent")),
            "novel_combination": sum(
                1 for item in judged if item.get("novel_combination")),
            "unsupported_claim": sum(
                1 for item in judged if item.get("unsupported_claim")),
        }
    return result


def summarize_rows(rows: Sequence[dict], *, corpus_provided: bool,
                   judge_enabled: bool, nll_requested: bool,
                   nll_error: str | None = None) -> dict:
    """Aggregate deterministic metrics and judge counts over all rows."""
    metrics = [row["metrics"] for row in rows]
    total = len(metrics)
    empty = sum(1 for item in metrics if item["empty"])
    novelty = [item["novel_3gram_ratio"] for item in metrics]
    response_3grams = sum(item["response_3grams"] for item in metrics)
    novel_3grams = sum(item["novel_3grams"] for item in metrics)
    keyword_rows = [item for item in metrics if item["keywords_total"]]
    keyword_coverage = [item["keyword_coverage"] for item in keyword_rows
                        if item["keyword_coverage"] is not None]
    fully_covered = sum(1 for value in keyword_coverage if value >= 1.0)
    copied_count = (sum(1 for item in metrics
                        if item["copied_training_sentence"])
                    if corpus_provided else None)

    nll_rows = [item["next_token_nll"] for item in metrics
                if item.get("next_token_nll")]
    nll_tokens = sum(item["tokens_scored"] for item in nll_rows)
    nll_total = sum(item["negative_log_likelihood"] for item in nll_rows)
    nll_mean = (round(nll_total / nll_tokens, 6)
                if nll_tokens else None)
    nll_ppl = (round(math.exp(min(nll_mean, 50)), 4)
               if nll_mean is not None else None)

    judged = [row["judge"] for row in rows
              if isinstance(row.get("judge"), dict)
              and "error" not in row["judge"]]
    judge_summary = None
    if judge_enabled:
        judge_summary = {
            "judged": len(judged),
            "errors": sum(1 for row in rows
                          if isinstance(row.get("judge"), dict)
                          and "error" in row["judge"]),
        }
        for key in JUDGE_BINARY_KEYS:
            count = sum(1 for item in judged if item.get(key))
            judge_summary[key] = count
            judge_summary[key + "_rate"] = _rate(count, len(judged))

    categories = sorted({row["category"] for row in rows})
    return {
        "prompts": total,
        "student_errors": sum(
            1 for row in rows
            if (row.get("student") or {}).get("error")),
        "empty_responses": empty,
        "empty_rate": _rate(empty, total),
        "mean_tokens": _mean(item["tokens"] for item in metrics) or 0.0,
        "mean_characters": _mean(item["characters"] for item in metrics) or 0.0,
        "mean_unique_tokens": (
            _mean(item["unique_tokens"] for item in metrics) or 0.0),
        "mean_unique_token_ratio": (
            _mean(item["unique_token_ratio"] for item in metrics) or 0.0),
        "mean_repeated_token_ratio": (
            _mean(item["repeated_token_ratio"] for item in metrics) or 0.0),
        "mean_max_token_repetition": (
            _mean(item["max_token_repetition"] for item in metrics) or 0.0),
        "max_token_repetition": (max(
            (item["max_token_repetition"] for item in metrics), default=0)),
        "mean_novel_3gram_ratio": _mean(novelty) or 0.0,
        "novel_3gram_ratio_micro": (
            round(novel_3grams / response_3grams, 4)
            if response_3grams else 0.0),
        "copied_training_sentences": copied_count,
        "copied_training_sentence_rate": (
            _rate(copied_count, total) if copied_count is not None else None),
        "keyword_prompts": len(keyword_rows),
        "mean_keyword_coverage": _mean(keyword_coverage),
        "fully_covered_prompts": fully_covered,
        "full_keyword_coverage_rate": _rate(
            fully_covered, len(keyword_coverage)),
        "prompts_in_corpus": (
            sum(1 for row in rows if row.get("prompt_in_corpus"))
            if corpus_provided else None),
        "probe_read_only_rows": sum(
            1 for row in rows
            if (row.get("student") or {}).get("probe") is True
            and (row.get("student") or {}).get("state_mutated") is False),
        "judge": judge_summary,
        "nll": {
            "requested": bool(nll_requested),
            "available": nll_tokens > 0,
            "prompts_scored": sum(1 for item in nll_rows
                                  if item["tokens_scored"]),
            "tokens_scored": nll_tokens,
            "mean_negative_log_likelihood": nll_mean,
            "perplexity": nll_ppl,
            "error": nll_error,
        },
        "by_category": {
            category: _category_summary(rows, category, judge_enabled)
            for category in categories
        },
    }


def _state_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in (
            "universe/mathematical_memory.sqlite3",
            "language/temporal_mirror.json",
            "language/layered_learning.json",
            "language/temporal_register.json",
            "language/mth_heads.json",
    ):
        path = root / relative
        if path.is_file():
            files.append(path)
    for pattern in ("conversations/*.jsonl",
                    "scratch_register/**/page-*.n16",
                    "universe/native/*.nmg"):
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def _state_fingerprint(root: Path) -> dict:
    entries: dict[str, dict] = {}
    for path in _state_files(root):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        entries[str(path.relative_to(root))] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": len(entries),
        "entries": entries,
    }


def _state_metadata(root: Path) -> dict:
    temporal_path = root / "language" / "temporal_mirror.json"
    temporal: dict[str, Any] = {
        "path": str(temporal_path),
        "exists": temporal_path.is_file(),
    }
    if temporal_path.is_file():
        temporal["bytes"] = temporal_path.stat().st_size
        try:
            data = json.loads(temporal_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                temporal.update({
                    "schema": data.get("schema"),
                    "vocabulary": len(data.get("vocab") or {}),
                    "exposed_sentences": len(data.get("seen") or []),
                    "bigram_contexts": len(data.get("bi") or {}),
                    "trigram_contexts": len(data.get("tri") or {}),
                })
        except (OSError, ValueError):
            temporal["parse_error"] = True

    layered_path = root / "language" / "layered_learning.json"
    layered: dict[str, Any] = {
        "path": str(layered_path),
        "exists": layered_path.is_file(),
    }
    if layered_path.is_file():
        layered["bytes"] = layered_path.stat().st_size
        try:
            data = json.loads(layered_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                layered["turns"] = data.get("turns")
                layered["rows"] = len(data.get("rows") or [])
        except (OSError, ValueError):
            layered["parse_error"] = True

    memory_path = root / "universe" / "mathematical_memory.sqlite3"
    return {
        "path": str(root),
        "temporal_mirror": temporal,
        "layered_learning": layered,
        "mathematical_memory": {
            "path": str(memory_path),
            "exists": memory_path.is_file(),
            "bytes": memory_path.stat().st_size
            if memory_path.is_file() else None,
        },
    }


def run_evaluation(state_dir: str | Path, prompts: Sequence[dict], *,
                   student: Any,
                   corpus: CorpusReference | None = None,
                   judge: Callable[[str, str, list[str]], dict] | None = None,
                   nll_model: Any = None,
                   nll_requested: bool = False,
                   nll_error: str | None = None,
                   prompt_meta: dict | None = None,
                   corpus_meta: dict | None = None,
                   judge_meta: dict | None = None) -> dict:
    """Run the student over every prompt and assemble the report.

    ``student`` is any callable/object returning a result dict with a
    ``response`` key.  ``judge`` receives ``(prompt, response, references)`` and
    is never used to produce the response.
    """
    root = Path(state_dir).expanduser()
    if not root.is_dir():
        raise ValueError("state dir does not exist: " + str(root))
    prompts = [_prompt_from_value(prompt, index)
               for index, prompt in enumerate(list(prompts), start=1)]
    if not prompts:
        raise ValueError("at least one prompt is required")

    state_before = _state_fingerprint(root)
    rows: list[dict] = []
    judge_references = (list(corpus.texts)
                        if corpus is not None else [])
    for index, prompt in enumerate(prompts, start=1):
        student_error: str | None = None
        result: Any = {}
        try:
            result = _reply_from_student(student, prompt["prompt"])
            if isinstance(result, str):
                result = {"response": result}
            if not isinstance(result, dict):
                raise TypeError("student result must be an object or string")
            response = str(result.get("response") or "")
        except Exception as exc:
            student_error = type(exc).__name__ + ": " + str(exc)[:1000]
            result = {}
            response = ""

        metrics = measure_response(
            response, prompt["prompt"],
            keyword_groups=prompt.get("keyword_groups") or [],
            corpus=corpus, nll_model=nll_model)
        student_data = _student_meta(result, student_error)
        student_data["interface"] = (
            getattr(student, "interface", None) or
            student_data.get("interface") or "custom")

        judge_result = None
        if judge is not None:
            if student_error is not None:
                judge_result = {
                    "error": "judge skipped because the student call failed",
                }
            else:
                try:
                    judge_result = normalise_judge_object(
                        judge(prompt["prompt"], response, judge_references))
                except Exception as exc:
                    judge_result = {
                        "error": "judge failed: %s: %s" % (
                            type(exc).__name__, str(exc)[:1000]),
                    }

        rows.append({
            "index": index,
            "id": prompt["id"],
            "category": prompt["category"],
            "prompt": prompt["prompt"],
            "keyword_groups": [list(group)
                               for group in prompt.get("keyword_groups") or []],
            "response": response,
            "prompt_in_corpus": metrics.get("prompt_in_corpus"),
            "student": student_data,
            "metrics": metrics,
            "judge": judge_result,
        })

    state_after = _state_fingerprint(root)
    changed_files = sorted(
        path for path in
        set(state_before["entries"]) | set(state_after["entries"])
        if state_before["entries"].get(path) != state_after["entries"].get(path))
    judge_enabled = judge is not None
    if judge_meta is None:
        judge_meta = (
            {
                "enabled": True,
                "url": getattr(judge, "url", None),
                "model": getattr(judge, "model", None),
                "rubric": list(JUDGE_REQUIRED_KEYS),
                "diagnostic_only": True,
                "ground_truth": False,
            } if judge_enabled else
            {"enabled": False, "diagnostic_only": True, "ground_truth": False}
        )

    summary = summarize_rows(
        rows, corpus_provided=corpus is not None,
        judge_enabled=judge_enabled, nll_requested=nll_requested,
        nll_error=nll_error)
    state_unchanged = (
        state_before["sha256"] == state_after["sha256"])
    boundaries = {
        "native_student_interface": getattr(student, "interface", "custom"),
        "llm_wrote_student_reply": False,
        "judge_is_diagnostic_only": True,
        "judge_is_ground_truth": False,
        "probe_read_only_requested": bool(
            getattr(student, "interface", None) == "nanlm_chat"),
        "probe_receipts_read_only": (
            summary["probe_read_only_rows"] == len(rows)),
        "state_fingerprint_unchanged": state_unchanged,
        "held_out_proven": False,
        "training_performed": False,
        "canonical_commit": False,
        "proof_claim": False,
        "notes": list(BOUNDARY_NOTES),
    }
    return {
        "schema": SCHEMA,
        "state_dir": str(root),
        "state": _state_metadata(root),
        "prompts": prompt_meta or {
            "source": "provided",
            "count": len(prompts),
            "sha256": _prompt_meta(prompts, "provided")["sha256"],
        },
        "corpus": corpus_meta or {
            "provided": corpus is not None,
        },
        "judge": judge_meta,
        "nll": {
            "requested": bool(nll_requested),
            "model": ("temporal_mirror_lm/1" if nll_model is not None else None),
            "error": nll_error,
        },
        "state_probe": {
            "fingerprint_before": state_before["sha256"],
            "fingerprint_after": state_after["sha256"],
            "files_checked": state_before["files"],
            "unchanged": state_unchanged,
            "changed_files": changed_files,
        },
        "rows": rows,
        "summary": summary,
        "boundaries": boundaries,
    }


def _json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _summary_table_lines(report: dict) -> list[str]:
    summary = report.get("summary") or {}
    lines = [
        "| metric | value |",
        "|---|---:|",
        "| prompts | %s |" % summary.get("prompts", 0),
        "| empty replies | %s (%.4f) |" % (
            summary.get("empty_responses", 0),
            summary.get("empty_rate", 0.0)),
        "| mean tokens | %s |" % summary.get("mean_tokens", 0.0),
        "| mean characters | %s |" % summary.get("mean_characters", 0.0),
        "| mean unique-token ratio | %s |" % (
            summary.get("mean_unique_token_ratio", 0.0)),
        "| mean repeated-token ratio | %s |" % (
            summary.get("mean_repeated_token_ratio", 0.0)),
        "| max token repetition | %s |" % (
            summary.get("max_token_repetition", 0)),
        "| mean novel 3-gram ratio | %s |" % (
            summary.get("mean_novel_3gram_ratio", 0.0)),
        "| novel 3-gram ratio (micro) | %s |" % (
            summary.get("novel_3gram_ratio_micro", 0.0)),
        "| mean keyword coverage | %s |" % (
            summary.get("mean_keyword_coverage")),
        "| copied training sentences | %s |" % (
            summary.get("copied_training_sentences")),
        "| probe read-only rows | %s |" % (
            summary.get("probe_read_only_rows", 0)),
    ]
    nll = summary.get("nll") or {}
    if nll.get("requested"):
        lines.append("| next-token NLL (mean) | %s |" % (
            nll.get("mean_negative_log_likelihood")))
        lines.append("| next-token perplexity | %s |" % nll.get("perplexity"))
    judge = summary.get("judge")
    if judge:
        lines.append("| judged replies | %s |" % judge.get("judged", 0))
        for key in JUDGE_BINARY_KEYS:
            lines.append("| judge %s | %s |" % (
                key, judge.get(key, 0)))
        lines.append("| judge errors | %s |" % judge.get("errors", 0))
    return lines


def _category_table_lines(report: dict) -> list[str]:
    summary = report.get("summary") or {}
    categories = summary.get("by_category") or {}
    lines = [
        "| category | prompts | empty | mean tokens | mean novel 3-gram | "
        "mean keyword coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category in sorted(categories):
        item = categories[category]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            category, item.get("prompts", 0),
            item.get("empty_responses", 0),
            item.get("mean_tokens", 0.0),
            item.get("mean_novel_3gram_ratio", 0.0),
            item.get("mean_keyword_coverage")))
    return lines


def render_markdown(report: dict) -> str:
    state = report.get("state") or {}
    temporal = state.get("temporal_mirror") or {}
    layered = state.get("layered_learning") or {}
    prompts = report.get("prompts") or {}
    corpus = report.get("corpus") or {}
    judge = report.get("judge") or {}
    nll = report.get("nll") or {}
    state_probe = report.get("state_probe") or {}

    lines = [
        "# corpus generation evaluation",
        "",
        "- schema: `%s`" % report.get("schema", SCHEMA),
        "- state dir: `%s`" % report.get("state_dir", ""),
        "- student interface: `%s` with `probe=True`" % (
            (report.get("boundaries") or {}).get(
                "native_student_interface", "custom")),
        "- prompts: %s from `%s`" % (
            prompts.get("count", 0), prompts.get("source", "unknown")),
        "- corpus: %s" % (
            "`%s` (%s sentences)" % (
                corpus.get("path"), corpus.get("sentences"))
            if corpus.get("provided") else "not supplied"),
        "- judge: %s" % (
            "`%s` at `%s` (diagnostic only)" % (
                judge.get("model"), judge.get("url"))
            if judge.get("enabled") else "not requested"),
        "- NLL/perplexity: %s" % (
            "requested (%s)" % (nll.get("error") or "available")
            if nll.get("requested") else "not requested"),
        "",
    ]
    if temporal.get("exists"):
        lines.append(
            "- temporal mirror: schema `%s`, vocabulary %s, exposed "
            "sentences %s, bigram contexts %s, trigram contexts %s" % (
                temporal.get("schema"), temporal.get("vocabulary"),
                temporal.get("exposed_sentences"),
                temporal.get("bigram_contexts"),
                temporal.get("trigram_contexts")))
    else:
        lines.append("- temporal mirror: **missing** at `%s`" % (
            temporal.get("path", "")))
    if layered.get("exists"):
        lines.append("- layered state: turns %s, rows %s" % (
            layered.get("turns"), layered.get("rows")))
    else:
        lines.append("- layered state: **missing** at `%s`" % (
            layered.get("path", "")))
    lines.extend([
        "- state probe fingerprint unchanged: %s (files checked: %s)" % (
            state_probe.get("unchanged"), state_probe.get("files_checked")),
        "",
        "## Aggregate summary",
        "",
    ])
    lines.extend(_summary_table_lines(report))
    lines.extend([
        "",
        "## Per-category summary",
        "",
    ])
    lines.extend(_category_table_lines(report))
    lines.extend([
        "",
        "## Per-prompt results",
        "",
    ])
    for row in report.get("rows") or []:
        lines.append("### %s — %s" % (row.get("id"), row.get("category")))
        lines.append("")
        lines.append("**Prompt:** %s" % row.get("prompt", ""))
        lines.append("")
        response = str(row.get("response") or "")
        if response.strip():
            lines.extend("> " + line for line in response.splitlines())
        else:
            lines.append("> (empty)")
        lines.append("")
        lines.append("- metrics: `%s`" % _json_line(row.get("metrics") or {}))
        lines.append("- student: `%s`" % _json_line(row.get("student") or {}))
        if row.get("prompt_in_corpus") is not None:
            lines.append("- exact prompt found in corpus: %s" % (
                row.get("prompt_in_corpus")))
        if row.get("judge") is not None:
            lines.append("- judge: `%s`" % _json_line(row.get("judge")))
        else:
            lines.append("- judge: not requested")
        lines.append("")

    lines.extend([
        "## Boundaries and interpretation",
        "",
    ])
    for note in (report.get("boundaries") or {}).get("notes") or BOUNDARY_NOTES:
        lines.append("- " + note)
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict, output: str | Path) -> tuple[Path, Path]:
    """Write the JSON report and sibling Markdown report atomically."""
    target = Path(str(output)).expanduser()
    if target.suffix.lower() == ".md":
        json_path = target.with_suffix(".json")
        markdown_path = target
    else:
        json_path = target
        markdown_path = target.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True,
                   ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    temporary.replace(json_path)

    temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary.write_text(render_markdown(report) + "\n", encoding="utf-8")
    temporary.replace(markdown_path)
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True,
                        help="trained NaNLM state directory")
    parser.add_argument("--prompts", default="",
                        help="JSONL/JSON probe file (default: built-in probes)")
    parser.add_argument("--corpus", default="",
                        help="training corpus file/directory for copy and "
                             "novelty references")
    parser.add_argument("--judge", action="store_true",
                        help="ask the optional Gemma judge for rubric labels")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="JSON report path (Markdown is written beside it)")
    parser.add_argument("--url", default=os.environ.get(
                            "NANLM_REVIEW_URL", DEFAULT_URL),
                        help="Ollama OpenAI-compatible chat completions URL")
    parser.add_argument("--model", default=os.environ.get(
                            "NANLM_REVIEW_MODEL", DEFAULT_MODEL),
                        help="judge model name")
    parser.add_argument("--nll", action="store_true",
                        help="also score replies with TemporalMirrorLM "
                             "next-token NLL/perplexity (read-only)")
    args = parser.parse_args(argv)

    root = Path(args.state_dir).expanduser()
    if not root.is_dir():
        parser.error("state dir does not exist: " + str(root))
    try:
        prompts, prompt_meta = load_prompts(args.prompts or None)
        corpus, corpus_meta = load_corpus(args.corpus or None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        student = NativeChatStudent(root)
    except Exception as exc:
        raise SystemExit("could not start native student: %s: %s" % (
            type(exc).__name__, exc)) from exc

    nll_model = None
    nll_error = None
    if args.nll:
        nll_model, nll_error = load_nll_model(root)

    judge = OllamaJudge(args.url, args.model) if args.judge else None
    report = run_evaluation(
        root, prompts, student=student, corpus=corpus, judge=judge,
        nll_model=nll_model, nll_requested=args.nll, nll_error=nll_error,
        prompt_meta=prompt_meta, corpus_meta=corpus_meta)
    json_path, markdown_path = write_report(report, args.output)
    print(json.dumps({
        "output": str(json_path),
        "markdown": str(markdown_path),
        "summary": report["summary"],
    }, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
