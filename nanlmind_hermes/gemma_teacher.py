"""Provider-universal LLM teacher for the HERMES layered learner.

Gemma is only one possible teacher.  The same bounded record contract works
with Hermes' configured provider, a local Ollama model, or any
OpenAI-compatible chat endpoint.  Teacher records are evidence, never
canonical truth; promotion remains the existing conservative gate.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from .layered_learning import LAYER_NAMES, LayeredLearningCoordinator

SCHEMA = "nanlm.llm-teacher/1"
MAX_RECORDS = 64
MAX_RESPONSE_BYTES = 262_144
MAX_TEXT = 512
TEACHER_PURPOSE = "nanlmind-relation-teacher-hermes"
_ALLOWED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _'.,;:/()\[\]{}#&%+\-=]{0,511}$")


class LLMTeacherError(ValueError):
    pass


# Backward-compatible spelling.
GemmaTeacherError = LLMTeacherError


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise LLMTeacherError(f"{name} must be a string")
    text = " ".join(value.split()).strip()
    if not text or len(text) > MAX_TEXT or _ALLOWED.fullmatch(text) is None:
        raise LLMTeacherError(f"{name} has an unsupported value")
    return text


def _layer(value: Any) -> int:
    if isinstance(value, str):
        try:
            return LAYER_NAMES.index(value.casefold())
        except ValueError as exc:
            raise LLMTeacherError("unknown L0-L9 layer") from exc
    if type(value) is not int or not 0 <= value <= 9:
        raise LLMTeacherError("layer must be 0..9")
    return value


def validate_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        value = value["records"]
    if not isinstance(value, list):
        raise LLMTeacherError("teacher response must be a JSON array or {records:[...]}")
    if len(value) > MAX_RECORDS:
        raise LLMTeacherError("teacher response exceeds 64 records")
    records = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise LLMTeacherError(f"record {index} must be an object")
        subject = _text(item.get("subject"), "subject")
        predicate = _text(item.get("predicate", "is"), "predicate")
        obj = _text(item.get("object"), "object")
        definition = item.get("definition")
        if definition is not None:
            definition = _text(definition, "definition")
        confidence = item.get("confidence", 0.5)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LLMTeacherError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise LLMTeacherError("confidence must be 0..1")
        evidence = item.get("evidence", "")
        if not isinstance(evidence, str) or len(evidence) > MAX_TEXT:
            raise LLMTeacherError("evidence is not bounded text")
        records.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "definition": definition,
            "layer": _layer(item.get("layer", 5)),
            "confidence": round(float(confidence), 6),
            "evidence": " ".join(evidence.split())[:MAX_TEXT],
        })
    return records


def parse_response(raw: str) -> list[dict[str, Any]]:
    if not isinstance(raw, str) or len(raw) > MAX_RESPONSE_BYTES:
        raise LLMTeacherError("teacher response is too large or not text")
    start_array, end_array = raw.find("["), raw.rfind("]")
    start_object, end_object = raw.find("{"), raw.rfind("}")
    if start_array >= 0 and end_array > start_array:
        candidate = raw[start_array:end_array + 1]
    elif start_object >= 0 and end_object > start_object:
        candidate = raw[start_object:end_object + 1]
    else:
        raise LLMTeacherError("teacher response does not contain a JSON value")
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMTeacherError("teacher response is not valid JSON") from exc
    return validate_records(value)


def build_prompt(topic: str, count: int) -> str:
    topic = _text(topic, "topic")
    if type(count) is not int or not 1 <= count <= MAX_RECORDS:
        raise LLMTeacherError("count must be 1..64")
    return (
        "You are a conservative lexical teacher for a local learning system. "
        "Return ONLY a JSON array of at most %d records. Each record must be "
        '{"subject": "...", "predicate": "...", "object": "...", '
        '"definition": "...", "layer": 0-9, "confidence": 0-1, '
        '"evidence": "..."}. Use short noun phrases. Do not invent private '
        "facts. Layer 0 is bit, 1 nibble, 2 byte, 3 phonetic, 4 morpheme, "
        "5 lexical, 6 word_completion, 7 sentence, 8 paragraph, 9 narrative. "
        "Topic: %s" % (count, topic)
    )


class HermesLLMTransport:
    """Use whichever provider Hermes already has configured."""

    def __init__(self, ctx, purpose: str = TEACHER_PURPOSE):
        if not hasattr(ctx, "llm"):
            raise LLMTeacherError("Hermes provider is not configured")
        self.ctx = ctx
        self.purpose = purpose

    def __call__(self, prompt: str) -> str:
        result = self.ctx.llm.complete(
            [{"role": "user", "content": prompt}], purpose=self.purpose)
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise LLMTeacherError("Hermes provider returned no text")
        return text


class OllamaTransport:
    def __init__(self, endpoint: str = "http://127.0.0.1:11434/api/generate",
                 model: str = "gemma4:31b-cloud", timeout: float = 120.0):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 2048},
        }, separators=(",", ":")).encode()
        request = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            raise LLMTeacherError("Ollama endpoint unavailable") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise LLMTeacherError("Ollama response exceeds the bound")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMTeacherError("Ollama endpoint returned invalid JSON") from exc
        answer = value.get("response")
        if not isinstance(answer, str):
            raise LLMTeacherError("Ollama response lacks text")
        return answer


class OpenAICompatibleTransport:
    """Any `/v1/chat/completions` endpoint, including local gateways."""

    def __init__(self, endpoint: str, model: str, *, api_key: str = "",
                 timeout: float = 120.0):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.endpoint, data=body, headers=headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            raise LLMTeacherError("OpenAI-compatible endpoint unavailable") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise LLMTeacherError("teacher response exceeds the bound")
        try:
            value = json.loads(raw)
            answer = value["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMTeacherError("teacher response lacks a valid completion") from exc
        if not isinstance(answer, str):
            raise LLMTeacherError("teacher completion is not text")
        return answer


class LLMTeacher:
    """Provider-universal teacher: validate, observe, never promote silently."""

    def __init__(self, transport: Callable[[str], str] | None = None):
        if transport is None:
            raise LLMTeacherError("a teacher transport is required")
        self.transport = transport

    def generate(self, topic: str, count: int = 8) -> list[dict[str, Any]]:
        return parse_response(self.transport(build_prompt(topic, count)))

    def teach(self, coordinator: LayeredLearningCoordinator, topic: str,
              count: int = 8, *, source: str = "llm-teacher") -> dict[str, Any]:
        records = self.generate(topic, count)
        relations = [
            {"subject": record["subject"], "predicate": record["predicate"],
             "object": record["object"]}
            for record in records
        ]
        observation = coordinator.observe(
            " ".join(
                " ".join((record["subject"], record["predicate"], record["object"]))
                for record in records
            ) or topic,
            source=source,
            relations=relations,
        )
        return {
            "schema": SCHEMA,
            "topic": topic,
            "provider_transport": self.transport.__class__.__name__,
            "records": len(records),
            "teacher_records": records,
            "observation": observation,
            "canonical_training_committed": False,
        }


def teacher_for_context(ctx) -> LLMTeacher:
    """Build the universal teacher from Hermes plugin configuration."""
    def setting(name, default):
        try:
            return ctx.get_config(name, default=default)
        except Exception:
            return default

    provider = str(setting("teacher_provider", "hermes")).casefold()
    if provider in ("hermes", "provider", "default"):
        return LLMTeacher(HermesLLMTransport(ctx))
    if provider in ("ollama", "gemma", "local"):
        return LLMTeacher(OllamaTransport(
            endpoint=str(setting("teacher_endpoint",
                                 setting("gemma_endpoint", "http://127.0.0.1:11434/api/generate"))),
            model=str(setting("teacher_model", setting("gemma_model", "gemma4:31b-cloud"))),
        ))
    if provider in ("openai", "openai-compatible", "compatible"):
        api_key = ""
        try:
            from .hermes_secrets import get_secret
            api_key = get_secret("LLM_TEACHER_API_KEY")
        except Exception:
            api_key = ""
        return LLMTeacher(OpenAICompatibleTransport(
            endpoint=str(setting("teacher_endpoint", "")),
            model=str(setting("teacher_model", "")),
            api_key=api_key,
        ))
    raise LLMTeacherError(f"unknown teacher provider: {provider}")


# Backward-compatible aliases.
GemmaTeacher = LLMTeacher
OllamaGemmaTransport = OllamaTransport


__all__ = [
    "GemmaTeacher",
    "GemmaTeacherError",
    "HermesLLMTransport",
    "LLMTeacher",
    "LLMTeacherError",
    "OllamaGemmaTransport",
    "OllamaTransport",
    "OpenAICompatibleTransport",
    "SCHEMA",
    "TEACHER_PURPOSE",
    "build_prompt",
    "parse_response",
    "teacher_for_context",
    "validate_records",
]
