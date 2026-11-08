"""Deterministic UTF-8 corpus ingestion and source-grounded retrieval.

This is a lexical retrieval store, not a language model. It deduplicates exact
chunks, assigns deterministic hash-based splits, and returns source passages
with their paths. Retrieval relevance is token overlap, not comprehension.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "with", "which", "who", "whose", "this", "these", "those", "was",
    "were", "will", "can", "not", "no",
}


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.casefold())
            if token not in STOPWORDS and len(token) > 1]


class CorpusStore:
    def __init__(self):
        self.documents: list[dict] = []

    def ingest(self, paths, max_tokens: int = 700) -> dict:
        files = []
        for given in paths:
            path = Path(given)
            if path.is_dir():
                files.extend(sorted(item for item in path.rglob("*.txt")
                                    if item.is_file()))
            else:
                files.append(path)
        seen = set()
        documents = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            paragraphs = re.split(r"\n\s*\n+", text)
            for paragraph in paragraphs:
                words = paragraph.split()
                if not words:
                    continue
                for start in range(0, len(words), max_tokens):
                    chunk = " ".join(words[start:start + max_tokens]).strip()
                    if not chunk:
                        continue
                    digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    bucket = int(digest[:8], 16) % 10
                    split = "train" if bucket < 8 else ("validation" if bucket == 8 else "test")
                    documents.append({
                        "sha256": digest,
                        "source": str(path),
                        "split": split,
                        "text": chunk,
                    })
        self.documents = documents
        return self.stats()

    def stats(self) -> dict:
        counts = {"train": 0, "validation": 0, "test": 0}
        for document in self.documents:
            counts[document["split"]] += 1
        return {"documents": len(self.documents), "splits": counts}

    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.documents, indent=2,
                                        ensure_ascii=False) + "\n",
                             encoding="utf-8")
        temporary.replace(target)

    def load(self, path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("corpus manifest must be a list")
        self.documents = [item for item in data
                          if isinstance(item, dict) and "text" in item]

    def _idf(self, split: str | None = None) -> dict[str, float]:
        documents = [doc for doc in self.documents
                     if split is None or doc["split"] == split]
        frequency = {}
        for document in documents:
            for token in set(tokens(document["text"])):
                frequency[token] = frequency.get(token, 0) + 1
        total = max(len(documents), 1)
        return {token: (total + 1) / (count + 1)
                for token, count in frequency.items()}

    def retrieve(self, query: str, split: str | None = None,
                 limit: int = 3) -> list[dict]:
        query_tokens = set(tokens(query))
        if not query_tokens:
            return []
        idf = self._idf(split)
        scored = []
        for document in self.documents:
            if split is not None and document["split"] != split:
                continue
            document_tokens = set(tokens(document["text"]))
            score = sum(idf.get(token, 0.0) for token in query_tokens & document_tokens)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1]["sha256"]))
        return [{"score": score, **document} for score, document in scored[:limit]]
