#!/usr/bin/env python3
"""Train the head-population next-letter model on WordNet 3.1 lemmas."""
from __future__ import annotations

import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.head_population import (  # noqa: E402
    ALPHABET,
    BigramBaseline,
    HeadPopulation,
    fnv,
)

ARCHIVE = "/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip"
FALLBACK = "/storage/emulated/0/Download/hyperllm/builder/wn3.1.dict.zip"
MAX_WORDS = 20_000


def load_lemmas() -> tuple[list[str], dict[str, int]]:
    path = ARCHIVE if Path(ARCHIVE).exists() else FALLBACK
    words: set[str] = set()
    frequency: dict[str, int] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if "/index." not in name or name.endswith("index.sense"):
                continue
            with archive.open(name) as handle:
                for raw in handle:
                    try:
                        line = raw.decode("latin-1")
                    except UnicodeDecodeError:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    lemma = parts[0].strip().lower()
                    if "_" in lemma or not lemma.isalpha() or not 2 <= len(lemma) <= 24:
                        continue
                    words.add(lemma)
                    try:
                        pointer_count = int(parts[3])
                        sense_index = 4 + pointer_count
                        tagsense = int(parts[sense_index + 1])
                        frequency[lemma] = max(frequency.get(lemma, 0), tagsense)
                    except (IndexError, ValueError):
                        frequency.setdefault(lemma, 0)
    ordered = sorted(words)
    if len(ordered) > MAX_WORDS:
        rng = random.Random(7)
        ordered = sorted(rng.sample(ordered, MAX_WORDS))
    return ordered, frequency


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/WORD_COMPLETION.md"
    words, frequency = load_lemmas()
    train = [word for word in words if fnv(word) % 5 != 0]
    test = [word for word in words if fnv(word) % 5 == 0]

    population = HeadPopulation(heads=256, active=8, prior_weight=0.0,
                                train_head0=False)
    for word in train:
        population.observe(word)
        # Explicit partial-word curriculum: emphasise early prefixes.
        for length in range(1, min(len(word), 5)):
            population.observe_partial(word[:length], word[length], weight=2.0)
    bigram = BigramBaseline()
    bigram.train(train)

    population_result = population.evaluate(test)
    baseline_result = bigram.evaluate(test)

    examples = {}
    for context in ("^^", "th", "qu", "in", "st", "er"):
        distribution = population.distribution(context)
        examples[context] = {
            "top": list(distribution.items())[:8],
            "head_votes": population.head_votes(context),
        }

    # All 256 heads, head 0 as the void/prior; frequency-weighted curriculum.
    all_heads = HeadPopulation(heads=256, active=255)
    weighted_train = sorted(train, key=lambda word: -frequency.get(word, 0))[:8000]
    for word in weighted_train:
        weight = 1.0 + min(4.0, frequency.get(word, 0) ** 0.5)
        all_heads.observe(word, weight=weight)
    all_heads_result = all_heads.evaluate(test)
    all_heads.prior_weight = 0.0
    all_heads_no_prior = all_heads.evaluate(test)
    all_heads.prior_weight = 1.0
    all_example = {}
    for context in ("th", "qu", "in"):
        all_example[context] = {
            "top": list(all_heads.distribution(context).items())[:6],
            "head_votes": all_heads.head_votes(context),
        }

    prefix = {}
    for length in (2, 3, 4, 5):
        correct = total = 0
        for word in test:
            if len(word) <= length:
                continue
            context = word[length - 2:length]
            target = word[length] if length < len(word) else "$"
            distribution = list(population.distribution(context))
            total += 1
            correct += bool(distribution) and distribution[0] == target
        prefix[length] = round(correct / total, 4) if total else 0.0

    report = {
        "schema": "nanlmind-hermes-word-completion/1",
        "archive": ARCHIVE if Path(ARCHIVE).exists() else FALLBACK,
        "words": len(words), "train": len(train), "test": len(test),
        "population": population.status(),
        "population_result": population_result,
        "bigram_result": baseline_result,
        "all_heads_result": all_heads_result,
        "all_heads_no_prior": all_heads_no_prior,
        "all_heads": all_heads.status(),
        "all_heads_examples": all_example,
        "frequency_weighted_train": len(weighted_train),
        "prefix_accuracy": prefix,
        "examples": examples,
        "superposition": population.superposition()[:8],
        "letter_to_head_mapping": False,
        "dictionary_in_superposition": True,
        "stored_words": 0,
        "note": ("each head holds a distribution over the whole alphabet; a "
                 "context activates a population and the distributions collapse; "
                 "the dictionary is the superposition, not a word list"),
    }
    json_path = output.with_name("word-completion.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# WordNet 3.1 head-population next-letter completion",
        "",
        "Letters are not mapped to heads. Each of the 256 heads holds a",
        "distribution over the whole alphabet (`a`–`z`, end-of-word); a context",
        "activates a population of heads and their distributions collapse into",
        "the next-letter counts. Head 0 is the shared NaNNet/polarity prior.",
        "",
        f"- lemmas: **{len(words):,}** (train {len(train):,}, test {len(test):,})",
        f"- heads with a letter distribution: **{report['population']['heads_with_distribution']}**",
        f"- active heads per context: **{report['population']['active_per_context']}**",
        f"- letter→head mapping: **none**",
        f"- dictionary in superposition: **{report['dictionary_in_superposition']}**; stored words: **{report['stored_words']}**",
        "", "Training streams words and partial words; no dictionary text is kept.",
        "",
        "## Accuracy",
        "",
        "| model | accuracy | top-3 |",
        "|---|---:|---:|",
        f"| head population | {population_result['accuracy']} | {population_result['top3']} |",
        f"| bigram baseline | {baseline_result['accuracy']} |  |",
        "",
        "## All-256-head run",
        "",
        "Head 0 is the void/NaNNet prior; heads 1–255 all participate. Training",
        "uses an 8k frequency-weighted curriculum (common words count more;",
        "counts are order-independent).",
        "",
        "```json",
        json.dumps({"result": all_heads_result, "status": all_heads.status(),
                    "examples": all_example}, indent=2, sort_keys=True)[:6000],
        "```",
        "",
        "## Collapse examples",
        "",
    ]
    for context, data in examples.items():
        top = ", ".join(f"{letter}:{value:.3f}" for letter, value in data["top"])
        votes = ", ".join(f"{letter}:{count}" for letter, count in
                          sorted(data["head_votes"].items(), key=lambda x: -x[1])[:8])
        lines.append(f"- `{context}` → {top}")
        lines.append(f"  - head votes: {votes}")
    lines += [
        "",
        "## Head superposition (sample)",
        "",
        "The dictionary is stored only as head letter distributions. No word",
        "text is retained.",
        "",
        "```json",
        json.dumps(report["superposition"], indent=2)[:5000],
        "```",
        "",
        "## Prefix accuracy",
        "",
        "```json",
        json.dumps(prefix, indent=2, sort_keys=True),
        "```",
        "",
        "## Honest read",
        "",
        "- The population is a hashed mixture: every head holds all letters, so no",
        "  one letter owns a head.",
        "- The head votes are diagnostics of the collapse; the answer is the",
        "  summed distribution, not a single head's choice.",
        "- This is still a statistical model; a learned Boolean/gate mechanism",
        "  should replace the hash activation next.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "json": str(json_path),
        "words": len(words), "population": population_result,
        "bigram": baseline_result, "prefix": prefix,
        "heads_with_distribution": report["population"]["heads_with_distribution"],
    }, indent=2))


if __name__ == "__main__":
    main()
