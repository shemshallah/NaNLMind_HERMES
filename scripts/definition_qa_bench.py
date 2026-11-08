#!/usr/bin/env python3
"""Definition-to-answer baseline over WordNet 3.1.

This is retrieval plus exact graph composition, not a learned generative decoder.
Queries use a definition prefix; the target synset identity is never supplied.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.definition_decoder import DefinitionAnswerer  # noqa: E402
from nanlmind_hermes.wordnet_reasoning import ancestors, parents  # noqa: E402
from nanlmind_hermes.wordnet_typed import iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/DEFINITION_QA.md"
    nouns = [record for record in iter_synsets(ARCHIVE)
             if record["pos"] == "n" and record["definition"]]
    by_offset = {record["offset"]: record for record in nouns}
    by_lemma = {}
    for record in nouns:
        for lemma in record["lemmas"]:
            by_lemma.setdefault(lemma, []).append(record["offset"])
    rng = random.Random(16016)
    sample = rng.sample(nouns, 2000)
    sample_offsets = {record["offset"] for record in sample}
    sample_index = {}
    for record in sample:
        for lemma in record["lemmas"]:
            sample_index.setdefault(lemma, []).append(record["offset"])
    answerer = DefinitionAnswerer({record["offset"]: record for record in sample},
                                  sample_index)
    pools = {}
    for record in sample:
        pools.setdefault(record["lexname"], []).append(record["offset"])

    queries = 100
    top1 = 0
    top1_same_category = 0
    for record in rng.sample(sample, queries):
        prefix = " ".join(record["definition"].split()[:6])
        results = answerer.search(prefix, candidates=pools[record["lexname"]],
                                  limit=5)
        if results and results[0]["offset"] == record["offset"]:
            top1 += 1
        if results and by_offset[record["offset"]]["lexname"] == \
                by_offset[results[0]["offset"]]["lexname"]:
            top1_same_category += 1

    ancestry_correct = 0
    direct_correct = 0
    positive = 0
    negative = 0
    full = DefinitionAnswerer(by_offset, by_lemma)
    full_pools = {}
    for record in nouns:
        full_pools.setdefault(record["lexname"], []).append(record["offset"])
    for record in rng.sample(nouns, 300):
        source = record["offset"]
        ancestor_map = ancestors(by_offset, source, max_depth=4)
        candidates = [offset for offset, depth in ancestor_map.items()
                      if 2 <= depth <= 4]
        if not candidates:
            continue
        target = candidates[rng.randrange(len(candidates))]
        ancestry_correct += int(full.answer_is_a(record["lemmas"][0],
                                                 by_offset[target]["lemmas"][0])["answer"])
        direct_correct += int(target in parents(record))
        positive += 1
        pool = full_pools[by_offset[target]["lexname"]]
        negative_target = None
        for _ in range(20):
            candidate = pool[rng.randrange(len(pool))]
            if candidate != source and candidate not in ancestor_map:
                negative_target = candidate
                break
        if negative_target is None:
            continue
        negative_answer = full.answer_is_a(
            record["lemmas"][0], by_offset[negative_target]["lemmas"][0])["answer"]
        ancestry_correct += int(not negative_answer)
        direct_correct += int(negative_target not in parents(record))
        negative += 1
    # One held-out new-word acquisition case.
    new_word = "sparklebeak"
    penguin = next(offset for offset, record in by_offset.items()
                   if record["lemmas"] and record["lemmas"][0] == "penguin")
    full.universe.learn_word(new_word, penguin)
    new_answer = full.answer_what(new_word)
    new_is_a_bird = full.answer_is_a(new_word, "bird")
    new_is_a = full.answer_is_a(new_word, "animal")
    new_is_a_vehicle = full.answer_is_a(new_word, "vehicle")

    report = {
        "schema": "nanlmind-hermes-definition-qa/1",
        "noun_synsets": len(nouns),
        "sample_synsets": len(sample),
        "definition_queries": queries,
        "definition_top1": top1 / queries,
        "definition_same_category": top1_same_category / queries,
        "ancestry_positive_queries": positive,
        "ancestry_negative_queries": negative,
        "ancestry_multi_hop_balanced_accuracy": ancestry_correct / max(positive + negative, 1),
        "direct_edge_only_balanced_accuracy": direct_correct / max(positive + negative, 1),
        "new_word": new_word,
        "new_word_definition": new_answer["definition"] if new_answer else None,
        "new_word_is_a_bird": new_is_a_bird["answer"],
        "new_word_is_a_animal": new_is_a["answer"],
        "new_word_is_a_vehicle": new_is_a_vehicle["answer"],
        "note": ("definition prefixes are retrieval queries, not free-form prose "
                 "understanding; ancestry uses exact typed graph traversal"),
    }
    json_path = output.with_name("definition-qa.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Definition-to-answer baseline",
        "",
        "A WordNet definition prefix retrieves candidate synsets by IDF-weighted",
        "token overlap. Ancestry answers use exact typed hypernym traversal. This",
        "is a structured retrieval/graph decoder, not a generated language model.",
        "",
        f"- noun synsets available: **{len(nouns):,}**",
        f"- sampled synsets: **{len(sample):,}**",
        f"- definition-prefix top-1 (same lexname pool): **{top1 / queries:.4f}**",
        f"- same-category top-1: **{top1_same_category / queries:.4f}**",
        f"- ancestry positives: **{positive}**",
        f"- ancestry same-lexname negatives: **{negative}**",
        f"- multi-hop ancestry balanced accuracy: **{ancestry_correct / max(positive + negative, 1):.4f}**",
        f"- direct-edge-only balanced accuracy: **{direct_correct / max(positive + negative, 1):.4f}**",
        "",
        "## New word",
        "",
        f"- **{new_word}** linked to the real `penguin` synset",
        f"- definition: {report['new_word_definition']}",
        f"- `is a bird`: **{new_is_a_bird['answer']}** (one hop)",
        f"- `is a animal`: **{new_is_a['answer']}** (two hops)",
        f"- `is a vehicle`: **{new_is_a_vehicle['answer']}** (negative control)",
        "",
        "## Boundary",
        "",
        "- Retrieval over supplied WordNet glosses is not learned prose parsing.",
        "- The decoder does not generate free-form answers; it returns structured",
        "  definitions, relation records and ancestry paths.",
        "- The new word is attached by a synset link, not learned from arbitrary",
        "  definition wording.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
