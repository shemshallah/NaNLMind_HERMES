#!/usr/bin/env python3
"""Round 012-style exact WordNet multi-hop ancestry benchmark.

The correct source synset is supplied. This measures symbolic graph composition,
not learned reading of definitions.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.wordnet_reasoning import ancestors, parents  # noqa: E402
from nanlmind_hermes.wordnet_typed import iter_synsets  # noqa: E402

ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/WORDNET_ANCESTRY.md"
    synsets = {}
    for record in iter_synsets(ARCHIVE):
        if record["pos"] == "n":
            synsets[record["offset"]] = record
    rng = random.Random(12012)
    sources = [offset for offset, record in synsets.items() if parents(record)]
    rng.shuffle(sources)
    pools = collections.defaultdict(list)
    for offset, record in synsets.items():
        pools[record["lexname"]].append(offset)

    positives = []
    negatives = []
    for source in sources:
        ancestor_map = ancestors(synsets, source, max_depth=4)
        candidates = [offset for offset, depth in ancestor_map.items()
                      if 2 <= depth <= 4]
        if not candidates:
            continue
        target = candidates[rng.randrange(len(candidates))]
        pool = pools[synsets[target]["lexname"]]
        negative = None
        for _ in range(25):
            candidate = pool[rng.randrange(len(pool))]
            if candidate != source and candidate not in ancestor_map:
                negative = candidate
                break
        if negative is None:
            continue
        positives.append((source, target, ancestor_map[target]))
        negatives.append((source, negative))
        if len(positives) >= 1000:
            break

    direct_correct = 0
    transitive_correct = 0
    for source, target, _depth in positives:
        direct_correct += int(target in parents(synsets[source]))
        transitive_correct += int(target in ancestors(synsets, source, max_depth=4))
    for source, target in negatives:
        direct_correct += int(target not in parents(synsets[source]))
        transitive_correct += int(target not in ancestors(synsets, source, max_depth=4))

    examples = []
    for source, target, depth in positives[:12]:
        examples.append({
            "source": synsets[source]["lemmas"][0],
            "target": synsets[target]["lemmas"][0],
            "hops": depth,
            "source_lexname": synsets[source]["lexname"],
            "target_lexname": synsets[target]["lexname"],
        })
    total = len(positives) + len(negatives)
    report = {
        "schema": "nanlmind-hermes-wordnet-ancestry/1",
        "synsets": len(synsets),
        "positive_queries": len(positives),
        "negative_queries": len(negatives),
        "direct_edge_only_balanced_accuracy": direct_correct / total,
        "multi_hop_balanced_accuracy": transitive_correct / total,
        "hop_distribution": dict(collections.Counter(depth for _s, _t, depth in positives)),
        "examples": examples,
        "caveat": ("The correct source synset is given; this is exact symbolic "
                   "composition, not definition understanding or a trained model."),
    }
    json_path = output.with_name("wordnet-ancestry.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# WordNet exact ancestry traversal",
        "",
        "The source synset is supplied. Positive targets are 2–4-hop ancestors;",
        "negative targets share the target's lexicographer class but are not",
        "ancestors. This is symbolic graph composition, not language learning.",
        "",
        f"- noun synsets: **{len(synsets):,}**",
        f"- positives: **{len(positives):,}**",
        f"- negatives: **{len(negatives):,}**",
        f"- direct-edge-only balanced accuracy: "
        f"**{report['direct_edge_only_balanced_accuracy']:.4f}**",
        f"- multi-hop balanced accuracy: "
        f"**{report['multi_hop_balanced_accuracy']:.4f}**",
        f"- hop distribution: `{report['hop_distribution']}`",
        "",
        "## Examples",
        "",
    ]
    for example in examples:
        lines.append(f"- {example['source']} → {example['target']} "
                     f"({example['hops']} hops; {example['source_lexname']} → "
                     f"{example['target_lexname']})")
    lines += [
        "",
        "The 50% direct baseline is expected: positives are deliberately not",
        "direct edges, while negatives are rejected by direct lookup. The",
        "multi-hop result only verifies traversal over the real graph.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
