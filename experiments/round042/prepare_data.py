#!/usr/bin/env python3
"""Prepare the compact HW4 WordNet mirror used by Round 042.

The input is the WordNet 3.1 ``data.noun`` file (or a Round 039 JSONL export).
The output preserves every synset needed to resolve typed pointers, while a
deterministic 25k/1.5k/1.5k subset carries the language-modelling glosses.

The canonical route is a read-only evidence address.  This is a local mirror,
not a live NANNET lease, and it does not authorize canonical merge.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW0:0::HW4:0"
KEEP_POINTERS = {"@", "@i", "~", "~i", "!", "#p", "#m", "#s", "%p", "%m", "%s"}


def split_for(identifier: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big") % 1000
    if bucket < 800:
        return "train"
    if bucket < 890:
        return "val"
    return "test"


def parse_data_noun(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line or line.startswith(" "):
            continue
        head, _, gloss = line.partition("|")
        parts = head.split()
        if len(parts) < 6:
            continue
        offset, lexfile, ss_type = parts[0], parts[1], parts[2]
        try:
            word_count = int(parts[3], 16)
            index = 4
            words: list[str] = []
            for _ in range(word_count):
                if index >= len(parts):
                    break
                words.append(parts[index].replace("_", " "))
                index += 2
            if index >= len(parts):
                continue
            pointer_count = int(parts[index], 16)
        except ValueError:
            continue
        index += 1
        pointers: list[list[str]] = []
        for _ in range(pointer_count):
            if index + 3 >= len(parts):
                break
            symbol, target, pos, _source = parts[index:index + 4]
            index += 4
            if symbol in KEEP_POINTERS:
                pointers.append([symbol, target, pos])
        identifier = f"n{offset}"
        rows.append({
            "id": identifier,
            "split": split_for(identifier),
            "lemmas": words[:4],
            "gloss": gloss.strip(),
            "pointers": pointers,
            "lexfile": lexfile,
            "route": ROUTE,
            "source": f"WordNet3.1:data.noun:{offset}",
        })
    return rows


def parse_round039(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        record = json.loads(line)
        if record.get("pos") != "noun" or not record.get("lemmas") or not record.get("gloss"):
            continue
        rows.append({
            "id": str(record["id"]),
            "split": str(record.get("split", "train")),
            "lemmas": list(record["lemmas"])[:4],
            "gloss": str(record["gloss"]).strip(),
            "pointers": [
                [str(kind), str(target), str(pos)]
                for kind, target, pos in record.get("pointers", [])
                if kind in KEEP_POINTERS
            ],
            "lexfile": str(record.get("lexfile", "")),
            "route": ROUTE,
            "source": str(record.get("source", "")),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wordnet",
        default="/sdcard/Download/LM/wn3.1.dict/dict/data.noun",
    )
    parser.add_argument("--round039", default="")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "data" / "wordnet_hw4.jsonl.gz")
    )
    parser.add_argument("--lm-train", type=int, default=25000)
    parser.add_argument("--lm-val", type=int, default=1500)
    parser.add_argument("--lm-test", type=int, default=1500)
    args = parser.parse_args()

    if args.round039:
        rows = parse_round039(Path(args.round039))
    else:
        rows = parse_data_noun(Path(args.wordnet))
    if not rows:
        raise SystemExit("no WordNet records parsed")

    # Deterministic LM subset: the first records in each split, preserving the
    # full synset table for pointer resolution.
    caps = {"train": args.lm_train, "val": args.lm_val, "test": args.lm_test}
    used = {"train": 0, "val": 0, "test": 0}
    selected = []
    for row in rows:
        split = row["split"]
        if split in caps and used[split] < caps[split]:
            row = dict(row)
            row["lm"] = True
            used[split] += 1
        else:
            row = dict(row)
            row["lm"] = False
        selected.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "route": ROUTE,
        "records": len(selected),
        "lm_records": used,
        "out": str(args.out),
        "bytes": Path(args.out).stat().st_size,
        "live_nannet_route": False,
        "canonical_merge": False,
    }, indent=2))


if __name__ == "__main__":
    main()
