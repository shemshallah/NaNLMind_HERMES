"""Typed WordNet 3.1 extraction for the sparse HW4 dictionary universe.

WordNet's pointer section is more than hypernyms: meronyms, attributes,
entailments, causes, domains, antonyms and derivational links are all typed
directed edges. This module preserves those types and the synset gloss without
materializing a dense vector per word.

The pointer regex is used instead of a strict field walk because some WordNet
POS files contain optional markers around word forms. A pointer is recognised
only by its stable shape: symbol, eight-digit target offset, target POS and the
four-hex source/target field.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

DATA_FILES = {
    "noun": "wn3.1.dict/dict/data.noun",
    "verb": "wn3.1.dict/dict/data.verb",
    "adj": "wn3.1.dict/dict/data.adj",
    "adv": "wn3.1.dict/dict/data.adv",
}

# WordNet 3.1 pointer symbols. Unknown symbols are retained as "other:<sym>".
RELATION_SYMBOLS = {
    "!": "antonym",
    "@": "hypernym",
    "@i": "instance_hypernym",
    "~": "hyponym",
    "~i": "instance_hyponym",
    "#m": "member_meronym",
    "#s": "substance_meronym",
    "#p": "part_meronym",
    "%m": "member_holonym",
    "%s": "substance_holonym",
    "%p": "part_holonym",
    "=": "attribute",
    "+": "derivationally_related_form",
    ";c": "domain_topic",
    ";r": "domain_region",
    ";u": "domain_usage",
    "-c": "member_of_domain_topic",
    "-r": "member_of_domain_region",
    "-u": "member_of_domain_usage",
    "*": "entailment",
    ">": "cause",
    "^": "also_see",
    "$": "verb_group",
    "&": "similar_to",
    "<": "participle_of",
    "\\": "pertainym",
}

LEXNAMES = {
    "00": "noun.Tops",
    "01": "noun.act",
    "02": "noun.animal",
    "03": "noun.artifact",
    "04": "noun.attribute",
    "05": "noun.body",
    "06": "noun.cognition",
    "07": "noun.communication",
    "08": "noun.event",
    "09": "noun.feeling",
    "10": "noun.food",
    "11": "noun.group",
    "12": "noun.location",
    "13": "noun.motive",
    "14": "noun.object",
    "15": "noun.person",
    "16": "noun.phenomenon",
    "17": "noun.plant",
    "18": "noun.possession",
    "19": "noun.process",
    "20": "noun.quantity",
    "21": "noun.relation",
    "22": "noun.shape",
    "23": "noun.state",
    "24": "noun.substance",
    "25": "noun.time",
    "26": "verb.body",
    "27": "verb.change",
    "28": "verb.cognition",
    "29": "verb.communication",
    "30": "verb.competition",
    "31": "verb.consumption",
    "32": "verb.contact",
    "33": "verb.creation",
    "34": "verb.emotion",
    "35": "verb.motion",
    "36": "verb.perception",
    "37": "verb.possession",
    "38": "verb.social",
    "39": "verb.stative",
    "40": "verb.weather",
    "41": "adj.all",
    "42": "adj.pert",
    "43": "adv.all",
    "44": "adj.ppl",
}

HEADER_RE = re.compile(r"^(\d{8}) (\d{2}) ([nvars]) ([0-9a-f]{2})(?:\s|$)")
POINTER_RE = re.compile(
    r"(?<!\S)([!?@~#$%^&*+=;:<>\-\\][a-z]?)\s+"
    r"(\d{8})\s+([nvars])\s+[0-9a-f]{4}(?!\S)"
)
EXAMPLE_RE = re.compile(r'"([^"]+)"')


def relation_name(symbol: str) -> str:
    return RELATION_SYMBOLS.get(symbol, f"other:{symbol}")


def parse_line(line: str):
    """Parse one data.<pos> record into a JSON-safe synset record."""
    line = line.strip()
    if "|" not in line:
        return None
    body, gloss = line.split("|", 1)
    header = HEADER_RE.match(body)
    if not header:
        return None
    offset = int(header.group(1))
    lex_filenum = header.group(2)
    pos = header.group(3)
    parts = body.split()
    if len(parts) < 5:
        return None
    word_count = int(parts[3], 16)
    index = 4
    words = []
    for _ in range(word_count):
        if index >= len(parts):
            break
        words.append(parts[index].replace("_", " "))
        index += 2
    pointer_text = " ".join(parts[index:])
    pointers = []
    for symbol, target, target_pos in POINTER_RE.findall(pointer_text):
        pointers.append({
            "relation": relation_name(symbol),
            "symbol": symbol,
            "target_pos": target_pos,
            "target_offset": int(target),
        })
    gloss = gloss.strip()
    definition = gloss.split('; "', 1)[0].strip()
    examples = EXAMPLE_RE.findall(gloss)
    return {
        "pos": pos,
        "offset": offset,
        "lex_filenum": lex_filenum,
        "lexname": LEXNAMES.get(lex_filenum, f"lex:{lex_filenum}"),
        "lemmas": [word.lower() for word in words],
        "gloss": gloss,
        "definition": definition,
        "examples": examples,
        "pointers": pointers,
    }


def iter_synsets(archive_path) -> iter:
    archive = zipfile.ZipFile(Path(archive_path))
    with archive:
        for data_file in DATA_FILES.values():
            with archive.open(data_file) as handle:
                for raw in handle:
                    record = parse_line(raw.decode("latin-1"))
                    if record is not None:
                        yield record


def audit(archive_path) -> dict:
    """Stream WordNet and return typed edge/lemma counts without storing the graph."""
    positions = {}
    relations = {}
    lemmas = 0
    gloss_chars = 0
    typed_edges = 0
    hypernym_edges = 0
    records = 0
    for record in iter_synsets(archive_path):
        records += 1
        pos = record["pos"]
        bucket = positions.setdefault(pos, {"synsets": 0, "lemmas": 0,
                                           "gloss_chars": 0, "pointers": 0})
        bucket["synsets"] += 1
        bucket["lemmas"] += len(record["lemmas"])
        bucket["gloss_chars"] += len(record["gloss"])
        lemmas += len(record["lemmas"])
        gloss_chars += len(record["gloss"])
        for pointer in record["pointers"]:
            relation = pointer["relation"]
            relations[relation] = relations.get(relation, 0) + 1
            bucket["pointers"] += 1
            typed_edges += 1
            if relation in {"hypernym", "instance_hypernym"}:
                hypernym_edges += 1
    return {
        "synsets": records,
        "lemmas": lemmas,
        "gloss_chars": gloss_chars,
        "typed_edges": typed_edges,
        "hypernym_edges": hypernym_edges,
        "positions": positions,
        "relations": dict(sorted(relations.items(),
                                 key=lambda item: (-item[1], item[0]))),
    }
