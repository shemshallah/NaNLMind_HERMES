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
    "00": "adj.all",
    "01": "adj.pert",
    "02": "adv.all",
    "03": "noun.Tops",
    "04": "noun.act",
    "05": "noun.animal",
    "06": "noun.artifact",
    "07": "noun.attribute",
    "08": "noun.body",
    "09": "noun.cognition",
    "10": "noun.communication",
    "11": "noun.event",
    "12": "noun.feeling",
    "13": "noun.food",
    "14": "noun.group",
    "15": "noun.location",
    "16": "noun.motive",
    "17": "noun.object",
    "18": "noun.person",
    "19": "noun.phenomenon",
    "20": "noun.plant",
    "21": "noun.possession",
    "22": "noun.process",
    "23": "noun.quantity",
    "24": "noun.relation",
    "25": "noun.shape",
    "26": "noun.state",
    "27": "noun.substance",
    "28": "noun.time",
    "29": "verb.body",
    "30": "verb.change",
    "31": "verb.cognition",
    "32": "verb.communication",
    "33": "verb.competition",
    "34": "verb.consumption",
    "35": "verb.contact",
    "36": "verb.creation",
    "37": "verb.emotion",
    "38": "verb.motion",
    "39": "verb.perception",
    "40": "verb.possession",
    "41": "verb.social",
    "42": "verb.stative",
    "43": "verb.weather",
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
