#!/usr/bin/env python3
"""Audit WordNet 3.1 topology: vertices, hypernym edges and cycle rank per POS."""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/storage/emulated/0/Download/NaNLMind/wn3.1.dict.zip")
DATA_FILES = {
    "noun": "wn3.1.dict/dict/data.noun",
    "verb": "wn3.1.dict/dict/data.verb",
    "adj": "wn3.1.dict/dict/data.adj",
    "adv": "wn3.1.dict/dict/data.adv",
}
HYPERNYM = {"@", "@i"}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first, second):
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self.parent[root_b] = root_a


def parse_graph(archive: zipfile.ZipFile, name: str):
    vertices = set()
    edges = set()
    with archive.open(DATA_FILES[name]) as handle:
        for raw in handle:
            line = raw.decode("latin-1").strip()
            if "|" not in line:
                continue
            body = line.split("|", 1)[0]
            parts = body.split()
            if len(parts) < 5 or not re.match(r"^\d{8} ", body):
                continue
            offset = int(parts[0])
            pos = parts[2]
            vertex = (pos, offset)
            vertices.add(vertex)
            for target, target_pos in re.findall(
                    r"@i?\s+(\d{8})\s+([nvars])\s+\d{4}", body):
                edges.add((vertex, (target_pos, int(target))))
    return vertices, edges


def cycle_rank(vertices, edges):
    union = UnionFind()
    for vertex in vertices:
        union.find(vertex)
    for first, second in edges:
        union.union(first, second)
    components = len({union.find(vertex) for vertex in vertices})
    return len(edges) - len(vertices) + components, components


def main():
    archive = zipfile.ZipFile(ARCHIVE)
    report = {"schema": "nanlmind-hermes-wordnet-topology/1", "graphs": {}}
    combined_vertices, combined_edges = set(), set()
    for name in DATA_FILES:
        vertices, edges = parse_graph(archive, name)
        rank, components = cycle_rank(vertices, edges)
        report["graphs"][name] = {
            "vertices": len(vertices),
            "undirected_edges": len(edges),
            "components": components,
            "cycle_rank": rank,
        }
        combined_vertices |= vertices
        combined_edges |= edges
    rank, components = cycle_rank(combined_vertices, combined_edges)
    report["graphs"]["all"] = {
        "vertices": len(combined_vertices),
        "undirected_edges": len(combined_edges),
        "components": components,
        "cycle_rank": rank,
    }
    report["complete_nan"] = {
        "vertices": 256,
        "undirected_edges": 32640,
        "cycle_rank": 32385,
    }
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/WORDNET_TOPOLOGY.md"
    json_path = output.with_name("wordnet-topology.json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# WordNet 3.1 topology audit",
        "",
        "WordNet hypernym graphs are directed, but this table reports the rank of",
        "the undirected graph: E − V + components. Cycle rank is not a measure of",
        "contradiction or quantum behavior.",
        "",
        "| graph | vertices | undirected edges | components | cycle rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in report["graphs"].items():
        lines.append(f"| {name} | {data['vertices']:,} | {data['undirected_edges']:,} | "
                     f"{data['components']} | {data['cycle_rank']:,} |")
    lines += [
        f"| complete NaN | {report['complete_nan']['vertices']:,} | "
        f"{report['complete_nan']['undirected_edges']:,} | 1 | "
        f"{report['complete_nan']['cycle_rank']:,} |",
        "",
        "The complete NaN graph is the latent computational substrate; WordNet",
        "supplies the much sparser known relationships embedded into it.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
