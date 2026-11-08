"""Foundational leaf algebra: bits -> bytes -> letters -> words -> relations.

Everything is a typed leaf with provenance; hyperedges construct larger leaves;
functions are registered mathematical transforms validated by domain/codomain.
No free-floating linguistic labels: letters are bit patterns, words are ordered
hyperedges over letter leaves, vectors are constructed leaves.

The 16 root Boolean functions are the substrate. 0 and 1 exist both as value
leaves and as Boolean-function leaves (ZERO/ONE), which is the distinction the
system needs between a value and a transform.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from .mth_layers import ROOT_NAMES, apply_bits

BASE = "nan://HW2:BS0:MC0:SC0:GM0"
PRINTABLE = tuple(chr(code) for code in range(32, 127))


def digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


@dataclass
class Leaf:
    leaf_id: str
    kind: str
    value: Any
    dtype: str = ""
    rank: int = 0
    shape: tuple = ()
    provenance: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)

    @property
    def address(self) -> str:
        return f"{BASE}::leaf/{self.kind.lower()}/{self.leaf_id[:16]}"


@dataclass
class Function:
    name: str
    domain: tuple[str, ...]
    codomain: str
    rule: Callable
    arity: int
    mask: int | None = None

    def apply(self, algebra: "LeafAlgebra", *leaves: Leaf) -> Leaf:
        if len(leaves) != self.arity:
            raise ValueError(f"{self.name}: arity {self.arity}, got {len(leaves)}")
        for spec, leaf in zip(self.domain, leaves):
            if spec != leaf.kind and spec != "*":
                raise TypeError(f"{self.name}: expected {spec}, got {leaf.kind}")
        return self.rule(algebra, *leaves)


class LeafAlgebra:
    def __init__(self):
        self.leaves: dict[str, Leaf] = {}
        self.functions: dict[str, Function] = {}
        self.hyperedges: dict[str, dict] = {}
        self.ledger: list[dict] = []
        self.heads: dict[int, list[str]] = {}
        self._register_boolean_functions()
        self.zero = self.value_leaf("BOOLEAN", 0, dtype="BOOLEAN")
        self.one = self.value_leaf("BOOLEAN", 1, dtype="BOOLEAN")

    # -- leaves ---------------------------------------------------------
    def _admit(self, leaf: Leaf) -> Leaf:
        self.leaves[leaf.leaf_id] = leaf
        head, mask = self.head_for(leaf)
        leaf.tags.setdefault("head", head)
        leaf.tags.setdefault("root_mask", mask)
        leaf.tags.setdefault("root_function", ROOT_NAMES[mask])
        self.heads.setdefault(head, []).append(leaf.leaf_id)
        self.ledger.append({"event": "admit", "leaf": leaf.leaf_id,
                            "kind": leaf.kind, "head": head, "mask": mask})
        return leaf

    def value_leaf(self, kind: str, value: Any, dtype: str = "",
                   provenance: dict | None = None) -> Leaf:
        leaf_id = digest(kind, repr(value), dtype)
        leaf = Leaf(leaf_id, kind, value, dtype=dtype,
                    provenance=provenance or {"constructor": "value"})
        return self._admit(leaf)

    def bitstring(self, bits: tuple[int, ...], kind: str = "BITSTRING",
                  provenance: dict | None = None) -> Leaf:
        value = tuple(int(bit) & 1 for bit in bits)
        leaf_id = digest(kind, value, provenance or {})
        leaf = Leaf(leaf_id, kind, value, dtype="BOOLEAN", rank=1,
                    shape=(len(value),),
                    provenance=provenance or {"constructor": "bitstring"},
                    tags={"bits": "".join(str(bit) for bit in value)})
        return self._admit(leaf)

    def byte(self, value: int) -> Leaf:
        bits = tuple((value >> shift) & 1 for shift in range(7, -1, -1))
        return self.bitstring(bits, kind="BYTE",
                              provenance={"constructor": "byte", "value": value})

    def letter(self, char: str) -> Leaf:
        if len(char) != 1:
            raise ValueError("letter requires one character")
        value = ord(char)
        byte_leaf = self.byte(value)
        leaf_id = digest("LETTER", char, value)
        leaf = Leaf(leaf_id, "LETTER", char, dtype="CHAR", rank=1, shape=(),
                    provenance={"constructor": "letter", "byte": byte_leaf.leaf_id,
                                "code": value},
                    tags={"byte_leaf": byte_leaf.leaf_id, "ascii": value})
        return self._admit(leaf)

    def hyperedge(self, members: list[Leaf], relation: str = "sequence") -> Leaf:
        ids = tuple(member.leaf_id for member in members)
        edge_id = digest("HYPEREDGE", ids, relation)
        self.hyperedges[edge_id] = {"members": ids, "relation": relation}
        leaf_id = digest("WORD", ids, relation)
        leaf = Leaf(leaf_id, "WORD", tuple(m.value for m in members),
                    dtype="SEQUENCE", rank=1, shape=(len(members),),
                    provenance={"constructor": "hyperedge", "hyperedge": edge_id,
                                "members": ids, "relation": relation},
                    tags={"hyperedge": edge_id})
        return self._admit(leaf)

    def word(self, text: str) -> Leaf:
        letters = [self.letter(char) for char in text]
        return self.hyperedge(letters, "spells")

    def vector(self, values: tuple, dtype: str = "RATIONAL",
               provenance: dict | None = None) -> Leaf:
        shape = (len(values),)
        leaf_id = digest("VECTOR", values, dtype)
        leaf = Leaf(leaf_id, "VECTOR", tuple(values), dtype=dtype, rank=1,
                    shape=shape, provenance=provenance or {"constructor": "PACK_VECTOR"})
        return self._admit(leaf)

    def matrix(self, rows: tuple[tuple, ...], dtype: str = "RATIONAL",
               provenance: dict | None = None) -> Leaf:
        rank = 2
        shape = (len(rows), len(rows[0]) if rows else 0)
        leaf_id = digest("MATRIX", rows, dtype)
        leaf = Leaf(leaf_id, "MATRIX", tuple(tuple(r) for r in rows), dtype=dtype,
                    rank=rank, shape=shape,
                    provenance=provenance or {"constructor": "OUTER_PRODUCT"})
        return self._admit(leaf)

    # -- functions ------------------------------------------------------
    def _register_boolean_functions(self) -> None:
        for mask in range(16):
            def rule(algebra, left, right, mask=mask):
                result = apply_bits(mask, int(left.value), int(right.value), 1)
                return algebra.value_leaf("BOOLEAN", result, dtype="BOOLEAN",
                                          provenance={"function": f"BOOL_{mask}",
                                                      "mask": mask,
                                                      "operands": [left.leaf_id,
                                                                   right.leaf_id]})
            self.functions[f"BOOL_{mask}"] = Function(
                f"BOOL_{mask}", ("BOOLEAN", "BOOLEAN"), "BOOLEAN", rule, 2, mask)
        self.functions["NEGATE"] = Function(
            "NEGATE", ("BOOLEAN",), "BOOLEAN",
            lambda algebra, value: algebra.value_leaf(
                "BOOLEAN", 1 - int(value.value), dtype="BOOLEAN",
                provenance={"function": "NEGATE", "operand": value.leaf_id}),
            1)
        self.functions["AFFIRM"] = Function(
            "AFFIRM", ("BOOLEAN",), "BOOLEAN",
            lambda algebra, value: algebra.value_leaf(
                "BOOLEAN", int(value.value), dtype="BOOLEAN",
                provenance={"function": "AFFIRM", "operand": value.leaf_id}),
            1)
        self.functions["CONCAT"] = Function(
            "CONCAT", ("LETTER", "LETTER"), "WORD",
            lambda algebra, first, second: algebra.hyperedge([first, second], "spells"),
            2)

    def apply(self, name: str, *leaves: Leaf) -> Leaf:
        function = self.functions.get(name)
        if function is None:
            raise ValueError(f"unregistered function {name}")
        result = function.apply(self, *leaves)
        self.ledger.append({"event": "transform", "function": name,
                            "inputs": [leaf.leaf_id for leaf in leaves],
                            "output": result.leaf_id})
        return result

    # -- heads ----------------------------------------------------------
    def head_for(self, leaf: Leaf) -> tuple[int, int]:
        # The byte layer is the structural head map: byte 0x00..0xFF maps to
        # heads 0..255 exactly once. Letters inherit their byte coordinate.
        if leaf.kind == "BYTE":
            bits = "".join(str(int(bit) & 1) for bit in leaf.value)
            value = int(bits, 2) if bits else 0
            return value & 0xFF, value & 0x0F
        if leaf.kind == "LETTER":
            code = ord(str(leaf.value)[0]) & 0xFF
            return code, code & 0x0F
        if leaf.kind == "BOOLEAN":
            return int(leaf.value) & 0xFF, 0x0F
        if leaf.kind == "WORD":
            head = 0
            for char in str(leaf.value):
                head = (head * 31 + ord(char)) & 0xFF
            return head, (head & 0x0F)
        # Other constructed leaves (vectors, matrices) fold by relation value.
        payload = repr(leaf.value) + leaf.kind
        data = payload.encode("utf-8")
        mask = (sum(data) + len(data)) % 16
        head = 0
        for index in range(0, len(data), 2):
            left = data[index]
            right = data[index + 1] if index + 1 < len(data) else 0
            head ^= apply_bits(mask, left, right, 8)
        return head & 0xFF, mask

    def self_modify(self, leaf: Leaf, note: str) -> dict:
        head = int(leaf.tags.get("head", 0)) & 0xFF
        event = {"event": "self_modify", "head": head, "leaf": leaf.leaf_id,
                 "note": note, "mask": leaf.tags.get("root_mask")}
        self.ledger.append(event)
        return event


def build_mapping_tree(algebra: LeafAlgebra) -> dict:
    """Initial matrix tree: bits -> functions -> bytes -> letters -> words."""
    levels = {"L0_bits": [], "L1_functions": [], "L2_bytes": [],
              "L3_letters": [], "L4_words": [], "L5_vectors": []}
    for bit in (0, 1):
        levels["L0_bits"].append(algebra.value_leaf("BOOLEAN", bit,
                                                    dtype="BOOLEAN").leaf_id)
    for name in algebra.functions:
        levels["L1_functions"].append(name)
    for value in range(256):
        levels["L2_bytes"].append(algebra.byte(value).leaf_id)
    for char in PRINTABLE:
        levels["L3_letters"].append(algebra.letter(char).leaf_id)
    for word in ("NO", "YES", "ON", "NOT", "ONE", "ZERO", "BOTH", "EITHER",
                 "DIFFERENT", "TRUE", "FALSE", "LEAF"):
        levels["L4_words"].append(algebra.word(word).leaf_id)
    pair = algebra.vector((1, 0), dtype="BOOLEAN",
                          provenance={"constructor": "PACK_VECTOR",
                                      "source": "polarity"})
    levels["L5_vectors"].append(pair.leaf_id)
    return levels


class ByteCompletion:
    """Practice next-bit prediction from an exposed byte corpus."""

    def __init__(self):
        self.counts: dict[tuple, dict[int, int]] = {}
        self.observed = 0

    def train(self, values: list[int]) -> None:
        for value in values:
            bits = tuple((value >> shift) & 1 for shift in range(7, -1, -1))
            self.observed += 1
            for index in range(8):
                prefix = bits[:index]
                target = bits[index]
                self.counts.setdefault(prefix, {}).setdefault(target, 0)
                self.counts[prefix][target] += 1

    def predict(self, prefix: tuple[int, ...]) -> int:
        options = self.counts.get(prefix)
        if not options:
            return 0
        return max(options.items(), key=lambda item: (item[1], -item[0]))[0]

    def evaluate(self, values: list[int]) -> dict:
        correct = total = 0
        by_prefix: dict[int, list[int]] = {}
        for value in values:
            bits = tuple((value >> shift) & 1 for shift in range(7, -1, -1))
            for index in range(8):
                prediction = self.predict(bits[:index])
                match = int(prediction == bits[index])
                correct += match
                total += 1
                by_prefix.setdefault(index, [0, 0])
                by_prefix[index][0] += match
                by_prefix[index][1] += 1
        return {"accuracy": round(correct / total, 4) if total else 0.0,
                "bits": total,
                "by_prefix": {length: round(data[0] / data[1], 4)
                              for length, data in by_prefix.items()}}
