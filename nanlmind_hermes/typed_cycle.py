"""Typed relation operators for path and loop composition.

This is a small classical prototype of a typed edge-to-cycle layer: each
WordNet relation type carries a 2x2 unitary, and a path composes them in order.
It is not trained on a language objective and is not a physical gauge field.
"""
from __future__ import annotations

import math

from nanlmind_hermes.relation_operators import (
    Matrix,
    dagger,
    identity,
    matmul,
    su2,
)

DEFAULT_OPERATORS = {
    "hypernym": su2((1.0, 0.0, 0.0), math.pi / 4.0),
    "instance_hypernym": su2((1.0, 0.0, 0.0), math.pi / 5.0),
    "hyponym": su2((-1.0, 0.0, 0.0), math.pi / 4.0),
    "part_meronym": su2((0.0, 1.0, 0.0), math.pi / 5.0),
    "member_meronym": su2((0.0, 1.0, 0.0), math.pi / 6.0),
    "attribute": su2((0.0, 0.0, 1.0), math.pi / 6.0),
    "cause": su2((1.0, 1.0, 0.0), math.pi / 3.0),
    "entailment": su2((0.0, 1.0, 1.0), math.pi / 4.0),
    "antonym": su2((0.0, 0.0, 1.0), math.pi),
    "similar_to": su2((1.0, 0.0, 1.0), math.pi / 7.0),
}


class TypedCycleOperators:
    def __init__(self, operators: dict[str, Matrix] | None = None):
        self.operators = dict(DEFAULT_OPERATORS)
        if operators:
            self.operators.update(operators)

    def operator(self, relation: str) -> Matrix:
        if relation in self.operators:
            return self.operators[relation]
        # Deterministic fallback: unknown relations must not silently become identity.
        seed = sum(ord(character) for character in relation)
        return su2((1.0, (seed % 7) / 7.0, (seed % 11) / 11.0),
                   math.pi / (3.0 + seed % 5))

    def path(self, relations: list[str]) -> Matrix:
        result = identity()
        for relation in relations:
            result = matmul(result, self.operator(relation))
        return result

    def reverse_path(self, relations: list[str]) -> Matrix:
        result = identity()
        for relation in reversed(relations):
            result = matmul(result, dagger(self.operator(relation)))
        return result

    def loop_trace(self, relations: list[str]) -> complex:
        product = self.path(relations)
        return product[0][0] + product[1][1]
