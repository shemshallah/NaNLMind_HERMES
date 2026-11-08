"""Boolean-gated self-modification of HERMES scratch leaves and relations.

The engine composes existing HERMES objects; it is not a second trainer:

* ``MTHLayerStack`` / ``MTHHeadMapper`` supply the root Boolean transform;
* ``LayeredLearningCoordinator`` owns L0-L9 scratch pages and relations;
* ``MirrorLeafBindings`` owns versioned MP -> leaf pointers;
* native MTH execution and canonical promotion remain separate gates.

Every mutation is evidence-bearing, head-scoped (1..255), CAS-versioned and
bounded.  NANNET0 (head 0) can never be a mutation target.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .layered_learning import (
    LAYER_NAMES,
    LayeredLearningCoordinator,
    LayeredLearningError,
    _relation,
)
from .mirror_topology import (
    HEADS,
    LAYERS,
    MirrorLeafBindings,
    MirrorTopology,
    MirrorTopologyError,
    NANNET_HEAD,
    pair_heads,
    pair_number,
)
from .mth_layers import MTHHeadMapper, MTHLayerStack, ROOT_NAMES, apply_bits
from .scratch_register import LAYER_NAMES as SCRATCH_LAYERS, parse_register_address

SELF_MODIFICATION_SCHEMA = "nanlm.hermes-self-modification/1"
MTH_TOOL_SCHEMA = "nanlm.hermes-mth-tools/1"
MAX_EVIDENCE_BYTES = 1_048_576
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class SelfModificationError(ValueError):
    pass


class StaleLeafGeneration(SelfModificationError):
    pass


@dataclass(frozen=True)
class SelfModificationPolicy:
    min_boolean_confidence: float = 0.5
    max_strength_q16: int = 65_536
    require_evidence: bool = True
    allow_negative_memory: bool = True
    require_approval_for_promotion: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_boolean_confidence <= 1:
            raise SelfModificationError("min_boolean_confidence must be in 0..1")
        if not 0 <= self.max_strength_q16 <= 65_536:
            raise SelfModificationError("max_strength_q16 must be in 0..65536")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "nanlm.hermes-self-modification-policy/1",
            "min_boolean_confidence": self.min_boolean_confidence,
            "max_strength_q16": self.max_strength_q16,
            "require_evidence": self.require_evidence,
            "allow_negative_memory": self.allow_negative_memory,
            "require_approval_for_promotion": self.require_approval_for_promotion,
        }


def _digest(value: Any) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise SelfModificationError("evidence_sha256 must be lowercase SHA-256")
    return value


def _head(value: Any, name: str = "head") -> int:
    if type(value) is not int or not 0 <= value < HEADS:
        raise SelfModificationError(f"{name} must be an integer 0..255")
    return value


def _source_head(value: Any) -> int:
    value = _head(value, "source_head")
    if value == NANNET_HEAD:
        raise SelfModificationError("NANNET0 cannot be a mutation source")
    return value


def _layer(value: Any) -> int:
    if isinstance(value, str):
        try:
            return SCRATCH_LAYERS.index(value.casefold())
        except ValueError as exc:
            raise SelfModificationError("unknown L0-L9 layer") from exc
    if type(value) is not int or not 0 <= value < LAYERS:
        raise SelfModificationError("layer must be 0..9")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class BooleanGate:
    """Evaluate a relation through a root Boolean truth table."""

    def __init__(self, stack: MTHLayerStack | None = None):
        self.stack = stack or MTHLayerStack()
        self.mapper = MTHHeadMapper(self.stack)

    def evaluate(self, relation: dict[str, str], *, head: int) -> dict[str, Any]:
        head = _source_head(head)
        subject = relation["subject"]
        predicate = relation["predicate"]
        obj = relation["object"]
        mask = (sum(ord(char) for char in predicate)
                + len(subject) + len(obj)) % 16
        subject_value = self.mapper.transform(subject, mask)
        object_value = self.mapper.transform(obj, mask)
        predicate_value = self.mapper.transform(predicate, mask)
        folded = apply_bits(mask, subject_value, object_value, 8)
        truth_bit = (folded >> (head & 7)) & 1
        truth_table = (folded >> (2 * (subject_value & 1) + (object_value & 1))) & 1
        score = (1.0 if truth_bit else 0.0) * 0.7 + (1.0 if truth_table else 0.0) * 0.3
        return {
            "schema": "nanlm.hermes-boolean-gate/1",
            "head": head,
            "mask": mask,
            "root_function": ROOT_NAMES[mask],
            "subject_value": subject_value,
            "predicate_value": predicate_value,
            "object_value": object_value,
            "folded_value": folded,
            "truth_bit": truth_bit,
            "truth_table_bit": truth_table,
            "confidence": round(score, 6),
            "allowed": score >= 0.5,
        }


class MthToolFacade:
    """Bounded local MTH planning facade over the ten L0-L9 coordinates.

    It exposes the same six read/compose tool names as the NaNROUTER HERMES MTH
    catalog.  It never claims native execution; ``nanlm_native_mth_u32`` and
    the NANNET lease remain the execution path.
    """

    def __init__(self, mapper: MTHHeadMapper | None = None):
        self.mapper = mapper or MTHHeadMapper(MTHLayerStack())

    @staticmethod
    def definitions() -> tuple[dict[str, str], ...]:
        return (
            {"name": "mth.catalog", "description": "List the ten L0-L9 coordinates and root Boolean functions."},
            {"name": "mth.inspect", "description": "Inspect one L0-L9 coordinate and its learned signal."},
            {"name": "mth.categories", "description": "List layer families and Boolean root families."},
            {"name": "mth.compose", "description": "Validate a bounded multi-layer plan without executing it."},
            {"name": "mth.prepare", "description": "Prepare one typed L0-L9 plan."},
            {"name": "mth.route", "description": "Resolve HW0:<head>::L<n> or HW0:MP<n>::L<n>."},
        )

    def invoke(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        if tool == "mth.catalog":
            return self._catalog(args)
        if tool == "mth.inspect":
            return self._inspect(args)
        if tool == "mth.categories":
            return self._categories()
        if tool == "mth.compose":
            return self._compose(args)
        if tool == "mth.prepare":
            return self._prepare(args)
        if tool == "mth.route":
            return self._route(args)
        raise SelfModificationError("unknown MTH tool")

    def _catalog(self, args: dict[str, Any]) -> dict[str, Any]:
        offset = int(args.get("offset", 0))
        limit = min(max(int(args.get("limit", 32)), 1), 100)
        rows = [
            {"layer": index, "name": name,
             "tool": f"mth.L{index}.{name}"}
            for index, name in enumerate(LAYER_NAMES)
        ]
        return {"schema": MTH_TOOL_SCHEMA, "tool": "mth.catalog",
                "total": len(rows), "offset": offset,
                "layers": rows[offset:offset + limit],
                "root_functions": list(ROOT_NAMES.values())}

    def _inspect(self, args: dict[str, Any]) -> dict[str, Any]:
        layer = _layer(args.get("layer"))
        return {"schema": MTH_TOOL_SCHEMA, "tool": "mth.inspect",
                "layer": layer, "name": LAYER_NAMES[layer],
                "root_functions": list(ROOT_NAMES.values()),
                "native_execution": False}

    def _categories(self) -> dict[str, Any]:
        return {"schema": MTH_TOOL_SCHEMA, "tool": "mth.categories",
                "layers": list(LAYER_NAMES),
                "root_functions": list(ROOT_NAMES.values()),
                "execution_available_here": False}

    def _compose(self, args: dict[str, Any]) -> dict[str, Any]:
        plan = args.get("plan")
        if not isinstance(plan, list) or not 1 <= len(plan) <= 64:
            raise SelfModificationError("plan must be a list of 1..64 layers")
        layers = [_layer(item.get("layer") if isinstance(item, dict) else item)
                  for item in plan]
        plan_id = str(args.get("plan_id", "hermes.mth"))
        if _PLAN_ID.fullmatch(plan_id) is None:
            raise SelfModificationError("plan_id is malformed")
        document = {"schema": "nanlm.hermes-mth-plan/1", "plan_id": plan_id,
                    "layers": layers, "steps": len(layers)}
        return {"schema": MTH_TOOL_SCHEMA, "tool": "mth.compose",
                "plan": document,
                "plan_digest": hashlib.sha256(_canonical(document)).hexdigest(),
                "native_execution": False}

    def _prepare(self, args: dict[str, Any]) -> dict[str, Any]:
        layer = _layer(args.get("layer"))
        return self.invoke("mth.compose", {
            "plan": [{"layer": layer}],
            "plan_id": args.get("plan_id", "hermes.mth.prepare"),
        })

    def _route(self, args: dict[str, Any]) -> dict[str, Any]:
        address = args.get("address")
        if not isinstance(address, str) or len(address) > 256:
            raise SelfModificationError("route address is required")
        # Reuse the recursive L0-L9 route parser.
        try:
            parsed = MirrorTopology.parse_route(address)
        except MirrorTopologyError as exc:
            raise SelfModificationError(str(exc)) from exc
        return {"schema": MTH_TOOL_SCHEMA, "tool": "mth.route",
                "route": parsed, "native_execution": False}


class SelfModificationEngine:
    def __init__(
        self,
        coordinator: LayeredLearningCoordinator,
        bindings: MirrorLeafBindings,
        *,
        policy: SelfModificationPolicy | None = None,
        gate: BooleanGate | None = None,
        mth: MthToolFacade | None = None,
    ):
        if not isinstance(coordinator, LayeredLearningCoordinator):
            raise SelfModificationError("layered learning coordinator required")
        if not isinstance(bindings, MirrorLeafBindings):
            raise SelfModificationError("mirror leaf bindings required")
        self.coordinator = coordinator
        self.bindings = bindings
        self.policy = policy or SelfModificationPolicy()
        self.gate = gate or BooleanGate(coordinator.stack)
        self.mth = mth or MthToolFacade(coordinator.mapper)

    def evaluate(self, relation: dict[str, Any], *, head: int) -> dict[str, Any]:
        parsed = _relation(relation)
        if parsed is None:
            raise SelfModificationError("subject, predicate and object are required")
        return self.gate.evaluate(parsed, head=head)

    def use_mth_tool(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.mth.invoke(tool, arguments)

    def propose_relation(
        self,
        relation: dict[str, Any],
        *,
        layer: int | str = "lexical",
        source_head: int,
        evidence_sha256: str,
        strength_q16: int = 32_768,
        polarity: int = 1,
        expected_generation: int = 0,
        tick: int = 0,
    ) -> dict[str, Any]:
        parsed = _relation(relation)
        if parsed is None:
            raise SelfModificationError("subject, predicate and object are required")
        source_head = _source_head(source_head)
        layer_number = _layer(layer)
        evidence = _digest(evidence_sha256) if self.policy.require_evidence else str(evidence_sha256)
        if type(strength_q16) is not int or not 0 <= strength_q16 <= self.policy.max_strength_q16:
            raise SelfModificationError("strength_q16 is outside the policy bound")
        if polarity not in (-1, 0, 1):
            raise SelfModificationError("polarity must be -1, 0 or 1")
        if type(expected_generation) is not int or expected_generation < 0:
            raise SelfModificationError("expected_generation must be >= 0")

        boolean = self.gate.evaluate(parsed, head=source_head)
        effective_polarity = polarity
        effective_strength = strength_q16
        if not boolean["allowed"]:
            if not self.policy.allow_negative_memory:
                raise SelfModificationError("Boolean gate rejected the mutation")
            effective_polarity = -1
            effective_strength = max(0, strength_q16 - 16_384)

        mapped = self.coordinator.mapper.map_relation({
            **parsed, "kind": "relation",
            "provenance": {"resolution": self.coordinator._resolution_for(parsed)},
        })
        leaf_id = f"self:L{layer_number}:H{source_head}:{mapped['digest']}"
        leaf_payload = {
            "schema": SELF_MODIFICATION_SCHEMA,
            "relation": parsed,
            "layer": layer_number,
            "source_head": source_head,
            "boolean": boolean,
            "polarity": effective_polarity,
            "strength_q16": effective_strength,
            "evidence_sha256": evidence,
            "mth_leaf": {
                "head": mapped["head"], "mirror": mapped["mirror"],
                "mask": mapped["mask"], "value": mapped["value"],
                "layer": mapped["layer"],
            },
        }
        leaf_digest = hashlib.sha256(_canonical(leaf_payload)).hexdigest()
        scratch = self.coordinator.space.append(
            layer=layer_number, head=source_head, leaf_id=leaf_id,
            payload=leaf_payload, kind="self-modification",
        )
        observation = self.coordinator.observe(
            " ".join((parsed["subject"], parsed["predicate"], parsed["object"])),
            source="self-modification",
            relations=[{
                **parsed,
                "polarity": effective_polarity,
                "strength_q16": effective_strength,
                "evidence_sha256": evidence,
            }],
            tick=tick,
        )
        binding = None
        target_head = mapped["head"] & 0xFF
        if target_head != source_head:
            mp = pair_number(source_head, target_head)
            try:
                binding = self.bindings.bind(
                    mp=mp, leaf_id=leaf_id, leaf_version=scratch["generation"],
                    leaf_digest=leaf_digest, source_head=source_head,
                    expected_generation=expected_generation, tick=tick,
                )
            except MirrorTopologyError as exc:
                raise SelfModificationError(str(exc)) from exc
        return {
            "schema": SELF_MODIFICATION_SCHEMA,
            "status": "mutated",
            "relation": parsed,
            "layer": layer_number,
            "source_head": source_head,
            "boolean": boolean,
            "effective_polarity": effective_polarity,
            "effective_strength_q16": effective_strength,
            "scratch": scratch,
            "leaf_digest": leaf_digest,
            "binding": binding,
            "observation_turn": observation["turn"],
            "mth_leaf": mapped,
            "policy": self.policy.to_dict(),
            "canonical_training_committed": False,
        }

    def modify_leaf(
        self,
        *,
        mp: int,
        leaf_id: str,
        payload: Any,
        source_head: int,
        evidence_sha256: str,
        expected_generation: int,
        tick: int = 0,
    ) -> dict[str, Any]:
        source_head = _source_head(source_head)
        _digest(evidence_sha256)
        current = self.bindings.resolve(mp)
        if current is None:
            if expected_generation != 0:
                raise SelfModificationError("new leaf must expect generation zero")
        elif current["generation"] != expected_generation:
            raise SelfModificationError("stale leaf generation")
        head_a, head_b = pair_heads(mp)
        target = head_b if source_head == head_a else head_a
        layer = 5
        boolean = self.gate.evaluate(
            {"subject": str(head_a), "predicate": "relates_to", "object": str(head_b)},
            head=source_head,
        )
        leaf_digest = hashlib.sha256(_canonical(payload)).hexdigest()
        scratch = self.coordinator.space.append(
            layer=layer, head=source_head, leaf_id=leaf_id,
            payload={"schema": SELF_MODIFICATION_SCHEMA, "payload": payload,
                     "boolean": boolean, "evidence_sha256": evidence_sha256,
                     "mp": mp, "target_head": target},
            kind="leaf-version",
        )
        binding = self.bindings.bind(
            mp=mp, leaf_id=leaf_id, leaf_version=scratch["generation"],
            leaf_digest=leaf_digest, source_head=source_head,
            expected_generation=expected_generation, tick=tick,
        )
        return {
            "schema": SELF_MODIFICATION_SCHEMA,
            "status": "leaf-modified",
            "mp": mp,
            "heads": list(pair_heads(mp)),
            "scratch": scratch,
            "boolean": boolean,
            "binding": binding,
            "leaf_digest": leaf_digest,
            "canonical_training_committed": False,
        }

    def promote(self, *, approved: bool = False) -> dict[str, Any]:
        if self.policy.require_approval_for_promotion and not approved:
            return {"schema": SELF_MODIFICATION_SCHEMA, "status": "approval_required",
                    "policy": self.policy.to_dict()}
        return self.coordinator.synchronize(approved=approved)

    def status(self) -> dict[str, Any]:
        return {
            "schema": SELF_MODIFICATION_SCHEMA,
            "policy": self.policy.to_dict(),
            "topology": MirrorTopology.status(),
            "bindings": self.bindings.stats(),
            "scratch": self.coordinator.space.stats(),
            "mth_tools": list(self.mth.definitions()),
            "nannet0_writable": False,
        }


__all__ = [
    "BooleanGate",
    "MthToolFacade",
    "SELF_MODIFICATION_SCHEMA",
    "SelfModificationEngine",
    "SelfModificationError",
    "SelfModificationPolicy",
    "StaleLeafGeneration",
]
