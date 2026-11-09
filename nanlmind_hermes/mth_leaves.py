"""Immutable MTH data leaves, hubs and constrained pairing.

Every leaf carries a function definition, action relations, mathematical
identities, constraints/constants and a hub ancestry. A leaf also has a real
NaN16 executable image (``.nan``) containing its metadata and payload, so the
data object can be run/inspected on the NaNVM0 boundary without a compiler.

Pairing is deliberately conservative:

* ``::`` with mode ``connect`` creates a relation leaf linking two leaves.
* ``::`` with an explicit mathematical operator (dot, cosine, add, sub) runs
  the admitted native vector kernel, checks invariants, and creates a new
  immutable hash leaf.
* Unknown/incompatible kinds, unknown operators, failed native execution or a
  violated invariant return an error and create/execute nothing.

There is no in-place mutation API. New evidence creates a new leaf; the old
leaf remains immutable. This is the self-modification guard for the data tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
from typing import Any

from .nanvm import MEM_WORDS as NANVM_WORDS, NanVM, make_word, parse_nan_image

STORE_SCHEMA = "nanlm.mth-leaf-store/1"
LEAF_SCHEMA = "nanlm.mth-leaf/1"
PAIR_SCHEMA = "nanlm.mth-pair/1"
HUB_SCHEMA = "nanlm.mth-hub/1"
AUDIT_SCHEMA = "nanlm.mth-leaf-audit/1"
HUB_TREE_LIMIT = 64

ROOT_HUB = "hub:mth.root"
VECTOR_HUB = "hub:mth.vector"
SCALAR_HUB = "hub:mth.scalar"
MATRIX_HUB = "hub:mth.matrix"
RELATION_HUB = "hub:mth.relation"
OPERATOR_HUB = "hub:mth.operator"
WORDNET_HUB = "hub:mth.wordnet"
CONSTRAINT_HUB = "hub:mth.constraints"
FUNCTION_HUB = "hub:mth.function"
FAMILY_HUB_PREFIX = "hub:mth.function."

WORDNET_ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW4:0"
WORDNET_VECTOR_REF = WORDNET_ROUTE + "::word/vector"
HUB_IDS = (ROOT_HUB, VECTOR_HUB, SCALAR_HUB, MATRIX_HUB, RELATION_HUB,
           OPERATOR_HUB, WORDNET_HUB, CONSTRAINT_HUB, FUNCTION_HUB)
KIND_TO_HUB = {
    "vector": VECTOR_HUB,
    "scalar": SCALAR_HUB,
    "matrix": MATRIX_HUB,
    "relation": RELATION_HUB,
    "kernel": FUNCTION_HUB,
}
VECTOR_CONSTRAINTS = (
    "vector.dimension == 256",
    "vector.precision == float32",
    "vector.immutable_inputs == true",
    "vector.no_dense_per_word_matrix == true",
    "vector.hub_parent == hub:mth.vector",
)
VECTOR_IDENTITIES = (
    "add(x,y) == add(y,x)",
    "add(add(x,y),z) == add(x,add(y,z))",
    "add(x,zero) == x",
    "sub(x,x) == zero",
    "dot(x,y) == dot(y,x)",
    "dot(x,x) >= 0",
    "cosine(x,y) == cosine(y,x)",
    "cosine(x,y) in [-1,1]",
)
OPERATOR_RULES = {
    "connect": {
        "input_kinds": ("vector", "vector"),
        "output_kind": "relation",
        "kernel": None,
        "identities": ("connect(a,b) is symmetric as an undirected relation",),
        "constraints": ("no numerical execution",),
    },
    "dot": {
        "input_kinds": ("vector", "vector"),
        "output_kind": "scalar",
        "kernel": "mth.vector.dot256.f32",
        "program_id": 900, "operator_id": 77,
        "identities": ("dot(x,y) == dot(y,x)", "dot(x,x) >= 0"),
        "constraints": ("dimension == 256", "finite inputs", "finite result"),
    },
    "cosine": {
        "input_kinds": ("vector", "vector"),
        "output_kind": "scalar",
        "kernel": "mth.vector.cosine256.f32",
        "program_id": 901, "operator_id": 78,
        "identities": ("cosine(x,y) == cosine(y,x)", "cosine in [-1,1]"),
        "constraints": ("dimension == 256", "finite inputs", "zero vector -> 0"),
    },
    "add": {
        "input_kinds": ("vector", "vector"),
        "output_kind": "vector",
        "kernel": "mth.vector.add256.f32",
        "program_id": 902, "operator_id": 79,
        "identities": ("add(x,y) == add(y,x)", "add(x,zero) == x"),
        "constraints": ("dimension == 256", "finite inputs", "finite result"),
    },
    "sub": {
        "input_kinds": ("vector", "vector"),
        "output_kind": "vector",
        "kernel": "mth.vector.sub256.f32",
        "program_id": 903, "operator_id": 80,
        "identities": ("sub(x,y) == -sub(y,x)", "sub(x,x) == zero"),
        "constraints": ("dimension == 256", "finite inputs", "finite result"),
    },
}
PAIR_RULES = {("vector", "vector"): tuple(OPERATOR_RULES)}
_LOCK = threading.RLock()
_CACHE: dict[str, "LeafStore"] = {}


class MthLeafError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "error": str(self),
                "details": self.details}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(values: list[float]) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(float(value))
               for value in values)


def _vector(values: Any, label: str = "vector") -> list[float]:
    if not isinstance(values, list) or len(values) != 256:
        raise MthLeafError(label + " must contain exactly 256 values", code="invalid_dimension")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MthLeafError(label + " values must be numeric", code="invalid_value")
        value = float(value)
        if not math.isfinite(value):
            raise MthLeafError(label + " values must be finite", code="non_finite")
        out.append(value)
    return out


def _f32_bits(values: list[float]) -> list[int]:
    return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _nan_line(payload: int, klass: int = 15) -> str:
    word = make_word(payload & 0xFFFFFFFF, klass)
    near = (word >> 48) & 0xFFFF
    far_le = word & 0xFFFF
    return "%04X_%08X_%04X" % (near, (word >> 16) & 0xFFFFFFFF, far_le)


def _leaf_nan_image(leaf: dict[str, Any]) -> dict[str, Any]:
    """Build a real NaN16 image: PUTC 'L' then HALT, with metadata/data words."""
    code = [
        _nan_line(0x0140004C),  # MOVI r4, 76 ('L')
        _nan_line(0x15400000),  # PUTC r4
        _nan_line(0xFF000000),  # HALT
    ]
    data: list[int] = []
    data.append(0x4C454146)  # 'LEAF'
    data.append(1)           # format version
    data.extend(struct.unpack("<8I", bytes.fromhex(leaf["content_sha256"])))
    metadata = _canonical({
        "leaf_id": leaf["leaf_id"], "kind": leaf["kind"], "hub": leaf["hub"],
        "parent": leaf["parent"], "function_definition": leaf["function_definition"],
        "action_relations": leaf["action_relations"],
        "identities": leaf["identities"], "constraints": leaf["constraints"],
        "constants": leaf["constants"], "wordnet": leaf["wordnet"],
    })
    data.append(len(metadata))
    for offset in range(0, len(metadata), 4):
        chunk = metadata[offset:offset + 4]
        data.append(int.from_bytes(chunk.ljust(4, b"\0"), "little"))
    payload = leaf.get("payload") or {}
    values = payload.get("values")
    if isinstance(values, list) and values and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
        data.append(len(values))
        data.extend(_f32_bits([float(value) for value in values]))
    elif isinstance(payload.get("value"), (int, float)) and not isinstance(payload.get("value"), bool):
        data.append(1)
        data.extend(_f32_bits([float(payload["value"])]))
    else:
        raw = _canonical(payload)
        data.append(len(raw))
        for offset in range(0, len(raw), 4):
            chunk = raw[offset:offset + 4]
            data.append(int.from_bytes(chunk.ljust(4, b"\0"), "little"))
    while len(code) < 0x100:
        code.append(_nan_line(0))
    image = code + [_nan_line(word) for word in data]
    text = "\n".join(image) + "\n"
    return {"text": text, "words": len(image),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def leaf_image_text(leaf: dict[str, Any]) -> str:
    return _leaf_nan_image(leaf)["text"]


def verify_leaf(leaf: dict[str, Any]) -> dict[str, Any]:
    """Execute the leaf image on the Python NaNVM0 boundary."""
    image = _leaf_nan_image(leaf)
    words = parse_nan_image(image["text"])
    if len(words) > NANVM_WORDS:
        raise MthLeafError("leaf image exceeds the Python NaNVM0 boundary",
                           code="image_too_large",
                           details={"words": len(words), "limit": NANVM_WORDS})
    vm = NanVM(words)
    vm.run(max_steps=1000)
    return {
        "schema": LEAF_SCHEMA,
        "leaf_id": leaf["leaf_id"],
        "nan_executable": vm.output.strip() == "L",
        "output": vm.output,
        "steps": vm.steps,
        "words": len(words),
        "image_sha256": image["sha256"],
        "boundary": "python-nanvm0",
        "native": False,
    }


def _hub_definitions() -> dict[str, dict[str, Any]]:
    return {
        ROOT_HUB: {
            "title": "MTH root data-leaf hub",
            "payload": {"defines": "root", "scope": "all MTH data leaves"},
            "identities": ("every leaf has one hub ancestry chain to this root",),
            "constraints": ("no leaf may rewrite a hub", "no leaf may delete a leaf"),
            "constants": {"schema": LEAF_SCHEMA, "immutable": True},
            "wordnet": {"route": WORDNET_ROUTE, "status": "declared"},
        },
        VECTOR_HUB: {
            "title": "Vector root hub",
            "payload": {"defines": "vector", "dimension": 256, "precision": "float32"},
            "identities": VECTOR_IDENTITIES,
            "constraints": VECTOR_CONSTRAINTS,
            "constants": {"dimension": 256, "dtype": "float32", "dense_per_word": False},
            "wordnet": {"route": WORDNET_ROUTE, "object_ref": WORDNET_VECTOR_REF,
                        "term": "vector", "status": "declared",
                        "note": "resolve the lexical definition through HW4; it is evidence, not authority"},
        },
        SCALAR_HUB: {
            "title": "Scalar root hub",
            "payload": {"defines": "scalar", "dimension": 0, "precision": "float32"},
            "identities": ("scalar identity and commutativity apply through the operator rule",),
            "constraints": ("result must be finite",),
            "constants": {"dtype": "float32"},
            "wordnet": {"route": WORDNET_ROUTE, "term": "scalar", "status": "declared"},
        },
        MATRIX_HUB: {
            "title": "Matrix root hub",
            "payload": {"defines": "matrix", "rows": 256, "columns": 256},
            "identities": ("matrix identity/associativity are operator-specific",),
            "constraints": ("no dense per-word allocation",),
            "constants": {"rows": 256, "columns": 256},
            "wordnet": {"route": WORDNET_ROUTE, "term": "matrix", "status": "declared"},
        },
        RELATION_HUB: {
            "title": "Relation root hub",
            "payload": {"defines": "relation"},
            "identities": ("connect(a,b) is an undirected relation",),
            "constraints": ("a relation leaf never executes a numerical kernel",),
            "constants": {},
            "wordnet": {"route": WORDNET_ROUTE, "term": "relation", "status": "declared"},
        },
        OPERATOR_HUB: {
            "title": "Operator root hub",
            "payload": {"defines": "operator", "operators": sorted(OPERATOR_RULES)},
            "identities": ("an operator is admitted only by its hub rule",),
            "constraints": ("unknown operators deny", "operator result creates a new leaf"),
            "constants": {"operators": sorted(OPERATOR_RULES)},
            "wordnet": {"route": WORDNET_ROUTE, "term": "operator", "status": "declared"},
        },
        WORDNET_HUB: {
            "title": "WordNet HW4 bridge hub",
            "payload": {"defines": "wordnet-link", "route": WORDNET_ROUTE},
            "identities": ("WordNet definitions are lexical evidence, not execution authority",),
            "constraints": ("never allocate a dense matrix per word", "missing word fails closed"),
            "constants": {"route": WORDNET_ROUTE, "link": "HW4:0", "nannet_port": "BS0:0"},
            "wordnet": {"route": WORDNET_ROUTE, "term": "wordnet", "status": "declared"},
        },
        CONSTRAINT_HUB: {
            "title": "Constraints and invariants hub",
            "payload": {"defines": "constraint", "invariants": VECTOR_IDENTITIES},
            "identities": VECTOR_IDENTITIES,
            "constraints": VECTOR_CONSTRAINTS,
            "constants": {"deny_by_default": True, "immutable": True},
            "wordnet": {"route": WORDNET_ROUTE, "term": "constraint", "status": "declared"},
        },
        FUNCTION_HUB: {
            "title": "MTH function/kernel root hub",
            "payload": {"defines": "function", "kernels": 91},
            "identities": ("every bundled kernel has exactly one function leaf",),
            "constraints": ("kernel source and fixture digests are pinned",
                            "catalog status and CI claim are recorded, not proven",
                            "kernel leaves are immutable"),
            "constants": {"kernel_count": 91, "catalog": "wave-capabilities.v1"},
            "wordnet": {"route": WORDNET_ROUTE, "term": "function", "status": "declared"},
        },
    }


@dataclass(frozen=True)
class PairResult:
    status: str
    hashleaf: str | None
    leaf: dict[str, Any] | None
    error: dict[str, Any] | None


class LeafStore:
    def __init__(self, state_dir: str | Path):
        self.root = Path(state_dir).expanduser() / "mth_leaves"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.index_path = self.root / "index.json"
        self.leaves_dir = self.root / "leaves"
        self.leaves_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.audit_path = self.root / "audit.jsonl"
        self._lock = threading.RLock()
        self._data = self._load()
        self._seed()

    def _load(self) -> dict[str, Any]:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("schema") == STORE_SCHEMA:
                    if not isinstance(data.get("leaves"), dict):
                        data["leaves"] = {}
                    if not isinstance(data.get("aliases"), dict):
                        data["aliases"] = {}
                    if not isinstance(data.get("order"), list):
                        data["order"] = []
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {"schema": STORE_SCHEMA, "generation": 0, "leaves": {},
                "aliases": {}, "order": [], "root_hub": ROOT_HUB}

    def _save(self) -> None:
        self._data["generation"] = int(self._data.get("generation", 0)) + 1
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, self.index_path)

    def _audit(self, action: str, outcome: str, **fields) -> None:
        record = {"schema": AUDIT_SCHEMA, "action": action, "outcome": outcome,
                  "at": _now(), **fields}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _hub_leaf(self, hub_id: str, definition: dict[str, Any],
                  *, parent: str | None = None,
                  save: bool = True) -> dict[str, Any]:
        existing = self._data["leaves"].get(hub_id)
        if existing is not None:
            return existing
        parent_id = None if hub_id == ROOT_HUB else (parent or ROOT_HUB)
        leaf = {
            "schema": LEAF_SCHEMA,
            "leaf_id": hub_id,
            "kind": "hub",
            "hub": hub_id if hub_id != ROOT_HUB else ROOT_HUB,
            "parent": parent_id,
            "generation": 0,
            "immutable": True,
            "payload": {"hub": hub_id, **definition["payload"]},
            "function_definition": {"kind": "hub",
                                    "defines": definition["payload"].get("defines")},
            "action_relations": ([{"action": "root_hub", "target": ROOT_HUB}]
                                 if hub_id == ROOT_HUB else
                                 [{"action": "specializes", "target": parent_id}]),
            "identities": list(definition["identities"]),
            "constraints": list(definition["constraints"]),
            "constants": definition["constants"],
            "wordnet": definition["wordnet"],
            "provenance": {"created_at": _now(), "source": "seed"},
        }
        leaf["content_sha256"] = _sha256({
            key: value for key, value in leaf.items()
            if key not in {"content_sha256", "nan_image"}})
        leaf["nan_image"] = self._write_image(leaf)
        self._data["leaves"][hub_id] = leaf
        self._data["order"].append(hub_id)
        if save:
            self._save()
        return leaf

    def _seed(self) -> None:
        with self._lock:
            hubs = _hub_definitions()
            changed = False
            for hub_id, definition in hubs.items():
                if hub_id in self._data["leaves"]:
                    continue
                self._hub_leaf(hub_id, definition, save=False)
                changed = True
            if changed:
                self._save()
                self._audit("seed", "ok", hubs=list(hubs))

    def _write_image(self, leaf: dict[str, Any]) -> dict[str, Any]:
        image = _leaf_nan_image(leaf)
        target = self.leaves_dir / (leaf["leaf_id"].replace(":", "_") + ".nan")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(image["text"], encoding="utf-8")
        os.replace(temporary, target)
        return {"path": str(target), "sha256": image["sha256"],
                "words": image["words"], "bytes": len(image["text"].encode("utf-8")),
                "format": "NaN16 .nan text", "python_nanvm0": image["words"] <= NANVM_WORDS}

    def _new_leaf(self, *, kind: str, payload: dict[str, Any],
                  function_definition: dict[str, Any],
                  action_relations: list[dict[str, Any]],
                  identities: list[str], constraints: list[str],
                  constants: dict[str, Any], wordnet: dict[str, Any],
                  parent: str | None, alias: str | None,
                  save: bool = True) -> dict[str, Any]:
        if alias and (alias in self._data["aliases"] or alias in self._data["leaves"]):
            raise MthLeafError("alias already exists: " + alias, code="alias_conflict")
        hub = KIND_TO_HUB[kind]
        core = {
            "schema": LEAF_SCHEMA,
            "kind": kind,
            "hub": hub,
            "parent": parent,
            "generation": 0,
            "immutable": True,
            "payload": payload,
            "function_definition": function_definition,
            "action_relations": action_relations,
            "identities": identities,
            "constraints": constraints,
            "constants": constants,
            "wordnet": wordnet,
            # Volatile provenance is added after the content hash so identical
            # semantic leaves are idempotent across restarts.
            "provenance": {"source": "mth-leaf-store/1"},
        }
        digest = _sha256(core)
        leaf_id = "leaf:" + digest[:32]
        if leaf_id in self._data["leaves"]:
            return self._data["leaves"][leaf_id]
        leaf = {**core, "leaf_id": leaf_id, "content_sha256": digest,
                "provenance": {**core["provenance"], "created_at": _now()}}
        leaf["nan_image"] = self._write_image(leaf)
        self._data["leaves"][leaf_id] = leaf
        self._data["order"].append(leaf_id)
        if alias:
            self._data["aliases"][alias] = leaf_id
        if save:
            self._save()
        return leaf

    def ensure_function_tree(self) -> dict[str, Any]:
        """Map every bundled MTH kernel to an immutable function leaf.

        Each kernel gets one family hub under ``hub:mth.function``, one kernel
        leaf under that family, aliases ``kernel:<kernel_id>``, links to the
        operator/function/WordNet/data hubs, and the manifest source/fixture
        digests. Mapping is idempotent and never rewrites an existing leaf.
        """
        from . import mth_catalog

        with self._lock:
            definitions = _hub_definitions()
            if FUNCTION_HUB not in self._data["leaves"]:
                self._hub_leaf(FUNCTION_HUB, definitions[FUNCTION_HUB], save=False)
            health = mth_catalog.health()
            for family in sorted(health["families"]):
                hub_id = FAMILY_HUB_PREFIX + family
                if hub_id in self._data["leaves"]:
                    continue
                definition = {
                    "title": "MTH function family hub: " + family,
                    "payload": {"defines": "function-family", "family": family,
                                "kernel_count": health["families"][family]},
                    "identities": (
                        "every kernel in this family has exactly one function leaf",
                        "family membership is pinned by the bundled manifest",
                    ),
                    "constraints": (
                        "source and fixture digests are immutable",
                        "catalog membership is not a proof or execution permission",
                    ),
                    "constants": {"family": family,
                                  "kernel_count": health["families"][family]},
                    "wordnet": {"route": WORDNET_ROUTE, "term": family,
                                "status": "declared"},
                }
                self._hub_leaf(hub_id, definition, parent=FUNCTION_HUB, save=False)
            linked = 0
            for kernel in mth_catalog.kernels():
                alias = "kernel:" + kernel.kernel_id
                if alias in self._data["aliases"]:
                    continue
                family_hub = FAMILY_HUB_PREFIX + kernel.family
                data_hub = (MATRIX_HUB if len(kernel.shape) >= 2
                            else VECTOR_HUB if len(kernel.shape) == 1
                            else SCALAR_HUB)
                identities = [
                    "kernel_id == " + kernel.kernel_id,
                    "precision == " + kernel.precision,
                    "shape == " + repr(list(kernel.shape)),
                    "workspace_words <= " + str(max(kernel.workspace_words, 0)),
                    "status == " + kernel.status,
                    "ci_verified == " + ("true" if kernel.ci_verified else "false"),
                    "marker == " + kernel.marker,
                    "source_digest is immutable",
                    "fixture_digest is immutable",
                ]
                if kernel.family == "vector":
                    identities.append("vector operator identities are inherited from hub:mth.vector")
                if kernel.family == "graph":
                    identities.append("graph contract is declared by manifest shape and workspace")
                constraints = [
                    "fixture must emit its declared marker",
                    "no canonical training, NANNET route, lease or merge commit",
                    "no arbitrary shell or caller native function pointer",
                    "workspace_words <= " + str(max(kernel.workspace_words, 0)),
                ]
                function_definition = {
                    "kind": "native_kernel",
                    "kernel_id": kernel.kernel_id,
                    "family": kernel.family,
                    "execution_mode": kernel.execution_mode,
                    "source": kernel.source,
                    "fixture": kernel.fixture,
                    "source_digest": kernel.source_digest,
                    "fixture_digest": kernel.fixture_digest,
                    "source_bytes": kernel.source_bytes,
                    "fixture_bytes": kernel.fixture_bytes,
                    "marker": kernel.marker,
                    "steps": kernel.steps,
                }
                self._new_leaf(
                    kind="kernel",
                    payload={"kernel_id": kernel.kernel_id,
                             "family": kernel.family,
                             "precision": kernel.precision,
                             "shape": list(kernel.shape),
                             "workspace_words": kernel.workspace_words},
                    function_definition=function_definition,
                    action_relations=[
                        {"action": "member_of_family", "target": family_hub},
                        {"action": "member_of_function_root", "target": FUNCTION_HUB},
                        {"action": "uses_operator_hub", "target": OPERATOR_HUB},
                        {"action": "operates_on", "target": data_hub},
                        {"action": "wordnet_link", "target": WORDNET_HUB},
                        {"action": "constraints", "target": CONSTRAINT_HUB},
                    ],
                    identities=identities,
                    constraints=constraints,
                    constants={"precision": kernel.precision,
                               "shape": list(kernel.shape),
                               "workspace_words": kernel.workspace_words,
                               "status": kernel.status,
                               "ci_verified": kernel.ci_verified,
                               "execution_mode": kernel.execution_mode},
                    wordnet={"route": WORDNET_ROUTE, "term": kernel.family,
                             "status": "declared"},
                    parent=family_hub, alias=alias, save=False)
                linked += 1
            self._save()
            self._audit("map_function_tree", "ok", linked=linked,
                        expected=health["total"], families=len(health["families"]))
            return self.function_tree_audit()

    def function_tree_audit(self, *, verify_images: bool = True) -> dict[str, Any]:
        from . import mth_catalog

        expected = {kernel.kernel_id: kernel for kernel in mth_catalog.kernels()}
        linked: dict[str, str] = {}
        duplicate_aliases = []
        for alias, leaf_id in self._data["aliases"].items():
            if not alias.startswith("kernel:"):
                continue
            kernel_id = alias[len("kernel:"):]
            if kernel_id in linked:
                duplicate_aliases.append(kernel_id)
                continue
            linked[kernel_id] = leaf_id
        missing = sorted(set(expected) - set(linked))
        extra = sorted(set(linked) - set(expected))
        dangling_aliases = []
        digest_mismatches = []
        orphans = []
        ancestry_failures = []
        image_failures = []
        image_ok = 0
        family_counts: dict[str, int] = {}
        for kernel_id, leaf_id in sorted(linked.items()):
            leaf = self._data["leaves"].get(leaf_id)
            if leaf is None:
                dangling_aliases.append(kernel_id)
                continue
            kernel = expected.get(kernel_id)
            if kernel is None:
                continue
            family_hub = FAMILY_HUB_PREFIX + kernel.family
            family_counts[kernel.family] = family_counts.get(kernel.family, 0) + 1
            definition = leaf.get("function_definition") or {}
            if (definition.get("source_digest") != kernel.source_digest
                    or definition.get("fixture_digest") != kernel.fixture_digest):
                digest_mismatches.append(kernel_id)
            if leaf.get("parent") != family_hub or family_hub not in self._data["leaves"]:
                orphans.append(kernel_id)
            family_leaf = self._data["leaves"].get(family_hub, {})
            function_root = self._data["leaves"].get(FUNCTION_HUB, {})
            if (family_leaf.get("parent") != FUNCTION_HUB
                    or function_root.get("parent") != ROOT_HUB):
                ancestry_failures.append(kernel_id)
            if verify_images:
                try:
                    if verify_leaf(leaf)["nan_executable"]:
                        image_ok += 1
                    else:
                        image_failures.append(kernel_id)
                except Exception:
                    image_failures.append(kernel_id)
        complete = not (missing or extra or duplicate_aliases or dangling_aliases
                        or digest_mismatches or orphans or ancestry_failures
                        or image_failures)
        return {
            "schema": "nanlm.mth-function-tree-audit/1",
            "expected": len(expected),
            "linked": len(linked),
            "missing": missing,
            "extra": extra,
            "duplicate_aliases": sorted(set(duplicate_aliases)),
            "dangling_aliases": sorted(set(dangling_aliases)),
            "digest_mismatches": digest_mismatches,
            "orphans": orphans,
            "ancestry_failures": ancestry_failures,
            "image_failures": image_failures,
            "images_verified": image_ok,
            "family_counts": dict(sorted(family_counts.items())),
            "complete": complete,
            "root": FUNCTION_HUB,
        }

    def resolve(self, reference: str) -> dict[str, Any]:
        if reference in self._data["leaves"]:
            return self._data["leaves"][reference]
        target = self._data["aliases"].get(reference)
        if target and target in self._data["leaves"]:
            return self._data["leaves"][target]
        raise MthLeafError("unknown MTH leaf: " + str(reference), code="unknown_leaf")

    def create(self, *, kind: str, payload: dict[str, Any] | None = None,
               function_definition: dict[str, Any] | None = None,
               action_relations: list[dict[str, Any]] | None = None,
               identities: list[str] | None = None,
               constraints: list[str] | None = None,
               constants: dict[str, Any] | None = None,
               wordnet: dict[str, Any] | None = None,
               parent: str | None = None, alias: str | None = None) -> dict[str, Any]:
        if kind not in KIND_TO_HUB:
            raise MthLeafError("unknown leaf kind: " + str(kind), code="unknown_kind")
        payload = dict(payload or {})
        if kind == "vector":
            if payload.get("values") is None and "fill" in payload:
                fill = payload.get("fill")
                if (isinstance(fill, bool) or not isinstance(fill, (int, float))
                        or not math.isfinite(float(fill))):
                    raise MthLeafError("vector fill must be a finite number",
                                       code="invalid_value")
                payload = {**payload, "values": [float(fill)] * 256}
            values = _vector(payload.get("values"))
            payload = {"values": values, "dimension": 256, "precision": "float32"}
            identities = list(VECTOR_IDENTITIES) + list(identities or [])
            constraints = list(VECTOR_CONSTRAINTS) + list(constraints or [])
            constants = {"dimension": 256, "precision": "float32", **(constants or {})}
        elif kind == "scalar":
            value = payload.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MthLeafError("scalar value must be finite", code="invalid_value")
            payload = {"value": float(value), "precision": "float32"}
            constants = {"precision": "float32", **(constants or {})}
        elif kind == "matrix":
            values = payload.get("values")
            if values is not None:
                if not isinstance(values, list) or len(values) != 256 or any(
                        not isinstance(row, list) or len(row) != 256 for row in values):
                    raise MthLeafError("matrix must be 256x256", code="invalid_dimension")
            constants = {"rows": 256, "columns": 256, **(constants or {})}
        elif kind == "relation":
            constants = dict(constants or {})
        hub = self._data["leaves"].get(KIND_TO_HUB[kind])
        if hub is None:
            raise MthLeafError("leaf hub is missing", code="hub_missing")
        if parent is not None:
            try:
                parent_leaf = self.resolve(parent)
            except MthLeafError as exc:
                raise MthLeafError("parent leaf is unknown: " + str(parent),
                                   code="unknown_parent") from exc
            if parent_leaf["kind"] == "hub":
                expected_hub = KIND_TO_HUB[kind]
                if parent_leaf["leaf_id"] == FUNCTION_HUB and kind != "kernel":
                    # A model may reasonably attach a data leaf to the function
                    # root; normalize it to the kind hub so the tree stays typed.
                    parent = expected_hub
                elif parent_leaf["leaf_id"] != expected_hub:
                    raise MthLeafError(
                        "parent hub is not valid for kind " + kind,
                        code="invalid_parent_hub",
                        details={"parent": parent, "expected": expected_hub})
        wordnet_link = dict(hub.get("wordnet") or {})
        wordnet_link.update(wordnet or {})
        with self._lock:
            leaf = self._new_leaf(
                kind=kind, payload=payload,
                function_definition=dict(function_definition or {"kind": "data"}),
                action_relations=list(action_relations or []),
                identities=list(identities or []),
                constraints=list(constraints or []),
                constants=constants, wordnet=wordnet_link,
                parent=parent, alias=alias)
            self._audit("create", "ok", leaf_id=leaf["leaf_id"],
                        kind=kind, alias=alias)
            return leaf

    def get(self, leaf_id: str, *, include_image: bool = False) -> dict[str, Any]:
        leaf = self.resolve(leaf_id)
        result = json.loads(json.dumps(leaf))
        if include_image:
            path = Path(leaf["nan_image"]["path"])
            result["nan_image"]["text"] = path.read_text(encoding="utf-8")
        return result

    def list(self, *, kind: str | None = None, limit: int = 64) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
            raise MthLeafError("limit must be 1..256", code="invalid_limit")
        values = []
        for leaf_id in self._data["order"]:
            leaf = self._data["leaves"].get(leaf_id)
            if leaf is None or (kind is not None and leaf["kind"] != kind):
                continue
            values.append({key: leaf[key] for key in
                           ("leaf_id", "kind", "hub", "parent", "generation",
                            "immutable", "content_sha256", "nan_image")})
        return values[-limit:]

    def hub_tree(self, *, limit: int = HUB_TREE_LIMIT) -> dict[str, Any]:
        """Return a bounded, read-only summary of every hub leaf."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            limit = HUB_TREE_LIMIT
        limit = min(limit, HUB_TREE_LIMIT)
        child_counts: dict[str, int] = {}
        for leaf in self._data["leaves"].values():
            parent = leaf.get("parent")
            if parent:
                child_counts[parent] = child_counts.get(parent, 0) + 1
        hubs = []
        for leaf_id in self._data["order"]:
            leaf = self._data["leaves"].get(leaf_id)
            if leaf is None or leaf.get("kind") != "hub":
                continue
            hubs.append({
                "leaf_id": leaf_id,
                "kind": leaf.get("kind"),
                "parent": leaf.get("parent"),
                "child_count": child_counts.get(leaf_id, 0),
            })
        hubs.sort(key=lambda item: item["leaf_id"])
        return {
            "schema": HUB_SCHEMA,
            "limit": limit,
            "total": len(hubs),
            "truncated": len(hubs) > limit,
            "hubs": hubs[:limit],
        }

    def status(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for leaf in self._data["leaves"].values():
            kinds[leaf["kind"]] = kinds.get(leaf["kind"], 0) + 1
        linked_kernels = sum(1 for alias in self._data.get("aliases", {})
                             if str(alias).startswith("kernel:"))
        return {
            "schema": STORE_SCHEMA,
            "generation": self._data.get("generation", 0),
            "root_hub": ROOT_HUB,
            "leaves": len(self._data["leaves"]),
            "kinds": dict(sorted(kinds.items())),
            "aliases": len(self._data.get("aliases", {})),
            "index": str(self.index_path),
            "audit": str(self.audit_path),
            "immutable": True,
            "pair_rules": {a + "::" + b: list(ops)
                           for (a, b), ops in PAIR_RULES.items()},
            "wordnet": {"route": WORDNET_ROUTE, "hub": WORDNET_HUB},
            "hub_tree": self.hub_tree(),
            "function_tree": {
                "root": FUNCTION_HUB,
                "mapped": FUNCTION_HUB in self._data["leaves"],
                "expected_kernels": 91,
                "linked_kernels": linked_kernels,
                "note": "call nanlm_mth_tree with action=map or audit for full coverage",
            },
        }

    def pair(self, left_ref: str, right_ref: str, *,
             operator: str | None = None, alias: str | None = None,
             check_identities: bool = True, ctx=None) -> PairResult:
        op = (operator or "connect").strip().lower()
        with self._lock:
            try:
                left = self.resolve(left_ref)
                right = self.resolve(right_ref)
                rule = self._check_pair(left, right, op)
                if op == "connect":
                    return self._pair_connect(left, right, rule, alias)
                return self._pair_math(left, right, op, rule, alias,
                                       check_identities=check_identities, ctx=ctx)
            except MthLeafError as exc:
                self._audit("pair", "denied", left=left_ref, right=right_ref,
                            operator=op, code=exc.code, error=str(exc))
                return PairResult("denied", None, None, exc.to_dict())

    def _check_pair(self, left: dict[str, Any], right: dict[str, Any],
                    operator: str) -> dict[str, Any]:
        if left["kind"] == "hub" or right["kind"] == "hub":
            raise MthLeafError("hub leaves cannot be paired as data",
                               code="hub_pairing_denied")
        if left.get("immutable") is not True or right.get("immutable") is not True:
            raise MthLeafError("input leaves must be immutable", code="mutable_input_denied")
        if left["hub"] == ROOT_HUB or right["hub"] == ROOT_HUB:
            raise MthLeafError("leaf is not attached to a typed hub",
                               code="hub_ancestry_denied")
        rule = PAIR_RULES.get((left["kind"], right["kind"]))
        if rule is None:
            raise MthLeafError(
                "illegal pairing: %s::%s" % (left["kind"], right["kind"]),
                code="illegal_pairing",
                details={"left_kind": left["kind"], "right_kind": right["kind"]})
        if operator not in OPERATOR_RULES:
            raise MthLeafError("unknown operator: " + operator,
                               code="unknown_operator",
                               details={"allowed": list(rule)})
        if operator not in rule:
            raise MthLeafError("operator not admitted for this pairing: " + operator,
                               code="operator_not_admitted",
                               details={"allowed": list(rule)})
        expected = OPERATOR_RULES[operator]["input_kinds"]
        if (left["kind"], right["kind"]) != expected:
            raise MthLeafError("operator input kinds do not match",
                               code="operator_kind_mismatch")
        return OPERATOR_RULES[operator]

    def _pair_connect(self, left, right, rule, alias) -> PairResult:
        payload = {"left": left["leaf_id"], "right": right["leaf_id"],
                   "relation": "connected", "operator": "connect"}
        leaf = self.create(
            kind="relation", payload=payload,
            function_definition={"kind": "relation", "operator": "connect"},
            action_relations=[
                {"action": "connects", "target": left["leaf_id"]},
                {"action": "connects", "target": right["leaf_id"]},
            ],
            identities=list(rule["identities"]),
            constraints=list(rule["constraints"]),
            constants={"operator": "connect"},
            parent=left["leaf_id"], alias=alias)
        self._audit("pair", "ok", left=left["leaf_id"], right=right["leaf_id"],
                    operator="connect", hashleaf=leaf["leaf_id"])
        return PairResult("ok", leaf["leaf_id"], leaf, None)

    def _pair_math(self, left, right, operator, rule, alias,
                   *, check_identities: bool, ctx) -> PairResult:
        from . import nan0s_runtime
        left_values = _vector(left["payload"].get("values"))
        right_values = _vector(right["payload"].get("values"))
        try:
            executed = nan0s_runtime.execute_vector_kernel(
                ctx, rule["kernel"], left_values, right_values)
        except Exception as exc:
            raise MthLeafError("MTH N16I execution failed: " + str(exc)[:500],
                               code="native_execution_failed") from exc
        if not executed.get("ok") or executed.get("status") != 1:
            raise MthLeafError("MTH N16I kernel did not return status 1",
                               code="native_execution_failed",
                               details={"run": executed.get("trap"),
                                        "steps": executed.get("steps")})
        result = executed["result"]
        self._check_result_invariants(operator, left_values, right_values, result)
        if check_identities:
            self._check_identity(rule, operator, left_values, right_values,
                                 result, ctx=ctx)
        output_kind = rule["output_kind"]
        if output_kind == "scalar":
            payload = {"value": float(result), "precision": "float32"}
        else:
            values = _vector(result)
            payload = {"values": values, "dimension": 256, "precision": "float32"}
        leaf = self.create(
            kind=output_kind, payload=payload,
            function_definition={
                "kind": "kernel", "kernel_id": rule["kernel"],
                "operator": operator, "program_id": rule["program_id"],
                "operator_id": rule["operator_id"],
                "execution": {"mode": executed["execution_mode"],
                              "steps": executed.get("steps"),
                              "trap": executed.get("trap"),
                              "status_register": executed.get("status"),
                              "n16i_used": bool(executed.get("n16i_used"))},
            },
            action_relations=[
                {"action": "left_input", "target": left["leaf_id"]},
                {"action": "right_input", "target": right["leaf_id"]},
                {"action": "uses_hub", "target": left["hub"]},
                {"action": "uses_operator_hub", "target": OPERATOR_HUB},
            ],
            identities=list(rule["identities"]),
            constraints=list(rule["constraints"]),
            constants={"operator": operator, "dimension": 256,
                       "precision": "float32"},
            parent=left["leaf_id"], alias=alias)
        self._audit("pair", "ok", left=left["leaf_id"], right=right["leaf_id"],
                    operator=operator, hashleaf=leaf["leaf_id"],
                    result_sha256=leaf["content_sha256"])
        return PairResult("ok", leaf["leaf_id"], leaf, None)

    @staticmethod
    def _check_result_invariants(operator: str, left: list[float],
                                 right: list[float], result: Any) -> None:
        if operator in {"dot", "cosine"}:
            value = float(result)
            if not math.isfinite(value):
                raise MthLeafError("scalar result is not finite",
                                   code="invariant_violation")
            if operator == "cosine" and not (-1.0001 <= value <= 1.0001):
                raise MthLeafError("cosine result is outside [-1,1]",
                                   code="invariant_violation",
                                   details={"value": value})
            if operator == "dot" and left == right and value < -1e-6:
                raise MthLeafError("dot(x,x) must be non-negative",
                                   code="invariant_violation",
                                   details={"value": value})
        else:
            values = _vector(result)
            if not _finite(values):
                raise MthLeafError("vector result is not finite",
                                   code="invariant_violation")

    @staticmethod
    def _check_identity(rule, operator: str, left: list[float],
                        right: list[float], result: Any, *, ctx=None) -> None:
        from . import nan0s_runtime
        try:
            swapped = nan0s_runtime.execute_vector_kernel(
                ctx, rule["kernel"], right, left)
        except Exception as exc:
            raise MthLeafError("identity check execution failed: " + str(exc)[:300],
                               code="identity_check_failed") from exc
        if not swapped.get("ok") or swapped.get("status") != 1:
            raise MthLeafError("identity check kernel failed",
                               code="identity_check_failed")
        other = swapped["result"]
        if operator in {"dot", "cosine"}:
            if abs(float(result) - float(other)) > 1e-5:
                raise MthLeafError("symmetry identity violated",
                                   code="identity_violation",
                                   details={"result": result, "swapped": other})
        else:
            values = _vector(result)
            other_values = _vector(other)
            if operator == "add":
                if any(abs(a - b) > 1e-6 for a, b in zip(values, other_values)):
                    raise MthLeafError("add commutativity identity violated",
                                       code="identity_violation")
            elif operator == "sub":
                if any(abs(a + b) > 1e-6 for a, b in zip(values, other_values)):
                    raise MthLeafError("sub antisymmetry identity violated",
                                       code="identity_violation")


def store_for_context(ctx=None, *, state_dir: str | Path | None = None) -> LeafStore:
    if state_dir is None:
        if ctx is not None:
            try:
                state_dir = ctx.get_config(
                    "state_dir", default="~/.local/share/nanlmind-hermes")
            except Exception:
                state_dir = "~/.local/share/nanlmind-hermes"
        else:
            state_dir = "~/.local/share/nanlmind-hermes"
    root = Path(str(state_dir)).expanduser()
    key = str(root.resolve())
    with _LOCK:
        existing = _CACHE.get(key)
        if existing is not None:
            return existing
        store = LeafStore(root)
        _CACHE[key] = store
        return store


def reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()
