"""NaNLMind Hermes plugin entry point."""
from . import tools


def register(ctx):
    from . import clocked_tools

    # Boot the ONE local NaN16 universe before advertising ANY plugin tool.
    # A corrupt N16SNAP1 checkpoint, a wrong-parent N16ROOT1 descriptor or
    # an unmountable chart must abort plugin startup. Silently suppressing
    # these failures would expose apparently working learning tools whose
    # underlying authoritative binary root was never mounted.
    from .universe_chart import LocalRunner
    LocalRunner.open(ctx)

    # Ship the NANNET/WordNet architecture contract with the plugin and load it
    # once at registration. The attached LLM receives this same permanent
    # session data on every provider turn; it is not reconstructed from user
    # text or from an external documentation fetch.
    try:
        from . import session_context
        session_context.open_session(ctx)
    except Exception:
        pass
    # A model-supplied approved=true cannot authorize external operations.
    # Hermes checks pre_tool_call on actual model tool dispatch and prompts
    # the human (or fails closed in unattended runs).
    from .catalog_approval import make_approval_hook
    human_approval_available = callable(getattr(ctx, "register_hook", None))
    if human_approval_available:
        ctx.register_hook("pre_tool_call", make_approval_hook(ctx))

    def _require_human_approval_route():
        if not human_approval_available:
            raise PermissionError(
                "This protected tool requires Hermes pre_tool_call human approval"
            )

    _learning_cache: dict[str, object] = {}

    def _learning_policy():
        from .layered_learning import LearningPolicy
        def setting(key, default):
            try:
                return ctx.get_config(key, default=default)
            except Exception:
                return default
        return LearningPolicy(
            min_support=int(setting("learning_min_support", 3)),
            min_confidence=float(setting("learning_min_confidence", 0.6)),
            max_candidates=int(setting("learning_max_candidates", 64)),
            max_promoted_leaves=int(setting("learning_max_promoted_leaves", 100)),
            require_explicit_approval=bool(setting("learning_require_approval", True)),
            auto_sync=bool(setting("learning_auto_sync", False)),
        )

    def _learning(ctx):
        from .layered_learning import coordinator_for_context
        from .universe_chart import LocalRunner
        runner = LocalRunner.open(ctx)
        coordinator = coordinator_for_context(ctx)
        # The local runner's chart is process-lifetime state. Keep the
        # coordinator bound to it without opening a second scratch view.
        if coordinator.chart is None:
            coordinator.bind_chart(runner.chart)
        return coordinator

    _resident_cache = {}
    _resident_field_cache = {}
    _resident_scratch_cache = {}
    _head_field_tool_cache = {}

    def _resident_logic_ok():
        try:
            from .head_boolean_field import logic_base_preserved
            return bool(logic_base_preserved())
        except Exception:
            return False

    def _resident_field_learner(universe, temporal):
        from .resident.field_learning import FieldLearner
        key = str(universe.path)
        cached = _resident_field_cache.get(key)
        if cached is None:
            cached = FieldLearner.from_cells(
                temporal, universe.cells,
                max_active_heads=64, max_couplings=128,
                max_writes_per_sentence=512,
                total_write_budget=1_000_000)
            _resident_field_cache.clear()
            _resident_field_cache[key] = cached
        return cached

    def _resident_scratch(universe):
        from .resident.scratch_layers import ScratchLayers
        key = str(universe.path)
        cached = _resident_scratch_cache.get(key)
        if cached is None:
            cached = ScratchLayers()
            _resident_scratch_cache.clear()
            _resident_scratch_cache[key] = cached
        return cached

    def _resident_config(key, default):
        try:
            return ctx.get_config(key, default=default)
        except TypeError:
            try:
                return ctx.get_config(key, default)
            except Exception:
                return default
        except Exception:
            return default

    def _resident_size():
        try:
            value = int(_resident_config(
                "resident_size_bytes", 256 * 1024 * 1024))
        except Exception:
            value = 256 * 1024 * 1024
        return max(16 * 1024 * 1024, min(value, 1024 * 1024 * 1024))

    def _resident_path():
        from pathlib import Path
        try:
            root = Path(str(_resident_config(
                "state_dir", "~/.local/share/nanlmind-hermes"))).expanduser()
        except Exception:
            root = Path("~/.local/share/nanlmind-hermes").expanduser()
        return root / "resident" / "universe.nlmres"

    def _resident_universe(create=False):
        from .resident.universe import ResidentUniverse
        path = _resident_path()
        key = str(path)
        cached = _resident_cache.get(key)
        if cached is not None:
            return cached
        if not create and not path.exists():
            return None
        universe = ResidentUniverse.open_or_create(path, size=_resident_size())
        _resident_cache.clear()
        _resident_cache[key] = universe
        return universe

    def _resident_status_optional():
        try:
            universe = _resident_universe(create=False)
            if universe is None:
                return {"opened": False, "path": str(_resident_path()),
                        "schema": "nanlm.resident-universe/1",
                        "canonical_merge": {"enabled": False, "attempted": False}}
            result = universe.status()
            result["opened"] = True
            try:
                from .resident.head_field import ResidentHeadField
                field = ResidentHeadField(
                    universe.cells, generation=universe.generation)
                result["head_field"] = field.snapshot()
            except Exception as exc:
                result["head_field"] = {
                    "status": "unavailable",
                    "error": type(exc).__name__ + ": " + str(exc)[:300],
                }
            scratch = _resident_scratch_cache.get(str(universe.path))
            if scratch is not None:
                try:
                    result["scratch_layers"] = scratch.stats()
                except Exception:
                    pass
            return result
        except Exception as exc:
            return {"opened": False, "path": str(_resident_path()),
                    "error": type(exc).__name__ + ": " + str(exc)[:300],
                    "canonical_merge": {"enabled": False, "attempted": False}}

    class _HeadFieldToolBackend:
        """Plugin adapter for field_tools over the resident head cells."""

        def __init__(self, universe):
            from .resident.head_field import ResidentHeadField
            from .resident.scratch_layers import ScratchLayers
            self.universe = universe
            self.field = ResidentHeadField(
                universe.cells, generation=universe.generation,
                commit=universe._commit, flush=universe.flush)
            self.scratch = ScratchLayers()
            self.layers = {}

        @property
        def generation(self):
            return int(self.universe.generation)

        def snapshot(self):
            activation = self.field.field_activation()
            heads = self.field.read_all()
            active = [item["head"] for item in heads if item.get("active")]
            status = self.universe.status()
            return {
                "schema": "nanlm.resident-head-field/1",
                "opened": True,
                "generation": self.universe.generation,
                "digest": self.universe.digest,
                "heads": heads,
                "active_heads": active,
                "field": {
                    "head_count": 256,
                    "active_heads": active,
                    "active_count": len(active),
                    "entropy_bits": activation.get("entropy_bits"),
                    "peak_head": activation.get("peak_head"),
                },
                "budget": {
                    "current_bytes": status.get("file_bytes"),
                    "used_bytes": status.get("committed_bytes"),
                    "hard_limit_bytes": status.get("hard_ceiling_bytes"),
                    "exceeded": False,
                },
                "canonical_merge": {"enabled": False, "attempted": False},
            }

        def activate(self, head, delta):
            result = self.field.activate(int(head), float(delta))
            return {
                "accepted": True,
                "generation": self.universe.generation,
                "score": result.get("score"),
                "state": result.get("state"),
                "active": True,
            }

        def propose(self, head, delta, reason=""):
            head = int(head)
            layer = head % 10
            self.layers[head] = layer
            try:
                self.scratch.write_layer(
                    head, layer,
                    {"delta": float(delta), "reason": str(reason)[:240]},
                    approved=False)
            except Exception:
                pass
            proposal = self.field.propose(head, float(delta))
            return {
                "token": proposal.get("token"),
                "generation": self.universe.generation,
                "reasons": ["scratch_layer_L%d" % layer],
                "head": head,
                "delta": float(delta),
            }

        def commit(self, token, expected_generation):
            if int(expected_generation) != int(self.universe.generation):
                return {"accepted": False,
                        "reasons": ["generation_mismatch"],
                        "generation": self.universe.generation}
            proposal = self.field.apply_proposal(token, approved=True)
            head = int(proposal["head"])
            layer = self.layers.get(head, head % 10)
            try:
                self.scratch.write_layer(
                    head, layer,
                    {"applied_token": token, "score": proposal.get("score")},
                    approved=True)
            except Exception:
                pass
            return {
                "accepted": True,
                "generation": self.universe.generation,
                "score": proposal.get("score"),
                "state": proposal.get("state"),
                "layer": layer,
            }

        def head_metrics(self, head):
            return self.field.read(int(head))

        def field_metrics(self):
            activation = self.field.field_activation()
            return {
                "active_heads": activation.get("active_heads"),
                "active_count": activation.get("active_heads"),
                "entropy_bits": activation.get("entropy_bits"),
                "peak_head": activation.get("peak_head"),
                "total": activation.get("total"),
                "generation": self.universe.generation,
            }

        def resonance(self, text, top_k):
            return list(self.universe.query(text, top_k=int(top_k)))

    def _head_field_backend():
        universe = _resident_universe(create=True)
        key = str(universe.path)
        cached = _head_field_tool_cache.get(key)
        if cached is None:
            cached = _HeadFieldToolBackend(universe)
            _head_field_tool_cache.clear()
            _head_field_tool_cache[key] = cached
        return cached

    def _observe_resident(message, answer, probe=False):
        """Online free-chat exposure into the resident pair geometry.

        Probe reads never mutate the resident file. If a chat turn explicitly
        reports state_mutated=False, it is also skipped. The canonical merge
        gate remains off.
        """
        if probe or not isinstance(message, str) or len(message.strip()) < 2:
            return None
        try:
            if not _resident_config("chat_learning", True):
                return None
            if not _resident_config("resident_learning_enabled", True):
                return None
            if isinstance(answer, dict) and answer.get("state_mutated") is False:
                return None
            from .temporal_lm import TemporalMirrorLM, fnv_head, tokens
            guard = TemporalMirrorLM()
            if not guard.observe(message):
                return None  # a question/request/duplicate is not exposure
            universe = _resident_universe(create=True)
            scratch = _resident_scratch(universe)
            for token in tokens(message)[:512]:
                head = fnv_head(token)
                try:
                    scratch.write_layer(
                        head, head % 10,
                        {"token": token, "source": "free-chat",
                         "stage": "before-main"},
                        approved=False)
                except Exception:
                    pass
            sequence = ["<s>"] + tokens(message) + ["</s>"]
            for index in range(1, len(sequence)):
                if sequence[index] == "<s>":
                    continue
                try:
                    universe.cells.observe_pair(
                        fnv_head(sequence[index - 1]),
                        fnv_head(sequence[index]), weight=1.0)
                except Exception:
                    pass
            result = universe.ingest_rows([{
                "text": message, "kind": "observation", "source": "free-chat",
            }])
            import hashlib as _hashlib
            source_digest = _hashlib.sha256(
                message.casefold().encode("utf-8")).hexdigest()
            # Reuse the deterministic, inspectable free-chat parser for explicit
            # X is Y / X has Y / X can Y evidence.  It is local parser output,
            # never a claim of independent fact verification.
            try:
                from .chat import _native_parse
                parsed = _native_parse(message, "resident", source_digest)
                relations = [row["item"] for row in parsed
                             if isinstance(row, dict)
                             and isinstance(row.get("item"), dict)
                             and row["item"].get("kind") == "relation"]
                if relations:
                    relation_result = universe.ingest_rows(
                        relations, source_digest=source_digest)
                    result["relations"] = relation_result.get("accepted", 0)
            except Exception:
                pass
            try:
                if not _resident_logic_ok():
                    raise RuntimeError("Boolean logic base is not preserved")
                field = _resident_field_learner(universe, guard)
                field_result = field.observe(message)
                result["field_learning"] = {
                    key: field_result.get(key)
                    for key in (
                        "active_heads", "active_head_count", "entropy",
                        "entropy_bits", "collisions", "writes",
                        "field_activation",
                    )
                }
                try:
                    field_stats = field.stats()
                    for key in ("accepted_proposals", "rejected_proposals",
                                "proposal_acceptance_rate", "writes",
                                "budget_rejections"):
                        result["field_learning"][key] = field_stats.get(key)
                except Exception:
                    pass
                try:
                    scratch_stats = scratch.stats()
                    result["field_learning"]["scratch"] = {
                        "records": scratch_stats.get("scratch_records"),
                        "writes": scratch_stats.get("scratch_writes"),
                        "generation": scratch_stats.get("scratch_generation"),
                        "allocated_layers": scratch_stats.get("allocated_layers"),
                    }
                except Exception:
                    pass
            except Exception as exc:
                result["field_learning"] = {
                    "status": "unavailable",
                    "error": type(exc).__name__ + ": " + str(exc)[:300],
                }
            universe._commit("chat", source_digest)
            universe.flush()
            return {"generation": universe.generation,
                    "accepted": result.get("accepted", 0),
                    "duplicates": result.get("duplicates", 0),
                    "field_learning": result.get("field_learning")}
        except Exception as exc:
            # Chat must survive resident-layer failure; report the boundary.
            return {"error": type(exc).__name__ + ": " + str(exc)[:300],
                    "canonical_commit": False}

    def _merge_enabled():
        """Local promotion/merge is opt-in and off unless the operator enables it."""
        try:
            value = ctx.get_config("merge_enabled", default=False)
        except Exception:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _merge_disabled(reason):
        from .layered_learning import SYNC_SCHEMA
        return {
            "schema": SYNC_SCHEMA,
            "status": "merge_disabled",
            "merge_enabled": False,
            "approved": False,
            "candidates": 0,
            "promoted": 0,
            "note": reason,
        }

    def handle_learning_cycle(params, **kwargs):
        _require_scope("training:execute")
        coordinator = _learning(ctx)
        if not _merge_enabled():
            observation = coordinator.observe(
                params["message"], source=params.get("source", "chat"),
                relations=params.get("relations"), tick=params.get("tick"))
            return {
                "observation": observation,
                "sync": _merge_disabled(
                    "merge_enabled is false; approved=true cannot promote"),
            }
        result = coordinator.cycle(
            params["message"], source=params.get("source", "chat"),
            relations=params.get("relations"), tick=params.get("tick"),
            approved=params.get("approved", False))
        return result

    def handle_learning_status(params, **kwargs):
        _require_scope("universe:read")
        return _learning(ctx).status()

    def handle_learning_sync(params, **kwargs):
        _require_scope("training:execute")
        if not _merge_enabled():
            return _merge_disabled(
                "merge_enabled is false; local promotion is disabled")
        return _learning(ctx).synchronize(approved=params.get("approved", False))

    def handle_wordnet_ingest(params, **kwargs):
        _require_scope("training:execute")
        coordinator = _learning(ctx)
        result = coordinator.ingest_word(
            params["word"], depth=params.get("depth", 2))
        if params.get("approved", False):
            if not _merge_enabled():
                result["sync"] = _merge_disabled(
                    "merge_enabled is false; WordNet ingest cannot promote")
            else:
                result["sync"] = coordinator.synchronize(approved=True)
        return result

    def handle_scratch_register(params, **kwargs):
        _require_scope("universe:read")
        coordinator = _learning(ctx)
        layer = params.get("layer")
        head = params.get("head")
        limit = int(params.get("limit", 32))
        records = list(coordinator.space.records(layer=layer, head=head))[-max(1, min(limit, 256)):]
        return {
            "stats": coordinator.space.stats(),
            "verify": coordinator.space.verify(),
            "records": records,
        }

    _self_modification_cache: dict[str, object] = {}

    def _self_modification(ctx):
        from .mirror_topology import MirrorLeafBindings
        from .self_modification import SelfModificationEngine, SelfModificationPolicy
        root, _corpora, universe = tools._paths(ctx)
        key = str(root)
        engine = _self_modification_cache.get(key)
        if engine is not None:
            return engine
        def setting(name, default):
            try:
                return ctx.get_config(name, default=default)
            except Exception:
                return default
        policy = SelfModificationPolicy(
            min_boolean_confidence=float(setting("self_mod_min_boolean", 0.5)),
            require_evidence=bool(setting("self_mod_require_evidence", True)),
            allow_negative_memory=bool(setting("self_mod_allow_negative", True)),
            require_approval_for_promotion=bool(setting("self_mod_require_approval", True)),
        )
        bindings = MirrorLeafBindings(universe / "mathematical_memory.sqlite3")
        engine = SelfModificationEngine(_learning(ctx), bindings, policy=policy)
        _self_modification_cache[key] = engine
        return engine

    def handle_mirror_topology(params, **kwargs):
        _require_scope("universe:read")
        from .mirror_topology import MirrorTopology
        result = MirrorTopology.status()
        if params.get("route"):
            result["route"] = MirrorTopology.parse_route(params["route"])
        elif params.get("head") is not None or params.get("mp") is not None:
            result["microtable"] = MirrorTopology.route(
                head=params.get("head"), mp=params.get("mp"),
                target_mp=params.get("target_mp"), layer=params.get("layer", 5))
        return result

    def handle_router_graph_verify(params, **kwargs):
        _require_scope("universe:read")
        from .router_graph_page import RouterGraphPageVerifier, RouterGraphPageError
        from .universe_chart import LocalRunner
        pages = params["pages"]
        if type(pages) is not list or not 1 <= len(pages) <= 128:
            raise RouterGraphPageError("requires 1..128 ordered graph pages")
        history_pages = params.get("history_pages")
        attestation = params.get("attestation")
        if attestation is not None and history_pages is None:
            raise RouterGraphPageError("signed graph requires complete history pages")
        chart = LocalRunner.open(ctx).chart
        if history_pages is None:
            verifier = RouterGraphPageVerifier(chart)
            for page in pages:
                result = verifier.add_page(page)
            if not result["state_verified"]:
                raise RouterGraphPageError("incomplete graph snapshot; no partial acceptance")
            return result
        if type(history_pages) is not list or not 1 <= len(history_pages) <= 128:
            raise RouterGraphPageError("requires 1..128 ordered history pages")
        from .router_graph_history import RouterGraphHistoryVerifier
        verifier = RouterGraphHistoryVerifier(chart)
        for page in pages:
            verifier.add_state_page(page)
        if not verifier.status()["state_verified"]:
            raise RouterGraphPageError("incomplete graph state; no partial acceptance")
        for page in history_pages:
            verifier.add_history_page(page)
        result = verifier.status()
        if not result["history_verified"]:
            raise RouterGraphPageError("incomplete graph history; no partial acceptance")
        if attestation is not None:
            # Public key and sender identity are independently pinned by the
            # installed HERMES configuration. Never trust request-provided
            # keys or caller claims as an identity/authority registry.
            from .router_graph_attestation import verify_graph_sender_attestation
            return verify_graph_sender_attestation(
                verifier, attestation,
                expected_router_id=ctx.get_config(
                    "router_graph_sender_id", default=""),
                expected_key_id=ctx.get_config(
                    "router_graph_sender_key_id", default=""),
                public_key_hex=ctx.get_config(
                    "router_graph_sender_public_key_hex", default=""),
            )
        return result

    def handle_router_graph_walk(params, **kwargs):
        """Read-only bounded graph navigation gated by full chart/history replay.

        Optional signature authentication reuses the operator-pinned router
        identity, never caller-provided trust material. No head, chart or
        graph write is exposed by this tool.
        """
        _require_scope("universe:read")
        from .router_graph_history import RouterGraphHistoryVerifier
        from .router_graph_page import RouterGraphPageError
        from .router_graph_walk import verified_candidate_walk
        from .universe_chart import LocalRunner
        pages = params["pages"]
        history_pages = params["history_pages"]
        if (type(pages) is not list or not 1 <= len(pages) <= 128
                or type(history_pages) is not list
                or not 1 <= len(history_pages) <= 128):
            raise RouterGraphPageError(
                "complete bounded state and history pages required")
        verifier = RouterGraphHistoryVerifier(LocalRunner.open(ctx).chart)
        for page in pages:
            verifier.add_state_page(page)
        if not verifier.status()["state_verified"]:
            raise RouterGraphPageError("incomplete graph state")
        for page in history_pages:
            verifier.add_history_page(page)
        if not verifier.status()["history_verified"]:
            raise RouterGraphPageError("incomplete graph history")
        attestation = params.get("attestation")
        auth = None
        if attestation is not None:
            from .router_graph_attestation import verify_graph_sender_attestation
            auth = verify_graph_sender_attestation(
                verifier, attestation,
                expected_router_id=ctx.get_config(
                    "router_graph_sender_id", default=""),
                expected_key_id=ctx.get_config(
                    "router_graph_sender_key_id", default=""),
                public_key_hex=ctx.get_config(
                    "router_graph_sender_public_key_hex", default=""))
        result = verified_candidate_walk(
            verifier, pages, start=params["start"],
            target=params.get("target"), layer=params.get("layer"),
            relation=params.get("relation"),
            max_hops=params.get("max_hops", 4),
            max_nodes=params.get("max_nodes", 256),
            max_edges=params.get("max_edges", 1024))
        if auth is not None:
            result["sender_authenticated"] = auth["sender_authenticated"]
            result["attestation_sha256"] = auth["attestation_sha256"]
            result["nonce_replay_protected"] = False
        return result

    def handle_self_modify(params, **kwargs):
        _require_scope("training:execute")
        engine = _self_modification(ctx)
        action = params.get("action", "status")
        if action == "status":
            return engine.status()
        if action == "relation":
            return engine.propose_relation(
                params["relation"], layer=params.get("layer", 5),
                source_head=params["source_head"],
                evidence_sha256=params["evidence_sha256"],
                strength_q16=params.get("strength_q16", 32768),
                polarity=params.get("polarity", 1),
                expected_generation=params.get("expected_generation", 0),
                tick=params.get("tick", 0))
        if action == "leaf":
            return engine.modify_leaf(
                mp=params["mp"], leaf_id=params["leaf_id"],
                payload=params["payload"], source_head=params["source_head"],
                evidence_sha256=params["evidence_sha256"],
                expected_generation=params["expected_generation"],
                tick=params.get("tick", 0))
        if action == "promote":
            return engine.promote(approved=params.get("approved", False))
        raise ValueError("unknown self-modification action")

    def handle_mth_tools(params, **kwargs):
        _require_scope("universe:read")
        engine = _self_modification(ctx)
        return engine.use_mth_tool(params["tool"], params.get("arguments"))

    def handle_teacher_teach(params, **kwargs):
        _require_scope("training:execute")
        from .gemma_teacher import teacher_for_context
        teacher = teacher_for_context(ctx)
        return teacher.teach(_learning(ctx), params["topic"], params.get("count", 8))

    def handle_clocked_plan(params, **kwargs):
        _require_scope("universe:read")
        return clocked_tools.clocked_plan(ctx, params["question"],
            params.get("sources", []), params.get("scratch_id", "default"))

    def handle_clocked_execute(params, **kwargs):
        _require_human_approval_route()
        _require_scope("universe:read")
        return clocked_tools.clocked_execute(ctx, params["question"],
            params.get("sources", []), params.get("scratch_id", "default"),
            True, params.get("plan_sha256", ""),
            params.get("max_requests", 16), params.get("max_bytes", 1048576))

    def handle_clocked_scratch(params, **kwargs):
        _require_scope("universe:read")
        return clocked_tools.scratch_show(ctx, params.get("scratch_id", "default"),
                                          params.get("lane"))

    def handle_head_process(params, **kwargs):
        _require_scope("universe:read")
        return clocked_tools.head_process(ctx, params["question"],
            params.get("scratch_id", "default"), params.get("purpose", "EVIDENCE_COGNITION"),
            params.get("draft"))

    def handle_clocked_synthesize(params, **kwargs):
        _require_scope("universe:read")
        return clocked_tools.clocked_synthesize(ctx, params["question"],
            params.get("scratch_id", "default"), params.get("max_tokens", 2048))

    def handle_head_candidate(params, **kwargs):
        _require_human_approval_route()
        _require_scope("training:execute")
        return clocked_tools.head_candidate(ctx, params["mutation_id"],
            params["head_index"], params["patch_mask"],
            params["base_topology_digest"], True,
            params.get("scratch_id", "default"))

    def handle_head_learn_candidate(params, **kwargs):
        _require_human_approval_route()
        _require_scope("training:execute")
        return clocked_tools.head_learn_candidate(ctx, params["mutation_id"],
            params.get("base_topology_digest"), True,
            params.get("scratch_id", "default"), params.get("epochs", 1))

    def handle_head_candidate_evaluate(params, **kwargs):
        _require_human_approval_route()
        _require_scope("training:execute")
        return clocked_tools.head_candidate_evaluate(ctx, params["mutation_id"],
            params["records"], True,
            params.get("scratch_id", "default"))

    def handle_status(params, **kwargs):
        result = tools.status(ctx)
        result["mathematical_memory"] = tools.memory_status(ctx)
        merge_gate = result.setdefault("merge_gate", {})
        merge_gate["canonical_merge_enabled"] = False
        merge_gate["merge_enabled"] = _merge_enabled()
        merge_gate["local_promotion_enabled"] = merge_gate["merge_enabled"]
        result["resident_universe"] = _resident_status_optional()
        return result

    def handle_session_context(params, **kwargs):
        from . import session_context
        result = session_context.status(ctx)
        document = params.get("document")
        if document:
            result["document"] = session_context.document_text(
                ctx, document, max_bytes=params.get("max_bytes", 16384))
        elif params.get("include_text", False):
            result["documents_text"] = [
                session_context.document_text(
                    ctx, item, max_bytes=params.get("max_bytes", 16384))
                for item in session_context.DOCUMENT_IDS
            ]
        return result

    def handle_chart_storage(params, **kwargs):
        action = params.get("action", "inspect")
        if action == "checkpoint":
            _require_scope("training:execute")
        return tools.chart_storage_tool(ctx, action, params.get("approved", False))

    def handle_resident_status(params, **kwargs):
        _require_scope("universe:read")
        result = _resident_status_optional()
        result["canonical_merge"] = {"enabled": False, "attempted": False}
        return result

    def handle_resident_ingest(params, **kwargs):
        _require_scope("training:execute")
        if params.get("approved") is not True:
            raise RuntimeError("resident ingest requires explicit approved=true")
        universe = _resident_universe(create=True)
        rows = params.get("rows")
        if not isinstance(rows, list) or not rows:
            rows = [{
                "text": params.get("message") or params.get("text"),
                "kind": params.get("kind", "observation"),
                "source": params.get("source", "tool"),
                "head": params.get("head"),
            }]
        result = universe.ingest_rows(rows)
        result["generation"] = universe.generation
        result["digest"] = universe.digest
        result["canonical_merge"] = {"enabled": False, "attempted": False}
        return result

    def handle_head_field(params, **kwargs):
        from .resident import field_tools
        action = params.get("action")
        if action not in field_tools.HANDLERS:
            raise ValueError("action must be status, activate, propose, commit or query")
        if action in {"status", "query"}:
            _require_scope("universe:read")
        else:
            _require_scope("training:execute")
            if not _resident_logic_ok():
                raise RuntimeError(
                    "Boolean logic base is not preserved; head-field writes are disabled")
        body = {key: value for key, value in params.items() if key != "action"}
        return field_tools.HANDLERS[action](
            body, backend=_head_field_backend())

    def handle_resident_query(params, **kwargs):
        _require_scope("universe:read")
        universe = _resident_universe(create=False)
        if universe is None:
            return {"opened": False, "hits": [], "leaves": [],
                    "relations": [],
                    "canonical_merge": {"enabled": False, "attempted": False}}
        text = params.get("text") or params.get("message")
        if text:
            result = universe.query(text, top_k=params.get("top_k", 8),
                                    mode=params.get("mode", "cosine"))
            rendered = {key: value for key, value in result.items()}
            rendered["hits"] = list(result)
            rendered["lookup"] = universe.lookup(
                text=text, limit=params.get("limit", 8))
            return rendered
        return universe.lookup(
            subject=params.get("subject"), predicate=params.get("predicate"),
            object=params.get("object"), head=params.get("head"),
            limit=params.get("limit", 16))

    def handle_resident_evidence(params, **kwargs):
        action = params.get("action")
        if action not in {"status", "state", "ask", "tell", "delete",
                          "save", "load"}:
            raise ValueError(
                "action must be status, state, ask, tell, delete, save or load")
        writes = action in {"tell", "delete", "load"}
        if writes:
            _require_scope("training:execute")
            if params.get("approved") is not True:
                raise RuntimeError(
                    "resident evidence writes require explicit approved=true")
        else:
            _require_scope("universe:read")
        universe = _resident_universe(create=writes)
        if universe is None:
            return {"opened": False, "action": action, "status": "unknown",
                    "records": 0,
                    "canonical_merge": {"enabled": False, "attempted": False}}
        session = universe.evidence_session()
        expected = params.get("expected_generation")
        if action == "status":
            result = session.status()
        elif action == "tell":
            result = session.tell(params["statement"],
                                  params.get("polarity"),
                                  expected_generation=expected).as_dict()
        elif action == "delete":
            result = session.delete(
                params["subject"], params.get("predicate", "is a"),
                params["object"], params.get("polarity"),
                expected_generation=expected)
        elif action == "ask":
            result = session.ask(params["question"]).as_dict()
        elif action == "state":
            result = session.lanes.state(
                params["subject"], params.get("predicate", "is a"),
                params["object"]).as_dict()
        elif action == "save":
            result = session.save()
        else:
            result = session.load(params["snapshot"],
                                  expected_generation=expected)
        result["opened"] = True
        result["canonical_merge"] = {"enabled": False, "attempted": False}
        return result

    def handle_curriculum(params, **kwargs):
        _require_scope("training:execute")
        return tools.generate_curriculum(ctx, params["topic"], params.get("records_per_stage", 12))

    def handle_curriculum_experiment(params, **kwargs):
        _require_scope("training:execute")
        return tools.run_curriculum_experiment(
            ctx, params["topic"], params.get("records_per_stage", 12),
            params.get("epochs", 3), params.get("max_samples_per_head", 4096))

    def handle_logic_session(params, **kwargs):
        _require_scope("training:execute")
        return tools.run_training_session(
            ctx, params.get("topic", "logic"), "logic",
            params.get("records_per_stage", 12),
            params.get("epochs", 3),
            params.get("max_samples_per_head", 4096))

    def handle_word_session(params, **kwargs):
        _require_scope("training:execute")
        return tools.run_training_session(
            ctx, params.get("topic", "words"), "words",
            params.get("records_per_stage", 12),
            params.get("epochs", 3),
            params.get("max_samples_per_head", 4096))

    def handle_sentence_session(params, **kwargs):
        _require_scope("training:execute")
        return tools.run_training_session(
            ctx, params.get("topic", "sentences"), "sentences",
            params.get("records_per_stage", 12),
            params.get("epochs", 3),
            params.get("max_samples_per_head", 4096))

    def handle_training_pipeline(params, **kwargs):
        _require_scope("training:execute")
        mode = params.get("mode", "pipeline")
        stages = params.get("stages")
        if isinstance(stages, list) and stages:
            return tools.run_curriculum_experiment(
                ctx, params.get("topic", "logic and language"),
                params.get("records_per_stage", 12),
                params.get("epochs", 3),
                params.get("max_samples_per_head", 4096), stages)
        return tools.run_training_session(
            ctx, params.get("topic", "logic and language"), mode,
            params.get("records_per_stage", 12),
            params.get("epochs", 3),
            params.get("max_samples_per_head", 4096))

    def _observe_chat(ctx, message, answer):
        """Compatibility observer for chat paths that did not already observe.

        ``tools.chat`` now performs the layered observation internally so that
        direct/script callers get the same behavior as plugin tools. Do not
        observe the same turn twice when a layered receipt is present.
        """
        try:
            if isinstance(answer, dict) and (answer.get("probe") is True
                                             or answer.get("state_mutated") is False):
                return None
            layered = answer.get("layered_learning") if isinstance(answer, dict) else None
            if isinstance(layered, dict) and layered.get("layers") == 10:
                return layered
            coordinator = _learning(ctx)
            records = []
            for key in ("learning", "native_learning", "reflection"):
                value = answer.get(key) if isinstance(answer, dict) else None
                if isinstance(value, dict) and isinstance(value.get("records"), list):
                    records.extend(item for item in value["records"] if isinstance(item, dict))
            return coordinator.observe(message, source="chat", relations=records)
        except Exception:
            # Best effort: canonical chat output must survive a scratch failure.
            return None

    def handle_chat(params, **kwargs):
        probe = bool(params.get("probe", False))
        read_authority = _require_scope("universe:read")
        try:
            learning_enabled = ctx.get_config("chat_learning", default=True) and not probe
        except Exception:
            learning_enabled = not probe
        learning_authority = None
        if learning_enabled:
            learning_authority = _require_scope("training:execute")
        answer = tools.chat(ctx, params["message"], params.get("session_id", "default"),
                            params.get("system_prompt"), params.get("temperature", 0.7),
                            params.get("max_tokens", 2048), params.get("sources"),
                            probe=probe)
        _observe_chat(ctx, params["message"], answer)
        resident_receipt = _observe_resident(
            params["message"], answer, probe=probe)
        if resident_receipt is not None:
            answer["resident_learning"] = resident_receipt
        answer["authorization"] = {
            "read": read_authority, "learning": learning_authority,
            "local_only": read_authority.get("mode") == "local-unverified",
        }
        return answer

    def handle_chat_train(params, **kwargs):
        """Mandatory Boolean foundation, then unrestricted dialogue and proposals."""
        authority = _require_scope("training:execute")
        foundation = tools.ensure_boolean_foundation(ctx)
        if foundation.get("status") != "passed":
            raise RuntimeError("free-chat learning blocked until native logic foundation passes")
        answer = tools.chat(
            ctx, params["message"], params.get("session_id", "training"),
            params.get("system_prompt"), params.get("temperature", 0.7),
            params.get("max_tokens", 2048), params.get("sources"))
        answer["mandatory_logic"] = {
            "status": "passed", "state_sha256": foundation.get("state_sha256"),
            "reused_checkpoint": foundation.get("reused_checkpoint", False)}
        answer["training_boundary"] = (
            "Conversation reflection is provisional; canonical head updates "
            "require the separate candidate evaluation and merge gates.")
        answer["authorization"] = authority
        _observe_chat(ctx, params["message"], answer)
        resident_receipt = _observe_resident(params["message"], answer,
                                             probe=False)
        if resident_receipt is not None:
            answer["resident_learning"] = resident_receipt
        return answer

    def handle_chat_history(params, **kwargs):
        return tools.chat_history(ctx, params.get("session_id", "default"),
                                  params.get("limit", 40))

    def handle_phone_call(params, **kwargs):
        read_authority = _require_scope("universe:read")
        try:
            learning_enabled = ctx.get_config("chat_learning", default=True)
        except Exception:
            learning_enabled = True
        learning_authority = (_require_scope("training:execute")
                              if learning_enabled else None)
        from . import phone_call
        result = phone_call.handle(ctx, params)
        if isinstance(result, dict):
            result["authorization"] = {
                "read": read_authority, "learning": learning_authority,
                "local_only": read_authority.get("mode") == "local-unverified",
            }
        return result

    # Load cryptography-backed NANNET handlers only when one is invoked.
    # This keeps ordinary Hermes chat/training importable on Android/Termux.
    def _auth_tools():
        try:
            from . import auth_tools
            return auth_tools
        except ImportError as exc:
            if exc.name == "cryptography" or "cryptography" in str(exc):
                raise RuntimeError(
                    "NANNET cryptographic operations require the optional "
                    "cryptography extra; install nanlmind-hermes[auth] on a "
                    "supported platform. Protected NANNET operations require it."
                ) from exc
            raise

    def _nannet_configured():
        try:
            url = str(ctx.get_config("nannet_auth_url", default="") or "").strip()
            issuer = str(ctx.get_config("nannet_issuer_public_key", default="") or "").strip()
            return bool(url and issuer)
        except Exception:
            return False

    def _require_scope(scope):
        # Local-only operation is allowed when no NANNET authority is
        # configured at all. It is explicitly reported as unverified and can
        # never authorize a remote router/lease/merge action. Once a NANNET
        # authority is configured, the paired installation token is mandatory
        # and fails closed. There is no environment-variable test bypass.
        if not _nannet_configured():
            return {"mode": "local-unverified", "scope": scope,
                    "remote": False, "nannet_configured": False}
        _auth_tools().require_scope(ctx, scope)
        return {"mode": "nannet", "scope": scope,
                "remote": True, "nannet_configured": True}

    def handle_auth_init(params, **kwargs): return _auth_tools().auth_init(ctx)
    def handle_auth_status(params, **kwargs): return _auth_tools().auth_status(ctx)
    def handle_auth_pair(params, **kwargs): return _auth_tools().auth_pair(ctx, params["pairing_code"])
    def handle_auth_revoke(params, **kwargs): return _auth_tools().auth_revoke(ctx)

    def handle_hw_register(params, **kwargs):
        from . import nannet_device_client
        return nannet_device_client.register(ctx, params.get("consent", False))

    def handle_hw_status(params, **kwargs):
        from . import nannet_device_client
        return nannet_device_client.status(ctx)

    def handle_hw_heartbeat(params, **kwargs):
        from . import nannet_device_client
        return nannet_device_client.heartbeat(ctx)

    def handle_hw_release(params, **kwargs):
        from . import nannet_device_client
        return nannet_device_client.release(ctx)

    def handle_hw_report(params, **kwargs):
        from . import nannet_device_client
        return nannet_device_client.report(
            ctx, params["report_type"], params["summary"],
            params.get("details", ""))

    def handle_hw2_bind(params, **kwargs):
        if not _nannet_configured():
            raise RuntimeError("NANNET authority is required for fixed HW2 binding")
        _require_scope("hermes:lease")
        from . import nannet_device_client
        return nannet_device_client.register_fixed_hw2(
            ctx, consent=params.get("consent", False))

    def handle_hw2_status(params, **kwargs):
        from . import nannet_device_client
        local = nannet_device_client.fixed_hw2_status(ctx)
        if params.get("remote", False):
            local = {"local": local,
                     "remote": nannet_device_client.fixed_hw2_remote_status(ctx)}
        return local

    def handle_hw2_report(params, **kwargs):
        from . import nannet_device_client
        return nannet_device_client.report_fixed_hw2(
            ctx, params["report_type"], params["summary"],
            params.get("details", ""))


    def handle_universe_status(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.status(ctx)

    def handle_universe_open(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.open_universe(
            ctx, params.get("alias", "primary"), params.get("ttl_seconds", 900))

    def handle_universe_resolve(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.resolve(
            ctx, params.get("alias", "primary"), params.get("head_index", 0))

    def handle_universe_renew(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.renew(
            ctx, params.get("alias", "primary"), params.get("ttl_seconds", 900))

    def handle_universe_release(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.release(ctx, params.get("alias", "primary"))

    def handle_universe_dispatch(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.dispatch(
            ctx, params.get("alias", "primary"),
            params.get("head_index", 0), params["dispatch_id"], params["frame_hex"])

    def handle_universe_ack(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.acknowledge(
            ctx, params.get("alias", "primary"), params.get("head_index", 0),
            params["dispatch_id"], params["response_sha256"])

    def handle_universe_inspect(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.inspect_dispatch(
            ctx, params.get("alias", "primary"), params.get("head_index", 0),
            params["dispatch_id"])

    def handle_native_mth_u32(params, **kwargs):
        from . import router_universe_client
        return router_universe_client.math_u32(
            ctx, params.get("alias", "primary"), params.get("head_index", 0),
            params.get("kernel_id", "mth.u32.add_checked"),
            params["a"], params["b"])

    def handle_mth_catalog(params, **kwargs):
        from . import mth_catalog
        return mth_catalog.list_kernels(
            family=params.get("family"), tags=params.get("tags"),
            offset=params.get("offset", 0), limit=params.get("limit", 32),
            require_ci=params.get("require_ci", False))

    def handle_mth_inspect(params, **kwargs):
        from . import mth_catalog
        return mth_catalog.inspect(params["kernel_id"])

    def handle_mth_runtime(params, **kwargs):
        from . import nan0s_runtime
        return nan0s_runtime.runtime_status(
            ctx, build=params.get("build", False))

    def handle_mth_verify(params, **kwargs):
        authority = _require_scope("training:execute")
        from . import nan0s_runtime
        result = nan0s_runtime.verify_kernel(
            ctx, params["kernel_id"], max_steps=params.get("max_steps"),
            timeout=params.get("timeout", 120))
        result["authorization"] = authority
        return result

    def handle_mth_execute(params, **kwargs):
        authority = _require_scope("training:execute")
        from . import nan0s_runtime
        result = nan0s_runtime.execute_vector_kernel(
            ctx, params["kernel_id"], params["x"], params["y"],
            timeout=params.get("timeout", 120))
        result["authorization"] = authority
        return result

    def handle_mth_n16i_selftest(params, **kwargs):
        authority = _require_scope("training:execute")
        from . import nan0s_runtime
        result = nan0s_runtime.n16i_self_test(
            ctx, timeout=params.get("timeout", 120))
        result["authorization"] = authority
        return result

    def handle_mth_dispatch(params, **kwargs):
        authority = _require_scope("training:execute")
        from . import nan0s_runtime
        result = nan0s_runtime.verify_kernel_n16i(
            ctx, params["kernel_id"],
            use_fixture=params.get("use_fixture", True),
            entry=params.get("entry", 0),
            max_steps=params.get("max_steps", 100_000_000),
            timeout=params.get("timeout", 120))
        result["authorization"] = authority
        return result

    def handle_mth_hub(params, **kwargs):
        from . import mth_leaves
        return mth_leaves.store_for_context(ctx).status()

    def handle_mth_tree(params, **kwargs):
        action = params.get("action", "status")
        authority = (_require_scope("universe:read") if action == "status"
                     else _require_scope("training:execute"))
        from . import mth_leaves
        store = mth_leaves.store_for_context(ctx)
        if action == "map":
            result = store.ensure_function_tree()
        elif action == "audit":
            if store.status()["function_tree"]["linked_kernels"] < 91:
                store.ensure_function_tree()
            result = store.function_tree_audit(
                verify_images=params.get("verify_images", True))
        else:
            result = store.status()
        result["authorization"] = authority
        return result

    def handle_mth_leaf_create(params, **kwargs):
        authority = _require_scope("training:execute")
        from . import mth_leaves
        store = mth_leaves.store_for_context(ctx)
        leaf = store.create(
            kind=params["kind"], payload=params.get("payload"),
            function_definition=params.get("function_definition"),
            action_relations=params.get("action_relations"),
            identities=params.get("identities"),
            constraints=params.get("constraints"),
            constants=params.get("constants"),
            wordnet=params.get("wordnet"),
            parent=params.get("parent"), alias=params.get("alias"))
        return {"status": "ok", "leaf": leaf, "authorization": authority}

    def handle_mth_leaf_get(params, **kwargs):
        from . import mth_leaves
        return mth_leaves.store_for_context(ctx).get(
            params["leaf_id"], include_image=params.get("include_image", False))

    def handle_mth_pair(params, **kwargs):
        authority = _require_scope("training:execute")
        from . import mth_leaves
        store = mth_leaves.store_for_context(ctx)
        result = store.pair(
            params["left"], params["right"], operator=params.get("operator"),
            alias=params.get("alias"),
            check_identities=params.get("check_identities", True), ctx=ctx)
        payload = {
            "schema": mth_leaves.PAIR_SCHEMA,
            "status": result.status,
            "hashleaf": result.hashleaf,
            "leaf": result.leaf,
            "error": result.error,
            "authorization": authority,
        }
        return payload

    def handle_mth_leaf_verify(params, **kwargs):
        from . import mth_leaves
        leaf = mth_leaves.store_for_context(ctx).resolve(params["leaf_id"])
        return mth_leaves.verify_leaf(leaf)

    def handle_nan16de_route(params, **kwargs):
        _require_scope("universe:read")
        from .auth import load_installation, decrypt_payload
        from .auth_tools import _master_key, _universe_dir, _config
        from .nan16de import NanWord, make_envelope, send_envelope
        import json as _json
        root = _universe_dir(ctx)
        doc, private = load_installation(root, _master_key(ctx))
        token_path = root / "nannet_token.json"
        if not token_path.exists():
            raise RuntimeError("NANNET paired token is missing; pair before routing")
        envelope = _json.loads(token_path.read_text(encoding="utf-8"))
        token = decrypt_payload(_master_key(ctx), envelope,
                                ("nannet-token-v1:" + doc["installation_id"]).encode()).decode()
        word = NanWord.from_bits(int(params.get("far_type", 0)),
                                 int(params.get("near_type", 0)),
                                 int(params.get("payload", 0)))
        env = make_envelope(params["route"], params.get("kind", "route"),
                            word, private, doc["installation_id"],
                            int(params.get("epoch", 0)))
        return send_envelope(_config(ctx, "nannet_auth_url"), token, env)

    def handle_generate(params, **kwargs):
        _require_scope("corpus:write")
        return tools.generate_corpus(ctx, params.get("topic", ""), params.get("records", 12))

    def handle_train(params, **kwargs):
        _require_scope("training:execute")
        return tools.train(ctx, tools.resolve_tool_corpus_path(ctx, params["corpus_path"]), params.get("duration_minutes"))

    def handle_math_train(params, **kwargs):
        _require_scope("training:execute")
        return tools.math_train(ctx, tools.resolve_tool_corpus_path(ctx, params["corpus_path"]), params.get("epochs", 3),
                                params.get("max_samples_per_head", 4096))

    def handle_relation_train(params, **kwargs):
        _require_scope("training:execute")
        return tools.structured_relation_train(ctx, params["records"], params.get("epochs", 1))

    def handle_evaluate(params, **kwargs):
        _require_scope("universe:read")
        return tools.evaluate(ctx, tools.resolve_tool_corpus_path(ctx, params["corpus_path"]))

    def handle_memory_ingest(params, **kwargs):
        _require_scope("training:execute")
        return tools.memory_ingest(ctx, params["discoveries"], params["source_digest"],
                                   params.get("head"), params.get("layer", 0))

    def handle_memory_reorganize(params, **kwargs):
        _require_scope("training:execute")
        return tools.memory_reorganize(ctx)

    def handle_memory_cross_resolution(params, **kwargs):
        _require_scope("training:execute")
        return tools.memory_cross_resolution(ctx, params.get("limit", 1000))

    def handle_memory_merge(params, **kwargs):
        _require_scope("training:execute")
        if not _merge_enabled():
            return {
                "schema": "nanlm.hermes-memory-merge/1",
                "status": "merge_disabled",
                "merge_enabled": False,
                "merge": {"accepted": 0, "duplicates": 0},
                "note": "merge_enabled is false; peer memory merge is disabled",
            }
        return tools.memory_merge(ctx, params["records"])

    def handle_memory_query(params, **kwargs):
        _require_scope("universe:read")
        return tools.memory_query(ctx, params.get("subject"), params.get("head"),
                                  params.get("layer"), params.get("limit", 100))

    def handle_memory_assessments(params, **kwargs):
        _require_scope("universe:read")
        return tools.memory_assessments(ctx, params.get("subject"), params.get("limit", 100))

    def handle_tree_inspect(params, **kwargs):
        _require_scope("universe:read")
        from . import experimental_register
        return experimental_register.inspect(
            ctx, params.get("head"), params.get("branch"),
            params.get("leaf"), params.get("limit", 128))

    def handle_tree_mutate(params, **kwargs):
        _require_scope("training:execute")
        from . import experimental_register
        return experimental_register.mutate(
            ctx, params["head"], params["branch"], params["leaf"],
            params["data_hex"], params.get("expected_revision", 0))

    def handle_tree_audit(params, **kwargs):
        _require_scope("universe:read")
        from . import experimental_register
        return experimental_register.audit(ctx, params.get("limit", 64))

    specs = [
        ("nanlm_mirror_topology", "Inspect the resident 256-head / 32,640-MP / 65,536-address mirror topology and recursive nan:// L0-L9 routes.",
         {"route": {"type": "string", "maxLength": 256},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "mp": {"type": "integer", "minimum": 0, "maximum": 32639},
          "target_mp": {"type": "integer", "minimum": 0, "maximum": 32639},
          "layer": {"type": "integer", "minimum": 0, "maximum": 9}},
         [], handle_mirror_topology),
        ("nanlm_router_graph_verify",
         "Read-only chart-bound NaNROUTER state/history verification. Optional attestation cryptographically authenticates sender against operator-pinned Ed25519 public key; no native execution or learning admission.",
         {"pages": {"type": "array", "minItems": 1, "maxItems": 128,
                    "items": {"type": "object"}},
          "history_pages": {"type": "array", "minItems": 1, "maxItems": 128,
                            "items": {"type": "object"}},
          "attestation": {"type": "object"}},
         ["pages"], handle_router_graph_verify),
        ("nanlm_router_graph_walk",
         "Read-only bounded candidate paths across verified NaNROUTER L0-L9 state and complete revision history in the ONE HERMES chart. Optional pinned Ed25519 sender authentication; no evidence truth, MTH execution or learning authority.",
         {"pages": {"type": "array", "minItems": 1, "maxItems": 128,
                    "items": {"type": "object"}},
          "history_pages": {"type": "array", "minItems": 1, "maxItems": 128,
                            "items": {"type": "object"}},
          "start": {"type": "integer", "minimum": 0, "maximum": 32895},
          "target": {"type": "integer", "minimum": 0, "maximum": 32895},
          "layer": {"type": "integer", "minimum": 0, "maximum": 9},
          "relation": {"type": "string", "enum": [
              "isa", "part_of", "defines", "supports", "refutes",
              "equivalent", "transforms", "causes", "depends_on",
              "similar", "contradicts", "observes", "routes_to"]},
          "max_hops": {"type": "integer", "minimum": 0, "maximum": 8},
          "max_nodes": {"type": "integer", "minimum": 1, "maximum": 1024},
          "max_edges": {"type": "integer", "minimum": 1, "maximum": 12800},
          "attestation": {"type": "object"}},
         ["pages", "history_pages", "start"], handle_router_graph_walk),
        ("nanlm_self_modify", "Boolean-gated self-modification of scratch leaves and MP relations: propose a relation, version a leaf, promote under the local approval gate, or inspect status.",
         {"action": {"type": "string", "enum": ["status", "relation", "leaf", "promote"]},
          "relation": {"type": "object"},
          "layer": {"type": "integer", "minimum": 0, "maximum": 9},
          "source_head": {"type": "integer", "minimum": 1, "maximum": 255},
          "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
          "strength_q16": {"type": "integer", "minimum": 0, "maximum": 65536},
          "polarity": {"type": "integer", "minimum": -1, "maximum": 1},
          "expected_generation": {"type": "integer", "minimum": 0},
          "mp": {"type": "integer", "minimum": 0, "maximum": 32639},
          "leaf_id": {"type": "string", "maxLength": 256},
          "payload": {},
          "approved": {"type": "boolean"},
          "tick": {"type": "integer", "minimum": 0}},
         ["action"], handle_self_modify),
        ("nanlm_mth_tools", "Use the bounded HERMES MTH tool facade: catalog, inspect, categories, compose, prepare or route over L0-L9.",
         {"tool": {"type": "string", "enum": ["mth.catalog", "mth.inspect", "mth.categories", "mth.compose", "mth.prepare", "mth.route"]},
          "arguments": {"type": "object"}},
         ["tool"], handle_mth_tools),
        ("nanlm_teacher_teach", "Ask the configured teacher provider (Hermes LLM, Ollama/Gemma, or OpenAI-compatible) for a bounded typed record batch and feed it into the L0-L9 scratch learner. Teacher records are evidence, never canonical truth.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 512},
          "count": {"type": "integer", "minimum": 1, "maximum": 64}},
         ["topic"], handle_teacher_teach),
        ("nanlm_learning_cycle", "Run one L0-L9 layered learning cycle: project the message through all ten layers, distribute it across all 255 HW0 heads, write paged scratch leaves, update concepts and optionally promote under the conservative policy.",
         {"message": {"type": "string", "minLength": 1, "maxLength": 1048576},
          "source": {"type": "string", "maxLength": 128},
          "relations": {"type": "array", "maxItems": 512},
          "tick": {"type": "integer", "minimum": 0},
          "approved": {"type": "boolean"}},
         ["message"], handle_learning_cycle),
        ("nanlm_learning_status", "Inspect the L0-L9 layered learner, 255-head ensemble, 1 GiB paged scratch register, concept matrix and local head register.",
         {}, [], handle_learning_status),
        ("nanlm_learning_sync", "Promote conservative scratch relation candidates into the local 8 MiB head register; canonical BS0/MC0 merge remains a separate gate.",
         {"approved": {"type": "boolean"}}, ["approved"], handle_learning_sync),
        ("nanlm_wordnet_ingest", "Read a word from the sparse HW4 WordNet universe, project definitions and typed hypernym relations into L0-L9 scratch leaves and optionally promote them.",
         {"word": {"type": "string", "minLength": 1, "maxLength": 128},
          "depth": {"type": "integer", "minimum": 1, "maximum": 8},
          "approved": {"type": "boolean"}},
         ["word"], handle_wordnet_ingest),
        ("nanlm_scratch_register", "Inspect bounded records in the 1 GiB logical scratch register space (128 x 8 MiB pages).",
         {"layer": {"type": "integer", "minimum": 0, "maximum": 9},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "limit": {"type": "integer", "minimum": 1, "maximum": 256}},
         [], handle_scratch_register),
        ("nanlm_chat_train", "Run free-form conversational learning after the mandatory 256-head native Boolean logic foundation; route user->Hermes->NaNLMind proposals and return evidence without bypassing canonical merge.",
         {"message": {"type": "string", "minLength": 1, "maxLength": 1048576},
          "session_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,80}$"},
          "system_prompt": {"type": "string", "maxLength": 16384},
          "temperature": {"type": "number", "minimum": 0, "maximum": 2},
          "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
          "sources": {"type": "array", "maxItems": 7}},
         ["message"], handle_chat_train),
        ("nanlm_tree_inspect", "Inspect the experimental HW0:1-255 head -> numbered math branch -> data leaf hierarchy, without claiming a canonical merge.",
         {"head": {"type": "integer", "minimum": 1, "maximum": 255},
          "branch": {"type": "string", "maxLength": 48},
          "leaf": {"type": "string", "maxLength": 40},
          "limit": {"type": "integer", "minimum": 1, "maximum": 256}},
         [], handle_tree_inspect),
        ("nanlm_tree_mutate", "Stage one experimental data leaf in HERMES isolated 8MiB scratch register; optimistic revision and SHA audit; never changes canonical heads.",
         {"head": {"type": "integer", "minimum": 1, "maximum": 255},
          "branch": {"type": "string", "pattern": "^(vector|hyperedge|matrix|transform|math|relation|word|logic)[0-9]{1,6}$"},
          "leaf": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,39}$"},
          "data_hex": {"type": "string", "maxLength": 131072},
          "expected_revision": {"type": "integer", "minimum": 0}},
         ["head", "branch", "leaf", "data_hex"], handle_tree_mutate),
        ("nanlm_tree_audit", "Inspect SHA-256 logged experimental register mutations without accessing canonical BS0 or MC0 merge gates.",
         {"limit": {"type": "integer", "minimum": 1, "maximum": 256}},
         [], handle_tree_audit),
        ("nanlm_clocked_plan", "Propose an autonomous local/CLOCKED mode and record a bounded multi-source plan without transmitting source text.",
         {"question": {"type": "string", "minLength": 1, "maxLength": 65536},
          "sources": {"type": "array", "maxItems": 7, "items": {"type": "object",
             "required": ["text"], "properties": {"id": {"type": "string", "maxLength": 80},
             "text": {"type": "string", "minLength": 1, "maxLength": 65536}}}},
          "scratch_id": {"type": "string", "maxLength": 80}},
         ["question"], handle_clocked_plan),
        ("nanlm_clocked_execute", "Execute a reviewed multi-source CLOCKED task plan through the scoped CLOCKED gateway. Requires Hermes human tool approval.",
         {"question": {"type": "string", "minLength": 1, "maxLength": 65536},
          "sources": {"type": "array", "maxItems": 7, "items": {"type": "object",
             "required": ["text"], "properties": {"id": {"type": "string", "maxLength": 80},
             "text": {"type": "string", "minLength": 1, "maxLength": 65536}}}},
          "scratch_id": {"type": "string", "maxLength": 80},
          "plan_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "max_requests": {"type": "integer", "minimum": 1, "maximum": 16, "default": 16},
          "max_bytes": {"type": "integer", "minimum": 1, "maximum": 1048576, "default": 1048576}},
         ["question", "plan_sha256"], handle_clocked_execute),
        ("nanlm_clocked_scratch", "Inspect a revisioned HERMES scratch or one writer-owned lane.",
         {"scratch_id": {"type": "string", "maxLength": 80},
          "lane": {"type": "string", "enum": ["user", "hermes_plan", "clocked_request", "clocked_result", "evidence_atoms", "conflict_map", "nan16_package", "head_response", "hermes_synthesis", "head_validation", "final", "learning_proposal", "phone_user", "phone_hermes", "phone_nanlm", "phone_event"]}},
         [], handle_clocked_scratch),
        ("nanlm_head_process", "Submit a frozen typed evidence package or answer draft to the NaNLMind head-cortex interface; HERMES never owns canonical head state.",
         {"question": {"type": "string", "minLength": 1, "maxLength": 65536},
          "scratch_id": {"type": "string", "maxLength": 80},
          "purpose": {"type": "string", "enum": ["EVIDENCE_COGNITION", "ANSWER_VALIDATION"]},
          "draft": {"type": "string", "maxLength": 65536}},
         ["question"], handle_head_process),
        ("nanlm_clocked_synthesize", "Synthesize a final answer from revisioned CLOCKED evidence, submit the draft through NaNLMind heads for answer validation, and compose the final response without treating head predictions as ground truth.",
         {"question": {"type": "string", "minLength": 1, "maxLength": 65536},
          "scratch_id": {"type": "string", "maxLength": 80},
          "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192, "default": 2048}},
         ["question"], handle_clocked_synthesize),
        ("nanlm_head_candidate", "Create an isolated Boolean-mask mutation candidate for one NaNLMind head. Requires Hermes human approval and training:execute scope; never changes canonical state.",
         {"mutation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 255},
          "patch_mask": {"type": "integer", "minimum": 1, "maximum": 65535},
          "base_topology_digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "scratch_id": {"type": "string", "maxLength": 80}},
         ["mutation_id", "head_index", "patch_mask", "base_topology_digest"],
         handle_head_candidate),
        ("nanlm_head_learn_candidate", "Transform CLOCKED observations in revisioned scratch into a weak-supervision training set and train an isolated NaNLMind candidate topology. Requires Hermes human approval and training:execute; CLOCKED is not ground truth and canonical state is never modified.",
         {"mutation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
          "base_topology_digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "scratch_id": {"type": "string", "maxLength": 80},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
         },
         ["mutation_id"], handle_head_learn_candidate),
        ("nanlm_head_candidate_evaluate", "Evaluate an isolated Boolean-head candidate against explicit held-out N16REL rows and record baseline/candidate metrics. Requires Hermes human approval and training:execute; never promotes.",
         {"mutation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
          "records": {"type": "array", "minItems": 1, "maxItems": 512},
          "scratch_id": {"type": "string", "maxLength": 80}},
         ["mutation_id", "records"], handle_head_candidate_evaluate),
        ("nanlm_status", "Inspect local NaNLMind universe, durable mathematical-memory status and permanent plugin-duration session context.", {}, [], handle_status),
        ("nanlm_session_context", "Inspect the permanent NANNET architecture/routing, WordNet HW4 and free-chat learning documents loaded at plugin start; optionally return one exact shipped document for re-intake.",
         {"document": {"type": "string", "enum": ["permanent-instructions", "nannet-architecture", "nannet-routing", "wordnet-hw4", "free-chat-learning", "mth-universal-runtime", "mth-data-leaves"]},
          "include_text": {"type": "boolean", "default": False},
          "max_bytes": {"type": "integer", "minimum": 1, "maximum": 16384}},
         [], handle_session_context),
        ("nanlm_chart_storage", "Inspect the SHA-verified local 8 MiB NaN16 chart snapshot or explicitly checkpoint it; this is never a canonical training or NANNET merge commit.",
         {"action": {"type": "string", "enum": ["inspect", "checkpoint"], "default": "inspect"},
          "approved": {"type": "boolean", "default": False}},
         [], handle_chart_storage),
        ("nanlm_resident_status", "Inspect the bounded one-file resident universe: mmap budget, generation/digest, leaves, relations, hyperedges, features, pair cells and the disabled canonical merge gate.",
         {}, [], handle_resident_status),
        ("nanlm_resident_ingest", "Explicitly ingest bounded local evidence rows into the resident universe. Requires approved=true; never performs canonical merge or promotion.",
         {"approved": {"type": "boolean", "const": True},
          "message": {"type": "string", "minLength": 1, "maxLength": 65536},
          "text": {"type": "string", "minLength": 1, "maxLength": 65536},
          "kind": {"type": "string", "maxLength": 32},
          "source": {"type": "string", "maxLength": 128},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "rows": {"type": "array", "maxItems": 64, "items": {"type": "object"}}},
         ["approved"], handle_resident_ingest),
        ("nanlm_resident_query", "Read-only resonance/superposition and exact-link query over the resident universe; probe reads do not mutate state.",
         {"text": {"type": "string", "maxLength": 65536},
          "subject": {"type": "string", "maxLength": 512},
          "predicate": {"type": "string", "maxLength": 512},
          "object": {"type": "string", "maxLength": 512},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "top_k": {"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
          "limit": {"type": "integer", "minimum": 1, "maximum": 64, "default": 16},
          "mode": {"type": "string", "enum": ["dot", "cosine", "phase"], "default": "cosine"}},
         [], handle_resident_query),
        ("nanlm_resident_evidence", "Inspect or explicitly append/delete/restore support and refute evidence in the full-cycle resident lanes; tell/delete/load require training scope and approved=true, while status/state/ask/save are read-only. Returns the four-state unknown/supports/refutes/both result without cancellation.",
         {"action": {"type": "string", "enum": ["status", "state", "ask", "tell", "delete", "save", "load"]},
          "approved": {"type": "boolean", "const": True},
          "statement": {"type": "string", "minLength": 1, "maxLength": 512},
          "question": {"type": "string", "minLength": 1, "maxLength": 512},
          "subject": {"type": "string", "minLength": 1, "maxLength": 512},
          "predicate": {"type": "string", "minLength": 1, "maxLength": 64, "default": "is a"},
          "object": {"type": "string", "minLength": 1, "maxLength": 512},
          "polarity": {"type": "string", "enum": ["supports", "refutes"]},
          "expected_generation": {"type": "integer", "minimum": 0},
          "snapshot": {"type": "object"}},
         ["action"], handle_resident_evidence),
        ("nanlm_head_field", "Inspect or explicitly operate the independent 256-head Boolean field: status/query are read-only; activate/propose/commit require training scope, with scratch-layer staging before an approved generation-CAS main write.",
         {"action": {"type": "string", "enum": ["status", "activate", "propose", "commit", "query"]},
          "approved": {"type": "boolean", "const": True},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "delta": {"type": "number", "minimum": -1.0, "maximum": 1.0},
          "reason": {"type": "string", "maxLength": 240},
          "token": {"type": "string", "maxLength": 256},
          "expected_generation": {"type": "integer", "minimum": 0},
          "text": {"type": "string", "maxLength": 4096},
          "top_k": {"type": "integer", "minimum": 1, "maximum": 32},
          "limit": {"type": "integer", "minimum": 1, "maximum": 256}},
         ["action"], handle_head_field),
        ("nanlm_chat", "Open-ended Hermes conversation; reflect evidence and, when explicit supervised relations are present, update native heads, mirror pairs and hyperedges in the shared universe.",
         {"message": {"type": "string", "minLength": 1, "maxLength": 1048576},
          "session_id": {"type": "string", "maxLength": 80}, "system_prompt": {"type": "string"},
          "temperature": {"type": "number", "minimum": 0, "maximum": 2},
          "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
          "probe": {"type": "boolean", "default": False,
                    "description": "Read-only evaluation probe: do not update memory, transcripts, native topology or layered scratch."},
          "sources": {"type": "array", "maxItems": 7, "items": {"type": "object",
              "required": ["text"], "properties": {"id": {"type": "string", "maxLength": 80},
              "text": {"type": "string", "minLength": 1, "maxLength": 65536}}}}},
         ["message"], handle_chat),
        ("nanlm_chat_history", "Read the durable transcript for an open-ended Hermes conversation.",
         {"session_id": {"type": "string", "maxLength": 80},
          "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
         [], handle_chat_history),
        ("nanlm_phone_call", "Three-way phone-call chat: user -> Hermes -> NaNLM -> user + Hermes. The shared scratch journal is the call window; user can address Hermes or NaNLM and interrupt between runs. The routing kernel is a raw NaN16 program.",
         {"session_id": {"type": "string", "maxLength": 64},
          "action": {"type": "string", "enum": ["start", "say", "run", "status", "end"]},
          "message": {"type": "string", "maxLength": 65536},
          "addressee": {"type": "string", "enum": ["hermes", "nanlm", "both"]},
          "rounds": {"type": "integer", "minimum": 1, "maximum": 64},
          "clocked": {"type": "boolean", "default": False}},
         [], handle_phone_call),
        ("nanlm_auth_init", "Create this installation's encrypted NANNET identity.", {}, [], handle_auth_init),
        ("nanlm_auth_status", "Inspect local NANNET identity and pairing status without revealing secrets.", {}, [], handle_auth_status),
        ("nanlm_auth_pair", "Pair this installation using a one-time code from the trusted NANNET authority.",
         {"pairing_code": {"type": "string", "minLength": 12, "maxLength": 256}}, ["pairing_code"], handle_auth_pair),
        ("nanlm_auth_revoke", "Revoke this installation token and erase its local encrypted copy.", {}, [], handle_auth_revoke),
        ("nanlm_hw_register", "Explicitly register a new HW5–HW255 NANNET hardware node using Android-compatible consent-based HTTPS enrollment; does not issue a guest universe.",
         {"consent": {"type": "boolean", "const": True}}, ["consent"], handle_hw_register),
        ("nanlm_hw_status", "Inspect only sanitized local NANNET hardware-token and lease metadata; never reveal the bearer.", {}, [], handle_hw_status),
        ("nanlm_hw_heartbeat", "Renew an enrolled NANNET hardware-node lease with its encrypted bearer; preserves the bound hardware identity.", {}, [], handle_hw_heartbeat),
        ("nanlm_hw_release", "Release a NANNET hardware-node lease after server approval; retain installation key for later explicit re-registration.", {}, [], handle_hw_release),
        ("nanlm_hw_report", "Send an authenticated bounded device/system issue report through NANNET.",
         {"report_type": {"type": "string", "enum": ["device_error", "system_error", "network_error", "security_concern", "other"]},
          "summary": {"type": "string", "minLength": 1, "maxLength": 240},
          "details": {"type": "string", "maxLength": 4000}},
         ["report_type", "summary"], handle_hw_report),
        ("nanlm_hw2_bind", "Bind the permanent fixed HERMES HW2/node-point-3 identity through the NANNET authority; requires explicit consent and an admin/fixed-binding scope. Never allocates HW5-HW255 and never returns the bearer.",
         {"consent": {"type": "boolean", "const": True}},
         ["consent"], handle_hw2_bind),
        ("nanlm_hw2_status", "Inspect sanitized local (and optionally remote) fixed HW2 binding metadata without revealing the guest bearer.",
         {"remote": {"type": "boolean", "default": False}},
         [], handle_hw2_status),
        ("nanlm_hw2_report", "Send a bounded report as the fixed HW2 HERMES identity through NANNET.",
         {"report_type": {"type": "string", "enum": ["device_error", "system_error", "network_error", "security_concern", "other"]},
          "summary": {"type": "string", "minLength": 1, "maxLength": 240},
          "details": {"type": "string", "maxLength": 4000}},
         ["report_type", "summary"], handle_hw2_report),
        ("nanlm_universe_status", "Inspect encrypted local HERMES router leases without returning bearer credentials or receipt MACs.", {}, [], handle_universe_status),
        ("nanlm_universe_open", "Request a distinct router-issued universe via NANNET's hermes:lease signed installation scope; never local allocation.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "ttl_seconds": {"type": "integer", "minimum": 60, "maximum": 3600, "default": 900}}, [], handle_universe_open),
        ("nanlm_universe_resolve", "Live-verify one local 0..254 head against the authenticated router lease; does not execute NaN16 guest code.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 254, "default": 0}}, [], handle_universe_resolve),
        ("nanlm_universe_renew", "Extend a router-issued lease while rotating its receipt revision; older receipts are rejected.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "ttl_seconds": {"type": "integer", "minimum": 60, "maximum": 3600, "default": 900}}, [], handle_universe_renew),
        ("nanlm_universe_release", "Revoke a router-issued universe and erase its local encrypted receipt after router confirmation.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"}}, [], handle_universe_release),
        ("nanlm_universe_dispatch", "Submit one bounded opaque N16T frame to router-authenticated native HERMES guest with stable dispatch_id for at-most-once processing; not a training commit.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 254, "default": 0},
          "dispatch_id": {"type": "string", "pattern": "^[A-Za-z0-9_.:-]{12,128}$"},
          "frame_hex": {"type": "string", "minLength": 72, "maxLength": 131144, "pattern": "^[0-9a-fA-F]+$"}},
         ["dispatch_id", "frame_hex"], handle_universe_dispatch),
        ("nanlm_universe_ack", "Confirm SHA-256 of a completed native N16T response in the durable router journal without implying model training.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 254, "default": 0},
          "dispatch_id": {"type": "string", "pattern": "^[A-Za-z0-9_.:-]{12,128}$"},
          "response_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
         ["dispatch_id", "response_sha256"], handle_universe_ack),
        ("nanlm_universe_inspect", "Recover a journaled native N16T dispatch without re-running ambiguous work; return the verified frame if completed.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 254, "default": 0},
          "dispatch_id": {"type": "string", "pattern": "^[A-Za-z0-9_.:-]{12,128}$"}},
         ["dispatch_id"], handle_universe_inspect),
        ("nanlm_native_mth_u32", "Run a real native NaN16 MTH checked uint32 add/subtract inside a signed HERMES leased hardware universe, with overflow status.",
         {"alias": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$", "default": "primary"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 254, "default": 0},
          "kernel_id": {"type": "string", "enum": ["mth.u32.add_checked", "mth.u32.sub_checked"], "default": "mth.u32.add_checked"},
          "a": {"type": "integer", "minimum": 0, "maximum": 4294967295},
          "b": {"type": "integer", "minimum": 0, "maximum": 4294967295}},
         ["a", "b"], handle_native_mth_u32),
        ("nanlm_mth_catalog", "Browse the 91 bundled native NaN16 MTH kernels by family, tags or CI status. Catalog discovery does not claim a general operand ABI.",
         {"family": {"type": "string", "maxLength": 48},
          "tags": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 80}},
          "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "default": 0},
          "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 32},
          "require_ci": {"type": "boolean", "default": False}},
         [], handle_mth_catalog),
        ("nanlm_mth_inspect", "Inspect one bundled MTH kernel's source/fixture digests, shape, precision, workspace, native execution mode and exact paths.",
         {"kernel_id": {"type": "string", "minLength": 5, "maxLength": 160,
                        "pattern": "^mth\\.[a-z0-9_.-]+$"}},
         ["kernel_id"], handle_mth_inspect),
        ("nanlm_mth_runtime", "Autodetect the host architecture, compiler, bundled/external NaN0S core, native CLI build state and N16I availability for the universal MTH runtime.",
         {"build": {"type": "boolean", "default": False}},
         [], handle_mth_runtime),
        ("nanlm_mth_verify", "Assemble and run one bundled MTH kernel's native NaN16 fixture on the host-built portable NaN0S core; returns the exact marker, steps, trap and honest non-claims.",
         {"kernel_id": {"type": "string", "minLength": 5, "maxLength": 160,
                        "pattern": "^mth\\.[a-z0-9_.-]+$"},
          "max_steps": {"type": "integer", "minimum": 1, "maximum": 500000000},
          "timeout": {"type": "integer", "minimum": 1, "maximum": 900, "default": 120}},
         ["kernel_id"], handle_mth_verify),
        ("nanlm_mth_execute", "Run one of the four admitted float32 vector kernels (dot, cosine, add, sub) with exactly 256 numeric x and y operands on the host-built native NaN0S core; no host Python math.",
         {"kernel_id": {"type": "string", "enum": ["mth.vector.dot256.f32", "mth.vector.cosine256.f32", "mth.vector.add256.f32", "mth.vector.sub256.f32"]},
          "x": {"type": "array", "minItems": 256, "maxItems": 256, "items": {"type": "number"}},
          "y": {"type": "array", "minItems": 256, "maxItems": 256, "items": {"type": "number"}},
          "timeout": {"type": "integer", "minimum": 1, "maximum": 900, "default": 120}},
         ["kernel_id", "x", "y"], handle_mth_execute),
        ("nanlm_mth_n16i_selftest", "Run the nan0s differential gate: the same NaN16 test image natively and under the raw-NaN N16I interpreter-in-NaN16; serial bytes must be identical.",
         {"timeout": {"type": "integer", "minimum": 1, "maximum": 900, "default": 120}},
         [], handle_mth_n16i_selftest),
        ("nanlm_mth_dispatch", "Assemble any bundled MTH kernel/fixture and dispatch the arbitrary NaN16 guest image under N16I, with bounded step budget and exact serial/output receipt.",
         {"kernel_id": {"type": "string", "minLength": 5, "maxLength": 160,
                        "pattern": "^mth\\.[a-z0-9_.-]+$"},
          "use_fixture": {"type": "boolean", "default": True},
          "entry": {"type": "integer", "minimum": 0, "default": 0},
          "max_steps": {"type": "integer", "minimum": 1, "maximum": 500000000, "default": 100000000},
          "timeout": {"type": "integer", "minimum": 1, "maximum": 900, "default": 120}},
         ["kernel_id"], handle_mth_dispatch),
        ("nanlm_mth_hub", "Inspect the immutable MTH data-leaf hub tree, vector root, WordNet HW4 link, constraints, invariants and pair rules.",
         {}, [], handle_mth_hub),
        ("nanlm_mth_tree", "Map or audit every one of the 91 bundled MTH kernels into immutable function leaves under family hubs, with source/fixture digests, hub ancestry and NaN-executable image verification.",
         {"action": {"type": "string", "enum": ["status", "map", "audit"], "default": "status"},
          "verify_images": {"type": "boolean", "default": True}},
         [], handle_mth_tree),
        ("nanlm_mth_leaf_create", "Create an immutable NaN-executable MTH data leaf with function definition, action relations, identities, constraints, constants and hub ancestry.",
         {"kind": {"type": "string", "enum": ["vector", "scalar", "matrix", "relation"]},
          "payload": {"type": "object", "properties": {
              "values": {"type": "array", "minItems": 256, "maxItems": 256,
                         "items": {"type": "number"}},
              "fill": {"type": "number",
                       "description": "vector convenience: repeat this finite value across all 256 elements"},
              "value": {"type": "number"},
              "dimension": {"type": "integer", "const": 256}}},
          "function_definition": {"type": "object"},
          "action_relations": {"type": "array", "maxItems": 64, "items": {"type": "object"}},
          "identities": {"type": "array", "maxItems": 64, "items": {"type": "string", "maxLength": 240}},
          "constraints": {"type": "array", "maxItems": 64, "items": {"type": "string", "maxLength": 240}},
          "constants": {"type": "object"},
          "wordnet": {"type": "object"},
          "parent": {"type": "string", "maxLength": 80},
          "alias": {"type": "string", "pattern": "^[A-Za-z0-9_.:-]{1,80}$"}},
         ["kind"], handle_mth_leaf_create),
        ("nanlm_mth_leaf_get", "Read one immutable MTH data leaf and its NaN16 image metadata; optionally include the exact .nan text.",
         {"leaf_id": {"type": "string", "minLength": 3, "maxLength": 80},
          "include_image": {"type": "boolean", "default": False}},
         ["leaf_id"], handle_mth_leaf_get),
        ("nanlm_mth_pair", "Resolve left::right through the hub rules. connect creates a relation leaf; dot/cosine/add/sub execute the native kernel, check identities and create a new immutable hash leaf. Illegal pairings deny without execution.",
         {"left": {"type": "string", "minLength": 1, "maxLength": 80},
          "right": {"type": "string", "minLength": 1, "maxLength": 80},
          "operator": {"type": "string", "enum": ["connect", "dot", "cosine", "add", "sub"], "default": "connect"},
          "alias": {"type": "string", "pattern": "^[A-Za-z0-9_.:-]{1,80}$"},
          "check_identities": {"type": "boolean", "default": True}},
         ["left", "right"], handle_mth_pair),
        ("nanlm_mth_leaf_verify", "Execute a leaf's real NaN16 image on the Python NaNVM0 boundary and verify its magic/output; no canonical commit.",
         {"leaf_id": {"type": "string", "minLength": 3, "maxLength": 80}},
         ["leaf_id"], handle_mth_leaf_verify),
        ("nanlm_generate_curriculum", "Generate a balanced 12-stage logic-to-language curriculum corpus, including Boolean inverses, deep relations, exceptions, word choice, sentences, paragraphs and stories.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 3, "maximum": 64, "default": 12}},
         ["topic"], handle_curriculum),
        ("nanlm_curriculum_experiment", "Generate teacher-labeled structured relations, train the native relation objective with frozen held-out challenges at each selected L0-L9 resolution, and merge training evidence into shared memory.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 3, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         ["topic"], handle_curriculum_experiment),
        ("nanlm_logic_session", "Explicitly train Boolean, nibble and byte relation tasks after the exact 1,024-check Boolean foundation gate. Uses separate held-out resolution metrics.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 3, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_logic_session),
        ("nanlm_word_session", "Explicitly train phonetic, morphemic, lexical and word-completion relation tasks with separate frozen held-out metrics.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 3, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_word_session),
        ("nanlm_sentence_session", "Explicitly train sentence, paragraph and narrative relation tasks with separate frozen held-out metrics.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 3, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_sentence_session),
        ("nanlm_training_pipeline", "Explicitly run structured teacher training for selected task stages. Each linguistic resolution has its own native relation objective and frozen held-out challenge; normal conversation uses nanlm_chat for online free-form relation learning.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "mode": {"type": "string", "enum": ["pipeline", "logic", "words", "sentences"], "default": "pipeline"},
          "stages": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
          "records_per_stage": {"type": "integer", "minimum": 3, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_training_pipeline),
        ("nanlm_nan16de_route", "Send one signed NaN16DE raw NaN16 double-endian envelope to the authenticated NANNET portal.",
         {"route": {"type": "string", "pattern": "^nan://HW[0-9]{1,3}:BS[0-9]{1,3}:MC[0-9]{1,3}:SC[0-9]{1,3}:GM[0-9]{1,3}:UM[0-9]{1,3}$"},
          "kind": {"type": "string", "minLength": 1, "maxLength": 64},
          "far_type": {"type": "integer", "minimum": 0, "maximum": 3},
          "near_type": {"type": "integer", "minimum": 0, "maximum": 3},
          "payload": {"type": "integer", "minimum": 0, "maximum": 4294967295},
          "epoch": {"type": "integer", "minimum": 0}},
         ["route"], handle_nan16de_route),
        ("nanlm_generate_corpus", "Generate a bounded N16TXT/1 corpus using Hermes' active model.",
         {"topic": {"type": "string"}, "records": {"type": "integer", "minimum": 1, "maximum": 64, "default": 24}}, ["topic"], handle_generate),
        ("nanlm_train", "Run the native NaNLMind Context2 bootstrap trainer locally.",
         {"corpus_path": {"type": "string"}, "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 360}}, ["corpus_path"], handle_train),
        ("nanlm_math_train", "Train all 256 native heads, resume the serialized global topology, export structured discoveries, and merge them into additive memory.",
         {"corpus_path": {"type": "string"}, "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         ["corpus_path"], handle_math_train),

        ("nanlm_structured_relation_train", "Train the shared native heads from schema-bound relation records across L0-L9. Train rows mutate state; test rows are held out. Every accepted event updates mirror-pair cells and a provenance-bound hyperedge.",
         {"records": {
              "type": "array", "minItems": 2, "maxItems": 10000,
              "items": {
                  "type": "object",
                  "required": ["split", "source_digest", "item"],
                  "properties": {
                      "split": {"type": "string", "enum": ["train", "test"]},
                      "source_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
                      "item": {
                          "type": "object",
                          "required": ["subject", "predicate", "object", "polarity", "confidence", "evidence", "provenance"],
                          "properties": {
                              "subject": {"type": "string", "minLength": 1},
                              "predicate": {"type": "string", "minLength": 1},
                              "object": {"type": "string", "minLength": 1},
                              "polarity": {"type": "string", "enum": ["supports", "refutes"]},
                              "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                              "evidence": {"type": "object"},
                              "provenance": {
                                  "type": "object", "required": ["resolution"],
                                  "properties": {"resolution": {"type": "string", "enum": ["bit", "nibble", "byte", "phonetic", "morpheme", "lexical", "word_completion", "sentence", "paragraph", "narrative"]}}
                              }
                          }
                      }
                  }
              }
          },
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 1}},
         ["records"], handle_relation_train),
        ("nanlm_evaluate", "Run native topology-byte, Kneser-Ney and Context2 controls locally on the same corpus.",
         {"corpus_path": {"type": "string"}}, ["corpus_path"], handle_evaluate),
        ("nanlm_memory_ingest", "Append structured mathematical discoveries to the durable additive universe memory.",
         {"discoveries": {"type": "array", "minItems": 1, "maxItems": 1000, "items": {"type": "object"}},
          "source_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "layer": {"type": "integer", "minimum": 0, "maximum": 9}},
         ["discoveries", "source_digest"], handle_memory_ingest),
        ("nanlm_memory_reorganize", "Rebuild deterministic head/layer memory projections without deleting discoveries.", {}, [], handle_memory_reorganize),
        ("nanlm_memory_cross_resolution", "Compute the ten-by-ten linguistic resolution interaction matrix, compare aligned evidence, preserve contradictions, and merge derived mathematical hyperedges into the shared universe.",
         {"limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 1000}},
         [], handle_memory_cross_resolution),
        ("nanlm_memory_merge", "Idempotently union peer discovery events into the local additive memory ledger.",
         {"records": {"type": "array", "minItems": 1, "maxItems": 1000,
                      "items": {"type": "object", "required": ["item", "source_digest"],
                                "properties": {"item": {"type": "object"},
                                               "source_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
                                               "head": {"type": "integer", "minimum": 0, "maximum": 255},
                                               "layer": {"type": "integer", "minimum": 0, "maximum": 9}},
                                "additionalProperties": False}}},
         ["records"], handle_memory_merge),
        ("nanlm_memory_query", "Query active mathematical-memory records by subject, head or layer.",
         {"subject": {"type": "string"}, "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "layer": {"type": "integer", "minimum": 0, "maximum": 9},
          "limit": {"type": "integer", "minimum": 1, "maximum": 1000}},
         [], handle_memory_query),
        ("nanlm_memory_assessments", "Inspect evidence-weighted support/refutation assessments; contested claims are promoted to the highest review layer.",
         {"subject": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 1000}},
         [], handle_memory_assessments),
    ]
    import json

    def tool_handler(handler):
        # Hermes tool handlers must return a JSON string. Internal helpers stay
        # Python-native so they can be composed and tested.
        def wrapped(params, **kwargs):
            return json.dumps(handler(params, **kwargs), sort_keys=True,
                              ensure_ascii=False, default=str)
        return wrapped

    for name, description, properties, required, handler in specs:
        ctx.register_tool(
            name=name, toolset="nanlmind", handler=tool_handler(handler),
            schema={"name": name, "description": description,
                    "parameters": {"type": "object", "properties": properties,
                                   "required": required, "additionalProperties": False}},
        )

    def handle_command(raw_args):
        import json as _json
        import shlex
        try:
            parts = shlex.split(raw_args or "")
        except ValueError:
            parts = (raw_args or "").split()
        command = parts[0].lower() if parts else "chat"
        rest = parts[1:]
        if command in {"help", "?"}:
            return _json.dumps({
                "usage": "/nanlm [chat|call|logic|words|sentences|pipeline|clocked|route|status] ...",
                "default": "chat",
                "note": "Bare /nanlm text is treated as a chat message.",
            }, sort_keys=True)
        if command == "status":
            return handle_status({})
        if command == "logic":
            return handle_logic_session({"topic": " ".join(rest) or "logic"})
        if command == "words":
            return handle_word_session({"topic": " ".join(rest) or "words"})
        if command == "sentences":
            return handle_sentence_session({"topic": " ".join(rest) or "sentences"})
        if command == "pipeline":
            return handle_training_pipeline({"topic": " ".join(rest) or "logic and language", "mode": "pipeline"})
        if command == "route":
            return _json.dumps(handle_nan16de_route({"route": rest[0] if rest else ""}), sort_keys=True)
        if command == "clocked":
            from pathlib import Path as _Path
            from . import clocked_tools as _clocked
            sub = rest[0].lower() if rest else "help"
            if sub == "scratch":
                return _json.dumps(_clocked.scratch_show(ctx, rest[1] if len(rest) > 1 else "default"), sort_keys=True, default=str)
            if sub == "plan" and len(rest) >= 2:
                data = _json.loads(_Path(rest[1]).expanduser().read_text(encoding="utf-8"))
                return _json.dumps(_clocked.clocked_plan(ctx, data["question"], data.get("sources", []), data.get("scratch_id", "default")), sort_keys=True, default=str)
            if sub == "run" and len(rest) >= 3 and "--approved" in rest[2:]:
                data = _json.loads(_Path(rest[1]).expanduser().read_text(encoding="utf-8"))
                return _json.dumps(_clocked.clocked_execute(ctx, data["question"], data.get("sources", []), data.get("scratch_id", "default"), True, data["plan_sha256"]), sort_keys=True, default=str)
            if sub == "head" and len(rest) >= 2:
                data = _json.loads(_Path(rest[1]).expanduser().read_text(encoding="utf-8"))
                return _json.dumps(_clocked.head_process(ctx, data["question"], data.get("scratch_id", "default"), data.get("purpose", "EVIDENCE_COGNITION"), data.get("draft")), sort_keys=True, default=str)
            if sub == "learn" and len(rest) >= 3 and "--approved" in rest[2:]:
                data = _json.loads(_Path(rest[1]).expanduser().read_text(encoding="utf-8"))
                return _json.dumps(_clocked.head_learn_candidate(
                    ctx, data["mutation_id"], data.get("base_topology_digest"),
                    True, data.get("scratch_id", "default"), data.get("epochs", 1)),
                    sort_keys=True, default=str)
            if sub == "evaluate" and len(rest) >= 3 and "--approved" in rest[2:]:
                data = _json.loads(_Path(rest[1]).expanduser().read_text(encoding="utf-8"))
                return _json.dumps(_clocked.head_candidate_evaluate(
                    ctx, data["mutation_id"], data["records"], True,
                    data.get("scratch_id", "default")), sort_keys=True, default=str)
            if sub == "synthesize" and len(rest) >= 2:
                question = " ".join(rest[1:])
                return _json.dumps(_clocked.clocked_synthesize(ctx, question), sort_keys=True, default=str)
            return _json.dumps({"usage": "/nanlm clocked [plan <request.json>|run <request.json> --approved|scratch [id]|head <request.json>|learn <request.json> --approved|evaluate <request.json> --approved|synthesize <question>]", "request_file": {"question": "string", "sources": [{"id": "source-a", "text": "source text"}], "scratch_id": "optional-id"}, "note": "Plan does not transmit source text. Run requires the explicit --approved flag."})
        if command == "call":
            from . import phone_call as _phone
            clocked = "--clocked" in rest
            rest = [token for token in rest if not token.startswith("--")]
            extra = {"clocked": True} if clocked else {}
            sub = rest[0].lower() if rest else "status"
            try:
                if sub == "start":
                    return _json.dumps(_phone.handle(ctx, {"action": "start", "message": " ".join(rest[1:]) or "Hello.", "addressee": "hermes", **extra}), sort_keys=True, default=str)
                if sub == "say":
                    return _json.dumps(_phone.handle(ctx, {"action": "say", "message": " ".join(rest[1:]) or "Hello.", "addressee": "hermes", **extra}), sort_keys=True, default=str)
                if sub in ("@hermes", "@nanlm", "@both"):
                    return _json.dumps(_phone.handle(ctx, {"action": "say", "message": " ".join(rest[1:]), "addressee": sub[1:], **extra}), sort_keys=True, default=str)
                if sub == "run":
                    rounds = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else 1
                    return _json.dumps(_phone.handle(ctx, {"action": "run", "rounds": rounds}), sort_keys=True, default=str)
                if sub == "status":
                    return _json.dumps(_phone.handle(ctx, {"action": "status"}), sort_keys=True, default=str)
                if sub == "end":
                    return _json.dumps(_phone.handle(ctx, {"action": "end"}), sort_keys=True, default=str)
                return _json.dumps({"usage": "/nanlm call [--clocked] [start <message>|say <message>|@hermes <message>|@nanlm <message>|@both <message>|run <rounds>|status|end]"})
            except Exception as exc:
                return _json.dumps({"error": type(exc).__name__ + ": " + str(exc)[:300]}, sort_keys=True)
        if command == "chat":
            message = " ".join(rest)
        else:
            message = raw_args.strip()
        if not message:
            return _json.dumps({"response": "NaNLMind chat is ready. Use /nanlm <message> or /nanlm help."})
        return _json.dumps(handle_chat({"message": message}), sort_keys=True, default=str)

    if hasattr(ctx, "register_command"):
        ctx.register_command(
            "nanlm", handle_command,
            description="NaNLMind default chat; explicit /nanlm logic|words|sentences|pipeline training modes.",
        )
