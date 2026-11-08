"""NaNLMind Hermes plugin entry point."""
from . import tools


def register(ctx):
    from . import clocked_tools

    # Boot the local-CPU runner immediately. The 8 MiB chart (genesis before
    # training) and the temporal register file are held in local memory so
    # heads can map from plugin boot, before any Boolean/local training.
    try:
        from .universe_chart import LocalRunner
        LocalRunner.open(ctx)
    except Exception:
        pass

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
        from .layered_learning import LayeredLearningCoordinator
        from .universe_chart import LocalRunner
        root, _corpora, _universe = tools._paths(ctx)
        key = str(root)
        coordinator = _learning_cache.get(key)
        if coordinator is not None:
            return coordinator
        runner = LocalRunner.open(ctx)
        dictionary = None
        try:
            archive = ctx.get_config("wordnet_archive", default="")
        except Exception:
            archive = ""
        if archive:
            try:
                from .wordnet_store import WordNetStore
                store = WordNetStore.discover(archive)
                dictionary = store.universe
            except Exception:
                dictionary = None
        coordinator = LayeredLearningCoordinator(
            root, chart=runner.chart, dictionary=dictionary,
            policy=_learning_policy())
        _learning_cache[key] = coordinator
        return coordinator

    def handle_learning_cycle(params, **kwargs):
        _require_scope("training:execute")
        coordinator = _learning(ctx)
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
        return _learning(ctx).synchronize(approved=params.get("approved", False))

    def handle_wordnet_ingest(params, **kwargs):
        _require_scope("training:execute")
        coordinator = _learning(ctx)
        result = coordinator.ingest_word(
            params["word"], depth=params.get("depth", 2))
        if params.get("approved", False):
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
        _require_scope("universe:read")
        return clocked_tools.clocked_execute(ctx, params["question"],
            params.get("sources", []), params.get("scratch_id", "default"),
            params.get("approved", False), params.get("plan_sha256", ""),
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
        _require_scope("training:execute")
        return clocked_tools.head_candidate(ctx, params["mutation_id"],
            params["head_index"], params["patch_mask"],
            params["base_topology_digest"], params.get("approved", False),
            params.get("scratch_id", "default"))

    def handle_head_learn_candidate(params, **kwargs):
        _require_scope("training:execute")
        return clocked_tools.head_learn_candidate(ctx, params["mutation_id"],
            params.get("base_topology_digest"), params.get("approved", False),
            params.get("scratch_id", "default"), params.get("epochs", 1))

    def handle_head_candidate_evaluate(params, **kwargs):
        _require_scope("training:execute")
        return clocked_tools.head_candidate_evaluate(ctx, params["mutation_id"],
            params["records"], params.get("approved", False),
            params.get("scratch_id", "default"))

    def handle_status(params, **kwargs):
        result = tools.status(ctx)
        result["mathematical_memory"] = tools.memory_status(ctx)
        return result

    def handle_chart_storage(params, **kwargs):
        action = params.get("action", "inspect")
        if action == "checkpoint":
            _require_scope("training:execute")
        return tools.chart_storage_tool(ctx, action, params.get("approved", False))

    def handle_curriculum(params, **kwargs):
        return tools.generate_curriculum(ctx, params["topic"], params.get("records_per_stage", 12))

    def handle_curriculum_experiment(params, **kwargs):
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

    def handle_chat(params, **kwargs):
        _require_scope("universe:read")
        try:
            learning_enabled = ctx.get_config("chat_learning", default=True)
        except Exception:
            learning_enabled = True
        if learning_enabled:
            _require_scope("training:execute")
        return tools.chat(ctx, params["message"], params.get("session_id", "default"),
                          params.get("system_prompt"), params.get("temperature", 0.7),
                          params.get("max_tokens", 2048), params.get("sources"))

    def handle_chat_train(params, **kwargs):
        """Mandatory Boolean foundation, then unrestricted dialogue and proposals."""
        _require_scope("training:execute")
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
        return answer

    def handle_chat_history(params, **kwargs):
        return tools.chat_history(ctx, params.get("session_id", "default"),
                                  params.get("limit", 40))

    def handle_phone_call(params, **kwargs):
        _require_scope("universe:read")
        try:
            learning_enabled = ctx.get_config("chat_learning", default=True)
        except Exception:
            learning_enabled = True
        if learning_enabled:
            _require_scope("training:execute")
        from . import phone_call
        return phone_call.handle(ctx, params)

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

    def _require_scope(scope):
        # Every protected action requires an encrypted paired installation token.
        # There is no environment-variable test bypass or implicit hardware registration.
        return _auth_tools().require_scope(ctx, scope)

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
        return tools.train(ctx, params["corpus_path"], params.get("duration_minutes"))

    def handle_math_train(params, **kwargs):
        _require_scope("training:execute")
        return tools.math_train(ctx, params["corpus_path"], params.get("epochs", 3),
                                params.get("max_samples_per_head", 4096))

    def handle_relation_train(params, **kwargs):
        _require_scope("training:execute")
        return tools.structured_relation_train(ctx, params["records"], params.get("epochs", 1))

    def handle_evaluate(params, **kwargs):
        _require_scope("universe:read")
        return tools.evaluate(ctx, params["corpus_path"])

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
        ("nanlm_clocked_execute", "Execute a reviewed multi-source CLOCKED task plan through the scoped CLOCKED gateway. Requires explicit approved=true.",
         {"question": {"type": "string", "minLength": 1, "maxLength": 65536},
          "sources": {"type": "array", "maxItems": 7, "items": {"type": "object",
             "required": ["text"], "properties": {"id": {"type": "string", "maxLength": 80},
             "text": {"type": "string", "minLength": 1, "maxLength": 65536}}}},
          "scratch_id": {"type": "string", "maxLength": 80},
          "approved": {"type": "boolean", "default": False},
          "plan_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "max_requests": {"type": "integer", "minimum": 1, "maximum": 16, "default": 16},
          "max_bytes": {"type": "integer", "minimum": 1, "maximum": 1048576, "default": 1048576}},
         ["question", "approved", "plan_sha256"], handle_clocked_execute),
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
        ("nanlm_head_candidate", "Create an isolated Boolean-mask mutation candidate for one NaNLMind head. Requires explicit approval and training:execute scope; never changes canonical state.",
         {"mutation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
          "head_index": {"type": "integer", "minimum": 0, "maximum": 255},
          "patch_mask": {"type": "integer", "minimum": 1, "maximum": 65535},
          "base_topology_digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "approved": {"type": "boolean", "default": False},
          "scratch_id": {"type": "string", "maxLength": 80}},
         ["mutation_id", "head_index", "patch_mask", "base_topology_digest", "approved"],
         handle_head_candidate),
        ("nanlm_head_learn_candidate", "Transform CLOCKED observations in revisioned scratch into a weak-supervision training set and train an isolated NaNLMind candidate topology. Requires explicit approval and training:execute; CLOCKED is not ground truth and canonical state is never modified.",
         {"mutation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
          "base_topology_digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
          "scratch_id": {"type": "string", "maxLength": 80},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
          "approved": {"type": "boolean", "default": False}},
         ["mutation_id", "approved"], handle_head_learn_candidate),
        ("nanlm_head_candidate_evaluate", "Evaluate an isolated Boolean-head candidate against explicit held-out N16REL rows and record baseline/candidate metrics. Requires approval and training:execute; never promotes.",
         {"mutation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
          "records": {"type": "array", "minItems": 1, "maxItems": 512},
          "approved": {"type": "boolean", "default": False},
          "scratch_id": {"type": "string", "maxLength": 80}},
         ["mutation_id", "records", "approved"], handle_head_candidate_evaluate),
        ("nanlm_status", "Inspect local NaNLMind universe and durable mathematical-memory status.", {}, [], handle_status),
        ("nanlm_chart_storage", "Inspect the SHA-verified local 8 MiB NaN16 chart snapshot or explicitly checkpoint it; this is never a canonical training or NANNET merge commit.",
         {"action": {"type": "string", "enum": ["inspect", "checkpoint"], "default": "inspect"},
          "approved": {"type": "boolean", "default": False}},
         [], handle_chart_storage),
        ("nanlm_chat", "Open-ended Hermes conversation; reflect evidence and, when explicit supervised relations are present, update native heads, mirror pairs and hyperedges in the shared universe.",
         {"message": {"type": "string", "minLength": 1, "maxLength": 1048576},
          "session_id": {"type": "string", "maxLength": 80}, "system_prompt": {"type": "string"},
          "temperature": {"type": "number", "minimum": 0, "maximum": 2},
          "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
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
