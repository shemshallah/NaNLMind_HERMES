"""NaNLMind Hermes plugin entry point."""
from . import tools


def register(ctx):
    from . import clocked_tools

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

    specs = [
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
