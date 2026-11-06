"""NaNLMind Hermes plugin entry point."""
import os

from . import tools


def register(ctx):
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
                          params.get("max_tokens", 2048))

    def handle_chat_history(params, **kwargs):
        return tools.chat_history(ctx, params.get("session_id", "default"),
                                  params.get("limit", 40))

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
                    "supported platform. Chat and local training do not require it."
                ) from exc
            raise

    def _hardware_scope(scope):
        portal = os.environ.get("NANNET_AUTH_URL", "").strip()
        if not portal:
            return False
        from .hw_hermes import authorize, ensure_hardware_key
        try:
            state = str(ctx.get_config("state_dir", "~/.local/share/nanlmind-hermes"))
        except Exception:
            state = "~/.local/share/nanlmind-hermes"
        key = ensure_hardware_key(portal, state)
        return authorize(portal, scope, key)

    def _device_scope(scope):
        """Android-style scoped device bearer, validated by NANNET."""
        token = os.environ.get("NANLM_DEVICE_TOKEN", "")
        base = (os.environ.get("NANNET_AUTH_URL") or
                os.environ.get("NANLM_INTAKE_URL", "")).rstrip("/")
        if not token or not base:
            return False
        import json as _json
        import urllib.request
        body = _json.dumps({"scope": scope}).encode()
        request = urllib.request.Request(
            base + "/v1/authorize", data=body,
            headers={"Authorization": "Bearer " + token,
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = _json.load(response)
        except Exception as exc:
            raise RuntimeError("NANNET device authorization failed; log in again") from exc
        if not result.get("authorized") or result.get("scope") != scope:
            raise RuntimeError("NANNET device token lacks the required scope")
        return True

    def _require_scope(scope):
        # NANNET login is required for chat and training. The only bypass is an
        # explicit test flag; production/local interactive use must pair first.
        if os.environ.get("NANLM_AUTH_TEST_BYPASS") == "1":
            return False
        if _hardware_scope(scope):
            return True
        if not os.environ.get("NANNET_AUTH_URL"):
            raise RuntimeError(
                "NANNET login is required before chat or training. "
                "Set NANNET_AUTH_URL, and optionally HW_HERMES_API; Hermes will "
                "register one hardware key if no system key is present."
            )
        raise RuntimeError("NANNET hardware registration or authorization failed")

    def handle_auth_init(params, **kwargs): return _auth_tools().auth_init(ctx)
    def handle_auth_status(params, **kwargs): return _auth_tools().auth_status(ctx)
    def handle_auth_pair(params, **kwargs): return _auth_tools().auth_pair(ctx, params["pairing_code"])
    def handle_auth_revoke(params, **kwargs): return _auth_tools().auth_revoke(ctx)

    def handle_nan16de_route(params, **kwargs):
        _require_scope("universe:read")
        from .auth import load_installation, decrypt_payload
        from .auth_tools import _master_key, _universe_dir
        from .nan16de import NanWord, make_envelope, send_envelope
        import json as _json
        root = _universe_dir(ctx)
        doc, private = load_installation(root, _master_key())
        token_path = root / "nannet_token.json"
        if not token_path.exists():
            raise RuntimeError("NANNET paired token is missing; pair before routing")
        envelope = _json.loads(token_path.read_text(encoding="utf-8"))
        token = decrypt_payload(_master_key(), envelope,
                                ("nannet-token-v1:" + doc["installation_id"]).encode()).decode()
        word = NanWord.from_bits(int(params.get("far_type", 0)),
                                 int(params.get("near_type", 0)),
                                 int(params.get("payload", 0)))
        env = make_envelope(params["route"], params.get("kind", "route"),
                            word, private, doc["installation_id"],
                            int(params.get("epoch", 0)))
        return send_envelope(os.environ["NANNET_AUTH_URL"], token, env)

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
        return tools.structured_relation_train(ctx, params["records"])

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
        ("nanlm_status", "Inspect local NaNLMind universe and durable mathematical-memory status.", {}, [], handle_status),
        ("nanlm_chat", "Open-ended Hermes conversation; reflect the exchange into the shared multi-resolution mathematical universe.",
         {"message": {"type": "string", "minLength": 1, "maxLength": 1048576},
          "session_id": {"type": "string", "maxLength": 80}, "system_prompt": {"type": "string"},
          "temperature": {"type": "number", "minimum": 0, "maximum": 2},
          "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192}},
         ["message"], handle_chat),
        ("nanlm_chat_history", "Read the durable transcript for an open-ended Hermes conversation.",
         {"session_id": {"type": "string", "maxLength": 80},
          "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
         [], handle_chat_history),
        ("nanlm_auth_init", "Create this installation's encrypted NANNET identity.", {}, [], handle_auth_init),
        ("nanlm_auth_status", "Inspect local NANNET identity and pairing status without revealing secrets.", {}, [], handle_auth_status),
        ("nanlm_auth_pair", "Pair this installation using a one-time code from the trusted NANNET authority.",
         {"pairing_code": {"type": "string", "minLength": 12, "maxLength": 256}}, ["pairing_code"], handle_auth_pair),
        ("nanlm_auth_revoke", "Revoke this installation token and erase its local encrypted copy.", {}, [], handle_auth_revoke),
        ("nanlm_generate_curriculum", "Generate a balanced 12-stage logic-to-language curriculum corpus, including Boolean inverses, deep relations, exceptions, word choice, sentences, paragraphs and stories.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12}},
         ["topic"], handle_curriculum),
        ("nanlm_curriculum_experiment", "Generate the balanced 12-stage corpus, train the native 256-head universe, and merge discoveries into shared mathematical memory.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         ["topic"], handle_curriculum_experiment),
        ("nanlm_logic_session", "Explicitly train the logic-only session: Boolean inverses and deep logic. Use only when the user asks for logic training; normal conversation uses nanlm_chat.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_logic_session),
        ("nanlm_word_session", "Explicitly train the word-only session: repetitive words, lexical words and word completion.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_word_session),
        ("nanlm_sentence_session", "Explicitly train the sentence/story session: sentence choice and narrative.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "records_per_stage": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         [], handle_sentence_session),
        ("nanlm_training_pipeline", "Explicitly run the full training pipeline, a preset, or a custom ordered stage list. Normal conversation must use nanlm_chat instead.",
         {"topic": {"type": "string", "minLength": 1, "maxLength": 500},
          "mode": {"type": "string", "enum": ["pipeline", "logic", "words", "sentences"], "default": "pipeline"},
          "stages": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
          "records_per_stage": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12},
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
         {"records": {"type": "array", "minItems": 2, "maxItems": 10000,
                      "items": {"type": "object",
                                "properties": {"split": {"type": "string", "enum": ["train", "test"]},
                                               "source_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
                                               "item": {"type": "object"}}}}},
         ["records"], handle_relation_train),
        ("nanlm_evaluate", "Run native topology-byte, Kneser-Ney and Context2 controls locally on the same corpus.",
         {"corpus_path": {"type": "string"}}, ["corpus_path"], handle_evaluate),
        ("nanlm_memory_ingest", "Append structured mathematical discoveries to the durable additive universe memory.",
         {"discoveries": {"type": "array", "minItems": 1, "maxItems": 1000, "items": {"type": "object"}},
          "source_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "layer": {"type": "integer", "minimum": 0, "maximum": 31}},
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
                                               "layer": {"type": "integer", "minimum": 0, "maximum": 31}},
                                "additionalProperties": False}}},
         ["records"], handle_memory_merge),
        ("nanlm_memory_query", "Query active mathematical-memory records by subject, head or layer.",
         {"subject": {"type": "string"}, "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "layer": {"type": "integer", "minimum": 0, "maximum": 31},
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
                "usage": "/nanlm [chat|logic|words|sentences|pipeline|route|status] ...",
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
