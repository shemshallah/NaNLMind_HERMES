"""NaNLMind Hermes plugin entry point."""
import os

from . import tools


def register(ctx):
    def handle_status(params, **kwargs):
        result = tools.status(ctx)
        result["mathematical_memory"] = tools.memory_status(ctx)
        return result

    def handle_chat(params, **kwargs):
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

    def _require_scope(scope):
        # In standalone/local mode there is no authority to consult, so avoid
        # importing the optional crypto stack at all. Paired/authority mode
        # still performs the full signed-token verification.
        if not (os.environ.get("NANNET_AUTH_URL") or
                os.environ.get("NANNET_ISSUER_PUBLIC_KEY")):
            return False
        return _require_scope( scope)

    def handle_auth_init(params, **kwargs): return _auth_tools().auth_init(ctx)
    def handle_auth_status(params, **kwargs): return _auth_tools().auth_status(ctx)
    def handle_auth_pair(params, **kwargs): return _auth_tools().auth_pair(ctx, params["pairing_code"])
    def handle_auth_revoke(params, **kwargs): return _auth_tools().auth_revoke(ctx)

    def handle_generate(params, **kwargs):
        _require_scope( "corpus:write")
        return tools.generate_corpus(ctx, params.get("topic", ""), params.get("records", 12))

    def handle_train(params, **kwargs):
        _require_scope( "training:execute")
        return tools.train(ctx, params["corpus_path"], params.get("duration_minutes"))

    def handle_math_train(params, **kwargs):
        _require_scope( "training:execute")
        return tools.math_train(ctx, params["corpus_path"], params.get("epochs", 3),
                                params.get("max_samples_per_head", 4096))

    def handle_evaluate(params, **kwargs):
        _require_scope( "universe:read")
        return tools.evaluate(ctx, params["corpus_path"])

    def handle_memory_ingest(params, **kwargs):
        _require_scope( "training:execute")
        return tools.memory_ingest(ctx, params["discoveries"], params["source_digest"],
                                   params.get("head"), params.get("layer", 0))

    def handle_memory_reorganize(params, **kwargs):
        _require_scope( "training:execute")
        return tools.memory_reorganize(ctx)

    def handle_memory_merge(params, **kwargs):
        _require_scope( "training:execute")
        return tools.memory_merge(ctx, params["records"])

    def handle_memory_query(params, **kwargs):
        _require_scope( "universe:read")
        return tools.memory_query(ctx, params.get("subject"), params.get("head"),
                                  params.get("layer"), params.get("limit", 100))

    def handle_memory_assessments(params, **kwargs):
        _require_scope( "universe:read")
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
        ("nanlm_generate_corpus", "Generate a bounded N16TXT/1 corpus using Hermes' active model.",
         {"topic": {"type": "string"}, "records": {"type": "integer", "minimum": 1, "maximum": 64, "default": 24}}, ["topic"], handle_generate),
        ("nanlm_train", "Run the native NaNLMind Context2 bootstrap trainer locally.",
         {"corpus_path": {"type": "string"}, "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 360}}, ["corpus_path"], handle_train),
        ("nanlm_math_train", "Train all 256 native heads, resume the serialized global topology, export structured discoveries, and merge them into additive memory.",
         {"corpus_path": {"type": "string"}, "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
          "max_samples_per_head": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 4096}},
         ["corpus_path"], handle_math_train),
        ("nanlm_evaluate", "Run native topology-byte, Kneser-Ney and Context2 controls locally on the same corpus.",
         {"corpus_path": {"type": "string"}}, ["corpus_path"], handle_evaluate),
        ("nanlm_memory_ingest", "Append structured mathematical discoveries to the durable additive universe memory.",
         {"discoveries": {"type": "array", "minItems": 1, "maxItems": 1000, "items": {"type": "object"}},
          "source_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
          "head": {"type": "integer", "minimum": 0, "maximum": 255},
          "layer": {"type": "integer", "minimum": 0, "maximum": 31}},
         ["discoveries", "source_digest"], handle_memory_ingest),
        ("nanlm_memory_reorganize", "Rebuild deterministic head/layer memory projections without deleting discoveries.", {}, [], handle_memory_reorganize),
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
    for name, description, properties, required, handler in specs:
        ctx.register_tool(
            name=name, toolset="nanlmind", handler=handler,
            schema={"name": name, "description": description,
                    "parameters": {"type": "object", "properties": properties,
                                   "required": required, "additionalProperties": False}},
        )
