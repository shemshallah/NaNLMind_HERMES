"""NaNLMind Hermes plugin entry point."""
from . import tools, auth_tools


def register(ctx):
    def handle_status(params, **kwargs):
        result = tools.status(ctx)
        result["mathematical_memory"] = tools.memory_status(ctx)
        return result

    def handle_auth_init(params, **kwargs): return auth_tools.auth_init(ctx)
    def handle_auth_status(params, **kwargs): return auth_tools.auth_status(ctx)
    def handle_auth_pair(params, **kwargs): return auth_tools.auth_pair(ctx, params["pairing_code"])
    def handle_auth_revoke(params, **kwargs): return auth_tools.auth_revoke(ctx)

    def handle_generate(params, **kwargs):
        auth_tools.require_scope(ctx, "corpus:write")
        return tools.generate_corpus(ctx, params.get("topic", ""), params.get("records", 12))

    def handle_train(params, **kwargs):
        auth_tools.require_scope(ctx, "training:execute")
        return tools.train(ctx, params["corpus_path"], params.get("duration_minutes"))

    def handle_math_train(params, **kwargs):
        auth_tools.require_scope(ctx, "training:execute")
        return tools.math_train(ctx, params["corpus_path"], params.get("epochs", 3),
                                params.get("max_samples_per_head", 4096))

    def handle_evaluate(params, **kwargs):
        auth_tools.require_scope(ctx, "universe:read")
        return tools.evaluate(ctx, params["corpus_path"])

    def handle_memory_ingest(params, **kwargs):
        auth_tools.require_scope(ctx, "training:execute")
        return tools.memory_ingest(ctx, params["discoveries"], params["source_digest"],
                                   params.get("head"), params.get("layer", 0))

    def handle_memory_reorganize(params, **kwargs):
        auth_tools.require_scope(ctx, "training:execute")
        return tools.memory_reorganize(ctx)

    def handle_memory_merge(params, **kwargs):
        auth_tools.require_scope(ctx, "training:execute")
        return tools.memory_merge(ctx, params["records"])

    def handle_memory_query(params, **kwargs):
        auth_tools.require_scope(ctx, "universe:read")
        return tools.memory_query(ctx, params.get("subject"), params.get("head"),
                                  params.get("layer"), params.get("limit", 100))

    def handle_memory_assessments(params, **kwargs):
        auth_tools.require_scope(ctx, "universe:read")
        return tools.memory_assessments(ctx, params.get("subject"), params.get("limit", 100))

    specs = [
        ("nanlm_status", "Inspect local NaNLMind universe and durable mathematical-memory status.", {}, [], handle_status),
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
