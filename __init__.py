"""NaNLMind Hermes plugin entry point."""
from . import tools


def register(ctx):
    """Register NaNLMind tools with Hermes."""
    def handle_status(params, **kwargs):
        return tools.status(ctx)

    def handle_auth_init(params, **kwargs):
        return tools.auth_init(ctx)

    def handle_auth_status(params, **kwargs):
        return tools.auth_status(ctx)

    def handle_generate(params, **kwargs):
        return tools.generate_corpus(ctx, params.get("topic", ""), params.get("records", 12))

    def handle_train(params, **kwargs):
        return tools.train(ctx, params["corpus_path"], params.get("duration_minutes"))

    def handle_evaluate(params, **kwargs):
        return tools.evaluate(ctx, params["corpus_path"])

    specs = [
        ("nanlm_status", "Inspect this installation's local NaNLMind universe and training configuration.", {}, [], handle_status),
        ("nanlm_auth_status", "Check local NANNET identity setup without exposing credentials.", {}, [], handle_auth_status),
        ("nanlm_auth_init", "Create a local NANNET identity using the master key already provisioned in the host environment.", {}, [], handle_auth_init),
        ("nanlm_generate_corpus", "Generate a bounded N16TXT/1 corpus using Hermes' active model.", {
            "topic": {"type": "string"}, "records": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12}
        }, ["topic"], handle_generate),
        ("nanlm_train", "Train this local universe for a bounded duration of at most six hours.", {
            "corpus_path": {"type": "string"}, "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 360}
        }, ["corpus_path"], handle_train),
        ("nanlm_evaluate", "Evaluate a corpus using the configured local runner.", {
            "corpus_path": {"type": "string"}
        }, ["corpus_path"], handle_evaluate),
    ]
    for name, description, properties, required, handler in specs:
        ctx.register_tool(name=name, toolset="nanlmind", schema={
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
        }, handler=handler)
