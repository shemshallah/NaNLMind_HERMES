"""NaNLMind Hermes plugin entry point."""
from . import tools


def register(ctx):
    """Register NaNLMind tools with Hermes."""
    def handle_status(params, **kwargs):
        return tools.status(ctx)

    def handle_generate(params, **kwargs):
        return tools.generate_corpus(
            ctx, params.get("topic", ""), params.get("records", 12)
        )

    def handle_train(params, **kwargs):
        return tools.train(
            ctx, params["corpus_path"], params.get("duration_minutes")
        )

    def handle_evaluate(params, **kwargs):
        return tools.evaluate(ctx, params["corpus_path"])

    ctx.register_tool(
        name="nanlm_status",
        toolset="nanlmind",
        schema={
            "name": "nanlm_status",
            "description": "Inspect this installation's local NaNLMind universe and training configuration.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        handler=handle_status,
    )
    ctx.register_tool(
        name="nanlm_generate_corpus",
        toolset="nanlmind",
        schema={
            "name": "nanlm_generate_corpus",
            "description": "Generate a bounded N16TXT/1 training corpus using Hermes' active model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic or skill area."},
                    "records": {"type": "integer", "minimum": 1, "maximum": 64, "default": 12},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
        handler=handle_generate,
    )
    ctx.register_tool(
        name="nanlm_train",
        toolset="nanlmind",
        schema={
            "name": "nanlm_train",
            "description": "Train this local universe for a bounded duration, at most six hours. A run may override the configured duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "corpus_path": {"type": "string", "description": "Path to an N16TXT/1 corpus file."},
                    "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 360},
                },
                "required": ["corpus_path"],
                "additionalProperties": False,
            },
        },
        handler=handle_train,
    )
    ctx.register_tool(
        name="nanlm_evaluate",
        toolset="nanlmind",
        schema={
            "name": "nanlm_evaluate",
            "description": "Run the configured local runner's evaluation mode against a corpus.",
            "parameters": {
                "type": "object",
                "properties": {"corpus_path": {"type": "string"}},
                "required": ["corpus_path"],
                "additionalProperties": False,
            },
        },
        handler=handle_evaluate,
    )
