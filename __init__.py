"""NaNLMind Hermes plugin entry point."""
from . import tools, auth_tools

def register(ctx):
    def handle_status(params, **kwargs): return tools.status(ctx)
    def handle_auth_init(params, **kwargs): return auth_tools.auth_init(ctx)
    def handle_auth_status(params, **kwargs): return auth_tools.auth_status(ctx)
    def handle_auth_pair(params, **kwargs): return auth_tools.auth_pair(ctx,params["pairing_code"])
    def handle_auth_revoke(params, **kwargs): return auth_tools.auth_revoke(ctx)
    def handle_generate(params, **kwargs):
        auth_tools.require_scope(ctx,"corpus:write")
        return tools.generate_corpus(ctx,params.get("topic",""),params.get("records",12))
    def handle_train(params, **kwargs):
        auth_tools.require_scope(ctx,"training:execute")
        return tools.train(ctx,params["corpus_path"],params.get("duration_minutes"))
    def handle_evaluate(params, **kwargs):
        auth_tools.require_scope(ctx,"universe:read")
        return tools.evaluate(ctx,params["corpus_path"])

    specs=[
      ("nanlm_status","Inspect local NaNLMind universe status.",{},[],handle_status),
      ("nanlm_auth_init","Create this installation's encrypted NANNET identity.",{},[],handle_auth_init),
      ("nanlm_auth_status","Inspect local NANNET identity and pairing status without revealing secrets.",{},[],handle_auth_status),
      ("nanlm_auth_pair","Pair this installation using a one-time code from the trusted NANNET authority.",{"pairing_code":{"type":"string","minLength":12,"maxLength":256}},["pairing_code"],handle_auth_pair),
      ("nanlm_auth_revoke","Revoke this installation token and erase its local encrypted copy.",{},[],handle_auth_revoke),
      ("nanlm_generate_corpus","Generate a bounded N16TXT/1 corpus using Hermes' active model.",{"topic":{"type":"string"},"records":{"type":"integer","minimum":1,"maximum":64,"default":24}},["topic"],handle_generate),
      ("nanlm_train","Run the native NaNLMind bootstrap trainer locally with a strict wall-clock limit; artifacts are reproducibility logs, not resumable weights.",{"corpus_path":{"type":"string"},"duration_minutes":{"type":"integer","minimum":1,"maximum":360}},["corpus_path"],handle_train),
      ("nanlm_evaluate","Run native topology-byte, Kneser-Ney and Context2 controls locally on the same corpus.",{"corpus_path":{"type":"string"}},["corpus_path"],handle_evaluate),
    ]
    for name,description,properties,required,handler in specs:
      ctx.register_tool(name=name,toolset="nanlmind",handler=handler,schema={"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":False}})
