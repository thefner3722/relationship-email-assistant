"""Registry. Every system is draft(case) -> {draft, prompt_tokens,
completion_tokens, latency_s, model, backend, prompt}. simple and layer0
call the same GEN_PROVIDER/GEN_MODEL; oss calls the open-source path.
All three use the same MAX_DRAFT_TOKENS reply cap. `prompt` is the
exact text sent to the generation model; specificity is derived from it
(metrics.extract_prompt_facts), never from a per-system fact list."""
from . import simple, oss, layer0

SYSTEMS = {
    "simple": simple.draft,
    "oss": oss.draft,
    "layer0": layer0.draft,
}
