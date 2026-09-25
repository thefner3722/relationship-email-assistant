"""Baseline 2: the exact same prompt as simple.py, open-source model.
Differs from simple ONLY in model/backend. Uses OLLAMA_MODEL when a local
Ollama is reachable, else OPENROUTER_MODEL. Same MAX_DRAFT_TOKENS cap as
every other backend, so output length is not a confound."""
import config
from providers import call_model, ollama_up
from .prompt import build_prompt
from .simple import history_for, _finish


def draft(case: dict) -> dict:
    prompt = build_prompt(case, history_for(case))
    if ollama_up():
        r = call_model("ollama", config.OLLAMA_MODEL, prompt,
                       max_tokens=config.MAX_DRAFT_TOKENS, temperature=config.GEN_TEMPERATURE)
    else:
        r = call_model("openrouter", config.OPENROUTER_MODEL, prompt,
                       max_tokens=config.MAX_DRAFT_TOKENS, temperature=config.GEN_TEMPERATURE)
    return _finish(prompt, r)
