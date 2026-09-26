"""Layer 0: fixed case block + ALL history messages, on the same primary
GEN_PROVIDER/GEN_MODEL as simple.py. The cap (config.MAX_CONTEXT_TOKENS)
is applied to the exact final prompt that is sent: the full prompt is
built, and the oldest history message is dropped and the prompt rebuilt
until estimate_tokens(prompt) <= cap. What is checked is what is sent.
If the prompt is still over the cap with no history left, it raises --
an over-limit prompt is never returned; nothing but history is ever
trimmed."""
import config
from providers import call_model
from .prompt import build_prompt, estimate_tokens
from .simple import _finish


def build_capped_prompt(case: dict, cap: int = None) -> tuple:
    """Returns (prompt, history_kept). Drops oldest history first until the
    complete prompt fits the cap. The fixed block, incoming email and
    wrapper text are never trimmed. Invariant on normal return:
    estimate_tokens(prompt) <= cap. If the history-free prompt still
    exceeds the cap, RuntimeError."""
    cap = config.MAX_CONTEXT_TOKENS if cap is None else cap
    msgs = list(case.get("history", []))
    prompt = build_prompt(case, msgs)
    while msgs and estimate_tokens(prompt) > cap:
        msgs = msgs[1:]
        prompt = build_prompt(case, msgs)
    tokens = estimate_tokens(prompt)
    if tokens > cap:
        raise RuntimeError(f"Layer 0 base prompt exceeds context cap: {tokens} > {cap}")
    return prompt, msgs


def draft(case: dict) -> dict:
    prompt, _ = build_capped_prompt(case)
    r = call_model(config.GEN_PROVIDER, config.GEN_MODEL, prompt,
                   max_tokens=config.MAX_DRAFT_TOKENS, temperature=config.GEN_TEMPERATURE,
                   reasoning_effort=config.GEN_REASONING_EFFORT)
    return _finish(prompt, r)
