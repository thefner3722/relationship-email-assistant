"""Baseline 0 (ablation): identical history depth to simple (last 5
messages, via simple.history_for -- imported, not reimplemented, so it
cannot silently drift), but the entire fixed instructions block is
removed: no global/standing instructions, no style guide, no relationship
objective, no memory facts, no "don't invent facts / honor commitments"
guidance. Isolates how much the fixed instructions matter, holding
history depth constant at simple's level. Uses the same GEN_PROVIDER/
GEN_MODEL as simple and layer0, so model choice is not a variable here
either -- only the presence or absence of instructions/context is."""
import config
from providers import call_model
from .simple import history_for
from .prompt import format_history, format_incoming

BARE_PROMPT = """Draft a reply to the incoming email below on behalf of {owner}.

EMAIL HISTORY WITH THIS CONTACT:
{history}

INCOMING EMAIL:
{incoming}
"""


def build_bare_prompt(case: dict) -> str:
    return BARE_PROMPT.format(
        owner=case["owner"],
        history=format_history(history_for(case)),
        incoming=format_incoming(case),
    )


def _finish(prompt: str, r: dict) -> dict:
    return {"draft": r["text"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "latency_s": r["latency_s"],
            "model": r["model"], "backend": r["backend"], "prompt": prompt}


def draft(case: dict) -> dict:
    prompt = build_bare_prompt(case)
    r = call_model(config.GEN_PROVIDER, config.GEN_MODEL, prompt,
                   max_tokens=config.MAX_DRAFT_TOKENS, temperature=config.GEN_TEMPERATURE,
                   reasoning_effort=config.GEN_REASONING_EFFORT)
    return _finish(prompt, r)
