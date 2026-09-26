"""Baseline 1: fixed case block + last 5 history messages, on the primary
generation provider/model (GEN_PROVIDER/GEN_MODEL — the same pair layer0
uses). Receives the SAME fixed instructions/style/objective/memory as
layer0; differs from layer0 only in history depth (see prompt.py)."""
import config
from providers import call_model
from .prompt import build_prompt

N_LAST = 5


def history_for(case: dict) -> list:
    return case.get("history", [])[-N_LAST:]


def _finish(prompt: str, r: dict) -> dict:
    return {"draft": r["text"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "latency_s": r["latency_s"],
            "model": r["model"], "backend": r["backend"], "prompt": prompt}


def draft(case: dict) -> dict:
    prompt = build_prompt(case, history_for(case))
    r = call_model(config.GEN_PROVIDER, config.GEN_MODEL, prompt,
                   max_tokens=config.MAX_DRAFT_TOKENS, temperature=config.GEN_TEMPERATURE,
                   reasoning_effort=config.GEN_REASONING_EFFORT)
    return _finish(prompt, r)
