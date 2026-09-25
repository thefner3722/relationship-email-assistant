"""One provider-agnostic model call with adapters underneath.

    call_model(provider, model, prompt, max_tokens, temperature)
      -> {"text", "prompt_tokens", "completion_tokens", "latency_s",
          "model", "backend"}

`backend` is the provider that actually answered, so cost is always
computed against the right price. Providers: openai, anthropic,
openrouter, ollama. Adding one = one adapter function + an entry in
ADAPTERS; nothing in systems/, metrics.py or harness.py changes.
"""
import json
import time
import urllib.request

import config

_clients = {}


def _openai(model, prompt, max_tokens, temperature):
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in eval/.env")
    if "openai" not in _clients:
        import openai
        _clients["openai"] = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    resp = _clients["openai"].chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content,
            resp.usage.prompt_tokens, resp.usage.completion_tokens)


def _anthropic(model, prompt, max_tokens, temperature):
    # The installed anthropic SDK (1.8.0, current-generation Messages API)
    # does not accept a `temperature` argument -- newer Claude models use
    # adaptive thinking instead of sampling temperature. temperature is
    # accepted here for interface symmetry with the other adapters but is
    # silently unused on this backend; GEN_TEMPERATURE/JUDGE_TEMPERATURE
    # have no effect when the provider is "anthropic".
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in eval/.env")
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = _clients["anthropic"].messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    # Adaptive-thinking models (5.x generation) can return a ThinkingBlock
    # ahead of the TextBlock; content[0] is not reliably the answer. Join
    # every "text" block and skip anything else (thinking, tool_use, ...).
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _openrouter(model, prompt, max_tokens, temperature):
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in eval/.env")
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {config.OPENROUTER_API_KEY}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    u = data.get("usage", {})
    return (data["choices"][0]["message"]["content"],
            u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def _ollama(model, prompt, max_tokens, temperature):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(config.OLLAMA_URL + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    return data["response"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)


ADAPTERS = {"openai": _openai, "anthropic": _anthropic,
            "openrouter": _openrouter, "ollama": _ollama}


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


def call_model(provider: str, model: str, prompt: str, *, max_tokens: int,
               temperature: float) -> dict:
    if provider not in ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}; one of {sorted(ADAPTERS)}")
    if not model:
        raise RuntimeError(f"no model id configured for provider {provider!r} (see eval/.env)")
    t0 = time.perf_counter()
    text, p_tok, c_tok = ADAPTERS[provider](model, prompt, max_tokens, temperature)
    return {
        "text": (text or "").strip(),
        "prompt_tokens": int(p_tok or 0),
        "completion_tokens": int(c_tok or 0),
        "latency_s": time.perf_counter() - t0,
        "model": model,
        "backend": provider,
    }
