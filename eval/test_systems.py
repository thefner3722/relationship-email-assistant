"""No-API tests for the three systems: identical fixed block across
systems, history depth as the only simple/layer0 difference, model as
the only simple/oss difference, equal output caps, context cap applied
to the exact final prompt, and the result shape."""
import pytest

import config
import providers
from systems import SYSTEMS, simple, layer0, oss
from systems.prompt import DRAFT_PROMPT, build_prompt, format_fixed, estimate_tokens

HIST = [{"from_name": "Priya", "date": f"2001-0{i+1}-01", "subject": f"s{i}",
         "body": f"message number {i}"} for i in range(8)]
CASE = {
    "owner": "Vince",
    "history": HIST,
    "incoming": {"from_name": "Priya", "date": "2001-09-01", "subject": "deck",
                 "body": "Checking on the Q3 deck?"},
    "context_profile": {
        "global_instructions": "Be concise.",
        "standing_instructions": "Always sign off as Vince.",
        "relationship_objective": "keep Priya's trust",
        "memory_facts": ["Q3 deck due Friday", "Priya is on the finance team"],
    },
}


def fake_call(calls):
    """Records every call_model invocation; returns a fixed reply."""
    def _call(provider, model, prompt, *, max_tokens, temperature):
        calls.append({"provider": provider, "model": model, "prompt": prompt,
                      "max_tokens": max_tokens, "temperature": temperature})
        return {"text": "Hi Priya, Friday as planned. Vince", "prompt_tokens": 100,
                "completion_tokens": 20, "latency_s": 0.5, "model": model, "backend": provider}
    return _call


def _patch_all(monkeypatch, calls):
    monkeypatch.setattr(config, "GEN_PROVIDER", "openai")
    monkeypatch.setattr(config, "GEN_MODEL", "gpt-test")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setattr(config, "MAX_DRAFT_TOKENS", 800)
    for mod in (simple, layer0, oss):
        monkeypatch.setattr(mod, "call_model", fake_call(calls))


def test_registry_has_three_systems():
    assert set(SYSTEMS) == {"simple", "oss", "layer0"}


def test_fixed_block_identical_across_systems(monkeypatch):
    calls = []; _patch_all(monkeypatch, calls)
    ps = simple.draft(CASE)["prompt"]
    pl = layer0.draft(CASE)["prompt"]
    fixed = format_fixed(CASE["context_profile"])
    assert fixed in ps and fixed in pl
    for s in ("Be concise.", "Always sign off as Vince", "keep Priya's trust", "- Q3 deck due Friday"):
        assert s in ps and s in pl


def test_simple_vs_layer0_differ_only_in_history_same_provider_model(monkeypatch):
    calls = []; _patch_all(monkeypatch, calls)
    ps = simple.draft(CASE)["prompt"]
    pl = layer0.draft(CASE)["prompt"]
    assert "message number 3" in ps and "message number 2" not in ps   # last 5
    assert all(f"message number {i}" in pl for i in range(8))         # all
    assert build_prompt(CASE, HIST[-5:]) == ps and build_prompt(CASE, HIST) == pl
    assert (calls[0]["provider"], calls[0]["model"]) == (calls[1]["provider"], calls[1]["model"]) == ("openai", "gpt-test")


def test_oss_uses_simple_prompt_and_reports_backend(monkeypatch):
    calls = []; _patch_all(monkeypatch, calls)
    ps = simple.draft(CASE)["prompt"]
    monkeypatch.setattr(oss, "ollama_up", lambda: True)
    out = oss.draft(CASE)
    assert out["prompt"] == ps
    assert (calls[-1]["provider"], calls[-1]["model"]) == ("ollama", "llama3.1:8b")
    assert out["backend"] == "ollama"
    monkeypatch.setattr(oss, "ollama_up", lambda: False)
    out = oss.draft(CASE)
    assert out["prompt"] == ps
    assert (calls[-1]["provider"], calls[-1]["model"]) == ("openrouter", "meta-llama/llama-3.1-8b-instruct")
    assert out["backend"] == "openrouter"


def test_every_backend_gets_same_draft_token_cap(monkeypatch):
    calls = []; _patch_all(monkeypatch, calls)
    monkeypatch.setattr(config, "MAX_DRAFT_TOKENS", 333)
    simple.draft(CASE); layer0.draft(CASE)
    monkeypatch.setattr(oss, "ollama_up", lambda: True); oss.draft(CASE)
    monkeypatch.setattr(oss, "ollama_up", lambda: False); oss.draft(CASE)
    assert [c["provider"] for c in calls] == ["openai", "openai", "ollama", "openrouter"]
    assert {c["max_tokens"] for c in calls} == {333}


def test_provider_adapters_pass_the_cap_through(monkeypatch):
    """The cap reaches each adapter's request: OpenAI/Anthropic/OpenRouter
    as max_tokens, Ollama as num_predict."""
    import json
    seen = {}

    class FakeResp:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
            self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1,
                                        "input_tokens": 1, "output_tokens": 1})()
            self.content = [type("T", (), {"text": text, "type": "text"})()]

    class FakeOpenAI:
        class chat:
            class completions:
                @staticmethod
                def create(**kw): seen["openai"] = kw; return FakeResp("x")

    class FakeAnthropic:
        class messages:
            @staticmethod
            def create(**kw): seen["anthropic"] = kw; return FakeResp("x")

    monkeypatch.setattr(config, "OPENAI_API_KEY", "k"); monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "k")
    monkeypatch.setitem(providers._clients, "openai", FakeOpenAI())
    monkeypatch.setitem(providers._clients, "anthropic", FakeAnthropic())

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data)
        key = "ollama" if "11434" in req.full_url else "openrouter"
        seen[key] = body
        payload = {"response": "x", "prompt_eval_count": 1, "eval_count": 1} if key == "ollama" else \
                  {"choices": [{"message": {"content": "x"}}], "usage": {}}
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return json.dumps(payload).encode()
        return R()
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(providers.json, "load", lambda r: json.loads(r.read()))

    for prov in ("openai", "anthropic", "openrouter", "ollama"):
        providers.call_model(prov, "m", "p", max_tokens=555, temperature=0)
    assert seen["openai"]["max_tokens"] == 555
    assert seen["anthropic"]["max_tokens"] == 555
    assert seen["openrouter"]["max_tokens"] == 555
    assert seen["ollama"]["options"]["num_predict"] == 555


def test_layer0_cap_applies_to_exact_final_prompt(monkeypatch):
    """The checked thing == the sent thing: estimate_tokens(prompt sent)
    never exceeds the cap, oldest messages go first, nothing but history
    is dropped."""
    calls = []; _patch_all(monkeypatch, calls)
    full = build_prompt(CASE, HIST)
    cap = estimate_tokens(full) - 10          # just below the full prompt
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", cap)
    out = layer0.draft(CASE)
    sent = calls[-1]["prompt"]
    assert sent == out["prompt"]
    assert estimate_tokens(sent) <= cap
    assert "message number 7" in sent and "message number 0" not in sent
    assert format_fixed(CASE["context_profile"]) in sent and "Checking on the Q3 deck?" in sent
    # cap that fits the history-free prompt but nothing more: all history dropped,
    # fixed block + incoming remain, still within cap
    base = estimate_tokens(build_prompt(CASE, []))
    p, kept = layer0.build_capped_prompt(CASE, cap=base)
    assert kept == [] and "(no prior messages)" in p and estimate_tokens(p) <= base
    # cap below the history-free prompt -> never returns an over-limit prompt
    with pytest.raises(RuntimeError, match="base prompt exceeds context cap"):
        layer0.build_capped_prompt(CASE, cap=1)
    # generous cap -> nothing dropped and the prompt is byte-identical to the uncapped build
    p, kept = layer0.build_capped_prompt(CASE, cap=10**9)
    assert kept == HIST and p == full


def test_prompt_template_and_result_shape(monkeypatch):
    assert "Do not invent facts" in DRAFT_PROMPT
    calls = []; _patch_all(monkeypatch, calls)
    out = layer0.draft(CASE)
    assert set(out) >= {"draft", "prompt_tokens", "completion_tokens",
                        "latency_s", "model", "backend", "prompt"}
    assert "prompt_facts" not in out
