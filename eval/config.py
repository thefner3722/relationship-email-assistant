"""Single entry point for keys, providers, model ids, limits and prices.

Every file that needs to call a model imports from HERE — nobody else
reads os.environ or hardcodes a provider or model. To change which
provider, key, model or limit is used, edit .env in this folder.

Provider/model pairs:
  GEN_PROVIDER / GEN_MODEL       simple.py and layer0.py drafting (same for both:
                                 that is what makes simple vs layer0 a controlled
                                 comparison of history depth)
  JUDGE_PROVIDER / JUDGE_MODEL   hard tests, specificity, rubric judge
  OLLAMA_MODEL                   oss.py when local Ollama is reachable ($0)
  OPENROUTER_MODEL               oss.py fallback via OpenRouter
Adding another provider = one adapter in providers.py + a key here; the
evaluation pipeline does not change.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

PROVIDERS = ("openai", "anthropic", "openrouter", "ollama")

GEN_PROVIDER = os.environ.get("GEN_PROVIDER", "openai")
GEN_MODEL = os.environ.get("GEN_MODEL")
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "openai")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _int(var, default):
    raw = os.environ.get(var, "").strip()
    return int(raw) if raw else default


# --- Limits ---------------------------------------------------------------
MAX_DRAFT_TOKENS = _int("MAX_DRAFT_TOKENS", 800)      # reply cap, identical on every backend
MAX_CONTEXT_TOKENS = _int("MAX_CONTEXT_TOKENS", 60000)  # layer0: cap on the exact final prompt
JUDGE_VALID_RUNS = _int("JUDGE_VALID_RUNS", 5)        # valid complete judge responses needed
JUDGE_MAX_ATTEMPTS = _int("JUDGE_MAX_ATTEMPTS", 15)   # give up (error) after this many calls
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0") or 0)
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0") or 0)


# --- Prices: USD per 1M tokens (input, output) ---------------------------
# Read from .env so pricing lives with the model ids it belongs to:
#   GEN_MODEL_PRICE=2.50,10.00
#   JUDGE_MODEL_PRICE=2.50,10.00
#   OPENROUTER_MODEL_PRICE=0.05,0.08
# Ollama models are always $0 and need no price line.

def _parse_price(var):
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    inp, out = raw.split(",")
    return float(inp), float(out)


PRICES = {}
for _model, _var in ((GEN_MODEL, "GEN_MODEL_PRICE"),
                     (JUDGE_MODEL, "JUDGE_MODEL_PRICE"),
                     (OPENROUTER_MODEL, "OPENROUTER_MODEL_PRICE")):
    _p = _parse_price(_var)
    if _model and _p is not None:
        PRICES[_model] = _p


def price_for(model: str, backend: str) -> tuple:
    """(input, output) USD per 1M tokens for the model/backend actually
    used. backend="ollama" is local and always $0. Any other backend's
    model id must have a *_PRICE line in .env; unknown ids raise rather
    than silently costing $0."""
    if backend == "ollama":
        return (0.0, 0.0)
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r} ({backend}); set its *_PRICE line in eval/.env")
    return PRICES[model]
