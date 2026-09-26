"""Scoring code for the evaluation harness.

Functions in this file, added incrementally per the M2 build order:
  - build_judge_prompt()  : loads rubric.md verbatim into the judge prompt   [done]
  - hard_tests()          : four pass/fail checks                            [done]
  - extract_prompt_facts() / specificity() : two-stage fact-use ratio        [done]
  - judge() / cost() / latency() : reliability-looped judge, cost, latency  [done]
"""
import hashlib
import json
from pathlib import Path

import config
from providers import call_model


class EvaluationError(RuntimeError):
    """An evaluator (judge model) response was malformed or reliability
    could not be reached. Never silently coerced into a score."""

RUBRIC_PATH = Path(__file__).parent / "rubric.md"

JUDGE_PROMPT_TEMPLATE = """You are scoring a drafted email reply against the rubric below.
Read the rubric carefully — it is the exact standard to apply. Do not
substitute your own judgment for what "good" means; use only the
rubric's own definitions of each level.

RUBRIC:
{rubric}

RELATIONSHIP OBJECTIVE FOR THIS CONTACT (fixed for this case — do not
re-infer your own; use exactly this, or "N/A" if none was found):
{relationship_objective}

CASE CONTEXT:
{context}

INCOMING EMAIL:
{incoming}

DRAFTED REPLY BEING SCORED:
{draft}

Score each of the four dimensions in the rubric using the rubric's own
level descriptors to decide the score.
- "instructions_and_style" and "accounts_for_context" are always 1-5.
- "aligns_with_objectives": {objective_instruction}
- "addresses_commitments": {commitments_instruction}
Applicability has been decided for you above; do not decide it yourself.
Scores are numbers between 1.0 and 5.0 inclusive. Decimals are allowed (e.g.
3.5) but a score must never be below 1 or above 5, and must be a JSON number,
not a string.
Return JSON only:
{{"instructions_and_style": <1.0-5.0>, "aligns_with_objectives": <1.0-5.0 or "N/A">,
  "addresses_commitments": <1.0-5.0 or "N/A">, "accounts_for_context": <1.0-5.0>,
  "reasoning": {{"instructions_and_style": "...", "aligns_with_objectives": "...",
                 "addresses_commitments": "...", "accounts_for_context": "..."}}}}
"""


def load_rubric_text() -> str:
    """Read rubric.md verbatim. No paraphrasing, no summarizing — the
    judge prompt gate requires the exact file text."""
    return RUBRIC_PATH.read_text()


def build_judge_prompt(context: str, incoming: str, draft: str,
                        relationship_objective: str | None,
                        commitment_relevant: bool = True) -> str:
    """Build the full judge prompt for one case, with rubric.md's real
    text inserted verbatim.

    relationship_objective must be inferred ONCE, before evaluation,
    from history available before the target case (see the rubric's
    dimension 2 note) — and passed in fixed here, the same for every
    judge run on this case (config.JUDGE_VALID_RUNS). Never let the
    judge infer its own; that was the bug this rewrite fixes. Pass None
    if no clear objective
    could be inferred; it renders as "N/A" in the prompt.

    commitment_relevant is likewise decided ONCE from case metadata
    (case["commitment_relevant"], set at golden-set build time), not by
    the judge. Applicability of both dimensions is fixed per case so the
    valid judge runs cannot disagree about it.
    """
    rubric_text = load_rubric_text()
    if relationship_objective:
        obj_instr = "score 1-5 against the objective given above."
    else:
        obj_instr = 'no objective applies to this case; return the string "N/A".'
    if commitment_relevant:
        com_instr = "a commitment is relevant to this case; score 1-5."
    else:
        com_instr = 'no commitment is relevant to this case; return the string "N/A", never 3.'
    return JUDGE_PROMPT_TEMPLATE.format(
        rubric=rubric_text,
        relationship_objective=relationship_objective or "N/A",
        objective_instruction=obj_instr,
        commitments_instruction=com_instr,
        context=context,
        incoming=incoming,
        draft=draft,
    )


# --- Hard tests ---------------------------------------------------------
# Four checks, combined into ONE judge-model call per vote (JUDGE_TEMPERATURE,
# default 0, JSON out), voted config.HARD_TEST_VOTES times (default 3);
# unanimous agreement per test is required to pass.
# Hard tests are independent binary checks. They do not modify or cap the
# qualitative 1-5 rubric score built above; the two are reported separately.
# Specificity (below) is a separate companion metric, also independent.
#
# no_new_commitment rule: a draft fails ONLY if it creates a commitment that
# is neither already established in prior context nor directly requested in
# the incoming email. Agreeing to something the sender asked for is allowed;
# repeating a commitment already in history is allowed; inventing a
# call/deadline/favor that was neither committed nor requested fails.

HARD_TEST_NAMES = ["no_fabrication", "no_contradicted_commitment",
                   "no_stale_item", "no_new_commitment"]

# All four checks in ONE prompt (one call instead of four) -- they all look
# at the exact same case material, so they can share a single question.
# Voted config.HARD_TEST_VOTES times for the same unanimous-agreement
# reliability as before; only the call count per vote changed, 4 -> 1.
COMBINED_HARD_TEST_PROMPT = """Check the drafted reply below against four independent \
criteria. Judge each one separately and honestly -- a reply can pass some and fail \
others.

1. no_fabrication: List every factual claim in the draft (names, dates, numbers, \
events, commitments). Is each one supported by CONTEXT or INCOMING? Passes only if \
every claim is supported.

2. no_contradicted_commitment: Given the commitments already made in this \
relationship's history (COMMITMENTS_IN_HISTORY), does the draft contradict, deny, or \
ignore any of them, given that INCOMING touches on them? Passes if none is \
contradicted, denied, or ignored.

3. no_stale_item: Given items already resolved/closed in this relationship's history \
(RESOLVED_ITEMS), does the draft raise, ask about, or re-offer any of them as if \
still open? Passes if none is re-raised as open. If INCOMING itself asks about a \
resolved item, a reply that correctly reports it as resolved is NOT re-raising it \
and passes.

4. no_new_commitment: Does the draft offer, promise, or propose anything (a meeting, \
a deliverable, a deadline, a favor) that is NOT already in COMMITMENTS_IN_HISTORY and \
NOT directly requested in INCOMING? Passes if nothing new is promised.

COMMITMENTS_IN_HISTORY:
{commitments_in_history}

RESOLVED_ITEMS:
{resolved_items}

CONTEXT:
{context}

INCOMING EMAIL:
{incoming}

DRAFTED REPLY:
{draft}

Return JSON only, exactly these four keys, each a JSON boolean, plus one reasoning \
string per key:
{{"no_fabrication": true|false, "no_contradicted_commitment": true|false, \
"no_stale_item": true|false, "no_new_commitment": true|false, \
"reasoning": {{"no_fabrication": "...", "no_contradicted_commitment": "...", \
"no_stale_item": "...", "no_new_commitment": "..."}}}}
"""


def call_judge_model(prompt: str, cacheable: bool = False) -> dict:
    """One judge call on the configured JUDGE_PROVIDER/JUDGE_MODEL at JUDGE_TEMPERATURE (default 0),
    JSON-parsed. Provider and model come from config.py (eval/.env) --
    nothing here hardcodes either. Every hard test, the specificity
    stages and the rubric judge route through this one function.
    cacheable=True marks the prompt as reused (repeated verbatim across
    hard-test votes or judge attempts) so provider adapters that support
    prompt caching (currently anthropic) bill repeats at a discount; a
    prompt only ever sent once (specificity) should leave this False.
    Raises EvaluationError if the response is not a JSON object."""
    # 4096 headroom: adaptive-thinking models can spend part of max_tokens
    # on an internal thinking block before the JSON answer; a small cap
    # risks truncating the JSON (seen in practice with Opus 5.5 on longer
    # prompts) and json.loads then fails, which is correctly treated as a
    # malformed response by the caller -- but a cap this tight makes that
    # failure common rather than exceptional. Raised to config.JUDGE_MAX_TOKENS
    # (default 16000): OpenAI reasoning models bill hidden reasoning tokens
    # against this exact same ceiling, so it needs more headroom than
    # Anthropic thinking once JUDGE_REASONING_EFFORT is non-"none".
    r = call_model(config.JUDGE_PROVIDER, config.JUDGE_MODEL, prompt,
                   max_tokens=config.JUDGE_MAX_TOKENS, temperature=config.JUDGE_TEMPERATURE,
                   cacheable=cacheable, reasoning_effort=config.JUDGE_REASONING_EFFORT)
    text = r["text"]
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise EvaluationError(f"judge returned non-JSON: {text[:120]!r}") from e
    if not isinstance(data, dict):
        raise EvaluationError(f"judge returned non-object JSON: {text[:120]!r}")
    return data


def valid_combined_vote(vote):
    """Per test name: True/False if that key holds a real JSON boolean;
    None if malformed for that key specifically. Independent per key --
    a malformed value on one test does not invalidate the other three's
    votes; they remain independent checks even though they share one
    call. If `vote` itself is not a dict, every key is None."""
    if not isinstance(vote, dict):
        return {name: None for name in HARD_TEST_NAMES}
    result = {}
    for name in HARD_TEST_NAMES:
        v = vote.get(name)
        result[name] = v if (v is True or v is False) else None
    return result


def hard_tests(case: dict, draft: str, *, call_fn=call_judge_model,
               votes: int = None, max_attempts: int = None) -> dict:
    """Run all four hard tests in ONE combined call per attempt (see
    COMBINED_HARD_TEST_PROMPT); unanimous agreement across `votes` valid
    votes (config.HARD_TEST_VOTES, default 3) is required to pass each
    test. `case` must have context, incoming, commitments_in_history,
    resolved_items keys — missing keys default to empty so a test with
    nothing to check against simply has nothing to flag. Every attempt
    reuses the identical prompt, so it is sent with cacheable=True.

    Each of the four tests accumulates its own valid boolean votes
    independently: a malformed value on ONE test's key in a given
    response does not consume that response as a vote for the other
    three, which keep whatever valid value they got from it. Retries
    (on a non-JSON call_fn response, or to top up any single test still
    short of `votes` valid votes) continue until every test has `votes`
    valid votes or `max_attempts` (config.HARD_TEST_MAX_ATTEMPTS,
    default 5) total calls have been made.

    If the cap is hit and ANY test still lacks enough valid votes,
    EvaluationError is raised for the whole case. An evaluator that
    cannot reliably answer is a failed EVALUATION, not a failed DRAFT —
    it must surface as a harness-level error (the same "n error" bucket
    APIConnectionError and IncompleteCaseError already land in), never
    silently reported as the draft failing a hard test."""
    votes_n = config.HARD_TEST_VOTES if votes is None else votes
    max_attempts_n = config.HARD_TEST_MAX_ATTEMPTS if max_attempts is None else max_attempts
    prompt = COMBINED_HARD_TEST_PROMPT.format(
        context=case.get("context", ""),
        incoming=case.get("incoming", ""),
        draft=draft,
        commitments_in_history=case.get("commitments_in_history", "(none)"),
        resolved_items=case.get("resolved_items", "(none)"),
    )
    per_test_valid = {name: [] for name in HARD_TEST_NAMES}
    invalid_counts = {name: 0 for name in HARD_TEST_NAMES}
    raw_votes, attempts = [], 0

    def _short():
        return [n for n in HARD_TEST_NAMES if len(per_test_valid[n]) < votes_n]

    while attempts < max_attempts_n and _short():
        attempts += 1
        try:
            resp = call_fn(prompt, cacheable=True)
        except EvaluationError:
            continue  # whole response unparsable: retried, not a vote for anyone
        raw_votes.append(resp)
        parsed = valid_combined_vote(resp)
        for name in HARD_TEST_NAMES:
            if parsed[name] is None:
                invalid_counts[name] += 1
            elif len(per_test_valid[name]) < votes_n:
                per_test_valid[name].append(parsed[name])

    short = _short()
    if short:
        raise EvaluationError(
            f"hard tests {short}: fewer than {votes_n} valid votes after "
            f"{attempts} attempts (case {case.get('id', '?')})")

    return {
        name: {
            "test": name,
            "pass": all(v is True for v in per_test_valid[name]),
            "votes": raw_votes,
            "invalid_votes": invalid_counts[name],
        }
        for name in HARD_TEST_NAMES
    }


FACT_EXTRACT_PROMPT = """Below is the complete prompt that was given to an email-drafting \
model. List every distinct factual item it contains: names, dates, numbers, deadlines, \
commitments, decisions, stated preferences, relationships, and any other concrete \
claim. Do not include instructions to the model, and do not include anything that is \
not stated in the prompt. Return JSON only: {{"facts": ["...", "..."]}}.

DRAFTING PROMPT:
{prompt}
"""

FACT_USE_PROMPT = """Here is a fixed, numbered list of factual items that were present in \
the prompt given to an email-drafting model:
{facts}

For the drafted reply below, decide which of these items are correctly represented in \
the reply. An item counts only if the reply uses it accurately. Return JSON only: \
{{"represented": [<item numbers>]}}. Return an empty list if none are used.

DRAFTED REPLY:
{draft}
"""

# Stage-1 cache: identical drafting prompt -> identical fact list, no re-extraction.
_FACT_CACHE: dict = {}


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def extract_prompt_facts(prompt: str, *, call_fn=call_judge_model) -> list:
    """Stage 1 of specificity. Input: the entire actual prompt sent to the
    drafting model for one system x case. The generated reply is NOT
    shown. Output: the frozen list of factual items in that prompt.
    Cached by prompt text, so a prompt reused across runs is extracted
    once."""
    key = _prompt_key(prompt)
    if key not in _FACT_CACHE:
        result = call_fn(FACT_EXTRACT_PROMPT.format(prompt=prompt))
        if not isinstance(result, dict) or "facts" not in result:
            raise EvaluationError(f"specificity stage 1: no 'facts' object in {str(result)[:120]!r}")
        facts = result["facts"]
        if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
            raise EvaluationError(f"specificity stage 1: 'facts' is not a list of strings: {str(facts)[:120]!r}")
        _FACT_CACHE[key] = [f for f in facts if f.strip()]
    return list(_FACT_CACHE[key])


def specificity(prompt: str, draft: str, *, call_fn=call_judge_model):
    """Companion metric, computed separately for each system x case.

        specificity = source facts correctly represented in the reply
                      / total source facts in the actual drafting prompt

    Two independent stages:
      1. extract_prompt_facts(prompt)  -- from the drafting prompt alone,
         reply not shown, cached per identical prompt
      2. one call with (frozen fact list, reply) deciding which items the
         reply represents correctly

    The source of truth is the exact prompt the generation model received,
    so different systems (different history depth) get different
    denominators automatically; nothing per-system is hand-maintained.

    Returns "N/A" if the prompt genuinely has zero factual items; 0.0 if
    facts exist but the reply represents none; else represented / total.
    A malformed evaluator response at either stage raises EvaluationError
    -- it is never turned into N/A or 0.0. Independent of the hard tests
    and of the 1-5 rubric score; diagnostic, not a quality metric."""
    facts = extract_prompt_facts(prompt, call_fn=call_fn)
    if not facts:
        return "N/A"
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    result = call_fn(FACT_USE_PROMPT.format(facts=numbered, draft=draft))
    if not isinstance(result, dict) or "represented" not in result:
        raise EvaluationError(f"specificity stage 2: no 'represented' object in {str(result)[:120]!r}")
    rep = result["represented"]
    if not isinstance(rep, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in rep):
        raise EvaluationError(f"specificity stage 2: 'represented' is not a list of ints: {str(rep)[:120]!r}")
    valid = {x for x in rep if 1 <= x <= len(facts)}
    return len(valid) / len(facts)


# --- Judge score, cost, latency -----------------------------------------
# Judge: one rubric call per run, JUDGE_VALID_RUNS valid runs (config default 3), median per dimension. "N/A"
# dimensions are excluded from the overall mean, never scored as 3.
# Independent of the hard tests and of specificity.

class IncompleteCaseError(ValueError):
    """A case is missing metadata that must be decided before judging."""


def valid_score(v) -> bool:
    """A rubric score is valid iff it is a JSON number (int or float, not
    bool) and 1.0 <= v <= 5.0. Decimals are fine."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 1.0 <= v <= 5.0


JUDGE_DIMENSIONS = ["instructions_and_style", "aligns_with_objectives",
                    "addresses_commitments", "accounts_for_context"]


def valid_judge_response(resp, applicable: dict) -> bool:
    """A judge response is valid only if EVERY applicable dimension holds a
    valid numeric score in [1.0, 5.0] and EVERY non-applicable dimension
    holds the string "N/A" (applicability is precomputed from case
    metadata). Anything else -- missing key, string number, out of range,
    a number where N/A was required -- makes the whole response invalid."""
    if not isinstance(resp, dict):
        return False
    for dim, is_app in applicable.items():
        v = resp.get(dim)
        if is_app and not valid_score(v):
            return False
        if not is_app and v != "N/A":
            return False
    return True


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def judge(case: dict, draft: str, *, call_fn=call_judge_model,
          valid_runs: int = None, max_attempts: int = None) -> dict:
    """Rubric judge with configurable reliability.

    Keeps calling the judge until `valid_runs` (config.JUDGE_VALID_RUNS)
    valid, complete responses are collected -- see valid_judge_response --
    stopping as soon as it has enough. Invalid responses -- parseable JSON
    that fails score validation, OR malformed/non-JSON output that makes
    call_judge_model raise EvaluationError -- are discarded, do not count,
    and each consumes one attempt. If `max_attempts` (config.JUDGE_MAX_ATTEMPTS) calls are
    made without reaching `valid_runs`, EvaluationError is raised.

    Final dimension score = median across the valid runs; overall = mean
    of the applicable dimension medians; N/A dimensions excluded.

    Applicability is decided ONCE from case metadata, never by the judge:
      aligns_with_objectives is N/A iff case["relationship_objective"] is empty
      addresses_commitments  is N/A iff case["commitment_relevant"] is False
    case["commitment_relevant"] is REQUIRED; a missing key raises
    IncompleteCaseError, never a silent default."""
    valid_runs = config.JUDGE_VALID_RUNS if valid_runs is None else valid_runs
    max_attempts = config.JUDGE_MAX_ATTEMPTS if max_attempts is None else max_attempts
    if "commitment_relevant" not in case:
        raise IncompleteCaseError(
            f"case {case.get('id', '?')} has no commitment_relevant flag; "
            "applicability must be set at golden-set build time")
    objective = case.get("relationship_objective") or None
    commitment_relevant = bool(case["commitment_relevant"])
    applicable = {
        "instructions_and_style": True,
        "aligns_with_objectives": objective is not None,
        "addresses_commitments": commitment_relevant,
        "accounts_for_context": True,
    }
    prompt = build_judge_prompt(
        context=case.get("context", ""),
        incoming=case.get("incoming", ""),
        draft=draft,
        relationship_objective=objective,
        commitment_relevant=commitment_relevant,
    )
    valid, invalid, attempts = [], [], 0
    while len(valid) < valid_runs:
        if attempts >= max_attempts:
            raise EvaluationError(
                f"judge: only {len(valid)}/{valid_runs} valid responses after "
                f"{attempts} attempts (case {case.get('id', '?')})")
        attempts += 1
        try:
            resp = call_fn(prompt, cacheable=True)
        except EvaluationError as e:          # non-JSON / non-object output
            invalid.append({"error": str(e)})   # consumes an attempt, retried
            continue
        (valid if valid_judge_response(resp, applicable) else invalid).append(resp)

    scores = {}
    for dim in JUDGE_DIMENSIONS:
        if not applicable[dim]:
            scores[dim] = "N/A"
        else:
            scores[dim] = _median([float(r[dim]) for r in valid])
    vals = [scores[d] for d in JUDGE_DIMENSIONS if scores[d] != "N/A"]
    scores["overall"] = sum(vals) / len(vals) if vals else "N/A"
    scores["runs"] = valid
    scores["attempts"] = attempts
    scores["invalid_attempts"] = len(invalid)
    return scores


def cost(system_result: dict) -> float:
    """USD for one draft, from the token counts the system reported:
    prompt_tokens * input_price + completion_tokens * output_price.
    Ollama backend costs 0; any other backend's model needs a price in .env."""
    inp, out = config.price_for(system_result["model"], system_result.get("backend", "api"))
    return (system_result["prompt_tokens"] * inp
            + system_result["completion_tokens"] * out) / 1_000_000


def latency(system_result: dict) -> float:
    """Seconds for one draft, as measured and reported by the system."""
    return float(system_result["latency_s"])


def aggregate(rows: list[dict]) -> dict:
    """Average every numeric key over rows; keys holding bools become
    pass rates; "N/A" values are skipped. Rows are the flat per-(system,
    case) dicts the harness writes."""
    out = {"n": len(rows)}
    keys = {k for r in rows for k in r}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] != "N/A"]
        if not vals:
            out[k] = "N/A"
        elif all(isinstance(v, bool) for v in vals):
            out[k] = sum(vals) / len(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = sum(vals) / len(vals)
    return out


if __name__ == "__main__":
    # smoke test: confirm the rubric's real text actually lands in the prompt,
    # and the N/A handling for objectives/commitments is wired in
    sample = build_judge_prompt(
        context="[sample history would go here]",
        incoming="[sample incoming email would go here]",
        draft="[sample draft would go here]",
        relationship_objective="build trust ahead of a Q3 budget approval",
    )
    assert "Instructions and style" in sample, "rubric text did not load into prompt"
    assert "Aligns with objectives" in sample, "rubric text did not load into prompt"
    assert "Addresses commitments" in sample, "rubric text did not load into prompt"
    assert "Accounts for context" in sample, "rubric text did not load into prompt"
    assert "N/A" in sample, "N/A handling instructions missing from prompt"
    assert "build trust ahead of a Q3 budget approval" in sample, "objective was not inserted"

    no_objective = build_judge_prompt(
        context="[sample history would go here]",
        incoming="[sample incoming email would go here]",
        draft="[sample draft would go here]",
        relationship_objective=None,
    )
    assert "FOR THIS CONTACT (fixed for this case" in no_objective and "\nN/A\n" in no_objective, \
        "None objective did not render as N/A"

    print("OK: judge prompt loads rubric.md verbatim, N/A handling wired in")
    print(f"prompt length: {len(sample)} chars")
