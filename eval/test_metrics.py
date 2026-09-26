"""Fixture tests for the four hard tests + specificity, with no live API
call — a mock call_fn stands in for call_judge_model.

Hard tests are independent binary checks, all four asked in one combined
call and voted config.HARD_TEST_VOTES times (unanimous agreement per test
required to pass); they do not modify or cap the 1-5 rubric score. Specificity is a
separate companion metric computed per system x case from the facts in
that system's actual drafting prompt.
"""
import pytest
import metrics
import config
from metrics import (hard_tests, specificity, extract_prompt_facts, judge, cost,
                     latency, aggregate, IncompleteCaseError, EvaluationError,
                     COMBINED_HARD_TEST_PROMPT, valid_combined_vote)

CASE = {
    "context": "Owner previously told Priya he'd send the Q3 deck by Friday. "
                "The Q2 budget question was resolved last month.",
    "incoming": "Hey, just checking on that Q3 deck?",
    "commitments_in_history": "1. Send Q3 deck by Friday (owner -> Priya)",
    "resolved_items": "1. Q2 budget question (resolved last month)",
    "commitment_relevant": True,   # REQUIRED case metadata; judge() raises without it
}

def make_mock_call_fn(responses):
    """responses: dict mapping a keyword found in the prompt -> the JSON
    dict to return. Lets each fixture case steer the combined hard-test
    prompt (and the specificity prompt) without a real model. cacheable
    is accepted and ignored -- these mocks don't touch a real API, so
    caching has nothing to do here."""
    def call_fn(prompt, cacheable=False):
        for keyword, response in responses.items():
            if keyword in prompt:
                return response
        raise AssertionError(f"mock got an unexpected prompt: {prompt[:80]}...")
    return call_fn


def make_combined_mock(no_fabrication=True, no_contradicted_commitment=True,
                       no_stale_item=True, no_new_commitment=True):
    """A hard-test mock that returns one combined {name: bool} response for
    every vote -- the new shape COMBINED_HARD_TEST_PROMPT expects back."""
    response = {
        "no_fabrication": no_fabrication,
        "no_contradicted_commitment": no_contradicted_commitment,
        "no_stale_item": no_stale_item,
        "no_new_commitment": no_new_commitment,
    }
    def call_fn(prompt, cacheable=False):
        return dict(response)
    return call_fn


def test_clean_draft_passes_everything():
    draft = "Hi Priya, the Q3 deck is on track for Friday as planned. Talk soon."
    mock = make_combined_mock()
    results = hard_tests(CASE, draft, call_fn=mock)
    assert set(results.keys()) == {
        "no_fabrication", "no_contradicted_commitment",
        "no_stale_item", "no_new_commitment",
    }
    assert all(r["pass"] for r in results.values()), results


def test_fabricating_draft_fails_no_fabrication():
    draft = "Hi Priya, the deck is done and I also looped in Legal on this."
    mock = make_combined_mock(no_fabrication=False)
    results = hard_tests(CASE, draft, call_fn=mock)
    assert results["no_fabrication"]["pass"] is False
    # the other three are independent checks and can still pass on this draft
    assert results["no_contradicted_commitment"]["pass"] is True
    assert results["no_stale_item"]["pass"] is True
    assert results["no_new_commitment"]["pass"] is True


# A drafting prompt as one system actually sent it. Specificity is scored
# against facts EXTRACTED from this text (stage 1), never against a list
# stored on CASE or declared by the system.
DRAFT_PROMPT_TEXT = ("You are drafting a reply for Vince.\nMEMORY:\n- Q3 deck due Friday\n"
                     "- Priya is on the finance team\nEMAIL HISTORY:\nPriya: Q2 budget resolved "
                     "last month.\nINCOMING: checking on the deck?")
THREE_FACTS = ["Q3 deck due Friday", "Priya is on the finance team", "Q2 budget resolved last month"]


def _spec_mock(facts, represented, log):
    """Stage-1 returns `facts` (from the prompt only); stage-2 returns
    `represented` (item numbers). Every call is logged."""
    def mock(prompt):
        log.append(prompt)
        if "List every distinct factual item" in prompt:
            assert "DRAFTED REPLY" not in prompt, "reply must not be shown to fact extraction"
            return {"facts": facts}
        if "correctly represented" in prompt:
            return {"represented": represented}
        raise AssertionError("unexpected prompt")
    return mock


@pytest.fixture(autouse=True)
def _clear_fact_cache():
    metrics._FACT_CACHE.clear()
    yield
    metrics._FACT_CACHE.clear()


def test_specificity_two_stage_fractional():
    """3 facts extracted from the prompt, reply represents 2 -> 2/3.
    Stage 1 sees the prompt (no reply); stage 2 sees the fixed list + reply."""
    log = []
    draft = "Hi Priya, deck's coming Friday; Q2 budget is closed."
    score = specificity(DRAFT_PROMPT_TEXT, draft, call_fn=_spec_mock(THREE_FACTS, [1, 3], log))
    assert len(log) == 2
    assert DRAFT_PROMPT_TEXT in log[0] and draft not in log[0]
    assert draft in log[1] and "1. Q3 deck due Friday" in log[1]
    assert abs(score - 2 / 3) < 1e-9


def test_specificity_zero_when_facts_present_but_none_used():
    log = []
    score = specificity(DRAFT_PROMPT_TEXT, "", call_fn=_spec_mock(THREE_FACTS, [], log))
    assert len(log) == 2, "stage-2 counting must run, not be skipped"
    assert score == 0.0


def test_specificity_na_when_prompt_has_zero_facts():
    log = []
    score = specificity("Reply politely.", "ok", call_fn=_spec_mock([], [], log))
    assert score == "N/A"
    assert len(log) == 1, "no stage-2 call when there is nothing to count"


def test_specificity_out_of_range_and_duplicate_indices_ignored():
    """Out-of-range and duplicate item numbers never inflate the numerator."""
    log = []
    score = specificity(DRAFT_PROMPT_TEXT, "x",
                        call_fn=_spec_mock(THREE_FACTS, [1, 1, 7, 0], log))
    assert abs(score - 1 / 3) < 1e-9


@pytest.mark.parametrize("bad", [None, "facts", [], {}, {"items": []}, {"facts": "a, b"},
                                 {"facts": [1, 2]}, {"facts": None}])
def test_specificity_malformed_stage1_raises(bad):
    """Malformed stage-1 output is an EvaluationError, never zero facts / N/A."""
    def mock(prompt):
        if "List every distinct factual item" in prompt:
            return bad
        raise AssertionError("stage 2 must not run after a bad stage 1")
    with pytest.raises(EvaluationError):
        specificity(DRAFT_PROMPT_TEXT, "d", call_fn=mock)
    assert not metrics._FACT_CACHE, "a bad extraction must not be cached"


@pytest.mark.parametrize("bad", [None, "1,2", [], {}, {"used": 2}, {"represented": "1"},
                                 {"represented": [True]}, {"represented": ["1"]},
                                 {"represented": None}])
def test_specificity_malformed_stage2_raises(bad):
    """Malformed stage-2 output is an EvaluationError, never 0.0."""
    def mock(prompt):
        if "List every distinct factual item" in prompt:
            return {"facts": THREE_FACTS}
        return bad
    with pytest.raises(EvaluationError):
        specificity(DRAFT_PROMPT_TEXT, "d", call_fn=mock)


def test_specificity_semantics_zero_vs_na_vs_error():
    log = []
    assert specificity("Reply politely.", "d", call_fn=_spec_mock([], [], log)) == "N/A"
    assert specificity(DRAFT_PROMPT_TEXT, "d", call_fn=_spec_mock(THREE_FACTS, [], log)) == 0.0


def test_fact_extraction_cached_per_identical_prompt():
    """Same drafting prompt twice -> one extraction call; a different prompt
    (e.g. another system with more history) -> its own extraction."""
    log = []
    mock = _spec_mock(THREE_FACTS, [1], log)
    assert extract_prompt_facts(DRAFT_PROMPT_TEXT, call_fn=mock) == THREE_FACTS
    assert extract_prompt_facts(DRAFT_PROMPT_TEXT, call_fn=mock) == THREE_FACTS
    assert len(log) == 1
    specificity(DRAFT_PROMPT_TEXT, "d", call_fn=mock)
    assert len(log) == 2                      # only the stage-2 call was added
    extract_prompt_facts(DRAFT_PROMPT_TEXT + " more history", call_fn=mock)
    assert len(log) == 3


def test_specificity_denominator_follows_the_prompt():
    """Two systems, same case: the one whose prompt held more facts gets
    the larger denominator, with nothing declared per system."""
    def mock(prompt):
        if "List every distinct factual item" in prompt:
            return {"facts": THREE_FACTS if "HISTORY" in prompt else THREE_FACTS[:1]}
        return {"represented": [1]}
    short = "MEMORY:\n- Q3 deck due Friday"
    assert specificity(short, "d", call_fn=mock) == 1.0
    assert abs(specificity(DRAFT_PROMPT_TEXT, "d", call_fn=mock) - 1 / 3) < 1e-9


def test_unanimous_pass_requires_all_votes():
    """Unanimous agreement across config.HARD_TEST_VOTES votes is required
    to pass; a single dissenting vote fails the test. Uses an explicit
    votes= of 3 so the test doesn't depend on whatever HARD_TEST_VOTES
    happens to default to."""
    draft = "Hi Priya, deck's coming Friday."
    calls = {"n": 0}

    def flaky_call_fn(prompt, cacheable=False):
        calls["n"] += 1
        # first two votes pass, third flips no_fabrication to fail
        return {
            "no_fabrication": calls["n"] != 3,
            "no_contradicted_commitment": True,
            "no_stale_item": True,
            "no_new_commitment": True,
        }

    results = hard_tests(CASE, draft, call_fn=flaky_call_fn, votes=3)
    assert results["no_fabrication"]["pass"] is False, (
        "one dissenting vote must fail the test"
    )
    assert len(results["no_fabrication"]["votes"]) == 3


def test_hard_test_votes_come_from_config(monkeypatch):
    """The number of votes per case is config.HARD_TEST_VOTES, not a
    hardcoded constant -- changing it changes real call count."""
    monkeypatch.setattr(config, "HARD_TEST_VOTES", 4)
    calls = {"n": 0}

    def mock(prompt, cacheable=False):
        calls["n"] += 1
        return {"no_fabrication": True, "no_contradicted_commitment": True,
                "no_stale_item": True, "no_new_commitment": True}

    hard_tests(CASE, "d", call_fn=mock)
    assert calls["n"] == 4


def test_hard_test_independence_one_bad_key_does_not_fail_others():
    """A malformed value on ONE of the four combined keys invalidates
    only that key's vote -- it must not drag the other three tests down
    with it. Before this fix, one bad field failed all four. The one
    flawed test tops up with a 4th call; the other three, already at 3
    valid votes after 3 calls, don't need it."""
    bad_vote = {"no_fabrication": True, "no_contradicted_commitment": "yes",  # malformed
               "no_stale_item": True, "no_new_commitment": True}
    good_vote = {"no_fabrication": True, "no_contradicted_commitment": True,
                "no_stale_item": True, "no_new_commitment": True}
    votes = iter([bad_vote, good_vote, good_vote, good_vote])
    r = hard_tests(CASE, "d", call_fn=lambda p, cacheable=False: next(votes), votes=3)
    assert r["no_fabrication"]["pass"] is True, "unaffected key must still pass"
    assert r["no_stale_item"]["pass"] is True, "unaffected key must still pass"
    assert r["no_new_commitment"]["pass"] is True, "unaffected key must still pass"
    assert r["no_contradicted_commitment"]["pass"] is True, "topped up to 3 valid votes, all True"
    assert r["no_contradicted_commitment"]["invalid_votes"] == 1, "the one malformed response is still counted"
    assert r["no_fabrication"]["invalid_votes"] == 0, "malformed sibling key must not count here"


def test_hard_test_retries_malformed_non_json_calls():
    """A call_fn raising EvaluationError (non-JSON judge output) is
    retried, not left to crash the whole case -- same reliability
    contract judge() already has. 1 error + 3 good votes succeeds within
    the attempt cap."""
    good = {"no_fabrication": True, "no_contradicted_commitment": True,
            "no_stale_item": True, "no_new_commitment": True}
    calls = {"n": 0}

    def mock(p, cacheable=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EvaluationError("judge returned non-JSON")
        return good

    r = hard_tests(CASE, "d", call_fn=mock, votes=3, max_attempts=5)
    assert calls["n"] == 4, "1 error + 3 valid votes = 4 calls"
    assert r["no_fabrication"]["pass"] is True
    assert r["no_fabrication"]["invalid_votes"] == 0

    # errors alone never reach `votes` valid responses for any test: this is
    # a failed EVALUATION, not a failed draft -- raise, never return False
    calls["n"] = 0
    def always_bad(p, cacheable=False):
        calls["n"] += 1
        raise EvaluationError("nope")
    with pytest.raises(EvaluationError):
        hard_tests(CASE, "d", call_fn=always_bad, votes=3, max_attempts=5)
    assert calls["n"] == 5


def test_new_commitment_rule_allows_requested_and_established():
    """no_new_commitment fails ONLY on commitments that are neither already
    in prior context nor directly requested in the incoming email. The
    combined prompt must state both exemptions, and the pass/fail result
    passes straight through."""
    p = COMBINED_HARD_TEST_PROMPT
    assert "NOT already in COMMITMENTS_IN_HISTORY" in p
    assert "NOT directly requested in INCOMING" in p

    requested_case = dict(CASE, incoming="Can you send this tomorrow?")
    draft = "Yes, I'll send it tomorrow."
    mock = make_combined_mock()
    assert hard_tests(requested_case, draft, call_fn=mock)["no_new_commitment"]["pass"] is True

    invented = "Sure -- and I'll also set up a call with Legal on Thursday."
    mock_fail = make_combined_mock(no_new_commitment=False)
    assert hard_tests(requested_case, invented, call_fn=mock_fail)["no_new_commitment"]["pass"] is False


def test_no_stale_item_judge_sees_incoming_email():
    """no_stale_item must be shown the incoming email, so a reply that
    correctly reports a resolved item because the sender asked about it
    is not mistaken for re-raising it."""
    p = COMBINED_HARD_TEST_PROMPT
    assert "INCOMING EMAIL:" in p and "{incoming}" in p

    case = dict(CASE, incoming="What happened with the Q2 budget question?")
    draft = "That was resolved last month."
    seen = {}

    def mock(prompt, cacheable=False):
        seen["prompt"] = prompt
        return {"no_fabrication": True, "no_contradicted_commitment": True,
                "no_stale_item": True, "no_new_commitment": True}

    results = hard_tests(case, draft, call_fn=mock)
    assert case["incoming"] in seen["prompt"]
    assert results["no_stale_item"]["pass"] is True


def test_hard_test_vote_must_be_json_boolean():
    """A string "false"/"true", a missing key, a non-bool value, or a
    non-dict vote makes THAT KEY's vote invalid, per key -- never
    coerced by truthiness, and never contaminating the other three keys
    (see test_hard_test_independence_one_bad_key_does_not_fail_others
    for the multi-key case). One malformed vote out of votes+1 supplied
    still succeeds (the flawed test tops up); malformed on every supplied
    vote for one key, with none left to top up with, raises instead of
    silently failing that test."""
    good = {"no_fabrication": True, "no_contradicted_commitment": True,
            "no_stale_item": True, "no_new_commitment": True}
    assert valid_combined_vote(good) == good
    assert valid_combined_vote(dict(good, no_fabrication="true")) == dict(good, no_fabrication=None)
    assert valid_combined_vote(dict(good, no_fabrication="false")) == dict(good, no_fabrication=None)
    assert valid_combined_vote(dict(good, no_fabrication=1)) == dict(good, no_fabrication=None)
    missing = {k: v for k, v in good.items() if k != "no_fabrication"}
    assert valid_combined_vote(missing) == dict(good, no_fabrication=None)
    # a non-dict vote can't be trusted for any key
    assert valid_combined_vote("yes") == {n: None for n in good}
    assert valid_combined_vote(None) == {n: None for n in good}

    for bad in (dict(good, no_fabrication="true"), dict(good, no_fabrication=1)):
        votes = iter([good, bad, good, good])   # 4th call tops up no_fabrication
        r = hard_tests(CASE, "d", call_fn=lambda p, cacheable=False: next(votes), votes=3)
        assert r["no_fabrication"]["pass"] is True, f"malformed vote {bad!r} should be topped up, not fail closed"
        assert r["no_fabrication"]["invalid_votes"] == 1

        votes2 = iter([good, bad, good])        # no 4th call available: can't reach 3 valid
        with pytest.raises(EvaluationError):
            hard_tests(CASE, "d", call_fn=lambda p, cacheable=False: next(votes2), votes=3, max_attempts=3)

    votes = iter([good] * 3)
    r = hard_tests(CASE, "d", call_fn=lambda p, cacheable=False: next(votes), votes=3)
    assert r["no_fabrication"]["pass"] is True


def test_judge_scores_validated_decimals_ok_out_of_range_invalid():
    """Decimals within [1,5] are valid; strings, bools, <1, >5 make the
    whole response invalid and it is retried, not counted."""
    from metrics import valid_score, valid_judge_response, build_judge_prompt
    assert valid_score(3.5) and valid_score(1) and valid_score(5.0)
    assert not valid_score(0.9) and not valid_score(5.1) and not valid_score("4")
    assert not valid_score(True) and not valid_score(None)
    p = build_judge_prompt("c", "i", "d", relationship_objective=None)
    assert "Decimals are allowed" in p and "never be below 1 or above 5" in p

    app = {"instructions_and_style": True, "aligns_with_objectives": False,
           "addresses_commitments": True, "accounts_for_context": True}
    good = {"instructions_and_style": 4.5, "aligns_with_objectives": "N/A",
            "addresses_commitments": 4, "accounts_for_context": 3}
    assert valid_judge_response(good, app)
    assert not valid_judge_response(dict(good, instructions_and_style="5"), app)
    assert not valid_judge_response(dict(good, instructions_and_style=0), app)
    assert not valid_judge_response(dict(good, instructions_and_style=6), app)
    assert not valid_judge_response(dict(good, aligns_with_objectives=3), app)   # number where N/A required
    assert not valid_judge_response({k: v for k, v in good.items() if k != "accounts_for_context"}, app)
    assert not valid_judge_response("nope", app)

    base = {"aligns_with_objectives": "N/A", "addresses_commitments": 4, "accounts_for_context": 3}
    runs = iter([dict(base, instructions_and_style=4.5), dict(base, instructions_and_style="5"),
                 dict(base, instructions_and_style=0), dict(base, instructions_and_style=6),
                 dict(base, instructions_and_style=3.5), dict(base, instructions_and_style=4.0)])
    r = judge(dict(CASE, relationship_objective=None), "d", call_fn=lambda p, cacheable=False: next(runs),
              valid_runs=3, max_attempts=15)
    assert r["instructions_and_style"] == 4.0      # median of 4.5, 3.5, 4.0
    assert r["attempts"] == 6 and r["invalid_attempts"] == 3


def test_judge_reliability_loop_retries_stops_and_fails():
    """Invalid attempts are retried; loop stops exactly when valid_runs are
    collected; EvaluationError after max_attempts."""
    good = {"instructions_and_style": 4, "aligns_with_objectives": 3,
            "addresses_commitments": 4, "accounts_for_context": 4}
    bad = {"instructions_and_style": "four"}
    case = dict(CASE, relationship_objective="keep trust")

    # 2 bad then 5 good, 3 more good available: stops at 7 calls, never uses the extras
    seq = [bad, bad] + [good] * 8
    calls = {"n": 0}
    def mock(p, cacheable=False):
        r = seq[calls["n"]]; calls["n"] += 1; return r
    r = judge(case, "d", call_fn=mock, valid_runs=5, max_attempts=15)
    assert calls["n"] == 7 and r["attempts"] == 7 and r["invalid_attempts"] == 2
    assert len(r["runs"]) == 5 and r["overall"] == 3.75

    # all good: exactly valid_runs calls
    calls["n"] = 0; seq[:] = [good] * 10
    r = judge(case, "d", call_fn=mock, valid_runs=5, max_attempts=15)
    assert calls["n"] == 5

    # never enough valid: error after max_attempts
    calls["n"] = 0; seq[:] = [bad] * 20
    with pytest.raises(EvaluationError):
        judge(case, "d", call_fn=mock, valid_runs=5, max_attempts=15)
    assert calls["n"] == 15

    # 4 valid then all bad -> still an error, valid count is not enough
    calls["n"] = 0; seq[:] = [good] * 4 + [bad] * 20
    with pytest.raises(EvaluationError):
        judge(case, "d", call_fn=mock, valid_runs=5, max_attempts=15)
    assert calls["n"] == 15


def test_judge_retries_malformed_non_json_calls():
    """A judge call that raises EvaluationError (non-JSON output) is an
    invalid attempt: retried, consumes an attempt, never fails the case
    on its own. 2 errors + 5 valid -> success after exactly 7 calls."""
    good = {"instructions_and_style": 4, "aligns_with_objectives": 3,
            "addresses_commitments": 4, "accounts_for_context": 4}
    case = dict(CASE, relationship_objective="keep trust")
    calls = {"n": 0}
    def mock(p, cacheable=False):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise EvaluationError("judge returned non-JSON")
        return good
    r = judge(case, "d", call_fn=mock, valid_runs=5, max_attempts=15)
    assert calls["n"] == 7 and r["attempts"] == 7 and r["invalid_attempts"] == 2
    assert len(r["runs"]) == 5 and r["overall"] == 3.75

    # errors alone never succeed: cap reached -> EvaluationError after 15 calls
    calls["n"] = 0
    def always_bad(p, cacheable=False):
        calls["n"] += 1
        raise EvaluationError("nope")
    with pytest.raises(EvaluationError):
        judge(case, "d", call_fn=always_bad, valid_runs=5, max_attempts=15)
    assert calls["n"] == 15


def test_judge_defaults_come_from_config(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_VALID_RUNS", 2)
    monkeypatch.setattr(config, "JUDGE_MAX_ATTEMPTS", 3)
    good = {"instructions_and_style": 4, "aligns_with_objectives": "N/A",
            "addresses_commitments": 4, "accounts_for_context": 4}
    calls = {"n": 0}
    def mock(p, cacheable=False): calls["n"] += 1; return good
    judge(dict(CASE, relationship_objective=None), "d", call_fn=mock)
    assert calls["n"] == 2
    calls["n"] = 0
    with pytest.raises(EvaluationError):
        judge(dict(CASE, relationship_objective=None), "d", call_fn=lambda p, cacheable=False: {})
    

def test_judge_requires_commitment_relevant():
    case = {k: v for k, v in CASE.items() if k != "commitment_relevant"}
    with pytest.raises(IncompleteCaseError):
        judge(case, "d", call_fn=lambda p, cacheable=False: {})
    ok = dict(CASE, commitment_relevant=False, relationship_objective=None)
    r = judge(ok, "d", call_fn=lambda p, cacheable=False: {"instructions_and_style": 4, "aligns_with_objectives": "N/A",
                                          "addresses_commitments": "N/A", "accounts_for_context": 4},
              valid_runs=2, max_attempts=4)
    assert r["addresses_commitments"] == "N/A" and r["overall"] == 4.0


def test_judge_median_and_na_handling():
    """5 runs -> median per dimension; a dimension that is N/A in every
    run stays N/A and is excluded from overall (not treated as 3)."""
    case = dict(CASE, relationship_objective=None)
    runs = iter([
        {"instructions_and_style": 5, "aligns_with_objectives": "N/A", "addresses_commitments": 4, "accounts_for_context": 3},
        {"instructions_and_style": 4, "aligns_with_objectives": "N/A", "addresses_commitments": 5, "accounts_for_context": 3},
        {"instructions_and_style": 5, "aligns_with_objectives": "N/A", "addresses_commitments": 4, "accounts_for_context": 4},
        {"instructions_and_style": 3, "aligns_with_objectives": "N/A", "addresses_commitments": 4, "accounts_for_context": 3},
        {"instructions_and_style": 5, "aligns_with_objectives": "N/A", "addresses_commitments": 2, "accounts_for_context": 5},
    ])
    r = judge(case, "Hi Priya, deck's coming Friday.", call_fn=lambda p, cacheable=False: next(runs),
              valid_runs=5, max_attempts=15)
    assert r["instructions_and_style"] == 5
    assert r["aligns_with_objectives"] == "N/A"
    assert r["addresses_commitments"] == 4
    assert r["accounts_for_context"] == 3
    assert r["overall"] == (5 + 4 + 3) / 3
    assert len(r["runs"]) == 5


def test_judge_applicability_comes_from_case_not_runs():
    """Applicability is fixed per case. If the case says the objective
    applies, a run returning N/A for it is invalid and retried; if the case
    says no commitment is relevant, a run must return N/A for it -- a
    number there is invalid too."""
    from metrics import build_judge_prompt
    case = dict(CASE, relationship_objective="keep Priya's trust", commitment_relevant=False)
    runs = iter([
        {"instructions_and_style": 4, "aligns_with_objectives": "N/A", "addresses_commitments": "N/A", "accounts_for_context": 4},  # invalid: objective applies
        {"instructions_and_style": 4, "aligns_with_objectives": 3, "addresses_commitments": 5, "accounts_for_context": 4},      # invalid: commitments N/A required
        {"instructions_and_style": 4, "aligns_with_objectives": 3, "addresses_commitments": "N/A", "accounts_for_context": 4},
        {"instructions_and_style": 4, "aligns_with_objectives": 5, "addresses_commitments": "N/A", "accounts_for_context": 4},
        {"instructions_and_style": 4, "aligns_with_objectives": 4, "addresses_commitments": "N/A", "accounts_for_context": 4},
    ])
    r = judge(case, "Hi Priya.", call_fn=lambda p, cacheable=False: next(runs), valid_runs=3, max_attempts=15)
    assert r["aligns_with_objectives"] == 4
    assert r["attempts"] == 5 and r["invalid_attempts"] == 2
    assert r["addresses_commitments"] == "N/A"
    assert r["overall"] == (4 + 4 + 4) / 3

    # and the prompt tells the judge the decision instead of asking it
    p = build_judge_prompt("c", "i", "d", relationship_objective=None, commitment_relevant=False)
    assert "no objective applies to this case" in p
    assert "no commitment is relevant to this case" in p
    assert "Applicability has been decided for you" in p


def test_judge_reasoning_effort_reaches_the_call(monkeypatch):
    """config.JUDGE_REASONING_EFFORT reaches call_model via
    call_judge_model -- this is what actually makes 'think harder when
    judging' real rather than just a config value nobody reads."""
    from metrics import call_judge_model
    seen = {}

    def fake_call_model(provider, model, prompt, *, max_tokens, temperature,
                        cacheable=False, reasoning_effort=None):
        seen["reasoning_effort"] = reasoning_effort
        return {"text": '{"pass": true}', "prompt_tokens": 1, "completion_tokens": 1,
                "latency_s": 0.0, "model": model, "backend": provider}

    monkeypatch.setattr(metrics, "call_model", fake_call_model)
    monkeypatch.setattr(config, "JUDGE_REASONING_EFFORT", "high")
    call_judge_model("any prompt")
    assert seen["reasoning_effort"] == "high"


def test_cost_and_latency_from_system_result():
    import config
    config.PRICES["test-model"] = (3.00, 15.00)  # prices come from .env via config
    api = {"draft": "x", "prompt_tokens": 1000, "completion_tokens": 200,
           "latency_s": 1.7, "model": "test-model", "backend": "anthropic"}
    assert abs(cost(api) - (1000 * 3.00 + 200 * 15.00) / 1_000_000) < 1e-12
    assert latency(api) == 1.7
    assert cost(dict(api, model="llama3.1:8b", backend="ollama")) == 0.0
    with pytest.raises(KeyError):        # openrouter model with no price line
        cost(dict(api, model="meta-llama/llama-3.1-8b-instruct", backend="openrouter"))


def test_aggregate_pass_rates_means_and_na():
    rows = [
        {"no_fabrication": True, "specificity": 0.5, "cost_usd": 0.01, "judge_overall": 4.0},
        {"no_fabrication": False, "specificity": "N/A", "cost_usd": 0.03, "judge_overall": "N/A"},
    ]
    a = aggregate(rows)
    assert a["n"] == 2
    assert a["no_fabrication"] == 0.5
    assert a["specificity"] == 0.5
    assert abs(a["cost_usd"] - 0.02) < 1e-12
    assert a["judge_overall"] == 4.0
