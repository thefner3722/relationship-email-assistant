"""End-to-end harness test with no API: fake system + fake judge.
Checks the row shape, error isolation, per-system specificity, and the
files the harness writes."""
import csv
import json

import harness
from test_metrics import make_mock_call_fn

HIST = [{"from_name": "Priya", "from": "priya@x.com", "to": ["vince@x.com"], "date": f"2001-0{i+1}-01",
         "subject": f"s{i}", "body": f"message number {i}"} for i in range(6)]
CASES = [
    {"id": "c001", "contact": "priya@x.com", "owner": "Vince", "bucket": "routine",
     "history": HIST, "incoming": dict(HIST[0], body="Checking on the Q3 deck?"),
     "reference_reply": {"body": "Friday as planned."},
     "commitments_in_history": ["Send Q3 deck by Friday"], "resolved_items": [],
     "commitment_relevant": True,
     "context_profile": {"relationship_objective": "keep Priya's trust",
                         "memory_facts": ["Q3 deck due Friday", "Priya is on finance"]}},
    {"id": "c002", "contact": "priya@x.com", "owner": "Vince", "bucket": "routine",
     "history": HIST, "incoming": dict(HIST[1], body="Thanks!"),
     "reference_reply": {"body": "Sure."},
     "commitments_in_history": [], "resolved_items": [], "commitment_relevant": False,
     "context_profile": {}},
    {"id": "c003", "contact": "priya@x.com", "owner": "Vince", "bucket": "routine",
     "history": HIST, "incoming": dict(HIST[2]), "reference_reply": {"body": "ok"},
     "commitments_in_history": [], "resolved_items": [], "commitment_relevant": False,
     "context_profile": {}},
    {"id": "c004", "contact": "priya@x.com", "owner": "Vince", "bucket": "routine",
     "history": HIST, "incoming": dict(HIST[3]), "reference_reply": {"body": "ok"},
     "commitments_in_history": [], "resolved_items": [], "context_profile": {}},  # no flag -> error row
]

def JUDGE(prompt):
    """Valid response shaped by the applicability the prompt announces."""
    return {"instructions_and_style": 4,
            "aligns_with_objectives": "N/A" if "no objective applies" in prompt else 4,
            "addresses_commitments": "N/A" if "no commitment is relevant" in prompt else 4,
            "accounts_for_context": 3}
MOCK = make_mock_call_fn({
    "List every factual claim": {"claims": [], "pass": True},
    "contradict, deny, or ignore": {"pass": True, "reason": "ok"},
    "raise, ask about, or re-offer": {"pass": True, "reason": "ok"},
    "offer, promise, or propose": {"pass": True, "new_commitments": []},
})


def _mock_with_spec(prompt):
    """Hard tests + judge from MOCK; specificity stage 1 extracts facts
    from the drafting prompt text (2 if it mentions MEMORY, else 0),
    stage 2 says item 1 is represented."""
    if "List every distinct factual item" in prompt:
        return {"facts": ["Q3 deck due Friday", "Priya is on finance"]} if "MEMORY" in prompt else {"facts": []}
    if "correctly represented" in prompt:
        return {"represented": [1]}
    if "You are scoring a drafted email reply" in prompt:
        return JUDGE(prompt)
    return MOCK(prompt)


def fake_layer0(case):
    if case["id"] == "c003":
        raise RuntimeError("boom")
    facts = (case.get("context_profile") or {}).get("memory_facts", [])
    prompt = ("MEMORY:\n" + "\n".join(facts)) if facts else "no facts here"
    return {"draft": "Hi Priya, Friday as planned.", "prompt_tokens": 500, "completion_tokens": 30,
            "latency_s": 1.2, "model": "llama3.1:8b", "backend": "ollama", "prompt": prompt}


def fake_simple(case):
    out = fake_layer0(case)
    return dict(out, prompt="short prompt, no facts")


def test_harness_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    ts = "TEST"
    rows0 = harness.run_system("layer0", CASES, ts, draft_fn=fake_layer0, call_fn=_mock_with_spec,
                               judge_runs=3, judge_reference=True, quiet=True)
    rows1 = harness.run_system("simple", CASES, ts, draft_fn=fake_simple, call_fn=_mock_with_spec,
                               judge_runs=3, judge_reference=False, quiet=True)

    # error isolation: c003 (system error) and c004 (missing commitment_relevant)
    # are error rows, but the run completes with 4 rows
    assert len(rows0) == 4 and rows0[2]["error"].startswith("RuntimeError")
    assert rows0[3]["error"].startswith("IncompleteCaseError")
    assert not rows0[0]["error"]

    # specificity from the actual prompt: layer0's prompt held 2 facts, 1 used -> 0.5;
    # simple's prompt held none -> N/A
    assert rows0[0]["specificity"] == 0.5 and rows0[0]["n_prompt_facts"] == 2
    assert rows1[0]["specificity"] == "N/A" and rows1[0]["n_prompt_facts"] == 0
    assert rows1[0]["ref_judge_overall"] == "N/A"

    # applicability from case: c001 has objective + commitment, c002 has neither
    assert rows0[0]["judge_aligns_with_objectives"] == 4
    assert rows0[1]["judge_aligns_with_objectives"] == "N/A"
    assert rows0[1]["judge_addresses_commitments"] == "N/A"
    assert rows0[1]["judge_overall"] == 3.5
    assert rows0[0]["ref_judge_overall"] == 3.75
    assert rows0[0]["cost_usd"] == 0.0 and rows0[0]["latency_s"] == 1.2

    # files
    assert (tmp_path / "TEST_layer0.csv").exists()
    drafts = [json.loads(l) for l in (tmp_path / "TEST_layer0_drafts.jsonl").open()]
    assert len(drafts) == 2 and drafts[0]["draft"].startswith("Hi Priya")

    s0 = harness.summarize("layer0", rows0, ts)
    s1 = harness.summarize("simple", rows1, ts)
    assert s0["n"] == 2 and s0["n_error"] == 2
    assert s0["no_fabrication"] == 1.0 and s0["specificity"] == 0.5
    assert s0["p50_latency_s"] == 1.2
    harness.append_summary([s0, s1])
    summ = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [r["system"] for r in summ] == ["layer0", "simple"]
    harness.print_table([s0, s1])


def test_summary_csv_fixed_schema_across_runs(tmp_path, monkeypatch):
    """Two appends with different optional fields present -> one header,
    every row has exactly SUMMARY_COLUMNS, optional metric is N/A when absent."""
    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    rows_ref = harness.run_system("layer0", CASES[:2], "R1", draft_fn=fake_layer0,
                                  call_fn=_mock_with_spec, judge_runs=3, judge_reference=True, quiet=True)
    rows_noref = harness.run_system("simple", CASES[:2], "R2", draft_fn=fake_simple,
                                    call_fn=_mock_with_spec, judge_runs=3, judge_reference=False, quiet=True)
    harness.append_summary([harness.summarize("layer0", rows_ref, "R1")])
    harness.append_summary([harness.summarize("simple", rows_noref, "R2")])
    with (tmp_path / "summary.csv").open(newline="") as f:
        raw = list(csv.reader(f))
    assert raw[0] == harness.SUMMARY_COLUMNS
    assert len(raw) == 3 and all(len(r) == len(harness.SUMMARY_COLUMNS) for r in raw)
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert rows[0]["ref_judge_overall"] not in ("", "N/A")
    assert rows[1]["ref_judge_overall"] == "N/A"
    # a file with a foreign header is refused, not corrupted
    (tmp_path / "summary.csv").write_text("a,b\n1,2\n")
    import pytest
    with pytest.raises(RuntimeError):
        harness.append_summary([harness.summarize("simple", rows_noref, "R3")])


def test_load_cases_filters_by_split(tmp_path, monkeypatch):
    g = tmp_path / "golden.jsonl"; s = tmp_path / "splits.json"
    g.write_text("".join(json.dumps(c) + "\n" for c in CASES))
    s.write_text(json.dumps({"dev": ["priya@x.com"], "test": []}))
    monkeypatch.setattr(harness, "GOLDEN", g); monkeypatch.setattr(harness, "SPLITS", s)
    assert len(harness.load_cases("dev", None)) == 4
    assert len(harness.load_cases("dev", 2)) == 2
    assert harness.load_cases("test", None) == []
