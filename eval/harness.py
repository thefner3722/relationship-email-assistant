"""Evaluation harness.

    python eval/harness.py --system simple --split dev [--limit N] [--judge-runs 3] [--judge-max-attempts 5]
    python eval/harness.py --system all --split dev --limit 5

For every (system, case): draft, then every metric, one flat row.
Errors on one case are recorded as {id, system, error} and the run
continues. Writes:
  eval/results/{ts}_{system}.csv          one row per case
  eval/results/{ts}_{system}_drafts.jsonl every draft + prompt, for error analysis
  eval/results/summary.csv                one aggregate row per (ts, system), appended
"""
import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics                       # noqa: E402
from systems import SYSTEMS          # noqa: E402
from systems.prompt import format_fixed, format_history  # noqa: E402

GOLDEN = HERE / "data" / "golden_set.jsonl"
SPLITS = HERE / "data" / "splits.json"
RESULTS = HERE / "results"

HARD = ["no_fabrication", "no_contradicted_commitment", "no_stale_item", "no_new_commitment"]
DIMS = metrics.JUDGE_DIMENSIONS

# Fixed schema for results/summary.csv. Every appended run writes exactly
# these columns in this order; a metric absent from a run (e.g. the
# optional reference-reply judge) is written as N/A.
SUMMARY_COLUMNS = [
    "ts", "system", "n", "n_error",
    *HARD, "all_hard_pass",
    "specificity", "mean_prompt_facts",
    "judge_overall", *[f"judge_{d}" for d in DIMS],
    "ref_judge_overall",
    "mean_cost_usd", "mean_prompt_tokens", "p50_latency_s",
]


def load_cases(split: str, limit: int | None) -> list[dict]:
    splits = json.load(SPLITS.open())
    keep = set(splits[split])
    cases = [json.loads(l) for l in GOLDEN.open() if l.strip()]
    cases = [c for c in cases if c["contact"] in keep]
    return cases[:limit] if limit else cases


def metric_case(case: dict) -> dict:
    """The case view the metrics need: flat strings for context,
    commitments, resolved items; applicability flags. The judge and hard
    tests see the same fixed case block every system saw plus the FULL
    history -- the evaluation truth is identical for every system.

    commitment_relevant is required case metadata (decided at golden-set
    build time); a case without it raises IncompleteCaseError."""
    if "commitment_relevant" not in case:
        raise metrics.IncompleteCaseError(
            f"case {case.get('id', '?')} has no commitment_relevant flag")
    profile = case.get("context_profile", {}) or {}
    context = format_fixed(profile) + "\n\nEMAIL HISTORY:\n\n" + format_history(case.get("history", []))
    inc = case["incoming"]
    incoming = inc if isinstance(inc, str) else (
        f"From: {inc.get('from_name') or inc.get('from', '')}\n"
        f"Date: {inc.get('date', '')}\nSubject: {inc.get('subject', '')}\n\n{inc.get('body', '')}")
    join = lambda xs: "\n".join(f"- {x}" for x in xs) if xs else "(none)"
    return {
        "context": context,
        "incoming": incoming,
        "commitments_in_history": join(case.get("commitments_in_history", [])),
        "resolved_items": join(case.get("resolved_items", [])),
        "relationship_objective": profile.get("relationship_objective") or None,
        "commitment_relevant": bool(case["commitment_relevant"]),
    }


def score_one(system: str, case: dict, *, draft_fn, call_fn, judge_runs: int,
              judge_reference: bool, judge_max_attempts: int = None) -> tuple[dict, dict]:
    """Returns (row, draft_record). draft_record always carries the draft
    and prompt once draft_fn succeeds, even if everything after it raises --
    an evaluator failure (hard tests, specificity, judge) must never destroy
    an already-generated draft, since those are exactly the cases most
    worth inspecting later. Only a draft_fn failure itself (no draft exists
    yet) is left to propagate to the caller, which is run_system()'s
    existing no-draft error path.

    wall_s is the full end-to-end time for the case, including the
    optional reference-reply judge call -- captured right before return,
    not before that call runs."""
    t0 = time.perf_counter()
    out = draft_fn(case)
    draft = out["draft"]
    record = {"id": case["id"], "system": system, "draft": draft, "prompt": out["prompt"]}

    try:
        mc = metric_case(case)
        ht = metrics.hard_tests(mc, draft, call_fn=call_fn)
        # specificity: facts are extracted from the exact prompt this system
        # sent (stage 1, cached), then matched against the draft (stage 2)
        prompt_facts = metrics.extract_prompt_facts(out["prompt"], call_fn=call_fn)
        spec = metrics.specificity(out["prompt"], draft, call_fn=call_fn)
        jd = metrics.judge(mc, draft, call_fn=call_fn, valid_runs=judge_runs,
                           max_attempts=judge_max_attempts)

        row = {
            "id": case["id"], "system": system, "bucket": case.get("bucket", ""),
            "contact": case["contact"], "model": out["model"], "backend": out.get("backend", ""),
            **{h: ht[h]["pass"] for h in HARD},
            "all_hard_pass": all(ht[h]["pass"] for h in HARD),
            "specificity": spec,
            "n_prompt_facts": len(prompt_facts),
            **{f"judge_{d}": jd[d] for d in DIMS},
            "judge_overall": jd["overall"],
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "cost_usd": metrics.cost(out), "latency_s": metrics.latency(out),
            "judge_attempts": jd["attempts"], "judge_invalid_attempts": jd["invalid_attempts"],
            "error": "",
        }
        row["ref_judge_overall"] = "N/A"
        record.update({"prompt_facts": prompt_facts, "hard_tests": ht, "judge": jd})
    except Exception as e:
        row = {"id": case["id"], "system": system, "bucket": case.get("bucket", ""),
               "contact": case["contact"], "error": f"{type(e).__name__}: {e}"}

    # Reference-reply judging is optional and independent: its failure must
    # never overwrite an otherwise-complete, valid primary row. Only runs
    # when the primary row succeeded (no point judging a reference against
    # a case that already errored) and reference_reply is actually present.
    if judge_reference and not row.get("error") and case.get("reference_reply"):
        try:
            mc = metric_case(case)
            ref = case["reference_reply"]
            ref_text = ref if isinstance(ref, str) else ref.get("body", "")
            rj = metrics.judge(mc, ref_text, call_fn=call_fn, valid_runs=judge_runs,
                               max_attempts=judge_max_attempts)
            row["ref_judge_overall"] = rj["overall"]
        except Exception as e:
            row["ref_judge_error"] = f"{type(e).__name__}: {e}"

    row["wall_s"] = time.perf_counter() - t0
    return row, record


def error_row(system: str, case: dict, exc: Exception) -> dict:
    return {"id": case["id"], "system": system, "bucket": case.get("bucket", ""),
            "contact": case["contact"], "error": f"{type(exc).__name__}: {exc}"}


def run_system(system: str, cases: list[dict], ts: str, *, draft_fn=None, call_fn=None,
               judge_runs: int = None, judge_reference: bool = False,
               judge_max_attempts: int = None, quiet=False) -> list[dict]:
    draft_fn = draft_fn or SYSTEMS[system]
    call_fn = call_fn or metrics.call_judge_model
    RESULTS.mkdir(exist_ok=True)
    rows, records = [], []
    for i, case in enumerate(cases, 1):
        try:
            row, rec = score_one(system, case, draft_fn=draft_fn, call_fn=call_fn,
                                 judge_runs=judge_runs, judge_reference=judge_reference,
                                 judge_max_attempts=judge_max_attempts)
            records.append(rec)
        except Exception as e:
            row = error_row(system, case, e)
            if not quiet:
                print(f"  ! {case['id']} {row['error']}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        rows.append(row)
        if not quiet:
            print(f"  {system} {i}/{len(cases)} {case['id']} "
                  f"{'ERR' if row.get('error') else 'ok'}", file=sys.stderr)
    write_rows(RESULTS / f"{ts}_{system}.csv", rows)
    with (RESULTS / f"{ts}_{system}_drafts.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return rows


def write_rows(path: Path, rows: list[dict]):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(system: str, rows: list[dict], ts: str) -> dict:
    """One row in the fixed SUMMARY_COLUMNS schema."""
    ok = [r for r in rows if not r.get("error")]
    agg = metrics.aggregate(ok) if ok else {}
    lat = [r["latency_s"] for r in ok]
    s = {c: "N/A" for c in SUMMARY_COLUMNS}
    s.update({"ts": ts, "system": system, "n": len(ok), "n_error": len(rows) - len(ok)})
    for h in HARD + ["all_hard_pass"]:
        s[h] = agg.get(h, "N/A")
    s["specificity"] = agg.get("specificity", "N/A")
    s["mean_prompt_facts"] = agg.get("n_prompt_facts", "N/A")
    s["judge_overall"] = agg.get("judge_overall", "N/A")
    for d in DIMS:
        s[f"judge_{d}"] = agg.get(f"judge_{d}", "N/A")
    s["ref_judge_overall"] = agg.get("ref_judge_overall", "N/A")
    s["mean_cost_usd"] = agg.get("cost_usd", "N/A")
    s["mean_prompt_tokens"] = agg.get("prompt_tokens", "N/A")
    s["p50_latency_s"] = statistics.median(lat) if lat else "N/A"
    return s


def append_summary(summaries: list[dict]):
    """Append to summary.csv using the fixed schema. If an existing file's
    header differs from SUMMARY_COLUMNS, refuse rather than corrupt it."""
    path = RESULTS / "summary.csv"
    if path.exists():
        with path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != SUMMARY_COLUMNS:
            raise RuntimeError(f"{path} has a different schema; move it aside before appending")
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="raise")
        if new:
            w.writeheader()
        for s in summaries:
            w.writerow({c: s.get(c, "N/A") for c in SUMMARY_COLUMNS})


def fmt(v):
    if v == "N/A" or v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}" if v >= 0.01 or v == 0 else f"{v:.4f}"
    return str(v)


def print_table(summaries: list[dict]):
    cols = [c for c in SUMMARY_COLUMNS if c not in ("ts", "mean_prompt_tokens")]
    if all(s.get("ref_judge_overall") == "N/A" for s in summaries):
        cols.remove("ref_judge_overall")
    short = {c: c.replace("judge_", "j_").replace("no_", "no ").replace("_", " ") for c in cols}
    widths = {c: max(len(short[c]), *(len(fmt(s.get(c, "N/A"))) for s in summaries)) for c in cols}
    print("  ".join(short[c].ljust(widths[c]) for c in cols))
    for s in summaries:
        print("  ".join(fmt(s.get(c, "N/A")).ljust(widths[c]) for c in cols))


def summarize_by(field: str, system: str, rows: list[dict], ts: str) -> list[dict]:
    """One summary row per distinct value of `field` (e.g. bucket) found in
    rows, each labeled "system/value" so it prints alongside the per-system
    table without being confused for another real system. Skips rows missing
    the field (empty bucket) rather than lumping them into a fake group."""
    groups = {}
    for r in rows:
        v = r.get(field)
        if not v:
            continue
        groups.setdefault(v, []).append(r)
    return [summarize(f"{system}/{v}", subset, ts) for v, subset in sorted(groups.items())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS) + ["all"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--judge-runs", type=int, default=None,
                    help=f"valid judge responses required (default JUDGE_VALID_RUNS={metrics.config.JUDGE_VALID_RUNS})")
    ap.add_argument("--judge-max-attempts", type=int, default=None,
                    help=f"give up after this many judge calls (default JUDGE_MAX_ATTEMPTS={metrics.config.JUDGE_MAX_ATTEMPTS})")
    ap.add_argument("--judge-reference", action="store_true",
                    help="also score the owner's real reply with the rubric (adds judge calls per case)")
    ap.add_argument("--report-by", choices=["bucket"], default=None,
                    help="also print a breakdown by this case field (currently: bucket)")
    a = ap.parse_args(argv)

    if a.split == "test":
        print("WARNING: scoring the TEST split. Dev is the working split.", file=sys.stderr)
    cases = load_cases(a.split, a.limit)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    systems = list(SYSTEMS) if a.system == "all" else [a.system]
    summaries, breakdowns = [], []
    for sysname in systems:
        rows = run_system(sysname, cases, ts, judge_runs=a.judge_runs,
                          judge_reference=a.judge_reference,
                          judge_max_attempts=a.judge_max_attempts)
        summaries.append(summarize(sysname, rows, ts))
        if a.report_by:
            breakdowns.extend(summarize_by(a.report_by, sysname, rows, ts))
    append_summary(summaries)
    print_table(summaries)
    if breakdowns:
        print(f"\nby {a.report_by}:")
        print_table(breakdowns)
        write_rows(RESULTS / f"{ts}_by_{a.report_by}.csv", breakdowns)
    print(f"\nresults: {RESULTS}/{ts}_*.csv, summary appended to {RESULTS}/summary.csv")
    if breakdowns:
        print(f"by-{a.report_by} breakdown: {RESULTS}/{ts}_by_{a.report_by}.csv")


if __name__ == "__main__":
    main()
