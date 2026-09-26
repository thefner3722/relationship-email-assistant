"""Pull real failing cases for error analysis (M2 tracker item: Failure pull).

Selects, from the most recent run's *_drafts.jsonl files, every case that
failed any hard test plus the 10 lowest judge-overall cases, deduped,
capped at 30 total. Writes a readable analysis/failures.md with, for
each: id, system, bucket, incoming (truncated), draft, which hard test(s)
failed, judge scores and reasoning where available.

Run from eval/:  python analysis/pull_failures.py
Writes: analysis/failures.md
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "failures.md")

HARD = ["no_fabrication", "no_contradicted_commitment", "no_stale_item", "no_new_commitment"]


def latest_timestamp():
    """The most recent run's timestamp, from whichever *_drafts.jsonl files
    share the latest prefix actually present in results/."""
    files = glob.glob(os.path.join(RESULTS, "*_drafts.jsonl"))
    if not files:
        sys.exit(f"No *_drafts.jsonl files found in {RESULTS} -- run the harness first.")
    ts_list = sorted({os.path.basename(f).split("_")[0] for f in files})
    return ts_list[-1]


def load_run(ts):
    """{(system, id): {**row, **record}} for every case across all systems
    that ran at this timestamp. row supplies bucket/hard-test pass/judge
    scores; record supplies the actual draft/prompt/reasoning."""
    combined = {}
    for csv_path in glob.glob(os.path.join(RESULTS, f"{ts}_*.csv")):
        system = os.path.basename(csv_path)[len(ts) + 1:-4]
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                combined[(system, row["id"])] = dict(row)
        drafts_path = csv_path[:-4] + "_drafts.jsonl"
        if os.path.exists(drafts_path):
            with open(drafts_path) as f:
                for line in f:
                    rec = json.loads(line)
                    key = (rec["system"], rec["id"])
                    if key in combined:
                        combined[key].update(rec)
    return combined


def select_failures(combined, cap=30, n_lowest_judge=10):
    """Every case failing >=1 hard test, plus the n_lowest_judge cases by
    judge_overall (numeric only -- N/A excluded from this second pool,
    since it isn't a quality signal to rank by), deduped, capped."""
    hard_fails = [c for c in combined.values() if c.get("error", "") == ""
                 and any(str(c.get(h)).lower() == "false" for h in HARD)]

    def judge_val(c):
        try:
            return float(c.get("judge_overall"))
        except (TypeError, ValueError):
            return None

    scored = [(judge_val(c), c) for c in combined.values() if c.get("error", "") == ""]
    scored = [(v, c) for v, c in scored if v is not None]
    scored.sort(key=lambda vc: vc[0])
    lowest_judge = [c for _, c in scored[:n_lowest_judge]]

    seen, out = set(), []
    for c in hard_fails + lowest_judge:
        key = (c["system"], c["id"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out[:cap]


def render(cases):
    lines = ["# Initial Failure Pull\n",
            f"{len(cases)} cases selected: every hard-test failure, plus the lowest "
            "judge-overall scores, deduped.\n"]
    for c in cases:
        failed = [h for h in HARD if str(c.get(h)).lower() == "false"]
        lines.append(f"## {c['id']} ({c.get('system', '?')}, bucket: {c.get('bucket', '?')})")
        lines.append(f"**Failed hard tests:** {', '.join(failed) if failed else '(none -- selected on low judge score)'}")
        lines.append(f"**Judge overall:** {c.get('judge_overall', 'N/A')}")
        incoming = (c.get("prompt", "") or "")[:300]
        lines.append(f"**Prompt (truncated):** {incoming}...")
        lines.append(f"**Draft:** {c.get('draft', '(not available)')}")
        ht = c.get("hard_tests")
        if ht:
            for h in failed:
                votes = ht.get(h, {}).get("votes") if isinstance(ht, dict) else None
                if votes:
                    lines.append(f"**{h} votes:** {votes}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ts = latest_timestamp()
    combined = load_run(ts)
    cases = select_failures(combined)
    text = render(cases)
    with open(OUT, "w") as f:
        f.write(text)
    print(f"{len(cases)} cases pulled from run {ts} -> {OUT}")
