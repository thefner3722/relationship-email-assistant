# Relationship Email Assistant

Per-contact email reply assistant, evaluated with a custom harness on real
Enron correspondence (Vince Kaminski's mailbox, `maildir/kaminski-v/`).

**Status:** Milestone 2 — AI Engineering track (CIS 5980, Fall 2026)

## Running the evaluation

### Install

```
cd eval
pip install -r requirements.txt
cp .env.example .env
```

### `.env` setup

Fill in `.env` with, at minimum:

```
GEN_PROVIDER=openai
GEN_MODEL=gpt-6-sol
GEN_MODEL_PRICE=2.00,10.00
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-6-sol
JUDGE_MODEL_PRICE=2.00,10.00
GEN_REASONING_EFFORT=none
JUDGE_REASONING_EFFORT=high
OPENAI_API_KEY=sk-...
```

`.env` is gitignored — never commit it. `GEN_REASONING_EFFORT=none` keeps
drafting at a fixed 800-token visible-output cap comparable across backends;
`JUDGE_REASONING_EFFORT=high` gives the judge model room to reason before
scoring. Any model id used in `*_MODEL` needs a matching `*_MODEL_PRICE` line,
or `cost()` raises rather than silently reporting $0.

### Run the tests (no API calls, no cost)

```
python -m pytest -q
```

68 tests across ingest, metrics, the three systems, and the harness — all run
against fake/mocked model responses, so they check the code itself, not a
live model.

### Data pipeline

The raw Enron corpus is not in this repo. Download `enron_mail_20150507.tar.gz`
from the CMU dataset page (https://www.cs.cmu.edu/~enron/) into
`eval/data/raw/`, then:

```
python eval/data/ingest_enron.py     # -> eval/data/processed/{contacts,messages}.jsonl
python eval/data/build_cases.py      # -> eval/data/candidates.jsonl, then golden_set.jsonl after review
```

`eval/data/processed/*.jsonl`, `splits.json`, and `golden_set.jsonl` are
already committed, so none of the above needs to be re-run just to score the
three systems — only if you want to rebuild the data from scratch.

### Golden set

70 cases in `eval/data/golden_set.jsonl`: routine 15, long_history 15,
buried_commitment 15, stale_item 15, sender_wrong 5, red_team 5.

### Run the harness

```
python harness.py --system simple --split dev --limit N   # sanity-check a small N first
python harness.py --system all --split dev                 # full 70-case run, all three systems
python harness.py --system all --split dev --report-by bucket  # adds a per-bucket breakdown table
```

Flags:
- `--system` — `simple`, `layer0`, `oss`, or `all`
- `--split` — `dev` (default) or `test` (held out; only run at the very end)
- `--limit N` — score only the first N dev-split cases, for a cheap sanity check
- `--judge-runs` / `--judge-max-attempts` — override the default valid-judge-response count and retry budget
- `--judge-reference` — also score the owner's real historical reply with the rubric, for comparison
- `--report-by bucket` — also print (and save to `{ts}_by_bucket.csv`) a breakdown by case bucket, not just by system

### Example command and real output

```
python harness.py --system all --split dev --limit 25
```

```
system  n   n error  fabrication  contradicted commitment  stale item  new commitment  all hard pass  specificity  mean prompt facts  j overall  j instructions and style  j aligns with objectives  j addresses commitments  j accounts for context  mean cost usd  p50 latency s
simple  25  0        0.68         0.84                     1.00        0.52            0.32           0.08         37.76              4.08       4.48                      4.64                      3.50                     3.40                    0.0042         1.45
oss     25  0        0.28         0.84                     1.00        0.28            0.12           0.07         37.76              2.85       3.24                      3.22                      2.50                     2.32                    0.00           8.22
layer0  25  0        0.80         0.92                     1.00        0.60            0.44           0.05         91.32              4.16       4.48                      4.64                      3.67                     3.62                    0.02           1.68

results: eval/results/20260926-045012_*.csv, summary appended to eval/results/summary.csv
```

### Where results land

- `eval/results/{ts}_{system}.csv` — one row per case
- `eval/results/{ts}_{system}_drafts.jsonl` — every draft, prompt, and judge/hard-test reasoning (preserved even when a case's evaluation fails partway through, for error analysis)
- `eval/results/summary.csv` — one aggregate row per (timestamp, system) run, appended
- `eval/results/{ts}_by_bucket.csv` — per-bucket breakdown, only written when `--report-by bucket` is used

### Pulling failures for error analysis

```
python analysis/pull_failures.py
```

Pulls every case that failed at least one hard test, plus the 10 lowest
judge-overall scores, deduped, capped at 30, from the most recent run. Writes
`analysis/failures.md`.

### How to add a system

Add a new file under `systems/` with a `draft(case) -> dict` function
returning `{"draft", "prompt_tokens", "completion_tokens", "latency_s",
"model", "backend", "prompt"}`, then register it in `systems/__init__.py`'s
`SYSTEMS` dict under a new name. It becomes available immediately as
`--system <name>` and is included automatically when running `--system all`.

## License

MIT
