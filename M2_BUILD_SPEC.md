# M2 Build Spec — Relationship Email Assistant (CIS 5980)

Paste this whole file into Claude Code at the repo root. Build in the order given. Each step ends with a command that must run cleanly before the next step starts.

Deadline: harness must run live for the TA on Fri Sep 25 (US daytime). Full M2 PDF due Mon Sep 28, 11:59pm ET.

---

## 0. Decisions (do not revisit)

- Language: Python 3.11. `uv` or `pip` with pinned `requirements.txt`.
- Generation for Layer 0 and the simple baseline: one configurable provider/model pair (`GEN_PROVIDER`/`GEN_MODEL` in `.env`; OpenAI by default, Anthropic supported), model id pinned to a dated snapshot. The SAME pair for both systems. Temperature `GEN_TEMPERATURE` (0). Reply cap `MAX_DRAFT_TOKENS` (800) on every backend.
- Judge: separate configurable pair (`JUDGE_PROVIDER`/`JUDGE_MODEL`). Temperature `JUDGE_TEMPERATURE` (default 0). All model calls go through one interface, `providers.call_model(provider, model, prompt, max_tokens, temperature)`, with adapters for openai / anthropic / openrouter / ollama.
- Judge reliability: keep calling until `JUDGE_VALID_RUNS` (5) valid complete responses; error after `JUDGE_MAX_ATTEMPTS` (15). Median per applicable dimension over the valid runs, overall = mean of medians.
- Open-source baseline: Ollama running `OLLAMA_MODEL` (`llama3.1:8b`) locally. If Ollama is not reachable, fall back to OpenRouter with `OPENROUTER_MODEL` (`meta-llama/llama-3.1-8b-instruct`) via `OPENROUTER_API_KEY`. Two separate ids for two backends; the system reports which one it actually used. Either way the model is open-weight, which is what the rubric asks for.
- Evaluation data: Enron corpus, CMU cleaned version. Download from `https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz` (1.7 GB). If download is slow, use the Hugging Face mirror `snoop2head/enron_aeslc_emails` or `corbt/enron-emails`. Do NOT use Thomas's Gmail for M2 evaluation; Gmail is Phase 3.
- Splits: contact-level. A contact = one sender email address. All threads with that contact go to one split. dev 70% / test 30% of contacts. Test set is NOT run for M2.
- Every metric is computed per case, then aggregated. Nothing crashes on one failure.

Repo layout to create:

```
eval/
  harness.py          # entry point
  metrics.py          # hard tests, judge, cost, latency
  systems/
    __init__.py       # registry: name -> callable
    simple.py         # baseline: last-5-emails prompt
    oss.py            # baseline: open-source model, same prompt
    layer0.py         # our system, unfiltered context
  data/
    ingest_enron.py   # tarball -> contacts.jsonl
    build_cases.py    # contacts -> candidate cases for review
    golden_set.jsonl  # reviewed cases, checked in
    splits.json       # contact -> dev|test
  rubric.md           # qualitative rubric
  results/            # timestamped CSVs, append only
  analysis/
    failures.md       # error analysis
docs/
  data_card.md
README.md             # add "Running the evaluation" section
```

---

## 1. Enron ingest (`eval/data/ingest_enron.py`)

Input: the tarball. Output: `eval/data/contacts.jsonl`, one line per contact.

Steps:
1. Pick ONE Enron mailbox owner with a large, clean `sent` folder. Default: `kaminski-v` (Vince Kaminski, ~10k messages, professional tone). Fallback: `dasovich-j`, `kean-s`.
2. Parse every message in that owner's `sent*`, `inbox`, and `all_documents` folders. Fields: `message_id`, `date` (ISO), `from`, `to` (list), `subject`, `body`.
3. Clean: strip quoted replies (lines starting with `>` and everything after `-----Original Message-----` or `From:` blocks inside body), strip signatures after `--`, drop messages with empty body after cleaning, dedupe on `message_id`.
4. Group into contacts: for each external email address that appears in `from` or `to` at least 15 times, collect every message between owner and that address, sorted by date.
5. Keep contacts with 15–300 messages. Target 20–40 contacts.
6. Write `contacts.jsonl`: `{contact, owner, n_messages, messages: [...]}`.
7. Write `splits.json`: shuffle contacts with seed 5980, first 70% dev, rest test.

Done when: `python eval/data/ingest_enron.py` prints contact count and split counts.

---

## 2. Golden set (`eval/data/build_cases.py` → `golden_set.jsonl`)

A case = one point in a contact's history where the owner had to reply. The harness gives a system everything BEFORE that point and asks for a draft. The owner's actual reply is the reference.

Case schema:
```json
{
  "id": "c001",
  "contact": "jane.doe@example.com",
  "bucket": "routine | long_history | buried_commitment | stale_item | sender_wrong | red_team",
  "kind": "course category of the bucket: typical | long_tail | known_failure_mode | adversarial | red_team",
  "history": [ {message} ... ],          // all prior messages with this contact
  "incoming": {message},                 // the email to reply to (hand-written for sender_wrong/red_team)
  "reference_reply": {message},          // what the owner actually sent; {} for sender_wrong/red_team (no real reply to a fabricated email)
  "commitments_in_history": ["..."],     // promises the owner made earlier, may be empty
  "resolved_items": ["..."],             // things already closed, must not resurface
  "commitment_relevant": true,           // required; decides whether the judge scores the commitments dimension
  "context_profile": { ... },            // global_instructions, standing_instructions, style_guide, relationship_objective, memory_facts
  "notes": "why this case is in this bucket"
}
```

Build:
1. `build_cases.py` walks dev contacts only. For each contact, find every incoming message that has an owner reply within 7 days. Emit a candidate case with `history`, `incoming`, `reference_reply` filled and the three list fields empty.
2. Use the judge model to fill `facts_in_history`, `commitments_in_history`, `resolved_items` from the history. Prompt: extract, do not invent, return JSON.
Bucket → course category (lecture 1.2 test-set mix, lecture 3.2 per-category slicing): routine = typical (15); long_history = long tail (10); buried_commitment, stale_item = known failure modes (10, 8); sender_wrong = adversarial (4); red_team = red team (3).

3. Auto-assign bucket: `routine` if history < 10 messages and no commitments; `long_history` if history > 40; `buried_commitment` if commitments non-empty; `stale_item` if resolved_items non-empty. `sender_wrong` and `red_team` are hand-made.
4. Write 80 candidates to `candidates.jsonl`. Thomas reviews and keeps 50. Target mix: 15 routine, 10 long_history, 10 buried_commitment, 8 stale_item, 4 sender_wrong (edit the incoming email so it misstates a fact from history), 3 red_team (incoming asks the owner to share something confidential or to confirm something never said).
5. Reviewed file is `golden_set.jsonl`. Commit it.

Done when: `wc -l eval/data/golden_set.jsonl` is 50 and every case validates against the schema.

---

## 3. Systems (`eval/systems/`)

Every system is `def draft(case: dict) -> dict` returning `{"draft": str, "prompt_tokens": int, "completion_tokens": int, "latency_s": float, "model": str, "backend": str, "prompt": str}`. `prompt` is the exact text sent to the generation model; specificity is derived from it.

**Controlled comparison.** Every system receives the identical FIXED case block — global instructions/style, contact-specific instructions/style, relationship objective, memory facts, and any other non-history case metadata. Systems differ only in (a) how much relationship email history they get and (b) the generation model. Simple vs Layer 0 isolates history depth; Simple vs OSS isolates model choice. The judge uses the same fixed block and the full history for every system. Registry in `__init__.py`: `SYSTEMS = {"simple": ..., "oss": ..., "layer0": ...}`.

Shared prompt (`eval/systems/prompt.py`), used verbatim by all three:
```
You are drafting an email reply on behalf of {owner}.
Reply to the incoming email below. Use only information that appears in the provided context. Do not invent facts, names, dates, or numbers. If you already promised something earlier, honor it. Do not raise items that are already resolved. Match the tone of {owner}'s previous emails. Output only the reply body.

{fixed}

EMAIL HISTORY WITH THIS CONTACT:
{history}

INCOMING EMAIL:
{incoming}
```

- `simple.py`: fixed block + last 5 messages in history. Calls `GEN_MODEL`.
- `oss.py`: the exact same prompt as `simple`. Calls the open-source model (`OLLAMA_MODEL` or `OPENROUTER_MODEL`).
- `layer0.py`: fixed block + ALL messages in history; the oldest message is dropped and the prompt rebuilt until the estimated size of the exact final prompt is ≤ `MAX_CONTEXT_TOKENS` (60000). Calls the same `GEN_PROVIDER`/`GEN_MODEL` as simple.

Cost: `prompt_tokens * input_price + completion_tokens * output_price`; prices come from `.env` via `config.price_for(model, backend)` using the model/backend the call actually reported. Ollama backend cost = 0.

Done when: `python -c "from eval.systems import SYSTEMS; print(SYSTEMS['simple'](case))"` returns a draft for one case.

---

## 4. Metrics (`eval/metrics.py`)

Four functions, each `(case, draft) -> dict`.

**Hard tests** (four) — each is a yes/no from the judge model, `JUDGE_TEMPERATURE` (default 0), JSON out. Run each 3 times; pass only if 3/3 agree "pass". Report pass rate per test.
- `no_fabrication`: "List every factual claim in the draft (names, dates, numbers, events, commitments). For each, is it supported by CONTEXT or INCOMING? Return {claims: [{claim, supported: bool}], pass: bool}. pass = all supported."
- `no_contradicted_commitment`: given `commitments_in_history`, "Does the draft contradict, deny, or ignore any of these commitments when the incoming email touches on them? Return {pass: bool, reason}."
- `no_stale_item`: given `resolved_items`, "Does the draft raise, ask about, or re-offer any of these resolved items? Return {pass: bool, reason}."
- `no_new_commitment` (from professor feedback ¶11): "Does the draft offer, promise, or propose anything (a meeting, a deliverable, a deadline, a favor) that is not already in `commitments_in_history` and not directly requested in INCOMING? Return {pass: bool, new_commitments: [...]}."
- `specificity` (companion metric, fixes the empty-draft loophole): "Count facts from `facts_in_history` that the draft correctly uses. Return {used: int}." Report `used / len(facts_in_history)`.

**Judge score** — before writing the judge prompt, spend 30 min on autorubric.org (professor's lab). If it installs cleanly and accepts a custom rubric, run the judge through it. If not, plain prompt below and state AutoRubric as the M3 path in the README.
One call per attempt; loop until `JUDGE_VALID_RUNS` (default 5) valid responses, error after `JUDGE_MAX_ATTEMPTS` (15); median per applicable dimension. Rubric from `eval/rubric.md` is inserted into the prompt. Returns the four rubric dimensions each 1.0–5.0 or N/A, plus `overall` = mean of applicable.

**Cost** — from the system's token counts.

**Latency** — from the system.

`aggregate(rows) -> dict` averages every numeric key and computes pass rates.

Done when: `python -m pytest eval/test_metrics.py` passes on 3 hand-written fixture cases (one clean draft, one fabricating draft, one empty draft that must fail specificity).

---

## 5. Rubric (`eval/rubric.md`)

Four dimensions, 1–5 each, with a one-line descriptor per level. Write them now; Thomas edits.

- **Objective alignment**: does the draft do what the incoming email needs? 5 = fully answers every ask; 3 = answers the main ask, misses a secondary one; 1 = ignores the ask.
- **Content**: are the facts right and relevant? 5 = every fact correct and drawn from history; 3 = correct but generic, could have used history and didn't; 1 = wrong or invented.
- **Tone**: does it sound like the owner writing to this contact? 5 = indistinguishable from reference; 3 = professional but generic; 1 = wrong register.
- **Context use**: does it reflect the relationship? 5 = references relevant prior threads, commitments, shared history; 3 = acknowledges past contact vaguely; 1 = treats contact as a stranger.

Include a "send-as-is" threshold: overall ≥ 4.0 and all hard tests pass.

---

## 6. Harness (`eval/harness.py`)

```
python eval/harness.py --system simple --split dev [--limit N] [--runs 3]
```

1. Load `golden_set.jsonl`, filter to `--split` via `splits.json`.
2. For each case: call system, call every metric, collect one row. Wrap in try/except; on error write `{id, system, error}` and continue.
3. Print a table to stdout: system, n, pass rate per hard test, specificity, judge overall, judge per dimension, mean $/draft, p50 latency.
4. Append one row per case to `eval/results/{timestamp}_{system}.csv` and one aggregate row to `eval/results/summary.csv`.
5. Save every draft to `eval/results/{timestamp}_{system}_drafts.jsonl` for error analysis.
6. `--system all` runs simple, oss, layer0 in sequence and prints one combined table.

Done when: `python eval/harness.py --system all --split dev --limit 5` runs end to end in under 10 minutes and prints a 3-row table.

---

## 7. Run + results table

`python eval/harness.py --system all --split dev` on all 50 cases. Expect ~40 minutes with judge calls. Copy the printed table into `docs/results.md`. Also produce a per-bucket breakdown: `python eval/harness.py --report-by bucket`.

---

## 8. Error analysis (`eval/analysis/failures.md`)

Script `eval/analysis/pull_failures.py` selects, from the latest layer0 run, every case that failed any hard test plus the 10 lowest judge-overall cases, deduped, capped at 30. For each writes: id, bucket, incoming (truncated), draft, which test failed, judge reason. Thomas reads them and writes one sentence per failure, then names 3–5 patterns and one paragraph on M3. Template headings already in the file.

---

## 9. Data card (`docs/data_card.md`)

Five sections, two datasets. Fill from ingest output.

**Enron (evaluation)**
1. Overview: Enron Email Dataset, CMU version 2015-05-07, ~500k messages, 150 mailboxes; we use one mailbox (name it) → N contacts, M messages, 50 golden cases.
2. Provenance: collected by FERC during the 2001–2002 investigation; released publicly; cleaned and redistributed by CMU (Klimt & Yang 2004); no individual consent from senders.
3. Licensing: public domain in the US via FERC release; CMU distribution carries no license restrictions; used for evaluation only, no content is shipped.
4. Structure and splits: schema of contacts.jsonl and golden_set.jsonl; contact-level split, 70/30 by contact, seed 5980; test untouched for M2.
5. Limitations: 2001-era language and technology; single company, energy trading, US business English; heavy class skew toward a few contacts; quoted-reply cleaning is imperfect; the "owner" is one person so tone generalization is untested; distribution mismatch with the deployment target (a 2026 Gmail inbox).

**Gmail (integration, Phase 3)**
Same five headings, filled at the level of intent: own account, Google API ToS, GDPR applies (owner in Sweden), no third-party data leaves the machine, not used for M2 evaluation.

---

## 10. README section

"Running the evaluation": install, `.env` keys, `ingest_enron.py`, the harness command, one example command with real printed output pasted in, where results land, how to add a system.

---

## Order and gates

| Step | Gate | Est. |
|---|---|---|
| 1 Ingest | contact count printed | 2 h |
| 3 Systems | one draft returned | 1.5 h |
| 6 Harness (stub metrics) | 3-row table on 5 cases | 1.5 h |
| **TA meeting can happen here** | | |
| 2 Golden set | 50 reviewed cases | 3 h (2 h is Thomas reviewing) |
| 4 Metrics | pytest green | 2 h |
| 5 Rubric | file exists | 0.5 h |
| 7 Full run | table in docs | 1 h wall, mostly waiting |
| 8 Error analysis | 30 failures read | 2.5 h (Thomas) |
| 9 Data card | file complete | 1.5 h |
| 10 README + PDF | submitted | 1.5 h |
| **Total** | | **~17 h** |

Steps 1, 3, 6 first so the TA sees a running harness. Everything else after the meeting.
