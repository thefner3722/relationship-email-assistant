"""Build candidate golden-set cases from the Enron dev contacts.

Two stages, so the free part runs before .env is filled in:

  python eval/data/build_cases.py extract    # no API: messages.jsonl -> candidates.jsonl
  python eval/data/build_cases.py annotate   # API (judge model): fills the LLM fields in place
  python eval/data/build_cases.py validate [file]   # schema check (default golden_set.jsonl)

A case = one point in a contact's history where the owner (Kaminski) had
to reply. history = everything before the incoming email. reference_reply =
what he actually sent within REPLY_WINDOW.

LLM-filled fields (annotate stage, judge model, extract-do-not-invent):
  context_profile.memory_facts     facts about this contact drawn from the history
                                   (the memory every system receives in its fixed block)
  commitments_in_history           promises the owner made earlier, still open
  resolved_items                   things already closed
  context_profile.{global_instructions, standing_instructions, style_guide,
                   relationship_objective}   synthetic, consistent with the history;
                                   relationship_objective is inferred ONCE here and
                                   held fixed for every system and every judge run
  commitment_relevant              True iff commitments_in_history is non-empty
                                   (REQUIRED by the judge; reviewable by hand)
  bucket, notes

Thomas reviews candidates.jsonl and keeps 50 -> golden_set.jsonl.
sender_wrong and red_team cases are hand-made, not generated here.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

HERE = Path(__file__).parent
MESSAGES = HERE / "processed" / "messages.jsonl"
SPLITS = HERE / "splits.json"
CANDIDATES = HERE / "candidates.jsonl"
GOLDEN = HERE / "golden_set.jsonl"

OWNER = "vince.kaminski@enron.com"
OWNER_NAME = "Vince Kaminski"
REPLY_WINDOW = timedelta(days=7)
MIN_HISTORY = 3          # need some history or there's nothing to test
MAX_PER_CONTACT = 4      # spread cases across contacts
TARGET = 80
SEED = 5980

# Bucket -> course category (lecture 1.2 test-set mix; lecture 3.2 per-category slicing)
BUCKET_KIND = {
    "routine":           "typical",
    "long_history":      "long_tail",
    "buried_commitment": "known_failure_mode",
    "stale_item":        "known_failure_mode",
    "sender_wrong":      "adversarial",
    "red_team":          "red_team",
}
BUCKET_TARGET = {"routine": 15, "long_history": 15, "buried_commitment": 15,
                 "stale_item": 15, "sender_wrong": 5, "red_team": 5}
# If the dev run gets too expensive/slow and the golden set needs to shrink:
# cut routine and/or long_history first. buried_commitment and stale_item are
# the most diagnostic buckets (they're what layer0's memory is actually for)
# and should stay full as long as possible.
BUCKETS = set(BUCKET_KIND)
PROFILE_KEYS = ("global_instructions", "standing_instructions", "style_guide",
                "relationship_objective", "memory_facts")


# ---------------------------------------------------------------- extract

def load_messages():
    by_contact = defaultdict(list)
    for line in MESSAGES.open():
        m = json.loads(line)
        by_contact[m["contact"]].append(m)
    for msgs in by_contact.values():
        msgs.sort(key=lambda m: (m["date"], m["message_id"]))
    return by_contact


def slim(m):
    return {k: m[k] for k in ("message_id", "date", "from", "from_name", "to",
                              "subject", "body", "direction")}


def parse(d):
    return datetime.fromisoformat(d)


def find_candidates(contact, msgs):
    out = []
    for i, m in enumerate(msgs):
        if m["direction"] != "received" or len(m["body"].strip()) < 40:
            continue
        t0 = parse(m["date"])
        reply = next(
            (r for r in msgs[i + 1:]
             if r["direction"] == "sent"
             and t0 < parse(r["date"]) <= t0 + REPLY_WINDOW
             and len(r["body"].strip()) >= 20),
            None,
        )
        if reply is None or i < MIN_HISTORY:
            continue
        out.append({
            "id": None,
            "contact": contact,
            "owner": OWNER_NAME,
            "bucket": None,
            "kind": None,
            "history": [slim(x) for x in msgs[:i]],
            "incoming": slim(m),
            "reference_reply": slim(reply),
            "commitments_in_history": [],
            "resolved_items": [],
            "commitment_relevant": None,
            "context_profile": {},
            "notes": "",
        })
    return out


def extract():
    dev = set(json.load(SPLITS.open())["dev"])
    by_contact = load_messages()
    rng = random.Random(SEED)
    pool = []
    for contact in sorted(dev):
        cands = find_candidates(contact, by_contact.get(contact, []))
        rng.shuffle(cands)
        pool.extend(cands[:MAX_PER_CONTACT])
    rng.shuffle(pool)
    pool = pool[:TARGET]
    for n, c in enumerate(pool, 1):
        c["id"] = f"c{n:03d}"
    with CANDIDATES.open("w") as f:
        for c in pool:
            f.write(json.dumps(c) + "\n")
    print(f"{len(pool)} candidates from {len({c['contact'] for c in pool})} contacts -> {CANDIDATES}")
    print("history length:", Counter(min(len(c["history"]) // 10 * 10, 60) for c in pool))


# ---------------------------------------------------------------- annotate

def fmt_history(history):
    return "\n\n".join(
        f"[{m['date'][:10]}] {m['from']} -> {', '.join(m['to'])}\nSubject: {m['subject']}\n{m['body']}"
        for m in history
    )


EXTRACT_PROMPT = """You are preparing a test case for an email-drafting evaluation.
Below is the full email history between {owner} (the OWNER) and one contact,
followed by a new INCOMING email the owner must reply to.

Extract the following. Do not invent anything. Every item must be traceable to a
specific email in the HISTORY (not the incoming email). Return JSON only, no prose,
no markdown fences.

{{
  "memory_facts": [3-8 short factual statements about this contact or the work
       between them, drawn from the history: names, roles, projects, dates,
       numbers, decisions, preferences],
  "commitments_in_history": [promises or offers the OWNER made earlier that are
       still open at the time of the incoming email; can be empty],
  "resolved_items": [questions or requests that were already answered or closed
       in the history and should not be raised again as open; can be empty],
  "notes": "one sentence on what makes this case interesting or routine"
}}

HISTORY:
{history}

INCOMING EMAIL:
[{date}] {sender}
Subject: {subject}
{body}
"""

PROFILE_PROMPT = """You are writing a realistic context profile for an email-reply assistant used by
{owner}. Base it only on the email history below so it is consistent with how the
owner actually writes and what the relationship is about. Return JSON only, no
markdown fences.

{{
  "global_instructions": "1-3 sentences: rules that apply to all of the owner's replies",
  "standing_instructions": "1-3 sentences: how the owner wants replies to THIS contact handled",
  "style_guide": "1-2 sentences describing the owner's tone, length and sign-off",
  "relationship_objective": "1 sentence: what the owner wants from this relationship, or \\"N/A\\" if the history gives no clear objective"
}}

HISTORY:
{history}
"""


def call(prompt) -> dict:
    from providers import call_model
    # see metrics.call_judge_model for why this is generous, not 1500
    r = call_model(config.JUDGE_PROVIDER, config.JUDGE_MODEL, prompt,
                   max_tokens=4096, temperature=0)
    text = r["text"]
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"annotator returned non-object: {text[:100]!r}")
    return data


SHORT_HISTORY_MAX = 10   # history shorter than this -> routine, regardless of commitments/resolved
LONG_HISTORY_MIN = 40    # history longer than this -> long_history, regardless of commitments/resolved


def assign_bucket(c):
    """History length is checked FIRST, at both ends, because almost every
    real ongoing-correspondence case has at least one commitment or
    resolved item -- checking those first (the original approach) starves
    long_history and routine almost entirely. commitments/resolved only
    decide the bucket for the middle-length range."""
    n = len(c["history"])
    if n > LONG_HISTORY_MIN:
        return "long_history"
    if n < SHORT_HISTORY_MAX:
        return "routine"
    if c["commitments_in_history"]:
        return "buried_commitment"
    if c["resolved_items"]:
        return "stale_item"
    return "routine"


def annotate():
    cases = [json.loads(l) for l in CANDIDATES.open()]
    for i, c in enumerate(cases, 1):
        if c["bucket"]:
            continue  # already annotated, resume
        hist = fmt_history(c["history"])
        inc = c["incoming"]
        ex = call(EXTRACT_PROMPT.format(
            owner=c["owner"], history=hist, date=inc["date"][:10],
            sender=inc["from"], subject=inc["subject"], body=inc["body"]))
        c["commitments_in_history"] = list(ex.get("commitments_in_history") or [])
        c["resolved_items"] = list(ex.get("resolved_items") or [])
        c["notes"] = str(ex.get("notes") or "")
        prof = call(PROFILE_PROMPT.format(owner=c["owner"], history=hist))
        obj = prof.get("relationship_objective")
        c["context_profile"] = {
            "global_instructions": str(prof.get("global_instructions") or ""),
            "standing_instructions": str(prof.get("standing_instructions") or ""),
            "style_guide": str(prof.get("style_guide") or ""),
            "relationship_objective": None if not obj or obj == "N/A" else str(obj),
            "memory_facts": [str(f) for f in (ex.get("memory_facts") or [])],
        }
        c["commitment_relevant"] = bool(c["commitments_in_history"])
        c["bucket"] = assign_bucket(c)
        c["kind"] = BUCKET_KIND[c["bucket"]]
        with CANDIDATES.open("w") as f:
            for x in cases:
                f.write(json.dumps(x) + "\n")
        print(f"{i}/{len(cases)} {c['id']} {c['bucket']}")
    print(Counter(c["bucket"] for c in cases))


# ---------------------------------------------------------------- validate

def validate_case(c: dict) -> list:
    """Return a list of problems (empty = valid). This is the schema the
    harness, systems and judge rely on."""
    p = []
    msg_keys = {"message_id", "date", "from", "to", "subject", "body", "direction"}
    for k in ("id", "contact", "owner", "bucket", "history", "incoming", "reference_reply",
              "commitments_in_history", "resolved_items", "commitment_relevant",
              "context_profile", "notes"):
        if k not in c:
            p.append(f"missing {k}")
    if p:
        return p
    if c["bucket"] not in BUCKETS:
        p.append(f"bad bucket {c['bucket']!r}")
    elif c.get("kind") != BUCKET_KIND[c["bucket"]]:
        p.append(f"kind must be {BUCKET_KIND[c['bucket']]!r} for bucket {c['bucket']!r}")
    if not isinstance(c["history"], list) or not all(msg_keys <= set(m) for m in c["history"]):
        p.append("history malformed")
    if not isinstance(c["incoming"], dict) or not msg_keys <= set(c["incoming"]):
        p.append("incoming malformed")
    # reference_reply is normally a real message; hand-made sender_wrong/red_team
    # cases have no real reply to a fabricated incoming, so {} (falsy, naturally
    # skipped by harness.py's --judge-reference check) is valid for those two
    # buckets specifically -- and only those.
    rr = c["reference_reply"]
    if rr == {}:
        if c.get("bucket") not in ("sender_wrong", "red_team"):
            p.append("reference_reply is empty but bucket is not hand-made (sender_wrong/red_team)")
    elif not isinstance(rr, dict) or not msg_keys <= set(rr):
        p.append("reference_reply malformed")
    for k in ("commitments_in_history", "resolved_items"):
        if not isinstance(c[k], list) or not all(isinstance(x, str) for x in c[k]):
            p.append(f"{k} not a list of strings")
    if not isinstance(c["commitment_relevant"], bool):
        p.append("commitment_relevant must be a bool (judge requires it)")
    prof = c["context_profile"]
    if not isinstance(prof, dict) or set(PROFILE_KEYS) - set(prof):
        p.append(f"context_profile must have {PROFILE_KEYS}")
    else:
        if not isinstance(prof["memory_facts"], list):
            p.append("memory_facts not a list")
        if prof["relationship_objective"] is not None and not isinstance(prof["relationship_objective"], str):
            p.append("relationship_objective must be a string or null")
    return p


def validate(path=None):
    path = Path(path) if path else GOLDEN
    cases = [json.loads(l) for l in path.open()]
    bad = {c.get("id"): validate_case(c) for c in cases}
    bad = {k: v for k, v in bad.items() if v}
    ids = [c.get("id") for c in cases]
    if len(set(ids)) != len(ids):
        bad["_"] = ["duplicate ids"]
    print(f"{len(cases)} cases, {len(bad)} invalid")
    for k, v in bad.items():
        print(f"  {k}: {'; '.join(v)}")
    counts = Counter(c.get("bucket") for c in cases)
    for bkt, target in BUCKET_TARGET.items():
        print(f"  {bkt:18s} {BUCKET_KIND[bkt]:20s} {counts.get(bkt, 0):3d} / {target}")
    return not bad


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if stage == "validate":
        sys.exit(0 if validate(sys.argv[2] if len(sys.argv) > 2 else None) else 1)
    {"extract": extract, "annotate": annotate}[stage]()
