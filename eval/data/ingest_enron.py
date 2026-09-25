"""Ingest one Enron mailbox into contacts.jsonl + messages.jsonl.

Usage (from repo root):
    python eval/data/ingest_enron.py [--min-messages 20] [--external-only]

Contact rule (per data card): mutual exchange (>=1 message each way)
and >=20 messages total; internal Enron contacts are included by
default (--external-only opts out of that).

Reads  eval/data/raw/maildir/kaminski-v/
Writes eval/data/processed/contacts.jsonl
       eval/data/processed/messages.jsonl
Prints counts needed for the data card.
"""
import argparse, email, hashlib, json, re, sys
from collections import defaultdict
from datetime import timezone
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

OWNER = "kaminski-v"
OWNER_ADDRS = {
    "vince.kaminski@enron.com", "vkamins@enron.com", "j.kaminski@enron.com",
    "vince.j.kaminski@enron.com", "kaminski@enron.com", "vkaminski@aol.com",
}
RAW = Path("eval/data/raw/maildir") / OWNER
SYSTEM_RE = re.compile(r"announce|arsystem|mailman|no-?reply|donotreply|listserv|newsletter|@enron\.com$", re.I)
def is_internal(a): return "enron" in a.split("@")[-1]
def is_system(a): return bool(re.search(r"announce|arsystem|mailman|no-?reply|donotreply|listserv|newsletter|daemon|notification", a, re.I))
OUT = Path("eval/data/processed")

QUOTE_MARKERS = [
    r"^-+\s*Original Message\s*-+$",
    r"^-+\s*Forwarded by .*$",
    r"^From:\s.*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}",   # "From: X on 01/02/2001 09:15 AM"
    r"^.*wrote:$",
]
QUOTE_RE = re.compile("|".join(QUOTE_MARKERS), re.M | re.I)


def clean_body(text: str) -> str:
    m = QUOTE_RE.search(text)
    if m:
        text = text[: m.start()]
    lines = [l for l in text.splitlines() if not l.lstrip().startswith(">")]
    return "\n".join(lines).strip()


def _decode(part) -> str:
    raw = part.get_payload(decode=True)
    if not raw:
        return ""
    charset = part.get_content_charset() or "latin-1"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("latin-1", errors="replace")


def extract_body(msg) -> str:
    """Text body of a message, robust to multipart.

    Single-part: decode the payload. Multipart: walk the parts, skip
    attachments (Content-Disposition: attachment, or a filename) and
    non-text parts, prefer the first text/plain; if there is none fall
    back to text/html with tags stripped; else empty."""
    if not msg.is_multipart():
        return _decode(msg)
    plain, html = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = (part.get("Content-Disposition") or "").lower()
        if disp.startswith("attachment") or part.get_filename():
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_decode(part))
        elif ctype == "text/html":
            html.append(_decode(part))
    if plain:
        return "\n".join(t for t in plain if t.strip())
    if html:
        return re.sub(r"<[^>]+>", " ", "\n".join(html))
    return ""


def addrs(msg, header):
    raw = msg.get_all(header, [])
    return [(n.strip(), a.lower().strip()) for n, a in getaddresses(raw) if a]


def parse_file(path: Path):
    try:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f)
    except Exception:
        return None
    mid = (msg.get("Message-ID") or "").strip()
    if not mid:
        return None
    try:
        dt = parsedate_to_datetime(msg.get("Date")).astimezone(timezone.utc)
    except Exception:
        return None
    if not (1998 <= dt.year <= 2002):
        return None
    frm = addrs(msg, "From")
    if not frm:
        return None
    xfrom = (msg.get("X-From") or "").strip()
    name = re.sub(r"\s*<.*$", "", xfrom).strip() or frm[0][0]
    body = clean_body(extract_body(msg))
    key = hashlib.md5(f"{dt.isoformat()}|{frm[0][1]}|{msg.get('Subject','')}|{body[:300]}".encode()).hexdigest()
    return {
        "message_id": mid,
        "dedup_key": key,
        "date": dt.isoformat(),
        "from_name": name, "from": frm[0][1],
        "to": [a for _, a in addrs(msg, "To")],
        "cc": [a for _, a in addrs(msg, "Cc")],
        "subject": (msg.get("Subject") or "").strip(),
        "body": body,
        "folder": path.relative_to(RAW).parts[0],
    }


def relationship_records(per_contact: dict, keep: set) -> list:
    """One record per (contact, message_id). An outgoing email sent to
    two retained contacts appears once in EACH contact's history; the
    same message is never written twice for the same contact. Dedup key
    is (contact, message_id), NOT message_id alone.

    Output order is deterministic: contacts sorted, then each contact's
    messages sorted by (date, message_id)."""
    seen, out = set(), []
    for a in sorted(keep):
        for m in sorted(per_contact[a], key=lambda x: (x.get("date", ""), x["message_id"])):
            key = (a, m["message_id"])
            if key in seen:
                continue
            seen.add(key)
            out.append({**m, "contact": a})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-messages", type=int, default=20)
    ap.add_argument("--external-only", action="store_true",
                    help="keep only non-@enron.com contacts")
    ap.add_argument("--min-each-way", type=int, default=1,
                    help="contact must have at least this many sent AND received")
    ap.add_argument("--max-recipients", type=int, default=5,
                    help="ignore sent mail with more direct recipients than this")
    args = ap.parse_args()

    if not RAW.exists():
        sys.exit(f"missing {RAW} — download per README")

    files = sorted(p for p in RAW.rglob("*") if p.is_file())   # deterministic order
    seen, msgs = set(), []
    for p in files:
        m = parse_file(p)
        if m and m["dedup_key"] not in seen:
            seen.add(m["dedup_key"])
            msgs.append(m)

    # attribute each message to counterpart contact(s)
    per_contact = defaultdict(list)
    names = {}
    for m in msgs:
        if m["from"] in OWNER_ADDRS:
            m["direction"] = "sent"
            if 0 < len(m["to"]) <= args.max_recipients:
                for a in m["to"]:
                    if a not in OWNER_ADDRS:
                        per_contact[a].append(m)
        else:
            m["direction"] = "received"
            per_contact[m["from"]].append(m)
            names.setdefault(m["from"], m["from_name"])

    contacts = []
    for a, ms in per_contact.items():
        internal = is_internal(a)
        if is_system(a):
            continue
        if args.external_only and internal:
            continue
        if len(ms) < args.min_messages:
            continue
        n_s = sum(x["direction"] == "sent" for x in ms)
        n_r = len(ms) - n_s
        if min(n_s, n_r) < args.min_each_way:
            continue
        ds = sorted(x["date"] for x in ms)
        contacts.append({
            "address": a, "name": names.get(a, ""),
            "internal": internal,
            "n_messages": len(ms),
            "n_sent": sum(x["direction"] == "sent" for x in ms),
            "n_received": sum(x["direction"] == "received" for x in ms),
            "first": ds[0], "last": ds[-1],
        })
    contacts.sort(key=lambda c: -c["n_messages"])
    keep = {c["address"] for c in contacts}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "contacts.jsonl", "w") as f:
        for c in contacts:
            f.write(json.dumps(c) + "\n")
    records = relationship_records(per_contact, keep)
    with open(OUT / "messages.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    written = {r["message_id"] for r in records}

    all_dates = sorted(m["date"] for m in msgs)
    n_sent = sum(m["direction"] == "sent" for m in msgs)
    print(f"files on disk        : {len(files)}")
    print(f"unique messages      : {len(msgs)}  (sent {n_sent}, received {len(msgs)-n_sent})")
    print(f"date range           : {all_dates[0][:10]} .. {all_dates[-1][:10]}")
    print(f"contacts >= {args.min_messages:<3}      : {len(contacts)}  "
          f"(internal {sum(c['internal'] for c in contacts)}, "
          f"external {sum(not c['internal'] for c in contacts)})")
    print(f"messages kept        : {len(written)} unique, {len(records)} contact-attributed records")
    print("contacts at other thresholds (same two-way rule):")
    for t in (15, 30, 50, 100):
        n = sum(c["n_messages"] >= t for c in contacts)
        e = sum(c["n_messages"] >= t and not c["internal"] for c in contacts)
        print(f"  >= {t:<3}: {n:3d}  (external {e})")
    print(f"wrote {OUT/'contacts.jsonl'}, {OUT/'messages.jsonl'}")
    print("\ntop 10 contacts:")
    for c in contacts[:10]:
        print(f"  {c['n_messages']:4d}  {'int' if c['internal'] else 'EXT'}  {c['address']}  {c['name']}")


if __name__ == "__main__":
    main()
