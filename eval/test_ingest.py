"""No-network tests for the Enron ingest: multipart body extraction and
(contact, message_id) attribution."""
import sys
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data"))
import ingest_enron as ing  # noqa: E402


def _multipart(body="Hi Vince, the deck is attached.", with_attachment=True):
    m = EmailMessage()
    m["From"] = "Priya <priya@x.com>"
    m["To"] = "vince.kaminski@enron.com"
    m["Subject"] = "deck"
    m["Date"] = "Mon, 01 Jan 2001 09:00:00 -0600"
    m["Message-ID"] = "<mp1@x.com>"
    m.set_content(body)
    m.add_alternative("<p>" + body + "</p>", subtype="html")
    if with_attachment:
        m.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="deck.pdf")
    return m


def test_multipart_text_body_is_not_empty():
    m = _multipart()
    assert m.is_multipart()
    body = ing.extract_body(m)
    assert "the deck is attached" in body
    assert "PDF" not in body            # attachment not mistaken for body
    assert "<p>" not in body            # text/plain preferred over html


def test_multipart_html_only_falls_back_to_stripped_html():
    m = EmailMessage()
    m["From"] = "a@x.com"; m["Message-ID"] = "<h@x>"
    m.add_alternative("<div>hello <b>there</b></div>", subtype="html")
    m.add_attachment(b"x", maintype="application", subtype="octet-stream", filename="f.bin")
    assert "hello" in ing.extract_body(m) and "<b>" not in ing.extract_body(m)


def test_singlepart_body_unchanged():
    m = EmailMessage()
    m["From"] = "a@x.com"
    m.set_content("plain text only")
    assert ing.extract_body(m).strip() == "plain text only"


def test_parse_file_multipart_end_to_end(tmp_path, monkeypatch):
    p = tmp_path / "raw" / "inbox" / "1."
    p.parent.mkdir(parents=True)
    p.write_bytes(_multipart().as_bytes())
    monkeypatch.setattr(ing, "RAW", tmp_path / "raw")
    rec = ing.parse_file(p)
    assert rec is not None and "the deck is attached" in rec["body"]


def test_multi_recipient_email_attributed_once_per_contact():
    """One email from the owner to Alice and Bob -> once in Alice's history,
    once in Bob's, never twice in Alice's."""
    sent = {"message_id": "<m1>", "from": "vince.kaminski@enron.com",
            "to": ["alice@x.com", "bob@x.com"], "direction": "sent", "body": "hi both"}
    per_contact = {"alice@x.com": [sent, sent], "bob@x.com": [sent]}  # alice listed twice on purpose
    recs = ing.relationship_records(per_contact, keep={"alice@x.com", "bob@x.com"})
    by_contact = {}
    for r in recs:
        by_contact.setdefault(r["contact"], []).append(r["message_id"])
    assert by_contact["alice@x.com"] == ["<m1>"]
    assert by_contact["bob@x.com"] == ["<m1>"]
    assert len(recs) == 2


def test_relationship_records_are_deterministic():
    """Same input, any dict/set order -> byte-identical ordered output:
    contacts sorted, messages sorted by (date, message_id)."""
    import json
    def m(mid, date): return {"message_id": mid, "date": date, "body": "x"}
    a1, a2, b1 = m("<a1>", "2001-02-01"), m("<a2>", "2001-01-01"), m("<b1>", "2001-01-15")
    per1 = {"zed@x.com": [a1, a2, a1], "amy@x.com": [b1]}
    per2 = {"amy@x.com": [b1], "zed@x.com": [a2, a1]}
    r1 = ing.relationship_records(per1, keep={"zed@x.com", "amy@x.com"})
    r2 = ing.relationship_records(per2, keep=set(["amy@x.com", "zed@x.com"]))
    assert json.dumps(r1) == json.dumps(r2)
    assert [(r["contact"], r["message_id"]) for r in r1] == [
        ("amy@x.com", "<b1>"), ("zed@x.com", "<a2>"), ("zed@x.com", "<a1>")]


def test_ingest_end_to_end_deterministic(tmp_path, monkeypatch):
    """Ingest the same tiny fixture mailbox twice -> identical files."""
    import json
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    (raw / "inbox").mkdir(parents=True); (raw / "sent").mkdir()
    def write(folder, name, frm, to, mid, date, body):
        m = EmailMessage(); m["From"] = frm; m["To"] = to; m["Subject"] = "s"
        m["Date"] = date; m["Message-ID"] = mid; m.set_content(body)
        (raw / folder / name).write_bytes(m.as_bytes())
    for i in range(12):
        write("inbox", f"{i}.", "Alice <alice@x.com>", "vince.kaminski@enron.com", f"<in{i}@x>",
              f"Mon, 0{1 + i % 9} Jan 2001 09:00:00 -0600", f"hello {i}")
        write("sent", f"{i}.", "vince.kaminski@enron.com", "alice@x.com, bob@x.com", f"<out{i}@x>",
              f"Mon, 0{1 + i % 9} Feb 2001 09:00:00 -0600", f"reply {i}")
    monkeypatch.setattr(ing, "RAW", raw); monkeypatch.setattr(ing, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["ingest", "--min-messages", "5", "--min-each-way", "1"])
    ing.main(); first = (out / "messages.jsonl").read_bytes(), (out / "contacts.jsonl").read_bytes()
    ing.main(); second = (out / "messages.jsonl").read_bytes(), (out / "contacts.jsonl").read_bytes()
    assert first == second
    recs = [json.loads(l) for l in (out / "messages.jsonl").open()]
    contacts = [r["contact"] for r in recs]
    assert contacts == sorted(contacts)
    for c in set(contacts):
        dates = [r["date"] for r in recs if r["contact"] == c]
        assert dates == sorted(dates)
