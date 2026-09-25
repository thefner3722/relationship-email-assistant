"""No-API tests for build_cases: candidate selection, bucket rule, schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data"))
import build_cases as bc  # noqa: E402


def msg(i, direction, day, body="x" * 50):
    return {"message_id": f"<m{i}>", "date": f"2001-01-{day:02d}T10:00:00+00:00",
            "from": "p@x.com" if direction == "received" else bc.OWNER, "from_name": "P",
            "to": [bc.OWNER] if direction == "received" else ["p@x.com"],
            "subject": "s", "body": body, "direction": direction, "contact": "p@x.com"}


def test_find_candidates_requires_reply_within_window_and_history():
    msgs = [msg(0, "received", 1), msg(1, "sent", 2), msg(2, "received", 3),
            msg(3, "received", 5),          # replied on day 6 -> candidate
            msg(4, "sent", 6),
            msg(5, "received", 10),         # replied on day 20 -> outside 7 days
            msg(6, "sent", 20),
            msg(7, "received", 21, body="short")]   # too short
    cands = bc.find_candidates("p@x.com", msgs)
    assert [c["incoming"]["message_id"] for c in cands] == ["<m3>"]
    c = cands[0]
    assert [m["message_id"] for m in c["history"]] == ["<m0>", "<m1>", "<m2>"]
    assert c["reference_reply"]["message_id"] == "<m4>"
    assert c["commitment_relevant"] is None and c["bucket"] is None


def test_assign_bucket_precedence():
    # length wins at both ends, even with commitments/resolved present
    long_with_commitment = {"history": [None] * 41, "commitments_in_history": ["x"], "resolved_items": []}
    assert bc.assign_bucket(long_with_commitment) == "long_history"
    short_with_resolved = {"history": [None] * 3, "commitments_in_history": [], "resolved_items": ["x"]}
    assert bc.assign_bucket(short_with_resolved) == "routine"
    # middle length: commitments/resolved decide
    mid = {"history": [None] * 20, "commitments_in_history": [], "resolved_items": []}
    assert bc.assign_bucket(mid) == "routine"
    assert bc.assign_bucket(dict(mid, resolved_items=["x"])) == "stale_item"
    assert bc.assign_bucket(dict(mid, commitments_in_history=["x"], resolved_items=["y"])) == "buried_commitment"
    # boundaries: exactly 10 and exactly 40 are middle-length, not short/long
    assert bc.assign_bucket({"history": [None] * 10, "commitments_in_history": [], "resolved_items": []}) == "routine"
    assert bc.assign_bucket({"history": [None] * 40, "commitments_in_history": [], "resolved_items": []}) == "routine"


def test_validate_case_schema():
    m = msg(0, "received", 1)
    good = {"id": "c001", "contact": "p@x.com", "owner": "V", "bucket": "routine",
            "history": [m], "incoming": m, "reference_reply": msg(1, "sent", 2),
            "commitments_in_history": [], "resolved_items": [], "commitment_relevant": False,
            "context_profile": {"global_instructions": "", "standing_instructions": "", "style_guide": "",
                                "relationship_objective": None, "memory_facts": ["f"]},
            "notes": "", "kind": "typical"}
    assert bc.validate_case(good) == []
    assert bc.validate_case(dict(good, kind="red_team"))
    assert bc.BUCKET_KIND == {"routine": "typical", "long_history": "long_tail",
                              "buried_commitment": "known_failure_mode", "stale_item": "known_failure_mode",
                              "sender_wrong": "adversarial", "red_team": "red_team"}
    assert sum(bc.BUCKET_TARGET.values()) == 70
    assert bc.validate_case(dict(good, commitment_relevant=None))
    assert bc.validate_case(dict(good, bucket="weird"))
    assert bc.validate_case({k: v for k, v in good.items() if k != "context_profile"})
    bad_prof = dict(good, context_profile={"memory_facts": []})
    assert bc.validate_case(bad_prof)


def test_validate_case_allows_empty_reference_reply_only_for_handmade_buckets():
    m = msg(0, "received", 1)
    prof = {"global_instructions": "", "standing_instructions": "", "style_guide": "",
            "relationship_objective": None, "memory_facts": []}
    def case(bucket, rr):
        return {"id": "c090", "contact": "p@x.com", "owner": "V", "bucket": bucket,
                "kind": bc.BUCKET_KIND[bucket], "history": [m], "incoming": m,
                "commitments_in_history": [], "resolved_items": [], "commitment_relevant": False,
                "context_profile": prof, "notes": "", "reference_reply": rr}
    assert bc.validate_case(case("red_team", {})) == []
    assert bc.validate_case(case("sender_wrong", {})) == []
    assert bc.validate_case(case("routine", {}))
    assert bc.validate_case(case("red_team", {"bad": 1}))
