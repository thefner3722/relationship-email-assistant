"""Shared drafting prompt and the FIXED case block.

Controlled comparison (M2 design):
  Every system receives the identical fixed case information — global
  instructions/style, contact-specific instructions/style, relationship
  objective, memory facts and any other non-history case metadata. The
  only things that differ between systems are (a) how much relationship
  email history is included and (b) which generation model is called:

    simple : fixed block + last 5 history messages,  GEN_MODEL
    oss    : fixed block + last 5 history messages,  open-source model
    layer0 : fixed block + full history (capped),    GEN_MODEL

  simple vs layer0 isolates history depth; simple vs oss isolates model.
"""

DRAFT_PROMPT = """You are drafting an email reply on behalf of {owner}.
Reply to the incoming email below. Use only information that appears in \
the provided context. Do not invent facts, names, dates, or numbers. If you \
already promised something earlier, honor it. Do not raise items that are \
already resolved. Match the tone of {owner}'s previous emails. Output only \
the reply body.

{fixed}

EMAIL HISTORY WITH THIS CONTACT:
{history}

INCOMING EMAIL:
{incoming}
"""


def format_message(m: dict) -> str:
    """One history message as plain text."""
    return (f"From: {m.get('from_name') or m.get('from', '')}\n"
            f"Date: {m.get('date', '')}\n"
            f"Subject: {m.get('subject', '')}\n\n"
            f"{m.get('body', '')}")


def format_incoming(case: dict) -> str:
    inc = case["incoming"]
    return format_message(inc) if isinstance(inc, dict) else str(inc)


def format_fixed(profile: dict) -> str:
    """The fixed, non-history case block. Identical for every system."""
    profile = profile or {}
    parts = []
    if profile.get("global_instructions"):
        parts.append("GLOBAL INSTRUCTIONS:\n" + profile["global_instructions"])
    if profile.get("standing_instructions"):
        parts.append("STANDING INSTRUCTIONS FOR THIS CONTACT:\n" + profile["standing_instructions"])
    if profile.get("style_guide"):
        parts.append("STYLE GUIDE:\n" + profile["style_guide"])
    if profile.get("relationship_objective"):
        parts.append("RELATIONSHIP OBJECTIVE:\n" + profile["relationship_objective"])
    facts = profile.get("memory_facts") or []
    if facts:
        parts.append("MEMORY (facts from prior correspondence):\n" +
                     "\n".join(f"- {f}" for f in facts))
    extra = profile.get("other_fixed_data")
    if extra:
        parts.append("OTHER CASE DATA:\n" + str(extra))
    return "\n\n".join(parts) if parts else "(no fixed instructions for this case)"


def estimate_tokens(text: str) -> int:
    """~4 chars per token. Used only for the layer0 cap on the exact
    final prompt; real counts come back from the provider."""
    return len(text) // 4


def format_history(msgs: list) -> str:
    return "\n\n---\n\n".join(format_message(m) for m in msgs) or "(no prior messages)"


def build_prompt(case: dict, history_msgs: list) -> str:
    """The exact prompt sent to the drafting model. Systems differ ONLY in
    which history_msgs they pass."""
    return DRAFT_PROMPT.format(
        owner=case["owner"],
        fixed=format_fixed(case.get("context_profile")),
        history=format_history(history_msgs),
        incoming=format_incoming(case),
    )
