# Rubric — Relationship Email Assistant (M2)

Each draft reply is scored 1–5 (decimals allowed; never below 1.0 or
above 5.0) on four qualitative dimensions by the LLM judge. Separate hard tests in `metrics.py` are independent binary
checks; they do not modify or cap the 1–5 rubric score. Specificity is a
separate companion metric (facts from the drafting prompt correctly
referenced / facts in the drafting prompt), also reported separately.

## 1. Instructions and style
Does the draft respect the applicable global instructions and this
contact's standing instructions, including any style/tone guide set for
them?

- **5** — Fully compliant with every applicable instruction and style setting.
- **4** — Compliant; a minor stylistic instruction is missed.
- **3** — Broadly compliant but generic; doesn't clearly reflect the specific instructions set. *(For Enron cases: no instruction was ever set, so this defaults to 5 unless the draft is unprofessional.)*
- **2** — Violates a minor instruction.
- **1** — Violates a clear, explicit instruction.

## 2. Aligns with objectives
Does the draft advance what the owner is actually trying to achieve with
this contact — the underlying relationship objective — not just answer
the literal email?

- **5** — Directly advances the relationship objective for this contact.
- **4** — Consistent with the objective; doesn't actively advance it.
- **3** — Neutral; doesn't work against the objective, doesn't use it either.
- **2** — Misses an opportunity the objective clearly called for.
- **1** — Works against the stated objective.

*(For M2/tonight only: the objective is not inferred independently by each judge run. It is inferred once, before evaluation, from message history available before the target case — the same way case facts are extracted at golden-set build time — and held fixed across all valid judge runs for that case. If no clear objective can be inferred from the available history, this dimension is marked N/A for that case rather than guessed. This is a stand-in because Enron has no real user to ask. In the actual product, the objective is not extracted — the user defines it explicitly for each contact, the same as any other standing instruction.)*

## 3. Addresses commitments
Does the draft honor promises the owner made, follow up on promises
others made to the owner, and correctly leave resolved items alone?

- **5** — Every relevant open commitment is handled correctly in both directions.
- **4** — Handled correctly; a resolved item is mentioned but not mishandled.
- **3** — Partially addresses the relevant commitment(s), but leaves an important aspect unresolved or unclear.
- **2** — A relevant open commitment, in either direction, is missed.
- **1** — Contradicts a commitment the owner made, or fails to follow up on something owed to the owner.
- **N/A** — No commitment is relevant to this case (decided from case metadata before judging, never by the judge). Excluded from the overall score; never enters the numeric mean.

## 4. Accounts for context
Does the draft actually draw on the available emails, uploads, past
chats, and memory — not just the incoming message in isolation?

- **5** — Uses all relevant available context correctly and selectively; incorporates specific prior context when it materially improves the reply.
- **4** — Uses the relevant context correctly but misses a minor opportunity to incorporate useful prior information.
- **3** — Context-agnostic; the same reply would work with no history at all.
- **2** — Ignores context that was directly relevant.
- **1** — Contradicts something established in context.

## Overall score
Hard tests (`metrics.py`) are separate binary correctness/safety checks,
reported separately. A hard-test failure does not modify or cap the
qualitative 1–5 rubric score below — the two are independent numbers.
Specificity is a third, separate companion metric and does not enter
the overall score either.

For each applicable dimension, take the median across the configured number of valid judge runs (`JUDGE_VALID_RUNS`, default: 5).
Overall = mean of the applicable dimension medians. Any dimension
marked N/A for a case (see above) is excluded from that mean.
