# Relationship Email Assistant

An email assistant organized around relationships, not threads. It maintains a persistent, per-contact memory built from Gmail history and produces context-aware reply drafts — approved by the user before anything is saved. The system never sends.

**Status:** Phase 1 — Proposal & Scope (CIS 5980 AI Capstone, Fall 2026, Group 10)

## How it works
- **Per-contact memory** — facts, commitments, and preferences extracted from email history, with source references
- **Three context-engineering layers** — unfiltered context → relevance filtering (inverted index + vector search) → importance scoring with time decay
- **Evaluation harness** — pass/fail correctness tests (no fabricated facts, no contradicted commitments, no stale items) plus LLM-as-judge quality scoring against thread-only and full-context baselines
- **Headline result** — - **Headline result** — - **Hes context strategies

## Tech stack
React · FastAPI · PostgreSQL + pgvector · Claude API · Gmail API · Docker · GitHub Actions

## Setup
In development — setup guide lands with Milestone 2.

## License
MIT
