# Contributing

## Setup
1. Clone the repo
2. Backend: `cd backend && pip install -r requirements.txt` (from M2)
3. Frontend: `cd frontend && npm install` (from M2)
4. Copy `.env.example` to `.env` and fill in keys — never commit `.env`

## Workflow
- Branch from `main`: `git switch -c feat/<short-name>`
- Use conventional commits: `fix:`, `feat:`, `docs:`, `refactor:`
- Run tests before pushing: `python -m pytest`
- Open a PR; at least one review before merge; CI must be green

## Questions
Open an issue or email thefner@engineering.upenn.edu
