# Repository Guidelines

Minimal repository; use this guide to keep contributions consistent and reviewable.

## Project Structure & Module Organization
- Use `src/` for application and library code; prefer a package namespace such as `src/rag_project/`.
- Mirror source files with tests in `tests/`, keeping identical module paths.
- Place one-off scripts in `scripts/`, experiments in `notebooks/`, and sample artifacts under `assets/`.
- For data-heavy work, store inputs in `data/raw/` and derived outputs in `data/processed/`; avoid committing large binaries; prefer Git LFS.

## Build, Test, and Development Commands
- Create a virtual environment (Python 3.11+ recommended): `python -m venv .venv && source .venv/bin/activate`.
- Install runtime dependencies: `pip install -r requirements.txt`; add a `requirements-dev.txt` for tooling and pin versions.
- Run the suite with `pytest` from the repo root; add `-k` or `-q` flags for targeted runs during iteration.
- Expose linters/formatters (e.g., `ruff`, `black`, `isort`) via scripts like `python -m ruff check src tests` to keep commands discoverable.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; keep functions small and pure where possible.
- Prefer type hints and docstrings for public interfaces; run `mypy` when static typing is introduced.
- Use snake_case for modules/functions, PascalCase for classes, and UPPER_SNAKE_CASE for constants.
- Keep imports ordered (stdlib, third-party, local); avoid wildcard imports.

## Testing Guidelines
- Write `pytest` tests in files named `test_*.py` colocated in `tests/`; use fixtures for shared setup.
- Cover edge cases (empty inputs, malformed queries, timeout/error paths) alongside happy paths.
- When fixing a bug, add a regression test first; prefer deterministic tests with mocked network/FS access.
- Target meaningful coverage across new code; do not skip tests without a clear TODO and owner.

## Commit & Pull Request Guidelines
- Use concise Conventional Commit messages (`feat:`, `fix:`, `chore:`, `docs:`) and keep each commit focused.
- PRs should summarize the change, link issues, describe testing, and note any data/model updates.
- Include screenshots or terminal output when altering APIs or tooling; keep PRs small and reviewable.

## Security & Configuration Tips
- Never commit secrets, tokens, or private datasets; reference them via environment variables and document defaults in `.env.example`.
- Remove transient outputs, caches, and checkpoints before committing; prefer reproducible scripts over checked-in results.
- Validate licenses for added models/datasets/deps and record provenance in the PR.
