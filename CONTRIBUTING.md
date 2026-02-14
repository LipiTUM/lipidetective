# Contributing to LipiDetective

Thank you for your interest in contributing to LipiDetective! This document provides guidelines
and instructions for contributing.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LipiTUM/lipidetective.git
   cd lipidetective
   ```

2. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```

3. **Run all checks** (formatting, linting, type-checking, tests):
   ```bash
   make all
   ```

## Running Tests

```bash
# Run all checks (lint + type-check + full test suite)
make all

# Run fast tests only (skips slow integration tests)
make test-fast

# Run tests with pytest directly
uv run pytest

# Run a specific test file
uv run pytest tests/test_lipid_library.py
```

## Code Style

This project uses automated tooling to enforce consistent code style:

- **[Ruff](https://docs.astral.sh/ruff/)** for formatting and linting (line length: 100)
- **[mypy](https://mypy-lang.org/)** for static type checking (runs on `src/` only)

Run the checks locally before submitting a PR:

```bash
# Format and lint
make lint

# Type-check
make typecheck
```

## Pull Request Process

1. **Branch from `develop`** — all feature work targets the `develop` branch, not `main`.
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — keep commits focused and descriptive.

3. **Ensure CI passes** — run `make all` locally before pushing.

4. **Open a PR** against `develop` with:
   - A clear description of what changed and why
   - Reference to any related issues (e.g., "Closes #42")

5. **Review** — a maintainer will review your PR. Please be responsive to feedback.

## Reporting Issues

- **Bug reports:** Use the [bug report template](https://github.com/LipiTUM/lipidetective/issues/new?template=bug_report.yml)
- **Feature requests:** Use the [feature request template](https://github.com/LipiTUM/lipidetective/issues/new?template=feature_request.yml)

## Questions?

If you have questions about contributing, feel free to open a
[discussion](https://github.com/LipiTUM/lipidetective/issues) or reach out to the maintainers.
