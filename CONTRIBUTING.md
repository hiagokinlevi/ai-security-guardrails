# Contributing to ai-security-guardrails

Thank you for your interest in contributing. This document explains how to set up a development
environment and submit contributions.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip + venv

### Setup

```bash
# Clone the repository
git clone https://github.com/hiagokinlevi/ai-security-guardrails.git
cd ai-security-guardrails

# Create virtual environment and install dependencies
uv sync --dev

# Copy environment configuration
cp .env.example .env
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=guardrails --cov-report=term-missing
```

### Code Style

This project uses `ruff` for linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

Type checking with mypy:

```bash
uv run mypy guardrails/
```

## Contribution Workflow

1. **Fork** the repository on GitHub.
2. **Create a branch** for your feature or fix: `git checkout -b feat/your-feature-name`
3. **Write tests** for any new functionality. PRs without tests for new features will not be merged.
4. **Ensure all tests pass** and the linter reports no errors.
5. **Commit** your changes with a descriptive commit message following
   [Conventional Commits](https://www.conventionalcommits.org/).
6. **Open a pull request** against the `main` branch.

## Pull Request Guidelines

- Keep PRs focused. One feature or fix per PR.
- Write a clear description of what the PR does and why.
- Reference related issues with `Closes #issue_number`.
- If your PR changes behavior, update the relevant documentation.

## Security Contributions

If your contribution addresses a security vulnerability, **do not open a public PR**. Follow the
process in [SECURITY.md](SECURITY.md) instead.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
