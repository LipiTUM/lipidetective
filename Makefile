.PHONY: help install install-dev test lint format type-check clean docs all

# Default target
help:
	@echo "LipiDetective Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         Auto-format code with ruff"
	@echo "  make lint           Check code style with ruff"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make test           Run pytest with coverage"
	@echo "  make all            Run format, lint, type-check, and test"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build Sphinx documentation"
	@echo "  make docs-serve     Build and serve documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and caches"
	@echo ""

# Installation
install:
	@echo "Installing production dependencies..."
	poetry install --without dev

install-dev:
	@echo "Installing development dependencies..."
	poetry install

# Code formatting
format:
	@echo "Formatting code with ruff..."
	poetry run ruff format src/ tests/

# Linting
lint:
	@echo "Checking code format..."
	poetry run ruff format --check src/ tests/
	@echo ""
	@echo "Running ruff linter..."
	poetry run ruff check src/ tests/

# Type checking
type-check:
	@echo "Running mypy type checking..."
	poetry run mypy src/

# Testing
test:
	@echo "Running tests with coverage..."
	poetry run pytest tests/ -v

test-fast:
	@echo "Running tests without coverage..."
	poetry run pytest tests/ -v --no-cov -m "not slow"

test-slow:
	@echo "Running slow tests..."
	poetry run pytest tests/ -v -m "slow"

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && poetry run sphinx-build -W -b html source build/html
	@echo ""
	@echo "Documentation built! Open docs/build/html/index.html"

docs-serve:
	@echo "Building and serving documentation..."
	cd docs && poetry run sphinx-build -W -b html source build/html
	@echo ""
	@echo "Serving documentation at http://localhost:8000"
	cd docs/build/html && python -m http.server 8000

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned!"

# Run all checks
all: format lint type-check test
	@echo ""
	@echo "✅ All checks passed!"
