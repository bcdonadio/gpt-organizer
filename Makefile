# Makefile for ChatGPT Organizer Project

# Variables
VENV_DIR := .venv
UV := uv
UV_RUN := UV_CACHE_DIR=.uv-cache $(UV) run --frozen

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Create venv with uv and install runtime dependencies"
	@echo "  install-dev - Run install and install dev dependencies too"
	@echo "  check       - Run all code quality checkers with strict settings"
	@echo "  update      - Update dependencies to latest available versions"
	@echo "  clean       - Remove temporary caches from analyzers and uv cache"
	@echo "  clean-all   - Run clean and remove virtual environment"

# Create venv and install runtime dependencies
.PHONY: install
install:
	$(UV) venv $(VENV_DIR)
	$(UV) sync --no-dev

# Install runtime and dev dependencies
.PHONY: install-dev
install-dev:
	$(UV) sync

# Run all code quality checks with strict settings
.PHONY: check
check:
	@echo "Running code quality checks..."
	@TESTS=pass; \
	$(UV_RUN) python -m black --check --diff . || TESTS=fail; \
	$(UV_RUN) python -m flake8 . || TESTS=fail; \
	$(UV_RUN) python -m mypy . --strict --show-error-codes --warn-unused-ignores || TESTS=fail; \
	$(UV_RUN) python -m pyright || TESTS=fail; \
	if [ "$$TESTS" = "pass" ]; then \
		echo 'All checks passed!'; \
	else \
		echo 'Some checks failed.'; \
		exit 1; \
	fi
	@echo "All checks completed!"

# Update dependencies to latest versions
.PHONY: update
update:
	$(UV) lock --upgrade
	$(UV) sync

# Clean temporary caches
.PHONY: clean
clean:
	@echo "Cleaning temporary caches..."
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	$(UV) cache clean 2>/dev/null || true
	@echo "Caches cleaned!"

# Clean everything including virtual environment
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)/
	@echo "Clean-all completed!"

# Format code (bonus target)
.PHONY: format

format:
	$(UV_RUN) python -m black .
	@echo "Code formatted!"

# Run tests (if any are added later)
.PHONY: test
test:
	$(UV_RUN) python -m pytest -v

# Check if virtual environment exists
.PHONY: check-venv
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
