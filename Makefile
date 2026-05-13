.PHONY: help install test test-slow lint format docs clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with dev + doc extras
	pip install -e ".[dev,doc]"

test:  ## Run fast tests
	pytest

test-slow:  ## Run all tests including fixture regression suite
	pytest --runslow

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Format with ruff
	ruff format src tests

docs:  ## Build Sphinx docs
	cd docs && $(MAKE) html

clean:  ## Remove build artifacts
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -rf docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
