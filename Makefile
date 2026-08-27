sh = uv run --no-sync --frozen

.DEFAULT_GOAL := check

.PHONY: install
install:
	uv sync --all-groups

.PHONY: lint
lint:
	$(sh) ruff format --check
	$(sh) ruff check

.PHONY: fix
fix:
	$(sh) ruff format
	$(sh) ruff check --fix

.PHONY: typing
typing:
	$(sh) mypy

.PHONY: test
test:
	$(sh) pytest

.PHONY: validate
validate:
	$(sh) python -m scripts.validate_benchmark

.PHONY: check
check: lint typing test validate
