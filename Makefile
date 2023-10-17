.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	python3 -m pip install .

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Linting code: Running pre-commit"
	pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	mypy .
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	python3 -m pytest --cov --cov-config=pyproject.toml --cov-report html --cov-report term

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	mkdocs serve

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

benchmark:
	$(MAKE) run -C tests/verificationCode

benchmark-clean:
	$(MAKE) clean -C tests/verificationCode

.DEFAULT_GOAL := help
