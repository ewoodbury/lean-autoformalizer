# Lean Autoformalizer Project Makefile
# =================================

# Default target - shows help
.DEFAULT_GOAL := help
.PHONY: help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project paths
ROOT_DIR := $(shell pwd)
SCRIPTS_DIR := $(ROOT_DIR)/scripts
PYTHON_SRC := $(ROOT_DIR)/src
TESTS_DIR := $(ROOT_DIR)/tests

# UV cache directory
export UV_CACHE_DIR := $(ROOT_DIR)/.uv-cache

##@ Help
help: ## Display this help message
	@echo "$(CYAN)Lean Autoformalizer Project$(RESET)"
	@echo "$(CYAN)===========================$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(YELLOW)Usage:$(RESET)\n  make $(CYAN)<target>$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Setup & Dependencies
bootstrap: bootstrap-python ## Bootstrap the entire development environment
	@echo "$(GREEN)✓ Project bootstrap completed$(RESET)"

bootstrap-python: ## Setup Python environment with uv
	@echo "$(CYAN)Setting up Python environment...$(RESET)"
	@$(SCRIPTS_DIR)/bootstrap_python.sh
	@echo "$(GREEN)✓ Python environment ready$(RESET)"

install-lean-deps: ## Install Lean dependencies via Lake
	@echo "$(CYAN)Installing Lean dependencies...$(RESET)"
	@lake update
	@echo "$(GREEN)✓ Lean dependencies installed$(RESET)"

##@ Building & Compilation
build: build-lean build-python ## Build both Lean and Python components
	@echo "$(GREEN)✓ Full build completed$(RESET)"

build-lean: ## Build Lean components
	@echo "$(CYAN)Building Lean components...$(RESET)"
	@lake build
	@echo "$(GREEN)✓ Lean build completed$(RESET)"

build-python: ## Build Python package
	@echo "$(CYAN)Building Python package...$(RESET)"
	@uv build
	@echo "$(GREEN)✓ Python package built$(RESET)"

check-lean: ## Check Lean code compilation and run basic tests
	@echo "$(CYAN)Checking Lean code...$(RESET)"
	@$(SCRIPTS_DIR)/check_lean.sh
	@echo "$(GREEN)✓ Lean checks passed$(RESET)"

##@ Development & Code Quality
dev-install: bootstrap ## Install development dependencies and setup
	@echo "$(CYAN)Setting up development environment...$(RESET)"
	@uv sync --group dev
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

fix-lint: ## Run all linting and formatting fixes
	@echo "$(CYAN)Running all linters and formatters fixes...$(RESET)"
	@uv run ruff check --fix $(PYTHON_SRC) $(TESTS_DIR)
	@uv run ruff format $(PYTHON_SRC) $(TESTS_DIR)
	@echo "$(GREEN)✓ Linting and formatting fixes applied$(RESET)"

test-lint: ## Run Python linting and formatting checks with ruff
	@echo "$(CYAN)Running Python linter...$(RESET)"
	@uv run ruff check $(PYTHON_SRC) $(TESTS_DIR)
	@uv run ruff format --check $(PYTHON_SRC) $(TESTS_DIR)
	@echo "$(GREEN)✓ Linting and formatting completed$(RESET)"

type-check: ## Run pyrefly type checking
	@echo "$(CYAN)Running type checking...$(RESET)"
	@uv run pyrefly check $(PYTHON_SRC)
	@echo "$(GREEN)✓ Type checking passed$(RESET)"

check-all: lint format-check type-check check-lean ## Run all code quality checks
	@echo "$(GREEN)✓ All checks passed$(RESET)"

##@ Testing
test: ## Run Python tests with pytest
	@echo "$(CYAN)Running Python tests...$(RESET)"
	@uv run pytest $(TESTS_DIR) -v
	@echo "$(GREEN)✓ Tests completed$(RESET)"

test-validate-proofs: ## Run tests that validate generated proofs
	@echo "$(CYAN)Running proof validation tests...$(RESET)"
	@uv run scripts/validate_dataset.py
	@echo "$(GREEN)✓ Proof validation tests completed$(RESET)"

test-watch: ## Run tests in watch mode
	@echo "$(CYAN)Running tests in watch mode...$(RESET)"
	@uv run pytest-watch $(TESTS_DIR) --verbose

test-coverage: ## Run tests with coverage report
	@echo "$(CYAN)Running tests with coverage...$(RESET)"
	@uv run pytest $(TESTS_DIR) --cov=$(PYTHON_SRC) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated$(RESET)"

test-eval: ## Run evals for buillding new proofs (requires OPENROUTER_API_KEY)
	@echo "$(CYAN)Running evaluation metrics...$(RESET)"
	@[ -n "$$OPENROUTER_API_KEY" ] || (echo "$(RED)OPENROUTER_API_KEY must be set to run evaluation$(RESET)" && exit 1)
	@uv run autoformalize evaluate
	@echo "$(GREEN)✓ Tests and evaluation metrics completed$(RESET)"

##@ Execution
run-cli: ## Run the autoformalizer CLI (pass args with ARGS="...")
	@echo "$(CYAN)Running autoformalizer CLI...$(RESET)"
	@uv run autoformalize $(ARGS)

decode: ## Interactively generate Lean code from an English statement
	@echo "$(CYAN)Starting interactive decoder...$(RESET)"
	@uv run autoformalize decode

run-lean-exe: ## Run the Lean executable
	@echo "$(CYAN)Running Lean executable...$(RESET)"
	@lake exe autoformalizer $(ARGS)

##@ Maintenance & Cleanup
clean: ## Clean build artifacts and caches
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	@rm -rf $(UV_CACHE_DIR)
	@lake clean
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup completed$(RESET)"

clean-lean: ## Clean only Lean build artifacts
	@echo "$(CYAN)Cleaning Lean artifacts...$(RESET)"
	@lake clean
	@echo "$(GREEN)✓ Lean cleanup completed$(RESET)"

clean-python: ## Clean only Python build artifacts
	@echo "$(CYAN)Cleaning Python artifacts...$(RESET)"
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	@rm -rf $(UV_CACHE_DIR)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Python cleanup completed$(RESET)"

update-deps: ## Update all dependencies
	@echo "$(CYAN)Updating dependencies...$(RESET)"
	@lake update
	@uv sync --upgrade
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

##@ Release & Distribution
release: clean check-all test build ## Prepare for release (clean, check, test, build)
	@echo "$(GREEN)✓ Release preparation completed$(RESET)"

##@ Utilities
show-info: ## Show project information
	@echo "$(CYAN)Project Information$(RESET)"
	@echo "$(CYAN)==================$(RESET)"
	@echo "Root Directory: $(ROOT_DIR)"
	@echo "Python Source: $(PYTHON_SRC)"
	@echo "Tests Directory: $(TESTS_DIR)"
	@echo "UV Cache: $(UV_CACHE_DIR)"
	@echo ""
	@echo "$(YELLOW)Lean Info:$(RESET)"
	@lake --version 2>/dev/null || echo "Lake not found"
	@echo ""
	@echo "$(YELLOW)Python Info:$(RESET)"
	@uv --version 2>/dev/null || echo "UV not found"
	@echo ""

env: ## Show current environment variables
	@echo "$(CYAN)Environment Variables$(RESET)"
	@echo "$(CYAN)====================$(RESET)"
	@echo "UV_CACHE_DIR=$(UV_CACHE_DIR)"
	@echo "PATH=$$PATH"
