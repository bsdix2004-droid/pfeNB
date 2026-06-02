.PHONY: help dev dev-d staging prod down down-staging down-prod down-all \
        logs logs-worker logs-prod logs-all \
        migrate migrate-staging migrate-prod migration db-shell \
        test test-unit test-integration \
        lint format typecheck \
        shell worker-shell \
        ollama-pull ollama-list \
        build build-nocache clean clean-all

DC_BASE    := docker compose -f docker/docker-compose.yml
DC_DEV     := $(DC_BASE) -f docker/docker-compose.dev.yml
DC_STAGING := $(DC_BASE) -f docker/docker-compose.staging.yml
DC_PROD    := $(DC_BASE) -f docker/docker-compose.prod.yml

IMAGE_TAG ?= latest
export IMAGE_TAG

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment boots ---

dev: ## Start DEV stack (hot-reload, debug tools)
	APP_ENV=dev $(DC_DEV) up --build

dev-d: ## Start DEV stack detached
	APP_ENV=dev $(DC_DEV) up --build -d

staging: ## Start STAGING stack
	APP_ENV=staging $(DC_STAGING) up --build -d

prod: ## Start PRODUCTION stack
	APP_ENV=prod $(DC_PROD) up -d

# --- Teardown ---

down: ## Stop dev containers
	APP_ENV=dev $(DC_DEV) down

down-staging: ## Stop staging containers
	APP_ENV=staging $(DC_STAGING) down

down-prod: ## Stop prod containers
	APP_ENV=prod $(DC_PROD) down --timeout 30

down-all: ## Stop ALL environments
	-APP_ENV=dev $(DC_DEV) down --remove-orphans
	-APP_ENV=staging $(DC_STAGING) down --remove-orphans
	-APP_ENV=prod $(DC_PROD) down --remove-orphans

# --- Logs ---

logs: ## Follow API logs (dev)
	APP_ENV=dev $(DC_DEV) logs -f api

logs-worker: ## Follow worker logs (dev)
	APP_ENV=dev $(DC_DEV) logs -f worker

logs-prod: ## Follow API logs (prod)
	APP_ENV=prod $(DC_PROD) logs -f api

logs-all: ## Follow all logs (dev)
	APP_ENV=dev $(DC_DEV) logs -f

# --- Database ---

migrate: ## Run Alembic migrations (dev)
	APP_ENV=dev $(DC_DEV) exec api alembic upgrade head

migrate-staging: ## Run Alembic migrations (staging)
	APP_ENV=staging $(DC_STAGING) exec api alembic upgrade head

migrate-prod: ## Run Alembic migrations (prod)
	@read -p "Run migrations on PRODUCTION? [y/N] " confirm && [ "$$confirm" = "y" ]
	APP_ENV=prod $(DC_PROD) exec api alembic upgrade head

migration: ## Create migration: make migration MSG="add user roles"
	APP_ENV=dev $(DC_DEV) exec api alembic revision --autogenerate -m "$(MSG)"

db-shell: ## Open psql shell (dev)
	APP_ENV=dev $(DC_DEV) exec postgres psql -U vision -d vision_ai_dev

# --- Testing ---

test: ## Run full test suite
	APP_ENV=dev $(DC_DEV) run --rm -e APP_ENV=test api \
	  pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

test-unit: ## Run unit tests only
	APP_ENV=dev $(DC_DEV) run --rm -e APP_ENV=test api \
	  pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	APP_ENV=dev $(DC_DEV) run --rm -e APP_ENV=test api \
	  pytest tests/integration/ -v --tb=short

# --- Code quality ---

lint: ## Run ruff linter
	APP_ENV=dev $(DC_DEV) run --rm api ruff check app/ tests/

format: ## Auto-format with ruff
	APP_ENV=dev $(DC_DEV) run --rm api ruff format app/ tests/

typecheck: ## Run mypy
	APP_ENV=dev $(DC_DEV) run --rm api mypy app/ --ignore-missing-imports

# --- Shell access ---

shell: ## Open bash in API container (dev)
	APP_ENV=dev $(DC_DEV) exec api bash

worker-shell: ## Open bash in worker container (dev)
	APP_ENV=dev $(DC_DEV) exec worker bash

# --- Ollama ---

ollama-pull: ## Pull default Ollama model
	APP_ENV=dev $(DC_DEV) exec ollama ollama pull llama3.2

ollama-list: ## List Ollama models
	APP_ENV=dev $(DC_DEV) exec ollama ollama list

# --- Build ---

build: ## Build Docker image
	docker build --target runtime -t vision_ai:$(IMAGE_TAG) .

build-nocache: ## Build without cache
	docker build --no-cache --target runtime -t vision_ai:$(IMAGE_TAG) .

# --- Cleanup ---

clean: ## Remove stopped containers and unused images
	docker system prune -f
	docker volume prune -f

clean-all: ## Remove everything including volumes (DESTRUCTIVE)
	@read -p "Delete ALL volumes and data? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker compose down -v --remove-orphans