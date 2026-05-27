# =============================================================================
# ConvoInsight Platform - Makefile
# =============================================================================
# Usage: make <target>
# Run `make help` to see all available targets.
# =============================================================================

.DEFAULT_GOAL := help

# --- Docker Commands ---
up: ## Start all services (Docker)
	docker compose up -d

down: ## Stop all services
	docker compose down

build: ## Build Docker images
	docker compose build

reset: ## Stop services and remove volumes (full reset)
	docker compose down -v

logs: ## Follow logs from all services
	docker compose logs -f

logs-web: ## Follow web service logs only
	docker compose logs -f web

# --- Django Commands (local) ---
migrate: ## Run database migrations
	python manage.py migrate

makemigrations: ## Create new migrations
	python manage.py makemigrations

seed: ## Load seed data into database
	python manage.py loaddata seed_data.json

shell: ## Open Django shell
	python manage.py shell

superuser: ## Create a superuser
	python manage.py createsuperuser

runserver: ## Run Django development server
	python manage.py runserver

# --- Database ---
db-shell: ## Open PostgreSQL shell
	docker compose exec postgres psql -U postgres -d convoinsight

# --- Code Quality ---
lint: ## Run linter (ruff)
	ruff check . --fix

format: ## Format code (ruff format)
	ruff format .

check: ## Run all checks (lint + format check)
	ruff check .
	ruff format --check .

# --- Testing ---
test: ## Run all tests
	pytest --tb=short -q

test-cov: ## Run tests with coverage report
	pytest --cov=apps --cov-report=term-missing --cov-report=html

test-verbose: ## Run tests with verbose output
	pytest -v

# --- Setup ---
setup: ## First-time setup (install deps + create env)
	cp -n .env.example .env || true
	pip install -r requirements.txt
	python manage.py migrate
	@echo "Setup complete! Edit .env with your API keys, then run 'make runserver'"

install: ## Install Python dependencies
	pip install -r requirements.txt

# --- Celery ---
celery-worker: ## Start Celery worker (local)
	celery -A config.celery worker --loglevel=info

celery-beat: ## Start Celery beat scheduler (local)
	celery -A config.celery beat --loglevel=info

# --- Help ---
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
