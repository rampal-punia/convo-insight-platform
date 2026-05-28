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

# --- Django Commands (local, run from backend/) ---
migrate: ## Run database migrations
	cd backend && python manage.py migrate

makemigrations: ## Create new migrations
	cd backend && python manage.py makemigrations

seed: ## Load seed data into database
	cd backend && python manage.py loaddata seed_data.json

shell: ## Open Django shell
	cd backend && python manage.py shell

superuser: ## Create a superuser
	cd backend && python manage.py createsuperuser

runserver: ## Run Django development server
	cd backend && python manage.py runserver

# --- Database ---
db-shell: ## Open PostgreSQL shell
	docker compose exec postgres psql -U postgres -d convoinsight

# --- Code Quality (run from backend/ where pyproject.toml lives) ---
lint: ## Run linter (ruff)
	cd backend && ruff check . --fix

format: ## Format code (ruff format)
	cd backend && ruff format .

check: ## Run all checks (lint + format check)
	cd backend && ruff check .
	cd backend && ruff format --check .

# --- Testing (run from backend/ where conftest.py + pyproject.toml live) ---
test: ## Run all tests
	cd backend && pytest --tb=short -q

test-cov: ## Run tests with coverage report
	cd backend && pytest --cov=apps --cov-report=term-missing --cov-report=html

test-verbose: ## Run tests with verbose output
	cd backend && pytest -v

# --- Setup ---
setup: ## First-time setup (install deps + create env)
	cp -n .env.example .env || true
	cd backend && pip install -r requirements.txt
	$(MAKE) download-nlp-data
	cd backend && python manage.py migrate
	@echo "Setup complete! Edit .env with your API keys, then run 'make runserver'"

download-nlp-data: ## Download NLP models (spacy, nltk)
	@echo "Downloading spacy en_core_web_sm model..."
	cd backend && python -m spacy download en_core_web_sm
	@echo "Downloading NLTK data..."
	cd backend && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
	@echo "NLP models downloaded successfully!"

install: ## Install Python dependencies
	cd backend && pip install -r requirements.txt

# --- Celery (local, run from backend/) ---
celery-worker: ## Start Celery worker (local)
	cd backend && celery -A config.celery worker --loglevel=info

celery-beat: ## Start Celery beat scheduler (local)
	cd backend && celery -A config.celery beat --loglevel=info

# --- Help ---
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
