# ConvoInsight: Customer Conversational Intelligence Platform

ConvoInsight is a Customer Conversational Intelligence Platform powered by Large Language Models (LLMs) and advanced NLP. It analyses customer interactions across diverse channels — chat, voice, email — to extract actionable insights, enabling businesses to optimise customer service and enhance customer experience.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [API Reference](#api-reference)
6. [Installation](#installation)
7. [Development](#development)
8. [Docker](#docker)
9. [Current State & Roadmap](#current-state--roadmap)
10. [Contributing](#contributing)
11. [License](#license)

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/rampal-punia/convo-insight-platform.git
cd convo-insight-platform
python3.12 -m venv venv
source venv/bin/activate

# 2. Install dependencies
cd backend && pip install -r requirements.txt

# 3. Configure environment
cp ../.env.example ../.env
# Edit .env — fill in at least OPENAI_API_KEY

# 4. Start infrastructure
cd .. && docker compose up -d postgres redis

# 5. Set up database
cd backend
python manage.py migrate
python manage.py seed_demo

# 5b. Seed NLP data (required for the playground's RAG classifier)
python manage.py create_intents
python manage.py create_topics
python manage.py populate_rag_store

# 6. Run the backend
python manage.py runserver
# → http://localhost:8000

# 7. Run the frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

**Demo credentials**: `demo_admin` / `demo12345` (admin) or `demo_user_01` / `demo12345` (regular user)

---

## Project Overview

ConvoInsight analyses customer conversations using three different NLP approaches:

1. **Fine-tuned BERT** — custom-trained models for sentiment, intent, and topic classification
2. **Few-shot GPT** — OpenAI GPT-4o-mini with prompt engineering for the same tasks
3. **RAG with PGVector** — retrieval-augmented generation using vector similarity search

The platform includes a **LangGraph-based support agent** that handles e-commerce customer support conversations with tool use (order tracking, modifications, cancellations) and real-time streaming responses via WebSockets.

### Key Capabilities

- Multi-channel conversation analysis (chat, voice, email)
- Real-time sentiment analysis with granular emotion detection
- Intent recognition and topic modeling
- LLM-powered customer support agent with tool use
- Agent performance evaluation
- Interactive dashboard with analytics
- E-commerce order management with integrated support chat

---

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Backend** | Django 6.0 + Django REST Framework 3.17 | Versioned REST API at `/api/v1/` |
| **Async** | Django Channels 4.3 + Daphne 4.2 | WebSocket support for real-time chat |
| **Auth** | simplejwt + allauth | JWT tokens + social login |
| **Database** | PostgreSQL 17 + pgvector | Relational + vector similarity search |
| **Task Queue** | Celery 5.6 + Redis 7 | Background processing |
| **LLM** | LangChain 1.3 + LangGraph 1.2 | Agent orchestration, tool use |
| **ML** | PyTorch 2.12 + Transformers 5.9 | Fine-tuned BERT models |
| **API Docs** | drf-spectacular 0.29 | OpenAPI 3 + Swagger + Redoc |
| **Frontend** | Next.js 15 (App Router, JSX) | TailwindCSS |
| **CI/CD** | GitHub Actions | Lint, test, build, security scan |
| **Containers** | Docker (multi-stage) + Compose | Postgres, Redis, Daphne, Celery |

---

## Project Structure

```
convo-insight-platform/
├── backend/                    # Django backend
│   ├── config/                 # Project settings, URLs, ASGI/WSGI, Celery
│   │   └── settings/
│   │       ├── base.py         # Shared settings (sys.path trick, DRF, JWT)
│   │       ├── development.py  # Local dev overrides
│   │       ├── production.py   # Production overrides
│   │       └── test.py         # Test overrides
│   ├── apps/
│   │   ├── accounts/           # User management
│   │   ├── products/           # Product catalogue + categories
│   │   ├── orders/             # Order management + LangGraph agent
│   │   ├── convochat/          # Core conversations, messages, NLP models
│   │   ├── analysis/           # Metrics, agent performance, recommendations
│   │   ├── api/                # DRF API layer
│   │   │   └── v1/             # Versioned endpoints (21 ViewSets)
│   │   ├── dashboard/          # Dashboard + seed_demo command
│   │   ├── playground/         # NLP playground (BERT, GPT, RAG methods)
│   │   ├── support_agent/      # LangGraph e-commerce support agent
│   │   ├── general_assistant/  # General AI assistant (text + voice)
│   │   └── llms/               # LLM fine-tuning + SageMaker integration
│   ├── data_processing/        # Data ingestion scripts
│   ├── scripts/                # DB init SQL
│   ├── ml_models/              # Trained model files (gitignored)
│   ├── static/                 # Static assets
│   ├── manage.py
│   ├── requirements.txt
│   ├── conftest.py             # Shared pytest fixtures
│   └── pyproject.toml          # Ruff, pytest, coverage config
├── frontend/                   # Next.js 15 (JSX, App Router)
│   ├── src/
│   │   ├── app/                # Pages (login, products)
│   │   └── lib/                # API client with JWT refresh
│   └── README.md
├── docs/                       # Project documentation
│   └── INTERN_ONBOARDING.md    # Intern onboarding guide
├── docker-compose.yml          # Postgres, Redis, Daphne, Celery
├── Dockerfile                  # Multi-stage build
├── Makefile                    # All common commands
├── .env.example                # Environment variable template
└── README.md
```

---

## API Reference

Interactive docs available at `http://localhost:8000/api/docs/` (Swagger) and `http://localhost:8000/api/redoc/` (Redoc) when the server is running.

### Authentication

```
POST /api/v1/auth/token/          # Obtain JWT pair (access + refresh)
POST /api/v1/auth/token/refresh/  # Refresh access token
POST /api/v1/auth/token/verify/   # Verify token is valid
POST /api/v1/auth/token/blacklist/ # Blacklist a refresh token
```

All authenticated endpoints require `Authorization: Bearer <access_token>`.

### Products & Categories

```
GET    /api/v1/products/           # List products (paginated, filterable)
POST   /api/v1/products/           # Create product
GET    /api/v1/products/{id}/      # Retrieve product
GET    /api/v1/products/in-stock/  # Products with stock > 0
GET    /api/v1/products/low-stock/ # Products with stock < 10
GET    /api/v1/categories/         # List categories
GET    /api/v1/categories/{id}/products/  # Products in a category
```

### Orders

```
GET    /api/v1/orders/             # List orders (paginated)
POST   /api/v1/orders/             # Create order
GET    /api/v1/orders/{id}/        # Retrieve order with items
GET    /api/v1/orders/{id}/tracking/     # Order tracking events
POST   /api/v1/orders/{id}/tracking/add/ # Add tracking event
GET    /api/v1/orders/{id}/items/        # Order line items
```

### Conversations & Messages

```
GET    /api/v1/conversations/      # List conversations
POST   /api/v1/conversations/      # Start a conversation
GET    /api/v1/conversations/{id}/ # Retrieve with messages
GET    /api/v1/conversations/{id}/messages/  # Paginated messages
POST   /api/v1/conversations/{id}/archive/   # Archive conversation
POST   /api/v1/conversations/{id}/end/       # End conversation
```

### NLP Analysis

```
POST   /api/v1/nlp/sentiment/     # Sentiment analysis (BERT / GPT / RAG)
POST   /api/v1/nlp/intent/        # Intent recognition
POST   /api/v1/nlp/topic/         # Topic modeling
POST   /api/v1/nlp/ner/           # Named entity recognition
GET    /api/v1/topics/trending/   # Trending topics
```

### Analysis & Metrics

```
GET    /api/v1/agent-performance/      # LLM agent performance records
GET    /api/v1/conversation-metrics/   # Conversation metrics
GET    /api/v1/recommendations/        # Agent recommendations
GET    /api/v1/intent-predictions/     # Intent prediction records
GET    /api/v1/topic-distributions/    # Topic distribution records
```

### Users

```
GET    /api/v1/users/              # List users (admin only)
GET    /api/v1/users/{id}/         # Retrieve user profile
```

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### 1. Clone and set up virtual environment

```bash
git clone https://github.com/rampal-punia/convo-insight-platform.git
cd convo-insight-platform
python3 -m venv venv
source venv/bin/activate
```

### 2. Install backend dependencies

```bash
cd backend && pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:
- `OPENAI_API_KEY` — required for GPT-based NLP and LangGraph agent
- `HUGGINGFACEHUB_API_TOKEN` — required for HuggingFace model access
- `TAVILY_API_KEY` — required for web search tool in support agent

### 4. Start infrastructure services

```bash
docker compose up -d postgres redis
```

This starts:
- **PostgreSQL 17** on `localhost:5433` with pgvector extension
- **Redis 7** on `localhost:6380`

### 5. Initialize database

```bash
cd backend
python manage.py migrate
python manage.py seed_demo
```

The `seed_demo` command creates sample categories, products, demo users, and orders.

Then seed the NLP playground data:

```bash
python manage.py create_intents
python manage.py create_topics
python manage.py populate_rag_store
```

These load intent/topic labels and generate vector embeddings used by the RAG-based classifier. The `vector` extension (pgvector) is enabled automatically by `migrate` — no manual setup needed.

### 6. Run the development server

```bash
python manage.py runserver
```

Visit:
- **Swagger UI**: http://localhost:8000/api/docs/
- **Admin**: http://localhost:8000/admin/ (`demo_admin` / `demo12345`)
- **API root**: http://localhost:8000/api/v1/

### 7. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

---

## Development

### Common Commands

All Python/Django commands run from `backend/`. Use `make` from the repo root for convenience.

```bash
# From repo root using Make
make runserver          # Start Django dev server
make migrate            # Run migrations
make makemigrations     # Create new migrations
make test               # Run pytest
make test-cov           # Run tests with coverage
make lint               # Lint and auto-fix with ruff
make seed               # Seed demo data
make shell              # Django shell

# Or directly from backend/
cd backend
python manage.py runserver
python manage.py migrate
pytest
ruff check .
celery -A config worker -l info
celery -A config beat -l info
```

### Running the Full Stack

```bash
# Terminal 1: Infrastructure
docker compose up -d postgres redis

# Terminal 2: Django
cd backend && python manage.py runserver

# Terminal 3: Celery worker
cd backend && celery -A config worker -l info

# Terminal 4: Celery beat (optional, for scheduled tasks)
cd backend && celery -A config beat -l info

# Terminal 5: Frontend
cd frontend && npm run dev
```

### Code Style

- **Line length**: 120, single quotes (enforced by ruff)
- **Imports**: `from <app> import ...` for sibling apps (the `apps/` directory is on `sys.path`)
- **JSX only** in the frontend — no TypeScript files
- Run `make lint` before committing

### Testing

```bash
cd backend

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest apps/api/v1/tests/test_smoke.py -v

# Run with coverage
pytest --cov=apps --cov-report=term-missing
```

---

## Docker

### Development (infrastructure only)

```bash
docker compose up -d postgres redis
```

### Full stack via Docker

```bash
docker compose up -d           # Starts all services
docker compose logs -f web     # Follow Django logs
docker compose down            # Stop everything
docker compose down -v         # Stop and remove volumes (full reset)
```

### Building

```bash
docker compose build           # Build the backend image
make build                     # Same thing via Make
```

---

## Current State & Roadmap

### What Works Now

- Full REST API with 21 ViewSets, JWT auth, and OpenAPI docs
- NLP analysis pipeline (sentiment, intent, topic, NER) via three methods
- LangGraph support agent with tool use and WebSocket streaming
- E-commerce models (products, orders, categories, tracking)
- Docker Compose for local infrastructure
- CI pipeline (lint, test, build, security scan)
- Next.js frontend skeleton with login and products page

### Roadmap

1. **Test coverage** — unit + integration tests for all apps (currently 12 smoke tests)
2. **Code cleanup** — remove dead code, replace `print()` with logging, fix typos
3. **Frontend pages** — dashboard, chat interface, orders management, analytics
4. **Multi-agent orchestration** — separate agents for billing, tech support, orders
5. **RAG enhancement** — chunking strategies, hybrid search, reranking
6. **Voice upgrade** — migrate to Whisper for speech recognition

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Key points:

- Branch from `development`, PR against `development`
- Every PR must include tests
- Run `make lint && make test` before pushing
- Single quotes, 120 char line length

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is intended as a learning exercise and demonstration of technology integration. It is not designed or tested for production use. Please perform thorough testing and security audits before considering any aspects for production environments.
