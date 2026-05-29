# ConvoInsight Backend — Vision

> The product north star for the ConvoInsight backend. This document defines **what the backend is, what it must become, and how we know we have arrived**. It is independent of any single release, contributor, or sprint.

For the build-out plan and active workstreams, see [ROADMAP](#) (forthcoming) and the README's _Current State & Roadmap_ section. For the frontend's equivalent, see [frontend/docs/VISION.md](../../frontend/docs/VISION.md).

---

## 1. Product Premise

ConvoInsight is a Customer Conversational Intelligence Platform. The backend is the system of record and the system of intelligence: it stores every conversation, runs the NLP that classifies them, orchestrates the LLM agent that resolves them, and exposes everything as a clean, versioned contract that any client — Next.js cockpit, Django templates, internal scripts, third-party integrations — can consume.

The backend's job is to make three things true at once:

1. **Conversations are first-class data.** Every message, intent, topic, sentiment, tool call, and agent decision is captured, indexable, and queryable.
2. **AI behaviour is reproducible.** The three classification methods (fine-tuned BERT, few-shot GPT, RAG over pgvector) and the LangGraph support agent can be re-run against any saved conversation and produce a comparable trace.
3. **The contract is the product.** REST and WebSocket interfaces are the public surface. Anything not behind those contracts is internal.

---

## 2. Vision Statement

> Build the most clear, observable, and contract-driven conversational-intelligence backend in the Django ecosystem — one where every AI decision is traceable, every endpoint is documented, and every async surface degrades safely under load.

We are not building a generic CRUD API. We are building a domain backend for real-time AI conversations, with three first-class concerns: **stateful AI agents, multi-method NLP, and operational analytics**.

---

## 3. Strategic Pillars

Every model, endpoint, consumer, task, and command must serve at least one of these five pillars.

### Pillar 1 — Stateful Agent Orchestration
A LangGraph-based agent (`support_agent/sa_utils/`) that handles e-commerce conversations with intent routing, context management, tool calling, and per-turn checkpointing. The agent is the platform's flagship AI surface and the cleanest reference implementation of a multi-step LLM workflow in the codebase.

### Pillar 2 — Multi-Method NLP Pipeline
Three independent classification methods (fine-tuned BERT, few-shot GPT, RAG with pgvector) across four tasks (sentiment, intent, topic, NER), each callable over REST and streamable over WebSocket. The pipeline is the platform's "show, don't tell" — the place that proves the science.

### Pillar 3 — Real-Time Conversation Backbone
Django Channels + Daphne carry every live conversation: streaming agent responses, playground analyses, multimodal general-assistant exchanges. The backbone is engineered to reconnect, authenticate via JWT, and survive worker restarts without losing state.

### Pillar 4 — Operational Analytics
First-class models for agent performance, conversation metrics, recommendations, topic distributions, and intent predictions. These are not bolt-ons; they are the substrate on which dashboards, evaluations, and product decisions are built.

### Pillar 5 — Contract-First Surface
A versioned REST API (`/api/v1/...`), an OpenAPI 3 schema, drf-spectacular Swagger and Redoc UIs, JWT auth with refresh and blacklist, and an explicit WebSocket protocol. Every client speaks to the platform exclusively through these contracts.

---

## 4. Non-Goals

To stay focused, the backend explicitly **does not** aim to be:

- A general-purpose helpdesk, ticketing system, or CRM.
- A vector database product (we use pgvector; we do not compete with it).
- A model training service (we fine-tune for the platform's own classifiers; we do not offer training-as-a-service).
- A multi-tenant SaaS control plane (single-organisation deployments are the target).
- A workflow builder, integration platform, or low-code engine.

These constraints are deliberate. They keep the codebase opinionated and the public contract small.

---

## 5. Engineering Principles

These principles bind every backend change.

1. **The contract is sacred.** REST routes under `/api/v1/` and WebSocket routes under `/ws/` are versioned. Breaking changes require a new version, not an in-place edit.
2. **OpenAPI is the source of truth.** Every endpoint appears in Swagger with accurate request/response schemas. Endpoints without schema entries do not exist.
3. **Async is for I/O, not parallelism.** WebSocket consumers and external API calls are async; CPU-bound work is offloaded to Celery, not awaited in a request.
4. **Models own their invariants.** Validation lives on the model and the serializer, never only on the view.
5. **Logging, not printing.** Every diagnostic message goes through Python's `logging`. `print()` is a code-smell flag in review.
6. **Migrations are forward-only.** No data loss, no destructive defaults; if a column must be removed, it is deprecated first.
7. **Tests describe behaviour.** Unit tests for utilities, API tests for endpoints, integration tests for WebSocket flows. The test suite is the executable specification.
8. **Secrets and prompts are configurable.** API keys come from environment variables. System prompts and intent definitions live in code or seeded data, never inlined in views.
9. **The `apps/` import trick is part of the architecture.** Sibling apps import as `from products.models import Product`, never `from apps.products.models import Product`. This is enforced by the `sys.path` setup in `config/settings/base.py`.
10. **Server-rendered Django templates are first-class.** The Django template tier (auth, staff workflows, internal dashboards) is not legacy and is not on a deprecation path. See §7.

---

## 6. Surface Tiers

The platform exposes three distinct surfaces, each with a fixed remit. This is the answer to "where does this page belong?" and to "why do we have Django templates and a Next.js client?"

| Tier | Implementation | Remit |
|---|---|---|
| **Admin** | Django admin (`/admin/`) | Power-user data ops, debugging, seeding inspection. Not for non-engineers. |
| **Staff / Operator (server-rendered)** | Django templates + Bootstrap 5 + crispy-forms (`backend/templates/`) | Auth (login, signup, password reset, profile), staff dashboards, internal CRUD that benefits from Django forms and the messages framework, quick server-rendered iteration surfaces for new features before they earn a cockpit slot. |
| **Product Cockpit (client-rendered)** | Next.js 15 App Router (`frontend/`) | Customer-facing chat, operator live-queue, NLP playground comparison view, analytics cockpit, anything that needs streaming, anything shipped to an end customer. |

A surface decision is made **once** per feature, recorded in the PR description, and not revisited unless a written proposal justifies it. See `frontend/docs/STRATEGY.md` and the contributing guide for the operational rules.

---

## 7. Target Users & Primary Jobs

| Persona | Primary Job-To-Be-Done | Backend Surfaces |
|---|---|---|
| **End customer** | Resolve an order issue without waiting for a human. | `ws/support_agent/`, `/api/v1/orders/`, `/api/v1/conversations/` |
| **Support operator** | Watch live conversations and step in when the agent struggles. | `/api/v1/conversations/`, `/api/v1/agent-performance/`, `ws/chat/` |
| **Operations manager** | Understand where conversations break and where the agent excels. | `/api/v1/conversation-metrics/`, `/api/v1/topics/trending/`, `/api/v1/recommendations/` |
| **NLP practitioner** | Compare classification methods on real text and iterate. | `/api/v1/nlp/*`, `ws/nlp_playground/`, fine-tuning commands under `apps/llms/` |
| **Administrator** | Manage users, products, orders, and platform configuration. | Django admin, `/api/v1/users/`, `/api/v1/products/`, `/api/v1/orders/` |
| **Backend developer / intern** | Ship a real feature in days without learning every layer. | Django templates tier, management commands, REST ViewSets |

---

## 8. Twelve-Month North Star Outcomes

A successful backend, twelve months from the first production deploy, demonstrably delivers:

1. **Contract stability.** Every endpoint in `/api/v1/` has a passing schema test; no breaking changes have been merged without a version bump or a deprecation window.
2. **Test coverage** of at least 80 % on `apps/api/v1/`, `apps/support_agent/sa_utils/`, and `apps/playground/`, with integration tests for every WebSocket consumer.
3. **Observability** in place: structured logs from every consumer and Celery task, a single error sink, and per-conversation traces that can be replayed.
4. **Reproducible AI.** Any saved conversation can be re-run through the support agent or any of the three NLP methods and produce a comparable trace and metric.
5. **Multi-agent orchestration shipped.** Beyond the e-commerce support agent, at least two additional specialised agents (e.g. billing, general assistant) coexist behind a routing layer with shared tooling.
6. **RAG pipeline maturity.** Configurable chunking, hybrid (vector + keyword) retrieval, reranking, and citation tracking — exposed as configuration, not hardcoded.
7. **Operational data pipeline** in `data_processing/` that ingests real conversation exports, runs the NLP pipeline, and stores results without manual orchestration.
8. **Production-ready deployment** documented end-to-end: container images, environment matrix, migration playbook, backup posture, rollback procedure.
9. **Performance budget enforced.** P95 latency on read endpoints under 200 ms on seeded data; time-to-first-agent-token under 1.5 s on the support agent over a warm connection.
10. **Zero `print()` statements**, every diagnostic via `logging`, and structured logs at INFO and above shipped to a centralised store.

---

## 9. Success Metrics

| Family | Indicator | Target |
|---|---|---|
| **Contract** | Endpoints with OpenAPI schema entries | 100 % |
| **Contract** | Endpoints covered by API tests | ≥ 90 % |
| **Quality** | Statement coverage on `apps/api/v1/` | ≥ 80 % |
| **Quality** | Statement coverage on `apps/support_agent/sa_utils/` | ≥ 80 % |
| **Reliability** | WebSocket reconnect success rate (server-side) | > 99 % |
| **Reliability** | Celery task failure rate | < 1 % |
| **Performance** | p95 latency on `/api/v1/*` list endpoints (seeded data) | < 200 ms |
| **Performance** | Time to first agent token (support agent, warm) | < 1.5 s |
| **AI** | Replayability of saved conversations through the agent | 100 % |
| **AI** | Agreement rate between BERT, GPT, and RAG on a held-out set | Tracked, trend up |

These targets drive engineering decisions; they are not gates for early releases.

---

## 10. Architectural Posture

The backend is, and will remain:

- **A Django 6 + DRF monolith** organised as a layered set of self-contained apps under `apps/`.
- **ASGI-first.** Daphne serves both HTTP and WebSocket. WSGI is not part of the production path.
- **PostgreSQL + pgvector** as the single primary datastore. Redis is broker and cache only. We do not introduce a second OLTP database.
- **Celery for background work.** No ad-hoc threading inside request handlers. Long work is enqueued.
- **LangChain 1.3 + LangGraph 1.2** for agent orchestration. Direct OpenAI SDK calls are confined to thin client wrappers; agent logic lives in graph nodes.
- **Three intentional surfaces** (admin, staff templates, product cockpit), with a fixed remit per §6.
- **Single repository, no shared package with the frontend.** The contract is the boundary.

Anything that does not fit this posture requires a written design proposal before implementation.

---

## 11. What "Great" Looks Like

A new contributor clones the repo, runs `docker compose up -d postgres redis && make setup`, and within twenty minutes can:

- Open the Django admin and inspect every model.
- Open Swagger UI and hit every endpoint with a JWT obtained from the same UI.
- Open a Django-template page (login, signup, dashboard) and complete an auth flow.
- Open the Next.js cockpit and sign in as a seeded user.
- Connect to `ws/support_agent/` from a browser console, send a message, and watch the LangGraph agent stream a response with visible tool calls.
- Run `pytest` and see a green suite that exercises models, viewsets, consumers, and agent tools.
- Read one focused doc per app under `backend/docs/lessons/project_apps/` and understand what that app owns.

Every one of those moments feels fast, deliberate, and explainable. That is the bar.
