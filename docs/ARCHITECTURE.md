# ConvoInsight — Architecture (one picture)

> One diagram. One source of truth. Read this before you read any per-app guide.
>
> For the build-out plans, see [backend/docs/VISION.md](../backend/docs/VISION.md), [frontend/docs/VISION.md](../frontend/docs/VISION.md), and [frontend/docs/STRATEGY.md](../frontend/docs/STRATEGY.md). For the surface-tier rules, see [backend/docs/SURFACE_TIERS.md](../backend/docs/SURFACE_TIERS.md).

---

## 1. The Whole Platform in One Diagram

```mermaid
flowchart TB
    %% =========================
    %% Clients (Tiers)
    %% =========================
    subgraph Clients["Clients (3 Tiers, 3 Audiences)"]
        direction LR
        ADMIN["Tier 1 — Admin<br/>Django Admin<br/>/admin/"]
        STAFF["Tier 2 — Staff (server-rendered)<br/>Django Templates + Bootstrap 5<br/>/accounts/*, /dashboard/"]
        COCKPIT["Tier 3 — Product Cockpit<br/>Next.js 16 App Router + Tailwind<br/>:3000"]
    end

    %% =========================
    %% Contracts
    %% =========================
    subgraph Contracts["Public Contract Surface"]
        direction LR
        REST["REST API<br/>/api/v1/  (22 ViewSets)"]
        WS["WebSocket<br/>ws/chat, ws/support_agent,<br/>ws/order_support, ws/general_assistant,<br/>ws/nlp_playground"]
        OAS["OpenAPI 3 schema<br/>/api/docs/  Swagger + Redoc"]
    end

    %% =========================
    %% Server
    %% =========================
    subgraph Server["Django 6 + Daphne (ASGI)"]
        direction TB
        DRF["DRF ViewSets + Serializers<br/>(apps/api/v1/)"]
        CONS["Channels Consumers<br/>(apps/*/consumers.py)"]
        DJVIEWS["Django Views + Crispy Forms<br/>(apps/accounts, apps/dashboard)"]
    end

    %% =========================
    %% Infrastructure Layers
    %% =========================
    subgraph Infra["Five Reusable Infrastructure Layers"]
        direction TB
        L1["1) Web Data Capture<br/>data_processing/ • scrapers"]
        L2["2) Document Intelligence &amp; Vectorization<br/>spaCy • NLTK • SentenceTransformers • pgvector"]
        L3["3) Agentic Workflow Orchestration<br/>LangGraph 1.2 • support_agent/sa_utils/"]
        L4["4) Search, Answers &amp; RAG<br/>pgvector retrieval • OpenAI GPT-4o-mini • RAG playground"]
        L5["5) Deployment Layer<br/>Daphne • Celery • Redis • Postgres • Docker • CI"]
    end

    %% =========================
    %% Data + External
    %% =========================
    subgraph Data["Datastores"]
        PG[("PostgreSQL 17<br/>+ pgvector")]
        REDIS[("Redis 7<br/>broker + cache")]
        FILES[("ml_models/  media/<br/>fine-tuned weights")]
    end

    subgraph External["External AI / NLP services"]
        OPENAI["OpenAI<br/>GPT-4o-mini + embeddings"]
        HF["Hugging Face<br/>models + tokenizers"]
        TAVILY["Tavily<br/>web search tool"]
    end

    subgraph Async["Background work"]
        CELERY["Celery worker"]
        BEAT["Celery beat"]
    end

    %% =========================
    %% Flows
    %% =========================
    ADMIN -->|Django auth session| DJVIEWS
    STAFF -->|Django auth session| DJVIEWS
    COCKPIT -->|JWT Bearer| REST
    COCKPIT -->|JWT in ?token=| WS

    REST --> DRF
    WS --> CONS
    DRF --> Infra
    CONS --> Infra
    DJVIEWS --> Infra

    Infra --> Data
    L3 --> OPENAI
    L4 --> OPENAI
    L2 --> HF
    L3 --> TAVILY

    DRF --> CELERY
    CONS --> CELERY
    CELERY --> REDIS
    BEAT --> REDIS

    OAS -. describes .- REST

    classDef tier fill:#0f172a,stroke:#0f172a,color:#fff
    classDef contract fill:#1e293b,stroke:#1e293b,color:#fff
    classDef infra fill:#0e7490,stroke:#0e7490,color:#fff
    classDef data fill:#334155,stroke:#334155,color:#fff
    class ADMIN,STAFF,COCKPIT tier
    class REST,WS,OAS contract
    class L1,L2,L3,L4,L5 infra
    class PG,REDIS,FILES data
```

---

## 2. The Three Tiers (Clients)

Three audiences → three tiers. Each tier has a fixed remit and **a feature lives in exactly one tier**.

| Tier | Implementation | Audience | Owns |
|---|---|---|---|
| **1 — Admin** | Django admin (`/admin/`) | Engineers, data ops | Power-user data ops, debugging, seed inspection |
| **2 — Staff (server-rendered)** | Django templates + Bootstrap 5 + `crispy-forms` (`backend/templates/`) | Internal users, staff | Auth (login, signup, password reset, profile), internal staff dashboard. **Frozen scope** — no new routes are added here; the existing pages are edited in place only. |
| **3 — Product Cockpit** | Next.js 16 + Tailwind + Auth.js (`frontend/`) | End customers, operators | All new product surfaces: customer chat, operator live-queue, NLP playground comparison view, analytics cockpit. **All new user-visible features land here.** |

Full rules and examples: [backend/docs/SURFACE_TIERS.md](../backend/docs/SURFACE_TIERS.md).

---

## 3. The Contract Surface

All three tiers ultimately speak to the same Django process, but through different boundaries:

- **Tiers 1 & 2** call Django views directly (same process, server-rendered, session cookies).
- **Tier 3** speaks only through the public contracts: **REST under `/api/v1/`** and **WebSockets under `/ws/`**, both JWT-authenticated.

| Contract | What it carries |
|---|---|
| `/api/v1/*` (22 ViewSets) | Products, orders, conversations, messages, sentiments, topics, intents, analysis, NLP analysis, users, JWT auth |
| `ws/chat/<id>/` | Core conversation channel |
| `ws/support_agent/<id>/` | LangGraph e-commerce agent with tool calls |
| `ws/order_support/<id>/` | Order-scoped support |
| `ws/general_assistant/<id>/` | Multimodal (text + image + voice) assistant |
| `ws/nlp_playground/` | Streaming NLP playground (BERT • GPT • RAG) |
| `/api/docs/`, `/api/redoc/` | Live OpenAPI 3 schema |

OpenAPI is the source of truth for the REST contract. WebSocket message shapes are documented inside each consumer module.

---

## 4. The Five Infrastructure Layers

The same five layers power every feature. We do not rebuild them per product; we configure them per use case.

| # | Layer | What it does | Where it lives |
|---|---|---|---|
| 1 | **Web Data Capture** | Ingest content from web pages, files, and APIs | `backend/data_processing/`, scraper utilities |
| 2 | **Document Intelligence & Vectorization** | Tokenize, embed, and store text for retrieval | `apps/playground/text_classification_vector_store.py`, pgvector models |
| 3 | **Agentic Workflow Orchestration** | Multi-step LLM workflows with tool calling | `apps/support_agent/sa_utils/` (9 modules, LangGraph StateGraph) |
| 4 | **Search, Answers & RAG** | Retrieval + LLM answer composition | `apps/playground/text_classification_rag_processor.py`, GPT pipelines |
| 5 | **Deployment Layer** | Async server, background jobs, schedulers, DB, containers, CI | Daphne, Celery, Redis, Postgres, Docker, GitHub Actions |

Each app under `backend/apps/` is a configuration of one or more of these layers.

---

## 5. Request Lifecycles

### REST (cockpit → backend)

```
Browser → Next.js (cockpit) → fetch /api/v1/...  (Authorization: Bearer <JWT>)
                                   ↓
                Daphne → DRF router → ViewSet → Serializer → Model → PostgreSQL
                                   ↓
                          JSON response → cockpit → UI
```

### WebSocket (cockpit → agent)

```
Cockpit opens ws://.../ws/support_agent/<id>/?token=<JWT>
                       ↓
       Daphne → JWTAuthMiddleware (apps/api/ws_auth.py) → scope['user']
                       ↓
       SupportAgentConsumer → LangGraph StateGraph (call_model ↔ tools)
                       ↓ stream
          Tokens + tool-call events → cockpit → UI
```

### Server-rendered (staff → backend)

```
Browser → Daphne → Django view → Template → HTML → Browser
                       ↑
              Django auth session cookie
```

### Background work

```
Anything slow → enqueue Celery task on Redis → Celery worker picks up → writes to Postgres
Scheduled jobs → Celery beat → Redis → Celery worker
```

---

## 6. Authentication Posture

| Surface | Mechanism | Notes |
|---|---|---|
| Django admin (Tier 1) | Django session | Default Django auth |
| Staff templates (Tier 2) | Django session + `allauth` | Login, signup, password reset, profile, social login |
| Cockpit REST (Tier 3) | JWT (access + refresh + blacklist) | Issued by `/api/v1/auth/token/`. Held by **Auth.js v5** on the cockpit in an encrypted session cookie (replaces the legacy `localStorage` flow). |
| Cockpit WS (Tier 3) | Same JWT via `?token=...` | Validated by `JWTAuthMiddleware` in `apps/api/ws_auth.py` |

Django is the **canonical identity store** for all tiers. Auth.js never owns identity; it owns the cockpit's session.

---

## 7. Tech Stack Snapshot (today)

| Layer | Technology | Version |
|---|---|---|
| Backend framework | Django | 6.0.5 |
| API | DRF + drf-spectacular | 3.17 / 0.29 |
| Async server | Daphne + Channels | 4.2 / 4.3 |
| Database | PostgreSQL + pgvector | 17 |
| Queue / cache | Celery + Redis | 5.6 / 7 |
| LLM orchestration | LangChain + LangGraph | 1.3 / 1.2 |
| ML / NLP | PyTorch + Transformers + spaCy + NLTK + BERTopic | current |
| Cockpit | Next.js + React | 16.2.6 / 19.2.6 |
| Cockpit auth | Auth.js (next-auth v5) | 5.0.0-beta.31 |
| Styling | TailwindCSS | 3.4 |
| Containers | Docker multi-stage + Compose | — |
| CI | GitHub Actions | lint → test → build + security |

---

## 8. Where to Go Next

| If you want to … | Read |
|---|---|
| Build a Day-1 mental map of every app | [docs/MENTAL_MODEL.md](./MENTAL_MODEL.md) |
| Understand why we ship three tiers | [backend/docs/SURFACE_TIERS.md](../backend/docs/SURFACE_TIERS.md) |
| Deep-dive the backend | [backend/docs/BACKEND_GUIDE.md](../backend/docs/BACKEND_GUIDE.md) |
| Deep-dive the cockpit | [frontend/docs/FRONTEND_GUIDE.md](../frontend/docs/FRONTEND_GUIDE.md) |
| Plan a contribution | [backend/docs/VISION.md](../backend/docs/VISION.md), [frontend/docs/STRATEGY.md](../frontend/docs/STRATEGY.md) |
| Implement cockpit auth with Auth.js | [frontend/docs/AUTHJS_INTEGRATION.md](../frontend/docs/AUTHJS_INTEGRATION.md) |
| Onboard as an intern | [docs/INTERN_ONBOARDING.md](./INTERN_ONBOARDING.md) |
