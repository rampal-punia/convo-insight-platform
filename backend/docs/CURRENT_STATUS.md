# ConvoInsight Backend — Current Status

> A factual snapshot of the backend as it stands today. This is the reference point that the [Vision](./VISION.md) and the README roadmap build from. Update this document in the same PR that changes structural facts.

---

## 1. At a Glance

| Area | State |
|---|---|
| Framework | Django 6.0.5 |
| API | Django REST Framework 3.17.1 + drf-spectacular 0.29 |
| Async server | Daphne 4.2 (ASGI) + Django Channels 4.3 |
| Auth | `simplejwt` (access + refresh + blacklist) + `allauth` |
| Database | PostgreSQL 17 with `pgvector` extension |
| Task queue | Celery 5.6 on Redis 7 |
| LLM stack | LangChain 1.3 + LangGraph 1.2, OpenAI GPT-4o-mini |
| ML stack | PyTorch 2.12 + Transformers 5.9 + BERTopic 0.17 + SentenceTransformers |
| NLP libs | spaCy 3.8 (`en_core_web_sm`) + NLTK 3.9 |
| API surface | 22 ViewSets across 6 resource files, 18 custom actions |
| WebSocket consumers | 5 consumers across 5 apps |
| Templates tier | Bootstrap 5 + crispy-forms (auth pages + dashboard) |
| Test suite | 9 smoke tests in a single file (under `apps/api/v1/tests/`) |
| Lint | `ruff` (config in `pyproject.toml`) |
| CI | GitHub Actions: lint → test → build + security scan |
| Containers | Multi-stage Dockerfile + Docker Compose (Postgres, Redis, Daphne, Celery, Beat) |

---

## 2. Repository Layout (`backend/`)

```
backend/
├── config/                         # Django project config
│   ├── settings/                   # base, development, production, test
│   ├── asgi.py                     # ASGI app composing 5 routing modules
│   ├── celery.py                   # Celery app + autodiscovery
│   ├── urls.py                     # Root URL conf
│   └── models.py                   # CreationModificationDateBase abstract model
├── apps/                           # Domain apps (sys.path-inserted)
│   ├── accounts/                   # Custom User + auth views + crispy forms
│   ├── products/                   # Category + Product
│   ├── orders/                     # Order + OrderItem + OrderTracking + WS consumer
│   ├── convochat/                  # Conversation, Message, UserText, AIText, Sentiment, Topic, Intent + WS consumer
│   ├── analysis/                   # Performance, metrics, recommendations, predictions
│   ├── api/v1/                     # Versioned API: 22 ViewSets + JWT routes + ws_auth middleware
│   ├── dashboard/                  # Dashboard view + seed_demo command
│   ├── playground/                 # NLP playground: BERT, GPT, RAG + WS consumer + management command
│   ├── support_agent/              # LangGraph agent (9-module sa_utils/) + WS consumer + ConversationSnapshot
│   ├── general_assistant/          # Multimodal assistant (text + image + voice) + WS consumer
│   └── llms/                       # Fine-tuning + SageMaker + 3 management commands
├── templates/                      # Live Django templates (base.html, index.html, accounts/, dashboard/)
├── templates_new/                  # In-progress restyle of the templates tier (parallel, not yet adopted)
├── static/                         # CSS, JS, images
├── media/                          # Uploaded media (gitignored content)
├── data_processing/                # Ingestion scripts (currently stubs, see §10)
├── ml_models/                      # Trained model files (gitignored)
├── logs/
├── scripts/                        # init-db.sql
├── docs/                           # Backend docs (this file lives here)
│   ├── BACKEND_GUIDE.md
│   ├── VISION.md
│   ├── CURRENT_STATUS.md
│   ├── langgraph_workflow_vs_agents.md
│   └── lessons/                    # Tutorial-style deep-dives, per app
├── manage.py
├── conftest.py
├── pyproject.toml
├── requirements.txt
└── docker-entrypoint.sh
```

### The `apps/` import trick

`config/settings/base.py` inserts `BASE_DIR / "apps"` into `sys.path`. Sibling apps import as `from products.models import Product`, never `from apps.products.models import Product`. This convention is load-bearing — do not work around it.

---

## 3. API Surface

The API lives under `/api/v1/`. Routing uses DRF's `DefaultRouter`. Interactive documentation is at:

- Swagger UI — `/api/docs/`
- Redoc — `/api/redoc/`
- Raw schema — `python manage.py spectacular --file schema.yml`

### ViewSet inventory (22 total)

| File | ViewSets | Custom actions |
|---|---|---|
| `views_products.py` | `CategoryViewSet`, `ProductViewSet` | `products`, `in_stock`, `low_stock`, `featured` |
| `views_orders.py` | `OrderViewSet`, `OrderItemViewSet`, `OrderTrackingViewSet` | `tracking`, `add_tracking`, `items`, `cancel`, `mark_shipped`, `update_status` |
| `views_conversations.py` | `ConversationViewSet`, `MessageViewSet`, `UserTextViewSet`, `AITextViewSet`, `IntentViewSet`, `TopicViewSet`, `SentimentViewSet`, `SentimentCategoryViewSet`, `GranularEmotionViewSet` | `trending`, `messages`, `archive`, `end` |
| `views_analysis.py` | `LLMAgentPerformanceViewSet`, `ConversationMetricsViewSet`, `RecommendationViewSet`, `TopicDistributionViewSet`, `IntentPredictionViewSet` | (read-only) |
| `views_accounts.py` | `UserViewSet` | `me` |
| `views_nlp.py` | `NLPAnalysisViewSet` | `analyze_sentiment`, `analyze_intent`, `analyze_topic`, `analyze_ner` |

### Authentication

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/auth/token/` | Obtain access + refresh |
| `POST /api/v1/auth/token/refresh/` | Refresh access |
| `POST /api/v1/auth/token/verify/` | Verify a token |
| `POST /api/v1/auth/token/blacklist/` | Blacklist a refresh token |

WebSocket authentication uses the same JWT, passed as a `?token=...` query parameter and validated by `apps/api/ws_auth.py`.

### Pagination, filtering, ordering

- All list endpoints use `StandardResultsSetPagination`.
- `django-filter`, `SearchFilter`, and `OrderingFilter` are enabled per ViewSet.

---

## 4. WebSocket Consumers

`config/asgi.py` composes routing modules from five apps:

| Route pattern | Consumer | Purpose |
|---|---|---|
| `ws/chat/<conversation_id>/` | `convochat.ChatConsumer` | Core conversation channel |
| `ws/support_agent/<conversation_id>/` | `support_agent.SupportAgentConsumer` | LangGraph e-commerce agent with tool calls |
| `ws/order_support/<conversation_id>/` | `orders.OrderSupportConsumer` | Order-scoped support consumer |
| `ws/general_assistant/<conversation_id>/` | `general_assistant.GeneralChatConsumer` | Multimodal (text + image + voice) assistant |
| `ws/nlp_playground/` | `playground.NLPPlaygroundConsumer` | Streaming NLP playground (BERT, GPT, RAG) |

Each consumer authenticates via the shared `JWTAuthMiddleware` in `apps/api/ws_auth.py`.

> Note: the live route patterns under `apps/*/routing.py` use underscores (`ws/support_agent/...`). Some older docs reference hyphenated forms (`ws/support-agent/...`). The routing files are authoritative.

---

## 5. Data Model

### Domain models (Django)

- **Accounts** — custom `User` model.
- **Products** — `Category`, `Product`.
- **Orders** — `Order`, `OrderItem`, `OrderTracking`, `OrderConversationLink`.
- **Convochat** — `Conversation` (UUID pk), `Message`, `UserText`, `AIText`, `Intent`, `Topic`, `SentimentCategory`, `GranularEmotion`, `Sentiment`.
- **Analysis** — `LLMAgentPerformance`, `ConversationMetrics`, `Recommendation`, `TopicDistribution`, `IntentPrediction`.
- **Support agent** — `ConversationSnapshot` (full state checkpoints).
- **General assistant** — `GeneralConversation`, `GeneralMessage`, `AudioMessage`, `ImageMessage`.

Most models inherit `CreationModificationDateBase` from `config/models.py`.

### Vector store

`RAGTextClassificationDocument` uses `pgvector` with 384-dimensional embeddings and an IVFFlat index for cosine similarity. It backs the RAG method in the playground.

A full ERD lives in [BACKEND_GUIDE.md](./BACKEND_GUIDE.md#database-schema).

---

## 6. Support Agent (LangGraph)

Location: `apps/support_agent/sa_utils/`. Nine modules:

- `state.py` — `InputState`, `ECommerceState`
- `configuration.py` — runtime config + Pydantic schemas
- `graph_builder.py` — `StateGraph` wiring, `call_model` and `tools` nodes
- `tool_manager.py` — tool definitions (`web_search`, order tools)
- `intent_router.py` — hybrid rule + LLM intent detection
- `prompt_manager.py` — system prompts keyed by intent
- `context_manager.py` — conversation context construction
- `flow_manager.py` — greeting / support / closing flow control
- `helper.py` — model loading helpers

Snapshots of full state are written to `ConversationSnapshot` (`AU`/`MN`/`EV`/`FN` types).

---

## 7. NLP Playground

Location: `apps/playground/`. Three methods × four tasks:

| Method | Backed by |
|---|---|
| Fine-tuned BERT | Local RoBERTa sentiment model + fine-tuned intent BERT + BERTopic |
| Few-shot GPT | OpenAI GPT-4o-mini via LangChain |
| RAG (pgvector) | SentenceTransformer embeddings → `RAGTextClassificationDocument` retrieval |

| Task | Endpoints |
|---|---|
| Sentiment | `POST /api/v1/nlp/sentiment/` + `ws/nlp_playground/` |
| Intent | `POST /api/v1/nlp/intent/` + `ws/nlp_playground/` |
| Topic | `POST /api/v1/nlp/topic/` + `ws/nlp_playground/` |
| NER | `POST /api/v1/nlp/ner/` + `ws/nlp_playground/` |

Seed commands (run after `migrate`):

- `python manage.py create_intents` — 11 intent labels.
- `python manage.py create_topics` — 12 topic categories.
- `python manage.py populate_rag_store` — embeds intents and topics into pgvector.

Model files for BERT and BERTopic methods are **not** in git; they are downloaded out-of-band (see the intern onboarding guide). Without them, the BERT and BERTopic tabs fail; the GPT and RAG tabs continue to work.

---

## 8. Templates Tier

The server-rendered tier covers auth and an internal dashboard. It is a first-class surface, not legacy.

| Route | Template | Source |
|---|---|---|
| `/accounts/login/` | `accounts/login.html` | `apps/accounts/views.py::CustomLoginView` |
| `/accounts/signup/` | `accounts/signup.html` | `apps/accounts/views.py::CustomSignupView` |
| `/accounts/password-reset/...` | `accounts/confirm_passwordreset.html` | `apps/accounts/views.py` |
| `/accounts/profile/` | `accounts/profile.html` | `apps/accounts/views.py` |
| `/dashboard/` | `dashboard/dashboard.html` | `apps/dashboard/views.py` |
| `/` | `index.html` | `config/urls.py` |

Styling uses Bootstrap 5 via CDN; forms use `django-crispy-forms` + `crispy-bootstrap5`. A parallel `templates_new/` directory holds in-progress restyle drafts — it is not yet wired into the URL conf.

---

## 9. Management Commands

| App | Command | Purpose |
|---|---|---|
| `dashboard` | `seed_demo` | Create categories, products, demo users, orders |
| `convochat` | `create_intents` | Load 11 intent labels |
| `convochat` | `create_topics` | Load 12 topic categories |
| `playground` | `populate_rag_store` | Embed seed labels into pgvector |
| `orders` | `generate_dummy_data` | Fixture-style order data |
| `accounts` | `create_random_users` | Seed users |
| `llms` | `fine_tune_llm` | Kick off a fine-tune job |
| `llms` | `train_deploy_model` | Train + deploy (SageMaker path) |
| `llms` | `monitor_model` | Monitor a deployed model |

---

## 10. Known Gaps

These are the items most likely to bite a new contributor. They are also the highest-leverage cleanup targets.

1. **Test coverage is thin.** Nine smoke tests cover a slice of the API; `support_agent`, `playground`, `orders`, `convochat`, `analysis`, and `llms` have no dedicated test suites.
2. **`print()` statements in production code.** Diagnostic calls should flow through `logging`; a sweep is outstanding.
3. **`data_processing/` is stubbed.** The three preprocess functions are placeholders and the example script's call signature does not match. The pipeline needs to be implemented end-to-end.
4. **`templates_new/` is parked.** It contains restyle drafts but is not referenced from any view. Either adopt or remove during the next templates pass.
5. **Tool registration gap in the agent.** Some tools are defined but not registered in the `TOOLS` list consumed by the LangGraph node — a check-and-fix task.
6. **`ECommerceState` defaults.** Several `Dict` fields default to `False` instead of `field(default_factory=dict)`. Safe today because of guarded reads; should be corrected.
7. **Dead code in the playground consumer.** A `process_request()` method duplicates `receive()` and is unreachable.
8. **Voice on the general assistant uses Google SpeechRecognition.** A migration to Whisper is on the roadmap.
9. **Single-agent only.** The support agent works; the routing layer for additional specialised agents (billing, tech support, etc.) is not yet implemented.
10. **Hyphen vs underscore in WS route docs.** Older guide text references `ws/support-agent/`; live routes use `ws/support_agent/`. Treat `apps/*/routing.py` as authoritative.

These gaps are tracked items, not blockers — every one of them is a labelled intern task in the roadmap.

---

## 11. Tooling & Operations

| Concern | Status |
|---|---|
| Lint | `ruff` configured in `pyproject.toml` (line length 120, single quotes) |
| Tests | `pytest` + `pytest-django`, configured in `pyproject.toml`, jsdom-free |
| Coverage | `pytest-cov` configured, target 70 %, not yet enforced as a gate |
| Pre-commit | Not installed |
| CI | GitHub Actions: lint → test → build + security scan |
| Container | Multi-stage Dockerfile (builder + runtime), non-root user, healthcheck |
| Compose | Postgres, Redis, Daphne, Celery worker, Celery beat |
| Migrations | Forward-only; `migrate` enables the `vector` extension automatically |

---

## 12. Environment & Configuration

Loaded via `django-environ` in `config/settings/base.py`. The repository ships `.env.example` at the root with all variables documented.

Mandatory for full functionality:

- `OPENAI_API_KEY` — GPT-based NLP + LangGraph agent.
- `HUGGINGFACEHUB_API_TOKEN` — Hugging Face model access.
- `TAVILY_API_KEY` — web search tool in the support agent.

Settings are split into `base.py`, `development.py`, `production.py`, `test.py`. `DJANGO_SETTINGS_MODULE` defaults to development locally and to production in the container image.

---

## 13. What Works End-to-End Today

A new contributor, after `docker compose up -d postgres redis && pip install -r requirements.txt && python manage.py migrate && python manage.py seed_demo && python manage.py create_intents && python manage.py create_topics && python manage.py populate_rag_store && python manage.py runserver`, can:

1. Open Swagger UI at `/api/docs/`, obtain a JWT from `/api/v1/auth/token/`, and hit every read endpoint with `Authorize`.
2. Open Django admin at `/admin/` (`demo_admin` / `demo12345`) and browse every model.
3. Sign in and use the Django-template dashboard at `/dashboard/`.
4. Connect to `ws://localhost:8000/ws/support_agent/<id>/?token=...` from a browser console and exchange messages with the LangGraph agent.
5. Run the playground NLP analyses over REST (`/api/v1/nlp/*`) and (for GPT and RAG methods) without any downloaded ML model files.
6. Run `pytest` and see the smoke suite pass.

That is the verified surface today. Everything beyond is on the roadmap.

---

_Last updated: keep this in sync with reality. If you change something structural, update the same PR that changes the code._
