# Intern Onboarding — ConvoInsight Platform

Welcome to ConvoInsight! This guide will get you from zero to shipping code in 4 weeks.
Read it in order. Ask questions whenever something doesn't make sense — that's how you learn.

---

## Day 1: Get It Running

### What you'll need

- Python 3.11+ installed (`python3 --version` to check)
- Node.js 18+ installed (`node --version` to check)
- Docker Desktop installed and running (`docker --version` to check)
- A code editor (VS Code recommended)
- A terminal

### Step 1: Clone the repo

```bash
git clone https://github.com/rampal-punia/convo-insight-platform.git
cd convo-insight-platform
```

### Step 2: Create a virtual environment

This keeps project dependencies isolated from your system Python.

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt. If you open a new terminal, you'll need to run `source venv/bin/activate` again from the project root.

### Step 3: Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

This takes a minute or two (there are ML packages including PyTorch). Go grab water.

### Step 3b: Download NLP models

The project uses spaCy and NLTK for text processing. Download the required models:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

> **If you skip this**, the server will crash with `Can't find model 'en_core_web_sm'`.

### Step 4: Set up environment variables

```bash
cd ..   # back to repo root
cp .env.example .env
```

Open `.env` in your editor. You **must** fill in real API keys for the AI features to work:
- `OPENAI_API_KEY` — get one at https://platform.openai.com/api-keys
- `HUGGINGFACEHUB_API_TOKEN` — get one at https://huggingface.co/settings/tokens
- `TAVILY_API_KEY` — get one at https://tavily.com (used by the support agent for web search)

> **If you don't have API keys yet**, the server will still start — you just won't be able to use the AI/NLP features until you add them.

### Step 5: Start the database and cache

```bash
# Make sure Docker Desktop is running first!
docker compose up -d postgres redis
```

Verify they're healthy:

```bash
docker compose ps
```

You should see both `convoinsight-postgres` and `convoinsight-redis` showing `healthy` (it might take 10-20 seconds).

### Step 6: Set up the database

```bash
cd backend
python manage.py migrate
```

You'll see a bunch of `Applying <something>... OK` lines. That's Django creating tables.

Now load sample data:

```bash
python manage.py seed_demo
```

This creates demo products, users, and orders so you have something to work with.

### Step 7: Start the backend server

```bash
python manage.py runserver
```

Open your browser:
- **Swagger UI** (API docs): http://localhost:8000/api/docs/
- **Django Admin**: http://localhost:8000/admin/
- **API root**: http://localhost:8000/api/v1/

**Admin login**: username `demo_admin`, password `demo12345`

> To stop the server, press `Ctrl+C` in the terminal.

### Step 8: Start the frontend (new terminal)

Open a **new terminal tab** and run:

```bash
cd convo-insight-platform/frontend
npm install
npm run dev
```

Open http://localhost:3000 — you should see the ConvoInsight landing page.

Log in with `demo_user_01` / `demo12345`, then visit http://localhost:3000/products to see the product listing.

### You did it! The full stack is running.

Take a screenshot. You earned it. Now let's understand what you just set up.

---

## Day 2-3: Understand the Codebase

### How the project is organised

```
convo-insight-platform/
├── backend/              ← All Django/Python code lives here
│   ├── config/           ← Django project settings, URLs, ASGI/WSGI config
│   │   └── settings/
│   │       ├── base.py   ← Main settings (DB, apps, DRF, JWT, logging)
│   │       ├── development.py  ← Local dev overrides
│   │       ├── production.py   ← Production overrides
│   │       └── test.py         ← Test overrides
│   ├── apps/             ← Each Django "app" is a self-contained module
│   │   ├── accounts/     ← User management
│   │   ├── products/     ← Product catalogue + categories
│   │   ├── orders/       ← Orders, order items, tracking
│   │   ├── convochat/    ← Conversations, messages, NLP utility functions
│   │   ├── analysis/     ← Metrics, agent performance, recommendations
│   │   ├── api/          ← The REST API layer
│   │   │   └── v1/       ← Versioned API (this is what the frontend calls)
│   │   ├── dashboard/    ← Dashboard views + the seed_demo command
│   │   ├── playground/   ← NLP playground (3 methods: BERT, GPT, RAG)
│   │   ├── support_agent/← LangGraph e-commerce support agent
│   │   ├── general_assistant/ ← General AI assistant
│   │   └── llms/         ← LLM fine-tuning + SageMaker
│   └── manage.py         ← Django management command entry point
├── frontend/             ← Next.js 15 (JSX only, no TypeScript)
│   └── src/
│       ├── app/          ← Pages (each folder = a route)
│       └── lib/          ← Shared utilities (API client)
└── docs/                 ← You are here
```

### The import trick

Open `backend/config/settings/base.py` and look at line 9:

```python
sys.path.insert(0, str(BASE_DIR / "apps"))
```

This adds the `apps/` directory to Python's module search path. That's why imports look like `from convochat.models import Conversation` instead of `from apps.convochat.models import Conversation`. When you create a new app, you use the short name.

### Django Templates Overview (for backend interns)

The project has both an API layer (DRF) **and** a server-rendered HTML layer (Django templates). While we're porting to Next.js, the Django templates are still used for auth pages and the dashboard.

**Template directory structure:**

```
backend/
├── templates/            ← Global templates (TEMPLATES DIRS in settings)
│   ├── base.html         ← Base layout (navbar, Bootstrap 5, messages)
│   ├── accounts/
│   │   ├── login.html       ← Custom login form (uses crispy-forms)
│   │   ├── signup.html      ← Custom signup form
│   │   ├── confirm_passwordreset.html  ← Password reset form
│   │   └── profile.html     ← Profile edit form
│   └── dashboard/
│       └── dashboard.html   ← Main dashboard (card grid with quick links)
└── static/               ← Static files (CSS, JS, images)
```

**How it fits together:**

1. **`config/settings/base.py`** sets `TEMPLATES[0]['DIRS']` to `BASE_DIR / "templates"` — so Django looks in `backend/templates/` first, then in each app's `templates/` directory (because `APP_DIRS = True`).

2. **`apps/accounts/views.py`** defines `CustomLoginView`, `CustomSignupView`, etc. Each specifies a `template_name` like `'accounts/login.html'` and a `form_class` using crispy-forms with Bootstrap 5.

3. **`apps/accounts/forms.py`** defines the forms with `FormHelper` and `Layout` from crispy-forms — this controls field ordering, placeholders, and submit buttons without writing HTML manually.

4. **`base.html`** uses Bootstrap 5 (via CDN), shows the navbar only when authenticated, and renders Django messages as alert banners.

**Key files to read in order:**

| File | What it does |
|------|-------------|
| `backend/templates/base.html` | Base layout — every other template extends this |
| `backend/apps/accounts/views.py` | Auth views (login, signup, profile) |
| `backend/apps/accounts/forms.py` | Crispy-forms definitions with Bootstrap 5 styling |
| `backend/config/urls.py` | URL routing — maps paths to views |
| `backend/apps/dashboard/views.py` | Dashboard view — renders `dashboard/dashboard.html` |

**To add a new template page:**

1. Create the view in the appropriate app's `views.py`
2. Add a URL path in `urls.py`
3. Create the template in `backend/templates/<appname>/` extending `base.html`
4. Use `{% load crispy_forms_tags %}` and `{% crispy form %}` for forms

### How a request flows

Let's trace `GET /api/v1/products/`:

1. **`config/urls.py`** — includes `api.urls`
2. **`apps/api/urls.py`** — includes `api.v1.urls` under `/api/v1/`
3. **`apps/api/v1/urls.py`** — registers `ProductViewSet` with the router at `products`
4. **`apps/api/v1/views_products.py`** — `ProductViewSet` handles the request
5. **`apps/api/v1/serializers_products.py`** — `ProductSerializer` converts models to JSON
6. **`apps/products/models.py`** — `Product` model fetches from PostgreSQL

Try it yourself:

```bash
cd backend
python manage.py shell
```

```python
from products.models import Product
Product.objects.count()        # should be 10 (from seed_demo)
Product.objects.first().name   # see the first product
exit()
```

### Explore the API

With the server running (`python manage.py runserver`), open http://localhost:8000/api/docs/ in your browser. This is **Swagger UI** — an interactive API explorer.

Try these:
1. Click **POST /api/v1/auth/token/** → enter `demo_user_01` / `demo12345` → click Execute → copy the `access` token
2. Click the **Authorize** button (top right) → enter `Bearer <paste-your-token>` → now all endpoints are authenticated
3. Click **GET /api/v1/products/** → Execute → see the product list
4. Click **GET /api/v1/orders/** → Execute → see the demo orders

### Explore the Django admin

Open http://localhost:8000/admin/ and log in as `demo_admin` / `demo12345`.

Walk through every model:
- **Products** → see the catalogue
- **Orders** → see order items and tracking
- **Conversations** → see messages, intents, topics, sentiments

This is your data model. Understanding these tables is understanding the domain.

### Run the tests

```bash
cd backend
pytest -v
```

All 12 smoke tests should pass. If they don't, ask for help — something in your setup is off.

---

## Week 1: Orientation Checklist

- [ ] Full stack runs locally (backend + frontend + Postgres + Redis)
- [ ] Can log in to Django admin and browse all models
- [ ] Can authenticate via Swagger UI and hit API endpoints
- [ ] Can trace a request from URL → viewset → serializer → model
- [ ] Tests pass locally (`cd backend && pytest -v`)
- [ ] Read `README.md` and `CONTRIBUTING.md`

### Concepts to understand by end of week 1

| Concept | Where to learn it |
|---------|-------------------|
| Django apps and models | `apps/products/models.py` — clean, simple example |
| DRF ViewSets + Serializers | `apps/api/v1/views_products.py` + `serializers_products.py` |
| DRF Router (auto-URL generation) | `apps/api/v1/urls.py` — see how `DefaultRouter` works |
| JWT auth flow | `apps/api/v1/urls_auth.py` + settings in `config/settings/base.py` |
| The `@action` decorator | `apps/api/v1/views_products.py` — see `in_stock`, `low_stock` actions |
| Django settings split | `config/settings/base.py` → `development.py` → `test.py` |

---

## Week 2: Ship Your First Feature

Pick **one** of these warm-up tasks (they're labelled `good-first-issue` on GitHub):

### Option A: Featured products endpoint

Add `GET /api/v1/products/featured/` that returns products with `stock > 50`.

**What you'll learn**: ViewSet actions, queryset filtering, API design.

```bash
# 1. Write the action in the ProductViewSet
#    File: backend/apps/api/v1/views_products.py
#    Add an @action(detail=False) method that filters Product.objects.filter(stock__gt=50)

# 2. Verify it works
cd backend
python manage.py runserver
# Visit http://localhost:8000/api/v1/products/featured/ in your browser

# 3. Write a test
#    File: backend/apps/api/v1/tests/test_smoke.py
#    Add a test function that hits the endpoint and checks the response

# 4. Run tests
pytest apps/api/v1/tests/test_smoke.py -v

# 5. Check the OpenAPI schema is clean
python manage.py spectacular --file /tmp/schema.yml
# Should not warn about your new endpoint
```

### Option B: Order refund action

Add `POST /api/v1/orders/{id}/refund/` that changes order status to `REFUNDED`.

**What you'll learn**: Detail actions, model state changes, permissions.

```bash
# 1. Check the Order model status choices
#    File: backend/apps/orders/models.py — look at the STATUS_CHOICES

# 2. Add @action(detail=True, methods=['post']) to OrderViewSet
#    File: backend/apps/api/v1/views_orders.py

# 3. Test it
cd backend
python manage.py runserver
# Use Swagger UI to POST to /api/v1/orders/{some_id}/refund/

# 4. Write a test
pytest apps/api/v1/tests/test_smoke.py -v

# 5. Verify schema
python manage.py spectacular --file /tmp/schema.yml
```

### Option C: My orders endpoint

Add `GET /api/v1/users/me/orders/` that returns the current user's orders.

**What you'll learn**: User-scoped queries, nested routes, authentication.

```bash
# 1. Add a @action on the UserViewSet or OrderViewSet
#    You'll need to filter orders by request.user

# 2. Test it via Swagger UI
cd backend
python manage.py runserver

# 3. Write tests (authenticated + unauthenticated cases)
pytest apps/api/v1/tests/test_smoke.py -v
```

### After you pick one

```bash
# Run all quality checks before committing
cd backend
ruff check . --fix       # Auto-fix lint issues
ruff format .            # Format code
pytest -v                # All tests pass

# Create a branch and commit
git checkout -b feature/my-first-endpoint
git add backend/apps/api/v1/
git commit -m "Add featured products endpoint"
git push origin feature/my-first-endpoint
# Then open a PR against development
```

---

## Week 3: WebSockets and Background Tasks

### WebSockets (real-time chat)

The platform uses Django Channels for WebSocket communication. The support agent streams responses token-by-token over WebSockets.

**Read these files in order:**
1. `backend/config/asgi.py` — how WebSockets are routed
2. `backend/apps/convochat/routing.py` — URL patterns for WS connections
3. `backend/apps/convochat/consumers.py` — handles incoming/outgoing messages
4. `backend/apps/api/ws_auth.py` — JWT authentication for WebSocket connections

**Try it live:**

```bash
# Start the server
cd backend
python manage.py runserver
```

Open your browser console (F12 → Console) and run:

```javascript
// First, get a JWT token
fetch('http://localhost:8000/api/v1/auth/token/', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({username: 'demo_user_01', password: 'demo12345'})
})
.then(r => r.json())
.then(data => {
  const token = data.access;

  // Connect to the support agent WebSocket
  const ws = new WebSocket(`ws://localhost:8000/ws/support-agent/?token=${token}`);

  ws.onopen = () => {
    console.log('Connected!');
    ws.send(JSON.stringify({
      type: 'chat_message',
      message: 'Where is my order #ORD-001?'
    }));
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    console.log('Received:', data);
  };

  ws.onerror = (e) => console.log('Error:', e);
  ws.onclose = (e) => console.log('Closed:', e.code, e.reason);
});
```

You should see the agent respond with streaming tokens. This is the LangGraph support agent processing your message, deciding to use the "track order" tool, and responding.

### Celery (background tasks)

Celery handles long-running tasks like NLP analysis and model training.

**Read these files:**
1. `backend/config/celery.py` — Celery app configuration
2. `backend/apps/orders/tasks.py` — example background tasks

**Try it:**

```bash
# Terminal 1: Start the server
cd backend && python manage.py runserver

# Terminal 2: Start a Celery worker
cd backend && celery -A config worker -l info

# Terminal 3: Trigger a task from Django shell
cd backend && python manage.py shell
```

```python
from orders.tasks import some_task_name  # pick any task
result = some_task_name.delay()          # .delay() sends it to Celery
result.id                                # see the task ID
result.status                            # check status
```

**Optional: Celery Beat (scheduled tasks)**

```bash
# In another terminal
cd backend && celery -A config beat -l info
```

---

## Week 4: Connect Frontend to Your Backend Feature

Now you'll build a UI page that uses the endpoint you built in Week 2.

### Set up

```bash
cd frontend
npm run dev
# → http://localhost:3000
```

**Read these files first:**
1. `frontend/src/lib/api.js` — the API client (JWT attach, auto-refresh on 401)
2. `frontend/src/app/login/page.jsx` — how login works
3. `frontend/src/app/products/page.jsx` — example of fetching and displaying data

### Build your page

If you built the featured products endpoint:

```bash
# Create the page
mkdir -p frontend/src/app/featured
```

Create `frontend/src/app/featured/page.jsx`:

```jsx
'use client';

import { useState, useEffect } from 'react';
import api from '@/lib/api';

export default function FeaturedProducts() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api('/api/v1/products/featured/')
      .then(data => setProducts(data.results || data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Featured Products</h1>
      <div className="grid grid-cols-3 gap-4">
        {products.map(p => (
          <div key={p.id} className="border rounded p-4">
            <h2 className="font-semibold">{p.name}</h2>
            <p className="text-gray-600">${p.price}</p>
            <p className="text-sm">Stock: {p.stock}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

Visit http://localhost:3000/featured to see it.

**Remember**: this project is **JSX only, no TypeScript**. Use `.jsx` files, not `.tsx`.

---

## Day-to-Day Commands

All Django/Python commands run from `backend/` with your venv activated.

```bash
# Always start here
cd convo-insight-platform
source venv/bin/activate
cd backend
```

### Server & Services

| What | Command |
|------|---------|
| Start infrastructure | `cd .. && docker compose up -d postgres redis` |
| Check services are running | `docker compose ps` |
| Stop everything | `docker compose down` |
| Full reset (deletes DB data) | `docker compose down -v` |
| Run Django dev server | `python manage.py runserver` |
| Download NLP models | `make download-nlp-data` (from repo root) |
| Start Celery worker | `celery -A config worker -l info` |
| Start Celery beat | `celery -A config beat -l info` |
| Start frontend (new terminal) | `cd ../frontend && npm run dev` |

### Database

| What | Command |
|------|---------|
| Run migrations | `python manage.py migrate` |
| Create new migrations | `python manage.py makemigrations` |
| Then apply them | `python manage.py migrate` |
| Seed demo data | `python manage.py seed_demo` |
| Reset and reseed | `python manage.py seed_demo --reset` |
| Create more users | `python manage.py create_random_users 20` |
| Generate dummy orders | `python manage.py generate_dummy_data` |
| Django shell | `python manage.py shell` |

### Testing & Quality

| What | Command |
|------|---------|
| Run all tests | `pytest` |
| Run with verbose output | `pytest -v` |
| Run specific file | `pytest apps/api/v1/tests/test_smoke.py -v` |
| Run with coverage | `pytest --cov=apps --cov-report=term-missing` |
| Lint + auto-fix | `ruff check . --fix` |
| Format code | `ruff format .` |
| Check without fixing | `ruff check .` |
| Generate OpenAPI schema | `python manage.py spectacular --file /tmp/schema.yml` |

### Or use Make from the repo root

```bash
cd convo-insight-platform    # repo root
make runserver               # start Django
make test                    # run tests
make lint                    # lint + format
make migrate                 # run migrations
make seed                    # seed demo data
make help                    # see all targets
```

---

## Code Conventions

- **Line length**: 120 characters, single quotes — enforced by ruff
- **Imports**: `from <app> import ...` for sibling apps (not `from apps.<app> import ...`)
- **JSX only** in frontend — never create `.ts` or `.tsx` files
- **Migrations**: always commit them. Never edit a migration that has been applied in CI
- **Secrets**: never commit `.env`. Use `.env.example` at repo root as the template
- **Tests**: every PR must include tests. Tests live next to code in `apps/<app>/tests/`

---

## Troubleshooting

### `docker compose up` fails with "port already in use"

Something else is using the port. Check what:

```bash
# Postgres port
lsof -i :5433
# Redis port
lsof -i :6380
```

Kill the process or change the port in `.env`.

### `python manage.py migrate` fails with connection error

Make sure Postgres is running: `docker compose ps`. If it says "starting", wait a few seconds.

Also check your `.env` has the right DB settings:
```
DB_HOST=localhost
DB_PORT=5433
```

### `ModuleNotFoundError: No module named 'xxx'`

You probably forgot to activate your venv:

```bash
source venv/bin/activate
```

Or you need to install deps:

```bash
cd backend && pip install -r requirements.txt
```

### `Can't find model 'en_core_web_sm'`

The spaCy NLP model hasn't been downloaded:

```bash
cd backend
python -m spacy download en_core_web_sm
```

### `ModuleNotFoundError: No module named 'convochat'`

You're running commands from the wrong directory. All Django commands must run from `backend/`:

```bash
cd backend
python manage.py runserver    # correct
```

Not from the repo root, not from inside `apps/`.

### Tests fail with "relation does not exist"

You need to run migrations first:

```bash
cd backend
python manage.py migrate
```

### Frontend shows "Failed to fetch"

The backend isn't running. Start it:

```bash
cd backend && python manage.py runserver
```

Also check `frontend/.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8000`.

---

## Where to Get Help

1. **Swagger UI** at http://localhost:8000/api/docs/ — try endpoints before guessing
2. **Django docs** at https://docs.djangoproject.com — always more reliable than Stack Overflow
3. **DRF docs** at https://www.django-rest-framework.org — for ViewSet, Serializer questions
4. **LangGraph docs** at https://langchain-ai.github.io/langgraph/ — for agent-related questions
5. **Ask your mentor** — no question is too basic. Seriously.

Good luck, and have fun!
