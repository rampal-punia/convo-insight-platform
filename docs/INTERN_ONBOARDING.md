# Intern Onboarding — ConvoInsight Platform

Welcome! This is a **3 to 4 week** path to get you productive on the codebase.
Read it in order. Stop and ask whenever something doesn't make sense.

---

## TL;DR — first hour

```bash
# 1. Clone, create venv, install
git clone <repo>
cd convo-insight-platform
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in OPENAI_API_KEY at minimum

# 2. Start the stack (Postgres + Redis)
docker compose up -d postgres redis

# 3. Migrate, seed, run
python manage.py migrate
python manage.py seed_demo
python manage.py runserver

# 4. Open the docs in your browser
#    http://localhost:8000/api/docs/      (Swagger UI)
#    http://localhost:8000/admin/         (Django admin — log in as demo_admin / demo12345)

# 5. Frontend
cd frontend && npm install && npm run dev
#    http://localhost:3000
```

---

## Week 1 — orientation

**Goal**: be able to read, run, and locate things.

- [ ] Read `README.md` and `CONTRIBUTING.md`.
- [ ] Run the full stack locally (steps above). Log in to admin, browse Swagger UI.
- [ ] Spend an hour clicking through Swagger — try `GET /api/v1/products/`, `POST /api/v1/auth/token/`.
- [ ] Run the test suite: `pytest apps/api/v1/tests/ -v`. All 12 tests should pass.
- [ ] Open the Django admin and walk through every model: Products, Orders, Conversations, Messages, Topics.
- [ ] Map the request lifecycle for one URL of your choice — from `config/urls.py` to a viewset to a serializer to the DB.

**Concepts to internalize**:

- Django apps (`apps/<app_name>/`) and the `apps/` sys.path trick (see `config/settings/base.py`).
- DRF viewsets, serializers, routers, `@action`.
- JWT auth (`/api/v1/auth/token/` → bearer header on every call).

---

## Week 2 — make a small backend change

**Goal**: ship a feature end-to-end.

Pick **one** from the warm-up list:

1. Add a `GET /api/v1/products/featured/` endpoint that returns products with `stock > 50`.
2. Add a `POST /api/v1/orders/{id}/refund/` action that flips the order to `REFUNDED`.
3. Add a `GET /api/v1/users/me/orders/` endpoint that returns the current user's orders.

For each one:

- Write the viewset action in `apps/api/v1/views_*.py`.
- Add a smoke test in `apps/api/v1/tests/test_smoke.py`.
- Re-run `pytest` and `python manage.py spectacular --file /tmp/schema.yml` (should not warn).
- Open Swagger UI and exercise the endpoint manually.

---

## Week 3 — touch the WebSocket / Celery side

**Goal**: understand the async surfaces.

- Read `apps/convochat/consumers.py` and `apps/convochat/routing.py`.
- Read `apps/api/ws_auth.py` — that's the JWT bridge for WebSockets.
- Run `python manage.py runserver` and open a browser console:

  ```js
  const access = '<paste-from-/api/v1/auth/token/>';
  const ws = new WebSocket(`ws://localhost:8000/ws/convochat/?token=${access}`);
  ws.onmessage = (e) => console.log(e.data);
  ws.onopen = () => ws.send(JSON.stringify({ type: 'ping' }));
  ```

- Read `apps/orders/tasks.py` and `config/celery.py`. Start a worker: `celery -A config worker -l info`.
- Trigger a Celery task from the Django shell.

---

## Week 4 — connect the frontend

**Goal**: ship a small UI feature backed by your week-2 endpoint.

- Read `frontend/README.md` and `frontend/src/lib/api.js`.
- Add a page under `frontend/src/app/<feature>/page.jsx` that hits your new endpoint.
- Hook up the login → fetch → display flow.

**Remember the constraint**: this project is **JSX only, no TypeScript**.

---

## Day-to-day commands

| Need                        | Command                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| Run dev server              | `python manage.py runserver`                                     |
| Run worker                  | `celery -A config worker -l info`                                |
| Run beat                    | `celery -A config beat -l info`                                  |
| Tests                       | `pytest`                                                         |
| Tests with coverage         | `coverage run -m pytest && coverage report`                      |
| Lint / format               | `ruff check . && ruff format .`                                  |
| Generate OpenAPI            | `python manage.py spectacular --file /tmp/schema.yml`            |
| Seed demo data              | `python manage.py seed_demo`                                     |
| Wipe + reseed               | `python manage.py seed_demo --reset`                             |
| Open shell                  | `python manage.py shell_plus`                                    |
| Make + apply migrations     | `python manage.py makemigrations && python manage.py migrate`    |
| Frontend dev                | `cd frontend && npm run dev`                                     |

---

## Code conventions

- **Line length**: 120, single quotes (enforced by ruff). See `pyproject.toml`.
- **Imports**: `from <app> import …` for sibling apps (the `apps/` directory is on sys.path).
- **Migrations**: always commit. Never edit a migration that has been applied in CI.
- **Secrets**: never commit `.env`. Use `.env.example` as the template.
- **Tests live next to code**: `apps/<app>/tests/test_*.py` or `apps/api/v1/tests/test_*.py`.

---

## Where to ask for help

- Look in `docs/` for design notes.
- Check the Swagger UI before guessing at API shapes.
- For anything Django-specific, the official docs at https://docs.djangoproject.com are
  always more reliable than Stack Overflow.

Good luck, and have fun!
