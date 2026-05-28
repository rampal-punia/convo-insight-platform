# Quick Win 01: Django Project Structure & How Things Connect

> Understand where everything lives and how a request travels through the codebase.

---

## Django in 30 Seconds

Django is a **MVT** framework — Model, View, Template:

- **Model** — Your database tables (Python classes)
- **View** — Your logic (what happens when a URL is hit)
- **Template** — Your HTML (what the user sees)

```
Request → URL → View → Model (database) → Template → Response
```

---

## Our Project Layout

```
backend/
├── config/                    # Project-level settings
│   ├── settings/
│   │   └── base.py           # All configuration lives here
│   ├── urls.py               # Root URL router
│   ├── asgi.py               # Async server (WebSocket + HTTP)
│   └── celery.py             # Background task runner
│
├── apps/                      # Each "app" = one domain
│   ├── accounts/             # Users, login, signup
│   ├── products/             # Product catalog
│   ├── orders/               # Orders, LangGraph agent
│   ├── convochat/            # Chat + NLP pipeline
│   ├── support_agent/        # LangGraph support agent
│   ├── general_assistant/    # Multimodal AI (text/image/voice)
│   ├── playground/           # NLP playground (BERT, RAG)
│   ├── analysis/             # Performance metrics
│   ├── llms/                 # Fine-tuning pipeline
│   ├── dashboard/            # Admin dashboard
│   └── api/                  # REST API (DRF ViewSets)
│
├── templates/                 # HTML templates
│   ├── base.html
│   ├── accounts/
│   └── dashboard/
│
├── manage.py                  # Django CLI tool
└── docs/                      # You are here
```

### The `sys.path` Trick

```python
# config/settings/base.py
import sys
sys.path.insert(0, str(BASE_DIR / "apps"))
```

This lets you write `from products.models import Product` instead of `from apps.products.models import Product`. Cleaner imports.

---

## How a Request Flows

### HTTP Request (REST API):

```
Browser sends: GET /api/v1/products/?category=2
                        │
                        ▼
              config/urls.py
              path('api/v1/', include('api.urls'))
                        │
                        ▼
              api/urls.py
              router.register('products', ProductViewSet)
                        │
                        ▼
              ProductViewSet.list()
              → filters queryset
              → paginates results
                        │
                        ▼
              ProductSerializer(queryset, many=True)
              → converts Python objects → JSON
                        │
                        ▼
              Response: { "count": 45, "results": [...] }
```

### WebSocket Connection:

```
Browser sends: ws://localhost:8000/ws/general-assistant/?token=eyJ...
                        │
                        ▼
              config/asgi.py
              ProtocolTypeRouter → "websocket" branch
                        │
                        ▼
              JWTAuthMiddleware
              → validates token → sets scope['user']
                        │
                        ▼
              GeneralAssistantConsumer
              → connect() → accept()
              → receive_json() → handle messages
              → send() → push data to browser
```

---

## The `manage.py` Commands You'll Use Daily

```bash
# Run the development server
python manage.py runserver

# Create database migrations (after changing models)
python manage.py makemigrations

# Apply migrations to the database
python manage.py migrate

# Create a superuser for the admin panel
python manage.py createsuperuser

# Open the Django shell (Python REPL with Django loaded)
python manage.py shell

# Run tests
python manage.py test
# or with pytest:
pytest

# Project-specific commands
python manage.py seed_demo              # Fill DB with test data
python manage.py create_random_users 5  # Create test users
```

---

## Quick Exercise

1. Run `python manage.py showmigrations` — see which apps have been migrated
2. Run `python manage.py shell` then:
   ```python
   from products.models import Product
   Product.objects.count()
   ```
3. Open `config/urls.py` and trace how `/api/v1/` connects to the API ViewSets
