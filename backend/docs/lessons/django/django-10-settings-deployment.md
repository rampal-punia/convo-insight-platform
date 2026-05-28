# Quick Win 10: Settings, Middleware & Deployment

> How settings, middleware, and the ASGI stack work together to run the whole platform.

---

## Settings Architecture

```
config/settings/
├── base.py     # Everything defined here
├── dev.py      # from .base import * + dev overrides
└── prod.py     # from .base import * + prod overrides
```

### The `base.py` pattern:

```python
import os
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# sys.path trick for clean imports
import sys
sys.path.insert(0, str(BASE_DIR / "apps"))

SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-fallback')

INSTALLED_APPS = [
    'daphne',                          # ASGI server (must be first)
    'django.contrib.admin',
    'django.contrib.auth',
    # ...
    'rest_framework',                  # DRF
    'rest_framework_simplejwt',        # JWT auth
    'channels',                        # WebSockets
    'django_celery_beat',              # Periodic tasks
    'django_filters',                  # Query filtering
    'crispy_forms',                    # Styled forms
    'crispy_bootstrap5',               # Bootstrap5 theme
    'pgvector.django',                 # Vector search

    # Our apps
    'accounts',
    'products',
    'orders',
    'convochat',
    'analysis',
    'playground',
    'support_agent',
    'general_assistant',
    'llms',
    'dashboard',
    'api',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'convo_insight'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}
```

### DRF configuration:

```python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
}
```

---

## Middleware Stack

Middleware runs on **every request**, in order. Think of it as a pipeline:

```
Request → Middleware 1 → Middleware 2 → ... → View → Response
Response ← Middleware 1 ← Middleware 2 ← ... ← View ←
```

### Our middleware stack:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

| Middleware | What It Does |
|-----------|-------------|
| `SecurityMiddleware` | SSL redirect, HSTS headers |
| `SessionMiddleware` | Manages session cookies |
| `CommonMiddleware` | URL normalization (trailing slashes) |
| `CsrfViewMiddleware` | CSRF protection for POST forms |
| `AuthenticationMiddleware` | Adds `request.user` to every request |
| `MessageMiddleware` | Flash messages between requests |

### Custom middleware for WebSockets:

```python
# apps/api/ws_auth.py
from channels.middleware import BaseMiddleware

class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        query_string = scope.get("query_string", b"").decode("utf-8")
        params = parse_qs(query_string)
        token = (params.get("token") or [None])[0]

        if token:
            scope = dict(scope)
            scope["user"] = await _get_user_from_token(token)

        return await super().__call__(scope, receive, send)
```

This is a **Channels middleware** — it wraps WebSocket connections, not HTTP requests.

---

## ASGI Application

The entry point that handles both HTTP and WebSocket:

```python
# config/asgi.py
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from api.ws_auth import JWTAuthMiddlewareStack

application = ProtocolTypeRouter({
    "http": get_asgi_application(),

    "websocket": JWTAuthMiddlewareStack(
        URLRouter([
            path("ws/general-assistant/", GeneralAssistantConsumer.as_asgi()),
            path("ws/playground/", NLPPlaygroundConsumer.as_asgi()),
            path("ws/support/", SupportAgentConsumer.as_asgi()),
        ])
    ),
})
```

**Flow:**
1. Incoming connection → `ProtocolTypeRouter` checks the protocol
2. If HTTP → standard Django handling
3. If WebSocket → `JWTAuthMiddlewareStack` authenticates → `URLRouter` routes to the right consumer

---

## Redis — The Multi-Purpose Server

Redis serves three roles in this project:

```python
# 1. Cache
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://localhost:6379/0',
    }
}

# 2. Celery broker
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# 3. Channels layer (WebSocket message bus)
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {"hosts": [('127.0.0.1', 6379)]},
    },
}
```

---

## Environment Variables

Never hardcode secrets in settings. Use environment variables:

```python
# settings/base.py
import os

SECRET_KEY = os.environ.get('SECRET_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
```

### `.env` file (local development):

```bash
SECRET_KEY=your-secret-key-here
DEBUG=True
OPENAI_API_KEY=sk-...
HUGGINGFACEHUB_API_TOKEN=hf_...
DB_NAME=convo_insight
DB_PASSWORD=postgres
```

---

## Deployment Checklist

### Services you need running:

```bash
# 1. PostgreSQL
pg_ctl start

# 2. Redis
redis-server

# 3. Migrate the database
python manage.py migrate

# 4. Start Daphne (ASGI server)
daphne config.asgi:application --port 8000 --bind 0.0.0.0

# 5. Start Celery worker (separate terminal)
celery -A config worker --loglevel=info --concurrency=4

# 6. Start Celery beat (separate terminal)
celery -A config beat --loglevel=info
```

### Production settings to verify:

```python
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com']
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

---

## Project-Wide Cheat Sheet

### The patterns used everywhere:

```python
# 1. Lazy imports (keep startup fast)
def analyze(text):
    import torch
    from transformers import AutoModel
    ...

# 2. asyncio.to_thread (run sync code without blocking)
result = await asyncio.to_thread(sync_function, args)

# 3. Error handling for external APIs
try:
    result = await api_call()
except (ConnectionError, TimeoutError):
    return fallback_response

# 4. Database operations in async context
from channels.db import database_sync_to_async

@database_sync_to_async
def save_to_db(obj):
    obj.save()

# 5. Celery tasks with lazy imports
@shared_task
def heavy_task(arg):
    from heavy_module import HeavyClass
    ...
```

---

## Quick Exercise

1. Read `config/settings/base.py` end-to-end — understand each section
2. Read `config/asgi.py` — trace how WebSocket connections are authenticated and routed
3. Check which environment variables are needed: `grep -r "os.environ.get" config/`
4. Start all services (PostgreSQL, Redis, Daphne, Celery worker) and verify everything connects
