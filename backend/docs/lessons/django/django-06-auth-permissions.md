# Quick Win 06: Authentication & Permissions

> How users log in, get tokens, and access only what they're allowed to.

---

## Two Auth Systems

This project uses **two** authentication methods:

1. **Session auth** — for Django templates (admin, dashboard)
2. **JWT auth** — for the REST API and WebSocket connections

---

## Session Auth (Django Built-in)

### How it works:

```
1. POST /accounts/login/ { username, password }
2. Django checks credentials
3. Sets a session cookie in the browser
4. Browser sends cookie with every subsequent request
5. Django identifies the user from the session
```

### Login View:

```python
# Django's built-in auth views handle this
# Just add to urls.py:
path('accounts/', include('django.contrib.auth.urls'))
```

This gives you `/accounts/login/`, `/accounts/logout/`, `/accounts/password_reset/`, etc.

### Checking auth in templates:

```html
{% if user.is_authenticated %}
    <p>Welcome, {{ user.username }}!</p>
    <a href="{% url 'accounts:logout' %}">Logout</a>
{% else %}
    <a href="{% url 'accounts:login' %}">Login</a>
{% endif %}
```

---

## JWT Auth (REST API)

### How it works:

```
1. POST /api/v1/auth/login/  { "username": "john", "password": "pass" }
2. Server returns: { "access": "eyJ...", "refresh": "eyJ..." }
3. Frontend stores tokens
4. Every API request: Authorization: Bearer eyJ...
5. When access token expires (60 min), use refresh token to get a new one
```

### Configuration:

```python
# config/settings/base.py
from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'AUTH_HEADER_TYPES': ('Bearer',),
}
```

### JWT URLs (auto-provided by SimpleJWT):

```python
# config/urls.py
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('api/v1/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/v1/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
```

### Testing with curl:

```bash
# Get tokens
curl -X POST http://localhost:8000/api/v1/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "john", "password": "pass123"}'

# Use access token
curl http://localhost:8000/api/v1/products/ \
  -H "Authorization: Bearer eyJhbGci..."

# Refresh when expired
curl -X POST http://localhost:8000/api/v1/auth/refresh/ \
  -H "Content-Type: application/json" \
  -d '{"refresh": "eyJhbGci..."}'
```

---

## Permissions

### DRF Permission Classes:

```python
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.IsAuthenticated):
    """Authenticated users can read; only the owner can modify."""

    def has_object_permission(self, request, view, obj):
        # Read permissions for any authenticated user
        if request.method in permissions.SAFE_METHODS:  # GET, HEAD, OPTIONS
            return True

        # Write permissions only for the owner
        owner = getattr(obj, "user", None) or getattr(obj, "owner", None)
        return owner == request.user
```

### Applying permissions:

```python
# Per-ViewSet
class OrderViewSet(viewsets.ModelViewSet):
    permission_classes = [IsOwnerOrReadOnly]

# Per-action
class ProductViewSet(viewsets.ModelViewSet):
    def get_permissions(self):
        if self.action in ['create', 'update', 'destroy']:
            return [permissions.IsAdminUser()]
        return [permissions.IsAuthenticatedOrReadOnly()]
```

### Built-in permission classes:

| Class | What It Allows |
|-------|---------------|
| `AllowAny` | Anyone (even unauthenticated) |
| `IsAuthenticated` | Any logged-in user |
| `IsAdminUser` | Only `is_staff=True` users |
| `IsAuthenticatedOrReadOnly` | Anyone can read, auth needed to write |

---

## WebSocket Auth

WebSockets don't support HTTP headers. This project sends JWT as a query parameter:

```
ws://localhost:8000/ws/general-assistant/?token=eyJhbGci...
```

### The middleware (`apps/api/ws_auth.py`):

```python
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

The consumer accesses the user via `self.scope["user"]`.

---

## Quick Exercise

1. Get a JWT token using `curl` (or Postman) and make an authenticated API call
2. Read `apps/api/permissions.py` — understand `IsOwnerOrReadOnly`
3. Try accessing `/api/v1/orders/` without a token — you should get `401 Unauthorized`
4. Read `apps/api/ws_auth.py` — trace how the token becomes `scope["user"]`
