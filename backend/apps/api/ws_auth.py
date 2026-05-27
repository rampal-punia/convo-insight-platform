"""WebSocket authentication middleware that resolves a SimpleJWT access token
from the query string and attaches the corresponding user to ``scope['user']``.

Usage from the Next.js client::

    const ws = new WebSocket(`ws://api.example.com/ws/orders/${id}/?token=${access}`);

Wire it in ``config/asgi.py`` by wrapping the URLRouter with
``JWTAuthMiddlewareStack`` *instead of* (or layered with) ``AuthMiddlewareStack``.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qs

from channels.auth import AuthMiddlewareStack
from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

logger = logging.getLogger("convochat")

User = get_user_model()


@database_sync_to_async
def _get_user_from_token(token: str):
    """Decode the JWT and return the corresponding user, or AnonymousUser."""
    # Imported lazily so this module can be imported even before Django apps
    # are fully populated (e.g. by ``manage.py check``).
    from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
    from rest_framework_simplejwt.tokens import UntypedToken

    if not token:
        return AnonymousUser()

    try:
        validated = UntypedToken(token)
    except (InvalidToken, TokenError) as exc:
        logger.warning("Rejected WS token: %s", exc)
        return AnonymousUser()

    user_id = validated.get("user_id")
    if not user_id:
        return AnonymousUser()

    try:
        return User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    """Channels middleware that pulls ``?token=...`` and authenticates the user."""

    async def __call__(self, scope, receive, send):
        query_string = scope.get("query_string", b"").decode("utf-8", errors="ignore")
        params = parse_qs(query_string)
        token_values = params.get("token") or params.get("access") or []
        token = token_values[0] if token_values else None

        # If a token is supplied we use it; otherwise fall through to whatever
        # previous middleware (session/cookie) already put in ``scope['user']``.
        if token:
            scope = dict(scope)
            scope["user"] = await _get_user_from_token(token)
        return await super().__call__(scope, receive, send)


def JWTAuthMiddlewareStack(inner):
    """Convenience wrapper combining JWT + session/cookie auth."""
    return JWTAuthMiddleware(AuthMiddlewareStack(inner))
