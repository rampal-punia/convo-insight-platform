"""Top-level URL routing for the public API.

Versioning lives here. Every released version is mounted under its own prefix
so the Next.js frontend can target a specific version safely.

Auto-discoverable endpoints (Step 1):
    /api/v1/                       Browsable API root (route map)
    /api/v1/<resource>/            CRUD for every registered ViewSet
    /api/v1/<resource>/<action>/   Every ``@action(url_path=...)`` decorator
    /api/schema/                   OpenAPI 3 schema (machine readable)
    /api/docs/                     Swagger UI
    /api/redoc/                    Redoc UI
"""

from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)

app_name = "api"

urlpatterns = [
    # Versioned API
    path("v1/", include("api.v1.urls", namespace="v1")),
    # OpenAPI schema + interactive docs (shared across versions for now)
    path("schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "docs/",
        SpectacularSwaggerView.as_view(url_name="api:schema"),
        name="docs",
    ),
    path(
        "redoc/",
        SpectacularRedocView.as_view(url_name="api:schema"),
        name="redoc",
    ),
]


# ---------------------------------------------------------------------------
# Quick reference (Step 1)
# ---------------------------------------------------------------------------
# Catalogue:
#   GET/POST            /api/v1/products/
#   GET/PUT/PATCH/DEL   /api/v1/products/{id}/
#   GET                 /api/v1/products/in-stock/
#   GET                 /api/v1/products/low-stock/?threshold=10
#   GET/POST            /api/v1/categories/
#   GET                 /api/v1/categories/{id}/products/
#
# Orders:
#   GET/POST            /api/v1/orders/
#   GET/PUT/PATCH/DEL   /api/v1/orders/{id}/
#   GET                 /api/v1/orders/{id}/tracking/
#   POST                /api/v1/orders/{id}/tracking/add/
#   GET                 /api/v1/orders/{id}/items/
#   GET                 /api/v1/order-items/
#   GET/POST            /api/v1/order-tracking/
#
# Conversations:
#   GET/POST            /api/v1/conversations/
#   GET                 /api/v1/conversations/{id}/messages/
#   POST                /api/v1/conversations/{id}/archive/
#   POST                /api/v1/conversations/{id}/end/
#   GET/POST            /api/v1/messages/
#   GET/POST            /api/v1/intents/
#   GET/POST            /api/v1/topics/
#   GET                 /api/v1/topics/trending/
#   GET                 /api/v1/sentiments/
#
# Analysis:
#   GET/POST            /api/v1/agent-performance/
#   GET/POST            /api/v1/conversation-metrics/
#   GET/POST            /api/v1/recommendations/
#   GET                 /api/v1/topic-distributions/
#   GET                 /api/v1/intent-predictions/
#
# Accounts:
#   GET                 /api/v1/users/
#   GET                 /api/v1/users/me/
#
# NLP playground:
#   POST                /api/v1/nlp/sentiment/
#   POST                /api/v1/nlp/intent/
#   POST                /api/v1/nlp/topic/
#   POST                /api/v1/nlp/ner/
