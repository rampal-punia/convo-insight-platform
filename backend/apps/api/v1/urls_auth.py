"""Authentication endpoints: JWT token obtain / refresh / verify / blacklist.

Frontend (Next.js) flow:
    1. POST /api/v1/auth/token/         { username, password }  -> { access, refresh }
    2. Send `Authorization: Bearer <access>` on every API call.
    3. When `access` expires, POST /api/v1/auth/token/refresh/  { refresh } -> new access.
    4. On logout, POST /api/v1/auth/token/blacklist/ { refresh } to invalidate.
"""

from django.urls import path
from rest_framework_simplejwt.views import (
    TokenBlacklistView,
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)

app_name = "auth"

urlpatterns = [
    path("token/", TokenObtainPairView.as_view(), name="token-obtain-pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token-refresh"),
    path("token/verify/", TokenVerifyView.as_view(), name="token-verify"),
    path("token/blacklist/", TokenBlacklistView.as_view(), name="token-blacklist"),
]
