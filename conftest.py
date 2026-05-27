"""Top-level pytest fixtures shared across the test suite."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def user(db):
    User = get_user_model()
    return User.objects.create_user(
        username="alice", email="alice@example.com", password="pw12345"
    )


@pytest.fixture
def staff_user(db):
    User = get_user_model()
    return User.objects.create_user(
        username="staff", email="staff@example.com", password="pw12345", is_staff=True
    )


@pytest.fixture
def auth_client(api_client, user) -> APIClient:
    """An APIClient pre-authenticated with a JWT access token for ``user``."""
    refresh = RefreshToken.for_user(user)
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {refresh.access_token}")
    return api_client


@pytest.fixture
def staff_client(api_client, staff_user) -> APIClient:
    refresh = RefreshToken.for_user(staff_user)
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {refresh.access_token}")
    return api_client
