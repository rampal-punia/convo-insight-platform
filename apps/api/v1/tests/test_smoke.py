"""Smoke tests for the v1 API surface.

These exercise the *routing*, *auth*, and *basic shape* of the new API.
Heavy domain tests live next to each app.
"""
from __future__ import annotations

import pytest
from django.urls import reverse
from rest_framework import status


pytestmark = pytest.mark.django_db


# --------------------------------------------------------------------------- #
# JWT auth                                                                    #
# --------------------------------------------------------------------------- #


def test_token_obtain_pair(api_client, user):
    url = reverse("api:v1:auth:token-obtain-pair")
    resp = api_client.post(url, {"username": user.username, "password": "pw12345"}, format="json")
    assert resp.status_code == status.HTTP_200_OK
    assert "access" in resp.data
    assert "refresh" in resp.data


def test_token_refresh(api_client, user):
    obtain = api_client.post(
        reverse("api:v1:auth:token-obtain-pair"),
        {"username": user.username, "password": "pw12345"},
        format="json",
    )
    refresh = obtain.data["refresh"]
    resp = api_client.post(reverse("api:v1:auth:token-refresh"), {"refresh": refresh}, format="json")
    assert resp.status_code == status.HTTP_200_OK
    assert "access" in resp.data


def test_token_obtain_with_wrong_password_fails(api_client, user):
    resp = api_client.post(
        reverse("api:v1:auth:token-obtain-pair"),
        {"username": user.username, "password": "WRONG"},
        format="json",
    )
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


# --------------------------------------------------------------------------- #
# Routing / docs                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/products/",
        "/api/v1/categories/",
        "/api/v1/orders/",
        "/api/v1/conversations/",
        "/api/v1/users/me/",
    ],
)
def test_endpoints_require_auth(api_client, path):
    """Anonymous requests should be 401/403, never 5xx."""
    resp = api_client.get(path)
    assert resp.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}


def test_openapi_schema_renders(api_client):
    resp = api_client.get("/api/schema/")
    assert resp.status_code == status.HTTP_200_OK
    body = resp.content.decode()
    assert "openapi:" in body or '"openapi"' in body


def test_swagger_ui_renders(api_client):
    resp = api_client.get("/api/docs/")
    assert resp.status_code == status.HTTP_200_OK


# --------------------------------------------------------------------------- #
# Authenticated reads                                                         #
# --------------------------------------------------------------------------- #


def test_authenticated_products_list(auth_client):
    resp = auth_client.get("/api/v1/products/")
    assert resp.status_code == status.HTTP_200_OK
    assert "results" in resp.data
    assert resp.data["results"] == []


def test_users_me(auth_client, user):
    resp = auth_client.get("/api/v1/users/me/")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.data["username"] == user.username
