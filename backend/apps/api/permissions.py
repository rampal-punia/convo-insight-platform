"""Reusable permission classes for the v1 API."""
from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.IsAuthenticated):
    """Authenticated users can read; only the owner can mutate.

    Inherits ``has_permission`` from ``IsAuthenticated`` so anonymous requests
    are rejected at the view level before object lookup.
    """

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        owner = getattr(obj, "user", None) or getattr(obj, "owner", None)
        return owner == request.user


class IsOwner(permissions.IsAuthenticated):
    """Authenticated users only; can read/mutate only their own records."""

    def has_object_permission(self, request, view, obj):
        owner = getattr(obj, "user", None) or getattr(obj, "owner", None)
        return owner == request.user
