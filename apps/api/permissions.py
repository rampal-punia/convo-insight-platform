"""Reusable permission classes for the v1 API."""
from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.BasePermission):
    """Object-level: only the owning user can mutate; everyone authenticated can read."""

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        owner = getattr(obj, "user", None) or getattr(obj, "owner", None)
        return owner == request.user


class IsOwner(permissions.BasePermission):
    """Object-level: only the owning user can read or mutate."""

    def has_object_permission(self, request, view, obj):
        owner = getattr(obj, "user", None) or getattr(obj, "owner", None)
        return owner == request.user
