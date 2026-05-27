"""ViewSets for the Accounts domain."""
from django.contrib.auth import get_user_model
from drf_spectacular.utils import extend_schema
from rest_framework import permissions, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..pagination import StandardResultsSetPagination
from .serializers_accounts import UserSerializer

User = get_user_model()


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only user directory. Non-staff users only see themselves."""

    queryset = User.objects.all().order_by("-date_joined")
    serializer_class = UserSerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        user = self.request.user
        if user.is_authenticated and not user.is_staff:
            qs = qs.filter(pk=user.pk)
        return qs

    @extend_schema(summary="Get the currently authenticated user", responses=UserSerializer)
    @action(detail=False, methods=["get"], url_path="me", url_name="me")
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)
