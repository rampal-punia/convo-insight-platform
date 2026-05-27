"""ViewSets for the Orders domain."""
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema
from rest_framework import filters, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from orders.models import Order, OrderItem, OrderTracking

from ..pagination import StandardResultsSetPagination
from ..permissions import IsOwnerOrReadOnly
from .serializers_orders import (
    OrderDetailSerializer,
    OrderItemSerializer,
    OrderListSerializer,
    OrderTrackingSerializer,
)


class OrderViewSet(viewsets.ModelViewSet):
    """CRUD on orders. Non-staff users only see their own orders."""

    queryset = Order.objects.select_related("user").prefetch_related(
        "items__product", "tracking_history"
    )
    pagination_class = StandardResultsSetPagination
    permission_classes = [IsOwnerOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["status", "shipping_method", "carrier", "user"]
    search_fields = ["tracking_number", "carrier", "delivery_address"]
    ordering_fields = ["created", "modified", "total_amount", "estimated_delivery"]

    def get_serializer_class(self):
        if self.action == "list":
            return OrderListSerializer
        return OrderDetailSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        user = self.request.user
        if user.is_authenticated and not user.is_staff:
            qs = qs.filter(user=user)
        return qs.order_by("-created")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @extend_schema(
        summary="Get the full tracking history for an order",
        responses=OrderTrackingSerializer(many=True),
    )
    @action(detail=True, methods=["get"], url_path="tracking", url_name="tracking")
    def tracking(self, request, pk=None):
        order = self.get_object()
        qs = order.tracking_history.all().order_by("-timestamp")
        serializer = OrderTrackingSerializer(qs, many=True)
        return Response(serializer.data)

    @extend_schema(
        summary="Append a tracking update to an order",
        request=OrderTrackingSerializer,
        responses=OrderTrackingSerializer,
    )
    @action(
        detail=True,
        methods=["post"],
        url_path="tracking/add",
        url_name="tracking-add",
    )
    def add_tracking(self, request, pk=None):
        order = self.get_object()
        payload = {**request.data, "order": order.pk}
        serializer = OrderTrackingSerializer(data=payload)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(
        summary="List items in an order",
        responses=OrderItemSerializer(many=True),
    )
    @action(detail=True, methods=["get"], url_path="items", url_name="items")
    def items(self, request, pk=None):
        order = self.get_object()
        qs = order.items.select_related("product").all()
        serializer = OrderItemSerializer(qs, many=True)
        return Response(serializer.data)


class OrderItemViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to order items across all orders for analytics."""

    queryset = OrderItem.objects.select_related("order", "product").all()
    serializer_class = OrderItemSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["order", "product"]
    ordering_fields = ["quantity", "price", "id"]

    def get_queryset(self):
        qs = super().get_queryset()
        user = self.request.user
        if user.is_authenticated and not user.is_staff:
            qs = qs.filter(order__user=user)
        return qs


class OrderTrackingViewSet(viewsets.ModelViewSet):
    """CRUD on individual tracking events (admin-heavy resource)."""

    queryset = OrderTracking.objects.select_related("order").all().order_by("-timestamp")
    serializer_class = OrderTrackingSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["order", "status", "location"]
    ordering_fields = ["timestamp", "status"]
