"""Serializers for the Orders domain."""
from rest_framework import serializers

from orders.models import Order, OrderItem, OrderTracking


class OrderItemSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source="product.name", read_only=True)

    class Meta:
        model = OrderItem
        fields = ["id", "product", "product_name", "quantity", "price"]


class OrderTrackingSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)

    class Meta:
        model = OrderTracking
        fields = [
            "id",
            "order",
            "status",
            "status_display",
            "location",
            "description",
            "timestamp",
            "created",
            "modified",
        ]
        read_only_fields = ["timestamp", "created", "modified"]


class OrderListSerializer(serializers.ModelSerializer):
    """Lean serializer for list endpoints."""

    status_display = serializers.CharField(source="get_status_display", read_only=True)
    shipping_method_display = serializers.CharField(
        source="get_shipping_method_display", read_only=True
    )
    user_username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = Order
        fields = [
            "id",
            "user",
            "user_username",
            "status",
            "status_display",
            "shipping_method",
            "shipping_method_display",
            "total_amount",
            "tracking_number",
            "carrier",
            "estimated_delivery",
            "shipped_date",
            "created",
            "modified",
        ]
        read_only_fields = ["created", "modified", "user"]


class OrderDetailSerializer(OrderListSerializer):
    """Detail serializer including nested items and tracking history."""

    items = OrderItemSerializer(many=True, read_only=True)
    tracking_history = OrderTrackingSerializer(many=True, read_only=True)
    current_status = serializers.SerializerMethodField()

    class Meta(OrderListSerializer.Meta):
        fields = OrderListSerializer.Meta.fields + [
            "delivery_address",
            "items",
            "tracking_history",
            "current_status",
        ]

    def get_current_status(self, obj: Order) -> str:
        return obj.get_tracking_status()
