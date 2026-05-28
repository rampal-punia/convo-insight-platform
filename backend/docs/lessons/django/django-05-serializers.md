# Quick Win 05: Serializers — Python Objects to JSON

> Serializers convert Django model instances to JSON for the API, and validate incoming JSON data.

---

## Why Serializers?

Your database stores Python objects. Your frontend wants JSON. Serializers bridge the gap.

```
Database → Model instance → Serializer → JSON → Frontend
Frontend → JSON → Serializer (validates) → Model instance → Database
```

---

## ModelSerializer (Most Common)

```python
# apps/api/v1/serializers_orders.py
from rest_framework import serializers
from orders.models import OrderItem

class OrderItemSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source="product.name", read_only=True)

    class Meta:
        model = OrderItem
        fields = ['id', 'product', 'product_name', 'quantity', 'price']
```

**What happens:**

```python
# Serialization: Python → JSON
item = OrderItem.objects.get(id=1)
serializer = OrderItemSerializer(item)
serializer.data
# → {'id': 1, 'product': 5, 'product_name': 'Widget', 'quantity': 2, 'price': '19.99'}

# Deserialization: JSON → Python (with validation)
serializer = OrderItemSerializer(data={'product': 5, 'quantity': 3, 'price': '29.99'})
serializer.is_valid()  # → True
serializer.validated_data  # → {'product': <Product: Widget>, 'quantity': 3, 'price': Decimal('29.99')}
```

---

## Key Serializer Features

### 1. `source` — Follow Relationships

```python
class OrderListSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    user_username = serializers.CharField(source="user.username", read_only=True)
```

`source="user.username"` follows the ForeignKey to User and gets the username. No extra database query if you use `select_related('user')` in the ViewSet.

### 2. Nested Serializers

```python
class OrderDetailSerializer(OrderListSerializer):
    items = OrderItemSerializer(many=True, read_only=True)
    tracking_history = OrderTrackingSerializer(many=True, read_only=True)
    current_status = serializers.SerializerMethodField()

    class Meta(OrderListSerializer.Meta):
        fields = OrderListSerializer.Meta.fields + [
            "delivery_address", "items", "tracking_history", "current_status",
        ]

    def get_current_status(self, obj: Order) -> str:
        return obj.get_tracking_status()
```

- `items = OrderItemSerializer(many=True)` — serializes the reverse relation (one order → many items)
- `SerializerMethodField` — calls `get_<fieldname>()` method for custom logic
- `read_only=True` — only output, not accepted in input

### 3. Different Serializers for List vs Detail

```python
class OrderViewSet(viewsets.ModelViewSet):
    def get_serializer_class(self):
        if self.action == 'retrieve':      # GET /api/v1/orders/{id}/
            return OrderDetailSerializer    # Full detail with nested items
        return OrderListSerializer          # GET /api/v1/orders/ — summary only
```

This avoids loading all order items when you just need a list.

---

## Validation

### Automatic Validation

The `ModelSerializer` automatically validates based on model field constraints:

```python
serializer = OrderItemSerializer(data={'quantity': -1})
serializer.is_valid()  # → False (PositiveIntegerField doesn't accept negatives)
serializer.errors      # → {'quantity': ['Ensure this value is greater than or equal to 0.']}
```

### Custom Validation in Forms

```python
# apps/orders/forms.py
class OrderItemForm(forms.ModelForm):
    class Meta:
        model = OrderItem
        fields = ['product', 'quantity']

    def clean(self):
        cleaned_data = super().clean()
        product = cleaned_data.get('product')
        quantity = cleaned_data.get('quantity')

        if product and quantity:
            if quantity > product.stock:
                raise ValidationError(
                    f"Not enough stock. Only {product.stock} available."
                )
        return cleaned_data
```

### Serializer-Level Validation

```python
class OrderCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = ['user', 'items', 'shipping_method']

    def validate_shipping_method(self, value):
        """Per-field validation."""
        if value == Order.ShippingMethod.OVERNIGHT:
            # Check cutoff time for overnight shipping
            if timezone.now().hour >= 15:
                raise serializers.ValidationError(
                    "Overnight shipping cutoff is 3 PM."
                )
        return value

    def validate(self, data):
        """Whole-object validation."""
        if data.get('shipping_method') == Order.ShippingMethod.LOCAL:
            # Verify delivery address is within range
            if not data.get('delivery_address'):
                raise serializers.ValidationError(
                    "Local delivery requires a delivery address."
                )
        return data
```

---

## Inline Formsets (Django Forms)

For editing related objects on the same page:

```python
# apps/orders/forms.py
from django.forms import inlineformset_factory

OrderItemFormSet = inlineformset_factory(
    Order,           # Parent model
    OrderItem,       # Child model
    form=OrderItemForm,
    extra=1,         # Number of blank rows
    can_delete=True  # Allow deleting items
)
```

This renders a table of OrderItem forms inside the Order form.

---

## Quick Exercise

1. Read `apps/api/v1/serializers_orders.py` — trace how `OrderDetailSerializer` nests `OrderItemSerializer`
2. Read `apps/api/v1/serializers_conversations.py` — find the nested serializer pattern there
3. Create a simple serializer:
   ```python
   from rest_framework import serializers
   from products.models import Product

   class ProductSummarySerializer(serializers.ModelSerializer):
       category_name = serializers.CharField(source='category.name')

       class Meta:
           model = Product
           fields = ['id', 'name', 'price', 'category_name']
   ```
   Try it in the shell:
   ```python
   from api.v1.serializers_orders import OrderListSerializer
   from orders.models import Order
   order = Order.objects.first()
   print(OrderListSerializer(order).data)
   ```
