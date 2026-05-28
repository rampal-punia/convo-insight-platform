# Lesson 2: Products, Orders & the DRF API Layer

> How Django REST Framework ViewSets, Serializers, filtering, and pagination expose data to the frontend.

---

## What You'll Learn

- DRF `ModelViewSet` — CRUD in one class
- Serializers — converting model instances to/from JSON
- Custom ViewSet actions — extending beyond basic CRUD
- Filtering, ordering, and pagination
- Nested relationships (Category → Products)

---

## 1. The Model Layer

### Product Model (`products/models.py`)

```python
class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    stock = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
```

### Order Model (`orders/models.py`)

```python
class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
```

### OrderItem Model

```python
class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_length=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)
```

**Relationship:**
```
User → Order → OrderItem → Product
                     └→ Product ← Category
```

---

## 2. DRF ViewSets

A `ModelViewSet` gives you all CRUD operations automatically:

```python
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticated]
    filterset_fields = ['category', 'is_active']
    ordering_fields = ['price', 'created_at', 'name']
    search_fields = ['name', 'description']
```

### What this single class provides:

| HTTP | URL | Method | Action |
|------|-----|--------|--------|
| GET | `/api/v1/products/` | `list()` | List all products (with pagination) |
| POST | `/api/v1/products/` | `create()` | Create a new product |
| GET | `/api/v1/products/{id}/` | `retrieve()` | Get one product |
| PUT | `/api/v1/products/{id}/` | `update()` | Replace a product |
| PATCH | `/api/v1/products/{id}/` | `partial_update()` | Update specific fields |
| DELETE | `/api/v1/products/{id}/` | `destroy()` | Delete a product |

### Custom Actions — Going Beyond CRUD

This project adds custom endpoints using the `@action` decorator:

```python
from rest_framework.decorators import action
from rest_framework.response import Response

class ProductViewSet(viewsets.ModelViewSet):
    # ... standard config ...

    @action(detail=False, methods=['get'])
    def featured(self, request):
        """GET /api/v1/products/featured/"""
        featured = self.queryset.filter(is_active=True).order_by('-created_at')[:10]
        serializer = self.get_serializer(featured, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def update_stock(self, request, pk=None):
        """POST /api/v1/products/{id}/update_stock/"""
        product = self.get_object()
        product.stock = request.data.get('stock', product.stock)
        product.save()
        return Response({'status': 'stock updated'})

    @action(detail=False, methods=['get'])
    def by_category(self, request):
        """GET /api/v1/products/by_category/?category_id=1"""
        category_id = request.query_params.get('category_id')
        products = self.queryset.filter(category_id=category_id)
        serializer = self.get_serializer(products, many=True)
        return Response(serializer.data)
```

**Key concept:** `detail=False` means the action works on the collection (no `{id}` in URL). `detail=True` means it works on a single instance.

---

## 3. Serializers

Serializers convert between Django model instances and JSON (or other formats).

### Basic Serializer

```python
from rest_framework import serializers
from products.models import Product

class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)

    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'price', 'category',
                  'category_name', 'stock', 'is_active', 'created_at']
        read_only_fields = ['created_at']
```

### Nested Serializer (Order with Items)

```python
class OrderItemSerializer(serializers.ModelSerializer):
    product_name = serializers.CharField(source='product.name', read_only=True)

    class Meta:
        model = OrderItem
        fields = ['id', 'product', 'product_name', 'quantity', 'price']


class OrderSerializer(serializers.ModelSerializer):
    items = OrderItemSerializer(many=True, read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)

    class Meta:
        model = Order
        fields = ['id', 'user', 'user_email', 'status', 'total_amount',
                  'items', 'created_at']
```

**How it works:**
- `source='category.name'` — follows the foreign key to get related data
- `many=True` — for reverse relations (one order has many items)
- `read_only=True` — computed from the model, not accepted in input

---

## 4. Filtering, Ordering & Pagination

### URL-based Filtering

```
GET /api/v1/products/?category=1&is_active=true
```

The `django-filter` package + `filterset_fields` handles this automatically.

### Ordering

```
GET /api/v1/products/?ordering=-price    # descending by price
GET /api/v1/products/?ordering=name       # ascending by name
```

### Pagination

All list endpoints are paginated:

```python
# config/settings/base.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}
```

Response format:
```json
{
    "count": 150,
    "next": "/api/v1/products/?page=2",
    "previous": null,
    "results": [
        { "id": 1, "name": "Widget", ... },
        { "id": 2, "name": "Gadget", ... }
    ]
}
```

---

## 5. Complete Request-Response Cycle

### Scenario: Browse products by category

```
1. Frontend:  GET /api/v1/products/?category=2&ordering=-price&page=1
              Authorization: Bearer eyJhbGci...

2. Daphne:    Routes to api.urls → router → ProductViewSet

3. DRF:       Permission check → IsAuthenticated → validates JWT token

4. ViewSet:   list() → applies filters (category=2)
           → applies ordering (-price)
           → applies pagination (page 1, 20 per page)

5. Serializer: ProductSerializer(queryset, many=True)
           → converts 20 Product instances to JSON dicts

6. Response:  200 OK
              {
                  "count": 45,
                  "next": "/api/v1/products/?category=2&ordering=-price&page=2",
                  "previous": null,
                  "results": [...]
              }
```

### Scenario: Create a new order

```
1. Frontend:  POST /api/v1/orders/
              Authorization: Bearer eyJhbGci...
              {
                  "items": [
                      {"product": 1, "quantity": 2},
                      {"product": 5, "quantity": 1}
                  ]
              }

2. DRF:       Permission check → JWT validated
              → Extract user from token

3. ViewSet:   create() → OrderSerializer.validate()
              → Calculate total_amount from product prices × quantities

4. Database:  INSERT INTO orders (...) VALUES (...)
              → INSERT INTO order_items (...) for each item

5. Response:  201 Created
              {
                  "id": 42,
                  "status": "pending",
                  "total_amount": "59.97",
                  "items": [...]
              }
```

---

## 6. API URL Structure

All API endpoints are versioned under `/api/v1/`:

```python
# api/urls.py
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'products', ProductViewSet)
router.register(r'categories', CategoryViewSet)
router.register(r'orders', OrderViewSet)
router.register(r'conversations', ConversationViewSet)
# ... more registrations

urlpatterns = [
    path('', include(router.urls)),
    path('auth/', include('rest_framework.urls')),
]
```

The `DefaultRouter` auto-generates URL patterns from the ViewSet.

---

## 7. Permissions

```python
# api/permissions.py
from rest_framework.permissions import BasePermission

class IsOwnerOrAdmin(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.user.is_staff:
            return True
        return obj.user == request.user
```

Applied per-ViewSet:
```python
class OrderViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated, IsOwnerOrAdmin]
```

This means regular users can only see/modify their own orders, while admins see everything.

---

## Exercises

1. **Add a "Low Stock" endpoint** — Create a custom action `@action(detail=False)` that returns products where `stock < 5`.
2. **Add reviews** — Create a `Review` model (user, product, rating, comment), a serializer, and add a nested `reviews` field to `ProductSerializer`.
3. **Order status transition validation** — Prevent invalid transitions (e.g., `delivered` → `pending`) in the serializer's `validate()` method.

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/products/models.py` | Product + Category models |
| `apps/orders/models.py` | Order + OrderItem + OrderStatus models |
| `api/views.py` | All 21 ViewSets |
| `api/serializers/` | 6 serializer files |
| `api/permissions.py` | Custom permission classes |
| `api/pagination.py` | Pagination configuration |
| `api/urls.py` | Router + URL registration |
