# Quick Win 02: Models — Your Database in Python

> Define database tables as Python classes. No SQL needed.

---

## The Basics

A Django **model** = one database table. Each attribute = one column.

```python
# apps/products/models.py
from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
```

**What this creates in PostgreSQL:**
```sql
CREATE TABLE products_category (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    description TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT NOW()
);
```

Django adds `id` automatically. You never write SQL.

---

## Field Types You'll See Everywhere

| Field | Python Type | Database | Use For |
|-------|------------|----------|---------|
| `CharField(max_length=N)` | `str` | `VARCHAR(N)` | Names, titles, short text |
| `TextField()` | `str` | `TEXT` | Long text, descriptions |
| `IntegerField()` | `int` | `INTEGER` | Counts, quantities |
| `DecimalField(max_digits, decimal_places)` | `Decimal` | `NUMERIC` | Prices, money |
| `BooleanField()` | `bool` | `BOOLEAN` | True/false flags |
| `DateTimeField(auto_now_add=True)` | `datetime` | `TIMESTAMP` | Created timestamps |
| `DateTimeField(auto_now=True)` | `datetime` | `TIMESTAMP` | Updated timestamps |
| `ForeignKey(OtherModel)` | `OtherModel` | `INTEGER (FK)` | Relationships |
| `JSONField()` | `dict/list` | `JSONB` | Flexible data |

---

## Relationships

### ForeignKey (Many-to-One)

```python
# apps/orders/models.py
class Order(models.Model):
    user = models.ForeignKey(
        User,                    # Related model
        on_delete=models.CASCADE  # Delete orders when user is deleted
    )
    status = models.CharField(
        max_length=2,
        choices=Status.choices,   # TextChoices enum
        default=Status.PENDING
    )
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"Order {self.id} by {self.user.username}"
```

**Accessing related data:**
```python
order = Order.objects.get(id=1)
order.user              # → User object
order.user.username     # → "john"

# Reverse: from user to orders
user = User.objects.get(id=1)
user.order_set.all()    # → QuerySet of all orders by this user
```

### Multiple ForeignKeys on one model:

```python
class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)
```

**`related_name='items'`** lets you write:
```python
order.items.all()       # Instead of order.orderitem_set.all()
```

---

## Choices (Enums)

```python
class Order(CreationModificationDateBase):
    class Status(models.TextChoices):
        PENDING = 'PE', _('Pending')
        PROCESSING = 'PR', _('Processing')
        SHIPPED = 'SH', _('Shipped')
        DELIVERED = 'DE', _('Delivered')
        CANCELLED = 'CA', _('Cancelled')

    status = models.CharField(
        max_length=2,
        choices=Status.choices,
        default=Status.PENDING
    )
```

**Using choices:**
```python
order = Order.objects.get(id=1)
order.status              # → 'PE' (the raw value)
order.get_status_display()  # → 'Pending' (human-readable)

# Filter by choice
Order.objects.filter(status=Order.Status.SHIPPED)
```

---

## The ORM — Querying the Database

```python
# Get all
Product.objects.all()

# Filter
Product.objects.filter(is_active=True)
Product.objects.filter(price__lt=50)           # price < 50
Product.objects.filter(name__icontains="phone") # name contains "phone"

# Get one
Product.objects.get(id=1)           # Raises DoesNotExist if not found
Product.objects.filter(id=1).first() # Returns None if not found

# Create
Product.objects.create(name="Widget", price=Decimal("9.99"))

# Update
product = Product.objects.get(id=1)
product.price = Decimal("12.99")
product.save()

# Delete
product.delete()

# Count
Product.objects.filter(is_active=True).count()

# Order
Product.objects.all().order_by('-price')  # descending

# Related object queries (JOIN)
Order.objects.select_related('user')              # FK join (single object)
Order.objects.prefetch_related('items__product')   # M2M/reverse (querysets)
```

### Chaining queries:

```python
# This does NOT hit the database yet
qs = Product.objects.filter(is_active=True)

# Still no database hit
qs = qs.filter(category_id=2).order_by('-price')

# NOW it hits the database (evaluation)
for product in qs:
    print(product.name)
```

QuerySets are **lazy** — they only hit the database when you actually need the data.

---

## Migrations

When you change a model, create and apply migrations:

```bash
# 1. Create migration files
python manage.py makemigrations

# 2. Apply to database
python manage.py migrate

# 3. See what SQL will run (without applying)
python manage.py sqlmigrate products 0001
```

---

## Quick Exercise

1. Open `apps/orders/models.py` — read the `Order`, `OrderItem`, and `OrderTracking` models
2. In the Django shell:
   ```python
   from orders.models import Order
   Order.objects.count()
   Order.objects.filter(status='PE').count()
   order = Order.objects.first()
   order.items.all()  # Related OrderItems
   ```
3. Add a new field to a model (e.g., `notes = models.TextField(blank=True)` to Order), run `makemigrations`, then `migrate`
