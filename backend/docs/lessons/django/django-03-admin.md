# Quick Win 03: Django Admin — Free CRUD Interface

> Django gives you a full admin panel for free. Just register your models.

---

## What is the Django Admin?

A ready-made web interface for managing your database. Create, read, update, delete any model — no frontend code needed.

Access it at: `http://localhost:8000/admin/`

---

## Registering Models

```python
# apps/products/admin.py
from django.contrib import admin
from .models import Product, Category

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'price', 'category', 'is_active', 'stock']
    list_filter = ['is_active', 'category']
    search_fields = ['name', 'description']
    ordering = ['-created_at']

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'description']
```

**What each option does:**
- `list_display` — columns shown in the list view
- `list_filter` — sidebar filters
- `search_fields` — enables a search box
- `ordering` — default sort order

---

## Key Admin Options

```python
@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    # List page
    list_display = ['id', 'user', 'status', 'total_amount', 'created_at']
    list_filter = ['status', 'shipping_method']
    list_editable = ['status']              # Editable right in the list!
    list_per_page = 25                       # Pagination
    search_fields = ['user__username', 'id']
    ordering = ['-created_at']

    # Detail page
    readonly_fields = ['created_at', 'updated_at']
    raw_id_fields = ['user']                 # Searchable FK dropdown
    date_hierarchy = 'created_at'            # Clickable date drill-down

    # Inline editing (edit OrderItems inside the Order page)
    inlines = [OrderItemInline]


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 1                                # 1 blank row for new items
    fields = ['product', 'quantity', 'price']
```

---

## The Admin Login

Create your first admin user:

```bash
python manage.py createsuperuser
# Enter username, email, password
```

This creates a `User` with `is_staff=True` and `is_superuser=True`, which grants admin access.

---

## Quick Exercise

1. Run `python manage.py createsuperuser` if you haven't already
2. Start the server and visit `/admin/`
3. Register a model that isn't registered yet (check each app's `admin.py`)
4. Add `list_editable` to an existing admin class and see how you can edit records inline
5. Try adding a `TabularInline` or `StackedInline` for a related model
