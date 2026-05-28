# Quick Win 04: Views & URLs — Handling Requests

> Views contain your logic. URLs route requests to the right view.

---

## Two Types of Views

### 1. Function-Based Views (FBV)

```python
# Simple and explicit
from django.http import HttpResponse

def hello(request):
    return HttpResponse("Hello, world!")
```

### 2. Class-Based Views (CBV) — Used in this project

```python
from django.views.generic import CreateView

class SignupView(CreateView):
    template_name = "accounts/signup.html"
    form_class = CustomSignupForm
    success_url = reverse_lazy("accounts:login")
```

---

## CBVs We Use

| CBV | What It Gives You | Example |
|-----|-------------------|---------|
| `CreateView` | Form + save new object | `SignupView` — create user |
| `TemplateView` | Just render a template | `LoginView`, `ProfileView` |
| `RedirectView` | Redirect to another URL | `LogoutView` |
| `ListView` | List of objects + pagination | Product list page |
| `DetailView` | Single object detail | Product detail page |
| `UpdateView` | Form + update existing object | Profile update |

### How a `CreateView` works:

```
GET  /accounts/signup/ → Show empty form
POST /accounts/signup/ → Validate form → Save to DB → Redirect
```

You don't write `get()` or `post()` — the CBV handles it. You just configure the template, form, and redirect URL.

---

## URL Routing

### URL patterns connect URLs to views:

```python
# apps/accounts/urls.py
app_name = 'accounts'

urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('profile/', ProfileView.as_view(), name='profile'),
]
```

### Including in the root URL config:

```python
# config/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('api.urls')),
    path('accounts/', include('accounts.urls')),
    path('dashboard/', include('dashboard.urls')),
]
```

### URL names → reverse in Python and templates:

```python
# In Python code
url = reverse('accounts:login')    # → '/accounts/login/'

# In templates
<a href="{% url 'accounts:login' %}">Login</a>
```

---

## DRF ViewSets (API Views)

For the REST API, we use DRF `ModelViewSet` — one class gives you all CRUD:

```python
# apps/api/v1/views_orders.py
class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.select_related("user").prefetch_related(
        "items__product", "tracking_history"
    )
    serializer_class = OrderListSerializer
    permission_classes = [IsOwnerOrReadOnly]
    pagination_class = StandardResultsSetPagination
```

### What this one class creates:

```
GET    /api/v1/orders/              → list all orders
POST   /api/v1/orders/              → create new order
GET    /api/v1/orders/{id}/         → get one order
PUT    /api/v1/orders/{id}/         → replace order
PATCH  /api/v1/orders/{id}/         → partial update
DELETE /api/v1/orders/{id}/         → delete order
```

### Custom actions beyond CRUD:

```python
from rest_framework.decorators import action

class ProductViewSet(viewsets.ModelViewSet):
    # ...

    @action(detail=False, methods=['get'])
    def featured(self, request):
        """GET /api/v1/products/featured/"""
        products = self.queryset.filter(is_active=True)[:10]
        serializer = self.get_serializer(products, many=True)
        return Response(serializer.data)
```

- `detail=False` → works on the collection (no `{id}`)
- `detail=True` → works on a single instance (has `{id}`)

### How URLs are auto-generated:

```python
# api/urls.py
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'orders', OrderViewSet)
router.register(r'products', ProductViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

The router reads the ViewSet and generates all the URL patterns automatically.

---

## Quick Exercise

1. Open `apps/accounts/views.py` and `apps/accounts/urls.py` — trace how `SignupView` connects to `/accounts/signup/`
2. Open `apps/api/v1/views_orders.py` and `api/urls.py` — find how the `OrderViewSet` is registered
3. Create a new simple view:
   ```python
   # In any app's views.py
   from django.http import JsonResponse

   def health_check(request):
       return JsonResponse({"status": "ok"})
   ```
   Add it to `urls.py` and test it in the browser.
