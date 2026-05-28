# Intern Assignment 01: Category-Aware Order Support Agent

**Track:** Backend — LangGraph Agent & Django REST API
**Difficulty:** Intermediate
**Estimated Effort:** 4–6 hours
**Prerequisites:** Complete the onboarding guide (`docs/lessons/`), understand LangGraph basics (`docs/langgraph_workflow_vs_agents.md`)

---

## Problem Statement

The e-commerce support agent can list a user's orders and track specific orders, but it **cannot understand or filter by product category**.

A user with these orders:

```
Order #12 — Shipped — $1048.00
  1x Aparsoft Phone X     (Category: Smartphones)
  1x StudioMic One         (Category: Audio)

Order #9 — Delivered — $346.00
  1x USB-C Hub             (Category: Accessories)
  2x PulseBand 3           (Category: Wearables)
  1x Fast Charger 65W      (Category: Accessories)
```

...should be able to ask the agent:
- "Show me my audio orders"
- "Which orders have electronics?"
- "Do I have any accessory items?"
- "What categories have I ordered?"

Right now the agent **cannot do any of this** because the category data never flows from the database through the tools to the LLM.

---

## Root Cause Analysis

The database already has the category data via this relationship chain:

```
Order → OrderItem → Product → Category
```

But **every layer** between the database and the LLM is missing category support. Here is the full gap analysis:

### Gap 1: Database Layer (`orders/utils/db_utils.py`)

| Method | Current Return | Missing |
|--------|---------------|---------|
| `get_recent_orders()` | `id, created_date, status, item_count, total_amount` | Items with product names **and categories** |
| `get_order_details()` | Items with `product_name, quantity, price` | `category` field on each item |
| *(missing method)* | — | `get_orders_by_category(user_id, category)` — filter orders by category |

### Gap 2: LangGraph Tools (`support_agent/sa_utils/tool_manager.py`)

| Tool | Current Behavior | Missing |
|------|-----------------|---------|
| `list_user_orders` | Lists recent orders (no items, no categories) | Include items with categories in response |
| *(missing tool)* | — | `search_orders_by_category(customer_id, category)` — filter by category |

### Gap 3: System Prompt (`support_agent/sa_utils/prompt_manager.py`)

The prompt does not mention category-based querying. The LLM doesn't know it can filter by category.

### Gap 4: REST API (`api/v1/views_orders.py`)

| ViewSet | Current Filters | Missing |
|---------|----------------|---------|
| `OrderViewSet` | `status, shipping_method, carrier, user` | `items__product__category` |
| `OrderItemViewSet` | `order, product` | `product__category` |

---

## Assignment Tasks

### Task 1: Enrich Database Operations (30 min)

**File:** `backend/apps/orders/utils/db_utils.py`

**1a.** Update `get_recent_orders()` to include items with categories:

```python
# CURRENT:
return [{
    'id': order.id,
    'created_date': order.created.strftime("%Y-%m-%d"),
    'status': order.get_status_display(),
    'item_count': order.items.count(),
    'total_amount': order.total_amount
} for order in recent_orders]

# TARGET:
return [{
    'id': order.id,
    'created_date': order.created.strftime("%Y-%m-%d"),
    'status': order.get_status_display(),
    'total_amount': order.total_amount,
    'items': [
        {
            'product_id': item.product.id,
            'product_name': item.product.name,
            'category': item.product.category.name,
            'quantity': item.quantity,
            'price': str(item.price),
        }
        for item in order.items.select_related('product__category').all()
    ],
} for order in recent_orders]
```

**Hint:** You need `.select_related('product__category')` on the OrderItem queryset to avoid N+1 queries.

**1b.** Update `get_order_details()` to include category on each item:

Add `'category': item.product.category.name` to each item dict in the items list.

**1c.** Create a new method `get_orders_by_category()`:

```python
@database_sync_to_async
def get_orders_by_category(self, category_name: str, limit: int = 10) -> List[Dict]:
    """Get orders containing products from a specific category."""
    orders = (
        Order.objects
        .filter(user=self.user, items__product__category__name__icontains=category_name)
        .distinct()
        .order_by('-created')[:limit]
    )
    # Return same format as get_recent_orders (with items + categories)
    ...
```

**Testing:** Write a test in `orders/tests/test_db_utils.py` (create if needed) that:
- Creates a user, categories, products, orders
- Verifies `get_recent_orders()` returns items with categories
- Verifies `get_orders_by_category('Audio')` returns only orders with audio products

---

### Task 2: Add Category-Aware Tools (30 min)

**File:** `backend/apps/support_agent/sa_utils/tool_manager.py`

**2a.** Update `list_user_orders` tool to format items with categories:

```python
# In the formatting loop, change from:
f"Items: {order['item_count']}\n"
# To listing each item with category:
for item in order.get('items', []):
    formatted_items.append(
        f"  - {item['quantity']}x {item['product_name']} "
        f"({item['category']}) — ${item['price']}"
    )
```

**2b.** Create a new tool `search_orders_by_category`:

```python
@tool
async def search_orders_by_category(
    customer_id: str,
    category: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Search a customer's orders filtered by product category.

    Use when a customer asks about orders for a specific product type.
    Examples: 'show my electronics orders', 'do I have any audio orders?',
    'what wearable items have I bought?'

    Args:
        customer_id: The ID of the customer
        category: The product category to filter by (e.g., 'Smartphones',
                  'Audio', 'Accessories', 'Wearables')
        config: RunnableConfig containing database operations

    Returns:
        Formatted list of matching orders with their items
    """
    ...
```

**2c.** Add the new tool to the `TOOLS` list:

```python
TOOLS: List[Callable[..., Any]] = [
    list_user_orders,
    search_orders_by_category,   # <-- new
    track_order,
    modify_order_quantity,
    cancel_order,
    web_search,
]
```

---

### Task 3: Update the System Prompt (15 min)

**File:** `backend/apps/support_agent/sa_utils/prompt_manager.py`

Add the new tool to the `## Available Tools` section in the system prompt:

```
- **search_orders_by_category**: Search orders filtered by product category.
  Call when user asks about orders for a specific product type.
  Examples: 'show my audio orders', 'any electronics?'.
```

Add a new routing rule:

```
6. If user asks about orders for a specific product type or category
   -> call search_orders_by_category with category="<the category>" and
      customer_id="{customer_id}".
```

---

### Task 4: Add Category Filter to REST API (20 min)

**File:** `backend/apps/api/v1/views_orders.py`

**4a.** Add `items__product__category` to `OrderViewSet.filterset_fields`:

```python
filterset_fields = ["status", "shipping_method", "carrier", "user",
                     "items__product__category"]
```

**4b.** Add `product__category` to `OrderItemViewSet.filterset_fields`:

```python
filterset_fields = ["order", "product", "product__category"]
```

**4c.** (Optional) Add a custom action `by-category` on `OrderViewSet`:

```python
@action(detail=False, methods=["get"], url_path="by-category/(?P<category>[^/.]+)")
def by_category(self, request, category=None):
    """List orders containing products from the given category."""
    qs = self.get_queryset().filter(items__product__category__name__icontains=category).distinct()
    page = self.paginate_queryset(qs)
    serializer = self.get_serializer(page, many=True)
    return self.get_paginated_response(serializer.data)
```

**Testing:** Write API tests in `api/v1/tests/test_orders_api.py` that verify:
- `GET /api/v1/orders/?items__product__category=Smartphones` returns only orders with smartphone products
- `GET /api/v1/order-items/?product__category=Audio` returns only audio items

---

### Task 5: End-to-End Verification (15 min)

After completing all tasks:

1. Start the server: `make run`
2. Open the support agent WebSocket at `/ws/support/<conversation_id>/`
3. Send these messages and verify the agent responds correctly:

| Message | Expected Behavior |
|---------|------------------|
| "show my orders" | Lists all orders with items and categories |
| "do I have any audio orders?" | Calls `search_orders_by_category` with category="Audio" |
| "show my electronics" | Calls `search_orders_by_category` with category="electronics" |
| "track order #12" | Shows tracking with items including categories |
| "what categories have I ordered?" | Lists categories from all order items |

4. Verify the REST API:

```bash
# List orders filtered by category
curl -H "Authorization: Token $TOKEN" \
  "http://localhost:8000/api/v1/orders/?items__product__category=Smartphones"

# List order items filtered by category
curl -H "Authorization: Token $TOKEN" \
  "http://localhost:8000/api/v1/order-items/?product__category=Audio"
```

---

## File Reference Map

All files you will touch, in order:

```
backend/
├── apps/
│   ├── orders/
│   │   ├── utils/
│   │   │   └── db_utils.py              ← Task 1 (enrich DB queries)
│   │   └── tests/
│   │       └── test_db_utils.py          ← Task 1 (write tests, create if needed)
│   ├── support_agent/
│   │   └── sa_utils/
│   │       ├── tool_manager.py           ← Task 2 (update + add tool)
│   │       └── prompt_manager.py         ← Task 3 (update prompt)
│   └── api/
│       └── v1/
│           ├── views_orders.py           ← Task 4 (add filters)
│           └── tests/
│               └── test_orders_api.py    ← Task 4 (write tests)
└── docs/
    └── lessons/
        └── assignments/
            └── 01_category_aware_order_agent.md  ← this file
```

---

## Key Concepts to Learn

1. **Django `select_related`** — Follows FK relationships in a single SQL query (avoids N+1)
   - `OrderItem.objects.select_related('product__category')` — joins through Product to Category
2. **Django `__icontains`** — Case-insensitive contains lookup
   - `Category.objects.filter(name__icontains='audio')` matches "Audio", "AUDIO", "audio equipment"
3. **LangGraph `@tool`** — The docstring is what the LLM reads to decide when to call the tool
   - Include examples in the docstring so the LLM knows exactly when to use it
4. **DjangoFilterBackend** — Automatically creates filter query params from `filterset_fields`
   - Adding `"items__product__category"` allows `?items__product__category=Smartphones`
5. **`InjectedToolArg`** — Marks tool parameters that LangGraph injects (not from the LLM)
   - The LLM provides `customer_id` and `category`, LangGraph injects `config`

---

## Submission Checklist

- [ ] `get_recent_orders()` returns items with `category` field
- [ ] `get_order_details()` returns items with `category` field
- [ ] `get_orders_by_category()` method works
- [ ] `list_user_orders` tool shows categories in output
- [ ] `search_orders_by_category` tool is created and in `TOOLS` list
- [ ] System prompt mentions category-based filtering
- [ ] `OrderViewSet` filters by `items__product__category`
- [ ] `OrderItemViewSet` filters by `product__category`
- [ ] All existing tests still pass (`pytest apps/ -v`)
- [ ] New tests added for category filtering
- [ ] End-to-end verification with WebSocket messages works

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
