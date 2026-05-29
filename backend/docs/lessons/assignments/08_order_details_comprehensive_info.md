# Intern Assignment 08: Wire the Order Details Button — Comprehensive Order Information Tool

**Track:** Backend — LangGraph Tools + Django ORM + Template Updates
**Difficulty:** Intermediate
**Estimated Effort:** 4–5 hours
**Prerequisites:** Understand the order detail page, comfortable with Django `select_related`/`prefetch_related`

---

## Problem Statement

The order detail page at `/orders/<id>/` has an **"Order Details"** button that sends the `order_detail` intent over WebSocket. When clicked, the agent should provide **comprehensive order information** — but it can't, because the data it receives is incomplete:

```python
# apps/orders/utils/db_utils.py — get_order_details() returns this:
{
    "order_id": "12",
    "status": "Shipped",
    "status_description": "SH",
    "user": <User object>,          # ← Serialized as string, useless
    "items": [
        {
            "product_id": "5",
            "product_name": "Wireless Headphones",
            "quantity": 2,
            "price": "99.99"
        }
    ],
    "total_amount": "199.98"
}
```

**What's missing:**

| Field | Why It Matters |
|-------|---------------|
| Product category | Agent can't answer "What category is this product?" |
| Shipping address | Agent can't confirm where the order is going |
| Carrier name | Agent can't say "Your order is shipped via FedEx" |
| Tracking number | Agent can't give the tracking number directly |
| Estimated delivery | Agent can't say when the order will arrive |
| Shipping method | Agent can't say "You chose express shipping" |
| Order date | Agent can't say "You placed this order on May 15" |
| Product image/description | Agent can't describe the product |

The `get_order_info` tool is even worse — it returns a **hardcoded text string** with no real data:

```python
# tool_manager.py — get_order_info returns static text
status_actions = {
    "Pending": ["- Modify quantities", "- Cancel order"],
    "Processing": ["- Modify quantities", "- Cancel order"],
    ...
}
return f"Order #{order_details['order_id']} Support Information\n" + ...
```

This means when a user clicks "Order Details" and asks "What's my shipping address?" or "When will my order arrive?" — the agent **doesn't know**.

---

## Root Cause Analysis

### Gap 1: `get_order_details()` Returns Incomplete Data

**File:** `apps/orders/utils/db_utils.py`, line ~443

The method only queries `Order.objects.get(id=order_id)` with **no `select_related` or `prefetch_related`**. It doesn't join:
- `OrderItem → Product → Category` (for product categories)
- `Order.tracking_history` (for latest tracking)
- No shipping fields: `shipping_method`, `carrier`, `tracking_number`, `delivery_address`, `estimated_delivery`

### Gap 2: `get_order_info` Tool Is a Static Text Generator

**File:** `apps/orders/utils/tool_manager.py`, line ~469

The tool calls `get_order_details` (which returns incomplete data) and then just formats a text response with hardcoded action lists per status. It doesn't use any of the rich data available in the `Order` model.

### Gap 3: The Prompt Doesn't Document the Tool Properly

**File:** `apps/orders/utils/prompt_manager.py`, `order_detail` prompt

The prompt tells the agent to "provide clear order summary" and "use track_order tool for status updates" but doesn't mention `get_order_info`. The LLM doesn't know what tool to use for comprehensive details.

---

## Assignment Tasks

### Task 1: Fix `get_order_details()` to Return Complete Data (30 min)

**File:** `apps/orders/utils/db_utils.py`

Rewrite `get_order_details` to fetch and return ALL relevant order information:

```python
@database_sync_to_async
def get_order_details(self, order_id: str) -> dict:
    """Fetch comprehensive order information including shipping, tracking, and product details."""
    try:
        order = Order.objects.select_related('user').prefetch_related(
            'items__product__category',
            'tracking_history',
        ).get(id=order_id)

        # Latest tracking event
        latest_tracking = order.tracking_history.order_by('-timestamp').first()

        # Build items list with categories
        items = []
        for item in order.items.all():
            items.append({
                "product_id": str(item.product.id),
                "product_name": item.product.name,
                "category": item.product.category.name if item.product.category else "Uncategorized",
                "quantity": item.quantity,
                "price": str(item.price),
                "subtotal": str(item.price * item.quantity),
            })

        return {
            "order_id": str(order.id),
            "status": order.get_status_display(),
            "status_code": order.status,
            "user_id": str(order.user.id),
            "username": order.user.username,
            "created": order.created.isoformat() if order.created else None,
            "modified": order.modified.isoformat() if order.modified else None,
            "items": items,
            "total_amount": str(order.total_amount),
            # Shipping details
            "shipping_address": order.delivery_address,
            "shipping_method": order.get_shipping_method_display(),
            "carrier": order.carrier,
            "tracking_number": order.tracking_number,
            "estimated_delivery": (
                order.estimated_delivery.isoformat()
                if order.estimated_delivery else None
            ),
            "shipped_date": (
                order.shipped_date.isoformat()
                if order.shipped_date else None
            ),
            # Latest tracking event
            "latest_tracking": {
                "status": latest_tracking.get_status_display() if latest_tracking else None,
                "location": latest_tracking.location if latest_tracking else None,
                "description": latest_tracking.description if latest_tracking else None,
                "timestamp": (
                    latest_tracking.timestamp.isoformat()
                    if latest_tracking and latest_tracking.timestamp else None
                ),
            } if latest_tracking else None,
        }
    except Order.DoesNotExist:
        return {"error": "Order not found"}
```

**Testing:** In a Django shell, verify the output:
```python
from orders.utils.db_utils import DatabaseOperations
import asyncio

async def test():
    db = DatabaseOperations(user)  # pass a real user
    result = await db.get_order_details("12")
    print(result)

asyncio.run(test())
```

Verify the result now includes `shipping_address`, `carrier`, `tracking_number`, `category`, etc.

---

### Task 2: Replace `get_order_info` with `get_order_summary` Tool (30 min)

**File:** `apps/orders/utils/tool_manager.py`

**2a.** Replace the `_create_get_order_info_tool` method with a comprehensive summary tool:

```python
def _create_get_order_info_tool(self):
    """Create the comprehensive order summary tool"""

    @tool(args_schema=GetSupportInfo)
    async def get_order_info(order_id: str, customer_id: str) -> str:
        """Get comprehensive order summary including items, shipping, tracking, and available actions.
        Use this tool when the customer wants to know details about their order."""
        try:
            order_details = await self.db_ops.get_order_details(order_id)
            if "error" in order_details:
                return "Order not found"

            # Build comprehensive summary
            summary_parts = []

            # Order header
            summary_parts.append(
                f"Order #{order_details['order_id']}\n"
                f"Status: {order_details['status']}\n"
                f"Placed: {order_details['created']}\n"
                f"Total: ${order_details['total_amount']}"
            )

            # Items with categories
            if order_details.get('items'):
                items_text = []
                for item in order_details['items']:
                    items_text.append(
                        f"  - {item['product_name']} "
                        f"(Category: {item['category']}) "
                        f"x{item['quantity']} @ ${item['price']} = ${item['subtotal']}"
                    )
                summary_parts.append("Items:\n" + "\n".join(items_text))

            # Shipping info
            shipping_parts = []
            if order_details.get('shipping_address'):
                shipping_parts.append(f"Address: {order_details['shipping_address']}")
            if order_details.get('shipping_method'):
                shipping_parts.append(f"Method: {order_details['shipping_method']}")
            if order_details.get('carrier'):
                shipping_parts.append(f"Carrier: {order_details['carrier']}")
            if order_details.get('tracking_number'):
                shipping_parts.append(f"Tracking #: {order_details['tracking_number']}")
            if order_details.get('estimated_delivery'):
                shipping_parts.append(f"Est. Delivery: {order_details['estimated_delivery']}")
            if shipping_parts:
                summary_parts.append("Shipping:\n" + "\n".join(f"  {s}" for s in shipping_parts))

            # Latest tracking event
            tracking = order_details.get('latest_tracking')
            if tracking:
                summary_parts.append(
                    f"Latest Tracking: {tracking['status']} "
                    f"at {tracking['location']} — {tracking['description']} "
                    f"({tracking['timestamp']})"
                )

            # Available actions based on status
            status_code = order_details.get('status_code', '')
            actions = {
                'PE': ["Modify quantities", "Cancel order"],
                'PR': ["Modify quantities", "Cancel order"],
                'SH': ["Track shipment", "Report issue"],
                'TR': ["Track shipment", "Report issue"],
                'DE': ["Return items (within 30 days)", "Report issue"],
            }
            available = actions.get(status_code, [])
            if available:
                summary_parts.append("Available Actions: " + ", ".join(available))

            return "\n\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Error getting order info: {str(e)}")
            logger.error(traceback.format_exc())
            return "Failed to get order information"

    return get_order_info
```

**Note:** The tool name stays `get_order_info` to avoid breaking existing references. The schema (`GetSupportInfo`) and registration in `_register_tools` stay the same — only the implementation changes.

---

### Task 3: Update the Order Details Prompt (20 min)

**File:** `apps/orders/utils/prompt_manager.py`

Replace the `order_detail` prompt with one that knows about the available data:

```python
'order_detail': ChatPromptTemplate.from_messages([
    ("system", """You are a customer support assistant for order details and information.

    Current order details: {order_info}
    Previous conversation: {conversation_history}

    ## Available Tools
    - **get_order_info**: Get a comprehensive order summary including items, categories,
      shipping details, tracking number, carrier, estimated delivery, and available actions.
      Use this FIRST when the customer asks about their order.
    - **track_order**: Get real-time tracking status and location.
    - **get_tracking_details**: Get full tracking history with all events.
    - **get_delivery_estimate**: Get estimated delivery date and confidence level.
    - **get_shipment_location**: Get current shipment location.

    ## Guidelines
    1. For ANY question about the order, call get_order_info first for complete context.
    2. Provide item details including product name, category, quantity, and price.
    3. For shipping questions, share the address, carrier, tracking number, and method.
    4. For delivery questions, use get_delivery_estimate for accurate timelines.
    5. For tracking questions, use get_tracking_details for full history.
    6. Always mention available actions based on order status:
       - Pending/Processing: Can modify or cancel
       - Shipped/In Transit: Can track or report issue
       - Delivered: Can return or report issue
    7. Format monetary values with $ and 2 decimal places.
    8. Format dates in a human-readable way (e.g., "May 28, 2026").
    """),
    ("human", "{user_input}"),
]),
```

---

### Task 4: Update the Template to Show More Order Details (20 min)

**File:** `apps/orders/templates/orders/order_detail.html`

The order items table (around line 55-91) currently shows: Product, Category, Quantity, Price, Total. But the tracking section and shipping details could show more. Add a small info panel that the agent can reference:

Find the Order Summary card (the `Shipping Details` and `Order Details` columns) and verify it includes these fields. If any are missing from the template display, add them:

```html
<!-- In the Order Details column, after estimated delivery, add: -->
{% if order.carrier %}
<p><strong>Carrier:</strong> {{ order.carrier }}</p>
{% endif %}
{% if order.tracking_number %}
<p><strong>Tracking #:</strong> {{ order.tracking_number }}</p>
{% endif %}
```

Also, ensure the JavaScript `OrderSupport` class has access to order details. Find where `order_id` is extracted and add a data attribute to the order card for status:

```html
<div class="card mb-3" data-order-id="{{ order.id }}" data-order-status="{{ order.status }}">
```

This lets the JavaScript check the order status for UI decisions.

---

### Task 5: Write Tests (45 min)

**File:** `apps/orders/tests/test_order_details.py` (create)

```python
import pytest
from django.contrib.auth import get_user_model

from orders.models import Order, OrderItem, OrderTracking
from products.models import Product, Category

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="detailuser", password="testpass")


@pytest.fixture
def category(db):
    return Category.objects.create(name="Electronics", slug="electronics")


@pytest.fixture
def product(db, category):
    return Product.objects.create(
        name="Wireless Headphones", slug="wireless-headphones",
        price=99.99, stock=50, category=category,
    )


@pytest.fixture
def shipped_order(user, product):
    order = Order.objects.create(
        user=user,
        status=Order.Status.SHIPPED,
        total_amount=199.98,
        shipping_method=Order.ShippingMethod.EXPRESS,
        carrier="FedEx",
        tracking_number="FX123456789",
        delivery_address="123 Main St, Springfield, IL 62701",
    )
    OrderItem.objects.create(order=order, product=product, quantity=2, price=99.99)
    OrderTracking.objects.create(
        order=order,
        status=OrderTracking.TrackingStatus.IN_TRANSIT,
        location="Chicago Hub",
        description="Package in transit",
    )
    return order


@pytest.mark.django_db
class TestGetOrderDetails:
    @pytest.mark.asyncio
    async def test_returns_complete_order_info(self, user, shipped_order):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.get_order_details(str(shipped_order.id))

        # Basic fields
        assert result["order_id"] == str(shipped_order.id)
        assert result["status"] == "Shipped"
        assert result["total_amount"] == "199.98"

        # Shipping fields (previously missing)
        assert result["shipping_address"] == "123 Main St, Springfield, IL 62701"
        assert result["carrier"] == "FedEx"
        assert result["tracking_number"] == "FX123456789"
        assert result["shipping_method"] == "Express"

        # Items with categories
        assert len(result["items"]) == 1
        assert result["items"][0]["category"] == "Electronics"
        assert result["items"][0]["product_name"] == "Wireless Headphones"
        assert result["items"][0]["subtotal"] == "199.98"

        # Latest tracking
        assert result["latest_tracking"] is not None
        assert result["latest_tracking"]["location"] == "Chicago Hub"

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_order(self, user):
        from orders.utils.db_utils import DatabaseOperations

        db_ops = DatabaseOperations(user)
        result = await db_ops.get_order_details("99999")
        assert "error" in result


@pytest.mark.django_db
class TestGetOrderInfoTool:
    @pytest.mark.asyncio
    async def test_tool_returns_comprehensive_summary(self, user, shipped_order):
        from orders.utils.db_utils import DatabaseOperations
        from orders.utils.tool_manager import create_tool_manager

        db_ops = DatabaseOperations(user)
        tool_manager = create_tool_manager(db_ops)

        # Find the get_order_info tool
        tool = next(
            (t for t in tool_manager.get_all_tools() if t.name == "get_order_info"),
            None,
        )
        assert tool is not None

        result = await tool.ainvoke({
            "order_id": str(shipped_order.id),
            "customer_id": str(user.id),
        })

        # Should include shipping and category info
        assert "FedEx" in result
        assert "FX123456789" in result
        assert "Electronics" in result
        assert "123 Main St" in result
        assert "Track shipment" in result  # Available action for shipped orders

    @pytest.mark.asyncio
    async def test_tool_handles_missing_order(self, user):
        from orders.utils.db_utils import DatabaseOperations
        from orders.utils.tool_manager import create_tool_manager

        db_ops = DatabaseOperations(user)
        tool_manager = create_tool_manager(db_ops)

        tool = next(
            (t for t in tool_manager.get_all_tools() if t.name == "get_order_info"),
            None,
        )
        result = await tool.ainvoke({
            "order_id": "99999",
            "customer_id": str(user.id),
        })
        assert "not found" in result.lower()


@pytest.mark.django_db
class TestOrderDetailsPrompt:
    def test_prompt_exists(self):
        from orders.utils.prompt_manager import PromptManager

        PromptManager.initialize()
        prompt = PromptManager.get_prompt('order_detail')
        assert prompt is not None

        # Verify prompt mentions the key tool
        system_content = prompt.messages[0].content
        assert "get_order_info" in system_content

    def test_prompt_mentiones_categories(self):
        from orders.utils.prompt_manager import PromptManager

        PromptManager.initialize()
        prompt = PromptManager.get_prompt('order_detail')
        system_content = prompt.messages[0].content
        assert "category" in system_content.lower()
```

Cover:
- `get_order_details` returns shipping address, carrier, tracking number, categories, latest tracking
- `get_order_info` tool includes all the new fields in its output
- Error handling for missing orders
- Prompt mentions `get_order_info` and categories

---

## File Reference Map

```
backend/
├── apps/orders/
│   ├── utils/
│   │   ├── db_utils.py                  ← Task 1 (fix get_order_details)
│   │   ├── tool_manager.py              ← Task 2 (rewrite get_order_info tool)
│   │   └── prompt_manager.py            ← Task 3 (update order_detail prompt)
│   ├── templates/orders/
│   │   └── order_detail.html            ← Task 4 (add carrier/tracking display)
│   └── tests/
│       └── test_order_details.py        ← Task 5 (create)
└── docs/lessons/assignments/
    └── 08_order_details_comprehensive_info.md  ← this file
```

---

## Key Concepts to Learn

1. **`select_related` and `prefetch_related`** — Follow FK chains (`OrderItem → Product → Category`) to avoid N+1 queries. Use `select_related` for ForeignKey, `prefetch_related` for reverse relations.
2. **Tool output formatting** — Format tool results as clear, structured text the LLM can understand and relay to the user.
3. **Prompt-driven tool usage** — The prompt must explicitly tell the LLM which tool to use and when. LLMs can't discover tools on their own.
4. **Data completeness** — When building APIs or tools, think about ALL the data a user might ask about, not just what you need for the main use case.
5. **Django model display values** — `get_status_display()` returns the human-readable label from `TextChoices`, not the stored code.

---

## Submission Checklist

- [ ] `get_order_details()` returns shipping address, carrier, tracking number, categories
- [ ] `get_order_details()` uses `select_related` and `prefetch_related` to avoid N+1
- [ ] `get_order_info` tool includes items with categories, shipping details, tracking
- [ ] `get_order_info` tool shows available actions based on order status
- [ ] `order_detail` prompt documents `get_order_info` as the primary tool
- [ ] Template shows carrier and tracking number in the summary
- [ ] Order card has `data-order-status` attribute
- [ ] All existing tests pass
- [ ] New tests in `test_order_details.py` pass

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
